"""Selections: an 8-bit mask, the things that build one, and floating pixels.

A selection is a mask rather than a rectangle, because everything interesting
about one -- a lasso, a wand, a feathered edge -- is not rectangular. Eight bits
rather than one, because feathering *is* the intermediate values: a mask at 128
means "write half of this pixel", which is the same arithmetic a brush stamp's
coverage already does, so the clip is one multiply and no special case.

The mask lives in canvas space at canvas size. That wastes a megabyte on a
tiny selection and buys the property that every op is a slice with no offset
arithmetic -- the same trade the layer model makes, for the same reason.

``contours()`` exists so the pane can draw marching ants without the engine
knowing what a frame is: it returns closed polylines in image coordinates, and
the pane walks dashes along them with its own clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ... import native
from . import tiling

# Anything at or above this counts as "in" for hit-tests and for the boundary
# the ants are drawn along. The mask itself stays continuous.
INSIDE = 128

COMBINE_OPS = ("replace", "add", "subtract", "intersect")

# Supersampling factor for the AA of an analytic shape. 4x is the point where a
# 45-degree lasso edge stops looking like a staircase; 8x costs four times as
# much and looks the same.
_SS = 4


def _shape_box(
    size: tuple[int, int], kind: str, geometry: Any
) -> tuple[int, int, int, int] | None:
    """The shape's own pixel-aligned bounding box, clipped to the canvas.

    ``None`` when the shape misses the canvas entirely, or is a polygon with
    too few points to fill. A one-pixel margin is added before clipping so a
    rounding question at the edge can never cost a covered pixel -- the box is
    a bound on the work, not a statement about the geometry.
    """
    width, height = size
    if kind == "polygon":
        points = list(geometry)
        if len(points) < 3:
            return None
        xs = [float(x) for x, _ in points]
        ys = [float(y) for _, y in points]
        lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    else:
        a, b, c, d = (float(v) for v in geometry)
        lo_x, hi_x, lo_y, hi_y = min(a, c), max(a, c), min(b, d), max(b, d)

    x0 = max(0, int(np.floor(lo_x)) - 1)
    y0 = max(0, int(np.floor(lo_y)) - 1)
    x1 = min(width, int(np.ceil(hi_x)) + 1)
    y1 = min(height, int(np.ceil(hi_y)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _draw_shape(
    size: tuple[int, int], kind: str, geometry: Any
) -> np.ndarray:
    """Rasterise a shape into an antialiased mask by supersampling.

    **Supersampled over the shape's own box, not the canvas.** The 4x buffer
    used to be canvas-sized -- 64 MB at 2048 square, allocated, drawn into and
    box-resized in full however small the shape was, so a fifty-pixel lasso
    paid for the whole document. The output is unchanged: an exact 4x BOX
    resize averages each 4x4 block independently, and the box is pixel-aligned,
    so every output pixel sees the same sixteen samples it saw before. Pixels
    outside the box are zero either way -- that is what makes it the box.
    """
    from PIL import Image, ImageDraw

    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    box = _shape_box(size, kind, geometry)
    if box is None:
        return mask
    bx0, by0, bx1, by1 = box

    big = Image.new("L", ((bx1 - bx0) * _SS, (by1 - by0) * _SS), 0)
    draw = ImageDraw.Draw(big)
    dx, dy = bx0 * _SS, by0 * _SS
    if kind == "rect":
        x0, y0, x1, y1 = (v * _SS for v in geometry)
        draw.rectangle((x0 - dx, y0 - dy, x1 - 1 - dx, y1 - 1 - dy), fill=255)
    elif kind == "ellipse":
        x0, y0, x1, y1 = (v * _SS for v in geometry)
        draw.ellipse((x0 - dx, y0 - dy, x1 - 1 - dx, y1 - 1 - dy), fill=255)
    elif kind == "polygon":
        draw.polygon([(x * _SS - dx, y * _SS - dy) for x, y in geometry], fill=255)
    else:  # pragma: no cover - programming error
        raise ValueError(f"unknown selection shape {kind!r}")

    small = big.resize((bx1 - bx0, by1 - by0), Image.BOX)
    mask[by0:by1, bx0:bx1] = np.asarray(small, dtype=np.uint8)
    return mask


@dataclass
class SelectionMask:
    mask: np.ndarray

    def __post_init__(self) -> None:
        if self.mask.dtype != np.uint8 or self.mask.ndim != 2:
            raise ValueError("a selection mask is (H, W) uint8")
        self._bounds: tuple[int, int, int, int] | None | Any = ...

    # -- construction ------------------------------------------------------

    @classmethod
    def full(cls, width: int, height: int) -> SelectionMask:
        return cls(np.full((int(height), int(width)), 255, dtype=np.uint8))

    @classmethod
    def from_rect(
        cls, size: tuple[int, int], rect: tuple[int, int, int, int]
    ) -> SelectionMask:
        # A rectangle is pixel-aligned by construction, so it gets the exact
        # path rather than the supersampled one: a marquee must not come back
        # with half-covered edge pixels the user did not ask for.
        width, height = size
        x0, y0, x1, y1 = rect
        mask = np.zeros((height, width), dtype=np.uint8)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(width, int(x1)), min(height, int(y1))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
        return cls(mask)

    @classmethod
    def from_ellipse(
        cls, size: tuple[int, int], rect: tuple[int, int, int, int]
    ) -> SelectionMask:
        return cls(_draw_shape(size, "ellipse", rect))

    @classmethod
    def from_polygon(
        cls, size: tuple[int, int], points: list[tuple[float, float]]
    ) -> SelectionMask:
        return cls(_draw_shape(size, "polygon", points))

    # -- queries -----------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return (self.mask.shape[1], self.mask.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.bounds is None

    @property
    def bounds(self) -> tuple[int, int, int, int] | None:
        """Half-open bbox of everything non-zero, or None. Cached, because the
        pane asks every frame and a full-canvas ``any`` is not free."""
        if self._bounds is ...:
            rows = np.flatnonzero(self.mask.any(axis=1))
            cols = np.flatnonzero(self.mask.any(axis=0))
            self._bounds = (
                None
                if rows.size == 0
                else (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
            )
        return self._bounds

    def contains(self, xy: tuple[int, int]) -> bool:
        x, y = int(xy[0]), int(xy[1])
        width, height = self.size
        if not (0 <= x < width and 0 <= y < height):
            return False
        return bool(self.mask[y, x] >= INSIDE)

    def copy(self) -> SelectionMask:
        return SelectionMask(self.mask.copy())

    # -- algebra -----------------------------------------------------------

    def combined(self, other: SelectionMask, op: str = "replace") -> SelectionMask:
        if op == "replace":
            return other.copy()
        if op == "add":
            return SelectionMask(np.maximum(self.mask, other.mask))
        if op == "subtract":
            keep = self.mask.astype(np.int16) - other.mask.astype(np.int16)
            return SelectionMask(np.clip(keep, 0, 255).astype(np.uint8))
        if op == "intersect":
            return SelectionMask(np.minimum(self.mask, other.mask))
        raise ValueError(f"unknown combine op {op!r}")

    def translated(self, dx: int, dy: int) -> SelectionMask:
        """The same mask slid by whole pixels, with zeros behind it.

        Zero-filled rather than rolled. A selection is a statement about a
        region of *this* canvas, so what leaves one edge has left -- wrapping it
        round to the other side would put coverage where the user dragged a
        selection away from, which is the one thing moving it is meant to
        avoid. Whole pixels, because a fractional slide would have to resample
        the mask and a hard-edged marquee would go soft the first time it was
        nudged.
        """
        out = np.zeros_like(self.mask)
        height, width = self.mask.shape
        dx, dy = int(dx), int(dy)
        sx, sy = max(0, -dx), max(0, -dy)
        tx, ty = max(0, dx), max(0, dy)
        copy_w = min(width - sx, width - tx)
        copy_h = min(height - sy, height - ty)
        if copy_w > 0 and copy_h > 0:
            out[ty : ty + copy_h, tx : tx + copy_w] = self.mask[
                sy : sy + copy_h, sx : sx + copy_w
            ]
        return SelectionMask(out)

    def inverted(self) -> SelectionMask:
        return SelectionMask((255 - self.mask).astype(np.uint8))

    def feathered(self, radius: float) -> SelectionMask:
        """Gaussian-blur the mask. Feather *is* the intermediate values -- there
        is nothing else to soften, which is why the mask is 8-bit."""
        if radius <= 0:
            return self.copy()
        from PIL import Image, ImageFilter

        blurred = Image.fromarray(self.mask, "L").filter(
            ImageFilter.GaussianBlur(float(radius))
        )
        return SelectionMask(np.asarray(blurred, dtype=np.uint8).copy())

    # -- morphology --------------------------------------------------------
    #
    # Grow, shrink and border, on the 8-bit mask rather than on a threshold of
    # it: a max filter over a soft edge moves the edge and keeps it soft, where
    # thresholding first would harden every antialiased selection the moment
    # somebody nudged it by a pixel.
    #
    # The neighbourhood is an *octagon*, built by alternating a 4-neighbour and
    # an 8-neighbour pass. A square kernel is one line shorter and grows a
    # circle into a rectangle with visibly square corners; a true disc needs a
    # distance transform, which is a scipy dependency this package does not
    # have and does not need for a control whose unit is whole pixels.

    def grown(self, radius: int) -> SelectionMask:
        """Expand the selection by *radius* pixels."""
        return SelectionMask(_morph(self.mask, int(radius), np.maximum))

    def shrunk(self, radius: int) -> SelectionMask:
        """Contract the selection by *radius* pixels."""
        return SelectionMask(_morph(self.mask, int(radius), np.minimum))

    def bordered(self, width: int) -> SelectionMask:
        """The band *width* pixels either side of the current edge.

        Centred on the edge rather than lying inside it, so bordering a
        selection and filling it strokes the line the marching ants are drawn
        on -- which is what somebody asking for a border of a selection means.
        """
        width = max(int(width), 0)
        if width <= 0:
            return SelectionMask(np.zeros_like(self.mask))
        outer = _morph(self.mask, width, np.maximum).astype(np.int16)
        inner = _morph(self.mask, width, np.minimum).astype(np.int16)
        return SelectionMask(np.clip(outer - inner, 0, 255).astype(np.uint8))

    # -- the outline -------------------------------------------------------

    def contours(self) -> list[list[tuple[int, int]]]:
        """Closed polylines around the ``INSIDE`` region, in image coordinates.

        Pixel-edge accurate rather than smoothed: the ants have to sit on the
        boundary the fill actually used, or a one-pixel selection looks like it
        selected its neighbour.

        The loop below is the reference and the fallback; the kernel traces the
        same boundary on the lattice instead of collecting unit segments and
        hashing them back together. What the two agree on is the *set* of unit
        edges -- loop order, starting vertex and winding are unspecified by this
        contract, and at a checkerboard corner the two legitimately partition
        the same edges into different loops.
        """
        if native.available():
            traced = _contours_native(self.mask)
            if traced is not None:
                return traced

        inside = np.pad(self.mask >= INSIDE, 1)
        ys, xs = np.nonzero(inside)
        if ys.size == 0:
            return []

        segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for dx, dy, a, b in (
            (-1, 0, (0, 0), (0, 1)),  # left edge
            (1, 0, (1, 0), (1, 1)),  # right edge
            (0, -1, (0, 0), (1, 0)),  # top edge
            (0, 1, (0, 1), (1, 1)),  # bottom edge
        ):
            # roll by -d, so the rolled plane holds the neighbour at (x+dx, y+dy)
            exposed = inside & ~np.roll(inside, (-dy, -dx), axis=(0, 1))
            for y, x in zip(*np.nonzero(exposed), strict=True):
                px, py = int(x) - 1, int(y) - 1
                segments.append(((px + a[0], py + a[1]), (px + b[0], py + b[1])))
        return _chain(segments)


def _contours_native(mask: np.ndarray) -> list[list[tuple[int, int]]] | None:
    """The traced loops, or None if the kernel could not answer.

    Every buffer is allocated here because the kernel allocates nothing, and
    the sizes are exact rather than guessed: the vertex count *is* the boundary
    edge count, which four vectorised comparisons produce far more cheaply than
    the loop they are replacing, and the shortest possible loop is four edges.
    """
    inside = mask >= INSIDE
    edges = _boundary_edges(inside)
    if edges == 0:
        return []

    height, width = mask.shape
    scratch = np.zeros(width * (height + 1) + (width + 1) * height, dtype=np.uint8)
    points = np.empty(edges * 2, dtype=np.int32)
    lengths = np.empty(edges // 4 + 1, dtype=np.int32)
    found = native.contours(
        np.ascontiguousarray(mask), INSIDE, scratch, points, lengths
    )
    if found < 0:  # pragma: no cover - the capacities above are exact
        return None

    loops: list[list[tuple[int, int]]] = []
    at = 0
    for count in lengths[:found].tolist():
        chunk = points[at * 2 : (at + count) * 2].reshape(count, 2)
        loops.append(
            list(zip(chunk[:, 0].tolist(), chunk[:, 1].tolist(), strict=True))
        )
        at += count
    return loops


def _boundary_edges(inside: np.ndarray) -> int:
    """How many pixel edges separate an inside pixel from an outside one.

    Exactly the number of lattice points the tracer will emit, so it sizes the
    buffer rather than estimating it. A single-row mask counts its top and its
    bottom in the same term, which is why the two borders are added rather than
    special-cased.
    """
    total = int(np.count_nonzero(inside[0])) + int(np.count_nonzero(inside[-1]))
    total += int(np.count_nonzero(inside[:, 0])) + int(np.count_nonzero(inside[:, -1]))
    if inside.shape[0] > 1:
        total += int(np.count_nonzero(inside[1:] != inside[:-1]))
    if inside.shape[1] > 1:
        total += int(np.count_nonzero(inside[:, 1:] != inside[:, :-1]))
    return total


def _chain(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    """Join unit edges end to end into closed loops."""
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in segments:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    unused = {frozenset((a, b)) for a, b in segments if a != b}
    loops: list[list[tuple[int, int]]] = []
    while unused:
        edge = next(iter(unused))
        start, nxt = tuple(edge)
        unused.discard(edge)
        loop = [start, nxt]
        while True:
            here = loop[-1]
            step = None
            for candidate in adjacency.get(here, ()):
                key = frozenset((here, candidate))
                if key in unused:
                    step = candidate
                    unused.discard(key)
                    break
            if step is None:
                break
            if step == loop[0]:
                break
            loop.append(step)
        if len(loop) > 2:
            loops.append(loop)
    return loops


# --- the magic wand ---------------------------------------------------------


def _morph(mask: np.ndarray, radius: int, combine) -> np.ndarray:
    """*radius* passes of an octagonal max (dilate) or min (erode) filter.

    Edge handling is what decides whether growing a selection that touches the
    canvas edge pulls away from it. A dilate pads with the border value (the
    selection continues off-canvas, which is what the user drew); an erode pads
    the same way, so a full-canvas selection shrinks from nothing rather than
    from four sides at once. ``mode="edge"`` gives both, which is why the
    padding is not conditional on the operation.
    """
    if radius <= 0:
        return mask.copy()
    fast = _morph_native(mask, radius, combine)
    if fast is not None:
        return fast
    out = mask
    for step in range(radius):
        out = _spread(out, combine, diagonal=step % 2 == 1)
    return out


# The kernel takes the operation as a number, and MORPH_MAX / MORPH_MIN in
# native/morph.c spell the same two out -- the C header owns the numbers.
# Written out rather than implied by branch order (the composite._MODE_IDS
# rule): a combine that is not in this map falls back to numpy, whereas two
# bare literals in an if/elif were one swapped pair away from handing the
# kernel a number it reads as the *other* operation -- silently, which is the
# one thing a fallback path must never be.
_MORPH_OPS: dict[Any, int] = {
    np.maximum: 0,  # MORPH_MAX -- dilate, what grown() asks for
    np.minimum: 1,  # MORPH_MIN -- erode, what shrunk() asks for
}


def _morph_native(mask: np.ndarray, radius: int, combine) -> np.ndarray | None:
    """The kernel's half of :func:`_morph`, or None to use the numpy body.

    None rather than an exception for every way of being unable to answer, and
    the caller falls through -- the rule every kernel seam in this repository
    follows. ``combine`` is matched by *identity* against the two functions the
    callers actually pass (``_MORPH_OPS``): anything else is somebody's third
    operation, and handing it to a kernel that only knows max and min would
    silently render it as one of them.
    """
    if not native.available():
        return None
    op = _MORPH_OPS.get(combine)
    if op is None:
        return None
    src = np.ascontiguousarray(mask, dtype=np.uint8)
    if src.ndim != 2 or not src.size:
        return None
    out = np.empty(src.shape, dtype=np.uint8)
    scratch = np.empty(src.shape, dtype=np.uint8)
    native.morph_u8(src, scratch, out, int(radius), op)
    return out


def _spread(mask: np.ndarray, combine, *, diagonal: bool) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    height, width = mask.shape
    offsets = [(0, 1), (2, 1), (1, 0), (1, 2)]
    if diagonal:
        offsets += [(0, 0), (0, 2), (2, 0), (2, 2)]
    out = mask.copy()
    for dy, dx in offsets:
        out = combine(out, padded[dy : dy + height, dx : dx + width])
    return out


def colour_distance(pixels: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """Chebyshev distance over RGBA, 0..255.

    Chebyshev rather than Euclidean because it is what a tolerance slider
    *means* to a user -- "no channel differs by more than this" -- and because
    the flood fill and the wand must agree, so there is exactly one predicate.
    """
    diff = np.abs(pixels.astype(np.int16) - colour.astype(np.int16))
    return diff.max(axis=2).astype(np.uint8)


def similar(pixels: np.ndarray, seed: tuple[int, int], tolerance: int) -> np.ndarray:
    """Boolean plane of everything within ``tolerance`` of the seed pixel."""
    x, y = int(seed[0]), int(seed[1])
    return colour_distance(pixels, pixels[y, x]) <= int(tolerance)


def colour_range(
    pixels: np.ndarray, colour: tuple[int, int, int, int], tolerance: int = 32
) -> SelectionMask:
    """Every pixel within ``tolerance`` of ``colour``, anywhere on the canvas.

    Colour-first where the wand is seed-first: you pick the colour and press
    the button, rather than having to find a pixel of it to click. Never
    contiguous -- selecting one palette entry everywhere is the whole gesture.

    Over :func:`colour_distance` rather than beside it, so this and the wand
    and the flood fill remain one predicate and a tolerance means the same
    thing wherever the user reads it.
    """
    plane = colour_distance(pixels, np.asarray(colour, dtype=np.uint8))
    return SelectionMask((plane <= int(tolerance)).astype(np.uint8) * 255)


def magic_wand(
    pixels: np.ndarray,
    seed: tuple[int, int],
    *,
    tolerance: int = 32,
    contiguous: bool = True,
    wrap: str | tuple[bool, bool] = "off",
) -> SelectionMask:
    """Everything similar to the seed pixel, optionally on a torus.

    ``wrap`` is one of :data:`.tiling.TILED_AXES` and only affects
    *contiguity*: similarity is per pixel and has no notion of an edge, so a
    non-contiguous wand is the same answer tiled or not. With it on, a region
    that runs off one edge and continues on the other is one region -- which is
    what makes filling a tile's background a single click rather than four.
    """
    height, width = pixels.shape[:2]
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < width and 0 <= y < height):
        return SelectionMask(np.zeros((height, width), dtype=np.uint8))
    plane = similar(pixels, (x, y), tolerance)
    if contiguous:
        plane = _contiguous(plane, (x, y), tiling.axes_of(wrap))
    return SelectionMask((plane.astype(np.uint8)) * 255)


def _contiguous(
    plane: np.ndarray, seed: tuple[int, int], axes: tuple[bool, bool] = (False, False)
) -> np.ndarray:
    """Keep only the region connected to the seed.

    Pillow's flood fill does the walk: it is C, and the alternative is either a
    Python BFS over a megapixel or a scipy dependency this app does not have.

    Toroidal contiguity is that same C flood run to a **fixpoint**: fill, ask
    :func:`.tiling.seam_seeds` which pixels on a wrapped edge the region has
    just made reachable from the opposite edge, fill from those, repeat. Two
    floods is the usual answer and the loop is bounded by the number of edge
    pixels, since every extra pass has to claim at least one new seed. Doing it
    this way rather than by padding the plane keeps the one predicate -- Pillow
    decides connectivity, here as before -- and costs no copy of the canvas.
    """
    from PIL import Image, ImageDraw

    # 0 outside, 1 inside, 2 reached. Filling for 2 from the seed cannot leave
    # the candidate region because everything else is a different value.
    # ``.copy()`` is load-bearing: an image built by ``fromarray`` wraps the
    # array's buffer read-only, and the fill then writes nothing at all --
    # silently, with no exception and an empty selection to show for it.
    canvas = Image.fromarray(plane.astype(np.uint8), "L").copy()
    ImageDraw.floodfill(canvas, seed, 2, thresh=0)
    if not (axes[0] or axes[1]):
        return np.asarray(canvas) == 2
    while True:
        reached = np.asarray(canvas) == 2
        seeds = tiling.seam_seeds(reached, plane, axes)
        if not seeds:
            return reached
        for point in seeds:
            ImageDraw.floodfill(canvas, point, 2, thresh=0)


# --- floating pixels --------------------------------------------------------


def render_transform(
    source: np.ndarray,
    mask: np.ndarray,
    angle: float,
    scale: tuple[float, float],
    shear: tuple[float, float],
    resample: str,
) -> tuple[np.ndarray, np.ndarray]:
    """One scale-then-shear-then-rotate render. Pure, and the only copy of it.

    **Scale, then shear, then rotate**, and the order is part of the contract
    because these do not commute -- shearing a turned rectangle and turning a
    sheared one are different pictures, and the numbers in the panel have to
    mean one of them. This order is the one that reads the way the fields are
    written: the scale is a statement about the lifted pixels, the shear slants
    that, and the rotation turns the whole result. It is also the order that
    keeps each field independent, since a rotate applied first would make the
    shear axes something other than the page's.

    ``resample`` reaches the *mask* as well as the pixels, and that is not an
    oversight to tidy up later: a nearest-neighbour rotation whose mask was
    filtered would keep a one-pixel band of partial coverage all the way round,
    so the hard edge the setting exists to preserve would survive in the
    colours and be thrown away at the composite.

    A module function rather than a method, because it has **two** callers that
    must not drift: the buffer's own live render, and the per-cel replay a
    ranged commit runs (``Document._replay_transform_on``). Two spellings of
    this would mean a transform that previewed one way and landed another.
    """
    from . import transform as tf

    height, width = source.shape[:2]
    target = (max(1, round(width * scale[0])), max(1, round(height * scale[1])))
    if target != (width, height):
        source = tf.scale(source, target, resample=resample)
        mask = tf.scale(mask, target, resample=resample)
    if shear != (0.0, 0.0):
        source = tf.shear(source, shear, resample=resample)
        mask = tf.shear(mask, shear, resample=resample)
    if abs(angle) > 1e-6:
        source = tf.rotate(source, angle, expand=True, resample=resample)
        mask = tf.rotate(mask, angle, expand=True, resample=resample)
    return source, mask


@dataclass
class FloatingBuffer:
    """Pixels lifted off a layer and hovering over it.

    The hole they left is already cut. Committing writes them back wherever
    they now are; cancelling is an undo of the lift, which is why the lift is a
    single edit rather than two.

    ``rev`` is here for the pane's texture cache, the same protocol the
    document uses -- moving a buffer changes where it draws but not what it
    contains, so a move must *not* bump it.
    """

    pixels: np.ndarray
    mask: np.ndarray
    offset: tuple[int, int]
    layer_uid: int
    rev: int = 0

    # The history entry cancelling this buffer has to reverse, or None. A lift
    # cut a hole and pushed one; a paste took pixels from the clipboard and
    # touched no layer, so there is nothing of its own to undo -- and undoing
    # anyway reverses whatever the user did *before* the paste.
    #
    # The entry itself rather than a flag, because a buffer floats across an
    # unbounded number of frames and selection ops push steps of their own
    # meanwhile: "the newest step" stops being this one, and reversing the
    # newest then destroys both the lifted pixels and an unrelated edit.
    lift_edit: Any = None

    # What a transform re-renders *from*. Kept so that dragging a scale handle
    # back and forth is not a chain of resamples: every adjustment starts again
    # from the pixels that were lifted, so only the final one is ever applied.
    source: np.ndarray | None = None
    source_mask: np.ndarray | None = None
    angle: float = 0.0
    scale: tuple[float, float] = (1.0, 1.0)
    #: Slant, in degrees per axis. Trailing and defaulted like the two above,
    #: and re-rendered from ``source`` with them rather than compounded -- a
    #: shear dragged back and forth is one shear, not a chain of them.
    shear: tuple[float, float] = (0.0, 0.0)

    # --- what a *ranged* commit replays -------------------------------------
    #
    # A transform is a gesture, and until Wave 2 the buffer only had to hold
    # its *result*. Committing it over a timeline range means running the same
    # gesture against a cel whose pixels the buffer has never seen, so the
    # three things below are what turn the result back into the recipe.

    #: The canvas rectangle a lift cut from, or None for a paste. It is what
    #: "the same gesture on another cel" needs and the only thing a paste
    #: cannot supply -- there is no source to re-cut -- so it doubles as the
    #: test for whether a ranged commit means anything at all.
    source_box: tuple[int, int, int, int] | None = None
    #: Flips, in the order they were asked for. A list rather than a pair of
    #: booleans because a flip is applied to the *source* and the replay has to
    #: run them before the affine, in sequence: flipping a turned rectangle and
    #: turning a flipped one are different pictures.
    flips: list[str] = field(default_factory=list)
    #: The filter the last render used. Recorded because ``flip`` re-renders,
    #: and re-rendering with the default silently threw a nearest-neighbour
    #: setting away mid-gesture -- a pixel-art transform that had been exact
    #: came back Lanczos-blurred the moment the user pressed the flip button.
    resample: str = "smooth"

    @property
    def lifted(self) -> bool:
        """Whether these pixels came off a layer rather than off the clipboard."""
        return self.lift_edit is not None

    @property
    def size(self) -> tuple[int, int]:
        return (self.pixels.shape[1], self.pixels.shape[0])

    @property
    def centre(self) -> tuple[float, float]:
        width, height = self.size
        return (self.offset[0] + width / 2.0, self.offset[1] + height / 2.0)

    @property
    def transformed(self) -> bool:
        return (
            abs(self.angle) > 1e-6
            or self.scale != (1.0, 1.0)
            or self.shear != (0.0, 0.0)
        )

    @property
    def base_size(self) -> tuple[int, int]:
        """The lifted pixels' own size -- what ``scale`` is a factor *of*.

        The numeric width and height fields need this: ``size`` is the size
        after the transform, so deriving a scale factor from it would compound
        every keystroke against the result of the last one.
        """
        source = self.pixels if self.source is None else self.source
        return (source.shape[1], source.shape[0])

    def transform(
        self,
        *,
        angle: float | None = None,
        scale: tuple[float, float] | None = None,
        shear: tuple[float, float] | None = None,
        resample: str = "smooth",
    ):
        """Re-render from the lifted pixels at a new angle, scale and slant.

        The centre is held fixed rather than the top-left: rotating about a
        corner sends the subject off across the canvas, which is not what
        grabbing a rotate handle means.

        The render itself is :func:`render_transform`, which owns the order the
        three are applied in and why. What stays here is the *state*: which
        pixels to render from, where to put the result, and -- since Wave 2 --
        what the recipe was, so a ranged commit can run it again on a cel this
        buffer has never held.
        """
        if self.source is None:
            self.source = self.pixels.copy()
            self.source_mask = self.mask.copy()
        if angle is not None:
            self.angle = float(angle)
        if scale is not None:
            self.scale = (max(0.01, float(scale[0])), max(0.01, float(scale[1])))
        if shear is not None:
            self.shear = (float(shear[0]), float(shear[1]))
        self.resample = resample

        cx, cy = self.centre
        pixels, mask = render_transform(
            self.source, self.source_mask, self.angle, self.scale, self.shear, resample
        )

        self.pixels, self.mask = pixels, mask
        new_h, new_w = pixels.shape[:2]
        self.offset = (round(cx - new_w / 2.0), round(cy - new_h / 2.0))
        # The pixels genuinely changed, so the texture has to be re-uploaded --
        # unlike a move, which changes only where they are drawn.
        self.rev += 1

    def flip(self, axis: str) -> None:
        """Flip in place. Applied to the *source* as well, so a later rotate
        does not undo it by re-rendering from the unflipped pixels.

        Recorded in ``flips`` in the order asked for, and re-rendered with the
        buffer's own ``resample`` rather than the default: a flip in the middle
        of a nearest-neighbour gesture used to re-render the whole thing
        smooth, which turned a pixel-art transform into a blurred one at the
        press of a button.
        """
        from . import transform as tf

        if self.source is None:
            self.source = self.pixels.copy()
            self.source_mask = self.mask.copy()
        self.source = tf.flip(self.source, axis)
        self.source_mask = tf.flip(self.source_mask, axis)
        self.flips.append(axis)
        self.transform(resample=self.resample)

    def moved(self, dx: int, dy: int) -> None:
        self.offset = (self.offset[0] + int(dx), self.offset[1] + int(dy))

    def contains(self, xy: tuple[int, int]) -> bool:
        x, y = int(xy[0]), int(xy[1])
        ox, oy = self.offset
        width, height = self.size
        if not (ox <= x < ox + width and oy <= y < oy + height):
            return False
        return bool(self.mask[y - oy, x - ox] >= INSIDE)


@dataclass
class Clipboard:
    """The app's own clipboard. Not the OS one: a paste has to bring its mask
    with it, and no platform clipboard format carries one."""

    pixels: np.ndarray | None = None
    mask: np.ndarray | None = None

    @property
    def empty(self) -> bool:
        return self.pixels is None

    def put(self, pixels: np.ndarray, mask: np.ndarray) -> None:
        self.pixels = pixels.copy()
        self.mask = mask.copy()

    def take(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.pixels is None or self.mask is None:
            return None
        return self.pixels.copy(), self.mask.copy()
