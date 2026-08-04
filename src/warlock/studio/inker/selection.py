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

from dataclasses import dataclass
from typing import Any

import numpy as np

# Anything at or above this counts as "in" for hit-tests and for the boundary
# the ants are drawn along. The mask itself stays continuous.
INSIDE = 128

COMBINE_OPS = ("replace", "add", "subtract", "intersect")

# Supersampling factor for the AA of an analytic shape. 4x is the point where a
# 45-degree lasso edge stops looking like a staircase; 8x costs four times as
# much and looks the same.
_SS = 4


def _draw_shape(
    size: tuple[int, int], kind: str, geometry: Any
) -> np.ndarray:
    """Rasterise a shape into an antialiased mask by supersampling."""
    from PIL import Image, ImageDraw

    width, height = size
    big = Image.new("L", (width * _SS, height * _SS), 0)
    draw = ImageDraw.Draw(big)
    if kind == "rect":
        x0, y0, x1, y1 = (v * _SS for v in geometry)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    elif kind == "ellipse":
        x0, y0, x1, y1 = (v * _SS for v in geometry)
        draw.ellipse((x0, y0, x1 - 1, y1 - 1), fill=255)
    elif kind == "polygon":
        points = [(x * _SS, y * _SS) for x, y in geometry]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    else:  # pragma: no cover - programming error
        raise ValueError(f"unknown selection shape {kind!r}")
    return np.asarray(big.resize((width, height), Image.BOX), dtype=np.uint8)


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

    # -- the outline -------------------------------------------------------

    def contours(self) -> list[list[tuple[int, int]]]:
        """Closed polylines around the ``INSIDE`` region, in image coordinates.

        Pixel-edge accurate rather than smoothed: the ants have to sit on the
        boundary the fill actually used, or a one-pixel selection looks like it
        selected its neighbour.
        """
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


def magic_wand(
    pixels: np.ndarray,
    seed: tuple[int, int],
    *,
    tolerance: int = 32,
    contiguous: bool = True,
) -> SelectionMask:
    height, width = pixels.shape[:2]
    x, y = int(seed[0]), int(seed[1])
    if not (0 <= x < width and 0 <= y < height):
        return SelectionMask(np.zeros((height, width), dtype=np.uint8))
    plane = similar(pixels, (x, y), tolerance)
    if contiguous:
        plane = _contiguous(plane, (x, y))
    return SelectionMask((plane.astype(np.uint8)) * 255)


def _contiguous(plane: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """Keep only the region connected to the seed.

    Pillow's flood fill does the walk: it is C, and the alternative is either a
    Python BFS over a megapixel or a scipy dependency this app does not have.
    """
    from PIL import Image, ImageDraw

    # 0 outside, 1 inside, 2 reached. Filling for 2 from the seed cannot leave
    # the candidate region because everything else is a different value.
    # ``.copy()`` is load-bearing: an image built by ``fromarray`` wraps the
    # array's buffer read-only, and the fill then writes nothing at all --
    # silently, with no exception and an empty selection to show for it.
    canvas = Image.fromarray(plane.astype(np.uint8), "L").copy()
    ImageDraw.floodfill(canvas, seed, 2, thresh=0)
    return np.asarray(canvas) == 2


# --- floating pixels --------------------------------------------------------


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
        return abs(self.angle) > 1e-6 or self.scale != (1.0, 1.0)

    def transform(self, *, angle: float | None = None, scale: tuple[float, float] | None = None):
        """Re-render from the lifted pixels at a new angle and scale.

        The centre is held fixed rather than the top-left: rotating about a
        corner sends the subject off across the canvas, which is not what
        grabbing a rotate handle means.
        """
        from . import transform as tf

        if self.source is None:
            self.source = self.pixels.copy()
            self.source_mask = self.mask.copy()
        if angle is not None:
            self.angle = float(angle)
        if scale is not None:
            self.scale = (max(0.01, float(scale[0])), max(0.01, float(scale[1])))

        cx, cy = self.centre
        pixels, mask = self.source, self.source_mask
        height, width = pixels.shape[:2]
        target = (max(1, round(width * self.scale[0])), max(1, round(height * self.scale[1])))
        if target != (width, height):
            pixels = tf.scale(pixels, target)
            mask = tf.scale(mask, target)
        if abs(self.angle) > 1e-6:
            pixels = tf.rotate(pixels, self.angle, expand=True)
            mask = tf.rotate(mask, self.angle, expand=True)

        self.pixels, self.mask = pixels, mask
        new_h, new_w = pixels.shape[:2]
        self.offset = (round(cx - new_w / 2.0), round(cy - new_h / 2.0))
        # The pixels genuinely changed, so the texture has to be re-uploaded --
        # unlike a move, which changes only where they are drawn.
        self.rev += 1

    def flip(self, axis: str) -> None:
        """Flip in place. Applied to the *source* as well, so a later rotate
        does not undo it by re-rendering from the unflipped pixels."""
        from . import transform as tf

        if self.source is None:
            self.source = self.pixels.copy()
            self.source_mask = self.mask.copy()
        self.source = tf.flip(self.source, axis)
        self.source_mask = tf.flip(self.source_mask, axis)
        self.transform()

    def moved(self, dx: int, dy: int) -> None:
        self.offset = (self.offset[0] + int(dx), self.offset[1] + int(dy))

    # Compatibility with the flat editor's ``Selection``: the old pane says
    # ``origin`` and hands ``chunk`` straight to a GL upload. Both go away with
    # the pane; neither is worth a second representation until then.

    @property
    def origin(self) -> tuple[int, int]:
        return self.offset

    @property
    def chunk(self) -> Any:
        from PIL import Image

        return Image.fromarray(self.pixels, "RGBA")

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
