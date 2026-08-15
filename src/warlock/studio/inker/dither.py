"""Palette conversion: nearest, error diffusion, and ordered thresholds.

What this is for is the pixel-art workflow's one genuinely destructive step --
taking a full-colour drawing (a photo, an SDXL reference, a smudged sketch) down
onto a table of a dozen colours -- and the whole argument for having three
methods is that no one of them is right twice. Nearest bands. Floyd-Steinberg
scatters the error and keeps the tone, at the cost of a noise floor that is
death on a 4x-zoomed sprite. An ordered matrix keeps the noise *regular*, which
at pixel-art sizes reads as texture rather than as dirt, and is what every
palette-first editor since Deluxe Paint has offered.

**This is deliberately not** ``pipelines/pixel.py::map_palette``, **and the
duplication is the decision.** That one is a *restyle of an export*: it works in
Oklab, on a PIL image, with an offset scaled by the palette's own mean
nearest-neighbour spacing, because the question it answers is "make this render
look like pixel art". This one is a *document conversion*: it works in straight
RGB, on the engine's ``(H, W, 4)`` uint8 planes, because the question it answers
is "put my drawing on exactly these swatches" -- and the answer has to agree,
pixel for pixel, with the snap that ``document._commit_patch`` performs on every
subsequent write. A conversion in Oklab followed by a lifetime of writes snapped
in RGB is a document that moves the moment you touch it. The engine also cannot
import ``pipelines``: this package is headless and pinned to stay that way, and
``map_palette`` reaches for PIL and the Oklab tables at module scope.

Alpha is never touched and fully transparent pixels are never read, for the
reasons ``indexed`` gives at length: a palette is a set of colours, not a set of
opacities, and a transparent pixel has no colour to convert.

Pure numpy and stdlib, like the rest of this package.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from . import indexed as ix

__all__ = [
    "BAYER_SIZES",
    "METHODS",
    "ORDERED",
    "bayer_matrix",
    "build_palette",
    "convert",
    "luma",
    "tile_matrix",
]

RGBA = tuple[int, int, int, int]

#: The ordered-dither matrices, by method name. Powers of two only, because the
#: recursive construction below is the definition of a Bayer matrix and it
#: doubles: 2, 4, 8 is the whole useful range at sprite sizes, and 16 is a
#: gradient of noise nobody can see the pattern of.
BAYER_SIZES = {"bayer2": 2, "bayer4": 4, "bayer8": 8}

#: Just the ordered methods, in size order -- what a UI offers when the choice
#: is *which* matrix rather than which family. The gradient tool's dither
#: option is exactly this list plus "none".
ORDERED = tuple(BAYER_SIZES)

#: Every conversion this module performs, in the order a UI should list them:
#: the two that need no matrix first, then the matrices by size.
METHODS = ("nearest", "floyd-steinberg", *ORDERED)

# Rec. 709 luma. Used for the deterministic output ordering of ``build_palette``
# and by the palette sort, so "sorted by brightness" means one thing in the
# whole app rather than one thing per caller.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# The Floyd-Steinberg kernel, as fractions of sixteen: the pixel ahead takes 7,
# and the row below takes 3 / 5 / 1 across the three cells under and beside it.
_FS = (7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0)


def luma(colour: Sequence[int]) -> float:
    """Rec. 709 brightness of one RGB(A) colour, 0..255."""
    return float(np.dot(_LUMA, np.asarray(tuple(colour)[:3], dtype=np.float64)))


def bayer_matrix(n: int) -> np.ndarray:
    """The ``n x n`` ordered-dither threshold matrix, float32 in ``(0, 1)``.

    Built by the recurrence that defines it -- ``M(2n)`` is four copies of
    ``M(n)`` scaled by four and offset by ``0, 2, 3, 1`` -- rather than by a
    written-out table, so there is one thing to be right about and no
    transcription to check.

    Thresholds are ``(index + 0.5) / n^2`` rather than ``index / n^2``: centring
    them in their cells is what makes a 50% mix come out as an even chequer, and
    it keeps every threshold strictly inside the open interval, so a parameter
    of exactly 0 always picks the low candidate and one of exactly 1 always
    picks the high one. Without that, a flat region at either end of a ramp
    picks up a sprinkle of the wrong colour.

    Public because two callers share it and must agree: :func:`convert`
    thresholds between two *palette* candidates, and ``gradient.render``
    thresholds between two adjacent *stops*. The candidates differ; the matrix
    is the same object, so a gradient dithered at ``bayer4`` interlocks with a
    document converted at ``bayer4`` instead of beating against it.
    """
    if n < 1 or n & (n - 1):
        raise ValueError(f"a Bayer matrix is a power of two, not {n}")
    matrix = np.zeros((1, 1), dtype=np.float64)
    size = 1
    while size < n:
        matrix = np.block(
            [
                [4.0 * matrix, 4.0 * matrix + 2.0],
                [4.0 * matrix + 3.0, 4.0 * matrix + 1.0],
            ]
        )
        size *= 2
    return ((matrix + 0.5) / float(n * n)).astype(np.float32)


def tile_matrix(matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """*matrix* repeated to cover ``(height, width)``, anchored at the origin.

    Anchored at the canvas origin rather than at the region being written, so a
    ramp drawn in two halves lines up with itself: an ordered dither whose phase
    followed the dirty rectangle would show its seam.
    """
    height, width = shape
    reps = (-(-height // matrix.shape[0]), -(-width // matrix.shape[1]))
    return np.tile(matrix, reps)[:height, :width]


def convert(pixels: np.ndarray, palette: Sequence[RGBA], method: str = "nearest") -> np.ndarray:
    """*pixels* rewritten onto *palette* by *method*. Returns a new array.

    Every output colour is an entry of *palette*, exactly -- that is what makes
    the result stable under the snap every later write goes through. Alpha rides
    through byte-identical and fully transparent pixels are returned verbatim.
    """
    if method not in METHODS:
        raise ValueError(f"unknown dither method {method!r}")
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("convert takes (H, W, 4) uint8")
    if not palette:
        raise ValueError("a conversion needs at least one colour")
    if method == "nearest":
        # Delegated rather than reimplemented: nearest *is* the snap, and a
        # second copy of it is how the conversion and the per-write constraint
        # come to disagree about one pixel on the boundary between two swatches.
        return ix.snap(pixels, palette)
    out = pixels.copy()
    if out.size == 0 or not (out[..., 3] > 0).any():
        return out
    table = np.asarray([tuple(c)[:3] for c in palette], dtype=np.int16)
    if method == "floyd-steinberg":
        return _floyd_steinberg(out, table)
    return _ordered(out, table, bayer_matrix(BAYER_SIZES[method]))


def _ordered(out: np.ndarray, table: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Two-candidate ordered dithering, over the region's distinct colours.

    For each colour in the image: the nearest entry ``p1``, then a partner
    ``p2`` chosen as the entry nearest to ``2c - p1`` -- the reflection of the
    colour through its own nearest swatch, which is the cheapest way to ask for
    an entry on the *other side* rather than for the second-nearest one (the
    second-nearest is routinely a third colour crowding the same side, and
    mixing those two brackets nothing). The mix is then where ``c`` falls on the
    segment between them, and the matrix decides which of the two each pixel
    gets.

    Vectorised over distinct colours through ``indexed.snap``'s packed-uint32
    ``np.unique`` idiom, so the cost is set by the palette and the number of
    distinct colours rather than by the pixel count.
    """
    visible = out[..., 3] > 0
    rgb = out[..., :3][visible]
    packed = (
        rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
    )
    keys, inverse = np.unique(packed, return_inverse=True)
    colours = np.stack(
        [(keys >> 16) & 0xFF, (keys >> 8) & 0xFF, keys & 0xFF], axis=1
    ).astype(np.int32)

    entries = table.astype(np.int32)
    first = np.argmin(
        ((colours[:, None, :] - entries[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    low = entries[first]
    # The reflection, clamped into the cube: an unclamped one pulls the search
    # towards a corner nothing is near and picks the same entry back.
    mirrored = np.clip(2 * colours - low, 0, 255)
    second = np.argmin(
        ((mirrored[:, None, :] - entries[None, :, :]) ** 2).sum(axis=2), axis=1
    )
    high = entries[second]

    span = (high - low).astype(np.float64)
    length = (span * span).sum(axis=1)
    along = ((colours - low).astype(np.float64) * span).sum(axis=1)
    mix = np.divide(along, length, out=np.zeros_like(along), where=length > 0.0)
    mix = np.clip(mix, 0.0, 1.0)

    threshold = tile_matrix(matrix, out.shape[:2])[visible].astype(np.float64)
    picked = np.where(mix[inverse] > threshold, second[inverse], first[inverse])
    out[..., :3][visible] = table[picked].astype(np.uint8)
    return out


def _floyd_steinberg(out: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Serpentine error diffusion in straight float32 RGB.

    A Python loop, and that is a decision rather than a gap: error diffusion is
    sequential by definition -- every pixel's input depends on its neighbours'
    already-quantised output -- so there is no vectorisation of it that is still
    Floyd-Steinberg. The preview above this memoises per (palette, method) for
    exactly this reason, the same way the blur filter does.

    Serpentine (alternate rows right-to-left) rather than raster order: a
    one-directional scan piles its error up towards one edge and leaves a visible
    lean on any large flat area, which is the classic artefact of the naive
    implementation.

    Transparent pixels are neither quantised nor used as error sinks; their dead
    RGB is not a colour and diffusing into it would be diffusing into nothing.
    """
    height, width = out.shape[:2]
    work = out[..., :3].astype(np.float32)
    visible = out[..., 3] > 0
    entries = table.astype(np.float32)

    for y in range(height):
        rightwards = y % 2 == 0
        columns = range(width) if rightwards else range(width - 1, -1, -1)
        step = 1 if rightwards else -1
        for x in columns:
            if not visible[y, x]:
                continue
            # Copied, not viewed: ``work[y, x]`` is a view into the plane, so
            # without this the assignment two lines down rewrites ``old`` as
            # well and every error comes out zero -- which is nearest wearing
            # this function's name, and passes every subset-of-the-palette
            # assertion while doing so.
            old = work[y, x].copy()
            delta = entries - old
            pick = int(np.argmin((delta * delta).sum(axis=1)))
            new = entries[pick]
            work[y, x] = new
            error = old - new
            ahead, below_back, below, below_ahead = _FS
            if 0 <= x + step < width:
                work[y, x + step] += error * ahead
            if y + 1 < height:
                if 0 <= x - step < width:
                    work[y + 1, x - step] += error * below_back
                work[y + 1, x] += error * below
                if 0 <= x + step < width:
                    work[y + 1, x + step] += error * below_ahead

    # Read back off ``work`` rather than recorded as it goes: every written
    # value is an exact palette entry, so this rounds nothing -- it is the one
    # copy, and clipping guards the diffused values that never became output.
    out[..., :3][visible] = np.clip(work, 0.0, 255.0)[visible].astype(np.uint8)
    return out


def build_palette(planes: Iterable[np.ndarray], max_colours: int = 32) -> list[RGBA]:
    """A table of at most *max_colours* drawn from the document's own pixels.

    Count-weighted median cut. Exact when the drawing already has few enough
    distinct colours -- which is the common case here, because the input is
    usually already pixel art and inventing an approximation of a palette the
    user hand-authored would be the worst possible answer.

    Fully deterministic: the boxes are chosen and split on written-down
    tie-breaks (widest extent, then pixel count, then the box's lowest packed
    colour), and the result is sorted by luma and then by packed RGB. The same
    document converts the same way twice, which is what makes a conversion
    something a test can pin at all.
    """
    if max_colours < 1:
        raise ValueError("a palette needs at least one colour")
    keys, counts = _distinct(planes)
    if keys.size == 0:
        # Nothing visible anywhere. A palette has to have an entry, and black is
        # the one colour that says "this document had none" without inventing a
        # hue the drawing never contained.
        return [(0, 0, 0, 255)]
    colours = np.stack(
        [(keys >> 16) & 0xFF, (keys >> 8) & 0xFF, keys & 0xFF], axis=1
    ).astype(np.int64)
    if keys.size <= max_colours:
        return _ordered_table(colours)
    return _ordered_table(_median_cut(colours, counts, max_colours))


def _distinct(planes: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Every distinct visible RGB across *planes*, packed, with pixel counts."""
    packed: list[np.ndarray] = []
    for plane in planes:
        if plane.dtype != np.uint8 or plane.ndim != 3 or plane.shape[2] != 4:
            raise ValueError("build_palette takes (H, W, 4) uint8 planes")
        if plane.size == 0:
            continue
        visible = plane[..., 3] > 0
        if not visible.any():
            continue
        rgb = plane[..., :3][visible]
        packed.append(
            rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
        )
    if not packed:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.int64)
    keys, counts = np.unique(np.concatenate(packed), return_counts=True)
    return keys, counts.astype(np.int64)


def _median_cut(colours: np.ndarray, counts: np.ndarray, want: int) -> np.ndarray:
    """Repeatedly split the widest box at its count-weighted median.

    Weighted by pixel count and not by distinct-colour count, which is the whole
    difference between a palette of the picture and a palette of its noise: a
    thousand near-identical antialiasing colours occupying two hundred pixels
    must not out-vote the flat fill behind them.
    """
    boxes = [np.arange(colours.shape[0])]
    while len(boxes) < want:
        pick = -1
        best: tuple[int, int, int] | None = None
        for index, box in enumerate(boxes):
            if box.size < 2:
                continue
            values = colours[box]
            extent = int((values.max(axis=0) - values.min(axis=0)).max())
            if extent == 0:
                continue
            rank = (extent, int(counts[box].sum()), -int(box.min()))
            if best is None or rank > best:
                best, pick = rank, index
        if pick < 0:
            break  # every box is one colour: there is nothing left to split
        box = boxes.pop(pick)
        values = colours[box]
        channel = int(np.argmax(values.max(axis=0) - values.min(axis=0)))
        # Sorted on (channel, packed) so equal values on the split channel keep
        # one fixed order, and ``kind="stable"`` so the sort itself adds none.
        order = np.lexsort((box, values[:, channel]))
        box = box[order]
        weights = counts[box]
        half = weights.sum() / 2.0
        cut = int(np.searchsorted(np.cumsum(weights), half, side="left") + 1)
        cut = max(1, min(cut, box.size - 1))
        boxes.append(box[:cut])
        boxes.append(box[cut:])

    out = []
    for box in boxes:
        weights = counts[box].astype(np.float64)
        mean = (colours[box].astype(np.float64) * weights[:, None]).sum(axis=0)
        out.append(np.floor(mean / weights.sum() + 0.5))
    return np.clip(np.asarray(out, dtype=np.int64), 0, 255)


def _ordered_table(colours: np.ndarray) -> list[RGBA]:
    """Distinct colours sorted by luma, then by packed RGB, as opaque RGBA.

    Deduplicated *after* the median cut as well as before it: two boxes can land
    on the same representative, and a table with the same swatch in it twice is
    a slot the user cannot tell from its neighbour and a wasted entry in the GIF
    colour table.
    """
    seen: dict[tuple[int, int, int], None] = {}
    for r, g, b in colours.tolist():
        seen.setdefault((int(r), int(g), int(b)), None)
    ordered = sorted(seen, key=lambda c: (luma(c), c[0] << 16 | c[1] << 8 | c[2]))
    return [(r, g, b, 255) for r, g, b in ordered]
