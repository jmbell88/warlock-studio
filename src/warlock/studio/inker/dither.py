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

**Grouped** is the fourth, and it is nearest's answer to nearest's own failure.
Nearest maps each colour to the absolutely-closest swatch, so a drawing with
fifteen mid-greys and a palette with two can land all fifteen on one of them and
lose every distinction the artist drew. Grouped instead clusters the drawing's
*own* distinct colours -- count-weighted, by the same median cut
:func:`build_palette` uses -- into at most as many groups as the palette has
entries, and then assigns groups to entries **one to one**, cheapest total
first. Dark greys land on the dark target and light greys on the light one
because that is what a min-distance injective assignment does; there is no hue
special-casing anywhere, and none is wanted. It is a flat per-colour map like
nearest, with no dither texture, so flat regions stay flat.

The metric is squared-Euclidean integer RGB throughout -- clustering and
assignment both. Oklab was considered and rejected *for now*: it would perceptually
improve the clustering, but this module's one-metric identity with the write-path
snap is worth more than that, exact integer arithmetic makes the tie-breaks
decidable rather than float-fragile, and the quality win here comes from the
clustering and the one-to-one assignment rather than from the space they happen in.

Alpha is never touched and fully transparent pixels are never read, for the
reasons ``indexed`` gives at length: a palette is a set of colours, not a set of
opacities, and a transparent pixel has no colour to convert.

Pure numpy and stdlib, like the rest of this package -- ``warlock.native`` is
the one import beyond that, and it is stdlib-only itself and returns ``None``
rather than raising when the DLL is absent, exactly as ``composite`` uses it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from ... import native
from . import index_plane as ixp
from . import indexed as ix

__all__ = [
    "BAYER_SIZES",
    "METHODS",
    "ORDERED",
    "bayer_matrix",
    "build_palette",
    "convert",
    "convert_indices",
    "grouped_index_table",
    "grouped_table",
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
#: the flat per-colour maps first, then error diffusion, then the matrices by
#: size. Only ``METHODS[0] == "nearest"`` is pinned elsewhere (the UI's default);
#: "grouped" sits beside it because the two answer the same question -- one
#: colour in, one colour out, no texture -- and differ only in how they choose.
METHODS = ("nearest", "grouped", "floyd-steinberg", *ORDERED)

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


def grouped_table(
    planes: Iterable[np.ndarray], palette: Sequence[RGBA]
) -> tuple[np.ndarray, np.ndarray]:
    """The grouped map for a whole conversion scope: ``(keys, targets)``.

    ``keys`` is the sorted packed-uint32 distinct visible RGBs across *planes*;
    ``targets`` is the exact palette RGB each of them lands on, ``(N, 3) uint8``.
    A table rather than a per-pixel function because the grouping is a property
    of the *scope*, not of a plane: two layers with disjoint grey ranges must be
    grouped against each other or the same grey ends up on different swatches in
    each, and the document comes apart along its layer boundaries. Building it
    once and applying it to every plane is what makes that impossible.

    The steps, and why each is the one it is:

    1. Distinct visible colours with pixel counts -- the same reduction
       :func:`build_palette` runs, so "the drawing's colours" means one thing.
    2. Clustering. With no more distinct colours than palette entries, every
       colour is its own group: that is the structure-preserving degenerate case
       and it makes the map *injective*, which is exactly what you want when a
       palettized drawing is put on a table it already fits.
    3. Otherwise :func:`_median_boxes`, count-weighted, so a thousand
       antialiasing colours do not out-vote the fill behind them. It may return
       fewer boxes than asked when colours run out; that is fine, since the
       assignment below only needs boxes <= targets.
    4. Each box's colour is :func:`_box_colour` -- one definition, shared.
    5. A box's rank is the lowest sorted-key index in it: distinct across boxes
       and stable, which is what makes the tie-breaks below deterministic.
    6. Assignment. Every (box, target) pair scored by exact integer
       squared-Euclidean distance, sorted by ``(distance, box rank, palette
       index)``, then walked greedily taking any pair whose box and target are
       both still free. Distance-zero pairs are first in that order, so a
       drawing already on the palette maps every colour to itself -- the
       identity a ``METHODS``-parametrized test forces on every method here.
       One-to-one is the point: it is what stops fifteen greys collapsing.

    Duplicate palette entries are kept as distinct slots on purpose. Two
    identical browns are two targets, and a scope with enough groups will use
    both; collapsing them would silently shrink the palette the user chose.
    """
    keys, counts = _distinct(planes)
    entries = np.asarray([tuple(c)[:3] for c in palette], dtype=np.int64)
    if keys.size == 0:
        return keys, np.zeros((0, 3), dtype=np.uint8)
    colours = np.stack(
        [(keys >> 16) & 0xFF, (keys >> 8) & 0xFF, keys & 0xFF], axis=1
    ).astype(np.int64)
    want = entries.shape[0]

    if keys.size <= want:
        boxes = [np.array([i], dtype=np.int64) for i in range(keys.size)]
    else:
        boxes = _median_boxes(colours, counts, want)

    reps = np.clip(
        np.asarray([_box_colour(colours, counts, box) for box in boxes], dtype=np.int64),
        0,
        255,
    )
    ranks = np.asarray([int(box.min()) for box in boxes], dtype=np.int64)

    delta = reps[:, None, :] - entries[None, :, :]
    d2 = (delta * delta).sum(axis=2)
    count = len(boxes)
    box_index = np.repeat(np.arange(count, dtype=np.int64), want)
    target_index = np.tile(np.arange(want, dtype=np.int64), count)
    # The last key is ``lexsort``'s primary one, so this is
    # (distance, box rank, palette index) read top to bottom.
    order = np.lexsort((target_index, ranks[box_index], d2.reshape(-1)))

    assigned = np.full(count, -1, dtype=np.int64)
    taken = np.zeros(want, dtype=bool)
    left = count
    for pair in order:
        if left == 0:
            break
        box_at = int(box_index[pair])
        target_at = int(target_index[pair])
        if assigned[box_at] >= 0 or taken[target_at]:
            continue
        assigned[box_at] = target_at
        taken[target_at] = True
        left -= 1

    targets = np.zeros((keys.size, 3), dtype=np.uint8)
    for index, box in enumerate(boxes):
        targets[box] = entries[assigned[index]].astype(np.uint8)
    return keys, targets


def _candidate_slots(count: int, transparent: int) -> tuple[int, list[int]]:
    """``(hole, candidate slots)`` for an indexed conversion of ``count`` slots.

    Extracted from :func:`convert_indices` so the rule exists once: the
    transparent slot is never a candidate, because dithering may pick any
    candidate for any pixel and a hole left in the running scatters holes through
    a solid area wherever its colour happens to win. :func:`grouped_index_table`
    needs the identical answer, and a second copy of the ``or list(range(count))``
    degenerate case is how the two would come to disagree on a one-entry palette.
    """
    hole = int(transparent) if 0 <= int(transparent) < count else 0
    return hole, [i for i in range(count) if i != hole] or list(range(count))


def grouped_index_table(
    planes: Iterable[np.ndarray],
    palette: Sequence[RGBA],
    *,
    transparent: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`grouped_table` over an indexed document's *candidate* slots.

    The transparent slot cannot be a target by construction rather than by a
    later filter: it is simply not in the table the assignment runs over.
    """
    count = len(palette)
    if not count:
        raise ValueError("a conversion needs at least one colour")
    _hole, slots = _candidate_slots(count, transparent)
    return grouped_table(planes, [tuple(palette[i]) for i in slots])


def _grouped(
    out: np.ndarray, keys: np.ndarray, targets: np.ndarray, entries: np.ndarray
) -> np.ndarray:
    """Apply a grouped table to one plane, in place, and return it.

    The visible mask is ``alpha > 0`` -- :func:`_ordered`'s rule exactly, and
    deliberately so: a semi-transparent pixel has a colour and is converted, a
    fully transparent one has none and rides through verbatim.

    A colour the table does not carry is snapped to its nearest palette entry
    rather than left alone. That keeps :func:`convert` *total* for a stale
    table -- a preview built before a stroke, say -- instead of returning a
    picture with a few off-palette pixels in it, which is precisely the thing
    the whole snap identity exists to make impossible.
    """
    visible = out[..., 3] > 0
    if not visible.any():
        return out
    rgb = out[..., :3][visible]
    packed = (
        rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
    )
    if keys.size:
        found = np.searchsorted(keys, packed)
        np.clip(found, 0, keys.size - 1, out=found)
        hit = keys[found] == packed
        mapped = targets[found].copy()
    else:
        hit = np.zeros(packed.shape, dtype=bool)
        mapped = np.zeros((packed.size, 3), dtype=np.uint8)
    if not hit.all():
        missing = ~hit
        colours = np.stack(
            [
                (packed[missing] >> 16) & 0xFF,
                (packed[missing] >> 8) & 0xFF,
                packed[missing] & 0xFF,
            ],
            axis=1,
        ).astype(np.int32)
        table = entries.astype(np.int32)
        delta = colours[:, None, :] - table[None, :, :]
        mapped[missing] = table[np.argmin((delta * delta).sum(axis=2), axis=1)].astype(
            np.uint8
        )
    out[..., :3][visible] = mapped
    return out

def convert(
    pixels: np.ndarray,
    palette: Sequence[RGBA],
    method: str = "nearest",
    *,
    table: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """*pixels* rewritten onto *palette* by *method*. Returns a new array.

    Every output colour is an entry of *palette*, exactly -- that is what makes
    the result stable under the snap every later write goes through. Alpha rides
    through byte-identical and fully transparent pixels are returned verbatim.

    ``table`` is a :func:`grouped_table` result and is only meaningful for the
    grouped method, where it carries the *scope*: the document-wide grouping the
    caller wants this plane converted under. Passing one with any other method is
    a ``ValueError`` rather than an ignored argument, because a caller who thinks
    it is applying a shared grouping and is silently not would produce a document
    whose layers disagree, with nothing anywhere to say so. Without a table,
    grouped builds one from this plane alone, which is the right answer for the
    single-plane callers (and for the seven ``METHODS``-parametrized suites).
    """
    if method not in METHODS:
        raise ValueError(f"unknown dither method {method!r}")
    if table is not None and method != "grouped":
        raise ValueError("a grouped table is only meaningful for the grouped method")
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
    entries = np.asarray([tuple(c)[:3] for c in palette], dtype=np.int16)
    if method == "grouped":
        keys, targets = grouped_table([out], palette) if table is None else table
        return _grouped(out, keys, targets, entries)
    if method == "floyd-steinberg":
        return _floyd_steinberg(out, entries)
    return _ordered(out, entries, bayer_matrix(BAYER_SIZES[method]))


def convert_indices(
    pixels: np.ndarray,
    palette: Sequence[RGBA],
    method: str = "nearest",
    *,
    transparent: int = 0,
    table: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """*pixels* converted onto *palette* as an ``(H, W) uint8`` **index plane**.

    :func:`convert`'s answer to the other question. That one returns the picture
    in RGBA, which is what palette-constrained RGB wants; this returns the slots,
    which is what a truly indexed document stores.

    **Delegated, never reimplemented**, for the reason ``convert`` gives about
    nearest being the snap: the dithering arithmetic is subtle -- a reflected
    second candidate, a canvas-anchored threshold matrix, a serpentine
    diffusion -- and two copies of it are how a document converted through one
    door and repainted through the other come to disagree along an edge. So the
    real work is ``convert`` on the *candidate* table, and this adds only the
    two things indices have and colours do not: which slot a colour came from,
    and where the holes are.

    The candidate table is the palette **without its transparent slot**. That is
    ``index_plane.resolve``'s rule applied one level up and it matters most
    here: dithering is allowed to pick any candidate for any pixel, so a
    transparent slot left in the running would scatter holes through a solid
    area wherever its colour happened to win.

    Alpha decides the holes, at ``index_plane.OPAQUE_THRESHOLD``, and it is read
    off the *input* -- ``convert`` passes alpha through byte-identically, so the
    two agree, and reading the input says so.

    ``table`` is :func:`grouped_index_table`'s result and rides straight through
    to :func:`convert`; it must have been built over the *candidate* slots, which
    is what that function does.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("convert_indices takes (H, W, 4) uint8")
    count = len(palette)
    if not count:
        raise ValueError("a conversion needs at least one colour")
    if count > ixp.MAX_COLOURS:
        raise ValueError(f"an indexed palette holds at most {ixp.MAX_COLOURS} colours")
    hole, slots = _candidate_slots(count, transparent)
    candidates = [tuple(palette[i]) for i in slots]

    out = np.full(pixels.shape[:2], hole, dtype=np.uint8)
    if pixels.size == 0:
        return out
    visible = pixels[..., 3] >= ixp.OPAQUE_THRESHOLD
    if not visible.any():
        return out

    # ``table`` rides through to the inner ``convert``, so every parity pin that
    # already compares this door's answer against that one covers grouped too.
    painted = convert(pixels, candidates, method, table=table)
    # ``transparent=None``: this sub-table has no hole in it, every entry is a
    # candidate, and the holes were labelled above off the input's alpha.
    local = ixp.resolve(painted, ixp.lut(candidates, transparent=-1), None)
    out[visible] = np.asarray(slots, dtype=np.uint8)[local[visible]]
    return out


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


def _diffuse_native(work: np.ndarray, visible: np.ndarray, entries: np.ndarray) -> bool:
    """Run the diffusion in C, or return False for "not this call's shape".

    A refusal is a fallback and never an error, the same as everywhere else at
    this seam. The arrays are all freshly built by the caller, so the contiguity
    tests below are formalities that keep the kernel free of strides it would
    never see -- but they are cheap and a stride the kernel indexed wrongly
    would be silent.
    """
    if not native.available():
        return False
    if work.dtype != np.float32 or entries.dtype != np.float32:
        return False
    if not work.flags["C_CONTIGUOUS"] or not entries.flags["C_CONTIGUOUS"]:
        return False
    if visible.dtype != np.bool_ or not visible.flags["C_CONTIGUOUS"]:
        return False
    if work.ndim != 3 or work.shape[2] != 3 or entries.ndim != 2 or entries.shape[1] != 3:
        return False
    if visible.shape != work.shape[:2] or not entries.shape[0]:
        return False
    native.dither_fs(work, visible, entries)
    return True


def _floyd_steinberg(out: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Serpentine error diffusion in straight float32 RGB.

    A Python loop, and that is a decision rather than a gap: error diffusion is
    sequential by definition -- every pixel's input depends on its neighbours'
    already-quantised output -- so there is no vectorisation of it that is still
    Floyd-Steinberg. The preview above this memoises per (palette, method) for
    exactly this reason, the same way the blur filter does.

    Which is also why this is the one kernel in ``native/`` whose reference is a
    Python loop: what it costs is not arithmetic but about ten numpy dispatches
    per pixel, measured at ~10 us/px and near enough flat in palette size, so a
    2048-square conversion took about 43 seconds. The loop below is never
    deleted -- it is the fallback and the thing ``tests/inker/test_dither_native.py``
    measures the kernel against, bit for bit.

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

    if _diffuse_native(work, visible, entries):
        out[..., :3][visible] = np.clip(work, 0.0, 255.0)[visible].astype(np.uint8)
        return out

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


def _median_boxes(colours: np.ndarray, counts: np.ndarray, want: int) -> list[np.ndarray]:
    """Repeatedly split the widest box at its count-weighted median.

    Weighted by pixel count and not by distinct-colour count, which is the whole
    difference between a palette of the picture and a palette of its noise: a
    thousand near-identical antialiasing colours occupying two hundred pixels
    must not out-vote the flat fill behind them.

    Returns the *boxes* -- disjoint index arrays into ``colours`` -- rather than
    their representatives, because two callers want different halves of this.
    :func:`_median_cut` wants one colour per box; :func:`grouped_table` wants the
    membership, since which distinct colours were grouped together is the whole
    of what it is asking. Fewer than ``want`` boxes come back when the colours
    run out, and both callers handle that.
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
    return boxes


def _box_colour(colours: np.ndarray, counts: np.ndarray, box: np.ndarray) -> np.ndarray:
    """One box's colour: the count-weighted mean, rounded half up and clipped.

    One definition, because :func:`_median_cut` and :func:`grouped_table` must
    agree about what a box *is* -- a grouped conversion whose representatives
    were rounded differently from the palette builder's would put a drawing on
    a table it did not quite come from.
    """
    weights = counts[box].astype(np.float64)
    mean = (colours[box].astype(np.float64) * weights[:, None]).sum(axis=0)
    return np.floor(mean / weights.sum() + 0.5)


def _median_cut(colours: np.ndarray, counts: np.ndarray, want: int) -> np.ndarray:
    """The colours :func:`_median_boxes`'s boxes stand for, one per box."""
    out = [_box_colour(colours, counts, box) for box in _median_boxes(colours, counts, want)]
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
