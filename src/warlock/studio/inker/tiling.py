"""Seamless tiled painting: one wrap helper every tool goes through.

A tiled document is a torus. Every tool that wants to honour that needs the
same two answers -- *which pieces of a rectangle land where* and *where does a
point fold to* -- and the whole reason this module exists is that there is
exactly one implementation of both. A brush that wrapped with its own
arithmetic and a fill that wrapped with a second one would disagree about the
seam by a pixel, and a pixel at the seam is the only pixel a tile artist is
looking at.

Three properties are worth stating, because everything above leans on them.

*The pieces are dest rects plus a source offset*, never a modulo applied to
coordinates. A stamp is an array, and slicing it against a destination slice of
the same shape is the whole of what wrapping costs -- no per-pixel index
arithmetic, no fancy indexing, no copy of the canvas.

*Overlap is safe.* When a rectangle is wider than the canvas, several pieces
land on the same destination. That is fine on the paint path because coverage
is accumulated with ``np.maximum`` and the layer is recomputed from the
pre-stroke pixels, and it is fine on the fill path because
:func:`fold_coverage` folds with ``np.maximum`` too. Nothing here sums.

*Wrapping is per axis.* Aseprite's tiled mode offers X, Y or both, and so does
this: a side-scroller's ground tile repeats horizontally and must not repeat
vertically, and a stroke near the bottom edge of one is supposed to be clipped.
Every function takes ``axes`` as ``(wrap_x, wrap_y)``, and ``off`` on an axis is
exactly the clamped behaviour the editor had before this module existed --
which is what makes "tiled off" byte-identical rather than merely similar.
"""

from __future__ import annotations

import math

import numpy as np

#: The four states the per-tab toggle offers, as a string enum in the style of
#: ``brush.NIBS`` and ``brush.SYMMETRY``. A string rather than a pair of bools
#: because it is one control with four positions, and because a document that
#: ever grows a saved copy of it wants a name in the file rather than two flags.
TILED_AXES = ("off", "x", "y", "both")


def axes_of(tiled: str | tuple[bool, bool]) -> tuple[bool, bool]:
    """``(wrap_x, wrap_y)`` for one of :data:`TILED_AXES`.

    A pair is passed through unchanged, so the engine's own field -- which is
    already a pair -- can be handed to any of these without a round trip
    through the name.
    """
    if isinstance(tiled, str):
        if tiled not in TILED_AXES:
            raise ValueError(f"unknown tiled mode {tiled!r}")
        return (tiled in ("x", "both"), tiled in ("y", "both"))
    return (bool(tiled[0]), bool(tiled[1]))


def spans(lo: int, hi: int, size: int, wrap: bool) -> list[tuple[int, int, int]]:
    """Where ``[lo, hi)`` lands in ``[0, size)``, as ``(dest_lo, dest_hi, src_lo)``.

    ``src_lo`` indexes the *source* interval, which is ``hi - lo`` long: the
    segment is ``src[src_lo : src_lo + (dest_hi - dest_lo)]``. That is the form
    a slice wants, and it is why this returns an offset into the source rather
    than the shift that produced it.

    With ``wrap`` false there is at most one segment and it is the plain clip --
    the arithmetic the brush did before wrapping existed, so an unwrapped stroke
    is not merely equivalent to the old one, it is the same slices in the same
    order.

    With ``wrap`` true every translate by a whole ``size`` that meets the canvas
    contributes one. An interval longer than the canvas therefore yields
    overlapping destinations; see the module docstring for why that is safe.
    """
    lo, hi, size = int(lo), int(hi), int(size)
    if hi <= lo or size <= 0:
        return []
    if not wrap:
        d0, d1 = max(0, lo), min(size, hi)
        return [] if d1 <= d0 else [(d0, d1, d0 - lo)]
    # The copies that meet the canvas: the smallest k with ``hi + k*size > 0``
    # and the largest with ``lo + k*size < size``. Derived rather than searched,
    # so a stamp far off-canvas costs no loop at all.
    first = (-hi) // size + 1
    last = (size - lo - 1) // size
    out: list[tuple[int, int, int]] = []
    for k in range(first, last + 1):
        shift = k * size
        d0, d1 = max(0, lo + shift), min(size, hi + shift)
        if d1 > d0:
            out.append((d0, d1, d0 - lo - shift))
    return out


def pieces(
    rect: tuple[int, int, int, int],
    size: tuple[int, int],
    axes: tuple[bool, bool] = (True, True),
) -> list[tuple[tuple[int, int, int, int], tuple[int, int]]]:
    """A rectangle's pieces on the canvas: ``(dest_rect, (src_x, src_y))``.

    The cartesian product of the two axes' :func:`spans`, in row-major order so
    the answer is deterministic -- two runs of the same stroke have to produce
    the same slices in the same order, or a test of "wrapping changes nothing
    when it is off" is testing float addition instead.
    """
    x0, y0, x1, y1 = (int(v) for v in rect)
    width, height = int(size[0]), int(size[1])
    columns = spans(x0, x1, width, bool(axes[0]))
    rows = spans(y0, y1, height, bool(axes[1]))
    return [
        ((dx0, dy0, dx1, dy1), (sx, sy))
        for dy0, dy1, sy in rows
        for dx0, dx1, sx in columns
    ]


def tile_offset(
    point: tuple[float, float],
    size: tuple[int, int],
    axes: tuple[bool, bool] = (True, True),
) -> tuple[float, float]:
    """Which whole tile a point is in, as the offset that takes it home.

    ``point - tile_offset(point)`` is :func:`canonical`. It is returned
    separately because a *drag* must subtract one offset for its whole life:
    folding every sample independently makes the brush jump a full tile the
    moment the cursor crosses a seam mid-stroke, which is the one thing a
    seamless painter must not do.
    """
    width, height = int(size[0]), int(size[1])
    dx = math.floor(point[0] / width) * width if axes[0] and width > 0 else 0
    dy = math.floor(point[1] / height) * height if axes[1] and height > 0 else 0
    return (dx, dy)


def canonical(
    point: tuple[float, float],
    size: tuple[int, int],
    axes: tuple[bool, bool] = (True, True),
) -> tuple[float, float]:
    """A point folded onto the canvas along the wrapped axes.

    Sub-pixel position is preserved: the offset is a whole number of tiles, so
    the fractional part a brush walks on comes through untouched.
    """
    dx, dy = tile_offset(point, size, axes)
    return (point[0] - dx, point[1] - dy)


def fold_coverage(
    mask: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    axes: tuple[bool, bool] = (True, True),
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Fold a coverage plane onto the canvas. -> ``(bbox, crop)`` or None.

    ``mask`` is a 2D plane whose top-left sits at ``origin`` in canvas
    coordinates and which may be larger than the canvas and start outside it.
    Everything that lands on the same canvas pixel is combined with
    ``np.maximum`` -- the same rule a stroke's coverage follows, so a shape that
    crosses a seam and overlaps itself is drawn once rather than twice as dark.

    The bounding box of what survived comes back with a crop of exactly that
    box, which is the pair ``Document.write_colour`` takes. None means the fold
    covered nothing at all.
    """
    height_src, width_src = mask.shape[:2]
    ox, oy = int(origin[0]), int(origin[1])
    width, height = int(size[0]), int(size[1])
    plane = np.zeros((height, width), dtype=mask.dtype)
    for (dx0, dy0, dx1, dy1), (sx, sy) in pieces(
        (ox, oy, ox + width_src, oy + height_src), (width, height), axes
    ):
        target = plane[dy0:dy1, dx0:dx1]
        np.maximum(target, mask[sy : sy + (dy1 - dy0), sx : sx + (dx1 - dx0)], out=target)
    rows = np.flatnonzero(plane.any(axis=1))
    cols = np.flatnonzero(plane.any(axis=0))
    if rows.size == 0:
        return None
    rect = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    return rect, plane[rect[1] : rect[3], rect[0] : rect[2]]


def seam_seeds(
    reached: np.ndarray, candidate: np.ndarray, axes: tuple[bool, bool]
) -> list[tuple[int, int]]:
    """Pixels on a wrapped edge that a flood should carry on from.

    A toroidal flood fill is the ordinary one plus this: wherever the region has
    reached one edge and the opposite edge holds an unreached candidate pixel in
    the same row (or column), that pixel is the seed of the region's
    continuation on the other side. Iterating the pair to a fixpoint is what
    makes contiguity toroidal, and in practice it converges in one or two extra
    floods.

    Seeds are produced **only on wrapped axes**, which is the per-axis negative
    control: with X-only tiling a region touching the top edge must not leak to
    the bottom.
    """
    seeds: list[tuple[int, int]] = []
    height, width = reached.shape[:2]
    open_now = candidate & ~reached
    if axes[0] and width > 1:
        for src, dst in ((0, width - 1), (width - 1, 0)):
            rows = np.flatnonzero(reached[:, src] & open_now[:, dst])
            seeds.extend((dst, int(y)) for y in rows.tolist())
    if axes[1] and height > 1:
        for src, dst in ((0, height - 1), (height - 1, 0)):
            cols = np.flatnonzero(reached[src, :] & open_now[dst, :])
            seeds.extend((int(x), dst) for x in cols.tolist())
    return seeds


# --- the numeric truth about a seam -----------------------------------------

#: Above this the wrap seam is a visible edge rather than part of the texture.
#:
#: Copied from ``pipelines/seam.py`` with its citation:
#: ``docs/measurements/2026-08-08-seam-threshold.md``. 72 units on sdxl-turbo put
#: the highest legitimately seamless tile at 2.50 and the lowest visible seam at
#: 5.52, an empty band whose geometric centre is 3.72; 3.5 is the round value
#: inside it. A copy at a second surface moves nothing -- **the same document
#: governs both**, and re-measuring before moving it applies here exactly as it
#: does there. A copy rather than an import because this package is headless and
#: pinned against reaching into ``pipelines`` (which imports PIL at module
#: scope), the same reason ``dither`` keeps its own conversion.
SEAM_MAX = 3.5

#: Below this many pixels on a side there is no interior to compare against.
SEAM_MIN_SIDE = 8


def seam_ratio(pixels: np.ndarray) -> tuple[float, float]:
    """``(horizontal, vertical)`` -- how hard the wrap join is, as a ratio.

    A numpy port of ``pipelines/seam.py::_ratios``, and the same statistic for
    the same reason: the edge difference divided by the *interior* difference,
    because an absolute number of levels means nothing without the picture's own
    grain to divide it by. ``horizontal`` compares the first column against the
    last, ``vertical`` the first row against the last.

    **A flat image is 0.0, and the mean is why.** ``interior`` is a mean of
    absolute adjacent differences, so a mean of zero means every adjacent pair
    is identical -- which makes the first column equal the last and the edge
    zero too, so the ``inf`` arm below cannot fire while the statistic is a mean.
    It stays because that implication is a property of the statistic and not of
    the idea: swap the mean for a median (tempting, to stop one bright speck
    dominating a flat texture) and a mostly-flat image with one hard join lands
    there with a real seam, where returning 0.0 would call it seamless.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("a seam is measured on (H, W, 3|4)")
    rgb = array[:, :, :3].astype(np.float64)
    if min(rgb.shape[:2]) < SEAM_MIN_SIDE:
        return (0.0, 0.0)

    def axis_ratio(plane: np.ndarray) -> float:
        # ``plane`` is (rows, columns, channels); the wrap seam is the first
        # column against the last, the interior every adjacent pair.
        edge = float(np.abs(plane[:, 0] - plane[:, -1]).mean())
        interior = float(np.abs(np.diff(plane, axis=1)).mean())
        if interior <= 0.0:
            return 0.0 if edge <= 0.0 else float("inf")
        return edge / interior

    return (axis_ratio(rgb), axis_ratio(rgb.transpose(1, 0, 2)))
