"""Finding a tile grid in an image that was drawn with separator lines.

A generated tilesheet -- SDXL asked for "a tileset", say -- comes back as one
1024x1024 image whose cells are ruled off by dark lines and are *not* quite
uniform: fourteen pixels here, sixteen there, and a one or two pixel border.
Slicing that on imposed rectangles produces a tileset in which every tile is
offset from its own art by a different amount. This module measures the ruled
lines instead, so the cells can be cut where they actually are and redrawn on a
uniform grid.

**Detection is a suggestion; the caller confirms it with the user.** That is
the same rule :mod:`warlock.studio.inker.sheetin` states from the other side --
there, a *generated* atlas is sliced on the rectangles it was generated on and
the seams are never re-detected, because one mis-registered generation would
otherwise become a document nobody can open. Nothing here contradicts that: this
runs only at the door where a user hands the editor a file nobody in this
program made, it never fires on a generated sheet's own import path, and what it
returns is offered in a popup that the user accepts or declines. A detector that
applied itself would be the thing sheetin refuses.

This is one of four measurements in the tree and they must not be merged:

* :mod:`warlock.pipelines.pixel` -- the luma *lattice* period and phase of an
  upscaled pixel-art render, PIL, on the export path.
* :mod:`warlock.studio.inker.transform` -- the same lattice measure inside the
  editor, on the open document.
* this module -- dark separator *bands* between cells, at the import doors.
* :mod:`warlock.studio.tilegrid.roles` -- which blob role each cell of an
  already-gridded sheet plays.

Different questions at different doors. A future reader looking at two of them
and seeing "grid detection" twice should read this paragraph before unifying
anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Separator lines are drawn dark but not always black, and how dark depends on
# the generator and the sheet. Auto mode tries each of these in turn rather than
# picking a middle value that finds ragged lines on one sheet and phantom ones
# on another.
DARK_THRESHOLDS = (20, 30, 40, 50, 60)
# A row is a separator when nearly all of it is dark. Not *all* of it: a tile
# whose art bleeds a pixel into the rule would otherwise break the band, and a
# single stray pixel is not evidence against a line that runs the whole width.
MIN_DARK_RATIO = 0.98
# How uneven the cells may be and still be called a grid. Generated sheets are
# irregular by a pixel or two; a "grid" whose cells vary by more than this is
# an image with dark regions in it, not a ruled sheet.
MAX_SIZE_CV = 0.15


@dataclass(frozen=True)
class SheetGrid:
    """Where the content is, between the separator lines.

    ``rows`` and ``cols`` are inclusive ``(start, end)`` pixel segments -- the
    *content*, not the rules -- in reading order. ``threshold`` is the darkness
    that found them, kept because the popup says it out loud: a user looking at
    a wrong answer is better served by "threshold 60" than by a bare refusal.
    """

    rows: tuple[tuple[int, int], ...]
    cols: tuple[tuple[int, int], ...]
    threshold: int

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, cols)`` in cells."""
        return len(self.rows), len(self.cols)

    @property
    def tile_count(self) -> int:
        return len(self.rows) * len(self.cols)


def find_bands(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of ``True`` in a 1-D mask, as inclusive ``(start, end)`` pairs."""
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return []
    # A band breaks wherever the index jumps by more than one.
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _segments(bands: list[tuple[int, int]], length: int) -> list[tuple[int, int]]:
    """The content between (and outside) the separator bands, in order.

    A sheet may or may not have a border rule, so the leading and trailing
    segments are produced the same way as the interior ones and then dropped if
    they are empty -- a border line makes no phantom cell, and its absence makes
    no missing one.
    """
    starts = [0] + [end + 1 for _, end in bands]
    ends = [start - 1 for start, _ in bands] + [length - 1]
    return [(s, e) for s, e in zip(starts, ends, strict=True) if e >= s]


def _size_cv(segments: list[tuple[int, int]]) -> float:
    """Coefficient of variation of the segment lengths (0.0 when uniform)."""
    if not segments:
        return float("inf")
    sizes = np.array([end - start + 1 for start, end in segments], dtype=np.float64)
    mean = float(sizes.mean())
    if mean <= 0:
        return float("inf")
    return float(sizes.std() / mean)


def _detect_at(
    dark: np.ndarray,
    threshold: int,
    min_dark_ratio: float,
    max_size_cv: float,
) -> SheetGrid | None:
    rows = _segments(find_bands(dark.mean(axis=1) >= min_dark_ratio), dark.shape[0])
    cols = _segments(find_bands(dark.mean(axis=0) >= min_dark_ratio), dark.shape[1])
    # Two of each: one segment on an axis is "the image", not a grid of it.
    if len(rows) < 2 or len(cols) < 2:
        return None
    if _size_cv(rows) > max_size_cv or _size_cv(cols) > max_size_cv:
        return None
    return SheetGrid(tuple(rows), tuple(cols), int(threshold))


def detect_grid(
    pixels: np.ndarray,
    *,
    threshold: int | None = None,
    min_dark_ratio: float = MIN_DARK_RATIO,
    max_size_cv: float = MAX_SIZE_CV,
) -> SheetGrid | None:
    """The ruled grid in ``pixels``, or ``None`` when there is not one.

    Darkness is ``max(R, G, B) <= threshold`` -- the *brightest* channel, so a
    saturated red line is not mistaken for a rule. A row or column is a
    separator when at least ``min_dark_ratio`` of it is dark.

    With ``threshold`` given, that darkness is used verbatim and the answer is
    a grid or nothing. In auto mode every value in :data:`DARK_THRESHOLDS` is
    tried and the best is kept: most tiles first, lowest threshold to break a
    tie. Most tiles because a threshold too low finds only the darkest part of
    a rule and merges cells across the rest; lowest to break the tie because a
    higher one that finds the same grid is finding it by including art.

    There is no fallback threshold and no relaxed second pass. The caller's
    fallback is the blind slice it would have done anyway, and a detector that
    kept trying until something matched would find a grid in every image.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("a sheet is an (h, w, 3) or (h, w, 4) array")
    rgb = array[:, :, :3]
    brightest = rgb.max(axis=2)
    if threshold is not None:
        dark = brightest <= int(threshold)
        return _detect_at(dark, int(threshold), min_dark_ratio, max_size_cv)
    best: SheetGrid | None = None
    for candidate in DARK_THRESHOLDS:
        found = _detect_at(brightest <= candidate, candidate, min_dark_ratio, max_size_cv)
        if found is None:
            continue
        if best is None or found.tile_count > best.tile_count:
            best = found
    return best


def _nearest_index(src: int, dst: int) -> np.ndarray:
    """Centre-sampled nearest-neighbour index map, matching PIL's NEAREST.

    ``((2i + 1) * src) // (2 * dst)`` samples the centre of each destination
    pixel, which is what PIL does; the naive ``i * src // dst`` samples the left
    edge and shifts the art half a destination pixel toward the origin.
    """
    return (np.arange(dst, dtype=np.int64) * 2 + 1) * src // (2 * dst)


def recompose(
    pixels: np.ndarray,
    grid: SheetGrid,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    """A clean atlas: every cell of ``grid``, redrawn at ``tile_w`` x ``tile_h``.

    The separator lines are gone by construction -- only the content segments
    are read -- and each cell is resized by nearest neighbour, so the result
    holds no colour that was not already in its cell. Returns a fresh
    ``(rows * tile_h, cols * tile_w, 4)`` array; the source is never written
    through, whatever its writeability.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("a sheet is an (h, w, 3) or (h, w, 4) array")
    tile_w, tile_h = int(tile_w), int(tile_h)
    if tile_w < 1 or tile_h < 1:
        raise ValueError("a tile must be at least one pixel across")
    if not grid.rows or not grid.cols:
        raise ValueError("that grid holds no cells")
    rows, cols = grid.shape
    out = np.zeros((rows * tile_h, cols * tile_w, 4), dtype=np.uint8)
    for r, (y0, y1) in enumerate(grid.rows):
        ys = y0 + _nearest_index(y1 - y0 + 1, tile_h)
        for c, (x0, x1) in enumerate(grid.cols):
            xs = x0 + _nearest_index(x1 - x0 + 1, tile_w)
            cell = array[np.ix_(ys, xs)]
            if cell.shape[2] == 3:
                out[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w, :3] = cell
                out[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w, 3] = 255
            else:
                out[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = cell
    return out
