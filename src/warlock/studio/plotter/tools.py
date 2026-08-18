"""What each tool computes, as arithmetic over one layer's array.

Every function here takes the layer's current gids and returns
``(x0, y0, region)`` -- the rectangle to write and what to write into it -- or
``None`` when there is nothing to do. Nothing mutates, nothing pushes a step and
nothing knows what a document is: :meth:`MapDoc.write_region` takes the diff and
owns the history, so a tool that computed a region identical to what is already
there costs a comparison rather than an undo step.

Clipping lives here rather than at the call site because every tool needs it and
the interesting case is the same for all of them: a brush dragged off the edge
of the map is a legitimate stroke whose visible part must land, not a refusal
and not an exception. Only a placement *entirely* outside the map returns None.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..tilegrid import gid as gidlib

Region = tuple[int, int, np.ndarray]


def flood_mask(match: np.ndarray, x: int, y: int) -> np.ndarray:
    """The four-connected run of ``match`` reachable from ``(x, y)``, as a mask.

    **Frontier dilation, not a per-cell queue.** The old shape was a ``deque``
    of coordinates popped one at a time, so a 512-square open room was a
    quarter of a million Python loop iterations with four bounds tests and two
    array index operations apiece -- inside a click handler. This grows the
    whole reached set one ring per pass: the OR of four one-cell shifts is
    "every neighbour of everything reached so far", intersecting that with the
    unvisited matching cells is the next ring, and the loop ends when a pass
    adds nothing.

    The number of passes is the *path distance* to the furthest reachable cell
    rather than the cell count, which is what turns the quarter-million into a
    few hundred whole-array operations. Semantics are identical to the queue's
    by construction: four-connected (the four shifts are the four neighbours,
    and a diagonal is only ever reached through one of them), seeded at the
    point, and bounded by ``match`` -- shifts fill with ``False`` at the edges,
    so nothing wraps.

    Shared by :func:`flood_fill` and :func:`terrain.fill_terrain`, which were a
    verbatim copy of one another differing only in what they compared.
    """
    height, width = match.shape
    seen = np.zeros((height, width), dtype=bool)
    if not (0 <= x < width and 0 <= y < height) or not match[y, x]:
        return seen
    seen[y, x] = True
    while True:
        grown = np.zeros_like(seen)
        grown[:-1, :] |= seen[1:, :]  # from below
        grown[1:, :] |= seen[:-1, :]  # from above
        grown[:, :-1] |= seen[:, 1:]  # from the right
        grown[:, 1:] |= seen[:, :-1]  # from the left
        grown &= match & ~seen
        if not grown.any():
            return seen
        seen |= grown


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """The all-octant integer form, endpoint inclusive."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    out: list[tuple[int, int]] = []
    x, y = x0, y0
    while True:
        out.append((x, y))
        if x == x1 and y == y1:
            return out
        step = 2 * err
        if step >= dy:
            err += dy
            x += sx
        if step <= dx:
            err += dx
            y += sy


def line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Every cell on the segment from one point to the other, both ends included.

    Integer Bresenham, eight-connected: consecutive cells differ by at most one
    in each axis, so a shallow diagonal steps rather than leaving gaps. **No
    clipping** -- the caller places a brush at each cell and every placement
    clips itself, which is what lets a line be drawn partly off the map exactly
    as a drag across the edge already is.

    Computed in a canonical direction and reversed when asked backwards, so
    ``line(a, b)`` is exactly ``reversed(line(b, a))``. Bresenham's error term
    resolves ties toward whichever end it started from, and a user who drags a
    line one way, undoes, and drags it the other should not get two different
    sets of cells out of the same two endpoints.
    """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if (x1, y1) < (x0, y0):
        return list(reversed(_bresenham(x1, y1, x0, y0)))
    return _bresenham(x0, y0, x1, y1)


def clip_region(region: Region, bounds: tuple[int, int, int, int]) -> Region | None:
    """A region trimmed to an inclusive ``(x0, y0, x1, y1)`` cell rect.

    What makes a selection constrain a tool without every tool learning about
    selections: the tools go on computing what they would do, and the region
    they hand back is cut down to what is allowed to land.

    ``None`` when nothing of it survives, which is the same answer every tool
    already gives for a placement entirely off the map.

    This is post-hoc and that is exactly right for a *placement* -- a stamp, a
    rectangle, an erase -- because those cover a rectangle and cutting it is
    still the same shape. It is the wrong tool for a flood, which is why
    :func:`flood_fill` takes its bounds up front instead: a fill clipped
    afterwards would be allowed to escape the selection, run around the outside
    and re-enter, and the cells it reached would then be trimmed to look as
    though it had never left.
    """
    x0, y0, block = region
    bx0, by0, bx1, by1 = (int(value) for value in bounds)
    height, width = block.shape
    lo_x, lo_y = max(int(x0), bx0), max(int(y0), by0)
    hi_x, hi_y = min(int(x0) + width - 1, bx1), min(int(y0) + height - 1, by1)
    if hi_x < lo_x or hi_y < lo_y:
        return None
    return (
        lo_x,
        lo_y,
        np.ascontiguousarray(
            block[lo_y - int(y0) : hi_y - int(y0) + 1, lo_x - int(x0) : hi_x - int(x0) + 1]
        ),
    )


def stamp(data: np.ndarray, x: int, y: int, brush: np.ndarray) -> Region | None:
    """Place a multi-tile brush with its top-left corner at ``(x, y)``.

    The brush replaces wholesale, empty cells included. A brush comes from a
    rectangular drag across the tileset palette, so it is always fully
    populated; treating a zero as "leave what is there" would make a
    single-tile eraser impossible to express and give the palette's own empty
    top-left corner a hidden meaning.
    """
    block = np.asarray(brush, dtype=gidlib.DTYPE)
    if block.ndim != 2 or block.size == 0:
        return None
    height, width = data.shape
    bh, bw = block.shape
    sx0, sy0 = max(0, -int(x)), max(0, -int(y))
    tx0, ty0 = max(0, int(x)), max(0, int(y))
    span_w = min(bw - sx0, width - tx0)
    span_h = min(bh - sy0, height - ty0)
    if span_w <= 0 or span_h <= 0:
        return None
    return tx0, ty0, np.ascontiguousarray(block[sy0 : sy0 + span_h, sx0 : sx0 + span_w])


#: The generator :func:`random_stamp` reaches for when a caller names none.
#: Held rather than built per call: the canvas places one cell per interpolated
#: point of a drag, so ``np.random.default_rng()`` inside the function was a
#: fresh seed sequence -- and an OS entropy read -- for every cell of every
#: scatter stroke. Built lazily so importing this module costs nothing, and
#: kept out of the signature's default so a caller that passes its own
#: generator is still exactly deterministic, which is what the tests do.
_RNG: Any = None


def _default_rng() -> Any:
    global _RNG
    if _RNG is None:
        _RNG = np.random.default_rng()
    return _RNG


def random_stamp(
    data: np.ndarray,
    x: int,
    y: int,
    brush: np.ndarray,
    *,
    rng: Any = None,
) -> Region | None:
    """Choose one non-empty encoded gid from a stamp and place it at a cell."""
    choices = np.asarray(brush, dtype=gidlib.DTYPE).reshape(-1)
    choices = choices[choices != gidlib.EMPTY]
    if not choices.size:
        return None
    generator = _default_rng() if rng is None else rng
    chosen = choices[int(generator.integers(0, len(choices)))]
    return stamp(data, x, y, np.asarray([[chosen]], dtype=gidlib.DTYPE))


def fill_rect(
    data: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int
) -> Region | None:
    """One gid across a rectangle given by any two opposite corners."""
    height, width = data.shape
    lo_x, hi_x = sorted((int(x0), int(x1)))
    lo_y, hi_y = sorted((int(y0), int(y1)))
    lo_x, lo_y = max(0, lo_x), max(0, lo_y)
    hi_x, hi_y = min(width - 1, hi_x), min(height - 1, hi_y)
    if hi_x < lo_x or hi_y < lo_y:
        return None
    region = np.full((hi_y - lo_y + 1, hi_x - lo_x + 1), gidlib.DTYPE(value), gidlib.DTYPE)
    return lo_x, lo_y, region


def fill_ellipse(
    data: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int
) -> Region | None:
    """One gid across the ellipse inscribed in the box those corners give.

    A cell is in when its *centre* is, which is the test that makes a circle
    look like one at small radii -- an any-overlap test grows every diameter by
    a cell and squares off the poles.

    **The ellipse is measured from the drawn box and clipped afterwards**, never
    the other way round: clipping the box first and inscribing into what
    survived would reshape the ellipse as it crossed the edge of the map, so
    dragging one half off-screen would change the half still on it.

    Degenerate drags need no special case. A one-cell-wide box puts every cell
    centre exactly on the vertical axis, so the x term vanishes and the test
    reduces to the column of cells -- a line, which is what a user dragging a
    one-wide ellipse means. The same holds transposed, and a 1x1 box is the one
    cell.

    Like :func:`flood_fill`, the returned region keeps the cells inside its
    bounding box that the shape does not cover: an ellipse must not rectangle
    over its own corners.
    """
    height, width = data.shape
    lo_x, hi_x = sorted((int(x0), int(x1)))
    lo_y, hi_y = sorted((int(y0), int(y1)))
    centre_x, centre_y = (lo_x + hi_x + 1) / 2.0, (lo_y + hi_y + 1) / 2.0
    semi_x, semi_y = (hi_x - lo_x + 1) / 2.0, (hi_y - lo_y + 1) / 2.0

    clip_x0, clip_y0 = max(0, lo_x), max(0, lo_y)
    clip_x1, clip_y1 = min(width - 1, hi_x), min(height - 1, hi_y)
    if clip_x1 < clip_x0 or clip_y1 < clip_y0:
        return None

    ys, xs = np.ogrid[clip_y0 : clip_y1 + 1, clip_x0 : clip_x1 + 1]
    inside = ((xs + 0.5 - centre_x) / semi_x) ** 2 + (
        (ys + 0.5 - centre_y) / semi_y
    ) ** 2 <= 1.0
    if not inside.any():
        return None
    region = np.array(
        data[clip_y0 : clip_y1 + 1, clip_x0 : clip_x1 + 1], dtype=gidlib.DTYPE
    )
    region[inside] = gidlib.DTYPE(value)
    return clip_x0, clip_y0, region


def erase(data: np.ndarray, x: int, y: int, w: int = 1, h: int = 1) -> Region | None:
    """Clear a rectangle. Erasing *is* filling with gid 0, and saying so once
    here is what keeps the two from drifting apart."""
    return fill_rect(data, x, y, int(x) + int(w) - 1, int(y) + int(h) - 1, gidlib.EMPTY)


def flood_fill(
    data: np.ndarray,
    x: int,
    y: int,
    value: int,
    *,
    bounds: tuple[int, int, int, int] | None = None,
) -> Region | None:
    """Four-connected fill of the contiguous run under ``(x, y)``.

    **The match is on the full encoded value**, flags included, so a
    horizontally mirrored wall tile bounds a fill of its unmirrored twin. That
    is the right answer for the same reason the flags are carried everywhere
    else: two cells that draw differently are two different cells, and a fill
    that spilled through the mirrored ones would cross exactly the seam a user
    drew them to make.

    Four-connected rather than eight, because a diagonal leak through a
    one-pixel gap is the classic way a fill escapes a room whose corner tiles
    only touch at a point.

    ``bounds`` is an inclusive cell rect the fill may not leave, and it is
    applied to the *match* rather than to the result. Clipping afterwards would
    let the flood run out of the selection, around the outside and back in, and
    then trim the cells it reached so the trip left no trace -- the fill would
    have crossed a wall the user drew a marquee to stop it at. Masking the match
    makes the boundary real: there is nothing to spread through. A seed outside
    the bounds fills nothing, which is what a click outside a selection means.
    """
    height, width = data.shape
    x, y = int(x), int(y)
    if not (0 <= x < width and 0 <= y < height):
        return None
    target = gidlib.DTYPE(data[y, x])
    fill = gidlib.DTYPE(value)
    if target == fill:
        return None

    match = np.asarray(data) == target
    if bounds is not None:
        allowed = np.zeros_like(match)
        bx0, by0, bx1, by1 = (int(entry) for entry in bounds)
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(width - 1, bx1), min(height - 1, by1)
        if bx1 < bx0 or by1 < by0:
            return None
        allowed[by0 : by1 + 1, bx0 : bx1 + 1] = True
        match &= allowed
    seen = flood_mask(match, x, y)
    if not seen.any():
        return None

    ys, xs = np.nonzero(seen)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    # The bounding box of what was reached, with the untouched cells inside it
    # left exactly as they were -- a fill of an L-shaped room must not
    # rectangle over the wall in its notch.
    region = np.array(data[y0:y1, x0:x1], dtype=gidlib.DTYPE)
    region[seen[y0:y1, x0:x1]] = fill
    return x0, y0, region


def pick(data: np.ndarray, x: int, y: int) -> int | None:
    """The encoded cell under a point, for an eyedropper. ``None`` off-map."""
    height, width = data.shape
    x, y = int(x), int(y)
    if not (0 <= x < width and 0 <= y < height):
        return None
    return int(data[y, x])


# --- transforming a brush -----------------------------------------------------
#
# These take a *brush* and return a brush, which is what makes them the odd ones
# out here: everything above answers with a Region to write into a layer. A
# brush transform moves two things at once -- where each tile sits in the block,
# and how that tile is drawn -- and both halves have to agree or a flipped brush
# stamps a mirrored arrangement of unmirrored tiles.
#
# The flag half is group algebra rather than a lookup. ``render.orient`` applies
# the flags as ``V^v . H^h . D^d`` (transpose, then mirror columns, then mirror
# rows), so transforming the *drawn result* by T means composing T on the left
# and pushing it back into that normal form. H and V commute with each other but
# not with D -- ``D.V = H.D`` and ``D.H = V.D`` -- which is the whole content of
# the three rules below:
#
#   * A horizontal mirror commutes all the way through and lands on H, so it
#     toggles FLIP_H and nothing else. Likewise V. This is worth stating because
#     the intuition that a mirror should "swap axes when FLIP_D is set" is
#     wrong: that describes mirroring the tile *before* its own transform, and
#     what a user flipping a brush means is the mirror of what they can see.
#   * A 90-degree clockwise rotation is ``H.D`` (transpose, then mirror
#     columns), so composing it gives ``h' = not v``, ``v' = h``, ``d' = not d``.
#     That sends an unflagged tile to FLIP_D | FLIP_H, which is exactly what
#     ``gid``'s own docstring says a clockwise quarter turn is.
#
# All three are pinned pixel-for-pixel against ``render.orient`` rather than
# against a hand-derived table, because a table is the thing being derived.


def _reflagged(brush: np.ndarray, h: np.ndarray, v: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Rebuild a block of cells from its ids and three boolean flag planes.

    Empty cells stay empty: gid 0 with a flag set is not an empty cell, it is a
    tile id nothing accounts for, and it would survive every round trip.
    """
    ids = np.asarray(brush, dtype=gidlib.DTYPE) & gidlib.DTYPE(gidlib.GID_MASK)
    out = ids.copy()
    for plane, bit in ((h, gidlib.FLIP_H), (v, gidlib.FLIP_V), (d, gidlib.FLIP_D)):
        out |= np.where(plane, gidlib.DTYPE(bit), gidlib.DTYPE(0))
    return np.where(ids != 0, out, gidlib.DTYPE(0)).astype(gidlib.DTYPE)


def _planes(brush: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(brush, dtype=gidlib.DTYPE)
    return (
        (arr & gidlib.DTYPE(gidlib.FLIP_H)) != 0,
        (arr & gidlib.DTYPE(gidlib.FLIP_V)) != 0,
        (arr & gidlib.DTYPE(gidlib.FLIP_D)) != 0,
    )


def flip_brush_h(brush: np.ndarray) -> np.ndarray:
    """Mirror a brush left-to-right, tiles and arrangement together."""
    arr = np.ascontiguousarray(np.fliplr(np.asarray(brush, dtype=gidlib.DTYPE)))
    h, v, d = _planes(arr)
    return _reflagged(arr, ~h, v, d)


def flip_brush_v(brush: np.ndarray) -> np.ndarray:
    """Mirror a brush top-to-bottom, tiles and arrangement together."""
    arr = np.ascontiguousarray(np.flipud(np.asarray(brush, dtype=gidlib.DTYPE)))
    h, v, d = _planes(arr)
    return _reflagged(arr, h, ~v, d)


def rotate_brush_cw(brush: np.ndarray) -> np.ndarray:
    """A quarter turn clockwise. Non-square brushes come back transposed."""
    arr = np.ascontiguousarray(np.rot90(np.asarray(brush, dtype=gidlib.DTYPE), k=-1))
    h, v, d = _planes(arr)
    return _reflagged(arr, ~v, h, ~d)
