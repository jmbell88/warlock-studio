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

from collections import deque

import numpy as np

from . import gid as gidlib

Region = tuple[int, int, np.ndarray]


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


def erase(data: np.ndarray, x: int, y: int, w: int = 1, h: int = 1) -> Region | None:
    """Clear a rectangle. Erasing *is* filling with gid 0, and saying so once
    here is what keeps the two from drifting apart."""
    return fill_rect(data, x, y, int(x) + int(w) - 1, int(y) + int(h) - 1, gidlib.EMPTY)


def flood_fill(data: np.ndarray, x: int, y: int, value: int) -> Region | None:
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
    """
    height, width = data.shape
    x, y = int(x), int(y)
    if not (0 <= x < width and 0 <= y < height):
        return None
    target = gidlib.DTYPE(data[y, x])
    fill = gidlib.DTYPE(value)
    if target == fill:
        return None

    seen = np.zeros((height, width), dtype=bool)
    seen[y, x] = True
    # A deque worked breadth-first: a recursive fill blows the stack on a large
    # open room, and this is bounded by the number of cells either way.
    queue: deque[tuple[int, int]] = deque([(x, y)])
    while queue:
        cx, cy = queue.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            inside = 0 <= nx < width and 0 <= ny < height
            if inside and not seen[ny, nx] and data[ny, nx] == target:
                seen[ny, nx] = True
                queue.append((nx, ny))

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
