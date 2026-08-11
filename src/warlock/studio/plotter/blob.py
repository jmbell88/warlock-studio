"""The 47-case blob collapse, as arithmetic over eight bits.

An autotiled cell picks its picture from which of its eight neighbours are the
same terrain, so the raw question has 256 answers. Only 47 of them are distinct
*pictures*, and the collapse is one rule: **a diagonal neighbour cannot change
the silhouette unless both of its flanking edges are also present.** A cell with
a north-east neighbour but no north and no east is a cell whose north-east
corner is open regardless -- the diagonal is behind the corner, not in it.

This module is separate from everything that has a colour, a tileset or a
document because that rule is the piece a test wants alone: that all 256 raw
cases land in 47, that the table is a bijection onto ``0..46``, that the last
index is the interior fill. It knows nothing about terrains or maps, and the
price of that independence is one import.

**The table is derived at import, never typed.** A hand-written list of 47
masks is a list somebody eventually renumbers by hand, and the atlas layout is
keyed on the order -- so the order is a property of the integers instead:
ascending numeric, which puts the isolated tile at index 0 and the interior fill
at index 46. That is the order every published blob-47 sheet already uses, so an
artist's reference sheet and this table agree without anyone having chosen it.
"""

from __future__ import annotations

import numpy as np

#: The eight neighbour bits, clockwise from north. The values are positional and
#: the atlas order derives from them, so they are not free to renumber.
N = 1
NE = 2
E = 4
SE = 8
S = 16
SW = 32
W = 64
NW = 128

#: ``(dx, dy, bit)`` clockwise from north, in the y-down convention every array
#: in this package uses -- north is ``dy = -1``.
NEIGHBOURS: tuple[tuple[int, int, int], ...] = (
    (0, -1, N),
    (1, -1, NE),
    (1, 0, E),
    (1, 1, SE),
    (0, 1, S),
    (-1, 1, SW),
    (-1, 0, W),
    (-1, -1, NW),
)

#: In the order the drawing code wants them: the four sides, then the four
#: corners with the pair of sides each one sits between.
EDGE_BITS: tuple[int, int, int, int] = (N, E, S, W)
CORNER_BITS: tuple[int, int, int, int] = (NE, SE, SW, NW)
#: ``(corner, flank, flank)`` -- the two edges a corner is only visible past.
CORNER_FLANKS: tuple[tuple[int, int, int], ...] = (
    (NE, N, E),
    (SE, E, S),
    (SW, S, W),
    (NW, W, N),
)


def normalise(mask: int) -> int:
    """Drop every diagonal bit whose two flanking edges are not both set.

    The whole collapse, and the only place it is expressed. Applying it twice is
    the same as applying it once -- clearing a corner never clears an edge, so
    no flank can stop being set part way through -- which is what lets callers
    normalise defensively without thinking about it.
    """
    value = int(mask) & 0xFF
    for corner, first, second in CORNER_FLANKS:
        if not (value & first and value & second):
            value &= ~corner
    return value


#: The 47 canonical masks, ascending. Derived, so the count is an observation
#: rather than a promise -- if the rule above ever changed, this would simply be
#: a different length and ``test_blob`` would say so.
BLOB_MASKS: tuple[int, ...] = tuple(sorted({normalise(m) for m in range(256)}))

TILE_COUNT: int = len(BLOB_MASKS)

#: Every raw mask to its index in :data:`BLOB_MASKS`. A ``uint8`` array rather
#: than a dict because the hot caller indexes it with a whole ``(h, w)`` mask
#: field at once, and a dict would put a Python loop in the middle of it.
BLOB_INDEX: np.ndarray = np.array(
    [BLOB_MASKS.index(normalise(m)) for m in range(256)], dtype=np.uint8
)
BLOB_INDEX.setflags(write=False)

#: The interior fill -- every neighbour present, so no edge and no corner is
#: open. Named because the generator, the tests and any caller wanting "just the
#: middle of a field" all reach for it.
FULL: int = int(BLOB_INDEX[0xFF])
#: The isolated cell -- no neighbour at all, so every edge is open.
LONE: int = int(BLOB_INDEX[0])


def open_edges(blob_index: int) -> tuple[bool, bool, bool, bool]:
    """``(north, east, south, west)`` -- is that side of the cell exposed?

    Exposed means the neighbour is *not* a member, which is the side that gets
    an outline drawn on it. Phrased as "open" rather than "has a neighbour"
    because every caller is asking the drawing question.
    """
    mask = BLOB_MASKS[int(blob_index)]
    return tuple(not mask & bit for bit in EDGE_BITS)  # type: ignore[return-value]


def open_corners(blob_index: int) -> tuple[bool, bool, bool, bool]:
    """``(ne, se, sw, nw)`` -- is that corner *notched*?

    Notched is the narrow case that separates the 47 from the 16: both flanking
    edges have a neighbour, so the cell is interior along both sides, and yet
    the diagonal does not -- leaving a single-pixel bite out of the corner. When
    a flank is open the corner is already covered by that side's outline, so it
    is reported as not notched and the drawing code needs no special case.
    """
    mask = BLOB_MASKS[int(blob_index)]
    return tuple(  # type: ignore[return-value]
        bool(mask & first and mask & second and not mask & corner)
        for corner, first, second in CORNER_FLANKS
    )


def masks_from(member: np.ndarray, *, outside: bool = True) -> np.ndarray:
    """The raw eight-bit mask of every cell in a membership field.

    Eight shifted-slice ORs over a padded copy rather than a loop over cells:
    a whole-map retile is one of the things a terrain stroke does on the frame
    thread, and per-cell Python would put a 200x200 map's worth of it there.

    ``outside`` is what the border counts as. It defaults to *member*, because
    the alternative outlines every terrain that runs to the edge of the map --
    drawing a frame around the whole thing, which is never what anyone drew.
    """
    field = np.asarray(member, dtype=bool)
    if field.ndim != 2:
        raise ValueError("a membership field is two-dimensional")
    height, width = field.shape
    padded = np.empty((height + 2, width + 2), dtype=bool)
    padded[:] = bool(outside)
    padded[1 : height + 1, 1 : width + 1] = field

    masks = np.zeros((height, width), dtype=np.uint8)
    for dx, dy, bit in NEIGHBOURS:
        window = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        masks |= np.where(window, np.uint8(bit), np.uint8(0))
    return masks


def indices_from(member: np.ndarray, *, outside: bool = True) -> np.ndarray:
    """:func:`masks_from`, collapsed to ``0..46``. The form a tile id wants."""
    return BLOB_INDEX[masks_from(member, outside=outside)]
