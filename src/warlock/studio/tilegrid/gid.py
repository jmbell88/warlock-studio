"""Tiled's global tile id, and its three transform flags, in one place.

A cell in a Tiled layer is a 32-bit number: the low 29 bits are a *global* tile
id -- an index into the map's tilesets, offset by each tileset's ``firstgid`` --
and the top three bits say the tile is mirrored horizontally, vertically, or
across its anti-diagonal. Together those encode all eight square symmetries,
which is why a rotation has no flag of its own: 90 degrees clockwise is
``FLIP_D | FLIP_H``.

**The array dtype is ``uint32`` and that is not a preference.** ``FLIP_H`` is
``0x80000000``, which does not fit in a signed 32-bit integer -- an ``int32``
layer would store it as a negative number, and every comparison, every mask and
every round trip through JSON would then be arguing about two's complement.

**Flags travel through the whole model bit-exactly.** Nothing between the reader
and the renderer strips them, and nothing re-derives them: a flipped tile is one
number, so a flood fill that matches on the full encoded value correctly treats
a mirrored tile as a different tile from its unmirrored twin, and a save writes
back exactly what was read. Only the code that actually draws a cell decodes.
"""

from __future__ import annotations

import numpy as np

# The three transform bits, in Tiled's own order.
FLIP_H = 0x80000000
FLIP_V = 0x40000000
FLIP_D = 0x20000000

# Tiled reserves a fourth bit (0x10000000) for hexagonal 120-degree rotation.
# It is deliberately absent, and the reason survived this editor learning to
# draw isometric maps: both projections it draws are *square-symmetry* lattices
# whose cells are exactly the eight transforms above, and the bit belongs to
# hexagonal grids, which :mod:`.tmx` still refuses at the door. Carrying a
# constant for a case that cannot arrive is how it eventually gets
# half-implemented -- so instead of a constant here, the doors probe for the
# bit and refuse it *by name*: ``tmx._finish`` for Tiled files and
# ``wmap._validate`` for hand-edited ``.wmap``s. It used to be caught only by luck: the mask
# below spans bit 28, so a hand-edited file carrying the flag read as a tile id
# 268435456 too large and was rejected as "a tile no tileset accounts for" --
# the right outcome under the wrong sentence.
FLAG_MASK = FLIP_H | FLIP_V | FLIP_D
GID_MASK = 0x1FFFFFFF

# The one dtype a layer is ever stored in. Named so the readers, the writers and
# the tests all say it once.
DTYPE = np.uint32

EMPTY = 0


def tile_ids(gids: np.ndarray) -> np.ndarray:
    """The global ids with the transform flags removed.

    Stays ``uint32``: the mask fits in the dtype, so numpy's value-based
    promotion has nothing to widen to.
    """
    return np.asarray(gids, dtype=DTYPE) & DTYPE(GID_MASK)


def flags(gids: np.ndarray) -> np.ndarray:
    """The transform bits alone, as a mask that can be OR-ed back on."""
    return np.asarray(gids, dtype=DTYPE) & DTYPE(FLAG_MASK)


def compose(
    tile_id: int,
    *,
    flip_h: bool = False,
    flip_v: bool = False,
    flip_d: bool = False,
) -> int:
    """One global id and its flags as the single number a cell holds.

    Raises on an id that does not fit the 29 bits available, rather than
    silently writing a flag: the id and the flags share one word, so a
    too-large id *is* a flag as far as every reader is concerned.
    """
    value = int(tile_id)
    if value < 0 or value > GID_MASK:
        raise ValueError(f"tile id {value} does not fit in {GID_MASK.bit_length()} bits")
    if flip_h:
        value |= FLIP_H
    if flip_v:
        value |= FLIP_V
    if flip_d:
        value |= FLIP_D
    return value


def decompose(gid: int) -> tuple[int, bool, bool, bool]:
    """The inverse of :func:`compose`, for one cell."""
    value = int(gid)
    return (
        value & GID_MASK,
        bool(value & FLIP_H),
        bool(value & FLIP_V),
        bool(value & FLIP_D),
    )


def empty_layer(width: int, height: int) -> np.ndarray:
    """A blank layer of the one shape and dtype everything here agrees on."""
    return np.zeros((int(height), int(width)), dtype=DTYPE)


# --- the flag algebra: transforming a *plane* of cells ------------------------
#
# Moved here from ``plotter/tools.py``, which had the only copy, when the Inker
# needed the same rules to flip and rotate a tilemap layer (``ASEPRITE_PARITY.md``
# Wave 3's "refs-aware flip/rotate (the flag algebra)" item). Neither engine may
# import the other, and a second copy of a group-theory table is exactly the kind
# of thing that drifts silently -- so it lives in the shared leaf both already
# depend on, and ``plotter.tools`` now delegates rather than restating.
#
# **The rules are group algebra, not a lookup.** The flags are applied as
# ``V^v . H^h . D^d`` -- transpose, then mirror columns, then mirror rows -- so
# transforming the *drawn result* by T means composing T on the left and pushing
# it back into that normal form. H and V commute with each other but not with D
# (``D.V = H.D`` and ``D.H = V.D``), which is the whole content of the three
# rules:
#
#   * A horizontal mirror commutes all the way through and lands on H, so it
#     toggles ``FLIP_H`` and nothing else. Likewise V. Worth stating because the
#     intuition that a mirror should "swap axes when FLIP_D is set" is wrong:
#     that describes mirroring the tile *before* its own transform, and what a
#     user flipping a map means is the mirror of what they can see.
#   * A 90-degree clockwise rotation is ``H.D``, so composing it gives
#     ``h' = not v``, ``v' = h``, ``d' = not d`` -- which sends an unflagged tile
#     to ``FLIP_D | FLIP_H``, exactly what a clockwise quarter turn is above.
#
# All three are pinned pixel-for-pixel against the renderer rather than against a
# hand-derived table, because a table is the thing being derived.


def planes(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(h, v, d)`` boolean planes for a plane of cells."""
    arr = np.asarray(cells, dtype=DTYPE)
    return (
        (arr & DTYPE(FLIP_H)) != 0,
        (arr & DTYPE(FLIP_V)) != 0,
        (arr & DTYPE(FLIP_D)) != 0,
    )


def reflagged(
    cells: np.ndarray, h: np.ndarray, v: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Rebuild a plane of cells from its ids and three boolean flag planes.

    Empty cells stay empty: gid 0 with a flag set is not an empty cell, it is a
    tile id nothing accounts for, and it would survive every round trip.
    """
    ids = np.asarray(cells, dtype=DTYPE) & DTYPE(GID_MASK)
    out = ids.copy()
    for plane, bit in ((h, FLIP_H), (v, FLIP_V), (d, FLIP_D)):
        out |= np.where(plane, DTYPE(bit), DTYPE(0))
    return np.where(ids != 0, out, DTYPE(0)).astype(DTYPE)


def flip_plane_h(cells: np.ndarray) -> np.ndarray:
    """Mirror a plane of cells left-to-right, tiles and arrangement together."""
    arr = np.ascontiguousarray(np.fliplr(np.asarray(cells, dtype=DTYPE)))
    h, v, d = planes(arr)
    return reflagged(arr, ~h, v, d)


def flip_plane_v(cells: np.ndarray) -> np.ndarray:
    """Mirror a plane of cells top-to-bottom, tiles and arrangement together."""
    arr = np.ascontiguousarray(np.flipud(np.asarray(cells, dtype=DTYPE)))
    h, v, d = planes(arr)
    return reflagged(arr, h, ~v, d)


def rotate_plane_cw(cells: np.ndarray) -> np.ndarray:
    """A quarter turn clockwise. Non-square planes come back transposed."""
    arr = np.ascontiguousarray(np.rot90(np.asarray(cells, dtype=DTYPE), k=-1))
    h, v, d = planes(arr)
    return reflagged(arr, ~v, h, ~d)


def rotate_plane_ccw(cells: np.ndarray) -> np.ndarray:
    """A quarter turn counter-clockwise -- ``np.rot90``'s own direction.

    The inverse of :func:`rotate_plane_cw`, and derived as one rather than
    spelled as three of them: applying the clockwise rule ``(h, v, d) ->
    (~v, h, ~d)`` three times gives ``(h, v, d) -> (v, ~h, ~d)``, which is what
    is written here. A test pins it against three clockwise turns so the
    derivation cannot be quietly wrong.

    Wanted because the Inker's geometry ops are counter-clockwise
    (``transform.rotate90`` is ``np.rot90``) where the Plotter's brush rotation
    is clockwise, and turning three times to go back one is the kind of thing
    that reads as a bug six months later.
    """
    arr = np.ascontiguousarray(np.rot90(np.asarray(cells, dtype=DTYPE), k=1))
    h, v, d = planes(arr)
    return reflagged(arr, v, ~h, ~d)
