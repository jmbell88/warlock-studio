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
