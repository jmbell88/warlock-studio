"""Tiled's global id encoding.

The whole of what this file is defending is that a flipped tile survives every
hop -- read, edit, save, export -- as the *same number*. Nothing downstream
re-derives the flags, so a single lost bit is a tile that silently draws the
wrong way round in an exported map and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.tilegrid import gid


def test_all_eight_symmetries_round_trip():
    """Three flags, eight combinations, and every one of them has to come back
    exactly -- the diagonal flip is the one an implementation forgets, because
    it is the only one that is not a mirror."""
    for h in (False, True):
        for v in (False, True):
            for d in (False, True):
                value = gid.compose(1234, flip_h=h, flip_v=v, flip_d=d)
                assert gid.decompose(value) == (1234, h, v, d)


def test_the_flag_bits_are_the_ones_tiled_writes():
    assert gid.FLIP_H == 0x80000000
    assert gid.FLIP_V == 0x40000000
    assert gid.FLIP_D == 0x20000000
    assert gid.GID_MASK == 0x1FFFFFFF
    assert gid.FLAG_MASK | gid.GID_MASK == 0xFFFFFFFF
    # No overlap: a bit that meant both an id and a flag would make the two
    # halves of every cell unrecoverable from each other.
    assert gid.FLAG_MASK & gid.GID_MASK == 0


def test_masking_a_layer_never_promotes_the_dtype():
    """``FLIP_H`` does not fit in a signed 32-bit integer. If a mask promoted
    the array to int64 the values would still be right and every later
    ``frombuffer``/``tobytes`` would silently double in width."""
    layer = np.array([[gid.compose(3, flip_h=True), 0]], dtype=gid.DTYPE)
    assert gid.tile_ids(layer).dtype == gid.DTYPE
    assert gid.flags(layer).dtype == gid.DTYPE
    assert gid.tile_ids(layer).tolist() == [[3, 0]]
    assert gid.flags(layer).tolist() == [[gid.FLIP_H, 0]]


def test_the_top_bit_is_storable_at_all():
    """The reason the dtype is unsigned, stated as a value rather than a
    comment: an int32 layer stores this as a negative number and every
    comparison after it is an argument about two's complement."""
    layer = gid.empty_layer(1, 1)
    layer[0, 0] = gid.compose(1, flip_h=True)
    assert int(layer[0, 0]) == 0x80000001


def test_an_id_too_large_to_fit_is_refused():
    """The id and the flags share one word, so an over-large id *is* a flag as
    far as every reader is concerned. Better to raise than to write one."""
    with pytest.raises(ValueError):
        gid.compose(gid.GID_MASK + 1)
    with pytest.raises(ValueError):
        gid.compose(-1)


def test_an_empty_layer_is_zeros_of_the_right_shape():
    layer = gid.empty_layer(5, 3)
    assert layer.shape == (3, 5)  # rows, columns
    assert layer.dtype == gid.DTYPE
    assert not layer.any()
