"""The 47-case collapse: that it *is* 47, and that the table is a bijection.

The atlas layout is keyed on this order, so these are not tidiness tests. If
``BLOB_MASKS`` ever reorders, every generated tileset in every saved ``.wmap``
starts meaning something else while every gid in them stays valid -- which is
the failure that leaves no trace anywhere to find it by.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.tilegrid import blob


def test_every_raw_mask_lands_in_the_table():
    assert blob.BLOB_INDEX.shape == (256,)
    assert set(blob.BLOB_INDEX.tolist()) == set(range(blob.TILE_COUNT))


def test_there_are_forty_seven_cases():
    """The number is an *observation* -- the table is derived from the bit
    constants -- and this is what makes it one."""
    assert blob.TILE_COUNT == 47
    assert len(blob.BLOB_MASKS) == 47
    assert len(set(blob.BLOB_MASKS)) == 47


def test_the_isolated_cell_is_first_and_the_interior_fill_is_last():
    """The ordering every published blob sheet uses, so an artist's reference
    and this table line up without anyone having chosen it."""
    assert blob.BLOB_MASKS[0] == 0
    assert blob.BLOB_MASKS[-1] == 0xFF
    assert blob.LONE == 0
    assert blob.FULL == blob.TILE_COUNT - 1


def test_a_corner_bit_survives_only_when_both_its_edges_do():
    """The whole collapse, stated on the one case it exists for: a diagonal
    neighbour behind an open corner cannot change the silhouette."""
    assert blob.normalise(blob.NE) == 0
    assert blob.normalise(blob.NE | blob.N) == blob.N
    assert blob.normalise(blob.NE | blob.E) == blob.E
    assert blob.normalise(blob.NE | blob.N | blob.E) == blob.NE | blob.N | blob.E


def test_normalising_twice_is_normalising_once():
    """Clearing a corner never clears an edge, so no flank can stop being set
    part way through -- which is what lets callers normalise defensively."""
    for mask in range(256):
        once = blob.normalise(mask)
        assert blob.normalise(once) == once


def test_a_notched_corner_needs_both_flanks_and_no_diagonal():
    """The narrow case that separates the forty-seven from the sixteen, and the
    one pixel of art that depends on it."""
    notched = blob.BLOB_INDEX[blob.N | blob.E]
    assert blob.open_corners(int(notched))[0] is True
    # With a flank open the corner is already covered by that side's outline.
    open_flank = blob.BLOB_INDEX[blob.N]
    assert blob.open_corners(int(open_flank))[0] is False


def test_open_edges_reports_the_side_with_no_neighbour():
    lone = blob.LONE
    assert blob.open_edges(lone) == (True, True, True, True)
    assert blob.open_edges(blob.FULL) == (False, False, False, False)
    north_only = int(blob.BLOB_INDEX[blob.N])
    assert blob.open_edges(north_only) == (False, True, True, True)


def test_a_field_of_members_is_all_interior():
    field = np.ones((4, 4), dtype=bool)
    assert (blob.indices_from(field) == blob.FULL).all()


def test_the_border_counts_as_a_member_by_default():
    """Or a terrain running to the edge of the map is outlined against the
    void, which draws a frame around the whole thing."""
    field = np.ones((3, 3), dtype=bool)
    assert (blob.indices_from(field, outside=True) == blob.FULL).all()
    assert blob.indices_from(field, outside=False)[0, 0] != blob.FULL


def test_a_lone_member_is_the_lone_case():
    field = np.zeros((3, 3), dtype=bool)
    field[1, 1] = True
    assert int(blob.indices_from(field)[1, 1]) == blob.LONE


def test_the_vectorised_masks_agree_with_the_neighbour_table():
    """``masks_from`` is eight shifted-slice ORs; this is the same answer worked
    out one cell at a time, which is the thing it is an optimisation of."""
    rng = np.random.default_rng(7)
    field = rng.random((6, 5)) > 0.5
    masks = blob.masks_from(field, outside=False)
    height, width = field.shape
    for y in range(height):
        for x in range(width):
            want = 0
            for dx, dy, bit in blob.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                inside = 0 <= nx < width and 0 <= ny < height
                if inside and field[ny, nx]:
                    want |= bit
            assert int(masks[y, x]) == want
