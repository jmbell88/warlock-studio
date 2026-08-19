"""Painting with a generic Wang set, beside the blob preset rather than over it.

The two rules this file exists to pin. **No match leaves the cell untouched**
rather than writing a near-miss -- a wrong tile is the silent half-read in
picture form, and a field with a hole in it is something the user can see and
fix. And the blob path is *untouched*: nothing here reaches it, which is what
lets the whole existing terrain corpus stay byte-identical.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter import terrain
from warlock.studio.tilegrid import gid as gidlib
from warlock.studio.tilegrid.tileset import Tileset, TilesetRef
from warlock.studio.tilegrid.wang import WangColour, WangSet


def _pixels(size: int = 64) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 3] = 255
    return out


def _corner_set() -> WangSet:
    """Two colours over the sixteen corner patterns, which is a complete corner
    set: every combination of the two colours at four corners."""
    tiles = {}
    for index in range(16):
        tiles[index] = (
            0,
            1 + ((index >> 0) & 1),
            0,
            1 + ((index >> 1) & 1),
            0,
            1 + ((index >> 2) & 1),
            0,
            1 + ((index >> 3) & 1),
        )
    return WangSet(
        name="Corners",
        kind="corner",
        colours=(WangColour("grass", "#00ff00"), WangColour("sand", "#ffff00")),
        tiles=tiles,
    )


def _ref(wangset: WangSet, firstgid: int = 1) -> TilesetRef:
    return TilesetRef(
        firstgid=firstgid,
        tileset=Tileset(
            name="w", pixels=_pixels(), tile_w=16, tile_h=16, wangsets=(wangset,)
        ),
    )


def _layer(width: int = 5, height: int = 5) -> np.ndarray:
    return gidlib.empty_layer(width, height)


def test_painting_a_colour_sets_the_cell_to_that_colours_interior() -> None:
    wangset = _corner_set()
    ref = _ref(wangset)
    data = _layer()

    region = terrain.paint_wang(data, 2, 2, 1, ref, wangset)
    assert region is not None
    x0, y0, block = region
    value = int(block[2 - y0, 2 - x0]) & gidlib.GID_MASK
    assert wangset.wangid_of(value - ref.firstgid) == (0, 1, 0, 1, 0, 1, 0, 1)


def test_a_second_colour_beside_the_first_reconciles_the_shared_corner() -> None:
    """Tiled's terrain brush: the painted cell asserts a colour and the ring
    reconciles against it."""
    wangset = _corner_set()
    ref = _ref(wangset)
    data = _layer()

    for x, colour in ((1, 1), (2, 2)):
        region = terrain.paint_wang(data, x, 2, colour, ref, wangset)
        assert region is not None
        x0, y0, block = region
        data[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block

    # Cell (1, 2)'s right-hand corners were re-chosen against cell (2, 2), whose
    # left-hand corners are colour 2.
    left = wangset.wangid_of((int(data[2, 1]) & gidlib.GID_MASK) - ref.firstgid)
    right = wangset.wangid_of((int(data[2, 2]) & gidlib.GID_MASK) - ref.firstgid)
    assert left[3] == right[5], "the shared bottom corner agrees"
    assert left[1] == right[7], "and so does the shared top one"


def test_no_match_leaves_the_cell_untouched() -> None:
    """A partial set: only the all-colour-1 tile exists, so a neighbour asking
    for anything else finds nothing and nothing is written."""
    partial = WangSet(
        kind="corner",
        colours=(WangColour("a"), WangColour("b")),
        tiles={0: (0, 1, 0, 1, 0, 1, 0, 1)},
    )
    ref = _ref(partial)
    data = _layer()
    assert terrain.paint_wang(data, 2, 2, 2, ref, partial) is None, (
        "colour 2 has no interior tile, so nothing is asserted at all"
    )


def test_a_cell_outside_the_map_paints_nothing() -> None:
    wangset = _corner_set()
    assert terrain.paint_wang(_layer(), 99, 99, 1, _ref(wangset), wangset) is None


def test_painting_the_same_cell_twice_changes_nothing_the_second_time() -> None:
    wangset = _corner_set()
    ref = _ref(wangset)
    data = _layer()
    region = terrain.paint_wang(data, 2, 2, 1, ref, wangset)
    x0, y0, block = region
    data[y0:y0 + block.shape[0], x0:x0 + block.shape[1]] = block
    assert terrain.paint_wang(data, 2, 2, 1, ref, wangset) is None


def test_the_region_covers_the_cell_and_its_ring() -> None:
    """One write and one undo step for the whole gesture."""
    wangset = _corner_set()
    region = terrain.paint_wang(_layer(), 2, 2, 1, _ref(wangset), wangset)
    x0, y0, block = region
    assert (x0, y0) == (1, 1)
    assert block.shape == (3, 3)


def test_the_ring_is_clipped_at_the_map_edge() -> None:
    wangset = _corner_set()
    region = terrain.paint_wang(_layer(), 0, 0, 1, _ref(wangset), wangset)
    x0, y0, block = region
    assert (x0, y0) == (0, 0)
    assert block.shape == (2, 2)


def test_a_wang_field_says_nothing_about_a_foreign_cell() -> None:
    """Off the map, an empty cell, and a tile the set does not describe all mean
    "nothing here has an opinion" -- which is what lets a field grow outward."""
    wangset = _corner_set()
    ref = _ref(wangset)
    data = _layer()
    data[0, 0] = 999  # no tileset accounts for it
    field_of = terrain.wang_field(data, ref, wangset)
    assert field_of(-1, 0) is None
    assert field_of(1, 1) is None, "an empty cell"
    assert field_of(0, 0) is None, "a gid this set does not describe"


def test_the_blob_path_is_untouched_by_any_of_this() -> None:
    """A tileset carrying a foreign set is *not* thereby a terrain set -- the
    positional preset and the general model never meet."""
    wangset = _corner_set()
    ref = _ref(wangset)
    assert ref.tileset.terrains == ()
    assert not ref.tileset.is_terrain_set
    # And the blob re-fit declines it rather than doing something to it.
    work = _layer()
    before = np.array(work)
    terrain._retile_into(work, ref, (0, 0, 5, 5), True)
    assert np.array_equal(work, before)
