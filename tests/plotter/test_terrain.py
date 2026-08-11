"""Terrains, read back off the cells that use them.

The rule under test is that there is **no stored terrain field**: everything
here derives a cell's terrain from its own gid, so the tests are written the
same way -- paint through the public functions, then ask the gids what happened.

The three-terrain junction is the load-bearing case. Blob autotiling cannot say
*which* not-self a neighbour is, and the answer is that a terrain's list
position is its precedence: membership is every neighbour ranked at or above the
cell's own. If that ever changes, this file is what says so.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter import blob, terrain, tilegen
from warlock.studio.plotter import gid as gidlib
from warlock.studio.plotter.tileset import TilesetRef


def _ref(count: int = 5) -> TilesetRef:
    spec = tilegen.GenSpec(terrains=terrain.DEFAULT_TERRAINS[:count], tile_w=8, tile_h=8)
    return TilesetRef(firstgid=1, tileset=tilegen.generate(spec))


def _apply(data: np.ndarray, region) -> None:
    x0, y0, block = region
    data[y0 : y0 + block.shape[0], x0 : x0 + block.shape[1]] = block


def _cases(data: np.ndarray, ref: TilesetRef) -> np.ndarray:
    local = gidlib.tile_ids(data).astype(np.int64) - ref.firstgid
    return local % blob.TILE_COUNT


def test_a_terrains_row_is_its_rank():
    ref = _ref()
    for rank in range(len(ref.tileset.terrains)):
        for case in (0, 13, blob.FULL):
            assert ref.tileset.terrain_of(ref.tileset.local_for(rank, case)) == rank


def test_an_empty_cell_ranks_below_every_terrain():
    ref = _ref()
    data = np.zeros((3, 3), dtype=gidlib.DTYPE)
    assert (terrain.rank_field(data, ref) == terrain.RANK_VOID).all()


def test_painting_one_cell_touches_only_it_and_its_ring():
    ref = _ref()
    data = np.zeros((7, 7), dtype=gidlib.DTYPE)
    region = terrain.paint_terrain(data, 3, 3, 0, ref)
    x0, y0, block = region
    assert (x0, y0) == (2, 2)
    assert block.shape == (3, 3)


def test_painting_at_the_edge_clips_rather_than_raising():
    ref = _ref()
    data = np.zeros((4, 4), dtype=gidlib.DTYPE)
    x0, y0, block = terrain.paint_terrain(data, 0, 0, 0, ref)
    assert (x0, y0) == (0, 0)
    assert block.shape == (2, 2)


def test_a_placement_wholly_off_the_map_is_nothing():
    ref = _ref()
    data = np.zeros((4, 4), dtype=gidlib.DTYPE)
    assert terrain.paint_terrain(data, 99, 99, 0, ref) is None


def test_painting_the_same_terrain_twice_changes_nothing():
    """The no-op rule ``tools`` states, so a drag that re-crosses a cell does
    not push a step for standing still."""
    ref = _ref()
    data = np.zeros((5, 5), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain(data, 2, 2, 0, ref))
    assert terrain.paint_terrain(data, 2, 2, 0, ref) is None


def test_a_lone_cell_gets_the_lone_case_and_a_filled_field_the_interior():
    ref = _ref()
    data = np.zeros((5, 5), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain(data, 2, 2, 0, ref))
    assert int(_cases(data, ref)[2, 2]) == blob.LONE

    full = np.zeros((5, 5), dtype=gidlib.DTYPE)
    cells = [(x, y) for y in range(5) for x in range(5)]
    _apply(full, terrain.paint_terrain_cells(full, cells, 0, ref))
    assert (_cases(full, ref) == blob.FULL).all()


def test_a_field_running_to_the_border_is_not_outlined_against_the_void():
    """Or the map wears a frame nobody drew."""
    ref = _ref()
    data = np.zeros((4, 4), dtype=gidlib.DTYPE)
    cells = [(x, y) for y in range(4) for x in range(4)]
    _apply(data, terrain.paint_terrain_cells(data, cells, 0, ref))
    assert int(_cases(data, ref)[0, 0]) == blob.FULL


def test_three_terrains_meeting_resolve_to_one_tile_each():
    """The case the whole precedence rule exists for. Grass is never outlined
    because everything ranks at or above it; sand is outlined against grass
    only; water against both."""
    ref = _ref()
    data = np.zeros((7, 7), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(7) for x in range(7)], 0, ref))
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(2, 6) for x in range(2, 6)], 2, ref))
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(3, 5) for x in range(3, 5)], 4, ref))

    ranks = terrain.rank_field(data, ref)
    assert ranks[0, 0] == 0 and ranks[2, 2] == 2 and ranks[3, 3] == 4
    cases = _cases(data, ref)
    # Grass sees every neighbour as a member, so it is interior everywhere.
    assert (cases[ranks == 0] == blob.FULL).all()
    # Sand's block is solid in its own membership (sand plus the water inside
    # it), so its corner is an outer corner rather than anything ragged.
    assert int(cases[2, 2]) == int(blob.BLOB_INDEX[blob.E | blob.S | blob.SE])
    # Water's 2x2 sees only itself.
    assert int(cases[3, 3]) == int(blob.BLOB_INDEX[blob.E | blob.S | blob.SE])


def test_erasing_re_fits_what_surrounded_the_hole():
    ref = _ref()
    data = np.zeros((5, 5), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(5) for x in range(5)], 0, ref))
    _apply(data, terrain.erase_terrain(data, 2, 2, ref))
    assert data[2, 2] == gidlib.EMPTY
    # The ring is no longer interior: it has grown an edge against the hole.
    assert int(_cases(data, ref)[1, 1]) != blob.FULL


def test_a_terrain_fill_crosses_its_own_blob_cases():
    """``tools.flood_fill`` matches on the encoded value and would stop at the
    boundary between two of one terrain's forty-seven cases, filling a ribbon
    along an edge. The fill is over the *rank* field for exactly that reason."""
    ref = _ref()
    data = np.zeros((5, 5), dtype=gidlib.DTYPE)
    block = [(x, y) for y in range(1, 4) for x in range(1, 4)]
    _apply(data, terrain.paint_terrain_cells(data, block, 0, ref))
    # An inset block has corners and edges, so its cells do *not* share a gid --
    # which is the situation a gid-matching fill stops dead in.
    painted = {int(data[y, x]) for x, y in block}
    assert len(painted) > 1
    _apply(data, terrain.fill_terrain(data, 1, 1, 3, ref))
    ranks = terrain.rank_field(data, ref)
    assert {int(ranks[y, x]) for x, y in block} == {3}


def test_filling_with_the_terrain_already_there_is_nothing():
    ref = _ref()
    data = np.zeros((4, 4), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(4) for x in range(4)], 1, ref))
    assert terrain.fill_terrain(data, 1, 1, 1, ref) is None


def test_terrain_at_reads_back_what_was_painted():
    ref = _ref()
    data = np.zeros((4, 4), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain(data, 1, 1, 3, ref))
    assert terrain.terrain_at(data, 1, 1, ref) == 3
    assert terrain.terrain_at(data, 99, 0, ref) is None


def test_a_cell_from_another_tileset_ranks_as_void():
    """A map may carry more than one tileset, and a foreign gid must not be
    read as some terrain of this one."""
    ref = _ref()
    data = np.zeros((3, 3), dtype=gidlib.DTYPE)
    data[1, 1] = gidlib.DTYPE(ref.last_gid + 500)
    assert terrain.rank_field(data, ref)[1, 1] == terrain.RANK_VOID
