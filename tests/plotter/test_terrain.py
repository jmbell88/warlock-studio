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

from warlock.studio.plotter import blob, terrain
from warlock.studio.plotter import gid as gidlib
from warlock.studio.plotter.tileset import TilesetRef

from ._terrainset import terrain_ref


def _ref(count: int = 5) -> TilesetRef:
    return terrain_ref(count=count)


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


# --- the retile window --------------------------------------------------------
#
# ``_retile_into`` recomputes the blob case of every cell in a box. It used to
# run each terrain's membership pass over the *whole layer* and then throw all
# but the box away, which made one painted cell cost the map size -- 7.9 ms on a
# 512-square map, and a terrain drag paints several cells a frame. It now runs
# over the box grown by one, which is every cell the answer can depend on. These
# tests are the equivalence: the window is an optimisation, not a rule change,
# and the map's own edge is where the two could most easily part company.


def _retile_whole_layer(work, ref, box, outside):
    """``_retile_into`` as it was: every pass over the entire layer."""
    tileset = ref.tileset
    if not tileset.is_terrain_set:
        return
    x0, y0, x1, y1 = box
    ranks = terrain.rank_field(work, ref)
    window = ranks[y0:y1, x0:x1]
    for rank in range(len(tileset.terrains)):
        chosen = window == rank
        if not chosen.any():
            continue
        cases = blob.indices_from(ranks >= rank, outside=outside)[y0:y1, x0:x1]
        base = ref.firstgid + rank * blob.TILE_COUNT
        work[y0:y1, x0:x1][chosen] = (base + cases[chosen]).astype(gidlib.DTYPE)


def _scattered_map(ref, size=24, seed=7):
    rng = np.random.default_rng(seed)
    data = gidlib.empty_layer(size, size)
    ranks = rng.integers(-1, len(ref.tileset.terrains), size=(size, size))
    for rank in range(len(ref.tileset.terrains)):
        where = ranks == rank
        data[where] = terrain.gid_for(ref, rank, blob.FULL)
    return data


def test_the_retile_window_agrees_with_the_whole_layer_pass():
    """Every box, at every position -- including the four corners, where the
    grown window runs off the map and the padding has to be the map's own."""
    ref = _ref()
    data = _scattered_map(ref)
    size = data.shape[0]
    boxes = [
        (0, 0, 3, 3),  # the corner
        (size - 3, 0, size, 3),
        (0, size - 3, 3, size),
        (size - 3, size - 3, size, size),
        (0, 0, size, size),  # the whole map
        (5, 7, 9, 12),  # the interior
        (0, 5, 1, 6),  # one cell against an edge
        (11, 11, 12, 12),  # one cell in the middle
    ]
    for outside in (True, False):
        for box in boxes:
            mine = np.array(data, dtype=gidlib.DTYPE)
            theirs = np.array(data, dtype=gidlib.DTYPE)
            terrain._retile_into(mine, ref, box, outside)
            _retile_whole_layer(theirs, ref, box, outside)
            assert np.array_equal(mine, theirs), f"{box} outside={outside}"


def test_painting_a_cell_against_each_edge_is_unchanged_by_the_window():
    """The public path, not the private one: a stroke along the border is
    exactly where a window that got its padding wrong would draw an outline
    around the map."""
    ref = _ref()
    base = _scattered_map(ref, seed=11)
    size = base.shape[0]
    for x, y in [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1),
                 (0, size // 2), (size // 2, 0), (size // 2, size // 2)]:
        got = terrain.paint_terrain(base, x, y, 3, ref)
        want_work = np.array(base, dtype=gidlib.DTYPE)
        want_box = terrain._box([(x, y)], (size, size), 1)
        want_work[y, x] = terrain.gid_for(ref, 3, blob.FULL)
        _retile_whole_layer(want_work, ref, want_box, True)
        assert got is not None
        x0, y0, block = got
        assert np.array_equal(block, want_work[y0 : y0 + block.shape[0], x0 : x0 + block.shape[1]])


def test_the_retile_cost_does_not_follow_the_map_size():
    """The property the window exists for, asserted as a property rather than a
    stopwatch: what the pass touches is the box and its ring, so the same paint
    on a map sixteen times the area reads the same number of cells."""
    ref = _ref()
    counted = []

    real = blob.indices_from

    def counting(member, *, outside=True):
        counted.append(int(np.asarray(member).size))
        return real(member, outside=outside)

    for size in (24, 96):
        data = _scattered_map(ref, size=size, seed=3)
        counted.clear()
        blob.indices_from = counting
        try:
            terrain.paint_terrain(data, size // 2, size // 2, 2, ref)
        finally:
            blob.indices_from = real
        assert counted, "a paint retiles"
        assert max(counted) <= 5 * 5, f"{size}: read {max(counted)} cells for one painted cell"


# -- phase variants ------------------------------------------------------------


def _phase_ref(k: int = 2, count: int = 3) -> TilesetRef:
    return terrain_ref(count=count, phases=k, name="ground")


def test_a_painted_field_carries_the_coordinate_phase():
    """Phase is the cell's absolute map coordinates mod k -- what makes the map
    show consecutive periods of one surface, with no RNG and repaint-stable."""
    k = 2
    ref = _phase_ref(k)
    data = np.zeros((6, 6), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(6) for x in range(6)], 1, ref))
    local = gidlib.tile_ids(data).astype(np.int64) - ref.firstgid
    subs = (local // blob.TILE_COUNT) % (k * k)
    for y in range(6):
        for x in range(6):
            assert int(subs[y, x]) == (y % k) * k + (x % k), (x, y)
    # And the terrain reads back through the phase arithmetic.
    assert (terrain.rank_field(data, ref) == 1).all()


def test_repainting_a_phased_field_changes_nothing():
    """Idempotence is what proves the phase comes from absolute coordinates:
    a window-local phase would shift under the smaller repaint box."""
    ref = _phase_ref(4)
    data = np.zeros((8, 8), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(8) for x in range(8)], 0, ref))
    assert terrain.paint_terrain_cells(
        data, [(x, y) for y in range(2, 5) for x in range(3, 6)], 0, ref
    ) is None


def test_a_phased_junction_still_resolves_to_one_tile_each():
    ref = _phase_ref(2)
    data = np.zeros((7, 7), dtype=gidlib.DTYPE)
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(7) for x in range(7)], 0, ref))
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(2, 6) for x in range(2, 6)], 1, ref))
    _apply(data, terrain.paint_terrain_cells(
        data, [(x, y) for y in range(3, 5) for x in range(3, 5)], 2, ref))
    ranks = terrain.rank_field(data, ref)
    assert ranks[0, 0] == 0 and ranks[2, 2] == 1 and ranks[3, 3] == 2
    cases = _cases(data, ref)
    assert (cases[ranks == 0] == blob.FULL).all()
    assert int(cases[2, 2]) == int(blob.BLOB_INDEX[blob.E | blob.S | blob.SE])
