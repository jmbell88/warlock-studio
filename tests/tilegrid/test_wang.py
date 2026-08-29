"""Wang sets as data, and the blob preset derived from them.

The load-bearing claim of the whole module is the last test here: the table
:func:`blob_wangset` derives is *exactly* the one the Tiled exporter has always
written for a blob terrain set. If those two ever disagree, a set generated here
opens in Tiled with a brush that paints the wrong tile, and nothing in either
file says so.
"""

from __future__ import annotations

import pytest

from warlock.studio.tilegrid import blob, wang


def _corner_set() -> wang.WangSet:
    """A two-colour corner set: four tiles covering the four corner patterns of
    one colour against the other, the smallest set a matcher can be wrong on."""
    return wang.WangSet(
        name="Corners",
        kind="corner",
        colours=(wang.WangColour("grass", "#00ff00"), wang.WangColour("sand", "#ffff00")),
        tiles={
            0: (0, 1, 0, 1, 0, 1, 0, 1),
            1: (0, 2, 0, 2, 0, 2, 0, 2),
            2: (0, 1, 0, 2, 0, 2, 0, 1),
            3: (0, 2, 0, 1, 0, 1, 0, 2),
        },
    )


# --- the record ----------------------------------------------------------------


def test_a_wangid_must_have_eight_slots() -> None:
    with pytest.raises(ValueError, match="8 slots"):
        wang.WangSet(colours=(wang.WangColour(),), tiles={0: (1, 1, 1)})


def test_a_slot_may_not_name_a_colour_the_set_lacks() -> None:
    with pytest.raises(ValueError, match="does not have"):
        wang.WangSet(colours=(wang.WangColour(),), tiles={0: (2, 0, 0, 0, 0, 0, 0, 0)})


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="wang set is one of"):
        wang.WangSet(kind="diagonal")


def test_a_negative_colour_probability_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        wang.WangColour(probability=-1.0)


def test_the_kind_decides_the_slots_not_the_data() -> None:
    """A corner set every one of whose tiles happens to have an unset edge is
    still a corner set; inferring the kind from the table would make an all-blank
    tile change what the set *is*."""
    assert _corner_set().slots == wang.CORNER_SLOTS
    assert wang.WangSet(kind="edge").slots == wang.EDGE_SLOTS
    assert wang.WangSet(kind="mixed").slots == tuple(range(8))


def test_a_tile_the_set_says_nothing_about_is_all_unset() -> None:
    assert _corner_set().wangid_of(99) == (0,) * 8


# --- matching -------------------------------------------------------------------


def test_an_exact_match_is_found() -> None:
    found = _corner_set().matching({1: 1, 3: 1, 5: 1, 7: 1})
    assert found == [0]


def test_an_unconstrained_slot_constrains_nothing() -> None:
    """A slot the caller left out is one nothing has an opinion about yet -- the
    edge of the map, or a neighbour holding a tile this set does not describe --
    and demanding a colour there would refuse every tile for no reason."""
    found = _corner_set().matching({1: 1})
    assert set(found) == {0, 2}


def test_no_match_returns_nothing_rather_than_a_near_miss() -> None:
    """A wrong tile is the silent half-read in picture form."""
    assert _corner_set().matching({1: 1, 3: 1, 5: 1, 7: 2}) == []


def test_ties_break_by_probability_then_lowest_id() -> None:
    heavy = wang.WangSet(
        kind="corner",
        colours=(wang.WangColour("a", probability=1.0), wang.WangColour("b", probability=5.0)),
        tiles={
            7: (0, 1, 0, 0, 0, 0, 0, 0),
            3: (0, 1, 0, 0, 0, 0, 0, 0),
            9: (0, 2, 0, 0, 0, 0, 0, 0),
        },
    )
    # Only tile 9 matches colour 2; among the two that match colour 1 the lower
    # id wins, because their weights are equal.
    assert heavy.matching({1: 2}) == [9]
    assert heavy.matching({1: 1}) == [3, 7]


def test_an_all_unset_tile_weighs_one_rather_than_zero() -> None:
    """Which keeps a set whose colours all weigh 1 in pure lowest-id order."""
    made = wang.WangSet(
        kind="mixed",
        colours=(wang.WangColour("a"),),
        tiles={5: (0,) * 8, 2: (0,) * 8},
    )
    assert made.matching({}) == [2, 5]


# --- constraints from neighbours -------------------------------------------------


def test_the_diagonal_neighbour_constrains_a_corner() -> None:
    """This cell's top-right corner is also the cell up-and-right's
    bottom-left."""
    wangset = _corner_set()

    def field_of(x: int, y: int):
        return (0, 0, 0, 0, 0, 2, 0, 0) if (x, y) == (1, -1) else None

    assert wang.constraints_from(field_of, 0, 0, wangset) == {1: 2}


def test_an_orthogonal_neighbour_constrains_a_corner_too() -> None:
    """The case consulting only the diagonal would miss, and the one that
    actually fires: a corner is a lattice point and four cells meet at it, so
    this cell's top-right corner is the right-hand neighbour's top-left."""
    wangset = _corner_set()

    def field_of(x: int, y: int):
        # The cell to the right, whose top-left and bottom-left corners are 2.
        return (0, 0, 0, 0, 0, 2, 0, 2) if (x, y) == (1, 0) else None

    assert wang.constraints_from(field_of, 0, 0, wangset) == {1: 2, 3: 2}


def test_an_edge_is_shared_with_exactly_one_neighbour() -> None:
    edges = wang.WangSet(kind="edge", colours=(wang.WangColour("a"),), tiles={})

    def field_of(x: int, y: int):
        # The cell above, whose bottom edge is 1.
        return (0, 0, 0, 0, 1, 0, 0, 0) if (x, y) == (0, -1) else None

    assert wang.constraints_from(field_of, 0, 0, edges) == {0: 1}


def test_a_missing_neighbour_constrains_nothing() -> None:
    """Which is what lets a field grow outward instead of refusing at its own
    edge."""
    assert wang.constraints_from(lambda x, y: None, 0, 0, _corner_set()) == {}


def test_a_neighbour_whose_shared_slot_is_unset_constrains_nothing() -> None:
    assert wang.constraints_from(lambda x, y: (0,) * 8, 0, 0, _corner_set()) == {}


def test_an_edge_set_is_never_constrained_by_a_corner() -> None:
    edges = wang.WangSet(kind="edge", colours=(wang.WangColour("a"),), tiles={})

    def field_of(x: int, y: int):
        # Every slot set, corners included.
        return (1,) * 8

    got = wang.constraints_from(field_of, 0, 0, edges)
    assert set(got) == set(wang.EDGE_SLOTS)


def test_the_opposite_table_is_an_involution() -> None:
    """A shared piece of the world is shared both ways round."""
    for slot in range(wang.POSITIONS):
        assert wang.OPPOSITE[wang.OPPOSITE[slot]] == slot


def test_an_edge_slot_has_exactly_one_sharer_and_it_is_the_opposite() -> None:
    for slot in wang.EDGE_SLOTS:
        sharers = wang.SHARERS[slot]
        assert len(sharers) == 1
        assert sharers[0][1] == wang.OPPOSITE[slot]


def test_a_corner_slot_has_exactly_three_sharers() -> None:
    """Four cells meet at every lattice point, so a corner is shared with the
    other three."""
    for slot in wang.CORNER_SLOTS:
        assert len(wang.SHARERS[slot]) == 3


def test_sharing_is_symmetric() -> None:
    """If this cell's slot is that cell's slot, then that cell's slot is this
    cell's slot -- which is the property a hand-written table gets wrong."""
    for slot in range(wang.POSITIONS):
        for (dx, dy), theirs in wang.SHARERS[slot]:
            back = wang.SHARERS[theirs]
            assert ((-dx, -dy), slot) in back, (slot, (dx, dy), theirs)


# --- the blob derivation ---------------------------------------------------------


def test_the_blob_preset_has_one_tile_per_case_per_colour() -> None:
    made = wang.blob_wangset(["grass", "sand"], ["#00ff00", "#ffff00"])
    assert len(made.tiles) == 2 * blob.TILE_COUNT
    assert made.kind == "mixed"
    assert [c.name for c in made.colours] == ["grass", "sand"]


def test_the_blob_preset_matches_the_exporters_own_table_exactly() -> None:
    """The load-bearing claim: the derivation and the table the Tiled exporter
    has always written are the same table. If they ever disagree, a set
    generated here opens in Tiled with a brush that paints the wrong tile, and
    nothing in either file says so."""
    from warlock.studio.plotter.tsx import _expected_wangids

    made = wang.blob_wangset(["a", "b"], ["#ff0000", "#00ff00"])
    expected = _expected_wangids(2, 1)
    assert len(expected) == len(made.tiles)
    for local, text in expected.items():
        assert made.tiles[local] == tuple(int(part) for part in text.split(","))


def test_the_blob_preset_matches_the_xml_a_tsx_export_actually_writes() -> None:
    """The same claim, made against the *file* rather than against the helper.

    ``_expected_wangids`` is shared by the reader and the writer, so the test
    above pins the two derivations of the table and nothing pins the element the
    exporter emits from it: the attribute names, the colour derived from a
    terrain's fill, and the order the colours are written in. Those are what a
    Tiled user's brush is built from, and they are the half a generated terrain
    set now depends on -- the set is landed from a record, so nobody looks at
    these tiles until Tiled opens them.
    """
    import xml.etree.ElementTree as ET

    from warlock.studio.plotter.tsx import write_wangsets
    from warlock.studio.tilegrid.tileset import TerrainSpec

    terrains = (
        TerrainSpec(name="wet grass", fill=(106, 153, 78, 255), outline=(63, 91, 46, 255)),
        TerrainSpec(name="dark water", fill=(58, 92, 140, 255), outline=(34, 55, 84, 255)),
    )
    root = ET.Element("tileset")
    write_wangsets(root, terrains)
    wangset = root.find("wangsets/wangset")
    assert wangset is not None

    written_colours = [
        (node.get("name"), node.get("color")) for node in wangset.findall("wangcolor")
    ]
    made = wang.blob_wangset(
        [name for name, _ in written_colours], [colour for _, colour in written_colours]
    )
    # The colours are the terrains, in order and by name -- a wangid slot is a
    # 1-based index into this list, so an order that differed would paint one
    # terrain's tiles for the other.
    assert [colour.name for colour in made.colours] == [t.name for t in terrains]
    assert written_colours[0][1] == "#6a994e"
    assert wangset.get("type") == made.kind

    written_tiles = {
        int(node.get("tileid")): tuple(
            int(part) for part in (node.get("wangid") or "").split(",")
        )
        for node in wangset.findall("wangtile")
    }
    assert written_tiles == made.tiles


def test_the_blob_presets_interior_fill_is_every_slot_set() -> None:
    made = wang.blob_wangset(["a"], ["#ff0000"])
    assert made.tiles[blob.FULL] == (1,) * 8


def test_the_blob_presets_lone_tile_is_every_slot_unset() -> None:
    made = wang.blob_wangset(["a"], ["#ff0000"])
    assert made.tiles[blob.LONE] == (0,) * 8


def test_the_blob_collapse_shows_up_as_corner_rules() -> None:
    """A corner only counts when both its flanking edges do -- which is exactly
    why the preset's kind is ``mixed`` rather than ``corner``."""
    made = wang.blob_wangset(["a"], ["#ff0000"])
    for local, wangid in made.tiles.items():
        del local
        for corner, (first, second) in zip(
            wang.CORNER_SLOTS, ((0, 2), (2, 4), (4, 6), (6, 0)), strict=True
        ):
            if wangid[corner]:
                assert wangid[first] and wangid[second], (
                    "a set corner always has both its flanking edges set"
                )
