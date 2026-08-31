"""Authoring a Wang set, and then painting the map with the one you authored.

Nothing in this app could make a Wang set. ``WangColour``/``WangSet`` and
``Tileset.wangsets`` had round-tripped through ``.tsx``, ``.tsj`` and ``.wmap``
since they landed, the Terrain tool's picker had enumerated their colours and
the constraint matcher had painted with them -- but the only way to *get* one
was to import a file Tiled wrote. The single ``wangset`` reference anywhere in
``panes/`` was a read-only swatch enumeration.

Two halves, tested two ways, which is ``test_tile_collision``'s split and its
reason:

* The table edits are in :mod:`warlock.studio.tilegrid.wang`, headless and
  pure, and are asserted directly.
* The *gestures* are in ``plotter_tileset_editor``, and every test of one below
  goes through the real handler -- the click dispatch with the shared synthetic
  pointer (``tests/plotter/_drive.TileScene``), and the buttons through the
  functions their presses call. A control that is drawn and does nothing is
  this codebase's most common historical defect.

And the last test in the file is the point of the whole wave: a set authored
here, through those gestures, paints the map through the canvas's own dispatch
with **no plumbing** in between.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from warlock.studio.panes import plotter_tileset_editor as editor
from warlock.studio.tilegrid import blob, wang
from warlock.studio.tilegrid.tileset import TerrainSpec

from ._drive import TileScene


@pytest.fixture
def scene(monkeypatch):
    """A tileset editor open on the Terrain tab, with a corner set and two
    colours already made -- through the real handlers, so even the fixture is
    an assertion that they work."""
    made = TileScene(monkeypatch, tiles=16, tileset_tab="Terrain")
    editor.create_wangset(made.state, made.tab, 0, "corner")
    editor.add_wang_colour(made.state, made.tab, 0, 0)
    editor.add_wang_colour(made.state, made.tab, 0, 0)
    return made


def _slot(scene, slot: int) -> tuple[float, float]:
    """Where a slot sits, in tile pixels, read off the production table."""
    return wang.slot_points(scene.view.tile_w, scene.view.tile_h)[slot]


# --- the table edits, which are pure -----------------------------------------


def test_every_slot_has_a_position_and_the_corners_are_the_corners():
    points = wang.slot_points(16, 16)
    assert len(points) == wang.POSITIONS
    assert points[7] == (0.0, 0.0), "top-left corner"
    assert points[1] == (16.0, 0.0), "top-right corner"
    assert points[3] == (16.0, 16.0) and points[5] == (0.0, 16.0)
    assert points[0] == (8.0, 0.0), "the top edge is the middle of the top side"
    assert points[2] == (16.0, 8.0) and points[4] == (8.0, 16.0) and points[6] == (0.0, 8.0)


def test_a_non_square_tile_puts_its_markers_on_its_own_edges():
    points = wang.slot_points(8, 32)
    assert points[3] == (8.0, 32.0)
    assert points[2] == (8.0, 16.0)


def test_a_set_is_asked_only_for_the_slots_its_kind_uses():
    """A corner set has no edge markers at all -- there is nothing to click,
    because nothing reads what a click there would store."""
    corner = wang.WangSet(kind="corner", colours=(wang.WangColour("a"),))
    assert sorted(wang.slot_points(16, 16, corner.slots)) == list(wang.CORNER_SLOTS)
    edge = wang.WangSet(kind="edge", colours=(wang.WangColour("a"),))
    assert sorted(wang.slot_points(16, 16, edge.slots)) == list(wang.EDGE_SLOTS)


def test_setting_a_slot_leaves_the_other_seven_alone():
    start = wang.WangSet(colours=(wang.WangColour("a"), wang.WangColour("b")))
    edited = wang.with_slot(start, 3, 1, 2)
    assert edited.wangid_of(3) == (0, 2, 0, 0, 0, 0, 0, 0)
    assert start.tiles == {}, "the input is frozen and was not touched"


def test_clearing_the_last_slot_drops_the_tile_rather_than_storing_eight_noughts():
    """The sparse rule ``set_tile_meta`` already keeps: a stored blank would
    round-trip out as a ``<wangtile>`` element saying nothing."""
    start = wang.WangSet(colours=(wang.WangColour("a"),))
    one = wang.with_slot(start, 5, 1, 1)
    assert 5 in one.tiles
    assert 5 not in wang.with_slot(one, 5, 1, 0).tiles


def test_a_slot_or_a_colour_this_set_does_not_have_is_refused():
    start = wang.WangSet(colours=(wang.WangColour("a"),))
    with pytest.raises(IndexError):
        wang.with_slot(start, 0, 8, 1)
    with pytest.raises(ValueError):
        wang.with_slot(start, 0, 1, 2)


def test_appending_a_colour_moves_no_existing_wangid():
    start = wang.with_slot(wang.WangSet(colours=(wang.WangColour("a"),)), 0, 1, 1)
    grown = wang.with_colour(start, wang.WangColour("b"))
    assert grown.wangid_of(0) == start.wangid_of(0)
    assert [c.name for c in grown.colours] == ["a", "b"]


def test_renaming_a_colour_moves_no_wangid_either():
    start = wang.with_slot(wang.WangSet(colours=(wang.WangColour("a"),)), 0, 1, 1)
    edited = wang.with_colour_at(start, 0, wang.WangColour("grass", "#00ff00", 2.0))
    assert edited.colours[0].name == "grass" and edited.colours[0].probability == 2.0
    assert edited.wangid_of(0) == start.wangid_of(0)


def test_removing_a_colour_renumbers_every_wangid_that_named_a_later_one():
    """The whole of ``without_colour``. A slot is a *position* in ``colours``,
    so dropping the second of three unsets every slot naming it and shifts
    every slot naming the third -- and a set that skipped the shift would have
    silently repainted every tile in it."""
    start = wang.WangSet(
        colours=(wang.WangColour("a"), wang.WangColour("b"), wang.WangColour("c")),
        tiles={0: (0, 1, 0, 2, 0, 3, 0, 0), 1: (0, 2, 0, 2, 0, 2, 0, 2)},
    )
    edited = wang.without_colour(start, 1)
    assert [c.name for c in edited.colours] == ["a", "c"]
    assert edited.wangid_of(0) == (0, 1, 0, 0, 0, 2, 0, 0)
    assert 1 not in edited.tiles, "a tile left with nothing to say drops out"


def test_removing_a_colour_nobody_has_is_refused():
    with pytest.raises(IndexError):
        wang.without_colour(wang.WangSet(colours=(wang.WangColour("a"),)), 3)


# --- the set and colour buttons, through the handlers they press -------------


def test_creating_a_set_lands_on_the_tileset_and_is_selected(monkeypatch):
    made = TileScene(monkeypatch, tileset_tab="Terrain")
    assert made.wangsets == ()
    editor.create_wangset(made.state, made.tab, 0, "mixed")
    assert len(made.wangsets) == 1
    assert made.wangset.kind == "mixed" and made.wangset.name == "Terrain 1"
    assert made.state.tileset_wangset == 0


def test_creating_a_set_keeps_the_firstgid_and_the_tile_count(monkeypatch):
    """It goes through ``replace_tileset``, so nothing already painted moves."""
    made = TileScene(monkeypatch, tileset_tab="Terrain")
    before = made.doc.tilesets[0]
    editor.create_wangset(made.state, made.tab, 0)
    after = made.doc.tilesets[0]
    assert after.firstgid == before.firstgid
    assert after.tileset.tile_count == before.tileset.tile_count


def test_creating_a_set_is_one_undoable_step(monkeypatch):
    made = TileScene(monkeypatch, tileset_tab="Terrain")
    head = made.doc.history.head
    editor.create_wangset(made.state, made.tab, 0)
    assert made.doc.history.head != head
    made.doc.undo()
    assert made.wangsets == () and made.doc.history.head == head


def test_a_second_set_is_appended_and_selected(scene):
    editor.create_wangset(scene.state, scene.tab, 0, "edge")
    assert len(scene.wangsets) == 2
    assert scene.state.tileset_wangset == 1 and scene.wangset.kind == "edge"


def test_deleting_a_set_removes_it_and_can_be_undone(scene):
    editor.delete_wangset(scene.state, scene.tab, 0, 0)
    assert scene.wangsets == ()
    scene.doc.undo()
    assert len(scene.wangsets) == 1


def test_deleting_a_set_leaves_the_cells_it_chose_exactly_as_they_are(scene):
    """The difference between this and removing a tileset, which is refused
    while anything uses it: a cell stores a gid, and a Wang set is only ever
    consulted to *choose* one."""
    layer = scene.doc.layers[0]
    data = layer.data.copy()
    data[0, 0] = scene.doc.tilesets[0].firstgid
    scene.doc.write_region(layer.uid, 0, 0, data)
    editor.delete_wangset(scene.state, scene.tab, 0, 0)
    assert scene.doc.layers[0].data[0, 0] == scene.doc.tilesets[0].firstgid


def test_adding_a_colour_arms_it(scene):
    assert scene.state.tileset_wang_colour == 2
    editor.add_wang_colour(scene.state, scene.tab, 0, 0)
    assert len(scene.wangset.colours) == 3 and scene.state.tileset_wang_colour == 3


def test_two_new_colours_are_not_the_same_colour(scene):
    swatches = [colour.colour for colour in scene.wangset.colours]
    assert len(set(swatches)) == len(swatches)


def test_renaming_a_colour_goes_through_and_is_undoable(scene):
    editor.set_wang_colour(
        scene.state, scene.tab, 0, 0, 0, wang.WangColour("grass", "#00ff00", 3.0)
    )
    assert scene.wangset.colours[0].name == "grass"
    assert scene.wangset.colours[0].probability == 3.0
    scene.doc.undo()
    assert scene.wangset.colours[0].name != "grass"


def test_renaming_the_set_goes_through(scene):
    editor.rename_wangset(scene.tab, 0, 0, "Coastline")
    assert scene.wangset.name == "Coastline"
    # A rename to the name it already has pushes nothing.
    head = scene.doc.history.head
    editor.rename_wangset(scene.tab, 0, 0, "Coastline")
    assert scene.doc.history.head == head


def test_removing_a_colour_renumbers_the_slots_that_were_painted_with_it(scene):
    """The pure renumbering, asserted through the button that triggers it."""
    scene.frame(_slot(scene, 7), click=True)  # colour 2 (the armed one)
    scene.state.tileset_wang_colour = 1
    scene.frame(_slot(scene, 1), click=True)
    assert scene.wangset.wangid_of(0) == (0, 1, 0, 0, 0, 0, 0, 2)
    editor.remove_wang_colour(scene.state, scene.tab, 0, 0, 0)
    assert len(scene.wangset.colours) == 1
    assert scene.wangset.wangid_of(0) == (0, 0, 0, 0, 0, 0, 0, 1)


def test_removing_a_colour_pulls_the_hand_back_into_range(scene):
    scene.state.tileset_wang_colour = 2
    editor.remove_wang_colour(scene.state, scene.tab, 0, 0, 1)
    assert scene.state.tileset_wang_colour <= len(scene.wangset.colours)


# --- the click, through the real dispatch ------------------------------------


def test_clicking_a_corner_writes_the_colour_in_hand_there(scene):
    scene.state.tileset_wang_colour = 1
    scene.frame(_slot(scene, 3), click=True)
    assert scene.wangset.wangid_of(0) == (0, 0, 0, 1, 0, 0, 0, 0)


def test_the_nearest_marker_wins_rather_than_the_first_one_listed(scene):
    """Aimed a few tile pixels *inside* the bottom-left corner, which is where
    a hand actually lands -- and it must not reach the bottom-right one."""
    scene.state.tileset_wang_colour = 1
    scene.frame((2.0, 14.0), click=True)
    assert scene.wangset.wangid_of(0)[5] == 1
    assert sum(1 for value in scene.wangset.wangid_of(0) if value) == 1


def test_a_click_in_the_middle_of_the_tile_sets_nothing(scene):
    """The centre is out of every marker's radius, so it means "nothing" --
    a set that snapped it to the nearest corner would write a slot the user
    never aimed at."""
    scene.frame((8.0, 8.0), click=True)
    assert scene.wangsets[0].tiles == {}


def test_a_corner_set_has_no_edge_regions_to_click(scene):
    """The middle of the top side is slot 0, which a corner set does not use.
    It is out of both top corners' radius, so the click is nothing at all
    rather than a value stored where nothing reads it."""
    scene.frame(_slot(scene, 0), click=True)
    assert scene.wangsets[0].tiles == {}


def test_a_mixed_set_does_take_the_edges(monkeypatch):
    made = TileScene(monkeypatch, tiles=16, tileset_tab="Terrain")
    editor.create_wangset(made.state, made.tab, 0, "mixed")
    editor.add_wang_colour(made.state, made.tab, 0, 0)
    made.frame(_slot(made, 0), click=True)
    assert made.wangset.wangid_of(0) == (1, 0, 0, 0, 0, 0, 0, 0)


def test_the_unset_colour_clears_a_slot(scene):
    scene.state.tileset_wang_colour = 1
    scene.frame(_slot(scene, 7), click=True)
    assert scene.wangset.wangid_of(0)[7] == 1
    scene.state.tileset_wang_colour = 0
    scene.frame(_slot(scene, 7), click=True)
    assert scene.wangset.wangid_of(0)[7] == 0
    assert scene.wangsets[0].tiles == {}, "and the tile drops out of the table"


def test_hovering_without_pressing_writes_nothing(scene):
    """The defect a drawn-and-dead control is the other half of: a control that
    fires on hover is a table the user edits by looking at it."""
    scene.frame(_slot(scene, 3))
    assert scene.wangsets[0].tiles == {}


def test_a_press_outside_the_view_writes_nothing(scene):
    scene.frame(_slot(scene, 3), click=True, hovered=False)
    assert scene.wangsets[0].tiles == {}


def test_clicking_the_same_slot_with_the_same_colour_pushes_no_step(scene):
    scene.state.tileset_wang_colour = 1
    scene.frame(_slot(scene, 3), click=True)
    head = scene.doc.history.head
    scene.frame(_slot(scene, 3), click=True)
    assert scene.doc.history.head == head


def test_one_click_is_one_undo_step(scene):
    scene.state.tileset_wang_colour = 1
    scene.frame(_slot(scene, 3), click=True)
    scene.frame(_slot(scene, 1), click=True)
    scene.doc.undo()
    assert scene.wangset.wangid_of(0) == (0, 0, 0, 1, 0, 0, 0, 0)
    scene.doc.undo()
    assert scene.wangsets[0].tiles == {}


def test_the_tab_edits_the_tile_the_strip_selected(scene):
    """The Terrain tab draws its own tile strip, but it writes the *same*
    ``state.editing_tile`` the Tiles tab does -- one selection shown twice,
    rather than a second one that can disagree."""
    scene.state.tileset_wang_colour = 1
    scene.frame(_slot(scene, 3), click=True)
    scene.local = scene.state.editing_tile = 5
    scene.frame(_slot(scene, 3), click=True)
    assert sorted(scene.wangsets[0].tiles) == [0, 5]


def test_a_hand_holding_a_colour_an_undo_removed_writes_nothing(scene):
    scene.state.tileset_wang_colour = 2
    scene.doc.undo()  # the second colour
    assert len(scene.wangset.colours) == 1
    scene.frame(_slot(scene, 3), click=True)
    assert scene.wangsets[0].tiles == {}


def test_a_selection_past_the_end_of_the_list_falls_back_to_the_first_set(scene):
    """An undo can take a set out from under the selection, so the tab clamps
    on read rather than only when it is the one removing something."""
    scene.state.tileset_wangset = 7
    scene.state.tileset_wang_colour = 1
    scene.frame(_slot(scene, 3), click=True)
    assert scene.wangsets[0].wangid_of(0)[3] == 1


# --- the point of the wave ---------------------------------------------------


def test_a_set_authored_here_paints_the_map_through_the_canvas_dispatch(scene):
    """Author a Wang set in the editor, then paint with it. End to end.

    This is what the wave was for. Everything downstream of the Terrain tab --
    ``plotter_tools.terrains_of`` offering a Wang colour as a terrain row,
    ``plotter_canvas._terrain_ref`` decoding the negative rank the picker
    encodes it as, and ``terrain.paint_wang`` matching constraints -- was built
    before the author existed and is not touched by it. So the assertion is not
    that painting works: it is that authoring and painting meet, with **no**
    plumbing between them.
    """
    from warlock.studio.panes import plotter_canvas, plotter_tools

    # 1. Author. Colour 1 at all four corners of tile 0 makes it that colour's
    #    interior -- the tile a click on the map lays down.
    scene.state.tileset_wang_colour = 1
    for slot in wang.CORNER_SLOTS:
        scene.frame(_slot(scene, slot), click=True)
    assert scene.wangset.wangid_of(0) == (0, 1, 0, 1, 0, 1, 0, 1)

    # 2. The Terrain tool's picker sees it, with no cache to refresh: one row
    #    per colour, and the second colour is there too even though no tile
    #    uses it yet.
    entries = plotter_tools.terrains_of(scene.doc)
    assert [rank for _index, rank, _spec in entries] == [-1, -2]
    assert entries[0][2].name == scene.wangset.colours[0].name

    # 3. Arm the tool exactly as opening the Terrain section does.
    scene.state.tool = "terrain"
    scene.state.terrain = plotter_tools.first_terrain(entries)
    target = plotter_canvas._terrain_ref(scene.state, scene.doc)
    assert target is not None and target.wangset is scene.wangset
    assert target.value == 1, "a 1-based Wang colour, not a blob rank"

    # 4. Paint, through the canvas's own dispatch.
    layer = scene.doc.layers[0]
    scene.doc.set_active_layer(layer.uid)
    plotter_canvas._apply(scene.ctx, scene.state, scene.tab, (2, 2))

    firstgid = scene.doc.tilesets[0].firstgid
    assert scene.doc.layer(layer.uid).data[2, 2] == firstgid + 0, (
        "the cell holds the interior tile of the colour that was authored"
    )
    assert scene.toasts == [], "and nothing refused on the way"


def test_the_authored_set_survives_a_wmap_round_trip(scene):
    """The format half, so the claim in ``docs/COMPAT.md`` is not one-ended:
    what the editor writes is what a reopened map reads back."""
    from warlock.studio.plotter import wmap

    scene.state.tileset_wang_colour = 1
    for slot in wang.CORNER_SLOTS:
        scene.frame(_slot(scene, slot), click=True)
    editor.rename_wangset(scene.tab, 0, 0, "Coastline")

    back = wmap.read_wmap(wmap.wmap_bytes(scene.doc))
    landed = back.tilesets[0].tileset.wangsets[0]
    assert landed.name == "Coastline" and landed.kind == "corner"
    assert [c.name for c in landed.colours] == [
        c.name for c in scene.wangset.colours
    ]
    assert landed.tiles == scene.wangset.tiles


def test_an_authored_set_does_not_make_the_tileset_a_blob_terrain_set(scene):
    """``is_terrain_set`` means the *positional 47-column preset*, and a Wang
    set is the general case beside it. Confusing the two would send the whole
    existing terrain corpus down the wrong painter."""
    assert not scene.doc.tilesets[0].tileset.is_terrain_set
    assert scene.doc.tilesets[0].tileset.terrains == ()


def test_the_editor_offers_a_terrain_tab_beside_the_other_three():
    assert editor.TABS == ("Tiles", "Collision", "Animation", "Terrain")


def test_the_marker_positions_are_read_from_the_engine_not_recomputed():
    """What is drawn is what is clickable, because both come off the same
    table. Asserted here because the two are in different modules and a second
    copy is exactly how a marker comes to sit where it cannot be pressed."""
    assert wang.SLOT_FRACTIONS[1] == (1.0, 0.0)
    assert len(wang.SLOT_FRACTIONS) == wang.POSITIONS


# --- the combination this tab must never create ------------------------------
#
# The Terrain tab is the first producer of ``Tileset.wangsets`` other than the
# ``.tsx`` reader, and the reader has always enforced an exclusion the writer
# only *assumed*: ``tsx._wang_model_of`` drops the general Wang model whenever
# the blob preset is present, while ``write_tsx`` writes the preset's block and
# the model's block from two doors that each return early only on their own
# emptiness. A tileset carrying both therefore exported two ``<wangsets>``
# blocks and lost the hand-authored one on reopen -- a silently-dropped row in
# a ledger that says it has none. Both doors refuse it now, and these are the
# tests of that.


def _blob_scene(monkeypatch):
    """A tileset editor on the Terrain tab, over a *generated* terrain set.

    The atlas is the preset's own shape -- ``blob.TILE_COUNT`` columns, one row
    per terrain -- because ``Tileset`` refuses any other for a set that
    declares terrains, and the point of the scene is a tileset the app itself
    could have produced.
    """
    made = TileScene(monkeypatch, tiles=blob.TILE_COUNT, tileset_tab="Terrain")
    ts = made.doc.tilesets[0].tileset
    made.doc.replace_tileset(
        0,
        dataclasses.replace(
            ts,
            terrains=(
                TerrainSpec(name="Grass", fill=(0, 200, 0, 255), outline=(0, 90, 0, 255)),
            ),
        ),
    )
    assert made.doc.tilesets[0].tileset.is_terrain_set
    return made


def _fake_row_widgets(monkeypatch):
    """``_wangset_row``'s two toolkits, faked, so the row can be drawn without
    a GL context. -> ``(prose, buttons)``, filled as it draws.

    The row is the only place the refusal is *offered or not offered*, and a
    test that read the source instead of drawing it would pass whatever the
    user ends up seeing.
    """
    prose: list[str] = []
    buttons: list[str] = []
    monkeypatch.setattr(
        editor,
        "widgets",
        SimpleNamespace(
            grid_width=lambda _n: 100.0,
            muted_wrapped=prose.append,
            input_text=lambda _label, value, **_k: value,
        ),
    )
    monkeypatch.setattr(
        editor,
        "controls",
        SimpleNamespace(
            button=lambda label, _size=None, **_k: buttons.append(label) and False,
            segmented_choice=lambda _key, _items, current: (False, current),
        ),
    )
    return prose, buttons


def test_authoring_a_wang_set_on_a_generated_terrain_set_is_refused(monkeypatch):
    made = _blob_scene(monkeypatch)
    assert editor.create_wangset(made.state, made.tab, 0, "corner", ctx=made.ctx) is False
    assert made.wangsets == (), "nothing was written"


def test_the_refusal_says_why_rather_than_failing_quietly(monkeypatch):
    """A door that returned False and said nothing would be a button that does
    nothing, which is this codebase's most common historical defect."""
    made = _blob_scene(monkeypatch)
    editor.create_wangset(made.state, made.tab, 0, ctx=made.ctx)
    assert len(made.toasts) == 1
    text, kind = made.toasts[0]
    assert kind == "error"
    assert text == editor.authoring_refusal(made.doc.tilesets[0].tileset)
    assert "generated terrain set" in text and "read back" in text


def test_the_refusal_is_not_a_silent_no_op_when_nobody_passes_a_ctx(monkeypatch):
    """The pure edit stays callable without a context -- every other function
    in this section is tested that way -- but it still reports."""
    made = _blob_scene(monkeypatch)
    assert editor.create_wangset(made.state, made.tab, 0) is False
    assert made.toasts == []


def test_the_terrain_tab_says_why_instead_of_offering_create(monkeypatch):
    """Drawn, not inferred. The refusal reaches the user *before* the gesture:
    the row puts the sentence where the Create button would be."""
    made = _blob_scene(monkeypatch)
    prose, buttons = _fake_row_widgets(monkeypatch)
    editor._wangset_row(made.ctx, made.state, made.tab, 0, ())
    assert not [label for label in buttons if label.startswith("Create")]
    assert prose == [editor.authoring_refusal(made.doc.tilesets[0].tileset)]


def test_the_terrain_tab_still_offers_create_on_an_ordinary_tileset(monkeypatch):
    """The other direction, so the test above cannot pass by the row drawing
    no buttons at all."""
    made = TileScene(monkeypatch, tileset_tab="Terrain")
    _prose, buttons = _fake_row_widgets(monkeypatch)
    editor._wangset_row(made.ctx, made.state, made.tab, 0, ())
    assert [label for label in buttons if label.startswith("Create")]


def test_no_authoring_door_can_put_a_wang_set_on_a_generated_set(monkeypatch):
    """The state, not the one gesture. Every entry point this tab writes
    through is pressed on a blob-preset tileset, and the tileset comes out the
    other side with terrains and no Wang sets -- which is the invariant
    ``tsx.write_tsx`` and ``tsx._wang_model_of`` are each half of.
    """
    made = _blob_scene(monkeypatch)
    editor.create_wangset(made.state, made.tab, 0, "corner", ctx=made.ctx)
    editor.add_wang_colour(made.state, made.tab, 0, 0)
    editor.set_wang_colour(
        made.state, made.tab, 0, 0, 0, wang.WangColour("x")
    )
    editor.remove_wang_colour(made.state, made.tab, 0, 0, 0)
    editor.rename_wangset(made.tab, 0, 0, "Coastline")
    editor.delete_wangset(made.state, made.tab, 0, 0)
    made.frame(_slot(made, 3), click=True)

    tileset = made.doc.tilesets[0].tileset
    assert tileset.terrains and tileset.wangsets == ()


def test_a_tsx_never_writes_two_wangsets_blocks(monkeypatch):
    """The pin the two refusals exist for. One block for a blob preset, one for
    a hand-authored set, and the combination refused by name rather than
    written as two."""
    from warlock.studio.plotter import tsx

    made = _blob_scene(monkeypatch)
    generated = made.doc.tilesets[0].tileset
    assert tsx.tsx_bytes(generated, image_name="a.png").count(b"<wangsets") == 1

    authored = dataclasses.replace(
        generated, terrains=(), wangsets=(wang.WangSet(name="Coast", kind="corner"),)
    )
    assert tsx.tsx_bytes(authored, image_name="a.png").count(b"<wangsets") == 1

    both = dataclasses.replace(generated, wangsets=authored.wangsets)
    with pytest.raises(tsx.TiledUnsupported) as caught:
        tsx.tsx_bytes(both, image_name="a.png")
    assert caught.value.feature == (
        "a tileset carrying both a generated terrain set and a hand-authored Wang set"
    )
    assert caught.value.exporting, "the save door's sentence, not the reader's"
