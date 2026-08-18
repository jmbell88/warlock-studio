"""The map document: uid addressing, the no-op rule, and what dirty means.

Three properties are pinned here because each of them was a real bug in one of
the editors this one is modelled on. An edit must survive a reorder. A step that
changes nothing must not be pushed. And "unsaved" must be a comparison against
the saved head rather than a latching flag, or undoing back to the saved state
leaves a document that asks to be saved forever.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter.tilemap import MapDoc, MapObject, ObjectLayer, TileLayer, new_uid
from warlock.studio.tilegrid import gid
from warlock.studio.tilegrid.tileset import Tileset


def _tileset(name: str = "t", tiles: int = 4) -> Tileset:
    pixels = np.zeros((16, 16 * tiles, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name=name, pixels=pixels, tile_w=16, tile_h=16)


def _doc() -> MapDoc:
    doc = MapDoc(8, 6, 16, 16)
    doc.add_tileset(_tileset())
    return doc


# --- shape --------------------------------------------------------------------


def test_a_new_layer_is_the_maps_shape_and_dtype():
    doc = _doc()
    layer = doc.add_tile_layer()
    assert layer.data.shape == (6, 8)
    assert layer.data.dtype == gid.DTYPE
    assert (doc.pixel_width, doc.pixel_height) == (128, 96)


def test_a_dimension_outside_the_bounds_is_refused():
    with pytest.raises(ValueError):
        MapDoc(0, 10, 16, 16)
    with pytest.raises(ValueError):
        MapDoc(10, 10, 16, 0)
    with pytest.raises(ValueError):
        MapDoc(99999, 10, 16, 16)


# --- tilesets -----------------------------------------------------------------


def test_firstgids_are_contiguous_and_one_based():
    """Zero means "no tile" in every Tiled file ever written, so the first real
    id is 1 and each set begins where the last one ended."""
    doc = MapDoc(4, 4, 16, 16)
    first = doc.add_tileset(_tileset("a", tiles=4))
    second = doc.add_tileset(_tileset("b", tiles=3))
    assert (first.firstgid, first.last_gid) == (1, 4)
    assert (second.firstgid, second.last_gid) == (5, 7)
    assert doc.next_firstgid == 8


def test_resolve_masks_the_flags_off_for_the_caller():
    """Callers never have to remember to mask -- forgetting is how a flipped
    tile comes to render as an out-of-range id."""
    doc = _doc()
    flipped = gid.compose(2, flip_h=True, flip_d=True)
    resolved = doc.resolve(flipped)
    assert resolved is not None and resolved[1] == 1  # local id, firstgid 1
    assert doc.resolve(0) is None
    assert doc.resolve(999) is None


# --- uid addressing -----------------------------------------------------------


def test_an_edit_survives_a_reorder_of_the_layer_list():
    """The rule the whole edit module is written around. After a move, an undo
    has to land on the layer the edit was made to, not on whatever now sits at
    the index it was made at."""
    doc = _doc()
    bottom, top = doc.add_tile_layer("bottom"), doc.add_tile_layer("top")
    doc.write_region(bottom.uid, 0, 0, np.array([[1]], gid.DTYPE))
    doc.move_layer(bottom.uid, 1)
    assert [layer.name for layer in doc.layers] == ["top", "bottom"]

    doc.undo()  # the move
    doc.undo()  # the write
    assert int(bottom.data[0, 0]) == 0
    assert int(top.data[0, 0]) == 0


def test_removing_a_layer_and_undoing_brings_back_the_same_object():
    """Re-insertion has to restore the *same* uid, or every patch recorded
    against the layer beforehand is stranded on a number nothing answers to."""
    doc = _doc()
    layer = doc.add_tile_layer("ground")
    doc.write_region(layer.uid, 1, 1, np.array([[3]], gid.DTYPE))
    doc.remove_layer(layer.uid)
    assert doc.layer(layer.uid) is None

    doc.undo()
    restored = doc.layer(layer.uid)
    assert restored is layer
    assert int(restored.data[1, 1]) == 3


def test_the_active_layer_follows_a_removal_by_uid():
    """By uid, so the fallback is a layer that still exists rather than
    whichever one slid into the removed one's index."""
    doc = _doc()
    first = doc.add_tile_layer("a")
    second = doc.add_tile_layer("b")
    assert doc.active_layer == second.uid
    doc.remove_layer(second.uid)
    assert doc.active_layer == first.uid
    doc.remove_layer(first.uid)
    assert doc.active_layer is None


def test_choosing_a_layer_pushes_no_step():
    """View state. An undoable "which layer am I on" would move the head and
    make a document ask to be saved because the user clicked a different row."""
    doc = _doc()
    a = doc.add_tile_layer("a")
    doc.add_tile_layer("b")
    doc.mark_saved()
    doc.set_active_layer(a.uid)
    assert not doc.dirty


# --- the no-op rule -----------------------------------------------------------


def test_writing_what_is_already_there_pushes_nothing():
    doc = _doc()
    layer = doc.add_tile_layer()
    block = np.array([[1, 2]], gid.DTYPE)
    assert doc.write_region(layer.uid, 0, 0, block) is True
    head = doc.history.head
    assert doc.write_region(layer.uid, 0, 0, block) is False
    assert doc.history.head == head


def test_renaming_a_layer_to_its_own_name_pushes_nothing():
    doc = _doc()
    layer = doc.add_tile_layer("ground")
    head = doc.history.head
    doc.set_layer_props(layer.uid, name="ground", visible=True)
    assert doc.history.head == head
    doc.set_layer_props(layer.uid, name="floor")
    assert doc.history.head != head


def test_setting_an_object_to_its_own_values_pushes_nothing():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="spawn", kind="point"))
    head = doc.history.head
    doc.set_object(layer.uid, obj.uid, name="spawn")
    assert doc.history.head == head


def test_a_resize_to_the_same_size_pushes_nothing():
    doc = _doc()
    doc.add_tile_layer()
    head = doc.history.head
    assert doc.resize(8, 6) is False
    assert doc.history.head == head


# --- dirty --------------------------------------------------------------------


def test_dirty_is_a_comparison_and_undo_can_clear_it():
    """A latching flag would call an undone document unsaved forever."""
    doc = _doc()
    layer = doc.add_tile_layer()
    doc.mark_saved()
    assert not doc.dirty
    doc.write_region(layer.uid, 0, 0, np.array([[1]], gid.DTYPE))
    assert doc.dirty
    doc.undo()
    assert not doc.dirty


def test_mark_saved_takes_the_head_the_encoder_actually_wrote():
    """The document routinely moves on while a save runs on a task thread, and
    marking the live head would call those later edits saved."""
    doc = _doc()
    layer = doc.add_tile_layer()
    captured = doc.history.head
    doc.write_region(layer.uid, 0, 0, np.array([[1]], gid.DTYPE))
    doc.mark_saved(captured)
    assert doc.dirty


# --- geometry -----------------------------------------------------------------


def test_a_resize_keeps_the_overlap_and_moves_the_objects_with_it():
    doc = _doc()
    tiles = doc.add_tile_layer()
    doc.write_region(tiles.uid, 0, 0, np.array([[7]], gid.DTYPE))
    objects = doc.add_object_layer()
    obj = doc.add_object(objects.uid, MapObject(uid=new_uid(), name="s", kind="point", x=0, y=0))

    doc.resize(10, 8, offset_x=2, offset_y=1)
    assert doc.layer(tiles.uid).data.shape == (8, 10)
    assert int(doc.layer(tiles.uid).data[1, 2]) == 7
    # Objects are absolute pixels; leaving them put would detach every trigger
    # volume from the geometry it was drawn around.
    assert (obj.x, obj.y) == (32.0, 16.0)


def test_a_resize_undoes_to_the_exact_arrays_it_replaced():
    doc = _doc()
    tiles = doc.add_tile_layer()
    doc.write_region(tiles.uid, 3, 3, np.array([[4]], gid.DTYPE))
    before = doc.layer(tiles.uid).data.copy()
    doc.resize(2, 2)
    doc.undo()
    assert np.array_equal(doc.layer(tiles.uid).data, before)
    assert (doc.width, doc.height) == (8, 6)


def test_a_resized_layer_does_not_alias_the_history():
    """The applied array is copied, or a later stamp writes into the undo
    step that is holding the previous shape."""
    doc = _doc()
    tiles = doc.add_tile_layer()
    doc.resize(4, 4)
    doc.write_region(tiles.uid, 0, 0, np.array([[9]], gid.DTYPE))
    doc.undo()  # the write
    doc.undo()  # the resize
    assert doc.layer(tiles.uid).data.shape == (6, 8)
    assert not doc.layer(tiles.uid).data.any()


# --- tile size ----------------------------------------------------------------
#
# The map's tile size is the *grid cell*, not the tileset's slicing. ``render``
# says so directly -- "a tileset whose tiles are larger than the map's grid is
# ordinary, a 32px map with 48px trees" -- so changing it re-places what is
# painted and never re-slices or invalidates it. That is the whole reason this
# is allowed on a map with content, where changing the *projection* is not.


def test_set_tile_size_moves_the_grid_and_leaves_the_painting_alone():
    doc = _doc()
    tiles = doc.add_tile_layer()
    doc.write_region(tiles.uid, 1, 1, np.array([[7]], gid.DTYPE))

    assert doc.set_tile_size(32, 32) is True
    assert (doc.tile_w, doc.tile_h) == (32, 32)
    assert (doc.pixel_width, doc.pixel_height) == (256, 192)
    # The gids are untouched: a gid names a tile, and which tile it names has
    # nothing to do with how big the cell it sits in is.
    assert int(doc.layer(tiles.uid).data[1, 1]) == 7
    assert doc.layer(tiles.uid).data.shape == (6, 8)


def test_set_tile_size_scales_objects_so_they_keep_their_cells():
    """``resize``'s rule, for the other way the grid can move: an object is
    absolute pixels, so a cell that changes size leaves it describing a
    different part of the map unless it travels with the geometry."""
    doc = _doc()
    objects = doc.add_object_layer()
    obj = doc.add_object(
        objects.uid, MapObject(uid=new_uid(), name="s", kind="rect", x=32, y=16, w=16, h=16)
    )
    doc.set_tile_size(32, 32)
    assert (obj.x, obj.y) == (64.0, 32.0)
    assert (obj.w, obj.h) == (32.0, 32.0)


def test_set_tile_size_undoes_to_the_size_and_objects_it_replaced():
    doc = _doc()
    objects = doc.add_object_layer()
    obj = doc.add_object(
        objects.uid, MapObject(uid=new_uid(), name="s", kind="rect", x=32, y=16, w=16, h=16)
    )
    doc.set_tile_size(8, 24)
    doc.undo()
    assert (doc.tile_w, doc.tile_h) == (16, 16)
    assert (obj.x, obj.y, obj.w, obj.h) == (32.0, 16.0, 16.0, 16.0)


def test_set_tile_size_scales_a_polygons_vertices_too():
    """Five of the eight shapes state an extent as ``w``/``h`` and two state it
    as vertices. A scale that knew only the first spelling would leave every
    polygon at its old size, which is the one case where the map visibly stops
    matching itself."""
    from warlock.studio.plotter._map_model import Polygon

    doc = _doc()
    objects = doc.add_object_layer()
    obj = doc.add_object(
        objects.uid,
        MapObject(
            uid=new_uid(),
            name="zone",
            shape=Polygon(points=((0.0, 0.0), (16.0, 0.0), (16.0, 16.0))),
        ),
    )
    doc.set_tile_size(32, 32)
    assert obj.shape.points == ((0.0, 0.0), (32.0, 0.0), (32.0, 32.0))
    doc.undo()
    assert obj.shape.points == ((0.0, 0.0), (16.0, 0.0), (16.0, 16.0))


def test_setting_the_tile_size_it_already_has_pushes_nothing():
    doc = _doc()
    head = doc.history.head
    assert doc.set_tile_size(16, 16) is False
    assert doc.history.head == head


@pytest.mark.parametrize("bad", [0, -4])
def test_set_tile_size_refuses_a_dimension_that_is_not_one(bad):
    doc = _doc()
    with pytest.raises(ValueError):
        doc.set_tile_size(bad, 16)


# --- objects ------------------------------------------------------------------


def test_objects_add_edit_and_remove_by_uid():
    doc = _doc()
    layer = doc.add_object_layer()
    first = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="rect", w=8, h=8))
    second = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="b", kind="point"))

    doc.set_object(layer.uid, first.uid, x=4.0, y=5.0, obj_class="Trigger")
    assert (first.x, first.y, first.obj_class) == (4.0, 5.0, "Trigger")

    doc.remove_object(layer.uid, second.uid)
    assert [o.uid for o in doc.layer(layer.uid).objects] == [first.uid]
    doc.undo()
    assert [o.uid for o in doc.layer(layer.uid).objects] == [first.uid, second.uid]
    doc.undo()
    assert (first.x, first.y) == (0.0, 0.0)


def test_an_unknown_object_kind_is_refused():
    """The string constructor covers Tiled's core shapes, and nothing else."""
    with pytest.raises(ValueError):
        MapObject(uid=new_uid(), kind="bezier")


# --- persistent ids -------------------------------------------------------


def test_a_new_layer_and_object_get_a_persistent_id_from_the_doc_counters():
    """``id`` is Tiled's own persistent id, distinct from ``uid``: the doc's
    two counters mint it at creation, one namespace for both layer kinds and
    a separate one for objects."""
    doc = _doc()
    tiles = doc.add_tile_layer()
    objects = doc.add_object_layer()
    assert (tiles.id, objects.id) == (1, 2)
    obj = doc.add_object(objects.uid, MapObject(uid=new_uid(), name="a", kind="point"))
    assert obj.id == 1
    assert (doc.next_layer_id, doc.next_object_id) == (3, 2)


def test_undo_of_an_add_does_not_reuse_its_id():
    """An object-typed property may go on naming a deleted object's id, so the
    counter must not walk back just because the add that minted it was
    undone."""
    doc = _doc()
    layer = doc.add_object_layer()
    first = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="point"))
    assert first.id == 1
    doc.undo()  # undoes the add
    assert doc.layer(layer.uid).objects == []
    second = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="b", kind="point"))
    assert second.id == 2
    assert doc.next_object_id == 3


def test_an_object_layer_carries_a_draw_order():
    doc = _doc()
    layer = doc.add_object_layer()
    assert layer.draworder == "topdown"
    doc.set_layer_props(layer.uid, draworder="index")
    assert doc.layer(layer.uid).draworder == "index"
    with pytest.raises(ValueError):
        doc.set_layer_props(layer.uid, draworder="sideways")
    doc.undo()
    assert doc.layer(layer.uid).draworder == "topdown"


def test_a_tile_layers_props_are_unaffected_by_the_draw_order():
    """One ``_apply_layer_props`` serves both layer kinds, and a tile layer's
    snapshot carries no ``draworder`` at all -- so the hook has to ask the
    values, not the layer."""
    doc = _doc()
    tiles = doc.add_tile_layer()
    assert "draworder" not in tiles.snapshot()
    doc.set_layer_props(tiles.uid, name="Ground", draworder="index")
    assert not hasattr(doc.layer(tiles.uid), "draworder")
    assert doc.layer(tiles.uid).name == "Ground"


def test_a_tile_op_on_an_object_layer_is_a_key_error():
    doc = _doc()
    layer = doc.add_object_layer()
    with pytest.raises(KeyError):
        doc.write_region(layer.uid, 0, 0, np.array([[1]], gid.DTYPE))


def test_a_region_outside_the_layer_is_refused():
    doc = _doc()
    layer = doc.add_tile_layer()
    with pytest.raises(ValueError):
        doc.write_region(layer.uid, 7, 0, np.array([[1, 1]], gid.DTYPE))
    with pytest.raises(ValueError):
        doc.write_region(layer.uid, -1, 0, np.array([[1]], gid.DTYPE))


# --- construction -------------------------------------------------------------


def test_a_document_built_by_construction_starts_clean():
    """What every reader relies on: a file that has just been opened is not
    unsaved, which means the layers go in directly rather than through the
    mutators that would push a step apiece."""
    layer = TileLayer(uid=new_uid(), name="l", data=gid.empty_layer(4, 4))
    other = ObjectLayer(uid=new_uid(), name="o")
    doc = MapDoc(4, 4, 16, 16, layers=[layer, other])
    assert not doc.dirty
    assert doc.active_layer == layer.uid


# --- strokes and projection ---------------------------------------------------
#
# A drag is one gesture, so it is one step. Before the stroke session existed
# the canvas called ``write_region`` on every frame the button was down, which
# made a forty-cell stamp forty things to undo and forty entries against the
# history's byte budget.


def _terrain_doc():
    from ._terrainset import terrain_tileset

    doc = MapDoc(8, 8, 8, 8)
    ref = doc.add_tileset(terrain_tileset(tile_w=8, tile_h=8))
    layer = doc.add_tile_layer("Ground")
    return doc, ref, layer


def test_a_stroke_is_one_undo_step_however_many_cells_it_wrote():
    from warlock.studio.plotter import terrain

    doc, ref, layer = _terrain_doc()
    # Depth, not ``head``: head is a process-wide *serial*, so comparing it to
    # ``head + 1`` is only accidentally right when nothing else pushed between.
    depth = len(doc.history)
    doc.begin_stroke(layer.uid)
    for x in range(6):
        region = terrain.paint_terrain(layer.data, x, 3, 0, ref)
        if region is not None:
            doc.stroke_write(*region)
    assert len(doc.history) == depth, "a stroke in progress pushes nothing"
    assert doc.end_stroke() is True
    assert len(doc.history) == depth + 1
    painted = layer.data.copy()
    doc.undo()
    assert not layer.data.any()
    doc.redo()
    assert np.array_equal(layer.data, painted)


def test_a_stroke_that_changed_nothing_pushes_nothing():
    doc, _ref, layer = _terrain_doc()
    depth = len(doc.history)
    doc.begin_stroke(layer.uid)
    assert doc.end_stroke() is False
    assert len(doc.history) == depth


def test_ending_a_stroke_that_was_never_begun_is_harmless():
    """A release can be missed -- focus lost, a tab switched, Esc, a save begun
    mid-drag -- and every recovery path would otherwise have to know whether a
    session was open."""
    doc, _ref, _layer = _terrain_doc()
    assert doc.stroking is False
    assert doc.end_stroke() is False
    assert doc.end_stroke() is False


def test_beginning_a_second_stroke_closes_the_first():
    from warlock.studio.plotter import terrain

    doc, ref, layer = _terrain_doc()
    depth = len(doc.history)
    doc.begin_stroke(layer.uid)
    doc.stroke_write(*terrain.paint_terrain(layer.data, 1, 1, 0, ref))
    doc.begin_stroke(layer.uid)
    assert len(doc.history) == depth + 1


def test_an_undo_during_a_stroke_commits_it_first():
    """A stroke writes the live array with no history, so undoing through an
    open session used to reverse the step *before* it and leave the
    uncommitted paint sitting on the layer."""
    from warlock.studio.plotter import terrain

    doc, ref, layer = _terrain_doc()
    doc.write_region(layer.uid, *terrain.paint_terrain(layer.data, 6, 6, 0, ref))
    settled = layer.data.copy()

    doc.begin_stroke(layer.uid)
    doc.stroke_write(*terrain.paint_terrain(layer.data, 1, 1, 0, ref))
    painted = layer.data.copy()
    assert not np.array_equal(painted, settled)

    assert doc.undo() is True
    assert np.array_equal(layer.data, settled), "the stroke is the step undone"
    assert doc.stroking is False
    doc.undo()
    assert not layer.data.any(), "and the step before it is the one before"
    doc.redo()
    doc.redo()
    assert np.array_equal(layer.data, painted)


def test_a_redo_during_a_stroke_commits_it_too():
    """Redoing with a session open would replay a step onto pixels the session
    has already changed underneath it."""
    from warlock.studio.plotter import terrain

    doc, ref, layer = _terrain_doc()
    doc.write_region(layer.uid, *terrain.paint_terrain(layer.data, 6, 6, 0, ref))
    doc.undo()
    doc.begin_stroke(layer.uid)
    doc.stroke_write(*terrain.paint_terrain(layer.data, 1, 1, 0, ref))
    painted = layer.data.copy()
    # Committing the stroke clears the redo branch, so there is nothing left to
    # replay -- and the paint survives, which is the whole point.
    assert doc.redo() is False
    assert doc.stroking is False
    assert np.array_equal(layer.data, painted)


def test_a_stroke_covers_only_the_cells_that_moved():
    """The step is the union of what was written, not the whole layer -- the
    dirty-rect rule ``TilePatchEdit`` exists for."""
    from warlock.studio.plotter import terrain

    doc, ref, layer = _terrain_doc()
    doc.begin_stroke(layer.uid)
    doc.stroke_write(*terrain.paint_terrain(layer.data, 2, 2, 0, ref))
    doc.end_stroke()
    assert doc.history.top.after.shape == (3, 3)


def test_replacing_a_tileset_keeps_every_painted_cell():
    from warlock.studio.plotter import terrain
    from warlock.studio.tilegrid.tileset import repolish

    doc, ref, layer = _terrain_doc()
    region = terrain.paint_terrain(layer.data, 3, 3, 2, ref)
    doc.write_region(layer.uid, *region)
    before = layer.data.copy()
    painted = np.array(ref.tileset.pixels)
    painted[..., 0] = 200
    doc.replace_tileset(0, repolish(ref.tileset, painted))
    assert np.array_equal(layer.data, before)
    assert int(doc.tilesets[0].tileset.pixels[..., 0].max()) == 200
    assert doc.tilesets[0].firstgid == ref.firstgid


def test_replacing_a_tileset_of_a_different_size_is_refused():
    """The one case that would renumber, and so the one that is named rather
    than accepted."""
    from ._terrainset import terrain_tileset

    doc, _ref, _layer = _terrain_doc()
    smaller = terrain_tileset(count=2, tile_w=8, tile_h=8)
    with pytest.raises(ValueError, match="renumber"):
        doc.replace_tileset(0, smaller)


def test_a_replace_undoes_back_to_the_original_art():
    from warlock.studio.tilegrid.tileset import repolish

    doc, ref, _layer = _terrain_doc()
    original = np.array(ref.tileset.pixels)
    painted = np.array(ref.tileset.pixels)
    painted[..., 0] = 200
    doc.replace_tileset(0, repolish(ref.tileset, painted))
    doc.undo()
    assert np.array_equal(doc.tilesets[0].tileset.pixels, original)


def test_the_maps_own_properties_are_undoable():
    from warlock.studio.plotter.tsx import Prop

    doc = _doc()
    depth = len(doc.history)
    doc.set_map_properties({"theme": Prop("string", "cave")})
    assert doc.properties == {"theme": Prop("string", "cave")}
    assert len(doc.history) == depth + 1

    doc.set_map_properties({"theme": Prop("string", "cave")})
    assert len(doc.history) == depth + 1, "a no-op pushes nothing"

    doc.undo()
    assert doc.properties == {}
    doc.redo()
    assert doc.properties == {"theme": Prop("string", "cave")}


def test_map_metadata_is_validated_and_undoable_as_one_step():
    doc = _doc()
    depth = len(doc.history)
    doc.set_map_settings(
        class_name="Dungeon",
        parallax_origin=(3, 5),
        renderorder="left-up",
        backgroundcolor="#102030",
        skew_x=8,
        skew_y=-2,
    )
    assert doc.map_settings() == {
        "class_name": "Dungeon",
        "parallax_origin": (3.0, 5.0),
        "renderorder": "left-up",
        "backgroundcolor": "#102030",
        "skew_x": 8,
        "skew_y": -2,
    }
    assert len(doc.history) == depth + 1
    doc.undo()
    assert doc.map_settings()["class_name"] == ""
    assert doc.map_settings()["backgroundcolor"] is None
    doc.redo()
    assert doc.map_settings()["class_name"] == "Dungeon"


def test_invalid_map_metadata_changes_nothing_and_pushes_nothing():
    doc = _doc()
    before, depth = doc.map_settings(), len(doc.history)
    with pytest.raises(ValueError, match="background colour"):
        doc.set_map_settings(class_name="Would leak", backgroundcolor="purple")
    assert doc.map_settings() == before
    assert len(doc.history) == depth


def test_a_map_properties_edit_owns_its_dicts():
    """Handed the live mapping, a "before" that moves with the document
    restores nothing."""
    from warlock.studio.plotter.tsx import Prop

    doc = _doc()
    doc.set_map_properties({"a": Prop("int", 1)})
    edit = doc.history.top
    assert edit.before is not doc.properties
    assert edit.after is not doc.properties

    doc.properties["b"] = Prop("int", 2)
    assert "b" not in edit.after
    doc.undo()
    assert doc.properties == {}


def test_a_layer_properties_edit_keeps_its_two_sides_apart():
    """``set_layer_props`` builds ``after`` by spreading ``before``, so the two
    shared one mapping whenever properties was not the field being changed.
    Nothing mutated it in place, so this is the hazard removed rather than a bug
    fixed -- but the two sides of an edit should never be one object."""
    from warlock.studio.plotter.tsx import Prop

    doc = _doc()
    layer = doc.add_tile_layer()
    layer.properties = {"a": Prop("int", 1)}
    doc.set_layer_props(layer.uid, name="Renamed")
    edit = doc.history.top
    assert edit.before["properties"] is not edit.after["properties"]
    assert edit.before["properties"] is not layer.properties


def test_layer_properties_survive_an_undo():
    from warlock.studio.plotter.tsx import Prop

    doc = _doc()
    layer = doc.add_tile_layer()
    doc.set_layer_props(layer.uid, properties={"kind": Prop("string", "floor")})
    assert layer.properties == {"kind": Prop("string", "floor")}
    doc.undo()
    assert layer.properties == {}


def test_locking_a_layer_is_undoable_and_a_no_op_pushes_nothing():
    """``set_layer_props`` filters through ``snapshot``, so the edit records the
    new field for free -- but undo replays through ``_apply_layer_props``, and a
    field missing from *that* is one the user can set and never take back."""
    doc = _doc()
    layer = doc.add_tile_layer()
    depth = len(doc.history)

    doc.set_layer_props(layer.uid, locked=True)
    assert layer.locked is True
    assert len(doc.history) == depth + 1

    doc.set_layer_props(layer.uid, locked=True)
    assert len(doc.history) == depth + 1, "a no-op pushes nothing"

    doc.undo()
    assert layer.locked is False, "the hook has to assign it, not just record it"
    doc.redo()
    assert layer.locked is True


def test_an_object_layer_locks_the_same_way():
    doc = _doc()
    layer = doc.add_object_layer("Things")
    doc.set_layer_props(layer.uid, locked=True)
    assert layer.locked is True
    doc.undo()
    assert layer.locked is False


def test_the_tileset_epoch_moves_for_every_way_the_list_changes():
    """The three hooks are the tileset list's only mutators, which is the whole
    licence for a cache keyed on this number."""
    from ._terrainset import terrain_tileset

    doc = MapDoc(6, 6, 16, 16)
    start = doc.tileset_epoch

    doc.add_tileset(_tileset("a"))
    after_add = doc.tileset_epoch
    assert after_add > start

    doc.replace_tileset(0, _tileset("b"))
    after_swap = doc.tileset_epoch
    assert after_swap > after_add

    # A projection change alone must *not* move it: it changes where a cell is
    # drawn, never which tileset owns an id.
    iso = terrain_tileset(tile_w=32, tile_h=16)
    doc.set_projection("orthogonal")
    assert doc.tileset_epoch == after_swap

    # ...but one arriving welded to a tileset does, because the tileset does.
    doc.set_projection("isometric", adding=iso)
    assert doc.tileset_epoch > after_swap


def test_undoing_a_tileset_change_moves_the_epoch_too():
    """A restored tileset list must not restore the epoch with it -- a cache
    that matched again on the way back would be serving the wrong answer."""
    doc = MapDoc(6, 6, 16, 16)
    doc.add_tileset(_tileset("a"))
    seen = doc.tileset_epoch

    doc.undo()
    assert not doc.tilesets
    assert doc.tileset_epoch > seen

    undone = doc.tileset_epoch
    doc.redo()
    assert len(doc.tilesets) == 1
    assert doc.tileset_epoch > undone


def test_a_failed_detach_leaves_the_epoch_alone():
    """The bump sits past the raise, so nothing invalidates on a no-op."""
    from warlock.studio.tilegrid.tileset import TilesetRef

    doc = MapDoc(6, 6, 16, 16)
    doc.add_tileset(_tileset("a"))
    settled = doc.tileset_epoch
    stranger = TilesetRef(firstgid=99, tileset=_tileset("elsewhere"), source="")
    with pytest.raises(KeyError):
        doc._detach_tileset(stranger)
    assert doc.tileset_epoch == settled


def test_a_projection_and_its_tileset_arrive_as_one_step():
    """Two steps would leave a Ctrl+Z on a map whose only tileset is drawn for
    the lattice it is no longer on."""
    from ._terrainset import terrain_tileset

    doc = MapDoc(6, 6, 32, 16)
    depth = len(doc.history)
    iso = terrain_tileset(tile_w=32, tile_h=16)
    doc.set_projection("isometric", adding=iso)
    assert len(doc.history) == depth + 1
    assert doc.projection == "isometric" and len(doc.tilesets) == 1
    doc.undo()
    assert doc.projection == "orthogonal" and not doc.tilesets


def test_an_unknown_projection_is_refused_by_name():
    with pytest.raises(ValueError, match="staggered"):
        MapDoc(4, 4, 16, 16, projection="staggered")


def test_an_isometric_map_is_taller_and_narrower_than_the_grid_suggests():
    doc = MapDoc(4, 4, 32, 16, projection="isometric")
    assert (doc.pixel_width, doc.pixel_height) == (128, 64)
    assert MapDoc(4, 4, 32, 16).pixel_width == 128
    assert MapDoc(4, 4, 32, 16).pixel_height == 64


# --- the object edit session --------------------------------------------------


def _object_doc():
    doc = _doc()
    layer = doc.add_object_layer("Things")
    obj = MapObject(uid=new_uid(), name="zone", kind="rect", x=1.0, y=2.0, w=4.0, h=5.0)
    doc.add_object(layer.uid, obj)
    return doc, layer, obj


def test_a_whole_object_drag_is_one_step():
    """``_map_paint``'s stroke session, for objects: a drag mutates live and
    pushes nothing, and one edit lands at release."""
    doc, layer, obj = _object_doc()
    depth = len(doc.history)

    doc.begin_object_edit(layer.uid, obj.uid)
    for step in range(1, 6):
        doc.place_object(x=float(step), y=float(step))
    assert len(doc.history) == depth, "nothing pushed while the drag is open"
    assert (obj.x, obj.y) == (5.0, 5.0), "but the object moved"

    assert doc.end_object_edit() is True
    assert len(doc.history) == depth + 1
    doc.undo()
    assert (obj.x, obj.y) == (1.0, 2.0)


def test_a_drag_that_moves_nothing_pushes_nothing():
    doc, layer, obj = _object_doc()
    depth = len(doc.history)
    doc.begin_object_edit(layer.uid, obj.uid)
    doc.place_object(x=obj.x, y=obj.y)
    assert doc.end_object_edit() is False
    assert len(doc.history) == depth


def test_ending_an_object_edit_is_idempotent():
    """A release can be missed to focus loss, a tab switch, Esc or a save
    beginning mid-drag, and none of those should have to know."""
    doc, layer, obj = _object_doc()
    doc.begin_object_edit(layer.uid, obj.uid)
    doc.place_object(x=9.0)
    assert doc.end_object_edit() is True
    assert doc.end_object_edit() is False
    assert doc.end_object_edit() is False


def test_re_opening_a_session_commits_the_previous_one():
    doc, layer, obj = _object_doc()
    depth = len(doc.history)
    doc.begin_object_edit(layer.uid, obj.uid)
    doc.place_object(x=9.0)
    doc.begin_object_edit(layer.uid, obj.uid)
    assert len(doc.history) == depth + 1


def test_undo_mid_drag_commits_the_drag_first():
    """The stroke session's rule: undoing *through* an open session would
    reverse the step before it and leave the uncommitted move on the object."""
    doc, layer, obj = _object_doc()
    doc.begin_object_edit(layer.uid, obj.uid)
    doc.place_object(x=9.0, y=9.0)
    doc.undo()
    assert doc.editing_object is False
    assert (obj.x, obj.y) == (1.0, 2.0), "the drag was committed, then undone"


def test_placing_an_object_outside_a_session_is_refused():
    """Rather than silently falling back to the undoable path, which would push
    a step per frame and only be caught by counting the stack."""
    doc, layer, obj = _object_doc()
    with pytest.raises(RuntimeError, match="no object edit"):
        doc.place_object(x=3.0)


def test_a_session_whose_object_was_deleted_commits_nothing():
    doc, layer, obj = _object_doc()
    doc.begin_object_edit(layer.uid, obj.uid)
    doc.place_object(x=9.0)
    doc.remove_object(layer.uid, obj.uid)
    depth = len(doc.history)
    assert doc.end_object_edit() is False
    assert len(doc.history) == depth
