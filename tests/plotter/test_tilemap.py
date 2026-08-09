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

from warlock.studio.plotter import gid
from warlock.studio.plotter.tilemap import MapDoc, MapObject, ObjectLayer, TileLayer, new_uid
from warlock.studio.plotter.tileset import Tileset


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
    with pytest.raises(ValueError):
        MapObject(uid=new_uid(), kind="ellipse")


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
