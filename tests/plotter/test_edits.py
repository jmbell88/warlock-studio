"""The two rules that travel with the undo engine, checked on this package.

An edit owns its arrays, and an edit reports its real cost. Both matter for the
same reason: ``cost`` is what eviction is driven by, and a numpy view reports
only its own ``nbytes`` while pinning the whole base alive -- a step that costs
four kilobytes by the budget's reckoning can hold sixteen megabytes.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter import gid
from warlock.studio.plotter.edits import (
    LayerAddEdit,
    LayerRemoveEdit,
    ObjectPropsEdit,
    ResizeEdit,
    TilePatchEdit,
    TilesetAddEdit,
)
from warlock.studio.plotter.tilemap import (
    Ellipse,
    MapDoc,
    MapObject,
    Polygon,
    Rect,
    TileLayer,
    new_uid,
)
from warlock.studio.plotter.tileset import Tileset, TilesetRef


def _tileset(tiles: int = 4) -> Tileset:
    pixels = np.zeros((16, 16 * tiles, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16)


def test_a_patch_copies_a_view_it_is_handed():
    """The document hands ``write_region`` a *slice* of the live layer as the
    before-image. Storing it would mean the before-image changed with the
    document and restored nothing."""
    layer = gid.empty_layer(8, 8)
    layer[2, 2] = 5
    edit = TilePatchEdit(
        layer_uid=1, x0=2, y0=2, before=layer[2:3, 2:3], after=np.array([[9]], gid.DTYPE)
    )
    layer[2, 2] = 77
    assert int(edit.before[0, 0]) == 5


def test_a_patch_costs_both_of_its_rectangles():
    before = gid.empty_layer(4, 4)
    after = gid.empty_layer(4, 4)
    edit = TilePatchEdit(layer_uid=1, x0=0, y0=0, before=before, after=after)
    assert edit.cost == before.nbytes + after.nbytes == 128


def test_a_layer_add_costs_the_array_it_would_hold_alone():
    """An undone add holds the only reference to a full-canvas buffer; a zero
    cost hides it from the byte budget entirely."""
    layer = TileLayer(uid=new_uid(), name="l", data=gid.empty_layer(64, 64))
    assert LayerAddEdit(layer=layer, index=0).cost == layer.data.nbytes
    assert LayerRemoveEdit(layer=layer, index=0).cost == layer.data.nbytes


def test_a_tileset_add_costs_its_pixels():
    """The one type whose cost is easy to leave at zero, and the most expensive
    thing an undo step in this package can be holding."""
    ts = _tileset()
    edit = TilesetAddEdit(ref=TilesetRef(firstgid=1, tileset=ts))
    assert edit.cost == ts.pixels.nbytes


def test_a_resize_costs_every_array_on_both_sides():
    before = {1: gid.empty_layer(8, 8)}
    after = {1: gid.empty_layer(16, 16)}
    edit = ResizeEdit(before_size=(8, 8), after_size=(16, 16), before=before, after=after)
    assert edit.cost == 8 * 8 * 4 + 16 * 16 * 4


def test_an_object_props_edit_deep_copies_its_property_dict():
    """A shallow copy would hand undo and redo the same live mapping, and the
    two would overwrite each other."""
    live = {"hp": 1}
    edit = ObjectPropsEdit(
        layer_uid=1, obj_uid=2, before={"properties": live}, after={"properties": live}
    )
    live["hp"] = 99
    assert edit.before["properties"] == {"hp": 1}
    assert edit.before["properties"] is not edit.after["properties"]


def test_the_budget_sees_what_the_history_is_really_holding():
    """End to end: the stack's own byte figure has to include a tileset's
    pixels, which is what would silently go unaccounted for."""
    doc = MapDoc(8, 8, 16, 16)
    ts = _tileset()
    doc.add_tileset(ts)
    assert doc.history.bytes >= ts.pixels.nbytes


def test_an_undone_edit_still_counts_against_the_budget():
    doc = MapDoc(8, 8, 16, 16)
    doc.add_tileset(_tileset())
    layer = doc.add_tile_layer()
    doc.write_region(layer.uid, 0, 0, np.array([[1]], gid.DTYPE))
    charged = doc.history.bytes
    doc.undo()
    # Still reachable through redo, so still held and still charged.
    assert doc.history.bytes == charged


def test_a_layer_and_object_id_survive_undo_and_redo_of_an_add():
    """``LayerAddEdit``/``ObjectAddEdit`` hold the layer or object itself, not
    a snapshot, so a redo re-inserts the very instance whose ``id`` was
    minted at creation -- no edit-class change is needed for the id to come
    back unchanged."""
    doc = MapDoc(8, 8, 16, 16)
    layer = doc.add_tile_layer()
    assert layer.id == 1
    doc.undo()
    doc.redo()
    assert layer.id == 1

    objects = doc.add_object_layer()
    obj = doc.add_object(objects.uid, MapObject(uid=new_uid(), name="a", kind="point"))
    assert obj.id == 1
    doc.undo()
    doc.redo()
    assert obj.id == 1
    assert doc.layer(objects.uid).objects[0] is obj


def test_an_object_edit_undoes_the_properties_as_well_as_the_position():
    doc = MapDoc(8, 8, 16, 16)
    layer = doc.add_object_layer()
    obj = doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="a", kind="rect", properties={"hp": 1})
    )
    doc.set_object(layer.uid, obj.uid, x=9.0, properties={"hp": 2})
    assert (obj.x, obj.properties) == (9.0, {"hp": 2})
    doc.undo()
    assert (obj.x, obj.properties) == (0.0, {"hp": 1})


def test_an_object_props_edit_holds_its_shapes_by_reference():
    """Geometry costs an edit one reference and no bytes.

    The shape is frozen, so the two sides of a step and the document can hold
    the same object without any of them being able to change it -- which is
    the whole reason ``snapshot`` can be called once per frame of a drag. A
    shape that got copied here would put a deep copy in that inner loop.
    """
    shape = Ellipse(4.0, 4.0)
    edit = ObjectPropsEdit(
        layer_uid=1,
        obj_uid=2,
        before={"shape": shape, "properties": {"hp": 1}},
        after={"shape": Rect(4.0, 4.0), "properties": {"hp": 1}},
    )
    assert edit.before["shape"] is shape
    # ...while ``properties`` is still copied one level, because a caller
    # routinely hands over the live dict it is about to write into.
    assert edit.before["properties"] is not edit.after["properties"]


def test_an_object_add_and_its_undo_carry_the_whole_object():
    """The add edit holds the object itself, so an undone remove brings back
    the same geometry and rotation rather than a rebuilt rect."""
    doc = MapDoc(8, 8, 16, 16)
    layer = doc.add_object_layer()
    shape = Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)))
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), shape=shape, rotation=30.0))
    doc.remove_object(layer.uid, obj.uid)
    doc.undo()
    restored = doc.layer(layer.uid).objects[0]
    assert (restored.shape, restored.rotation) == (shape, 30.0)
    assert restored.shape is shape
