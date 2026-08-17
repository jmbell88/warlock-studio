"""What an object's geometry *is*, now that it is a tagged shape.

This package used to model two geometries, with the rest refused at the
reader's door. **Those refusals are gone**: the document models all eight, the
canvas draws and hit-tests each, and both Tiled spellings carry them -- which
is exactly the flip the refusals existed to be flipped onto. The docstring here
said they still stood, directly above the tests asserting they do not.

Seven of the eight are Tiled's; ``capsule`` is Warlock dialect and no Tiled
release reads it back. See ``docs/PLOTTER_COMPAT.md``.

The rule the file is written around: ``kind``/``w``/``h`` are **derived** from
the shape and never stored beside it. Two fields that must agree is a bug
waiting for the one code path that updates one of them, and the compat
properties exist so that the panes, the writers and the older tests keep
reading an object the way they always have while there is exactly one place the
answer comes from.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest

from warlock.studio.plotter import gid, tmx
from warlock.studio.plotter.tilemap import (
    Capsule,
    Ellipse,
    MapDoc,
    MapObject,
    ObjectLayer,
    Point,
    Polygon,
    Polyline,
    Rect,
    Text,
    TileShape,
    new_uid,
)
from warlock.studio.plotter.tileset import Tileset


def _doc() -> MapDoc:
    return MapDoc(8, 8, 16, 16)


def _tileset(tiles: int = 4) -> Tileset:
    pixels = np.zeros((16, 16 * tiles, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16)


# --- the union ----------------------------------------------------------------


def test_every_shape_names_its_kind():
    """The compat property is the only place a kind string comes from."""
    cases = [
        (Rect(4, 5), "rect"),
        (Point(), "point"),
        (Ellipse(4, 5), "ellipse"),
        (Capsule(4, 5), "capsule"),
        (Polygon(((0, 0), (4, 0), (4, 4))), "polygon"),
        (Polyline(((0, 0), (4, 0))), "polyline"),
        (TileShape(gid=1, w=16, h=16), "tile"),
        (Text("hi"), "text"),
    ]
    for shape, kind in cases:
        obj = MapObject(uid=new_uid(), shape=shape)
        assert obj.kind == kind
        assert obj.shape is shape


def test_the_size_of_a_shape_that_has_none_is_zero():
    for shape in (Point(), Polygon(((0, 0), (4, 0), (4, 4))), Polyline(((0, 0), (4, 0)))):
        obj = MapObject(uid=new_uid(), shape=shape)
        assert (obj.w, obj.h) == (0.0, 0.0)


def test_the_four_sized_shapes_report_their_size():
    for shape in (Rect(4, 5), Ellipse(4, 5), TileShape(gid=1, w=4, h=5), Text("x", 4, 5)):
        obj = MapObject(uid=new_uid(), shape=shape)
        assert (obj.w, obj.h) == (4.0, 5.0)


def test_shapes_are_frozen_so_a_snapshot_can_share_one():
    """The reason ``snapshot`` costs nothing for geometry: there is no way to
    change a shape in place, so undo and the document can hold the same one."""
    shape = Rect(4, 5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        shape.w = 9  # type: ignore[misc]


def test_a_polygon_needs_three_points_and_a_polyline_two():
    with pytest.raises(ValueError):
        Polygon(((0, 0), (4, 0)))
    with pytest.raises(ValueError):
        Polyline(((0, 0),))
    assert len(Polygon(((0, 0), (4, 0), (4, 4))).points) == 3
    assert len(Polyline(((0, 0), (4, 0))).points) == 2


def test_polygon_points_are_normalized_to_tuples_of_floats():
    """Handed a list of lists -- which is what every JSON reader produces --
    the shape stores the hashable, comparable form."""
    poly = Polygon([[0, 0], [4, 0], [4, 4]])
    assert poly.points == ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))
    assert poly == Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)))


def test_a_negative_size_is_refused_on_every_shape_that_has_one():
    for build in (
        lambda: Rect(-1, 0),
        lambda: Ellipse(0, -1),
        lambda: TileShape(gid=1, w=-1, h=0),
        lambda: Text("x", 0, -1),
    ):
        with pytest.raises(ValueError):
            build()


def test_a_tile_shape_carries_its_flip_flags_in_the_gid():
    """Exactly like a cell: one number, flags in the top bits, nothing between
    here and the renderer strips them."""
    packed = gid.compose(3, flip_h=True, flip_d=True)
    obj = MapObject(uid=new_uid(), shape=TileShape(gid=packed, w=16, h=16))
    assert obj.shape.gid == packed
    assert gid.decompose(obj.shape.gid) == (3, True, False, True)


def test_text_defaults_mirror_tiled():
    """Tiled 1.12.2's own defaults for a text object, field for field. A
    different default here would make every unstyled text object export as
    styled."""
    text = Text("hello")
    assert (text.text, text.w, text.h) == ("hello", 0.0, 0.0)
    assert (text.family, text.pixel_size, text.wrap) == ("sans-serif", 16, False)
    assert (text.color, text.halign, text.valign) == ("#000000", "left", "top")
    assert (text.bold, text.italic, text.underline) == (False, False, False)
    assert (text.strikeout, text.kerning) == (False, True)


# --- the compat constructor ---------------------------------------------------


def test_the_old_construction_form_still_builds_a_rect():
    obj = MapObject(uid=new_uid(), name="zone", kind="rect", x=1, y=2, w=8, h=4)
    assert obj.shape == Rect(8.0, 4.0)
    assert (obj.kind, obj.x, obj.y, obj.w, obj.h) == ("rect", 1.0, 2.0, 8.0, 4.0)


def test_an_object_with_no_shape_and_no_kind_is_a_rect():
    assert MapObject(uid=new_uid()).shape == Rect(0.0, 0.0)


def test_the_old_construction_form_still_builds_a_point():
    obj = MapObject(uid=new_uid(), name="spawn", kind="point", x=1, y=2)
    assert obj.shape == Point()
    assert (obj.kind, obj.w, obj.h) == ("point", 0.0, 0.0)


def test_every_ui_creatable_kind_builds_a_useful_default_shape():
    kinds = ("rect", "point", "ellipse", "capsule", "polygon", "polyline", "tile", "text")
    for kind in kinds:
        assert MapObject(uid=new_uid(), kind=kind, w=4, h=5).kind == kind


def test_a_shape_and_a_size_cannot_both_be_given():
    """Two ways to say the geometry in one call is two chances to disagree."""
    with pytest.raises(ValueError):
        MapObject(uid=new_uid(), kind="rect", w=4, h=4, shape=Ellipse(8, 8))
    with pytest.raises(ValueError):
        MapObject(uid=new_uid(), shape=Point(), kind="rect")


def test_something_that_is_not_a_shape_is_refused():
    with pytest.raises(ValueError):
        MapObject(uid=new_uid(), shape="rect")  # type: ignore[arg-type]


def test_every_field_is_a_parameter_of_the_hand_written_constructor():
    """``MapObject.__init__`` is written by hand -- ``kind``/``w``/``h`` go in
    and are never stored, which ``dataclass`` has no way to express -- so
    nothing generated keeps the two lists in agreement. A field added to the
    class and forgotten in the signature breaks ``dataclasses.replace`` and
    nothing else, which is a failure that surfaces a long way from its cause.
    """
    fields = {f.name for f in dataclasses.fields(MapObject)}
    parameters = set(inspect.signature(MapObject.__init__).parameters)
    assert fields <= parameters, f"not constructible: {sorted(fields - parameters)}"


def test_replace_rebuilds_an_object_around_one_changed_field():
    """What the agreement above buys. ``replace`` passes every field by name,
    so it is what actually exercises the whole signature -- and it re-runs
    ``__post_init__``, so the copy is validated rather than assembled."""
    obj = MapObject(
        uid=new_uid(),
        name="zone",
        shape=Ellipse(4, 6),
        rotation=15.0,
        properties={"hp": 1},
    )
    moved = dataclasses.replace(obj, x=5.0)
    assert (moved.x, moved.y) == (5.0, 0.0)
    assert (moved.name, moved.rotation, moved.shape) == ("zone", 15.0, Ellipse(4.0, 6.0))
    assert (moved.kind, moved.w, moved.h) == ("ellipse", 4.0, 6.0)
    assert moved.properties == {"hp": 1}
    assert obj.x == 0.0, "the original is untouched"
    with pytest.raises(ValueError):
        dataclasses.replace(obj, shape="rect")  # type: ignore[arg-type]


# --- rotation -----------------------------------------------------------------


def test_rotation_defaults_to_zero_and_is_stored_as_a_float():
    assert MapObject(uid=new_uid()).rotation == 0.0
    assert MapObject(uid=new_uid(), rotation=45).rotation == 45.0


def test_a_snapshot_carries_rotation_and_the_shape_itself():
    shape = Polygon(((0, 0), (4, 0), (4, 4)))
    obj = MapObject(uid=new_uid(), shape=shape, rotation=30.0)
    snap = obj.snapshot()
    assert snap["rotation"] == 30.0
    # Passed through, not copied: a frozen shape is safe to share, and
    # deep-copying one per drag frame is the cost this buys back.
    assert snap["shape"] is shape
    assert (snap["kind"], snap["w"], snap["h"]) == ("polygon", 0.0, 0.0)


# --- editing ------------------------------------------------------------------


def test_a_shape_change_round_trips_through_undo():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), kind="rect", w=8, h=8))
    doc.set_object(layer.uid, obj.uid, shape=Ellipse(6, 6))
    assert (obj.shape, obj.kind, obj.w) == (Ellipse(6.0, 6.0), "ellipse", 6.0)
    doc.undo()
    assert (obj.shape, obj.kind) == (Rect(8.0, 8.0), "rect")
    doc.redo()
    assert obj.shape == Ellipse(6.0, 6.0)


def test_a_rotation_change_round_trips_through_undo():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), kind="rect", w=8, h=8))
    doc.set_object(layer.uid, obj.uid, rotation=90.0)
    assert obj.rotation == 90.0
    doc.undo()
    assert obj.rotation == 0.0


def test_an_object_opacity_change_round_trips_through_undo():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), opacity=0.8))
    doc.set_object(layer.uid, obj.uid, opacity=0.25)
    assert obj.opacity == 0.25
    doc.undo()
    assert obj.opacity == 0.8
    doc.redo()
    assert obj.opacity == 0.25


@pytest.mark.parametrize("opacity", [-0.01, 1.01])
def test_an_object_opacity_outside_zero_to_one_is_refused(opacity):
    with pytest.raises(ValueError, match="opacity"):
        MapObject(uid=new_uid(), opacity=opacity)


def test_resizing_by_w_and_h_keeps_the_shape_it_was():
    """The canvas resizes by ``w``/``h`` and knows nothing about shapes. An
    ellipse dragged by a handle stays an ellipse."""
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), shape=Ellipse(4, 4)))
    doc.set_object(layer.uid, obj.uid, w=10.0, h=12.0)
    assert obj.shape == Ellipse(10.0, 12.0)
    doc.undo()
    assert obj.shape == Ellipse(4.0, 4.0)


def test_resizing_a_shape_that_has_no_size_changes_nothing():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), kind="point"))
    doc.history.clear()
    doc.set_object(layer.uid, obj.uid, w=10.0, h=12.0)
    assert obj.shape == Point()
    assert doc.history.can_undo is False


def test_the_kind_door_still_refuses_an_exotic_kind_on_an_edit():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), kind="rect", w=8, h=8))
    with pytest.raises(ValueError):
        doc.set_object(layer.uid, obj.uid, kind="bezier")


def test_a_kind_change_rebuilds_the_shape():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), kind="rect", w=8, h=8))
    doc.set_object(layer.uid, obj.uid, kind="point")
    assert obj.shape == Point()
    doc.undo()
    assert obj.shape == Rect(8.0, 8.0)


def test_a_kind_change_can_rebuild_a_core_non_rect_shape():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), kind="rect", w=8, h=6))
    doc.set_object(layer.uid, obj.uid, kind="capsule")
    assert obj.shape == Capsule(8.0, 6.0)


def test_setting_the_same_shape_pushes_nothing():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), shape=Ellipse(4, 4)))
    doc.history.clear()
    doc.set_object(layer.uid, obj.uid, shape=Ellipse(4, 4))
    assert doc.history.can_undo is False


def test_a_drag_session_can_reshape_and_pushes_one_step():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), shape=Ellipse(4, 4)))
    doc.history.clear()
    doc.begin_object_edit(layer.uid, obj.uid)
    for size in (5.0, 6.0, 7.0):
        doc.place_object(x=1.0, y=1.0, w=size, h=size)
    assert doc.history.can_undo is False
    assert doc.end_object_edit() is True
    assert obj.shape == Ellipse(7.0, 7.0)
    doc.undo()
    assert (obj.shape, obj.x) == (Ellipse(4.0, 4.0), 0.0)


def test_a_rotation_gesture_ends_in_one_step():
    doc = _doc()
    layer = doc.add_object_layer()
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), kind="rect", w=4, h=4))
    doc.history.clear()
    doc.begin_object_edit(layer.uid, obj.uid)
    doc.place_object(rotation=15.0)
    doc.place_object(rotation=30.0)
    assert doc.end_object_edit() is True
    assert obj.rotation == 30.0
    doc.undo()
    assert obj.rotation == 0.0


# --- draworder ----------------------------------------------------------------


def test_an_object_layer_draws_top_down_by_default():
    doc = _doc()
    assert doc.add_object_layer().draworder == "topdown"


def test_draworder_round_trips_through_undo():
    doc = _doc()
    layer = doc.add_object_layer()
    doc.set_layer_props(layer.uid, draworder="index")
    assert doc.layer(layer.uid).draworder == "index"
    doc.undo()
    assert doc.layer(layer.uid).draworder == "topdown"


def test_an_unknown_draworder_is_refused():
    with pytest.raises(ValueError):
        ObjectLayer(uid=new_uid(), name="o", draworder="sideways")


# --- gid validation -----------------------------------------------------------


def test_a_tile_object_whose_gid_nothing_accounts_for_is_refused():
    """The check every cell already gets, reaching the one other place a gid
    can appear. Unit-tested by construction: no importer makes one yet."""
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    layer = doc.add_object_layer("Things")
    layer.objects.append(MapObject(uid=new_uid(), shape=TileShape(gid=99, w=16, h=16)))
    with pytest.raises(ValueError, match="99"):
        tmx._finish(doc)


def test_a_tile_object_whose_gid_resolves_is_accepted_flags_and_all():
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    layer = doc.add_object_layer("Things")
    packed = gid.compose(2, flip_h=True, flip_v=True)
    layer.objects.append(MapObject(uid=new_uid(), shape=TileShape(gid=packed, w=16, h=16)))
    tmx._finish(doc)
    assert layer.objects[0].shape.gid == packed


def test_a_tile_object_gid_of_zero_is_the_empty_tile_and_passes():
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(_tileset())
    layer = doc.add_object_layer("Things")
    layer.objects.append(MapObject(uid=new_uid(), shape=TileShape(gid=0)))
    tmx._finish(doc)
