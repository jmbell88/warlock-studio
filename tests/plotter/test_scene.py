"""The layer tree and the resolver that flattens it.

Three properties are what this file exists for, and all three are the tree
restating rules the flat stack already had.

**A group is not a layer with children drawn onto it.** Nothing composites a
group; a group is a *state* its descendants inherit, so the same six numbers --
offset, parallax, opacity, visibility, tint, lock -- have to combine the same
way for every consumer or the canvas and an export start disagreeing about what
a nested layer looks like. :mod:`warlock.studio.plotter.scene` is that one
answer and both renderers iterate it.

**A uid still addresses.** The whole point of never recording an index is that
an edit survives the list moving under it, and a tree gives the list more ways
to move: a layer can now change *parent*. An edit recorded before a reparent
has to land afterwards, which is what ``_an_edit_inside_a_moved_group`` pins.

**A subtree travels as one object.** Removing a group is one step whose cost is
every array under it, and undoing it puts the whole subtree back where it was --
not the group with its children stranded somewhere else.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from warlock.studio.plotter import scene
from warlock.studio.plotter.tilemap import (
    GroupLayer,
    ImageLayer,
    MapDoc,
    MapObject,
    ObjectLayer,
    TileLayer,
    new_uid,
)
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


def _picture(w: int = 4, h: int = 4) -> np.ndarray:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[..., 0] = 200
    pixels[..., 3] = 255
    return pixels


# --- the model ----------------------------------------------------------------


def test_every_layer_kind_carries_the_same_decorations():
    """One vocabulary for four classes. They are declared field by field rather
    than inherited (a base class would force ``data`` to carry a default), so
    the only thing keeping the four in step is this."""
    doc = _doc()
    kinds = [
        doc.add_tile_layer("t"),
        doc.add_object_layer("o"),
        doc.add_group_layer("g"),
        doc.add_image_layer("i"),
    ]
    for layer in kinds:
        for name in scene.DECORATION_FIELDS:
            assert hasattr(layer, name), f"{type(layer).__name__} has no {name}"
        assert layer.tint == (255, 255, 255, 255)
        assert (layer.parallax_x, layer.parallax_y) == (1.0, 1.0)
        assert (layer.offset_x, layer.offset_y) == (0.0, 0.0)
        assert layer.class_name == ""
        snapshot = layer.snapshot()
        for name in scene.DECORATION_FIELDS:
            assert name in snapshot


def test_an_image_layers_pixels_are_frozen_like_a_tilesets():
    layer = ImageLayer(uid=1, name="sky", pixels=_picture())
    assert layer.pixels.shape == (4, 4, 4)
    with pytest.raises(ValueError):
        layer.pixels[0, 0, 0] = 1


def test_an_image_layer_with_no_picture_is_legal_and_empty():
    layer = ImageLayer(uid=1, name="sky")
    assert layer.pixels.shape == (0, 0, 4)
    assert (layer.source, layer.repeat_x, layer.repeat_y) == ("", False, False)


def test_a_tint_outside_the_channel_range_is_refused():
    with pytest.raises(ValueError):
        TileLayer(uid=1, name="t", data=gid.empty_layer(2, 2), tint=(255, 300, 0, 255))


@pytest.mark.parametrize("tint", [(255, 300, 0, 255), (1, 2), (0, 0, 0, -1), "red"])
def test_the_setter_refuses_every_tint_the_constructor_refuses(tint):
    """The door and the constructor have to agree, or the setter is a way in
    for a value the type will not accept.

    ``(1, 2)`` is the case that made this urgent: it is not merely wrong, it
    survived being stored and then failed *elsewhere and later*, inside
    ``scene._tint_product``'s strict ``zip``, on the frame thread, one resolve
    after anybody could have connected the two.
    """
    doc = _doc()
    layer = doc.add_tile_layer("t")
    head = doc.history.head
    with pytest.raises(ValueError):
        doc.set_layer_props(layer.uid, tint=tint)
    assert layer.tint == (255, 255, 255, 255), "the refusal left the layer alone"
    assert doc.history.head == head, "a refused change must push no step"
    assert scene.resolve(doc)[0].tint == (255, 255, 255, 255)


def test_a_tint_spelled_as_a_list_is_the_same_tint_and_pushes_nothing():
    """The coercion runs before the no-op test, so an unchanged value spelled
    differently is still an unchanged value."""
    doc = _doc()
    layer = doc.add_tile_layer("t")
    head = doc.history.head
    doc.set_layer_props(layer.uid, tint=[255, 255, 255, 255])
    assert doc.history.head == head
    doc.set_layer_props(layer.uid, tint=[128, 255, 255, 255])
    assert layer.tint == (128, 255, 255, 255)
    assert isinstance(layer.tint, tuple)


def test_a_decoration_that_is_not_a_number_is_refused_before_the_push():
    doc = _doc()
    layer = doc.add_tile_layer("t")
    head = doc.history.head
    with pytest.raises(ValueError):
        doc.set_layer_props(layer.uid, offset_x="over there")
    assert doc.history.head == head
    assert layer.offset_x == 0.0


# --- the tree -----------------------------------------------------------------


def test_a_layer_added_into_a_group_is_not_in_the_root_list():
    doc = _doc()
    group = doc.add_group_layer("G")
    inner = doc.add_tile_layer("inner", parent_uid=group.uid)
    assert doc.layers == [group]
    assert group.children == [inner]
    assert doc.layer(inner.uid) is inner
    assert doc.parent_uid_of(inner.uid) == group.uid
    assert doc.index_of(inner.uid) == 0


def test_tile_layers_are_the_leaves_in_depth_first_paint_order():
    doc = _doc()
    floor = doc.add_tile_layer("floor")
    group = doc.add_group_layer("G")
    inner_a = doc.add_tile_layer("a", parent_uid=group.uid)
    inner_b = doc.add_tile_layer("b", parent_uid=group.uid)
    roof = doc.add_tile_layer("roof")
    assert [layer.name for layer in doc.tile_layers()] == ["floor", "a", "b", "roof"]
    assert doc.tile_layers() == [floor, inner_a, inner_b, roof]
    assert [layer.name for layer in doc.all_layers()] == ["floor", "G", "a", "b", "roof"]


def test_a_group_cannot_be_moved_inside_its_own_subtree():
    doc = _doc()
    outer = doc.add_group_layer("outer")
    inner = doc.add_group_layer("inner", parent_uid=outer.uid)
    with pytest.raises(ValueError):
        doc.move_layer(outer.uid, 0, parent_uid=inner.uid)
    with pytest.raises(ValueError):
        doc.move_layer(outer.uid, 0, parent_uid=outer.uid)
    assert doc.layers == [outer]
    assert outer.children == [inner]


def test_a_reparent_is_one_step_and_undoes_to_the_old_parent():
    doc = _doc()
    group = doc.add_group_layer("G")
    layer = doc.add_tile_layer("t")
    head = doc.history.head
    doc.move_layer(layer.uid, 0, parent_uid=group.uid)
    assert doc.history.head == head + 1
    assert group.children == [layer]
    assert doc.layers == [group]
    doc.undo()
    assert group.children == []
    assert doc.layers == [group, layer]


def test_moving_a_layer_where_it_already_is_pushes_nothing():
    doc = _doc()
    group = doc.add_group_layer("G")
    layer = doc.add_tile_layer("t", parent_uid=group.uid)
    head = doc.history.head
    doc.move_layer(layer.uid, 0)
    doc.move_layer(layer.uid, 0, parent_uid=group.uid)
    assert doc.history.head == head


# --- the three the plan names -------------------------------------------------


def test_group_visibility_and_opacity_resolve_through_ancestors():
    """The five combination rules, all at once and all through one group.

    Hidden wins by AND, opacity multiplies, offsets sum, parallax multiplies and
    a lock spreads downward -- so a layer that is itself visible, unlocked and
    at full opacity resolves to none of those things under a group that is not.
    """
    doc = _doc()
    group = doc.add_group_layer("G")
    inner = doc.add_tile_layer("inner", parent_uid=group.uid)
    doc.set_layer_props(
        group.uid,
        opacity=0.5,
        locked=True,
        offset_x=10.0,
        offset_y=4.0,
        parallax_x=0.5,
        tint=(255, 128, 128, 255),
    )
    doc.set_layer_props(
        inner.uid, opacity=0.5, offset_x=1.0, parallax_x=0.5, tint=(128, 255, 255, 255)
    )

    (entry,) = scene.resolve(doc)
    assert entry.layer is inner
    assert entry.opacity == pytest.approx(0.25)
    assert entry.offset == (11.0, 4.0)
    assert entry.parallax == (0.25, 1.0)
    assert entry.tint == (128, 128, 128, 255)
    assert entry.locked is True
    assert entry.visible is True

    doc.set_layer_props(group.uid, visible=False)
    assert scene.resolve(doc) == []
    hidden = scene.resolve(doc, include_hidden=True)
    assert [e.layer for e in hidden] == [inner]
    assert hidden[0].visible is False
    # The leaf's own flag never moved -- the group is what is hidden.
    assert inner.visible is True


def test_an_edit_inside_a_moved_group_still_lands_by_uid():
    """The travelling rule, under the one thing a tree adds: a change of parent.

    The patch is recorded while the layer sits in one group and undone after it
    has been moved into another, at a different depth and a different index.
    """
    doc = _doc()
    left = doc.add_group_layer("left")
    right = doc.add_group_layer("right")
    inner = doc.add_tile_layer("inner", parent_uid=left.uid)
    doc.add_tile_layer("decoy", parent_uid=right.uid)

    value = gid.compose(2)
    doc.write_region(inner.uid, 1, 1, np.full((2, 2), value, gid.DTYPE))
    assert int(inner.data[1, 1]) == value

    doc.move_layer(inner.uid, 0, parent_uid=right.uid)
    assert doc.parent_uid_of(inner.uid) == right.uid

    doc.undo()  # the move
    doc.undo()  # the patch, recorded two parents ago
    assert int(inner.data[1, 1]) == 0
    assert doc.parent_uid_of(inner.uid) == left.uid

    doc.redo()
    assert int(inner.data[1, 1]) == value


def test_removing_a_group_removes_and_restores_the_subtree_as_one_step():
    doc = _doc()
    keep = doc.add_tile_layer("keep")
    group = doc.add_group_layer("G")
    inner = doc.add_tile_layer("inner", parent_uid=group.uid)
    nested = doc.add_group_layer("nested", parent_uid=group.uid)
    deep = doc.add_tile_layer("deep", parent_uid=nested.uid)
    doc.write_region(deep.uid, 0, 0, np.full((1, 1), gid.compose(3), gid.DTYPE))

    head = doc.history.head
    doc.remove_layer(group.uid)
    assert doc.history.head == head + 1
    assert doc.layers == [keep]
    assert doc.layer(inner.uid) is None
    assert doc.layer(deep.uid) is None

    # The cost is the whole subtree, not the group's own nothing.
    step = doc.history.top
    assert step.cost == inner.data.nbytes + deep.data.nbytes

    doc.undo()
    assert doc.layers == [keep, group]
    assert doc.layer(deep.uid) is deep
    assert doc.parent_uid_of(deep.uid) == nested.uid
    assert doc.index_of(nested.uid) == 1
    assert int(deep.data[0, 0]) == gid.compose(3)

    doc.redo()
    assert doc.layers == [keep]


def test_removing_a_group_the_active_layer_is_inside_falls_back_to_a_real_layer():
    doc = _doc()
    keep = doc.add_tile_layer("keep")
    group = doc.add_group_layer("G")
    inner = doc.add_tile_layer("inner", parent_uid=group.uid)
    doc.set_active_layer(inner.uid)
    doc.remove_layer(group.uid)
    assert doc.active_layer == keep.uid


# --- resolve ------------------------------------------------------------------


def test_resolve_yields_only_leaves_in_paint_order():
    doc = _doc()
    bottom = doc.add_tile_layer("bottom")
    group = doc.add_group_layer("G")
    inner = doc.add_object_layer("inner", parent_uid=group.uid)
    image = doc.add_image_layer("sky", pixels=_picture(), parent_uid=group.uid)
    top = doc.add_tile_layer("top")
    assert [e.layer for e in scene.resolve(doc)] == [bottom, inner, image, top]
    assert all(not isinstance(e.layer, GroupLayer) for e in scene.resolve(doc))


def test_an_empty_group_resolves_to_nothing_at_all():
    doc = _doc()
    doc.add_group_layer("G")
    assert scene.resolve(doc) == []


def test_a_hidden_group_is_not_descended_into_unless_hidden_are_asked_for():
    doc = _doc()
    group = doc.add_group_layer("G")
    doc.add_tile_layer("inner", parent_uid=group.uid)
    doc.set_layer_props(group.uid, visible=False)
    assert scene.resolve(doc) == []
    assert len(scene.resolve(doc, include_hidden=True)) == 1


def test_resolved_for_answers_about_a_group_as_well_as_a_leaf():
    doc = _doc()
    group = doc.add_group_layer("G")
    inner = doc.add_tile_layer("inner", parent_uid=group.uid)
    doc.set_layer_props(group.uid, offset_x=8.0)
    assert scene.resolved_for(doc, group.uid).offset == (8.0, 0.0)
    assert scene.resolved_for(doc, inner.uid).offset == (8.0, 0.0)
    assert scene.resolved_for(doc, 99999) is None


def test_a_resolved_state_is_frozen():
    doc = _doc()
    doc.add_tile_layer("t")
    (entry,) = scene.resolve(doc)
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.opacity = 0.5


# --- both renderers -----------------------------------------------------------


def test_a_group_opacity_reaches_the_flat_render():
    from warlock.studio.plotter import render as plotter_render

    doc = _doc()
    group = doc.add_group_layer("G")
    inner = doc.add_tile_layer("inner", parent_uid=group.uid)
    doc.write_region(inner.uid, 0, 0, np.full((1, 1), gid.compose(1), gid.DTYPE))
    full = plotter_render.render_map(doc)
    assert int(full[0, 0, 3]) == 255
    doc.set_layer_props(group.uid, opacity=0.5)
    faded = plotter_render.render_map(doc)
    assert 100 < int(faded[0, 0, 3]) < 200


def test_a_hidden_group_hides_its_children_from_the_export_and_the_minimap():
    from warlock.studio.plotter import render as plotter_render

    doc = _doc()
    group = doc.add_group_layer("G")
    inner = doc.add_tile_layer("inner", parent_uid=group.uid)
    doc.write_region(inner.uid, 0, 0, np.full((1, 1), gid.compose(1), gid.DTYPE))
    assert int(plotter_render.render_map(doc)[0, 0, 3]) == 255
    assert int(plotter_render.minimap(doc)[0, 0, 3]) > 0
    doc.set_layer_props(group.uid, visible=False)
    assert int(plotter_render.render_map(doc)[0, 0, 3]) == 0
    assert int(plotter_render.minimap(doc)[0, 0, 3]) == 0


def test_a_layer_offset_moves_what_the_flat_render_draws():
    from warlock.studio.plotter import render as plotter_render

    doc = _doc()
    layer = doc.add_tile_layer("t")
    doc.write_region(layer.uid, 0, 0, np.full((1, 1), gid.compose(1), gid.DTYPE))
    doc.set_layer_props(layer.uid, offset_x=16.0, offset_y=16.0)
    out = plotter_render.render_map(doc)
    assert int(out[0, 0, 3]) == 0
    assert int(out[16, 16, 3]) == 255


def test_an_image_layer_composites_into_the_flat_render():
    from warlock.studio.plotter import render as plotter_render

    doc = _doc()
    doc.add_image_layer("sky", pixels=_picture(8, 8))
    out = plotter_render.render_map(doc)
    assert int(out[0, 0, 0]) == 200
    assert int(out[7, 7, 3]) == 255
    assert int(out[8, 8, 3]) == 0


def test_a_repeating_image_layer_fills_the_map():
    from warlock.studio.plotter import render as plotter_render

    doc = _doc()
    doc.add_image_layer("sky", pixels=_picture(8, 8), repeat_x=True, repeat_y=True)
    out = plotter_render.render_map(doc)
    assert int(out[-1, -1, 3]) == 255


def test_a_layer_tint_multiplies_the_flat_render():
    from warlock.studio.plotter import render as plotter_render

    doc = _doc()
    doc.add_image_layer("sky", pixels=_picture(4, 4), tint=(128, 255, 255, 255))
    out = plotter_render.render_map(doc)
    assert int(out[0, 0, 0]) == 100


# --- the writer doors ---------------------------------------------------------


def test_a_wmap_of_a_document_holding_a_group_stores_the_tree():
    """Flipped. This was a ``WmapUnstorable`` while the manifest's ``layers``
    was a flat list; version 3's entries are recursive, so the refusal moved
    for the only reason a refusal here is ever allowed to move -- the format
    learned to hold the thing."""
    from warlock.studio.plotter import wmap

    doc = _doc()
    group = doc.add_group_layer("G")
    doc.add_tile_layer("t", parent_uid=group.uid)
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    outer = back.layers[-1]
    assert outer.name == "G"
    assert [child.name for child in outer.children] == ["t"]


def test_a_wmap_of_a_document_holding_an_image_layer_stores_the_picture():
    """Flipped, with the group case above: the pixels are an ``images/N.png``
    member now, embedded the way a tileset's atlas already was."""
    import numpy as np

    from warlock.studio.plotter import wmap

    doc = _doc()
    doc.add_image_layer("sky", pixels=_picture(), source="art/sky.png", repeat_x=True)
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    sky = back.layers[-1]
    assert (sky.name, sky.source, sky.repeat_x, sky.repeat_y) == (
        "sky", "art/sky.png", True, False
    )
    assert np.array_equal(sky.pixels, _picture())


def test_the_wmap_writer_door_has_a_name_of_its_own():
    """The door outlives the four refusals it was built for. Version 3 stores
    the tree, the pictures and the decorations, so nothing in an ordinary
    document reaches this any more -- but a refusal the save path can catch *by
    type* is what stops a handler on bare ``ValueError`` dressing every genuine
    encoder defect up as a polite refusal the user is meant to act on, and the
    milestones ahead (chunked storage, a fifth layer kind) put more behind it.
    The remaining raise is the unknown-kind fallthrough, which nothing but a
    future layer type can reach -- so it is provoked with one, and a bare
    object is enough precisely because the writer decides the kind *before* it
    asks a layer for anything."""
    from warlock.studio.plotter import wmap

    assert issubclass(wmap.WmapUnstorable, ValueError)

    class FutureLayer:
        """A fifth layer kind, arriving before the container can hold it."""

    doc = _doc()
    doc.layers.append(FutureLayer())
    with pytest.raises(wmap.WmapUnstorable, match="no entry for"):
        wmap.wmap_bytes(doc)


def test_the_wmap_door_is_the_encoder_rather_than_the_json_formatter():
    """Flipped from "the refusal came before the manifest was built" to its
    other half: with nothing left to refuse for a document like this, the
    encoder now reaches the formatter -- and the formatter is where the one
    remaining refusal lives, which is why ``wmap_bytes`` still builds the whole
    manifest *before* it opens the archive. A refusal raised inside the ``with``
    would leave a half-written zip behind it."""
    import warlock.studio.plotter.wmap as wmap

    doc = _doc()
    doc.add_group_layer("G")
    called: list[int] = []
    original = wmap.manifest_json
    try:
        wmap.manifest_json = lambda d: called.append(1) or original(d)  # type: ignore[assignment]
        assert wmap.wmap_bytes(doc)
    finally:
        wmap.manifest_json = original
    assert called == [1], "the encoder formats through the one manifest writer"


def test_a_flat_document_still_writes_a_wmap():
    from warlock.studio.plotter import wmap

    doc = _doc()
    doc.add_tile_layer("t")
    doc.add_object_layer("o")
    assert wmap.read_wmap(wmap.wmap_bytes(doc)).width == doc.width


def test_groups_and_image_layers_are_written_to_both_tiled_formats():
    import json
    import xml.etree.ElementTree as ET

    from warlock.studio.plotter import tmx

    doc = _doc()
    group = doc.add_group_layer("G")
    doc.add_image_layer("sky", pixels=_picture(), parent_uid=group.uid)

    xml_files = tmx.tmx_export(doc)
    root = ET.fromstring(xml_files["map.tmx"])
    assert root.find("group/imagelayer") is not None
    assert any(name.endswith(".png") for name in xml_files)

    payload = json.loads(tmx.tmj_export(doc)["map.tmj"])
    assert payload["layers"][0]["type"] == "group"
    assert payload["layers"][0]["layers"][0]["type"] == "imagelayer"


@pytest.mark.parametrize(
    ("values", "xml_attr", "json_key"),
    [
        ({"offset_y": 8.0}, "offsety", "offsety"),
        ({"tint": (255, 0, 0, 255)}, "tintcolor", "tintcolor"),
        ({"parallax_x": 0.5}, "parallaxx", "parallaxx"),
        ({"class_name": "Ground"}, "class", "class"),
        ({"blend_mode": "multiply"}, "mode", "mode"),
    ],
)
def test_decorated_layers_are_written_to_both_tiled_formats(values, xml_attr, json_key):
    import json
    import xml.etree.ElementTree as ET

    from warlock.studio.plotter import tmx

    doc = _doc()
    layer = doc.add_tile_layer("t")
    doc.set_layer_props(layer.uid, **values)
    node = ET.fromstring(tmx.tmx_export(doc)["map.tmx"]).find("layer")
    assert node is not None and node.get(xml_attr) is not None
    payload = json.loads(tmx.tmj_export(doc)["map.tmj"])
    assert json_key in payload["layers"][0]


@pytest.mark.parametrize(
    "values",
    [
        {"offset_y": 8.0},
        {"tint": (255, 0, 0, 255)},
        {"parallax_x": 0.5},
        {"class_name": "Ground"},
    ],
)
def test_a_wmap_of_a_decorated_layer_round_trips_since_v3(values):
    """Flipped from ``..._is_refused_until_v3``. Version 3 is the "until", and
    the four decorations the ``.tmx`` door still refuses by name are stored
    here field for field -- which is what makes the *other* door's message
    honest, since ".wmap holds it, Tiled cannot" is now a true sentence."""
    from warlock.studio.plotter import wmap

    doc = _doc()
    layer = doc.add_tile_layer("t")
    doc.set_layer_props(layer.uid, **values)
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    stored = back.layers[-1]
    for name, value in values.items():
        assert getattr(stored, name) == value


def test_an_undecorated_document_is_not_caught_by_either_door():
    """The guard that keeps the four refusals above from being a size limit on
    every map: identity values are not decorations."""
    from warlock.studio.plotter import tmx, wmap

    doc = _doc()
    doc.add_tile_layer("t")
    doc.add_object_layer("o")
    assert wmap.wmap_bytes(doc)
    assert tmx.tmx_export(doc)
    assert tmx.tmj_export(doc)


# --- the rest of the document keeps working -----------------------------------


def test_a_resize_moves_objects_on_a_nested_layer_too():
    doc = _doc()
    group = doc.add_group_layer("G")
    layer = doc.add_object_layer("o", parent_uid=group.uid)
    doc.add_object(layer.uid, MapObject(uid=new_uid(), name="spawn", x=16.0, y=16.0))
    doc.resize(10, 8, offset_x=1, offset_y=1)
    assert (layer.objects[0].x, layer.objects[0].y) == (32.0, 32.0)


def test_a_nested_tile_layer_is_resized_with_the_rest():
    doc = _doc()
    group = doc.add_group_layer("G")
    layer = doc.add_tile_layer("t", parent_uid=group.uid)
    doc.resize(10, 8)
    assert layer.data.shape == (8, 10)


def test_the_layer_kinds_are_all_in_the_union():
    from warlock.studio.plotter import _map_model

    assert set(_map_model.LEAF_LAYERS) == {TileLayer, ObjectLayer, ImageLayer}
    assert GroupLayer not in _map_model.LEAF_LAYERS
