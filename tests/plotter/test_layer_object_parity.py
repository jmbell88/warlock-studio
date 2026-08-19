"""Object and layer parity: duplicates, merge down, image pixels, vertices.

Two rules run through the whole file. A duplicate takes **fresh identities** --
uid and persistent id both -- because a uid is how every recorded patch addresses
a layer and an id is what a `.tmx` writes, so a copy sharing either is the
original as far as undo or a round trip is concerned. And a merge is a **data**
merge at gid level, never a pixel composite: opacity, tint and blend stay
presentation, because a merge that baked them would disagree with the renderer
the first time a tint changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter.tilemap import MapDoc, MapObject, new_uid
from warlock.studio.tilegrid import gid as gidlib


def _doc(width: int = 8, height: int = 8) -> MapDoc:
    doc = MapDoc(width, height, 16, 16)
    doc.add_tile_layer("Tiles")
    return doc


# --- duplicate a layer --------------------------------------------------------


def test_a_duplicated_tile_layer_shares_no_array_with_its_original() -> None:
    doc = _doc()
    original = doc.tile_layers()[0]
    doc.write_region(original.uid, 0, 0, np.array([[5]], gidlib.DTYPE))

    copy = doc.duplicate_layer(original.uid)
    assert copy.uid != original.uid
    assert copy.id != original.id
    assert np.array_equal(copy.data, original.data)

    doc.write_region(copy.uid, 1, 1, np.array([[9]], gidlib.DTYPE))
    assert int(doc.layer(original.uid).data[1, 1]) == 0, "painting one must not paint both"


def test_a_duplicate_lands_directly_above_its_original() -> None:
    doc = _doc()
    original = doc.tile_layers()[0]
    copy = doc.duplicate_layer(original.uid)
    # ``children_of`` is top-first, so "above" is the lower index.
    order = [layer.uid for layer in doc.children_of(None)]
    assert order.index(copy.uid) + 1 == order.index(original.uid)


def test_duplicating_a_group_mints_fresh_ids_all_the_way_down() -> None:
    doc = _doc()
    group = doc.add_group_layer("Group")
    inner = doc.add_tile_layer("Inner", parent_uid=group.uid)

    copy = doc.duplicate_layer(group.uid)
    inner_copy = copy.children[0]
    assert inner_copy.uid != inner.uid
    assert inner_copy.id != inner.id
    assert copy.uid != group.uid and copy.id != group.id


def test_a_duplicated_object_layer_mints_fresh_object_ids() -> None:
    doc = _doc()
    layer = doc.add_object_layer("Objects")
    obj = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="rect"))

    copy = doc.duplicate_layer(layer.uid)
    made = copy.objects[0]
    assert made.uid != obj.uid
    assert made.id != obj.id, "ids are never reused"


def test_a_duplicate_is_one_undo_step() -> None:
    doc = _doc()
    original = doc.tile_layers()[0]
    before = len(doc.children_of(None))
    doc.duplicate_layer(original.uid)
    assert len(doc.children_of(None)) == before + 1
    assert doc.undo() is True
    assert len(doc.children_of(None)) == before


# --- merge down ---------------------------------------------------------------


def test_merge_down_is_nonzero_above_wins_per_cell() -> None:
    doc = _doc()
    below = doc.tile_layers()[0]
    above = doc.add_tile_layer("Above", index=0)
    doc.write_region(below.uid, 0, 0, np.array([[1, 2]], gidlib.DTYPE))
    doc.write_region(above.uid, 0, 0, np.array([[0, 9]], gidlib.DTYPE))

    assert doc.merge_down(above.uid) is True
    merged = doc.layer(below.uid).data
    assert int(merged[0, 0]) == 1, "an empty cell above lets the one below show"
    assert int(merged[0, 1]) == 9, "a painted cell above wins"


def test_merge_down_removes_the_layer_it_merged() -> None:
    doc = _doc()
    below = doc.tile_layers()[0]
    above = doc.add_tile_layer("Above", index=0)
    doc.merge_down(above.uid)
    assert [layer.uid for layer in doc.children_of(None)] == [below.uid]


def test_merge_down_leaves_presentation_alone() -> None:
    """A gid-level merge, deliberately not a pixel composite: there is no gid
    that means "this tile at 40%", and baking one would disagree with the
    renderer the first time the tint changed."""
    doc = _doc()
    below = doc.tile_layers()[0]
    above = doc.add_tile_layer("Above", index=0)
    doc.set_layer_props(above.uid, opacity=0.4)
    doc.write_region(above.uid, 0, 0, np.array([[9]], gidlib.DTYPE))

    doc.merge_down(above.uid)
    assert int(doc.layer(below.uid).data[0, 0]) == 9, "the gid lands whole"
    assert doc.layer(below.uid).opacity == 1.0, "the layer below keeps its own"


def test_merge_down_is_one_undo_step() -> None:
    doc = _doc()
    below = doc.tile_layers()[0]
    above = doc.add_tile_layer("Above", index=0)
    doc.write_region(above.uid, 0, 0, np.array([[9]], gidlib.DTYPE))
    doc.write_region(below.uid, 1, 1, np.array([[3]], gidlib.DTYPE))

    doc.merge_down(above.uid)
    assert doc.undo() is True
    assert len(doc.children_of(None)) == 2
    assert int(doc.layer(below.uid).data[0, 0]) == 0
    assert int(doc.layer(below.uid).data[1, 1]) == 3


def test_merge_down_refuses_with_nothing_below() -> None:
    doc = _doc()
    only = doc.tile_layers()[0]
    with pytest.raises(ValueError, match="no layer below"):
        doc.merge_down(only.uid)


def test_merge_down_refuses_across_kinds_by_name() -> None:
    doc = _doc()
    doc.add_object_layer("Objects")  # lands at the end, below the tile layer
    above = doc.tile_layers()[0]
    with pytest.raises(ValueError, match="not a tile layer"):
        doc.merge_down(above.uid)


def test_merge_down_refuses_a_non_tile_layer() -> None:
    doc = _doc()
    objects = doc.add_object_layer("Objects", index=0)
    with pytest.raises(ValueError, match="not a tile layer"):
        doc.merge_down(objects.uid)


# --- image-layer pixels -------------------------------------------------------


def _picture(size: int = 4) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., :3] = 120
    out[..., 3] = 255
    return out


def test_attaching_a_picture_is_undoable() -> None:
    doc = _doc()
    layer = doc.add_image_layer("Picture")
    assert doc.set_image_pixels(layer.uid, _picture(), source="a.png") is True
    assert doc.layer(layer.uid).pixels.shape == (4, 4, 4)
    assert doc.layer(layer.uid).source == "a.png"

    doc.undo()  # the source change
    doc.undo()  # the pixels
    assert doc.layer(layer.uid).pixels.shape != (4, 4, 4)


def test_the_attached_picture_is_frozen() -> None:
    """The UI keys a texture upload on the array's identity, so an in-place edit
    would leave the cache holding a live key over stale pixels."""
    doc = _doc()
    layer = doc.add_image_layer("Picture")
    doc.set_image_pixels(layer.uid, _picture())
    with pytest.raises(ValueError):
        doc.layer(layer.uid).pixels[0, 0, 0] = 9


def test_attaching_the_same_picture_twice_pushes_nothing() -> None:
    doc = _doc()
    layer = doc.add_image_layer("Picture")
    picture = _picture()
    doc.set_image_pixels(layer.uid, picture, source="a.png")
    depth = len(doc.history)
    assert doc.set_image_pixels(layer.uid, picture, source="a.png") is False
    assert len(doc.history) == depth


def test_attaching_to_a_non_image_layer_is_refused() -> None:
    doc = _doc()
    with pytest.raises(KeyError):
        doc.set_image_pixels(doc.tile_layers()[0].uid, _picture())


# --- vertices -----------------------------------------------------------------


def test_a_moved_vertex_is_a_new_frozen_shape() -> None:
    """Shapes are frozen records, so a moved vertex is a new Polygon through
    ``merged_object_values`` -- the one reconciliation door."""
    from warlock.studio.plotter.tilemap import Polygon

    doc = _doc()
    layer = doc.add_object_layer("Objects")
    obj = doc.add_object(
        layer.uid,
        MapObject(
            uid=new_uid(),
            shape=Polygon(((0.0, 0.0), (16.0, 0.0), (16.0, 16.0))),
        ),
    )
    before = obj.shape

    doc.set_object(
        layer.uid, obj.uid, shape=Polygon(((0.0, 0.0), (32.0, 0.0), (16.0, 16.0)))
    )
    assert doc.layer(layer.uid).objects[0].shape != before
    assert before.points == ((0.0, 0.0), (16.0, 0.0), (16.0, 16.0)), "frozen"


def test_a_polygon_floors_at_three_points() -> None:
    from warlock.studio.plotter.tilemap import Polygon

    with pytest.raises(ValueError):
        Polygon(((0.0, 0.0), (1.0, 1.0)))


def test_a_polyline_floors_at_two_points() -> None:
    from warlock.studio.plotter.tilemap import Polyline

    with pytest.raises(ValueError):
        Polyline(((0.0, 0.0),))


def test_a_vertex_converts_through_the_rotation_not_around_it() -> None:
    """The ``_resized`` trap by name: the outline is drawn by rotating each point
    about the object's origin, so a map point taken straight as a vertex lands
    somewhere the outline does not go."""
    from warlock.studio.panes import plotter_canvas as canvas

    point = (10.0, 0.0)
    turned = canvas._rotated_about(point, 90.0)
    assert abs(turned[0]) < 1e-9 and abs(turned[1] - 10.0) < 1e-9
    back = canvas._unrotated_about(turned, 90.0)
    assert abs(back[0] - 10.0) < 1e-9 and abs(back[1]) < 1e-9
