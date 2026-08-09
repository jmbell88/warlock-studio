"""The flat renderer -- what an export and a library thumbnail are made of.

Deliberately not what the canvas draws: the canvas issues one textured quad per
*visible* cell through imgui's draw list. What both share is where a cell lands
and which way round it is, and both get those from ``gid`` and ``tile_rect``.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter import gid, render
from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.plotter.tileset import Tileset


# A 2x1 tileset of 2x2 tiles: tile 0 has one opaque red pixel at its top-left,
# tile 1 is solid blue. Small enough to assert pixel by pixel.
def _tileset() -> Tileset:
    pixels = np.zeros((2, 4, 4), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0, 255)
    pixels[:, 2:4] = (0, 0, 255, 255)
    return Tileset(name="t", pixels=pixels, tile_w=2, tile_h=2)


def _doc(width: int = 2, height: int = 1) -> MapDoc:
    doc = MapDoc(width, height, 2, 2)
    doc.add_tileset(_tileset())
    return doc


def test_an_empty_map_renders_transparent_at_its_pixel_size():
    doc = _doc(3, 2)
    doc.add_tile_layer()
    out = render.render_map(doc)
    assert out.shape == (4, 6, 4)
    assert not out.any()


def test_a_tile_lands_in_its_own_cell():
    doc = _doc()
    layer = doc.add_tile_layer()
    doc.write_region(layer.uid, 0, 0, np.array([[1, 2]], gid.DTYPE))
    out = render.render_map(doc)
    assert tuple(out[0, 0]) == (255, 0, 0, 255)   # tile 0's marked pixel
    assert tuple(out[0, 1]) == (0, 0, 0, 0)
    assert tuple(out[0, 2]) == (0, 0, 255, 255)   # tile 1 fills its cell
    assert tuple(out[1, 3]) == (0, 0, 255, 255)


def test_each_flag_moves_the_marked_pixel_where_it_should():
    """The eight symmetries, checked on the one asymmetric tile -- which is the
    only way to tell a horizontal flip from a vertical one."""
    corners = {
        (False, False, False): (0, 0),
        (True, False, False): (0, 1),
        (False, True, False): (1, 0),
        (True, True, False): (1, 1),
        # A diagonal flip is a transpose, so on a tile whose only mark is at
        # (0, 0) it alone changes nothing -- which is why it is checked in
        # combination.
        (False, False, True): (0, 0),
        (True, False, True): (0, 1),
        (False, True, True): (1, 0),
    }
    for (h, v, d), (row, column) in corners.items():
        doc = _doc(1, 1)
        layer = doc.add_tile_layer()
        cell = gid.compose(1, flip_h=h, flip_v=v, flip_d=d)
        doc.write_region(layer.uid, 0, 0, np.array([[cell]], gid.DTYPE))
        out = render.render_map(doc)
        assert tuple(out[row, column]) == (255, 0, 0, 255), (h, v, d)
        assert int(out[..., 3].sum()) == 255


def test_layers_composite_bottom_first():
    doc = _doc(1, 1)
    bottom = doc.add_tile_layer("bottom")
    top = doc.add_tile_layer("top")
    doc.write_region(bottom.uid, 0, 0, np.array([[2]], gid.DTYPE))  # solid blue
    doc.write_region(top.uid, 0, 0, np.array([[1]], gid.DTYPE))     # one red pixel
    out = render.render_map(doc)
    assert tuple(out[0, 0]) == (255, 0, 0, 255)  # the top layer wins where it paints
    assert tuple(out[0, 1]) == (0, 0, 255, 255)  # and the bottom shows elsewhere


def test_a_hidden_layer_is_not_drawn():
    """One flag decides what you see and what comes out; two answers to that is
    how an export starts disagreeing with the screen."""
    doc = _doc(1, 1)
    layer = doc.add_tile_layer()
    doc.write_region(layer.uid, 0, 0, np.array([[2]], gid.DTYPE))
    doc.set_layer_props(layer.uid, visible=False)
    assert not render.render_map(doc).any()
    assert render.render_map(doc, include_hidden=True).any()


def test_a_zero_opacity_layer_is_skipped_entirely():
    doc = _doc(1, 1)
    layer = doc.add_tile_layer()
    doc.write_region(layer.uid, 0, 0, np.array([[2]], gid.DTYPE))
    doc.set_layer_props(layer.uid, opacity=0.0)
    assert not render.render_map(doc).any()


def test_a_half_opaque_layer_composites_rather_than_replacing():
    doc = _doc(1, 1)
    bottom = doc.add_tile_layer("bottom")
    top = doc.add_tile_layer("top")
    doc.write_region(bottom.uid, 0, 0, np.array([[2]], gid.DTYPE))
    doc.write_region(top.uid, 0, 0, np.array([[2]], gid.DTYPE))
    doc.set_layer_props(top.uid, opacity=0.5)
    out = render.render_map(doc)
    assert tuple(out[0, 0]) == (0, 0, 255, 255)  # same colour, still opaque


def test_an_object_layer_contributes_no_pixels():
    """Objects are metadata an engine reads; drawing the editor's handles into
    an export would be drawing the ruler onto the drawing."""
    from warlock.studio.plotter.tilemap import MapObject, new_uid

    doc = _doc(1, 1)
    layer = doc.add_object_layer()
    doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="rect", w=2, h=2))
    assert not render.render_map(doc).any()


def test_a_tile_larger_than_the_grid_grows_upward_out_of_its_cell():
    """Tiled anchors an oversized tile by its bottom-left, which is what makes
    a 48px tree sit on a 32px floor rather than float above it."""
    big = np.zeros((4, 4, 4), dtype=np.uint8)
    big[...] = (0, 255, 0, 255)
    doc = MapDoc(2, 2, 2, 2)
    doc.add_tileset(Tileset(name="big", pixels=big, tile_w=4, tile_h=4))
    layer = doc.add_tile_layer()
    doc.write_region(layer.uid, 0, 1, np.array([[1]], gid.DTYPE))
    out = render.render_map(doc)
    # The cell is rows 2-3; the tile is 4 tall, so it occupies rows 0-3.
    assert tuple(out[0, 0]) == (0, 255, 0, 255)
    assert tuple(out[3, 3]) == (0, 255, 0, 255)


def test_rendering_twice_gives_the_same_pixels():
    doc = _doc()
    layer = doc.add_tile_layer()
    doc.write_region(layer.uid, 0, 0, np.array([[1, 2]], gid.DTYPE))
    assert np.array_equal(render.render_map(doc), render.render_map(doc))
