"""The flat renderer -- what an export and a library thumbnail are made of.

Deliberately not what the canvas draws: the canvas issues one textured quad per
*visible* cell through imgui's draw list. What both share is where a cell lands
and which way round it is, and both get those from ``gid`` and ``tile_rect``.
"""

from __future__ import annotations

import numpy as np
import pytest

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


def test_a_map_too_big_to_composite_is_refused_before_the_allocation(monkeypatch):
    """Both factors are the document's own numbers, so a legal map can ask for a
    quarter of a terabyte in one ``np.zeros``. The constant is lowered rather
    than such a map being built, which is why it is read at call time."""
    doc = _doc(4, 4)
    monkeypatch.setattr(render, "MAX_RENDER_PIXELS", 4)
    with pytest.raises(ValueError) as exc:
        render.render_map(doc)
    assert "past the 4" in str(exc.value)


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


def test_the_flat_render_places_an_isometric_cell_where_the_canvas_does():
    """The "two renderers agree" rule, asserted rather than intended. Both take
    placement from ``project`` now, so this is what would catch one of them
    growing its own arithmetic."""
    from warlock.studio.plotter import project

    doc = MapDoc(3, 3, 32, 16, projection="isometric")
    pixels = np.zeros((16, 32, 4), dtype=np.uint8)
    pixels[:, :] = (255, 0, 0, 255)
    tileset = Tileset(name="t", pixels=pixels, tile_w=32, tile_h=16)
    ref = doc.add_tileset(tileset)
    layer = doc.add_tile_layer("G")
    layer.data[1, 2] = gid.compose(ref.firstgid)

    out = render.render_map(doc)
    assert out.shape[:2] == (doc.pixel_height, doc.pixel_width)
    x0, y0 = project.cell_origin(project.Lattice("isometric", 3, 3, 32, 16), 2, 1)
    # The centre of that cell's diamond is opaque; the map's top-left corner,
    # which no diamond covers, is not.
    assert out[int(y0) + 8, int(x0) + 16, 3] == 255
    assert out[0, 0, 3] == 0


def test_an_isometric_render_is_the_size_the_projection_says():
    doc = MapDoc(4, 6, 32, 16, projection="isometric")
    doc.add_tile_layer("G")
    out = render.render_map(doc)
    # (rows, columns): the height comes from tile_h and the width from tile_w,
    # so a 2:1 cell makes the map twice as wide as it is tall.
    assert out.shape[:2] == ((4 + 6) * 16 // 2, (4 + 6) * 32 // 2)


def test_a_transformed_brush_stamps_what_the_flat_renderer_then_draws():
    """The two-renderer family, reached the other way round: the brush
    transforms in ``tools`` and the flag decoding in ``render`` have to agree,
    or a rotated brush stamps an arrangement whose tiles face the wrong way.

    Whole-pipeline rather than unit: transform a brush, stamp it, composite the
    map, and compare against the same rotation applied to the pixels of the map
    the untransformed brush produced.
    """
    from warlock.studio.plotter import tools

    # A 2x2 map of 2x2 tiles, so the composite is a 4x4 image a numpy rotation
    # can be compared against directly.
    def painted(brush: np.ndarray) -> np.ndarray:
        doc = MapDoc(2, 2, 2, 2)
        doc.add_tileset(_tileset())
        layer = doc.add_tile_layer("G")
        doc.write_region(layer.uid, 0, 0, np.asarray(brush, gid.DTYPE))
        return render.render_map(doc)

    # Built in the tileset's own id space, and with every cell filled. Offsetting
    # ids *after* a transform would not work: an empty cell deliberately never
    # gains a flag, so adding a firstgid afterwards turns "nothing here" into a
    # real unflipped tile and only in one of the two renders being compared.
    tile_a, tile_b = gid.compose(1), gid.compose(2)
    brush = np.array([[tile_a, tile_b], [tile_b, tile_a]], gid.DTYPE)
    flat = painted(brush)

    assert np.array_equal(painted(tools.flip_brush_h(brush)), flat[:, ::-1])
    assert np.array_equal(painted(tools.flip_brush_v(brush)), flat[::-1, :])
    assert np.array_equal(painted(tools.rotate_brush_cw(brush)), np.rot90(flat, k=-1))


# --- the minimap --------------------------------------------------------------


def test_a_minimap_is_one_pixel_per_cell():
    """Not a downscale of ``render_map``: that composites at full size, which
    for a large map is a nine-figure blend guarded by MAX_RENDER_PIXELS."""
    doc = MapDoc(6, 4, 32, 32)
    ref = doc.add_tileset(_tileset())
    layer = doc.add_tile_layer("G")
    layer.data[1, 2] = gid.compose(ref.firstgid + 1)  # the solid blue tile

    out = render.minimap(doc)
    assert out.shape == (4, 6, 4)
    assert tuple(out[1, 2][:3]) == (0, 0, 255)
    assert out[0, 0, 3] == 0, "an empty cell stays empty"


def test_a_minimap_ignores_the_flip_flags():
    """Every transform is a permutation of a tile's pixels, and a mean is
    invariant under all eight."""
    doc = MapDoc(2, 1, 32, 32)
    ref = doc.add_tileset(_tileset())
    layer = doc.add_tile_layer("G")
    layer.data[0, 0] = gid.compose(ref.firstgid)
    layer.data[0, 1] = gid.compose(ref.firstgid, flip_h=True, flip_v=True, flip_d=True)
    out = render.minimap(doc)
    assert np.array_equal(out[0, 0], out[0, 1])


def test_a_minimap_skips_hidden_and_transparent_layers():
    doc = MapDoc(2, 1, 32, 32)
    ref = doc.add_tileset(_tileset())
    hidden = doc.add_tile_layer("H")
    hidden.data[0, 0] = gid.compose(ref.firstgid + 1)
    doc.set_layer_props(hidden.uid, visible=False)
    assert not render.minimap(doc).any()

    doc.set_layer_props(hidden.uid, visible=True, opacity=0.0)
    assert not render.minimap(doc).any()


def test_a_minimap_composites_bottom_first():
    doc = MapDoc(1, 1, 32, 32)
    ref = doc.add_tileset(_tileset())
    bottom = doc.add_tile_layer("bottom")
    top = doc.add_tile_layer("top")
    bottom.data[0, 0] = gid.compose(ref.firstgid)  # red-cornered tile
    top.data[0, 0] = gid.compose(ref.firstgid + 1)  # solid blue
    assert tuple(render.minimap(doc)[0, 0][:3]) == (0, 0, 255), "the top layer wins"


def test_the_colour_table_is_memoised_on_the_pixel_array():
    """``Tileset`` freezes its pixels, so a matching id on a *live* array is the
    same art, and a replaced tileset is a new array alongside the old one."""
    tileset = _tileset()
    first = render.tile_colours(tileset)
    assert render.tile_colours(tileset) is first
    assert render.tile_colours(_tileset()) is not first


def test_the_colour_table_is_dropped_when_its_array_dies():
    """An id names the same art only for as long as that array is alive.
    CPython hands a freed block straight back out and ``replace_tileset``
    refuses a different tile count, so an entry outliving its array would pass
    both the key and the length check and serve the superseded set's colours to
    whichever atlas landed on its address."""
    import gc

    tileset = _tileset()
    key = id(tileset.pixels)
    render.tile_colours(tileset)
    assert key in render._LUT_CACHE

    del tileset
    gc.collect()
    assert key not in render._LUT_CACHE


def test_a_tile_colour_is_weighted_by_alpha():
    """An unweighted mean over a mostly-transparent tile is dominated by
    whatever is stored under the zero alpha -- routinely black, which would draw
    a sparse map as a dark smear."""
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0, 255)  # one opaque red pixel, three clear
    tileset = Tileset(name="sparse", pixels=pixels, tile_w=2, tile_h=2)
    colour = render.tile_colours(tileset)[0]
    assert tuple(colour[:3]) == (255, 0, 0), "the visible pixel decides the hue"
    assert 0 < int(colour[3]) < 255, "and coverage lives in the alpha"


def test_a_minimap_of_a_map_with_no_tilesets_is_empty():
    doc = MapDoc(3, 3, 16, 16)
    doc.add_tile_layer("G")
    assert render.minimap(doc).shape == (3, 3, 4)
    assert not render.minimap(doc).any()
