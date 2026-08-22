"""Tile-tool parity: capture, pattern and random fill, masked selections, map ops.

The two properties everything here leans on: a 1x1 brush must be byte-identical
to the plain fill it replaces (so nothing that already worked moved), and a
bounded flood must be bounded *up front* (so a fill cannot escape a concave
selection, run around the outside and come back in with the trip hidden).
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter import tools
from warlock.studio.tilegrid import gid as gidlib


def _map(width: int = 8, height: int = 8) -> np.ndarray:
    return gidlib.empty_layer(width, height)


# --- K1: capture --------------------------------------------------------------


def test_a_capture_round_trips_byte_equal_with_its_flags() -> None:
    """Encoded gids verbatim: a captured wall that was mirrored stamps back
    mirrored, and stripping the flags would un-mirror half of anything."""
    data = _map()
    data[2, 3] = gidlib.compose(7, flip_h=True)
    data[2, 4] = gidlib.compose(8, flip_v=True, flip_d=True)
    data[3, 3] = gidlib.compose(9)

    block = tools.capture(data, 3, 2, 4, 3)
    assert block is not None
    assert block.shape == (2, 2)
    assert np.array_equal(block, data[2:4, 3:5])

    target = _map()
    x0, y0, region = tools.stamp(target, 0, 0, block)
    assert np.array_equal(region, block)


def test_an_all_empty_capture_is_refused() -> None:
    """It would arm a brush that erases everything it touches while calling
    itself a stamp."""
    assert tools.capture(_map(), 1, 1, 4, 4) is None


def test_a_capture_off_the_map_is_refused() -> None:
    data = _map()
    data[0, 0] = 5
    assert tools.capture(data, 20, 20, 25, 25) is None


def test_a_capture_clips_to_the_map() -> None:
    data = _map(4, 4)
    data[3, 3] = 5
    block = tools.capture(data, 2, 2, 99, 99)
    assert block is not None
    assert block.shape == (2, 2)


# --- K2: pattern and random fill ---------------------------------------------


def test_a_one_by_one_brush_fill_is_byte_identical_to_the_plain_one() -> None:
    """The regression pin: nothing that already worked moved."""
    data = _map()
    plain = tools.fill_rect(data, 1, 1, 5, 4, 9)
    patterned = tools.fill_rect(data, 1, 1, 5, 4, 9, brush=np.array([[9]], gidlib.DTYPE))
    assert plain is not None and patterned is not None
    assert plain[:2] == patterned[:2]
    assert np.array_equal(plain[2], patterned[2])


def test_a_pattern_fill_tiles_the_brush_over_map_coordinates() -> None:
    data = _map()
    brush = np.array([[1, 2], [3, 4]], gidlib.DTYPE)
    x0, y0, region = tools.fill_rect(data, 0, 0, 3, 3, 0, brush=brush)
    assert (x0, y0) == (0, 0)
    assert np.array_equal(region, np.tile(brush, (2, 2)))


def test_two_separate_fills_continue_one_pattern() -> None:
    """Anchored to *map* coordinates, not to each fill's own origin -- the
    "position mod k" rule terrain phases already use."""
    data = _map()
    brush = np.array([[1, 2], [3, 4]], gidlib.DTYPE)
    left = tools.fill_rect(data, 0, 0, 1, 1, 0, brush=brush)
    right = tools.fill_rect(data, 2, 0, 3, 1, 0, brush=brush)
    assert np.array_equal(left[2], right[2]), "the second fill resumes the pattern"

    odd = tools.fill_rect(data, 1, 0, 2, 1, 0, brush=brush)
    # Starting on an odd column picks up the pattern's second column first.
    assert odd[2][0, 0] == 2


def test_brush_at_is_the_pattern_rule() -> None:
    brush = np.array([[1, 2, 3], [4, 5, 6]], gidlib.DTYPE)
    assert tools.brush_at(brush, 0, 0) == 1
    assert tools.brush_at(brush, 4, 3) == 5  # brush[3 % 2, 4 % 3] = brush[1, 1]
    assert tools.brush_at(brush, 2, 0) == 3


def test_a_random_fill_writes_only_brush_members() -> None:
    data = _map()
    brush = np.array([[11, 12], [13, 0]], gidlib.DTYPE)
    rng = np.random.default_rng(3)
    _x, _y, region = tools.fill_rect(data, 0, 0, 7, 7, 0, brush=brush, rng=rng)
    assert set(np.unique(region).tolist()) <= {11, 12, 13}
    assert 0 not in set(np.unique(region).tolist()), "empty members are never chosen"


def test_a_random_fill_is_deterministic_under_a_given_generator() -> None:
    data = _map()
    brush = np.array([[11, 12, 13]], gidlib.DTYPE)
    first = tools.fill_rect(
        data, 0, 0, 7, 7, 0, brush=brush, rng=np.random.default_rng(5)
    )
    second = tools.fill_rect(
        data, 0, 0, 7, 7, 0, brush=brush, rng=np.random.default_rng(5)
    )
    assert np.array_equal(first[2], second[2])


def test_a_pattern_ellipse_leaves_its_corners_alone() -> None:
    data = _map()
    data[:] = 99
    brush = np.array([[1, 2], [3, 4]], gidlib.DTYPE)
    _x, _y, region = tools.fill_ellipse(data, 0, 0, 5, 5, 0, brush=brush)
    assert int(region[0, 0]) == 99, "an ellipse must not rectangle over its corners"
    assert int(region[3, 3]) in (1, 2, 3, 4)


def test_a_pattern_flood_fills_exactly_the_room() -> None:
    """The brush decides what is written, never what bounds the fill: the match
    is still the single target cell's encoded gid."""
    data = _map()
    data[:, 4] = 99  # a wall
    brush = np.array([[1, 2], [3, 4]], gidlib.DTYPE)
    x0, y0, region = tools.flood_fill(data, 0, 0, 1, brush=brush)
    assert (x0, y0) == (0, 0)
    assert region.shape == (8, 4)
    assert set(np.unique(region).tolist()) == {1, 2, 3, 4}


# --- K3: masked selections ----------------------------------------------------


def test_a_masked_clip_puts_the_refused_cells_back() -> None:
    """Writing them unchanged rather than not writing them keeps a placement one
    rectangular write, which undo and the dirty rect are both built around."""
    data = _map()
    data[:] = 99
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    region = tools.fill_rect(data, 0, 0, 7, 7, 5)
    out = tools.clip_region_mask(region, data, (0, 0, 7, 7), mask)
    assert out is not None
    x0, y0, block = out
    assert (x0, y0) == (0, 0)
    assert int(block[2, 2]) == 5
    assert int(block[0, 0]) == 99


def test_a_clip_with_no_mask_is_the_rectangle_trim_it_always_was() -> None:
    data = _map()
    region = tools.fill_rect(data, 0, 0, 7, 7, 5)
    with_mask = tools.clip_region_mask(region, data, (2, 2, 5, 5), None)
    without = tools.clip_region(region, (2, 2, 5, 5))
    assert with_mask[:2] == without[:2]
    assert np.array_equal(with_mask[2], without[2])


def test_a_clip_whose_mask_holds_nothing_is_nothing() -> None:
    data = _map()
    region = tools.fill_rect(data, 0, 0, 7, 7, 5)
    mask = np.zeros((8, 8), dtype=bool)
    assert tools.clip_region_mask(region, data, (0, 0, 7, 7), mask) is None


def test_a_bounded_flood_cannot_escape_a_concave_mask() -> None:
    """The escape the up-front bound exists to prevent: a fill trimmed at the
    end would leave the selection, run around the outside and come back in, and
    the trim would hide the trip.
    """
    data = _map(9, 9)  # entirely empty, so everything is connected
    # A C-shaped selection: the two arms are only joined through cells outside
    # it, so a fill seeded in one arm must not reach the other.
    mask = np.zeros((9, 9), dtype=bool)
    mask[0, 0:4] = True
    mask[0:9, 0] = True
    mask[8, 0:4] = True

    x0, y0, region = tools.flood_fill(data, 3, 0, 7, mask=mask)
    reached = np.zeros((9, 9), dtype=bool)
    reached[y0 : y0 + region.shape[0], x0 : x0 + region.shape[1]] = region == 7
    assert reached[0, 3], "the seed's own arm fills"
    assert reached[8, 0], "and the arms really are joined inside the mask"
    assert not reached[4, 4], "nothing outside the mask is ever touched"
    assert (reached & ~mask).sum() == 0


def test_a_flood_seeded_outside_the_mask_fills_nothing() -> None:
    data = _map()
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:2, 0:2] = True
    assert tools.flood_fill(data, 5, 5, 7, mask=mask) is None


def test_the_wand_set_is_the_floods_reached_set() -> None:
    """One helper, three callers -- and the C kernel behind it serves all."""
    data = _map()
    data[:, 4] = 99
    same = data == data[0, 0]
    reached = tools.flood_mask(same, 0, 0)
    assert reached[:, :4].all()
    assert not reached[:, 4:].any()


# --- K4/K5: offset and autocrop ----------------------------------------------


def _doc(width: int = 8, height: int = 8):
    from warlock.studio.plotter.tilemap import MapDoc

    doc = MapDoc(width, height, 16, 16)
    doc.add_tile_layer("Tiles")
    return doc


def test_four_quarter_wraps_are_the_identity() -> None:
    doc = _doc()
    layer = doc.tile_layers()[0]
    doc.write_region(layer.uid, 1, 1, np.array([[5, 6]], gidlib.DTYPE))
    before = np.array(doc.tile_layers()[0].data)

    for _ in range(4):
        assert doc.offset(2, 2, wrap=True) is True
    assert np.array_equal(doc.tile_layers()[0].data, before)


def test_an_unwrapped_offset_zero_fills_what_it_vacates() -> None:
    doc = _doc()
    layer = doc.tile_layers()[0]
    doc.write_region(layer.uid, 0, 0, np.array([[5]], gidlib.DTYPE))

    assert doc.offset(2, 0, wrap=False) is True
    data = doc.tile_layers()[0].data
    assert int(data[0, 2]) == 5
    assert int(data[0, 0]) == 0


def test_an_unwrapped_offset_drops_what_leaves_the_map() -> None:
    doc = _doc()
    layer = doc.tile_layers()[0]
    doc.write_region(layer.uid, 7, 0, np.array([[5]], gidlib.DTYPE))
    assert doc.offset(2, 0, wrap=False) is True
    assert int(np.asarray(doc.tile_layers()[0].data).sum()) == 0


def test_a_zero_offset_does_nothing() -> None:
    doc = _doc()
    assert doc.offset(0, 0) is False
    assert doc.offset(8, 8, wrap=True) is False, "a whole map is a zero offset"


def test_an_offset_is_one_undo_step() -> None:
    doc = _doc()
    layer = doc.tile_layers()[0]
    doc.write_region(layer.uid, 1, 1, np.array([[5]], gidlib.DTYPE))
    before = np.array(doc.tile_layers()[0].data)
    doc.offset(3, 2)
    assert doc.undo() is True
    assert np.array_equal(doc.tile_layers()[0].data, before)


def test_only_whole_map_scope_moves_objects() -> None:
    from warlock.studio.plotter.tilemap import MapObject, new_uid

    doc = _doc()
    layer = doc.add_object_layer("Objects")
    doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="a", kind="rect", x=32.0, y=48.0)
    )

    doc.offset(1, 0, scope="layer")
    assert doc.layer(layer.uid).objects[0].x == 32.0

    tile = doc.tile_layers()[0]
    doc.set_active_layer(tile.uid)
    doc.offset(1, 0, scope="map")
    assert doc.layer(layer.uid).objects[0].x == 48.0


def test_a_wrapped_offset_wraps_objects_with_the_cells() -> None:
    """The cells are normalized by modulo, so an object shifted by the
    normalized amount *un*-wrapped rode ``offset(-1)`` seven tiles right on an
    8-wide map -- off the geometry it annotates."""
    from warlock.studio.plotter.tilemap import MapObject, new_uid

    doc = _doc()  # 8x8 of 16px cells: 128px across
    layer = doc.add_object_layer("Objects")
    doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="a", kind="rect", x=32.0, y=48.0)
    )

    doc.offset(-1, 0, wrap=True)
    assert doc.layer(layer.uid).objects[0].x == 16.0, "one tile left, not seven right"
    doc.offset(3, 0, wrap=True)  # 16 + 48 stays inside the 128px map
    assert doc.layer(layer.uid).objects[0].x == 64.0
    doc.offset(5, 0, wrap=True)  # 64 + 80 wraps past the edge
    assert doc.layer(layer.uid).objects[0].x == 16.0


def test_a_wrapped_offset_and_back_is_the_identity_for_objects_too() -> None:
    """The docstring's identity claim, stated for the half of the document that
    used to break it."""
    from warlock.studio.plotter.tilemap import MapObject, new_uid

    doc = _doc()
    layer = doc.add_object_layer("Objects")
    doc.add_object(
        layer.uid, MapObject(uid=new_uid(), name="a", kind="rect", x=32.0, y=48.0)
    )
    doc.offset(1, 2, wrap=True)
    doc.offset(-1, -2, wrap=True)
    obj = doc.layer(layer.uid).objects[0]
    assert (obj.x, obj.y) == (32.0, 48.0)


def test_an_offset_scope_must_be_named() -> None:
    with pytest.raises(ValueError):
        _doc().offset(1, 1, scope="everything")


def test_autocrop_is_the_tight_box() -> None:
    doc = _doc(10, 10)
    layer = doc.tile_layers()[0]
    doc.write_region(layer.uid, 3, 4, np.array([[5, 6], [7, 8]], gidlib.DTYPE))

    assert doc.autocrop() is True
    assert (doc.width, doc.height) == (2, 2)
    assert np.array_equal(
        doc.tile_layers()[0].data, np.array([[5, 6], [7, 8]], gidlib.DTYPE)
    )


def test_autocrop_spans_every_tile_layer() -> None:
    doc = _doc(10, 10)
    first = doc.tile_layers()[0]
    second = doc.add_tile_layer("Second")
    doc.write_region(first.uid, 1, 1, np.array([[5]], gidlib.DTYPE))
    doc.write_region(second.uid, 8, 8, np.array([[6]], gidlib.DTYPE))

    assert doc.autocrop() is True
    assert (doc.width, doc.height) == (8, 8)


def test_autocrop_refuses_an_empty_map() -> None:
    doc = _doc()
    assert doc.autocrop() is False
    assert (doc.width, doc.height) == (8, 8)


def test_content_bounds_of_an_empty_map_is_none() -> None:
    assert _doc().content_bounds() is None


# --- terrain on an offset lattice (Part P4) -----------------------------------


def test_the_blob_neighbourhood_is_a_square_lattices() -> None:
    """Why terrain painting is refused by name on a staggered or hexagonal map
    rather than done wrongly: ``blob.masks_from`` reads the eight *shifted
    slices* of an array, and on an offset lattice those are not the
    neighbouring cells -- every other row is pushed sideways, and a hexagon has
    six neighbours rather than eight."""
    from warlock.studio.tilegrid import blob

    field = np.zeros((3, 3), dtype=bool)
    field[1, 1] = True
    masks = blob.masks_from(field, outside=False)
    # The cell up-and-left sees its south-east neighbour, which is only true
    # because the lattice is square.
    assert masks[0, 0] & blob.SE
    assert not masks[0, 0] & blob.N


# -- an infinite map has no edge, so it neither wraps nor clears --------------


def _infinite(width: int = 8, height: int = 8):
    doc = _doc(width, height)
    doc.set_infinite(True)
    return doc


def test_wrapping_an_infinite_map_is_refused_by_name() -> None:
    """``np.roll`` over a *window* whose size and position are an artifact of
    painting history: two documents with identical true content offset
    differently depending on how they happened to be framed."""

    import pytest

    doc = _infinite()
    with pytest.raises(ValueError, match="no edge to wrap around"):
        doc.offset(2, 0, wrap=True)


def test_an_infinite_whole_map_offset_slides_the_origin_and_loses_nothing() -> None:
    """The clear-what-was-vacated branch discarded content pushed past a line
    that does not exist. On an infinite map the shift is a translation: the
    window's corner moves and every array stays exactly as it was."""

    doc = _infinite()
    layer = doc.tile_layers()[0]
    doc.write_region(layer.uid, 0, 0, np.array([[5]], gidlib.DTYPE))
    before = np.array(doc.tile_layers()[0].data)
    origin = (doc.origin_x, doc.origin_y)

    assert doc.offset(-3, -2, wrap=False) is True
    assert np.array_equal(doc.tile_layers()[0].data, before), "no cell was lost"
    assert (doc.origin_x, doc.origin_y) == (origin[0] + 3, origin[1] + 2)


def test_an_infinite_offset_is_one_undo_step_that_puts_the_origin_back() -> None:
    """The old ``ResizeEdit`` push omitted both origin fields, so even where
    the shift happened to be lossless the undo left every cell at a different
    true coordinate than it was painted at."""

    doc = _infinite()
    origin = (doc.origin_x, doc.origin_y)
    assert doc.offset(2, 3, wrap=False) is True
    assert doc.undo() is True
    assert (doc.origin_x, doc.origin_y) == origin


def test_an_infinite_layer_offset_grows_the_window_rather_than_clipping() -> None:
    """Layer scope cannot be an origin slide -- that would move every layer --
    so the window grows by exactly what the shifted layer needs, as one step,
    and the untouched layers keep their true coordinates."""

    doc = _infinite()
    lower = doc.tile_layers()[0]
    upper = doc.add_tile_layer("Upper")
    doc.write_region(lower.uid, 0, 0, np.array([[9]], gidlib.DTYPE))
    doc.write_region(upper.uid, 7, 0, np.array([[5]], gidlib.DTYPE))
    doc.set_active_layer(upper.uid)
    origin_x = doc.origin_x

    assert doc.offset(3, 0, wrap=False, scope="layer") is True
    assert doc.width == 11, "the window grew to hold the shifted cell"
    assert int(np.asarray(doc.layer(upper.uid).data).sum()) == 5, "nothing was lost"
    # The lower layer did not move, and the origin did not have to move either:
    # the growth was all on the right-hand side.
    assert doc.origin_x == origin_x
    assert int(doc.layer(lower.uid).data[0, 0]) == 9
    assert doc.undo() is True
    assert doc.width == 8
