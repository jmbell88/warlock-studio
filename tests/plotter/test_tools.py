"""What each tool computes, from plain arrays.

Two things carry most of the weight. Clipping: a brush dragged off the edge is a
legitimate stroke whose visible part must land, and only a placement *entirely*
outside the map is nothing. And the flood fill matching on the **full encoded
value**, which is what makes a mirrored wall tile bound a fill of its unmirrored
twin -- the seam the user drew them to make.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter import gid, tools


def _layer(width: int = 6, height: int = 4) -> np.ndarray:
    return gid.empty_layer(width, height)


def _brush(value: int, w: int = 2, h: int = 2) -> np.ndarray:
    return np.full((h, w), value, gid.DTYPE)


# --- stamp --------------------------------------------------------------------


def test_a_stamp_lands_whole_when_it_fits():
    result = tools.stamp(_layer(), 1, 1, _brush(7))
    assert result is not None
    x0, y0, region = result
    assert (x0, y0) == (1, 1)
    assert region.tolist() == [[7, 7], [7, 7]]


def test_a_stamp_hanging_off_the_top_left_is_clipped_not_refused():
    result = tools.stamp(_layer(), -1, -1, _brush(7, 3, 3))
    assert result is not None
    x0, y0, region = result
    assert (x0, y0) == (0, 0)
    assert region.shape == (2, 2)


def test_a_stamp_hanging_off_the_bottom_right_is_clipped():
    result = tools.stamp(_layer(6, 4), 5, 3, _brush(7))
    assert result is not None
    assert result[2].shape == (1, 1)


def test_a_stamp_entirely_off_the_map_is_nothing():
    assert tools.stamp(_layer(), 20, 20, _brush(7)) is None
    assert tools.stamp(_layer(), -5, 0, _brush(7)) is None


def test_a_stamp_replaces_wholesale_including_empty_cells():
    """A zero meaning "leave what is there" would make a single-tile eraser
    impossible to express and give the palette's own empty corner a hidden
    meaning."""
    brush = np.array([[5, 0]], gid.DTYPE)
    result = tools.stamp(_layer(), 0, 0, brush)
    assert result is not None and result[2].tolist() == [[5, 0]]


# --- rectangles ---------------------------------------------------------------


def test_a_rectangle_accepts_its_corners_in_any_order():
    a = tools.fill_rect(_layer(), 4, 3, 1, 1, 2)
    b = tools.fill_rect(_layer(), 1, 1, 4, 3, 2)
    assert a is not None and b is not None
    assert (a[0], a[1]) == (b[0], b[1]) == (1, 1)
    assert a[2].shape == b[2].shape == (3, 4)


def test_a_rectangle_is_clipped_to_the_map():
    result = tools.fill_rect(_layer(6, 4), -3, -3, 1, 1, 2)
    assert result is not None
    assert (result[0], result[1]) == (0, 0)
    assert result[2].shape == (2, 2)


def test_a_rectangle_entirely_off_the_map_is_nothing():
    assert tools.fill_rect(_layer(), 10, 10, 20, 20, 2) is None


def test_erasing_is_filling_with_zero():
    result = tools.erase(_layer(), 1, 1, 2, 2)
    assert result is not None
    assert not result[2].any()


# --- flood fill ---------------------------------------------------------------


def test_a_fill_covers_the_contiguous_run_and_nothing_else():
    layer = _layer(5, 3)
    layer[1, :] = 9  # a wall across the middle
    result = tools.flood_fill(layer, 0, 0, 4)
    assert result is not None
    x0, y0, region = result
    assert (x0, y0) == (0, 0)
    assert region.tolist() == [[4, 4, 4, 4, 4]]


def test_a_fill_leaves_untouched_cells_inside_its_bounding_box():
    """An L-shaped room must not be rectangled over the wall in its notch."""
    layer = _layer(3, 3)
    layer[0, 2] = 9
    layer[1, 2] = 9
    result = tools.flood_fill(layer, 0, 0, 4)
    assert result is not None
    assert result[2].tolist() == [[4, 4, 9], [4, 4, 9], [4, 4, 4]]


def test_a_fill_matches_on_the_full_encoded_value_so_a_flip_is_a_wall():
    """The property the whole flags-are-carried-everywhere design buys: two
    cells that draw differently are two different cells."""
    layer = _layer(4, 1)
    plain = gid.compose(3)
    mirrored = gid.compose(3, flip_h=True)
    layer[0, :] = plain
    layer[0, 2] = mirrored
    result = tools.flood_fill(layer, 0, 0, 8)
    assert result is not None
    assert result[2].tolist() == [[8, 8]]


def test_a_fill_is_four_connected():
    """Eight-connected leaks through a corner where two walls only touch at a
    point -- exactly the seam a room's corner tiles make."""
    layer = _layer(3, 3)
    layer[0, 1] = 9
    layer[1, 0] = 9
    result = tools.flood_fill(layer, 0, 0, 4)
    assert result is not None
    assert result[2].tolist() == [[4]]


def test_a_fill_with_what_is_already_there_is_nothing():
    layer = _layer()
    assert tools.flood_fill(layer, 0, 0, 0) is None


def test_a_fill_off_the_map_is_nothing():
    assert tools.flood_fill(_layer(), 99, 0, 1) is None


def test_a_fill_never_promotes_the_dtype():
    layer = _layer()
    result = tools.flood_fill(layer, 0, 0, gid.compose(1, flip_h=True))
    assert result is not None and result[2].dtype == gid.DTYPE


# --- pick ---------------------------------------------------------------------


def test_pick_returns_the_encoded_cell_or_none():
    layer = _layer()
    value = gid.compose(6, flip_v=True)
    layer[2, 3] = value
    assert tools.pick(layer, 3, 2) == value
    assert tools.pick(layer, 99, 0) is None


# --- nothing mutates ----------------------------------------------------------


def test_no_tool_touches_the_array_it_is_given():
    """Every tool is a pure function; the document owns the write and the diff,
    which is what gets the no-op rule applied in one place."""
    layer = _layer()
    original = layer.copy()
    tools.stamp(layer, 0, 0, _brush(3))
    tools.fill_rect(layer, 0, 0, 2, 2, 3)
    tools.erase(layer, 0, 0, 2, 2)
    tools.flood_fill(layer, 0, 0, 3)
    assert np.array_equal(layer, original)
