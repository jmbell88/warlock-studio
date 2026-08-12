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


# --- the mask under both fills ------------------------------------------------
#
# ``flood_mask`` replaced a per-cell ``deque`` that ``terrain.fill_terrain``
# carried a verbatim copy of. The bar is that it reaches exactly the same cells,
# so it is checked against a hand-rolled queue rather than against itself.


def _bfs(match: np.ndarray, x: int, y: int) -> np.ndarray:
    """The shape ``flood_mask`` replaced, kept here as the thing it must equal."""
    from collections import deque

    height, width = match.shape
    seen = np.zeros((height, width), dtype=bool)
    if not match[y, x]:
        return seen
    seen[y, x] = True
    queue = deque([(x, y)])
    while queue:
        cx, cy = queue.popleft()
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < width and 0 <= ny < height and not seen[ny, nx] and match[ny, nx]:
                seen[ny, nx] = True
                queue.append((nx, ny))
    return seen


def test_the_mask_reaches_what_a_queue_reaches_over_random_fields():
    rng = np.random.default_rng(20260811)
    for _ in range(12):
        match = rng.random((64, 64)) < 0.55
        x, y = int(rng.integers(64)), int(rng.integers(64))
        assert np.array_equal(tools.flood_mask(match, x, y), _bfs(match, x, y))


def test_the_mask_turns_a_pinch_the_way_a_queue_does():
    """An L with a one-cell waist: the case a dilation gets wrong if it grows
    diagonally, and the case a fill escapes through if it is eight-connected."""
    match = np.zeros((5, 5), dtype=bool)
    match[0, :] = True
    match[:, 0] = True
    match[4, 4] = True  # reachable only diagonally, so not reachable at all
    seen = tools.flood_mask(match, 0, 0)
    assert np.array_equal(seen, _bfs(match, 0, 0))
    assert not seen[4, 4]


def test_the_mask_of_a_point_that_does_not_match_is_empty():
    match = np.zeros((3, 3), dtype=bool)
    assert not tools.flood_mask(match, 1, 1).any()


# --- pick ---------------------------------------------------------------------


def test_pick_returns_the_encoded_cell_or_none():
    layer = _layer()
    value = gid.compose(6, flip_v=True)
    layer[2, 3] = value
    assert tools.pick(layer, 3, 2) == value
    assert tools.pick(layer, 99, 0) is None


# --- transforming a brush -----------------------------------------------------


def _corner_tile() -> np.ndarray:
    """A 2x2 tile with four distinct pixels, so every one of the eight square
    symmetries produces a different picture."""
    tile = np.zeros((2, 2, 4), dtype=np.uint8)
    for index, (row, column) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        tile[row, column] = (index + 1, 0, 0, 255)
    return tile


def _drawn(cell: int, tile: np.ndarray) -> np.ndarray:
    """What one encoded cell actually renders as."""
    from warlock.studio.plotter.render import orient

    _id, flip_h, flip_v, flip_d = gid.decompose(int(cell))
    return orient(tile, flip_h, flip_v, flip_d)


def test_flipping_a_brush_agrees_with_the_renderer_for_every_flag():
    """The oracle, and the reason there is no hand-written permutation table:
    for all eight starting flag masks, the transformed cell must *draw* as the
    numpy transform of what the original drew."""
    tile = _corner_tile()
    for mask in range(8):
        cell = gid.compose(
            5,
            flip_h=bool(mask & 1),
            flip_v=bool(mask & 2),
            flip_d=bool(mask & 4),
        )
        brush = np.array([[cell]], gid.DTYPE)
        before = _drawn(cell, tile)

        got_h = _drawn(int(tools.flip_brush_h(brush)[0, 0]), tile)
        assert np.array_equal(got_h, before[:, ::-1]), f"flip_h wrong for mask {mask}"

        got_v = _drawn(int(tools.flip_brush_v(brush)[0, 0]), tile)
        assert np.array_equal(got_v, before[::-1, :]), f"flip_v wrong for mask {mask}"

        got_r = _drawn(int(tools.rotate_brush_cw(brush)[0, 0]), tile)
        assert np.array_equal(got_r, np.rot90(before, k=-1)), f"rotate wrong for mask {mask}"


def test_a_quarter_turn_of_an_unflagged_tile_is_flip_d_and_flip_h():
    """What ``gid``'s own docstring says a clockwise turn is."""
    brush = np.array([[gid.compose(3)]], gid.DTYPE)
    assert int(tools.rotate_brush_cw(brush)[0, 0]) == gid.compose(3, flip_d=True, flip_h=True)


def test_four_quarter_turns_and_double_mirrors_are_the_identity():
    cells = [
        gid.compose(9, flip_h=bool(m & 1), flip_v=bool(m & 2), flip_d=bool(m & 4))
        for m in range(8)
    ]
    brush = np.array(cells, gid.DTYPE).reshape(2, 4)
    assert np.array_equal(tools.flip_brush_h(tools.flip_brush_h(brush)), brush)
    assert np.array_equal(tools.flip_brush_v(tools.flip_brush_v(brush)), brush)
    turned = brush
    for _ in range(4):
        turned = tools.rotate_brush_cw(turned)
    assert np.array_equal(turned, brush)


def test_a_brush_transform_moves_the_arrangement_as_well_as_the_tiles():
    brush = np.array([[1, 2], [3, 4]], gid.DTYPE)
    assert np.array_equal(gid.tile_ids(tools.flip_brush_h(brush)), [[2, 1], [4, 3]])
    assert np.array_equal(gid.tile_ids(tools.flip_brush_v(brush)), [[3, 4], [1, 2]])
    assert np.array_equal(gid.tile_ids(tools.rotate_brush_cw(brush)), [[3, 1], [4, 2]])


def test_a_non_square_brush_comes_back_transposed():
    brush = np.array([[1, 2, 3]], gid.DTYPE)
    turned = tools.rotate_brush_cw(brush)
    assert turned.shape == (3, 1)
    assert np.array_equal(gid.tile_ids(turned), [[1], [2], [3]])


def test_an_empty_cell_never_gains_a_flag():
    """gid 0 with a flag set is not an empty cell -- it is a tile id nothing
    accounts for, and it would survive every round trip."""
    brush = np.array([[0, gid.compose(4)], [0, 0]], gid.DTYPE)
    for transform in (tools.flip_brush_h, tools.flip_brush_v, tools.rotate_brush_cw):
        out = transform(brush)
        assert int((out[gid.tile_ids(out) == 0]).max(initial=0)) == 0


def test_a_brush_transform_leaves_its_input_alone():
    brush = np.array([[gid.compose(1, flip_d=True), 2]], gid.DTYPE)
    original = brush.copy()
    tools.flip_brush_h(brush)
    tools.flip_brush_v(brush)
    tools.rotate_brush_cw(brush)
    assert np.array_equal(brush, original)


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
