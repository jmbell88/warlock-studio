"""What each tool computes, from plain arrays.

Two things carry most of the weight. Clipping: a brush dragged off the edge is a
legitimate stroke whose visible part must land, and only a placement *entirely*
outside the map is nothing. And the flood fill matching on the **full encoded
value**, which is what makes a mirrored wall tile bound a fill of its unmirrored
twin -- the seam the user drew them to make.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.plotter import tools
from warlock.studio.tilegrid import gid


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


def test_random_mode_chooses_one_nonempty_tile_from_the_stamp():
    brush = np.array([[0, 5], [7, 9]], gid.DTYPE)
    rng = np.random.default_rng(4)
    seen = {
        int(tools.random_stamp(_layer(), 2, 1, brush, rng=rng)[2][0, 0])
        for _ in range(20)
    }
    assert seen <= {5, 7, 9}
    assert len(seen) > 1


def test_random_mode_with_no_nonempty_choice_is_nothing():
    assert tools.random_stamp(_layer(), 0, 0, np.zeros((2, 2), gid.DTYPE)) is None


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


# --- clipping a region --------------------------------------------------------


def test_a_region_is_cut_down_to_the_bounds():
    block = np.arange(1, 13, dtype=gid.DTYPE).reshape(3, 4)
    got = tools.clip_region((2, 1, block), (3, 2, 4, 2))
    assert got is not None
    x0, y0, cut = got
    assert (x0, y0) == (3, 2)
    assert np.array_equal(cut, [[6, 7]])


def test_a_region_wholly_inside_the_bounds_is_unchanged():
    block = np.ones((2, 2), gid.DTYPE)
    x0, y0, cut = tools.clip_region((1, 1, block), (0, 0, 9, 9))
    assert (x0, y0) == (1, 1) and np.array_equal(cut, block)


def test_a_region_wholly_outside_the_bounds_is_nothing():
    block = np.ones((2, 2), gid.DTYPE)
    assert tools.clip_region((10, 10, block), (0, 0, 3, 3)) is None


def test_a_clipped_region_is_its_own_array():
    """The tools return views of their inputs in places; a clipped region has
    to be safe to hand to ``write_region``, which keeps it."""
    block = np.ones((3, 3), gid.DTYPE)
    _x, _y, cut = tools.clip_region((0, 0, block), (0, 0, 1, 1))
    cut[0, 0] = 42
    assert int(block[0, 0]) == 1


# --- a bounded flood ----------------------------------------------------------


def test_a_bounded_fill_cannot_escape_its_bounds():
    layer = _layer(6, 6)  # all zeros, so the whole map is one connected run
    region = tools.flood_fill(layer, 1, 1, 7, bounds=(0, 0, 2, 2))
    got = _filled(region, 6, 6)
    assert int(got.sum()) == 9
    assert got[0:3, 0:3].all()


def test_a_bounded_fill_cannot_leave_and_come_back():
    """The reason bounds are applied to the *match* and not to the result.

    Two rooms on the top row, joined only by a corridor along the bottom one.
    Bounded to the top two rows, the corridor is unreachable and so is the far
    room -- where a fill clipped *afterwards* would have travelled through the
    corridor, lit up both rooms, and had the trip trimmed away invisibly.
    """
    layer = _layer(9, 3)
    layer[0] = [0, 0, 0, 1, 1, 1, 0, 0, 0]  # left room, wall, right room
    layer[1] = [0, 1, 1, 1, 1, 1, 1, 1, 0]  # a way down at each end
    layer[2] = 0  # the corridor joining them

    # The rooms really are connected: unbounded, the far one fills. Without this
    # the test below would pass with the bounds ignored entirely.
    everywhere = _filled(tools.flood_fill(layer, 0, 0, 7), 9, 3)
    assert everywhere[0, 6:9].all()

    got = _filled(tools.flood_fill(layer, 0, 0, 7, bounds=(0, 0, 8, 1)), 9, 3)
    assert got[0, 0:3].all(), "the seeded room fills"
    assert not got[0, 6:9].any(), "the far room is not reachable inside the bounds"
    assert not got[2, :].any(), "and the corridor is out of bounds entirely"


def test_a_fill_seeded_outside_its_bounds_does_nothing():
    layer = _layer(6, 6)
    assert tools.flood_fill(layer, 5, 5, 7, bounds=(0, 0, 1, 1)) is None


def test_bounds_are_themselves_clipped_to_the_map():
    layer = _layer(4, 4)
    region = tools.flood_fill(layer, 0, 0, 7, bounds=(-5, -5, 99, 99))
    assert int(_filled(region, 4, 4).sum()) == 16


def test_an_unbounded_fill_is_exactly_what_it_always_was():
    layer = _layer(5, 5)
    layer[2, :] = 1
    plain = tools.flood_fill(layer, 0, 0, 7)
    same = tools.flood_fill(layer, 0, 0, 7, bounds=None)
    assert np.array_equal(_filled(plain, 5, 5), _filled(same, 5, 5))


# --- ellipses -----------------------------------------------------------------


def _filled(region, width: int, height: int) -> np.ndarray:
    """A whole-layer boolean of what a region would write."""
    out = np.zeros((height, width), dtype=bool)
    x0, y0, block = region
    out[y0 : y0 + block.shape[0], x0 : x0 + block.shape[1]] = block != 0
    return out


def test_an_ellipse_fills_its_bounding_box_but_not_its_corners():
    layer = _layer(8, 8)
    region = tools.fill_ellipse(layer, 0, 0, 7, 7, 5)
    assert region is not None
    got = _filled(region, 8, 8)
    # The four corners of the box are outside the inscribed circle; the middle
    # of every edge is inside it.
    assert not got[0, 0] and not got[0, 7] and not got[7, 0] and not got[7, 7]
    assert got[0, 3] and got[7, 4] and got[3, 0] and got[4, 7]
    assert got[4, 4]


def test_an_ellipse_leaves_the_cells_it_does_not_cover_alone():
    """It must not rectangle over its own corners -- ``flood_fill``'s rule."""
    layer = _layer(6, 6)
    layer[0, 0] = 99
    region = tools.fill_ellipse(layer, 0, 0, 5, 5, 5)
    x0, y0, block = region
    assert (x0, y0) == (0, 0)
    assert int(block[0, 0]) == 99


def test_a_one_wide_ellipse_is_the_line_of_cells():
    """No special case: a one-cell-wide box puts every centre on the axis, so
    the x term vanishes and the test reduces to the column."""
    layer = _layer(6, 6)
    got = _filled(tools.fill_ellipse(layer, 2, 1, 2, 4, 5), 6, 6)
    assert [bool(got[y, 2]) for y in range(6)] == [False, True, True, True, True, False]
    assert not got[:, 3].any() and not got[:, 1].any()


def test_a_one_cell_ellipse_is_that_cell():
    layer = _layer(6, 6)
    got = _filled(tools.fill_ellipse(layer, 3, 3, 3, 3, 5), 6, 6)
    assert int(got.sum()) == 1 and got[3, 3]


def test_an_ellipse_keeps_its_shape_when_dragged_off_the_map():
    """Measured from the drawn box and clipped afterwards. Inscribing into what
    survived clipping would reshape the half still on screen."""
    layer = _layer(8, 8)
    whole = _filled(tools.fill_ellipse(layer, 0, 0, 7, 7, 5), 8, 8)
    # The same circle, with its left half dragged off the left edge.
    shifted = tools.fill_ellipse(layer, -4, 0, 3, 7, 5)
    assert shifted is not None
    got = _filled(shifted, 8, 8)
    # Column 0 of the shifted ellipse is column 4 of the whole one.
    assert list(got[:, 0]) == list(whole[:, 4])


def test_an_ellipse_entirely_off_the_map_is_nothing():
    layer = _layer(6, 6)
    assert tools.fill_ellipse(layer, -20, -20, -10, -10, 5) is None


def test_an_ellipse_takes_its_corners_in_any_order():
    layer = _layer(8, 8)
    forward = tools.fill_ellipse(layer, 1, 1, 6, 5, 5)
    backward = tools.fill_ellipse(layer, 6, 5, 1, 1, 5)
    assert np.array_equal(_filled(forward, 8, 8), _filled(backward, 8, 8))


# --- lines --------------------------------------------------------------------


def test_a_line_includes_both_ends_and_steps_one_cell_at_a_time():
    cells = tools.line(0, 0, 5, 2)
    assert cells[0] == (0, 0) and cells[-1] == (5, 2)
    for (ax, ay), (bx, by) in zip(cells[:-1], cells[1:], strict=True):
        assert max(abs(bx - ax), abs(by - ay)) == 1, "eight-connected: no gaps"


def test_a_line_is_the_same_cells_drawn_either_way():
    """Bresenham resolves ties toward the end it started from, so a user who
    drags one way, undoes, and drags back must not get a different line."""
    for a, b in (((0, 0), (5, 2)), ((1, 7), (9, 2)), ((3, 3), (-4, 8)), ((0, 0), (7, 7))):
        assert tools.line(*a, *b) == list(reversed(tools.line(*b, *a)))


def test_a_degenerate_line_is_the_one_cell():
    assert tools.line(4, 4, 4, 4) == [(4, 4)]


def test_steep_shallow_and_axis_aligned_lines_all_span_their_endpoints():
    steep = tools.line(0, 0, 1, 6)
    shallow = tools.line(0, 0, 6, 1)
    vertical = tools.line(2, 0, 2, 4)
    horizontal = tools.line(0, 2, 4, 2)
    assert len(steep) == 7 and len(shallow) == 7
    assert vertical == [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)]
    assert horizontal == [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]


def test_a_line_is_not_clipped_to_anything():
    """Every placement clips itself, which is what lets a line be drawn partly
    off the map exactly as a drag across the edge already is."""
    cells = tools.line(-2, -2, 2, 2)
    assert (-2, -2) in cells and (0, 0) in cells


def test_a_perfect_diagonal_is_pure_diagonal_steps():
    assert tools.line(0, 0, 4, 4) == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


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
