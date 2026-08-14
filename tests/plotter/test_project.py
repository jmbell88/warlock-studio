"""Where a cell is, in both lattices, and how a click gets back.

The canvas and the flat renderer both take placement from here, so an error in
this module is an export that stops being a picture of the screen. The
round-trip tests are the load-bearing ones: the inverse is claimed to be *exact*
rather than a hit test, and that claim is only worth anything asserted.
"""

from __future__ import annotations

import pytest

from warlock.studio.plotter import project

SHAPES = [(7, 5, 32, 32), (7, 5, 32, 16), (4, 9, 24, 12), (1, 1, 16, 16)]


@pytest.mark.parametrize("projection", project.PROJECTIONS)
@pytest.mark.parametrize("shape", SHAPES)
def test_every_cell_round_trips_from_its_own_centre(projection, shape):
    width, height, tile_w, tile_h = shape
    lat = project.Lattice(projection, width, height, tile_w, tile_h)
    for column in range(width):
        for row in range(height):
            x, y = project.cell_origin(lat, column, row)
            back = project.cell_at(lat, x + tile_w / 2, y + tile_h / 2)
            assert back == (column, row)


@pytest.mark.parametrize("shape", SHAPES)
def test_an_orthogonal_cell_lands_where_it_always_did(shape):
    """The old arithmetic, kept honest: this projection's numbers are not
    allowed to move at all, or every existing map shifts."""
    width, height, tile_w, tile_h = shape
    lat = project.Lattice(project.ORTHOGONAL, width, height, tile_w, tile_h)
    for column in range(width):
        for row in range(height):
            assert project.cell_origin(lat, column, row) == (column * tile_w, row * tile_h)
    assert project.map_size(lat) == (
        width * tile_w,
        height * tile_h,
    )


def test_an_isometric_map_is_the_bounding_box_of_its_diamonds():
    width, height, tile_w, tile_h = 6, 4, 32, 16
    lat = project.Lattice(project.ISOMETRIC, width, height, tile_w, tile_h)
    size = project.map_size(lat)
    assert size == ((width + height) * tile_w // 2, (width + height) * tile_h // 2)
    corners = [
        project.cell_corner(lat, c, r)
        for c, r in ((0, 0), (width, 0), (width, height), (0, height))
    ]
    xs = [x for x, _y in corners]
    ys = [y for _x, y in corners]
    assert (min(xs), min(ys)) == (0.0, 0.0)
    assert (max(xs), max(ys)) == (float(size[0]), float(size[1]))


@pytest.mark.parametrize("projection", project.PROJECTIONS)
def test_the_four_cells_meeting_at_a_node_each_own_their_own_side(projection):
    """``floor`` breaks the tie the same way everywhere, which is the whole
    reason the inverse is arithmetic rather than a polygon test.

    The probe steps from the node *toward each cell's centre* rather than along
    the screen axes: in an isometric lattice the four cells around a node sit on
    the diagonals, so an axis-aligned nudge lands in two of them, not four.
    """
    lat = project.Lattice(projection, 5, 5, 32, 16)
    node = project.cell_corner(lat, 2, 2)
    owners = set()
    for column, row in ((1, 1), (2, 1), (1, 2), (2, 2)):
        centre = project.cell_corner(lat, column + 0.5, row + 0.5)
        probe = (
            node[0] + (centre[0] - node[0]) * 0.1,
            node[1] + (centre[1] - node[1]) * 0.1,
        )
        assert project.cell_at(lat, *probe) == (column, row)
        owners.add((column, row))
    assert len(owners) == 4


def test_visible_bounds_take_four_corners_not_two():
    """A screen rectangle is a *rhombus* in isometric cell space, so the min and
    max of one diagonal pair culls cells that are on screen."""
    lat = project.Lattice(project.ISOMETRIC, 8, 8, 32, 16)
    size = project.map_size(lat)
    c0, r0, c1, r1 = project.cell_bounds(lat, 0, 0, size[0], size[1])
    assert (c0, r0) == (0, 0)
    assert (c1, r1) == (7, 7)


def test_bounds_are_clamped_to_the_map():
    lat = project.Lattice(project.ORTHOGONAL, 4, 3, 16, 16)
    assert project.cell_bounds(lat, -500, -500, 5000, 5000) == (0, 0, 3, 2)


@pytest.mark.parametrize("projection", project.PROJECTIONS)
def test_draw_order_visits_every_cell_once(projection):
    cells = list(project.draw_order(project.Lattice(projection, 5, 4, 32, 16)))
    assert len(cells) == 20
    assert len(set(cells)) == 20


def test_isometric_draw_order_is_back_to_front():
    """Row-major is *not* monotone in isometric screen depth, which is the whole
    reason this function exists: cell (width-1, 0) sits in front of (0, 1)."""
    lat = project.Lattice(project.ISOMETRIC, 6, 4, 32, 16)
    depths = [c + r for c, r in project.draw_order(lat)]
    assert depths == sorted(depths)


def test_orthogonal_draw_order_is_row_major():
    lat = project.Lattice(project.ORTHOGONAL, 3, 2, 32, 16)
    assert list(project.draw_order(lat)) == [
        (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)
    ]


def test_an_unknown_projection_is_refused_by_name():
    with pytest.raises(ValueError, match="staggered"):
        project.check("staggered")


@pytest.mark.parametrize("projection", project.PROJECTIONS)
def test_an_object_round_trips_through_tileds_coordinate_space(projection):
    """Tiled measures an isometric object in tile-space units of the tile
    *height*, not in the projected plane. A position that made the trip
    untouched would reopen somewhere else and say nothing about it."""
    lat = project.Lattice(projection, 6, 5, 32, 16)
    for point in ((0.0, 0.0), (96.0, 40.0), (17.5, 3.25)):
        stored = project.object_from_pixels(lat, *point)
        back = project.object_to_pixels(lat, *stored)
        assert back == pytest.approx(point)


def test_an_orthogonal_object_is_not_converted_at_all():
    lat = project.Lattice(project.ORTHOGONAL, 6, 5, 32, 16)
    assert project.object_to_pixels(lat, 12.5, 7.0) == (12.5, 7.0)
    assert project.object_from_pixels(lat, 12.5, 7.0) == (12.5, 7.0)
