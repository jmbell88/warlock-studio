"""Go to coordinate: the arithmetic under a navigation that had no control.

A large map could only be reached by dragging or by aiming at a fraction of the
minimap, so "put me at 148, 96" -- the coordinate that arrives from a design
document, an engine log or a bug report -- was answerable only by dragging until
the status bar agreed.

Two pure pieces hold it up and both are here: *where is the middle of a cell*,
which is a different sum on each of the five projections, and *what pan puts a
point in the middle of a pane*, which is the sum the minimap was already doing
in a line of its own.
"""

from __future__ import annotations

import pytest

from warlock.studio import plotter_state
from warlock.studio.plotter import project
from warlock.studio.plotter.tilemap import MapDoc


@pytest.mark.parametrize(
    "projection",
    ["orthogonal", "isometric", "hexagonal", "staggered"],
)
def test_a_cell_centre_is_inside_that_cell_on_every_lattice(projection):
    """The property that makes a jump land where it says.

    ``corner + half a tile`` is only the middle on an orthogonal map: an
    isometric corner is the diamond's *top vertex*, and every other row of a
    staggered or hexagonal lattice is pushed sideways. The centroid of the
    cell's own outline is one answer that is right for all of them, and the way
    to check it is to ask the lattice which cell the point lands in.
    """
    doc = MapDoc(8, 8, 32, 16, projection=projection)
    for column, row in ((0, 0), (3, 2), (7, 7), (2, 5)):
        point = doc.cell_centre(column, row)
        assert doc.cell_at(*point) == (column, row), (projection, column, row)


def test_the_centre_of_an_orthogonal_cell_is_where_anyone_would_put_it():
    """One case written out, so the property test above cannot be satisfied by
    an answer that is merely self-consistent."""
    doc = MapDoc(4, 4, 32, 16)
    assert doc.cell_centre(0, 0) == (16.0, 8.0)
    assert doc.cell_centre(2, 3) == (80.0, 56.0)


def test_an_oblique_cell_centre_follows_the_skew():
    """The dialect projection is not exempt: its cells lean, and a centre that
    ignored the skew would be outside them the further down the map you go."""
    doc = MapDoc(6, 6, 32, 32)
    doc.projection = "oblique"
    doc.skew_x, doc.skew_y = 16, 0
    point = doc.cell_centre(1, 4)
    assert doc.cell_at(*point) == (1, 4)
    # Leaning right by half a tile per row means row 4 is two tiles across.
    assert point[0] == pytest.approx(project.cell_centre(doc._lattice(), 1, 4)[0])
    assert point[0] > doc.cell_centre(1, 0)[0]


def test_the_pan_puts_the_point_in_the_middle_of_the_pane():
    region = (800.0, 600.0)
    pan = plotter_state.centre_pan(region, (100.0, 50.0), 2.0)
    # Screen position of the point is pan + point * zoom, which must be the
    # middle of the region.
    assert (pan[0] + 100.0 * 2.0, pan[1] + 50.0 * 2.0) == (400.0, 300.0)


def test_the_pan_is_the_one_the_minimap_already_used():
    """Written once because two copies is how one of them comes to be half a
    tile out; pinned so a future edit to either cannot quietly fork them."""
    region, point, zoom = (640.0, 480.0), (77.0, 33.0), 0.75
    assert plotter_state.centre_pan(region, point, zoom) == (
        region[0] / 2.0 - point[0] * zoom,
        region[1] / 2.0 - point[1] * zoom,
    )
