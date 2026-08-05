"""The scale gizmo's hit test and drag arithmetic, without a GL context.

``Gizmo`` only touches ``ctx`` inside ``draws()``, so everything about hitting
and dragging is assertable headlessly -- which is the whole reason the handles
are analytic rather than an id-buffer readback. ``place()`` is skipped for the
same reason: it needs a camera to work out a screen-constant size, and what the
arithmetic below actually depends on is the ``origin`` and ``scale`` it leaves
behind, which are set here directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.viewer import gizmo as gz


def _scale_gizmo(origin=(0.0, 0.0, 0.0), scale: float = 1.0) -> gz.ScaleGizmo:
    g = gz.ScaleGizmo(None, {})
    g.origin = np.array(origin, dtype="f8")
    g.scale = scale
    return g


def _ray_at(point, direction=(0.0, 0.0, -1.0)):
    """A ray pointing at ``point`` from a long way down +Z."""
    origin = np.array(point, dtype="f8") - np.array(direction, dtype="f8") * 20.0
    return origin, np.array(direction, dtype="f8")


# --- hit testing -------------------------------------------------------------


@pytest.mark.parametrize(("axis", "unit"), [("x", 0), ("y", 1), ("z", 2)])
def test_each_axis_handle_is_hit_along_its_own_axis(axis: str, unit: int) -> None:
    g = _scale_gizmo()
    point = np.zeros(3)
    point[unit] = 0.8
    # Look at the handle from a direction that is not its own axis, or the ray
    # runs down the handle and every point on it is equidistant.
    direction = np.zeros(3)
    direction[(unit + 1) % 3] = -1.0
    assert g.hit(*_ray_at(point, direction)) == axis


def test_the_centre_handle_wins_at_the_origin() -> None:
    """The three axis handles all start at the origin, so without testing the
    uniform handle first every click on it would be stolen by whichever axis
    the loop reached first."""
    g = _scale_gizmo()
    assert g.hit(*_ray_at((0.0, 0.0, 0.0))) == "xyz"


def test_a_ray_nowhere_near_the_gizmo_hits_nothing() -> None:
    g = _scale_gizmo()
    assert g.hit(*_ray_at((6.0, 6.0, 0.0))) is None


def test_a_hit_past_the_end_of_a_handle_is_not_a_hit() -> None:
    """Only the drawn part is grabbable. A handle that carried on forever would
    steal clicks from every object behind the gizmo."""
    g = _scale_gizmo()
    assert g.hit(*_ray_at((4.0, 0.0, 0.0), (0.0, -1.0, 0.0))) is None


def test_the_gizmo_moves_with_its_origin() -> None:
    g = _scale_gizmo(origin=(5.0, 5.0, 5.0))
    assert g.hit(*_ray_at((5.8, 5.0, 5.0), (0.0, -1.0, 0.0))) == "x"
    assert g.hit(*_ray_at((0.8, 0.0, 0.0), (0.0, -1.0, 0.0))) is None


# --- dragging ----------------------------------------------------------------


def test_an_axis_drag_reports_a_factor_on_that_axis_alone() -> None:
    g = _scale_gizmo()
    assert g.begin("x", *_ray_at((1.0, 0.0, 0.0), (0.0, -1.0, 0.0))) is True

    factors = g.update(*_ray_at((2.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    assert factors is not None
    assert np.allclose(factors, [2.0, 1.0, 1.0])


def test_a_factor_is_relative_to_where_the_drag_started() -> None:
    """Relative, so the caller multiplies it onto the scale it recorded at the
    start of the drag -- the same ``was``-shaped contract ``set_transform``
    takes, and the reason the gizmo needs to remember nothing about the object."""
    g = _scale_gizmo()
    g.begin("y", *_ray_at((0.0, 2.0, 0.0)))
    assert np.allclose(g.update(*_ray_at((0.0, 1.0, 0.0))), [1.0, 0.5, 1.0])
    assert np.allclose(g.update(*_ray_at((0.0, 4.0, 0.0))), [1.0, 2.0, 1.0])


def test_the_uniform_handle_scales_all_three_axes_together() -> None:
    g = _scale_gizmo()
    g.begin("xyz", *_ray_at((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    factors = g.update(*_ray_at((3.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    assert factors is not None
    assert np.allclose(factors, [3.0, 3.0, 3.0])


def test_dragging_through_the_origin_never_reports_a_negative_factor() -> None:
    """A negative scale is a mirror, and a mirror has to be baked into the mesh
    or glTF readers disagree about the winding -- which is exactly why
    ``build.ops.mirror`` exists. A gizmo must not be able to create one."""
    g = _scale_gizmo()
    g.begin("x", *_ray_at((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    factors = g.update(*_ray_at((-3.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    assert factors is not None
    assert (factors > 0.0).all()


def test_a_drag_that_began_at_the_origin_reports_nothing_rather_than_infinity() -> None:
    """There is no ratio to take from a start distance of zero, and the answer
    a naive divide gives is an infinity that reaches the document."""
    g = _scale_gizmo()
    g.begin("x", *_ray_at((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    assert g.update(*_ray_at((2.0, 0.0, 0.0), (0.0, -1.0, 0.0))) is None


def test_update_before_a_drag_begins_reports_nothing() -> None:
    assert _scale_gizmo().update(*_ray_at((1.0, 0.0, 0.0))) is None


def test_ending_a_drag_clears_it() -> None:
    g = _scale_gizmo()
    g.begin("x", *_ray_at((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
    assert g.dragging is True
    g.end_drag()
    assert g.dragging is False
    assert g.update(*_ray_at((2.0, 0.0, 0.0), (0.0, -1.0, 0.0))) is None


def test_the_handle_geometry_is_lines_and_starts_at_the_origin() -> None:
    """The overlay pass is unlit and depth-less, so everything in it is lines --
    a shaded box handle reads as a flat blob. Asserted because ``draws()`` is
    the one part of the gizmo a headless test cannot reach."""
    verts = gz.scale_handle("x")
    assert verts.dtype == np.dtype("f4")
    assert len(verts) % 2 == 0
    assert np.allclose(verts[0], [0.0, 0.0, 0.0])
    assert verts[:, 0].max() == pytest.approx(1.0)
