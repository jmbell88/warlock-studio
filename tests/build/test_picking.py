"""Ray-triangle picking: the one thing the analytic pickers could never do.

Everything here is arithmetic, so all of it is asserted without a window. The
tests are mostly about the ways a Moller-Trumbore implementation goes wrong
quietly rather than loudly: a hit behind the camera, a NaN out of a degenerate
triangle, a grazing ray in the triangle's own plane, and the wrong one of two
triangles that lie along the same ray.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.build import mesh as bm
from warlock.studio.build import primitives as bp
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer import picking

# A unit quad in the z = 0 plane, spanning x and y in [0, 1].
QUAD = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype="f8")
QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype="i4")

DOWN_Z = np.array([0.0, 0.0, -1.0])


def _hit(origin, direction, positions=QUAD, tris=QUAD_TRIS):
    return picking.ray_triangles(np.asarray(origin, dtype="f8"), direction, positions, tris)


# --- the basic solve ---------------------------------------------------------


def test_a_ray_down_minus_z_hits_the_quad_at_the_exact_distance() -> None:
    hit = _hit([0.5, 0.5, 4.0], DOWN_Z)
    assert hit is not None
    t, index = hit
    assert t == pytest.approx(4.0)
    assert index in (0, 1)


def test_a_ray_that_misses_returns_none() -> None:
    assert _hit([5.0, 5.0, 4.0], DOWN_Z) is None


def test_a_hit_behind_the_ray_origin_is_not_a_hit() -> None:
    """A click never selects what is behind the camera. The quad is at z = 0
    and the ray starts below it pointing further down."""
    assert _hit([0.5, 0.5, -4.0], DOWN_Z) is None


def test_the_nearer_of_two_triangles_along_one_ray_is_returned() -> None:
    positions = np.array(
        [
            [0, 0, 0], [2, 0, 0], [0, 2, 0],  # near, at z = 0
            [0, 0, -5], [2, 0, -5], [0, 2, -5],  # far, at z = -5
        ],
        dtype="f8",
    )
    tris = np.array([[3, 4, 5], [0, 1, 2]], dtype="i4")  # far one listed first

    hit = _hit([0.25, 0.25, 3.0], DOWN_Z, positions, tris)
    assert hit is not None
    t, index = hit
    assert t == pytest.approx(3.0)
    assert index == 1  # the nearer triangle's own index, not its position in a sort


def test_a_ray_from_inside_hits_the_back_of_a_face() -> None:
    """Two-sided on purpose: clicking a box you are standing inside must select
    it, and a blockout mesh is routinely inspected from within."""
    hit = _hit([0.5, 0.5, -4.0], np.array([0.0, 0.0, 1.0]))
    assert hit is not None
    assert hit[0] == pytest.approx(4.0)


# --- the quiet failures ------------------------------------------------------


def test_a_degenerate_triangle_is_skipped_rather_than_dividing_by_zero() -> None:
    positions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype="f8")
    tris = np.array([[0, 1, 2]], dtype="i4")
    assert _hit([0.0, 0.0, 4.0], DOWN_Z, positions, tris) is None


def test_a_grazing_ray_in_the_triangles_plane_returns_none_rather_than_a_nan() -> None:
    hit = _hit([-3.0, 0.5, 0.0], np.array([1.0, 0.0, 0.0]))
    assert hit is None


def test_a_degenerate_triangle_beside_a_real_one_does_not_poison_the_result() -> None:
    """The vectorised solve computes every triangle at once, so a NaN from one
    of them can propagate into the comparison that picks the nearest."""
    positions = np.vstack([QUAD, np.array([[9, 9, 9], [9, 9, 9], [9, 9, 9]], dtype="f8")])
    tris = np.array([[4, 5, 6], [0, 1, 2], [0, 2, 3]], dtype="i4")

    hit = _hit([0.5, 0.5, 4.0], DOWN_Z, positions, tris)
    assert hit is not None
    assert hit[0] == pytest.approx(4.0)


def test_an_empty_triangle_array_returns_none() -> None:
    assert (
        picking.ray_triangles(
            np.zeros(3), DOWN_Z, np.zeros((0, 3)), np.zeros((0, 3), dtype="i4")
        )
        is None
    )


# --- the AABB prefilter ------------------------------------------------------


def test_a_ray_through_a_box_hits_its_aabb() -> None:
    assert picking.ray_aabb(np.array([0.5, 0.5, 4.0]), DOWN_Z, np.zeros(3), np.ones(3))


def test_a_ray_beside_a_box_misses_its_aabb() -> None:
    assert not picking.ray_aabb(np.array([5.0, 0.5, 4.0]), DOWN_Z, np.zeros(3), np.ones(3))


def test_a_ray_running_exactly_along_a_face_of_the_aabb_still_counts() -> None:
    """A zero direction component makes the slab test divide by zero; the
    infinities that come out are the correct answer, and only become a NaN if
    the origin is exactly on the slab. A prefilter must never reject something
    the triangle test would have hit."""
    assert picking.ray_aabb(np.array([0.0, 0.5, 4.0]), DOWN_Z, np.zeros(3), np.ones(3))


def test_the_aabb_is_not_hit_when_the_box_is_behind_the_ray() -> None:
    assert not picking.ray_aabb(np.array([0.5, 0.5, -4.0]), DOWN_Z, np.zeros(3), np.ones(3))


def test_a_ray_starting_inside_the_box_hits_it() -> None:
    assert picking.ray_aabb(np.array([0.5, 0.5, 0.5]), DOWN_Z, np.zeros(3), np.ones(3))


# --- object space ------------------------------------------------------------


def _box_arrays():
    mesh = bp.box(size=(1.0, 1.0, 1.0))
    tris, _face = bm.triangulate(mesh)
    return mesh.positions.astype("f8"), tris


def test_a_rotated_and_scaled_box_is_hit_where_it_looks_like_it_is() -> None:
    """The whole point of doing this in object space: the ray goes through the
    inverse TRS once, rather than every triangle going through the TRS."""
    positions, tris = _box_arrays()
    matrix = m3.compose(
        m3.vec3(10.0, 0.0, 0.0),
        m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), np.pi / 4),
        m3.vec3(4.0, 4.0, 4.0),
    )
    # The box is a unit cube scaled by four and moved to x = 10, so its top
    # face sits at y = 2 and a ray straight down the y axis meets it there.
    hit = picking.ray_object(
        np.array([10.0, 9.0, 0.0]), np.array([0.0, -1.0, 0.0]), matrix, positions, tris
    )
    assert hit is not None
    t, _index = hit
    assert t == pytest.approx(7.0)


def test_a_ray_that_misses_a_placed_box_returns_none() -> None:
    positions, tris = _box_arrays()
    matrix = m3.compose(m3.vec3(10.0, 0.0, 0.0), m3.quat_identity(), m3.vec3(1.0, 1.0, 1.0))
    hit = picking.ray_object(
        np.array([0.0, 9.0, 0.0]), np.array([0.0, -1.0, 0.0]), matrix, positions, tris
    )
    assert hit is None


def test_the_returned_distance_is_in_world_units_under_a_non_uniform_scale() -> None:
    """``t`` has to be comparable between two objects, or the nearer-object
    test picks by whichever happens to be scaled up. The object-space direction
    is deliberately left un-normalised, which keeps the parameter the world
    ray's own."""
    positions, tris = _box_arrays()
    matrix = m3.compose(m3.vec3(), m3.quat_identity(), m3.vec3(0.1, 6.0, 0.1))
    hit = picking.ray_object(
        np.array([0.0, 20.0, 0.0]), np.array([0.0, -1.0, 0.0]), matrix, positions, tris
    )
    assert hit is not None
    assert hit[0] == pytest.approx(17.0)  # the top face is at y = 3


def test_an_object_behind_the_camera_is_not_picked() -> None:
    positions, tris = _box_arrays()
    matrix = m3.compose(m3.vec3(0.0, -50.0, 0.0), m3.quat_identity(), m3.vec3(1.0, 1.0, 1.0))
    hit = picking.ray_object(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), matrix, positions, tris
    )
    assert hit is None


def test_a_singular_transform_is_refused_rather_than_raising() -> None:
    """A zero scale is a value the properties panel will accept, and inverting
    it is a ``LinAlgError`` in the middle of a mouse move."""
    positions, tris = _box_arrays()
    matrix = m3.compose(m3.vec3(), m3.quat_identity(), m3.vec3(0.0, 0.0, 0.0))
    assert (
        picking.ray_object(
            np.array([0.0, 9.0, 0.0]), np.array([0.0, -1.0, 0.0]), matrix, positions, tris
        )
        is None
    )
