"""The concavity screen, the ear search, and the promise that a fan is kept."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import earclip as ec
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as prim

# An L-shaped hexagon in the XZ plane, wound counter-clockwise seen from +Y, so
# its Newell normal is +Y. Corner 2 is the reflex one.
L_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
    ],
    dtype="f4",
)[::-1].copy()


def _one_face(positions: np.ndarray) -> bm.Mesh:
    n = len(positions)
    return bm.Mesh(
        positions=positions,
        loops=np.arange(n, dtype="i4"),
        starts=np.array([0, n], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )


def _tri_area_sum(positions: np.ndarray, tris: np.ndarray) -> float:
    a, b, c = (positions[tris[:, i]].astype("f8") for i in range(3))
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * 0.5)


def _polygon_area(positions: np.ndarray) -> float:
    p = positions.astype("f8")
    total = np.zeros(3)
    for i in range(len(p)):
        total += np.cross(p[i], p[(i + 1) % len(p)])
    return float(np.linalg.norm(total) * 0.5)


def test_convex_faces_are_never_suspect() -> None:
    for m in (
        prim.box(),
        prim.cylinder(segments=12),
        prim.cone(segments=7),
        prim.uv_sphere(segments=6, rings=4),
        prim.torus(segments=6, sides=5),
    ):
        mask = ec.concave_faces(m.positions, m.loops, m.starts, bm._face_normals(m))
        assert not mask.any()


def test_a_reflex_corner_is_caught() -> None:
    m = _one_face(L_POSITIONS)
    mask = ec.concave_faces(m.positions, m.loops, m.starts, bm._face_normals(m))
    assert mask.tolist() == [True]


def test_a_convex_mesh_triangulates_byte_identically_to_the_plain_fan() -> None:
    for m in (prim.box(), prim.uv_sphere(segments=5, rings=3), prim.torus()):
        want, want_face = bm._fan_corners(m)
        got, got_face = bm._corner_triangles(m)
        assert np.array_equal(got, want)
        assert np.array_equal(got_face, want_face)


def test_an_l_shape_is_triangulated_inside_itself() -> None:
    m = _one_face(L_POSITIONS)
    tris, tri_face = bm.triangulate(m)
    assert len(tris) == 4
    assert tri_face.tolist() == [0, 0, 0, 0]
    # Every corner used, no repeats within a triangle, and the areas add up to
    # the polygon's -- which a fan across the reflex corner would overshoot.
    assert set(tris.reshape(-1).tolist()) == set(range(6))
    assert _tri_area_sum(m.positions, tris) == pytest.approx(_polygon_area(m.positions), rel=1e-5)


def test_the_triangle_count_is_n_minus_2_even_on_a_degenerate_face() -> None:
    # All six corners collinear: no ear exists, so the search stalls and fans.
    line = np.stack(
        [np.arange(6.0), np.zeros(6), np.zeros(6)],
        axis=1,
    ).astype("f4")
    m = _one_face(line)
    tris, _ = bm.triangulate(m)
    assert len(tris) == 4


def test_a_repeated_corner_does_not_break_the_search() -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],  # duplicated
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ],
        dtype="f4",
    )[::-1].copy()
    m = _one_face(pts)
    tris, _ = bm.triangulate(m)
    assert len(tris) == 5


def test_render_arrays_and_triangulate_agree_on_a_concave_face() -> None:
    m = _one_face(L_POSITIONS)
    tris, _ = bm.triangulate(m)
    _, _, _, indices = bm.render_arrays(m)
    # Flat shading splits one vertex per corner in loop order, so a render
    # index is its corner index; mapping back through loops must give the same
    # triangles the picker will use.
    assert np.array_equal(m.loops[indices.reshape(-1, 3)], tris)


def test_a_face_pointing_down_a_negative_axis_still_clips() -> None:
    flipped = L_POSITIONS[::-1].copy()  # normal now -Y
    m = _one_face(flipped)
    assert bm._face_normals(m)[0][1] < 0
    tris, _ = bm.triangulate(m)
    assert _tri_area_sum(m.positions, tris) == pytest.approx(_polygon_area(m.positions), rel=1e-5)


def test_reversed_corner_perm_reverses_each_face_in_place() -> None:
    m = prim.cylinder(segments=5)
    perm = bm.reversed_corner_perm(m.starts)
    flipped = m.loops[perm]
    for i in range(bm.face_count(m)):
        lo, hi = m.starts[i], m.starts[i + 1]
        assert flipped[lo:hi].tolist() == m.loops[lo:hi].tolist()[::-1]
    assert bm.reversed_corner_perm(np.zeros(1, dtype="i4")).shape == (0,)
