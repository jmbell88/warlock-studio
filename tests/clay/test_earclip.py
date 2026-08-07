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


# --- the constant factor ------------------------------------------------------


def _earclip_reference(pts: np.ndarray) -> np.ndarray:
    """The ear search as it stood before the boxing and ``rest`` were removed.

    Kept verbatim here rather than in the module: the change claimed to move
    speed only, and the only way to hold it to that is to run the old code and
    compare. A star polygon exercises every branch -- reflex corners, a
    rejected ear, the pointer walking back after a clip.
    """
    n = len(pts)
    if n < 3:
        return np.zeros((0, 3), dtype="i8")
    idx = list(range(n))
    if sum(ec._cross2(pts[0], pts[i], pts[i + 1]) for i in range(1, n - 1)) < 0.0:
        idx.reverse()

    tris: list[tuple[int, int, int]] = []
    p = 0
    stalled = 0
    while len(idx) > 3 and stalled <= len(idx):
        m = len(idx)
        a, b, c = idx[p % m], idx[(p + 1) % m], idx[(p + 2) % m]
        rest = [i for i in idx if i not in (a, b, c)]
        if ec._cross2(pts[a], pts[b], pts[c]) > 0.0 and not any(
            ec._inside(pts, a, b, c, i) for i in rest
        ):
            tris.append((a, b, c))
            del idx[(p + 1) % m]
            p = max(0, p - 1)
            stalled = 0
        else:
            p += 1
            stalled += 1

    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    else:
        tris.extend((idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1))
    return np.array(tris, dtype="i8").reshape(-1, 3)


def _star(points: int, inner: float, rng) -> np.ndarray:
    angles = np.arange(points * 2) * (np.pi / points)
    radius = np.where(np.arange(points * 2) % 2 == 0, 1.0, inner)
    jitter = 1.0 + (rng.random(points * 2) - 0.5) * 0.1
    return np.stack(
        [np.cos(angles) * radius * jitter, np.sin(angles) * radius * jitter], axis=1
    ).astype("f8")


@pytest.mark.parametrize("points", [3, 4, 5, 8, 13])
def test_the_faster_ear_search_triangulates_exactly_as_the_old_one_did(points):
    rng = np.random.default_rng(1000 + points)
    for inner in (0.35, 0.7):
        pts = _star(points, inner, rng)
        assert np.array_equal(ec._earclip(pts), _earclip_reference(pts))
        # And reversed, which is what sends it through the winding flip.
        assert np.array_equal(ec._earclip(pts[::-1]), _earclip_reference(pts[::-1]))


def test_a_degenerate_loop_still_falls_back_exactly_as_it_did():
    """All-collinear corners stall the search; the fan-the-remainder path is
    the contract every caller indexes by, so it has to be unchanged too."""
    pts = np.stack([np.arange(6, dtype="f8"), np.zeros(6)], axis=1)
    assert np.array_equal(ec._earclip(pts), _earclip_reference(pts))
