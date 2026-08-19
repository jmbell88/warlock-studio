"""The picking BVH: a complexity change that must not be a behaviour change.

Every test here compares the narrowed sweep against the full one on the same
ray, because the tree is only allowed to make picking *faster*: the candidate
set is a superset of every hit, so the answer -- including which triangle index
comes back and which of two equally distant ones wins -- has to be identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives
from warlock.studio.viewer import picking as pk


def _sphere(segments: int = 64, rings: int = 48):
    mesh = primitives.uv_sphere(segments=segments, rings=rings)
    tris, tri_face = bm.triangulate(mesh)
    return mesh, mesh.positions.astype("f8"), tris, tri_face


def _rays(count: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        origin = -direction * 5.0 + rng.normal(scale=0.4, size=3)
        yield origin, direction


def test_a_small_mesh_gets_no_tree() -> None:
    """Below the threshold the vectorised sweep is a handful of numpy calls,
    where a tree walk is a Python loop per node."""
    mesh = primitives.box()
    tris, _tri_face = bm.triangulate(mesh)
    assert len(tris) < pk.BVH_MIN_TRIS
    assert pk.build_bvh(mesh.positions.astype("f8"), tris) is None


def test_a_large_mesh_gets_one() -> None:
    _mesh, positions, tris, _tf = _sphere()
    bvh = pk.build_bvh(positions, tris)
    assert bvh is not None
    assert len(bvh.order) == len(tris)
    # Every triangle appears exactly once across the leaves.
    assert sorted(bvh.order.tolist()) == list(range(len(tris)))


def test_the_narrowed_sweep_agrees_with_the_full_one() -> None:
    _mesh, positions, tris, _tf = _sphere()
    bvh = pk.build_bvh(positions, tris)
    for origin, direction in _rays(300, seed=1):
        assert pk.ray_triangles(origin, direction, positions, tris, bvh) == (
            pk.ray_triangles(origin, direction, positions, tris)
        )


def test_it_agrees_on_rays_that_miss_entirely() -> None:
    _mesh, positions, tris, _tf = _sphere()
    bvh = pk.build_bvh(positions, tris)
    for offset in (10.0, -10.0):
        origin = np.array([offset, offset, -5.0])
        direction = np.array([0.0, 0.0, 1.0])
        assert pk.ray_triangles(origin, direction, positions, tris, bvh) is None
        assert pk.ray_triangles(origin, direction, positions, tris) is None


def test_it_agrees_on_a_ray_from_inside() -> None:
    """Two-sided picking: clicking a shape you are inside must select it."""
    _mesh, positions, tris, _tf = _sphere()
    bvh = pk.build_bvh(positions, tris)
    origin = np.zeros(3)
    for direction in np.eye(3):
        assert pk.ray_triangles(origin, direction, positions, tris, bvh) == (
            pk.ray_triangles(origin, direction, positions, tris)
        )


def test_the_returned_index_names_a_triangle_in_the_callers_list() -> None:
    """The narrowed sweep works over a subset, so the index has to be mapped
    back -- ``tri_face`` is indexed by it and a wrong index selects a face
    somewhere else on the mesh."""
    _mesh, positions, tris, tri_face = _sphere()
    bvh = pk.build_bvh(positions, tris)
    hits = 0
    for origin, direction in _rays(60, seed=4):
        hit = pk.ray_triangles(origin, direction, positions, tris, bvh)
        if hit is None:
            continue
        hits += 1
        _distance, index = hit
        assert 0 <= index < len(tris)
        assert 0 <= int(tri_face[index]) < len(tri_face)
    assert hits > 0


def test_a_degenerate_mesh_with_coincident_centroids_still_builds() -> None:
    """Every centroid in one place: the split cannot separate them, so the node
    stays a leaf rather than recursing forever."""
    count = pk.BVH_MIN_TRIS + 16
    positions = np.zeros((3, 3), dtype="f8")
    positions[1] = (1.0, 0.0, 0.0)
    positions[2] = (0.0, 1.0, 0.0)
    tris = np.tile(np.array([[0, 1, 2]], dtype="i8"), (count, 1))
    bvh = pk.build_bvh(positions, tris)
    assert bvh is not None
    assert sorted(bvh.order.tolist()) == list(range(count))


def test_the_cache_is_stamped_on_the_mesh_object() -> None:
    """A Mesh is immutable, so the same object is the same geometry -- where
    keying on an array's id would key on an allocation numpy may recycle."""
    mesh, positions, tris, _tf = _sphere()
    first = pk.cached_bvh(mesh, positions, tris)
    assert first is not None
    assert pk.cached_bvh(mesh, positions, tris) is first

    other = primitives.uv_sphere(segments=64, rings=48)
    other_tris, _ = bm.triangulate(other)
    second = pk.cached_bvh(other, other.positions.astype("f8"), other_tris)
    assert second is not first, "a different mesh object gets its own tree"


def test_a_cached_small_mesh_records_the_none() -> None:
    """``None`` is a real answer and must be memoised as one, or every pick on
    a small mesh re-runs the size check and the triangulation gather."""
    mesh = primitives.box()
    tris, _tf = bm.triangulate(mesh)
    positions = mesh.positions.astype("f8")
    assert pk.cached_bvh(mesh, positions, tris) is None
    assert mesh in pk._BVH_CACHE


def test_ray_object_agrees_with_and_without_the_tree() -> None:
    mesh, positions, tris, _tf = _sphere()
    bvh = pk.build_bvh(positions, tris)
    matrix = np.eye(4)
    matrix[:3, 3] = (0.5, -0.25, 0.0)
    for origin, direction in _rays(80, seed=6):
        assert pk.ray_object(origin, direction, matrix, positions, tris, bvh=bvh) == (
            pk.ray_object(origin, direction, matrix, positions, tris)
        )


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_it_agrees_on_a_box_stack(seed: int) -> None:
    """Several separated boxes: the tree's split actually separates them, which
    is the case a single-AABB prefilter cannot help with."""
    rng = np.random.default_rng(seed)
    verts: list[np.ndarray] = []
    tris: list[list[int]] = []
    cube = primitives.box()
    cube_tris, _ = bm.triangulate(cube)
    for _ in range(60):
        base = len(verts)
        offset = rng.uniform(-8.0, 8.0, size=3)
        verts.extend(cube.positions.astype("f8") + offset)
        tris.extend((cube_tris + base).tolist())
    positions = np.asarray(verts, dtype="f8")
    triangles = np.asarray(tris, dtype="i8")
    bvh = pk.build_bvh(positions, triangles)
    assert bvh is not None
    for origin, direction in _rays(60, seed=seed):
        assert pk.ray_triangles(origin, direction, positions, triangles, bvh) == (
            pk.ray_triangles(origin, direction, positions, triangles)
        )
