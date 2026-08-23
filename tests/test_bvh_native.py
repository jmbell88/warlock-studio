"""The BVH build in C: a tree that is *not* the numpy one, held to what is.

This is the second kernel in the tree whose bar is not output identity, and the
only one added since ``contours.c``. The licence is written out in
``docs/INVARIANTS.md``, in ``native/bvh.c`` and beside ``_build_bvh_native``,
and it comes down to one fact: ``np.argpartition`` is introselect, so its
permutation among *equal* keys is unspecified. A C median split therefore
separates coincident centroids differently and builds a different -- equally
valid -- tree, and asserting node-for-node equality would be asserting a
property the numpy path does not have either.

So the bar moves to what is guaranteed, and this file is where that is spelled
out:

* **The pick result.** A ray returns the same triangle through the tree as
  through the full linear sweep, whose tie-break (lowest triangle index) is
  already pinned so that the two agree. That is the property every caller
  actually depends on.
* **The structural invariants.** Every triangle in exactly one leaf, and a
  child's box inside its parent's -- true of any correct tree whatever its
  shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import native
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives
from warlock.studio.viewer import picking as pk

needs_dll = pytest.mark.skipif(
    not native.available(), reason="warlockc.dll not built"
)


def _sphere(segments: int = 96, rings: int = 72):
    """Big enough to be past ``BVH_MIN_TRIS`` several times over."""
    mesh = primitives.uv_sphere(segments=segments, rings=rings)
    tris, _ = bm.triangulate(mesh)
    return mesh.positions.astype("f8"), tris


def _rays(count: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        origin = -direction * 5.0 + rng.normal(scale=0.4, size=3)
        yield origin, direction


def _numpy_bvh(monkeypatch, positions, tris):
    """The reference tree, with the seam forced open.

    ``monkeypatch`` on ``native.available`` rather than on the wrapper, so what
    is under test is the *seam* -- the branch in ``build_bvh`` that chooses --
    and not a helper called directly by the test and by nothing else.
    """
    monkeypatch.setattr(native, "available", lambda: False)
    return pk.build_bvh(positions, tris)


@needs_dll
def test_the_fallback_is_genuinely_taken_when_the_seam_is_closed(monkeypatch):
    """Without this the monkeypatch could stop toggling anything -- a renamed
    predicate, a hoisted handle -- and every parity test below would be
    comparing the same code against itself and passing beautifully."""
    calls: list[int] = []
    real = native.bvh_build
    monkeypatch.setattr(native, "bvh_build", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    positions, tris = _sphere()
    pk.build_bvh(positions, tris)
    assert calls, "the kernel was never reached with the DLL present"

    monkeypatch.setattr(native, "available", lambda: False)
    calls.clear()
    pk.build_bvh(positions, tris)
    assert not calls, "the numpy path still called the kernel"


@needs_dll
def test_a_ray_picks_the_same_triangle_through_either_tree(monkeypatch):
    """The bar. Not the tree, the answer."""
    positions, tris = _sphere()
    native_tree = pk.build_bvh(positions, tris)
    assert native_tree is not None
    numpy_tree = _numpy_bvh(monkeypatch, positions, tris)
    assert numpy_tree is not None

    seen = 0
    for origin, direction in _rays(120):
        a = pk.ray_triangles(origin, direction, positions, tris, native_tree)
        b = pk.ray_triangles(origin, direction, positions, tris, numpy_tree)
        assert a == b
        seen += a is not None
    assert seen > 40, "a sweep that mostly misses proves very little"


@needs_dll
def test_the_native_tree_agrees_with_the_full_linear_sweep():
    """The stronger form of the same claim: the tree is only allowed to make
    picking *faster*, so its answer must be the unnarrowed one."""
    positions, tris = _sphere()
    tree = pk.build_bvh(positions, tris)
    assert tree is not None
    for origin, direction in _rays(120, seed=3):
        narrowed = pk.ray_triangles(origin, direction, positions, tris, tree)
        full = pk.ray_triangles(origin, direction, positions, tris, None)
        assert narrowed == full


@needs_dll
def test_every_triangle_sits_in_exactly_one_leaf():
    positions, tris = _sphere()
    tree = pk.build_bvh(positions, tris)
    assert tree is not None
    leaves = np.flatnonzero(tree.left < 0)
    held = np.concatenate(
        [tree.order[tree.first[n] : tree.first[n] + tree.count[n]] for n in leaves]
    )
    assert sorted(held.tolist()) == list(range(len(tris)))
    # An interior node holds none of its own, which is what makes the above a
    # partition rather than a double count.
    assert (tree.count[tree.left >= 0] == 0).all()


@needs_dll
def test_a_child_box_is_contained_in_its_parents():
    positions, tris = _sphere()
    tree = pk.build_bvh(positions, tris)
    assert tree is not None
    for node in range(len(tree.lo)):
        for child in (tree.left[node], tree.right[node]):
            if child < 0:
                continue
            assert (tree.lo[child] >= tree.lo[node]).all()
            assert (tree.hi[child] <= tree.hi[node]).all()


@needs_dll
def test_a_leaf_holds_no_more_than_the_leaf_size_unless_its_centroids_coincide():
    """The one legitimate oversized leaf: when every centroid in a span is the
    same point there is no split that separates them, and both implementations
    stop rather than recursing forever."""
    positions, tris = _sphere()
    tree = pk.build_bvh(positions, tris)
    assert tree is not None
    corners = positions[tris]
    centroid = (corners.min(axis=1) + corners.max(axis=1)) * 0.5
    for node in np.flatnonzero(tree.left < 0):
        span = tree.order[tree.first[node] : tree.first[node] + tree.count[node]]
        if len(span) <= pk.BVH_LEAF:
            continue
        points = centroid[span]
        assert (points.max(axis=0) == points.min(axis=0)).all()


@needs_dll
def test_a_mesh_of_coincident_triangles_does_not_hang_or_overrun():
    """Degenerate on purpose: every centroid is the same point, so the split
    that would separate them does not exist. The kernel has to make one leaf of
    the lot rather than recurse, which is also the case that would overrun the
    node budget if it did not."""
    positions = np.zeros((3, 3), dtype="f8")
    positions[1] = (1.0, 0.0, 0.0)
    positions[2] = (0.0, 1.0, 0.0)
    tris = np.tile(np.array([[0, 1, 2]], dtype="i8"), (4000, 1))
    tree = pk.build_bvh(positions, tris)
    assert tree is not None
    assert len(tree.lo) == 1
    assert int(tree.count[0]) == 4000


@needs_dll
def test_the_node_arrays_are_big_enough_for_a_worst_case_split():
    """The sizing argument in ``_build_bvh_native`` -- a span of nine or more
    leaves at least four a side -- is what makes the budget safe. If it were
    wrong the kernel would return -1 and picking would silently fall back to a
    tree it takes a second to build, so it is asserted rather than reasoned
    about.

    A ramp of collinear centroids is the shape that splits most evenly, which
    is the worst case for node count.
    """
    n = 5000
    positions = np.zeros((n + 2, 3), dtype="f8")
    positions[:, 0] = np.arange(n + 2, dtype="f8")
    tris = np.stack(
        [np.arange(n), np.arange(n) + 1, np.arange(n) + 2], axis=1
    ).astype("i8")
    tree = pk.build_bvh(positions, tris)
    assert tree is not None
    assert len(tree.lo) <= 2 * (n // 4 + 2) + 8
