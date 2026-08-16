"""The derived half-edge tables, and the weak cache that holds them."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as prim


def _plane_grid() -> bm.Mesh:
    """Two quads sharing one interior edge -- the smallest open manifold."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype="f4",
    )
    return bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 4, 3, 1, 2, 5, 4], dtype="i4"),
        starts=np.array([0, 4, 8], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )


def test_corner_walks_wrap_within_their_own_face() -> None:
    a = adj.adjacency(_plane_grid())
    # Face 0 is corners 0..3, face 1 is 4..7; the last corner of each wraps to
    # its own first, never into the next face.
    assert list(a.next_corner) == [1, 2, 3, 0, 5, 6, 7, 4]
    assert list(a.prev_corner) == [3, 0, 1, 2, 7, 4, 5, 6]
    assert list(a.corner_face) == [0, 0, 0, 0, 1, 1, 1, 1]


def test_edge_uses_separate_boundary_from_interior() -> None:
    a = adj.adjacency(_plane_grid())
    shared = a.edge_ids(np.array([[1, 4]]))[0]
    assert a.edge_uses[shared] == 2
    # Seven edges: six on the boundary, one interior.
    assert a.n_edges == 7
    assert sorted(a.edge_uses.tolist()) == [1, 1, 1, 1, 1, 1, 2]


def test_a_closed_box_has_every_edge_used_twice_and_every_corner_twinned() -> None:
    a = adj.adjacency(prim.box())
    assert a.n_edges == 12
    assert set(a.edge_uses.tolist()) == {2}
    assert (a.twin >= 0).all()
    # A twin is symmetric and never itself.
    assert (a.twin[a.twin] == np.arange(len(a.twin))).all()


def test_a_boundary_corner_has_no_twin() -> None:
    a = adj.adjacency(prim.plane())
    assert list(a.twin) == [-1, -1, -1, -1]
    assert a.flipped_pairs.shape == (0, 2)


def test_a_flipped_neighbour_is_reported_and_denied_a_twin() -> None:
    # Two quads sharing edge (1, 4), the second wound the same way round that
    # edge as the first -- one of them is inside out.
    grid = _plane_grid()
    flipped = bm.Mesh(
        positions=grid.positions,
        loops=np.array([0, 1, 4, 3, 4, 5, 2, 1], dtype="i4"),
        starts=grid.starts,
        material=grid.material,
        smooth=grid.smooth,
    )
    bm.validate(flipped)
    a = adj.adjacency(flipped)
    assert a.flipped_pairs.shape == (1, 2)
    c1, c2 = a.flipped_pairs[0]
    assert a.corner_edge[c1] == a.corner_edge[c2]
    assert a.twin[c1] == -1 and a.twin[c2] == -1


def test_a_non_manifold_edge_gets_three_uses_and_no_twin() -> None:
    # Three triangles hinged on the edge (0, 1).
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype="i4"),
        starts=np.array([0, 3, 6, 9], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    a = adj.adjacency(m)
    hinge = a.edge_ids(np.array([[0, 1]]))[0]
    assert a.edge_uses[hinge] == 3
    for c in np.flatnonzero(a.corner_edge == hinge):
        assert a.twin[c] == -1


def test_vertex_corners_are_grouped_and_complete() -> None:
    m = prim.cylinder(segments=6)
    a = adj.adjacency(m)
    for v in range(len(m.positions)):
        corners = a.vertex_corners(v)
        assert (m.loops[corners] == v).all()
    assert len(a.vc_corners) == len(m.loops)
    assert a.vc_starts[0] == 0 and a.vc_starts[-1] == len(m.loops)


def test_edge_ids_maps_pairs_either_way_round_and_misses_cleanly() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    e = a.edge_verts[3]
    assert a.edge_ids(np.array([e]))[0] == 3
    assert a.edge_ids(np.array([e[::-1]]))[0] == 3
    # A pair that is a diagonal of the box, not an edge of it.
    assert a.edge_ids(np.array([[0, 6]]))[0] == -1
    assert a.edge_ids(np.zeros((0, 2), dtype="i4")).shape == (0,)


def test_edge_verts_agrees_with_the_meshs_own_edge_list() -> None:
    for m in (prim.box(), prim.cone(segments=5), prim.torus(segments=6, sides=5)):
        assert np.array_equal(adj.adjacency(m).edge_verts, bm.edges(m))


def test_every_table_is_read_only() -> None:
    a = adj.adjacency(prim.box())
    for name in ("next_corner", "corner_edge", "twin", "edge_verts", "vc_corners"):
        with pytest.raises(ValueError):
            getattr(a, name)[0] = 0


def test_an_empty_mesh_builds_empty_tables() -> None:
    m = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    a = adj.adjacency(m)
    assert a.n_edges == 0
    assert len(a.next_corner) == 0
    assert a.edge_ids(np.array([[0, 1]]))[0] == -1


def test_the_same_mesh_hands_back_the_same_adjacency_object() -> None:
    m = prim.box()
    assert adj.adjacency(m) is adj.adjacency(m)
    assert adj.cached_triangulation(m) is adj.cached_triangulation(m)
    # A different mesh object with identical arrays is a different entry: the
    # cache is keyed by identity, which is what the GPU cache keys on too.
    assert adj.adjacency(prim.box()) is not adj.adjacency(m)


def test_a_dead_mesh_takes_its_adjacency_with_it() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    ref = __import__("weakref").ref(a)
    del m, a
    gc.collect()
    assert ref() is None


def test_a_closed_mesh_has_no_boundary_at_all() -> None:
    rings, pinched = adj.boundary_loops(prim.box())
    assert rings == []
    assert len(pinched) == 0


def test_a_boundary_ring_is_wound_so_a_cap_built_from_it_is_consistent() -> None:
    # The plane is one quad; its border is one ring of four, and a face wound
    # in that order traverses every shared edge opposite to the quad.
    m = prim.plane()
    rings, pinched = adj.boundary_loops(m)
    assert len(rings) == 1 and len(pinched) == 0
    ring = rings[0]
    assert sorted(ring.tolist()) == [0, 1, 2, 3]

    capped = bm.Mesh(
        positions=m.positions,
        loops=np.concatenate([m.loops, ring]),
        starts=np.array([0, 4, 8], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    bm.validate(capped)
    a = adj.adjacency(capped)
    assert set(a.edge_uses.tolist()) == {2}
    assert a.flipped_pairs.shape == (0, 2), "the cap is wound against its neighbour"


def test_a_tube_has_two_rings() -> None:
    m = prim.cylinder(segments=6)
    open_tube = bm.Mesh(
        positions=m.positions,
        loops=m.loops[: 6 * 4],
        starts=m.starts[:7],
        material=m.material[:6],
        smooth=m.smooth[:6],
    )
    rings, pinched = adj.boundary_loops(open_tube)
    assert sorted(len(r) for r in rings) == [6, 6]
    assert len(pinched) == 0


def test_a_pinched_boundary_is_named() -> None:
    # Two triangles meeting at vertex 0 only: an hourglass, so two boundary
    # edges leave vertex 0 and the ring through it is ambiguous.
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [-1, 0, 0], [-1, 0, -1]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 3, 4], dtype="i4"),
        starts=np.array([0, 3, 6], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    _, pinched = adj.boundary_loops(m)
    assert pinched.tolist() == [0]


def test_a_clean_closed_mesh_reports_clean() -> None:
    assert adj.check_manifold(prim.box()).clean
    assert adj.check_manifold(prim.torus(segments=6, sides=5)).clean


def test_the_report_names_a_boundary_without_calling_it_broken() -> None:
    r = adj.check_manifold(prim.plane())
    assert len(r.boundary_edges) == 4
    assert len(r.nonmanifold_edges) == 0
    assert not r.clean, "strict reading: an open sheet is not closed"


def test_the_report_finds_duplicates_repeats_strays_and_flips() -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [9, 9, 9]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        # Two copies of the same quad (opposite windings, so also flipped), and
        # a triangle that uses vertex 1 twice. Vertex 4 is used by nothing.
        loops=np.array([0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 1], dtype="i4"),
        starts=np.array([0, 4, 8, 11], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    r = adj.check_manifold(m)
    assert r.duplicate_faces.tolist() == [0, 1]
    assert r.repeated_corner_faces.tolist() == [2]
    assert r.unused_verts.tolist() == [4]
    assert len(r.nonmanifold_edges) > 0
    assert not r.clean


def test_a_flipped_pair_shows_up_as_a_flipped_edge() -> None:
    grid = _plane_grid()
    flipped = bm.Mesh(
        positions=grid.positions,
        loops=np.array([0, 1, 4, 3, 4, 5, 2, 1], dtype="i4"),
        starts=grid.starts,
        material=grid.material,
        smooth=grid.smooth,
    )
    r = adj.check_manifold(flipped)
    assert r.flipped_edges.tolist() == [[1, 4]]


def test_cached_triangulation_matches_the_uncached_one_and_is_frozen() -> None:
    m = prim.uv_sphere(segments=5, rings=3)
    tris, tri_face = adj.cached_triangulation(m)
    want_tris, want_face = bm.triangulate(m)
    assert np.array_equal(tris, want_tris)
    assert np.array_equal(tri_face, want_face)
    with pytest.raises(ValueError):
        tris[0, 0] = 0


def test_cached_positions_f8_is_the_conversion_memoised_and_frozen() -> None:
    """Picking works in f8 and the mesh stores f4, so every ray cast converted
    the whole vertex array first -- once per object, on every mouse move, for a
    mesh that is frozen and cannot have changed."""
    m = prim.uv_sphere(segments=5, rings=3)
    got = adj.cached_positions_f8(m)
    assert got.dtype == np.dtype("f8")
    assert np.array_equal(got, m.positions.astype("f8"))
    assert adj.cached_positions_f8(m) is got
    with pytest.raises(ValueError):
        got[0, 0] = 0.0


def test_cached_positions_do_not_outlive_their_mesh() -> None:
    m = prim.box()
    adj.cached_positions_f8(m)
    assert len(adj._F8) >= 1
    del m
    gc.collect()
    assert len(adj._F8) == 0
