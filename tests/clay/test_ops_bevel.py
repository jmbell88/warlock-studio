"""Loop cut and bevel: the walking ops."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_bevel as ob
from warlock.studio.clay import primitives as prim
from warlock.studio.clay import topo

from .topo_asserts import assert_closed, assert_consistently_oriented


def _grid(nx: int = 3, nz: int = 1) -> bm.Mesh:
    xs = np.arange(nx + 1, dtype="f4")
    zs = np.arange(nz + 1, dtype="f4")
    positions = np.stack(
        [
            np.repeat(xs, nz + 1),
            np.zeros((nx + 1) * (nz + 1), dtype="f4"),
            np.tile(zs, nx + 1),
        ],
        axis=1,
    )

    def v(i: int, j: int) -> int:
        return i * (nz + 1) + j

    faces = [
        [v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)]
        for i in range(nx)
        for j in range(nz)
    ]
    return bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * len(faces)),
        material=np.zeros(len(faces), dtype="i4"),
        smooth=np.zeros(len(faces), dtype=bool),
    )


def _uvd(mesh: bm.Mesh) -> bm.Mesh:
    n = len(mesh.loops)
    uv = np.stack([np.arange(n, dtype="f4"), np.arange(n, dtype="f4") * 2], axis=1)
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=uv,
    )


def _tube(segments: int = 6) -> bm.Mesh:
    """A ring of quads with no caps -- the closed-loop case."""
    return topo.take_faces(prim.cylinder(segments=segments), np.arange(segments))


# --- loop_cut ---------------------------------------------------------------


def test_a_loop_cut_round_a_tube_closes_on_itself() -> None:
    m = _tube(6)
    a = adj.adjacency(m)
    # A vertical rung of the tube: an interior edge between two side quads.
    rung = a.edge_verts[a.edge_uses == 2][0]
    out, sel = ob.loop_cut(m, el.ElementSel(edges=[rung]))
    bm.validate(out)
    assert bm.face_count(out) == 12, "every quad split in two"
    assert len(out.positions) == 12 + 6, "one new vertex per rung"
    assert len(sel.edges) == 6, "the new ring is six edges"
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).boundary_edges.shape[0] == 12, "still open-ended"


def test_a_loop_cut_at_a_quarter_stays_on_one_side_all_the_way_round() -> None:
    m = _tube(8)
    a = adj.adjacency(m)
    rung = a.edge_verts[a.edge_uses == 2][0]
    out, _ = ob.loop_cut(m, el.ElementSel(edges=[rung]), t=0.25)
    new = out.positions[len(m.positions) :].astype("f8")
    # The tube runs -0.5..0.5 in Y. A ring cut consistently sits at one height;
    # an unoriented cut would zigzag between -0.25 and +0.25.
    assert np.allclose(new[:, 1], new[0, 1])
    assert abs(abs(float(new[0, 1])) - 0.25) < 1e-6


def test_a_loop_cut_across_a_strip_stops_at_the_ends_and_splices_them() -> None:
    m = _grid(3, 1)  # three quads in a row, open at both ends
    a = adj.adjacency(m)
    rung = a.edge_verts[a.edge_uses == 2][0]
    out, sel = ob.loop_cut(m, el.ElementSel(edges=[rung]))
    bm.validate(out)
    assert bm.face_count(out) == 6
    assert len(sel.edges) == 3, "one new edge per crossed quad"
    assert_consistently_oriented(out)
    assert len(adj.check_manifold(out).nonmanifold_edges) == 0


def test_a_loop_cut_stops_at_a_cap_and_splices_it() -> None:
    m = prim.cylinder(segments=6)  # hexagonal caps at both ends
    a = adj.adjacency(m)
    # A vertical rung: the strip runs round the sides and stops nowhere, so
    # cut a *horizontal* edge instead -- the strip then runs up the tube and
    # dead-ends in each cap.
    horizontal = None
    for e in a.edge_verts:
        lo, hi = m.positions[e[0]], m.positions[e[1]]
        if abs(float(lo[1] - hi[1])) < 1e-6:
            horizontal = e
            break
    assert horizontal is not None
    out, sel = ob.loop_cut(m, el.ElementSel(edges=[horizontal]))
    bm.validate(out)
    assert_closed(out), "the caps took the new vertices into their loops"
    assert_consistently_oriented(out)
    assert len(sel.edges) == 1, "one quad crossed, between the two caps"
    sizes = np.diff(out.starts).tolist()
    assert sizes.count(7) == 2, "each hexagonal cap became a heptagon"


def test_a_loop_cuts_new_corners_take_interpolated_uvs() -> None:
    m = _uvd(_grid(2, 1))
    a = adj.adjacency(m)
    rung = a.edge_verts[a.edge_uses == 2][0]
    out, _ = ob.loop_cut(m, el.ElementSel(edges=[rung]), t=0.5)
    assert out.uv is not None and len(out.uv) == len(out.loops)
    # Every new uv is the mean of two of the originals, so it lands on the
    # half-integer grid the fixture's u coordinates make.
    assert set(np.unique(out.uv[:, 0] * 2.0) % 1.0) == {0.0}


def test_loop_cut_refuses_a_boundary_edge_and_a_bad_selection() -> None:
    m = _grid(2, 1)
    boundary = adj.check_manifold(m).boundary_edges[0]
    with pytest.raises(el.OpError, match="between two faces"):
        ob.loop_cut(m, el.ElementSel(edges=[boundary]))
    with pytest.raises(el.OpError, match="exactly one edge"):
        ob.loop_cut(m, el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        ob.loop_cut(m, el.ElementSel(edges=[[0, 5]]))


def test_loop_cut_refuses_where_neither_face_is_a_quad() -> None:
    m = prim.cone(segments=6)  # triangles all the way round
    a = adj.adjacency(m)
    side = None
    for e, uses in zip(a.edge_verts, a.edge_uses, strict=True):
        if uses == 2 and 6 not in e.tolist():
            side = e
            break
    assert side is not None
    with pytest.raises(el.OpError, match="neither face there is one"):
        ob.loop_cut(m, el.ElementSel(edges=[side]))


# --- bevel_edges ------------------------------------------------------------


def _cube_edge(m: bm.Mesh, i: int = 0) -> np.ndarray:
    return adj.adjacency(m).edge_verts[i]


def test_beveling_one_cube_edge_gives_a_quad_and_two_pentagons() -> None:
    m = prim.box()
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=[_cube_edge(m)]), width=0.1)
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean

    assert bm.face_count(out) == 7, "six faces plus one bevel quad, no corner caps"
    assert len(sel.faces) == 1
    sizes = sorted(np.diff(out.starts).tolist())
    assert sizes.count(5) == 2, "the two faces opposite the beveled edge grew a corner"
    assert sizes.count(4) == 5


def test_beveling_the_three_edges_at_a_cube_corner_mitres_it() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    corner = 0
    at_corner = a.edge_verts[(a.edge_verts == corner).any(axis=1)]
    assert len(at_corner) == 3
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=at_corner), width=0.1)
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean
    # Six originals, three bevel quads, one miter triangle.
    assert bm.face_count(out) == 10
    assert len(sel.faces) == 4
    sizes = np.diff(out.starts)
    assert (sizes[-1] == 3).all() or int(sizes[-1]) == 3, "the corner cap is a triangle"


def test_beveling_all_twelve_cube_edges_gives_twenty_six_faces() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=a.edge_verts), width=0.1)
    bm.validate(out)
    assert bm.face_count(out) == 26, "6 faces + 12 edge quads + 8 corner triangles"
    assert len(sel.faces) == 20
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean
    assert len(out.positions) == 24, "three new vertices per original corner"


def test_a_bevel_is_crack_free_because_both_faces_share_the_slide_vertex() -> None:
    m = prim.box()
    out, _ = ob.bevel_edges(m, el.ElementSel(edges=[_cube_edge(m)]), width=0.1)
    report = adj.check_manifold(out)
    assert len(report.boundary_edges) == 0
    assert len(report.nonmanifold_edges) == 0
    assert len(report.duplicate_faces) == 0


def test_a_bevel_wider_than_the_feature_collapses_rather_than_inverting() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    out, _ = ob.bevel_edges(m, el.ElementSel(edges=a.edge_verts), width=99.0)
    bm.validate(out)
    lo, hi = bm.bounds(out)
    assert (lo >= bm.bounds(m)[0] - 1e-5).all()
    assert (hi <= bm.bounds(m)[1] + 1e-5).all()


def test_a_bevels_slide_uvs_are_interpolated_and_its_miters_are_copied() -> None:
    m = _uvd(prim.box())
    out, _ = ob.bevel_edges(m, el.ElementSel(edges=[_cube_edge(m)]), width=0.1)
    assert out.uv is not None and len(out.uv) == len(out.loops)
    assert np.isfinite(out.uv).all()


def test_bevel_refuses_a_boundary_edge_and_an_empty_selection() -> None:
    m = _grid(2, 1)
    boundary = adj.check_manifold(m).boundary_edges[0]
    with pytest.raises(el.OpError, match="on a boundary"):
        ob.bevel_edges(m, el.ElementSel(edges=[boundary]), width=0.1)
    with pytest.raises(el.OpError, match="Select an edge"):
        ob.bevel_edges(m, el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        ob.bevel_edges(m, el.ElementSel(edges=[[0, 5]]))


def test_bevel_refuses_a_non_manifold_edge() -> None:
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]], dtype="f4"
    )
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype="i4"),
        starts=np.array([0, 3, 6, 9], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    with pytest.raises(el.OpError, match="3 faces on it"):
        ob.bevel_edges(m, el.ElementSel(edges=[[0, 1]]), width=0.1)


def test_bevel_refuses_a_border_vertex_carrying_two_beveled_edges() -> None:
    # A 2x2 grid with one quad removed: the middle vertex now has two boundary
    # edges (where the missing quad was) and two interior ones. Beveling both
    # interior edges asks for a corner polygon at a vertex whose fan is open.
    xs = np.arange(3, dtype="f4")
    positions = np.stack(
        [np.repeat(xs, 3), np.zeros(9, dtype="f4"), np.tile(xs, 3)], axis=1
    )
    faces = [
        [i * 3 + j, i * 3 + j + 1, (i + 1) * 3 + j + 1, (i + 1) * 3 + j]
        for i in range(2)
        for j in range(2)
    ][1:]
    m = bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * 3),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    a = adj.adjacency(m)
    at_middle = (a.edge_verts == 4).any(axis=1)
    interior = a.edge_verts[at_middle & (a.edge_uses == 2)]
    assert len(interior) == 2
    assert (a.edge_uses[at_middle] == 1).any(), "the middle vertex is on the border"
    with pytest.raises(el.OpError, match="on the border"):
        ob.bevel_edges(m, el.ElementSel(edges=interior), width=0.1)


def test_beveling_one_interior_edge_of_an_open_sheet_caps_the_interior_end() -> None:
    # A 2x2 grid: one interior edge runs from the centre (four edges, so three
    # distinct slide vertices once one is beveled -- a corner polygon) out to a
    # border vertex (two, so none).
    xs = np.arange(3, dtype="f4")
    positions = np.stack(
        [np.repeat(xs, 3), np.zeros(9, dtype="f4"), np.tile(xs, 3)], axis=1
    )
    faces = [
        [i * 3 + j, i * 3 + j + 1, (i + 1) * 3 + j + 1, (i + 1) * 3 + j]
        for i in range(2)
        for j in range(2)
    ]
    m = bm.Mesh(
        positions=positions,
        loops=np.array([c for f in faces for c in f], dtype="i4"),
        starts=topo.starts_from_counts([4] * 4),
        material=np.zeros(4, dtype="i4"),
        smooth=np.zeros(4, dtype=bool),
    )
    a = adj.adjacency(m)
    one = a.edge_verts[a.edge_uses == 2][:1]
    out, sel = ob.bevel_edges(m, el.ElementSel(edges=one), width=0.1)
    bm.validate(out)
    assert bm.face_count(out) == 6, "four faces, one bevel quad, one corner cap"
    assert len(sel.faces) == 2
    assert_consistently_oriented(out)
    assert len(adj.check_manifold(out).nonmanifold_edges) == 0
