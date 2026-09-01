"""Dissolve: merging across an element, and refusing when the outline forks."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_dissolve as dis
from warlock.studio.clay import primitives as prim
from warlock.studio.clay import topo

from .topo_asserts import assert_closed, assert_consistently_oriented


def _grid(nx: int = 3, nz: int = 3) -> bm.Mesh:
    """An ``nx`` by ``nz`` grid of quads in the XZ plane."""
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
        [v(i, j), v(i, j + 1), v(i + 1, j + 1), v(i + 1, j)] for i in range(nx) for j in range(nz)
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


# --- dissolve_edges ---------------------------------------------------------


def test_dissolving_an_edge_merges_the_two_faces_across_it() -> None:
    m = _grid(2, 1)  # two quads side by side
    shared = np.array([[1, 3]])  # x = 1 column
    a = adj.adjacency(m)
    shared = a.edge_verts[a.edge_uses == 2]
    out, sel = dis.dissolve_edges(m, el.ElementSel(edges=shared))
    bm.validate(out)
    assert bm.face_count(out) == 1
    assert np.diff(out.starts)[0] == 6, "a hexagon: the two ends stay as corners"
    assert sel.faces.tolist() == [0]
    assert_consistently_oriented(out)


def test_a_dissolved_quad_pair_keeps_its_outline_and_drops_nothing_else() -> None:
    m = _grid(2, 1)
    a = adj.adjacency(m)
    out, _ = dis.dissolve_edges(m, el.ElementSel(edges=a.edge_verts[a.edge_uses == 2]))
    assert len(out.positions) == 6, "every vertex is still on the outline"
    lo, hi = bm.bounds(out)
    assert lo.tolist() == bm.bounds(m)[0].tolist()
    assert hi.tolist() == bm.bounds(m)[1].tolist()


def test_dissolving_two_box_edges_in_a_row_merges_three_faces() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    # Two edges of the -Y face, each shared with a different side.
    quad = m.loops[m.starts[0] : m.starts[1]]
    edges = np.array([[quad[0], quad[1]], [quad[1], quad[2]]])
    out, sel = dis.dissolve_edges(m, el.ElementSel(edges=edges))
    bm.validate(out)
    assert bm.face_count(out) == 4, "three faces became one"
    assert len(sel.faces) == 1
    assert_closed(out)
    assert_consistently_oriented(out)
    assert len(a.edge_verts) == 12


def test_dissolve_edges_refuses_a_boundary_edge() -> None:
    m = prim.plane()
    with pytest.raises(el.OpError, match="on a boundary"):
        dis.dissolve_edges(m, el.ElementSel(edges=[[0, 1]]))


def test_dissolve_edges_refuses_a_non_manifold_edge() -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype="i4"),
        starts=np.array([0, 3, 6, 9], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    with pytest.raises(el.OpError, match="non-manifold"):
        dis.dissolve_edges(m, el.ElementSel(edges=[[0, 1]]))


def test_dissolve_edges_refuses_an_empty_or_foreign_selection() -> None:
    with pytest.raises(el.OpError, match="Select an edge"):
        dis.dissolve_edges(prim.box(), el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        dis.dissolve_edges(prim.box(), el.ElementSel(edges=[[0, 6]]))


# --- dissolve_faces ---------------------------------------------------------


def test_dissolving_a_block_of_faces_gives_one_face_along_its_outline() -> None:
    m = _grid(3, 3)
    out, sel = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 3, 4]))
    bm.validate(out)
    assert bm.face_count(out) == 6, "nine faces less four, plus one"
    assert np.diff(out.starts)[-1] == 8, "the 2x2 block's outline is eight corners"
    assert len(sel.faces) == 1
    assert_consistently_oriented(out)


def test_dissolving_two_separate_blocks_gives_two_faces() -> None:
    m = _grid(3, 3)
    out, sel = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 7, 8]))
    bm.validate(out)
    assert len(sel.faces) == 2
    assert bm.face_count(out) == 7


def test_dissolve_refuses_a_selection_that_rings_a_face_it_leaves_out() -> None:
    m = _grid(3, 3)
    ring = [f for f in range(9) if f != 4]
    with pytest.raises(el.OpError, match="surrounds a face it does not include"):
        dis.dissolve_faces(m, el.ElementSel(faces=ring))


def test_dissolve_refuses_a_bowtie_outline() -> None:
    # A 4x4 grid with two diagonally-touching faces left out: the two holes
    # meet at one vertex, so the region's outline passes through it twice and
    # the merged n-gon would fold onto itself there.
    m = _grid(4, 4)
    region = [f for f in range(16) if f not in (5, 10)]
    with pytest.raises(el.OpError, match="appears twice on the outline"):
        dis.dissolve_faces(m, el.ElementSel(faces=region))


def test_dissolving_a_whole_closed_surface_is_refused() -> None:
    with pytest.raises(el.OpError, match="closed surface"):
        dis.dissolve_faces(prim.box(), el.ElementSel(faces=range(6)))


def test_dissolve_faces_refuses_an_empty_selection() -> None:
    with pytest.raises(el.OpError, match="Select the faces"):
        dis.dissolve_faces(prim.box(), el.empty())


def test_a_lone_selected_face_has_nothing_to_dissolve() -> None:
    with pytest.raises(el.OpError, match="Nothing there to dissolve"):
        dis.dissolve_faces(_grid(3, 3), el.ElementSel(faces=[0]))


# --- dissolve_verts ---------------------------------------------------------


def test_dissolving_an_interior_vertex_merges_its_fan() -> None:
    m = _grid(2, 2)
    centre = int(np.flatnonzero((m.positions == [1.0, 0.0, 1.0]).all(axis=1))[0])
    out, sel = dis.dissolve_verts(m, el.ElementSel(verts=[centre]))
    bm.validate(out)
    assert bm.face_count(out) == 1
    assert np.diff(out.starts)[0] == 8
    assert len(out.positions) == 8, "the dissolved vertex is gone"
    assert len(sel.faces) == 1
    assert_consistently_oriented(out)


def test_dissolving_a_box_corner_merges_its_three_faces() -> None:
    m = prim.box()
    out, sel = dis.dissolve_verts(m, el.ElementSel(verts=[0]))
    bm.validate(out)
    assert bm.face_count(out) == 4
    assert len(out.positions) == 7
    assert len(sel.faces) == 1
    assert_closed(out)
    assert_consistently_oriented(out)


def test_dissolve_verts_refuses_a_boundary_vertex() -> None:
    with pytest.raises(el.OpError, match="on a boundary"):
        dis.dissolve_verts(_grid(2, 2), el.ElementSel(verts=[0]))


def test_dissolve_verts_refuses_a_non_manifold_vertex() -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, -1]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype="i4"),
        starts=np.array([0, 3, 6, 9], dtype="i4"),
        material=np.zeros(3, dtype="i4"),
        smooth=np.zeros(3, dtype=bool),
    )
    with pytest.raises(el.OpError, match="boundary|non-manifold"):
        dis.dissolve_verts(m, el.ElementSel(verts=[0]))


def test_dissolve_verts_refuses_an_empty_or_foreign_selection() -> None:
    with pytest.raises(el.OpError, match="Select a vertex"):
        dis.dissolve_verts(prim.box(), el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        dis.dissolve_verts(prim.box(), el.ElementSel(verts=[99]))


# --- shared behaviour -------------------------------------------------------


def test_a_dissolve_keeps_each_border_corners_own_uv() -> None:
    m = _uvd(_grid(2, 1))
    a = adj.adjacency(m)
    out, _ = dis.dissolve_edges(m, el.ElementSel(edges=a.edge_verts[a.edge_uses == 2]))
    assert out.uv is not None and len(out.uv) == len(out.loops)
    assert all(row.tolist() in m.uv.tolist() for row in out.uv)


def test_a_dissolve_may_produce_a_concave_face_that_still_triangulates_inside() -> None:
    # An L of three quads: dissolving them leaves a genuinely concave octagon.
    m = _grid(2, 2)
    out, _ = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 2]))
    bm.validate(out)
    normals = bm._face_normals(out)
    from warlock.studio.clay import earclip as ec

    assert ec.concave_faces(out.positions, out.loops, out.starts, normals).any()
    tris, tri_face = bm.triangulate(out)
    merged = tris[tri_face == bm.face_count(out) - 1]
    area = float(
        np.linalg.norm(
            np.cross(
                out.positions[merged[:, 1]] - out.positions[merged[:, 0]],
                out.positions[merged[:, 2]] - out.positions[merged[:, 0]],
            ),
            axis=1,
        ).sum()
        * 0.5
    )
    assert area == pytest.approx(3.0, rel=1e-5), "three unit quads, no overshoot"


def test_the_merged_face_inherits_material_and_smoothing() -> None:
    m = _grid(2, 1)
    marked = bm.Mesh(
        positions=m.positions,
        loops=m.loops,
        starts=m.starts,
        material=np.array([7, 7], dtype="i4"),
        smooth=np.array([True, True]),
    )
    a = adj.adjacency(marked)
    out, sel = dis.dissolve_edges(marked, el.ElementSel(edges=a.edge_verts[a.edge_uses == 2]))
    assert out.material[sel.faces[0]] == 7
    assert bool(out.smooth[sel.faces[0]])


# --- the growth ceiling ------------------------------------------------------


def test_a_dissolve_whose_outline_is_too_big_is_refused(monkeypatch) -> None:
    """``ops_subdiv`` refuses past ``MAX_SUBDIVIDED_FACES``; this is the other
    unbounded growth an edit can ask for.

    The n-gon goes to ``earclip``, whose ear search is quadratic in the corner
    count and runs in Python on the frame thread -- so the refusal has to come
    before the merge, not after it. Driven with the ceiling lowered rather than
    with a twenty-thousand-corner mesh, which would cost more to build than the
    thing it is testing.
    """
    m = _grid(3, 3)
    monkeypatch.setattr(dis, "MAX_DISSOLVED_RING", 4)
    with pytest.raises(el.OpError, match="corners, past the"):
        dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 3, 4]))


def test_the_ceiling_is_far_above_any_ordinary_dissolve() -> None:
    """It must not be felt: the 2x2 block above is an eight-corner outline."""
    assert dis.MAX_DISSOLVED_RING >= 20_000
    m = _grid(3, 3)
    out, sel = dis.dissolve_faces(m, el.ElementSel(faces=[0, 1, 3, 4]))
    bm.validate(out)
    assert len(sel.faces) == 1
