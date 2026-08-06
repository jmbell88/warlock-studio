"""The topology substrate and the two ops that prove it: delete and flip."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_topo as ops
from warlock.studio.clay import primitives as prim
from warlock.studio.clay import topo

from .topo_asserts import (
    assert_closed,
    assert_consistently_oriented,
    assert_wound_outward,
    directed_edge_counts,
)


def _uvd(mesh: bm.Mesh) -> bm.Mesh:
    """The same mesh with a distinct, recognisable uv per corner."""
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


# --- topo.py ----------------------------------------------------------------


def test_take_faces_keeps_order_materials_and_uvs() -> None:
    m = _uvd(prim.box())
    sub = topo.take_faces(m, [4, 1])
    bm.validate(sub)
    assert bm.face_count(sub) == 2
    assert sub.loops[:4].tolist() == m.loops[m.starts[4] : m.starts[5]].tolist()
    assert sub.uv[:4].tolist() == m.uv[m.starts[4] : m.starts[5]].tolist()
    assert len(sub.positions) == len(m.positions), "vertices are left alone"


def test_compact_vertices_renumbers_and_reports_the_map() -> None:
    m = topo.take_faces(prim.box(), [0])
    compact, old_to_new = topo.compact_vertices(m)
    bm.validate(compact)
    assert len(compact.positions) == 4
    assert (old_to_new[m.loops] == compact.loops).all()
    assert (old_to_new >= 0).sum() == 4
    # A mesh that uses everything is handed straight back.
    whole = prim.box()
    same, ident = topo.compact_vertices(whole)
    assert same is whole
    assert ident.tolist() == list(range(8))


def test_splice_corners_grows_the_right_face_and_keeps_uvs_in_step() -> None:
    m = _uvd(prim.plane())  # one quad, corners 0..3
    loops, starts, uv = topo.splice_corners(
        m.loops,
        m.starts,
        m.uv,
        after=np.array([1]),
        values=np.array([2]),
        counts=np.array([1]),
        value_uv=np.array([[9.0, 9.0]], dtype="f4"),
    )
    assert loops.tolist() == [0, 1, 2, 2, 3]
    assert starts.tolist() == [0, 5]
    assert uv[2].tolist() == [9.0, 9.0]
    assert uv[3].tolist() == m.uv[2].tolist()


def test_splice_corners_handles_unsorted_insertion_points() -> None:
    m = prim.plane()
    loops, starts, _ = topo.splice_corners(
        m.loops,
        m.starts,
        None,
        after=np.array([2, 0]),
        values=np.array([70, 71, 80]),  # two after corner 2, one after corner 0
        counts=np.array([2, 1]),
    )
    assert loops.tolist() == [0, 80, 1, 2, 70, 71, 3]
    assert starts.tolist() == [0, 7]


def test_splice_corners_with_nothing_to_insert_is_the_identity() -> None:
    m = prim.plane()
    loops, starts, uv = topo.splice_corners(
        m.loops, m.starts, None, after=np.zeros(0), values=np.zeros(0), counts=np.zeros(0)
    )
    assert loops is m.loops and starts is m.starts and uv is None


def test_region_boundary_corners_is_the_border_of_the_selection() -> None:
    m = prim.box()
    # One face: all four of its corners are on the region border.
    assert len(topo.region_boundary_corners(m, [0])) == 4
    # Two adjacent faces share one edge, so the border is six corners, not
    # eight -- the shared edge belongs to both and is interior to the region.
    faces = [0, 2]
    border = topo.region_boundary_corners(m, faces)
    assert len(border) == 6
    # The whole closed box has no border at all.
    assert len(topo.region_boundary_corners(m, range(6))) == 0
    assert len(topo.region_boundary_corners(m, [])) == 0


def test_rebuild_demands_a_uv_decision() -> None:
    m = prim.plane()
    with pytest.raises(TypeError):
        topo.rebuild(m.positions, m.loops, m.starts, m.material, m.smooth)  # type: ignore[call-arg]


# --- delete_faces -----------------------------------------------------------


def test_delete_faces_removes_the_faces_and_the_orphaned_vertices() -> None:
    m = _uvd(prim.box())
    out, sel = ops.delete_faces(m, el.ElementSel(faces=[0]))
    bm.validate(out)
    assert bm.face_count(out) == 5
    assert len(out.positions) == 8, "every box vertex is still used by another face"
    assert el.is_empty(sel), "what was selected no longer exists"

    # A face whose corners nothing else uses takes its vertices with it.
    lone = topo.take_faces(prim.box(), [0, 1])
    out2, _ = ops.delete_faces(lone, el.ElementSel(faces=[0]))
    assert len(out2.positions) == 4


def test_delete_faces_preserves_the_surviving_corners_uvs() -> None:
    m = _uvd(prim.box())
    out, _ = ops.delete_faces(m, el.ElementSel(faces=[0]))
    assert out.uv.tolist() == m.uv[4:].tolist()


def test_deleting_everything_leaves_a_valid_empty_mesh() -> None:
    m = _uvd(prim.box())
    out, sel = ops.delete_faces(m, el.ElementSel(faces=range(6)))
    bm.validate(out)
    assert bm.face_count(out) == 0
    assert len(out.positions) == 0
    assert out.uv is not None and out.uv.shape == (0, 2)
    assert el.is_empty(sel)


def test_delete_faces_refuses_an_empty_selection() -> None:
    with pytest.raises(el.OpError, match="at least one face"):
        ops.delete_faces(prim.box(), el.empty())


def test_deleting_a_box_face_leaves_a_consistent_open_shell() -> None:
    out, _ = ops.delete_faces(prim.box(), el.ElementSel(faces=[3]))
    assert_consistently_oriented(out)


# --- flip_normals -----------------------------------------------------------


def test_flipping_every_face_inverts_the_shell_but_keeps_it_consistent() -> None:
    m = prim.box()
    out, sel = ops.flip_normals(m, el.empty())
    bm.validate(out)
    assert_closed(out)
    assert_consistently_oriented(out)
    with pytest.raises(AssertionError):
        assert_wound_outward(out)
    assert sel is not None
    # Flipping twice is the identity.
    again, _ = ops.flip_normals(out, el.empty())
    assert again.loops.tolist() == m.loops.tolist()


def test_flipping_one_face_leaves_the_rest_alone_and_is_reported() -> None:
    m = prim.box()
    out, sel = ops.flip_normals(m, el.ElementSel(faces=[2]))
    bm.validate(out)
    assert out.loops[8:12].tolist() == m.loops[8:12].tolist()[::-1]
    assert out.loops[:8].tolist() == m.loops[:8].tolist()
    assert max(directed_edge_counts(out).values()) == 2, "deliberately inconsistent"
    assert sel.faces.tolist() == [2], "the face stays selected so it can be flipped back"


def test_flipping_carries_the_uvs_with_their_corners() -> None:
    m = _uvd(prim.box())
    out, _ = ops.flip_normals(m, el.ElementSel(faces=[0]))
    assert out.uv[:4].tolist() == m.uv[:4].tolist()[::-1]
    assert out.uv[4:].tolist() == m.uv[4:].tolist()


def test_flipping_a_mesh_with_no_faces_refuses() -> None:
    empty_mesh = bm.Mesh(
        positions=np.zeros((0, 3), dtype="f4"),
        loops=np.zeros(0, dtype="i4"),
        starts=np.zeros(1, dtype="i4"),
        material=np.zeros(0, dtype="i4"),
        smooth=np.zeros(0, dtype=bool),
    )
    with pytest.raises(el.OpError, match="no faces"):
        ops.flip_normals(empty_mesh, el.empty())


# --- extrude_faces ----------------------------------------------------------


def test_extruding_one_box_face_at_zero_adds_four_walls_and_moves_nothing() -> None:
    m = prim.box()
    out, sel = ops.extrude_faces(m, el.ElementSel(faces=[0]))
    bm.validate(out)
    assert bm.face_count(out) == 10, "six faces plus four walls"
    assert len(out.positions) == 12, "four new verts stacked on the old ones"
    assert sel.faces.tolist() == [0], "the cap keeps its index and stays selected"
    # Nothing moved: the new block is a copy of the old corners.
    lo, hi = bm.bounds(out)
    assert lo.tolist() == bm.bounds(m)[0].tolist()
    assert hi.tolist() == bm.bounds(m)[1].tolist()


def test_an_extruded_cap_is_wound_the_same_way_and_the_walls_agree() -> None:
    m = prim.box()
    out, _ = ops.extrude_faces(m, el.ElementSel(faces=[0]), offset=0.5)
    assert_consistently_oriented(out)
    assert_closed(out)
    assert_wound_outward(out)


def test_extruding_a_two_face_region_walls_only_its_outside() -> None:
    m = prim.box()
    out, sel = ops.extrude_faces(m, el.ElementSel(faces=[0, 2]), offset=0.25)
    bm.validate(out)
    # Six original faces plus six border walls -- not eight, which is what a
    # per-face extrude of two adjacent faces would produce.
    assert bm.face_count(out) == 12
    assert sel.faces.tolist() == [0, 2]
    assert_consistently_oriented(out)
    assert_closed(out)


def test_an_offset_extrude_moves_only_the_cap() -> None:
    m = prim.box()
    out, _ = ops.extrude_faces(m, el.ElementSel(faces=[1]), offset=0.5)
    normal = bm._face_normals(m)[1]
    normal = normal / np.linalg.norm(normal)
    cap = out.positions[out.loops[out.starts[1] : out.starts[2]]]
    original = m.positions[m.loops[m.starts[1] : m.starts[2]]]
    assert np.allclose(cap - original, normal * 0.5, atol=1e-5)


def test_an_extrudes_walls_inherit_their_source_corners_uvs() -> None:
    m = _uvd(prim.box())
    out, _ = ops.extrude_faces(m, el.ElementSel(faces=[0]))
    assert out.uv[:24].tolist() == m.uv.tolist(), "the caps keep their uvs verbatim"
    wall = out.uv[24:28]
    assert wall[0].tolist() == wall[3].tolist(), "a and new_a share a source corner"
    assert wall[1].tolist() == wall[2].tolist()


def test_extruding_an_open_sheet_leaves_a_closed_box() -> None:
    out, _ = ops.extrude_faces(prim.plane(), el.ElementSel(faces=[0]), offset=1.0)
    bm.validate(out)
    assert bm.face_count(out) == 5, "the plane plus four walls -- still open at the back"
    assert_consistently_oriented(out)


def test_extrude_refuses_an_empty_selection() -> None:
    with pytest.raises(el.OpError, match="at least one face"):
        ops.extrude_faces(prim.box(), el.empty())


# --- inset_faces ------------------------------------------------------------


def test_insetting_one_face_keeps_its_index_and_rings_it() -> None:
    m = prim.box()
    out, sel = ops.inset_faces(m, el.ElementSel(faces=[0]), thickness=0.1)
    bm.validate(out)
    assert bm.face_count(out) == 10, "six faces plus a four-quad rim"
    assert len(out.positions) == 12
    assert sel.faces.tolist() == [0]
    assert_consistently_oriented(out)
    assert_closed(out)
    assert_wound_outward(out)


def test_an_inset_face_is_strictly_smaller_and_still_flat() -> None:
    m = prim.box()
    out, _ = ops.inset_faces(m, el.ElementSel(faces=[0]), thickness=0.1)
    inner = out.positions[out.loops[out.starts[0] : out.starts[1]]].astype("f8")
    outer = m.positions[m.loops[m.starts[0] : m.starts[1]]].astype("f8")
    assert np.allclose(inner[:, 1], outer[:, 1]), "no depth means no movement in Y"
    span_in = inner[:, [0, 2]].max(axis=0) - inner[:, [0, 2]].min(axis=0)
    span_out = outer[:, [0, 2]].max(axis=0) - outer[:, [0, 2]].min(axis=0)
    assert (span_in < span_out).all()


def test_an_inset_thicker_than_the_face_collapses_rather_than_inverting() -> None:
    m = prim.box()
    out, _ = ops.inset_faces(m, el.ElementSel(faces=[0]), thickness=99.0)
    inner = out.positions[out.loops[out.starts[0] : out.starts[1]]].astype("f8")
    assert np.allclose(inner, inner[0]), "every corner landed on the centroid"


def test_inset_depth_pushes_the_inner_ring_along_the_face_normal() -> None:
    m = prim.box()
    out, _ = ops.inset_faces(m, el.ElementSel(faces=[1]), thickness=0.1, depth=-0.2)
    normal = bm._face_normals(m)[1]
    normal = normal / np.linalg.norm(normal)
    inner = out.positions[out.loops[out.starts[1] : out.starts[2]]].astype("f8")
    outer = m.positions[m.loops[m.starts[1] : m.starts[2]]].astype("f8")
    along = (inner - outer) @ normal
    assert np.allclose(along, -0.2, atol=1e-5)


def test_an_insets_uvs_are_lerped_within_the_face_that_made_them() -> None:
    m = _uvd(prim.box())
    out, _ = ops.inset_faces(m, el.ElementSel(faces=[0]), thickness=0.1)
    face_uv = m.uv[:4].astype("f8")
    inner = out.uv[:4].astype("f8")
    centroid = face_uv.mean(axis=0)
    # Each inner uv sits on the segment from its own corner to the face's uv
    # centroid -- never reading another face's corners, so a seam survives.
    for i in range(4):
        d = inner[i] - face_uv[i]
        to_centre = centroid - face_uv[i]
        assert abs(d[0] * to_centre[1] - d[1] * to_centre[0]) < 1e-3
        assert float(d @ to_centre) > 0.0


def test_insetting_two_adjacent_faces_gives_each_its_own_rim() -> None:
    m = prim.box()
    out, sel = ops.inset_faces(m, el.ElementSel(faces=[0, 2]), thickness=0.05)
    bm.validate(out)
    assert bm.face_count(out) == 14, "six faces plus two four-quad rims"
    assert sel.faces.tolist() == [0, 2]


def test_a_region_inset_shares_its_ring_between_the_faces() -> None:
    m = prim.box()
    out, sel = ops.inset_faces(m, el.ElementSel(faces=[0, 2]), thickness=0.05, region=True)
    bm.validate(out)
    # Extrude@0 of the two-face region: six faces plus six border walls, and no
    # doubled edge down the middle.
    assert bm.face_count(out) == 12
    assert sel.faces.tolist() == [0, 2]
    assert_consistently_oriented(out)


def test_a_region_inset_pulls_each_corner_toward_its_local_centroid() -> None:
    m = prim.plane(size=(2.0, 2.0))
    out, _ = ops.inset_faces(m, el.ElementSel(faces=[0]), thickness=0.25, region=True)
    inner = out.positions[out.loops[out.starts[0] : out.starts[1]]].astype("f8")
    outer = m.positions[m.loops[m.starts[0] : m.starts[1]]].astype("f8")
    # ``thickness`` is the distance travelled toward the centroid, not the
    # perpendicular offset from the edge -- the documented approximation, and
    # the same rule the per-face mode follows.
    assert np.allclose(np.linalg.norm(inner - outer, axis=1), 0.25, atol=1e-5)
    span = inner[:, [0, 2]].max(axis=0) - inner[:, [0, 2]].min(axis=0)
    assert (span < 2.0).all()


def test_inset_refuses_an_empty_selection() -> None:
    with pytest.raises(el.OpError, match="at least one face"):
        ops.inset_faces(prim.box(), el.empty())


# --- weld -------------------------------------------------------------------


def _split_box() -> bm.Mesh:
    """A box whose -Y face uses its own four coincident copies of the corners."""
    m = prim.box()
    extra = m.positions[m.loops[0:4]]
    loops = m.loops.astype("i8").copy()
    loops[0:4] = len(m.positions) + np.arange(4)
    return bm.Mesh(
        positions=np.concatenate([m.positions, extra]),
        loops=loops,
        starts=m.starts,
        material=m.material,
        smooth=m.smooth,
    )


def test_weld_closes_a_split_seam_and_leaves_the_shell_closed() -> None:
    m = _split_box()
    assert len(m.positions) == 12
    assert not adj.check_manifold(m).clean, "the split face has a boundary"
    out, sel = ops.weld(m, el.empty())
    bm.validate(out)
    assert len(out.positions) == 8
    assert bm.face_count(out) == 6
    assert adj.check_manifold(out).clean
    assert el.is_empty(sel)


def test_weld_puts_the_representative_at_the_cluster_centroid() -> None:
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype="f4",
    )
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 2, 3, 1, 2, 3], dtype="i4"),
        starts=np.array([0, 3, 6], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    out, _ = ops.weld(m, el.empty(), eps=0.1)
    assert len(out.positions) == 3
    assert out.positions[0].tolist() == pytest.approx([0.01, 0.0, 0.0])


def test_weld_only_touches_the_selection() -> None:
    m = _split_box()
    out, _ = ops.weld(m, el.ElementSel(verts=[0, 8]), eps=1e-3)
    assert len(out.positions) == 11, "one pair merged, the other three left split"


def test_weld_drops_the_faces_it_degenerates() -> None:
    # A quad whose corners merge in pairs is a line, and goes away.
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 0, 1e-6], [0.0, 0, 1e-6]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 3], dtype="i4"),
        starts=np.array([0, 4], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )
    out, _ = ops.weld(m, el.empty(), eps=1e-3)
    bm.validate(out)
    assert bm.face_count(out) == 0


def test_weld_preserves_each_surviving_corners_own_uv() -> None:
    m = _uvd(_split_box())
    out, _ = ops.weld(m, el.empty())
    assert out.uv.tolist() == m.uv.tolist(), "a seam in UV survives a weld in space"


def test_weld_refuses_a_nonsense_distance_or_a_single_vertex() -> None:
    m = prim.box()
    with pytest.raises(el.OpError, match="greater than zero"):
        ops.weld(m, el.empty(), eps=0.0)
    with pytest.raises(el.OpError, match="at least two"):
        ops.weld(m, el.ElementSel(verts=[0]))


def test_the_gridded_path_agrees_with_the_search_on_ordinary_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = _split_box()
    want, _ = ops.weld(m, el.empty())
    monkeypatch.setattr(ops, "WELD_SEARCH_LIMIT", 0)
    got, _ = ops.weld(m, el.empty())
    assert len(got.positions) == len(want.positions)


# --- collapse ---------------------------------------------------------------


def test_collapsing_an_edge_merges_its_ends_at_the_midpoint() -> None:
    m = prim.box()
    edge = [int(m.loops[0]), int(m.loops[1])]
    midpoint = m.positions[edge].astype("f8").mean(axis=0)
    out, sel = ops.collapse(m, el.ElementSel(edges=[edge]))
    bm.validate(out)
    assert len(out.positions) == 7
    assert any(np.allclose(p, midpoint, atol=1e-6) for p in out.positions)
    assert el.is_empty(sel)


def test_collapsing_a_chain_of_edges_gives_one_vertex_not_three() -> None:
    m = prim.cylinder(segments=6)
    ring = [[i, (i + 1) % 6] for i in range(3)]
    out, _ = ops.collapse(m, el.ElementSel(edges=ring))
    bm.validate(out)
    assert len(out.positions) == 12 - 3, "four vertices became one"


def test_collapsing_a_face_pulls_it_to_a_point() -> None:
    m = prim.box()
    out, _ = ops.collapse(m, el.ElementSel(faces=[0]))
    bm.validate(out)
    assert len(out.positions) == 5
    assert bm.face_count(out) == 5, "the collapsed face itself is gone"


def test_collapse_points_a_vertex_selection_at_weld() -> None:
    with pytest.raises(el.OpError, match="use Weld"):
        ops.collapse(prim.box(), el.ElementSel(verts=[0, 1]))
    with pytest.raises(el.OpError, match="Select an edge"):
        ops.collapse(prim.box(), el.empty())


# --- fill_hole --------------------------------------------------------------


def _open_tube(segments: int = 6) -> bm.Mesh:
    m = prim.cylinder(segments=segments)
    return topo.take_faces(m, np.arange(segments))


def test_filling_a_hole_caps_it_with_one_n_gon_wound_the_right_way() -> None:
    m = _open_tube()
    boundary = adj.check_manifold(m).boundary_edges
    out, sel = ops.fill_hole(m, el.ElementSel(edges=boundary[:1]))
    bm.validate(out)
    assert bm.face_count(out) == 7, "one cap, not a fan of six triangles"
    assert np.diff(out.starts)[-1] == 6
    assert sel.faces.tolist() == [6]
    assert_consistently_oriented(out)


def test_filling_both_ends_of_a_tube_closes_it() -> None:
    m = _open_tube()
    boundary = adj.check_manifold(m).boundary_edges
    out, sel = ops.fill_hole(m, el.ElementSel(edges=boundary))
    bm.validate(out)
    assert len(sel.faces) == 2
    assert_closed(out)
    assert_consistently_oriented(out)
    assert adj.check_manifold(out).clean


def test_fill_hole_refuses_an_interior_edge() -> None:
    m = prim.box()
    with pytest.raises(el.OpError, match="not on a boundary"):
        ops.fill_hole(m, el.ElementSel(edges=[[int(m.loops[0]), int(m.loops[1])]]))


def test_fill_hole_refuses_a_pinched_boundary() -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [-1, 0, 0], [-1, 0, -1]], dtype="f4")
    m = bm.Mesh(
        positions=positions,
        loops=np.array([0, 1, 2, 0, 3, 4], dtype="i4"),
        starts=np.array([0, 3, 6], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    with pytest.raises(el.OpError, match="two boundary edges"):
        ops.fill_hole(m, el.ElementSel(edges=[[0, 1]]))


def test_fill_hole_refuses_an_empty_selection_and_a_foreign_edge() -> None:
    m = _open_tube()
    with pytest.raises(el.OpError, match="Select an edge"):
        ops.fill_hole(m, el.empty())
    with pytest.raises(el.OpError, match="not part of this mesh"):
        ops.fill_hole(m, el.ElementSel(edges=[[0, 3]]))


def test_a_filled_caps_uvs_come_from_the_surface_around_it() -> None:
    m = _uvd(_open_tube())
    boundary = adj.check_manifold(m).boundary_edges
    out, _ = ops.fill_hole(m, el.ElementSel(edges=boundary[:1]))
    cap = out.uv[len(m.uv) :]
    assert len(cap) == 6
    assert all(row.tolist() in m.uv.tolist() for row in cap)
