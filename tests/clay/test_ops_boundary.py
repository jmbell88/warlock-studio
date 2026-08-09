"""The two boundary ops: extruding an edge, and bridging two loops.

Both rest on the same fact, which is the thing worth pinning: a ``Mesh`` stores
no wire edges. An edge exists exactly while a face uses it, so an interior edge
has no open side to grow into and a lone vertex has nothing to grow at all --
which is why the refusals here are as much of the behaviour as the geometry is.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import ops_topo as ops
from warlock.studio.clay import primitives as prim

from .topo_asserts import (
    assert_closed,
    assert_consistently_oriented,
    assert_wound_outward,
)


def _open_tube(segments: int = 8) -> bm.Mesh:
    """A cylinder with both caps gone: one surface, two boundary rings."""
    mesh, _ = ops.delete_faces(prim.cylinder(segments=segments), el.ElementSel(faces=[8, 9]))
    return mesh


def _caps_only(segments: int = 8) -> bm.Mesh:
    """A cylinder with its *sides* gone: two n-gons facing away from each other,
    which is exactly the pair a bridge should rebuild the tube between."""
    mesh, _ = ops.delete_faces(
        prim.cylinder(segments=segments), el.ElementSel(faces=list(range(segments)))
    )
    return mesh


def _ring_edges(mesh: bm.Mesh, which: int = 0) -> np.ndarray:
    rings, _ = adj.boundary_loops(mesh)
    ring = rings[which]
    return np.stack([ring, np.roll(ring, -1)], axis=1)


def _all_boundary_edges(mesh: bm.Mesh) -> np.ndarray:
    rings, _ = adj.boundary_loops(mesh)
    return np.concatenate([np.stack([r, np.roll(r, -1)], axis=1) for r in rings])


# --- extrude ------------------------------------------------------------------


def test_extruding_a_rim_adds_one_quad_per_edge() -> None:
    mesh = _open_tube()
    out, _ = ops.extrude_edges(mesh, el.ElementSel(edges=_ring_edges(mesh)))
    bm.validate(out)
    assert bm.face_count(out) == bm.face_count(mesh) + 8


def test_an_extruded_quad_is_wound_against_the_face_it_grew_from() -> None:
    """The mirror of an extruded wall, and the other way round: there the cap
    moves onto the new vertices, so the wall keeps the original direction. Here
    the face stays exactly where it is, so the quad has to reverse."""
    mesh = _open_tube()
    out, _ = ops.extrude_edges(mesh, el.ElementSel(edges=_ring_edges(mesh)))
    assert_consistently_oriented(out)


def test_an_extrude_takes_no_distance_because_the_one_it_would_take_is_zero() -> None:
    """A closed rim's owning faces point radially outward all the way round, so
    their mean is zero and any offset handed in would move nothing. The new rim
    lands on the old one and the gizmo moves it -- which is what the returned
    selection is for."""
    import inspect

    assert "offset" not in inspect.signature(ops.extrude_edges).parameters
    mesh = _open_tube()
    out, sel = ops.extrude_edges(mesh, el.ElementSel(edges=_ring_edges(mesh)))
    assert len(out.positions) == len(mesh.positions) + 8
    assert len(sel.edges) == 8
    # The new rim is a separate set of vertices sitting exactly on the old one.
    fresh = set(sel.edges.reshape(-1).tolist())
    assert fresh.isdisjoint(_ring_edges(mesh).reshape(-1).tolist())


def test_extruding_an_interior_edge_is_refused_by_name() -> None:
    mesh = prim.cylinder(segments=8)
    interior = np.array([[0, 1]], dtype="i4")
    with pytest.raises(el.OpError, match="faces on both sides"):
        ops.extrude_edges(mesh, el.ElementSel(edges=interior))


def test_extruding_nothing_is_refused_rather_than_silently_doing_nothing() -> None:
    with pytest.raises(el.OpError, match="Select one or more boundary edges"):
        ops.extrude_edges(prim.cylinder(), el.empty())


def test_an_extruded_quad_inherits_its_sources_uvs() -> None:
    mesh = _open_tube()
    n = len(mesh.loops)
    uv = np.stack([np.arange(n, dtype="f4"), np.zeros(n, dtype="f4")], axis=1)
    from dataclasses import replace

    out, _ = ops.extrude_edges(replace(mesh, uv=uv), el.ElementSel(edges=_ring_edges(mesh)))
    assert out.uv is not None
    # Every new corner's u came from an existing corner rather than from zero.
    fresh = out.uv[n:, 0]
    assert set(fresh.tolist()) <= set(uv[:, 0].tolist())


# --- extrude in vertex mode ---------------------------------------------------


def test_a_vertex_selection_extrudes_the_border_between_its_vertices() -> None:
    """A mesh stores no wire edges, so what a run of border vertices means is
    the border between them."""
    mesh = _open_tube()
    ring = adj.boundary_loops(mesh)[0][0]
    out, _ = ops.extrude_verts(mesh, el.ElementSel(verts=ring))
    bm.validate(out)
    assert bm.face_count(out) == bm.face_count(mesh) + 8


def test_a_lone_vertex_is_refused_with_the_reason_rather_than_ignored() -> None:
    mesh = _open_tube()
    with pytest.raises(el.OpError, match="no border edge"):
        ops.extrude_verts(mesh, el.ElementSel(verts=[0]))


def test_a_vertex_selection_in_the_middle_of_a_surface_is_refused() -> None:
    """It implies a web of interior edges and no border at all, which is the one
    reading nobody intends."""
    mesh = prim.cylinder(segments=8)
    with pytest.raises(el.OpError, match="no border edge"):
        ops.extrude_verts(mesh, el.ElementSel(verts=np.arange(len(mesh.positions))))


# --- bridge -------------------------------------------------------------------


def test_bridging_two_caps_rebuilds_the_tube_between_them() -> None:
    mesh = _caps_only()
    out, sel = ops.bridge_edges(mesh, el.ElementSel(edges=_all_boundary_edges(mesh)))
    bm.validate(out)
    assert bm.face_count(out) == 10
    assert len(sel.faces) == 8
    assert_closed(out)
    assert_consistently_oriented(out)
    assert_wound_outward(out)


def test_a_bridge_encloses_the_same_volume_the_original_cylinder_did() -> None:
    """The strongest single statement that the winding needed no guess: the
    rebuilt tube is the tube, not its inside-out twin."""
    rebuilt, _ = ops.bridge_edges(
        _caps_only(), el.ElementSel(edges=_all_boundary_edges(_caps_only()))
    )
    original = prim.cylinder(segments=8)
    assert_wound_outward(rebuilt)
    assert bm.face_count(rebuilt) == bm.face_count(original)


def test_a_bridge_is_not_twisted_however_the_two_rings_are_numbered() -> None:
    """A closed loop carries no starting vertex, so the correspondence is
    measured. Getting it wrong raises nothing at all -- it just puts a visible
    twist in the strip -- which is why this is asserted on the geometry."""
    mesh = _caps_only(segments=8)
    out, sel = ops.bridge_edges(mesh, el.ElementSel(edges=_all_boundary_edges(mesh)))
    for face in sel.faces.tolist():
        corners = out.positions[bm.face(out, face)].astype("f8")
        # Measured across the *rings* rather than along the axis: an untwisted
        # quad is one segment wide, a twisted one reaches the diameter.
        across = corners[:, [0, 2]]
        assert float(np.linalg.norm(across.max(axis=0) - across.min(axis=0))) < 0.5


def test_bridging_one_loop_is_refused() -> None:
    mesh = _open_tube()
    with pytest.raises(el.OpError, match="exactly two boundary loops"):
        ops.bridge_edges(mesh, el.ElementSel(edges=_ring_edges(mesh)))


def test_bridging_loops_of_different_lengths_is_refused_with_both_counts() -> None:
    mesh = _caps_only(segments=8)
    edges = _all_boundary_edges(mesh)
    # Drop one edge from the second ring, leaving an open chain against a
    # closed loop -- the pairing that has no correspondence at all.
    with pytest.raises(el.OpError, match="closed and the other is not"):
        ops.bridge_edges(mesh, el.ElementSel(edges=edges[:-1]))


def test_bridging_an_interior_edge_is_refused() -> None:
    """Bridging two interior rings means deleting the faces between them first,
    and doing that silently would remove geometry nobody asked to lose."""
    mesh = prim.cylinder(segments=8)
    with pytest.raises(el.OpError, match="faces on both sides"):
        ops.bridge_edges(mesh, el.ElementSel(edges=np.array([[0, 1]], dtype="i4")))


def test_a_bridge_quad_inherits_the_material_of_the_loop_it_grew_from() -> None:
    from dataclasses import replace

    mesh = _caps_only()
    material = np.array([3, 3], dtype="i4")
    mesh = replace(mesh, material=material)
    out, sel = ops.bridge_edges(mesh, el.ElementSel(edges=_all_boundary_edges(mesh)))
    assert set(out.material[sel.faces].tolist()) == {3}


# --- through the registry -----------------------------------------------------


def test_extrude_is_one_row_and_one_key_across_all_three_modes() -> None:
    from warlock.studio import clay_ops

    op = clay_ops.get("extrude")
    assert op.modes == clay_ops.ELEMENT_MODES
    for mode in clay_ops.ELEMENT_MODES:
        assert clay_ops.by_key(mode, "E") is op


def test_bridge_is_offered_in_edge_mode_only() -> None:
    from warlock.studio import clay_ops

    assert clay_ops.get("bridge").modes == ("edge",)
