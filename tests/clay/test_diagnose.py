"""The manifold report as rows a panel draws and a click selects."""

from __future__ import annotations

import numpy as np

from warlock.studio.clay import diagnose
from warlock.studio.clay import elements as el
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as prim


def _one_quad() -> bm.Mesh:
    """A single face: an open sheet, so four boundary edges in one loop."""
    return bm.Mesh(
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            dtype="f4",
        ),
        loops=np.array([0, 1, 2, 3], dtype="i4"),
        starts=np.array([0, 4], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )


def _kinds(mesh: bm.Mesh) -> list[str]:
    return [f.kind for f in diagnose.findings(mesh)]


def test_a_closed_primitive_has_nothing_to_report() -> None:
    assert diagnose.findings(prim.box()) == []
    assert diagnose.findings(prim.uv_sphere()) == []


def test_one_open_quad_is_one_hole_not_four() -> None:
    """The count is boundary *loops*; the selection is the boundary edges.

    Counting edges would call a single open face four holes, which is a wrong
    answer rather than a coarse one -- the reason the two halves come from
    different functions.
    """
    rows = diagnose.findings(_one_quad())
    assert [f.kind for f in rows] == ["hole"]
    assert rows[0].count == 1
    assert rows[0].label == "1 hole"
    assert len(rows[0].sel.edges) == 4


def test_two_separate_holes_are_counted_separately() -> None:
    """A cylinder with both caps removed: two independent boundary rings."""
    caps = prim.cylinder(segments=8)
    keep = np.array([i for i in range(bm.face_count(caps)) if len(bm.face(caps, i)) == 4])
    mesh, _ = _keep_faces(caps, keep)
    rows = diagnose.findings(mesh)
    holes = [f for f in rows if f.kind == "hole"]
    assert [f.count for f in holes] == [2]
    assert holes[0].label == "2 holes"


def _keep_faces(mesh: bm.Mesh, faces: np.ndarray) -> tuple[bm.Mesh, None]:
    from warlock.studio.clay import ops_topo

    drop = np.setdiff1d(np.arange(bm.face_count(mesh)), faces).astype("i4")
    out, _ = ops_topo.delete_faces(mesh, el.ElementSel(faces=drop))
    return out, None


def test_a_face_reusing_one_vertex_is_reported_in_face_mode() -> None:
    mesh = bm.Mesh(
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype="f4"
        ),
        loops=np.array([0, 1, 2, 1], dtype="i4"),
        starts=np.array([0, 4], dtype="i4"),
        material=np.zeros(1, dtype="i4"),
        smooth=np.zeros(1, dtype=bool),
    )
    rows = {f.kind: f for f in diagnose.findings(mesh)}
    assert "repeated" in rows
    row = rows["repeated"]
    assert row.mode == "face"
    assert row.label == "1 face with a repeated vertex"
    assert row.sel.faces.tolist() == [0]


def test_a_vertex_no_face_uses_is_reported_in_vertex_mode() -> None:
    quad = _one_quad()
    mesh = bm.Mesh(
        positions=np.vstack([quad.positions, np.array([[5.0, 5.0, 5.0]], dtype="f4")]),
        loops=quad.loops,
        starts=quad.starts,
        material=quad.material,
        smooth=quad.smooth,
    )
    row = next(f for f in diagnose.findings(mesh) if f.kind == "unused")
    assert row.mode == "vertex"
    assert row.label == "1 unused vertex"
    assert row.sel.verts.tolist() == [4]


def test_every_row_selects_in_a_mode_its_selection_is_non_empty_in() -> None:
    """A row whose mode and selection disagree draws an empty overlay.

    The pane sets the element mode from ``mode`` and the selection from ``sel``
    in one go, so the two have to name the same elements or the click looks
    like it did nothing.
    """
    mesh = bm.Mesh(
        positions=np.vstack([_one_quad().positions, np.array([[5.0, 5.0, 5.0]], dtype="f4")]),
        loops=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype="i4"),
        starts=np.array([0, 4, 8], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    rows = diagnose.findings(mesh)
    assert rows, "this mesh is a duplicate face plus a stray vertex"
    for row in rows:
        assert row.sel.count(row.mode) > 0, row.kind
        assert row.count > 0
        assert row.label.startswith(str(row.count))


def test_duplicate_faces_are_reported_in_face_mode() -> None:
    mesh = bm.Mesh(
        positions=_one_quad().positions,
        loops=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype="i4"),
        starts=np.array([0, 4, 8], dtype="i4"),
        material=np.zeros(2, dtype="i4"),
        smooth=np.zeros(2, dtype=bool),
    )
    row = next(f for f in diagnose.findings(mesh) if f.kind == "duplicate")
    assert row.mode == "face"
    assert row.label == "2 duplicate faces"
    assert row.sel.faces.tolist() == [0, 1]


def test_rows_for_does_not_measure_twice() -> None:
    """A caller with a report in hand gets the same rows without re-checking."""
    from warlock.studio.clay import adjacency as adj

    mesh = _one_quad()
    report = adj.check_manifold(mesh)
    assert [f.kind for f in diagnose.rows_for(mesh, report)] == _kinds(mesh)
