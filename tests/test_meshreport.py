from __future__ import annotations

import trimesh

from warlock import meshreport


def _write(tmp_path, mesh, name="m.glb"):
    path = tmp_path / name
    trimesh.Scene(mesh).export(path)
    return path


def test_a_clean_box_is_ready(tmp_path):
    path = _write(tmp_path, trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    report = meshreport.build(path)
    assert report["status"] in ("ready", "review")
    assert report["watertight"] is True
    assert report["triangles"] == 12
    assert report["components"] == 1
    assert report["boundary_edges"] == 0


def test_an_open_surface_is_not_watertight_and_is_flagged(tmp_path):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.faces = box.faces[:-2]          # tear a hole
    box.remove_unreferenced_vertices()
    path = _write(tmp_path, box)
    report = meshreport.build(path)
    assert report["watertight"] is False
    assert report["boundary_edges"] > 0
    assert report["status"] == "review"
    assert any("watertight" in r for r in report["reasons"])


def _seam_split_cube():
    """A cube whose vertices have been duplicated along a UV seam.

    This is what xatlas hands back: to lay a chart flat it splits a vertex in
    two and gives each copy its own UV, leaving the *positions* identical. The
    surface is still closed -- nothing about it has a hole -- but every face on
    one side of the seam now references a different index than its neighbour,
    so an unwelded read of it counts boundary edges and calls it open.
    """
    import numpy as np
    import trimesh

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    vertices = np.asarray(box.vertices, dtype=np.float64)
    faces = np.asarray(box.faces, dtype=np.int64).copy()
    # A seam through one corner: the faces that touch vertex 0 are split into
    # two groups and the second group is given its own copy of that position.
    touching = np.flatnonzero((faces == 0).any(axis=1))
    twin = len(vertices)
    vertices = np.vstack([vertices, vertices[0]])
    for index in touching[len(touching) // 2 :]:
        faces[index][faces[index] == 0] = twin
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_a_seam_split_cube_is_open_raw_and_watertight_welded(tmp_path):
    """The whole point of the welded analysis copy.

    ``build`` loads with ``process=False`` because the UV and material checks
    need the unwelded mesh -- so the raw watertight figure on a textured trellis
    reconstruction was mostly counting seam splits, not holes.
    """
    path = _write(tmp_path, _seam_split_cube())
    report = meshreport.build(path)

    assert report["watertight"] is False
    assert report["boundary_edges"] > 0
    assert report["welded_watertight"] is True
    assert report["welded_boundary_edges"] == 0
    assert report["welded_components"] == 1
    assert not any("watertight" in r for r in report["reasons"])


def test_the_watertight_reason_names_the_welded_numbers(tmp_path):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.faces = box.faces[:-2]          # tear a real hole
    box.remove_unreferenced_vertices()
    report = meshreport.build(_write(tmp_path, box))

    reason = next(r for r in report["reasons"] if "watertight" in r)
    assert str(report["welded_boundary_edges"]) in reason
    assert "weld" in reason


def test_size_and_grounding_are_measured(tmp_path):
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box.apply_translation((0.0, 1.0, 0.0))   # glTF is Y-up: min Y == 0
    path = _write(tmp_path, box)
    report = meshreport.build(path, target_size_m=2.0)
    assert report["grounded"] is True
    assert abs(report["achieved_size_m"] - 2.0) < 1e-6


def test_an_unparseable_file_is_invalid(tmp_path):
    path = tmp_path / "broken.glb"
    path.write_bytes(b"not a glb")
    report = meshreport.build(path)
    assert report["status"] == "invalid"
    assert report["reasons"]


def test_component_count_sees_separate_shells_and_lone_faces(tmp_path):
    """The count is taken off the face-adjacency graph rather than mesh.split()
    -- building a full Trimesh per shell to compute one integer is a large
    transient on a 500k-triangle reconstruction. The graph omits faces with no
    neighbour, so a floating triangle is the case that pins the `nodes` arg."""
    import numpy as np
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    box = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
    box.apply_translation([2.0, 0.0, 0.0])
    lone = trimesh.Trimesh(
        vertices=np.array([[5.0, 0.0, 0.0], [5.5, 0.0, 0.0], [5.0, 0.5, 0.0]]),
        faces=np.array([[0, 1, 2]]),
    )
    path = tmp_path / "shells.glb"
    trimesh.util.concatenate([sphere, box, lone]).export(path)

    assert meshreport.build(path)["components"] == 3
