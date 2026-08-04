"""hole_fraction must find see-through gaps without inventing them on solid meshes.

The false-positive direction is the one that matters. This measurement exists to
decide whether a generation-flag change actually helped, so a harness that
reports holes on a watertight sphere would send the whole investigation the
wrong way -- which is exactly what happened with the checks it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from warlock.meshaudit import DEFAULT_VIEWS, hole_fraction


@pytest.fixture
def sphere_glb(tmp_path):
    """A watertight icosphere -- the no-false-positives control."""
    path = tmp_path / "sphere.glb"
    trimesh.creation.icosphere(subdivisions=4, radius=0.5).export(path)
    return path


def test_watertight_sphere_reports_no_holes(sphere_glb):
    result = hole_fraction(sphere_glb, resolution=512)
    assert result["worst"] == pytest.approx(0.0, abs=1e-4)
    assert all(view["blobs"] == 0 for view in result["views"])


def test_watertight_cube_reports_no_holes(tmp_path):
    path = tmp_path / "cube.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
    assert hole_fraction(path, resolution=512)["worst"] == pytest.approx(0.0, abs=1e-4)


def test_a_hole_punched_through_both_walls_is_found(tmp_path):
    """Both caps removed, so a ray down +z passes through the mesh entirely.

    Removing only the near cap would not be see-through: with no backface
    culling the far wall still covers those pixels, and correctly so.
    """
    path = tmp_path / "pierced.glb"
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    keep = np.abs(sphere.face_normals[:, 2]) < 0.9
    trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[keep]).export(path)

    result = hole_fraction(path, views=((0.0, 0.0, 1.0),), resolution=512)
    assert result["worst"] > 0.01
    assert result["views"][0]["blobs"] >= 1


def test_a_solid_mesh_stays_solid_from_every_default_view(sphere_glb):
    result = hole_fraction(sphere_glb, resolution=256)
    assert len(result["views"]) == len(DEFAULT_VIEWS)
    for view in result["views"]:
        assert view["hole_px"] == 0
        assert view["silhouette_px"] > 0


def test_higher_resolution_does_not_manufacture_holes(sphere_glb):
    """Sub-pixel splatting is the one approximation here; it must not leak in."""
    for resolution in (256, 512, 1024):
        assert hole_fraction(sphere_glb, resolution=resolution)["worst"] == pytest.approx(
            0.0, abs=1e-4
        )


def test_reports_face_count_and_both_summary_statistics(sphere_glb):
    result = hole_fraction(sphere_glb, resolution=256)
    assert result["faces"] == 5120  # icosphere(subdivisions=4)
    assert result["mean"] <= result["worst"]


def test_chunking_the_rasteriser_is_bit_identical_to_one_pass(sphere_glb, monkeypatch):
    """The (n, k, k) working set is chunked to bound a commit spike on a
    500k-triangle mesh. `covered` is written in place, so the chunk size must
    be invisible in the result -- forcing it down to one triangle per pass is
    the strongest version of that claim."""
    import warlock.meshaudit as meshaudit

    unchunked = hole_fraction(sphere_glb, resolution=256)
    monkeypatch.setattr(meshaudit, "_BATCH_MAX_CELLS", 1)
    chunked = hole_fraction(sphere_glb, resolution=256)
    assert chunked == unchunked
