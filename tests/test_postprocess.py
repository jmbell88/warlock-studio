"""scale_glb must resize the mesh without re-encoding the file's binary payload."""

from __future__ import annotations

import os
import struct
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from warlock.pipelines.postprocess import (
    _split_glb,
    glb_to_obj_zip,
    glb_to_stl,
    scale_glb,
)


@pytest.fixture
def cube_glb(tmp_path):
    """A 1 m cube exported as GLB."""
    path = tmp_path / "model.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
    return path


def _extent(path):
    return float(trimesh.load(path).extents.max())


def test_scales_longest_axis_to_the_target(cube_glb):
    factor = scale_glb(cube_glb, 2.5)
    assert factor == pytest.approx(2.5)
    assert _extent(cube_glb) == pytest.approx(2.5, rel=1e-4)


def test_scales_down_too(tmp_path):
    path = tmp_path / "model.glb"
    trimesh.creation.box(extents=(4.0, 1.0, 2.0)).export(path)
    scale_glb(path, 1.0)
    mesh = trimesh.load(path)
    assert mesh.extents.max() == pytest.approx(1.0, rel=1e-4)
    # Proportions are preserved -- it is a uniform scale, not a fit-to-box.
    assert sorted(mesh.extents) == pytest.approx([0.25, 0.5, 1.0], rel=1e-4)


def test_binary_payload_is_untouched(cube_glb):
    _, _, before = _split_glb(cube_glb.read_bytes())
    scale_glb(cube_glb, 3.0)
    _, _, after = _split_glb(cube_glb.read_bytes())
    assert before == after


def test_rewritten_container_stays_well_formed(cube_glb):
    scale_glb(cube_glb, 3.0)
    data = cube_glb.read_bytes()
    _magic, _version, length = struct.unpack_from("<III", data, 0)
    assert length == len(data)
    json_len = struct.unpack_from("<I", data, 12)[0]
    assert json_len % 4 == 0  # the spec requires 4-byte-aligned chunks


def test_scaling_is_idempotent_when_already_at_target(cube_glb):
    assert scale_glb(cube_glb, 1.0) == 1.0
    assert _extent(cube_glb) == pytest.approx(1.0, rel=1e-4)


def test_repeated_scaling_composes(cube_glb):
    scale_glb(cube_glb, 2.0)
    scale_glb(cube_glb, 0.5)
    assert _extent(cube_glb) == pytest.approx(0.5, rel=1e-4)


def test_degenerate_target_is_a_no_op(cube_glb):
    assert scale_glb(cube_glb, 0.0) == 1.0
    assert _extent(cube_glb) == pytest.approx(1.0, rel=1e-4)


def test_flat_mesh_with_zero_extent_axis_still_scales(tmp_path):
    """A plane has a zero-thickness axis; only the *longest* axis matters."""
    path = tmp_path / "model.glb"
    plane = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0]], dtype=float),
        faces=np.array([[0, 1, 2]]),
    )
    plane.export(path)
    scale_glb(path, 1.0)
    assert trimesh.load(path).extents.max() == pytest.approx(1.0, rel=1e-4)


def test_derived_exports_see_the_scale(cube_glb, tmp_path):
    """The STL/OBJ paths load the GLB with trimesh, so they must agree with it."""
    scale_glb(cube_glb, 3.0)
    stl = glb_to_stl(cube_glb, tmp_path / "model.stl")
    assert trimesh.load(stl).extents.max() == pytest.approx(3.0, rel=1e-4)


def test_rejects_a_file_that_is_not_a_glb(tmp_path):
    path = tmp_path / "model.glb"
    path.write_bytes(b"not a glb at all, really")
    with pytest.raises(ValueError):
        scale_glb(path, 2.0)


def test_write_is_atomic_rename_not_in_place_truncation(cube_glb, monkeypatch):
    """A concurrent reader must never observe a 0-byte/partial file mid-scale.

    write_bytes() opens "wb", which truncates immediately; scale_glb must
    instead write to a temp file and os.replace() it into place.
    """
    seen_sizes = []
    real_replace = os.replace

    def spying_replace(src, dst):
        # If the rewrite ever truncated glb_path in place before this call,
        # dst would already be 0 bytes by now.
        seen_sizes.append(Path(dst).stat().st_size if Path(dst).exists() else None)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spying_replace)
    original_size = cube_glb.stat().st_size

    scale_glb(cube_glb, 3.0)

    assert seen_sizes == [original_size]
    assert not cube_glb.with_suffix(".glb.tmp").exists()


# --- on-demand exports ------------------------------------------------------
#
# STL and OBJ are produced by the file-serving route the first time they are
# requested, so two requests for the same artifact (one double-clicked download
# link) can run concurrently over the same destination path.


def test_obj_zip_contains_the_model(cube_glb, tmp_path):
    zip_path = glb_to_obj_zip(cube_glb, tmp_path / "model_obj.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "model.obj" in names


def test_obj_zip_leaves_no_working_directory_behind(cube_glb, tmp_path):
    """The export workdir used to be a fixed 'obj_export' next to the zip, kept
    forever -- a second copy of the model and every texture in each job dir."""
    out = tmp_path / "out"
    out.mkdir()
    glb_to_obj_zip(cube_glb, out / "model_obj.zip")
    assert [p.name for p in out.iterdir()] == ["model_obj.zip"]


def test_obj_zip_includes_nested_texture_files(cube_glb, tmp_path, monkeypatch):
    """trimesh may write textures into a subdirectory; the old non-recursive
    iterdir() would hand zf.write a directory and blow up."""
    import warlock.pipelines.postprocess as pp

    real_load = pp.trimesh.load

    def load_with_nested_export(path, *args, **kwargs):
        scene = real_load(path, *args, **kwargs)
        real_export = scene.export

        def export(target, *a, **k):
            result = real_export(target, *a, **k)
            nested = Path(target).parent / "textures"
            nested.mkdir(exist_ok=True)
            (nested / "albedo.png").write_bytes(b"fake-texture")
            return result

        scene.export = export
        return scene

    monkeypatch.setattr(pp.trimesh, "load", load_with_nested_export)
    zip_path = glb_to_obj_zip(cube_glb, tmp_path / "model_obj.zip")
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/albedo.png" in zf.namelist()


@pytest.mark.parametrize(
    "export, name",
    [(glb_to_stl, "model.stl"), (glb_to_obj_zip, "model_obj.zip")],
)
def test_exports_land_atomically(cube_glb, tmp_path, monkeypatch, export, name):
    """The destination must not exist at all until the file is complete."""
    dest = tmp_path / name
    real_replace = os.replace
    observed = []

    def spying_replace(src, dst):
        if Path(dst) == dest:
            observed.append(dest.exists())
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spying_replace)
    export(cube_glb, dest)

    assert observed == [False], "the export wrote to the destination before finishing"
    assert dest.stat().st_size > 0
    # No temp files left over.
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


@pytest.mark.parametrize(
    "export, name",
    [(glb_to_stl, "model.stl"), (glb_to_obj_zip, "model_obj.zip")],
)
def test_concurrent_exports_produce_a_valid_file(cube_glb, tmp_path, export, name):
    dest = tmp_path / name
    errors: list[BaseException] = []
    start = threading.Barrier(4)

    def run():
        start.wait()
        try:
            export(cube_glb, dest)
        except BaseException as exc:  # noqa: BLE001 - any failure is the finding
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    # Whichever writer won, the result is a complete file, not a torn one.
    if name.endswith(".zip"):
        with zipfile.ZipFile(dest) as zf:
            assert zf.testzip() is None
            assert "model.obj" in zf.namelist()
    else:
        assert trimesh.load(dest).extents.max() == pytest.approx(1.0, rel=1e-4)
