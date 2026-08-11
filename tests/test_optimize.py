from __future__ import annotations

import subprocess

import pytest

from warlock import winjob
from warlock.pipelines import optimize


def test_profiles_cover_the_named_tiers():
    assert optimize.PROFILES["draft"] == 20_000
    assert optimize.PROFILES["standard"] == 50_000
    assert optimize.PROFILES["detailed"] == 100_000
    assert optimize.PROFILES["raw"] is None


def test_raw_profile_copies_without_invoking_the_exe(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("gltfpack must not run for the raw profile")

    monkeypatch.setattr(winjob, "run", explode)

    def no_counting(path):
        raise AssertionError("the raw profile must not load the mesh to count it")

    monkeypatch.setattr(optimize, "_triangles", no_counting)
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    result = optimize.run(
        src, tmp_path / "model.glb", target_triangles=None, exe=tmp_path / "missing.exe"
    )
    assert (tmp_path / "model.glb").read_bytes() == b"glb"
    # Not measured rather than guessed: nothing in this path needs the count,
    # and the mesh report measures the finished model a step later anyway.
    assert result["requested"] is None
    assert result["achieved"] is None
    assert result["source_triangles"] is None


def test_command_uses_the_documented_flags(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        # gltfpack writes its output; stand that in.
        from pathlib import Path

        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(winjob, "run", fake_run)
    monkeypatch.setattr(
        optimize, "_triangles", lambda p: 100_000 if p.name == "source.glb" else 50_000
    )
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    result = optimize.run(src, tmp_path / "model.glb", target_triangles=50_000, exe=exe)

    argv = seen["argv"]
    assert argv[0] == str(exe)
    assert "-noq" in argv and "-ke" in argv and "-km" in argv
    assert argv[argv.index("-si") + 1] == "0.5"
    assert result["requested"] == 50_000
    assert result["achieved"] == 50_000
    assert (tmp_path / "model.glb").read_bytes() == b"optimised"


def test_a_failing_exe_raises_rather_than_leaving_a_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        winjob,
        "run",
        lambda argv, **k: subprocess.CompletedProcess(argv, 1, "", "boom"),
    )
    monkeypatch.setattr(optimize, "_triangles", lambda p: 100_000)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    with pytest.raises(optimize.OptimizeError):
        optimize.run(src, tmp_path / "model.glb", target_triangles=50_000, exe=exe)
    assert not (tmp_path / "model.glb").exists()


def test_a_missing_exe_raises_and_does_not_silently_ship_the_source(tmp_path, monkeypatch):
    # The raw profile is the way to opt out of optimizing. A budget that was
    # asked for and not applied must be loud, not a copy wearing the name.
    monkeypatch.setattr(optimize, "_triangles", lambda p: 100_000)
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    with pytest.raises(optimize.OptimizeError):
        optimize.run(
            src, tmp_path / "model.glb", target_triangles=50_000, exe=tmp_path / "nope.exe"
        )


def test_a_failed_raw_copy_never_corrupts_an_existing_model(tmp_path, monkeypatch):
    """The within-budget path overwrites a model.glb the file route may be
    serving concurrently (POST /optimize runs on a *done* job), so the copy
    must be staged: a failure mid-copy leaves the old file intact rather
    than truncated."""
    from pathlib import Path

    monkeypatch.setattr(optimize, "_triangles", lambda p: 7)
    src = tmp_path / "source.glb"
    src.write_bytes(b"new")
    dest = tmp_path / "model.glb"
    dest.write_bytes(b"old-and-complete")

    def partial_copy(a, b, *args, **kwargs):
        Path(b).write_bytes(b"par")
        raise OSError("disk full")

    monkeypatch.setattr(optimize.shutil, "copyfile", partial_copy)
    with pytest.raises(OSError):
        optimize.run(src, dest, target_triangles=None, exe=tmp_path / "missing.exe")
    assert dest.read_bytes() == b"old-and-complete"


def test_staged_copy_replaces_the_dest_atomically(tmp_path):
    src = tmp_path / "a"
    src.write_bytes(b"new")
    dest = tmp_path / "b"
    dest.write_bytes(b"old")
    optimize.staged_copy(src, dest)
    assert dest.read_bytes() == b"new"
    assert not list(tmp_path.glob("*.tmp"))


def test_resolve_maps_names_and_validates_custom():
    assert optimize.resolve("draft") == 20_000
    assert optimize.resolve("raw") is None
    assert optimize.resolve("custom", 30_000) == 30_000
    with pytest.raises(ValueError):
        optimize.resolve("custom", 1)
    with pytest.raises(ValueError):
        optimize.resolve("nonsense")


def test_an_unmeasurable_output_leaves_no_staging_file_behind(tmp_path, monkeypatch):
    """Every other exit from run() unlinks the .glb.opt.tmp; the tail did not.
    A gltfpack output trimesh cannot parse raised straight past it and left the
    staging file sitting beside the served model."""
    def fake_run(argv, **kwargs):
        from pathlib import Path

        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(winjob, "run", fake_run)
    calls: list[str] = []

    def counting(path):
        calls.append(path.name)
        if len(calls) > 1:
            raise ValueError("this is not a mesh")
        return 100_000

    monkeypatch.setattr(optimize, "_triangles", counting)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    dest = tmp_path / "model.glb"
    with pytest.raises(ValueError, match="not a mesh"):
        optimize.run(src, dest, target_triangles=50_000, exe=exe)
    assert not dest.exists()
    assert list(tmp_path.glob("*.opt.tmp")) == []
