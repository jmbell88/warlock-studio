from __future__ import annotations

import subprocess

import pytest

from warlock.pipelines import optimize


def test_profiles_cover_the_named_tiers():
    assert optimize.PROFILES["draft"] == 20_000
    assert optimize.PROFILES["standard"] == 50_000
    assert optimize.PROFILES["detailed"] == 100_000
    assert optimize.PROFILES["raw"] is None


def test_raw_profile_copies_without_invoking_the_exe(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("gltfpack must not run for the raw profile")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(optimize, "_triangles", lambda p: 7)
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    result = optimize.run(
        src, tmp_path / "model.glb", target_triangles=None, exe=tmp_path / "missing.exe"
    )
    assert (tmp_path / "model.glb").read_bytes() == b"glb"
    assert result["requested"] is None
    assert result["achieved"] == 7


def test_command_uses_the_documented_flags(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        # gltfpack writes its output; stand that in.
        from pathlib import Path

        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
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
        subprocess,
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


def test_resolve_maps_names_and_validates_custom():
    assert optimize.resolve("draft") == 20_000
    assert optimize.resolve("raw") is None
    assert optimize.resolve("custom", 30_000) == 30_000
    with pytest.raises(ValueError):
        optimize.resolve("custom", 1)
    with pytest.raises(ValueError):
        optimize.resolve("nonsense")
