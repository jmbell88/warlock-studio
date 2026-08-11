"""Freeing a bench run's disk space without losing what it measured.

The whole point of the feature is the asymmetry: a sweep's item directories are
almost entirely pixels and meshes, and the findings are almost entirely
kilobytes of JSONL. So every test here is really one of two assertions -- the
big things went, and the small things that a verdict or a report is built from
did not.
"""

from __future__ import annotations

import json

from warlock.bench import runner as runner_mod


def _make_run(tmp_path, units=("a--s1", "b--s2")):
    """A run directory with everything a real one has, at one byte per file."""
    run_dir = tmp_path / "runs" / "20260805-120000-base-sweep"
    (run_dir / "items").mkdir(parents=True)
    for name in ("manifest.json", "items.jsonl", "scores.json"):
        (run_dir / name).write_text("{}", encoding="utf-8")
    for unit in units:
        item = run_dir / "items" / unit
        (item / "views").mkdir(parents=True)
        for name in (*runner_mod.ARTIFACTS, runner_mod.SOURCE_ARTIFACT):
            (item / name).write_bytes(b"x" * 1024)
        (item / "job.json").write_text('{"params": {}}', encoding="utf-8")
        for angle in range(8):
            (item / "views" / f"{angle:03d}.png").write_bytes(b"y" * 512)
        (item / "views" / "views.json").write_text("{}", encoding="utf-8")
    return run_dir


def _survivors(run_dir):
    return {p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()}


def test_purge_removes_the_meshes_the_images_and_the_rendered_views(tmp_path):
    run_dir = _make_run(tmp_path)
    runner_mod.purge_media(run_dir)

    left = _survivors(run_dir)
    for unit in ("a--s1", "b--s2"):
        for name in (*runner_mod.ARTIFACTS, runner_mod.SOURCE_ARTIFACT):
            assert f"items/{unit}/{name}" not in left
        assert not [p for p in left if p.startswith(f"items/{unit}/views/") and p.endswith(".png")]


def test_purge_keeps_everything_a_finding_is_built_from(tmp_path):
    run_dir = _make_run(tmp_path)
    runner_mod.purge_media(run_dir)

    left = _survivors(run_dir)
    for name in ("manifest.json", "items.jsonl", "scores.json"):
        assert name in left
    for unit in ("a--s1", "b--s2"):
        # job.json carries params["mesh_report"]/["mesh_audit"] -- the mesh
        # measurements the report reads -- and views.json says what was
        # rendered, so a scored run stays interpretable.
        assert f"items/{unit}/job.json" in left
        assert f"items/{unit}/views/views.json" in left


def test_a_purge_reports_the_bytes_it_actually_freed(tmp_path):
    """The receipt is the measurement now that ``media_bytes`` is gone: it was
    a second way to ask the same question with no caller outside this file, and
    ``freed_bytes`` is what anything downstream reads."""
    run_dir = _make_run(tmp_path)
    on_disk = sum(p.stat().st_size for p in runner_mod._media_files(run_dir))
    assert on_disk > 0

    doc = runner_mod.purge_media(run_dir)
    assert doc["freed_bytes"] == on_disk
    # And nothing is left to free.
    assert runner_mod.purge_media(run_dir)["freed_bytes"] == on_disk
    assert not runner_mod._media_files(run_dir)


def test_purge_writes_its_receipt_and_is_idempotent(tmp_path):
    run_dir = _make_run(tmp_path)
    assert not runner_mod.is_purged(run_dir)

    first = runner_mod.purge_media(run_dir)
    assert runner_mod.is_purged(run_dir)
    stored = json.loads((run_dir / runner_mod.PURGED_FILE).read_text("utf-8"))
    assert stored == first
    assert stored["purged_at"] and stored["files"]

    # A second purge frees nothing and must not rewrite the run's history to
    # say so: what it freed is a fact about the run, not about the call.
    again = runner_mod.purge_media(run_dir)
    assert again["freed_bytes"] == first["freed_bytes"]
    assert again["purged_at"] == first["purged_at"]


def test_purge_survives_a_run_with_no_items_directory(tmp_path):
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)
    assert runner_mod.purge_media(run_dir)["freed_bytes"] == 0
