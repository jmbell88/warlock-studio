"""Library integrity and backup.

``verify`` is a report and nothing else: what is pinned here is that each of the
five findings is raised by the damage it names, that the cases that merely
*look* like damage are not raised, and that a healthy library says so. The
false-positive half matters more than the other one -- a verify that reports
every queued job and every rig as broken is a verify nobody runs twice.

``backup`` is pinned on the asymmetry that is easy to mistake for a bug: the
store is copied and the asset tree is not, unless asked.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from warlock.service import library
from warlock.service.errors import Invalid


def _job(svc, job_id, *, kind="text", stage="model", status="done", params=None):
    """A row plus the directory a real job of that shape would own."""
    svc.store.create(kind, "a goblin", params or {}, job_id, stage=stage, status=status)
    return svc.job_dir(job_id)


def _finished(svc, job_id, *, stage="model", artifact="model.glb"):
    job_dir = _job(svc, job_id, stage=stage)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / artifact).write_bytes(b"glb")
    return job_dir


# --- the healthy case ---------------------------------------------------------


def test_an_intact_library_is_reported_intact(svc):
    _finished(svc, "aaaaaaaaaaaa")
    _finished(svc, "bbbbbbbbbbbb", stage="reference", artifact="input.png")
    report = library.verify(svc)
    assert report["ok"] is True
    assert report["findings"] == 0
    assert report["checked"] == 2


def test_an_empty_library_is_intact(svc):
    report = library.verify(svc)
    assert report["ok"] is True
    assert report["checked"] == 0


# --- the five findings --------------------------------------------------------


def test_a_finished_job_with_no_directory_is_reported(svc):
    """The row still lists, offers exports and reads as an asset. Nothing about
    the library says the files behind it are gone."""
    _job(svc, "cccccccccccc")  # row, no mkdir
    report = library.verify(svc)
    assert [f["id"] for f in report["missing_dirs"]] == ["cccccccccccc"]
    assert report["ok"] is False


def test_a_directory_with_no_row_is_reported_and_sized(svc):
    """Invisible to the library *and* to the prune, so it is disk nothing will
    ever reclaim on its own. The size is the number that decides whether the
    user cares, so it is measured rather than counted."""
    stray = svc.config.data_dir / "dddddddddddd"
    stray.mkdir(parents=True)
    (stray / "model.glb").write_bytes(b"x" * 1234)
    report = library.verify(svc)
    assert [f["id"] for f in report["orphan_dirs"]] == ["dddddddddddd"]
    assert report["orphan_bytes"] == 1234


def test_a_finished_job_whose_output_is_missing_is_reported(svc):
    job_dir = _job(svc, "eeeeeeeeeeee")
    job_dir.mkdir(parents=True)
    (job_dir / "input.png").write_bytes(b"png")  # the input survived; the mesh did not
    report = library.verify(svc)
    assert [f["artifact"] for f in report["missing_artifacts"]] == ["model.glb"]


def test_a_verdict_naming_a_vanished_job_is_reported(svc):
    """2026-08-09: 117 verdicts, 100 of them naming directories that no longer
    existed, after a button whose confirmation truthfully said the verdicts
    would be kept. They were. The pixels were not."""
    svc.store.create("text", "a goblin", {}, "ffffffffffff", stage="model", status="done")
    svc.store.add_verdict(
        "ffffffffffff",
        source="human",
        verdict="accept",
        reasons=[],
        vector={},
        stage="model",
    )
    report = library.verify(svc)
    assert [f["id"] for f in report["stale_verdicts"]] == ["ffffffffffff"]


def test_a_params_blob_that_will_not_parse_is_reported(svc):
    """The store answers ``{}`` for these on purpose -- one blob raising out of
    ``next_queued`` starved every job behind it forever. The cost of that
    tolerance is that nothing else can see the damage."""
    _finished(svc, "a1a1a1a1a1a1")
    with sqlite3.connect(svc.config.db_path) as conn:
        conn.execute("UPDATE jobs SET params = ? WHERE id = ?", ("{not json", "a1a1a1a1a1a1"))
    report = library.verify(svc)
    assert report["unreadable_params"] == ["a1a1a1a1a1a1"]


# --- what must *not* be reported ----------------------------------------------


def test_a_queued_job_has_no_directory_yet_and_that_is_not_a_finding(svc):
    _job(svc, "b2b2b2b2b2b2", status="queued")
    assert library.verify(svc)["ok"] is True


def test_a_failed_job_is_expected_to_have_produced_nothing(svc):
    _job(svc, "c3c3c3c3c3c3", status="error")
    assert library.verify(svc)["ok"] is True


def test_a_rig_never_owned_a_directory_of_its_own(svc):
    """A rig, a sheet and a re-texture all write into the *source* job's
    directory. Reporting them as missing would be true of every one of them and
    would mean nothing."""
    _finished(svc, "d4d4d4d4d4d4")
    _job(svc, "e5e5e5e5e5e5", kind="rig", params={"source_job": "d4d4d4d4d4d4"})
    assert library.verify(svc)["ok"] is True


def test_an_unknown_stage_is_not_guessed_at(svc):
    """A stage with no entry in ``PRIMARY`` is not checked. The alternative is
    findings against jobs that are perfectly intact."""
    job_dir = _job(svc, "f6f6f6f6f6f6", stage="something-new")
    job_dir.mkdir(parents=True)
    assert library.verify(svc)["missing_artifacts"] == []


def test_the_app_s_own_files_are_not_mistaken_for_orphans(svc):
    """``JOB_ID_RE`` rather than an exclusion list, so a new sibling directory
    needs no maintenance here to stay unreported."""
    (svc.config.data_dir / "autosave").mkdir(parents=True, exist_ok=True)
    (svc.config.data_dir / "warlock.log").write_text("hello", encoding="utf-8")
    (svc.config.data_dir / "some-future-thing").mkdir(exist_ok=True)
    assert library.verify(svc)["orphan_dirs"] == []


def test_a_half_written_mesh_is_not_a_mesh_here_either(svc):
    """Through ``files.ready`` rather than ``Path.exists``: the same question
    the exporter and the library ask, so the three cannot drift."""
    job_dir = _job(svc, "a7a7a7a7a7a7", status="running")
    job_dir.mkdir(parents=True)
    (job_dir / "model.glb").write_bytes(b"partial")
    # Running, so not checked at all -- and once it is done, ready() agrees.
    assert library.verify(svc)["ok"] is True


def test_the_walk_pages_through_a_history_longer_than_one_page(svc, monkeypatch):
    """``prune_jobs``' keyset cursor, for its reason: a library longer than a
    single page must still be verifiable, and that is exactly the library that
    needs it."""
    monkeypatch.setattr(library, "_PAGE", 3)
    for i in range(7):
        _finished(svc, f"{i:012x}")
    assert library.verify(svc)["checked"] == 7


# --- backup -------------------------------------------------------------------


def test_a_backup_copies_the_store_and_it_opens(svc, tmp_path):
    _finished(svc, "b8b8b8b8b8b8")
    out = library.backup(svc, tmp_path / "bk")
    assert out["store_bytes"] > 0
    with sqlite3.connect(out["store"]) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", ("b8b8b8b8b8b8",)).fetchone()
    assert json.loads(row["params"]) == {}
    assert row["prompt"] == "a goblin"


def test_the_asset_tree_is_left_alone_unless_asked(svc, tmp_path):
    """The store is megabytes and irreplaceable; the tree is tens of gigabytes
    of ordinary files the user's own backup already handles. Copying it by
    default would turn a one-second operation into an hour-long one."""
    _finished(svc, "c9c9c9c9c9c9")
    out = library.backup(svc, tmp_path / "bk")
    assert out["included_assets"] is False
    assert out["assets"] == 0
    assert not (tmp_path / "bk" / "assets").exists()


def test_asking_for_the_assets_copies_them(svc, tmp_path):
    _finished(svc, "dadadadadada")
    out = library.backup(svc, tmp_path / "bk", include_assets=True)
    assert out["assets"] == 1
    assert (tmp_path / "bk" / "assets" / "dadadadadada" / "model.glb").read_bytes() == b"glb"


def test_a_backup_onto_a_file_refuses_rather_than_failing_later(svc, tmp_path):
    target = tmp_path / "not-a-folder"
    target.write_text("occupied", encoding="utf-8")
    with pytest.raises(Invalid, match="folder"):
        library.backup(svc, target)
