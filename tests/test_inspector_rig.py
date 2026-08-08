"""The rig section's two readings of a rig that lives in another job's row.

`TODO.md` §6, and the whole difficulty is one sentence: 0d wrote the weighting
outcome onto the **rig** job, but the Rig & Pose tab opens on the **model** job,
because that is the asset a user selects and the mesh the rig belongs to. So the
natural selection showed nothing at all -- a degraded rig looked exactly like a
good one again, which is the failure 0d existed to end.

The fix reads ``rig.json`` from the selected job's own directory, where the
worker already puts it (a rig writes into the *source* job's directory -- the
rig belongs to the mesh). That is a file read on the frame thread, so it is
cached, and the cache stamp is a mtime -- which on Windows is coarse enough to
miss a write 155 times in 200. Hence the racily-clean rule, and hence these
tests: a re-rig that lands inside the stamped mtime's own tick must still be
seen, or the pane reports last week's weighting forever.

The alternative considered and rejected was making ``queue._rig`` a second
writer onto the model job's row. It is refused because that row has a
single-owner invariant and ``set_params`` is last-write-wins.
"""

from __future__ import annotations

import json
from typing import Any

from warlock.studio.panes import inspector
from warlock.studio.state import AppState


class FakeCtx:
    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = AppState()

    def submit(self, key: str, run: Any, *args: Any) -> bool:  # pragma: no cover
        raise AssertionError("the rig summary is a frame-thread read, not a task")


def _rigged(svc, **meta):
    """A finished mesh with a rig beside it, the way the worker leaves one."""
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.glb").write_bytes(b"glTF-not-really")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "template": "humanoid", "bones": [], **meta}), "utf-8"
    )
    return job_id


def _job(svc, job_id):
    return svc.store.get(job_id)


# --- reading the rig's own record --------------------------------------------


def test_the_weighting_outcome_is_read_off_the_rig_beside_the_mesh(svc):
    """The verdict the *rig* job recorded, shown against the *model* job the
    user actually selects."""
    ctx = FakeCtx(svc)
    job_id = _rigged(svc, weighting="envelope", weighting_reason="bone-heat failed: no volume")

    meta = inspector.rig_meta(ctx, _job(svc, job_id))

    assert meta is not None
    assert meta["weighting"] == "envelope"
    assert meta["weighting_reason"] == "bone-heat failed: no volume"


def test_an_unrigged_mesh_has_no_rig_summary_and_no_error(svc):
    ctx = FakeCtx(svc)
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)

    assert inspector.rig_meta(ctx, _job(svc, job_id)) is None


def test_the_rig_verdict_prefers_the_rig_file_over_the_selected_rows_params(svc):
    """A model row carries no ``weighting`` -- the rig job does -- so the file is
    the only source for the selection the tab actually opens on. Where a row
    *does* carry one (the rig job itself, selected in the library), it still
    wins: it is the row the user is looking at."""
    ctx = FakeCtx(svc)
    job_id = _rigged(svc, weighting="automatic-welded")

    job = _job(svc, job_id)
    assert inspector.weighting_verdict(job.get("params") or {}) is None
    assert inspector.rig_weighting(ctx, job) == (
        inspector.weighting_verdict({"weighting": "automatic-welded"})
    )


def test_a_rig_jobs_own_row_still_answers_for_itself(svc):
    """The rig job has the params and no rig.json of its own."""
    ctx = FakeCtx(svc)
    job_id = svc.store.create(
        "rig", "a chest", {"weighting": "envelope"}, stage="model", status="done"
    )
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)

    verdict = inspector.rig_weighting(ctx, _job(svc, job_id))
    assert verdict is not None
    assert "envelope" in verdict[1]


# --- the cache ---------------------------------------------------------------


def _settled(monkeypatch, job_dir):
    """Move the clock a tick past the directory's mtime.

    Every test that wants a *cache hit* has to do this, and that is the rule
    working rather than a wrinkle in it: a directory written a moment ago is
    answered correctly and deliberately not remembered, so a test that skipped
    this would be asserting against the racy window the rule exists to refuse.
    """
    import warlock.studio.panes.inspector as inspector_mod
    from warlock.service.files import MTIME_RACE_NS

    settled = job_dir.stat().st_mtime_ns + MTIME_RACE_NS * 2
    monkeypatch.setattr(inspector_mod.time, "time_ns", lambda: settled)


def test_the_rig_file_is_read_once_and_then_cached(svc, monkeypatch):
    """Frame-thread work, at 60 fps, for as long as the tab is open."""
    ctx = FakeCtx(svc)
    job_id = _rigged(svc, weighting="automatic")
    job = _job(svc, job_id)
    _settled(monkeypatch, svc.job_dir(job_id))
    reads: list[Any] = []

    from warlock import rigging

    real = rigging.read_rig
    monkeypatch.setattr(rigging, "read_rig", lambda d: (reads.append(d), real(d))[1])

    for _ in range(10):
        inspector.rig_meta(ctx, job)

    assert len(reads) == 1


def test_a_rig_written_a_moment_ago_is_not_cached_at_all(svc, monkeypatch):
    """The other half of the same rule, and the one that makes it safe: inside
    the racy window the answer is re-read every frame rather than remembered."""
    ctx = FakeCtx(svc)
    job_id = _rigged(svc, weighting="automatic")
    job = _job(svc, job_id)
    reads: list[Any] = []

    from warlock import rigging

    real = rigging.read_rig
    monkeypatch.setattr(rigging, "read_rig", lambda d: (reads.append(d), real(d))[1])

    for _ in range(3):
        assert inspector.rig_meta(ctx, job)["weighting"] == "automatic"

    assert len(reads) == 3
    assert ctx.state.rig_meta_cache == {}


def test_a_re_rig_inside_the_mtime_tick_is_still_seen(svc, monkeypatch):
    """Git's racily-clean rule. Windows writes a directory's mtime from the
    system clock, whose tick is 15.6 ms, so a write landing after our read but
    inside the stamped mtime's tick is invisible to the stamp -- and invisible
    *forever*, because every later comparison keeps matching. A stamp is
    therefore only remembered once its mtime is safely in the past.

    Simulated by freezing the clock at the stamp's own mtime, which is the
    worst case the rule exists for: nothing may be cached at all.
    """
    ctx = FakeCtx(svc)
    job_id = _rigged(svc, weighting="automatic")
    job = _job(svc, job_id)
    job_dir = svc.job_dir(job_id)

    import warlock.studio.panes.inspector as inspector_mod

    monkeypatch.setattr(
        inspector_mod.time, "time_ns", lambda: job_dir.stat().st_mtime_ns
    )
    assert inspector.rig_meta(ctx, job)["weighting"] == "automatic"

    # A re-rig, published the way finalize_rig publishes one.
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "template": "humanoid", "weighting": "envelope"}), "utf-8"
    )

    assert inspector.rig_meta(ctx, job)["weighting"] == "envelope"


def test_the_cache_is_per_job_so_a_new_selection_reads_its_own_rig(svc):
    ctx = FakeCtx(svc)
    first = _rigged(svc, weighting="automatic")
    second = _rigged(svc, weighting="envelope")

    assert inspector.rig_meta(ctx, _job(svc, first))["weighting"] == "automatic"
    assert inspector.rig_meta(ctx, _job(svc, second))["weighting"] == "envelope"
    assert inspector.rig_meta(ctx, _job(svc, first))["weighting"] == "automatic"


def test_the_cache_lives_on_the_app_state_not_in_a_module_global(svc, monkeypatch):
    """A module global would be shared by every window and outlive the data
    directory it describes; ``AppState`` is the app's own per-session state."""
    ctx = FakeCtx(svc)
    job_id = _rigged(svc, weighting="automatic")
    _settled(monkeypatch, svc.job_dir(job_id))
    inspector.rig_meta(ctx, _job(svc, job_id))

    assert job_id in ctx.state.rig_meta_cache


# --- the QA sheet ------------------------------------------------------------


def test_the_deformation_sheet_is_offered_when_its_sidecar_exists(svc):
    """``rig_qa.png`` reaches ``job["files"]`` through ``LISTED`` and is
    exportable by name, but the inspector's download grid is a hardcoded list --
    so nothing appeared. It is gated on the *sidecar* for the reason rig.glb is
    gated on rig.json: the sheet is written before the JSON that describes it.
    """
    job_id = _rigged(svc)
    job_dir = svc.job_dir(job_id)
    job = _job(svc, job_id)

    assert inspector.deform_qa_path(svc, job) is None

    (job_dir / "rig_qa.png").write_bytes(b"png-not-really")
    assert inspector.deform_qa_path(svc, job) is None

    (job_dir / "rig_qa.json").write_text("{}", "utf-8")
    assert inspector.deform_qa_path(svc, job) == job_dir / "rig_qa.png"
