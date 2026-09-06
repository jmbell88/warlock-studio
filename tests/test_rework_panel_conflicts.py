"""A sibling rework job greys the button before the press, not after.

The 2026-09-06 audit, finding create-02: ``retarget_panel``, ``remesh_panel``
and ``texture_panel`` greyed their submit button only on their own
client-side ``busy`` flag, never on whether some *other* job (a rig, a sheet,
or a sibling rework) is already queued or running against the same asset --
``_jobs_lifecycle.dependent_jobs`` is the service's own answer to that
question, and ``_jobs_rework._require_no_dependents`` already refuses on it.
So the queued/running collision was discovered only after a wasted press and
a ``Conflict`` toast, instead of as a pre-stated reason the way the stale-rig
warning is shown before the button -- the framing Manual chapter 23 promises
("the panel will not hide from you").

Each panel's ``dependent_job_reason`` is a pure function over
``ctx.cache.jobs`` (the frame-thread-safe job list ``jobs_cache`` already
refreshes on its own timer), so it is testable here with no imgui and no
sqlite in play.
"""

from __future__ import annotations

from typing import Any

from warlock.studio.panes import remesh_panel, retarget_panel, texture_panel


def _job(status: str, source_job: str | None, kind: str = "retexture") -> dict[str, Any]:
    return {"id": "x", "status": status, "kind": kind, "params": {"source_job": source_job}}


def test_retarget_panel_greys_rebuild_button_while_a_dependent_retexture_is_running():
    jobs = [_job("running", "mesh1", kind="retexture")]
    assert retarget_panel.dependent_job_reason(jobs, "mesh1") is not None
    assert retarget_panel.dependent_job_reason(jobs, "mesh2") is None


def test_remesh_panel_greys_button_while_a_dependent_rig_is_queued():
    jobs = [_job("queued", "mesh1", kind="rig")]
    assert remesh_panel.dependent_job_reason(jobs, "mesh1") is not None


def test_texture_panel_greys_button_while_a_dependent_remesh_is_running():
    jobs = [_job("running", "mesh1", kind="remesh")]
    assert texture_panel.dependent_job_reason(jobs, "mesh1") is not None


def test_a_finished_dependent_job_does_not_grey_the_button():
    """The refusal is about *unfinished* work only: ``dependent_jobs`` (the
    service side of this) reads ``active_jobs()``, which is ``queued`` and
    ``running`` rows -- a ``done`` sibling has already stopped writing."""
    jobs = [_job("done", "mesh1", kind="retexture")]
    assert retarget_panel.dependent_job_reason(jobs, "mesh1") is None
    assert remesh_panel.dependent_job_reason(jobs, "mesh1") is None
    assert texture_panel.dependent_job_reason(jobs, "mesh1") is None


def test_an_unrelated_job_does_not_grey_the_button():
    """A queued job against a *different* asset must not ring this one's
    button -- the count is per-``source_job``, not "anything in the queue"."""
    jobs = [_job("running", "some-other-mesh", kind="retexture")]
    assert retarget_panel.dependent_job_reason(jobs, "mesh1") is None
    assert remesh_panel.dependent_job_reason(jobs, "mesh1") is None
    assert texture_panel.dependent_job_reason(jobs, "mesh1") is None
