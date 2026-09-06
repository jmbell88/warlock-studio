"""A create-side cap greys the button before the press, not after.

The 2026-09-06 audit, second run, findings create2-06 and create2-08: the
same defect in two files. ``sheet_panel.validate()`` stated every refusal
``create_sheet`` can raise except the ``MAX_SHEETS`` cap, and
``sprite_panel._submit`` graded its button without ever reading
``MAX_SPRITE_DRAFTS``, so both caps were discovered only after a wasted
press as a fieldless ``Conflict`` toast. Both panes now carry a pure
``*_cap_reason`` function beside the draw call, following the shape
``retarget_panel.dependent_job_reason`` established earlier the same day:
a function over state the frame already has (``ctx.state.preview["sheets"]``
/ ``draft_records`` and ``ctx.cache.jobs``), with no sqlite and no directory
listing of its own -- testable here with no imgui in play.
"""

from __future__ import annotations

from typing import Any

from warlock import rigging
from warlock.studio.panes import sheet_panel, sprite_panel


def _job(status: str, source_job: str, kind: str, **extra: Any) -> dict[str, Any]:
    params = {"source_job": source_job, **extra}
    return {"id": "x", "status": status, "kind": kind, "params": params}


def test_sheet_panel_states_the_sheet_cap_before_the_button_is_pressed():
    saved = [{"id": str(i)} for i in range(rigging.MAX_SHEETS)]
    assert sheet_panel.sheet_cap_reason(saved, [], "mesh1") is not None
    # One short of the cap: no reason yet.
    assert sheet_panel.sheet_cap_reason(saved[:-1], [], "mesh1") is None


def test_sheet_cap_counts_queued_sibling_sheet_jobs_too():
    """The cap counts on-disk sheets *plus* unfinished jobs that will land
    one, matching ``service.sheets.queued_sheets`` -- a rapid double-press
    must not both read the same under-cap count."""
    saved = [{"id": str(i)} for i in range(rigging.MAX_SHEETS - 1)]
    queued = [_job("queued", "mesh1", "sheet")]
    assert sheet_panel.sheet_cap_reason(saved, queued, "mesh1") is not None


def test_sheet_cap_counts_a_queued_troupe_rig_marked_for_a_sheet():
    saved = [{"id": str(i)} for i in range(rigging.MAX_SHEETS - 1)]
    queued = [_job("running", "mesh1", "rig", troupe_sheet=True)]
    assert sheet_panel.sheet_cap_reason(saved, queued, "mesh1") is not None


def test_sheet_cap_ignores_an_unrelated_or_finished_job():
    saved = [{"id": str(i)} for i in range(rigging.MAX_SHEETS - 1)]
    other_asset = [_job("queued", "mesh2", "sheet")]
    finished = [_job("done", "mesh1", "sheet")]
    plain_rig = [_job("queued", "mesh1", "rig")]
    assert sheet_panel.sheet_cap_reason(saved, other_asset, "mesh1") is None
    assert sheet_panel.sheet_cap_reason(saved, finished, "mesh1") is None
    assert sheet_panel.sheet_cap_reason(saved, plain_rig, "mesh1") is None


def test_sprite_panel_states_the_draft_cap_before_the_button_is_pressed():
    records = [{"id": str(i)} for i in range(rigging.MAX_SPRITE_DRAFTS)]
    assert sprite_panel.sprite_draft_cap_reason(records, [], "draw1") is not None
    assert sprite_panel.sprite_draft_cap_reason(records[:-1], [], "draw1") is None


def test_sprite_draft_cap_counts_a_queued_sibling_synthesis_too():
    records = [{"id": str(i)} for i in range(rigging.MAX_SPRITE_DRAFTS - 1)]
    queued = [_job("running", "draw1", "sprite_synthesis")]
    assert sprite_panel.sprite_draft_cap_reason(records, queued, "draw1") is not None


def test_sprite_draft_cap_ignores_an_unrelated_or_finished_job():
    records = [{"id": str(i)} for i in range(rigging.MAX_SPRITE_DRAFTS - 1)]
    other_asset = [_job("queued", "draw2", "sprite_synthesis")]
    finished = [_job("done", "draw1", "sprite_synthesis")]
    assert sprite_panel.sprite_draft_cap_reason(records, other_asset, "draw1") is None
    assert sprite_panel.sprite_draft_cap_reason(records, finished, "draw1") is None
