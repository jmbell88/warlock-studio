"""The inspector's graded verdict: any finished asset is gradeable, and the
staged tags belong to the asset that was on screen when they were chosen."""

from __future__ import annotations

from typing import Any

from warlock.studio.panes import inspector
from warlock.studio.state import AppState


class FakeCtx:
    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = AppState()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


def _done_mesh(svc, **params):
    return svc.store.create("image", "a chest", params, stage="model", status="done")


def test_staged_tags_are_keyed_by_job_so_a_new_selection_drops_them(svc):
    ctx = FakeCtx(svc)
    inspector.toggle_tag(ctx.state, "aaaaaaaaaaaa", "holes")

    assert inspector.staged_tags(ctx.state, "aaaaaaaaaaaa") == ["holes"]
    # Selecting a different asset is not "still staged" -- the tag would ride
    # onto a verdict about something the user never looked at. Nothing has to
    # *notice* the change: a different id simply has nothing staged.
    assert inspector.staged_tags(ctx.state, "bbbbbbbbbbbb") == []


def test_toggling_the_same_tag_twice_unstages_it(svc):
    ctx = FakeCtx(svc)
    inspector.toggle_tag(ctx.state, "aaaaaaaaaaaa", "clean-shape")
    inspector.toggle_tag(ctx.state, "aaaaaaaaaaaa", "holes")
    assert inspector.staged_tags(ctx.state, "aaaaaaaaaaaa") == ["clean-shape", "holes"]

    inspector.toggle_tag(ctx.state, "aaaaaaaaaaaa", "clean-shape")
    assert inspector.staged_tags(ctx.state, "aaaaaaaaaaaa") == ["holes"]


def test_a_tag_the_vocabulary_does_not_carry_is_never_staged(svc):
    """It would make ``record_verdict`` refuse the whole verdict, which loses a
    grade to a typo in a key map."""
    ctx = FakeCtx(svc)
    inspector.toggle_tag(ctx.state, "aaaaaaaaaaaa", "ugly")
    assert inspector.staged_tags(ctx.state, "aaaaaaaaaaaa") == []


def test_starting_on_another_asset_replaces_rather_than_appends(svc):
    ctx = FakeCtx(svc)
    inspector.toggle_tag(ctx.state, "aaaaaaaaaaaa", "holes")
    inspector.toggle_tag(ctx.state, "bbbbbbbbbbbb", "broken")

    assert inspector.staged_tags(ctx.state, "bbbbbbbbbbbb") == ["broken"]
    assert inspector.staged_tags(ctx.state, "aaaaaaaaaaaa") == []


def test_recording_writes_a_graded_verdict_clears_and_queues_the_findings(svc):
    ctx = FakeCtx(svc)
    job_id = _done_mesh(svc, lora_weight=0.9)
    inspector.toggle_tag(ctx.state, job_id, "holes")

    assert inspector.record_verdict(ctx, job_id, -4, ("holes",)) is True

    row = svc.store.latest_verdicts()[0]
    assert (row["job_id"], row["grade"], row["reasons"]) == (job_id, -4, ["holes"])
    # The word is derived from the grade by one writer, never asserted here.
    assert row["verdict"] == "reject"
    assert row["vector"] == {"lora_weight": 0.9, "stage": "model"}
    assert ctx.state.inspector_tags_job is None
    assert ctx.state.inspector_tags == []
    # The recompute reads every verdict and writes a file, so it is asked for
    # rather than done here -- through Review's own request, so the inspector
    # keeps no second spelling of Review's task key.
    assert ctx.state.findings_dirty is True
    assert ctx.submitted == []


def test_the_toast_names_the_grade_and_not_the_derived_word(svc):
    """The word answers a coarser question than the buttons asked."""
    ctx = FakeCtx(svc)
    inspector.record_verdict(ctx, _done_mesh(svc), 5)
    assert ctx.toasts[-1] == ("Recorded: +5.", "info")

    inspector.record_verdict(ctx, _done_mesh(svc), 0)
    assert ctx.toasts[-1] == ("Recorded: 0.", "info")


def test_a_refused_verdict_toasts_and_writes_nothing(svc):
    ctx = FakeCtx(svc)
    assert inspector.record_verdict(ctx, "ffffffffffff", 3) is False
    assert ctx.toasts[-1][1] == "error"
    assert svc.store.latest_verdicts() == []


def test_a_grade_outside_the_scale_is_refused_rather_than_clamped(svc):
    ctx = FakeCtx(svc)
    job_id = _done_mesh(svc)
    assert inspector.record_verdict(ctx, job_id, 9) is False
    assert svc.store.latest_verdicts() == []


def test_an_inspector_verdict_feeds_the_same_findings_a_sweep_does(svc):
    from warlock.service import findings as svc_findings

    ctx = FakeCtx(svc)
    for _ in range(5):
        inspector.record_verdict(ctx, _done_mesh(svc, platform="pc"), 4)

    doc = svc_findings.aggregate(svc.store)
    assert doc["params"]["platform"]["pc"] == {
        "n": 5, "accepts": 5, "accept_rate": 1.0,
        "wilson_low": svc_findings.wilson_low(5, 5),
        "graded_n": 5,
        "mean_grade": 4.0,
        "grades": {"4": 5},
        "tags": {"good": [], "bad": []},
        "sources": {"human": {"accept": 5, "reject": 0}},
        "top_reasons": [],
    }
    assert len(svc_findings.presets(doc)) == 1
