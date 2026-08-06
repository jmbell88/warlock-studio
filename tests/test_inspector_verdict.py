"""The inspector's Accept/Reject: any finished asset is verdictable, and the
armed state belongs to the asset that was on screen when it was armed."""

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


def test_the_armed_flag_is_keyed_by_job_so_a_new_selection_disarms(svc):
    ctx = FakeCtx(svc)
    inspector.arm_verdict(ctx.state, "aaaaaaaaaaaa")

    assert inspector.verdict_armed(ctx.state, "aaaaaaaaaaaa") is True
    # Selecting a different asset is not "still armed" -- the reason buttons
    # would file a rejection against something the user never looked at.
    assert inspector.verdict_armed(ctx.state, "bbbbbbbbbbbb") is False


def test_recording_writes_a_verdict_disarms_and_queues_the_findings(svc):
    ctx = FakeCtx(svc)
    job_id = _done_mesh(svc, lora_weight=0.9)
    inspector.arm_verdict(ctx.state, job_id)

    assert inspector.record_verdict(ctx, job_id, "reject", ("holes",)) is True

    row = svc.store.latest_verdicts()[0]
    assert (row["job_id"], row["verdict"], row["reasons"]) == (job_id, "reject", ["holes"])
    assert row["vector"] == {"lora_weight": 0.9, "stage": "model"}
    assert ctx.state.inspector_reject_armed is None
    # The recompute reads every verdict and writes a file, so it is asked for
    # rather than done here -- through Review's own request, so the inspector
    # keeps no second spelling of Review's task key.
    assert ctx.state.findings_dirty is True
    assert ctx.submitted == []


def test_a_refused_verdict_toasts_and_writes_nothing(svc):
    ctx = FakeCtx(svc)
    assert inspector.record_verdict(ctx, "ffffffffffff", "accept") is False
    assert ctx.toasts[-1][1] == "error"
    assert svc.store.latest_verdicts() == []


def test_an_inspector_verdict_feeds_the_same_findings_a_sweep_does(svc):
    from warlock.service import findings as svc_findings

    ctx = FakeCtx(svc)
    for _ in range(5):
        inspector.record_verdict(ctx, _done_mesh(svc, platform="pc"), "accept")

    doc = svc_findings.aggregate(svc.store)
    assert doc["params"]["platform"]["pc"] == {
        "n": 5, "accepts": 5, "accept_rate": 1.0,
        "wilson_low": svc_findings.wilson_low(5, 5),
        "sources": {"human": {"accept": 5, "reject": 0}},
        "top_reasons": [],
    }
    assert len(svc_findings.presets(doc)) == 1
