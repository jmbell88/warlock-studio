"""The verdict and sweep tables: latest-wins, sources coexisting, and the one
property the denormalized vector exists for -- surviving the job it names."""

from __future__ import annotations

import pytest

from warlock.service import findings as svc_findings
from warlock.service import jobs as svc_jobs
from warlock.service import verdicts as svc_verdicts
from warlock.service.errors import Invalid, NotFound


def _model_job(store, **kwargs):
    return store.create("image", "a chest", {"lora_weight": 0.9}, stage="model", **kwargs)


def test_latest_verdict_per_job_and_source_wins(store):
    job_id = _model_job(store)
    store.add_verdict(job_id, source="human", verdict="reject", reasons=["holes"], vector={})
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})

    latest = store.latest_verdicts()
    assert len(latest) == 1
    assert latest[0]["verdict"] == "accept"


def test_a_human_and_an_ai_verdict_sit_side_by_side(store):
    job_id = _model_job(store)
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})
    store.add_verdict(job_id, source="ai:demo", verdict="reject", reasons=["broken"], vector={})

    latest = {r["source"]: r for r in store.latest_verdicts()}
    assert latest["human"]["verdict"] == "accept"
    assert latest["ai:demo"]["verdict"] == "reject"

    for_job = store.verdicts_for([job_id])
    assert set(for_job) == {(job_id, "human"), (job_id, "ai:demo")}
    only_human = store.verdicts_for([job_id], source="human")
    assert set(only_human) == {(job_id, "human")}


def test_a_verdicts_vector_outlives_the_job_row(store):
    job_id = _model_job(store)
    store.add_verdict(
        job_id, source="human", verdict="accept", reasons=[], vector={"lora_weight": 0.9}
    )
    store.delete(job_id)

    # This is the whole reason the vector is denormalized: prune_jobs deletes
    # rows, and the corpus has to survive it.
    assert store.get(job_id) is None
    latest = store.latest_verdicts()
    assert latest[0]["vector"] == {"lora_weight": 0.9}


def test_unverdicted_models_skips_sweep_units_and_judged_jobs(store):
    plain = _model_job(store)
    judged = _model_job(store)
    unit = _model_job(store, sweep_id="deadbeefcafe", sweep_unit="lora_weight=0.6 s42")
    store.create("text", "a picture", {}, stage="reference")
    store.add_verdict(judged, source="human", verdict="accept", reasons=[], vector={})
    for job_id in (plain, judged, unit):
        store.set_status(job_id, "done")

    ids = [job["id"] for job in store.unverdicted_models()]
    assert ids == [plain]


def test_unverdicted_models_is_per_source(store):
    job_id = _model_job(store)
    store.set_status(job_id, "done")
    store.add_verdict(job_id, source="human", verdict="accept", reasons=[], vector={})

    assert [j["id"] for j in store.unverdicted_models(source="human")] == []
    assert [j["id"] for j in store.unverdicted_models(source="ai:demo")] == [job_id]


def test_sweep_rows_carry_their_unit_counts(store):
    sweep_id = store.create_sweep("lora weight", "a chest", {"axes": []})
    running = store.create("text", "a chest", {}, sweep_id=sweep_id, sweep_unit="baseline s1")
    finished = store.create("text", "a chest", {}, sweep_id=sweep_id, sweep_unit="baseline s2")
    store.set_status(finished, "done")
    store.create("text", "unrelated", {})

    listed = store.list_sweeps()
    assert len(listed) == 1
    assert listed[0]["id"] == sweep_id
    assert listed[0]["label"] == "lora weight"
    assert listed[0]["spec"] == {"axes": []}
    assert (listed[0]["units"], listed[0]["done"], listed[0]["todo"]) == (2, 1, 1)
    assert [j["id"] for j in store.sweep_jobs(sweep_id)] == [running, finished]


def test_deleting_a_sweep_leaves_its_jobs_to_the_caller(store):
    sweep_id = store.create_sweep("s", "p", {})
    job_id = store.create("text", "a chest", {}, sweep_id=sweep_id, sweep_unit="u")
    store.delete_sweep(sweep_id)

    assert store.list_sweeps() == []
    assert store.get(job_id) is not None


def test_next_queued_breaks_created_at_ties_on_id(store):
    """A sweep submits N rows in one loop, so time.time() genuinely ties."""
    ids = sorted(store.create("text", "a chest", {}, f"{i:012x}") for i in range(5))
    for row in store.list(10):
        store._conn.execute(
            "UPDATE jobs SET created_at = 100.0 WHERE id = ?", (row["id"],)
        )
    store._conn.commit()

    assert store.next_queued()["id"] == ids[0]


def test_record_verdict_snapshots_the_jobs_vector(svc):
    job_id = svc.store.create(
        "image", "a chest", {"lora_weight": 0.9, "seed": 12, "composed_prompt": "x"},
        stage="model", status="done",
    )
    svc_verdicts.record_verdict(svc, job_id, verdict="reject", reasons=["holes"])

    row = svc.store.latest_verdicts()[0]
    assert row["verdict"] == "reject"
    assert row["reasons"] == ["holes"]
    # The allowlist, not "everything in params": no seed, no derived value.
    assert row["vector"] == {"lora_weight": 0.9, "stage": "model"}


def test_record_verdict_refuses_unknown_verdicts_reasons_and_jobs(svc):
    job_id = svc.store.create("image", "a chest", {}, stage="model", status="done")
    with pytest.raises(Invalid):
        svc_verdicts.record_verdict(svc, job_id, verdict="maybe")
    with pytest.raises(Invalid):
        svc_verdicts.record_verdict(svc, job_id, verdict="reject", reasons=["ugly"])
    with pytest.raises(NotFound):
        svc_verdicts.record_verdict(svc, "ffffffffffff", verdict="accept")
    assert svc.store.latest_verdicts() == []


def test_a_verdict_on_a_pruned_job_still_feeds_the_findings(svc):
    job_id = svc.store.create(
        "image", "a chest", {"lora_weight": 0.9}, stage="model", status="done"
    )
    svc_verdicts.record_verdict(svc, job_id, verdict="accept")
    svc_jobs.prune_jobs(svc, keep=0)

    doc = svc_findings.aggregate(svc.store)
    assert doc["params"]["lora_weight"]["0.9"]["accepts"] == 1
