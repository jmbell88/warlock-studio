"""The observations table: append-only machine evidence, one row per finished
model job, deliberately outliving the job rows it was measured from."""

from __future__ import annotations

import pytest

from warlock.db import JobStore


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite")
    yield s
    s.close()


def _observe(store, job_id, *, sweep_id=None, sweep_unit="", seed=42, **metrics):
    return store.add_observation(
        job_id,
        sweep_id=sweep_id,
        sweep_unit=sweep_unit,
        seed=seed,
        prompt_hash="abc123abc123",
        vector={"lora_weight": 0.6, "stage": "model"},
        metrics=metrics,
    )


def test_an_observation_round_trips_its_json_columns(store):
    _observe(store, "job1", sweep_id="sw1", sweep_unit="baseline s42",
             hole_worst=0.034, watertight=True, triangles=24120)
    record = store.latest_observations()[0]
    assert record["job_id"] == "job1"
    assert record["sweep_id"] == "sw1"
    assert record["sweep_unit"] == "baseline s42"
    assert record["seed"] == 42
    assert record["prompt_hash"] == "abc123abc123"
    assert record["vector"] == {"lora_weight": 0.6, "stage": "model"}
    assert record["metrics"] == {
        "hole_worst": 0.034, "watertight": True, "triangles": 24120,
    }


def test_the_latest_observation_per_job_wins(store):
    """A re-audit (a retarget rewrites the mesh) may someday write a second
    row; the newest by id describes the artifact that exists now."""
    _observe(store, "job1", hole_worst=0.5)
    _observe(store, "job1", hole_worst=0.01)
    _observe(store, "job2", hole_worst=0.2)

    records = {r["job_id"]: r for r in store.latest_observations()}
    assert len(records) == 2
    assert records["job1"]["metrics"] == {"hole_worst": 0.01}
    assert records["job2"]["metrics"] == {"hole_worst": 0.2}


def test_observations_outlive_the_job_and_the_sweep_rows(store):
    """The corpus argument: prune_jobs and delete_sweep remove job and sweep
    rows, and what was measured must survive both -- same rule as the verdict
    vector snapshot."""
    job_id = store.create("image", "a chest", {}, stage="model", status="done",
                          sweep_id="sw1", sweep_unit="baseline s42")
    sweep_id = store.create_sweep("s", "a chest", {})
    _observe(store, job_id, sweep_id=sweep_id, hole_worst=0.02)

    store.delete(job_id)
    store.delete_sweep(sweep_id)

    records = store.latest_observations()
    assert len(records) == 1
    assert records[0]["job_id"] == job_id
    assert records[0]["metrics"] == {"hole_worst": 0.02}
