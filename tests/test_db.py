from __future__ import annotations

from animancer3d.db import JobStore


def test_create_and_get(store):
    job_id = store.create("text", "a sword", {"seed": 7, "resolution": 1024})
    job = store.get(job_id)
    assert job is not None
    assert job["kind"] == "text"
    assert job["status"] == "queued"
    assert job["prompt"] == "a sword"
    assert job["params"] == {"seed": 7, "resolution": 1024}


def test_status_transitions(store):
    job_id = store.create("image", None, {})
    store.set_status(job_id, "running")
    assert store.get(job_id)["started_at"] is not None
    store.set_status(job_id, "done")
    job = store.get(job_id)
    assert job["status"] == "done"
    assert job["finished_at"] is not None


def test_next_queued_fifo(store):
    first = store.create("text", "a", {})
    store.create("text", "b", {})
    assert store.next_queued()["id"] == first
    store.set_status(first, "running")
    assert store.next_queued()["prompt"] == "b"


def test_error_records_message(store):
    job_id = store.create("image", None, {})
    store.set_status(job_id, "error", "boom")
    job = store.get(job_id)
    assert job["status"] == "error"
    assert job["error"] == "boom"


def test_delete(store):
    job_id = store.create("text", "x", {})
    store.delete(job_id)
    assert store.get(job_id) is None
    assert store.list() == []


def test_claim_succeeds_on_queued_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    assert store.claim(job_id) is True
    assert store.get(job_id)["status"] == "running"
    store.close()


def test_claim_fails_on_already_claimed_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    assert store.claim(job_id) is True
    assert store.claim(job_id) is False
    store.close()


def test_claim_fails_on_cancelled_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.set_status(job_id, "cancelled")
    assert store.claim(job_id) is False
    store.close()


def test_cancel_succeeds_on_queued_or_running_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    assert store.cancel(job_id) is True
    assert store.get(job_id)["status"] == "cancelled"
    store.close()


def test_cancel_fails_on_already_terminal_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.set_status(job_id, "done")
    assert store.cancel(job_id) is False
    assert store.get(job_id)["status"] == "done"
    store.close()


def test_reconcile_startup_marks_running_jobs_as_interrupted(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.claim(job_id)
    store.reconcile_startup()
    job = store.get(job_id)
    assert job["status"] == "error"
    assert job["error"] == "interrupted by shutdown"
    store.close()


def test_reconcile_startup_leaves_other_statuses_alone(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    queued_id = store.create("text", "x", {})
    store.reconcile_startup()
    assert store.get(queued_id)["status"] == "queued"
    store.close()


def test_set_status_preserves_error_when_not_overwritten(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.set_status(job_id, "error", "boom")
    store.set_status(job_id, "cancelled")
    assert store.get(job_id)["error"] == "boom"
    assert store.get(job_id)["status"] == "cancelled"
    store.close()


def test_create_accepts_explicit_job_id(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {}, job_id="myid123")
    assert job_id == "myid123"
    assert store.get("myid123") is not None
    store.close()


def test_create_generates_id_when_not_given(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    assert job_id
    assert store.get(job_id) is not None
    store.close()
