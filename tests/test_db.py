from __future__ import annotations


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
