from __future__ import annotations

from warlock.db import JobStore


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


def test_cancel_succeeds_on_running_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.claim(job_id)
    assert store.cancel(job_id) is True
    assert store.get(job_id)["status"] == "cancelled"
    store.close()


def test_finish_succeeds_on_running_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.claim(job_id)
    assert store.finish(job_id, "done") is True
    assert store.get(job_id)["status"] == "done"
    store.close()


def test_finish_fails_if_job_was_cancelled_underneath(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.claim(job_id)
    store.cancel(job_id)
    assert store.finish(job_id, "done") is False
    assert store.get(job_id)["status"] == "cancelled"
    store.close()


def test_finish_records_error_message(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job_id = store.create("text", "x", {})
    store.claim(job_id)
    assert store.finish(job_id, "error", "boom") is True
    job = store.get(job_id)
    assert job["status"] == "error"
    assert job["error"] == "boom"
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


def test_create_records_stage_and_parent(store):
    parent = store.create("text", "a barrel", {}, stage="reference")
    child = store.create("text", "a barrel", {}, parent_id=parent)
    assert store.get(parent)["stage"] == "reference"
    assert [c["id"] for c in store.children(parent)] == [child]


def test_create_defaults_stage_to_model(store):
    job_id = store.create("text", "x", {})
    assert store.get(job_id)["stage"] == "model"
    assert store.get(job_id)["parent_id"] is None


def test_set_stage(store):
    job_id = store.create("text", "x", {})
    store.set_stage(job_id, "reference")
    assert store.get(job_id)["stage"] == "reference"


def test_children_ordered_oldest_first(store):
    parent = store.create("text", "x", {})
    first = store.create("text", "x", {}, parent_id=parent)
    second = store.create("text", "x", {}, parent_id=parent)
    assert [c["id"] for c in store.children(parent)] == [first, second]


def test_set_meta_updates_name_tags_and_favorite(store):
    job_id = store.create("text", "a barrel", {})
    assert store.set_meta(job_id, name="Oak barrel", tags="prop,fantasy", favorite=True)
    row = store.get(job_id)
    assert row["name"] == "Oak barrel"
    assert row["tags"] == "prop,fantasy"
    assert row["favorite"] == 1


def test_set_meta_leaves_unspecified_fields_alone(store):
    job_id = store.create("text", "a barrel", {})
    store.set_meta(job_id, name="Oak barrel")
    store.set_meta(job_id, favorite=True)
    row = store.get(job_id)
    assert row["name"] == "Oak barrel"
    assert row["favorite"] == 1


def test_set_meta_on_a_missing_job_is_false(store):
    assert store.set_meta("0" * 12, name="x") is False


def test_new_jobs_default_to_empty_metadata(store):
    row = store.get(store.create("text", "a barrel", {}))
    assert row["name"] == ""
    assert row["tags"] == ""
    assert row["favorite"] == 0


def test_merge_params_keeps_keys_it_was_not_asked_about(store):
    """The reason this exists at all: params is one JSON blob, so a writer
    holding a copy read earlier and calling set_params destroys whatever
    landed in between. Merging under the lock is what makes two writers on
    one row (the worker recording derived values, POST /optimize rewriting a
    budget) not a lost update."""
    job_id = store.create("text", "a barrel", {"seed": 7, "mesh_report": {"status": "ready"}})
    written = store.merge_params(job_id, {"profile": "raw"}, remove=("mesh_report",))
    assert written == {"seed": 7, "profile": "raw"}
    assert store.get(job_id)["params"] == {"seed": 7, "profile": "raw"}


def test_merge_params_on_a_missing_job_is_none(store):
    assert store.merge_params("0" * 12, {"profile": "raw"}) is None


def test_created_at_is_indexed(store):
    """list() and next_queued() both sort on it, and next_queued runs on every
    dispatch tick. Asserted through the schema rather than a timing, which is
    the only thing that stays true on an empty test database."""
    names = {r[1] for r in store._conn.execute("PRAGMA index_list(jobs)")}
    assert "idx_jobs_created" in names


# --- keyset pagination -------------------------------------------------------


def test_list_pages_through_the_whole_history(store):
    ids = [store.create("text", f"j{i}", {}) for i in range(25)]
    seen = []
    cursor = None
    while True:
        page = store.list(10, cursor)
        if not page:
            break
        seen += [j["id"] for j in page]
        cursor = (page[-1]["created_at"], page[-1]["id"])
    # Every job exactly once, newest first.
    assert seen == list(reversed(ids))


def test_list_breaks_created_at_ties_by_id(store):
    """time.time() genuinely ties across rows created in quick succession, so
    without the id tiebreak a cursor would either skip or repeat rows."""
    ids = [store.create("text", f"j{i}", {}) for i in range(6)]
    store._conn.execute("UPDATE jobs SET created_at = 100.0")
    store._conn.commit()

    seen = []
    cursor = None
    while True:
        page = store.list(2, cursor)
        if not page:
            break
        seen += [j["id"] for j in page]
        cursor = (page[-1]["created_at"], page[-1]["id"])
    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen)) == 6


def test_list_with_a_cursor_past_the_end_is_empty(store):
    store.create("text", "j", {})
    assert store.list(10, (0.0, "0" * 12)) == []


def test_count_reports_every_job_not_just_a_page(store):
    assert store.count() == 0
    for i in range(7):
        store.create("text", f"j{i}", {})
    assert store.count() == 7
    assert len(store.list(3)) == 3


def test_a_batch_of_jobs_dispatches_in_a_deterministic_order(store):
    """A sweep submits N rows in one loop, so ``time.time()`` genuinely ties
    across them. Correctness never depended on the order -- the worker restarts
    trellis-server whenever the running config does not match the job in hand --
    but how many restarts a sweep costs does."""
    ids = [store.create("text", "a chest", {}, f"{i:012x}") for i in range(6)]
    for job_id in ids:
        store._conn.execute(
            "UPDATE jobs SET created_at = 100.0 WHERE id = ?", (job_id,)
        )
    store._conn.commit()

    seen = []
    while (job := store.next_queued()) is not None:
        seen.append(job["id"])
        store.claim(job["id"])
    assert seen == sorted(ids)
