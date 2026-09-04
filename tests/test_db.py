from __future__ import annotations

import ast
import inspect

import warlock.db as db_mod
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
    assert store.get(child)["parent_id"] == parent


def test_create_defaults_stage_to_model(store):
    job_id = store.create("text", "x", {})
    assert store.get(job_id)["stage"] == "model"
    assert store.get(job_id)["parent_id"] is None


def test_set_stage(store):
    job_id = store.create("text", "x", {})
    store.set_stage(job_id, "reference")
    assert store.get(job_id)["stage"] == "reference"


def test_a_child_records_its_parent(store):
    parent = store.create("text", "x", {})
    first = store.create("text", "x", {}, parent_id=parent)
    second = store.create("text", "x", {}, parent_id=parent)
    assert {store.get(j)["parent_id"] for j in (first, second)} == {parent}


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


def test_malformed_params_are_tolerated_by_get_list_and_dispatch(store):
    job_id = store.create("text", "a barrel", {"seed": 7})
    store._conn.execute("UPDATE jobs SET params = ? WHERE id = ?", ("{broken", job_id))
    store._conn.commit()

    assert store.get(job_id)["params"] == {}
    assert store.list()[0]["params"] == {}
    assert store.next_queued()["params"] == {}


def test_merge_params_repairs_a_non_object_params_blob(store):
    job_id = store.create("text", "a barrel", {})
    store._conn.execute("UPDATE jobs SET params = ? WHERE id = ?", ("[1, 2]", job_id))
    store._conn.commit()

    assert store.merge_params(job_id, {"profile": "raw"}) == {"profile": "raw"}
    assert store.get(job_id)["params"] == {"profile": "raw"}


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


def test_merge_param_entry_does_not_lose_a_concurrent_sibling_entry(store):
    """The nested read-modify-write, under one hold.

    ``merge_params`` makes a top-level key safe; this makes a key *inside* one
    safe, and the difference is what ``followups.persist`` used to get wrong: it
    read the row, built the new nested dict from what it read, then wrote the
    whole dict back through ``merge_params`` -- two separate holds, so two
    callers adding two different entries could each read the same dict and the
    second write would drop the first.

    Interleaved by hand rather than with threads, because the defect is not a
    timing accident: it is that the read and the write are separable at all, and
    reading both copies before either writes is exactly what a thread schedule
    is allowed to do.
    """
    job_id = store.create("text", "a barrel", {})

    store.merge_param_entry(job_id, "followup_failures", "rig", {"why": "no bpy"})
    store.merge_param_entry(job_id, "followup_failures", "sheet", {"why": "no vram"})

    failures = store.get(job_id)["params"]["followup_failures"]
    assert failures == {"rig": {"why": "no bpy"}, "sheet": {"why": "no vram"}}

    # An existing entry is replaced, not duplicated, and its siblings stay.
    store.merge_param_entry(job_id, "followup_failures", "rig", {"why": "retried"})
    failures = store.get(job_id)["params"]["followup_failures"]
    assert failures["rig"] == {"why": "retried"}
    assert failures["sheet"] == {"why": "no vram"}

    # A key that is not a dict yet -- or is something else entirely -- is
    # replaced by one rather than raising.
    store.merge_params(job_id, {"notes": "a string"})
    assert store.merge_param_entry(job_id, "notes", "a", 1) is not None
    assert store.get(job_id)["params"]["notes"] == {"a": 1}

    assert store.merge_param_entry("nope", "followup_failures", "rig", {}) is None


# Methods that touch ``self._conn`` deliberately outside ``self._lock``, and
# why. Anything not listed here must guard every touch -- see the class
# docstring at db.py:467.
_LOCK_EXEMPT = {
    # Takes a *separate* connection through sqlite3's own online-backup
    # locking specifically so a page walk to disk never holds the store lock
    # across blocking I/O -- see its own docstring at db.py:519.
    "backup_to",
}


def test_every_public_jobstore_method_guards_self_conn_with_self_lock():
    """Every public ``JobStore`` method that reaches ``self._conn`` does so only
    inside a ``with self._lock:`` block, or is named in ``_LOCK_EXEMPT``.

    The class docstring states this as the whole reason the lock exists --
    ``check_same_thread=False`` disables sqlite3's own guard, so the lock is
    the only thing left serialising writes from the executor pool a request
    reaches this store through. A new method that reads or writes ``self._conn``
    without taking the lock first would reintroduce exactly the race that
    disabling ``check_same_thread`` opened up, silently, and nothing else in
    the suite would catch it. Walked with ``ast`` rather than asserted by hand
    per method so a future method is covered automatically.
    """
    source = inspect.getsource(db_mod)
    tree = ast.parse(source)
    (class_node,) = [
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "JobStore"
    ]

    def is_self_lock(expr: ast.expr) -> bool:
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr == "_lock"
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "self"
        )

    def is_self_conn(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "_conn"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        )

    def guarded(n: ast.AST, parents: dict[ast.AST, ast.AST], stop: ast.AST) -> bool:
        cur = parents.get(n)
        while cur is not None and cur is not stop:
            if isinstance(cur, ast.With) and any(
                is_self_lock(item.context_expr) for item in cur.items
            ):
                return True
            cur = parents.get(cur)
        return False

    violations = []
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if name.startswith("_") or name in _LOCK_EXEMPT:
            continue

        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(node):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for n in ast.walk(node):
            if is_self_conn(n) and not guarded(n, parents, node):
                violations.append(f"{name} (line {n.lineno})")

    assert violations == [], (
        "JobStore methods touching self._conn outside self._lock: "
        f"{violations} -- add the guard, or allow-list with a reason in _LOCK_EXEMPT"
    )
