"""Phase 1 data-layer behaviours: WAL, the read-path indexes (migration 8),
the params-parse memo, and batched sweep-unit commits."""

from __future__ import annotations

import sqlite3

from warlock.db import MIGRATIONS, JobStore


def _index_sql(path, name):
    conn = sqlite3.connect(path)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else ""


def test_the_store_opens_in_wal_mode_with_normal_durability(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    try:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        sync = store._conn.execute("PRAGMA synchronous").fetchone()[0]
        assert mode == "wal"
        assert sync == 1  # NORMAL
    finally:
        store.close()


def test_migration_8_indexes_exist_on_fresh_and_pre_existing_dbs(tmp_path):
    """The new read-path indexes land on a fresh DB and on one migrated from
    an older shape, and the sweep index really is partial."""
    fresh = tmp_path / "fresh.sqlite"
    JobStore(fresh).close()

    old = tmp_path / "old.sqlite"
    conn = sqlite3.connect(old)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            prompt TEXT, params TEXT NOT NULL DEFAULT '{}', error TEXT,
            created_at REAL NOT NULL, started_at REAL, finished_at REAL
        );
        """
    )
    conn.commit()
    conn.close()
    JobStore(old).close()

    for path in (fresh, old):
        conn = sqlite3.connect(path)
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        conn.close()
        assert {
            "idx_verdicts_latest",
            "idx_verdicts_source",
            "idx_observations_latest",
            "idx_jobs_dispatch",
            "idx_jobs_sweep",
        } <= names
        # Partial: list_sweeps' tally only ever looks at sweep rows, and
        # almost every jobs row has sweep_id NULL.
        assert "WHERE sweep_id IS NOT NULL" in _index_sql(path, "idx_jobs_sweep")


def test_migration_9_is_the_last_entry(tmp_path):
    """Append-only bookkeeping: user_version lands on len(MIGRATIONS), and the
    index entry is number 9 (the trash's ``deleted_at``) -- anything planned as
    'migration 9' elsewhere is now 10."""
    assert len(MIGRATIONS) == 9
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    store.close()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 9
    conn.close()


def test_list_memoizes_the_params_parse_but_hands_out_independent_dicts(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    try:
        a = store.create("text", "one", {"seed": 1, "resolution": 1024})
        store.create("text", "two", {"seed": 1, "resolution": 1024})
        first = store.list(10)
        second = store.list(10)
        assert first[0]["params"] == second[0]["params"]
        # Top-level dicts are copies: a caller replacing a key must not edit
        # what the next tick hands somebody else.
        first[0]["params"]["seed"] = 999
        assert store.list(10)[0]["params"]["seed"] == 1
        # A rewritten blob is re-parsed, not served from the memo.
        store.set_params(a, {"seed": 7})
        by_id = {j["id"]: j for j in store.list(10)}
        assert by_id[a]["params"] == {"seed": 7}
    finally:
        store.close()


def test_deferred_commits_batches_creates_and_always_commits(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    try:
        with store.deferred_commits():
            for i in range(3):
                store.create("text", f"unit {i}", {})
        # Visible from a second connection only if the batch committed.
        other = sqlite3.connect(path)
        count = other.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        other.close()
        assert count == 3
        # And the finally-commit runs even when the block raises, so the
        # rollback path can read the rows back to delete them.
        try:
            with store.deferred_commits():
                store.create("text", "doomed", {})
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        other = sqlite3.connect(path)
        count = other.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        other.close()
        assert count == 4
    finally:
        store.close()


def test_unverdicted_and_unlabelled_agree_with_the_not_in_spelling(tmp_path):
    """The NOT EXISTS rewrite answers exactly what NOT IN did."""
    store = JobStore(tmp_path / "jobs.sqlite")
    try:
        judged = store.create("text", "judged", {}, status="done")
        waiting = store.create("text", "waiting", {}, status="done")
        refused = store.create("text", "refused", {}, status="error", stage="reference")
        store.add_verdict(
            judged, source="human", verdict="accept", reasons=[], vector={}
        )
        assert [j["id"] for j in store.unverdicted_models()] == [waiting]
        # The reference-stage listing includes the errored row and excludes
        # nothing for a *model*-stage verdict on it.
        store.add_verdict(
            refused, source="human", verdict="reject", reasons=[], vector={},
            stage="model",
        )
        listed = {j["id"] for j in store.unlabelled_references(stage="reference")}
        assert refused in listed
        store.add_verdict(
            refused, source="human", verdict="reject", reasons=[], vector={},
            stage="reference",
        )
        listed = {j["id"] for j in store.unlabelled_references(stage="reference")}
        assert refused not in listed
    finally:
        store.close()
