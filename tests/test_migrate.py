from __future__ import annotations

import sqlite3

import warlock.db as db_mod
from warlock.db import _SCHEMA, MIGRATIONS, JobStore


def test_fresh_db_lands_on_the_latest_migration_version(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    store.close()
    conn = sqlite3.connect(path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert version == len(MIGRATIONS)


def test_hand_built_v0_db_migrates_and_keeps_its_row(tmp_path):
    path = tmp_path / "jobs.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT INTO jobs (id, kind, status, prompt, params, created_at) VALUES (?,?,?,?,?,?)",
        ("preexisting", "text", "done", "old job", "{}", 0.0),
    )
    conn.commit()
    conn.close()

    store = JobStore(path)
    job = store.get("preexisting")
    store.close()

    assert job is not None
    assert job["status"] == "done"
    assert job["prompt"] == "old job"


def test_fresh_and_migrated_dbs_have_identical_schema(tmp_path):
    fresh = JobStore(tmp_path / "fresh.sqlite")
    fresh.close()

    hand_built_path = tmp_path / "handbuilt.sqlite"
    conn = sqlite3.connect(hand_built_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    migrated = JobStore(hand_built_path)
    migrated.close()

    def table_info(path):
        c = sqlite3.connect(path)
        info = c.execute("PRAGMA table_info(jobs)").fetchall()
        c.close()
        return info

    assert table_info(tmp_path / "fresh.sqlite") == table_info(hand_built_path)


def test_pre_migration_db_gains_stage_and_parent(tmp_path):
    """A DB created before the columns existed must converge on the same shape
    as a fresh one, with existing rows defaulted to stage='model'."""
    old_path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(old_path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            prompt TEXT, params TEXT NOT NULL DEFAULT '{}', error TEXT,
            created_at REAL NOT NULL, started_at REAL, finished_at REAL
        );
        """
    )
    conn.execute(
        "INSERT INTO jobs (id, kind, status, params, created_at)"
        " VALUES ('aaaaaaaaaaaa', 'text', 'done', '{}', 1.0)"
    )
    conn.commit()
    conn.close()

    store = JobStore(old_path)
    try:
        row = store.list(10)[0]
        assert row["stage"] == "model"
        assert row["parent_id"] is None
    finally:
        store.close()

    fresh_path = tmp_path / "fresh.sqlite"
    fresh = JobStore(fresh_path)
    fresh.close()

    def table_info(path):
        c = sqlite3.connect(path)
        info = c.execute("PRAGMA table_info(jobs)").fetchall()
        c.close()
        return info

    def user_version(path):
        c = sqlite3.connect(path)
        v = c.execute("PRAGMA user_version").fetchone()[0]
        c.close()
        return v

    def index_names(path):
        c = sqlite3.connect(path)
        names = {r[1] for r in c.execute("PRAGMA index_list(jobs)")}
        c.close()
        return names

    # Full column tuples (name, type, notnull, default, pk), not just names --
    # a type or default drift between the two paths would otherwise pass.
    assert table_info(old_path) == table_info(fresh_path)
    assert user_version(old_path) == len(MIGRATIONS)
    assert user_version(fresh_path) == len(MIGRATIONS)
    assert "idx_jobs_parent" in index_names(old_path)
    assert "idx_jobs_parent" in index_names(fresh_path)


def test_add_column_guard_skips_the_column_less_add_spelling_on_a_fresh_db(
    tmp_path, monkeypatch
):
    """SQLite's ALTER TABLE ADD does not require the COLUMN keyword. A
    migration written that way must still be skipped on a fresh DB, exactly
    like the canonical "ADD COLUMN" spelling -- otherwise a fresh database
    fails to start the moment a future migration uses this legal alternative.

    "status" is a column _SCHEMA already creates, so a fresh DB replaying
    this statement verbatim (i.e. the hardening reverted to a bare
    ``.startswith("ALTER TABLE jobs ADD COLUMN")`` check) raises
    sqlite3.OperationalError: duplicate column name.
    """
    fake_migrations = [
        [*MIGRATIONS[0]],
        ["ALTER TABLE jobs ADD status TEXT NOT NULL DEFAULT 'queued'"],
    ]
    monkeypatch.setattr(db_mod, "MIGRATIONS", fake_migrations)

    path = tmp_path / "jobs.sqlite"
    store = JobStore(path)
    try:
        conn = sqlite3.connect(path)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        assert version == len(fake_migrations)
    finally:
        store.close()
