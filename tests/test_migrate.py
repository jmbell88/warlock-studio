from __future__ import annotations

import sqlite3

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
