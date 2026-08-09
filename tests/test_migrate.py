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


def test_a_db_without_metadata_columns_gains_them(tmp_path):
    """The workshop metadata columns are additive: a DB that predates them must
    converge on the fresh shape with every existing row defaulted to blank."""
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
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

    store = JobStore(path)
    try:
        row = store.list(10)[0]
        assert row["name"] == ""
        assert row["tags"] == ""
        assert row["favorite"] == 0
    finally:
        store.close()


def test_a_v4_db_gains_verdict_context_and_the_observations_table(tmp_path):
    """Migration 5 is additive over a v4 database: the four context columns
    land on ``verdicts`` defaulted to "not part of a sweep", the observations
    table appears, and existing verdict rows survive untouched."""
    path = tmp_path / "v4.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            prompt TEXT, params TEXT NOT NULL DEFAULT '{}', error TEXT,
            created_at REAL NOT NULL, started_at REAL, finished_at REAL,
            stage TEXT NOT NULL DEFAULT 'model', parent_id TEXT,
            name TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',
            favorite INTEGER NOT NULL DEFAULT 0,
            sweep_id TEXT, sweep_unit TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE sweeps (
            id TEXT PRIMARY KEY, label TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '', spec TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        CREATE TABLE verdicts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
            source TEXT NOT NULL, verdict TEXT NOT NULL,
            reasons TEXT NOT NULL DEFAULT '[]', vector TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO verdicts (job_id, source, verdict, reasons, vector, created_at)"
        " VALUES ('aaaaaaaaaaaa', 'human', 'accept', '[]', '{\"platform\": \"pc\"}', 1.0)"
    )
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()

    store = JobStore(path)
    try:
        record = store.latest_verdicts()[0]
        assert record["vector"] == {"platform": "pc"}
        # Migration 10 rode along: a binary accept is now the mildest grade that
        # still means "usable", and the derived verdict word is untouched.
        assert record["grade"] == 3
        assert record["verdict"] == "accept"
        assert record["sweep_id"] is None
        assert record["sweep_unit"] == ""
        assert record["seed"] is None
        assert record["prompt_hash"] == ""
        # The new table is usable, not merely present.
        store.add_observation(
            "aaaaaaaaaaaa", sweep_id=None, sweep_unit="", seed=7,
            prompt_hash="", vector={"platform": "pc"}, metrics={"hole_worst": 0.1},
        )
        assert store.latest_observations()[0]["metrics"] == {"hole_worst": 0.1}
    finally:
        store.close()

    fresh_path = tmp_path / "fresh.sqlite"
    fresh = JobStore(fresh_path)
    fresh.close()

    def shape(p):
        c = sqlite3.connect(p)
        info = {
            table: c.execute(f"PRAGMA table_info({table})").fetchall()
            for table in ("jobs", "sweeps", "verdicts", "observations")
        }
        version = c.execute("PRAGMA user_version").fetchone()[0]
        indexes = {
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
                " AND name LIKE 'idx_observations%'"
            )
        }
        c.close()
        return info, version, indexes

    migrated_shape, migrated_version, migrated_indexes = shape(path)
    fresh_shape, fresh_version, fresh_indexes = shape(fresh_path)
    assert migrated_shape == fresh_shape
    assert migrated_version == fresh_version == len(MIGRATIONS)
    # (job_id) and (job_id, id): the second serves latest_observations'
    # MAX(id) GROUP BY job_id as an index-only scan (migration 8). There is
    # still deliberately no index on sweep_id -- nothing queries observations
    # by sweep (the grouping happens in findings._comparisons, in Python, over
    # the whole set), so it would be a B-tree maintained on every insert to
    # answer a question nobody asks.
    assert (
        migrated_indexes
        == fresh_indexes
        == {"idx_observations_job", "idx_observations_latest"}
    )


def test_a_v3_db_gains_the_sweep_columns_and_the_new_tables(tmp_path):
    """Migration 4 is additive over a v3 database: the two sweep columns land
    on ``jobs`` defaulted to "not part of a sweep", and the sweeps/verdicts
    tables appear."""
    path = tmp_path / "v3.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
            prompt TEXT, params TEXT NOT NULL DEFAULT '{}', error TEXT,
            created_at REAL NOT NULL, started_at REAL, finished_at REAL,
            stage TEXT NOT NULL DEFAULT 'model', parent_id TEXT,
            name TEXT NOT NULL DEFAULT '', tags TEXT NOT NULL DEFAULT '',
            favorite INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.execute(
        "INSERT INTO jobs (id, kind, status, params, created_at)"
        " VALUES ('aaaaaaaaaaaa', 'text', 'done', '{}', 1.0)"
    )
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    store = JobStore(path)
    try:
        row = store.list(10)[0]
        assert row["sweep_id"] is None
        assert row["sweep_unit"] == ""
        # The new tables are usable, not merely present.
        sweep_id = store.create_sweep("s", "p", {})
        assert [s["id"] for s in store.list_sweeps()] == [sweep_id]
        store.add_verdict("aaaaaaaaaaaa", source="human", verdict="accept",
                          reasons=[], vector={"platform": "pc"})
        assert store.latest_verdicts()[0]["vector"] == {"platform": "pc"}
    finally:
        store.close()

    fresh_path = tmp_path / "fresh.sqlite"
    fresh = JobStore(fresh_path)
    fresh.close()

    def shape(p):
        c = sqlite3.connect(p)
        info = {
            table: c.execute(f"PRAGMA table_info({table})").fetchall()
            for table in ("jobs", "sweeps", "verdicts")
        }
        version = c.execute("PRAGMA user_version").fetchone()[0]
        indexes = {r[1] for r in c.execute("PRAGMA index_list(jobs)")}
        c.close()
        return info, version, indexes

    migrated_shape, migrated_version, migrated_indexes = shape(path)
    fresh_shape, fresh_version, fresh_indexes = shape(fresh_path)
    assert migrated_shape == fresh_shape
    assert migrated_version == fresh_version == len(MIGRATIONS)
    assert "idx_jobs_sweep" in migrated_indexes
    assert "idx_jobs_sweep" in fresh_indexes


def _grades(store):
    return dict(store._conn.execute("SELECT id, grade FROM verdicts"))


def test_migration_10_backfills_only_the_mesh_rows_and_replays_idempotently(tmp_path):
    """The grade column, and the three properties its guards buy.

    A pre-migration database is v9 with three verdict rows: a mesh accept, a
    mesh reject and an image label. After migration the two mesh rows carry the
    backfill and the image row carries NULL -- permanently, because an image
    label feeds a binary probe and a grade would be thresholded straight back to
    a bit.

    The ``grade IS NULL`` guards are what make a *partial* replay safe: DDL runs
    in autocommit here, so an entry that failed part-way is re-run whole, and an
    unguarded UPDATE would overwrite a real grade recorded since. Replaying the
    statements by hand is the only way to exercise that -- the store will not
    re-run a migration it has already versioned past.
    """
    path = tmp_path / "v9.sqlite"
    store = JobStore(path)
    mesh_ok = store.add_verdict("aaaaaaaaaaaa", source="human", verdict="accept",
                                reasons=[], vector={}, stage="model")
    mesh_bad = store.add_verdict("bbbbbbbbbbbb", source="human", verdict="reject",
                                 reasons=["holes"], vector={}, stage="model")
    image = store.add_verdict("cccccccccccc", source="human", verdict="accept",
                              reasons=[], vector={}, stage="reference")
    # Rewind to the state migration 10 was written to find, then reopen.
    store._conn.execute("UPDATE verdicts SET grade = NULL")
    store._conn.execute("PRAGMA user_version = 9")
    store._conn.commit()
    store.close()

    store = JobStore(path)
    try:
        grades = _grades(store)
        assert grades[mesh_ok] == 3
        assert grades[mesh_bad] == -3
        assert grades[image] is None

        # A human regrades the accepted mesh to a +5 it never had...
        store._conn.execute("UPDATE verdicts SET grade = 5 WHERE id = ?", (mesh_ok,))
        store._conn.commit()
        # ...and a partial replay of the entry must not walk it back to +3.
        for stmt in MIGRATIONS[9][1:]:
            store._conn.execute(stmt)
        store._conn.commit()
        regraded = _grades(store)
        assert regraded[mesh_ok] == 5
        assert regraded[mesh_bad] == -3
        assert regraded[image] is None
    finally:
        store.close()
