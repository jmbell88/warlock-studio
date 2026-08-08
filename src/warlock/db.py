"""SQLite job store. Synchronous sqlite3 behind asyncio.to_thread — job volume is tiny."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,              -- 'text' | 'image' | 'rig' | 'sheet'
    status      TEXT NOT NULL,              -- queued | running | done | error | cancelled
    prompt      TEXT,
    params      TEXT NOT NULL DEFAULT '{}', -- JSON: seed, resolution, ...
    error       TEXT,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    stage       TEXT NOT NULL DEFAULT 'model',  -- 'reference' | 'tile' | 'model'
    parent_id   TEXT,                           -- the reference job this was promoted from
    name        TEXT NOT NULL DEFAULT '',       -- user-given title; the prompt is the fallback
    tags        TEXT NOT NULL DEFAULT '',       -- comma-separated, normalized lowercase
    favorite    INTEGER NOT NULL DEFAULT 0,
    sweep_id    TEXT,                           -- NULL for an ordinary job
    sweep_unit  TEXT NOT NULL DEFAULT '',       -- display label, e.g. "lora_weight=0.6 s42"
    candidate_group TEXT,                       -- NULL once decided, and for an ordinary job
    candidate_index INTEGER NOT NULL DEFAULT 0  -- which candidate of the group this was
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS sweeps (
    id          TEXT PRIMARY KEY,
    label       TEXT NOT NULL DEFAULT '',
    prompt      TEXT NOT NULL DEFAULT '',
    spec        TEXT NOT NULL DEFAULT '{}',     -- JSON: base vector, axes, vectors, seeds, stage
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS verdicts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    source      TEXT NOT NULL,                  -- 'human' | 'ai:<model>'
    verdict     TEXT NOT NULL,                  -- accept | reject
    reasons     TEXT NOT NULL DEFAULT '[]',     -- JSON list
    vector      TEXT NOT NULL DEFAULT '{}',     -- JSON: the config vector, denormalized
    created_at  REAL NOT NULL,
    sweep_id    TEXT,                           -- denormalized: pairs outlive delete_sweep
    sweep_unit  TEXT NOT NULL DEFAULT '',       -- display label only; pairing never parses it
    seed        INTEGER,                        -- params["seed"] at record time
    prompt_hash TEXT NOT NULL DEFAULT '',       -- sha1[:12]; counts distinct prompts, nothing else
    stage       TEXT NOT NULL DEFAULT 'model'   -- 'reference' | 'blank' | 'model': which question
);
CREATE INDEX IF NOT EXISTS idx_verdicts_job ON verdicts(job_id);

CREATE TABLE IF NOT EXISTS observations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    sweep_id    TEXT,                           -- NULL for an ordinary job
    sweep_unit  TEXT NOT NULL DEFAULT '',
    seed        INTEGER,
    prompt_hash TEXT NOT NULL DEFAULT '',
    vector      TEXT NOT NULL DEFAULT '{}',     -- JSON: config vector, verdicts' canonical form
    metrics     TEXT NOT NULL DEFAULT '{}',     -- JSON: vectors.observation_metrics output
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observations_job ON observations(job_id);
"""
# No index on observations(sweep_id), deliberately: the one read of this table
# (latest_observations) has no WHERE at all -- it groups by job_id and the
# sweep grouping happens in findings._comparisons, in Python, over the whole
# set. An index nothing can use is a B-tree maintained on every insert to
# answer a query nobody asks.
# idx_jobs_parent is created by the migration below, not here: _SCHEMA's
# CREATE TABLE IF NOT EXISTS is a no-op against a pre-existing table that
# predates the parent_id column, and an index on a column that doesn't exist
# yet would fail before _migrate ever runs.

# Append-only. Each entry is a list of SQL statements, applied in order and
# followed by bumping PRAGMA user_version. Never edit an entry once it has
# shipped — only append. A fresh DB gets _SCHEMA then replays every entry here,
# so fresh and pre-existing DBs converge on the same shape.
#
# Not one transaction, whatever the ordering suggests: Python 3.12's sqlite3
# runs DDL and PRAGMA in autocommit, so an entry that fails part-way leaves the
# statements before it applied and the version unbumped, and the next open
# replays the whole entry. What makes that safe is the per-statement ADD COLUMN
# guard below, not atomicity.
MIGRATIONS: list[list[str]] = [
    # 1 -- approve-reference-first. A row that predates the split was a
    # single-stage generate, which is what stage='model' means.
    [
        "ALTER TABLE jobs ADD COLUMN stage TEXT NOT NULL DEFAULT 'model'",
        "ALTER TABLE jobs ADD COLUMN parent_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_id)",
    ],
    # 2 -- workshop metadata. A job's identity used to be its raw prompt
    # forever. Columns rather than params keys because these are what the list
    # filters and sorts on, and params is a JSON string sqlite cannot index
    # into.
    [
        "ALTER TABLE jobs ADD COLUMN name TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN tags TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE jobs ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_jobs_favorite ON jobs(favorite)",
    ],
    # 3 -- created_at index. Both hot reads sort by it: list() pages the whole
    # history newest-first and next_queued() runs on every dispatch tick, and
    # without an index each is a full scan plus a sort of every row ever made.
    [
        "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)",
    ],
    # 4 -- in-app sweeps and verdicts. Sweep membership is *columns* rather
    # than params keys for the reason the workshop metadata is: the list filters
    # on it, and params is a JSON string sqlite cannot index into. It also
    # sidesteps DERIVED_PARAMS entirely -- rerun_job and promote_to_model copy
    # params and nothing else, so membership can never leak onto a reroll.
    #
    # The two new tables are also in _SCHEMA (executed on every open), so
    # replaying them here is a no-op; they are repeated for the append-only
    # contract's sake, so a reader of this list sees the whole shape of the
    # change. The index on sweep_id is *not* in _SCHEMA, exactly as
    # idx_jobs_parent is not: CREATE TABLE IF NOT EXISTS is a no-op against a
    # pre-existing table, so the column may not exist when _SCHEMA runs.
    [
        "ALTER TABLE jobs ADD COLUMN sweep_id TEXT",
        "ALTER TABLE jobs ADD COLUMN sweep_unit TEXT NOT NULL DEFAULT ''",
        "CREATE INDEX IF NOT EXISTS idx_jobs_sweep ON jobs(sweep_id)",
        "CREATE TABLE IF NOT EXISTS sweeps ("
        " id TEXT PRIMARY KEY,"
        " label TEXT NOT NULL DEFAULT '',"
        " prompt TEXT NOT NULL DEFAULT '',"
        " spec TEXT NOT NULL DEFAULT '{}',"
        " created_at REAL NOT NULL)",
        "CREATE TABLE IF NOT EXISTS verdicts ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " job_id TEXT NOT NULL,"
        " source TEXT NOT NULL,"
        " verdict TEXT NOT NULL,"
        " reasons TEXT NOT NULL DEFAULT '[]',"
        " vector TEXT NOT NULL DEFAULT '{}',"
        " created_at REAL NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_verdicts_job ON verdicts(job_id)",
    ],
    # 5 -- machine evidence and matched pairs. Verdicts gain a denormalized
    # sweep context (sweep_id/sweep_unit/seed/prompt_hash) so findings can pair
    # a baseline verdict against an axis verdict -- same prompt, same seed, one
    # param differing -- *after* delete_sweep has removed the job rows, which
    # is the designed cleanup path. Observations are the verdict table's
    # pattern applied to what the worker measures on every finished model job
    # (hole fraction, watertight, triangles): one append-only row per
    # generation, carrying its own vector snapshot, deleted by nothing.
    #
    # The observations table is also in _SCHEMA (executed on every open), so
    # replaying it here is a no-op; it is repeated for the append-only
    # contract's sake. Its index can live in both places -- unlike
    # idx_jobs_sweep, the table either pre-exists with all its columns or was
    # just created whole by _SCHEMA. The verdict ALTERs are the statements the
    # _ADD_COLUMN_RE guard exists for: on a fresh DB the columns came from
    # _SCHEMA and the replay must skip them.
    #
    # sweep_unit is carried on both tables and read by nothing, on purpose: it
    # is the forensic half of the denormalization. Pairing is computed from the
    # vectors and must never parse a label (findings._one_key_diff), but after
    # delete_sweep has taken the job rows, the label is the only thing left
    # that says which unit of which sweep a surviving row came from -- for
    # somebody reading the table, not for the code.
    [
        "ALTER TABLE verdicts ADD COLUMN sweep_id TEXT",
        "ALTER TABLE verdicts ADD COLUMN sweep_unit TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE verdicts ADD COLUMN seed INTEGER",
        "ALTER TABLE verdicts ADD COLUMN prompt_hash TEXT NOT NULL DEFAULT ''",
        "CREATE TABLE IF NOT EXISTS observations ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " job_id TEXT NOT NULL,"
        " sweep_id TEXT,"
        " sweep_unit TEXT NOT NULL DEFAULT '',"
        " seed INTEGER,"
        " prompt_hash TEXT NOT NULL DEFAULT '',"
        " vector TEXT NOT NULL DEFAULT '{}',"
        " metrics TEXT NOT NULL DEFAULT '{}',"
        " created_at REAL NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_observations_job ON observations(job_id)",
    ],
    # 6 -- mesh candidates. Two columns, for exactly the reasons migration 4
    # gives for sweep membership: the library filters on it, params is a JSON
    # string sqlite cannot index into, and -- the half that matters most --
    # ``rerun_job``/``promote_to_model`` copy *params* and nothing else, so
    # membership can never leak onto a reroll. That sidesteps DERIVED_PARAMS
    # rather than bending it.
    #
    # ``candidate_group`` is NULL for an ordinary job *and* for a candidate the
    # user has decided about: "Keep" dissolves the group (service.jobs.
    # keep_candidate) rather than flagging a winner, so ``Filters.matches``
    # asks the same single question it asks of sweep_id and no row can end up
    # hidden with nothing left to reach it by. ``candidate_index`` survives
    # that dissolve and is read by nothing, deliberately: it is the forensic
    # half, the way ``sweep_unit`` is -- it says which candidate of a group an
    # asset used to be, for somebody reading the table.
    #
    # This entry took 6 by landing first. Migrations are append-only and never
    # edited once shipped, so anything else planned for "migration 6" is now 7.
    [
        "ALTER TABLE jobs ADD COLUMN candidate_group TEXT",
        "ALTER TABLE jobs ADD COLUMN candidate_index INTEGER NOT NULL DEFAULT 0",
        "CREATE INDEX IF NOT EXISTS idx_jobs_candidate ON jobs(candidate_group)",
    ],
    # 7 -- which question a verdict answers. Three values, 'reference' |
    # 'blank' | 'model', and the third value is the load-bearing one.
    #
    # The instinct is a two-value column mirroring ``jobs.stage`` (already
    # exactly 'reference' | 'model') and recovering the intent by joining back
    # to the job. That breaks for the reason ``verdicts.vector`` is
    # denormalized at all: ``prune_jobs`` deletes job rows, and the corpus has
    # to outlive the assets it was learned from. A label whose meaning depends
    # on a row that no longer exists is uninterpretable exactly when it
    # matters. So intent is stored, not derived.
    #
    # Two of the values are *intents over the same artifact*: in 2D mode the
    # image is the deliverable, and the same PNG is sometimes the product and
    # sometimes the input to the next machine -- where "good" means opposite
    # things (rich and dramatic against single-subject on plain background). A
    # reference-stage image therefore takes two independent labels, which is
    # also why intent may not live in ``source``: ``latest_verdicts`` keys on
    # (job_id, source) and one answer would silently overwrite the other.
    #
    # The DEFAULT backfills every existing row to 'model', which is what every
    # verdict recorded to date is about.
    #
    # This is the entry `TODO.md` and BUILD_PLAN both called "migration 6".
    # Migration 6 is the candidate columns above: it landed first, and
    # migrations are append-only and never renumbered.
    [
        "ALTER TABLE verdicts ADD COLUMN stage TEXT NOT NULL DEFAULT 'model'",
    ],
]


# Matches any legal SQLite "ALTER TABLE <table> ADD [COLUMN] <name> ...": the
# COLUMN keyword is optional, whitespace is free-form, and both names may be
# quoted or bracketed. MIGRATIONS[0] was written to match a literal
# ``.startswith("ALTER TABLE jobs ADD COLUMN")`` + ``split()`` template; that
# template is correct for the one statement it was copied from but wrong for
# anything a future migration might legally write, and a miss means a fresh
# database tries to re-add an existing column and fails to start. This regex
# removes the class rather than the instance -- see ``service.jobs``'s job-id
# checks for the same reasoning applied to identifiers. It hardened again for
# migration 5, which is the first to ALTER a table other than jobs: a
# jobs-only pattern would have let a fresh DB replay the verdicts ALTERs
# against columns _SCHEMA already created.
_ADD_COLUMN_RE = re.compile(
    r"^ALTER\s+TABLE\s+[\"'\[]?(\w+)[\"'\]]?\s+ADD\s+(?:COLUMN\s+)?[\"'\[]?(\w+)",
    re.IGNORECASE,
)


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    # Each table's columns are snapshotted once, on the first ALTER that names
    # it. Safe because no migration in this loop removes a column: a snapshot
    # can only go stale by missing a column a later entry adds, which just
    # means that entry's own ADD COLUMN runs instead of being skipped -- never
    # a false skip.
    columns: dict[str, set[str]] = {}
    for i in range(version, len(MIGRATIONS)):
        for stmt in MIGRATIONS[i]:
            # A fresh DB got these columns from _SCHEMA; replaying the ALTER
            # would fail on it. Skipping the statement rather than the whole
            # entry keeps fresh and migrated DBs converging, which is the
            # property the append-only contract exists to protect.
            match = _ADD_COLUMN_RE.match(stmt)
            if match:
                table, column = match.group(1), match.group(2)
                if table not in columns:
                    columns[table] = {
                        r[1] for r in conn.execute(f"PRAGMA table_info({table})")
                    }
                if column in columns[table]:
                    continue
            conn.execute(stmt)
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()


class JobStore:
    """One sqlite3 connection, guarded by an RLock.

    Every method that touches ``self._conn`` takes ``self._lock`` first. The lock
    is not decoration: callers reach this store through ``asyncio.to_thread``,
    whose default executor is a *multi*-worker pool, so two requests really can
    land in ``execute``/``commit`` on different threads at the same instant.
    ``check_same_thread=False`` disables sqlite3's own guard against exactly
    that, which makes the lock the only thing left serialising writes.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        _migrate(self._conn)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(
        self,
        kind: str,
        prompt: str | None,
        params: dict[str, Any],
        job_id: str | None = None,
        *,
        stage: str = "model",
        parent_id: str | None = None,
        status: str = "queued",
        sweep_id: str | None = None,
        sweep_unit: str = "",
        candidate_group: str | None = None,
        candidate_index: int = 0,
    ) -> str:
        """Insert a job row. ``status`` is queued for everything the worker
        runs.

        The one caller that passes anything else is an *import*: pixels the
        user painted are already the artifact, so the row is born ``done``
        rather than being created queued and immediately finished. Written in
        one statement deliberately -- a create-then-claim pair leaves a window
        in which the worker's poll can pick the job up and try to run it.
        """
        job_id = job_id or uuid.uuid4().hex[:12]
        now = time.time()
        finished = now if status in ("done", "error", "cancelled") else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, prompt, params, created_at,"
                " stage, parent_id, started_at, finished_at, sweep_id, sweep_unit,"
                " candidate_group, candidate_index)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    kind,
                    status,
                    prompt,
                    json.dumps(params),
                    now,
                    stage,
                    parent_id,
                    finished,
                    finished,
                    sweep_id,
                    sweep_unit,
                    candidate_group,
                    int(candidate_index),
                ),
            )
            self._conn.commit()
        return job_id

    def set_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE jobs SET stage = ? WHERE id = ?", (stage, job_id))
            self._conn.commit()

    def children(self, parent_id: str) -> list[dict[str, Any]]:
        """Every job promoted from this one, oldest first. ``id`` is a
        secondary sort key: ``time.time()`` can tie across rows created in
        quick succession, and rowid order isn't guaranteed by SQL, so
        created_at alone leaves ties order-undefined."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE parent_id = ? ORDER BY created_at, id", (parent_id,)
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._to_dict(row) if row else None

    def list(
        self, limit: int = 100, before: tuple[float, str] | None = None
    ) -> list[dict[str, Any]]:
        """The newest ``limit`` jobs, or the newest after ``before``.

        ``before`` is a keyset cursor -- the (created_at, id) of the last row of
        the previous page -- rather than an offset, so rows deleted while
        paging (which is exactly what prune does) cannot make the walk skip
        entries. ``id`` is in both the cursor and the ORDER BY because
        ``time.time()`` genuinely ties across rows created in quick succession,
        and created_at alone leaves those ties order-undefined.
        """
        sql = "SELECT * FROM jobs"
        args: list[Any] = []
        if before is not None:
            sql += " WHERE (created_at < ?) OR (created_at = ? AND id < ?)"
            args += [before[0], before[0], before[1]]
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._to_dict(r) for r in rows]

    def count(self) -> int:
        """How many jobs exist, for the "showing newest N of M" row."""
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0])

    def set_status(self, job_id: str, status: str, error: str | None = None) -> None:
        """Update status. ``error`` is only written when explicitly given —
        an unrelated status transition (e.g. running -> cancelled) must not
        wipe out a previously recorded error message."""
        now = time.time()
        stamp_col = {"running": "started_at", "done": "finished_at", "error": "finished_at",
                     "cancelled": "finished_at"}.get(status)
        sets = ["status = ?"]
        args: list[Any] = [status]
        if error is not None:
            sets.append("error = ?")
            args.append(error)
        if stamp_col:
            sets.append(f"{stamp_col} = ?")
            args.append(now)
        args.append(job_id)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", args)
            self._conn.commit()

    def set_params(self, job_id: str, params: dict[str, Any]) -> None:
        """Replace the params blob. Used by the worker to record derived values
        (the composed prompt, the scale factor actually applied) so a finished
        job carries everything needed to explain or reproduce it."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET params = ? WHERE id = ?", (json.dumps(params), job_id)
            )
            self._conn.commit()

    def merge_params(
        self,
        job_id: str,
        changes: dict[str, Any],
        *,
        remove: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        """Apply ``changes`` (and drop ``remove``) onto the stored params blob.

        The read and the write happen under one hold of the lock, which is the
        whole point: params is a single JSON blob, so ``set_params`` from a copy
        read earlier is a last-write-wins update that silently discards whatever
        landed in between. Two writers really do exist -- the worker records
        derived values while POST /optimize can rewrite the budget of the same
        row -- and neither touches the keys the other cares about, so a merge
        loses nothing a full-blob write wouldn't have destroyed.

        Returns the params as written, or None if the job is gone.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT params FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            params = json.loads(row["params"] or "{}")
            for key in remove:
                params.pop(key, None)
            params.update(changes)
            self._conn.execute(
                "UPDATE jobs SET params = ? WHERE id = ?", (json.dumps(params), job_id)
            )
            self._conn.commit()
        return params

    def set_meta(
        self,
        job_id: str,
        *,
        name: str | None = None,
        tags: str | None = None,
        favorite: bool | None = None,
    ) -> bool:
        """Update the user-facing metadata. Only the fields given are written.

        Partial by design: the UI's star button and its rename field are
        separate actions on the same row, and a full-row write from either
        would silently clobber whatever the other just did.
        """
        sets: list[str] = []
        args: list[Any] = []
        for column, value in (("name", name), ("tags", tags)):
            if value is not None:
                sets.append(f"{column} = ?")
                args.append(value)
        if favorite is not None:
            sets.append("favorite = ?")
            args.append(1 if favorite else 0)
        if not sets:
            return self.get(job_id) is not None
        args.append(job_id)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", args
            )
            self._conn.commit()
            return cur.rowcount > 0

    def claim(self, job_id: str) -> bool:
        """Atomically transition queued -> running. False if the job was
        already claimed, cancelled, or deleted since it was fetched —
        closes the race between next_queued() and a concurrent cancel."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?"
                " WHERE id = ? AND status = 'queued'",
                (now, job_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def cancel(self, job_id: str) -> bool:
        """Atomically transition queued|running -> cancelled. False if the
        job already reached a terminal state (done/error/cancelled) --
        closes the race between this write and the worker's own terminal
        write in Worker._process's finally block."""
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'cancelled', finished_at = ?"
                " WHERE id = ? AND status IN ('queued', 'running')",
                (now, job_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def finish(self, job_id: str, status: str, error: str | None = None) -> bool:
        """Atomically transition running -> a terminal status (done/error).
        False if the job was cancelled out from under it between claim()
        and this call -- the caller must not overwrite that outcome or
        leave a viewable artifact for a job the DB already says was
        cancelled."""
        now = time.time()
        sets = ["status = ?", "finished_at = ?"]
        args: list[Any] = [status, now]
        if error is not None:
            sets.append("error = ?")
            args.append(error)
        args.append(job_id)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET {', '.join(sets)} WHERE id = ? AND status = 'running'",
                args,
            )
            self._conn.commit()
            return cur.rowcount > 0

    def reconcile_startup(self) -> None:
        """Any job still 'running' at process start was orphaned by a crash
        or an unclean shutdown — not silently re-run, made visibly an error."""
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = 'error', error = ?, finished_at = ?"
                " WHERE status = 'running'",
                ("interrupted by shutdown", now),
            )
            self._conn.commit()

    def next_queued(self) -> dict[str, Any] | None:
        """The oldest queued job. ``id`` is a secondary sort key for the reason
        it is in ``children`` and ``list``: ``time.time()`` genuinely ties
        across rows inserted in quick succession, which is exactly what a sweep
        submit is -- N rows in one loop. A tie left order-undefined makes
        dispatch order vary between runs of the same sweep, which in turn makes
        the worker's config grouping (and so the number of trellis restarts)
        vary. Correctness never depended on it; the restart count does."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
        return self._to_dict(row) if row else None

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def delete_if_not_running(self, job_id: str) -> bool:
        """Delete unless the worker owns the row. -> whether the row went.

        The status check and the delete are one statement under the lock --
        the same shape as ``claim`` -- because a caller that checks a snapshot
        and then calls ``delete`` races the worker's claim() in the gap, and
        loses a live reconstruction's row while the run keeps writing.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE id = ? AND status != 'running'", (job_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def active_jobs(self) -> list[dict[str, Any]]:
        """Every queued or running row, oldest first.

        Unbounded on purpose and cheap in practice: one job runs at a time and
        a queue is a handful of rows, so the caller that has to ask "is anything
        unfinished going to write into this directory" can filter these in
        Python rather than the store learning to search inside the params blob.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running')"
                " ORDER BY created_at, id"
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    # --- sweeps ---------------------------------------------------------------

    def create_sweep(self, label: str, prompt: str, spec: dict[str, Any]) -> str:
        sweep_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO sweeps (id, label, prompt, spec, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (sweep_id, label, prompt, json.dumps(spec), time.time()),
            )
            self._conn.commit()
        return sweep_id

    def list_sweeps(self) -> list[dict[str, Any]]:
        """Every sweep, newest first, each with its unit counts.

        The counts come from one grouped join rather than a query per sweep:
        this is read on every Review rescan, and the store is one serialized
        connection every other reader is queued behind.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sweeps ORDER BY created_at DESC, id DESC"
            ).fetchall()
            counts = self._conn.execute(
                "SELECT sweep_id,"
                " COUNT(*) AS units,"
                " SUM(CASE WHEN status IN ('done', 'error', 'cancelled') THEN 1 ELSE 0 END)"
                "   AS done"
                " FROM jobs WHERE sweep_id IS NOT NULL GROUP BY sweep_id"
            ).fetchall()
        tally = {r["sweep_id"]: (int(r["units"]), int(r["done"] or 0)) for r in counts}
        out: list[dict[str, Any]] = []
        for row in rows:
            units, done = tally.get(row["id"], (0, 0))
            entry = dict(row)
            entry["spec"] = json.loads(entry["spec"] or "{}")
            entry["units"] = units
            entry["done"] = done
            entry["todo"] = units - done
            out.append(entry)
        return out

    def get_sweep(self, sweep_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sweeps WHERE id = ?", (sweep_id,)
            ).fetchone()
        if row is None:
            return None
        entry = dict(row)
        entry["spec"] = json.loads(entry["spec"] or "{}")
        return entry

    def sweep_jobs(self, sweep_id: str) -> list[dict[str, Any]]:
        """A sweep's units in submission order -- the order the worker will
        dispatch them in, which is the order they were grouped into."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE sweep_id = ? ORDER BY created_at, id", (sweep_id,)
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    def delete_sweep(self, sweep_id: str) -> None:
        """The sweeps row only. Its jobs and their verdicts are the caller's
        business: ``service.sweeps.delete_sweep`` cancels and removes the jobs,
        and the verdict rows are deliberately *kept* -- each carries its own
        config-vector snapshot, which is the whole point of denormalizing it."""
        with self._lock:
            self._conn.execute("DELETE FROM sweeps WHERE id = ?", (sweep_id,))
            self._conn.commit()

    # --- candidates -----------------------------------------------------------

    def candidate_jobs(self, group: str) -> list[dict[str, Any]]:
        """A candidate group's members, in the order they were submitted.

        ``candidate_index`` first rather than ``created_at``: the index is what
        the seed rule is stated in terms of (candidate 0 keeps the requested
        seed), so it has to be what the picker numbers them by. ``id`` breaks a
        tie for the same reason it does everywhere else here -- rowid order is
        not guaranteed by SQL, and the whole group is inserted inside one
        ``time.time()`` tick.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE candidate_group = ? ORDER BY candidate_index, id",
                (group,),
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    def resolve_candidates(self, group: str) -> int:
        """Dissolve a candidate group: every member becomes an ordinary job.

        One statement, so no member can be left in a group the others have
        left. ``candidate_index`` is deliberately *not* cleared -- it is the
        forensic record of which candidate an asset was, the way ``sweep_unit``
        is kept on a verdict row after ``delete_sweep``.

        -> how many rows stopped being candidates.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET candidate_group = NULL WHERE candidate_group = ?", (group,)
            )
            self._conn.commit()
            return cur.rowcount

    # --- verdicts -------------------------------------------------------------

    def add_verdict(
        self,
        job_id: str,
        *,
        source: str,
        verdict: str,
        reasons: list[str],
        vector: dict[str, Any],
        sweep_id: str | None = None,
        sweep_unit: str = "",
        seed: int | None = None,
        prompt_hash: str = "",
        stage: str = "model",
    ) -> int:
        """Append one verdict. Append-only: a changed mind is a new row, and
        ``latest_verdicts`` takes the highest id per (job, source, stage), so the
        history of what a reviewer thought survives the correction.

        The sweep context is denormalized for the same reason the vector is:
        matched-pair comparisons must survive ``delete_sweep`` taking the job
        rows. Rows recorded before migration 5 have no context and simply
        never enter a comparison.

        ``stage`` defaults to ``'model'`` because every existing caller records a
        mesh verdict and every row written before migration 7 is one.
        """
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO verdicts (job_id, source, verdict, reasons, vector,"
                " created_at, sweep_id, sweep_unit, seed, prompt_hash, stage)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    source,
                    verdict,
                    json.dumps(list(reasons)),
                    json.dumps(vector),
                    time.time(),
                    sweep_id,
                    sweep_unit,
                    seed,
                    prompt_hash,
                    stage,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def latest_verdicts(self) -> list[dict[str, Any]]:
        """One row per (job_id, source, stage) -- the newest, by id.

        ``id`` rather than ``created_at``: the column is an AUTOINCREMENT
        rowid, so it is strictly increasing even when two verdicts land inside
        one ``time.time()`` tick, which at the rate a reviewer presses A they
        genuinely do.

        ``stage`` joined the grouping with migration 7 and had to: the same
        reference image takes two independent labels (is it a good 2D asset, will
        it reconstruct), and grouping without stage would let one answer
        supersede the other. The consequence for readers is that a job can now
        contribute more than one row, which is why ``findings`` filters by stage
        rather than averaging whatever it finds.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM verdicts WHERE id IN ("
                " SELECT MAX(id) FROM verdicts GROUP BY job_id, source, stage)"
                " ORDER BY id"
            ).fetchall()
        return [self._verdict_to_dict(r) for r in rows]

    def verdicts_for(
        self, job_ids: list[str], *, source: str | None = None, stage: str | None = None
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """``{(job_id, source): latest verdict}`` for the jobs named.

        Chunked because sqlite caps a statement at SQLITE_MAX_VARIABLE_NUMBER
        (999 on the builds Python ships); a sweep of a few hundred units is
        already within one chunk, but the ceiling is not this module's to
        assume.

        The key stays ``(job_id, source)`` rather than growing a third element,
        so ``stage`` is a filter rather than a dimension: every caller is asking
        one question at a time (Review asks about meshes, the labelling grid
        about one intent), and a caller that passed nothing would otherwise get
        two answers under one key and keep whichever the row order handed it
        last. Passing nothing therefore means *every* stage, which only a caller
        with one stage in play may do.
        """
        out: dict[tuple[str, str], dict[str, Any]] = {}
        ids = list(job_ids)
        if not ids:
            return out
        with self._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                marks = ",".join("?" * len(chunk))
                sql = (
                    f"SELECT * FROM verdicts WHERE job_id IN ({marks})"
                    " AND id IN (SELECT MAX(id) FROM verdicts"
                    " GROUP BY job_id, source, stage)"
                )
                args: list[Any] = list(chunk)
                if source is not None:
                    sql += " AND source = ?"
                    args.append(source)
                if stage is not None:
                    sql += " AND stage = ?"
                    args.append(stage)
                for row in self._conn.execute(sql, args).fetchall():
                    record = self._verdict_to_dict(row)
                    out[(record["job_id"], record["source"])] = record
        return out

    def unverdicted_models(self, *, source: str = "human", limit: int = 50) -> list[
        dict[str, Any]
    ]:
        """Finished ordinary meshes nobody has judged, newest first.

        ``sweep_id IS NULL`` because a sweep's units are reviewed under their
        own sweep; this is the "daily use feeds the same findings pool" half,
        and mixing the two would list every sweep unit twice.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status = 'done' AND stage = 'model'"
                " AND sweep_id IS NULL"
                " AND id NOT IN (SELECT job_id FROM verdicts WHERE source = ?"
                " AND stage = 'model')"
                " ORDER BY created_at DESC, id DESC LIMIT ?",
                (source, limit),
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    # Which population each image label is a question *about*. Not decoration:
    # in 2D mode the image is the deliverable, so a reference-stage job's image
    # is a product ("rich, styled, dramatic"), while a model-stage job's is the
    # blank trellis consumed ("single subject, nothing else, plain background").
    # The two definitions of good are opposed -- a dramatic plate with pillars
    # and a cast shadow is a better asset and a worse blank -- so a probe pointed
    # at the wrong population learns the average of two opposed objectives and is
    # useless for each.
    LABEL_POPULATION = {"reference": "reference", "blank": "model"}

    def unlabelled_references(
        self, *, stage: str, source: str = "human", limit: int = 200
    ) -> list[dict[str, Any]]:
        """Images with no label under ``stage`` yet, newest first.

        Deliberately **not** ``unverdicted_models`` with a parameter, and it
        cannot be fixed with one. That query filters ``status = 'done' AND stage
        = 'model' AND sweep_id IS NULL``, which excludes the two things a
        labelling pass most needs:

        * **Errored jobs.** A reference refused at the composition gate for
          multi-object is the most informative negative available, and it is a
          model-stage job that *failed* -- so ``status = 'done'`` throws away
          precisely the rows a blank probe has to learn from.
        * **Sweep units.** They are the entire corpus this work is built on.

        ``queued`` and ``running`` are still excluded: there is nothing to look
        at yet. And the exclusion is per ``(source, stage)``, so a mesh verdict
        on a job leaves its blank still waiting to be judged -- two different
        claims about one row, which is the whole reason ``stage`` is a column.
        """
        population = self.LABEL_POPULATION.get(stage)
        if population is None:
            # ``model`` is the mesh question and ``unverdicted_models`` is its
            # query. Answering it here would be a second, subtly different
            # spelling of one listing.
            raise ValueError(
                f"unlabelled_references does not serve stage {stage!r}; "
                f"expected one of {sorted(self.LABEL_POPULATION)}"
            )
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE stage = ?"
                " AND status IN ('done', 'error')"
                " AND id NOT IN (SELECT job_id FROM verdicts WHERE source = ?"
                " AND stage = ?)"
                " ORDER BY created_at DESC, id DESC LIMIT ?",
                (population, source, stage, limit),
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    # --- observations ---------------------------------------------------------

    def add_observation(
        self,
        job_id: str,
        *,
        sweep_id: str | None,
        sweep_unit: str,
        seed: int | None,
        prompt_hash: str,
        vector: dict[str, Any],
        metrics: dict[str, Any],
    ) -> int:
        """Append one machine-evidence row -- what the worker measured about a
        finished model job, snapshotted with its config vector so it outlives
        the job row. Nothing deletes these; that is the point."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO observations (job_id, sweep_id, sweep_unit, seed,"
                " prompt_hash, vector, metrics, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    sweep_id,
                    sweep_unit,
                    seed,
                    prompt_hash,
                    json.dumps(vector),
                    json.dumps(metrics),
                    time.time(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def latest_observations(self) -> list[dict[str, Any]]:
        """One row per job -- the newest, by id, matching ``latest_verdicts``:
        a retarget that re-audits a mesh may someday append a second row, and
        the newest is the one describing the artifact that exists."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM observations WHERE id IN ("
                " SELECT MAX(id) FROM observations GROUP BY job_id)"
                " ORDER BY id"
            ).fetchall()
        return [self._observation_to_dict(r) for r in rows]

    @staticmethod
    def _blob(text: Any, empty: Any) -> Any:
        """A JSON column: its value, ``empty`` if the column is blank, or
        ``None`` if it will not parse at all.

        ``aggregate`` is careful to skip a row whose payload is the wrong
        *type*, but that check never ran on a payload that was not JSON at all:
        the decode happens here, and an unparseable blob raised out of the
        store, taking the whole findings recompute -- and ``verdicts_for``, and
        so Review's own list -- down with it. ``None`` rather than an empty
        container on purpose: every reader already treats a non-dict vector as
        a row to skip and a falsy ``reasons`` as none, whereas ``{}`` would be
        a *readable* row describing the empty configuration and would rank as
        one. A row nobody can read is one row of evidence lost; it is not a
        reason to lose the rest.
        """
        if not text:
            return empty
        try:
            return json.loads(text)
        except ValueError:
            log.warning("unreadable JSON column; that row is being skipped")
            return None

    @staticmethod
    def _observation_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["vector"] = JobStore._blob(d["vector"], {})
        d["metrics"] = JobStore._blob(d["metrics"], {})
        return d

    @staticmethod
    def _verdict_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["reasons"] = JobStore._blob(d["reasons"], [])
        d["vector"] = JobStore._blob(d["vector"], {})
        return d

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        return d
