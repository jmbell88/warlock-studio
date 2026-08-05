"""SQLite job store. Synchronous sqlite3 behind asyncio.to_thread — job volume is tiny."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

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
    sweep_unit  TEXT NOT NULL DEFAULT ''        -- display label, e.g. "lora_weight=0.6 s42"
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
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verdicts_job ON verdicts(job_id);
"""
# idx_jobs_parent is created by the migration below, not here: _SCHEMA's
# CREATE TABLE IF NOT EXISTS is a no-op against a pre-existing table that
# predates the parent_id column, and an index on a column that doesn't exist
# yet would fail before _migrate ever runs.

# Append-only. Each entry is a list of SQL statements applied in one
# transaction, bumping PRAGMA user_version by one. Never edit an entry once
# it has shipped — only append. A fresh DB gets _SCHEMA then replays every
# entry here, so fresh and pre-existing DBs converge on the same shape.
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
]


# Matches any legal SQLite "ALTER TABLE jobs ADD [COLUMN] <name> ...": the
# COLUMN keyword is optional, whitespace is free-form, and the name may be
# quoted or bracketed. MIGRATIONS[0] was written to match a literal
# ``.startswith("ALTER TABLE jobs ADD COLUMN")`` + ``split()`` template; that
# template is correct for the one statement it was copied from but wrong for
# anything a future migration might legally write, and a miss means a fresh
# database tries to re-add an existing column and fails to start. This regex
# removes the class rather than the instance -- match _check_job_id in
# app.py for the same reasoning applied to job ids.
_ADD_COLUMN_RE = re.compile(
    r"^ALTER\s+TABLE\s+jobs\s+ADD\s+(?:COLUMN\s+)?[\"'\[]?(\w+)", re.IGNORECASE
)


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    # Snapshotted once, before any migration runs. Safe because no migration
    # in this loop removes a column: a snapshot can only go stale by missing
    # a column a later entry adds, which just means that entry's own ADD
    # COLUMN runs instead of being skipped -- never a false skip.
    columns = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    for i in range(version, len(MIGRATIONS)):
        for stmt in MIGRATIONS[i]:
            # A fresh DB got these columns from _SCHEMA; replaying the ALTER
            # would fail on it. Skipping the statement rather than the whole
            # entry keeps fresh and migrated DBs converging, which is the
            # property the append-only contract exists to protect.
            match = _ADD_COLUMN_RE.match(stmt)
            if match and match.group(1) in columns:
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
                " stage, parent_id, started_at, finished_at, sweep_id, sweep_unit)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    # --- verdicts -------------------------------------------------------------

    def add_verdict(
        self,
        job_id: str,
        *,
        source: str,
        verdict: str,
        reasons: list[str],
        vector: dict[str, Any],
    ) -> int:
        """Append one verdict. Append-only: a changed mind is a new row, and
        ``latest_verdicts`` takes the highest id per (job, source), so the
        history of what a reviewer thought survives the correction."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO verdicts (job_id, source, verdict, reasons, vector, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    source,
                    verdict,
                    json.dumps(list(reasons)),
                    json.dumps(vector),
                    time.time(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def latest_verdicts(self) -> list[dict[str, Any]]:
        """One row per (job_id, source) -- the newest, by id.

        ``id`` rather than ``created_at``: the column is an AUTOINCREMENT
        rowid, so it is strictly increasing even when two verdicts land inside
        one ``time.time()`` tick, which at the rate a reviewer presses A they
        genuinely do.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM verdicts WHERE id IN ("
                " SELECT MAX(id) FROM verdicts GROUP BY job_id, source)"
                " ORDER BY id"
            ).fetchall()
        return [self._verdict_to_dict(r) for r in rows]

    def verdicts_for(self, job_ids: list[str], *, source: str | None = None) -> dict[
        tuple[str, str], dict[str, Any]
    ]:
        """``{(job_id, source): latest verdict}`` for the jobs named.

        Chunked because sqlite caps a statement at SQLITE_MAX_VARIABLE_NUMBER
        (999 on the builds Python ships); a sweep of a few hundred units is
        already within one chunk, but the ceiling is not this module's to
        assume.
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
                    " AND id IN (SELECT MAX(id) FROM verdicts GROUP BY job_id, source)"
                )
                args: list[Any] = list(chunk)
                if source is not None:
                    sql += " AND source = ?"
                    args.append(source)
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
                " AND id NOT IN (SELECT job_id FROM verdicts WHERE source = ?)"
                " ORDER BY created_at DESC, id DESC LIMIT ?",
                (source, limit),
            ).fetchall()
        return [self._to_dict(r) for r in rows]

    @staticmethod
    def _verdict_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["reasons"] = json.loads(d["reasons"] or "[]")
        d["vector"] = json.loads(d["vector"] or "{}")
        return d

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        return d
