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
    favorite    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
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
                " stage, parent_id, started_at, finished_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
        return self._to_dict(row) if row else None

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        return d
