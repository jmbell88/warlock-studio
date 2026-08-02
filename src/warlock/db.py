"""SQLite job store. Synchronous sqlite3 behind asyncio.to_thread — job volume is tiny."""

from __future__ import annotations

import json
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
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

# Append-only. Each entry is a list of SQL statements applied in one
# transaction, bumping PRAGMA user_version by one. Never edit an entry once
# it has shipped — only append. A fresh DB gets _SCHEMA then replays every
# entry here, so fresh and pre-existing DBs converge on the same shape.
MIGRATIONS: list[list[str]] = []


def _migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i in range(version, len(MIGRATIONS)):
        for stmt in MIGRATIONS[i]:
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
        self, kind: str, prompt: str | None, params: dict[str, Any], job_id: str | None = None
    ) -> str:
        job_id = job_id or uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, prompt, params, created_at)"
                " VALUES (?, ?, 'queued', ?, ?, ?)",
                (job_id, kind, prompt, json.dumps(params), time.time()),
            )
            self._conn.commit()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._to_dict(row) if row else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._to_dict(r) for r in rows]

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
