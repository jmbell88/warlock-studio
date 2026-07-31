"""SQLite job store. Synchronous sqlite3 behind asyncio.to_thread — job volume is tiny."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,              -- 'text' | 'image'
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


class JobStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def create(self, kind: str, prompt: str | None, params: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO jobs (id, kind, status, prompt, params, created_at)"
            " VALUES (?, ?, 'queued', ?, ?, ?)",
            (job_id, kind, prompt, json.dumps(params), time.time()),
        )
        self._conn.commit()
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._to_dict(row) if row else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._to_dict(r) for r in rows]

    def set_status(self, job_id: str, status: str, error: str | None = None) -> None:
        now = time.time()
        stamp_col = {"running": "started_at", "done": "finished_at", "error": "finished_at",
                     "cancelled": "finished_at"}.get(status)
        stamp_sql = f", {stamp_col} = ?" if stamp_col else ""
        args: list[Any] = [status, error]
        if stamp_col:
            args.append(now)
        args.append(job_id)
        self._conn.execute(
            f"UPDATE jobs SET status = ?, error = ?{stamp_sql} WHERE id = ?", args
        )
        self._conn.commit()

    def next_queued(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        return self._to_dict(row) if row else None

    def delete(self, job_id: str) -> None:
        self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._conn.commit()

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        return d
