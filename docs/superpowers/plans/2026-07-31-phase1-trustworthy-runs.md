# Phase 1: Trustworthy Runs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cancel actually stops the GPU within seconds and leaves no viewable artifact; Ctrl+C exits within 20s; a crash in one job no longer bricks the worker permanently; `animancer3d doctor` reports what's missing before a 2-minute job wastes GPU time; failures read as sentences instead of tracebacks.

**Architecture:** No new processes, no new schema columns. Cancellation is a `threading.Event` set by the API route and read by the worker's `_process`/`_generate` loop and by the text2image step callback (mirrors the existing "who's allowed to touch what thread" split the codebase already has for `progress.py`). The only mechanism that actually frees the GPU mid-run is killing `trellis-server.exe` (`TrellisServer.stop()`), which already exists and is already used for the VRAM handoff — there is no cancel/abort HTTP endpoint on trellis-server.exe itself. A `db.py` migration harness (`PRAGMA user_version` + an append-only `MIGRATIONS` list) ships now with zero migrations queued, so Phase 3's `stage` column lands without inventing the harness under pressure.

**Tech Stack:** Python 3.12+, FastAPI, sqlite3 (stdlib), asyncio, pytest + pytest-asyncio (`asyncio_mode = auto`), ruff. No new dependencies.

## Global Constraints

- Every commit in this plan is a single commit per task (or per fix round), with subject line exactly `Animancer3D v0.0.1` — the version number does not change unless a human explicitly asks for a bump. Put the actual description in the commit body, not the subject.
- Run `uv run pytest` and `uv run ruff check .` before every commit; both must be clean. Report actual output, not "tests pass" from memory.
- **The single-event-loop invariant (`db.py` / `queue.py`):** `JobStore` wraps one unsynchronized `sqlite3` connection with `check_same_thread=False`. This is only safe because every call is funneled through `asyncio.to_thread` from the single asyncio event loop — that serializes DB access onto one worker thread at a time. Any new `JobStore` call site added by this plan must go through `asyncio.to_thread` from `app.py`, and may be called directly (no `to_thread`) from `queue.py`'s own synchronous test helpers/fixtures since those aren't racing the event loop.
- **VRAM handoff order (`queue.py:1-5, 158-174`):** the 3D server and SDXL-Turbo never run concurrently. For a text job: `trellis.stop()` first, then load+run SDXL, then `t2i.unload()` in a `finally` before the next job can start `trellis-server.exe` again. Nothing in this plan changes that order — cancellation hooks into it, it does not replace it.
- **Cancelled ⇒ no artifact.** If `model.glb` landed before a cancel took effect, it must be unlinked. A cancelled job that still renders a model in the UI is a bug.
- No new third-party dependencies. No new sqlite columns or schema changes in this phase — the migration harness ships with an empty `MIGRATIONS` list.
- Windows environment: no POSIX-only APIs (no `signal.SIGKILL`, no `os.fork`).

---

### Task 1: `errors.py` — friendly error messages

**Files:**
- Create: `src/animancer3d/errors.py`
- Test: `tests/test_errors.py`

**Interfaces:**
- Produces: `friendly(exc: Exception) -> str`, `write_error_log(job_dir: Path, exc: Exception) -> None`. Task 6 (`queue.py`) calls both from `_process`'s exception handler.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_errors.py`:

```python
from __future__ import annotations

import httpx

from animancer3d.errors import friendly, write_error_log


def test_oom_message_is_friendly():
    exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    assert "resolution 512" in friendly(exc)


def test_transport_error_points_at_the_log():
    exc = httpx.TransportError("connection reset")
    assert "trellis.log" in friendly(exc)


def test_unknown_exception_falls_back_to_str():
    assert friendly(ValueError("weird")) == "weird"


def test_exception_with_no_message_falls_back_to_class_name():
    assert friendly(RuntimeError()) == "RuntimeError"


def test_write_error_log_captures_full_traceback(tmp_path):
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        write_error_log(tmp_path / "job1", exc)
    content = (tmp_path / "job1" / "error.log").read_text()
    assert "RuntimeError: boom" in content
    assert "Traceback" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'animancer3d.errors'`

- [ ] **Step 3: Implement `errors.py`**

Create `src/animancer3d/errors.py`:

```python
"""Turn raw exceptions into a sentence a user can act on.

The full traceback always goes to the job's error.log; only the short,
friendly sentence goes in the DB and the UI.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import httpx


def friendly(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "GPU out of memory — try resolution 512, or close other GPU apps."
    if isinstance(exc, httpx.TransportError):
        return "The 3D engine stopped unexpectedly. See assets/trellis.log."
    return str(exc) or exc.__class__.__name__


def write_error_log(job_dir: Path, exc: Exception) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "error.log").write_text(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_errors.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`
Expected: no errors

```bash
git add src/animancer3d/errors.py tests/test_errors.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Add errors.py: friendly(exc) turns raw exceptions into a sentence a
user can act on (GPU OOM, trellis-server crash), write_error_log
captures the full traceback to the job directory.
EOF
)"
```

---

### Task 2: `db.py` — migration harness, claim, reconcile_startup, error preservation

**Files:**
- Modify: `src/animancer3d/db.py`
- Test: `tests/test_db.py` (new)
- Create: `tests/test_migrate.py`

**Interfaces:**
- Produces: `JobStore.claim(job_id: str) -> bool`, `JobStore.reconcile_startup() -> None`, `JobStore.create(kind, prompt, params, job_id: str | None = None) -> str`, `JobStore.set_status(job_id, status, error=None)` (now preserves an existing `error` column value when `error` is not passed). Module-level `MIGRATIONS: list[list[str]]` (starts empty). Task 6 (`queue.py`) calls `claim`; Task 8 (`app.py`) calls `reconcile_startup` and passes a pre-generated `job_id` to `create`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db.py`:

```python
from __future__ import annotations

from animancer3d.db import JobStore


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
```

Create `tests/test_migrate.py`:

```python
from __future__ import annotations

import sqlite3

from animancer3d.db import _SCHEMA, MIGRATIONS, JobStore


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py tests/test_migrate.py -v`
Expected: FAIL — `AttributeError: 'JobStore' object has no attribute 'claim'` (and `ImportError: cannot import name 'MIGRATIONS'`)

- [ ] **Step 3: Implement the migration harness, claim, reconcile_startup, error-preserving set_status, and job_id-accepting create**

Replace the full contents of `src/animancer3d/db.py`:

```python
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
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        _migrate(self._conn)

    def close(self) -> None:
        self._conn.close()

    def create(
        self, kind: str, prompt: str | None, params: dict[str, Any], job_id: str | None = None
    ) -> str:
        job_id = job_id or uuid.uuid4().hex[:12]
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
        self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", args)
        self._conn.commit()

    def claim(self, job_id: str) -> bool:
        """Atomically transition queued -> running. False if the job was
        already claimed, cancelled, or deleted since it was fetched —
        closes the race between next_queued() and a concurrent cancel."""
        now = time.time()
        cur = self._conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ? AND status = 'queued'",
            (now, job_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def reconcile_startup(self) -> None:
        """Any job still 'running' at process start was orphaned by a crash
        or an unclean shutdown — not silently re-run, made visibly an error."""
        now = time.time()
        self._conn.execute(
            "UPDATE jobs SET status = 'error', error = ?, finished_at = ? WHERE status = 'running'",
            ("interrupted by shutdown", now),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py tests/test_migrate.py tests/test_api.py -v`
Expected: PASS. (`tests/test_api.py` is included because `db.py` is shared state — confirm nothing there broke.)

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`

```bash
git add src/animancer3d/db.py tests/test_db.py tests/test_migrate.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Add a migration harness (PRAGMA user_version + append-only MIGRATIONS)
so Phase 3's stage column has somewhere to land. Add JobStore.claim()
to close the cancel/claim race between next_queued and the running
write, reconcile_startup() to mark orphaned running jobs as errored on
restart instead of silently re-running them, and job_id= on create()
so the caller can write files before the row becomes visible. Stop
set_status() from clobbering a recorded error on every later status
transition.
EOF
)"
```

---

### Task 3: `pipelines/text2image.py` — cancellable generation

**Files:**
- Modify: `src/animancer3d/pipelines/text2image.py`

**Interfaces:**
- Produces: `class JobCancelled(Exception)`. `Text2Image.generate(..., cancel_event: threading.Event | None = None)`. Task 6 (`queue.py`) passes its cancellation event in; Task 7's `conftest.py` fake mirrors this signature exactly.

No dedicated test file: `generate()` calls `self.load()`, which imports `torch`/`diffusers` — unavailable in this dev environment (no `text2image` extra installed) and untestable without a GPU. The cancellation *behavior* is exercised end-to-end in Task 7's `test_queue.py` against a fake that mimics this exact signature. This task is verified by inspection plus the full suite staying green (nothing here is reachable without the `text2image` extra, so no existing test touches it).

- [ ] **Step 1: Add `JobCancelled` and thread a `cancel_event` through `generate()`**

In `src/animancer3d/pipelines/text2image.py`, add near the top (after the module docstring, alongside the other imports):

```python
import threading
```

Add this class definition after the `STEPS = 4` / `PROMPT_TEMPLATE` block, before `class Text2Image:`:

```python
class JobCancelled(Exception):
    """Raised from inside a diffusers step callback to abort mid-sample."""
```

Replace the `generate` method with:

```python
    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        seed: int = 42,
        on_state: Callable[[str], None] | None = None,
        on_step: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        import torch

        self.load(on_state)
        assert self._pipe is not None
        # load()/download() have no interruption point of their own; check
        # once here so a cancel requested during either isn't silently lost.
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled
        if on_state is not None:
            on_state("sample")

        def step_cb(_pipe, i, _t, kwargs):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled
            if on_step is not None:
                on_step(i + 1, STEPS)
            return kwargs  # diffusers requires the kwargs dict back, not None

        image = self._pipe(
            PROMPT_TEMPLATE.format(prompt=prompt),
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            width=self._image_size,
            height=self._image_size,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=step_cb,
        ).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path
```

- [ ] **Step 2: Run the full suite to confirm nothing broke**

Run: `uv run pytest -v`
Expected: PASS (this module isn't imported by any test that runs without the `text2image` extra)

- [ ] **Step 3: Lint and commit**

Run: `uv run ruff check .`

```bash
git add src/animancer3d/pipelines/text2image.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Add JobCancelled and a cancel_event parameter to Text2Image.generate:
checked once after load() (download/load have no interruption point of
their own) and on every diffusers step callback during sampling.
EOF
)"
```

---

### Task 4: `pipelines/trellis.py` + `config.py` — reap crashed server, `--webp off` default

**Files:**
- Modify: `src/animancer3d/pipelines/trellis.py`
- Modify: `src/animancer3d/config.py`
- Create: `tests/test_trellis.py`

**Interfaces:**
- Consumes: `Config.trellis_webp: bool` (new field, Task added here).
- Produces: `TrellisServer.__init__(..., webp: bool = False)`, `TrellisServer._argv() -> list[str]`, `TrellisServer._reap_if_dead() -> None`. Task 6 (`queue.py`) passes `config.trellis_webp` when constructing `TrellisServer`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trellis.py`:

```python
from __future__ import annotations

import subprocess
import sys

from animancer3d.pipelines.trellis import TrellisServer


def test_argv_defaults_webp_off(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    assert srv._argv()[-2:] == ["--webp", "off"]


def test_argv_webp_on_when_configured(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971, webp=True)
    assert srv._argv()[-2:] == ["--webp", "on"]


def test_argv_includes_models_host_port(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    argv = srv._argv()
    assert argv[0] == str(tmp_path / "exe")
    assert "--models" in argv and str(tmp_path / "models") in argv
    assert "--port" in argv and "17971" in argv


def test_reap_if_dead_clears_an_already_exited_process(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    srv._proc = dead
    srv._reap_if_dead()
    assert srv._proc is None


def test_reap_if_dead_leaves_a_running_process_alone(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    running = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    try:
        srv._proc = running
        srv._reap_if_dead()
        assert srv._proc is running
    finally:
        running.kill()
        running.wait()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trellis.py -v`
Expected: FAIL — `AttributeError: 'TrellisServer' object has no attribute '_argv'`

- [ ] **Step 3: Implement `_argv()`, `_reap_if_dead()`, and the `webp` flag**

In `src/animancer3d/config.py`, add a new field to `Config` (after `t2i_image_size`):

```python
    # trellis-server.exe's WebP textures declare EXT_texture_webp as required,
    # which Godot's glTF importer does not implement (it refuses the file
    # rather than skip the extension). Off is the correct default.
    trellis_webp: bool = field(
        default_factory=lambda: os.environ.get("ANIMANCER3D_TRELLIS_WEBP", "off").lower()
        in ("1", "true", "on")
    )
```

In `src/animancer3d/pipelines/trellis.py`, change `TrellisServer.__init__` to accept and store `webp`:

```python
    def __init__(
        self,
        exe: Path,
        models_dir: Path,
        port: int,
        log_path: Path | None = None,
        webp: bool = False,
    ) -> None:
        self._exe = exe
        self._models_dir = models_dir
        self._port = port
        self._log_path = log_path
        self._webp = webp
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = asyncio.Lock()
        self.last_used = 0.0
        # Called for every decoded stdout line, on the reader thread.
        self.on_line: Callable[[str], None] | None = None
        self._reader: threading.Thread | None = None
        self._logfh = None
```

Add `_argv()` and `_reap_if_dead()` as new methods (place them right before `ensure_started`):

```python
    def _argv(self) -> list[str]:
        return [
            str(self._exe),
            "--models", str(self._models_dir),
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "--require-gpu",
            "--webp", "on" if self._webp else "off",
        ]

    def _reap_if_dead(self) -> None:
        """A self-crashed server otherwise leaks the old log handle and
        reader thread the next time _proc/_reader/_logfh are overwritten."""
        if self._proc is not None and self._proc.poll() is not None:
            log.warning(
                "trellis-server exited with code %s; reaping", self._proc.returncode
            )
            self.stop()
```

Update `ensure_started` to call `_reap_if_dead()` and to build its argv from `_argv()`. Replace:

```python
    async def ensure_started(self) -> None:
        async with self._lock:
            if self.running:
                return
```

with:

```python
    async def ensure_started(self) -> None:
        async with self._lock:
            self._reap_if_dead()
            if self.running:
                return
```

and replace the `subprocess.Popen([...])` call's argument list:

```python
            self._proc = subprocess.Popen(
                self._argv(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
```

- [ ] **Step 4: Wire `Config.trellis_webp` into `Worker`'s `TrellisServer` construction**

In `src/animancer3d/queue.py`, in `Worker.__init__`, change:

```python
        self.trellis = TrellisServer(
            config.trellis_server_exe,
            config.trellis_models_dir,
            config.trellis_port,
            log_path=config.data_dir / "trellis.log",
        )
```

to:

```python
        self.trellis = TrellisServer(
            config.trellis_server_exe,
            config.trellis_models_dir,
            config.trellis_port,
            log_path=config.data_dir / "trellis.log",
            webp=config.trellis_webp,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_trellis.py tests/test_api.py -v`
Expected: PASS

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check .`

```bash
git add src/animancer3d/pipelines/trellis.py src/animancer3d/config.py src/animancer3d/queue.py tests/test_trellis.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Default --webp off: trellis-server.exe's GLBs declare
EXT_texture_webp as required, which Godot's importer refuses rather
than skip. Gated by ANIMANCER3D_TRELLIS_WEBP for anyone who wants it
back. Also reap a self-crashed trellis-server in ensure_started
instead of leaking its log handle and reader thread when _proc is
overwritten.
EOF
)"
```

---

### Task 5: `doctor.py` + `animancer3d doctor` CLI command

**Files:**
- Create: `src/animancer3d/doctor.py`
- Modify: `src/animancer3d/cli.py`
- Test: `tests/test_doctor.py`

**Interfaces:**
- Consumes: `Config` (Task 4's `trellis_webp` field is irrelevant here; uses `trellis_server_exe`, `trellis_models_dir`, `trellis_port`, `data_dir`, `t2i_model_id`).
- Produces: `@dataclass Check(name: str, ok: bool, detail: str, fatal: bool)`, `run_checks(config: Config) -> list[Check]`. Task 8 (`app.py`) calls `run_checks` for `/api/health` and at startup.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_doctor.py`:

```python
from __future__ import annotations

import socket

from animancer3d.config import Config
from animancer3d.doctor import run_checks


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config(tmp_path, **overrides) -> Config:
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        trellis_port=_free_port(),
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def test_exe_check_reports_missing_exe_as_fatal(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["trellis-server.exe"].ok is False
    assert checks["trellis-server.exe"].fatal is True


def test_exe_check_passes_when_exe_exists(tmp_path):
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_server_exe=exe))}
    assert checks["trellis-server.exe"].ok is True


def test_gguf_check_finds_weight_files(tmp_path):
    models = tmp_path / "models"
    models.mkdir(parents=True)
    (models / "trellis.gguf").write_bytes(b"")
    checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_models_dir=models))}
    assert checks["TRELLIS GGUF weights"].ok is True


def test_gguf_check_reports_missing_dir_as_fatal(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["TRELLIS GGUF weights"].ok is False
    assert checks["TRELLIS GGUF weights"].fatal is True


def test_birefnet_check_is_not_fatal_when_missing(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["birefnet.gguf (background removal)"].ok is False
    assert checks["birefnet.gguf (background removal)"].fatal is False


def test_port_check_reports_a_free_port_as_ok(tmp_path):
    checks = {c.name: c for c in run_checks(_config(tmp_path))}
    assert checks["trellis port"].ok is True


def test_port_check_reports_a_bound_port_as_not_ok(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        checks = {c.name: c for c in run_checks(_config(tmp_path, trellis_port=port))}
        assert checks["trellis port"].ok is False


def test_run_checks_returns_seven_checks(tmp_path):
    assert len(run_checks(_config(tmp_path))) == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_doctor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'animancer3d.doctor'`

- [ ] **Step 3: Implement `doctor.py`**

Create `src/animancer3d/doctor.py`:

```python
"""Preflight checks: what's missing before you waste two minutes on a GPU job."""

from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

from .config import Config

MIN_FREE_DISK_GB = 5.0


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool


def run_checks(config: Config) -> list[Check]:
    return [
        _exe_check(config),
        _gguf_check(config),
        _birefnet_check(config),
        _cuda_check(),
        _disk_check(config),
        _port_check(config),
        _sdxl_cache_check(config),
    ]


def _exe_check(config: Config) -> Check:
    ok = config.trellis_server_exe.exists()
    detail = str(config.trellis_server_exe) if ok else f"not found at {config.trellis_server_exe}"
    return Check("trellis-server.exe", ok, detail, fatal=True)


def _gguf_check(config: Config) -> Check:
    ok = config.trellis_models_dir.exists() and any(config.trellis_models_dir.glob("*.gguf"))
    detail = (
        str(config.trellis_models_dir)
        if ok
        else f"no *.gguf found in {config.trellis_models_dir}"
    )
    return Check("TRELLIS GGUF weights", ok, detail, fatal=True)


def _birefnet_check(config: Config) -> Check:
    path = config.trellis_models_dir / "birefnet.gguf"
    ok = path.exists()
    detail = (
        str(path)
        if ok
        else f"missing at {path} -- background matting falls back to a threshold cutout"
    )
    return Check("birefnet.gguf (background removal)", ok, detail, fatal=False)


def _cuda_check() -> Check:
    try:
        import torch
    except ImportError:
        return Check(
            "CUDA", False, "torch not installed (uv sync --extra text2image)", fatal=False
        )
    ok = torch.cuda.is_available()
    return Check("CUDA", ok, "available" if ok else "torch.cuda.is_available() is False", fatal=False)


def _disk_check(config: Config) -> Check:
    free_gb = shutil.disk_usage(config.data_dir).free / (1024**3)
    ok = free_gb >= MIN_FREE_DISK_GB
    return Check("free disk space", ok, f"{free_gb:.1f} GB free in {config.data_dir}", fatal=False)


def _port_check(config: Config) -> Check:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", config.trellis_port))
            ok, detail = True, f"port {config.trellis_port} is free"
        except OSError as exc:
            ok, detail = False, f"port {config.trellis_port} unavailable: {exc}"
    return Check("trellis port", ok, detail, fatal=False)


def _sdxl_cache_check(config: Config) -> Check:
    home = os.environ.get("HF_HOME")
    hub = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"
    ok = (hub / f"models--{config.t2i_model_id.replace('/', '--')}").exists()
    detail = "cached" if ok else "not cached yet -- first text job downloads ~7 GB"
    return Check("SDXL-Turbo cache", ok, detail, fatal=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_doctor.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Wire `animancer3d doctor` into `cli.py`**

Replace the full contents of `src/animancer3d/cli.py`:

```python
"""Entry point: `animancer3d` starts the local server; `animancer3d doctor` checks setup."""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Animancer3D — local AI 3D asset generator")
    parser.add_argument(
        "command", nargs="?", choices=["doctor"], default=None,
        help="omit to start the server; 'doctor' checks dependencies and configuration",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.command == "doctor":
        _run_doctor()
        return

    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


def _run_doctor() -> None:
    from .config import get_config
    from .doctor import run_checks

    config = get_config()
    checks = run_checks(config)
    for check in checks:
        status = "OK" if check.ok else ("FATAL" if check.fatal else "WARN")
        print(f"[{status}] {check.name}: {check.detail}")
    if any(not c.ok and c.fatal for c in checks):
        raise SystemExit(1)
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 7: Lint and commit**

Run: `uv run ruff check .`

```bash
git add src/animancer3d/doctor.py src/animancer3d/cli.py tests/test_doctor.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Add doctor.py: seven pure preflight checks (exe, GGUF weights,
birefnet, CUDA, disk space, port, SDXL cache) that tell you what's
missing before a job wastes two minutes of GPU time. Wire it up as
`animancer3d doctor`.
EOF
)"
```

---

### Task 6: `queue.py` — real cancellation, crash resilience, clean shutdown

**Files:**
- Modify: `src/animancer3d/queue.py`

**Interfaces:**
- Consumes: `JobStore.claim` (Task 2), `Text2Image.generate(..., cancel_event=...)` and `JobCancelled` (Task 3, imported lazily alongside `Text2Image`), `errors.friendly` / `errors.write_error_log` (Task 1), `TrellisServer(..., webp=...)` (Task 4, already wired).
- Produces: `Worker.request_cancel(job_id: str) -> None` (async), `Worker.fatal: BaseException | None`, `Worker.alive: bool` (property). Task 7's `test_queue.py` exercises all three; Task 8 (`app.py`) calls `request_cancel` from the cancel route and reads `alive`/`fatal` from `/api/health`.

This task has no dedicated unit test file of its own — `Worker` needs the `fake_pipelines` fixture (Task 7) to be testable without a GPU, and that fixture needs `Worker`'s final shape to patch against correctly. Verify this task by re-running the existing suite (nothing here is covered yet) and by a careful read-through against the steps below; Task 7 is the real test coverage for this file and must not be skipped.

- [ ] **Step 1: Rewrite `queue.py`**

Replace the full contents of `src/animancer3d/queue.py`:

```python
"""Single-worker GPU job queue.

One job runs at a time. VRAM handoff for text jobs: the trellis server is stopped
before Flux loads (both don't fit alongside each other), and Flux is unloaded
before the trellis server starts.

Cancellation has no HTTP counterpart on trellis-server.exe (it exposes exactly
/generate and /health) and aborting the client request does not stop the GPU.
The only mechanism that actually frees VRAM mid-run is killing the subprocess
(TrellisServer.stop()), which this module already does for the VRAM handoff.
A cancel during the text2image stage instead sets a threading.Event that the
diffusers step callback checks every step (see pipelines/text2image.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import errors
from .config import Config
from .db import JobStore
from .pipelines.trellis import TrellisServer
from .progress import ProgressBus, TrellisProgressParser

log = logging.getLogger(__name__)

POLL_INTERVAL = 1.0
SHUTDOWN_TIMEOUT = 20.0

# Labels for the text-to-image phases, which have no trace of their own.
T2I_PHASES = {
    "download": ("t2i_download", "Downloading SDXL-Turbo (~7 GB, first run only)"),
    "load": ("t2i_load", "Loading image model"),
    "sample": ("t2i_sample", "Drawing reference image"),
}


@dataclass
class _Cancel:
    job_id: str
    event: threading.Event = field(default_factory=threading.Event)


class Worker:
    def __init__(self, config: Config, store: JobStore) -> None:
        self.config = config
        self.store = store
        self.trellis = TrellisServer(
            config.trellis_server_exe,
            config.trellis_models_dir,
            config.trellis_port,
            log_path=config.data_dir / "trellis.log",
            webp=config.trellis_webp,
        )
        self._text2image = None  # lazy: torch/diffusers may not be installed
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.current_job_id: str | None = None
        self._cancel: _Cancel | None = None
        self.fatal: BaseException | None = None
        self.progress = ProgressBus()
        self._parser = TrellisProgressParser(self._emit_progress)
        self.trellis.on_line = self._parser.feed

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="gpu-worker")
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.fatal = exc
            log.critical("gpu worker task died", exc_info=exc)

    async def request_cancel(self, job_id: str) -> None:
        """No-op unless job_id is the job currently running."""
        if job_id != self.current_job_id or self._cancel is None:
            return
        self._cancel.event.set()
        snapshot = self.progress.snapshot()
        phase = snapshot["phase"] if snapshot else None
        if phase == "trellis":
            # The only real "abort" trellis-server.exe has: kill it. The
            # in-flight client.post then dies with a TransportError, which
            # _process below turns into a cancelled status because the
            # cancel event is already set.
            await asyncio.to_thread(self.trellis.stop)
        # t2i_sample: the diffusers step callback checks the event itself.
        # t2i_load / t2i_download: not interruptible; the event is checked
        # once between load() and sampling in Text2Image.generate().

    async def shutdown(self) -> None:
        self._stop.set()
        if self.current_job_id is not None:
            await self.request_cancel(self.current_job_id)
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=SHUTDOWN_TIMEOUT)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        self.trellis.stop()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = await asyncio.to_thread(self.store.next_queued)
                if job is None:
                    self._maybe_evict_idle()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._stop.wait(), timeout=POLL_INTERVAL)
                    continue
                await self._process(job)
            except Exception:
                # A crash here used to kill the worker permanently and
                # silently -- next_queued or a DB hiccup would strand every
                # future job in 'queued' forever with no error surfaced.
                log.exception("worker loop iteration failed")
            if self._stop.is_set():
                break

    def _maybe_evict_idle(self) -> None:
        if (
            self.trellis.running
            and time.monotonic() - self.trellis.last_used > self.config.trellis_idle_timeout
        ):
            log.info("evicting idle trellis-server")
            self.trellis.stop()

    # --- progress plumbing ---

    def _emit_progress(
        self,
        phase: str,
        label: str,
        inner: float,
        inner_next: float | None,
        nominal: float,
        fields: dict[str, Any],
    ) -> None:
        """Sink for the trellis parser. Runs on the stdout reader thread."""
        job_id = self.current_job_id
        if job_id is None:
            return  # server chatter outside a job (startup banner, idle logs)
        self.progress.update(
            job_id,
            phase=phase,
            label=label,
            inner=inner,
            inner_next=inner_next,
            nominal=nominal,
            **fields,
        )

    def _t2i_state(self, job_id: str, state: str) -> None:
        phase, label = T2I_PHASES.get(state, ("t2i_load", "Preparing"))
        # Download and load have no measurable inner progress; creep across the
        # whole phase so the bar still moves.
        self.progress.update(
            job_id, phase=phase, label=label, inner=0.0, inner_next=1.0,
            nominal=90.0 if state == "download" else 20.0, detail="",
        )

    def _t2i_step(self, job_id: str, step: int, total: int) -> None:
        self.progress.update(
            job_id,
            phase="t2i_sample",
            label="Drawing reference image",
            inner=step / max(total, 1),
            inner_next=(step + 1) / max(total, 1),
            nominal=1.0,
            detail=f"step {step}/{total}",
            step=step,
            step_total=total,
        )

    async def _process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        claimed = await asyncio.to_thread(self.store.claim, job_id)
        if not claimed:
            # Cancelled or deleted between next_queued() and here.
            return
        self.current_job_id = job_id
        self._cancel = _Cancel(job_id)
        # A cold trellis server loads ~8 GB inside its first stage; text jobs stop
        # the server outright, so they are always cold.
        self.progress.begin(
            job_id, job["kind"], cold=job["kind"] == "text" or not self.trellis.running
        )
        error: str | None = None
        try:
            await self._generate(job)
        except Exception as exc:
            if not self._cancel.event.is_set():
                log.exception("job %s failed", job_id)
                errors.write_error_log(self.config.job_dir(job_id), exc)
                error = errors.friendly(exc)
        finally:
            if self._cancel.event.is_set():
                await asyncio.to_thread(self.store.set_status, job_id, "cancelled")
                glb = self.config.job_dir(job_id) / "model.glb"
                with contextlib.suppress(OSError):
                    glb.unlink()
            elif error is not None:
                await asyncio.to_thread(self.store.set_status, job_id, "error", error)
            else:
                await asyncio.to_thread(self.store.set_status, job_id, "done")
            self.current_job_id = None
            self._cancel = None
            self.progress.end(job_id)

    async def _generate(self, job: dict[str, Any]) -> None:
        job_dir = self.config.job_dir(job["id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        params = job["params"]
        seed = int(params.get("seed", 42))
        resolution = int(params.get("resolution", 1024))
        image_path = job_dir / "input.png"

        job_id = job["id"]
        assert self._cancel is not None
        if job["kind"] == "text":
            # Free VRAM held by the 3D server, run Flux, then free Flux.
            self.trellis.stop()
            t2i = self._get_text2image()
            try:
                await asyncio.to_thread(
                    functools.partial(
                        t2i.generate,
                        job["prompt"],
                        image_path,
                        seed=seed,
                        on_state=lambda s: self._t2i_state(job_id, s),
                        on_step=lambda i, n: self._t2i_step(job_id, i, n),
                        cancel_event=self._cancel.event,
                    )
                )
            finally:
                await asyncio.to_thread(t2i.unload)
        elif not image_path.exists():
            raise RuntimeError("image job has no uploaded input.png")

        if self._cancel.event.is_set():
            return

        self.progress.update(
            job_id,
            phase="trellis",
            label="Starting 3D engine" if not self.trellis.running else "Sending image",
            inner=0.0,
            inner_next=0.02,
            nominal=6.0,
            detail="",
        )
        await self.trellis.generate(
            image_path, job_dir / "model.glb", seed=seed, resolution=resolution
        )

    def _get_text2image(self):
        if self._text2image is None:
            try:
                from .pipelines.text2image import Text2Image
            except ImportError as exc:
                raise RuntimeError(
                    "text-to-3D requires the text2image extra: uv sync --extra text2image"
                ) from exc
            self._text2image = Text2Image(
                self.config.t2i_model_id, self.config.t2i_image_size
            )
        return self._text2image
```

Notes on the diff from the current file: `_process` now claims the job atomically instead of unconditionally setting it to `running`; the terminal status is derived from `self._cancel.event`, not a second DB read; `_generate` returns early (without starting `trellis.generate`) if a text-job cancel landed during t2i; `_run` wraps its body in `try/except Exception` and breaks immediately if `_stop` is set right after processing; `start()`/`shutdown()` gained the fatal-tracking and bounded-wait behavior described in the module docstring.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS — this only changes internal `Worker` behavior; no existing test constructs a real `Worker` against a real `trellis-server.exe`, so nothing should break. `test_progress_reports_the_running_job` and friends manipulate `worker.progress`/`worker.current_job_id` directly and remain valid.

- [ ] **Step 3: Lint and commit**

Run: `uv run ruff check .`

```bash
git add src/animancer3d/queue.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Make cancellation real: request_cancel() dispatches per phase (kills
trellis-server.exe mid-3D-stage, sets a threading.Event checked by the
text2image step callback and once after load), and _process derives
the terminal status from that event instead of a second DB read, so a
cancelled job never leaves a viewable model.glb behind. Claim jobs
atomically via JobStore.claim() to close the race with a concurrent
cancel. Wrap the worker loop in try/except so a single job's crash
can't strand every future job in 'queued' forever, and give shutdown()
a bounded wait instead of hanging on Ctrl+C.
EOF
)"
```

---

### Task 7: `conftest.py` fixture + `test_queue.py`

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_queue.py`

**Interfaces:**
- Consumes: `Worker`, `_Cancel` behavior from Task 6; `JobCancelled` from Task 3.
- Produces: `fake_pipelines` pytest fixture (autouse-free; requested explicitly by tests that construct a `Worker`).

- [ ] **Step 1: Write `conftest.py`**

Create `tests/conftest.py`:

```python
"""Shared fixtures. fake_pipelines replaces the GPU-bound pieces (a real
trellis-server.exe subprocess, real torch/diffusers) with in-process fakes
so Worker's control flow -- cancellation, crash recovery, shutdown -- is
testable without a GPU."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from animancer3d.pipelines.text2image import JobCancelled


class FakeTrellisServer:
    """Stands in for TrellisServer: no subprocess, no GPU, no HTTP."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.running = False
        self.last_used = 0.0
        self.on_line = None
        self.stop_calls = 0
        self.generate_calls: list[dict] = []
        self.slices = 5
        self.sleep_per_slice = 0.02
        self.should_raise: Exception | None = None

    async def generate(
        self, image_path: Path, output_path: Path, *, seed: int = 42, resolution: int = 1024
    ) -> Path:
        self.generate_calls.append(
            {"image_path": image_path, "seed": seed, "resolution": resolution}
        )
        if self.should_raise is not None:
            exc, self.should_raise = self.should_raise, None
            raise exc
        self.running = True
        for _ in range(self.slices):
            await asyncio.sleep(self.sleep_per_slice)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-glb")
        self.last_used = time.monotonic()
        return output_path

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


class FakeText2Image:
    """Stands in for Text2Image: no torch, no diffusers."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.loaded = False
        self.unload_calls = 0
        self.steps = 3
        self.sleep_per_step = 0.02

    def generate(
        self,
        prompt,
        output_path,
        *,
        seed=42,
        on_state=None,
        on_step=None,
        cancel_event=None,
    ):
        if on_state is not None:
            on_state("load")
        self.loaded = True
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled
        if on_state is not None:
            on_state("sample")
        for i in range(self.steps):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled
            time.sleep(self.sleep_per_step)
            if on_step is not None:
                on_step(i + 1, self.steps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-png")
        return output_path

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


@pytest.fixture
def fake_pipelines(monkeypatch):
    """Patch the GPU pipeline classes at their definition. Worker.__init__
    constructs `TrellisServer(...)` via the name imported into queue.py's
    namespace; Worker._get_text2image does `from .pipelines.text2image
    import Text2Image` fresh on every call, so patching the attribute on
    the text2image module is picked up immediately without touching queue.py."""
    import animancer3d.pipelines.text2image as text2image_mod
    import animancer3d.queue as queue_mod

    monkeypatch.setattr(queue_mod, "TrellisServer", FakeTrellisServer)
    monkeypatch.setattr(text2image_mod, "Text2Image", FakeText2Image)
```

- [ ] **Step 2: Write `test_queue.py`**

Create `tests/test_queue.py`:

```python
from __future__ import annotations

import asyncio
import time

import pytest

from animancer3d.config import Config
from animancer3d.db import JobStore
from animancer3d.queue import Worker

pytestmark = pytest.mark.asyncio


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    yield w
    store.close()


async def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


def _make_image_job(worker: Worker) -> str:
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    return job_id


async def test_shutdown_with_job_running_returns_promptly(worker):
    # Written first: cancelling from inside teardown while the fake "GPU"
    # work is in flight is the fiddliest path in this file.
    _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.trellis.running)

    start = time.monotonic()
    await asyncio.wait_for(worker.shutdown(), timeout=20.0)
    assert time.monotonic() - start < 20.0


async def test_cancel_mid_trellis_stops_process_and_leaves_no_glb(worker):
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.trellis.running)

    await worker.request_cancel(job_id)
    await _wait_until(lambda: worker.store.get(job_id)["status"] != "running")
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "cancelled"
    assert worker.trellis.stop_calls >= 1
    assert not (worker.config.job_dir(job_id) / "model.glb").exists()


async def test_cancel_mid_t2i_never_starts_trellis(worker):
    job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
    worker.start()
    await _wait_until(lambda: worker.current_job_id == job_id)

    await worker.request_cancel(job_id)
    await _wait_until(lambda: worker.store.get(job_id)["status"] != "running")
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "cancelled"
    assert worker.trellis.generate_calls == []


async def test_exception_in_generate_marks_error_and_worker_survives(worker):
    bad_id = _make_image_job(worker)
    worker.trellis.should_raise = RuntimeError("boom")
    good_id = _make_image_job(worker)

    worker.start()
    await _wait_until(lambda: worker.store.get(good_id)["status"] in ("done", "error"))
    await worker.shutdown()

    bad_job = worker.store.get(bad_id)
    assert bad_job["status"] == "error"
    assert bad_job["error"] == "boom"
    assert worker.store.get(good_id)["status"] == "done"


async def test_worker_picks_up_the_next_queued_job_after_a_completed_one(worker):
    first = _make_image_job(worker)
    second = _make_image_job(worker)

    worker.start()
    await _wait_until(lambda: worker.store.get(second)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(first)["status"] == "done"
    assert worker.store.get(second)["status"] == "done"
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_queue.py -v`
Expected: PASS (6 passed). If `test_shutdown_with_job_running_returns_promptly` or the cancel tests hang, check that `Worker.request_cancel` in Task 6 is actually `async def` and that `shutdown()` awaits it before the bounded wait — a sync `request_cancel` silently returning a coroutine object without awaiting it would look like it worked but never call `trellis.stop()`.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS, all files

- [ ] **Step 5: Lint and commit**

Run: `uv run ruff check .`

```bash
git add tests/conftest.py tests/test_queue.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Add a fake_pipelines fixture (no subprocess, no GPU, no torch) and
test_queue.py covering the paths that had zero coverage before this
phase: cancel mid-trellis stops the process and leaves no model.glb,
cancel mid-t2i never starts trellis, an exception in one job doesn't
strand the worker, and shutdown() with a job running returns well
under its 20s budget.
EOF
)"
```

---

### Task 8: `app.py` — creation ordering, alpha handling, cancel dispatch, health, startup wiring

**Files:**
- Modify: `src/animancer3d/app.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `JobStore.create(..., job_id=...)`, `JobStore.reconcile_startup` (Task 2); `doctor.run_checks` (Task 5); `Worker.request_cancel`, `Worker.alive`, `Worker.fatal` (Task 6).

- [ ] **Step 1: Write the new/changed tests**

In `tests/test_api.py`, replace the `test_cancel_and_delete` test (it currently accepts either of two outcomes and verifies nothing) with a deterministic one, and add four new tests. Replace:

```python
def test_cancel_and_delete(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code in (200, 409)
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
```

with:

```python
def test_cancel_and_delete(client):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_cancel_running_job_calls_request_cancel(client, monkeypatch):
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    worker = client.app.state.worker
    worker.store.set_status(job_id, "running")
    worker.current_job_id = job_id
    called = []

    async def fake_request_cancel(jid):
        called.append(jid)

    monkeypatch.setattr(worker, "request_cancel", fake_request_cancel)
    try:
        r = client.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 200
        assert called == [job_id]
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"
    finally:
        worker.current_job_id = None


def test_input_png_written_before_db_row_is_created(client, tmp_path, monkeypatch):
    from animancer3d.db import JobStore

    original_create = JobStore.create
    seen = {}

    def spying_create(self, kind, prompt, params, job_id=None):
        if job_id is not None:
            path = tmp_path / "assets" / job_id / "input.png"
            seen["exists"] = path.exists()
        return original_create(self, kind, prompt, params, job_id=job_id)

    monkeypatch.setattr(JobStore, "create", spying_create)

    r = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert r.status_code == 200
    assert seen.get("exists") is True


def test_opaque_jpeg_upload_preserves_rgb(client, tmp_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buf, "JPEG")
    r = client.post(
        "/api/jobs",
        data={"kind": "image"},
        files={"image": ("ref.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")},
    )
    assert r.status_code == 200
    stored = tmp_path / "assets" / r.json()["id"] / "input.png"
    assert Image.open(stored).mode == "RGB"


def test_health_reports_worker_alive_and_doctor_checks(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["worker_alive"] is True
    assert body["fatal"] is None
    assert isinstance(body["checks"], list)
    assert len(body["checks"]) == 7
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `test_health_reports_worker_alive_and_doctor_checks` (no `worker_alive`/`checks` keys yet), `test_cancel_running_job_calls_request_cancel` (`request_cancel` not yet awaited from the route), `test_input_png_written_before_db_row_is_created` (current ordering writes the file after `store().create`)

- [ ] **Step 3: Implement the `app.py` changes**

Replace the full contents of `src/animancer3d/app.py`:

```python
"""FastAPI app: job API + static web UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import doctor
from .config import get_config
from .db import JobStore
from .queue import Worker

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_RESOLUTIONS = {512, 1024, 1536}


def create_app() -> FastAPI:
    config = get_config()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        store = JobStore(config.db_path)
        # A job still 'running' at process start was orphaned by a crash or
        # an unclean shutdown -- surface it instead of silently re-running
        # a 2-minute GPU job on every restart.
        await asyncio.to_thread(store.reconcile_startup)
        for check in await asyncio.to_thread(doctor.run_checks, config):
            if not check.ok:
                level = log.critical if check.fatal else log.warning
                level("doctor: %s -- %s", check.name, check.detail)
        worker = Worker(config, store)
        worker.start()
        app.state.store = store
        app.state.worker = worker
        try:
            yield
        finally:
            await worker.shutdown()
            store.close()

    app = FastAPI(title="Animancer3D", lifespan=lifespan)

    def store() -> JobStore:
        return app.state.store

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        worker: Worker = app.state.worker
        checks = await asyncio.to_thread(doctor.run_checks, config)
        return {
            "ok": worker.alive and worker.fatal is None,
            "worker_alive": worker.alive,
            "fatal": str(worker.fatal) if worker.fatal else None,
            "trellis_running": worker.trellis.running,
            "checks": [asdict(c) for c in checks],
        }

    @app.get("/api/progress")
    async def progress() -> dict[str, Any]:
        """Live progress for the running job. Cheap by design: no DB, no disk.

        The UI polls this every ~600 ms while a job is active, and the full job
        list only every few seconds.
        """
        worker = app.state.worker
        return {
            "job_id": worker.current_job_id,
            "progress": worker.progress.snapshot(),
            # Lets the client correct for clock skew when rendering elapsed time.
            "server_time": time.time(),
        }

    @app.post("/api/jobs")
    async def create_job(
        kind: Annotated[str, Form()],
        prompt: Annotated[str | None, Form()] = None,
        seed: Annotated[int, Form()] = 42,
        resolution: Annotated[int, Form()] = 1024,
        image: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        if kind not in ("text", "image"):
            raise HTTPException(400, "kind must be 'text' or 'image'")
        if resolution not in ALLOWED_RESOLUTIONS:
            raise HTTPException(400, f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
        if kind == "text" and not (prompt and prompt.strip()):
            raise HTTPException(400, "text jobs require a prompt")
        if kind == "image" and image is None:
            raise HTTPException(400, "image jobs require an image upload")

        normalized: bytes | None = None
        if image is not None:
            data = await image.read()
            try:
                normalized = await asyncio.to_thread(_to_png, data)
            except Exception as exc:
                raise HTTPException(400, "could not decode uploaded image") from exc

        # Write the file before the row exists: the worker's next_queued()
        # poll can otherwise claim an image job in the gap and find no
        # input.png on disk yet.
        job_id = uuid.uuid4().hex[:12]
        if normalized is not None:
            job_dir = config.job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "input.png").write_bytes(normalized)
        params = {"seed": seed, "resolution": resolution}
        await asyncio.to_thread(store().create, kind, prompt, params, job_id)
        return {"id": job_id}

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        jobs = await asyncio.to_thread(store().list)
        for job in jobs:
            _attach_files(job, config.job_dir(job["id"]))
            _attach_progress(job, app.state.worker)
        return jobs

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        _attach_files(job, config.job_dir(job_id))
        _attach_progress(job, app.state.worker)
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] not in ("queued", "running"):
            raise HTTPException(409, f"job is {job['status']}")
        if job["status"] == "running":
            await app.state.worker.request_cancel(job_id)
        # A running job finishes its current GPU stage; the worker preserves the
        # cancelled status instead of marking it done.
        await asyncio.to_thread(store().set_status, job_id, "cancelled")
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] == "running":
            raise HTTPException(409, "cancel the job before deleting it")
        await asyncio.to_thread(store().delete, job_id)
        shutil.rmtree(config.job_dir(job_id), ignore_errors=True)
        return {"ok": True}

    _MEDIA = {
        "model.glb": "model/gltf-binary",
        "input.png": "image/png",
        "model.stl": "model/stl",
        "model_obj.zip": "application/zip",
    }

    @app.get("/api/jobs/{job_id}/files/{name}")
    async def get_file(job_id: str, name: str) -> FileResponse:
        if name not in _MEDIA:
            raise HTTPException(404, "unknown file")
        job_dir = config.job_dir(job_id)
        path = job_dir / name
        glb = job_dir / "model.glb"
        if not path.exists() and name in ("model.stl", "model_obj.zip") and glb.exists():
            from .pipelines import postprocess

            convert = (
                postprocess.glb_to_stl if name == "model.stl" else postprocess.glb_to_obj_zip
            )
            await asyncio.to_thread(convert, glb, path)
        if not path.exists():
            raise HTTPException(404, "file not ready")
        return FileResponse(path, media_type=_MEDIA[name], filename=f"{job_id}_{name}")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _to_png(data: bytes) -> bytes:
    """Re-encode any uploaded image as PNG; trellis.cpp only decodes PNG/JPEG.

    Alpha is preserved only when the source already had it, so a pre-matted
    upload (RGBA/LA/PA, or a palette image with a transparency entry) keeps
    its alpha for the server's bg-removal auto-detection, without forcing an
    opaque photo through the same path.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        out = io.BytesIO()
        im.convert("RGBA" if has_alpha else "RGB").save(out, "PNG")
        return out.getvalue()


def _attach_files(job: dict[str, Any], job_dir: Path) -> None:
    job["files"] = [
        name for name in ("input.png", "model.glb") if (job_dir / name).exists()
    ]


def _attach_progress(job: dict[str, Any], worker: Any) -> None:
    """Live progress, only ever for the job actually running."""
    job["progress"] = worker.progress.snapshot(job["id"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (all tests, including the 4 new ones and the rewritten `test_cancel_and_delete`)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff check .`

```bash
git add src/animancer3d/app.py tests/test_api.py
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Write input.png before the DB row exists, closing a race where the
worker's poll could claim an image job with no file on disk yet.
Preserve upload alpha only when the source actually had it, instead of
forcing every JPEG through bg-removal's pre-matted path. Cancel now
awaits Worker.request_cancel for a running job so the GPU actually
stops. Run reconcile_startup() before the worker starts so a crash
doesn't silently re-run a 2-minute job on restart, and log doctor
warnings at startup. Rewrite /api/health to report worker_alive,
fatal, and the seven doctor checks instead of an unconditional
ok: True.
EOF
)"
```

---

### Task 9: `static/app.js` + `static/index.html` — cancel feedback, disconnected banner, doctor banner

**Files:**
- Modify: `src/animancer3d/static/index.html`
- Modify: `src/animancer3d/static/app.js`

**Interfaces:** none (pure UI; no test runner for this project's frontend — see `CLAUDE.md`'s no-build-step rule). Verify by eye per the manual steps below; there is no automated test for this task.

- [ ] **Step 1: Add banner markup and styles to `index.html`**

In `src/animancer3d/static/index.html`, add a banner style block right before the closing `</style>` (after the `#downloads a:hover` rule, around line 119):

```css
  #banner {
    position: fixed; top: 0; left: 0; right: 0; z-index: 20;
    display: flex; align-items: center; gap: 12px;
    background: var(--err); color: #fff; font-size: 13px;
    padding: 8px 16px;
  }
  #banner button {
    margin-left: auto; background: none; border: none; color: inherit;
    cursor: pointer; font-size: 16px; line-height: 1; padding: 0 4px;
  }
```

Add the banner element as the first child of `<body>` (immediately after `<body>` on line 122, before `<aside>`):

```html
<div id="banner" hidden>
  <span id="banner-text"></span>
  <button id="banner-close" type="button" aria-label="dismiss">&times;</button>
</div>
```

- [ ] **Step 2: Cancel button shows "cancelling…" and disables during the request**

In `src/animancer3d/static/app.js`, find the `act.addEventListener("click", ...)` block (around line 201) and replace it:

```js
  act.addEventListener("click", async (e) => {
    e.stopPropagation();
    const job = jobsById.get(id);
    if (!job) return;
    const active = job.status === "queued" || job.status === "running";
    if (active) {
      act.disabled = true;
      setText(act, "cancelling…");
    }
    try {
      await fetch(`/api/jobs/${id}${active ? "/cancel" : ""}`, {
        method: active ? "POST" : "DELETE",
      });
    } finally {
      act.disabled = false;
    }
    if (!active && selected === id) {
      selected = null;
      hideOverlay();
      downloads.style.display = "none";
    }
    poll(true);
  });
```

- [ ] **Step 3: Add the banner helpers and wire them into `poll()`**

Near the top of `src/animancer3d/static/app.js`, after the existing top-level `const`s used for DOM lookups (find where `nodes`/`jobsById` or similar module-level state is declared, and add alongside it — search for `const nodes = new Map();` around line 170), add:

```js
// --- banner --------------------------------------------------------------
// A single dismissible slot; the last shown "kind" wins so a doctor warning
// isn't clobbered by a transient disconnect and vice versa.

const banner = document.getElementById("banner");
const bannerText = document.getElementById("banner-text");
const bannerClose = document.getElementById("banner-close");
const dismissedBanners = new Set();
let activeBannerKind = null;

function showBanner(kind, text) {
  if (dismissedBanners.has(kind)) return;
  activeBannerKind = kind;
  setText(bannerText, text);
  banner.hidden = false;
}

function hideBanner(kind) {
  if (activeBannerKind !== kind) return;
  banner.hidden = true;
  activeBannerKind = null;
}

bannerClose.addEventListener("click", () => {
  if (activeBannerKind) dismissedBanners.add(activeBannerKind);
  banner.hidden = true;
  activeBannerKind = null;
});
```

In the `poll()` function (around line 315), add failure tracking. Replace:

```js
async function poll(forceList = false) {
  if (forceList) wantList = true;
  if (inFlight) return;
  inFlight = true;
  try {
    const r = await fetch("/api/progress", { signal: abort.signal });
    const data = await r.json();
    skew = data.server_time - Date.now() / 1000;
```

with:

```js
let pollFailures = 0;

async function poll(forceList = false) {
  if (forceList) wantList = true;
  if (inFlight) return;
  inFlight = true;
  try {
    const r = await fetch("/api/progress", { signal: abort.signal });
    const data = await r.json();
    pollFailures = 0;
    hideBanner("disconnected");
    skew = data.server_time - Date.now() / 1000;
```

and replace the `catch` block:

```js
  } catch (e) {
    if (e.name !== "AbortError") console.error("poll failed", e);
  } finally {
```

with:

```js
  } catch (e) {
    if (e.name !== "AbortError") {
      console.error("poll failed", e);
      pollFailures++;
      if (pollFailures >= 3) showBanner("disconnected", "Lost connection to the server — retrying…");
    }
  } finally {
```

- [ ] **Step 4: Check `/api/health` once at startup for fatal doctor findings**

At the bottom of `src/animancer3d/static/app.js`, right before the `(function loop() { ... })();` IIFE (around line 405), add:

```js
async function checkDoctor() {
  try {
    const health = await (await fetch("/api/health")).json();
    const bad = (health.checks || []).filter((c) => c.fatal && !c.ok);
    if (bad.length) {
      showBanner("doctor", `Setup problem: ${bad.map((c) => c.detail).join("; ")}`);
    }
  } catch (e) {
    console.error("doctor check failed", e);
  }
}
checkDoctor();
```

- [ ] **Step 5: Manual verification (no automated test for this file)**

Run: `uv run animancer3d --port 8420`, open `http://127.0.0.1:8420` in a browser:
- Submit a text job, click "cancel" while it's queued/running — the button should read "cancelling…" and be disabled for the duration of the request, then the job's status badge should read "cancelling" (existing UI logic) until the worker finishes tearing down.
- Rename `vendor/trellis/trellis-server.exe` temporarily and reload the page — a red banner reading "Setup problem: ..." should appear at the top, dismissible with the × button. Rename it back afterward.
- Stop the server process while the page is open and confirm a "Lost connection to the server — retrying…" banner appears within ~2 seconds (3 failed polls at 600ms).

- [ ] **Step 6: Run the backend suite once more (nothing here should have touched Python)**

Run: `uv run pytest -v && uv run ruff check .`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/animancer3d/static/index.html src/animancer3d/static/app.js
git commit -m "$(cat <<'EOF'
Animancer3D v0.0.1

Cancel button shows "cancelling…" and disables for the duration of
the request. Add a dismissible top banner: a doctor check at startup
surfaces fatal setup problems (missing exe, missing weights), and
three consecutive failed /api/progress polls surface a "lost
connection" warning instead of only logging to the console.
EOF
)"
```

---

## Manual checks after all tasks (cannot be automated; requires the real exe and a GPU)

These are the plan's own acceptance criteria for Phase 1 — run them once Tasks 1-9 are all committed and reviewed, using the real `vendor/trellis/trellis-server.exe`:

1. Start a res-1536 image job, cancel it around 50% through the trellis phase. Confirm GPU utilization (Task Manager / `nvidia-smi`) drops within a few seconds of clicking cancel, the job's final status is `cancelled`, and no `model.glb` exists in its job directory.
2. Start a text job, Ctrl+C the server process while it's running. Confirm the process exits within ~20 seconds instead of hanging.
3. Rename `vendor/trellis/trellis-server.exe` to something else, run `uv run animancer3d doctor` — confirm it prints `[FATAL] trellis-server.exe: not found at ...` and exits non-zero. Rename it back.
4. Confirm `uv run animancer3d doctor` reports the birefnet check as a non-fatal `[WARN]` when `birefnet.gguf` is absent from `models/trellis2-gguf/`.
