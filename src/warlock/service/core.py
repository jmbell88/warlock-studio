"""The handle every service function takes: config, store, worker, loop.

Deliberately thin. It owns exactly the three things a service function cannot
get for itself -- the process's config and store, the worker (which lives on
another thread's event loop and so needs a hop to reach), and the per-artifact
lock table -- and no business logic at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from ..config import Config
from ..db import JobStore
from .errors import NotFound
from .validation import check_job_id


class WarlockService:
    """Shared state for the service functions.

    ``worker`` and ``loop`` are optional so a test (or a headless tool) can
    exercise the pure-DB half without standing up the GPU queue; the functions
    that genuinely need the worker say so by raising rather than by having a
    non-optional field nothing else can satisfy.
    """

    def __init__(
        self,
        config: Config,
        store: JobStore,
        worker: Any = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.worker = worker
        self.loop = loop
        # One lock per derived artifact, so the same STL/OBJ/posed GLB is
        # never converted twice concurrently. Created lazily and never
        # evicted: an idle lock is a few dozen bytes and the key space is
        # jobs x a handful.
        #
        # threading.Lock rather than asyncio.Lock: every caller is now a plain
        # worker thread, the protected work is blocking anyway (Blender is
        # ~1 s, a trimesh export longer), and this keeps those waits off the
        # event loop that drives dispatch and cancellation.
        self._convert_locks: dict[tuple[str, str], threading.Lock] = {}
        self._locks_lock = threading.Lock()
        # service.system's doctor-check cache. Here rather than in that module
        # so it dies with the process's config, not with the interpreter.
        self.health_cache: dict[str, Any] = {}

    # -- paths -------------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        return self.config.job_dir(job_id)

    # -- locks -------------------------------------------------------------

    def convert_lock(self, job_id: str, name: str) -> threading.Lock:
        """The lock guarding one derived artifact of one job.

        Get-or-create under a meta-lock: unlike the asyncio version this
        replaces, two threads genuinely can reach the ``setdefault`` at once,
        and two locks for one artifact is exactly the double conversion the
        table exists to prevent.
        """
        key = (job_id, name)
        with self._locks_lock:
            lock = self._convert_locks.get(key)
            if lock is None:
                lock = self._convert_locks[key] = threading.Lock()
            return lock

    # -- worker ------------------------------------------------------------

    def wake_worker(self) -> None:
        """Dispatch immediately instead of on the next poll tick.

        Every function that inserts a queued row calls this. It is advisory:
        the worker still polls on its own interval, so a missed call costs a
        second of latency rather than a stranded job -- which is also why a
        loop that has already stopped is not an error here.
        """
        if self.worker is None:
            return
        if self.loop is None:
            self.worker.wake()
            return
        # A loop that has already closed is not an error: the worker's poll
        # interval is the backstop this call only ever shortens.
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.worker.wake)

    def call_on_loop(self, coro_factory: Any, timeout: float = 30.0) -> Any:
        """Run a worker coroutine on the loop thread and wait for it.

        Blocking by construction -- it is only ever called from a service
        function, which is only ever called from a thread that is allowed to
        block.
        """
        if self.worker is None:
            return None
        if self.loop is None:
            # No loop thread: run it here. A service standing on its own (a
            # test, a headless tool) still has to *do* the thing rather than
            # silently skip it -- a cancel that quietly does not cancel is the
            # worst of the three possible behaviours.
            return asyncio.run(coro_factory())
        fut: Future[Any] = asyncio.run_coroutine_threadsafe(coro_factory(), self.loop)
        return fut.result(timeout)

    # -- jobs --------------------------------------------------------------

    def require_job(self, job_id: str) -> dict[str, Any]:
        """The job row, or NotFound -- with the id guard applied first."""
        check_job_id(job_id)
        job = self.store.get(job_id)
        if job is None:
            raise NotFound("no such job")
        return job

    def attach_progress(self, job: dict[str, Any]) -> None:
        """Live progress, only ever for the job actually running."""
        job["progress"] = (
            self.worker.progress.snapshot(job["id"]) if self.worker is not None else None
        )
