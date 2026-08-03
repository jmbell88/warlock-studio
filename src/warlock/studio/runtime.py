"""Process startup and shutdown: the FastAPI lifespan, without FastAPI.

Three threads, and the split between them is the whole design:

===================  ==========================================  ===============
Thread               Runs                                        May block
===================  ==========================================  ===============
main (pygame)        events, imgui, the viewport, JobStore reads  no (~16 ms)
warlock-loop         the asyncio loop hosting Worker              asyncio only
TaskRunner pool      service calls: exports, bakes, prune         yes
===================  ==========================================  ===============

``JobStore`` is safe to read from the main thread -- one connection behind an
RLock -- so the frame loop reads jobs directly and only the genuinely slow work
goes to the pool. The worker stays on its own loop because that is where it was
written to live: ``wake`` is loop-affine, cancellation is a coroutine, and
moving either would mean rewriting queue.py rather than porting the shell.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from .. import doctor
from ..config import Config, get_config
from ..db import JobStore
from ..service import WarlockService
from .tasks import TaskRunner

log = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30.0


class Runtime:
    """Owns the store, the worker's loop thread and the task pool."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.store: JobStore | None = None
        self.worker: Any = None
        self.svc: WarlockService | None = None
        self.tasks = TaskRunner()
        self.checks: list[doctor.Check] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> WarlockService:
        self.store = JobStore(self.config.db_path)
        # A job still 'running' at process start was orphaned by a crash or an
        # unclean shutdown -- surface it instead of silently re-running a
        # 2-minute GPU job on every launch.
        self.store.reconcile_startup()
        self.checks = doctor.run_checks(self.config)
        for check in self.checks:
            if not check.ok:
                level = log.critical if check.fatal else log.warning
                level("doctor: %s -- %s", check.name, check.detail)

        self._thread = threading.Thread(target=self._run_loop, name="warlock-loop", daemon=True)
        self._thread.start()
        self._ready.wait(10.0)
        if self._loop is None:
            raise RuntimeError("the worker loop did not start")

        # Constructed *on* the loop: Worker.start creates a task, and a task
        # has to be created from the thread that will run it.
        self.worker = self._submit(self._make_worker()).result(10.0)
        self.svc = WarlockService(self.config, self.store, self.worker, self._loop)
        return self.svc

    async def _make_worker(self) -> Any:
        from ..queue import Worker

        worker = Worker(self.config, self.store)
        worker.start()
        return worker

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            # Anything still pending at stop() gets a chance to unwind before
            # the loop closes, or asyncio complains about destroyed pending
            # tasks on the way out and hides whatever the real error was.
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    def _submit(self, coro: Any) -> Any:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        """Stop everything, in the order that lets each stage finish cleanly."""
        if self.worker is not None and self._loop is not None:
            try:
                self._submit(self.worker.shutdown()).result(SHUTDOWN_TIMEOUT)
            except Exception:
                log.exception("the worker did not shut down cleanly")
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(SHUTDOWN_TIMEOUT)
        # After the loop, so a task still calling into the service finds the
        # store open rather than closed underneath it.
        self.tasks.shutdown()
        if self.store is not None:
            self.store.close()
        self.worker = self.svc = self.store = None
        self._loop = self._thread = None

    def __enter__(self) -> WarlockService:
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.shutdown()

    # -- worker access -----------------------------------------------------

    def progress(self, job_id: str | None = None) -> dict[str, Any] | None:
        """Live progress, read straight off the worker's in-memory bus.

        No polling protocol and no DB: ProgressBus is its own lock and never
        touches sqlite, so the frame loop calls this every frame -- which makes
        the percentage's easing smoother than the browser's 600 ms poll ever
        managed.
        """
        if self.worker is None:
            return None
        return self.worker.progress.snapshot(job_id)

    @property
    def current_job_id(self) -> str | None:
        return None if self.worker is None else self.worker.current_job_id

    @property
    def alive(self) -> bool:
        return self.worker is not None and self.worker.alive

    @property
    def fatal(self) -> BaseException | None:
        return None if self.worker is None else self.worker.fatal
