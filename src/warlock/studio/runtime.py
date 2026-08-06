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

from .. import doctor, vram
from ..config import Config, get_config
from ..db import JobStore
from ..service import WarlockService
from .tasks import TaskRunner

log = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30.0


def _loop_error(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Log what asyncio would otherwise print to stderr and forget.

    The default handler writes to stderr, which on the console-script launch
    path goes to a window that closes with the process. A task on warlock-loop
    dying is the worker dying, and it must leave a record in warlock.log.
    """
    exc = context.get("exception")
    log.critical(
        "unhandled error on the worker loop: %s",
        context.get("message", ""),
        exc_info=exc if exc is not None else False,
    )


class Runtime:
    """Owns the store, the worker's loop thread and the task pool."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.store: JobStore | None = None
        self.worker: Any = None
        self.svc: WarlockService | None = None
        self.tasks = TaskRunner()
        self.checks: list[doctor.Check] = []
        self.vram_plan: vram.Plan | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> WarlockService:
        try:
            return self._start()
        except BaseException:
            # Everything start() opens is opened before it can fail: the store
            # connection, the loop thread, the worker's GPU-side tasks. Without
            # this, a doctor check or a worker constructor that raised left all
            # of them running with nothing holding a reference. shutdown() is
            # idempotent and null-guarded per resource, so a partial start
            # unwinds through exactly the same path a full one does.
            self.shutdown()
            raise

    def _start(self) -> WarlockService:
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
        # Before the Worker exists, and after run_checks (which has already
        # paid for the torch import). This is what preserves the documented
        # "read once at startup" invariant: queue.py goes on reading a plain
        # bool off the config and never learns that it was resolved.
        self.vram_plan = self._resolve_vram()

        self._thread = threading.Thread(target=self._run_loop, name="warlock-loop", daemon=True)
        self._thread.start()
        self._ready.wait(10.0)
        if self._loop is None:
            raise RuntimeError("the worker loop did not start")

        # Constructed *on* the loop: Worker.start creates a task, and a task
        # has to be created from the thread that will run it.
        self.worker = self._submit(self._make_worker()).result(10.0)
        self.svc = WarlockService(self.config, self.store, self.worker, self._loop)
        self.svc.vram_plan = self.vram_plan
        return self.svc

    def _resolve_vram(self) -> vram.Plan:
        """Measure the card, choose the mode, write the decision back."""
        plan = vram.plan(
            exclusive=self.config.vram_exclusive,
            budget_gib=self.config.vram_budget_gib,
            total_gib=self.config.vram_total_gib,
            device=vram.probe(),
            explicit=self.config.vram_exclusive_explicit,
        )
        self.config.vram_exclusive = plan.exclusive
        self.config.vram_budget_gib = plan.budget_gib
        # Recorded before the write above makes it underivable; see the field.
        self.config.vram_exclusive_explicit = plan.explicit
        log.info("vram: %s", plan.reason)
        return plan

    async def _make_worker(self) -> Any:
        from ..queue import Worker

        worker = Worker(self.config, self.store)
        worker.start()
        return worker

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(_loop_error)
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
        loop_stopped = True
        if self._thread is not None:
            self._thread.join(SHUTDOWN_TIMEOUT)
            # Checked, not assumed. ``join`` with a timeout returns whether it
            # waited long enough only via ``is_alive``, and the worker runs on
            # this thread -- so a loop that has not unwound is a second writer
            # the store-closing decision below has to account for, exactly as
            # the TaskRunner's ``drained`` is.
            loop_stopped = not self._thread.is_alive()
            if not loop_stopped:
                log.warning("the worker loop did not stop within %.0fs", SHUTDOWN_TIMEOUT)
        # After the loop, so a task still calling into the service finds the
        # store open rather than closed underneath it.
        # Bounded: a task parked in a never-dismissed native dialog would
        # otherwise hold the process open after the window has already gone.
        drained = self.tasks.shutdown(timeout=SHUTDOWN_TIMEOUT) and loop_stopped
        if self.store is not None:
            if drained:
                self.store.close()
            else:
                # The two comments above are only compatible while the pool
                # actually drains. It did not, so something is still running --
                # closing the connection under it turns a task that was going to
                # finish into a ProgrammingError captured in a future nobody
                # will poll. The process is exiting; sqlite releases the file
                # either way, and leaking a handle for the last second of a
                # shutdown costs less than corrupting the last write.
                log.warning("work still running at shutdown; leaving the store open")
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
