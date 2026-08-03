"""Running blocking work off the frame thread, and getting the answer back.

The rule the whole UI is built on: **nothing that can block runs on the main
thread**. A trimesh export is seconds, a Blender pose bake is about one, and a
16 ms frame budget has room for neither. Everything that might goes through
here and is collected on a later frame.

A task is keyed, and the key is what the UI binds a spinner to -- ``is_busy
("derive:model.stl")`` is how a download button knows to disable itself. Keys
also deduplicate: submitting the same key twice while the first is in flight is
a no-op, which is what stops a double-clicked button starting two exports.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

# Private, and stable for years: the registry concurrent.futures' atexit hook
# joins on. See TaskRunner.shutdown.
from concurrent.futures.thread import _threads_queues
from dataclasses import dataclass, field
from typing import Any

from ..service.errors import ServiceError

log = logging.getLogger(__name__)

# Four: enough that a slow export does not block a thumbnail decode, small
# enough that a fistful of them cannot starve the loop thread of CPU.
WORKERS = 4


@dataclass
class Done:
    """A finished task, handed to the UI on the frame it completed."""

    key: str
    result: Any = None
    error: BaseException | None = None
    # What a toast should say. Set for a ServiceError, whose message is written
    # for a person; an unexpected exception gets a generic line and a log entry
    # instead, because its str() is usually a traceback fragment.
    message: str | None = None
    tag: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class _Pending:
    future: Future
    tag: Any = None
    started: float = field(default=0.0)


class TaskRunner:
    """A small pool plus a per-frame collection point."""

    def __init__(self, workers: int = WORKERS) -> None:
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="warlock-task")
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    def submit(self, key: str, fn: Callable[..., Any], *args: Any, tag: Any = None, **kwargs: Any):
        """Run ``fn`` off-thread under ``key``. -> whether it was accepted.

        Refused rather than queued when the key is already in flight: two
        exports of the same artifact would fight over the same file, and the
        second click almost always means "I did not see the first one work".
        """
        with self._lock:
            if key in self._pending:
                return False
            future = self._pool.submit(fn, *args, **kwargs)
            self._pending[key] = _Pending(future=future, tag=tag)
        return True

    def is_busy(self, key: str) -> bool:
        with self._lock:
            return key in self._pending

    def any_busy(self, prefix: str) -> bool:
        with self._lock:
            return any(k.startswith(prefix) for k in self._pending)

    @property
    def busy_keys(self) -> set[str]:
        with self._lock:
            return set(self._pending)

    def poll(self) -> list[Done]:
        """Collect whatever finished since the last frame.

        Never blocks: a task that is still running is simply not in the result,
        and the UI keeps its spinner for another frame.
        """
        finished: list[Done] = []
        with self._lock:
            ready = [(k, p) for k, p in self._pending.items() if p.future.done()]
            for key, _pending in ready:
                del self._pending[key]
        for key, pending in ready:
            error = pending.future.exception()
            if error is None:
                finished.append(Done(key=key, result=pending.future.result(), tag=pending.tag))
            elif isinstance(error, ServiceError):
                finished.append(
                    Done(key=key, error=error, message=error.message, tag=pending.tag)
                )
            else:
                log.exception("task %s failed", key, exc_info=error)
                finished.append(
                    Done(
                        key=key,
                        error=error,
                        message="Something went wrong; see the log for details.",
                        tag=pending.tag,
                    )
                )
        return finished

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> None:
        """Stop the pool, optionally giving running tasks only ``timeout`` s.

        Bounded because a task can be parked on something that never returns:
        the native file dialogs block until the user dismisses them, and by the
        time this runs the window they belong to has already been destroyed. An
        unbounded wait there is a process that never exits.
        """
        if timeout is None:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
            with self._lock:
                self._pending.clear()
            return

        with self._lock:
            futures = [p.future for p in self._pending.values()]
        # ThreadPoolExecutor.shutdown takes no timeout, so the wait happens on
        # the futures and the pool is then told not to wait at all.
        concurrent.futures.wait(futures, timeout=timeout)
        self._pool.shutdown(wait=False, cancel_futures=True)
        # A task still parked in a never-dismissed dialog keeps its worker
        # thread alive, and concurrent.futures' atexit hook joins every live
        # pool thread on the way out -- which would hang the process after the
        # window is already gone. The threads cannot be daemonized after the
        # fact (setting .daemon on a started thread raises), so the documented
        # workaround is to drop them from that registry.
        for thread in list(self._pool._threads):
            _threads_queues.pop(thread, None)
        with self._lock:
            self._pending.clear()
