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
from dataclasses import dataclass
from typing import Any

from ..service.errors import Failed, ServiceError

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
    # What the toast should offer besides its text -- see ``state.Toast``. Set
    # to "log" for exactly the failures whose message defers to the log file,
    # which is to say the unexpected ones: a ServiceError names its own remedy
    # and the log has nothing to add to it.
    action: str | None = None
    tag: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class _Pending:
    # No ``started`` timestamp: one was declared here and never written or
    # read. Elapsed time per task has exactly one consumer -- the progress
    # readout -- and that is served by ``_progress``, which carries a percent
    # and a label the task itself reports rather than a figure inferred from a
    # clock this class would have to keep current.
    future: Future
    tag: Any = None


class TaskRunner:
    """A small pool plus a per-frame collection point."""

    def __init__(self, workers: int = WORKERS) -> None:
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="warlock-task")
        self._pending: dict[str, _Pending] = {}
        # Live progress per key, written by whatever thread is running the task
        # and read on the frame thread. Under the same lock as _pending: the
        # frame loop asks "is this busy" and "how far" in consecutive lines, and
        # an answer assembled from two locks can say "not busy, 40%".
        #
        # A dict rather than a protocol, for the reason the worker's ProgressBus
        # is one: the reader is in the same process and the writer is a
        # callback, so anything more is a message queue nobody needed. Done
        # arrives only at completion, which is enough for a two-second export
        # and not for a 16 GB download.
        self._progress: dict[str, dict[str, Any]] = {}
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
            # Cleared here rather than only on collection: a key reused after a
            # finished run would otherwise open showing the last run's bar.
            self._progress.pop(key, None)
            future = self._pool.submit(fn, *args, **kwargs)
            self._pending[key] = _Pending(future=future, tag=tag)
        return True

    def set_progress(self, key: str, percent: float, label: str = "") -> None:
        """Report how far ``key`` has got. Safe from any thread.

        Only recorded while the key is actually in flight: a task thread that
        reports one last time as it finishes must not resurrect an entry the
        frame loop has already collected and stopped drawing.
        """
        with self._lock:
            if key not in self._pending:
                return
            self._progress[key] = {
                "percent": max(0.0, min(100.0, float(percent))),
                "label": str(label),
            }

    def progress(self, key: str) -> dict[str, Any] | None:
        """``{"percent": float, "label": str}`` or None -- the shape
        ``Runtime.progress`` returns, so a pane draws either the same way."""
        with self._lock:
            found = self._progress.get(key)
            return dict(found) if found is not None else None

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
                self._progress.pop(key, None)
        for key, pending in ready:
            error = pending.future.exception()
            if error is None:
                finished.append(Done(key=key, result=pending.future.result(), tag=pending.tag))
            elif isinstance(error, ServiceError):
                # ``Failed`` is the one ServiceError whose message cannot name a
                # remedy (E52): it means a subprocess or a conversion that
                # should have worked didn't -- Blender is installed and the bake
                # still died, gltfpack ran and returned garbage -- so the next
                # thing to look at is the log, exactly as for an unexpected
                # exception. Every other ServiceError names its own remedy and
                # would be worse for the extra button.
                finished.append(
                    Done(
                        key=key,
                        error=error,
                        message=error.message,
                        action="log" if isinstance(error, Failed) else None,
                        tag=pending.tag,
                    )
                )
            else:
                log.exception("task %s failed", key, exc_info=error)
                finished.append(
                    Done(
                        key=key,
                        error=error,
                        message="Something went wrong; see the log for details.",
                        action="log",
                        tag=pending.tag,
                    )
                )
        return finished

    def shutdown(self, wait: bool = True, timeout: float | None = None) -> bool:
        """Stop the pool, optionally giving running tasks only ``timeout`` s.

        -> whether everything actually finished. Bounded because a task can be
        parked on something that never returns: the native file dialogs block
        until the user dismisses them, and by the time this runs the window they
        belong to has already been destroyed. An unbounded wait there is a
        process that never exits.

        The answer is returned rather than discarded because the caller's next
        move depends on it. ``Runtime.shutdown`` closes the store immediately
        afterwards, on the stated grounds that "a task still calling into the
        service finds the store open rather than closed underneath it" -- which
        is true only when the pool drained. A task that outlives the grace
        period resumes against a closed sqlite connection and gets a
        ``ProgrammingError`` captured into a future nobody will ever poll.
        """
        if timeout is None:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
            with self._lock:
                self._pending.clear()
                self._progress.clear()
            return True

        with self._lock:
            futures = [p.future for p in self._pending.values()]
        # ThreadPoolExecutor.shutdown takes no timeout, so the wait happens on
        # the futures and the pool is then told not to wait at all.
        _done, not_done = concurrent.futures.wait(futures, timeout=timeout)
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
            self._progress.clear()
        return not not_done
