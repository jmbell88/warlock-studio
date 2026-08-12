"""A readers-writer lease over the model store, held by *threads*.

The primitive the audit found missing. Every existing guard around model
mutation was either a UI convention (the Settings pane disables the other
buttons) or a coroutine-level check on the worker loop -- and neither is a lock
over the thing that actually matters.

The structural point, stated once here because every user of this module
depends on it: **event-loop serialization does not serialize the underlying
model thread.** ``Worker._generate`` spends its whole life awaiting
``asyncio.to_thread``, so while a checkpoint is being read by
``from_pretrained`` or a sampler is running, the event loop is *idle* and will
happily run anything scheduled onto it -- including an unload, or a service
callable that is about to delete the weights being read. Checking
``current_job_id`` on the loop proves something about the loop, not about the
thread inside the model.

So the lease is a plain ``threading`` object and is taken *inside* the worker
thread, around the model operation itself:

* **use** (shared) -- held by every model operation: load, generate, unload.
  Any number at once, because the worker runs one job at a time anyway and the
  point is not mutual exclusion between jobs.
* **maintain** (exclusive) -- held by download, uninstall and repair for the
  whole mutation, and by shutdown before it tears a pipe down. Waits for every
  outstanding *use* to finish, which is the guarantee that nothing is inside
  ``from_pretrained`` when the files move.

Writer-preferring: once a maintainer is waiting, new users queue behind it.
A model store that could never be maintained because jobs keep arriving is the
same bug from the other side, and the queue's own admission already refuses
work while a maintenance is in flight.

Deliberately process-local, and deliberately *not* re-entrant across the two
modes. A second process mutating the same store is a different problem with a
different answer (a per-home OS lock); a lease here would not have seen it and
pretending otherwise would be worse than saying so.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator

log = logging.getLogger(__name__)

# How long a maintainer waits for in-flight model operations before giving up.
# Generous because the operation it is waiting for is a checkpoint load or a
# sampler pass -- tens of seconds is normal and refusing at five would make the
# lease useless -- and bounded because a wait with no end is a hang with a
# nicer name.
MAINTAIN_TIMEOUT = 180.0


class ModelLease:
    """Shared/exclusive access to the model store, for threads."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._users = 0
        self._maintaining = False
        self._waiting_maintainers = 0
        # Per-thread nesting depth, because ``use`` has to be re-entrant.
        # ``Text2Image.generate`` can reach ``load``, and both take the lease --
        # so without this, a maintainer arriving between the two would deadlock
        # exactly as a non-reentrant readers-writer lock always does: the outer
        # use waits for the maintainer to clear, and the maintainer waits for
        # the outer use to end.
        self._local = threading.local()

    @property
    def users(self) -> int:
        """How many model operations are in flight. Diagnostics only."""
        with self._cond:
            return self._users

    @contextlib.contextmanager
    def use(self) -> Iterator[None]:
        """Hold a shared lease for one model operation.

        Taken on the thread that does the work, never on the event loop. A
        caller that takes this on the loop has re-created the bug this module
        exists for, and will also block the frame loop's own dispatch.
        """
        depth = getattr(self._local, "depth", 0)
        if depth:
            # Already inside one on this thread; the outer lease covers this.
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth -= 1
            return
        with self._cond:
            # Wait out an active maintainer *and* any queued one: weights may be
            # moving right now, and a user that jumped a waiting maintainer
            # could starve it indefinitely.
            self._cond.wait_for(
                lambda: not self._maintaining and self._waiting_maintainers == 0
            )
            self._users += 1
        self._local.depth = 1
        try:
            yield
        finally:
            self._local.depth = 0
            with self._cond:
                self._users -= 1
                self._cond.notify_all()

    @contextlib.contextmanager
    def maintain(self, timeout: float | None = None) -> Iterator[None]:
        """Hold the exclusive lease. Raises ``TimeoutError`` rather than waiting
        forever.

        The caller turns that into whatever refusal its layer speaks -- the
        service layer raises ``Conflict``, because "a job is using these weights,
        try again in a moment" is a sentence the user can act on, and deleting
        the files anyway is not.

        ``timeout`` defaults to ``MAINTAIN_TIMEOUT`` read *at call time* rather
        than bound as a default argument, so a test can shorten the wait without
        reaching inside this class.
        """
        if timeout is None:
            timeout = MAINTAIN_TIMEOUT
        with self._cond:
            self._waiting_maintainers += 1
            try:
                if not self._cond.wait_for(
                    lambda: self._users == 0 and not self._maintaining, timeout
                ):
                    raise TimeoutError(
                        f"model operations still in flight after {timeout:.0f}s"
                    )
                self._maintaining = True
            finally:
                self._waiting_maintainers -= 1
                # Notify even on the failure path: a user parked behind this
                # maintainer must be released when it gives up.
                self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._maintaining = False
                self._cond.notify_all()


# One lease per process, because there is one model store per process. Module
# state rather than something hung off a service or a worker: the two ends of it
# live in layers that may not import each other (``queue`` may not import
# ``service``), which is the same reason ``VECTOR_PARAMS`` lives in
# ``vectors.py``.
MODELS = ModelLease()
