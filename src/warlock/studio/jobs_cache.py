"""The job list the UI reads, refreshed on a timer rather than per frame.

``JobStore.list`` is a real sqlite query behind a lock, and at 60 fps calling
it every frame would put the single connection under 60 reads a second for
data that changes when a job changes status. Every 500 ms, or immediately when
something the UI did makes it stale, is the same tradeoff the browser made with
its poll -- minus the HTTP.

Status transitions are diffed here rather than watched for elsewhere: this is
the one place that sees both the old list and the new one, which makes it the
only place that can tell "finished" from "was already finished".
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ..service import jobs as svc_jobs

log = logging.getLogger(__name__)

REFRESH_SECONDS = 0.5
LIST_LIMIT = 200


class JobsCache:
    """A recent job list plus what changed since the frame before."""

    def __init__(self, svc: Any, limit: int = LIST_LIMIT) -> None:
        self.svc = svc
        self.limit = limit
        self.jobs: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.storage: dict[str, Any] = {}
        # How many jobs exist at all, so the library can say "showing newest N
        # of M" rather than silently presenting a truncated history as whole.
        self.total = 0
        self.error: str | None = None
        self._last_status: dict[str, str] = {}
        self._next_refresh = 0.0
        self._dirty = True

    def invalidate(self) -> None:
        """Refresh on the next tick. Called after anything the UI did that
        changes the list -- a submit, a delete, a rename."""
        self._dirty = True

    def load_more(self) -> None:
        """Widen the window by one page.

        A bigger single read rather than a merge of pages: tick() is one
        list_jobs call by design, and the per-row attach_files cost only grows
        when the user asks to see further back. MAX_LIST_LIMIT is the service's
        ceiling on a single read and clamps this.
        """
        self.limit += LIST_LIMIT
        self.invalidate()

    def tick(self, on_transition: Callable[[dict[str, Any], str | None], None] | None = None):
        """-> whether the list was re-read this frame."""
        now = time.monotonic()
        if not self._dirty and now < self._next_refresh:
            return False
        self._dirty = False
        self._next_refresh = now + REFRESH_SECONDS
        try:
            jobs = svc_jobs.list_jobs(self.svc, self.limit)
        except Exception as exc:  # a locked DB, a vanished file
            log.exception("could not read the job list")
            self.error = str(exc)
            return False
        self.error = None
        self.jobs = jobs
        try:
            self.total = self.svc.store.count()
        except Exception:  # a count is not worth failing the refresh over
            log.exception("could not count the job list")
            self.total = len(jobs)
        self.by_id = {j["id"]: j for j in jobs}
        if on_transition is not None:
            for job in jobs:
                previous = self._last_status.get(job["id"])
                if previous is not None and previous != job["status"]:
                    on_transition(job, previous)
        self._last_status = {j["id"]: j["status"] for j in jobs}
        return True

    def refresh_storage(self) -> None:
        try:
            self.storage = svc_jobs.storage(self.svc)
        except Exception:
            log.exception("could not measure storage")

    # -- queries -----------------------------------------------------------

    def get(self, job_id: str | None) -> dict[str, Any] | None:
        return None if job_id is None else self.by_id.get(job_id)

    def visible(self, filters: Any) -> list[dict[str, Any]]:
        return [j for j in self.jobs if filters.matches(j)]

    def children(self, job_id: str) -> list[dict[str, Any]]:
        return [j for j in self.jobs if j.get("parent_id") == job_id]

    @property
    def active(self) -> dict[str, Any] | None:
        """The job worth narrating: whatever is running, else whatever is queued."""
        for status in ("running", "queued"):
            for job in self.jobs:
                if job["status"] == status:
                    return job
        return None


def transition_message(job: dict[str, Any], previous: str | None) -> tuple[str, str] | None:
    """-> (text, level) for a status change worth a toast, or None.

    Only terminal transitions: a queued job becoming running is what the
    progress card is for, and a toast for it would fire on every submit.
    """
    name = job.get("name") or job.get("prompt") or job["id"]
    name = name if len(name) <= 40 else name[:37] + "..."
    if job["status"] == "done":
        return f"{name} finished.", "info"
    if job["status"] == "error":
        return f"{name} failed: {job.get('error') or 'unknown error'}", "error"
    if job["status"] == "cancelled" and previous == "running":
        return f"{name} cancelled.", "info"
    return None
