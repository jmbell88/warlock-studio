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
from ..service.files import dir_size

log = logging.getLogger(__name__)

REFRESH_SECONDS = 0.5
# The poll while nothing is queued or running (L102): the list can only change
# through this UI, and everything the UI does calls ``invalidate`` -- so the
# slow tick is a backstop against a missed invalidation, not the signal path.
IDLE_REFRESH_SECONDS = 3.0
# How often COUNT(*) is re-run while the page is full (A3). When the page is
# not full the count is exact for free (total == len(jobs)).
COUNT_SECONDS = 5.0
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
        # Why ``total`` is not to be trusted, when it is not (E43). A failed
        # COUNT(*) falls back to ``len(jobs)``, which on a full page is exactly
        # the value that means "this is the whole history" -- so the fallback
        # silently retracts "Load older" and presents a truncated window as
        # complete. The library reads this and says so instead.
        self.count_error: str | None = None
        # Likewise for the storage walk (E45). Set on the task thread and read
        # on the frame thread, as ``_dir_sizes`` beside it already is: a plain
        # attribute write is atomic enough for a flag nothing branches twice on.
        self.storage_error: str | None = None
        self._last_status: dict[str, str] = {}
        self._next_refresh = 0.0
        self._next_count = 0.0
        self._dirty = True
        # Bumped every time ``jobs`` is replaced -- the key the per-generation
        # memos below (visible, failures) hang off.
        self._generation = 0
        self._visible_memo: tuple[Any, list[dict[str, Any]]] | None = None
        self._failures_memo: tuple[Any, int] | None = None
        # {job dir name: bytes} from the last full walk, so a finished job can
        # fold its own directory back in without re-walking everything (C33).
        self._dir_sizes: dict[str, int] = {}
        # {job_id: ((status, dir mtime), names)} for attach_files -- the frame
        # loop's largest syscall cost, and the one that grew without limit as
        # "load more" widened the window. Pruned to the page below, so it can
        # never outgrow what is being shown.
        self._files: dict[str, tuple[tuple[Any, int], list[str]]] = {}

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
        was_dirty = self._dirty
        self._dirty = False
        try:
            jobs = svc_jobs.list_jobs(self.svc, self.limit, files_cache=self._files)
        except Exception as exc:  # a locked DB, a vanished file
            log.exception("could not read the job list")
            self.error = str(exc)
            return False
        self.error = None
        self.jobs = jobs
        self._generation += 1
        # Adaptive cadence (L102): fast only while a job is live -- that is the
        # only time a row can change without the UI having called invalidate.
        live = any(j.get("status") in ("queued", "running") for j in jobs)
        self._next_refresh = now + (REFRESH_SECONDS if live else IDLE_REFRESH_SECONDS)
        # Whatever fell off the page cannot be asked for again without a
        # re-read, so its entry is dead weight.
        if len(self._files) > len(jobs):
            live_ids = {j["id"] for j in jobs}
            self._files = {k: v for k, v in self._files.items() if k in live_ids}
        # COUNT(*) only when it can disagree with len(jobs) (A3): a page that
        # is not full *is* the whole history. A full page re-counts on a longer
        # cadence, and immediately after anything the UI did (invalidate).
        if len(jobs) < self.limit:
            self.total = len(jobs)
            self.count_error = None
        elif was_dirty or now >= self._next_count:
            self._next_count = now + COUNT_SECONDS
            try:
                self.total = self.svc.store.count()
            except Exception as exc:  # a count is not worth failing the refresh over
                log.exception("could not count the job list")
                self.total = len(jobs)
                self.count_error = str(exc)
            else:
                self.count_error = None
        self.by_id = {j["id"]: j for j in jobs}
        if on_transition is not None:
            for job in jobs:
                previous = self._last_status.get(job["id"])
                if previous is not None and previous != job["status"]:
                    on_transition(job, previous)
        self._last_status = {j["id"]: j["status"] for j in jobs}
        return True

    def refresh_storage(self) -> None:
        """Measure the data directory now. **Blocking** -- callers off the
        frame thread only; the app submits :meth:`measure` as a task instead
        (the startup call this used to serve is deferred too, C32)."""
        self.storage = self.measure()

    def measure(self) -> Any:
        """The full measurement, safe to call from a task thread.

        Decomposed per job directory so :meth:`measure_one` can later fold a
        single finished job back in without re-walking the whole tree.
        """
        try:
            sizes = svc_jobs.storage_sizes(self.svc)
        except Exception as exc:
            log.exception("could not measure storage")
            self.storage_error = str(exc)
            return self.storage
        self.storage_error = None
        self._dir_sizes = sizes
        return {"job_dirs": len(sizes), "bytes": sum(sizes.values())}

    def measure_one(self, job_id: str) -> Any:
        """Incremental storage accounting (C33): re-measure one job directory.

        Sound because job directories are disjoint: only this job's entry can
        have changed when this job finished. Falls back to the full walk when
        no baseline exists yet -- adding one directory to a total that was
        never measured would present a single job as the whole workshop.
        """
        if not self._dir_sizes:
            return self.measure()
        try:
            path = self.svc.job_dir(job_id)
            size = dir_size(path)
            sizes = self._dir_sizes
            if size or path.exists():
                sizes[path.name] = size
            else:
                sizes.pop(path.name, None)
        except Exception as exc:
            log.exception("could not measure job dir %s", job_id)
            self.storage_error = str(exc)
            return self.storage
        self.storage_error = None
        return {"job_dirs": len(sizes), "bytes": sum(sizes.values())}

    # -- queries -----------------------------------------------------------

    def get(self, job_id: str | None) -> dict[str, Any] | None:
        return None if job_id is None else self.by_id.get(job_id)

    def _filters_key(self, filters: Any) -> Any:
        """A hashable snapshot: the generation plus every filter field. The
        fields are strings and bools, so ``vars`` is the whole state."""
        return (self._generation, tuple(sorted(vars(filters).items())))

    def visible(self, filters: Any) -> list[dict[str, Any]]:
        """The filtered, ordered list -- memoized per (list generation,
        filter state), because the library recomputes it every frame (B19)."""
        key = self._filters_key(filters)
        memo = self._visible_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        out = filters.order([j for j in self.jobs if filters.matches(j)])
        self._visible_memo = (key, out)
        return out

    def failures(self, filters: Any) -> int:
        """``filters.failures`` over the loaded page, memoized like
        :meth:`visible` and for the same reason."""
        key = self._filters_key(filters)
        memo = self._failures_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        count = filters.failures(self.jobs)
        self._failures_memo = (key, count)
        return count

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
        # ``success`` rather than ``info`` (H68): a finished job is the one
        # unambiguously good thing this function reports, and it spent its
        # whole life in the same neutral grey as "settings copied to the form".
        return f"{name} finished.", "success"
    if job["status"] == "error":
        return f"{name} failed: {job.get('error') or 'unknown error'}", "error"
    if job["status"] == "cancelled" and previous == "running":
        return f"{name} cancelled.", "info"
    return None


def sweep_summary(jobs: list[dict[str, Any]], sweep_id: str) -> tuple[str, str] | None:
    """-> (text, level) once every loaded unit of a sweep has finished (N109).

    ``None`` while any of them is still queued or running, which is what makes
    one call per finished unit collapse into exactly one toast at the end.

    Judged against the *loaded* window, and that is a real limitation rather
    than an oversight: the list is the newest N of M, so a sweep longer than
    the window could be declared finished early. It is the honest trade -- the
    alternative is a ``COUNT`` on the frame thread, which the single-connection
    rule forbids -- and it fails in the safe direction: an early summary is a
    slightly wrong number, where a missed one is silence at the end of an
    overnight run.
    """
    units = [job for job in jobs if job.get("sweep_id") == sweep_id]
    if not units or any(job.get("status") in ("queued", "running") for job in units):
        return None
    done = sum(1 for job in units if job.get("status") == "done")
    failed = sum(1 for job in units if job.get("status") == "error")
    cancelled = len(units) - done - failed
    parts = [f"{done} done"]
    if failed:
        # *Refused* rather than *failed*: the dominant error here is the
        # composition gate declining the reference, which is a measurement
        # rather than a fault, and the word decides whether the user goes
        # looking for a bug.
        parts.append(f"{failed} refused")
    if cancelled:
        parts.append(f"{cancelled} cancelled")
    level = "warn" if failed and not done else "success"
    return f"Sweep finished - {', '.join(parts)}.", level
