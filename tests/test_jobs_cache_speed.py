"""Phase 1 job-list cache behaviours: count cadence (A3), the adaptive poll
(L102), the per-generation visible/failures memos (B19), and incremental
storage accounting (C33)."""

from __future__ import annotations

from warlock.service import jobs as svc_jobs
from warlock.studio.jobs_cache import IDLE_REFRESH_SECONDS, REFRESH_SECONDS, JobsCache
from warlock.studio.state import Filters


def test_count_is_exact_for_free_when_the_page_is_not_full(svc, monkeypatch):
    for i in range(3):
        svc_jobs.create_job(svc, kind="text", prompt=f"job {i}")
    cache = JobsCache(svc)
    calls = []
    real = svc.store.count
    monkeypatch.setattr(svc.store, "count", lambda: calls.append(1) or real())
    assert cache.tick() is True
    assert cache.total == 3
    # Three rows against a 200 limit: the page *is* the history, so COUNT(*)
    # never ran.
    assert calls == []


def test_the_poll_backs_off_when_nothing_is_live_and_not_when_something_is(svc):
    a = svc_jobs.create_job(svc, kind="text", prompt="queued job")["id"]
    cache = JobsCache(svc)
    import time

    now = time.monotonic()
    assert cache.tick() is True
    # A queued job keeps the fast cadence.
    assert cache._next_refresh - now < REFRESH_SECONDS + 0.5
    svc.store.cancel(a)
    cache.invalidate()
    now = time.monotonic()
    assert cache.tick() is True
    # Nothing queued or running: the backstop cadence.
    assert cache._next_refresh - now > REFRESH_SECONDS + 0.5
    assert cache._next_refresh - now <= IDLE_REFRESH_SECONDS + 0.5


def test_visible_and_failures_are_memoized_per_generation(svc):
    svc_jobs.create_job(svc, kind="text", prompt="one")
    cache = JobsCache(svc)
    cache.tick()
    filters = Filters()
    first = cache.visible(filters)
    assert cache.visible(filters) is first  # same generation, same filters
    assert cache.failures(filters) == cache.failures(filters)
    filters.text = "one"
    assert cache.visible(filters) is not first  # the filter state is in the key
    cache.invalidate()
    cache.tick()
    assert cache.visible(Filters()) is not first  # a re-read mints a generation


def test_measure_one_folds_a_single_directory_into_the_totals(svc):
    a = svc_jobs.create_job(svc, kind="text", prompt="a")["id"]
    cache = JobsCache(svc)
    # Reading then adopting, which is the split the app runs across two
    # threads: the walk on a task, the amendment on the frame loop (T3).
    cache.adopt_storage(cache.measure())
    baseline = cache.storage
    assert baseline["job_dirs"] >= 0
    # A finished job writes into its own directory; only it is re-walked.
    job_dir = svc.job_dir(a)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"x" * 1000)
    cache.adopt_storage(cache.measure_one(a))
    updated = cache.storage
    assert updated["bytes"] >= baseline["bytes"] + 1000
    # No baseline yet falls back to the full walk rather than presenting one
    # directory as the whole workshop.
    fresh = JobsCache(svc)
    fresh.adopt_storage(fresh.measure_one(a))
    assert fresh.storage["bytes"] == updated["bytes"]
