from __future__ import annotations

import asyncio
import time

import pytest

from animancer3d.config import Config
from animancer3d.db import JobStore
from animancer3d.queue import SHUTDOWN_TIMEOUT, Worker

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


async def test_shutdown_forces_cancel_after_timeout_when_trellis_ignores_stop(worker):
    # If shutdown()'s forced task.cancel() fallback (the branch that fires
    # after asyncio.wait_for(self._task, timeout=SHUTDOWN_TIMEOUT) raises
    # TimeoutError) were deleted, this test would hang past its own
    # asyncio.wait_for budget and fail with TimeoutError instead of
    # completing -- unlike test_shutdown_with_job_running_returns_promptly,
    # whose fake job always finishes on its own in ~0.1s regardless of
    # whether cancellation ever reached it.
    _make_image_job(worker)
    worker.trellis.ignore_stop = True
    worker.trellis.slices = 100
    worker.trellis.sleep_per_slice = 1.0  # total >> SHUTDOWN_TIMEOUT
    worker.start()
    await _wait_until(lambda: worker.trellis.running)

    start = time.monotonic()
    await asyncio.wait_for(worker.shutdown(), timeout=SHUTDOWN_TIMEOUT + 5.0)
    elapsed = time.monotonic() - start

    # Bounded by the forced-cancel fallback (~SHUTDOWN_TIMEOUT), not by the
    # fake job's own ~100s runtime.
    assert elapsed < SHUTDOWN_TIMEOUT + 5.0
    assert worker.trellis.stop_calls >= 1


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


async def test_worker_finish_does_not_overwrite_a_cancel_that_raced_it(worker):
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.current_job_id == job_id)
    # Simulate the lost-cancel race: cancel lands at the DB level without
    # ever reaching self._cancel.event (as if request_cancel ran before
    # self._cancel existed to observe it, e.g. between claim() and the
    # synchronous self._cancel = _Cancel(job_id) assignment).
    worker.store.cancel(job_id)
    await _wait_until(lambda: worker.store.get(job_id)["status"] != "running")
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "cancelled"
    assert not (worker.config.job_dir(job_id) / "model.glb").exists()


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
