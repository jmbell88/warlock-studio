from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from warlock.config import Config
from warlock.db import JobStore
from warlock.queue import POLL_INTERVAL, SHUTDOWN_TIMEOUT, Worker

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


async def test_guidance_is_folded_into_the_image_prompt(worker):
    job_id = worker.store.create(
        "text",
        "a plasma rifle",
        {"seed": 1, "resolution": 512, "genre": "scifi", "art_style": "lowpoly"},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    from warlock import guidance

    prompt = worker._text2image.prompts[0]
    assert prompt.startswith("a plasma rifle, ")
    assert guidance.GENRES["scifi"].prompt in prompt
    assert guidance.ART_STYLES["lowpoly"].prompt in prompt
    # Recorded on the job so a finished asset can explain how it was made.
    assert worker.store.get(job_id)["params"]["composed_prompt"] == prompt


async def test_composed_prompt_is_read_from_last_prompt_not_recomputed(worker, monkeypatch):
    """queue.py must record t2i.last_prompt, not its own local `composed` --
    otherwise the UI's "prompt sent" row would show the pre-trigger,
    pre-PROMPT_TEMPLATE string forever, as it did before this change.

    To prove this: monkeypatch FakeText2Image.generate() to set last_prompt to a
    value that differs from the raw composed string (simulating what the real
    Text2Image would do by wrapping with PROMPT_TEMPLATE). If queue.py reads
    the wrong variable (composed instead of last_prompt), the assertion fails.
    """
    from conftest import FakeText2Image

    # Store the original generate method
    original_generate = FakeText2Image.generate

    def generate_with_wrapped_prompt(self, prompt, *args, **kwargs):
        # Call the original to maintain all behavior
        result = original_generate(self, prompt, *args, **kwargs)
        # Simulate the real Text2Image: last_prompt would be the template-wrapped version
        # (in reality it includes PROMPT_TEMPLATE and LoRA trigger wrapping)
        self.last_prompt = f"{prompt} [TEMPLATE_WRAPPED]"
        return result

    monkeypatch.setattr(FakeText2Image, "generate", generate_with_wrapped_prompt)

    job_id = worker.store.create(
        "text", "a barrel", {"seed": 1, "resolution": 512, "genre": "scifi"},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    stored_prompt = worker.store.get(job_id)["params"]["composed_prompt"]

    # This assertion would pass ONLY if queue.py reads t2i.last_prompt.
    # If queue.py read the raw composed string instead, stored_prompt would
    # NOT contain "[TEMPLATE_WRAPPED]" and the assertion would fail.
    assert "[TEMPLATE_WRAPPED]" in stored_prompt


async def test_resolution_from_params_reaches_trellis(worker):
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 1536})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.generate_calls[0]["resolution"] == 1536


async def test_worker_forwards_bg_removal_to_trellis(worker):
    job_id = worker.store.create("image", None, {"seed": 1, "bg_removal": "birefnet"})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.generate_calls[0]["bg_removal"] == "birefnet"


async def test_worker_passes_negative_prompt(worker):
    job_id = worker.store.create(
        "text", "a barrel", {"seed": 1, "negative_prompt": "blurry, two objects"}
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.negatives[-1] == "blurry, two objects"


async def test_a_reference_job_never_reaches_trellis(worker):
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.generate_calls == []
    assert (worker.config.job_dir(job_id) / "input.png").exists()
    # Nothing pays for a trellis run just because the reference finished.
    assert not (worker.config.job_dir(job_id) / "model.glb").exists()


async def test_worker_uses_each_stage_seed(worker):
    # reference_seed and mesh_seed diverge from the legacy seed here, so the
    # test would pass by accident if the worker still read plain "seed" for
    # both -- 1 would show up somewhere and mask a real bug.
    job_id = worker.store.create(
        "text", "a barrel", {"seed": 1, "reference_seed": 11, "mesh_seed": 22}
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.seeds[-1] == 11
    assert worker.trellis.generate_calls[-1]["seed"] == 22


async def test_worker_falls_back_to_legacy_seed_for_both_stages(worker):
    # A job row written before the split has only "seed" -- no reference_seed
    # / mesh_seed keys at all. Both stages must still receive that one seed,
    # not the int(params.get(..., 42)) default, or a legacy row stops
    # reproducing its original output exactly.
    job_id = worker.store.create("text", "a barrel", {"seed": 7})
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.seeds[-1] == 7
    assert worker.trellis.generate_calls[-1]["seed"] == 7


async def test_finished_model_is_scaled_to_the_requested_size(worker, monkeypatch):
    import warlock.pipelines.postprocess as postprocess_mod

    calls = []
    monkeypatch.setattr(
        postprocess_mod,
        "normalize_glb",
        lambda path, target: (
            calls.append((path, target)),
            {"scale": 2.5, "translation": [0.0, -1.0, 0.0], "achieved_size_m": 2.5},
        )[1],
    )
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512, "size_m": 2.5})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert calls == [(job_dir / "model.glb", 2.5)]
    assert worker.store.get(job_id)["params"]["scale_factor"] == 2.5


async def test_no_size_still_grounds_but_does_not_rescale(worker, monkeypatch):
    # Grounding is not conditional on a target size: a pivot at the
    # reconstruction volume's centre is a manual fixup on every import.
    # Without a size the scale factor must still come back exactly 1.0.
    import warlock.pipelines.postprocess as postprocess_mod

    calls = []
    monkeypatch.setattr(
        postprocess_mod,
        "normalize_glb",
        lambda path, target: (
            calls.append((path, target)),
            {"scale": 1.0, "translation": [0.0, -0.5, 0.0], "achieved_size_m": 1.0},
        )[1],
    )
    job_id = _make_image_job(worker)  # params carry no size_m
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    assert [target for _, target in calls] == [None]
    assert job["params"]["scale_factor"] == 1.0
    assert job["params"]["transform"]["translation"] == [0.0, -0.5, 0.0]


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


async def test_an_image_reference_job_never_reaches_trellis(worker):
    """The service refuses to *mint* this combination now, but the worker's own
    guard is what stopped a painted reference -- kind=image, stage=reference --
    from falling past the text branch's early return into a full two-minute
    trellis run. The existing coverage only ever drove kind=text, which takes
    the other return.
    """
    job_id = worker.store.create(
        "image", None, {"seed": 1, "resolution": 512}, stage="reference"
    )
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.generate_calls == []
    assert not (job_dir / "model.glb").exists()


async def test_a_failed_error_log_still_records_the_job_as_errored(worker, monkeypatch):
    """``write_error_log`` ran *before* ``error`` was computed, so an OSError
    from it -- a full or read-only disk, which is also how a job fails in the
    first place -- left ``error`` None and the finally recorded the job as
    **done**: a successful-looking job with no model.glb and no message."""
    from warlock import errors as errors_mod

    def cannot_write(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(errors_mod, "write_error_log", cannot_write)

    job_id = _make_image_job(worker)
    worker.trellis.should_raise = RuntimeError("boom")

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "error"
    assert job["error"] == "boom"


async def test_a_failed_terminal_write_does_not_wedge_the_worker(worker, monkeypatch):
    """The four resets sat *after* ``store.finish`` in the same finally, so a
    raise from it skipped them: current_job_id kept pointing at a dead job, the
    stale _Cancel stayed, the ProgressBus entry never ended, and every later
    job's trellis output was attributed to the job that was already gone."""
    real_finish = worker.store.finish
    calls = {"n": 0}

    def finish_once_badly(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database is locked")
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(worker.store, "finish", finish_once_badly)

    doomed = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: calls["n"] >= 1)

    # The worker must let go of the job whose terminal write blew up...
    await _wait_until(lambda: worker.current_job_id != doomed)
    assert worker._cancel is None
    assert worker.progress.snapshot(doomed) is None

    # ...and still run the next one to completion.
    good = _make_image_job(worker)
    await _wait_until(lambda: worker.store.get(good)["status"] == "done")
    await worker.shutdown()


def _make_worker(tmp_path, **config_overrides) -> Worker:
    """Like the worker fixture, but with Config overrides (vram_exclusive etc.)."""
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        **config_overrides,
    )
    return Worker(config, JobStore(config.db_path))


async def test_coexist_text_job_keeps_trellis_and_sdxl_resident(worker):
    job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")

    # Checked before shutdown() (which legitimately stops trellis): the job
    # itself must not have stopped the 3D server or unloaded the image model.
    assert worker.trellis.stop_calls == 0
    assert worker._text2image.unload_calls == 0
    assert worker._text2image.loaded is True
    await worker.shutdown()


async def test_exclusive_text_job_restores_handoff(tmp_path, fake_pipelines):
    worker = _make_worker(tmp_path, vram_exclusive=True)
    try:
        job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        await worker.shutdown()

        assert worker.trellis.stop_calls >= 1
        assert worker._text2image.unload_calls == 1
        assert worker._text2image.loaded is False
    finally:
        worker.store.close()


async def test_a_job_is_refused_at_dispatch_when_the_host_is_out_of_commit(
    tmp_path, fake_pipelines, monkeypatch
):
    """The submit-time gate cannot see a host that filled up while the job
    waited, and Windows does not raise on commit exhaustion -- it kills us."""
    import warlock.queue as queue_mod

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: 0.97)
    worker = _make_worker(tmp_path)
    try:
        job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "error")
        assert "97% committed" in worker.store.get(job_id)["error"]
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_a_job_is_refused_at_dispatch_when_the_card_has_since_filled(
    tmp_path, fake_pipelines, monkeypatch
):
    import warlock.queue as queue_mod
    from warlock import vram

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 1.0))
    worker = _make_worker(tmp_path)
    try:
        job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "error")
        assert "GiB of VRAM" in worker.store.get(job_id)["error"]
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_a_resident_sdxl_pipe_is_not_charged_twice_at_dispatch(
    tmp_path, fake_pipelines, monkeypatch
):
    """The steady state coexist is designed for -- the pipe warm between jobs --
    is exactly when free VRAM is lowest. What the pipe holds is already inside
    `need`, so it must be credited back rather than demanded a second time."""
    import warlock.queue as queue_mod
    from warlock import vram

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(queue_mod, "vram_gib", lambda: (7.1, 7.52))
    free = {"gib": 30.0}
    monkeypatch.setattr(
        vram, "device_memory", lambda: vram.DeviceMemory(32.0, free["gib"])
    )
    worker = _make_worker(tmp_path)
    try:
        first = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(first)["status"] == "done")
        assert worker._text2image.loaded is True

        # The 2026-08-04 session: pipe resident, free down by what it holds.
        free["gib"] = 22.6
        second = worker.store.create("text", "a crate", {"seed": 2, "resolution": 512})
        await _wait_until(lambda: worker.store.get(second)["status"] == "done")
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_an_image_job_gets_no_credit_for_the_resident_pipe(
    tmp_path, fake_pipelines, monkeypatch
):
    """An image job's `need` carries no SDXL term, and under coexist the pipe
    stays resident beside trellis -- crediting it would overstate headroom by
    the pipe's whole size on exactly the tightest path."""
    import warlock.queue as queue_mod
    from warlock import vram

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(queue_mod, "vram_gib", lambda: (7.1, 7.52))
    monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 15.0))
    worker = _make_worker(tmp_path)
    try:
        worker._text2image = SimpleNamespace(loaded=True)
        job = {"kind": "image", "stage": "model", "params": {"resolution": 1024}}
        with pytest.raises(RuntimeError, match="GiB of VRAM"):
            worker._check_resources(job)
    finally:
        worker.store.close()


async def test_dispatch_still_refuses_past_what_the_resident_models_explain(
    tmp_path, fake_pipelines, monkeypatch
):
    import warlock.queue as queue_mod
    from warlock import vram

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(queue_mod, "vram_gib", lambda: (7.1, 7.5))
    monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 0.5))
    worker = _make_worker(tmp_path)
    try:
        worker.trellis.running = True
        worker._text2image = SimpleNamespace(loaded=True)
        job = {"kind": "text", "stage": "model", "params": {"resolution": 1536}}
        with pytest.raises(RuntimeError, match="GiB of VRAM"):
            worker._check_resources(job)
    finally:
        worker.store.close()


async def test_idle_eviction_unloads_sdxl_in_coexist_mode(tmp_path, fake_pipelines):
    worker = _make_worker(tmp_path, trellis_idle_timeout=0.05)
    try:
        job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        await _wait_until(lambda: worker._text2image.unload_calls == 1)
        assert worker._text2image.loaded is False
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_exclusive_handoff_stops_trellis_off_the_event_loop(tmp_path, fake_pipelines):
    # The handoff stop() fires the instant a text job starts -- exactly when
    # the user is watching the bar. TrellisServer.stop() blocks for up to ~20 s
    # (terminate, wait(15), reader join(5)), so running it on the loop freezes
    # /api/progress and every other route for the duration.
    worker = _make_worker(tmp_path, vram_exclusive=True)
    loop_thread = threading.get_ident()
    try:
        job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        # Sampled before shutdown(), whose own stop() is already threaded and
        # would mask a regression here.
        handoff_threads = list(worker.trellis.stop_threads)
        unload_threads = list(worker._text2image.unload_threads)
        await worker.shutdown()

        assert handoff_threads, "the exclusive handoff should have stopped trellis"
        assert loop_thread not in handoff_threads
        assert unload_threads and loop_thread not in unload_threads
    finally:
        worker.store.close()


async def test_idle_eviction_runs_off_the_event_loop(tmp_path, fake_pipelines):
    worker = _make_worker(tmp_path, trellis_idle_timeout=0.05)
    loop_thread = threading.get_ident()
    try:
        job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        await _wait_until(lambda: worker.trellis.stop_threads != [])
        await _wait_until(lambda: worker._text2image.unload_threads != [])

        assert loop_thread not in worker.trellis.stop_threads
        assert loop_thread not in worker._text2image.unload_threads
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_mesh_audit_is_recorded_on_the_finished_job(worker, monkeypatch):
    import warlock.meshaudit as meshaudit_mod

    calls = []

    def fake_hole_fraction(path, views, resolution):
        calls.append((path, views, resolution))
        return {"worst": 0.11, "mean": 0.05, "faces": 1234, "resolution": resolution, "views": []}

    monkeypatch.setattr(meshaudit_mod, "hole_fraction", fake_hole_fraction)
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    glb = worker.config.job_dir(job_id) / "model.glb"
    assert [c[0] for c in calls] == [glb]
    # Half the diagnostic default: the audit is superlinear in resolution and
    # this runs on every job.
    assert calls[0][2] == meshaudit_mod.REQUEST_PATH_RESOLUTION
    # Only the summary is stored -- per-view detail would ride on every row of
    # the job list.
    assert worker.store.get(job_id)["params"]["mesh_audit"] == {
        "worst": 0.11, "mean": 0.05, "faces": 1234, "resolution": 512,
    }


async def test_trellis_output_is_kept_as_source_glb(worker, monkeypatch):
    # The on-disk contract: source.glb is what trellis returned and is never
    # overwritten; model.glb is derived from it, so re-targeting a budget later
    # never has to pay for another trellis run.
    import warlock.pipelines.optimize as optimize_mod

    monkeypatch.setattr(
        optimize_mod,
        "run",
        lambda source, dest, **k: (
            dest.write_bytes(source.read_bytes()),
            {
                "requested": 50_000,
                "achieved": 50_000,
                "source_triangles": 90_000,
                "bytes": 1,
            },
        )[1],
    )
    job_id = worker.store.create(
        "image", None, {"seed": 1, "resolution": 512, "profile": "standard"}
    )
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert (job_dir / "source.glb").exists()
    assert (job_dir / "model.glb").exists()
    assert worker.store.get(job_id)["params"]["optimize"]["achieved"] == 50_000


async def test_a_failing_optimize_still_ships_the_reconstruction(worker, monkeypatch):
    # The reconstruction is on disk and usable; losing the budget costs file
    # size, and failing the job would cost the user the mesh.
    import warlock.pipelines.optimize as optimize_mod

    def explode(*_args, **_kwargs):
        raise optimize_mod.OptimizeError("gltfpack not found")

    monkeypatch.setattr(optimize_mod, "run", explode)
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()

    job = worker.store.get(job_id)
    job_dir = worker.config.job_dir(job_id)
    assert job["status"] == "done"
    assert "optimize" not in job["params"]
    assert (job_dir / "model.glb").read_bytes() == (job_dir / "source.glb").read_bytes()


async def test_finished_job_carries_a_mesh_report(worker, monkeypatch):
    import warlock.meshaudit as meshaudit_mod
    import warlock.meshreport as meshreport_mod

    monkeypatch.setattr(
        meshaudit_mod,
        "hole_fraction",
        lambda path, views, resolution: {
            "worst": 0.0, "mean": 0.0, "faces": 1, "resolution": resolution,
        },
    )
    monkeypatch.setattr(
        meshreport_mod,
        "build",
        lambda *a, **k: {"status": "ready", "reasons": [], "triangles": 42},
    )
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(job_id)["params"]["mesh_report"]["status"] == "ready"


async def test_a_failing_mesh_report_does_not_fail_the_job(worker, monkeypatch):
    # Same rule the audit already follows: a diagnostic must never fail a job
    # whose mesh is already on disk.
    import warlock.meshaudit as meshaudit_mod
    import warlock.meshreport as meshreport_mod

    monkeypatch.setattr(
        meshaudit_mod,
        "hole_fraction",
        lambda path, views, resolution: {
            "worst": 0.0, "mean": 0.0, "faces": 1, "resolution": resolution,
        },
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("trimesh said no")

    monkeypatch.setattr(meshreport_mod, "build", explode)
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    assert "mesh_report" not in job["params"]
    # The silhouette measurement is unaffected by the report failing.
    assert job["params"]["mesh_audit"]["worst"] == 0.0


async def test_a_failing_mesh_audit_does_not_fail_the_job(worker, monkeypatch):
    # The mesh is already on disk and fine by the time this runs; a diagnostic
    # blowing up must not retroactively turn a good job into an errored one.
    import warlock.meshaudit as meshaudit_mod

    def explode(*_args, **_kwargs):
        raise RuntimeError("trimesh said no")

    monkeypatch.setattr(meshaudit_mod, "hole_fraction", explode)
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    assert job["error"] is None
    # No badge is better than a wrong one.
    assert "mesh_audit" not in job["params"]


async def test_mesh_audit_runs_after_scaling(worker, monkeypatch):
    # Ordering is load-bearing: the audit must measure the mesh the user will
    # actually download, not the pre-scale one.
    import warlock.meshaudit as meshaudit_mod
    import warlock.pipelines.postprocess as postprocess_mod

    order = []
    monkeypatch.setattr(
        postprocess_mod,
        "normalize_glb",
        lambda path, target: (
            order.append("scale"),
            {"scale": 2.0, "translation": [0.0, 0.0, 0.0], "achieved_size_m": 2.0},
        )[1],
    )
    monkeypatch.setattr(
        meshaudit_mod,
        "hole_fraction",
        lambda path, views, resolution: (
            order.append("audit"),
            {"worst": 0.0, "mean": 0.0, "faces": 1, "resolution": resolution},
        )[1],
    )
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512, "size_m": 2.0})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert order == ["scale", "audit"]


async def test_a_failing_normalize_does_not_fail_the_job(worker, monkeypatch):
    # Grounding runs on every job, including ones that never asked for a size,
    # so an unparseable mesh must not turn a job that produced a GLB into an
    # errored one.
    import warlock.pipelines.postprocess as postprocess_mod

    def explode(*_args, **_kwargs):
        raise ValueError("incorrect header on GLB file")

    monkeypatch.setattr(postprocess_mod, "normalize_glb", explode)
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512, "size_m": 2.0})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done"
    assert job["error"] is None
    assert "transform" not in job["params"]


async def test_worker_picks_up_the_next_queued_job_after_a_completed_one(worker):
    first = _make_image_job(worker)
    second = _make_image_job(worker)

    worker.start()
    await _wait_until(lambda: worker.store.get(second)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(first)["status"] == "done"
    assert worker.store.get(second)["status"] == "done"


# --- base-model switching ---------------------------------------------------


async def test_same_base_model_across_jobs_reuses_the_resident_pipe(tmp_path, fake_pipelines):
    """The pipe is process-lifetime cached, so two jobs on one base must not
    pay a 7 GB reload between them."""
    worker = _make_worker(tmp_path)
    try:
        first = worker.store.create(
            "text", "a barrel", {"seed": 1, "resolution": 512, "base_model": "turbo"}
        )
        second = worker.store.create(
            "text", "a crate", {"seed": 2, "resolution": 512, "base_model": "turbo"}
        )
        worker.start()
        await _wait_until(lambda: worker.store.get(second)["status"] == "done")
        assert worker.store.get(first)["status"] == "done"
        assert worker._text2image.unload_calls == 0
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_switching_base_model_unloads_the_previous_one_exactly_once(
    tmp_path, fake_pipelines
):
    """A 32 GB card holds trellis plus *one* SDXL-class pipe. If the switch
    leaks the old pipe instead of freeing it, the second load OOMs -- and only
    under real VRAM, never here -- so the unload is what gets asserted."""
    worker = _make_worker(tmp_path)
    try:
        first = worker.store.create(
            "text", "a barrel", {"seed": 1, "resolution": 512, "base_model": "turbo"}
        )
        worker.start()
        # Queued only after the first finishes, so grabbing the resident pipe
        # here cannot race the swap.
        await _wait_until(lambda: worker.store.get(first)["status"] == "done")
        first_pipe = worker._text2image
        assert first_pipe.spec.key == "turbo"
        second = worker.store.create(
            "text", "a crate", {"seed": 2, "resolution": 512, "base_model": "sdxl"}
        )
        await _wait_until(lambda: worker.store.get(second)["status"] == "done")

        assert first_pipe is not worker._text2image, "the switch must build a new pipe"
        assert first_pipe.unload_calls == 1
        assert worker._t2i_key == "sdxl"
        assert worker._text2image.spec.key == "sdxl"
        assert worker._text2image.unload_calls == 0
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_base_model_switch_unloads_off_the_event_loop(tmp_path, fake_pipelines):
    """unload() pays a gc.collect() plus empty_cache(); on the event loop that
    freezes /api/progress and every other route for its duration."""
    worker = _make_worker(tmp_path)
    try:
        first = worker.store.create(
            "text", "a barrel", {"seed": 1, "resolution": 512, "base_model": "turbo"}
        )
        loop_thread = threading.get_ident()
        worker.start()
        await _wait_until(lambda: worker.store.get(first)["status"] == "done")
        # The swap makes the old pipe unreachable from the worker, so hold on
        # to it before queueing the job that triggers the switch.
        first_pipe = worker._text2image
        second = worker.store.create(
            "text", "a crate", {"seed": 2, "resolution": 512, "base_model": "sdxl"}
        )
        await _wait_until(lambda: worker.store.get(second)["status"] == "done")
        await worker.shutdown()
        assert first_pipe.unload_threads
        assert loop_thread not in first_pipe.unload_threads
    finally:
        worker.store.close()


async def test_style_lora_and_weight_reach_the_pipeline(tmp_path, fake_pipelines):
    worker = _make_worker(tmp_path)
    try:
        job_id = worker.store.create(
            "text",
            "a barrel",
            {"seed": 1, "resolution": 512, "style_lora": "ps1", "lora_weight": 0.55},
        )
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        assert worker._text2image.lora_calls == [("ps1", 0.55)]
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_unknown_base_model_in_params_falls_back_rather_than_failing(
    tmp_path, fake_pipelines
):
    """Params outlive the registry: a row written before an entry was renamed
    must still generate, since the user cannot edit a stored job's params."""
    worker = _make_worker(tmp_path)
    try:
        job_id = worker.store.create(
            "text", "a barrel", {"seed": 1, "resolution": 512, "base_model": "retired"}
        )
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        assert worker._t2i_key == worker.config.t2i_model
        await worker.shutdown()
    finally:
        worker.store.close()


async def test_wake_ends_the_idle_wait_immediately(worker):
    """Dispatch used to sleep out a full POLL_INTERVAL between looks at the
    queue, so every submit paid up to a second of dead time before anything
    started. wake() is the other half of that: the timeout stays as the
    backstop, but a submit no longer waits for it."""
    started = time.monotonic()
    task = asyncio.ensure_future(worker._wait_for_work())
    await asyncio.sleep(0)
    worker.wake()
    await asyncio.wait_for(task, timeout=POLL_INTERVAL)
    assert time.monotonic() - started < POLL_INTERVAL


async def test_the_wake_flag_is_cleared_after_it_is_observed(worker):
    """Otherwise the loop spins: a permanently-set event makes every
    subsequent wait return instantly with no work to do."""
    worker.wake()
    await asyncio.wait_for(worker._wait_for_work(), timeout=POLL_INTERVAL)
    assert not worker._wake.is_set()


async def test_a_wake_during_the_queue_read_is_not_swallowed(worker):
    """next_queued() runs in a thread, so a submit can land while the loop is
    already past its check. Clearing the flag before the wait rather than
    after would turn that into a full poll interval of sleep."""
    worker.wake()   # stands in for a submit that landed during the DB read
    started = time.monotonic()
    await asyncio.wait_for(worker._wait_for_work(), timeout=POLL_INTERVAL)
    assert time.monotonic() - started < POLL_INTERVAL


# --- conditioning -----------------------------------------------------------


def _ref_png(path, size=(64, 64), box=(16, 16, 47, 47)):
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, (200, 200, 200))
    ImageDraw.Draw(im).rectangle(list(box), fill=(40, 90, 160))
    im.save(path)
    return path


async def test_an_unconditioned_job_hands_the_pipeline_none(worker):
    """The bit-identity contract, asserted at the boundary that decides it: a
    job with no reference must pass conditioning=None, not an empty object the
    pipeline would have to interpret."""
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.conditionings == [None]


async def test_a_conditioned_job_reaches_the_pipeline_with_its_scales(worker):
    params = {
        "seed": 1, "base_model": "sdxl_cfg",
        "ip_adapter": "plus", "ip_scale": 0.9,
        "control": "canny", "control_scale": 0.4, "control_end": 0.5,
    }
    job_id = worker.store.create("text", "a barrel", params, stage="reference")
    _ref_png(worker.config.job_dir(job_id) / "ref.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    cond = worker._text2image.conditionings[-1]
    assert cond is not None
    assert cond.ip_adapter == "plus" and cond.ip_scale == 0.9
    assert cond.control == "canny" and cond.control_scale == 0.4
    assert cond.control_end == 0.5
    assert cond.ip_image == worker.config.job_dir(job_id) / "ref.png"
    assert cond.control_image.exists(), "the hint has to be written before the call"
    assert worker.store.get(job_id)["params"]["control_hint"]["kind"] == "canny"


async def test_an_unknown_conditioning_key_is_dropped_rather_than_failing(worker):
    """Params can predate a registry rename, and the user cannot fix a row that
    already exists -- the same tolerance base_model already gets."""
    params = {"seed": 1, "base_model": "sdxl_cfg", "ip_adapter": "gone", "control": "gone"}
    job_id = worker.store.create("text", "a barrel", params, stage="reference")
    _ref_png(worker.config.job_dir(job_id) / "ref.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "done"
    assert worker._text2image.conditionings[-1] is None


async def test_a_control_on_a_distilled_base_is_dropped_not_run(worker):
    params = {"seed": 1, "base_model": "turbo", "control": "canny"}
    job_id = worker.store.create("text", "a barrel", params, stage="reference")
    _ref_png(worker.config.job_dir(job_id) / "ref.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.conditionings[-1] is None
    assert not (worker.config.job_dir(job_id) / "control.png").exists()


async def test_a_conditioned_model_stage_job_stops_trellis_before_loading(worker):
    """A ControlNet plus the CLIP-ViT-H encoder is ~6 GB over the unconditioned
    budget, which does not fit beside a resident trellis on a 32 GB card."""
    params = {"seed": 1, "base_model": "sdxl_cfg", "ip_adapter": "plus"}
    job_id = worker.store.create("text", "a barrel", params)
    _ref_png(worker.config.job_dir(job_id) / "ref.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.stop_calls >= 1
    assert worker._text2image.unload_calls >= 1


async def test_a_conditioned_reference_job_leaves_trellis_alone(worker):
    """The stage the UI actually offers conditioning on never involves trellis,
    so it must not pay for a restart."""
    params = {"seed": 1, "base_model": "sdxl_cfg", "ip_adapter": "plus"}
    job_id = worker.store.create("text", "a barrel", params, stage="reference")
    _ref_png(worker.config.job_dir(job_id) / "ref.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    # Read before shutdown, which stops trellis for its own reasons.
    stops = worker.trellis.stop_calls
    await worker.shutdown()

    assert stops == 0


# --- the reference trellis actually sees ------------------------------------


async def test_trellis_receives_reference_png_not_input_png(worker):
    job_id = _make_image_job(worker)
    _ref_png(worker.config.job_dir(job_id) / "input.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    call = worker.trellis.generate_calls[-1]
    assert call["image_path"].name == "reference.png"
    assert call["image_path"].exists()
    assert worker.store.get(job_id)["params"]["reference_report"]["ok"] is True


async def test_a_rejected_reference_fails_before_trellis_runs(worker):
    """The rejection has to cost the request, not two minutes of GPU."""
    from PIL import Image

    job_id = _make_image_job(worker)
    # An empty frame: no subject at all.
    path = worker.config.job_dir(job_id) / "input.png"
    Image.new("RGB", (64, 64), (200, 200, 200)).save(path)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "error")
    await worker.shutdown()

    assert worker.trellis.generate_calls == []
    assert worker.store.get(job_id)["params"]["reference_report"]["ok"] is False


async def test_an_unreadable_reference_is_passed_on_rather_than_rejected(worker):
    """The rules are about composition, and none of them can be evaluated on
    bytes that will not decode -- trellis is the authority on that."""
    job_id = _make_image_job(worker)  # input.png is b"fake-png"
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.generate_calls
    report = worker.store.get(job_id)["params"]["reference_report"]
    assert report["ok"] is True
    assert report["measured"] is False


async def test_a_finished_job_records_its_trellis_recipe(worker):
    job_id = _make_image_job(worker)
    _ref_png(worker.config.job_dir(job_id) / "input.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    recipe = worker.store.get(job_id)["params"]["recipe"]["trellis"]
    assert recipe["seed"] == 1
    assert "versions" in recipe


async def test_a_cancelled_job_keeps_the_users_reference(worker):
    """ref.png is a user-supplied input; keeping it is what makes "cancel,
    tweak, resubmit" work. The two images this run derived from it go."""
    job_id = _make_image_job(worker)
    job_dir = worker.config.job_dir(job_id)
    _ref_png(job_dir / "input.png")
    _ref_png(job_dir / "ref.png")
    worker.start()
    await _wait_until(lambda: worker.trellis.running)
    # Written by _conditioning on a text job; placed here because this one is
    # an image job, and what is under test is which paths a cancel removes.
    _ref_png(job_dir / "control.png")

    await worker.request_cancel(job_id)
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "cancelled")
    await worker.shutdown()

    assert (job_dir / "ref.png").exists()
    assert not (job_dir / "reference.png").exists()
    assert not (job_dir / "control.png").exists()


async def test_a_finished_reference_carries_a_rank(worker):
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    rank = worker.store.get(job_id)["params"]["rank"]
    assert 0.0 <= rank["score"] <= 1.0
    # No ref.png, so nothing to compare against -- the composition half stands
    # on its own rather than the whole thing being absent.
    assert rank["anchor"] is None


async def test_a_failing_rank_does_not_fail_the_job(worker, monkeypatch):
    # Same rule as the mesh audit: a diagnostic must never be able to fail a
    # job whose artifact is already on disk.
    import warlock.pipelines.rank as rank_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rank_mod, "score", boom)
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "done"
    assert "rank" not in worker.store.get(job_id)["params"]


async def test_a_mesh_job_is_not_ranked(worker):
    # The score is about choosing between reference candidates; a mesh job has
    # nothing to choose between and the measurement would be noise in params.
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert "rank" not in worker.store.get(job_id)["params"]


def _bad_then_good(monkeypatch, failures: int):
    """reference.measure_file that refuses the first `failures` calls."""
    import warlock.pipelines.reference as reference_mod

    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        ok = calls["n"] > failures
        return reference_mod.Report(
            ok=ok, reasons=() if ok else ("the subject runs off the frame",)
        )

    monkeypatch.setattr(reference_mod, "measure_file", fake)
    return calls


async def test_without_the_setting_a_bad_reference_is_not_rerolled(worker, monkeypatch):
    _bad_then_good(monkeypatch, failures=99)
    job_id = worker.store.create("text", "a barrel", {"seed": 5}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.seeds == [5]
    assert "reference_attempts" not in worker.store.get(job_id)["params"]


async def test_a_bad_reference_is_rerolled_once_with_a_fresh_seed(
    tmp_path, fake_pipelines, monkeypatch
):
    from warlock.config import Config
    from warlock.db import JobStore

    _bad_then_good(monkeypatch, failures=1)
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        reference_retries=1,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("text", "a barrel", {"seed": 5}, stage="reference")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    # Read the teardown counters before shutdown(), which unloads the pipe
    # itself and would mask what the job did.
    unloads, trims = w._text2image.unload_calls, w._text2image.trim_calls
    await w.shutdown()

    seeds = w._text2image.seeds
    assert len(seeds) == 2
    assert seeds[0] == 5 and seeds[1] != 5
    attempts = store.get(job_id)["params"]["reference_attempts"]
    assert [a["ok"] for a in attempts] == [False, True]
    assert [a["seed"] for a in attempts] == seeds
    # One load and one teardown around both samples: the retry must not repeat
    # the VRAM handoff, which is the whole reason it lives inside the try.
    assert unloads == 0
    assert trims == 1
    store.close()


async def test_the_retry_budget_is_a_ceiling_not_a_loop(
    tmp_path, fake_pipelines, monkeypatch
):
    from warlock.config import Config
    from warlock.db import JobStore

    _bad_then_good(monkeypatch, failures=99)
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        reference_retries=2,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("text", "a barrel", {"seed": 5}, stage="reference")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    # Three samples, then it stops and hands the user the last one -- the
    # report is a heuristic, and refusing to finish would be worse.
    assert len(w._text2image.seeds) == 3
    assert store.get(job_id)["status"] == "done"
    store.close()


async def test_a_model_stage_job_measures_nothing_when_the_reroll_is_off(worker, monkeypatch):
    # The default path pays for no *extra* measurement. The one call the whole
    # job makes is reference.prepare's, further down and unchanged -- the loop
    # adds none of its own, which is what the early break is for. Drop that
    # break and this is 2.
    calls = _bad_then_good(monkeypatch, failures=0)
    job_id = worker.store.create("text", "a barrel", {"seed": 5})
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert calls["n"] == 1
    assert worker._text2image.seeds == [5]


async def test_a_cancel_between_samples_stops_the_reroll(
    tmp_path, fake_pipelines, monkeypatch
):
    """A refused report is not worth another four seconds of a job the user
    has already given up on."""
    import warlock.pipelines.reference as reference_mod
    from warlock.config import Config
    from warlock.db import JobStore

    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        reference_retries=1,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)

    def fake(path):
        # The cancel lands while the first sample's report is being measured:
        # the budget still has a retry in it, and it must not be spent.
        w._cancel.event.set()
        return reference_mod.Report(ok=False, reasons=("the subject runs off the frame",))

    monkeypatch.setattr(reference_mod, "measure_file", fake)
    job_id = store.create("text", "a barrel", {"seed": 5}, stage="reference")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "cancelled")
    await w.shutdown()

    assert w._text2image.seeds == [5]
    store.close()


async def test_a_failed_measurement_still_records_the_seed_that_shipped(
    tmp_path, fake_pipelines, monkeypatch
):
    """The provenance is about which image is on disk, not about the verdict.

    A reroll draws seed B, and the measurement of B then breaks. Stopping is
    right -- no verdict, no further reroll -- but the attempt happened and B is
    what the user is looking at, so params must say so. Recording nothing left
    reference_seed naming seed A, which no longer reproduces the image.
    """
    import warlock.pipelines.reference as reference_mod
    from warlock.config import Config
    from warlock.db import JobStore

    calls = {"n": 0}

    def fake(path):
        calls["n"] += 1
        if calls["n"] == 1:
            return reference_mod.Report(ok=False, reasons=("the subject runs off the frame",))
        raise RuntimeError("the measurement itself broke")

    monkeypatch.setattr(reference_mod, "measure_file", fake)
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        reference_retries=1,
    )
    store = JobStore(config.db_path)
    w = Worker(config, store)
    job_id = store.create("text", "a barrel", {"seed": 5}, stage="reference")
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] in ("done", "error"))
    await w.shutdown()

    params = store.get(job_id)["params"]
    seeds = w._text2image.seeds
    assert len(seeds) == 2
    # A measurement that breaks is advisory, so the job still finishes.
    assert store.get(job_id)["status"] == "done"
    # The seed on record is the one whose image is on disk, not the refused one.
    assert params["reference_seed"] == seeds[1]
    attempts = params["reference_attempts"]
    assert [a["seed"] for a in attempts] == seeds
    assert attempts[0] == {
        "seed": seeds[0],
        "ok": False,
        "reasons": ["the subject runs off the frame"],
    }
    # Recorded as unmeasured rather than as a refusal: no verdict was reached.
    assert attempts[1]["measured"] is False
    assert attempts[1]["ok"] is True
    store.close()


# --- the opt-in remesh -------------------------------------------------------


def _audits(monkeypatch, worsts: list[float]):
    """meshaudit.hole_fraction returning a scripted sequence."""
    import warlock.meshaudit as meshaudit_mod

    seen = {"n": 0}

    def fake(path, views, resolution):
        value = worsts[min(seen["n"], len(worsts) - 1)]
        seen["n"] += 1
        return {
            "worst": value, "mean": value / 2, "faces": 1000,
            "resolution": resolution, "views": [],
        }

    monkeypatch.setattr(meshaudit_mod, "hole_fraction", fake)
    return seen


def _retry_worker(tmp_path, **config_kwargs):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        **config_kwargs,
    )
    store = JobStore(config.db_path)
    return Worker(config, store), store


def _retry_job(worker: Worker, store: JobStore, **params) -> str:
    job_id = store.create("image", None, {"seed": 3, "resolution": 512, **params})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    return job_id


def _seeded_glb_bytes(worker: Worker) -> None:
    """Make the fake reconstruction distinguishable per attempt.

    The stock fake writes the same eight bytes every time, which cannot tell
    the kept attempt's GLB from the losing one's -- and that is exactly what
    the restore has to get right, for source.glb as well as for model.glb.
    """
    inner = worker.trellis.generate

    async def generate(image_path, output_path, *, seed=42, **kwargs):
        result = await inner(image_path, output_path, seed=seed, **kwargs)
        output_path.write_bytes(f"glb-{seed}".encode())
        return result

    worker.trellis.generate = generate


async def test_without_the_setting_a_holey_mesh_is_kept(worker, monkeypatch):
    _audits(monkeypatch, [0.5])
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert len(worker.trellis.generate_calls) == 1
    assert "mesh_attempts" not in worker.store.get(job_id)["params"]
    # Off by default means literally nothing extra on disk either.
    assert not (worker.config.job_dir(job_id) / "best.glb").exists()


async def test_a_holey_mesh_is_remeshed_with_a_fresh_seed(
    tmp_path, fake_pipelines, monkeypatch
):
    _audits(monkeypatch, [0.5, 0.01])
    w, store = _retry_worker(tmp_path, mesh_retries=1, mesh_hole_max=0.1)
    job_id = _retry_job(w, store, mesh_seed=3)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    seeds = [c["seed"] for c in w.trellis.generate_calls]
    assert len(seeds) == 2 and seeds[0] == 3 and seeds[1] != 3
    params = store.get(job_id)["params"]
    assert [a["worst"] for a in params["mesh_attempts"]] == [0.5, 0.01]
    assert [a["seed"] for a in params["mesh_attempts"]] == seeds
    assert params["mesh_audit"]["worst"] == 0.01
    store.close()


async def test_a_mesh_that_passes_is_never_remeshed(tmp_path, fake_pipelines, monkeypatch):
    _audits(monkeypatch, [0.01])
    w, store = _retry_worker(tmp_path, mesh_retries=2, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert len(w.trellis.generate_calls) == 1
    assert "mesh_attempts" not in store.get(job_id)["params"]
    store.close()


async def test_the_remesh_budget_is_a_ceiling_not_a_loop(
    tmp_path, fake_pipelines, monkeypatch
):
    # Every attempt fails the threshold. The job still ends, having run the
    # trellis stage exactly retries + 1 times.
    _audits(monkeypatch, [0.5])
    w, store = _retry_worker(tmp_path, mesh_retries=2, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert len(w.trellis.generate_calls) == 3
    assert len(store.get(job_id)["params"]["mesh_attempts"]) == 3
    store.close()


async def test_the_best_attempt_is_the_one_kept_even_when_it_is_the_first(
    tmp_path, fake_pipelines, monkeypatch
):
    # A reroll can be worse. Keeping the newest would then have spent two
    # minutes of GPU to make the asset worse than it already was.
    _audits(monkeypatch, [0.3, 0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=1, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    _seeded_glb_bytes(w)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    params = store.get(job_id)["params"]
    assert params["mesh_audit"]["worst"] == 0.3
    # The seed on record is the one that reproduces the GLB that shipped, not
    # the last one tried -- provenance, the same rule reference_seed follows.
    first = w.trellis.generate_calls[0]["seed"]
    assert params["mesh_seed"] == first
    assert params["recipe"]["trellis"]["seed"] == first
    job_dir = w.config.job_dir(job_id)
    # Both halves of the on-disk contract: model.glb is what shipped and
    # source.glb is what it was derived from. A later retarget re-derives
    # model.glb from source.glb, so a source left behind by the losing
    # attempt would quietly undo the choice made here.
    assert (job_dir / "model.glb").read_bytes() == f"glb-{first}".encode()
    assert (job_dir / "source.glb").read_bytes() == f"glb-{first}".encode()
    assert not (job_dir / "best.glb").exists()
    assert not (job_dir / "best.source.glb").exists()
    store.close()


async def test_the_last_attempt_wins_when_it_is_the_best(
    tmp_path, fake_pipelines, monkeypatch
):
    _audits(monkeypatch, [0.9, 0.2])
    w, store = _retry_worker(tmp_path, mesh_retries=1, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    _seeded_glb_bytes(w)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    params = store.get(job_id)["params"]
    last = w.trellis.generate_calls[-1]["seed"]
    assert params["mesh_audit"]["worst"] == 0.2
    assert params["mesh_seed"] == last
    job_dir = w.config.job_dir(job_id)
    assert (job_dir / "source.glb").read_bytes() == f"glb-{last}".encode()
    store.close()


async def test_an_unmeasurable_mesh_is_never_remeshed(
    tmp_path, fake_pipelines, monkeypatch
):
    # No verdict is not a bad verdict: the audit blowing up leaves the mesh
    # that is already on disk alone, exactly as it does with the retry off.
    import warlock.meshaudit as meshaudit_mod

    def explode(*_args, **_kwargs):
        raise RuntimeError("trimesh said no")

    monkeypatch.setattr(meshaudit_mod, "hole_fraction", explode)
    w, store = _retry_worker(tmp_path, mesh_retries=3, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert len(w.trellis.generate_calls) == 1
    store.close()


async def test_a_cancel_stops_the_retry(tmp_path, fake_pipelines, monkeypatch):
    _audits(monkeypatch, [0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=3, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    w.start()
    await _wait_until(lambda: w.trellis.running)
    await w.request_cancel(job_id)
    await _wait_until(lambda: store.get(job_id)["status"] == "cancelled")
    await w.shutdown()

    assert len(w.trellis.generate_calls) == 1
    store.close()


async def test_a_cancelled_retry_leaves_no_scratch_glb(
    tmp_path, fake_pipelines, monkeypatch
):
    # best.glb is scratch belonging to this run, so _discard_artifacts owes it
    # the same cleanup model.glb and source.glb already get.
    _audits(monkeypatch, [0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=3, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    job_dir = w.config.job_dir(job_id)
    w.start()
    await _wait_until(lambda: w.trellis.running)
    (job_dir / "best.glb").write_bytes(b"scratch")
    (job_dir / "best.source.glb").write_bytes(b"scratch")
    await w.request_cancel(job_id)
    await _wait_until(lambda: store.get(job_id)["status"] == "cancelled")
    await w.shutdown()

    assert not (job_dir / "best.glb").exists()
    assert not (job_dir / "best.source.glb").exists()
    store.close()
