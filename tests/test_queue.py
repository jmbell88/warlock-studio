from __future__ import annotations

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import pytest

from warlock.config import Config
from warlock.db import JobStore
from warlock.queue import POLL_INTERVAL, Worker

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


# Twenty seconds, up from five. The fakes now emit a *real* PNG and a *real*
# GLB (ART-02), so a queue job does the post-processing it always claimed to do
# -- measure the reference, parse the mesh, ground it, audit the silhouette --
# instead of having every one of those steps raise on a marker string and be
# swallowed. That is the point of the change, and it costs seconds rather than
# milliseconds on a conditioned job. Still bounded, because a hang has to fail
# rather than run the suite out of time.
async def _wait_until(predicate, timeout: float = 20.0) -> None:
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


async def test_shutdown_forces_cancel_after_timeout_when_trellis_ignores_stop(
    worker, monkeypatch
):
    # shutdown()'s forced task.cancel() fallback -- the branch that fires after
    # asyncio.wait_for(self._task, timeout=SHUTDOWN_TIMEOUT) raises
    # TimeoutError -- against a trellis that ignores every stop, so the run can
    # only end by being cancelled.
    #
    # It cost 20 real seconds, the most of any test in the suite, waiting out
    # the true SHUTDOWN_TIMEOUT. Nothing here can tell one budget from another,
    # so the timeout is patched down instead. Patched on the module because
    # ``shutdown`` reads the global at call time; the bound below is written
    # against the local name because a module-level ``from ... import
    # SHUTDOWN_TIMEOUT`` would not see the patch.
    #
    # What this actually pins, established by mutation on 2026-08-16, is the
    # *bound* -- not the explicit ``self._task.cancel()`` pair the name says.
    # Deleting that pair changes nothing observable, because ``asyncio.wait_for``
    # cancels its awaitable on timeout by itself (3.11+); the mutant passed at
    # 20 s and at 1 s alike, and ``task.done()`` is true either way. The branch
    # is belt-and-braces over what wait_for already guarantees.
    #
    # Replace the ``wait_for`` with a bare ``await self._task``, though, and this
    # fails in about six seconds instead of sitting through the fake's ~100 s
    # reconstruction. That is the regression worth owning, and the assertions
    # below are written for it: shutdown returns inside the budget, and it does
    # not leave the dispatch task running behind it.
    import warlock.queue as queue_mod

    timeout = 1.0
    monkeypatch.setattr(queue_mod, "SHUTDOWN_TIMEOUT", timeout)
    _make_image_job(worker)
    worker.trellis.ignore_stop = True
    worker.trellis.slices = 100
    worker.trellis.sleep_per_slice = 1.0  # total >> the patched SHUTDOWN_TIMEOUT
    worker.start()
    await _wait_until(lambda: worker.trellis.running)
    task = worker._task

    start = time.monotonic()
    await asyncio.wait_for(worker.shutdown(), timeout=timeout + 5.0)
    elapsed = time.monotonic() - start

    # Bounded by the forced-cancel fallback (~the patched SHUTDOWN_TIMEOUT),
    # not by the fake job's own ~100s runtime.
    assert elapsed < timeout + 5.0
    assert worker.trellis.stop_calls >= 1
    # And it did not simply walk away from the job: an unbounded wait would
    # never reach here, but a shutdown that gave up on the task without ending
    # it would, leaving ~99 s of fake reconstruction running behind it.
    assert task.done(), "shutdown returned with the dispatch task still running"


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


async def test_stale_taxonomy_params_do_not_reach_the_image_prompt(worker):
    # A stored job from before the taxonomy retirement still runs; its
    # fragments are simply gone from the composed prompt.
    job_id = worker.store.create(
        "text",
        "a plasma rifle",
        {"seed": 1, "resolution": 512, "genre": "scifi", "art_style": "ps1"},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    prompt = worker._text2image.prompts[0]
    assert prompt == "a plasma rifle"
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


async def test_a_load_is_refused_when_commit_is_short_in_bytes_not_in_percent(
    tmp_path, fake_pipelines, monkeypatch
):
    """MDL-04: the ceiling is a percentage, and a percentage is not a quantity.

    An offloaded FLUX.2 klein is ~16 GiB of *host* weights. A machine at 85%
    commit passes the 90% ceiling with far less than 16 GiB left, is admitted,
    and crosses the limit during checkpoint allocation -- which on Windows is
    the OS terminating the process, i.e. exactly the failure the check exists to
    prevent. So the bytes the next load needs are asked for by name, immediately
    before it.
    """
    import warlock.queue as queue_mod
    from warlock import memlog, models

    klein = next(
        key
        for key, spec in models.BASE_MODELS.items()
        if spec.residency == models.OFFLOAD
    )
    spec = models.BASE_MODELS[klein]
    assert spec.host_peak_gib > spec.vram_gib, "offload is why the host figure is larger"

    # Comfortably under the ceiling...
    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: 0.85)
    # ...and 6 GiB free against a ~16 GiB load.
    monkeypatch.setattr(
        memlog, "system_memory", lambda: memlog.SystemMemory(54.0, 60.0)
    )
    worker = _make_worker(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="host memory"):
            await worker._acquire_t2i(spec, klein)
        assert worker._text2image is None, "nothing may be loaded after a refusal"

        # With room for it, the same load goes ahead.
        monkeypatch.setattr(
            memlog, "system_memory", lambda: memlog.SystemMemory(10.0, 60.0)
        )
        pipe, _handoff = await worker._acquire_t2i(spec, klein)
        assert pipe is not None
    finally:
        worker.store.close()


def _commit_scenario(monkeypatch):
    """MDL-04's memory setup, made switchable mid-test.

    Returns a dict whose ``free`` key the test rewrites; the monkeypatched
    ``system_memory`` reads it on every call, so a test can be short before an
    unload and roomy after it.
    """
    import warlock.queue as queue_mod
    from warlock import memlog

    state = {"free": 40.0}
    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: 0.85)
    monkeypatch.setattr(
        memlog,
        "system_memory",
        lambda: memlog.SystemMemory(60.0 - state["free"], 60.0),
    )
    return state


def _offloaded_key():
    from warlock import models

    return next(
        key
        for key, spec in models.BASE_MODELS.items()
        if spec.residency == models.OFFLOAD
    )


async def test_a_base_switch_unloads_the_old_pipe_before_the_commit_check(
    tmp_path, fake_pipelines, monkeypatch
):
    """The live klein refusal: 13.2 GiB free with ~12 GiB of the shortfall
    being the resident SDXL pipe the switch was guaranteed to unload one call
    later. The check must measure a host that has already given those weights
    back, not charge the outgoing model against the incoming one."""
    from warlock import models

    state = _commit_scenario(monkeypatch)
    klein = _offloaded_key()
    worker = _make_worker(tmp_path)
    try:
        await worker._acquire_t2i(models.BASE_MODELS[models.DEFAULT_BASE_MODEL],
                                  models.DEFAULT_BASE_MODEL)
        old = worker._text2image
        assert old is not None

        # Short while the old pipe is resident, roomy once it has unloaded --
        # exactly the live machine's shape.
        def free():
            return 6.0 if old.unload_calls == 0 else 40.0

        from warlock import memlog
        monkeypatch.setattr(
            memlog, "system_memory",
            lambda: memlog.SystemMemory(60.0 - free(), 60.0),
        )
        del state

        pipe, _handoff = await worker._acquire_t2i(models.BASE_MODELS[klein], klein)
        assert old.unload_calls == 1
        assert pipe is not old
        assert worker._t2i_key == klein
    finally:
        worker.store.close()


async def test_a_switch_still_short_after_the_unload_is_refused(
    tmp_path, fake_pipelines, monkeypatch
):
    """Re-measure, never credit: if the host is genuinely short even after the
    stale pipe is gone, the refusal still fires -- against post-unload numbers."""
    from warlock import models

    state = _commit_scenario(monkeypatch)
    klein = _offloaded_key()
    worker = _make_worker(tmp_path)
    try:
        await worker._acquire_t2i(models.BASE_MODELS[models.DEFAULT_BASE_MODEL],
                                  models.DEFAULT_BASE_MODEL)
        old = worker._text2image
        state["free"] = 6.0

        with pytest.raises(RuntimeError, match="host memory"):
            await worker._acquire_t2i(models.BASE_MODELS[klein], klein)
        assert old.unload_calls == 1
        assert worker._text2image is None, "nothing may be loaded after a refusal"
    finally:
        worker.store.close()


async def test_a_warm_same_key_pipe_is_never_commit_checked(
    tmp_path, fake_pipelines, monkeypatch
):
    """The weights are in commit either way; refusing a warm hit would fail a
    job for memory it is not about to ask for."""
    from warlock import models

    state = _commit_scenario(monkeypatch)
    worker = _make_worker(tmp_path)
    try:
        await worker._acquire_t2i(models.BASE_MODELS[models.DEFAULT_BASE_MODEL],
                                  models.DEFAULT_BASE_MODEL)
        first = worker._text2image
        state["free"] = 0.0

        pipe, _handoff = await worker._acquire_t2i(
            models.BASE_MODELS[models.DEFAULT_BASE_MODEL], models.DEFAULT_BASE_MODEL
        )
        assert pipe is first
        assert first.unload_calls == 0
    finally:
        worker.store.close()


async def test_a_store_generation_bump_reloads_through_the_commit_check(
    tmp_path, fake_pipelines, monkeypatch
):
    """The previously unchecked path: a generation-forced rebuild of a same-key
    pipe skipped the bytes check entirely, because the gate keyed on the base
    key alone. It now unloads first and then passes through the check."""
    from warlock import fetch, memlog, models

    _commit_scenario(monkeypatch)
    worker = _make_worker(tmp_path)
    try:
        await worker._acquire_t2i(models.BASE_MODELS[models.DEFAULT_BASE_MODEL],
                                  models.DEFAULT_BASE_MODEL)
        old = worker._text2image
        fetch.bump_store_generation()

        # Short before the unload, roomy after: passes only if the check runs
        # on the far side of the eviction.
        monkeypatch.setattr(
            memlog, "system_memory",
            lambda: memlog.SystemMemory(
                60.0 - (6.0 if old.unload_calls == 0 else 40.0), 60.0
            ),
        )

        pipe, _handoff = await worker._acquire_t2i(
            models.BASE_MODELS[models.DEFAULT_BASE_MODEL], models.DEFAULT_BASE_MODEL
        )
        assert old.unload_calls == 1
        assert pipe is not old, "the rebuilt pipe must be a new object"
    finally:
        worker.store.close()


async def test_idle_cache_eviction_runs_again_for_a_cache_loaded_off_the_queue(
    tmp_path, fake_pipelines, monkeypatch
):
    """MDL-12: the eviction was latched, and the latch only reopened on a GPU job.

    Matting is loaded by exports and matte previews through ``TaskRunner``,
    never through the queue -- so on an idle session the first pass evicted,
    the latch closed, an export then loaded BiRefNet's ~1.5 GB child, and
    nothing ever dropped it again. Throttling by time instead means the next
    idle window gets its own pass.
    """
    import warlock.queue as queue_mod

    drops: list[int] = []
    monkeypatch.setattr(
        queue_mod.asyncio, "to_thread", _counting_to_thread(drops)
    )
    worker = _make_worker(tmp_path, trellis_idle_timeout=1.0)
    try:
        # Idle for longer than the timeout: the first pass runs.
        worker._last_job_at = time.monotonic() - 10.0
        await worker._maybe_evict_caches()
        assert len(drops) == 1

        # Immediately after, the throttle holds it off -- this is what the
        # latch was for and it is preserved.
        await worker._maybe_evict_caches()
        assert len(drops) == 1

        # A whole idle window later -- during which an export may well have
        # loaded matting through the task pool -- it runs again.
        worker._caches_evicted_at = time.monotonic() - 10.0
        worker._last_job_at = time.monotonic() - 10.0
        await worker._maybe_evict_caches()
        assert len(drops) == 2, "a cache loaded off the queue path must still be dropped"
    finally:
        worker.store.close()


def _counting_to_thread(seen: list[int]):
    async def _to_thread(fn, *args, **kwargs):
        seen.append(1)
        return fn(*args, **kwargs)

    return _to_thread


async def test_installing_a_model_while_a_pipe_is_warm_rebuilds_it(worker):
    """MDL-05: the pipe freezes its adapter set at ``load()`` and stays warm for
    600 s, so a style LoRA installed in Settings between two jobs was invisible
    to the second one -- which then generated with no style while recording
    that it had used one.

    The fix is a model-store generation the service bumps on every successful
    download or uninstall; the worker compares it and rebuilds rather than
    handing back a pipe that predates the weights.
    """
    from warlock import fetch

    first = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
    worker.start()
    await _wait_until(lambda: worker.store.get(first)["status"] == "done")
    warm = worker._text2image
    assert warm.loaded is True
    unloads = warm.unload_calls

    # Same base, same key: without the generation this is a plain cache hit.
    same = await worker._get_text2image(worker._t2i_key)
    assert same is warm
    assert warm.unload_calls == unloads

    # Now the store changes under it, exactly as a Settings install does.
    fetch.bump_store_generation()
    rebuilt = await worker._get_text2image(worker._t2i_key)
    assert rebuilt is not warm, "a pipe older than the store must not be reused"
    assert warm.unload_calls == unloads + 1

    # And the rebuilt pipe is current, so the next job is a cache hit again.
    assert await worker._get_text2image(worker._t2i_key) is rebuilt
    await worker.shutdown()


async def test_an_offloaded_base_hands_off_even_in_coexist_mode(tmp_path, fake_pipelines):
    """The driving defect, and it is measured rather than theorised.

    "Coexist" was written when every image model was RESIDENT. An offloaded
    checkpoint is the opposite shape: ``enable_model_cpu_offload()`` keeps the
    whole ~16 GiB in *host* RAM for the life of the pipe and streams one
    submodule to the device, so its declared ``vram_gib`` is honest about VRAM
    and says nothing about commit -- and under WDDM trellis' ~16 GiB device
    allocation is charged against the same commit limit. Three consecutive
    FLUX.2 klein jobs took Python private commit from 24.4 to 45.2 GiB and
    system commit to 99%.

    So the handoff is mandatory for OFFLOAD, whatever WARLOCK_VRAM_EXCLUSIVE
    says -- and the teardown must be unload(), not trim(): trim() returns the
    CUDA caching allocator's pool, and none of an offloaded pipe's cost is in
    it.
    """
    worker = _make_worker(tmp_path)  # coexist: vram_exclusive unset
    assert not worker.config.vram_exclusive
    try:
        job_id = worker.store.create(
            "text",
            "a barrel",
            {"seed": 1, "resolution": 512, "base_model": "flux_klein"},
        )
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        stop_calls = worker.trellis.stop_calls
        unload_calls = worker._text2image.unload_calls
        trim_calls = worker._text2image.trim_calls
        await worker.shutdown()

        assert stop_calls >= 1, "an offloaded base must stop trellis before loading"
        assert unload_calls == 1
        assert trim_calls == 0
    finally:
        worker.store.close()


async def test_a_resident_base_still_coexists(tmp_path, fake_pipelines):
    """The other half of the same decision: nothing about SDXL changed.

    Stated as its own test rather than trusted to
    ``test_coexist_text_job_keeps_trellis_and_sdxl_resident``, because that one
    does not name a base and so would go on passing if the new term were
    applied to the registry's default by accident.
    """
    worker = _make_worker(tmp_path)
    try:
        job_id = worker.store.create(
            "text", "a barrel", {"seed": 1, "resolution": 512, "base_model": "sdxl"}
        )
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
        assert worker.trellis.stop_calls == 0
        assert worker._text2image.unload_calls == 0
        assert worker._text2image.trim_calls == 1
        assert worker._text2image.loaded is True
        await worker.shutdown()
    finally:
        worker.store.close()


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


async def test_the_3d_stage_is_withheld_when_commit_crosses_after_the_image(
    tmp_path, fake_pipelines, monkeypatch
):
    """The commit gate ran once, at dispatch -- before the stage that moves it.

    Loading an image model is the single largest thing this process does to
    host commit, and on an offloaded checkpoint it is ~16 GiB of it. Nothing
    re-asked the question across stop-trellis / load / generate / unload /
    start-trellis: the only observation point was ``_log_mem``'s log.critical,
    which records the wall and then walks into it (the 2026-08-03
    Resource-Exhaustion crash).

    The wording is half the fix and is asserted here: the image is finished and
    on disk, and a user told "out of memory" after a successful two-minute
    generation will assume they lost it.
    """
    import warlock.queue as queue_mod

    # Two healthy readings, then the wall: dispatch asks, and so does the check
    # immediately before the checkpoint load (MDL-04). The one under test is the
    # third, after the image stage.
    readings = iter([0.50, 0.50])
    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: next(readings, 0.97))
    worker = _make_worker(tmp_path)
    try:
        job_id = worker.store.create("text", "a barrel", {"seed": 1, "resolution": 512})
        worker.start()
        await _wait_until(lambda: worker.store.get(job_id)["status"] == "error")
        error = worker.store.get(job_id)["error"]
        await worker.shutdown()

        assert "97% committed after the image stage" in error
        assert "3D stage was withheld" in error
        # Not a claim about the picture: it is on disk and the job says so.
        assert (worker.config.job_dir(job_id) / "input.png").exists()
        # And nothing reached trellis -- which is the whole point of standing
        # here rather than one line later.
        assert worker.trellis.generate_calls == []
        assert worker.trellis.config_calls == []
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


@pytest.mark.parametrize(
    "kind,params",
    [
        ("sprite_synthesis", {"base_model": "sdxl_cfg"}),
        ("pixel_sheet", {"base_model": "sdxl_cfg"}),
        ("retexture", {"base_model": "sdxl_cfg"}),
    ],
)
async def test_every_kind_that_is_charged_for_a_checkpoint_is_credited_for_one(
    tmp_path, fake_pipelines, monkeypatch, kind, params
):
    """MDL-06: the credit was gated on ``kind == "text"`` and three kinds that
    are charged an image-model term were not on the list.

    The flow that broke is the natural one on the flagship card: generate a 2D
    reference (which leaves a ~7.5 GiB pipe warm *by design*, so free VRAM is at
    its lowest exactly here), then start a sprite synthesis over it. Need came
    to ~26.7 against ~23.5 free and the job was refused at dispatch with "close
    other GPU applications" -- for reusing the very pipe it was being charged
    for. The true incremental need is the ControlNet and IP encoder, ~3.7 GiB.

    Waiting out the 600 s idle eviction made it work, which is what made this
    read as a phantom leak rather than as an accounting bug.
    """
    import warlock.queue as queue_mod
    from warlock import vram

    monkeypatch.setattr(queue_mod, "commit_fraction", lambda: None)
    monkeypatch.setattr(queue_mod, "vram_gib", lambda: (7.1, 7.52))
    # Trellis warm and the pipe warm: the coexist steady state.
    monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 23.5))
    worker = _make_worker(tmp_path)
    try:
        worker.trellis.running = True
        worker._text2image = SimpleNamespace(loaded=True)
        job = {"kind": kind, "stage": "model", "params": params}
        # The estimate does charge a checkpoint term for this kind...
        _need, image_term = vram.estimate_job_parts(job)
        assert image_term > 0
        # ...so the resident pipe is credited and the job is admitted.
        worker._check_resources(job)

        # And the credit is not a blanket pass. With trellis *not* yet running
        # its ~16 GiB is still in `need` under coexist but not yet given back by
        # the running-server branch, so a nearly full card refuses exactly as
        # before -- the credit covers the resident checkpoint and nothing else.
        worker.trellis.running = False
        monkeypatch.setattr(vram, "device_memory", lambda: vram.DeviceMemory(32.0, 0.5))
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
    # Whatever the request path measures at -- which is the constant's job to
    # decide, not this test's. test_meshaudit pins the value and the
    # measurement document that justifies it.
    assert calls[0][2] == meshaudit_mod.REQUEST_PATH_RESOLUTION
    # Only the summary is stored -- per-view detail would ride on every row of
    # the job list.
    assert worker.store.get(job_id)["params"]["mesh_audit"] == {
        "worst": 0.11,
        "mean": 0.05,
        "faces": 1234,
        "resolution": meshaudit_mod.REQUEST_PATH_RESOLUTION,
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
    # ``model.glb`` is the reconstruction with the budget *not* applied -- and
    # then grounded, which is a separate step that still runs. This used to
    # assert the two files were byte-identical, which was only ever true because
    # the fake wrote a marker that ``normalize_glb`` could not parse: the
    # grounding failed silently and the "happy path" was the degraded one
    # (ART-02). Both files exist and neither is empty; the transform is what
    # makes them differ.
    assert (job_dir / "model.glb").stat().st_size > 0
    assert (job_dir / "source.glb").stat().st_size > 0
    assert job["params"]["transform"], "grounding still runs when optimize fails"
    # And the failure is *recorded* rather than only logged, so a user can see
    # why this mesh is at full density (ART-01).
    assert "optimize" in job["params"]["degraded"]


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


async def test_a_pipeline_fed_control_reaching_a_text_job_is_dropped_not_run(worker):
    """The registry's depth entry is fed by the re-texture pipeline's own
    Blender render; a text job has nothing to render a hint from, and calling
    write_hint(kind=None) would fail the job with the checkpoint already in
    VRAM. guidance.normalize refuses it at the door -- this is the tolerance
    for a params row that never went through the door."""
    params = {"seed": 1, "base_model": "sdxl_cfg", "control": "depth"}
    job_id = worker.store.create("text", "a barrel", params, stage="reference")
    _ref_png(worker.config.job_dir(job_id) / "ref.png")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "failed"))
    await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "done"
    assert worker._text2image.conditionings[-1] is None
    assert not (worker.config.job_dir(job_id) / "control.png").exists()


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


async def test_a_refused_reference_is_rerolled_before_the_job_fails(worker, monkeypatch):
    """Reference retries, and the config default is the whole of it.

    Seed 11's baseline errored in the 2026-08-07 rogue sweep, which did not
    cost one mesh: ``findings.comparisons`` pairs two rows only when they share
    a sweep, a source and a *seed*, so the baseline at seed 11 was one side of
    nine prospective pairs -- one per axis value at that seed -- and every one
    of them went with it. Which of three reference draws happened to avoid a
    character sheet is not a setting anyone is trying to measure.

    ``mesh_seed`` is what must not move: a unit's identity in the corpus is its
    config vector and its mesh seed, so a reroll that changed both would be a
    different unit rather than a second attempt at this one.
    """
    import warlock.pipelines.reference as reference_mod

    _bad_then_good(monkeypatch, failures=1)
    # The post-normalisation gate passes; this test is about the composition
    # reroll above it, which is the half that can act on a refusal at all.
    monkeypatch.setattr(
        reference_mod, "prepare", lambda *a, **k: reference_mod.Report(ok=True)
    )
    job_id = worker.store.create("text", "a barrel", {"seed": 11})
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    reference_seeds = worker._text2image.seeds
    assert len(reference_seeds) == 2, "the refused draw was not rerolled"
    assert reference_seeds[0] == 11 and reference_seeds[1] != 11
    assert worker.trellis.generate_calls[0]["seed"] == 11


async def test_without_the_setting_a_bad_reference_is_not_rerolled(worker, monkeypatch):
    # Pins the reroll *off*, rather than trusting the default to be off -- it
    # is 2 now, and this test is about the other setting.
    monkeypatch.setattr(worker.config, "reference_retries", 0)
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
    monkeypatch.setattr(worker.config, "reference_retries", 0)
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
        # Stage-and-rename, exactly as the real server client's _atomic_write
        # does: every writer of source.glb replaces the directory entry rather
        # than rewriting the inode, and the remesh staging hard-links against
        # precisely that contract (C37). A bare write_bytes here would model a
        # writer the app does not have -- and scribble through the link.
        tmp = output_path.with_suffix(".glb.tmp")
        tmp.write_bytes(f"glb-{seed}".encode())
        os.replace(tmp, output_path)
        return result

    worker.trellis.generate = generate


async def test_a_mesh_exactly_at_the_threshold_is_kept(
    tmp_path, fake_pipelines, monkeypatch
):
    # The comparison is <=, so the threshold is the last acceptable value
    # rather than the first rejected one.
    _audits(monkeypatch, [0.1])
    w, store = _retry_worker(tmp_path, mesh_retries=2, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert len(w.trellis.generate_calls) == 1
    store.close()


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


async def test_a_failed_remesh_does_not_fail_a_job_that_already_had_a_mesh(
    tmp_path, fake_pipelines, monkeypatch
):
    # The retry is this code's own idea, so it may not retroactively fail a job
    # whose GLB is already on disk -- the rule the audit, the report and the
    # grounding all follow. The half-written source.glb the failed run left is
    # replaced by the kept attempt's, not trusted.
    _audits(monkeypatch, [0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=2, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    _seeded_glb_bytes(w)
    inner = w.trellis.generate

    async def generate(image_path, output_path, **kwargs):
        if w.trellis.generate_calls:
            # A failed run leaves *something untrusted* at source.glb. Written
            # through a rename like every real writer (the client's
            # _atomic_write never leaves a torn file, and the staging
            # hard-links against exactly that contract, C37) -- but the bytes
            # are garbage, which is what the restore must replace.
            tmp = output_path.with_suffix(".glb.tmp")
            tmp.write_bytes(b"failed-run-leftovers")
            os.replace(tmp, output_path)
            raise RuntimeError("trellis-server died")
        return await inner(image_path, output_path, **kwargs)

    w.trellis.generate = generate
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] in ("done", "error"))
    await w.shutdown()

    job = store.get(job_id)
    assert job["status"] == "done"
    assert job["error"] is None
    job_dir = w.config.job_dir(job_id)
    assert (job_dir / "model.glb").read_bytes() == b"glb-3"
    assert (job_dir / "source.glb").read_bytes() == b"glb-3"
    assert store.get(job_id)["params"]["mesh_audit"]["worst"] == 0.9
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


async def test_a_cancel_during_a_retry_kills_the_server(
    tmp_path, fake_pipelines, monkeypatch
):
    # request_cancel reads the progress *phase* to decide whether to kill
    # trellis-server, and killing it is the only abort trellis has. The retry's
    # generate therefore has to re-declare phase="trellis": left at "audit" --
    # what _audit_mesh set -- a cancel would set the event, skip the kill and
    # then sit in the retry for a whole reconstruction.
    _audits(monkeypatch, [0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=3, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    inner = w.trellis.generate

    async def generate(image_path, output_path, **kwargs):
        if w.trellis.generate_calls:
            # Long enough to still be inside the retry when the cancel lands.
            w.trellis.slices = 100
        return await inner(image_path, output_path, **kwargs)

    w.trellis.generate = generate
    w.start()
    await _wait_until(lambda: len(w.trellis.generate_calls) == 2)
    await w.request_cancel(job_id)

    assert w.trellis.stop_calls == 1
    assert w.progress.snapshot()["phase"] == "trellis"
    await _wait_until(lambda: store.get(job_id)["status"] == "cancelled", timeout=10.0)
    await w.shutdown()
    store.close()


async def test_a_failed_staging_copy_gives_up_on_retrying_rather_than_on_the_job(
    tmp_path, fake_pipelines, monkeypatch
):
    # A failed staging must cost the retry, never the mesh that is already on
    # disk. The staging is a hard link now (C37), so the realistic failures
    # are a filesystem that refuses links *and* no room for the fallback copy
    # -- both halves are refused here so the OSError actually escapes
    # _stage_link rather than being absorbed by its fallback.
    import warlock.queue as queue_mod

    real = queue_mod.shutil.copyfile
    real_link = queue_mod.os.link

    def link(src, dst, *args, **kwargs):
        if str(src).endswith("model.glb"):
            raise OSError("links not supported here")
        return real_link(src, dst, *args, **kwargs)

    def copyfile(src, dst, *args, **kwargs):
        # ``in``, not ``endswith``: the fallback stages through a temp sibling
        # and renames now, so the copy's destination is ``best.glb.copy.<hex>
        # .tmp`` rather than the served name itself (CON-04). Matching only the
        # final name would let this copy succeed and the test would silently
        # stop exercising a failed staging at all.
        if "best.glb" in str(dst):
            raise OSError(28, "No space left on device")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(queue_mod.os, "link", link)
    monkeypatch.setattr(queue_mod.shutil, "copyfile", copyfile)
    _audits(monkeypatch, [0.9])
    w, store = _retry_worker(tmp_path, mesh_retries=2, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] in ("done", "error"))
    await w.shutdown()

    job = store.get(job_id)
    assert job["status"] == "done"
    assert job["error"] is None
    assert len(w.trellis.generate_calls) == 1
    assert job["params"]["mesh_audit"]["worst"] == 0.9
    store.close()


async def test_no_scratch_copy_is_taken_for_an_attempt_that_cannot_be_retried(
    tmp_path, fake_pipelines, monkeypatch
):
    # Two whole GLBs. The copy belongs below the break check, so the attempt
    # that ends the loop -- here the first, which passes -- never pays for one.
    import warlock.queue as queue_mod

    real = queue_mod.shutil.copyfile
    copies: list[str] = []

    def copyfile(src, dst, *args, **kwargs):
        copies.append(str(dst))
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(queue_mod.shutil, "copyfile", copyfile)
    _audits(monkeypatch, [0.01])
    w, store = _retry_worker(tmp_path, mesh_retries=2, mesh_hole_max=0.1)
    job_id = _retry_job(w, store)
    w.start()
    await _wait_until(lambda: store.get(job_id)["status"] == "done")
    await w.shutdown()

    assert not [c for c in copies if "best" in c]
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


# --- tiles ------------------------------------------------------------------


async def test_a_tile_job_never_reaches_trellis(worker):
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.generate_calls == []
    assert (worker.config.job_dir(job_id) / "input.png").exists()


async def test_a_tile_job_asks_the_pipeline_to_tile(worker):
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.tiles == [True]


async def test_an_ordinary_reference_does_not_tile(worker):
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.tiles == [False]


async def test_the_prompt_preview_mirror_agrees_with_the_worker_and_the_pipeline(worker):
    # prompt.build() is a *mirror* of an assembly split across two modules --
    # queue.py composes the subject, text2image.generate picks the template
    # -- and nothing else checks that the two agree.
    from warlock import guidance
    from warlock.pipelines import prompt as prompt_lib

    params = guidance.normalize({})
    params["seed"] = 1
    tile_id = worker.store.create("text", "cobblestone", dict(params), stage="tile")
    ref_id = worker.store.create("text", "cobblestone", dict(params), stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(tile_id)["status"] == "done")
    await _wait_until(lambda: worker.store.get(ref_id)["status"] == "done")
    await worker.shutdown()

    # Keyed by the job rather than by position: both prompts are recorded on
    # one fake pipe, and unpacking them in order would silently swap the two
    # assertions -- passing for the wrong reason -- if the worker ever stopped
    # dispatching in creation order. composed_prompt is what the job itself
    # recorded, so it cannot be attributed to the wrong row.
    tile_composed = worker.store.get(tile_id)["params"]["composed_prompt"]
    ref_composed = worker.store.get(ref_id)["params"]["composed_prompt"]
    assert {tile_composed, ref_composed} == set(worker._text2image.prompts)
    # The template half is what generate() applies to the string the worker
    # handed it; tests/test_tiling.py pins that it applies exactly these two.
    assert prompt_lib.build("cobblestone", params, tile=True) == (
        prompt_lib.TILE_TEMPLATE.format(prompt=tile_composed)
    )
    assert prompt_lib.build("cobblestone", params) == (
        prompt_lib.PROMPT_TEMPLATE.format(prompt=ref_composed)
    )


async def test_a_tile_is_measured_for_seams_not_for_composition(worker, monkeypatch):
    from warlock.pipelines import seam as seam_mod

    monkeypatch.setattr(
        seam_mod,
        "report",
        lambda path: {
            "horizontal": 1.1,
            "vertical": 1.2,
            "worst": 1.2,
            "seamless": True,
            "threshold": 2.0,
        },
    )
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    params = worker.store.get(job_id)["params"]
    assert params["seam_report"]["seamless"] is True
    # A tile has no subject, so the composition report would be a verdict about
    # something that is deliberately not in the picture.
    assert "reference_report" not in params
    assert "rank" not in params


async def test_a_tile_is_never_rerolled_for_its_composition(worker, monkeypatch):
    # The reroll's rules are all about where a *subject* sits, so a tile that
    # entered the loop would be redrawn for failing a test it cannot pass.
    from warlock.pipelines import reference as reference_mod

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("a tile was measured for composition")

    monkeypatch.setattr(reference_mod, "measure_file", _forbidden)
    monkeypatch.setattr(worker.config, "reference_retries", 3)
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert len(worker._text2image.prompts) == 1


async def test_a_failing_seam_measurement_does_not_fail_the_job(worker, monkeypatch):
    from warlock.pipelines import seam as seam_mod

    monkeypatch.setattr(
        seam_mod, "report", lambda path: (_ for _ in ()).throw(ValueError("too small"))
    )
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "done"
    assert "seam_report" not in worker.store.get(job_id)["params"]


# --- the per-job trellis-server config --------------------------------------


def _make_image_job_with(worker: Worker, **params) -> str:
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512, **params})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    return job_id


async def test_a_job_that_pins_a_server_axis_restarts_a_warm_server(worker):
    """The first job leaves the server warm on the config's own settings; the
    second pins a different band and must get its own server."""
    first = _make_image_job(worker)
    second = _make_image_job_with(worker, trellis_band=8)

    worker.start()
    await _wait_until(lambda: worker.store.get(second)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(first)["status"] == "done"
    assert worker.trellis.config_calls == [
        (worker.config.trellis_tex_res, worker.config.trellis_band),
        (worker.config.trellis_tex_res, 8),
    ]
    assert worker.trellis.restarts == 1
    assert worker.trellis.band == 8


async def test_a_job_matching_the_running_config_restarts_nothing(worker):
    _make_image_job(worker)
    _make_image_job(worker)

    worker.start()
    await _wait_until(
        lambda: all(j["status"] == "done" for j in worker.store.list(10))
    )
    await worker.shutdown()

    assert worker.trellis.restarts == 0


async def test_an_ordinary_job_after_a_pinned_one_restores_the_config(worker):
    """Nothing has to remember that a sweep changed the server: the check is
    against what is running, and every model-stage job resolves its own."""
    pinned = _make_image_job_with(worker, trellis_band=8, trellis_tex_res=2048)
    plain = _make_image_job(worker)

    worker.start()
    await _wait_until(lambda: worker.store.get(plain)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(pinned)["status"] == "done"
    assert worker.trellis.config_calls[-1] == (
        worker.config.trellis_tex_res,
        worker.config.trellis_band,
    )
    assert worker.trellis.restarts == 1


async def test_ensure_config_never_runs_on_the_event_loop(worker):
    """It calls stop(), which blocks for up to ~20 s in the real server."""
    _make_image_job_with(worker, trellis_band=8)
    loop_thread = threading.get_ident()

    worker.start()
    await _wait_until(lambda: worker.trellis.config_threads)
    await worker.shutdown()

    assert loop_thread not in worker.trellis.config_threads


async def test_a_reference_job_never_touches_the_server_config(worker):
    job_id = worker.store.create("text", "a chest", {"seed": 1}, stage="reference")

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.config_calls == []


async def test_ensure_config_stops_only_a_running_server_with_a_different_config(tmp_path):
    """The real thing, not the fake: the fields feed _argv, and there is no new
    spawn site -- a restart is a stop plus the existing lazy start."""
    from warlock.pipelines.trellis import TrellisServer

    server = TrellisServer(tmp_path / "x.exe", tmp_path / "models", 9999, tex_res=512)
    assert server.ensure_config(tex_res=1024, band=8) is False  # nothing running
    assert "--band" in server._argv()
    assert server._argv()[server._argv().index("--band") + 1] == "8"
    assert server._argv()[server._argv().index("--tex-res") + 1] == "1024"

    calls = []
    server.stop = lambda: calls.append(1)
    server._proc = SimpleNamespace(poll=lambda: None, pid=1)
    assert server.ensure_config(tex_res=1024, band=8) is False
    assert calls == []
    assert server.ensure_config(tex_res=1024, band=None) is True
    assert calls == [1]
    assert "--band" not in server._argv()
    server._proc = None


# --- observations: machine evidence written at completion ---------------------


def _fake_audit(monkeypatch):
    import warlock.meshaudit as meshaudit_mod

    monkeypatch.setattr(
        meshaudit_mod,
        "hole_fraction",
        lambda path, views, resolution: {
            "worst": 0.11, "mean": 0.05, "faces": 1234,
            "resolution": resolution, "views": [],
        },
    )


async def test_a_finished_model_job_leaves_one_observation_row(worker, monkeypatch):
    """The evidence corpus accumulates on every generation, not every review:
    a finished model job appends one observation carrying its config vector
    and the machine measurements, keyed to survive the job row's deletion."""
    _fake_audit(monkeypatch)
    job_id = _make_image_job(worker)

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    rows = worker.store.latest_observations()
    assert [r["job_id"] for r in rows] == [job_id]
    row = rows[0]
    assert row["seed"] == 1
    assert row["vector"]["stage"] == "model"
    assert row["vector"]["resolution"] == 512
    assert row["metrics"]["hole_worst"] == 0.11
    assert row["metrics"]["hole_mean"] == 0.05


async def test_a_job_refused_at_the_composition_gate_leaves_an_observation(
    worker, monkeypatch
):
    """Observations on refusal. The 17 refusals in the 2026-08-07 sweep wrote nothing
    at all, so findings.json could only ever report a checkpoint's accept rate
    *among the references that survived* -- which flatters exactly the
    checkpoints that fail most often. ``sdxl_cfg`` refused 3 of 5 and
    ``playground`` 0 of 5, and no reader could say so."""
    import warlock.pipelines.reference as reference_mod

    monkeypatch.setattr(
        reference_mod,
        "prepare",
        lambda *a, **k: reference_mod.Report(
            ok=False,
            reasons=("There is more than one object in the reference.",),
            codes=("multi_object",),
        ),
    )
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "error")
    await worker.shutdown()

    row = worker.store.latest_observations()[0]
    assert row["job_id"] == job_id
    assert row["metrics"]["refused"] == 1.0
    assert row["metrics"]["refused_multi_object"] == 1.0
    # And the settings it was refused under, or the rate is attributable to
    # nothing -- which is the whole point of recording it.
    assert row["vector"]["resolution"] == 512


async def test_an_errored_job_still_records_that_its_reference_passed(worker):
    """A job that cleared the gate and then died in trellis is a *passing*
    reference, and the refusal rate needs it: a mean over refusals alone is
    1.0 by construction. It carries no mesh metrics, because there is no mesh.
    """
    bad_id = _make_image_job(worker)
    worker.trellis.should_raise = RuntimeError("boom")

    worker.start()
    await _wait_until(lambda: worker.store.get(bad_id)["status"] == "error")
    await worker.shutdown()

    row = worker.store.latest_observations()[0]
    assert row["metrics"]["refused"] == 0.0
    assert "hole_worst" not in row["metrics"]


async def test_a_cancelled_job_leaves_no_observation(worker, monkeypatch):
    """A cancel is the user changing their mind, not a measurement of
    anything. It is also the one terminal status that discards artifacts, so a
    row about them would outlive what it describes."""
    import warlock.pipelines.reference as reference_mod

    monkeypatch.setattr(
        reference_mod, "prepare", lambda *a, **k: reference_mod.Report(ok=True)
    )
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.trellis.running)
    await worker.request_cancel(job_id)
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "cancelled")
    await worker.shutdown()

    assert worker.store.latest_observations() == []


async def test_a_reference_job_leaves_no_observation(worker):
    job_id = worker.store.create("text", "a chest", {"seed": 1}, stage="reference")

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.store.latest_observations() == []


async def test_observe_finished_skips_a_job_with_no_measurements(worker):
    """A mesh whose audit and report both failed says nothing measurable; an
    empty metrics row would be a bucket that dilutes every mean it joins."""
    from warlock.queue import _observe_finished

    job_id = worker.store.create(
        "image", None, {"seed": 1}, stage="model", status="done"
    )
    assert _observe_finished(worker.store, job_id) is False
    assert worker.store.latest_observations() == []


async def test_observe_finished_snapshots_the_sweep_context(worker):
    from warlock.queue import _observe_finished

    job_id = worker.store.create(
        "image", "a chest", {"seed": 7, "mesh_audit": {"worst": 0.1, "mean": 0.05}},
        stage="model", status="done",
        sweep_id="deadbeefcafe", sweep_unit="baseline s7",
    )
    assert _observe_finished(worker.store, job_id) is True
    row = worker.store.latest_observations()[0]
    assert row["sweep_id"] == "deadbeefcafe"
    assert row["sweep_unit"] == "baseline s7"
    assert row["seed"] == 7
    assert row["metrics"] == {"hole_worst": 0.1, "hole_mean": 0.05}


async def test_a_failing_observation_write_does_not_fail_the_job(worker, monkeypatch):
    """Same rule as _audit_mesh: a diagnostic must never fail a job whose mesh
    is already on disk."""
    _fake_audit(monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(worker.store, "add_observation", boom, raising=False)
    job_id = _make_image_job(worker)

    worker.start()
    await _wait_until(
        lambda: worker.store.get(job_id)["status"] in ("done", "error")
    )
    await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "done"
    assert worker.store.latest_observations() == []


async def test_rank_candidates_off_skips_the_anchor_and_keeps_the_composition_score(
    tmp_path, monkeypatch
):
    """``WARLOCK_RANK=off`` had one reader and nothing asserting it.

    The switch is about the *anchor* half only: that half needs a ref.png and an
    optional DINOv2 download, so it is opportunistic three ways over. The
    composition half is free -- the report was measured either way -- so turning
    ranking off must not turn scoring off.
    """
    from warlock.bench import metrics

    worker = _make_worker(tmp_path, rank_candidates=False)
    job_dir = tmp_path / "assets" / "j"
    job_dir.mkdir(parents=True)
    (job_dir / "ref.png").write_bytes(b"not read")

    def refuse(*_args, **_kwargs):
        raise AssertionError("the anchor half ran with ranking off")

    monkeypatch.setattr(metrics, "dino_available", refuse)

    scored = worker._rank_reference(job_dir / "input.png", {"reference_report": {}})
    assert "cosine" not in scored or scored["cosine"] is None
    assert scored, "the composition half is free and still reported"


async def test_rank_candidates_on_consults_the_anchor_when_one_exists(tmp_path, monkeypatch):
    from warlock.bench import metrics

    worker = _make_worker(tmp_path, rank_candidates=True)
    job_dir = tmp_path / "assets" / "j"
    job_dir.mkdir(parents=True)
    (job_dir / "ref.png").write_bytes(b"not read")

    asked: list[bool] = []
    monkeypatch.setattr(metrics, "dino_available", lambda *_a, **_k: asked.append(True))

    worker._rank_reference(job_dir / "input.png", {"reference_report": {}})
    assert asked == [True]


# --- the layering rule, pinned ----------------------------------------------


async def test_the_worker_never_imports_service_studio_or_imgui():
    """``queue.py`` may not import ``service`` -- and now neither may the
    ``_q_*.py`` siblings that hold the rest of ``Worker``'s methods.

    The rule is old and the reason has not changed: the worker is the layer
    *below* the service, it runs on its own thread with no imgui context, and
    an import in the other direction would make the dependency a cycle the
    moment ``service`` reached for a queue constant. Function-body imports
    count -- this package's house style uses them for deferral, and one of
    those is exactly how the rule would be broken by accident.

    There are **no allowances**, and there is history in that. ``_q_ground``
    used to hold one -- it reached into ``studio.plotter.groundtex`` for the
    forty-seven-case compositor, on the argument that a second copy of the tile
    geometry in ``pipelines/`` is how a worker's atlas comes to disagree with
    the editor that paints with it. That module was deleted on 2026-08-18 when
    tile sheets moved to Create, and its replacement needs nothing from the
    studio: sixty-four crops and a paste live in ``pipelines.tilesheet``, where
    the queue is allowed to look. So the allowlist is empty and the rule is now
    absolute. An entry here is a decision to argue for, not a line to add.

    ``async def`` only because this module marks every test asyncio; there is
    nothing to await.
    """
    import ast
    from pathlib import Path

    src = Path(__import__("warlock").__file__).parent
    files = [src / "queue.py"] + sorted(src.glob("_q_*.py"))
    assert len(files) >= 6, "the mixin siblings are not being scanned"

    allowed: set[tuple[str, str]] = set()
    offenders: list[str] = []
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level>0 is a relative import; ``from . import x`` has no
                # module, and ``from .service import jobs`` has module
                # "service" -- both spellings have to be caught.
                base = node.module or ""
                names = [base] if base else []
                names += [f"{base}.{a.name}" if base else a.name for a in node.names]
            for name in names:
                if (path.name, name) in allowed:
                    continue
                root = name.split(".")[0]
                if root in ("service", "studio") or "imgui" in name:
                    offenders.append(f"{path.name}: {name}")
                if name.startswith(("warlock.service", "warlock.studio")):
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []


async def test_a_happy_path_job_really_ran_its_post_processing(worker):
    """ART-02: a ``done`` assertion used to prove almost nothing.

    Every post-processing step -- parse, normalize, audit, report -- is wrapped
    in a catch-everything handler, and the fakes wrote ``b"fake-glb"``. So the
    canonical "the job finished" test was asserting that the *degraded* path
    completes: nothing parsed, every step raised and was swallowed, and the row
    reached ``done`` anyway. Which is precisely what happens when a real
    reconstruction comes back malformed, so the happy-path and degraded-path
    tests were indistinguishable.

    With a valid artifact and ART-01's health record there is finally something
    to assert: the steps ran, and *nothing* was swallowed.
    """
    job_id = _make_image_job(worker)
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()

    job = worker.store.get(job_id)
    assert job["status"] == "done", job["error"]
    params = job["params"]
    assert "degraded" not in params, (
        f"a step was swallowed on the happy path: {params.get('degraded')}"
    )
    # And the evidence each step leaves behind, so this cannot pass by the
    # steps having been skipped rather than having succeeded.
    assert params["transform"], "grounding did not run"
    assert params["mesh_audit"], "the silhouette audit did not run"
    assert params["mesh_report"], "the mesh report did not run"
