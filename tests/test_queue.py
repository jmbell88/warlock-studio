from __future__ import annotations

import asyncio
import threading
import time

import pytest

from warlock.config import Config
from warlock.db import JobStore
from warlock.queue import SHUTDOWN_TIMEOUT, Worker

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


async def test_composed_prompt_is_read_from_last_prompt_not_recomputed(worker):
    """queue.py must record t2i.last_prompt, not its own local `composed` --
    otherwise the UI's "prompt sent" row would show the pre-trigger,
    pre-PROMPT_TEMPLATE string forever, as it did before this change."""
    job_id = worker.store.create(
        "text", "a barrel", {"seed": 1, "resolution": 512, "genre": "scifi"},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert (
        worker.store.get(job_id)["params"]["composed_prompt"]
        == worker._text2image.last_prompt
    )


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
