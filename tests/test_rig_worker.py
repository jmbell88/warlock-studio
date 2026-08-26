"""Worker dispatch for rig jobs, with the Blender subprocess faked out.

The subprocess boundary itself is covered in test_rigging.py; what matters
here is that the queue treats a rig job as a first-class kind -- right
artifacts, right cancellation, and no collateral damage to the mesh it reads.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from warlock import rigging
from warlock.config import Config
from warlock.db import JobStore
from warlock.queue import Worker

# No module-level asyncio mark: pyproject sets asyncio_mode = "auto", which
# already collects every async test here. Applying it to the module as well
# also applied it to this file's one *synchronous* test, which pytest-asyncio
# reports as a mistake -- and it is one.


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        # Pinned empty, not left at PROJECT_ROOT/models. The landmark gate
        # stats this root for the pose weights, so without it every assertion
        # below about the detector *not* being consulted would depend on
        # whether whoever ran the suite happens to have downloaded vitpose --
        # the same rule conftest's svc fixture keeps for gltfpack. Tests that
        # want the detector present fake it (_fake_detection).
        t2i_model_root=tmp_path / "t2i-models",
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


def _mesh_job(worker: Worker, **params) -> str:
    """A finished text job with a mesh on disk -- something worth rigging."""
    job_id = worker.store.create("text", "a knight", {"seed": 1, **params})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    worker.store.set_status(job_id, "done")
    return job_id


class _Calls(list):
    """Rig runs, with the deformation-QA sheet runs kept to one side.

    A rig now ends in a second Blender run -- the QA sheet -- and mixing the
    two into one list would make every ``len(calls) == 1`` here a statement
    about the sheet as well as about the rig, which is not what any of them
    mean.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sheets: list[dict] = []


def _fake_worker_run(monkeypatch, *, result=None, side_effect=None, hold=None):
    """Replace rigging.run_worker, recording the spec it was handed."""
    calls = _Calls()

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0):
        if spec.get("op") == "sheet":
            # What op_sheet does, minus EEVEE: one square transparent PNG per
            # cell, which is all the host-side packer reads.
            from PIL import Image

            calls.sheets.append({"spec": spec, "timeout": timeout})
            size = int(spec["frame_size"])
            for cell in spec["cells"]:
                frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
                frame.save(Path(spec["frames_dir"]) / f"{cell['index']:04d}.png")
            return {"ok": True, "frames": [c["index"] for c in spec["cells"]]}
        calls.append({"spec": spec, "timeout": timeout})
        if on_progress is not None:
            on_progress(0.4, "Computing weights")
        if hold is not None:
            hold.wait(timeout=10)
        if side_effect is not None:
            raise side_effect
        # The real worker writes these; the queue's cancel bookkeeping keys
        # off them, so the fake must too.
        Path(spec["out_glb"]).write_bytes(b"fake-rig")
        Path(spec["out_json"]).write_bytes(b'{"bones": []}')
        return result or {"ok": True, "weighting": "automatic", "bones": 19}

    monkeypatch.setattr(rigging, "run_worker", fake)
    return calls


async def test_rig_job_runs_blender_against_the_source_job_dir(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source, "template": "quadruped"})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert len(calls) == 1
    spec = calls[0]["spec"]
    # The rig belongs to the mesh: artifacts land beside model.glb in the
    # source job's directory, not in the rig job's own. The worker writes to
    # temp names there and the queue renames them into place on success.
    source_dir = worker.config.job_dir(source)
    assert spec["source_glb"] == str(source_dir / "model.glb")
    assert Path(spec["out_glb"]).parent == source_dir
    assert (source_dir / "rig.glb").exists()
    assert (source_dir / "rig.json").exists()
    assert spec["template"] == "quadruped"
    assert calls[0]["timeout"] == worker.config.rig_timeout
    # Recorded on the job so the history row can say "envelope" without
    # every card fetching rig.json.
    assert worker.store.get(rig_id)["params"]["weighting"] == "automatic"
    await worker.shutdown()


async def test_an_envelope_fallback_records_why_beside_the_weighting(worker, monkeypatch):
    """The reason used to exist only as a print inside the subprocess, so a rig
    that quietly degraded looked exactly like one that did not."""
    _fake_worker_run(
        monkeypatch,
        result={
            "ok": True,
            "weighting": "envelope",
            "weighting_reason": "bone-heat weighting failed: produced no vertex weights",
            "bones": 19,
        },
    )
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")
    params = worker.store.get(rig_id)["params"]
    assert params["weighting"] == "envelope"
    assert params["weighting_reason"] == "bone-heat weighting failed: produced no vertex weights"
    await worker.shutdown()


async def test_rig_job_falls_back_to_the_configured_template(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")
    assert calls[0]["spec"]["template"] == worker.config.rig_template
    await worker.shutdown()


async def test_rig_failure_is_reported_as_an_error_not_a_dead_worker(worker, monkeypatch):
    _fake_worker_run(monkeypatch, side_effect=rigging.BlenderError("bpy exploded"))
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "error")
    assert worker.alive
    # The mesh it was reading is untouched.
    assert (worker.config.job_dir(source) / "model.glb").exists()
    await worker.shutdown()


async def test_cancelling_a_rig_never_deletes_the_source_mesh(worker, monkeypatch):
    """The regression this guards: a cancelled job used to unconditionally
    unlink model.glb. For a rig job that file is the *input*, belonging to a
    different, successful job."""
    hold = threading.Event()
    _fake_worker_run(monkeypatch, hold=hold)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.current_job_id == rig_id)
    await worker.request_cancel(rig_id)
    hold.set()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "cancelled")

    assert (worker.config.job_dir(source) / "model.glb").exists()
    # ...but the half-finished rig is gone.
    assert not (worker.config.job_dir(source) / "rig.glb").exists()
    assert not (worker.config.job_dir(source) / "rig.json").exists()
    await worker.shutdown()


async def test_cancelling_a_rig_kills_the_blender_subprocess(worker, monkeypatch):
    killed: list[bool] = []

    class FakeProc:
        def poll(self):
            return None

        def kill(self):
            killed.append(True)

    hold = threading.Event()

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0):
        if on_progress is not None:
            on_progress(0.4, "Computing weights")  # puts the phase in "rig"
        if on_start is not None:
            on_start(FakeProc())
        hold.wait(timeout=10)
        raise rigging.BlenderError("killed")

    monkeypatch.setattr(rigging, "run_worker", fake)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.progress.snapshot(rig_id) is not None
                      and worker.progress.snapshot(rig_id)["phase"] == "rig")
    await worker.request_cancel(rig_id)
    hold.set()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "cancelled")
    assert killed, "cancel must kill the subprocess; bpy checks nothing itself"
    await worker.shutdown()


async def test_rig_progress_uses_the_whole_bar(worker, monkeypatch):
    """PHASES_RIG gives the rig phase 0..1, so a worker fraction of 0.4 must
    read as 40% -- not as a slice of some other pipeline's budget."""
    hold = threading.Event()
    _fake_worker_run(monkeypatch, hold=hold)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(
        lambda: (snap := worker.progress.snapshot(rig_id)) is not None
        and snap["percent"] >= 40.0
    )
    hold.set()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")
    await worker.shutdown()


async def test_cancelling_a_rerig_keeps_the_previous_rig(worker, monkeypatch):
    """A re-rig is a correction of an existing, successful rig. Cancelling the
    correction must not destroy the artifacts the first rig job produced --
    every saved pose bakes against that rig.glb."""
    hold = threading.Event()
    _fake_worker_run(monkeypatch, hold=hold)
    source = _mesh_job(worker)
    source_dir = worker.config.job_dir(source)
    (source_dir / "rig.glb").write_bytes(b"old-rig")
    (source_dir / "rig.json").write_bytes(b'{"bones": ["old"]}')
    rig_id = worker.store.create("rig", None, {"source_job": source, "adjusted": True})

    worker.start()
    await _wait_until(lambda: worker.current_job_id == rig_id)
    await worker.request_cancel(rig_id)
    hold.set()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "cancelled")

    assert (source_dir / "rig.glb").read_bytes() == b"old-rig"
    assert (source_dir / "rig.json").read_bytes() == b'{"bones": ["old"]}'
    await worker.shutdown()


async def test_the_worker_is_never_handed_the_served_rig_paths(worker, monkeypatch):
    """Blender writes its GLB in place over seconds, and rig.json is the
    completion gate -- which an earlier rig already satisfied. So the worker
    must write to temp names and the queue renames them into place on success,
    or a re-rig serves a truncated rig.glb to anyone who asks mid-run."""
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    source_dir = worker.config.job_dir(source)
    (source_dir / "rig.glb").write_bytes(b"old-rig")
    (source_dir / "rig.json").write_bytes(b'{"bones": ["old"]}')
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    spec = calls[0]["spec"]
    assert spec["out_glb"] != str(source_dir / "rig.glb")
    assert spec["out_json"] != str(source_dir / "rig.json")
    # ...but the served names end up carrying the new artifacts, atomically.
    assert (source_dir / "rig.glb").read_bytes() == b"fake-rig"
    assert (source_dir / "rig.json").read_bytes() == b'{"bones": []}'
    # No temp files left behind in the source job's directory.
    assert not (source_dir / rigging.RIG_GLB_TMP).exists()
    assert not (source_dir / rigging.RIG_JSON_TMP).exists()
    await worker.shutdown()


# --- the opt-in checkbox ----------------------------------------------------


async def test_a_generate_job_with_rig_set_queues_a_follow_up(worker, monkeypatch):
    _fake_worker_run(monkeypatch)
    job_id = worker.store.create(
        "image", None, {"seed": 1, "resolution": 512, "rig": True, "rig_template": "quadruped"}
    )
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await _wait_until(lambda: any(j["kind"] == "rig" for j in worker.store.list()))

    rig_job = next(j for j in worker.store.list() if j["kind"] == "rig")
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["template"] == "quadruped"
    assert rig_job["params"]["auto"] is True
    await worker.shutdown()


async def test_no_follow_up_without_the_flag(worker, monkeypatch):
    _fake_worker_run(monkeypatch)
    job_id = worker.store.create("image", None, {"seed": 1, "resolution": 512})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")

    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await asyncio.sleep(0.2)
    assert not any(j["kind"] == "rig" for j in worker.store.list())
    await worker.shutdown()


async def test_a_rig_enqueue_failure_is_recorded_on_the_finished_mesh(
    worker, monkeypatch
):
    _fake_worker_run(monkeypatch)
    real_create = worker.store.create

    def fail_rig(kind, *args, **kwargs):
        if kind == "rig":
            raise OSError("database is read-only")
        return real_create(kind, *args, **kwargs)

    job_id = worker.store.create(
        "image", None, {"seed": 1, "resolution": 512, "rig": True}
    )
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.png").write_bytes(b"fake-png")
    monkeypatch.setattr(worker.store, "create", fail_rig)

    worker.start()
    await _wait_until(
        lambda: "followup_failures" in worker.store.get(job_id)["params"]
    )
    job = worker.store.get(job_id)
    assert job["status"] == "done"
    failure = job["params"]["followup_failures"]["rig"]
    assert failure["error_type"] == "OSError"
    assert failure["message"] == "database is read-only"
    await worker.shutdown()


async def test_a_rig_job_does_not_recursively_queue_another(worker, monkeypatch):
    """params flows from the generate job, so a rig job can inherit rig=True.
    It must not spawn a rig of a rig."""
    _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    worker.store.create("rig", None, {"source_job": source, "rig": True})

    worker.start()
    await _wait_until(lambda: all(j["status"] != "queued" for j in worker.store.list()))
    await asyncio.sleep(0.2)
    assert sum(1 for j in worker.store.list() if j["kind"] == "rig") == 1
    await worker.shutdown()


def test_fbx_spec_names_the_op_and_paths(tmp_path):
    from warlock import rigging

    spec = rigging.fbx_spec(tmp_path / "model.glb", tmp_path / "model.fbx", tmp_path)
    assert spec["op"] == "fbx"
    assert spec["source_glb"].endswith("model.glb")
    assert spec["out_fbx"].endswith("model.fbx")
    assert spec["result_path"].startswith(str(tmp_path))


async def test_a_rig_job_passes_corrected_joints_through_to_the_worker(worker, monkeypatch):
    """The adjust pass re-skins with the user's joints; the queue is the only
    thing between the route and the worker, so it must not drop them."""
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    rig_id = worker.store.create(
        "rig",
        None,
        {"source_job": source, "template": "humanoid", "bones": fitted, "adjusted": True},
    )

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert calls[0]["spec"]["bones"] == fitted
    await worker.shutdown()


async def test_an_ordinary_rig_job_sends_no_bones_so_the_worker_fits(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert "bones" not in calls[0]["spec"]
    await worker.shutdown()


# --- landmark-informed joint placement ---------------------------------------
#
# The mesh is a reconstruction of the source job's input.png, so that image
# says where the subject's joints are. What is tested here is only the gating:
# when the queue asks the detector at all, and that no answer it gives -- none,
# nonsense, or an exception -- can cost the rig. Where a joint lands is
# tests/test_pose2d.py's subject.


def _fake_detection(monkeypatch, *, bones="landmarks", available=True, raises=None):
    """Stand in for the whole pose2d model half, recording every call."""
    from warlock.pipelines import pose2d

    calls: list[Path] = []

    def fake_detect(image_path, subject_bbox, config=None):
        calls.append(Path(image_path))
        if raises is not None:
            raise raises
        return [pose2d.Keypoint(n, 1.0, 1.0, 0.9) for n in pose2d.COCO_KEYPOINTS]

    monkeypatch.setattr(pose2d, "available", lambda config=None: available)
    monkeypatch.setattr(pose2d, "detect", fake_detect)
    monkeypatch.setattr(
        pose2d,
        "refit",
        lambda template, keypoints, bbox: (
            None if bones is None else [dict(b) for b in template.bones]
        ),
    )
    return calls


def _rigged_reference(worker, **params) -> str:
    """A finished job with both a mesh and the reference it was made from."""
    job_id = _mesh_job(worker, **params)
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (128, 256), (240, 240, 240))
    ImageDraw.Draw(image).rectangle((40, 30, 88, 226), fill=(20, 20, 20))
    image.save(worker.config.job_dir(job_id) / "input.png")
    return job_id


async def test_a_humanoid_rig_asks_the_reference_where_the_joints_are(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    detections = _fake_detection(monkeypatch)
    source = _rigged_reference(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source, "template": "humanoid"})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert detections == [worker.config.job_dir(source) / "input.png"]
    spec = calls[0]["spec"]
    assert spec["template_bones"]
    assert spec["fit"]["method"] == "pose2d"
    assert spec["fit"]["model"]
    assert 0.0 <= spec["fit"]["confidence"] <= 1.0
    await worker.shutdown()


async def test_a_mesh_with_no_reference_image_gets_the_bbox_fit(worker, monkeypatch):
    """An imported or hand-modelled asset has no input.png, and there is
    nothing to read joints off. It rigs exactly as it always did."""
    calls = _fake_worker_run(monkeypatch)
    detections = _fake_detection(monkeypatch)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert detections == []
    assert "template_bones" not in calls[0]["spec"]
    await worker.shutdown()


def test_a_mesh_with_no_reference_is_not_offered_to_the_detector(worker, monkeypatch):
    """The gate itself, rather than what it prevents.

    ``measure_file`` would decline a missing file too, so the test above passes
    either way -- what this one pins is that a rig of an imported mesh does not
    pay a thread hop and a model load to be told there is no image, which is
    the only thing the check buys.
    """
    _fake_detection(monkeypatch)
    source_dir = worker.config.job_dir(rigging.new_id())
    source_dir.mkdir(parents=True, exist_ok=True)

    assert worker._wants_landmarks(source_dir, "humanoid", {}) is False
    (source_dir / "input.png").write_bytes(b"a reference, of some description")
    assert worker._wants_landmarks(source_dir, "humanoid", {}) is True


async def test_joints_the_user_corrected_are_never_second_guessed(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    detections = _fake_detection(monkeypatch)
    source = _rigged_reference(worker)
    fitted = rigging.fit_template(rigging.get_template("humanoid"), [-1, -1, 0], [1, 1, 2])
    rig_id = worker.store.create(
        "rig", None, {"source_job": source, "bones": fitted, "adjusted": True}
    )

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert detections == [], "adjust-joints is the user overruling every fit"
    assert "template_bones" not in calls[0]["spec"]
    await worker.shutdown()


async def test_the_kill_switch_stops_the_detector_being_asked(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    detections = _fake_detection(monkeypatch)
    worker.config.pose_fit = False
    source = _rigged_reference(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert detections == []
    assert "template_bones" not in calls[0]["spec"]
    await worker.shutdown()


async def test_a_non_humanoid_template_is_never_asked(worker, monkeypatch):
    """COCO-17 is a human skeleton. There is no mapping from it onto a
    quadruped's shoulder, so the question is not worth asking."""
    calls = _fake_worker_run(monkeypatch)
    detections = _fake_detection(monkeypatch)
    source = _rigged_reference(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source, "template": "quadruped"})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert detections == []
    assert "template_bones" not in calls[0]["spec"]
    await worker.shutdown()


async def test_no_weights_means_the_rig_runs_exactly_as_it_did_before(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    detections = _fake_detection(monkeypatch, available=False)
    source = _rigged_reference(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert detections == []
    assert "template_bones" not in calls[0]["spec"]
    assert "fit" not in calls[0]["spec"]
    await worker.shutdown()


async def test_a_refused_detection_falls_back_to_the_bbox_fit(worker, monkeypatch):
    """refit returning None is a sanity gate refusing -- a figure the landmarks
    do not describe. The rig still happens, on the fit that always works."""
    calls = _fake_worker_run(monkeypatch)
    _fake_detection(monkeypatch, bones=None)
    source = _rigged_reference(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert "template_bones" not in calls[0]["spec"]
    await worker.shutdown()


async def test_a_detector_that_raises_never_fails_the_rig(worker, monkeypatch):
    """Log and swallow, the rule every diagnostic in the worker follows: a
    better skeleton is an optimisation, and an optimisation must not be able to
    fail a job whose mesh is already on disk."""
    calls = _fake_worker_run(monkeypatch)
    _fake_detection(monkeypatch, raises=RuntimeError("the model exploded"))
    source = _rigged_reference(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert "template_bones" not in calls[0]["spec"]
    assert (worker.config.job_dir(source) / "rig.json").exists()
    await worker.shutdown()


def test_the_kill_switch_reads_the_environment(monkeypatch):
    import warlock.config as config_mod

    monkeypatch.setenv("WARLOCK_POSE_FIT", "0")
    monkeypatch.setattr(config_mod, "_config", None)
    assert config_mod.get_config().pose_fit is False
    monkeypatch.setenv("WARLOCK_POSE_FIT", "1")
    monkeypatch.setattr(config_mod, "_config", None)
    assert config_mod.get_config().pose_fit is True


# --- the deformation QA sheet ------------------------------------------------
#
# Human-reviewable and nothing more: this scores nothing and gates nothing. A
# weighting method that *reports* success says only that the solve produced
# numbers, and looking at the mesh bent is how you find out whether those
# numbers deform it sensibly. Scoring is a later phase's problem.


async def test_a_rig_renders_the_deformation_battery_beside_the_mesh(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    source_dir = worker.config.job_dir(source)
    # Beside model.glb, for the reason the rig itself is: it describes that
    # mesh, not the request that produced it.
    assert (source_dir / "rig_qa.png").exists()
    meta = json.loads(rigging.rig_qa_path(source_dir).read_text(encoding="utf-8"))
    assert meta["rows"] == len(rigging.deform_battery("humanoid"))
    # Rendered *through* the rig, so the sheet depicts the weights under test.
    assert calls.sheets[0]["spec"]["source_glb"] == str(source_dir / "rig.glb")
    assert calls.sheets[0]["spec"]["lighting"] == "lit"
    # And attached to the rig job, so a card can say a battery was rendered
    # without opening a file per row.
    assert worker.store.get(rig_id)["params"]["deform_qa"]["poses"][0] == "squat"
    await worker.shutdown()


async def test_every_battery_row_is_posed_rather_than_the_first_one_four_times(
    worker, monkeypatch
):
    """The cells carry their rotations inline -- the worker cannot read a job
    directory -- and a row is identified by its pose id, which is exactly what
    a battery of id-less rows would collapse."""
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    cells = calls.sheets[0]["spec"]["cells"]
    assert len({tuple(sorted(c["bones"])) for c in cells}) == len(
        rigging.deform_battery("humanoid")
    )
    assert all(c["bones"] for c in cells), "a battery row with no rotations is the rest pose"
    await worker.shutdown()


async def test_a_battery_that_fails_never_fails_the_rig(worker, monkeypatch):
    """Log and swallow, the ``_audit_mesh`` rule: the rig is already published,
    and a review render is the improvement nobody asked for."""
    calls = _fake_worker_run(monkeypatch)
    real = rigging.run_worker

    def explode(spec, **kw):
        if spec.get("op") == "sheet":
            raise RuntimeError("EEVEE fell over")
        return real(spec, **kw)

    monkeypatch.setattr(rigging, "run_worker", explode)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    job = worker.store.get(rig_id)
    assert job["status"] == "done"
    assert job["params"]["weighting"] == "automatic"
    assert "deform_qa" not in job["params"]
    assert not (worker.config.job_dir(source) / "rig_qa.json").exists()
    assert calls, "the rig itself still ran"
    await worker.shutdown()


async def test_the_battery_kill_switch_leaves_a_rig_byte_for_byte_as_it_was(
    worker, monkeypatch
):
    calls = _fake_worker_run(monkeypatch)
    worker.config.deform_qa = False
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert calls.sheets == []
    assert not (worker.config.job_dir(source) / "rig_qa.json").exists()
    await worker.shutdown()


async def test_a_template_with_no_battery_simply_renders_nothing(worker, monkeypatch):
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    rig_id = worker.store.create("rig", None, {"source_job": source, "template": "quadruped"})

    worker.start()
    await _wait_until(lambda: worker.store.get(rig_id)["status"] == "done")

    assert calls.sheets == []
    assert worker.store.get(rig_id)["params"]["weighting"] == "automatic"
    await worker.shutdown()


def test_the_battery_kill_switch_reads_the_environment(monkeypatch):
    import warlock.config as config_mod

    monkeypatch.setenv("WARLOCK_DEFORM_QA", "0")
    monkeypatch.setattr(config_mod, "_config", None)
    assert config_mod.get_config().deform_qa is False
    monkeypatch.setenv("WARLOCK_DEFORM_QA", "1")
    monkeypatch.setattr(config_mod, "_config", None)
    assert config_mod.get_config().deform_qa is True


# --- the worker's own stdin/result contract ----------------------------------


def test_a_spec_that_is_not_json_exits_two_with_a_sentence(monkeypatch, capsys, tmp_path):
    """fetch_worker's rule on the other worker: a malformed spec is reported in
    a sentence and an exit code, never as a traceback the user cannot read."""
    import io
    import sys as _sys

    from warlock.pipelines import blender_worker

    monkeypatch.setattr(_sys, "stdin", io.StringIO("not json"))
    assert blender_worker.main() == 2
    assert "not usable" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_a_spec_with_no_result_path_exits_two_rather_than_running_the_op(monkeypatch, capsys):
    import io
    import sys as _sys

    from warlock.pipelines import blender_worker

    def explode(_bpy, _spec):
        raise AssertionError("the op ran for a spec that could not hand anything back")

    monkeypatch.setitem(blender_worker.OPS, "rig", explode)
    monkeypatch.setattr(_sys, "stdin", io.StringIO(json.dumps({"op": "rig"})))
    assert blender_worker.main() == 2
    assert "not usable" in capsys.readouterr().err


def test_a_good_run_stages_its_result_and_leaves_no_tmp_sibling(monkeypatch, tmp_path):
    """The host polls for this file's existence, so a partial write is a result
    it would parse as a failure of the op rather than of the write."""
    import io
    import sys as _sys
    import types

    from warlock.pipelines import blender_worker

    result = tmp_path / ".blender_result.json"
    monkeypatch.setitem(blender_worker.OPS, "rig", lambda _bpy, _spec: {"ok": True, "bones": 3})
    monkeypatch.setitem(_sys.modules, "bpy", types.ModuleType("bpy"))
    monkeypatch.setattr(
        _sys, "stdin", io.StringIO(json.dumps({"op": "rig", "result_path": str(result)}))
    )
    assert blender_worker.main() == 0
    assert json.loads(result.read_text(encoding="utf-8")) == {"ok": True, "bones": 3}
    assert [p.name for p in tmp_path.iterdir()] == [result.name]


async def test_a_cancel_after_the_rig_is_published_still_records_it_as_done(
    worker, monkeypatch
):
    """The window between ``finalize_rig`` and the QA tail.

    ``_discard_artifacts`` deliberately removes only the temps -- a cancelled
    re-rig must not destroy an earlier rig -- so once finalize has renamed onto
    the served names the artifact is real and cannot be taken back. Recording
    the row as "cancelled" then leaves a valid rig under a row saying it never
    happened, and every follow-up is gated on ``status == "done"``, so the
    Troupe chain stalls with a perfectly good rig nobody points at.
    """
    _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    source_dir = worker.config.job_dir(source)
    rig_id = worker.store.create("rig", None, {"source_job": source})

    published = threading.Event()
    release = threading.Event()
    real_finalize = rigging.finalize_rig

    def finalize(*args, **kwargs):
        out = real_finalize(*args, **kwargs)
        published.set()
        release.wait(10.0)
        return out

    monkeypatch.setattr(rigging, "finalize_rig", finalize)

    worker.start()
    await _wait_until(published.is_set)
    await worker.request_cancel(rig_id)
    release.set()
    await _wait_until(
        lambda: worker.store.get(rig_id)["status"] in ("done", "cancelled", "error")
    )

    job = worker.store.get(rig_id)
    assert job["status"] == "done", "a published rig must not be recorded as cancelled"
    assert (source_dir / "rig.glb").read_bytes() == b"fake-rig"
    await worker.shutdown()


def test_a_publish_that_lands_the_glb_but_not_the_json_leaves_no_stale_marker(
    tmp_path, monkeypatch
):
    """``finalize_rig`` renames GLB then JSON, and each retries on its own. If
    the JSON rename exhausts its retries after the GLB already landed, the
    served directory held a *new* rig.glb beside the *old* rig.json -- the
    completion marker advertising a skeleton that no longer matched the mesh.
    The honest state is "not rigged": the stale marker goes, and the caller's
    ``discard_rig_temps`` takes the unpublished JSON."""
    job_dir = tmp_path
    (job_dir / "rig.glb").write_bytes(b"old")
    (job_dir / "rig.json").write_text('{"old": true}')
    (job_dir / rigging.RIG_GLB_TMP).write_bytes(b"new")
    (job_dir / rigging.RIG_JSON_TMP).write_text('{"new": true}')
    real_replace = os.replace

    def replace(src, dst):
        if str(dst).endswith("rig.json"):
            raise PermissionError("mid-stream reader")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        rigging.finalize_rig(job_dir)
    assert (job_dir / "rig.glb").read_bytes() == b"new"
    assert not (job_dir / "rig.json").exists()
    assert (job_dir / rigging.RIG_JSON_TMP).exists()
