"""Worker dispatch for rig jobs, with the Blender subprocess faked out.

The subprocess boundary itself is covered in test_rigging.py; what matters
here is that the queue treats a rig job as a first-class kind -- right
artifacts, right cancellation, and no collateral damage to the mesh it reads.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from warlock import rigging
from warlock.config import Config
from warlock.db import JobStore
from warlock.queue import Worker

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


def _mesh_job(worker: Worker, **params) -> str:
    """A finished text job with a mesh on disk -- something worth rigging."""
    job_id = worker.store.create("text", "a knight", {"seed": 1, **params})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    worker.store.set_status(job_id, "done")
    return job_id


def _fake_worker_run(monkeypatch, *, result=None, side_effect=None, hold=None):
    """Replace rigging.run_worker, recording the spec it was handed."""
    calls: list[dict] = []

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0):
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
