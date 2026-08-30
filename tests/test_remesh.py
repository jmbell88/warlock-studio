"""The game-ready remesh: the pure module, the door, and the worker stage.

Blender is faked at ``rigging.run_worker`` exactly as ``test_rig_worker.py``
does; what is under test is that the queue treats a remesh as a rework of
the *source* job -- publishes over its ``model.glb`` by rename, invalidates
every derived export, never touches ``source.glb`` -- and that the door
refuses what the worker could not do.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from warlock import followups, progress, rigging
from warlock.config import Config
from warlock.db import JobStore
from warlock.pipelines import blender_worker, remesh
from warlock.queue import Worker
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Conflict, Invalid
from warlock.service.validation import DERIVED_PARAMS
from warlock.studio import asset_open
from warlock.studio.panes import remesh_panel

# --- the pure module -------------------------------------------------------------


def test_profiles_resolve_to_their_quad_counts():
    for key, faces in remesh.FACE_PROFILES.items():
        assert remesh.resolve(key) == faces
    assert remesh.resolve("custom", 1234) == 1234


@pytest.mark.parametrize("bad", [None, "x", remesh.FACES_MIN - 1, remesh.FACES_MAX + 1])
def test_a_custom_budget_outside_the_range_is_refused_by_name(bad):
    with pytest.raises(ValueError):
        remesh.resolve("custom", bad)


def test_an_unknown_profile_names_the_choices():
    with pytest.raises(ValueError, match="custom"):
        remesh.resolve("enormous")


def test_the_label_is_derived_from_the_table():
    assert remesh.profile_label("low") == "Low (2k quads)"
    assert remesh.profile_label("custom") == "Custom..."


def test_the_report_line_never_calls_a_decimated_mesh_quads():
    quad = remesh.report_line(
        {"faces": 8000, "quads": 0.98, "method": "quadriflow", "texture_size": 1024}
    )
    assert "98% quads" in quad and "1024 px" in quad
    tri = remesh.report_line({"faces": 16000, "quads": 0.0, "method": "decimate"})
    assert "decimated" in tri and "quads" not in tri
    assert remesh.report_line(None) is None
    assert remesh.report_line({"method": "quadriflow"}) is None


def test_a_failed_tiercheck_is_said_out_loud():
    line = remesh.report_line(
        {"faces": 100, "quads": 1.0, "method": "quadriflow",
         "tiercheck": {"ok": False, "failures": ["uv_primitives"]}}
    )
    assert "lost: uv_primitives" in line


def test_the_worker_side_constants_match_the_host_side():
    # The bpy side may not import the host package, so the numbers are restated.
    assert blender_worker.VOXEL_FRACTION == remesh.VOXEL_FRACTION
    assert blender_worker.BAKE_MARGIN_PX == remesh.BAKE_MARGIN_PX


def test_the_derived_list_is_the_services_own():
    from warlock.pipelines import retexture
    from warlock.service import files

    assert set(remesh.GEOMETRY_DERIVED) == set(files.DERIVED)
    assert set(retexture.SURFACE_DERIVED) <= set(remesh.GEOMETRY_DERIVED)


def test_the_op_is_registered_and_the_spec_names_it(tmp_path):
    assert blender_worker.OPS["remesh"] is blender_worker.op_remesh
    spec = rigging.remesh_spec(
        tmp_path / "model.glb", tmp_path / ".remesh.tmp.glb", tmp_path,
        target_faces=8000, texture_size=1024, close_holes=True, seed=3,
    )
    assert spec["op"] == "remesh"
    assert spec["target_faces"] == 8000 and spec["texture_size"] == 1024
    assert spec["close_holes"] is True and spec["seed"] == 3
    assert Path(spec["result_path"]).parent == tmp_path


def test_the_kind_is_in_every_stage_keyed_table():
    assert "remesh" in DERIVED_PARAMS
    assert progress.phases_for("remesh") is progress.PHASES_REMESH
    assert followups.PRODUCTS["remesh"] == "Remesh"
    assert asset_open.FOLLOWUP_STAGES["remesh"] == "mesh"


def test_the_pure_module_imports_nothing_heavy():
    src = Path(remesh.__file__).read_text(encoding="utf-8")
    for name in ("bpy", "numpy", "trimesh", "service", "queue", "studio"):
        assert f"import {name}" not in src and f"from .{name}" not in src


# --- the door ----------------------------------------------------------------------


def _finished_mesh(svc, **params):
    job_id = svc.store.create("text", "a crate", {"seed": 1, **params})
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"glb")
    (job_dir / "source.glb").write_bytes(b"src")
    svc.store.set_status(job_id, "done")
    return job_id


@pytest.fixture
def blender_present(monkeypatch):
    from warlock import doctor

    class _Ok:
        ok = True

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _Ok())


def test_the_door_queues_a_remesh_row_against_the_source(svc, blender_present):
    job_id = _finished_mesh(svc)
    out = svc_jobs.remesh_job(svc, job_id, profile="low", texture_size=512, close_holes=True)
    row = svc.store.get(out["id"])
    assert row["kind"] == "remesh" and row["status"] == "queued"
    assert row["params"]["source_job"] == job_id
    assert row["params"]["target_faces"] == remesh.FACE_PROFILES["low"]
    assert row["params"]["remesh_profile"] == "low"
    assert row["params"]["texture_size"] == 512
    assert row["params"]["close_holes"] is True
    # The gltfpack tier key is not borrowed: a quad budget must not land in the
    # findings corpus as a triangle tier.
    assert "profile" not in row["params"]
    assert out["stale"] == []


def test_the_door_reports_the_rig_it_will_orphan(svc, blender_present):
    job_id = _finished_mesh(svc)
    (svc.job_dir(job_id) / "rig.glb").write_bytes(b"rig")
    (svc.job_dir(job_id) / "rig.json").write_bytes(b"{}")
    out = svc_jobs.remesh_job(svc, job_id)
    assert "rig.glb" in out["stale"]


def test_the_door_refuses_without_blender(svc, monkeypatch):
    from warlock import doctor

    class _No:
        ok = False

    monkeypatch.setattr(doctor, "blender_check", lambda *a, **k: _No())
    job_id = _finished_mesh(svc)
    with pytest.raises(Invalid) as info:
        svc_jobs.remesh_job(svc, job_id)
    assert "Blender" in str(info.value)


def test_the_door_refuses_a_running_job_and_a_custom_budget_out_of_range(svc, blender_present):
    job_id = _finished_mesh(svc)
    svc.store.set_status(job_id, "running")
    with pytest.raises(Conflict):
        svc_jobs.remesh_job(svc, job_id)
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid) as info:
        svc_jobs.remesh_job(svc, job_id, profile="custom", custom_faces=1)
    assert info.value.field == "custom_faces"
    with pytest.raises(Invalid) as info:
        svc_jobs.remesh_job(svc, job_id, texture_size=777)
    assert info.value.field == "texture_size"


def test_the_door_refuses_a_job_with_no_mesh(svc, blender_present):
    job_id = svc.store.create("text", "nothing", {"seed": 1})
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid):
        svc_jobs.remesh_job(svc, job_id)


# --- the worker -------------------------------------------------------------------


@pytest.fixture
def worker(tmp_path, fake_pipelines):
    config = Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
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


def _mesh_job(worker: Worker) -> str:
    job_id = worker.store.create("text", "a crate", {"seed": 1, "size_m": 1.0})
    job_dir = worker.config.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"old-model")
    (job_dir / "source.glb").write_bytes(b"reconstruction")
    for name in ("model.stl", "model.fbx", "textures.zip"):
        (job_dir / name).write_bytes(b"stale")
    worker.store.set_status(job_id, "done")
    return job_id


def _fake_worker_run(monkeypatch, *, write=True, side_effect=None, hold=None):
    calls = []

    def fake(spec, *, on_progress=None, on_start=None, timeout=0.0):
        calls.append({"spec": spec, "timeout": timeout})
        if on_progress is not None:
            on_progress(0.5, "Baking colour")
        if hold is not None:
            hold.wait(timeout=10)
        if side_effect is not None:
            raise side_effect
        if write:
            Path(spec["out_glb"]).write_bytes(b"new-model")
        return {"ok": True, "method": "quadriflow", "faces": 8000, "faces_before": 300000,
                "quads": 0.97, "texture_size": spec["texture_size"], "metallic": 0.0}

    monkeypatch.setattr(rigging, "run_worker", fake)
    return calls


@pytest.fixture
def _no_normalize(monkeypatch):
    # The fake GLB is bytes, not a glTF: grounding is not the subject here.
    from warlock.pipelines import postprocess

    monkeypatch.setattr(postprocess, "normalize_glb", lambda *a, **k: {"scale": 1.0})


async def test_a_remesh_publishes_over_the_source_mesh_and_drops_its_exports(
    worker, monkeypatch, _no_normalize
):
    calls = _fake_worker_run(monkeypatch)
    source = _mesh_job(worker)
    job_id = worker.store.create(
        "remesh", "a crate", {"source_job": source, "target_faces": 8000, "texture_size": 512}
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()
    row = worker.store.get(job_id)
    assert row["status"] == "done", row.get("error")
    source_dir = worker.config.job_dir(source)
    assert (source_dir / "model.glb").read_bytes() == b"new-model"
    assert (source_dir / "source.glb").read_bytes() == b"reconstruction"
    assert not (source_dir / rigging.REMESH_GLB_TMP).exists()
    for name in ("model.stl", "model.fbx", "textures.zip"):
        assert not (source_dir / name).exists()
    spec = calls[0]["spec"]
    assert spec["op"] == "remesh" and spec["target_faces"] == 8000
    assert spec["texture_size"] == 512
    assert Path(spec["source_glb"]) == source_dir / "model.glb"
    report = row["params"]["remesh"]
    assert report["faces"] == 8000 and report["method"] == "quadriflow"
    assert "tiercheck" in report
    source_row = worker.store.get(source)
    assert source_row["params"]["remesh"]["faces"] == 8000
    assert "optimize" not in source_row["params"]


async def test_a_worker_that_wrote_nothing_fails_the_job_and_keeps_the_old_mesh(
    worker, monkeypatch, _no_normalize
):
    _fake_worker_run(monkeypatch, write=False)
    source = _mesh_job(worker)
    job_id = worker.store.create("remesh", "a crate", {"source_job": source, "target_faces": 8000})
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()
    assert worker.store.get(job_id)["status"] == "error"
    source_dir = worker.config.job_dir(source)
    assert (source_dir / "model.glb").read_bytes() == b"old-model"
    assert (source_dir / "model.stl").exists()


async def test_a_blender_failure_leaves_no_temp_behind(worker, monkeypatch, _no_normalize):
    _fake_worker_run(monkeypatch, side_effect=rigging.BlenderError("boom"))
    source = _mesh_job(worker)
    job_id = worker.store.create("remesh", "a crate", {"source_job": source, "target_faces": 8000})
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] in ("done", "error"))
    await worker.shutdown()
    assert worker.store.get(job_id)["status"] == "error"
    source_dir = worker.config.job_dir(source)
    assert not (source_dir / rigging.REMESH_GLB_TMP).exists()
    assert (source_dir / "model.glb").read_bytes() == b"old-model"


def test_the_cancel_sweep_names_only_the_temp(worker):
    source = _mesh_job(worker)
    source_dir = worker.config.job_dir(source)
    (source_dir / rigging.REMESH_GLB_TMP).write_bytes(b"half")
    job = {"id": "abc", "kind": "remesh", "params": {"source_job": source}}
    worker._discard_artifacts(job)
    assert not (source_dir / rigging.REMESH_GLB_TMP).exists()
    assert (source_dir / "model.glb").read_bytes() == b"old-model"


# --- the panel -----------------------------------------------------------------------


def test_the_panel_offers_every_profile_and_submits_the_doors_arguments():
    keys = [k for k, _ in remesh_panel.PROFILES]
    assert keys == [*remesh.FACE_PROFILES, "custom"]
    form = {
        "job_id": "x", "remesh_profile": "custom", "custom_faces": 4000,
        "texture_size": "2048", "close_holes": True,
    }
    assert remesh_panel.validate(form) == []
    assert remesh_panel.submit_kwargs(form) == {
        "profile": "custom", "custom_faces": 4000, "texture_size": 2048, "close_holes": True,
    }
    form["custom_faces"] = 1
    assert remesh_panel.validate(form)
    form.update(remesh_profile="low", texture_size="")
    assert remesh_panel.submit_kwargs(form)["texture_size"] is None
    assert remesh_panel.submit_kwargs(form)["custom_faces"] is None
