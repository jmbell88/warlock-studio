"""Pose storage and its HTTP surface.

No Blender: everything except baking a posed GLB is decidable without bpy, and
the one route that needs it is exercised with the worker stubbed out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import warlock.config as config_mod
from warlock import rigging
from warlock.app import create_app
from warlock.db import JobStore

BONES = ["hips", "spine", "head"]
IDENTITY = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def pose_client(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(assets))
    monkeypatch.setenv("WARLOCK_DB", str(assets / "jobs.sqlite"))
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(config_mod, "_config", None)
    # Same reason as test_rig_api: a live worker would claim the source job and
    # race every assertion here.
    monkeypatch.setattr(JobStore, "next_queued", lambda self: None)
    with TestClient(create_app()) as c:
        yield c, assets


def _rigged_job(client, assets) -> str:
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "a knight"}).json()["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "bones": [{"name": n} for n in BONES]})
    )
    client.app.state.store.set_status(job_id, "done")
    return job_id


def _pose(name="idle", **bones):
    return {"name": name, "bones": {b: bones.get(b, IDENTITY) for b in BONES}}


# --- storage ----------------------------------------------------------------


def test_a_saved_pose_round_trips(tmp_path):
    record = rigging.save_pose(tmp_path, {"name": "idle", "bones": {"hips": IDENTITY}})
    assert rigging.is_valid_id(record["id"])
    assert rigging.read_pose(tmp_path, record["id"]) == record
    assert rigging.list_poses(tmp_path) == [record]


def test_poses_list_oldest_first(tmp_path):
    ids = [
        rigging.save_pose(tmp_path, {"name": f"p{i}", "bones": {"hips": IDENTITY}})["id"]
        for i in range(3)
    ]
    assert [p["id"] for p in rigging.list_poses(tmp_path)] == ids


def test_saving_over_an_id_replaces_it_and_drops_the_baked_glb(tmp_path):
    record = rigging.save_pose(tmp_path, {"name": "idle", "bones": {"hips": IDENTITY}})
    glb = rigging.pose_glb_path(tmp_path, record["id"])
    glb.write_bytes(b"stale")
    rigging.save_pose(tmp_path, {"name": "crouch", "bones": {"hips": IDENTITY}}, record["id"])
    assert len(rigging.list_poses(tmp_path)) == 1
    assert rigging.read_pose(tmp_path, record["id"])["name"] == "crouch"
    # The cached bake depicted the old pose; leaving it would serve a lie.
    assert not glb.exists()


def test_a_corrupt_pose_file_costs_only_itself(tmp_path):
    good = rigging.save_pose(tmp_path, {"name": "idle", "bones": {"hips": IDENTITY}})
    (rigging.pose_dir(tmp_path) / "0123456789ab.json").write_text("{not json")
    assert [p["id"] for p in rigging.list_poses(tmp_path)] == [good["id"]]


def test_listing_a_job_with_no_poses_is_empty_not_an_error(tmp_path):
    assert rigging.list_poses(tmp_path) == []


@pytest.mark.parametrize("bad", ["..", "../x", "not-an-id", "ABCDEF012345", ""])
def test_pose_paths_reject_ids_that_are_not_ours(tmp_path, bad):
    """This is the only place a caller-supplied pose id becomes a path."""
    with pytest.raises(ValueError):
        rigging.pose_path(tmp_path, bad)


def test_delete_pose_removes_both_files(tmp_path):
    record = rigging.save_pose(tmp_path, {"name": "idle", "bones": {"hips": IDENTITY}})
    rigging.pose_glb_path(tmp_path, record["id"]).write_bytes(b"baked")
    assert rigging.delete_pose(tmp_path, record["id"]) is True
    assert rigging.list_poses(tmp_path) == []
    assert not rigging.pose_glb_path(tmp_path, record["id"]).exists()
    assert rigging.delete_pose(tmp_path, record["id"]) is False


# --- HTTP -------------------------------------------------------------------


def test_poses_require_a_rig(pose_client):
    client, assets = pose_client
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    assert client.get(f"/api/jobs/{job_id}/poses").status_code == 404
    assert client.post(f"/api/jobs/{job_id}/poses", json=_pose()).status_code == 404


def test_listing_reports_the_rigs_bones(pose_client):
    """The editor needs the bone list to know what it may send back."""
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    body = client.get(f"/api/jobs/{job_id}/poses").json()
    assert body["bones"] == BONES
    assert body["poses"] == []


def test_save_then_list_then_delete(pose_client):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    record = client.post(f"/api/jobs/{job_id}/poses", json=_pose("crouch")).json()
    assert record["name"] == "crouch"
    assert client.get(f"/api/jobs/{job_id}/poses").json()["poses"] == [record]
    assert client.delete(f"/api/jobs/{job_id}/poses/{record['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/poses").json()["poses"] == []
    assert client.delete(f"/api/jobs/{job_id}/poses/{record['id']}").status_code == 404


def test_saving_with_an_id_overwrites_in_place(pose_client):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    first = client.post(f"/api/jobs/{job_id}/poses", json=_pose("idle")).json()
    body = _pose("idle", head=[0.0, 0.0, 0.7071068, 0.7071068])
    body["id"] = first["id"]
    second = client.post(f"/api/jobs/{job_id}/poses", json=body).json()
    assert second["id"] == first["id"]
    assert client.get(f"/api/jobs/{job_id}/poses").json()["poses"] == [second]


def test_saving_against_an_unknown_id_is_404(pose_client):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    body = _pose()
    body["id"] = "0123456789ab"
    assert client.post(f"/api/jobs/{job_id}/poses", json=body).status_code == 404


def test_a_pose_naming_a_bone_the_rig_lacks_is_rejected(pose_client):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    body = {"name": "idle", "bones": {"tail_01": IDENTITY}}
    r = client.post(f"/api/jobs/{job_id}/poses", json=body)
    assert r.status_code == 400
    assert "tail_01" in r.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"name": "", "bones": {"hips": IDENTITY}},
        {"name": "x" * 65, "bones": {"hips": IDENTITY}},
        {"name": "idle", "bones": {}},
        {"name": "idle", "bones": {"hips": [0, 0, 0]}},
        {"name": "idle", "bones": {"hips": [0, 0, 0, 0]}},
        {"name": "idle", "bones": {"hips": ["a", 0, 0, 1]}},
    ],
)
def test_malformed_pose_payloads_are_400(pose_client, body):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    assert client.post(f"/api/jobs/{job_id}/poses", json=body).status_code == 400


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", "0123456789abcd"])
def test_pose_routes_reject_malformed_pose_ids(pose_client, bad):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    assert client.delete(f"/api/jobs/{job_id}/poses/{bad}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/poses/{bad}/model.glb").status_code == 404


def test_pose_cap_is_enforced(pose_client, monkeypatch):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    monkeypatch.setattr(rigging, "MAX_POSES", 1)
    assert client.post(f"/api/jobs/{job_id}/poses", json=_pose("a")).status_code == 200
    r = client.post(f"/api/jobs/{job_id}/poses", json=_pose("b"))
    assert r.status_code == 409


# --- the derived posed GLB --------------------------------------------------


def _fake_bake(monkeypatch, *, side_effect=None):
    """Stand in for the Blender subprocess; record what it was asked to do."""
    calls = []

    def run_worker(spec, **kwargs):
        calls.append((spec, kwargs))
        if side_effect is not None:
            raise side_effect
        Path(spec["out_glb"]).write_bytes(b"posed-glb")
        return {"ok": True, "bones": len(spec["bones"]), "unknown": []}

    monkeypatch.setattr(rigging, "run_worker", run_worker)
    return calls


def test_posed_glb_is_baked_once_and_then_cached(pose_client, monkeypatch):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    record = client.post(f"/api/jobs/{job_id}/poses", json=_pose()).json()
    calls = _fake_bake(monkeypatch)

    url = f"/api/jobs/{job_id}/poses/{record['id']}/model.glb"
    first = client.get(url)
    assert first.status_code == 200
    assert first.content == b"posed-glb"
    assert first.headers["content-type"] == "model/gltf-binary"
    assert client.get(url).content == b"posed-glb"
    # Second request served the cached file rather than running Blender again.
    assert len(calls) == 1
    spec, kwargs = calls[0]
    assert spec["op"] == "pose"
    assert spec["rig_glb"].endswith("rig.glb")
    assert spec["bones"].keys() == set(BONES)
    assert kwargs["timeout"] == config_mod.get_config().pose_timeout


def test_posed_glb_404s_for_an_unknown_pose(pose_client):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    assert client.get(f"/api/jobs/{job_id}/poses/0123456789ab/model.glb").status_code == 404


def test_a_failed_bake_is_a_500_not_a_hang(pose_client, monkeypatch):
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    record = client.post(f"/api/jobs/{job_id}/poses", json=_pose()).json()
    _fake_bake(monkeypatch, side_effect=rigging.BlenderError("boom"))
    r = client.get(f"/api/jobs/{job_id}/poses/{record['id']}/model.glb")
    assert r.status_code == 500
    # Nothing cached, so a retry after fixing the cause still works.
    assert not rigging.pose_glb_path(assets / job_id, record["id"]).exists()


def test_re_saving_a_pose_invalidates_its_bake(pose_client, monkeypatch):
    """The cached GLB depicts the old rotations; serving it after an edit would
    show the user a pose they replaced."""
    client, assets = pose_client
    job_id = _rigged_job(client, assets)
    record = client.post(f"/api/jobs/{job_id}/poses", json=_pose()).json()
    calls = _fake_bake(monkeypatch)
    url = f"/api/jobs/{job_id}/poses/{record['id']}/model.glb"
    client.get(url)
    body = _pose("idle", hips=[0.0, 0.0, 0.7071068, 0.7071068])
    body["id"] = record["id"]
    client.post(f"/api/jobs/{job_id}/poses", json=body)
    client.get(url)
    assert len(calls) == 2
