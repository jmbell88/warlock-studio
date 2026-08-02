"""The rig HTTP surface. No Blender: every route here is reachable, and every
error it can return is reachable, without bpy installed."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import warlock.config as config_mod
from warlock.app import create_app
from warlock.db import JobStore


@pytest.fixture
def rig_client(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(assets))
    monkeypatch.setenv("WARLOCK_DB", str(assets / "jobs.sqlite"))
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(config_mod, "_config", None)
    # The worker never claims anything here. These tests are about the HTTP
    # surface, and a live worker would race every one of them: it would pick
    # up the source job, try to load SDXL, and rewrite the very status the
    # test just set. Worker behaviour has its own tests (test_queue.py).
    monkeypatch.setattr(JobStore, "next_queued", lambda self: None)
    with TestClient(create_app()) as c:
        yield c, assets


def _finished_mesh_job(client, assets) -> str:
    """A job the API will agree is rigg-able: status done, model.glb on disk."""
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "a barrel"}).json()["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    client.app.state.store.set_status(job_id, "done")
    return job_id


# --- catalogue --------------------------------------------------------------


def test_templates_route_lists_both_skeletons(rig_client):
    client, _ = rig_client
    body = client.get("/api/rig/templates").json()
    assert {t["key"] for t in body["templates"]} >= {"humanoid", "quadruped"}
    assert body["default"] == "humanoid"
    # available reflects bpy, which CI may or may not have -- but it must be a
    # bool and carry a reason either way, because the UI keys off it.
    assert isinstance(body["available"], bool)
    assert body["detail"]


def test_health_reports_rigging_without_failing_startup(rig_client):
    client, _ = rig_client
    body = client.get("/api/health").json()
    names = {c["name"]: c for c in body["checks"]}
    assert "Blender (rigging)" in names
    assert names["Blender (rigging)"]["fatal"] is False
    # A machine without bpy still has a healthy app.
    assert body["ok"] is True


# --- queueing a rig ---------------------------------------------------------


def test_rig_unknown_job_is_404(rig_client):
    client, _ = rig_client
    assert client.post("/api/jobs/0123456789ab/rig").status_code == 404


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", "0123456789abcd", "%2e%2e"])
def test_rig_routes_reject_malformed_job_ids(rig_client, bad):
    """config.job_dir() is a bare path join, so every id-bearing route needs
    the same guard get_file has."""
    client, _ = rig_client
    assert client.post(f"/api/jobs/{bad}/rig").status_code == 404
    assert client.get(f"/api/jobs/{bad}/rig").status_code == 404


def test_rig_requires_a_finished_mesh(rig_client):
    client, _ = rig_client
    job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
    r = client.post(f"/api/jobs/{job_id}/rig")
    assert r.status_code == 400
    assert "finished mesh" in r.json()["detail"]


def test_rig_rejects_an_unknown_template(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    r = client.post(f"/api/jobs/{job_id}/rig", data={"template": "dragon"})
    assert r.status_code == 400
    assert "dragon" in r.json()["detail"]


def test_rig_queues_a_job_pointing_back_at_its_source(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    r = client.post(f"/api/jobs/{job_id}/rig", data={"template": "quadruped"})
    assert r.status_code == 200
    rig_id = r.json()["id"]
    assert rig_id != job_id
    rig_job = client.get(f"/api/jobs/{rig_id}").json()
    assert rig_job["kind"] == "rig"
    # source_job is what lets the UI attach this to the parent card instead of
    # listing it as an unrelated row.
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["template"] == "quadruped"


def test_a_rig_job_cannot_itself_be_rigged(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    rig_id = client.post(f"/api/jobs/{job_id}/rig").json()["id"]
    r = client.post(f"/api/jobs/{rig_id}/rig")
    assert r.status_code == 400


def test_generate_form_can_opt_into_rigging(rig_client):
    client, _ = rig_client
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "a knight", "rig": "true",
                           "rig_template": "humanoid"}
    )
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["rig"] is True
    assert params["rig_template"] == "humanoid"


def test_generate_form_rejects_an_unknown_template_up_front(rig_client):
    """Better to lose the request than to lose the rig 90 seconds after the
    mesh finishes."""
    client, _ = rig_client
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "x", "rig": "true",
                           "rig_template": "dragon"}
    )
    assert r.status_code == 400


def test_generate_without_the_flag_records_no_rig_intent(rig_client):
    client, _ = rig_client
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "x"})
    assert "rig" not in client.get(f"/api/jobs/{r.json()['id']}").json()["params"]


# --- reading a rig back -----------------------------------------------------


def test_rig_metadata_is_404_until_rigged(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    assert client.get(f"/api/jobs/{job_id}/rig").status_code == 404


def test_rig_metadata_is_served_once_written(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    meta = {"version": 1, "template": "humanoid", "weighting": "envelope", "bones": []}
    (assets / job_id / "rig.json").write_text(json.dumps(meta))
    assert client.get(f"/api/jobs/{job_id}/rig").json() == meta


def test_rig_glb_is_hidden_until_rig_json_lands(rig_client):
    """rig.json is written last, so it -- not rig.glb's own existence -- is
    what says the rig finished. Otherwise a GET can catch a half-exported GLB."""
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    (assets / job_id / "rig.glb").write_bytes(b"half-written")
    assert "rig.glb" not in client.get(f"/api/jobs/{job_id}").json()["files"]
    (assets / job_id / "rig.json").write_text('{"bones": []}')
    assert "rig.glb" in client.get(f"/api/jobs/{job_id}").json()["files"]


def test_downloading_rig_glb_is_gated_on_rig_json_too(rig_client):
    """The same gate as the files listing, on the route that actually serves
    bytes: the Blender export is not atomic, so a direct GET aimed at a
    known URL must not be able to walk in ahead of rig.json and collect a
    truncated GLB."""
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    (assets / job_id / "rig.glb").write_bytes(b"half-written")
    assert client.get(f"/api/jobs/{job_id}/files/rig.glb").status_code == 404


def test_rig_glb_downloads(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    (assets / job_id / "rig.glb").write_bytes(b"glb-bytes")
    (assets / job_id / "rig.json").write_text('{"bones": []}')
    r = client.get(f"/api/jobs/{job_id}/files/rig.glb")
    assert r.status_code == 200
    assert r.content == b"glb-bytes"
    assert r.headers["content-type"] == "model/gltf-binary"


def test_rig_glb_404s_when_absent(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    assert client.get(f"/api/jobs/{job_id}/files/rig.glb").status_code == 404


# --- corrected joints ---------------------------------------------------------


def _rigged_job(client, assets) -> tuple[str, list[dict]]:
    """A job with a rig on disk, and the fitted bones the editor would show."""
    from warlock import rigging

    job_id = _finished_mesh_job(client, assets)
    job_dir = assets / job_id
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid", "bones": fitted}), encoding="utf-8"
    )
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    return job_id, fitted


def test_adjust_joints_queues_a_rerig(rig_client):
    client, assets = rig_client
    job_id, fitted = _rigged_job(client, assets)

    r = client.post(
        f"/api/jobs/{job_id}/rig/joints",
        json={
            "bones": [
                {"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted
            ]
        },
    )
    assert r.status_code == 200
    rig_job = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert rig_job["kind"] == "rig"
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["bones"]
    assert rig_job["params"]["adjusted"] is True


def test_adjust_joints_on_an_unrigged_job_is_a_400(rig_client):
    client, assets = rig_client
    job_id = _finished_mesh_job(client, assets)
    assert client.post(f"/api/jobs/{job_id}/rig/joints", json={"bones": []}).status_code == 400


def test_adjust_joints_on_a_missing_job_is_a_404(rig_client):
    client, _ = rig_client
    r = client.post("/api/jobs/0123456789ab/rig/joints", json={"bones": []})
    assert r.status_code == 404


def test_adjust_joints_rejects_a_partial_skeleton(rig_client):
    client, assets = rig_client
    job_id, fitted = _rigged_job(client, assets)
    r = client.post(
        f"/api/jobs/{job_id}/rig/joints",
        json={"bones": [{"name": fitted[0]["name"], "head": [0, 0, 0], "tail": [0, 0, 1]}]},
    )
    assert r.status_code == 400
    assert "missing bone" in r.json()["detail"]
