"""The rig surface. No Blender: every entry point here is reachable, and every
error it can return is reachable, without bpy installed."""

from __future__ import annotations

import json

import pytest

from warlock.service import Invalid, NotFound, NotReady
from warlock.service import derive as svc_derive
from warlock.service import jobs as svc_jobs
from warlock.service import rig as svc_rig
from warlock.service import system as svc_system


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def _finished_mesh_job(svc, assets) -> str:
    """A job the service will agree is rigg-able: status done, model.glb on disk."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a barrel")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    svc.store.set_status(job_id, "done")
    return job_id


# --- catalogue --------------------------------------------------------------


def test_the_template_catalogue_lists_both_skeletons(svc):
    body = svc_rig.rig_templates(svc)
    assert {t["key"] for t in body["templates"]} >= {"humanoid", "quadruped"}
    assert body["default"] == "humanoid"
    # available reflects bpy, which a given machine may or may not have -- but
    # it must be a bool and carry a reason either way, because the UI keys off it.
    assert isinstance(body["available"], bool)
    assert body["detail"]


def test_health_reports_rigging_without_failing_startup(svc):
    body = svc_system.health(svc)
    names = {c["name"]: c for c in body["checks"]}
    assert "Blender (rigging)" in names
    assert names["Blender (rigging)"]["fatal"] is False


# --- queueing a rig ---------------------------------------------------------


def test_rigging_an_unknown_job_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_rig.create_rig(svc, "0123456789ab")


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", "0123456789abcd", "../.."])
def test_the_rig_entry_points_reject_malformed_job_ids(svc, bad):
    """config.job_dir() is a bare path join, so every id-bearing entry point
    needs the same guard get_file has."""
    with pytest.raises(NotFound):
        svc_rig.create_rig(svc, bad)
    with pytest.raises(NotFound):
        svc_rig.get_rig(svc, bad)


def test_rigging_requires_a_finished_mesh(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(Invalid, match="finished mesh"):
        svc_rig.create_rig(svc, job_id)


def test_rigging_rejects_an_unknown_template(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(Invalid, match="dragon"):
        svc_rig.create_rig(svc, job_id, template="dragon")


def test_a_rig_job_points_back_at_its_source(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    out = svc_rig.create_rig(svc, job_id, template="quadruped")
    rig_job = svc.store.get(out["id"])
    assert out["id"] != job_id
    assert rig_job["kind"] == "rig"
    # source_job is what lets the UI attach this to the parent card instead of
    # listing it as an unrelated row.
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["template"] == "quadruped"


def test_a_rig_job_cannot_itself_be_rigged(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    rig_id = svc_rig.create_rig(svc, job_id)["id"]
    with pytest.raises(Invalid):
        svc_rig.create_rig(svc, rig_id)


def test_a_submit_can_opt_into_rigging(svc):
    out = svc_jobs.create_job(
        svc, kind="text", prompt="a knight", rig=True, rig_template="humanoid"
    )
    params = svc.store.get(out["id"])["params"]
    assert params["rig"] is True
    assert params["rig_template"] == "humanoid"


def test_a_submit_rejects_an_unknown_template_up_front(svc):
    """Better to lose the request than to lose the rig 90 seconds after the
    mesh finishes."""
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="text", prompt="x", rig=True, rig_template="dragon")


def test_a_submit_without_the_flag_records_no_rig_intent(svc):
    out = svc_jobs.create_job(svc, kind="text", prompt="x")
    assert "rig" not in svc.store.get(out["id"])["params"]


# --- reading a rig back -----------------------------------------------------


def test_rig_metadata_is_absent_until_rigged(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(NotFound):
        svc_rig.get_rig(svc, job_id)


def test_rig_metadata_is_served_once_written(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    meta = {"version": 1, "template": "humanoid", "weighting": "envelope", "bones": []}
    (assets / job_id / "rig.json").write_text(json.dumps(meta))
    assert svc_rig.get_rig(svc, job_id) == meta


def test_rig_glb_is_hidden_until_rig_json_lands(svc, assets):
    """rig.json is written last, so it -- not rig.glb's own existence -- is
    what says the rig finished. Otherwise a read can catch a half-exported GLB."""
    job_id = _finished_mesh_job(svc, assets)
    (assets / job_id / "rig.glb").write_bytes(b"half-written")
    assert "rig.glb" not in svc_jobs.get_job(svc, job_id)["files"]
    (assets / job_id / "rig.json").write_text('{"bones": []}')
    assert "rig.glb" in svc_jobs.get_job(svc, job_id)["files"]


def test_reading_rig_glb_is_gated_on_rig_json_too(svc, assets):
    """The same gate as the files listing, on the call that actually hands back
    a path: the Blender export is not atomic, so a direct read must not be able
    to walk in ahead of rig.json and collect a truncated GLB."""
    job_id = _finished_mesh_job(svc, assets)
    (assets / job_id / "rig.glb").write_bytes(b"half-written")
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "rig.glb")


def test_rig_glb_is_readable_once_complete(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    (assets / job_id / "rig.glb").write_bytes(b"glb-bytes")
    (assets / job_id / "rig.json").write_text('{"bones": []}')
    path = svc_derive.get_file(svc, job_id, "rig.glb")
    assert path.read_bytes() == b"glb-bytes"


def test_rig_glb_is_not_ready_when_absent(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "rig.glb")


# --- corrected joints -------------------------------------------------------


def _rigged_job(svc, assets) -> tuple[str, list[dict]]:
    """A job with a rig on disk, and the fitted bones the editor would show."""
    from warlock import rigging

    job_id = _finished_mesh_job(svc, assets)
    job_dir = assets / job_id
    template = rigging.get_template("humanoid")
    fitted = rigging.fit_template(template, [-1, -1, 0], [1, 1, 2])
    (job_dir / "rig.json").write_text(
        json.dumps({"template": "humanoid", "bones": fitted}), encoding="utf-8"
    )
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    return job_id, fitted


def test_adjusting_joints_queues_a_rerig(svc, assets):
    job_id, fitted = _rigged_job(svc, assets)
    out = svc_rig.adjust_joints(
        svc,
        job_id,
        {"bones": [{"name": b["name"], "head": b["head"], "tail": b["tail"]} for b in fitted]},
    )
    rig_job = svc.store.get(out["id"])
    assert rig_job["kind"] == "rig"
    assert rig_job["params"]["source_job"] == job_id
    assert rig_job["params"]["bones"]
    assert rig_job["params"]["adjusted"] is True


def test_adjusting_joints_on_an_unrigged_job_is_refused(svc, assets):
    job_id = _finished_mesh_job(svc, assets)
    with pytest.raises(Invalid):
        svc_rig.adjust_joints(svc, job_id, {"bones": []})


def test_adjusting_joints_on_a_missing_job_is_not_found(svc):
    with pytest.raises(NotFound):
        svc_rig.adjust_joints(svc, "0123456789ab", {"bones": []})


def test_adjusting_joints_rejects_a_partial_skeleton(svc, assets):
    job_id, fitted = _rigged_job(svc, assets)
    with pytest.raises(Invalid, match="missing bone"):
        svc_rig.adjust_joints(
            svc,
            job_id,
            {"bones": [{"name": fitted[0]["name"], "head": [0, 0, 0], "tail": [0, 0, 1]}]},
        )
