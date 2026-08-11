"""Pose storage and the service calls over it.

No Blender: everything except baking a posed GLB is decidable without bpy, and
the one path that needs it is exercised with the worker stubbed out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import warlock.config as config_mod
from warlock import rigging
from warlock.service import Conflict, Failed, Invalid, NotFound
from warlock.service import jobs as svc_jobs
from warlock.service import rig as svc_rig

BONES = ["hips", "spine", "head"]
IDENTITY = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def _rigged_job(svc, assets) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    (job_dir / "rig.json").write_text(
        json.dumps({"version": 1, "bones": [{"name": n} for n in BONES]})
    )
    svc.store.set_status(job_id, "done")
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


# --- the service surface ----------------------------------------------------


def test_poses_require_a_rig(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    with pytest.raises(NotFound):
        svc_rig.list_poses(svc, job_id)
    with pytest.raises(NotFound):
        svc_rig.save_pose(svc, job_id, _pose())


def test_listing_reports_the_rigs_bones(svc, assets):
    """The editor needs the bone list to know what it may send back."""
    job_id = _rigged_job(svc, assets)
    body = svc_rig.list_poses(svc, job_id)
    assert body["bones"] == BONES
    assert body["poses"] == []


def test_save_then_list_then_delete(svc, assets):
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose("crouch"))
    assert record["name"] == "crouch"
    assert svc_rig.list_poses(svc, job_id)["poses"] == [record]
    assert svc_rig.delete_pose(svc, job_id, record["id"]) == {"ok": True}
    assert svc_rig.list_poses(svc, job_id)["poses"] == []
    with pytest.raises(NotFound):
        svc_rig.delete_pose(svc, job_id, record["id"])


def test_saving_with_an_id_overwrites_in_place(svc, assets):
    job_id = _rigged_job(svc, assets)
    first = svc_rig.save_pose(svc, job_id, _pose("idle"))
    body = _pose("idle", head=[0.0, 0.0, 0.7071068, 0.7071068])
    body["id"] = first["id"]
    second = svc_rig.save_pose(svc, job_id, body)
    assert second["id"] == first["id"]
    assert svc_rig.list_poses(svc, job_id)["poses"] == [second]


def test_saving_against_an_unknown_id_is_not_found(svc, assets):
    job_id = _rigged_job(svc, assets)
    body = _pose()
    body["id"] = "0123456789ab"
    with pytest.raises(NotFound):
        svc_rig.save_pose(svc, job_id, body)


def test_a_pose_naming_a_bone_the_rig_lacks_is_rejected(svc, assets):
    job_id = _rigged_job(svc, assets)
    with pytest.raises(Invalid, match="tail_01"):
        svc_rig.save_pose(svc, job_id, {"name": "idle", "bones": {"tail_01": IDENTITY}})


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
def test_malformed_pose_payloads_are_refused(svc, assets, body):
    job_id = _rigged_job(svc, assets)
    with pytest.raises(Invalid):
        svc_rig.save_pose(svc, job_id, body)


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", "0123456789abcd"])
def test_the_pose_entry_points_reject_malformed_pose_ids(svc, assets, bad):
    job_id = _rigged_job(svc, assets)
    with pytest.raises(NotFound):
        svc_rig.delete_pose(svc, job_id, bad)
    with pytest.raises(NotFound):
        svc_rig.posed_model(svc, job_id, bad)


def test_the_pose_cap_is_enforced(svc, assets, monkeypatch):
    job_id = _rigged_job(svc, assets)
    monkeypatch.setattr(rigging, "MAX_POSES", 1)
    svc_rig.save_pose(svc, job_id, _pose("a"))
    with pytest.raises(Conflict):
        svc_rig.save_pose(svc, job_id, _pose("b"))


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


def test_a_posed_glb_is_baked_once_and_then_cached(svc, assets, monkeypatch):
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose())
    calls = _fake_bake(monkeypatch)

    first = svc_rig.posed_model(svc, job_id, record["id"])
    assert first.read_bytes() == b"posed-glb"
    assert svc_rig.posed_model(svc, job_id, record["id"]).read_bytes() == b"posed-glb"
    # The second call served the cached file rather than running Blender again.
    assert len(calls) == 1
    spec, kwargs = calls[0]
    assert spec["op"] == "pose"
    assert spec["rig_glb"].endswith("rig.glb")
    assert spec["bones"].keys() == set(BONES)
    assert kwargs["timeout"] == config_mod.get_config().pose_timeout


def test_a_posed_glb_for_an_unknown_pose_is_not_found(svc, assets):
    job_id = _rigged_job(svc, assets)
    with pytest.raises(NotFound):
        svc_rig.posed_model(svc, job_id, "0123456789ab")


def test_a_failed_bake_raises_rather_than_hanging(svc, assets, monkeypatch):
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose())
    _fake_bake(monkeypatch, side_effect=rigging.BlenderError("boom"))
    with pytest.raises(Failed):
        svc_rig.posed_model(svc, job_id, record["id"])
    # Nothing cached, so a retry after fixing the cause still works.
    assert not rigging.pose_glb_path(assets / job_id, record["id"]).exists()


def test_a_bake_that_dies_part_way_leaves_no_partial_glb(svc, assets, monkeypatch):
    """The sharper half of the failure above: Blender writing *something* and
    then dying -- a pose_timeout, a kill-on-close at shutdown.

    Existence is this artifact's whole freshness test, so an unstaged bake
    would leave a truncated GLB that ``posed_model`` then serves under this
    pose id forever, with no way for a retry to get past it.
    """
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose())

    def run_worker(spec, **kwargs):
        Path(spec["out_glb"]).write_bytes(b"half a gl")
        raise rigging.BlenderError("killed")

    monkeypatch.setattr(rigging, "run_worker", run_worker)
    with pytest.raises(Failed):
        svc_rig.posed_model(svc, job_id, record["id"])

    assert not rigging.pose_glb_path(assets / job_id, record["id"]).exists()
    # And the staging file went with it: nothing sweeps a pose directory, so a
    # stranded dotfile would live as long as the job.
    pose_dir = rigging.pose_glb_path(assets / job_id, record["id"]).parent
    assert [p.name for p in pose_dir.iterdir() if p.name.startswith(".")] == []


def test_re_saving_a_pose_invalidates_its_bake(svc, assets, monkeypatch):
    """The cached GLB depicts the old rotations; serving it after an edit would
    show the user a pose they replaced."""
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose())
    calls = _fake_bake(monkeypatch)
    svc_rig.posed_model(svc, job_id, record["id"])
    body = _pose("idle", hips=[0.0, 0.0, 0.7071068, 0.7071068])
    body["id"] = record["id"]
    svc_rig.save_pose(svc, job_id, body)
    svc_rig.posed_model(svc, job_id, record["id"])
    assert len(calls) == 2


def test_a_delete_waits_for_an_in_flight_bake_and_leaves_no_orphan_glb(
    svc, assets, monkeypatch
):
    """delete_pose took no lock, so it could land mid-bake: the .json went and
    the bake then wrote a .glb nothing could reach or clean up."""
    import threading

    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose())
    in_bake = threading.Event()
    release = threading.Event()

    def run_worker(spec, **kwargs):
        in_bake.set()
        assert release.wait(5)
        Path(spec["out_glb"]).write_bytes(b"posed-glb")
        return {"ok": True, "bones": len(spec["bones"]), "unknown": []}

    monkeypatch.setattr(rigging, "run_worker", run_worker)

    baker = threading.Thread(
        target=lambda: svc_rig.posed_model(svc, job_id, record["id"])
    )
    baker.start()
    assert in_bake.wait(5)

    deleted: list = []

    def delete():
        deleted.append(svc_rig.delete_pose(svc, job_id, record["id"]))

    deleter = threading.Thread(target=delete)
    deleter.start()
    deleter.join(timeout=0.3)
    assert deleter.is_alive(), "the delete must wait for the bake's lock"

    release.set()
    baker.join(timeout=5)
    deleter.join(timeout=5)
    assert deleted == [{"ok": True}]
    assert not rigging.pose_glb_path(assets / job_id, record["id"]).exists()
    assert not rigging.pose_path(assets / job_id, record["id"]).exists()


def test_a_bake_of_a_pose_deleted_first_is_not_found(svc, assets, monkeypatch):
    """The pose is read under the bake lock, so the check and the bake are
    atomic against a delete."""
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose())
    calls = _fake_bake(monkeypatch)
    svc_rig.delete_pose(svc, job_id, record["id"])
    with pytest.raises(NotFound):
        svc_rig.posed_model(svc, job_id, record["id"])
    assert calls == []


def test_a_failed_pose_write_leaves_the_previous_pose_intact(svc, assets, monkeypatch):
    """A pose file is the only record of its rotations; a torn overwrite would
    lose them."""
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose("idle"))
    path = rigging.pose_path(assets / job_id, record["id"])
    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        rigging.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    body = _pose("wave", hips=[0.0, 0.0, 0.7071068, 0.7071068])
    body["id"] = record["id"]
    with pytest.raises(OSError):
        svc_rig.save_pose(svc, job_id, body)

    assert path.read_text(encoding="utf-8") == before
    # And no staging file left lying beside it.
    leftovers = [p for p in path.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_a_snapshot_with_a_root_offset_bakes_with_the_root_kwargs(
    svc, assets, monkeypatch
):
    """posed_model scales a library snapshot's root translation onto this
    rig's own height and hands it to the bake -- and only then."""
    job_id = _rigged_job(svc, assets)
    job_dir = assets / job_id
    (job_dir / "rig.json").write_text(
        json.dumps(
            {
                "version": 1,
                "template": "humanoid",
                "root": "hips",
                "bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 2.0]},
                "bones": [{"name": n} for n in BONES],
            }
        )
    )
    pose = rigging.validate_pose(_pose("leap"), BONES)
    record = rigging.save_pose(
        job_dir, pose, extra={"root_translation": [0.1, 0.0, -0.25]}
    )
    calls = _fake_bake(monkeypatch)
    svc_rig.posed_model(svc, job_id, record["id"])
    spec, _ = calls[0]
    assert spec["root_bone"] == "hips"
    # Character-height units times a 2-unit-tall rig.
    assert spec["root_offset"] == pytest.approx([0.2, 0.0, -0.5])


def test_a_pose_without_a_root_offset_bakes_the_spec_it_always_did(
    svc, assets, monkeypatch
):
    job_id = _rigged_job(svc, assets)
    record = svc_rig.save_pose(svc, job_id, _pose())
    calls = _fake_bake(monkeypatch)
    svc_rig.posed_model(svc, job_id, record["id"])
    spec, _ = calls[0]
    assert "root_bone" not in spec
    assert "root_offset" not in spec


def test_a_root_offset_the_rig_cannot_scale_costs_the_offset_not_the_bake(
    svc, assets, monkeypatch
):
    """_rigged_job's rig.json has no bounds and no root -- the pre-library
    shape -- so the offset is dropped with a log line and the bake proceeds."""
    job_id = _rigged_job(svc, assets)
    pose = rigging.validate_pose(_pose("leap"), BONES)
    record = rigging.save_pose(
        assets / job_id, pose, extra={"root_translation": [0.1, 0.0, 0.0]}
    )
    calls = _fake_bake(monkeypatch)
    path = svc_rig.posed_model(svc, job_id, record["id"])
    assert path.exists()
    spec, _ = calls[0]
    assert "root_offset" not in spec


# --- shipped preset library -------------------------------------------------


def test_the_preset_library_is_readable_per_template():
    names = [p["name"] for p in svc_rig.template_presets("humanoid")["poses"]]
    assert "idle" in names


def test_an_unknown_templates_presets_are_refused():
    with pytest.raises(Invalid):
        svc_rig.template_presets("nope")
