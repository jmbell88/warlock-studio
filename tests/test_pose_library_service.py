"""The pose-library service surface: CRUD, applying to an asset, the preview.

No Blender anywhere: the preview path is exercised with the worker stubbed and
the doctor probe faked, the same way test_poses_api.py treats the pose bake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warlock import doctor, poselib, rigging
from warlock.doctor import Check
from warlock.service import Conflict, Failed, Invalid, NotFound
from warlock.service import jobs as svc_jobs
from warlock.service import poses as svc_poses

IDENTITY = [0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def assets(svc):
    return svc.config.data_dir


def _payload(name="Crouch", template="humanoid", root=None, **over):
    bones = {b["name"]: IDENTITY for b in rigging.get_template(template).bones}
    body = {"name": name, "template": template, "bones": bones}
    if root is not None:
        body["root_translation"] = root
    body.update(over)
    return body


def _rigged_job(svc, assets, template="humanoid") -> str:
    """A done job wearing a real-shaped rig.json: template, bounds, root."""
    job_id = svc_jobs.create_job(svc, kind="text", prompt="a knight")["id"]
    job_dir = assets / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    (job_dir / "rig.glb").write_bytes(b"fake-rig")
    t = rigging.get_template(template)
    (job_dir / "rig.json").write_text(
        json.dumps(
            {
                "version": 1,
                "template": t.key,
                "root": t.root,
                "bounds": {"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 2.0]},
                "bones": [{"name": b["name"]} for b in t.bones],
            }
        )
    )
    svc.store.set_status(job_id, "done")
    return job_id


# --- CRUD -------------------------------------------------------------------


def test_create_read_list_delete(svc):
    stored = svc_poses.create_library_pose(svc, _payload())
    assert rigging.is_valid_id(stored["id"])
    assert svc_poses.read_library_pose(svc, stored["id"]) == stored
    assert svc_poses.list_library(svc)["poses"] == [stored]
    assert svc_poses.list_library(svc, "fish")["poses"] == []
    assert svc_poses.delete_library_pose(svc, stored["id"]) == {"ok": True}
    with pytest.raises(NotFound):
        svc_poses.delete_library_pose(svc, stored["id"])
    assert svc_poses.list_library(svc)["poses"] == []


def test_a_duplicate_name_is_a_conflict_naming_the_control(svc):
    svc_poses.create_library_pose(svc, _payload("Crouch"))
    with pytest.raises(Conflict, match="(?i)crouch") as info:
        svc_poses.create_library_pose(svc, _payload("  crouch "))
    assert info.value.field == "name"
    # The same name on a different skeleton is a different pose.
    svc_poses.create_library_pose(svc, _payload("Crouch", template="fish"))


def test_the_cap_is_enforced(svc, monkeypatch):
    monkeypatch.setattr(poselib, "MAX_LIBRARY_POSES", 1)
    svc_poses.create_library_pose(svc, _payload("A"))
    with pytest.raises(Conflict, match="at most"):
        svc_poses.create_library_pose(svc, _payload("B"))


def test_update_overwrites_in_place(svc):
    stored = svc_poses.create_library_pose(svc, _payload("Crouch"))
    updated = svc_poses.update_library_pose(
        svc, stored["id"], _payload("Crouch low", root=[0.1, 0.0, 0.0])
    )
    assert updated["id"] == stored["id"]
    assert updated["created"] == stored["created"]
    assert updated["root_translation"] == [0.1, 0.0, 0.0]
    assert [p["name"] for p in svc_poses.list_library(svc)["poses"]] == ["Crouch low"]


def test_rename_to_self_is_allowed_and_to_a_taken_name_is_not(svc):
    a = svc_poses.create_library_pose(svc, _payload("A"))
    svc_poses.create_library_pose(svc, _payload("B"))
    # Uniqueness excludes self: re-saving under your own name is not a
    # collision with yourself.
    assert svc_poses.rename_library_pose(svc, a["id"], "A")["name"] == "A"
    assert svc_poses.rename_library_pose(svc, a["id"], "a")["name"] == "a"
    with pytest.raises(Conflict):
        svc_poses.rename_library_pose(svc, a["id"], "B")
    with pytest.raises(Invalid) as info:
        svc_poses.rename_library_pose(svc, a["id"], "")
    assert info.value.field == "name"


def test_the_template_is_immutable(svc):
    stored = svc_poses.create_library_pose(svc, _payload())
    with pytest.raises(Invalid, match="cannot be changed") as info:
        svc_poses.update_library_pose(svc, stored["id"], _payload(template="fish"))
    assert info.value.field == "template"


def test_duplicate_defaults_to_the_copy_name_idiom(svc):
    stored = svc_poses.create_library_pose(svc, _payload("Crouch"))
    copy = svc_poses.duplicate_library_pose(svc, stored["id"])
    assert copy["name"] == "Crouch.001"
    assert copy["id"] != stored["id"]
    assert copy["bones"] == stored["bones"]
    again = svc_poses.duplicate_library_pose(svc, stored["id"])
    assert again["name"] == "Crouch.002"
    with pytest.raises(Conflict):
        svc_poses.duplicate_library_pose(svc, stored["id"], "Crouch.001")


def test_duplicate_respects_the_cap(svc, monkeypatch):
    stored = svc_poses.create_library_pose(svc, _payload("A"))
    monkeypatch.setattr(poselib, "MAX_LIBRARY_POSES", 1)
    with pytest.raises(Conflict, match="at most"):
        svc_poses.duplicate_library_pose(svc, stored["id"])


def test_a_malformed_create_carries_its_field(svc):
    body = _payload()
    body["template"] = "dragon"
    with pytest.raises(Invalid) as info:
        svc_poses.create_library_pose(svc, body)
    assert info.value.field == "template"
    with pytest.raises(Invalid) as info:
        svc_poses.create_library_pose(svc, _payload(root=[9.0, 0.0, 0.0]))
    assert info.value.field == "root_translation"


def test_a_partial_record_is_refused_naming_the_field(svc):
    """A library pose carries the whole skeleton -- the retargeting premise --
    and the refusal arrives framed and addressed like every other bad field."""
    body = _payload()
    body["bones"] = {"hips": IDENTITY}
    with pytest.raises(Invalid, match="missing") as info:
        svc_poses.create_library_pose(svc, body)
    assert info.value.field == "bones"


@pytest.mark.parametrize("bad", ["not-an-id", "ABCDEF012345", ""])
def test_malformed_ids_are_not_found(svc, bad):
    with pytest.raises(NotFound):
        svc_poses.read_library_pose(svc, bad)
    with pytest.raises(NotFound):
        svc_poses.delete_library_pose(svc, bad)


# --- applying to an asset ---------------------------------------------------


def test_apply_snapshots_into_the_jobs_own_poses(svc, assets):
    job_id = _rigged_job(svc, assets)
    stored = svc_poses.create_library_pose(svc, _payload("Leap", root=[0.0, 0.0, 0.25]))

    local = svc_poses.apply_library_pose(svc, job_id, stored["id"])
    assert local["id"] != stored["id"], "the snapshot is a fresh local pose"
    assert local["name"] == "Leap"
    assert local["root_translation"] == [0.0, 0.0, 0.25]
    assert local["source_pose"] == {
        "id": stored["id"],
        "name": "Leap",
        "updated": stored["updated"],
    }
    assert [p["id"] for p in rigging.list_poses(assets / job_id)] == [local["id"]]


def test_editing_or_deleting_the_library_pose_leaves_the_snapshot_alone(svc, assets):
    """Immutability is automatic because applying is a copy -- the whole
    provenance model in one assertion."""
    job_id = _rigged_job(svc, assets)
    stored = svc_poses.create_library_pose(svc, _payload("Leap"))
    local = svc_poses.apply_library_pose(svc, job_id, stored["id"])
    path = rigging.pose_path(assets / job_id, local["id"])
    before = path.read_bytes()

    svc_poses.update_library_pose(svc, stored["id"], _payload("Leap far", root=[0.5, 0, 0]))
    svc_poses.delete_library_pose(svc, stored["id"])
    assert path.read_bytes() == before


def test_apply_refuses_a_template_mismatch_by_label(svc, assets):
    job_id = _rigged_job(svc, assets, template="humanoid")
    fish = svc_poses.create_library_pose(svc, _payload("Swim", template="fish"))
    with pytest.raises(Conflict, match="Fish"):
        svc_poses.apply_library_pose(svc, job_id, fish["id"])


def test_apply_to_an_unrigged_job_is_not_found(svc, assets):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    stored = svc_poses.create_library_pose(svc, _payload())
    with pytest.raises(NotFound, match="not rigged"):
        svc_poses.apply_library_pose(svc, job_id, stored["id"])


def test_apply_respects_the_per_job_pose_cap(svc, assets, monkeypatch):
    job_id = _rigged_job(svc, assets)
    stored = svc_poses.create_library_pose(svc, _payload())
    monkeypatch.setattr(rigging, "MAX_POSES", 0)
    with pytest.raises(Conflict, match="at most"):
        svc_poses.apply_library_pose(svc, job_id, stored["id"])


# --- the template preview ---------------------------------------------------


def _fake_probe(monkeypatch, ok=True, detail="bpy 5.2.0"):
    monkeypatch.setattr(
        doctor, "blender_check", lambda **kw: Check("Blender (rigging)", ok, detail, False)
    )


def _fake_worker(monkeypatch):
    calls = []

    def run_worker(spec, **kwargs):
        calls.append((spec, kwargs))
        Path(spec["out_glb"]).write_bytes(b"armature-glb")
        return {"ok": True, "bones": 1}

    monkeypatch.setattr(rigging, "run_worker", run_worker)
    return calls


def test_the_preview_builds_once_and_is_then_a_cache_hit(svc, monkeypatch):
    _fake_probe(monkeypatch)
    calls = _fake_worker(monkeypatch)
    first = svc_poses.template_preview(svc, "humanoid")
    assert first == poselib.preview_path(svc.config, "humanoid")
    assert first.read_bytes() == b"armature-glb"
    assert svc_poses.template_preview(svc, "humanoid") == first
    assert len(calls) == 1, "the second request served the cache"
    spec, kwargs = calls[0]
    assert spec["op"] == "armature"
    assert spec["template"] == "humanoid"
    assert spec["out_glb"].endswith(".glb"), "the exporter appends .glb to anything else"
    assert kwargs["timeout"] == svc.config.pose_timeout
    # The sidecar landed last and records what the cache is keyed on.
    sidecar = poselib.read_preview_sidecar(svc.config, "humanoid")
    assert sidecar["blender_version"] == "bpy 5.2.0"


def test_a_blender_upgrade_rebuilds_the_preview(svc, monkeypatch):
    _fake_probe(monkeypatch, detail="bpy 5.2.0")
    calls = _fake_worker(monkeypatch)
    svc_poses.template_preview(svc, "humanoid")
    _fake_probe(monkeypatch, detail="bpy 6.0.0")
    svc_poses.template_preview(svc, "humanoid")
    assert len(calls) == 2


def test_no_blender_is_a_failed_with_the_probes_own_words(svc, monkeypatch):
    _fake_probe(monkeypatch, ok=False, detail="bpy import failed -- install with: uv sync")
    with pytest.raises(Failed, match="install with"):
        svc_poses.template_preview(svc, "humanoid")


def test_an_unknown_template_preview_is_invalid(svc, monkeypatch):
    _fake_probe(monkeypatch)
    with pytest.raises(Invalid):
        svc_poses.template_preview(svc, "dragon")


def test_a_failed_build_caches_nothing(svc, monkeypatch):
    _fake_probe(monkeypatch)

    def boom(spec, **kwargs):
        raise rigging.BlenderError("exploded")

    monkeypatch.setattr(rigging, "run_worker", boom)
    with pytest.raises(Failed):
        svc_poses.template_preview(svc, "humanoid")
    assert not poselib.preview_path(svc.config, "humanoid").exists()
    assert poselib.read_preview_sidecar(svc.config, "humanoid") is None
