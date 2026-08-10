"""The global pose library's pure half: records, names, paths, preview cache.

Everything here runs with a bare ``data_dir`` and no service, which is the
point of ``poselib`` being its own module: every rule about what a stored pose
is stays assertable without Blender, a DB, or a Runtime.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from warlock import poselib, rigging


def _config(tmp_path):
    return SimpleNamespace(data_dir=tmp_path)


def _full_bones(template="humanoid"):
    """Every bone of the template at identity -- what ``get_pose()`` saves.

    A record must carry the whole skeleton (the retargeting premise in the
    module docstring), so the payload helper mirrors the only writer.
    """
    return {b["name"]: [0.0, 0.0, 0.0, 1.0] for b in rigging.get_template(template).bones}


def _payload(**over):
    base = {
        "name": "Crouch",
        "template": "humanoid",
        "bones": _full_bones(),
        "root_translation": [0.1, 0.0, -0.2],
    }
    base.update(over)
    return base


# --- records ----------------------------------------------------------------


def test_a_record_round_trips(tmp_path):
    config = _config(tmp_path)
    record = poselib.validate_record(_payload())
    stored = poselib.save_record(config, record)
    assert rigging.is_valid_id(stored["id"])
    assert stored["created"] == stored["updated"]

    back = poselib.read_record(config, stored["id"])
    assert back == stored
    assert back["name"] == "Crouch"
    assert back["template"] == "humanoid"
    assert back["bones"] == _full_bones()
    assert back["root_translation"] == [0.1, 0.0, -0.2]
    assert [r["id"] for r in poselib.list_records(config)] == [stored["id"]]


def test_overwrite_keeps_created_and_bumps_updated(tmp_path, monkeypatch):
    config = _config(tmp_path)
    record = poselib.validate_record(_payload())
    monkeypatch.setattr(poselib.time, "time", lambda: 100.0)
    stored = poselib.save_record(config, record)
    monkeypatch.setattr(poselib.time, "time", lambda: 200.0)
    again = poselib.save_record(config, poselib.validate_record(_payload(name="Crouch low")),
                               stored["id"])
    assert again["id"] == stored["id"]
    assert again["created"] == 100.0
    assert again["updated"] == 200.0
    assert poselib.read_record(config, stored["id"])["name"] == "Crouch low"


def test_a_failed_write_leaves_the_existing_record_intact(tmp_path, monkeypatch):
    """The mkstemp+replace staging: a save that dies mid-write must not
    truncate the only copy, and must not leave its temp behind."""
    config = _config(tmp_path)
    stored = poselib.save_record(config, poselib.validate_record(_payload()))

    def boom(src, dest):
        raise OSError("disk full")

    monkeypatch.setattr(poselib.os, "replace", boom)
    with pytest.raises(OSError):
        poselib.save_record(config, poselib.validate_record(_payload(name="B")), stored["id"])
    monkeypatch.undo()
    assert poselib.read_record(config, stored["id"])["name"] == "Crouch"
    leftovers = [p for p in poselib.library_dir(config).iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_a_corrupt_file_costs_itself_not_the_list(tmp_path):
    config = _config(tmp_path)
    good = poselib.save_record(config, poselib.validate_record(_payload()))
    bad_id = rigging.new_id()
    poselib.library_dir(config).joinpath(f"{bad_id}.json").write_text("{not json")
    assert [r["id"] for r in poselib.list_records(config)] == [good["id"]]


def test_list_records_filters_by_template_and_sorts_by_name(tmp_path):
    config = _config(tmp_path)
    b = poselib.save_record(config, poselib.validate_record(_payload(name="bravo")))
    a = poselib.save_record(config, poselib.validate_record(_payload(name="Alpha")))
    fish = poselib.save_record(
        config,
        poselib.validate_record(
            _payload(name="Swim", template="fish", bones=_full_bones("fish"))
        ),
    )
    # Case-insensitive by name: "Alpha" before "bravo" despite the capital.
    assert [r["id"] for r in poselib.list_records(config)] == [a["id"], b["id"], fish["id"]]
    assert [r["id"] for r in poselib.list_records(config, "fish")] == [fish["id"]]


def test_delete_record(tmp_path):
    config = _config(tmp_path)
    stored = poselib.save_record(config, poselib.validate_record(_payload()))
    assert poselib.delete_record(config, stored["id"]) is True
    assert poselib.delete_record(config, stored["id"]) is False
    assert poselib.list_records(config) == []


# --- ids and paths ----------------------------------------------------------


@pytest.mark.parametrize("bad", ["..", "../../etc", "ABCDEF012345", "", "0123456789abc"])
def test_pose_path_rejects_traversal_and_malformed_ids(tmp_path, bad):
    with pytest.raises(ValueError, match="malformed pose id"):
        poselib.pose_path(_config(tmp_path), bad)


def test_the_library_lives_under_data_dir_poser(tmp_path):
    config = _config(tmp_path)
    pose_id = rigging.new_id()
    assert poselib.pose_path(config, pose_id) == tmp_path / "poser" / "poses" / f"{pose_id}.json"
    assert poselib.preview_path(config, "humanoid") == (
        tmp_path / "poser" / "previews" / "humanoid.glb"
    )


def test_preview_path_refuses_an_unknown_template(tmp_path):
    """The registry is the path sanitiser: a template key never becomes a
    filename unless the registry knows it."""
    with pytest.raises(ValueError, match="unknown skeleton template"):
        poselib.preview_path(_config(tmp_path), "../evil")


# --- validation -------------------------------------------------------------


def test_validate_record_normalizes(tmp_path):
    out = poselib.validate_record(_payload(name="  Crouch  "))
    assert out == {
        "name": "Crouch",
        "template": "humanoid",
        "bones": _full_bones(),
        "root_translation": [0.1, 0.0, -0.2],
    }


def test_validate_record_defaults_an_absent_root_translation_to_zeros():
    out = poselib.validate_record(_payload(root_translation=None))
    assert out["root_translation"] == [0.0, 0.0, 0.0]
    out = poselib.validate_record({k: v for k, v in _payload().items()
                                   if k != "root_translation"})
    assert out["root_translation"] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "over, field, match",
    [
        ({"template": "dragon"}, "template", "unknown skeleton template"),
        ({"template": ""}, "template", "unknown skeleton template"),
        ({"name": ""}, "name", "requires a name"),
        ({"name": "x" * 65}, "name", "at most"),
        ({"root_translation": [1.0, 0.0]}, "root_translation", "3-vector"),
        ({"root_translation": ["x", 0.0, 0.0]}, "root_translation", "not numeric"),
        ({"root_translation": [3.0, 0.0, 0.0]}, "root_translation", "character heights"),
        ({"root_translation": [0.0, -2.5, 0.0]}, "root_translation", "character heights"),
    ],
)
def test_validate_record_refusals_carry_their_field(over, field, match):
    with pytest.raises(poselib.RecordError, match=match) as info:
        poselib.validate_record(_payload(**over))
    assert info.value.field == field


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_root_translation_is_finite_checked_before_range_checked(bad):
    """abs(nan) <= 2.0 is False, so the range message would blame a value that
    is not a number -- the finite check has to win."""
    with pytest.raises(poselib.RecordError, match="not numeric") as info:
        poselib.validate_record(_payload(root_translation=[bad, 0.0, 0.0]))
    assert info.value.field == "root_translation"


def test_validate_record_rejects_a_bone_the_template_lacks():
    bones = _full_bones()
    bones["tentacle"] = [0, 0, 0, 1]
    with pytest.raises(poselib.RecordError, match="unknown bone"):
        poselib.validate_record(_payload(bones=bones))


def test_validate_record_refuses_a_partial_record():
    """Exactly the template's bone set, not a subset: the module docstring's
    retargeting argument stands on a record carrying every bone, and a partial
    record applied over a posed editor keeps stale rotations which get_pose()
    would then save as authored. Completeness is the right bar for what is on
    disk too -- Poser's only writer is get_pose(), which is always complete."""
    bones = _full_bones()
    bones.pop("hips")
    bones.pop("spine")
    with pytest.raises(poselib.RecordError, match="missing 2 of") as info:
        poselib.validate_record(_payload(bones=bones))
    assert info.value.field == "bones"


def test_record_error_is_a_value_error():
    """The GuidanceError shape: every caller that already catches ValueError
    keeps working, and invalid_from picks the field up off the instance."""
    assert issubclass(poselib.RecordError, ValueError)


# --- names ------------------------------------------------------------------


def test_find_name_is_casefolded_and_per_template(tmp_path):
    config = _config(tmp_path)
    stored = poselib.save_record(config, poselib.validate_record(_payload()))
    assert poselib.find_name(config, "humanoid", "crouch") == stored["id"]
    assert poselib.find_name(config, "humanoid", "  CROUCH  ") == stored["id"]
    # Same name on another template is not a collision.
    assert poselib.find_name(config, "fish", "crouch") is None
    # A rename or overwrite skips colliding with itself.
    assert poselib.find_name(config, "humanoid", "crouch", exclude=stored["id"]) is None


def test_next_copy_name_counts_up_and_skips_taken():
    assert poselib.next_copy_name("Crouch") == "Crouch.001"
    assert poselib.next_copy_name("Crouch", ["Crouch.001"]) == "Crouch.002"
    assert poselib.next_copy_name("Crouch.001") == "Crouch.002"
    # Case-insensitive, like the uniqueness rule it exists to satisfy.
    assert poselib.next_copy_name("Crouch", ["crouch.001"]) == "Crouch.002"


def test_the_library_cap_matches_the_per_job_one():
    assert poselib.MAX_LIBRARY_POSES == rigging.MAX_POSES == 500


# --- mirroring --------------------------------------------------------------


def test_mirror_root_translation_reflects_x_only():
    assert poselib.mirror_root_translation([0.3, -0.1, 0.7]) == [-0.3, -0.1, 0.7]


def test_mirroring_the_root_twice_is_the_identity():
    v = [0.3, -0.1, 0.7]
    assert poselib.mirror_root_translation(poselib.mirror_root_translation(v)) == v


# --- the preview cache ------------------------------------------------------


def _seed_preview(config, key="humanoid", version="bpy 4.5.0"):
    poselib.preview_path(config, key).parent.mkdir(parents=True, exist_ok=True)
    poselib.preview_path(config, key).write_bytes(b"glb")
    poselib.write_preview_sidecar(config, key, version)


def test_a_fresh_preview_is_valid(tmp_path):
    config = _config(tmp_path)
    _seed_preview(config)
    assert poselib.preview_valid(config, "humanoid", "bpy 4.5.0") is True


def test_a_missing_glb_invalidates_whatever_the_sidecar_claims(tmp_path):
    config = _config(tmp_path)
    _seed_preview(config)
    poselib.preview_path(config, "humanoid").unlink()
    assert poselib.preview_valid(config, "humanoid", "bpy 4.5.0") is False


def test_a_missing_or_corrupt_sidecar_invalidates(tmp_path):
    config = _config(tmp_path)
    _seed_preview(config)
    poselib.preview_sidecar_path(config, "humanoid").write_text("{not json")
    assert poselib.preview_valid(config, "humanoid", "bpy 4.5.0") is False
    poselib.preview_sidecar_path(config, "humanoid").unlink()
    assert poselib.preview_valid(config, "humanoid", "bpy 4.5.0") is False


def test_a_blender_upgrade_invalidates(tmp_path):
    config = _config(tmp_path)
    _seed_preview(config, version="bpy 4.5.0")
    assert poselib.preview_valid(config, "humanoid", "bpy 5.0.1") is False


def test_a_template_edit_invalidates(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _seed_preview(config)
    monkeypatch.setattr(poselib, "template_digest", lambda key: "different")
    assert poselib.preview_valid(config, "humanoid", "bpy 4.5.0") is False


def test_an_epoch_bump_invalidates(tmp_path, monkeypatch):
    """The hand-bumped constant for _build_armature/_export changes: nothing
    else in the sidecar can see our own exporter code change."""
    config = _config(tmp_path)
    _seed_preview(config)
    monkeypatch.setattr(poselib, "PREVIEW_EPOCH", poselib.PREVIEW_EPOCH + 1)
    assert poselib.preview_valid(config, "humanoid", "bpy 4.5.0") is False


def test_a_vanished_template_file_answers_stale_not_a_raise(tmp_path, monkeypatch):
    """The registry loads once per process, so the file can go missing under
    it. A cache-validity question must answer "rebuild", never throw a
    FileNotFoundError out of service.poses.template_preview unframed."""
    config = _config(tmp_path)
    _seed_preview(config)
    rigging.get_template("humanoid")  # the registry is loaded and cached
    monkeypatch.setattr(rigging, "TEMPLATE_DIR", tmp_path / "gone")
    assert poselib.template_digest("humanoid") is None
    assert poselib.preview_valid(config, "humanoid", "bpy 4.5.0") is False


def test_template_digest_tracks_the_file_bytes(tmp_path, monkeypatch):
    (tmp_path / "humanoid.json").write_text(json.dumps({
        "key": "humanoid", "label": "H", "root": "a",
        "bones": [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
    }))
    monkeypatch.setattr(rigging, "TEMPLATE_DIR", tmp_path)
    monkeypatch.setattr(rigging, "_templates", None)
    first = poselib.template_digest("humanoid")
    (tmp_path / "humanoid.json").write_text(
        (tmp_path / "humanoid.json").read_text() + " "
    )
    assert poselib.template_digest("humanoid") != first
    monkeypatch.setattr(rigging, "_templates", None)


def test_the_sidecar_records_all_three_invalidation_axes(tmp_path):
    config = _config(tmp_path)
    _seed_preview(config)
    sidecar = poselib.read_preview_sidecar(config, "humanoid")
    assert set(sidecar) == {"template_sha256", "blender_version", "preview_epoch"}
    assert sidecar["preview_epoch"] == poselib.PREVIEW_EPOCH
    assert sidecar["blender_version"] == "bpy 4.5.0"
    assert sidecar["template_sha256"] == poselib.template_digest("humanoid")


def test_unit_box_constants():
    """Pinned because two things stand on them: fit_template's place() is the
    identity over exactly this box, and a root_translation is in
    character-height units only while the armature is exactly 1 tall."""
    assert poselib.UNIT_LO == (-0.5, -0.5, 0.0)
    assert poselib.UNIT_HI == (0.5, 0.5, 1.0)
    assert poselib.MAX_ROOT_TRANSLATION == 2.0


def test_copy_names_count_up_the_way_clay_ops_names_do():
    """The four-line judgement -- count up, never prefix -- is stated twice on
    purpose (this module may not import studio, says the comment beside
    ``_SUFFIX``); this pins that the two spellings still agree."""
    from warlock.studio.clay.ops import _next_name

    cases = (
        ("Crouch", ()),
        ("Crouch", ("Crouch.001",)),
        ("Crouch.002", ()),
        ("Crouch.002", ("Crouch.003", "Crouch.004")),
    )
    for name, taken in cases:
        assert poselib.next_copy_name(name, taken) == _next_name(name, taken)
