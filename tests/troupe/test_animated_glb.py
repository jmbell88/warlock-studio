"""``animated.glb``: the clip library, leaving the app as 3D animation.

Warlock has authored clips, a clip editor, interpolation and an EEVEE renderer
that poses an armature per cell -- and until now the *only* consumer of any of
it was Troupe's 2D sheet. ``blender_worker._export`` hard-coded
``export_animations=False``, so a walk cycle could be authored, edited and
rendered, and never played by an engine.

The pure half is asserted here, in the default lane, because the interpolation
and the timing live on the host by construction. The bake itself needs a real
Blender and is asserted below through the service's own door, skipping without
``bpy`` exactly as ``test_rigging.py`` does -- it is not in the ``gpu`` lane,
which means "requires a local GPU and model weights", and this needs neither.
"""

from __future__ import annotations

import inspect
import json
import struct
from pathlib import Path

import pytest

from warlock import clips, rigging
from warlock.pipelines import charsheet
from warlock.pipelines import sheet as sheetlib
from warlock.service import derive, files
from warlock.studio import artifacts

#: A rig import plus five keyed actions and a glTF export. Well past the
#: suite's 120 s hang net, and still a hang net rather than a budget.
pytestmark = pytest.mark.timeout(600)


# --- the host half -----------------------------------------------------------


def test_a_track_carries_the_authors_own_timing_not_troupes():
    """``expand_clips`` resamples to the frame count the sheet's grid asks for,
    because a sheet has cells to fill. An exported animation has no grid, so it
    keeps the clip's own segment lengths -- otherwise the file would carry
    Troupe's layout as if it were the animation."""
    library = rigging.clip_library("humanoid")
    by_name = {clip["name"]: clip for clip in library["clips"]}
    tracks = {track["name"]: track for track in clips.animation_tracks("humanoid")}
    assert set(tracks) == set(by_name)

    for name, clip in by_name.items():
        authored = sheetlib.interpolate_clip(
            rigging.clip_keys("humanoid", name),
            clip["segments"],
            closed=bool(clip["closed"]),
            easing=str(clip["easing"]),
            space=str(clip["space"]),
            clip_id=name,
        )
        assert len(tracks[name]["frames"]) == len(authored), name
        assert tracks[name]["frames"][0]["bones"] == authored[0]["bones"], name


def test_the_timing_table_has_one_home():
    """Per-frame duration and the loop flag already exist, once, in
    ``charsheet.ANIMATIONS``. A second copy would be one edit from disagreeing
    about how fast a walk cycle is."""
    table = {name: (loop, ms) for name, _frames, loop, ms in charsheet.ANIMATIONS}
    for track in clips.animation_tracks("humanoid"):
        loop, duration_ms = table[track["name"]]
        assert track["loop"] is bool(loop)
        assert track["step"] == clips.ANIMATION_FPS * duration_ms / 1000.0


def test_the_timebase_divides_every_authored_tempo_exactly():
    """glTF stores sample times in seconds against one scene rate, so five
    clips with five tempos share one timebase. 100 fps is the number because a
    10 ms frame divides 150, 100, 60, 80 and 100 exactly -- no clip's tempo is
    rounded, which is the whole reason it is not 24 or 30."""
    for _name, _frames, _loop, duration_ms in charsheet.ANIMATIONS:
        step = clips.ANIMATION_FPS * duration_ms / 1000.0
        assert step == int(step), duration_ms


def test_the_spec_refuses_a_skeleton_with_nothing_authored(tmp_path):
    """Before a subprocess is spent: a file whose whole reason is its
    animations, carrying none, answers the question wrongly rather than not at
    all."""
    with pytest.raises(ValueError, match="fish"):
        clips.animate_spec(tmp_path, "fish", tmp_path / "out.glb", tmp_path)
    with pytest.raises(ValueError, match="dragon"):
        clips.animate_spec(tmp_path, "dragon", tmp_path / "out.glb", tmp_path)

    spec = clips.animate_spec(tmp_path, "humanoid", tmp_path / "out.glb", tmp_path)
    assert spec["op"] == "animate"
    assert spec["rig_glb"] == str(tmp_path / "rig.glb")
    assert len(spec["clips"]) == len(charsheet.ANIMATIONS)


def test_blender_does_no_interpolation():
    """The host/worker split ``fit_template`` establishes: frames arrive
    resolved, so the timing stays under test with no ``bpy``."""
    from warlock.pipelines import blender_worker

    # The docstring says the word; the body must not.
    body = inspect.getsource(blender_worker.op_animate).split('"""')[-1]
    assert "interpolate" not in body
    assert "sheet." not in body
    assert "animate" in blender_worker.OPS


def test_an_animated_export_carries_the_rest_armature():
    """The inverse of ``_export``'s own rule, and one decision rather than two
    knobs: a track is a rotation *from rest*, so a file carrying one has to
    carry the rest armature to play it against. A posed bake still must not."""
    from warlock.pipelines import blender_worker

    source = inspect.getsource(blender_worker._export)
    assert "export_animations=animations" in source
    assert "export_rest_position_armature=animations" in source
    assert "animations: bool = False" in source


# --- the doors ---------------------------------------------------------------


def _rigged(svc, *, template="humanoid", rig=True):
    job_id = svc.store.create("image", "a hooded ranger", {}, stage="model")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "model.glb").write_bytes(b"fake-glb")
    if rig:
        (job_dir / "rig.glb").write_bytes(b"fake-rig")
        (job_dir / "rig.json").write_text(json.dumps({"template": template}), "utf-8")
    svc.store.set_status(job_id, "done")
    return job_id


def test_the_artifact_is_offered_beside_the_rig_it_is_made_of():
    names = [name for name, _label in artifacts.ARTIFACTS]
    assert names.index("animated.glb") == names.index("rig.glb") + 1
    assert files.MEDIA["animated.glb"] == "model/gltf-binary"
    assert "animated.glb" in files.DERIVED_RIG
    # Not in DERIVED: that tuple is keyed on model.glb, and a name in it would
    # gate this on a mesh being finished while ignoring the rig it is made of.
    assert "animated.glb" not in files.DERIVED


def test_an_unrigged_mesh_cannot_be_animated(svc):
    job_id = _rigged(svc, rig=False)
    job = svc.store.get(job_id)
    job_dir = svc.job_dir(job_id)
    assert not files.ready(job, job_dir, "animated.glb")
    assert "not been rigged" in files.unready_reason(job, job_dir, "animated.glb")


def test_a_rig_with_no_clips_says_so_rather_than_baking_an_empty_file(svc):
    job_id = _rigged(svc, template="fish")
    job = svc.store.get(job_id)
    job_dir = svc.job_dir(job_id)
    assert not files.ready(job, job_dir, "animated.glb")
    reason = files.unready_reason(job, job_dir, "animated.glb")
    assert "nothing is authored" in reason


def test_a_rigged_mesh_may_be_animated(svc):
    job_id = _rigged(svc)
    job = svc.store.get(job_id)
    assert files.ready(job, svc.job_dir(job_id), "animated.glb")


def test_a_retarget_reports_the_animation_as_stale_with_the_rig(svc):
    """It is baked *from* rig.glb, so a retarget leaves it describing a
    skeleton skinned to a mesh that no longer exists."""
    from warlock.service import _jobs_rework

    job_id = _rigged(svc)
    job_dir = svc.job_dir(job_id)
    (job_dir / "animated.glb").write_bytes(b"fake")
    assert "animated.glb" in _jobs_rework.stale_rig_artifacts(job_dir)


def test_the_bake_is_staged_under_a_name_the_exporter_will_not_rename():
    """Blender's glTF exporter appends ``.glb`` to a path that does not end in
    it, so ``_staged``'s default ``.animated.glb.tmp`` would be written as
    ``.animated.glb.tmp.glb`` and the rename would find nothing --
    ``rigging.RIG_GLB_TMP``'s rule, met a second time. Existence is the
    freshness test for this artifact, which is why it is staged at all."""
    source = inspect.getsource(derive.get_file)
    assert 'tmp_name=".animated.tmp.glb"' in source
    assert "convert_lock(job_id, name)" in source


# --- the bake itself ---------------------------------------------------------


def _gltf_json(path: Path) -> dict:
    """The JSON chunk of a GLB, without a glTF library.

    Twelve-byte header, then length-prefixed chunks; the first is always JSON.
    """
    data = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "not a GLB"
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == 0x4E4F534A, "first chunk is not JSON"
    return json.loads(data[20 : 20 + chunk_len].decode("utf-8"))


def test_every_authored_clip_comes_back_as_a_named_glTF_animation(svc, tmp_path):
    """The end-to-end claim, through the service's own door.

    The subject is the template's own armature rather than a reconstruction --
    ``op_armature`` builds it with the same code path a real rig uses -- which
    is the honest scope: this asks whether the clips bake and land as named,
    playable tracks, not whether a 300k-face mesh survives the export.
    """
    pytest.importorskip("bpy")

    job_id = _rigged(svc, rig=False)
    job_dir = svc.job_dir(job_id)
    rigging.run_worker(
        rigging.armature_spec("humanoid", job_dir / "rig.glb", tmp_path),
        timeout=600,
    )
    assert (job_dir / "rig.glb").exists()
    (job_dir / "rig.json").write_text(json.dumps({"template": "humanoid"}), "utf-8")

    out = derive.get_file(svc, job_id, "animated.glb")
    assert out.exists() and out.name == "animated.glb"
    # The staging file is gone: a stranded dotfile lives for the life of the
    # job directory, where nothing ever looks again.
    assert not (job_dir / ".animated.tmp.glb").exists()

    gltf = _gltf_json(out)
    names = [str(anim.get("name") or "") for anim in gltf.get("animations") or ()]
    assert {name for name, _f, _l, _ms in charsheet.ANIMATIONS} <= set(names), names
    for anim in gltf["animations"]:
        assert anim["channels"], anim.get("name")
        assert anim["samplers"], anim.get("name")
