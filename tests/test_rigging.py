"""Everything about rigging that is decidable without Blender.

The bpy-dependent half lives behind importorskip at the bottom, matching how
tests/test_offline.py guards the heavy paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from warlock import rigging

# --- template registry ------------------------------------------------------


def test_both_templates_load():
    keys = set(rigging.templates())
    assert {"humanoid", "quadruped"} <= keys


@pytest.mark.parametrize("key", ["humanoid", "quadruped"])
def test_template_hierarchy_is_well_formed(key):
    t = rigging.get_template(key)
    names = {b["name"] for b in t.bones}
    assert len(names) == len(t.bones)
    assert t.root in names
    for bone in t.bones:
        assert bone["parent"] is None or bone["parent"] in names
    # Exactly one root: a second parentless bone would export as a detached
    # skeleton the pose editor has no way to present.
    assert sum(1 for b in t.bones if b["parent"] is None) == 1
    for a, b in t.mirror_pairs:
        assert a in names and b in names


@pytest.mark.parametrize("key", ["humanoid", "quadruped"])
def test_mirror_pairs_are_actually_mirrored(key):
    """A .L/.R pair must be reflections in X, or the future mirror-pose button
    silently produces a lopsided pose."""
    t = rigging.get_template(key)
    by_name = {b["name"]: b for b in t.bones}
    for left, right in t.mirror_pairs:
        for end in ("head", "tail"):
            lx, ly, lz = by_name[left][end]
            rx, ry, rz = by_name[right][end]
            assert lx == pytest.approx(-rx)
            assert ly == pytest.approx(ry)
            assert lz == pytest.approx(rz)


def test_get_template_rejects_unknown():
    with pytest.raises(ValueError, match="unknown skeleton template"):
        rigging.get_template("dragon")


def test_catalog_shape():
    entries = rigging.catalog()
    assert all(set(e) == {"key", "label"} for e in entries)


def test_malformed_template_is_skipped_not_fatal(tmp_path, monkeypatch):
    (tmp_path / "good.json").write_text(
        json.dumps(
            {
                "key": "good",
                "label": "Good",
                "root": "a",
                "bones": [{"name": "a", "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1]}],
            }
        )
    )
    (tmp_path / "broken.json").write_text('{"key": "broken"}')
    monkeypatch.setattr(rigging, "TEMPLATE_DIR", tmp_path)
    monkeypatch.setattr(rigging, "_templates", None)
    assert set(rigging.templates()) == {"good"}


def test_template_with_unknown_parent_is_rejected():
    with pytest.raises(ValueError, match="unknown parent"):
        rigging._parse_template(
            {
                "key": "x",
                "label": "X",
                "root": "a",
                "bones": [{"name": "a", "parent": "ghost", "head": [0, 0, 0], "tail": [0, 0, 1]}],
            }
        )


# --- fitting ----------------------------------------------------------------


def test_fit_maps_normalized_coords_onto_the_bbox():
    t = rigging.get_template("humanoid")
    lo, hi = [-1.0, -0.5, 0.0], [1.0, 0.5, 4.0]
    fitted = rigging.fit_template(t, lo, hi)
    by_name = {b["name"]: b for b in fitted}
    # head's tail is z=1.0 normalized -> the top of the bbox; hips sit on the
    # bbox centre in x and y.
    assert by_name["head"]["tail"][2] == pytest.approx(4.0)
    assert by_name["hips"]["head"][0] == pytest.approx(0.0)
    assert by_name["hips"]["head"][2] == pytest.approx(0.53 * 4.0)
    # x spans -0.5..0.5 of a 2.0-wide box, so 0.07 normalized is 0.14 world.
    assert by_name["thigh.L"]["head"][0] == pytest.approx(0.14)


def test_fit_respects_an_off_origin_bbox():
    t = rigging.get_template("humanoid")
    fitted = rigging.fit_template(t, [10.0, 20.0, 30.0], [11.0, 21.0, 31.0])
    for bone in fitted:
        for point in (bone["head"], bone["tail"]):
            assert 9.9 <= point[0] <= 11.1
            assert 19.9 <= point[1] <= 21.1
            assert 29.9 <= point[2] <= 31.1


@pytest.mark.parametrize("key", ["humanoid", "quadruped"])
def test_fit_never_produces_a_zero_length_bone(key):
    """Blender deletes zero-length bones on leaving edit mode, taking their
    children with them -- so a flat bbox must not be able to produce one."""
    t = rigging.get_template(key)
    for lo, hi in [
        ([-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]),
        ([-1.0, 0.0, 0.0], [1.0, 0.0, 2.0]),  # zero depth
        ([0.0, 0.0, 0.0], [0.0, 0.0, 2.0]),  # a line
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),  # a point
    ]:
        for bone in rigging.fit_template(t, lo, hi):
            assert rigging._distance(bone["head"], bone["tail"]) > 0.0


def test_fit_preserves_parent_names():
    t = rigging.get_template("quadruped")
    fitted = rigging.fit_template(t, [-1, -2, 0], [1, 2, 1])
    assert [b["parent"] for b in fitted] == [b["parent"] for b in t.bones]


# --- pose payloads ----------------------------------------------------------


def test_validate_pose_accepts_a_unit_quaternion():
    pose = rigging.validate_pose({"name": "idle", "bones": {"hips": [0, 0, 0, 1]}})
    assert pose == {"name": "idle", "bones": {"hips": [0.0, 0.0, 0.0, 1.0]}}


def test_validate_pose_renormalizes_drifted_quaternions():
    """Accumulated gizmo drags leave the unit sphere by float noise; refusing
    the save over that would be indefensible."""
    pose = rigging.validate_pose({"name": "wave", "bones": {"hips": [0, 0, 0, 2.0]}})
    assert pose["bones"]["hips"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"name": "", "bones": {"hips": [0, 0, 0, 1]}}, "requires a name"),
        ({"name": "x" * 65, "bones": {"hips": [0, 0, 0, 1]}}, "at most"),
        ({"name": "a", "bones": {}}, "non-empty"),
        ({"name": "a"}, "non-empty"),
        ({"name": "a", "bones": {"hips": [0, 0, 1]}}, "4-element"),
        ({"name": "a", "bones": {"hips": ["x", 0, 0, 1]}}, "not numeric"),
        ({"name": "a", "bones": {"hips": [0, 0, 0, 0]}}, "degenerate"),
    ],
)
def test_validate_pose_rejects_bad_payloads(payload, match):
    with pytest.raises(ValueError, match=match):
        rigging.validate_pose(payload)


def test_validate_pose_rejects_unknown_bones_when_told_the_skeleton():
    with pytest.raises(ValueError, match="unknown bone"):
        rigging.validate_pose({"name": "a", "bones": {"tentacle": [0, 0, 0, 1]}}, ["hips"])


# --- ids and paths ----------------------------------------------------------


def test_new_id_is_accepted_by_its_own_validator():
    assert rigging.is_valid_id(rigging.new_id())


@pytest.mark.parametrize(
    "bad", ["..", "../../etc", "ABCDEF012345", "0123456789ab/x", "", "0123456789abc"]
)
def test_traversal_and_malformed_ids_are_rejected(bad):
    assert not rigging.is_valid_id(bad)


# --- the worker boundary ----------------------------------------------------


def test_rig_spec_validates_the_template_before_spawning_anything(tmp_path):
    with pytest.raises(ValueError):
        rigging.rig_spec(tmp_path, "dragon")


def test_rig_spec_paths_all_land_in_the_job_dir(tmp_path):
    spec = rigging.rig_spec(tmp_path, "humanoid")
    assert spec["op"] == "rig"
    for key in ("source_glb", "out_glb", "out_json", "result_path"):
        assert tmp_path in type(tmp_path)(spec[key]).parents


def test_read_rig_returns_none_without_a_rig(tmp_path):
    assert rigging.read_rig(tmp_path) is None
    assert rigging.rig_bone_names(tmp_path) is None


def test_read_rig_survives_corrupt_json(tmp_path):
    (tmp_path / "rig.json").write_text("{not json")
    assert rigging.read_rig(tmp_path) is None


def test_read_rig_round_trips(tmp_path):
    (tmp_path / "rig.json").write_text(json.dumps({"bones": [{"name": "hips"}]}))
    assert rigging.rig_bone_names(tmp_path) == ["hips"]


def test_run_worker_failure_is_a_blender_error_not_a_traceback(tmp_path):
    """Whether bpy is installed or not, rigging a mesh that isn't there must
    surface as one typed error carrying the worker's output."""
    spec = rigging.rig_spec(tmp_path, "humanoid")
    with pytest.raises(rigging.BlenderError):
        rigging.run_worker(spec, timeout=300)


def test_run_worker_kills_a_worker_that_hangs_without_closing_stdout(tmp_path, monkeypatch):
    """The timeout has to cover the *run*, not just the wait after EOF.

    A bpy process wedged in a weight solve or an EEVEE render writes nothing
    more and never closes stdout, so reading it inline would block forever and
    take the single serial job queue down with it -- no rig, pose or sheet job
    would ever start again.
    """
    real_popen = subprocess.Popen

    def fake_popen(_cmd, **kw):
        # Holds stdout open and never exits, exactly like a wedged worker.
        return real_popen([sys.executable, "-c", "import time; time.sleep(120)"], **kw)

    monkeypatch.setattr(rigging.subprocess, "Popen", fake_popen)
    started: list[subprocess.Popen] = []
    began = time.monotonic()
    with pytest.raises(rigging.BlenderError, match="timed out"):
        rigging.run_worker(
            {"op": "rig", "result_path": str(tmp_path / "r.json")},
            timeout=1.0,
            on_start=started.append,
        )
    assert time.monotonic() - began < 30, "the deadline did not fire"
    assert started and started[0].poll() is not None, "the hung process was left running"


def test_run_worker_rejects_an_unknown_op(tmp_path):
    with pytest.raises(rigging.BlenderError, match="code 2"):
        rigging.run_worker(
            {"op": "sculpt", "result_path": str(tmp_path / "r.json")}, timeout=60
        )


# --- with Blender actually installed ----------------------------------------


@pytest.mark.gpu
def test_end_to_end_rig_of_a_generated_cube(tmp_path):
    pytest.importorskip("bpy")
    import bpy  # noqa: F401  -- only to confirm the same interpreter has it

    from warlock.pipelines import blender_worker

    # A cube is manifold, so this exercises the heat-weighting path rather
    # than the envelope fallback.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    source = tmp_path / "model.glb"
    bpy.ops.export_scene.gltf(filepath=str(source), export_format="GLB")

    spec = rigging.rig_spec(tmp_path, "humanoid")
    result = blender_worker.op_rig(bpy, spec)
    assert result["ok"] is True
    assert result["weighting"] in ("automatic", "envelope")
    assert (tmp_path / "rig.glb").exists()
    rig = rigging.read_rig(tmp_path)
    assert rig["template"] == "humanoid"
    assert len(rig["bones"]) == result["bones"]


def _glb_node_rotations(path) -> dict[str, list[float]]:
    """Every named node's local rotation, read straight out of the GLB.

    Deliberately not via bpy: the point of the test below is that the *file*
    the browser downloads carries the rotations the browser asked for, so
    reading it back through Blender would beg the question.
    """
    import struct

    data = path.read_bytes()
    offset = 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        if kind == 0x4E4F534A:   # 'JSON'
            doc = json.loads(data[offset + 8 : offset + 8 + length])
            return {
                n["name"]: n.get("rotation", [0.0, 0.0, 0.0, 1.0])
                for n in doc["nodes"]
                if "name" in n
            }
        offset += 8 + length + (-length % 4)
    raise AssertionError("GLB has no JSON chunk")


@pytest.mark.gpu
def test_a_posed_glb_carries_back_exactly_the_rotations_it_was_given(tmp_path):
    """The contract the whole pose feature rests on.

    The browser only ever sees glTF node-local rotations, and Blender only ever
    wants pose-bone bases. blender_worker._rest_local_rotation claims the two
    are one composition apart with no per-bone axis correction; if that were
    wrong, poses would come back subtly rotated and nothing else would notice.
    """
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.context.object.scale = (0.3, 0.2, 1.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")
    blender_worker.op_rig(bpy, rigging.rig_spec(tmp_path, "humanoid"))

    # A parent and its child, so bone-chain accumulation is exercised too.
    want = {
        "upper_arm.L": [0.3826834, 0.0, 0.0, 0.9238795],   # 45 deg about local X
        "forearm.L": [0.0, 0.0, 0.2588190, 0.9659258],     # 30 deg about local Z
    }
    out = tmp_path / "posed.glb"
    result = blender_worker.op_pose(
        bpy, {"rig_glb": str(tmp_path / "rig.glb"), "out_glb": str(out), "bones": want}
    )
    assert result == {"ok": True, "bones": 2, "unknown": []}

    got = _glb_node_rotations(out)
    for name, target in want.items():
        # q and -q are the same rotation, and the exporter is free to pick either.
        deltas = [abs(a - b) for a, b in zip(got[name], target, strict=True)]
        sums = [abs(a + b) for a, b in zip(got[name], target, strict=True)]
        assert max(deltas) < 1e-4 or max(sums) < 1e-4, f"{name}: {got[name]} != {target}"


@pytest.mark.gpu
def test_posing_a_bone_the_rig_lacks_is_reported_not_fatal(tmp_path):
    """A pose saved against one skeleton should still mostly apply after a
    re-rig, so an unknown name is data, not an error."""
    pytest.importorskip("bpy")
    import bpy

    from warlock.pipelines import blender_worker

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(tmp_path / "model.glb"), export_format="GLB")
    blender_worker.op_rig(bpy, rigging.rig_spec(tmp_path, "humanoid"))

    result = blender_worker.op_pose(
        bpy,
        {
            "rig_glb": str(tmp_path / "rig.glb"),
            "out_glb": str(tmp_path / "posed.glb"),
            "bones": {"hips": [0.0, 0.0, 0.0, 1.0], "tail_01": [0.0, 0.0, 0.0, 1.0]},
        },
    )
    assert result["bones"] == 1
    assert result["unknown"] == ["tail_01"]
    assert (tmp_path / "posed.glb").exists()


def test_a_glb_round_trips_through_the_worker_into_a_real_fbx(tmp_path):
    """op_fbx actually produces a file Unity/Unreal would accept.

    The spec-shape test above only pins the wire format; this runs the real
    Blender subprocess, because the failure mode worth catching is an export
    operator that rejects one of its keyword arguments -- which no amount of
    dict-shape assertion would find.
    """
    pytest.importorskip("bpy")
    trimesh = pytest.importorskip("trimesh")

    source = tmp_path / "model.glb"
    trimesh.Scene(trimesh.creation.box(extents=(1.0, 2.0, 1.0))).export(source)
    out = tmp_path / "model.fbx"

    rigging.run_worker(rigging.fbx_spec(source, out, tmp_path), timeout=300)

    assert out.exists()
    # "Kaydara FBX Binary" is the magic every FBX reader looks for.
    assert out.read_bytes()[:18] == b"Kaydara FBX Binary"
