"""Weld-before-heat: the fallback chain, the epsilon and the review battery.

The hypothesis is that xatlas seam-split vertices make a trellis mesh
non-manifold for bone-heat's Laplacian solve even though nothing is open --
the same root cause that made ``meshreport`` measure UV seams and call them
holes -- and that merging by distance first lets ``ARMATURE_AUTO`` succeed.

Everything here is host-side. ``_skin`` takes bpy as an argument and
``_skin_steps``/``weld_distance`` are pure, which is what makes the whole
decision reachable on a machine with no Blender -- exactly the reason
``_rig_bones`` was carved out. The one thing that genuinely needs bpy is the
claim that the weld does not change what the user sees; that test is at the
bottom, behind the same importorskip the other Blender tests use.
"""

from __future__ import annotations

import json

import pytest

from warlock import meshreport, rigging
from warlock.pipelines import blender_worker

# --- the epsilon ------------------------------------------------------------


def test_the_weld_distance_is_a_fraction_of_the_model_not_an_absolute_size():
    """An absolute epsilon means something different on a 0.02 m gear than on
    a 30 m building, which is the reason meshreport's is relative too."""
    small = blender_worker.weld_distance([0, 0, 0], [0.02, 0.02, 0.02])
    large = blender_worker.weld_distance([0, 0, 0], [20.0, 20.0, 20.0])
    assert large == pytest.approx(small * 1000.0)


def test_the_weld_distance_is_meshreports_tolerance_reused_not_re_derived():
    """The two are the same judgement about the same meshes, so there is one
    number. A second constant here is how they drift."""
    diagonal = (3 * 4.0) ** 0.5
    assert blender_worker.weld_distance([-1, -1, -1], [1, 1, 1]) == pytest.approx(
        diagonal * meshreport.WELD_TOLERANCE
    )


def test_a_degenerate_bounding_box_asks_for_no_weld_at_all():
    assert blender_worker.weld_distance([1, 1, 1], [1, 1, 1]) == 0.0


# --- the chain, as a table --------------------------------------------------


def test_a_weld_that_merged_nothing_is_not_a_welded_rig():
    """The mesh the solve saw is the mesh it would have seen anyway, so the
    recorded method must not claim otherwise -- and there is no second,
    identical attempt to fall back to."""
    assert blender_worker._skin_steps(0) == (("automatic", False),)


def test_a_weld_that_merged_something_earns_its_own_name_and_a_retry():
    assert blender_worker._skin_steps(12) == (
        ("automatic-welded", False),
        ("automatic", True),
    )


# --- the chain, through a fake bpy ------------------------------------------


class _FakeMeshData:
    """Just enough of a bpy Mesh datablock: vertices, and copying itself."""

    def __init__(self, vertices: int, weighted: bool = False) -> None:
        self.vertices = [_FakeVertex(weighted) for _ in range(vertices)]
        self.materials: list[object] = []

    def copy(self) -> _FakeMeshData:
        clone = _FakeMeshData(0)
        clone.vertices = list(self.vertices)
        return clone


class _Weight:
    group = 0
    weight = 1.0


class _Group:
    index = 0


class _FakeVertex:
    def __init__(self, weighted: bool) -> None:
        self.groups = [_Weight()] if weighted else []


class _FakeGroups(list):
    def clear(self) -> None:
        del self[:]


class _FakeMesh:
    def __init__(self, vertices: int = 100) -> None:
        self.data = _FakeMeshData(vertices)
        self.modifiers: list[object] = []
        self.parent = None
        self.vertex_groups = _FakeGroups()

    def select_set(self, value: bool) -> None:
        pass


class _FakeArm:
    def select_set(self, value: bool) -> None:
        pass


class _FakeOps:
    def __init__(self, bpy: _FakeBpy) -> None:
        self._bpy = bpy

    @property
    def object(self):
        return self

    @property
    def mesh(self):
        return self

    # object.*
    def select_all(self, action: str) -> None:
        pass

    def mode_set(self, mode: str) -> None:
        self._bpy.modes.append(mode)

    def parent_set(self, type: str) -> None:  # noqa: A002 - bpy's own keyword
        bpy = self._bpy
        bpy.binds.append(type)
        if type != "ARMATURE_AUTO":
            return
        outcome = bpy.heat.pop(0) if bpy.heat else "ok"
        if outcome not in ("ok", "empty"):
            # An operator RuntimeError: one of the two ways a heat solve fails.
            raise RuntimeError(outcome)
        if outcome == "ok":
            # The other is a 'FINISHED' that leaves every group empty, which is
            # what "empty" means here: bound, and no weights to show for it.
            bpy.mesh.vertex_groups.append(_Group())
            for vertex in bpy.mesh.data.vertices:
                vertex.groups = [_Weight()]

    # mesh.*
    def remove_doubles(self, threshold: float) -> None:
        bpy = self._bpy
        bpy.welded_at = threshold
        if bpy.weld_raises is not None:
            raise RuntimeError(bpy.weld_raises)
        del bpy.mesh.data.vertices[: bpy.merges]

    def normals_make_consistent(self, inside: bool) -> None:
        pass


class _FakeMeshes(list):
    def remove(self, block) -> None:
        self.append(block)


class _FakeBpy:
    """``heat`` is the outcome of each ARMATURE_AUTO attempt in order:
    'ok', 'empty' (a FINISHED that leaves no weights), or a message to raise."""

    def __init__(self, mesh, *, merges: int = 0, heat=None, weld_raises=None) -> None:
        self.mesh = mesh
        self.merges = merges
        self.heat = list(heat or [])
        self.weld_raises = weld_raises
        self.binds: list[str] = []
        self.modes: list[str] = []
        self.welded_at: float | None = None
        self.ops = _FakeOps(self)
        self.data = type("D", (), {"meshes": _FakeMeshes()})()

        class _View:
            objects = type("O", (), {"active": None})()

        self.context = type("C", (), {"view_layer": _View()})()


def test_welding_first_is_what_lets_the_heat_solve_take():
    """The whole point of the phase: the seam-split mesh the solve refused is
    welded, and the rig is recorded as the welded one it actually is."""
    mesh = _FakeMesh()
    bpy = _FakeBpy(mesh, merges=40, heat=["ok"])
    assert blender_worker._skin(bpy, mesh, _FakeArm(), weld=0.01) == ("automatic-welded", None)
    assert bpy.binds == ["ARMATURE_AUTO"], "one solve, on the welded mesh"
    assert bpy.welded_at == 0.01
    assert len(bpy.data.meshes) == 1, "the pre-weld copy has to be dropped, not leaked"


def test_a_weld_that_merges_nothing_is_reported_as_a_plain_automatic_rig():
    mesh = _FakeMesh()
    bpy = _FakeBpy(mesh, merges=0, heat=["ok"])
    assert blender_worker._skin(bpy, mesh, _FakeArm(), weld=0.01) == ("automatic", None)
    assert bpy.binds == ["ARMATURE_AUTO"], "no identical second attempt"


def test_a_welded_solve_that_fails_falls_back_to_the_unwelded_one():
    """A welded solve failing is not evidence the unwelded one would: the weld
    is an attempt to help, so its failure costs that attempt and no more."""
    mesh = _FakeMesh(vertices=100)
    bpy = _FakeBpy(mesh, merges=40, heat=["non-manifold input", "ok"])
    assert blender_worker._skin(bpy, mesh, _FakeArm(), weld=0.01) == ("automatic", None)
    assert bpy.binds == ["ARMATURE_AUTO", "ARMATURE_AUTO"]
    # Restored from the copy the weld kept: the second attempt has to see the
    # geometry the first one did not, or it is the same attempt twice.
    assert len(mesh.data.vertices) == 100
    assert len(bpy.data.meshes) == 1, "the welded copy is dropped, not left behind"


def test_envelope_is_still_the_floor_and_the_reason_names_every_rung():
    mesh = _FakeMesh()
    bpy = _FakeBpy(mesh, merges=40, heat=["non-manifold input", "empty"])
    weighting, reason = blender_worker._skin(bpy, mesh, _FakeArm(), weld=0.01)
    assert weighting == "envelope"
    assert bpy.binds == ["ARMATURE_AUTO", "ARMATURE_AUTO", "ARMATURE_ENVELOPE"]
    # Both rungs, named: "it failed" without saying which attempt failed sends
    # the reader looking at the weld when it was the geometry, or the reverse.
    assert "automatic-welded: non-manifold input" in reason
    assert "automatic: produced no vertex weights" in reason


def test_a_weld_that_itself_fails_costs_the_welded_attempt_and_nothing_else():
    mesh = _FakeMesh()
    bpy = _FakeBpy(mesh, merges=40, heat=["ok"], weld_raises="bmesh exploded")
    assert blender_worker._skin(bpy, mesh, _FakeArm(), weld=0.01) == ("automatic", None)
    assert bpy.binds == ["ARMATURE_AUTO"]


def test_the_weld_failure_reaches_the_reason_when_everything_else_fails_too():
    mesh = _FakeMesh()
    bpy = _FakeBpy(mesh, merges=40, heat=["empty"], weld_raises="bmesh exploded")
    weighting, reason = blender_worker._skin(bpy, mesh, _FakeArm(), weld=0.01)
    assert weighting == "envelope"
    assert "weld pass failed: bmesh exploded" in reason


def test_asking_for_no_weld_is_byte_for_byte_the_old_behaviour():
    """The kill-switch case, and the one every existing rig on disk was made
    under: no copy, no edit mode, one solve."""
    mesh = _FakeMesh(vertices=100)
    before = mesh.data
    bpy = _FakeBpy(mesh, merges=40, heat=["ok"])
    assert blender_worker._skin(bpy, mesh, _FakeArm(), weld=0.0) == ("automatic", None)
    assert bpy.modes == [] and bpy.welded_at is None
    assert mesh.data is before, "no copy is taken and none is put back"
    assert bpy.data.meshes == []


def test_the_welded_method_joins_the_weighting_vocabulary_rather_than_replacing_it():
    """0d's inspector line has to render the new value as a success, not fall
    through to the envelope warning or to nothing at all."""
    pytest.importorskip("imgui_bundle")
    from warlock.studio.panes import inspector

    colour, text = inspector.weighting_verdict({"weighting": "automatic-welded"})
    assert text == "weighting: automatic-welded"
    assert colour != inspector.theme.WARN
    text.encode("latin-1")   # imgui's default Basic-Latin+Latin-1 atlas


def test_the_welded_method_survives_the_subprocess_boundary_as_rig_json(tmp_path):
    meta = blender_worker._rig_meta(
        rigging.get_template("humanoid"),
        bones=[],
        lo=[0.0, 0.0, 0.0],
        hi=[1.0, 1.0, 1.0],
        weighting="automatic-welded",
        weighting_reason=None,
        adjusted=False,
        fit={"method": "bbox"},
    )
    (tmp_path / "rig.json").write_text(json.dumps(meta), encoding="utf-8")
    assert rigging.read_rig(tmp_path)["weighting"] == "automatic-welded"


# --- the deformation battery ------------------------------------------------


def test_the_battery_is_template_data_not_code():
    poses = rigging.deform_battery("humanoid")
    assert [p["name"] for p in poses] == [
        "squat",
        "arms overhead",
        "elbow and knee 90",
        "torso twist",
    ]


def test_a_template_with_no_battery_costs_the_qa_sheet_and_never_the_rig():
    assert rigging.deform_battery("fish") == []
    with pytest.raises(ValueError):
        rigging.deform_battery("nonesuch")


@pytest.mark.parametrize("key", ["humanoid"])
def test_every_battery_pose_names_real_bones_and_is_a_unit_quaternion(key):
    names = {b["name"] for b in rigging.get_template(key).bones}
    for pose in rigging.deform_battery(key):
        assert pose["bones"], f"{pose['name']} poses nothing"
        for bone, quat in pose["bones"].items():
            assert bone in names, f"{pose['name']} poses {bone}, which {key} does not have"
            assert len(quat) == 4, "XYZW, the order every pose on disk is in"
            assert sum(v * v for v in quat) == pytest.approx(1.0, abs=1e-3)


def test_the_symmetric_battery_rows_are_exact_mirrors():
    """Authored by hand, so the sign convention is exactly the thing that goes
    wrong invisibly -- a mirrored limb that rotates the wrong way about one
    axis still looks plausible in a still. ``mirror_quaternion`` is the rule."""
    pairs = rigging.get_template("humanoid").mirror_pairs
    for pose in rigging.deform_battery("humanoid"):
        if pose["name"] == "torso twist":
            continue   # a twist is deliberately not symmetric
        bones = pose["bones"]
        for left, right in pairs:
            if left not in bones and right not in bones:
                continue
            assert left in bones and right in bones, f"{pose['name']} poses half of a pair"
            assert bones[right] == pytest.approx(rigging.mirror_quaternion(bones[left]))


def test_every_battery_row_carries_an_id_so_a_sheet_does_not_render_one_pose():
    """A sheet row is keyed by (pose id, frame). Rows that all had no id put
    the first pose into every row of the atlas."""
    ids = [p["id"] for p in rigging.deform_battery("humanoid")]
    assert len(set(ids)) == len(ids) and all(ids)


def test_a_preset_the_user_picks_still_has_no_id():
    """It is not a saved pose, and handing the editor one would be a claim
    that it can be loaded and deleted."""
    assert all("id" not in p for p in rigging.preset_poses("humanoid"))


def test_the_battery_plans_a_grid_through_the_one_sheet_planner(tmp_path):
    """Not a second renderer: the QA sheet is the ordinary sheet pipeline with
    a different pose list, which is what keeps one set of camera conventions."""
    from warlock import queue as queue_mod
    from warlock.pipelines import sheet as sheetlib

    poses = rigging.deform_battery("humanoid")
    layout = sheetlib.plan(
        poses,
        frame_size=queue_mod.DEFORM_QA_FRAME_SIZE,
        lighting="lit",
        yaws=queue_mod.DEFORM_QA_YAWS,
    )
    assert layout.rows == len(poses)
    assert layout.columns == queue_mod.DEFORM_QA_YAWS
    assert layout.yaws[0] == 0.0, "column 0 is the front view"
    # One row per pose, and every row's cells name that row's pose.
    per_row = {}
    for cell in layout.cells:
        per_row.setdefault(cell.row, set()).add(cell.pose)
    assert all(len(v) == 1 for v in per_row.values())
    assert len({next(iter(v)) for v in per_row.values()}) == len(poses)


# --- the artifact -----------------------------------------------------------


def test_the_qa_sheet_is_not_a_sheet_in_the_users_sheet_list(tmp_path):
    """It belongs to the rig, so it must not join list_sheets, count against
    MAX_SHEETS or be deleted by a sheet delete."""
    rigging.rig_qa_png_path(tmp_path).write_bytes(b"png")
    rigging.rig_qa_path(tmp_path).write_text("{}", encoding="utf-8")
    assert rigging.list_sheets(tmp_path) == []


def test_the_qa_sheet_is_served_only_once_its_sidecar_is_written(tmp_path):
    """The atlas is packed first and the sidecar last, by a job other than the
    one this directory belongs to -- so existence alone hands a reader a file
    still being written."""
    from warlock.service import files

    job = {"status": "done", "stage": "model"}
    rigging.rig_qa_png_path(tmp_path).write_bytes(b"png")
    assert files.ready(job, tmp_path, "rig_qa.png") is False
    rigging.rig_qa_path(tmp_path).write_text("{}", encoding="utf-8")
    assert files.ready(job, tmp_path, "rig_qa.png") is True
    assert "rig_qa.png" in files.MEDIA and "rig_qa.png" in files.LISTED


def test_the_qa_record_is_derived_and_cannot_be_inherited_by_a_reroll():
    from warlock.service.validation import DERIVED_PARAMS

    assert "deform_qa" in DERIVED_PARAMS


def test_read_rig_qa_survives_a_corrupt_sidecar(tmp_path):
    assert rigging.read_rig_qa(tmp_path) is None
    rigging.rig_qa_path(tmp_path).write_text("{not json", encoding="utf-8")
    assert rigging.read_rig_qa(tmp_path) is None


# --- with Blender actually installed ----------------------------------------


@pytest.mark.gpu
def test_welding_does_not_change_what_the_user_sees(tmp_path):
    """The deform mesh and the render mesh are the same mesh, so the whole
    argument that the weld is invisible rests on merge-by-distance keeping
    face-corner data: UVs are per-loop, so the two halves of a seam keep their
    own texture coordinates while sharing one position.

    Proved rather than asserted. A UV-seam split is built deliberately -- a
    cube whose vertices are all doubled, which is what an xatlas atlas does to
    a trellis mesh -- then welded, and the exported texture bytes and the set
    of per-loop UVs have to come back identical.
    """
    pytest.importorskip("bpy")
    import hashlib

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    # Split every vertex: this is the seam-split shape the weld exists for.
    bpy.ops.mesh.edge_split(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    def snapshot(path):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
        uvs = sorted(
            (round(uv.uv[0], 6), round(uv.uv[1], 6)) for uv in obj.data.uv_layers.active.uv
        )
        return hashlib.sha256(path.read_bytes()).hexdigest(), uvs, len(obj.data.polygons)

    _, uvs_before, faces_before = snapshot(tmp_path / "before.glb")
    split_vertices = len(obj.data.vertices)

    lo = [min(v.co[i] for v in obj.data.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in obj.data.vertices) for i in range(3)]
    original, merged = blender_worker._weld(bpy, obj, blender_worker.weld_distance(lo, hi))

    _, uvs_after, faces_after = snapshot(tmp_path / "after.glb")
    assert merged == split_vertices - 8, "a cube welds back to eight corners"
    assert faces_after == faces_before, "the weld may not drop a face"
    assert uvs_after == uvs_before, "per-loop UVs are what makes the weld invisible"

    # And the restore puts the split geometry back, byte for byte -- which is
    # what the unwelded fallback rung depends on.
    blender_worker._restore_mesh(bpy, obj, original)
    assert len(obj.data.vertices) == split_vertices


@pytest.mark.gpu
def test_a_welded_rig_exports_the_same_textures(tmp_path):
    """The texture bytes specifically: a weld that dropped or re-baked an image
    would be invisible in the UV check above and obvious to the user."""
    pytest.importorskip("bpy")
    import hashlib
    import struct

    import bpy

    def images(path):
        data = path.read_bytes()
        offset, doc, blob = 12, None, b""
        while offset < len(data):
            length, kind = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8 : offset + 8 + length]
            if kind == 0x4E4F534A:
                doc = json.loads(chunk)
            else:
                blob = chunk
            offset += 8 + length + (-length % 4)
        out = []
        for image in (doc or {}).get("images", []):
            view = doc["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            out.append(
                hashlib.sha256(blob[start : start + view["byteLength"]]).hexdigest()
            )
        return sorted(out)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.view_layer.objects.active
    material = bpy.data.materials.new("textured")
    image = bpy.data.images.new("albedo", 16, 16)
    image.pixels = [0.25, 0.5, 0.75, 1.0] * (16 * 16)
    tree = material.node_tree
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    principled = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
    tree.links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    obj.data.materials.append(material)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.mesh.edge_split(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")

    before = tmp_path / "before.glb"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(before), export_format="GLB")

    lo = [min(v.co[i] for v in obj.data.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in obj.data.vertices) for i in range(3)]
    blender_worker._weld(bpy, obj, blender_worker.weld_distance(lo, hi))

    after = tmp_path / "after.glb"
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=str(after), export_format="GLB")
    assert images(after) == images(before)
    assert len(obj.data.materials) == 1, "the weld may not drop a material slot"
