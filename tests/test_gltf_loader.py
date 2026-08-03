"""The GLB loader, against a hand-built asset and (when present) a real one.

The hand-built one is the pin: it is the only way to assert what a skin
*should* decode to without a 30 MB fixture in the repo. The real-asset tests
run only in the main checkout, where assets/ exists -- they are what catch a
loader that is self-consistently wrong.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from warlock.glbio import rebuild_glb
from warlock.studio.viewer import gltf
from warlock.studio.viewer import math3d as m3

REAL_MESH = Path("assets/2b058522940a/model.glb")
REAL_RIG = Path("assets/2b058522940a/rig.glb")
needs_real = pytest.mark.skipif(
    not REAL_RIG.exists(), reason="needs the main checkout's assets/ dir"
)


def _glb(gltf_json: dict, binary: bytes) -> bytes:
    """Wrap a JSON chunk and a BIN chunk as a GLB, the way an exporter would."""
    binary += b"\x00" * (-len(binary) % 4)
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    return rebuild_glb(header, gltf_json, chunk)


@pytest.fixture
def skinned_glb(tmp_path):
    """A two-triangle strip bound to two joints, one above the other.

    Small enough to assert on by hand: vertex 0 belongs entirely to the root
    joint, vertex 2 entirely to the tip, so rotating the tip must move exactly
    one of them.
    """
    positions = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype="<f4"
    )
    normals = np.tile(np.array([0, 0, 1], dtype="<f4"), (4, 1))
    uvs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype="<f4")
    joints = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype="<u1")
    weights = np.tile(np.array([1, 0, 0, 0], dtype="<f4"), (4, 1))
    indices = np.array([0, 1, 2, 1, 3, 2], dtype="<u4")
    # Column-major, as glTF stores them: the tip joint sits at y=1, so its
    # inverse bind matrix translates by -1.
    ibm = np.array(
        [np.eye(4, dtype="<f4").T, m3.translation(m3.vec3(0, -1, 0)).T.astype("<f4")]
    )

    blobs = [
        positions.tobytes(),
        normals.tobytes(),
        uvs.tobytes(),
        joints.tobytes(),
        weights.tobytes(),
        indices.tobytes(),
        ibm.tobytes(),
    ]
    views, binary = [], b""
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob + b"\x00" * (-len(blob) % 4)

    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": [
            {"name": "mesh_node", "mesh": 0, "skin": 0},
            {"name": "root", "children": [2]},
            {"name": "tip", "translation": [0, 1, 0]},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TEXCOORD_0": 2,
                            "JOINTS_0": 3,
                            "WEIGHTS_0": 4,
                        },
                        "indices": 5,
                        "material": 0,
                    }
                ]
            }
        ],
        "skins": [{"joints": [1, 2], "inverseBindMatrices": 6}],
        "materials": [
            {
                "name": "test",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.8, 0.2, 0.1, 1.0],
                    "metallicFactor": 0.25,
                    "roughnessFactor": 0.75,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5121, "count": 4, "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC4"},
            {"bufferView": 5, "componentType": 5125, "count": 6, "type": "SCALAR"},
            {"bufferView": 6, "componentType": 5126, "count": 2, "type": "MAT4"},
        ],
    }
    path = tmp_path / "skinned.glb"
    path.write_bytes(_glb(doc, binary))
    return path


# --- geometry ---------------------------------------------------------------


def test_a_plain_mesh_loads_with_its_indices_and_normals(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(path)
    model = gltf.load(path)

    (_, prims) = model.mesh_instances()[0]
    prim = prims[0]
    assert prim.positions.shape[1] == 3
    assert prim.positions.dtype == np.float32
    assert len(prim.indices) == model.triangle_count * 3
    assert prim.indices.dtype == np.uint32


def test_a_root_transform_is_not_discarded(tmp_path):
    """The whole reason this loader exists rather than trimesh's: normalize_glb
    puts the grounding transform on a node, and trimesh drops it."""
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
    from warlock.pipelines.postprocess import normalize_glb

    normalize_glb(path, 4.0)
    lo, hi = gltf.load(path).bounds()
    assert (hi - lo).max() == pytest.approx(4.0, rel=1e-4)
    # Grounded: min-Y sits at zero on every job, size or no size.
    assert lo[1] == pytest.approx(0.0, abs=1e-6)


def test_a_matrix_node_decomposes_to_the_same_transform(tmp_path):
    """A node gives either TRS or a matrix; both have to land in the same place."""
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "n", "matrix": list(m3.translation(m3.vec3(1, 2, 3)).T.flatten())}],
    }
    path = tmp_path / "m.glb"
    path.write_bytes(_glb(doc, b""))
    node = gltf.load(path).nodes[0]
    assert node.translation == pytest.approx([1, 2, 3])
    assert node.world[:3, 3] == pytest.approx([1, 2, 3])


# --- materials --------------------------------------------------------------


def test_material_factors_survive_the_load(skinned_glb):
    prim = gltf.load(skinned_glb).meshes[0][0]
    assert prim.material.base_color_factor == pytest.approx((0.8, 0.2, 0.1, 1.0))
    assert prim.material.metallic_factor == 0.25
    assert prim.material.roughness_factor == 0.75
    assert prim.material.double_sided is True


# --- skins ------------------------------------------------------------------


def test_joint_indices_are_promoted_to_int32(skinned_glb):
    """JOINTS_0 arrives as uint8; GLSL 3.30 indexes a palette with an ivec4."""
    prim = gltf.load(skinned_glb).meshes[0][0]
    assert prim.joints.dtype == np.int32
    assert prim.weights.dtype == np.float32
    assert prim.weights.sum(axis=1) == pytest.approx(np.ones(4))


def test_the_rest_palette_is_the_identity(skinned_glb):
    """world(joint) @ inverseBind(joint) cancels at rest, by construction --
    which is what makes a rest-pose render look like the unskinned mesh."""
    model = gltf.load(skinned_glb)
    node = model.mesh_instances()[0][0]
    for mat in model.joint_palette(node):
        assert mat == pytest.approx(np.eye(4), abs=1e-6)


def test_rotating_a_joint_moves_only_its_own_vertices(skinned_glb):
    model = gltf.load(skinned_glb)
    node = model.mesh_instances()[0][0]
    assert model.set_rotation("tip", m3.quat_from_axis_angle(m3.vec3(0, 0, 1), np.pi / 2))
    model.update_world()
    root_mat, tip_mat = model.joint_palette(node)

    assert root_mat == pytest.approx(np.eye(4), abs=1e-6)
    # The tip's own origin is its pivot, so a vertex there does not move...
    assert (tip_mat @ np.array([0.0, 1.0, 0.0, 1.0]))[:3] == pytest.approx([0, 1, 0], abs=1e-6)
    # ...while one beside it swings a quarter turn about z.
    assert (tip_mat @ np.array([1.0, 1.0, 0.0, 1.0]))[:3] == pytest.approx([0, 2, 0], abs=1e-6)


def test_an_unknown_bone_is_reported_not_ignored(skinned_glb):
    model = gltf.load(skinned_glb)
    assert model.set_rotation("no_such_bone", [0, 0, 0, 1]) is False


def test_a_pose_reports_only_what_moved(skinned_glb):
    model = gltf.load(skinned_glb)
    assert model.pose() == {}
    q = m3.quat_from_axis_angle(m3.vec3(1, 0, 0), 0.3)
    model.set_rotation("tip", q)
    assert list(model.pose()) == ["tip"]
    assert model.pose()["tip"] == pytest.approx(list(q))
    model.reset_all()
    assert model.pose() == {}


def test_a_pose_round_trips_as_xyzw(skinned_glb):
    """The wire order is three.js's, which is the order the Blender worker
    converts from -- not the order it converts to."""
    model = gltf.load(skinned_glb)
    q = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 1.1)
    model.set_rotation("tip", q)
    assert model.get_rotation("tip")[3] == pytest.approx(np.cos(0.55))


# --- refusals ---------------------------------------------------------------


def test_a_primitive_with_no_positions_is_refused(tmp_path):
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {}}]}],
    }
    path = tmp_path / "empty.glb"
    path.write_bytes(_glb(doc, b""))
    with pytest.raises(ValueError, match="POSITION"):
        gltf.load(path)


def test_a_line_primitive_is_refused_rather_than_drawn_as_triangles(tmp_path):
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "mode": 1}]}],
        "buffers": [{"byteLength": 12}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3"}],
    }
    path = tmp_path / "lines.glb"
    path.write_bytes(_glb(doc, b"\x00" * 12))
    with pytest.raises(ValueError, match="mode"):
        gltf.load(path)


def test_not_a_glb_is_refused_by_the_container_reader(tmp_path):
    path = tmp_path / "bad.glb"
    path.write_bytes(b"this is not a glb at all, really")
    with pytest.raises(ValueError, match="not a GLB"):
        gltf.load(path)


# --- the real thing ---------------------------------------------------------


@needs_real
def test_the_real_rig_has_twenty_joints_including_the_synthetic_one():
    model = gltf.load(REAL_RIG)
    node = model.mesh_instances()[0][0]
    assert len(model.joint_palette(node)) == 20
    assert model.skins[0].inverse_bind.shape == (20, 4, 4)
    # Blender's exporter invents neutral_bone for vertices with no group; it
    # is a real joint in the palette and dropping it shifts every index.
    assert "neutral_bone" in model.by_name


@needs_real
def test_the_real_rig_is_skinned_with_normalized_weights():
    prim = gltf.load(REAL_RIG).meshes[0][0]
    assert prim.joints.dtype == np.int32
    assert prim.joints.max() < 20
    assert prim.weights.sum(axis=1) == pytest.approx(np.ones(len(prim.weights)), abs=1e-5)


@needs_real
def test_the_real_mesh_is_grounded_and_carries_both_pbr_maps():
    model = gltf.load(REAL_MESH)
    lo, _hi = model.bounds()
    # normalize_glb puts min-Y at zero on every job.
    assert lo[1] == pytest.approx(0.0, abs=1e-4)
    material = model.meshes[0][0].material
    assert material.base_color is not None
    assert material.metallic_roughness is not None


@needs_real
def test_the_real_meshs_transform_is_on_a_child_not_the_root():
    """postprocess inserts the scale/translation below the root precisely so a
    loader that drops root transforms still gets it."""
    doc, _ = __import__("warlock.glbio", fromlist=["read_glb"]).read_glb(REAL_MESH)
    root = doc["nodes"][doc["scenes"][doc.get("scene", 0)]["nodes"][0]]
    assert "scale" not in root and "translation" not in root
    child = doc["nodes"][root["children"][0]]
    assert "scale" in child and "translation" in child


@needs_real
def test_loading_the_real_mesh_is_fast_enough_to_do_on_a_click():
    import time

    start = time.perf_counter()
    gltf.load(REAL_MESH)
    assert time.perf_counter() - start < 3.0


def test_json_only_glbs_are_still_valid(tmp_path):
    """A GLB with no BIN chunk is legal; the loader must not assume one."""
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": []}]}
    path = tmp_path / "empty.glb"
    path.write_bytes(rebuild_glb(struct.pack("<III", 0x46546C67, 2, 0), doc, b""))
    model = gltf.load(path)
    assert model.nodes == []
    assert json.loads("{}") == {}  # the doc really was JSON-only
