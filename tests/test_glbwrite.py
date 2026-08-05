"""The GLB writer, verified by round-tripping through the loader that ships.

A writer tested against its own idea of the format is a writer that agrees with
itself. The strongest test available here is the loader in ``viewer/gltf.py``:
it is independently written, it is what every other GLB in this project is read
by, and it decodes accessors from the bytes rather than from the intent. So
almost everything here authors a ``gltf.Model``, writes it and loads it back.

The exceptions are the assertions about the *container* -- magic, chunk lengths,
alignment, the absence of an optional attribute -- which a round trip cannot
see, because the loader is tolerant of exactly the things a strict third-party
reader is not.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from warlock.glbio import read_glb
from warlock.studio.viewer import glbwrite, gltf
from warlock.studio.viewer import math3d as m3


def _quad(offset: float = 0.0, *, uvs: bool = False) -> gltf.Primitive:
    positions = np.array(
        [[0, 0, offset], [1, 0, offset], [0, 1, offset], [1, 1, offset]], dtype="f4"
    )
    return gltf.Primitive(
        positions=positions,
        indices=np.array([0, 1, 2, 1, 3, 2], dtype="u4"),
        normals=np.tile(np.array([0, 0, 1], dtype="f4"), (4, 1)),
        uvs=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype="f4") if uvs else None,
    )


def _materials() -> list[gltf.Material]:
    return [
        gltf.Material(
            name="red",
            base_color_factor=(0.9, 0.1, 0.2, 1.0),
            metallic_factor=0.25,
            roughness_factor=0.75,
        ),
        gltf.Material(
            name="glow",
            base_color_factor=(0.1, 0.1, 0.1, 1.0),
            metallic_factor=0.0,
            roughness_factor=0.5,
            emissive_factor=(0.4, 0.6, 0.8),
        ),
        gltf.Material(
            name="leaf",
            base_color_factor=(0.2, 0.7, 0.3, 0.5),
            metallic_factor=1.0,
            roughness_factor=0.1,
            double_sided=True,
        ),
    ]


def _model() -> tuple[gltf.Model, list[gltf.Material]]:
    """Two nodes with distinct TRS; the first carries two primitives."""
    mats = _materials()
    a, b, c = _quad(0.0), _quad(1.0), _quad(2.0)
    a.material, b.material, c.material = mats[0], mats[1], mats[2]
    nodes = [
        gltf.Node(
            name="first",
            translation=m3.vec3(1.0, 2.0, 3.0),
            rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.5),
            scale=m3.vec3(2.0, 0.5, 1.5),
            mesh=0,
        ),
        gltf.Node(
            name="second",
            translation=m3.vec3(-4.0, 0.0, 0.25),
            rotation=m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), -1.1),
            scale=m3.vec3(1.0, 1.0, 1.0),
            mesh=1,
        ),
    ]
    return gltf.Model(nodes, [0, 1], [[a, b], [c]], []), mats


def _roundtrip(model: gltf.Model) -> gltf.Model:
    return gltf.load(glbwrite.write_glb(model))


# --- geometry ----------------------------------------------------------------


def test_positions_normals_and_indices_survive_exactly() -> None:
    """Exact, not ``allclose``: everything here is already f4 on the way in, so
    any drift means a conversion happened that should not have."""
    model, _ = _model()
    out = _roundtrip(model)

    assert len(out.meshes) == 2
    assert [len(prims) for prims in out.meshes] == [2, 1]
    for before, after in zip(
        [p for prims in model.meshes for p in prims],
        [p for prims in out.meshes for p in prims],
        strict=True,
    ):
        assert np.array_equal(before.positions, after.positions)
        assert np.array_equal(before.normals, after.normals)
        assert np.array_equal(before.indices, after.indices)


def test_each_nodes_trs_survives() -> None:
    model, _ = _model()
    out = _roundtrip(model)

    assert [n.name for n in out.nodes] == ["first", "second"]
    assert out.roots == [0, 1]
    for before, after in zip(model.nodes, out.nodes, strict=True):
        assert np.allclose(before.translation, after.translation)
        assert np.allclose(before.rotation, after.rotation)  # XYZW
        assert np.allclose(before.scale, after.scale)
        assert before.mesh == after.mesh


def test_a_node_hierarchy_survives() -> None:
    parent = gltf.Node(name="parent", translation=m3.vec3(0.0, 5.0, 0.0), children=[1])
    child = gltf.Node(name="child", translation=m3.vec3(1.0, 0.0, 0.0), mesh=0)
    out = _roundtrip(gltf.Model([parent, child], [0], [[_quad()]], []))

    assert out.nodes[0].children == [1]
    assert out.roots == [0]
    # The child's world transform is its parent's composed with its own, which
    # is the property a flat re-parenting would quietly destroy.
    assert np.allclose(out.nodes[1].world[:3, 3], [1.0, 5.0, 0.0])


# --- materials ---------------------------------------------------------------


def test_every_material_factor_survives() -> None:
    model, mats = _model()
    out = _roundtrip(model)
    written = [p.material for prims in out.meshes for p in prims]

    for before, after in zip(mats, written, strict=True):
        assert after.name == before.name
        assert np.allclose(after.base_color_factor, before.base_color_factor)
        assert after.metallic_factor == pytest.approx(before.metallic_factor)
        assert after.roughness_factor == pytest.approx(before.roughness_factor)
        assert np.allclose(after.emissive_factor, before.emissive_factor)
        assert after.double_sided == before.double_sided


def test_the_primitive_to_material_assignment_survives() -> None:
    model, mats = _model()
    out = _roundtrip(model)
    assert [p.material.name for p in out.meshes[0]] == ["red", "glow"]
    assert [p.material.name for p in out.meshes[1]] == ["leaf"]
    assert [m.name for m in mats] == ["red", "glow", "leaf"]


def test_a_shared_material_is_written_once() -> None:
    """``to_model`` hands every primitive the palette entry itself, so identity
    de-duplication here is what keeps a twelve-object document at one material
    rather than twelve copies of one."""
    shared = gltf.Material(name="shared", base_color_factor=(0.3, 0.3, 0.3, 1.0))
    a, b = _quad(0.0), _quad(1.0)
    a.material = b.material = shared
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[a, b]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert len(doc["materials"]) == 1
    assert {p["material"] for p in doc["meshes"][0]["primitives"]} == {0}


def test_two_equal_but_distinct_materials_are_written_separately() -> None:
    """De-duplication is by identity, not by value -- the palette is a list of
    slots the user edits, and collapsing two that happen to match today would
    make editing one of them change the other."""
    a, b = _quad(0.0), _quad(1.0)
    a.material = gltf.Material(name="same")
    b.material = gltf.Material(name="same")
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[a, b]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert len(doc["materials"]) == 2


# --- optional attributes -----------------------------------------------------


def test_a_model_without_uvs_writes_no_texcoord_and_still_loads() -> None:
    """TEXCOORD_0 is optional in the spec, and Clay Phase 1 has no textures.
    Writing a zero-filled one would be four bytes a vertex of a lie."""
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[_quad(uvs=False)]], [])
    data = glbwrite.write_glb(model)

    doc, _ = read_glb(data)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert "POSITION" in attrs
    assert "TEXCOORD_0" not in attrs

    out = gltf.load(data)
    assert out.meshes[0][0].uvs is None


def test_uvs_survive_when_they_are_there() -> None:
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[_quad(uvs=True)]], [])
    out = _roundtrip(model)
    assert np.array_equal(out.meshes[0][0].uvs, _quad(uvs=True).uvs)


def test_a_primitive_without_normals_writes_no_normal_attribute() -> None:
    prim = _quad()
    prim.normals = None
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[prim]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert "NORMAL" not in doc["meshes"][0]["primitives"][0]["attributes"]
    assert gltf.load(glbwrite.write_glb(model)).meshes[0][0].normals is None


# --- the container -----------------------------------------------------------


def test_the_bytes_are_a_well_formed_glb() -> None:
    model, _ = _model()
    data = glbwrite.write_glb(model)

    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67  # "glTF"
    assert version == 2
    assert total == len(data)

    offset = 12
    seen = []
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        seen.append(kind)
        assert offset + 8 + length <= len(data)
        assert length % 4 == 0
        offset += 8 + length
    assert offset == len(data)
    assert seen == [0x4E4F534A, 0x004E4942]  # JSON then BIN, in that order


def test_the_json_chunk_is_padded_with_spaces_and_parses() -> None:
    """Spaces, not zeros -- the spec says so, and a strict reader that hands the
    chunk straight to a JSON parser chokes on a trailing NUL."""
    model, _ = _model()
    data = glbwrite.write_glb(model)
    length, _kind = struct.unpack_from("<II", data, 12)
    chunk = data[20 : 20 + length]

    assert chunk.rstrip(b" ") == chunk.rstrip()
    assert json.loads(chunk.decode())["asset"]["version"] == "2.0"


def test_every_buffer_view_is_four_byte_aligned() -> None:
    """An accessor is read with ``np.frombuffer`` at the view's offset, and an
    f4 read from an odd offset is a misaligned read on the platforms that still
    care and a silently wrong number on the ones that do not."""
    model, _ = _model()
    doc, binary = read_glb(glbwrite.write_glb(model))

    for view in doc["bufferViews"]:
        assert view.get("byteOffset", 0) % 4 == 0
        assert view.get("byteOffset", 0) + view["byteLength"] <= len(binary)
    assert doc["buffers"][0]["byteLength"] == len(binary)


def test_the_position_accessor_carries_min_and_max() -> None:
    """Required by the spec, and most viewers frame the scene from them: without
    them the model loads and sits somewhere unhelpful in the frame."""
    model, _ = _model()
    doc, _ = read_glb(glbwrite.write_glb(model))

    prim = doc["meshes"][0]["primitives"][0]
    accessor = doc["accessors"][prim["attributes"]["POSITION"]]
    assert accessor["min"] == [0.0, 0.0, 0.0]
    assert accessor["max"] == [1.0, 1.0, 0.0]
    # And only there: min/max on every accessor is legal but is noise.
    assert "min" not in doc["accessors"][prim["indices"]]


def test_a_small_mesh_writes_unsigned_short_indices() -> None:
    model, _ = _model()
    doc, _ = read_glb(glbwrite.write_glb(model))
    prim = doc["meshes"][0]["primitives"][0]
    assert doc["accessors"][prim["indices"]]["componentType"] == 5123  # USHORT


def test_a_mesh_past_65535_vertices_writes_unsigned_int_indices() -> None:
    """The componentType has to agree with the bytes actually written, and the
    boundary is the one place a writer can disagree with itself."""
    n = 70000
    prim = gltf.Primitive(
        positions=np.zeros((n, 3), dtype="f4"),
        indices=np.array([0, 1, n - 1], dtype="u4"),
    )
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[prim]], [])
    data = glbwrite.write_glb(model)

    doc, _ = read_glb(data)
    accessor = doc["accessors"][doc["meshes"][0]["primitives"][0]["indices"]]
    assert accessor["componentType"] == 5125  # UINT
    assert np.array_equal(gltf.load(data).meshes[0][0].indices, [0, 1, n - 1])


def test_an_empty_model_writes_no_buffer_at_all() -> None:
    """``byteLength`` has a minimum of one in the spec, so an empty buffer is
    not a degenerate file but an invalid one -- and a glTF with no buffer is
    perfectly legal. An empty scene must not be rejected for a reason that has
    nothing to do with what is in it."""
    data = glbwrite.write_glb(gltf.Model([], [], [], []))
    doc, binary = read_glb(data)
    assert "buffers" not in doc
    assert binary == b""

    out = gltf.load(data)
    assert out.nodes == []
    assert out.meshes == []


def test_a_primitive_with_no_geometry_is_written_without_accessors() -> None:
    """An object in the outliner with nothing in it yet is a state the document
    can be in, and it must not produce a zero-count accessor: the spec requires
    a count of at least one, so a strict reader rejects the whole file."""
    prim = gltf.Primitive(
        positions=np.zeros((0, 3), dtype="f4"), indices=np.zeros(0, dtype="u4")
    )
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[prim]], [])

    doc, _ = read_glb(glbwrite.write_glb(model))
    assert doc["meshes"][0]["primitives"] == []
    assert gltf.load(glbwrite.write_glb(model)).meshes == [[]]


def test_a_model_with_a_skin_is_refused_rather_than_silently_flattened() -> None:
    """Clay has no skins. Dropping one would export a rig-shaped file with
    no rig in it, which fails a long way from here."""
    model = gltf.Model([gltf.Node(name="n")], [0], [], [gltf.Skin([0], np.eye(4)[None])])
    with pytest.raises(ValueError, match="skin"):
        glbwrite.write_glb(model)
