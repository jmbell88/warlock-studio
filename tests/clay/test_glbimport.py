"""GLB into Clay, and the round trip back out again.

The round trip is the point. ``jobs.import_mesh`` mints a library asset from a
Clay document and this turns one back into a document, so an asset can be
exported, re-opened, edited and exported again -- and every one of those steps
has to preserve what the one before it wrote. The end-to-end test at the bottom
is the whole claim in one assertion chain.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import document as bd
from warlock.studio.clay import glbimport
from warlock.studio.clay import mesh as bm
from warlock.studio.clay import primitives as bp
from warlock.studio.clay.elements import OpError
from warlock.studio.viewer import glbwrite, gltf
from warlock.studio.viewer import math3d as m3


def _tex(seed: int = 0) -> tuple[int, int, bytes]:
    return (2, 2, bytes((seed + i) % 256 for i in range(16)))


def _uvd(mesh: bm.Mesh) -> bm.Mesh:
    n = len(mesh.loops)
    uv = np.stack(
        [np.arange(n, dtype="f4") / max(n, 1), np.full(n, 0.5, dtype="f4")], axis=1
    )
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=mesh.material,
        smooth=mesh.smooth,
        uv=uv,
    )


def _glb(doc: bd.ClayDoc) -> bytes:
    return glbwrite.write_glb(bd.to_model(doc))


def _one_box(**kw: object) -> bd.ClayDoc:
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box(), **kw))  # type: ignore[arg-type]
    return doc


# --- the merge --------------------------------------------------------------


def test_an_exported_cube_comes_back_with_eight_vertices_not_twenty_four() -> None:
    """The bitwise merge: an exporter's split copies are bit-identical, so
    ``np.unique`` over the raw rows puts them back exactly."""
    out = glbimport.glb_to_claydoc(_glb(_one_box()))
    assert len(out.objects) == 1
    mesh = out.objects[0].mesh
    bm.validate(mesh)
    assert len(mesh.positions) == 8
    assert bm.face_count(mesh) == 12, "quads arrive triangulated; glTF has only triangles"
    assert sorted(map(tuple, mesh.positions.tolist())) == sorted(
        map(tuple, bp.box().positions.tolist())
    )


def test_the_merge_is_exact_and_never_welds_a_near_miss() -> None:
    a = bp.box()
    nudged = np.array(a.positions)
    nudged[0] += 1e-5
    doc = bd.ClayDoc()
    doc.objects.append(
        bd.Obj(
            uid=bd.new_uid(),
            name="B",
            mesh=bm.Mesh(
                positions=nudged,
                loops=a.loops,
                starts=a.starts,
                material=a.material,
                smooth=a.smooth,
            ),
        )
    )
    out = glbimport.glb_to_claydoc(_glb(doc))
    assert len(out.objects[0].mesh.positions) == 8, "a hair apart is not the same vertex"


def test_per_corner_uvs_survive_the_merge() -> None:
    doc = bd.ClayDoc()
    doc.objects.append(bd.Obj(uid=bd.new_uid(), name="P", mesh=_uvd(bp.plane())))
    out = glbimport.glb_to_claydoc(_glb(doc))
    mesh = out.objects[0].mesh
    assert mesh.uv is not None
    assert len(mesh.uv) == len(mesh.loops)
    assert float(mesh.uv[:, 0].min()) >= 0.0 and float(mesh.uv[:, 0].max()) <= 1.0
    assert np.allclose(mesh.uv[:, 1], 0.5)


def test_a_flat_shaded_export_comes_back_flat_and_a_smooth_one_smooth() -> None:
    flat = glbimport.glb_to_claydoc(_glb(_one_box()))
    assert not flat.objects[0].mesh.smooth.any()

    a = bp.uv_sphere(segments=8, rings=4)
    smooth_doc = bd.ClayDoc()
    smooth_doc.objects.append(
        bd.Obj(
            uid=bd.new_uid(),
            name="S",
            mesh=bm.Mesh(
                positions=a.positions,
                loops=a.loops,
                starts=a.starts,
                material=a.material,
                smooth=np.ones(bm.face_count(a), dtype=bool),
            ),
        )
    )
    out = glbimport.glb_to_claydoc(_glb(smooth_doc))
    assert out.objects[0].mesh.smooth.any()


def test_a_mesh_with_no_normals_is_taken_as_smooth() -> None:
    prim = gltf.Primitive(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="f4"),
        indices=np.array([0, 1, 2], dtype="u4"),
        material=gltf.Material(name="m"),
    )
    model = gltf.Model([gltf.Node(name="n", mesh=0)], [0], [[prim]], [])
    out = glbimport.glb_to_claydoc(glbwrite.write_glb(model))
    assert out.objects[0].mesh.smooth.tolist() == [True]


# --- structure --------------------------------------------------------------


def test_one_object_per_primitive_with_names_deduped() -> None:
    doc = bd.ClayDoc(materials=[bd.default_material("a"), bd.default_material("b")])
    two_tone = bp.box()
    material = np.array([0, 0, 0, 1, 1, 1], dtype="i4")
    doc.objects.append(
        bd.Obj(
            uid=bd.new_uid(),
            name="Box",
            mesh=bm.Mesh(
                positions=two_tone.positions,
                loops=two_tone.loops,
                starts=two_tone.starts,
                material=material,
                smooth=two_tone.smooth,
            ),
        )
    )
    out = glbimport.glb_to_claydoc(_glb(doc))
    assert len(out.objects) == 2, "one primitive per material becomes one object each"
    assert [o.name for o in out.objects] == ["Box", "Box.001"]
    assert {o.mesh.material[0] for o in out.objects} == {0, 1}
    assert len(out.materials) == 2


def test_a_node_transform_is_decomposed_onto_the_object() -> None:
    doc = _one_box(
        translation=m3.vec3(1.0, 2.0, 3.0),
        scale=m3.vec3(2.0, 2.0, 2.0),
    )
    out = glbimport.glb_to_claydoc(_glb(doc))
    obj = out.objects[0]
    assert np.allclose(obj.translation, [1.0, 2.0, 3.0])
    assert np.allclose(obj.scale, [2.0, 2.0, 2.0])


def test_an_imported_object_is_frozen_and_the_document_is_clean() -> None:
    out = glbimport.glb_to_claydoc(_glb(_one_box()))
    assert out.objects[0].generator is None
    assert out.objects[0].params == {}
    assert len(out.history) == 0
    assert out.dirty is False
    assert out.selection == set()
    assert out.element_sel == {}


# --- refusals ---------------------------------------------------------------


def test_a_rigged_glb_is_refused_by_name() -> None:
    data = _glb(_one_box())
    model = gltf.load(data)
    model.skins = [object()]  # type: ignore[list-item]

    import warlock.studio.clay.glbimport as mod

    original = gltf.load
    try:
        mod.gltf.load = lambda _data: model  # type: ignore[assignment]
        with pytest.raises(OpError, match="rigged"):
            glbimport.glb_to_claydoc(data)
    finally:
        mod.gltf.load = original  # type: ignore[assignment]


def test_a_mesh_past_the_triangle_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glbimport, "MAX_TRIANGLES", 4)
    with pytest.raises(OpError, match="past the"):
        glbimport.glb_to_claydoc(_glb(_one_box()))


def test_a_glb_with_no_meshes_is_refused() -> None:
    model = gltf.Model([gltf.Node(name="empty")], [0], [], [])
    with pytest.raises(OpError, match="no meshes"):
        glbimport.glb_to_claydoc(glbwrite.write_glb(model))


def test_bytes_that_are_not_a_glb_are_refused_rather_than_raised_through() -> None:
    with pytest.raises(OpError, match="could not be read"):
        glbimport.glb_to_claydoc(b"not a glb at all")


# --- the round trip ---------------------------------------------------------


def test_an_authored_textured_model_survives_a_whole_round_trip() -> None:
    """Authored -> GLB -> Clay -> model -> GLB -> loaded, with everything equal.

    Positions, uvs, texture pixels and the world transform: the four things a
    round trip is capable of losing, each lost by a different half of the path.
    """
    image = _tex(3)
    doc = bd.ClayDoc(materials=[gltf.Material(name="baked", base_color=image)])
    doc.objects.append(
        bd.Obj(
            uid=bd.new_uid(),
            name="Panel",
            mesh=_uvd(bp.plane(size=(2.0, 2.0))),
            translation=m3.vec3(0.5, 1.5, -0.25),
        )
    )

    first = _glb(doc)
    imported = glbimport.glb_to_claydoc(first)
    second = glbwrite.write_glb(bd.to_model(imported))
    loaded = gltf.load(second)

    assert len(loaded.meshes) == 1
    prim = loaded.meshes[0][0]
    assert prim.uvs is not None
    assert prim.material.base_color == image, "the pixels came back byte for byte"
    assert np.allclose(loaded.nodes[0].translation, [0.5, 1.5, -0.25])

    # And the geometry survives: same bounds, same triangle count.
    original = gltf.load(first).meshes[0][0]
    assert len(prim.indices) == len(original.indices)
    assert np.allclose(prim.positions.min(axis=0), original.positions.min(axis=0))
    assert np.allclose(prim.positions.max(axis=0), original.positions.max(axis=0))


def test_a_palette_edit_keeps_the_baked_textures() -> None:
    """The properties panel replaces a material rather than rebuilding one.

    A fresh ``Material`` from the three factors the panel shows would silently
    drop the other five fields -- every texture slot an import carried in.
    """
    from dataclasses import replace

    material = gltf.Material(name="m", base_color=_tex(), normal=_tex(9))
    edited = replace(material, base_color_factor=(0.1, 0.2, 0.3, 1.0))
    assert edited.base_color is material.base_color
    assert edited.normal is material.normal
