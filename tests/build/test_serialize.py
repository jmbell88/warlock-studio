"""The ``.wblk`` document format: a zip of ``scene.json`` plus one npz an object.

The tests that matter are the ones about what a *reload* is allowed to lose.
Nothing, is the answer -- uids especially, because an undo step recorded before
a save names its object by uid and a reload that reissued them would silently
retarget the history. The rest pin the two ways a half-read file is worse than
a refused one: an unknown version, and a scene that names a mesh the archive
does not carry.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pytest

from warlock.studio.build import document as bd
from warlock.studio.build import primitives as bp
from warlock.studio.build import serialize as ser
from warlock.studio.viewer import gltf
from warlock.studio.viewer import math3d as m3


def _doc() -> bd.BuildDoc:
    """Three objects and a three-entry palette, with nothing at a default."""
    materials = [
        gltf.Material(name="red", base_color_factor=(0.9, 0.1, 0.2, 1.0)),
        gltf.Material(
            name="glow",
            base_color_factor=(0.1, 0.2, 0.3, 0.4),
            metallic_factor=0.25,
            roughness_factor=0.75,
            emissive_factor=(0.4, 0.5, 0.6),
            double_sided=True,
        ),
        gltf.Material(name="leaf", alpha_mode="MASK", alpha_cutoff=0.25),
    ]
    doc = bd.BuildDoc(materials=materials)
    doc.objects.append(
        bd.Obj(
            uid=bd.new_uid(),
            name="Cyl",
            mesh=bp.cylinder(radius=0.4, height=2.0, segments=12),
            translation=m3.vec3(1.0, 2.0, 3.0),
            rotation=m3.quat_from_axis_angle(m3.vec3(0.0, 1.0, 0.0), 0.7),
            scale=m3.vec3(2.0, 0.5, 1.5),
            generator="cylinder",
            params={"radius": 0.4, "height": 2.0, "segments": 12},
            visible=True,
            material=2,
        )
    )
    doc.objects.append(
        bd.Obj(uid=bd.new_uid(), name="Hidden", mesh=bp.box(), visible=False, material=1)
    )
    doc.objects.append(
        bd.Obj(uid=bd.new_uid(), name="Frozen", mesh=bp.cone(), generator=None)
    )
    return doc


def _roundtrip(doc: bd.BuildDoc) -> bd.BuildDoc:
    return ser.read_wblk(ser.wblk_bytes(doc))


# --- the round trip ----------------------------------------------------------


def test_every_object_field_survives_a_round_trip() -> None:
    doc = _doc()
    out = _roundtrip(doc)

    assert len(out.objects) == len(doc.objects)
    for before, after in zip(doc.objects, out.objects, strict=True):
        assert after.uid == before.uid
        assert after.name == before.name
        assert after.generator == before.generator
        assert after.params == before.params
        assert after.visible == before.visible
        assert after.material == before.material
        assert np.allclose(after.translation, before.translation)
        assert np.allclose(after.rotation, before.rotation)  # XYZW
        assert np.allclose(after.scale, before.scale)


def test_object_order_survives_because_it_is_what_the_outliner_shows() -> None:
    doc = _doc()
    assert [o.name for o in _roundtrip(doc).objects] == ["Cyl", "Hidden", "Frozen"]


def test_every_mesh_array_survives_exactly() -> None:
    """Exact, not ``allclose``: the arrays go to disk in their own dtypes, so
    any drift means a conversion happened on the way through."""
    doc = _doc()
    out = _roundtrip(doc)

    for before, after in zip(doc.objects, out.objects, strict=True):
        for name in ("positions", "loops", "starts", "material", "smooth"):
            mine, theirs = getattr(before.mesh, name), getattr(after.mesh, name)
            assert theirs.dtype == mine.dtype
            assert np.array_equal(theirs, mine)


def test_every_material_factor_survives() -> None:
    doc = _doc()
    out = _roundtrip(doc)

    assert len(out.materials) == 3
    for before, after in zip(doc.materials, out.materials, strict=True):
        assert after.name == before.name
        assert np.allclose(after.base_color_factor, before.base_color_factor)
        assert after.metallic_factor == pytest.approx(before.metallic_factor)
        assert after.roughness_factor == pytest.approx(before.roughness_factor)
        assert np.allclose(after.emissive_factor, before.emissive_factor)
        assert after.double_sided == before.double_sided
        assert after.alpha_mode == before.alpha_mode
        assert after.alpha_cutoff == pytest.approx(before.alpha_cutoff)


def test_a_restored_document_is_clean_with_an_empty_history() -> None:
    """A file just opened is not unsaved. Restoring through the document's own
    ``add_object`` would push a step per object and open every file dirty."""
    doc = _doc()
    doc.set_props(doc.objects[0].uid, name="edited")

    out = _roundtrip(doc)
    assert len(out.history) == 0
    assert out.history.can_undo is False
    assert out.dirty is False


def test_a_restored_document_has_nothing_selected() -> None:
    doc = _doc()
    doc.select([o.uid for o in doc.objects])
    assert _roundtrip(doc).selection == set()


def test_an_empty_scene_is_a_legal_thing_to_save() -> None:
    out = _roundtrip(bd.BuildDoc(materials=[]))
    assert out.objects == []
    assert out.materials == []
    assert out.dirty is False


def test_a_restored_object_owns_its_arrays() -> None:
    out = _roundtrip(_doc())
    assert out.objects[0].translation.base is None
    assert out.objects[0].mesh.positions.flags.writeable is False


# --- uids --------------------------------------------------------------------


def test_a_uid_minted_after_a_load_cannot_collide_with_a_restored_one() -> None:
    """The uid is the address every undo step is written against. Reading a
    document whose uids run past the process counter and then minting a fresh
    one would hand back a number an object in the outliner already wears."""
    doc = bd.BuildDoc()
    high = max(bd.new_uid() for _ in range(3)) + 10_000
    doc.objects.append(bd.Obj(uid=high, name="A", mesh=bp.box()))

    out = _roundtrip(doc)
    assert out.objects[0].uid == high
    assert bd.new_uid() > high


# --- refusals ----------------------------------------------------------------


def _rewrite(data: bytes, edit) -> bytes:
    """The same archive with ``edit`` applied to its ``scene.json`` dict."""
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for name in src.namelist():
            if name == ser.SCENE:
                scene = json.loads(src.read(name))
                edit(scene)
                dst.writestr(name, json.dumps(scene))
            else:
                dst.writestr(name, src.read(name))
    return out.getvalue()


def test_an_unknown_future_version_is_refused_rather_than_half_read() -> None:
    def bump(scene: dict) -> None:
        scene["version"] = ser.VERSION + 1

    with pytest.raises(ValueError, match="newer version"):
        ser.read_wblk(_rewrite(ser.wblk_bytes(_doc()), bump))


def test_a_scene_naming_a_mesh_the_archive_does_not_carry_is_refused() -> None:
    """Silently substituting an empty mesh would open the file, show an object
    with nothing in it, and let the user save that over their work."""
    data = ser.wblk_bytes(_doc())
    stripped = BytesIO()
    with zipfile.ZipFile(BytesIO(data)) as src, zipfile.ZipFile(stripped, "w") as dst:
        for name in src.namelist():
            if not name.endswith(".npz"):
                dst.writestr(name, src.read(name))

    with pytest.raises(ValueError, match="mesh"):
        ser.read_wblk(stripped.getvalue())


def test_bytes_that_are_not_a_zip_are_refused() -> None:
    with pytest.raises(ValueError, match="not a Warlock build document"):
        ser.read_wblk(b"this is not a zip file at all")


def test_a_zip_without_a_scene_is_refused() -> None:
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("something.txt", "hello")
    with pytest.raises(ValueError, match="not a Warlock build document"):
        ser.read_wblk(out.getvalue())


# --- the file itself ---------------------------------------------------------


def test_two_saves_of_an_unchanged_document_are_byte_identical() -> None:
    """Diffable, and reproducible. Every timestamp in the archive is fixed for
    this reason: a zip stamps each member with the wall clock by default, which
    would make a document that has not changed produce a different file every
    time it is written."""
    doc = _doc()
    assert ser.wblk_bytes(doc) == ser.wblk_bytes(doc)


def test_scene_json_is_sorted_so_the_file_is_diffable() -> None:
    with zipfile.ZipFile(BytesIO(ser.wblk_bytes(_doc()))) as zf:
        raw = zf.read(ser.SCENE).decode()
    scene = json.loads(raw)

    assert raw == json.dumps(scene, sort_keys=True, indent=2)
    assert scene["version"] == ser.VERSION
    assert [o["name"] for o in scene["objects"]] == ["Cyl", "Hidden", "Frozen"]


def test_the_archive_holds_one_mesh_per_object() -> None:
    doc = _doc()
    with zipfile.ZipFile(BytesIO(ser.wblk_bytes(doc))) as zf:
        names = set(zf.namelist())
    assert names == {ser.SCENE} | {f"meshes/{o.uid}.npz" for o in doc.objects}
