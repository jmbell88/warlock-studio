"""The Build document: objects addressed by uid, and the one glTF conversion.

Every test here pins a decision rather than an implementation. The uid rule is
the first of them and the one the whole history depends on: an undo issued
after a reorder must still land on the object the edit was made to.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from warlock.studio.build import document as bd
from warlock.studio.build import mesh as bm
from warlock.studio.build import primitives as bp


def _obj(name: str, mesh: bm.Mesh | None = None, **kwargs: object) -> bd.Obj:
    return bd.Obj(
        uid=bd.new_uid(),
        name=name,
        mesh=bp.box() if mesh is None else mesh,
        **kwargs,  # type: ignore[arg-type]
    )


def _last_edit(doc: bd.BuildDoc) -> Any:
    """The newest step on ``doc``'s history, for asserting how it is addressed."""
    return doc.history._done[-1]


def _recoloured(mesh: bm.Mesh, material: list[int]) -> bm.Mesh:
    """The same geometry with a per-face material assignment of our choosing."""
    out = bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops,
        starts=mesh.starts,
        material=np.asarray(material, dtype="i4"),
        smooth=mesh.smooth,
    )
    bm.validate(out)
    return out


# --- the uid rule ------------------------------------------------------------


def test_undo_lands_on_the_right_object_after_a_reorder() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B", bp.cylinder()))
    a_mesh, b_mesh = a.mesh, b.mesh

    doc.set_mesh(a.uid, bp.cone())
    assert doc.by_uid(a.uid).mesh is not a_mesh

    # The outliner moves A below B. Nothing about the recorded edit changes.
    doc.objects.reverse()
    assert [o.name for o in doc.objects] == ["B", "A"]

    assert doc.undo() is True
    assert doc.by_uid(a.uid).mesh is a_mesh
    assert doc.by_uid(b.uid).mesh is b_mesh


def test_undo_lands_on_the_right_object_after_a_delete_and_re_add() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    a_mesh = a.mesh
    doc.set_mesh(a.uid, bp.torus())

    doc.remove_object(a.uid)
    assert doc.objects == []
    # The remove edit holds the object itself, so undoing it puts *that* object
    # back -- same uid, and therefore still the address the mesh edit named.
    assert doc.undo() is True
    assert doc.by_uid(a.uid) is a

    assert doc.undo() is True
    assert doc.by_uid(a.uid).mesh is a_mesh


def test_an_object_edit_is_addressed_by_uid_not_by_index() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.add_object(_obj("B"))

    assert doc.set_transform(a.uid, translation=(1.0, 2.0, 3.0)) is True
    edit = _last_edit(doc)
    assert edit.obj_uid == a.uid
    assert not hasattr(edit, "index")

    assert doc.set_mesh(a.uid, bp.cone()) is True
    assert _last_edit(doc).obj_uid == a.uid
    assert doc.set_props(a.uid, name="renamed") is True
    assert _last_edit(doc).obj_uid == a.uid


# --- dirty, saving and no-ops ------------------------------------------------


def test_dirty_is_a_comparison_against_the_history_head() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.mark_saved()
    assert doc.dirty is False

    doc.set_transform(a.uid, translation=(0.0, 1.0, 0.0))
    assert doc.dirty is True

    # An undo is a change -- ``rev`` counts it -- but the document is back at
    # the step it was saved at, so it is clean again.
    doc.undo()
    assert doc.dirty is False


def test_a_no_op_transform_pushes_no_step() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A", translation=(1.0, 0.0, 0.0)))
    doc.mark_saved()

    depth = len(doc.history)
    assert doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0)) is False
    assert len(doc.history) == depth
    assert doc.dirty is False


def test_a_no_op_property_change_pushes_no_step() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.mark_saved()

    assert doc.set_props(a.uid, visible=True) is False
    assert doc.dirty is False
    assert doc.set_props(a.uid, visible=False) is True
    assert doc.dirty is True


def test_setting_a_mesh_to_the_one_already_there_pushes_no_step() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.mark_saved()

    assert doc.set_mesh(a.uid, a.mesh) is False
    assert doc.dirty is False


# --- selection ---------------------------------------------------------------


def test_selection_is_not_undoable() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    b = doc.add_object(_obj("B"))
    doc.set_transform(a.uid, translation=(5.0, 0.0, 0.0))

    doc.select([a.uid])
    doc.select([b.uid])
    head = doc.history.head

    # The undo reverses the last *edit*, and the selection is left alone.
    assert doc.undo() is True
    assert doc.selection == {b.uid}
    assert doc.history.head != head
    assert np.allclose(doc.by_uid(a.uid).translation, [0.0, 0.0, 0.0])


def test_selecting_does_not_move_the_history_head() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.mark_saved()
    doc.select([a.uid])
    assert doc.dirty is False


def test_removing_an_object_deselects_it() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.select([a.uid])
    doc.remove_object(a.uid)
    assert doc.selection == set()


# --- uids --------------------------------------------------------------------


def test_new_uid_never_reuses() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    first = a.uid
    doc.remove_object(a.uid)
    b = doc.add_object(_obj("B"))
    assert b.uid != first


def test_by_uid_raises_for_an_object_that_is_not_there() -> None:
    doc = bd.BuildDoc()
    with pytest.raises(KeyError):
        doc.by_uid(12345)


# --- the glTF conversion -----------------------------------------------------


def test_to_model_emits_one_node_per_object_carrying_its_trs() -> None:
    doc = bd.BuildDoc()
    doc.add_object(
        _obj(
            "A",
            translation=(1.0, 2.0, 3.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            scale=(2.0, 2.0, 2.0),
        )
    )
    doc.add_object(_obj("B", bp.cylinder()))

    model = bd.to_model(doc)
    assert [n.name for n in model.nodes] == ["A", "B"]
    assert model.roots == [0, 1]
    assert np.allclose(model.nodes[0].translation, [1.0, 2.0, 3.0])
    assert np.allclose(model.nodes[0].scale, [2.0, 2.0, 2.0])
    assert np.allclose(model.nodes[0].rotation, [0.0, 0.0, 0.0, 1.0])
    assert [n.mesh for n in model.nodes] == [0, 1]


def test_to_model_emits_one_primitive_per_object_and_material_group() -> None:
    doc = bd.BuildDoc(materials=[bd.default_material("a"), bd.default_material("b")])
    two_toned = _recoloured(bp.box(), [0, 0, 0, 1, 1, 1])
    doc.add_object(_obj("A", two_toned))

    model = bd.to_model(doc)
    assert len(model.nodes) == 1
    prims = model.meshes[0]
    assert len(prims) == 2
    assert [p.material.name for p in prims] == ["a", "b"]
    # Three quads a side, fanned into two triangles each.
    assert [len(p.indices) for p in prims] == [18, 18]
    for prim in prims:
        assert prim.positions.dtype == np.dtype("f4")
        assert prim.normals is not None
        assert len(prim.normals) == len(prim.positions)


def test_a_single_material_object_is_one_primitive() -> None:
    doc = bd.BuildDoc()
    doc.add_object(_obj("A"))
    prims = bd.to_model(doc).meshes[0]
    assert len(prims) == 1
    assert len(prims[0].indices) == 36


def test_a_hidden_object_is_not_in_the_model() -> None:
    doc = bd.BuildDoc()
    doc.add_object(_obj("A"))
    doc.add_object(_obj("B", visible=False))

    model = bd.to_model(doc)
    assert [n.name for n in model.nodes] == ["A"]
    assert len(model.meshes) == 1
    assert model.roots == [0]


def test_to_model_de_duplicates_materials_by_identity() -> None:
    palette = [bd.default_material("shared")]
    doc = bd.BuildDoc(materials=palette)
    doc.add_object(_obj("A"))
    doc.add_object(_obj("B", bp.cylinder()))

    model = bd.to_model(doc)
    prims = [p for mesh in model.meshes for p in mesh]
    assert len(prims) == 2
    # The very object out of the palette, so ``id(material)`` de-duplication in
    # GpuMaterial collapses the two draws onto one upload with no extra work.
    assert all(p.material is palette[0] for p in prims)


def test_a_material_index_past_the_palette_falls_back_rather_than_raising() -> None:
    doc = bd.BuildDoc(materials=[bd.default_material("only")])
    doc.add_object(_obj("A", _recoloured(bp.box(), [0, 0, 0, 7, 7, 7])))

    prims = bd.to_model(doc).meshes[0]
    assert len(prims) == 2
    assert prims[1].material is bd.FALLBACK_MATERIAL


def test_to_primitives_splits_a_mesh_without_touching_it() -> None:
    mesh = _recoloured(bp.box(), [0, 1, 0, 1, 0, 1])
    obj = _obj("A", mesh)
    prims = bd.to_primitives(obj, [bd.default_material("a"), bd.default_material("b")])
    assert len(prims) == 2
    assert obj.mesh is mesh
    # Every face is accounted for exactly once: six quads, twelve triangles.
    assert sum(len(p.indices) for p in prims) == 36


def test_an_empty_document_converts_to_an_empty_model() -> None:
    model = bd.to_model(bd.BuildDoc())
    assert model.nodes == []
    assert model.meshes == []
    assert model.skins == []


# --- palette edits -----------------------------------------------------------


def test_a_material_change_is_undoable_by_palette_index() -> None:
    doc = bd.BuildDoc()
    doc.add_object(_obj("A"))
    before = doc.materials[0]
    doc.mark_saved()

    changed = bd.default_material("red")
    changed.base_color_factor = (1.0, 0.0, 0.0, 1.0)
    assert doc.set_material(0, changed) is True
    assert doc.materials[0] is changed
    assert bd.to_model(doc).meshes[0][0].material is changed

    doc.undo()
    assert doc.materials[0] is before
    assert bd.to_model(doc).meshes[0][0].material is before
    assert doc.dirty is False


# --- objects own their arrays ------------------------------------------------


def test_an_objects_transform_arrays_are_owned_not_aliased() -> None:
    source = np.array([1.0, 2.0, 3.0])
    obj = _obj("A", translation=source)
    source[0] = 99.0
    assert obj.translation[0] == 1.0
    assert obj.translation.base is None


def test_a_transform_edit_restores_the_values_it_recorded() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.set_transform(a.uid, translation=(1.0, 0.0, 0.0), scale=(3.0, 3.0, 3.0))
    assert np.allclose(a.scale, [3.0, 3.0, 3.0])

    doc.undo()
    assert np.allclose(a.translation, [0.0, 0.0, 0.0])
    assert np.allclose(a.scale, [1.0, 1.0, 1.0])

    doc.redo()
    assert np.allclose(a.translation, [1.0, 0.0, 0.0])
    assert np.allclose(a.scale, [3.0, 3.0, 3.0])


def test_was_records_the_values_a_live_drag_started_from() -> None:
    """A gizmo mutates the object as the mouse moves and asks for the step on
    release, by which time the object no longer knows where it started."""
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    started_at = a.trs()

    a.translation = np.array([4.0, 0.0, 0.0])  # the drag, applied live
    assert doc.set_transform(a.uid, was=started_at) is True

    doc.undo()
    assert np.allclose(a.translation, [0.0, 0.0, 0.0])


def test_was_records_a_property_dict_that_was_edited_in_place() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A", generator="cylinder", params={"radius": 0.5}))
    started_at = {"params": dict(a.params)}

    a.params["radius"] = 1.0  # edited in place, so obj.params *is* the new value
    assert doc.set_props(a.uid, was=started_at, params=a.params) is True

    doc.undo()
    assert doc.by_uid(a.uid).params == {"radius": 0.5}


def test_a_mesh_edit_costs_both_meshes() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(_obj("A"))
    doc.set_mesh(a.uid, bp.uv_sphere())
    edit = doc.history._done[-1]
    assert edit.cost > int(a.mesh.positions.nbytes)


def test_the_generator_field_survives_a_regenerate() -> None:
    doc = bd.BuildDoc()
    a = doc.add_object(
        _obj("A", generator="cylinder", params={"radius": 0.5, "height": 1.0})
    )
    doc.set_props(a.uid, params={"radius": 1.0, "height": 1.0})
    doc.set_mesh(a.uid, bp.cylinder(radius=1.0))
    assert doc.by_uid(a.uid).generator == "cylinder"
    assert doc.by_uid(a.uid).params == {"radius": 1.0, "height": 1.0}

    doc.undo()
    doc.undo()
    assert doc.by_uid(a.uid).params == {"radius": 0.5, "height": 1.0}


def test_every_mesh_the_document_converts_is_valid() -> None:
    doc = bd.BuildDoc()
    for name, (defaults, build) in bp.GENERATORS.items():
        doc.add_object(_obj(name, build(**defaults), generator=name, params=defaults))
    model = bd.to_model(doc)
    assert len(model.nodes) == len(bp.GENERATORS)
    for prims in model.meshes:
        for prim in prims:
            assert len(prim.indices) % 3 == 0
