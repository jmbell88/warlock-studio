"""Removing a palette slot renumbers *every* reference to one, reversibly.

A slot is an index -- the per-face ``material`` array on every mesh names it,
and so does ``Obj.material``, the slot a new face is given -- so removing one
shifts everything above it down and the undo shifts it back. Three things were
missing from that and each is a different way for a document to come back wrong:

* ``Obj.material`` was never shifted at all, so an object's default silently
  moved onto a different material, or past the end of a shortened palette.
* the shift swapped the whole ``Obj`` through ``dataclasses.replace``, which
  left any ``ObjectAddEdit`` or ``ObjectRemoveEdit`` on the stack holding the
  *previous* object -- the identity those steps exist to preserve.
* objects held out of the document by a step (an undone add, a done remove)
  were not walked at all, so a slot they alone used counted zero users and the
  removal that permitted renumbered their faces onto another material.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from warlock.studio.clay import document as bd
from warlock.studio.clay import primitives as bp


def _obj(material: int = 0, faces: int | None = None) -> bd.Obj:
    """A box whose faces all point at one slot, and whose default does too."""
    mesh = bp.box()
    count = len(mesh.starts) - 1
    at = material if faces is None else faces
    mesh = replace(mesh, material=np.full(count, at, dtype=mesh.material.dtype))
    return bd.Obj(uid=bd.new_uid(), name="B", mesh=mesh, material=material)


def _palette(doc: bd.ClayDoc, count: int) -> None:
    while len(doc.materials) < count:
        doc.add_material()
    doc.history.clear()


# --- Obj.material ------------------------------------------------------------


def test_removing_a_slot_shifts_the_new_face_default_down():
    doc = bd.ClayDoc()
    _palette(doc, 4)
    obj = _obj(material=3)
    doc.objects.append(obj)

    assert doc.remove_material(1) is True
    assert obj.material == 2, "the default followed its material down a slot"
    assert int(obj.mesh.material.max()) == 2


def test_the_default_never_ends_up_past_the_end_of_the_palette():
    """The symptom the miss produced: a default naming a slot that is gone, and
    a new face rendering as the magenta fallback."""
    doc = bd.ClayDoc()
    _palette(doc, 3)
    obj = _obj(material=2)
    doc.objects.append(obj)

    assert doc.remove_material(0) is True
    assert 0 <= obj.material < len(doc.materials)


def test_the_default_comes_back_where_it_was_on_an_undo():
    doc = bd.ClayDoc()
    _palette(doc, 4)
    obj = _obj(material=3)
    doc.objects.append(obj)

    doc.remove_material(1)
    doc.undo()
    assert obj.material == 3
    assert int(obj.mesh.material.max()) == 3
    assert len(doc.materials) == 4


# --- object identity ---------------------------------------------------------


def test_the_shift_keeps_the_object_the_history_is_holding():
    """``ObjectAddEdit`` holds the object rather than a copy, which is what
    keeps its uid alive across an undo. Swapping in a replacement left the step
    holding a stale one, and a redo re-inserted it over the document's."""
    doc = bd.ClayDoc()
    _palette(doc, 3)
    obj = _obj(material=2)
    doc.add_object(obj)
    edit = doc.history.top

    doc.remove_material(0)
    assert doc.objects[0] is obj
    assert edit is not None
    assert edit.obj is obj  # type: ignore[attr-defined]


def test_an_add_undone_and_redone_across_a_palette_change_keeps_its_numbering():
    doc = bd.ClayDoc()
    _palette(doc, 4)
    doc.add_object(_obj(material=3))

    doc.remove_material(1)
    doc.undo()  # the removal
    doc.undo()  # the add
    assert not doc.objects
    doc.redo()  # the add
    doc.redo()  # the removal

    obj = doc.objects[0]
    assert int(obj.mesh.material.max()) == 2
    assert obj.material == 2
    assert 0 <= obj.material < len(doc.materials)


# --- objects the history is holding out of the document ----------------------


def test_a_slot_used_only_by_a_deleted_object_is_still_in_use():
    doc = bd.ClayDoc()
    _palette(doc, 3)
    obj = _obj(material=2)
    doc.add_object(obj)
    doc.remove_object(obj.uid)

    assert not doc.objects
    assert doc.material_users(2) == len(obj.mesh.starts) - 1
    assert doc.remove_material(2) is False


def test_a_slot_used_only_by_an_undone_add_is_still_in_use():
    doc = bd.ClayDoc()
    _palette(doc, 3)
    obj = _obj(material=2)
    doc.add_object(obj)
    doc.undo()

    assert not doc.objects
    assert doc.remove_material(2) is False


def test_a_deleted_objects_faces_shift_with_everyone_elses():
    """Deleting a slot *below* the one a held object uses is allowed, and the
    held object has to be renumbered with the rest -- or it comes back naming a
    material one slot along from the one it was drawn with."""
    doc = bd.ClayDoc()
    _palette(doc, 3)
    obj = _obj(material=2)
    doc.add_object(obj)
    doc.remove_object(obj.uid)

    assert doc.remove_material(1) is True
    assert int(obj.mesh.material.max()) == 1

    doc.undo()  # the palette removal
    doc.undo()  # the object removal
    assert doc.objects[0] is obj
    assert int(obj.mesh.material.max()) == 2
    assert len(doc.materials) == 3


def test_an_object_is_shifted_once_even_though_two_things_name_it():
    """It is in the document *and* named by the add step on the stack. The walk
    deduplicates by identity; without that it would shift twice and the faces
    would land two slots away."""
    doc = bd.ClayDoc()
    _palette(doc, 4)
    obj = _obj(material=3)
    doc.add_object(obj)

    doc.remove_material(0)
    assert int(obj.mesh.material.max()) == 2
    assert obj.material == 2


# --- and the append case, which must stay free -------------------------------


def test_appending_a_material_renumbers_nothing():
    doc = bd.ClayDoc()
    _palette(doc, 2)
    obj = _obj(material=1)
    doc.objects.append(obj)
    mesh = obj.mesh

    doc.add_material()
    assert obj.mesh is mesh, "no mesh is rebuilt, so no GPU buffer is thrown away"
    assert obj.material == 1
