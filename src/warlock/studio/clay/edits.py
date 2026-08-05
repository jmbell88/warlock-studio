"""Clay's undo steps: what changed, and how to put it back.

The engine underneath these -- ``Edit``, ``CompoundEdit``, ``UndoStack``, the
serial counter and the byte budget -- is ``studio/undo.py``, shared with the
raster editor and with no opinion about what an edit edits. What is here is the
half that *is* about objects: a mesh swap, a transform, an object arriving or
leaving, a property change, a palette entry. It is the second consumer of that
engine, and the reason it was extracted out of the raster editor at all.

Two rules travel down from the engine unchanged, and both were learned the hard
way on layers before a mesh ever needed them.

**Every edit addresses its object by uid, never by index.** An index stops
naming the thing it named the moment anything moves, and the outliner can move
an object at any time -- including between the edit being recorded and the undo
being asked for. So a ``MeshEdit`` carries ``obj_uid`` and looks the object up,
and the two edits that genuinely *are* about a position in a list
(:class:`ObjectAddEdit`, :class:`ObjectRemoveEdit`, :class:`MaterialEdit`) hold
an index only for where to put something *back*, never for finding it: the add
and remove pair still delete by uid.

**An edit owns its data.** ``cost`` is what eviction is driven by, and a numpy
view reports its own small ``nbytes`` while pinning the whole base array alive.
``Mesh`` copies every array it is handed, so a mesh held here already owns its
bytes; the transform arrays are copied here for the same reason, cheap as they
are, because a caller that keeps drag-mutating the array it passed in would
otherwise rewrite history behind the stack's back.

A mesh edit is a *snapshot* pair rather than an inverse operation. That is
affordable because ``Mesh`` is immutable and every op is ``Mesh -> Mesh``: the
before mesh is the very object the document was holding, not a copy of it, so
the only cost is keeping it alive while it sits on the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from ..undo import Edit
from .mesh import Mesh


def mesh_bytes(mesh: Mesh) -> int:
    """Every array a mesh owns, added up -- the number eviction spends.

    Derived from the dataclass's fields rather than listed, so a Phase 2 field
    (UVs, creases) is counted the day it is added rather than the day someone
    notices the budget has stopped matching reality.
    """
    return int(sum(getattr(mesh, f.name).nbytes for f in fields(mesh)))


def _own(value: Any) -> Any:
    """A copy of an array, so nothing the caller still holds can rewrite it."""
    return np.array(value, dtype="f8", copy=True)


@dataclass
class MeshEdit(Edit):
    """One object's geometry, before and after.

    The cost is both meshes, because both are alive: the after mesh is what the
    document renders and the before mesh is what this step exists to restore.
    """

    obj_uid: int
    before: Mesh
    after: Mesh

    def __post_init__(self) -> None:
        self.cost = mesh_bytes(self.before) + mesh_bytes(self.after)

    def _put(self, doc: Any, mesh: Mesh) -> None:
        doc.by_uid(self.obj_uid).mesh = mesh
        doc.touch()

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)


@dataclass
class TransformEdit(Edit):
    """Translation, rotation and scale as one step, at near-zero cost.

    One step rather than three because a gizmo drag moves more than one of them
    at once and splitting that across three Ctrl+Z presses shows the user a
    pose the object was never in. Rotation is XYZW, as every quaternion in this
    project is.
    """

    obj_uid: int
    before: tuple[Any, Any, Any]
    after: tuple[Any, Any, Any]

    def __post_init__(self) -> None:
        self.before = tuple(_own(v) for v in self.before)  # type: ignore[assignment]
        self.after = tuple(_own(v) for v in self.after)  # type: ignore[assignment]
        self.cost = int(sum(v.nbytes for v in self.before + self.after))

    def _put(self, doc: Any, trs: tuple[Any, Any, Any]) -> None:
        obj = doc.by_uid(self.obj_uid)
        obj.translation, obj.rotation, obj.scale = (_own(v) for v in trs)
        doc.touch()

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)


@dataclass
class ObjectAddEdit(Edit):
    """The object itself is held, not a copy of it.

    Re-inserting the same object is what keeps its uid, and keeping its uid is
    what lets a mesh edit recorded before the delete still find its target
    after the undo. The index says only where to put it back.
    """

    index: int
    obj: Any

    def undo(self, doc: Any) -> None:
        doc.objects.pop(doc.index_of(self.obj.uid))
        doc.selection.discard(self.obj.uid)
        doc.touch()

    def redo(self, doc: Any) -> None:
        doc.objects.insert(self.index, self.obj)
        doc.touch()


@dataclass
class ObjectRemoveEdit(Edit):
    """The mirror of :class:`ObjectAddEdit`, and the one that costs bytes.

    While this step is on the stack the object is in no document, so the stack
    is the only thing keeping its mesh alive -- which is exactly the case the
    byte budget exists to bound.
    """

    index: int
    obj: Any

    def __post_init__(self) -> None:
        self.cost = mesh_bytes(self.obj.mesh)

    def undo(self, doc: Any) -> None:
        doc.objects.insert(self.index, self.obj)
        doc.touch()

    def redo(self, doc: Any) -> None:
        doc.objects.pop(doc.index_of(self.obj.uid))
        doc.selection.discard(self.obj.uid)
        doc.touch()


@dataclass
class ObjectPropsEdit(Edit):
    """Name, visibility, default material, generator and its parameters --
    anything about an object that is neither its geometry nor its transform."""

    obj_uid: int
    before: dict[str, Any]
    after: dict[str, Any]

    def _apply(self, doc: Any, props: dict[str, Any]) -> None:
        obj = doc.by_uid(self.obj_uid)
        for key, value in props.items():
            setattr(obj, key, value)
        doc.touch()

    def undo(self, doc: Any) -> None:
        self._apply(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._apply(doc, self.after)


@dataclass
class MaterialEdit(Edit):
    """One palette entry replaced.

    Addressed by index because a palette slot *is* an index: the per-face
    ``material`` array on every mesh in the document names it, and a uid would
    be a second name for the same thing that those arrays cannot carry. It puts
    back the entry object itself, so a primitive already holding that material
    is the same identity again and ``GpuMaterial``'s ``id(material)``
    de-duplication is undisturbed by the undo.
    """

    index: int
    before: Any
    after: Any

    def _put(self, doc: Any, material: Any) -> None:
        doc.materials[self.index] = material
        doc.touch()

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)
