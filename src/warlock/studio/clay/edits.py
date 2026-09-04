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
being asked for. So a ``MeshEdit`` carries ``obj_uid`` and looks the object up.
Three edits genuinely *are* about a position in a list, and they split two ways.
:class:`ObjectAddEdit` and :class:`ObjectRemoveEdit` hold an index only for
where to put an object *back*, never for finding it -- both still delete by uid.
:class:`MaterialEdit` and :class:`MaterialListEdit` are the exception that
proves the rule rather than a lapse from it: a palette slot **is** an index,
named as one by the per-face ``material`` array on every mesh in the document,
so there is no uid for one to be addressed by and ``_put`` finds its slot by
the only name it has.

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

    Derived from the dataclass's fields rather than listed, so a later field
    (creases, per-corner colours) is counted the day it is added rather than
    the day someone notices the budget has stopped matching reality. An
    optional field that is absent -- ``uv`` on an authored mesh -- is skipped
    rather than summed, which is the whole reason the sum is not a bare
    generator over the fields.
    """
    return int(
        sum(
            arr.nbytes
            for arr in (getattr(mesh, f.name) for f in fields(mesh))
            if arr is not None
        )
    )


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

    Costs bytes for the same reason its mirror does, just on the other side of
    the toggle: while this step is *undone* the object is in no document, and
    ``self.obj`` is the only thing keeping its mesh alive.
    """

    index: int
    obj: Any

    def __post_init__(self) -> None:
        self.cost = mesh_bytes(self.obj.mesh)

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
class ObjectMoveEdit(Edit):
    """One object's position in the list, before and after.

    The third of the three edits that genuinely are about a position -- and it
    still finds its object by uid, exactly as add and remove do. The indices say
    only where it goes; ``index_of`` says which object is being moved, so a
    reorder recorded before some other reorder still moves the right thing when
    it is undone.

    It costs nothing: two integers, and the object never leaves the document.
    """

    obj_uid: int
    before: int
    after: int

    def _put(self, doc: Any, index: int) -> None:
        obj = doc.objects.pop(doc.index_of(self.obj_uid))
        doc.objects.insert(index, obj)
        doc.touch()

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)


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


@dataclass
class MaterialListEdit(Edit):
    """A palette entry added or removed.

    A slot *is* an index -- every mesh's per-face ``material`` array names it --
    so adding or removing one in the middle would renumber the array in every
    object in the document. That is why the document only ever appends, and only
    ever removes a slot **no face uses**: an append renumbers nothing, and a
    removal shifts the indices above it down, which is a pure index arithmetic
    the undo reverses exactly by shifting them back up. No mesh pixels or
    positions are stored here at all, which is what keeps a palette edit's undo
    cost at one material object.
    """

    index: int
    material: Any
    added: bool

    def _insert(self, doc: Any) -> None:
        doc.materials.insert(self.index, self.material)
        _shift_materials(doc, self.index, +1)
        doc.touch()

    def _remove(self, doc: Any) -> None:
        del doc.materials[self.index]
        _shift_materials(doc, self.index, -1)
        doc.touch()

    def undo(self, doc: Any) -> None:
        self._remove(doc) if self.added else self._insert(doc)

    def redo(self, doc: Any) -> None:
        self._insert(doc) if self.added else self._remove(doc)


def _shift_materials(doc: Any, index: int, delta: int) -> None:
    """Renumber every reference to a palette slot at or above *index*.

    An object that names no slot that high is skipped, which is what makes the
    append case -- the common one -- cost nothing: there is nothing above the
    last slot to move, and rebuilding every mesh in the document to change
    nothing would throw away the whole GPU cache on every palette add.

    Three things about *what* gets renumbered, each of which was once missed.

    **Both references, not just the array.** A mesh's per-face ``material``
    array is the obvious one; ``Obj.material`` -- the slot a *new* face is given
    -- is the other, it is written to the file by ``serialize`` like any other
    field, and leaving it behind pointed an object's default past the end of a
    shortened palette.

    **In place, never through ``dataclasses.replace`` on the object.** The mesh
    is immutable and is replaced, as every mesh op replaces one; the ``Obj``
    around it is not. Swapping in a new ``Obj`` left the old one held by any
    :class:`ObjectAddEdit` or :class:`ObjectRemoveEdit` on the stack, so a redo
    re-inserted an object the document had since moved on from -- the identity
    that ``ObjectAddEdit``'s docstring says is the whole reason it holds the
    object rather than a copy.

    **Everything the history is holding, as well as everything in the
    document.** An undone add and a done remove both keep an object alive in no
    document at all, and a renumbering that walked ``doc.objects`` alone left
    those naming the palette as it stood before it. Deduplicated by identity,
    because an object that is both in the document and named by a step on the
    stack is one object and must shift once.
    """
    from dataclasses import replace as _replace

    import numpy as np

    def shift(mesh: Any) -> Any:
        material = mesh.material
        if not len(material) or int(material.max()) < index:
            return mesh
        return _replace(mesh, material=np.where(material >= index, material + delta, material))

    for obj in _material_holders(doc):
        obj.mesh = shift(obj.mesh)
        if int(obj.material) >= index:
            # The same threshold the face arrays use, and exactly reversible
            # for every default *above* the slot. A default that names the
            # removed slot itself is the one lossy case, and it is lossy
            # because the position it named stopped existing -- no arithmetic
            # can put that back. It is reachable from one place, the properties
            # panel's own Remove, which re-points that object's default in the
            # very next call and records it as an ``ObjectPropsEdit``; so the
            # undo restores it exactly, from the step that owns it rather than
            # from here. The clamp is the floor under the arithmetic for a slot
            # 0 removal, not a choice about which material to substitute.
            obj.material = max(0, int(obj.material) + delta)


def _material_holders(doc: Any) -> Any:
    """Every ``Obj`` this document's state space contains, each exactly once.

    The objects in the document, and the ones a step on the undo stack could put
    back into it -- an undone add, a done remove. Deduplicated by identity,
    which is not a tidiness measure in either caller: one object is routinely
    named by the document *and* by an add step still on the stack, and counting
    it twice overstates a slot's users while shifting it twice moves its faces
    two slots instead of one.
    """
    seen: set[int] = set()
    for obj in doc.objects:
        seen.add(id(obj))
        yield obj
    for edit in doc.history.edits():
        obj = getattr(edit, "obj", None)
        # ``hasattr(obj, "mesh")`` rather than an isinstance check, because the
        # shared history engine carries edits from three other packages and this
        # asks only whether the thing it holds has materials to renumber.
        if obj is not None and hasattr(obj, "mesh") and id(obj) not in seen:
            seen.add(id(obj))
            yield obj
