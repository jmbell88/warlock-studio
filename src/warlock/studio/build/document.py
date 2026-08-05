"""The Build document -- objects, a material palette, and the history over them.

A Build document is a flat list of objects. There is no hierarchy in Phase 1:
parenting is a feature with a cost (a transform that is not the one you typed,
an outliner that has to explain itself) and nothing in "place a few primitives
and export them" needs it, so ``Obj`` carries a TRS and no parent. The one
conversion out of here -- :func:`to_model` -- is what the viewport draws, what
the exporter writes and what the trellis render photographs; there is exactly
one of it, so those three can never disagree about what the document *is*.

**Materials are ``viewer.gltf.Material`` objects, not a new type.** They are
already pure data, they already carry ``base_color_factor`` /
``metallic_factor`` / ``roughness_factor`` / ``emissive_factor`` /
``double_sided``, and a parallel Build-only material type would have bought
nothing but a conversion function and a place for the two to drift. It also
pays off on the GPU for free: ``GpuMaterial`` de-duplicates by
``id(material)``, and :func:`to_primitives` hands every primitive the palette
entry *itself*, so a document whose twelve objects share one material uploads
one material. The texture slots stay ``None`` throughout -- Build mode paints
no textures, and a slot that is ``None`` is a slot the renderer skips.

**Rotation is XYZW**, matching ``viewer.math3d``, ``viewer.gltf``, the pose
files on disk and glTF itself. There is no other quaternion order anywhere in
this project and this is not the place to introduce one.

**``generator`` is the live-until-frozen field.** An object placed from the
primitive registry keeps the generator's name and the parameters it was built
with, so the properties panel shows "Cylinder: radius, height, segments" and a
change regenerates the mesh as one :class:`~.edits.MeshEdit`. Phase 2's first
topology edit will set ``generator = None`` and the panel switches to a vertex
and face count with a "frozen" note. **Phase 1 never freezes anything** -- but
the field exists from day one, so Phase 2 adds a line rather than a migration
of every document already saved.

**Dirty is a comparison against ``history.head``, not a flag.** ``rev`` counts
changes and an undo is a change, so a rev-based check calls an undone document
unsaved forever. :attr:`BuildDoc.saved_head` records the head at save time and
:attr:`BuildDoc.dirty` compares against it, which is the same rule the raster
editor's tabs follow and for the same reason.

**Selection is not undoable.** The raster editor makes a selection undoable
because a lasso around a character's hand is minutes of work that a stray click
destroys. Clicking an object in a 3D viewport is not that, and Blender's object
mode agrees -- Ctrl+Z there reverses the last *edit*, not the last click. It
matters beyond taste: an undoable selection would push a step, the step would
move ``history.head``, and a document would ask to be saved because the user
looked at a different object.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..undo import UndoStack
from ..viewer import gltf
from ..viewer import math3d as m3
from . import mesh as bm
from .edits import (
    MaterialEdit,
    MeshEdit,
    ObjectAddEdit,
    ObjectPropsEdit,
    ObjectRemoveEdit,
    TransformEdit,
)

_uids = itertools.count(1)


def new_uid() -> int:
    """Mint an object uid. Never reused, for the life of the process.

    The uid is the *address* every undo step is written against, so reuse is
    not a tidiness question: a recycled uid would let a step recorded against a
    deleted object land on a different one that happens to wear its number. A
    process-wide counter rather than a per-document one, so the same rule holds
    across a document that was closed and reopened in the same session.
    """
    return next(_uids)


def reserve_uid(uid: int) -> None:
    """Guarantee that :func:`new_uid` never hands back ``uid`` or anything below.

    A saved document carries its uids, and a reload that reissued them would
    silently retarget every undo step recorded against one. But the counter is
    per *process*, not per document, so a file whose uids run past where this
    session happens to have got to would otherwise collide the moment the user
    adds an object -- the new object and a restored one wearing one number, and
    the first edit to either landing on whichever ``index_of`` reaches first.
    So the reader raises the floor as it restores. Deliberately monotonic: the
    counter never moves backwards, so loading a small document after a large
    one cannot undo the protection the large one bought.
    """
    global _uids
    _uids = itertools.count(max(int(uid) + 1, next(_uids)))


def default_material(name: str = "Material") -> gltf.Material:
    """A plain untextured dielectric -- the palette entry a new object gets.

    ``gltf.Material``'s own defaults are the glTF spec's, which are fully
    metallic and fully rough: correct as a file format default and useless as
    an editing default, because a metal with no environment behind it renders
    as a black shape. This is the mid-grey a modelling package opens on.
    """
    return gltf.Material(
        name=name,
        base_color_factor=(0.8, 0.8, 0.8, 1.0),
        metallic_factor=0.0,
        roughness_factor=0.6,
    )


# A face whose material index names no palette entry still has to draw. It gets
# this one -- a single shared object, so the identity de-duplication holds for
# it too, and a visibly wrong magenta rather than a silent grey, because a mesh
# pointing off the end of the palette is a bug somewhere upstream.
FALLBACK_MATERIAL = gltf.Material(
    name="missing",
    base_color_factor=(1.0, 0.0, 1.0, 1.0),
    metallic_factor=0.0,
    roughness_factor=1.0,
)


@dataclass
class Obj:
    """One object: a mesh, where it sits, and how it was made."""

    uid: int
    name: str
    mesh: bm.Mesh
    translation: Any = field(default_factory=m3.vec3)
    rotation: Any = field(default_factory=m3.quat_identity)  # XYZW
    scale: Any = field(default_factory=lambda: m3.vec3(1.0, 1.0, 1.0))
    generator: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    # The default for *new* faces only. A face's actual material lives on the
    # mesh, one index per face, because a two-toned box is one object.
    material: int = 0

    def __post_init__(self) -> None:
        # Own the transform arrays rather than aliasing whatever was passed in,
        # for the reason ``edits`` states: a caller that keeps mutating the
        # array it handed over would rewrite a recorded step behind the undo
        # stack's back, and a view would misreport its own size to eviction.
        self.translation = np.array(self.translation, dtype="f8", copy=True)
        self.rotation = np.array(self.rotation, dtype="f8", copy=True)
        self.scale = np.array(self.scale, dtype="f8", copy=True)

    def trs(self) -> tuple[Any, Any, Any]:
        return self.translation, self.rotation, self.scale


class BuildDoc:
    """Objects, the palette they share, the selection and the history."""

    def __init__(
        self,
        objects: Iterable[Obj] | None = None,
        materials: Iterable[gltf.Material] | None = None,
    ) -> None:
        self.objects: list[Obj] = list(objects or [])
        self.materials: list[gltf.Material] = (
            [default_material()] if materials is None else list(materials)
        )
        self.selection: set[int] = set()  # object uids
        self.history = UndoStack()
        # A change counter, for anything that caches off the document -- the
        # viewport's GPU upload, the outliner's row list. Deliberately *not*
        # what "dirty" is derived from; see the module docstring.
        self.rev = 0
        self.saved_head = self.history.head

    # -- lookup ------------------------------------------------------------

    def index_of(self, uid: int) -> int:
        for i, obj in enumerate(self.objects):
            if obj.uid == uid:
                return i
        raise KeyError(f"no object with uid {uid}")

    def by_uid(self, uid: int) -> Obj:
        return self.objects[self.index_of(uid)]

    def touch(self) -> None:
        self.rev += 1

    # -- saving ------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.history.head != self.saved_head

    def mark_saved(self) -> None:
        self.saved_head = self.history.head

    # -- history -----------------------------------------------------------

    def undo(self) -> bool:
        return self.history.undo(self)

    def redo(self) -> bool:
        return self.history.redo(self)

    # -- objects -----------------------------------------------------------

    def add_object(self, obj: Obj, index: int | None = None) -> Obj:
        """Insert an object and record the step. Returns the object it was given
        so a caller can place and keep hold of one in a single expression."""
        at = len(self.objects) if index is None else index
        self.objects.insert(at, obj)
        self.history.push(ObjectAddEdit(at, obj))
        self.touch()
        return obj

    def remove_object(self, uid: int) -> bool:
        index = self.index_of(uid)
        obj = self.objects.pop(index)
        self.selection.discard(uid)
        self.history.push(ObjectRemoveEdit(index, obj))
        self.touch()
        return True

    def set_mesh(self, uid: int, mesh: bm.Mesh) -> bool:
        """Replace one object's geometry as one step.

        Identity, not equality, decides whether anything happened: ``Mesh`` is
        ``eq=False`` because numpy arrays have no truthy ``==``, and every op
        is ``Mesh -> Mesh``, so the same object *is* the same geometry.
        """
        obj = self.by_uid(uid)
        if mesh is obj.mesh:
            return False
        before, obj.mesh = obj.mesh, mesh
        self.history.push(MeshEdit(uid, before, mesh))
        self.touch()
        return True

    def set_transform(
        self,
        uid: int,
        *,
        translation: Sequence[float] | None = None,
        rotation: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        was: tuple[Any, Any, Any] | None = None,
    ) -> bool:
        """Move, rotate and scale as one step, pushing nothing for a no-op.

        ``was`` is for a gizmo that mutates the object live so the viewport
        follows the drag and only asks for the step when the drag is released:
        by then the object already holds the new values, so reading "before"
        off it would compare a value against itself and record nothing. What
        such a caller passes must be values the drag cannot reach: ``trs()``
        hands back the object's live arrays, so a gizmo that writes through
        them (``obj.translation[0] = x``) rather than rebinding would find its
        own ``was`` had moved with it. Rebind, as this method does. Setting
        a value to the one already there pushes no step at all, because dirty
        is a comparison against the head and a no-op step makes a saved
        document ask to be saved again.
        """
        obj = self.by_uid(uid)
        before = tuple(np.array(v, dtype="f8", copy=True) for v in (was or obj.trs()))
        after = tuple(
            obj.trs()[i] if new is None else np.array(new, dtype="f8", copy=True)
            for i, new in enumerate((translation, rotation, scale))
        )
        if all(np.array_equal(a, b) for a, b in zip(before, after, strict=True)):
            return False
        obj.translation, obj.rotation, obj.scale = after
        self.history.push(TransformEdit(uid, before, after))  # type: ignore[arg-type]
        self.touch()
        return True

    def set_props(self, uid: int, *, was: dict[str, Any] | None = None, **props: Any) -> bool:
        """Name, visibility, generator, params, default material -- one step.

        ``was`` is the counterpart of :meth:`set_transform`'s, and the trap it
        avoids is sharper here because ``params`` is a dict: a panel that edits
        the object's own dict in place and then passes it back would hand this
        the very object it is comparing against, so "before" and "after" would
        be the same value and the change would record nothing at all. Such a
        caller passes the values it started with as ``was``; a caller that
        builds a fresh dict -- which is what a widget reading a form does --
        needs none of this.
        """
        obj = self.by_uid(uid)
        source = {} if was is None else was
        before = {key: source.get(key, getattr(obj, key)) for key in props}
        if before == props:
            return False
        for key, value in props.items():
            setattr(obj, key, value)
        self.history.push(ObjectPropsEdit(uid, before, dict(props)))
        self.touch()
        return True

    # -- palette -----------------------------------------------------------

    def set_material(self, index: int, material: gltf.Material) -> bool:
        before = self.materials[index]
        if material is before:
            return False
        self.materials[index] = material
        self.history.push(MaterialEdit(index, before, material))
        self.touch()
        return True

    # -- selection (not undoable) ------------------------------------------

    def select(self, uids: Iterable[int]) -> None:
        """Replace the selection. Pushes no step, by design; see the module
        docstring. It still bumps ``rev``, because the viewport draws the
        selected object's outline and has to know to redraw it."""
        self.selection = {int(u) for u in uids}
        self.touch()


# --- the one conversion to glTF ----------------------------------------------


def _material_at(materials: Sequence[gltf.Material], index: int) -> gltf.Material:
    if 0 <= index < len(materials):
        return materials[index]
    return FALLBACK_MATERIAL


def _submesh(mesh: bm.Mesh, faces: np.ndarray) -> bm.Mesh:
    """The mesh restricted to ``faces``, with its vertex array left whole.

    Leaving the positions alone rather than compacting them is deliberate and
    free: ``render_arrays`` emits a vertex only for a corner that some face in
    *this* submesh uses, so an untouched position costs nothing downstream, and
    not re-indexing means not having a second place that could get the mapping
    wrong.
    """
    counts = np.diff(mesh.starts).astype("i8")[faces]
    starts = np.concatenate([[0], np.cumsum(counts)]).astype("i4")
    if len(faces):
        loops = np.concatenate([bm.face(mesh, int(f)) for f in faces])
    else:
        loops = np.zeros(0, dtype="i4")
    return bm.Mesh(
        positions=mesh.positions,
        loops=loops,
        starts=starts,
        material=mesh.material[faces],
        smooth=mesh.smooth[faces],
    )


def to_primitives(obj: Obj, materials: Sequence[gltf.Material]) -> list[gltf.Primitive]:
    """One :class:`~gltf.Primitive` per material the object's faces use.

    A draw call carries one material, so a two-toned box has to be two
    primitives however it is stored -- which is why the split happens here, on
    the way out, rather than in the mesh: the mesh stays one object the user
    can select faces across, and the renderer gets the grouping it needs.

    Groups come out in palette-index order, so the same document produces the
    same primitive order every time -- an exporter's output is diffable, and a
    GPU cache keyed on position does not shuffle.
    """
    mesh = obj.mesh
    if bm.face_count(mesh) == 0:
        return []
    prims = []
    for index in np.unique(mesh.material):
        faces = np.flatnonzero(mesh.material == index)
        positions, normals, indices = bm.render_arrays(_submesh(mesh, faces))
        prims.append(
            gltf.Primitive(
                positions=positions,
                indices=indices,
                normals=normals,
                material=_material_at(materials, int(index)),
            )
        )
    return prims


def to_model(doc: BuildDoc) -> gltf.Model:
    """The document as a :class:`~gltf.Model`: one node per visible object.

    Every consumer goes through here -- the viewport, the GLB writer, the
    render that gets handed to trellis -- so "what does this document look
    like" has exactly one answer. The nodes are all roots: Build mode has no
    hierarchy, and a flat list of roots is what that *is* in glTF terms.

    **A hidden object is simply not here.** ``visible=False`` means it does not
    render, does not export and cannot be picked, and one flag enforcing all
    three in one place is the only way those three can never disagree.
    """
    nodes: list[gltf.Node] = []
    meshes: list[list[gltf.Primitive]] = []
    for obj in doc.objects:
        if not obj.visible:
            continue
        meshes.append(to_primitives(obj, doc.materials))
        nodes.append(
            gltf.Node(
                name=obj.name,
                translation=np.array(obj.translation, dtype="f8", copy=True),
                rotation=np.array(obj.rotation, dtype="f8", copy=True),
                scale=np.array(obj.scale, dtype="f8", copy=True),
                mesh=len(meshes) - 1,
            )
        )
    return gltf.Model(nodes, list(range(len(nodes))), meshes, [])
