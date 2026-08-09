"""The Clay document -- objects, a material palette, and the history over them.

A Clay document is a flat list of objects. There is no hierarchy in Phase 1:
parenting is a feature with a cost (a transform that is not the one you typed,
an outliner that has to explain itself) and nothing in "place a few primitives
and export them" needs it, so ``Obj`` carries a TRS and no parent. The one
conversion out of here -- :func:`to_model` -- is what the viewport draws, what
the exporter writes and what the trellis render photographs; there is exactly
one of it, so those three can never disagree about what the document *is*.

**Materials are ``viewer.gltf.Material`` objects, not a new type.** They are
already pure data, they already carry ``base_color_factor`` /
``metallic_factor`` / ``roughness_factor`` / ``emissive_factor`` /
``double_sided``, and a parallel Clay-only material type would have bought
nothing but a conversion function and a place for the two to drift. It also
pays off on the GPU for free: ``GpuMaterial`` de-duplicates by
``id(material)``, and :func:`to_primitives` hands every primitive the palette
entry *itself*, so a document whose twelve objects share one material uploads
one material. The texture slots stay ``None`` throughout -- Clay paints
no textures, and a slot that is ``None`` is a slot the renderer skips.

**Rotation is XYZW**, matching ``viewer.math3d``, ``viewer.gltf``, the pose
files on disk and glTF itself. There is no other quaternion order anywhere in
this project and this is not the place to introduce one.

**``generator`` is the live-until-frozen field.** An object placed from the
primitive registry keeps the generator's name and the parameters it was built
with, so the properties panel shows "Cylinder: radius, height, segments" and a
change regenerates the mesh as one :class:`~.edits.MeshEdit`. The first
topology edit clears it -- an extruded box is not describable as "box, size 1",
and a panel still offering a size field would discard the edit the moment it
was touched. ``clay_ops`` does that in one place for every op, so no op has to
remember to.

**Dirty is a comparison against ``history.head``, not a flag.** ``rev`` counts
changes and an undo is a change, so a rev-based check calls an undone document
unsaved forever. :attr:`ClayDoc.saved_head` records the head at save time and
:attr:`ClayDoc.dirty` compares against it, which is the same rule the raster
editor's tabs follow and for the same reason.

**Element mode is per document, not per app.** ``element_mode`` and
``element_sel`` live here rather than on ``ClayState`` because the mode is the
*interpretation key* for the selection, and the selection is a property of the
document. An app-level mode would reinterpret every open tab's selection the
moment the user switched tabs -- a face selection in one document read as a
vertex selection in another -- and neither half would have done anything wrong.
Neither the mode nor the selection is serialized: a stored element selection
would describe indices into a mesh the file might reopen with, and a stored mode
would put the user in a mode they did not choose.

In an element mode ``selection`` is **derived**: it holds exactly the uids with
a non-empty ``element_sel``, which is Wings3D's model (a body selection *is* a
selection of everything in it). That invariant is what keeps
``frame_selection``, the properties pane and ``world_bounds`` working with no
per-mode branch anywhere in them.

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

from ..undo import CompoundEdit, Edit, UndoStack
from ..viewer import gltf
from ..viewer import math3d as m3
from . import elements as el
from . import mesh as bm
from .edits import (  # noqa: F401
    MaterialEdit,
    MaterialListEdit,
    MeshEdit,
    ObjectAddEdit,
    ObjectPropsEdit,
    ObjectRemoveEdit,
    TransformEdit,
    _material_holders,
    _shift_materials,
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


class ClayDoc:
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
        # Element mode and what is selected inside each object. See the module
        # docstring: per document, derived-from rather than parallel-to
        # ``selection``, and never written to a file.
        self.element_mode: str = "object"
        self.element_sel: dict[int, el.ElementSel] = {}
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
        """Reverse the newest step, dropping any element selection it invalidates.

        The step is read *before* it is applied, because by the time ``undo``
        returns the stack has already moved it out of reach. Only mesh and
        object edits clear anything: a rename or a gizmo drag leaves the same
        vertices selected on the same mesh, and clearing there would make every
        Ctrl+Z in element mode feel like it deselected something at random.
        """
        edit = self.history.top
        if not self.history.undo(self):
            return False
        self._forget_elements(edit)
        return True

    def redo(self) -> bool:
        edit = self.history.redo_top
        if not self.history.redo(self):
            return False
        self._forget_elements(edit)
        return True

    def _forget_elements(self, edit: Edit | None) -> None:
        for uid in _geometry_uids(edit):
            gone = self.element_sel.pop(uid, None) is not None
            if gone and self.element_mode != "object":
                self.selection.discard(uid)
        self.touch()

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
        self.element_sel.pop(uid, None)
        self.history.push(ObjectRemoveEdit(index, obj))
        self.touch()
        return True

    def set_mesh(
        self,
        uid: int,
        mesh: bm.Mesh,
        *,
        select: el.ElementSel | None = None,
        keep_generator: bool = False,
    ) -> bool:
        """Replace one object's geometry as one step, and freeze its generator.

        Identity, not equality, decides whether anything happened: ``Mesh`` is
        ``eq=False`` because numpy arrays have no truthy ``==``, and every op
        is ``Mesh -> Mesh``, so the same object *is* the same geometry.

        ``select`` is the selection the op wants shown next -- extrude hands
        back its caps so the user can drag them straight away. It is applied
        *after* the push and pushes nothing of its own, because selection is
        not undoable; undoing the mesh edit then drops it, which is the right
        answer for a selection describing geometry that no longer exists.

        **The freeze lives here rather than in the op layer**, which is where
        it was and where it was only half applied: ``clay_ops.run_mesh_op`` and
        Smooth cleared ``generator``, while Delete, Bake Transform, Mirror and
        an element drag did not. An object that still claims to be "box, size
        1" keeps offering that size field, and touching it rebuilds a pristine
        box -- so the deletion, the bake, the mirror or the drag vanished with
        no warning. Geometry that is no longer what a generator would build is
        a fact about ``set_mesh``, not about which caller remembered.

        The freeze is pushed *with* the mesh edit, as one ``CompoundEdit``:
        two steps meant one Ctrl+Z restored the generator claim over the
        still-edited mesh -- the exact state the freeze exists to prevent --
        and the user had to press again to get out of it.

        ``keep_generator`` is the single exception, for the properties panel's
        own rebuild: there the new mesh *is* what the generator makes, which is
        the one case where the claim is still true.
        """
        obj = self.by_uid(uid)
        if mesh is obj.mesh:
            return False
        before, obj.mesh = obj.mesh, mesh
        edits: list[Any] = [MeshEdit(uid, before, mesh)]
        if not keep_generator and obj.generator is not None:
            # One step, not two. Pushed separately, a single Ctrl+Z restored
            # the generator claim over the still-edited mesh -- the exact state
            # the freeze exists to prevent -- and only a second press undid the
            # edit it belongs to.
            was = {"generator": obj.generator, "params": obj.params}
            obj.generator, obj.params = None, {}
            edits.append(ObjectPropsEdit(uid, was, {"generator": None, "params": {}}))
        self.history.push(edits[0] if len(edits) == 1 else CompoundEdit(edits))
        if select is not None:
            self.set_element_sel(uid, select)
        self.touch()
        return True

    def join_objects(self, target_uid: int, mesh: bm.Mesh, others: Iterable[int]) -> bool:
        """Adopt a merged mesh and drop the objects it absorbed, as **one** step.

        One ``CompoundEdit`` and not a ``set_mesh`` followed by N
        ``remove_object`` calls, for the reason ``set_mesh``'s own freeze is one
        step: a Ctrl+Z that put back one of the absorbed objects while the
        target still carried the merged geometry would show the user a document
        in which that shape exists twice -- a state that never happened, and one
        it takes another N presses to leave.

        The removals are recorded in *descending* index order so each
        ``ObjectRemoveEdit`` names the index the object actually sat at when it
        was popped; ``CompoundEdit`` undoes in reverse, which re-inserts them
        ascending, which is the only order in which those indices are all still
        correct.

        The generator freeze applies here exactly as it does in ``set_mesh``:
        a box merged with a sphere is not a box, and leaving the claim would let
        the properties panel rebuild a pristine box over the merge.
        """
        obj = self.by_uid(target_uid)
        doomed = sorted({int(u) for u in others} - {target_uid}, key=self.index_of, reverse=True)
        if mesh is obj.mesh and not doomed:
            return False
        before, obj.mesh = obj.mesh, mesh
        edits: list[Any] = [MeshEdit(target_uid, before, mesh)]
        if obj.generator is not None:
            was = {"generator": obj.generator, "params": obj.params}
            obj.generator, obj.params = None, {}
            edits.append(ObjectPropsEdit(target_uid, was, {"generator": None, "params": {}}))
        for uid in doomed:
            index = self.index_of(uid)
            gone = self.objects.pop(index)
            self.selection.discard(uid)
            self.element_sel.pop(uid, None)
            edits.append(ObjectRemoveEdit(index, gone))
        # The target's own element selection names vertices of the mesh that
        # has just been replaced, so it describes geometry that is no longer
        # there -- the same reason ``_forget_elements`` drops one after an undo.
        self.element_sel.pop(target_uid, None)
        self.history.push(CompoundEdit(edits))
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

    def add_material(self, material: gltf.Material | None = None) -> int:
        """Append a palette entry and return its index.

        Appended, never inserted: a slot is an index that every mesh's per-face
        ``material`` array names, so inserting in the middle would renumber
        those arrays in every object in the document. An append renumbers
        nothing.
        """
        index = len(self.materials)
        entry = material if material is not None else default_material(f"Material {index + 1}")
        self.materials.append(entry)
        self.history.push(MaterialListEdit(index, entry, added=True))
        self.touch()
        return index

    def material_users(self, index: int) -> int:
        """How many faces point at this slot, over everything an undo can reach.

        The document's own objects, and also every object a step on the stack is
        holding out of it -- an undone add, a done remove. Those are in no
        document, so a count over ``self.objects`` alone said zero for a slot
        that a single Ctrl+Z would put faces back onto, and the removal that
        answer permitted renumbered those faces onto whichever material had
        taken the slot's place: the silent reassignment :meth:`remove_material`
        exists to refuse, arriving later and by a different door.

        The cost is that a palette entry stays undeletable for as long as a
        deleted object that used it is still undoable. That is the safe
        direction of a bad trade -- the entry becomes deletable again once the
        step is evicted or the redo branch is dropped, whereas a face silently
        repainted is discovered three edits later with nothing to say what did
        it.

        It counts *faces*, deliberately, and not the ``Obj.material`` default:
        the properties panel's Remove button offers exactly the selected
        object's own default slot and re-points it immediately afterwards, so
        counting defaults would disable the only control that reaches this.
        """
        return sum(
            int((obj.mesh.material == index).sum()) for obj in _material_holders(self)
        )

    def remove_material(self, index: int) -> bool:
        """Drop an *unused* palette entry. -> whether it went.

        Refused while any face points at it -- including a face on an object
        only the undo stack is still holding, see :meth:`material_users` -- and
        refused for the last entry. Reassigning those faces to some other slot
        is the alternative, and it is a silent change to how part of the model
        looks, which is exactly the kind of thing a user discovers three edits
        later. Refusing lets the panel say which objects are in the way.
        """
        if not 0 <= index < len(self.materials) or len(self.materials) <= 1:
            return False
        if self.material_users(index):
            return False
        entry = self.materials[index]
        del self.materials[index]
        _shift_materials(self, index, -1)
        self.history.push(MaterialListEdit(index, entry, added=False))
        self.touch()
        return True

    def set_shading(self, uid: int, faces: Any, smooth: bool) -> bool:
        """Set the per-face shading flag on some of one object's faces.

        Shading is not geometry -- the positions and the topology are untouched
        -- so this keeps the object's generator, exactly as an unwrap does. A
        smooth-shaded box is still a box.
        """
        import numpy as np

        obj = self.by_uid(uid)
        flags = np.array(obj.mesh.smooth, dtype=bool)
        if faces is None:
            flags[:] = smooth
        else:
            picked = np.asarray(faces, dtype="i8")
            if not len(picked):
                return False
            flags[picked] = smooth
        if np.array_equal(flags, obj.mesh.smooth):
            return False
        from dataclasses import replace as _replace

        self.set_mesh(uid, _replace(obj.mesh, smooth=flags), keep_generator=True)
        return True

    # -- selection (not undoable) ------------------------------------------

    def select(self, uids: Iterable[int]) -> None:
        """Replace the selection. Pushes no step, by design; see the module
        docstring. It still bumps ``rev``, because the viewport draws the
        selected object's outline and has to know to redraw it."""
        self.selection = {int(u) for u in uids}
        self.touch()

    # -- element mode (also not undoable) ----------------------------------

    def set_element_mode(self, mode: str) -> None:
        """Switch to object/vertex/edge/face mode, converting what is selected.

        Leaving an element mode keeps the objects selected -- the user was
        working on those objects and is still working on them. *Entering* one
        from object mode selects nothing, because there is nothing to convert
        and the invariant says the object selection in an element mode is the
        set of objects with something selected inside them.
        """
        if mode not in el.MODES:
            raise ValueError(f"unknown element mode {mode!r}")
        if mode == self.element_mode:
            return
        if mode == "object":
            self.element_sel = {}
        else:
            converted: dict[int, el.ElementSel] = {}
            for uid, sel in self.element_sel.items():
                out = el.convert(self.by_uid(uid).mesh, sel, mode)
                if not el.is_empty(out):
                    converted[uid] = out
            self.element_sel = converted
            self.selection = set(converted)
        self.element_mode = mode
        self.touch()

    def set_element_sel(self, uid: int, sel: el.ElementSel | None) -> None:
        """Replace what is selected inside one object, keeping the invariant.

        An empty selection removes the entry *and* the object from
        ``selection``: "selected with nothing selected inside it" is a state the
        invariant does not allow, and letting it exist is how a gizmo ends up
        drawn at the centroid of nothing.
        """
        if el.is_empty(sel):
            self.element_sel.pop(uid, None)
            if self.element_mode != "object":
                self.selection.discard(uid)
        else:
            assert sel is not None
            self.element_sel[uid] = sel
            if self.element_mode != "object":
                self.selection.add(uid)
        self.touch()

    def element_sel_of(self, uid: int) -> el.ElementSel:
        """What is selected inside *uid* -- an empty selection when nothing is."""
        return self.element_sel.get(uid) or el.empty()

    def clear_element_sel(self) -> None:
        self.element_sel = {}
        if self.element_mode != "object":
            self.selection = set()
        self.touch()


def _geometry_uids(edit: Edit | None) -> set[int]:
    """The uids whose *geometry* an edit changes, walking compounds.

    Deliberately narrow. A ``TransformEdit`` moves an object without changing a
    single vertex index and a rename changes nothing at all -- an element
    selection survives both, and clearing it there would make undo feel
    arbitrary. Only a mesh replacement, an add or a remove can leave a stored
    selection pointing at indices the mesh no longer has.
    """
    if edit is None:
        return set()
    if isinstance(edit, CompoundEdit):
        return {uid for child in edit.edits for uid in _geometry_uids(child)}
    if isinstance(edit, MeshEdit):
        return {edit.obj_uid}
    if isinstance(edit, ObjectAddEdit | ObjectRemoveEdit):
        return {edit.obj.uid}
    return set()


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
    wrong. The same does *not* go for ``uv``, which is indexed by corner rather
    than by vertex and so has to be gathered alongside ``loops``.

    The corner gather is arithmetic rather than a loop over faces because this
    runs once per material per object per rebuild, and an imported mesh has
    hundreds of thousands of faces: a Python-level pass over them here was the
    single worst hot spot in the rebuild.
    """
    counts = np.diff(mesh.starts).astype("i8")[faces]
    starts = np.concatenate([[0], np.cumsum(counts)]).astype("i4")
    total = int(starts[-1]) if len(starts) else 0
    if total:
        # For every output corner, the index of the corner it came from:
        # its face's start, plus how far into that face it sits.
        face_of_corner = np.repeat(np.arange(len(faces), dtype="i8"), counts)
        within = np.arange(total, dtype="i8") - starts[:-1].astype("i8")[face_of_corner]
        corners = mesh.starts[:-1].astype("i8")[faces][face_of_corner] + within
    else:
        corners = np.zeros(0, dtype="i8")
    return bm.Mesh(
        positions=mesh.positions,
        loops=mesh.loops[corners],
        starts=starts,
        material=mesh.material[faces],
        smooth=mesh.smooth[faces],
        uv=None if mesh.uv is None else mesh.uv[corners],
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
        positions, normals, uvs, indices = bm.render_arrays(_submesh(mesh, faces))
        prims.append(
            gltf.Primitive(
                positions=positions,
                indices=indices,
                normals=normals,
                uvs=uvs,
                material=_material_at(materials, int(index)),
            )
        )
    return prims


def to_model(doc: ClayDoc) -> gltf.Model:
    """The document as a :class:`~gltf.Model`: one node per visible object.

    Every consumer goes through here -- the viewport, the GLB writer, the
    render that gets handed to trellis -- so "what does this document look
    like" has exactly one answer. The nodes are all roots: Clay has no
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
