"""One registry of everything Clay can *do*, and no imgui anywhere in it.

Three surfaces invoke Clay's operations -- the right-mouse context menu, the
buttons in the tools pane, and ``clay_mode.handle_key`` -- and before this
module they each had their own list. Three lists means three answers to "is
Extrude available in vertex mode", and the interesting one is always the one
nobody updated: a key that fires an op the menu greys out, or a menu row for an
op the key path never learned about.

So there is one list. :data:`OPS` is the whole of what is invocable, each entry
carrying the modes it applies to, whether it is enabled right now, the key that
fires it and how it groups in the menu. The menu renders it, the pane renders a
subset of it, and the key handler looks up by key -- and none of them decides
anything.

**Nothing here imports imgui**, which is what keeps the registry testable: an
``Op``'s ``enabled`` predicate is a function of a document, so "Fill Hole is
greyed out with a face selected" is a plain assertion rather than a screenshot.
The pane layer (``panes/clay_menu.py``) is the only thing that knows a popup
exists.

**An op that changes geometry freezes the object's generator.** A box whose
faces have been extruded is no longer describable by "box, size 1" -- the
properties panel would offer a size field that silently discards the edit the
moment it was touched. The freeze is ``Document.set_mesh``'s, not any op's:
saying "``run`` clears it in one place, for every op" was not true of ``run``
at all -- only ``run_mesh_op`` and Smooth did it, while Delete, Bake Transform
and Mirror went straight to ``set_mesh`` and kept a generator that would
rebuild over them. Putting it where the geometry actually changes is what
makes the sentence true for ops that do not exist yet.

**A refusal is a toast, not an exception.** Every op raises
:class:`~.clay.elements.OpError` with a sentence naming what it refused and what
to do instead; :func:`run` catches exactly that, shows it and records no edit.
Anything else propagates, because an ``IndexError`` out of a topology op is a
bug and swallowing it would leave a half-built mesh on screen with no clue why.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "OPS",
    "Op",
    "Param",
    "by_key",
    "defaults_for",
    "menu",
    "register",
    "run",
    "run_mesh_op",
    "run_object_op",
]

# The element modes an op can appear in. "object" is the fourth and is not an
# element mode; ops that name it are the object-level ones (duplicate, bake,
# mirror) that were previously hardcoded in the tools pane.
ALL_MODES: tuple[str, ...] = ("object", "vertex", "edge", "face")
ELEMENT_MODES: tuple[str, ...] = ("vertex", "edge", "face")


@dataclass(frozen=True)
class Param:
    """One number an op takes, and everything a widget needs to offer it.

    Described rather than drawn so the pane builds every popup from one loop.
    ``warn`` is shown under the field -- Catmull-Clark at two levels multiplies
    a mesh by sixteen, and a user who finds that out by waiting is a user who
    lost their document to the undo budget.
    """

    name: str
    label: str
    default: float
    step: float = 0.01
    low: float = 0.0
    high: float = 1e6
    integer: bool = False
    warn: str = ""


@dataclass(frozen=True)
class Op:
    """One invocable operation.

    ``run(ctx, doc)`` does the work and is free to push whatever steps it needs;
    ``enabled(doc)`` answers whether it can right now, and is what greys the
    menu row and disables the button. ``key`` is the shortcut *label* as well as
    the binding, so the menu and the shortcuts sheet cannot disagree about it.
    """

    name: str
    label: str
    modes: tuple[str, ...]
    run: Callable[..., Any]
    enabled: Callable[[Any], bool] = lambda doc: True
    key: str = ""
    separator_before: bool = False
    params: tuple[Param, ...] = field(default=())
    """The numbers the pane pops a dialog for, or empty for a bare action."""
    hint: str = ""
    """One sentence under the dialog's title, about the op rather than a field.

    ``Param.warn`` is about a *number* -- what two levels of Catmull-Clark will
    do to a mesh -- and belongs under the field it qualifies. This is about the
    op: what it is for, and when the op next to it is the one you meant. Only
    the parameterised ops can show it, because only they open a dialog, which
    is the right restriction: a bare action gives no moment to read anything.
    """


OPS: list[Op] = []


def register(op: Op) -> Op:
    """Add an op to the registry, refusing a duplicate name."""
    if any(existing.name == op.name for existing in OPS):
        raise ValueError(f"an op named {op.name!r} is already registered")
    OPS.append(op)
    return op


def menu(mode: str) -> list[Op]:
    """Every op that applies in *mode*, in registration order."""
    return [op for op in OPS if mode in op.modes]


def by_key(mode: str, key: str) -> Op | None:
    """The op *key* fires in *mode*, or ``None``."""
    for op in menu(mode):
        if op.key and op.key == key:
            return op
    return None


def get(name: str) -> Op:
    for op in OPS:
        if op.name == name:
            return op
    raise KeyError(f"no op named {name!r}")


# --- running ----------------------------------------------------------------


def toast(ctx: Any, message: str) -> None:
    """A refusal, shown. ``Ctx.toast(text, level)`` is the whole API.

    It used to reach for a ``ctx.toasts`` attribute that the real ``Ctx`` has
    never had, so every refusal this module raises -- "can't delete the last
    object", every per-object one -- was raised, caught, formatted and thrown
    away in the running app, and only the test doubles ever saw one.
    """
    ctx.toast(message, "error")


def defaults_for(op: Op) -> dict[str, float]:
    """The op's parameters at their defaults, as a fresh dict."""
    return {param.name: param.default for param in op.params}


def format_for(param: Param) -> str:
    """The printf format ``input_float`` should draw this parameter with.

    imgui's own default is ``"%.3f"``, and both weld distances in this registry
    default to 1e-4 -- so the field read ``0.000``, the step arrows moved it by
    an amount no digit shown could express, and the number a user typed came
    back as a different one. A control whose value cannot be read is not a
    control.

    **Derived from the parameter rather than declared on it.** A ``format``
    field would be one more thing to remember, and the next sub-millimetre
    parameter would arrive without it and land in exactly the same hole; the
    step and the default already say what precision the number is kept at.
    Widened only downwards, so every parameter that was legible at three
    decimals still reads exactly as it did.
    """
    scale = min(abs(param.step) or 1.0, abs(param.default) or 1.0)
    decimals = 3
    while decimals < 9 and scale < 10.0**-decimals:
        decimals += 1
    return f"%.{decimals}f"


def run(ctx: Any, doc: Any, op: Op, **params: Any) -> bool:
    """Invoke an op, turning a refusal into a toast. -> whether it ran.

    Missing parameters fall back to their declared defaults, so a caller that
    has no remembered values -- the key path, a test -- gets the same result the
    popup would have produced with the fields untouched.

    **Declared parameters are clamped here.** The popup clamps its live fields
    too (``panes/clay_menu.py``), but that is a UX affordance on one surface --
    the key path, the tools pane and every test call arrive with whatever the
    caller had remembered, and a subdivision at ``levels=99`` is not a refusal
    an op should have to write for itself. ``run`` is the choke point all three
    surfaces funnel through, so the range a ``Param`` declares is enforced once,
    where it cannot be bypassed.
    """
    from .clay.elements import OpError

    if not op.enabled(doc):
        return False
    values = defaults_for(op) | params
    for param in op.params:
        value = min(max(float(values[param.name]), param.low), param.high)
        values[param.name] = int(value) if param.integer else value
    depth = len(doc.history)
    head = doc.history.head
    before = _op_context(doc)
    try:
        result = op.run(ctx, doc, **values)
    except OpError as error:
        toast(ctx, str(error))
        return False
    # An op that refuses *per object* -- ``run_mesh_op`` toasts and carries on
    # to the next one -- says so by returning False rather than by raising, so
    # a caller still learns that nothing happened.
    if result is False:
        return False
    _one_step(doc, op, depth, head)
    _remember(ctx, doc, op, values, depth, before)
    return True


def _op_context(doc: Any) -> tuple[str, dict, set]:
    """What the op is about to run against, snapshotted before it runs.

    Shallow copies, which is enough because both an ``ElementSel`` and a uid
    are immutable -- what changes is which of them the document is holding, and
    that is exactly what the copies pin.
    """

    return (
        str(getattr(doc, "element_mode", "object")),
        dict(getattr(doc, "element_sel", {})),
        set(getattr(doc, "selection", ())),
    )


def _one_step(doc: Any, op: Op, depth: int, head: int) -> None:
    """Fold everything the op pushed into a single, named step.

    **One press, one Ctrl+Z.** An op is free to push whatever steps it needs and
    most push one, but the composed ones do not: Mirror pushed a transform and a
    mesh change, and ``run_mesh_op`` pushes one ``set_mesh`` *per object*, so
    extruding faces across three objects cost three presses to undo -- a
    gesture the user made once. ``collapse_since`` is the primitive that was
    built for exactly this and had one caller.

    Then the name. A fold reads as "compound" in the history panel, which says
    nothing about what is being undone; the op knows what it was, and
    ``Op.label`` is already the word on the button. The trailing ellipsis of a
    parameterised op's label goes -- "Bevel..." is an invitation to a dialog,
    and this is a record of something that happened.

    ``head`` is the stack's head **before the op ran**, and it has to be taken
    by the caller rather than read here: read at the top of this function it is
    already the post-op head, so the guard below only fired when the *collapse*
    had pushed -- which is to say a multi-object op got its name and a
    single-object one silently kept "mesh". Serials rather than a depth
    comparison, because the byte budget can evict an older step while this one
    is pushed and leave the two counts equal.
    """

    history = getattr(doc, "history", None)
    if history is None:  # pragma: no cover - every document has one
        return
    history.collapse_since(depth)
    top = history.top
    # Only when the op actually pushed something: a select-all or a frame
    # changes no document, and labelling the *previous* step with this op's
    # name would be a lie in the one place a user goes to read what happened.
    if top is not None and history.head != head:
        top.label = op.label.rstrip(".")


def _remember(
    ctx: Any, doc: Any, op: Op, values: dict, depth: int, before: tuple
) -> None:
    """Record the run, for the adjust card and for Repeat.

    Reached through ``getattr`` rather than by importing ``clay_mode``: this
    module is the one every surface funnels through and it must not depend on
    any of them. A ctx with no Clay state -- a test's fake, a headless call --
    simply records nothing, which is the right answer rather than a guard the
    caller has to write.
    """

    from .clay_state import LastOp

    state = getattr(getattr(ctx, "state", None), "clay", None)
    if state is None:
        return
    mode, element_sel, selection = before
    state.last_op = LastOp(
        name=op.name,
        params=dict(values),
        depth_before=depth,
        head_after=getattr(doc.history, "head", 0),
        element_mode=mode,
        element_sel=element_sel,
        selection=selection,
    )


def run_mesh_op(
    ctx: Any, doc: Any, func: Callable[..., Any], /, **params: Any
) -> bool:
    """Apply a ``(mesh, sel) -> (mesh, sel)`` op to every object with a selection.

    The loop, the refusal handling and the generator freeze all live here, so an
    op is registered by naming its function rather than by writing this out
    again -- and a refusal on one object does not abandon the others, which is
    what a user selecting faces across two objects means by pressing the button
    once.
    """
    from .clay.elements import OpError

    ran = False
    for uid in list(doc.element_sel):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        try:
            mesh, sel = func(obj.mesh, doc.element_sel_of(uid), **params)
        except OpError as error:
            toast(ctx, str(error))
            continue
        doc.set_mesh(uid, mesh, select=sel)
        ran = True
    return ran


def run_object_op(
    ctx: Any, doc: Any, func: Callable[[Any, Any], Any], /, *, uids: Iterable[int] | None = None
) -> bool:
    """Apply ``func(doc, obj)`` to every selected object. -> whether any ran.

    ``run_mesh_op``'s sibling for the object-level ops, and it exists for the
    same reason: seven ops here wrote out the same four lines -- snapshot the
    selection, look each uid up, tolerate one that has gone, toast a refusal and
    carry on -- and only two of them wrote out all four. The ones that skipped
    the ``KeyError`` guard crash on an object deleted between the snapshot and
    the loop; the ones that skipped the ``OpError`` catch abandon the rest of
    the selection when one object refuses.

    The snapshot is taken before anything runs because ``doc.selection`` is live
    and several of these ops change it.
    """
    from .clay.elements import OpError

    ran = False
    for uid in list(doc.selection if uids is None else uids):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        try:
            func(doc, obj)
        except OpError as error:
            toast(ctx, str(error))
            continue
        ran = True
    return ran


# --- predicates -------------------------------------------------------------


def has_objects(doc: Any) -> bool:
    return bool(doc.selection)


def has_elements(doc: Any) -> bool:
    return bool(doc.element_sel)


def has_two_visible(doc: Any) -> bool:
    """Two selected objects the user can actually see. Merge's predicate, and
    the only one that has to look past ``selection`` at what is in it."""
    return sum(1 for obj in doc.objects if obj.uid in doc.selection and obj.visible) >= 2


def in_mode(*modes: str) -> Callable[[Any], bool]:
    def check(doc: Any) -> bool:
        return doc.element_mode in modes and bool(doc.element_sel)

    return check


# --- the object-level ops (moved out of the tools pane) ---------------------


def _element(dotted: str) -> Callable[..., None]:
    """Register a ``clay.ops_*`` function as an op, resolved lazily by name.

    Lazy so importing the registry does not drag every topology module in at
    startup, and by name so the table below reads as a list of ops rather than a
    list of imports.
    """
    module, func = dotted.rsplit(".", 1)

    def call(ctx: Any, doc: Any, **params: Any) -> bool:
        import importlib

        target = getattr(importlib.import_module(f".clay.{module}", __package__), func)
        return run_mesh_op(ctx, doc, target, **params)

    return call


def _dissolve(ctx: Any, doc: Any, **_: Any) -> None:
    """Dissolve, dispatched on the mode -- one menu row rather than three.

    "Dissolve" means the same thing to a user in every mode (get rid of this,
    and heal what it separated); it is only the implementation that differs, so
    splitting it into three rows would be exposing the implementation.
    """
    from .clay import ops_dissolve

    which = {
        "vertex": ops_dissolve.dissolve_verts,
        "edge": ops_dissolve.dissolve_edges,
        "face": ops_dissolve.dissolve_faces,
    }.get(doc.element_mode)
    if which is not None:
        run_mesh_op(ctx, doc, which)


def _extrude(ctx: Any, doc: Any, **params: Any) -> bool:
    """Extrude, dispatched on the mode -- one row and one key, as Dissolve is.

    "Extrude" means the same thing in all three modes (pull this out and wall in
    the gap it leaves) and only the implementation differs, so three rows would
    be exposing the implementation. The vertex branch is the one that is not a
    simple rename: a mesh stores no wire edges, so it extrudes the *border*
    edges the selection implies -- see ``ops_topo.extrude_verts``.
    """
    from .clay import ops_topo

    which = {
        "vertex": ops_topo.extrude_verts,
        "edge": ops_topo.extrude_edges,
        "face": ops_topo.extrude_faces,
    }.get(doc.element_mode)
    return which is not None and run_mesh_op(ctx, doc, which, **params)


def _smooth(ctx: Any, doc: Any, levels: float = 1.0, **_: Any) -> None:
    """Catmull-Clark over every selected object, whatever the mode.

    It ignores the element selection deliberately -- a smoothing subdivision
    moves the original vertices, and moving only some of them tears the surface
    along the edge of the selection. See ``ops_subdiv.catmull_clark``.
    """
    from .clay import elements as el
    from .clay import ops_subdiv

    def one(doc: Any, obj: Any) -> None:
        mesh, sel = ops_subdiv.catmull_clark(obj.mesh, el.empty(), levels=int(levels))
        doc.set_mesh(obj.uid, mesh, select=sel)

    run_object_op(ctx, doc, one)


def _duplicate(ctx: Any, doc: Any, **_: Any) -> None:
    from .clay import selection

    del ctx
    selection.duplicate_selected(doc)


def _bake(ctx: Any, doc: Any, **_: Any) -> None:
    """Fold each selected object's transform into its geometry.

    Two steps rather than one compound: the mesh and the transform are separate
    edits in this document's vocabulary, and keeping them separate means an undo
    of a bake is two presses rather than a compound type that exists for one
    button.
    """
    from .clay import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        baked = clay_ops_geom.bake_transform(obj)
        doc.set_mesh(obj.uid, baked.mesh)
        doc.set_transform(
            obj.uid,
            translation=baked.translation,
            rotation=baked.rotation,
            scale=baked.scale,
        )

    run_object_op(ctx, doc, one)


def _join(ctx: Any, doc: Any, weld: float = 1e-4, **_: Any) -> None:
    """Merge every selected object into the topmost one in the outliner.

    Document order rather than "the one clicked last": ``doc.selection`` is a
    set and carries no order at all, so a target read out of it would be
    whichever object the hash happened to put first -- and the name, transform
    and material default that the merge keeps are the *target's*, which makes
    that an arbitrary answer to a question the user can see the answer to.

    The geometry is ``clay.ops.join``'s and the bookkeeping is
    ``ClayDoc.join_objects``'s, which is this module's usual boundary; the one
    thing that happens here is the selection afterwards, and it is deliberately
    the survivor rather than nothing -- the user has one shape now and the
    gizmo should be on it.

    **Visible objects only**, matching ``visible=False``'s documented meaning
    that an object does not render, does not export and cannot be picked. A
    merge is the one op where absorbing an unseen object is not merely odd but
    invisible in its result too: the geometry arrives inside the survivor, which
    *is* shown. ``_select_all`` no longer hands over hidden objects, so this is
    the second half -- an object hidden after it was selected.
    """
    from .clay import ops as clay_ops_geom

    del ctx
    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    mesh = clay_ops_geom.join([doc.by_uid(uid) for uid in uids], eps=float(weld))
    doc.join_objects(uids[0], mesh, uids[1:])
    doc.select([uids[0]])


def _union(ctx: Any, doc: Any, **_: Any) -> None:
    """Boolean-union every selected object into the topmost one.

    ``_join``'s shape exactly -- topmost visible as the target, the geometry
    from ``clay.ops_boolean``, the bookkeeping from ``ClayDoc.join_objects``,
    the survivor left selected -- and every one of those reasons carries over
    unchanged. What is different is only which function computes the mesh, and
    that a refusal is possible: a union is defined over closed solids, so
    ``ops_boolean.union`` raises :class:`OpError` where ``ops.join`` cannot.
    That is caught where every other element-op refusal is, in ``run``.

    In-process and synchronous, unlike a mesh pipeline: manifold is CPU and
    fast at the scale Clay authors at, and handing this to ``TaskRunner`` would
    mean a document edit landing from another thread.
    """
    from .clay import ops_boolean

    del ctx
    uids = [obj.uid for obj in doc.objects if obj.uid in doc.selection and obj.visible]
    mesh = ops_boolean.union([doc.by_uid(uid) for uid in uids])
    doc.join_objects(uids[0], mesh, uids[1:])
    doc.select([uids[0]])


def mirror(ctx: Any, doc: Any, axis: int, **_: Any) -> None:
    from .clay import ops as clay_ops_geom

    def one(doc: Any, obj: Any) -> None:
        doc.set_mesh(obj.uid, clay_ops_geom.mirror(obj, axis).mesh)

    run_object_op(ctx, doc, one)


def _delete(ctx: Any, doc: Any, **_: Any) -> None:
    from .clay import selection

    for message in selection.delete_selected(doc):
        toast(ctx, message)


def _unwrap(ctx: Any, doc: Any, **_: Any) -> None:
    """Give every selected object a fresh box projection.

    Whole objects rather than the selected faces, and that is the decision:
    unwrapping half a mesh leaves the other half's coordinates from whenever
    they were last computed, so the two islands are at different texel
    densities and a checker map says so immediately. Per-face unwrapping is a
    real feature and it needs a seam tool first.

    It does **not** freeze the generator. UVs are not geometry -- the positions,
    the topology and the parameters are all untouched -- so a box that has been
    unwrapped is still describable as "box, size 1", and re-editing the size
    correctly rebuilds it with the generator's own canonical coordinates.
    """
    from .clay import uv as uv_mod

    def one(doc: Any, obj: Any) -> None:
        doc.set_mesh(obj.uid, uv_mod.box_unwrap(obj.mesh), keep_generator=True)

    run_object_op(ctx, doc, one)


def _shade(smooth: bool) -> Callable[..., None]:
    """Set the shading flag on the selected faces, or on whole objects.

    Both modes, because both readings are real: in face mode a user means
    "these faces", and in object mode they mean "this whole shape". The flag is
    per face either way -- there is no object-level shading setting that would
    have to be kept in agreement with it.
    """

    def run(ctx: Any, doc: Any, **_: Any) -> None:
        if doc.element_mode == "object":
            run_object_op(ctx, doc, lambda doc, obj: doc.set_shading(obj.uid, None, smooth))
            return
        from .clay import elements as el

        del ctx
        for uid in list(doc.element_sel):
            faces = el.convert(doc.by_uid(uid).mesh, doc.element_sel_of(uid), "face")
            doc.set_shading(uid, faces.faces, smooth)

    return run


def _shade_auto(ctx: Any, doc: Any, angle: float = 30.0, **_: Any) -> None:
    """Smooth every face whose *every* neighbour agrees with it to within *angle*.

    Per face rather than per edge, because ``smooth`` is a per-face flag and
    there is nowhere to record "smooth along this edge only" -- so a face is
    smooth only when it has no sharp edge at all. That is a real limitation and
    it is stated here because the result surprises people: **a capped cylinder
    comes out entirely flat**, since every side quad meets a cap at a right
    angle.

    That is also the *correct* answer for this renderer rather than a gap in
    the rule. A smooth face takes accumulated vertex normals, so smoothing the
    band while the caps stay flat would average the cap normals into the rim
    and round the very edge the caps are there to define. Blender avoids this
    with per-edge split normals, which is a different mesh format.

    What it does do well is exactly what it should: a sphere or a torus goes
    smooth throughout, a box stays flat, and a mesh that mixes the two gets the
    right answer per region. The measurement is the cosine between adjacent
    face normals -- the same question ``glbimport`` asks of an imported mesh's
    supplied normals -- and a boundary edge has no neighbour to disagree with,
    so it makes nothing sharp.
    """
    from dataclasses import replace

    import numpy as np

    from .clay import mesh as bm
    from .clay.adjacency import adjacency

    limit = float(np.cos(np.radians(max(0.0, min(180.0, float(angle))))))

    def one(doc: Any, obj: Any) -> None:
        mesh = obj.mesh
        faces = bm.face_count(mesh)
        if faces == 0:
            return
        # Normalised: ``face_normals`` returns Newell normals, whose length is
        # proportional to face area -- a dot product of two of those is not a
        # cosine, and comparing it against one silently called every pair sharp.
        normals = np.asarray(bm.face_normals(mesh), dtype="f8")
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)
        counts = np.diff(np.asarray(mesh.starts, dtype="i8"))
        face_of = np.repeat(np.arange(faces, dtype="i8"), counts)
        twin = np.asarray(adjacency(mesh).twin, dtype="i8")

        paired = np.flatnonzero(twin >= 0)
        smooth = np.ones(faces, dtype=bool)
        if len(paired):
            left, right = face_of[paired], face_of[twin[paired]]
            sharp = np.einsum("ij,ij->i", normals[left], normals[right]) < limit
            smooth[left[sharp]] = False
            smooth[right[sharp]] = False
        if np.array_equal(smooth, mesh.smooth):
            return
        # One step per object, and only for an object this changed: going
        # through ``set_shading`` first and then writing the array would push
        # two, and a Ctrl+Z would land halfway.
        doc.set_mesh(obj.uid, replace(mesh, smooth=smooth), keep_generator=True)

    # The whole document when nothing is selected: this is the one op here that
    # means "tidy the shading", and a user with no selection means all of it.
    run_object_op(ctx, doc, one, uids=list(doc.selection) or [obj.uid for obj in doc.objects])


def _frame(ctx: Any, doc: Any, **_: Any) -> None:
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.frame_selection(doc)


def _select_all(ctx: Any, doc: Any, **_: Any) -> None:
    from .clay import selection

    del ctx
    selection.select_all(doc)


def _select_none(ctx: Any, doc: Any, **_: Any) -> None:
    del ctx
    doc.clear_element_sel()


def _invert(ctx: Any, doc: Any, **_: Any) -> None:
    from .clay import selection

    del ctx
    selection.invert(doc)


def _register_defaults() -> None:
    """Build the registry once, at import.

    A function rather than module-level statements so the tests can assert the
    registry is *complete* by calling it on a fresh list, and so a duplicate
    import cannot register everything twice.
    """
    if OPS:
        return

    register(
        Op(
            name="select-all",
            label="Select All",
            modes=ELEMENT_MODES,
            run=_select_all,
            key="Ctrl+A",
        )
    )
    register(
        Op(
            name="select-none",
            label="Select None",
            modes=ELEMENT_MODES,
            run=_select_none,
            enabled=has_elements,
            key="Esc",
        )
    )
    register(
        Op(
            name="select-invert",
            label="Invert Selection",
            modes=ELEMENT_MODES,
            run=_invert,
            key="Ctrl+Shift+I",
        )
    )

    register(
        Op(
            name="duplicate",
            label="Duplicate",
            modes=("object",),
            run=_duplicate,
            enabled=has_objects,
            key="Ctrl+D",
            separator_before=True,
        )
    )
    for smooth, label in ((True, "Shade Smooth"), (False, "Shade Flat")):
        register(
            Op(
                name=f"shade-{'smooth' if smooth else 'flat'}",
                label=label,
                modes=("object", "face"),
                run=_shade(smooth),
                enabled=has_objects,
                separator_before=smooth,
            )
        )
    register(
        Op(
            name="shade-auto",
            label="Shade Auto...",
            modes=("object",),
            run=_shade_auto,
            enabled=has_objects,
            params=(
                Param(
                    "angle",
                    "sharp above (deg)",
                    30.0,
                    5.0,
                    low=0.0,
                    high=180.0,
                    warn="0 makes everything flat; 180 makes everything smooth.",
                ),
            ),
        )
    )
    register(
        Op(
            name="unwrap",
            label="Box Unwrap",
            modes=("object",),
            run=_unwrap,
            enabled=has_objects,
        )
    )
    register(
        Op(
            name="bake",
            label="Bake Transform",
            modes=("object",),
            run=_bake,
            enabled=has_objects,
        )
    )
    register(
        Op(
            name="join",
            label="Merge Objects...",
            modes=("object",),
            run=_join,
            # Two, not one: merging a single object is the identity, and an
            # enabled button that does nothing is worse than a greyed one.
            # Counted over the *visible* selection for that same reason, since
            # that is what ``_join`` will actually merge -- a row enabled by an
            # object the merge then skips is the greyed one's problem again.
            enabled=has_two_visible,
            key="Ctrl+J",
            # The pointer at its counterpart, here rather than in the menu: this
            # dialog is the one moment the user has committed to "make these one
            # object" and can still choose which meaning of it they wanted, and
            # the moment the weld's cost -- the walls it is about to bury inside
            # the result -- has not been paid yet.
            hint=(
                "Welds the shapes and keeps the surfaces inside the overlap. For "
                "shapes that interpenetrate, Union Objects... (Ctrl+Shift+J) cuts "
                "those away instead -- at the cost of the UVs and the n-gons."
            ),
            params=(
                Param(
                    "weld",
                    "weld distance (m)",
                    1e-4,
                    1e-4,
                    low=0.0,
                    warn="0 keeps the shapes as separate shells inside one object.",
                ),
            ),
        )
    )
    # Beside *Merge Objects...*, never instead of it. The two answer different
    # questions and the manual says which is which: a merge welds and keeps the
    # geometry inside the overlap, a union removes it -- and pays for that with
    # the UVs and the n-gons, which is exactly why the weld cannot simply be
    # retired in its favour. Same predicate, because "fewer than two visible" is
    # the identity for both.
    #
    # And the same key, shifted. Union spent its first release in the context
    # menu with no binding at all while the weld beside it held Ctrl+J, so of
    # the pair the discoverable one was the one that leaves the interior walls
    # in -- users found *Merge*, got z-fighting inside the overlap, and had no
    # reason to suspect the other row existed. Shift is the right modifier for
    # it under the rule Ctrl+Shift+Z and Ctrl+Shift+I already follow here: the
    # same question, answered the other way.
    register(
        Op(
            name="union",
            label="Union Objects...",
            modes=("object",),
            run=_union,
            enabled=has_two_visible,
            key="Ctrl+Shift+J",
        )
    )
    for axis, label in enumerate(("X", "Y", "Z")):
        register(
            Op(
                name=f"mirror-{label.lower()}",
                label=f"Mirror {label}",
                modes=("object",),
                run=(lambda a: lambda ctx, doc, **kw: mirror(ctx, doc, a, **kw))(axis),
                enabled=has_objects,
                separator_before=axis == 0,
            )
        )

    register(
        Op(
            name="extrude",
            label="Extrude",
            modes=ELEMENT_MODES,
            run=_extrude,
            enabled=has_elements,
            key="E",
            separator_before=True,
        )
    )
    register(
        Op(
            name="bridge",
            label="Bridge Loops",
            modes=("edge",),
            run=_element("ops_topo.bridge_edges"),
            enabled=in_mode("edge"),
        )
    )
    register(
        Op(
            name="inset",
            label="Inset Faces...",
            modes=("face",),
            run=_element("ops_topo.inset_faces"),
            enabled=in_mode("face"),
            params=(
                Param("thickness", "thickness (m)", 0.1, 0.01),
                Param("depth", "depth (m)", 0.0, 0.01, low=-1e6),
            ),
        )
    )
    register(
        Op(
            name="bevel",
            label="Bevel Edges...",
            modes=("edge",),
            run=_element("ops_bevel.bevel_edges"),
            enabled=in_mode("edge"),
            params=(Param("width", "width (m)", 0.05, 0.01),),
        )
    )
    register(
        Op(
            name="loop-cut",
            label="Loop Cut...",
            modes=("edge",),
            run=_element("ops_bevel.loop_cut"),
            enabled=in_mode("edge"),
            params=(Param("t", "position", 0.5, 0.05, low=0.0, high=1.0),),
        )
    )
    register(
        Op(
            name="dissolve",
            label="Dissolve",
            modes=ELEMENT_MODES,
            run=_dissolve,
            enabled=has_elements,
            separator_before=True,
        )
    )
    # Deliberately a second name for ``dissolve`` in face mode rather than a
    # second implementation. ``ops_dissolve.dissolve_faces`` has always merged a
    # connected block of faces into one n-gon; what it lacked was a name anyone
    # would look for. "Dissolve" is the modelling word and stays, because it is
    # what the vertex and edge modes do too and splitting it would be exposing
    # the implementation -- but a user who wants to merge two faces searches for
    # "merge", finds nothing, and concludes the editor cannot do it.
    #
    # Registering it (rather than adding a button) is what gets it into all
    # three surfaces at once: the Clay menu, the tools pane and the bare-letter
    # keys all read this table.
    register(
        Op(
            name="merge_faces",
            label="Merge Faces",
            modes=("face",),
            run=_element("ops_dissolve.dissolve_faces"),
            enabled=in_mode("face"),
        )
    )
    register(
        Op(
            name="collapse",
            label="Collapse",
            modes=("edge", "face"),
            run=_element("ops_topo.collapse"),
            enabled=in_mode("edge", "face"),
        )
    )
    register(
        Op(
            name="weld",
            label="Weld...",
            modes=("vertex",),
            run=_element("ops_topo.weld"),
            enabled=in_mode("vertex"),
            params=(Param("eps", "distance (m)", 1e-4, 1e-4, low=1e-9),),
        )
    )
    register(
        Op(
            name="fill-hole",
            label="Fill Hole",
            modes=("edge",),
            run=_element("ops_topo.fill_hole"),
            enabled=in_mode("edge"),
        )
    )
    register(
        Op(
            name="flip",
            label="Flip Normals",
            modes=("face",),
            run=_element("ops_topo.flip_normals"),
            enabled=in_mode("face"),
            separator_before=True,
        )
    )
    register(
        Op(
            name="subdivide",
            label="Subdivide",
            modes=("face",),
            run=_element("ops_subdiv.subdivide"),
            enabled=in_mode("face"),
        )
    )
    register(
        Op(
            name="smooth",
            label="Smooth (Catmull-Clark)...",
            modes=ALL_MODES,
            run=_smooth,
            enabled=lambda doc: bool(doc.selection),
            params=(
                Param(
                    "levels",
                    "levels",
                    1.0,
                    1.0,
                    low=1.0,
                    high=4.0,
                    integer=True,
                    warn="Each level multiplies the face count by four.",
                ),
            ),
        )
    )
    register(
        Op(
            name="frame",
            label="Frame Selection",
            modes=ALL_MODES,
            run=_frame,
            separator_before=True,
        )
    )
    register(
        Op(
            name="delete",
            label="Delete",
            modes=ALL_MODES,
            run=_delete,
            enabled=lambda doc: bool(doc.selection),
            key="Del",
            separator_before=True,
        )
    )


_register_defaults()
