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

from collections.abc import Callable
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
    """
    from .clay.elements import OpError

    if not op.enabled(doc):
        return False
    values = defaults_for(op) | params
    try:
        result = op.run(ctx, doc, **values)
    except OpError as error:
        toast(ctx, str(error))
        return False
    # An op that refuses *per object* -- ``run_mesh_op`` toasts and carries on
    # to the next one -- says so by returning False rather than by raising, so
    # a caller still learns that nothing happened.
    return result is not False


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


def _smooth(ctx: Any, doc: Any, levels: float = 1.0, **_: Any) -> None:
    """Catmull-Clark over every selected object, whatever the mode.

    It ignores the element selection deliberately -- a smoothing subdivision
    moves the original vertices, and moving only some of them tears the surface
    along the edge of the selection. See ``ops_subdiv.catmull_clark``.
    """
    from .clay import elements as el
    from .clay import ops_subdiv

    for uid in list(doc.selection):
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            continue
        try:
            mesh, sel = ops_subdiv.catmull_clark(
                obj.mesh, el.empty(), levels=int(levels)
            )
        except el.OpError as error:
            toast(ctx, str(error))
            continue
        doc.set_mesh(uid, mesh, select=sel)


def _duplicate(ctx: Any, doc: Any, **_: Any) -> None:
    from . import clay_mode

    clay_mode._duplicate_selection(ctx, clay_mode.ensure(ctx), doc)


def _bake(ctx: Any, doc: Any, **_: Any) -> None:
    """Fold each selected object's transform into its geometry.

    Two steps rather than one compound: the mesh and the transform are separate
    edits in this document's vocabulary, and keeping them separate means an undo
    of a bake is two presses rather than a compound type that exists for one
    button.
    """
    from .clay import ops as clay_ops_geom

    del ctx
    for uid in list(doc.selection):
        baked = clay_ops_geom.bake_transform(doc.by_uid(uid))
        doc.set_mesh(uid, baked.mesh)
        doc.set_transform(
            uid,
            translation=baked.translation,
            rotation=baked.rotation,
            scale=baked.scale,
        )


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


def mirror(ctx: Any, doc: Any, axis: int, **_: Any) -> None:
    from .clay import ops as clay_ops_geom

    del ctx
    for uid in list(doc.selection):
        doc.set_mesh(uid, clay_ops_geom.mirror(doc.by_uid(uid), axis).mesh)


def _delete(ctx: Any, doc: Any, **_: Any) -> None:
    from . import clay_mode

    clay_mode._delete(ctx, doc)


def _frame(ctx: Any, doc: Any, **_: Any) -> None:
    view = getattr(ctx, "clay_view", None)
    if view is not None:
        view.frame_selection(doc)


def _select_all(ctx: Any, doc: Any, **_: Any) -> None:
    from . import clay_mode

    del ctx
    clay_mode._select_all(doc)


def _select_none(ctx: Any, doc: Any, **_: Any) -> None:
    del ctx
    doc.clear_element_sel()


def _invert(ctx: Any, doc: Any, **_: Any) -> None:
    from . import clay_mode

    del ctx
    clay_mode._invert(doc)


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
            modes=("face",),
            run=_element("ops_topo.extrude_faces"),
            enabled=in_mode("face"),
            key="E",
            separator_before=True,
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
