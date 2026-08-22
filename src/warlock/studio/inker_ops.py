"""One registry of everything the Inker can *do*, and no imgui anywhere in it.

``clay_ops``'s idea, for the editor that needed it more. Inker's verbs were
spread across five surfaces -- the bridge panel's five blocks, the layers
pane's icon strip, the tools pane's selection section, the canvas's file row
and ``inker_mode.handle_key`` -- and each surface decided for itself whether a
verb was available and what to say when it was not. Five lists means five
answers to "can I flatten while a save is in flight", and the interesting one
is always the one nobody updated.

So there is one list. :data:`OPS` holds what is invocable, each entry carrying
the menu it belongs to, the key that fires it, the context it applies in, and
the sentence to show when it cannot run. The menu strip renders it, the context
bar renders a slice of it, a status-bar tip's remedy *names* an entry in it,
and none of them decides anything.

**Nothing here imports imgui**, which is what keeps it testable: "Flatten is
greyed out on a one-layer document" is a plain assertion rather than a
screenshot. It cannot live in ``inker_mode`` either -- that module is 3.8k
lines and ``main.shortcut_sections`` must stay free of imgui and pygame -- nor
in ``inker_state``, which ``inker_mode`` imports. ``inker_mode`` is therefore
imported *lazily, inside* each ``run``, exactly as ``clay_ops`` does.

**A dialog is a pending flag, not a call.** An op that needs a popup writes
``state.pending_dialog``; the pane that owns the popup opens it on the next
frame. An imgui popup belongs to the window that began it, so a registry that
tried to open one would either have to know about windows or open it in the
wrong one -- which is the bug where a menu row silently does nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MENUS",
    "OPS",
    "Op",
    "Param",
    "by_key",
    "defaults_for",
    "get",
    "menu",
    "register",
    "run",
]

#: The menu strip, in order -- which is Aseprite's order.
MENUS: tuple[str, ...] = (
    "File",
    "Edit",
    "Sprite",
    "Layer",
    "Frame",
    "Select",
    "View",
)

#: The key contexts an op can be bound in; see ``inker_state.key_context``.
#: ``""`` means every context, which is what a menu row wants when no modal
#: gesture is in flight.
def _contexts() -> tuple[str, ...]:
    """The context names, read from the table that decides them.

    Derived rather than written out: a name here that ``key_context`` can never
    return is a binding that can never fire, and the failure would be silence.
    """
    from .inker_state import KEY_CONTEXTS

    return ("", *(name for name, _applies in KEY_CONTEXTS))


CONTEXTS: tuple[str, ...] = _contexts()


@dataclass(frozen=True)
class Param:
    """One number an op takes, and everything a dialog needs to offer it."""

    name: str
    label: str
    default: float
    low: float = 0.0
    high: float = 1e6
    step: float = 1.0
    integer: bool = True
    warn: str = ""


def _always(state: Any, tab: Any) -> bool:
    return True


@dataclass(frozen=True)
class Op:
    """One invocable operation.

    ``run(ctx, tab, **params)`` does the work. ``enabled(state, tab)`` answers
    whether it can right now and is what greys the menu row; ``reason`` is the
    sentence shown when it cannot, and lives here rather than at each call site
    so the menu, the context bar and a tip's remedy button cannot disagree
    about *why*.

    ``key`` is the shortcut label as well as the binding, so a menu row and the
    shortcuts sheet are the same string. ``context`` is which key context the
    binding lives in; ``""`` is "wherever a document is open".
    """

    name: str
    label: str
    run: Callable[..., Any]
    menu: str = ""
    key: str = ""
    context: str = ""
    enabled: Callable[[Any, Any], bool] = _always
    reason: str = ""
    separator_before: bool = False
    params: tuple[Param, ...] = field(default=())
    hint: str = ""


OPS: list[Op] = []


def register(op: Op) -> Op:
    """Add an op, refusing a duplicate name, an unknown menu or context."""

    if any(existing.name == op.name for existing in OPS):
        raise ValueError(f"an op named {op.name!r} is already registered")
    if op.menu and op.menu not in MENUS:
        raise ValueError(f"{op.name!r} names a menu that does not exist: {op.menu!r}")
    if op.context not in CONTEXTS:
        raise ValueError(f"{op.name!r} names an unknown key context: {op.context!r}")
    OPS.append(op)
    return op


def menu(name: str) -> list[Op]:
    """Every op in one menu, in registration order -- which is the drawn order."""

    return [op for op in OPS if op.menu == name]


def get(name: str) -> Op:
    for op in OPS:
        if op.name == name:
            return op
    raise KeyError(f"no op named {name!r}")


def by_key(key: str, context: str = "") -> Op | None:
    """The op *key* fires, preferring one bound to *context*.

    Context-specific first, then the context-free ops: that ordering is what
    lets Enter mean "close the polygon" inside a gesture and "play" outside one
    without either binding having to know the other exists.
    """

    if context:
        for op in OPS:
            if op.key == key and op.context == context:
                return op
    for op in OPS:
        if op.key == key and not op.context:
            return op
    return None


def defaults_for(op: Op) -> dict[str, float]:
    """The op's parameters at their defaults, as a fresh dict."""

    return {param.name: param.default for param in op.params}


def run(ctx: Any, op: Op, **params: Any) -> bool:
    """Invoke an op against the active tab. -> whether it ran.

    The choke point all four surfaces funnel through, which is what lets the
    ``enabled`` gate and the declared parameter ranges be enforced once. A
    missing parameter falls back to its default, so the key path and a test get
    what the dialog would have produced with its fields untouched.

    A refused op **says why** rather than doing nothing: the reason is already
    written down for the greyed menu row, and the keyboard is exactly the
    surface where the user cannot see that the row was grey.
    """

    state = getattr(ctx.state, "inker", None)
    if state is None:
        return False
    tab = state.active
    if not op.enabled(state, tab):
        if op.reason:
            state.say(op.reason)
        return False
    values = defaults_for(op) | params
    for param in op.params:
        value = min(max(float(values[param.name]), param.low), param.high)
        values[param.name] = int(value) if param.integer else value
    return op.run(ctx, tab, **values) is not False


# --- the predicates ---------------------------------------------------------
#
# Named rather than written out as lambdas at each registration: the same few
# questions gate two thirds of the registry, and a copy of one that drifted is
# exactly the class of defect this module exists to remove.

BUSY = "The document is busy -- a save, an export or playback is still running."
NO_DOC = "Nothing is open."
NO_SELECTION = "Nothing is selected."


def has_doc(state: Any, tab: Any) -> bool:
    return tab is not None


def ready(state: Any, tab: Any) -> bool:
    """A document that can be restructured right now.

    ``busy`` rather than ``saving``: playback is the second reason a document
    may not be restructured, and it is the same list of verbs for the same
    reason (``inker_mode._MUTATING_CTRL``). A floating transform counts too --
    a save mid-transform would commit it with no confirm.
    """

    return tab is not None and not tab.busy and not state.transforming


def has_selection(state: Any, tab: Any) -> bool:
    return tab is not None and tab.doc.mask is not None


def can_reselect(state: Any, tab: Any) -> bool:
    # Off the *memory* rather than off "there is no selection": the useful case
    # is re-selecting after something else was selected, and a mask the canvas
    # has outgrown is refused by the engine.
    return tab is not None and tab.doc._last_mask is not None


def animated(state: Any, tab: Any) -> bool:
    return tab is not None and tab.doc.anim is not None


def not_animated(state: Any, tab: Any) -> bool:
    return tab is not None and tab.doc.anim is None


def can_undo(state: Any, tab: Any) -> bool:
    return tab is not None and tab.doc.history.can_undo


def can_redo(state: Any, tab: Any) -> bool:
    return tab is not None and tab.doc.history.can_redo


def many_layers(state: Any, tab: Any) -> bool:
    return tab is not None and len(tab.doc.stack) > 1


def can_merge_down(state: Any, tab: Any) -> bool:
    """Whether there is a layer under the active one, inside its own group."""

    if tab is None:
        return False
    doc = tab.doc
    index = doc.stack.active_index
    if index <= 0:
        return False
    return doc.group_of.get(_uid_at(doc, index)) == doc.group_of.get(
        _uid_at(doc, index - 1)
    )


def _uid_at(doc: Any, index: int) -> int | None:
    try:
        return doc.stack[index].uid
    except (IndexError, AttributeError):  # pragma: no cover - defensive
        return None


# --- the runners ------------------------------------------------------------


def dialog(name: str) -> Callable[..., Any]:
    """An op that asks a pane to open a popup. See the module docstring."""

    def _open(ctx: Any, tab: Any, **_: Any) -> None:
        ctx.state.inker.pending_dialog = name

    return _open


def _mode(verb: str, **kwargs: Any) -> Callable[..., Any]:
    """An op that is one ``inker_mode`` function of ``(ctx, tab)``."""

    def _run(ctx: Any, tab: Any, **params: Any) -> Any:
        from . import inker_mode

        return getattr(inker_mode, verb)(ctx, tab, **(kwargs | params))

    return _run


def _mode_ctx(verb: str) -> Callable[..., Any]:
    """An op that is one ``inker_mode`` function of ``(ctx)`` alone."""

    def _run(ctx: Any, tab: Any, **_: Any) -> Any:
        from . import inker_mode

        return getattr(inker_mode, verb)(ctx)

    return _run


def _doc(verb: str, *args: Any, **kwargs: Any) -> Callable[..., Any]:
    """An op that is one method on the document."""

    def _run(ctx: Any, tab: Any, **params: Any) -> Any:
        return getattr(tab.doc, verb)(*args, **(kwargs | params))

    return _run


def _state_doc(verb: str, attr: str) -> Callable[..., Any]:
    """A document method taking one app-level tool setting as its argument."""

    def _run(ctx: Any, tab: Any, **_: Any) -> Any:
        return getattr(tab.doc, verb)(getattr(ctx.state.inker, attr))

    return _run


def _view(verb: str, *args: Any) -> Callable[..., Any]:
    def _run(ctx: Any, tab: Any, **_: Any) -> Any:
        from . import inker_state

        return getattr(inker_state, verb)(tab.view, *args)

    return _run


# --- File -------------------------------------------------------------------

register(Op("new", "New...", dialog("new-canvas"), menu="File", key="Ctrl+N"))
register(Op("open", "Open...", _mode_ctx("ask_open"), menu="File", key="Ctrl+O"))
register(
    Op(
        "save",
        "Save",
        _mode("save"),
        menu="File",
        key="Ctrl+S",
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)
register(
    Op(
        "save_as",
        "Save As...",
        _mode("save_as"),
        menu="File",
        key="Ctrl+Shift+S",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "export_png",
        "Export PNG...",
        _mode("export_png"),
        menu="File",
        key="Ctrl+Shift+E",
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)
register(
    Op(
        "export_sheet",
        "Export sprite sheet...",
        _mode("export_sheet"),
        menu="File",
        enabled=lambda state, tab: ready(state, tab) and animated(state, tab),
        reason="This drawing has no frames yet -- Animate it first.",
    )
)
register(
    Op(
        "export_gif",
        "Export GIF...",
        _mode("export_gif"),
        menu="File",
        enabled=lambda state, tab: ready(state, tab) and animated(state, tab),
        reason="This drawing has no frames yet -- Animate it first.",
    )
)
register(
    Op(
        "export_slices",
        "Export slices...",
        _mode("export_slices"),
        menu="File",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "import_sheet",
        "Import sprite sheet...",
        _mode_ctx("ask_import_sheet"),
        menu="File",
        separator_before=True,
    )
)
register(
    Op(
        "import_aseprite",
        "Import .aseprite...",
        _mode_ctx("ask_import_aseprite"),
        menu="File",
    )
)
register(
    Op(
        "import_tileset",
        "Import tileset (.tsx)...",
        _mode("import_tileset"),
        menu="File",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "save_as_reference",
        "Save as reference",
        _mode("save_as_reference"),
        menu="File",
        key="Ctrl+E",
        enabled=lambda state, tab: ready(state, tab) and not tab.linked,
        reason=(
            "This document is already in the library -- it is a reference "
            "opened for editing, so Ctrl+S is the write it wants."
        ),
        separator_before=True,
    )
)
register(
    Op(
        "send_to_3d",
        "Make 3D",
        _mode("send_to_3d"),
        menu="File",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "add_to_packwright",
        "Add to Packwright",
        lambda ctx, tab, **_: _packwright(ctx, tab),
        menu="File",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "revert",
        "Revert to original",
        _mode("revert"),
        menu="File",
        enabled=lambda state, tab: ready(state, tab) and tab.linked and tab.has_original,
        reason=(
            "There is no original kept for this document: it is not a "
            "reference, or it has never been edited."
        ),
    )
)
register(
    Op(
        "close",
        "Close",
        _mode("request_close"),
        menu="File",
        key="Ctrl+W",
        enabled=has_doc,
        reason=NO_DOC,
        separator_before=True,
    )
)


def _packwright(ctx: Any, tab: Any) -> Any:
    from . import packwright_mode

    return packwright_mode.add_inker_document(ctx, tab)


# --- Edit -------------------------------------------------------------------

register(
    Op(
        "undo",
        "Undo",
        _doc("undo"),
        menu="Edit",
        key="Ctrl+Z",
        enabled=can_undo,
        reason="Nothing to undo yet.",
    )
)
register(
    Op(
        "redo",
        "Redo",
        _doc("redo"),
        menu="Edit",
        key="Ctrl+Y",
        enabled=can_redo,
        reason="Nothing to redo: this is the newest step.",
    )
)
register(
    Op(
        "cut",
        "Cut",
        _doc("cut"),
        menu="Edit",
        key="Ctrl+X",
        enabled=has_selection,
        reason=NO_SELECTION,
        separator_before=True,
    )
)
register(
    Op(
        "copy",
        "Copy",
        _doc("copy"),
        menu="Edit",
        key="Ctrl+C",
        enabled=has_selection,
        reason=NO_SELECTION,
    )
)
register(
    Op(
        "paste",
        "Paste",
        lambda ctx, tab, **_: _paste(ctx, tab, as_layer=False),
        menu="Edit",
        key="Ctrl+V",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "paste_as_layer",
        "Paste as new layer",
        lambda ctx, tab, **_: _paste(ctx, tab, as_layer=True),
        menu="Edit",
        key="Ctrl+Shift+V",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "transform",
        "Free transform",
        _mode("begin_transform"),
        menu="Edit",
        key="Ctrl+T",
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)
register(
    Op(
        "capture_brush",
        "New brush from selection",
        _mode_ctx("capture_brush"),
        menu="Edit",
        key="Ctrl+B",
        enabled=has_selection,
        reason=NO_SELECTION,
    )
)
register(
    Op(
        "clear_brush",
        "Drop the captured brush",
        _mode_ctx("clear_brush"),
        menu="Edit",
        enabled=lambda state, tab: state.stamp is not None,
        reason="No brush has been captured.",
    )
)
register(
    Op(
        "copy_merged",
        "Copy merged",
        _doc("copy_merged"),
        menu="Edit",
        key="Ctrl+Shift+C",
        enabled=has_selection,
        reason=NO_SELECTION,
        hint=(
            "What is visible inside the selection rather than one layer of it: "
            "an ordinary copy moves a drawing between layers, this moves a part "
            "of the picture between documents."
        ),
    )
)
register(
    Op(
        "new_from_selection",
        "New document from selection",
        _mode("new_from_selection"),
        menu="Edit",
        enabled=has_selection,
        reason=NO_SELECTION,
    )
)
register(
    Op(
        "fill_selection",
        "Fill selection",
        lambda ctx, tab, **_: tab.doc.fill_selection(ctx.state.inker.fg),
        menu="Edit",
        enabled=has_selection,
        reason=NO_SELECTION,
        separator_before=True,
    )
)
register(
    Op(
        "stroke_selection",
        "Stroke selection...",
        lambda ctx, tab, **params: tab.doc.stroke_selection(
            ctx.state.inker.fg, int(params["width"])
        ),
        menu="Edit",
        enabled=has_selection,
        reason=NO_SELECTION,
        params=(Param("width", "Width", 1, 1, 32),),
        hint=(
            "The selection's own outline, drawn inside it -- an outline that "
            "grew past the edge would paint pixels you did not select."
        ),
    )
)
register(
    Op(
        "shift_selected",
        "Shift pixels...",
        lambda ctx, tab, **params: tab.doc.shift_selected(
            int(params["dx"]), int(params["dy"])
        ),
        menu="Edit",
        enabled=has_selection,
        reason=NO_SELECTION,
        params=(
            Param("dx", "Right", 1, -4096, 4096),
            Param("dy", "Down", 0, -4096, 4096),
        ),
        hint=(
            "Moves the selected pixels and leaves a hole behind, which is what "
            "the move tool does -- it goes through the same floating buffer, so "
            "it is one undo step."
        ),
    )
)
register(
    Op(
        "filter",
        "Filter...",
        dialog("inker-filter"),
        menu="Edit",
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)


def _paste(ctx: Any, tab: Any, *, as_layer: bool) -> Any:
    from . import inker_mode

    inker_mode.paste_from_os(ctx, tab)
    if as_layer:
        return tab.doc.paste_as_layer()
    result = tab.doc.paste()
    ctx.state.inker.set_tool("move")
    return result


# --- Sprite -----------------------------------------------------------------

register(
    Op(
        "resize",
        "Canvas size...",
        dialog("inker-resize"),
        menu="Sprite",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "crop_to_selection",
        "Crop to selection",
        _doc("crop_to_selection"),
        menu="Sprite",
        enabled=lambda state, tab: ready(state, tab) and has_selection(state, tab),
        reason=NO_SELECTION,
    )
)
register(
    Op(
        "flip_h",
        "Flip horizontal",
        _doc("flip", "horizontal"),
        menu="Sprite",
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)
register(
    Op(
        "flip_v",
        "Flip vertical",
        _doc("flip", "vertical"),
        menu="Sprite",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "rotate90",
        "Rotate 90 degrees",
        _doc("rotate90"),
        menu="Sprite",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "convert_to_tilemap",
        "Convert to tilemap...",
        dialog("inker-to-tilemap"),
        menu="Sprite",
        enabled=lambda state, tab: _tiles().can_convert(state, tab),
        reason="The active layer is already a tilemap layer.",
        separator_before=True,
    )
)
register(
    Op(
        "tile_auto",
        "Tileset follows the drawing",
        lambda ctx, tab, **_: setattr(tab.doc, "tile_behavior", "auto"),
        menu="Sprite",
        enabled=lambda state, tab: (
            tab is not None
            and tab.doc.active_tilemap_uid() is not None
            and tab.doc.tile_behavior != "auto"
        ),
        reason=(
            "This is not a tilemap layer, or it already updates its tileset "
            "as you draw."
        ),
        hint=(
            "Auto: painting on a tilemap layer edits the tile under the brush. "
            "Manual leaves the tileset alone and reverts the cell instead."
        ),
    )
)
register(
    Op(
        "convert_colour_mode",
        "Colour mode...",
        dialog("inker-convert"),
        menu="Sprite",
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)


# --- Layer ------------------------------------------------------------------

register(
    Op(
        "add_layer",
        "New layer",
        _doc("add_layer"),
        menu="Layer",
        key="Ctrl+Shift+N",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "duplicate_layer",
        "Duplicate layer",
        _doc("duplicate_layer"),
        menu="Layer",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "delete_layer",
        "Delete layer",
        _doc("remove_layer"),
        menu="Layer",
        enabled=lambda state, tab: ready(state, tab) and many_layers(state, tab),
        reason="A document keeps at least one layer.",
    )
)
register(
    Op(
        "rename_layer",
        "Rename layer...",
        dialog("inker-rename-layer"),
        menu="Layer",
        enabled=has_doc,
        reason=NO_DOC,
    )
)
register(
    Op(
        "group_layers",
        "Group layer",
        _doc("group_layers"),
        menu="Layer",
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)
register(
    Op(
        "merge_down",
        "Merge down",
        _doc("merge_down"),
        menu="Layer",
        enabled=lambda state, tab: ready(state, tab) and can_merge_down(state, tab),
        reason="There is no layer under this one to merge into.",
    )
)
register(
    Op(
        "flatten",
        "Flatten",
        _doc("flatten_layers"),
        menu="Layer",
        enabled=lambda state, tab: ready(state, tab) and many_layers(state, tab),
        reason="There is only one layer.",
    )
)
register(
    Op(
        "layer_up",
        "Move layer up",
        lambda ctx, tab, **_: _move_layer(tab, 1),
        menu="Layer",
        key="Ctrl+Shift+Up",
        enabled=lambda state, tab: ready(state, tab) and many_layers(state, tab),
        reason="There is only one layer.",
        separator_before=True,
    )
)
register(
    Op(
        "layer_down",
        "Move layer down",
        lambda ctx, tab, **_: _move_layer(tab, -1),
        menu="Layer",
        key="Ctrl+Shift+Down",
        enabled=lambda state, tab: ready(state, tab) and many_layers(state, tab),
        reason="There is only one layer.",
    )
)
register(
    Op(
        "show_layer",
        "Show this layer",
        lambda ctx, tab, **_: tab.doc.set_layer_props(
            tab.doc.stack.active_index, visible=True
        ),
        menu="Layer",
        enabled=lambda state, tab: tab is not None and not tab.doc.stack.active.visible,
        reason="This layer is already visible.",
        separator_before=True,
    )
)
register(
    Op(
        "layer_properties",
        "Properties...",
        dialog("inker-layer-properties"),
        menu="Layer",
        enabled=has_doc,
        reason=NO_DOC,
        separator_before=True,
    )
)


def _move_layer(tab: Any, delta: int) -> Any:
    index = tab.doc.stack.active_index
    return tab.doc.move_layer(index, index + delta)


# --- Frame ------------------------------------------------------------------

register(
    Op(
        "animate",
        "Animate this drawing",
        _mode("animate"),
        menu="Frame",
        enabled=lambda state, tab: ready(state, tab) and not_animated(state, tab),
        reason="This document is already animated.",
    )
)
register(
    Op(
        "play",
        "Play / stop",
        _mode("toggle_play"),
        menu="Frame",
        key="Enter",
        context="Normal",
        enabled=animated,
        reason="This drawing has no frames yet -- Animate it first.",
    )
)
register(
    Op(
        "next_frame",
        "Next frame",
        _mode("step_frame", delta=1),
        menu="Frame",
        key=".",
        enabled=animated,
        reason="This drawing has no frames yet -- Animate it first.",
        separator_before=True,
    )
)
register(
    Op(
        "prev_frame",
        "Previous frame",
        _mode("step_frame", delta=-1),
        menu="Frame",
        key=",",
        enabled=animated,
        reason="This drawing has no frames yet -- Animate it first.",
    )
)


# --- Select -----------------------------------------------------------------

register(
    Op(
        "select_all",
        "All",
        _doc("select_all"),
        menu="Select",
        key="Ctrl+A",
        enabled=has_doc,
        reason=NO_DOC,
    )
)
register(
    Op(
        "deselect",
        "Deselect",
        _doc("deselect"),
        menu="Select",
        key="Ctrl+D",
        enabled=has_selection,
        reason=NO_SELECTION,
    )
)
register(
    Op(
        "reselect",
        "Reselect",
        _doc("reselect"),
        menu="Select",
        key="Ctrl+Shift+D",
        enabled=can_reselect,
        reason="Nothing has been deselected yet.",
    )
)
register(
    Op(
        "invert_selection",
        "Inverse",
        _doc("invert_selection"),
        menu="Select",
        key="Ctrl+Shift+I",
        enabled=has_doc,
        reason=NO_DOC,
    )
)
register(
    Op(
        "select_layer_alpha",
        "This layer's pixels",
        _doc("select_layer_alpha"),
        menu="Select",
        enabled=has_doc,
        reason=NO_DOC,
        separator_before=True,
    )
)
register(
    Op(
        "select_colour_range",
        "Colour range",
        lambda ctx, tab, **_: _colour_range(ctx, tab),
        menu="Select",
        enabled=has_doc,
        reason=NO_DOC,
        hint=(
            "Every pixel close to the foreground colour, anywhere on the "
            "canvas -- not contiguous, so one press takes a palette entry "
            "wherever it was used. The tolerance is the magic wand's."
        ),
    )
)
register(
    Op(
        "copy_to_layer",
        "Copy to new layer",
        _doc("layer_from_selection", cut=False),
        menu="Select",
        key="Ctrl+J",
        enabled=has_selection,
        reason=NO_SELECTION,
        separator_before=True,
    )
)
register(
    Op(
        "move_to_layer",
        "Move to new layer",
        _doc("layer_from_selection", cut=True),
        menu="Select",
        key="Ctrl+Shift+J",
        enabled=has_selection,
        reason=NO_SELECTION,
    )
)
register(
    Op(
        "feather",
        "Feather...",
        lambda ctx, tab, **params: tab.doc.feather_selection(params["radius"]),
        menu="Select",
        enabled=has_selection,
        reason=NO_SELECTION,
        params=(Param("radius", "Radius", 2.0, 0.0, 32.0, 0.5, integer=False),),
        separator_before=True,
    )
)
register(
    Op(
        "grow",
        "Grow...",
        lambda ctx, tab, **params: tab.doc.grow_selection(params["steps"]),
        menu="Select",
        enabled=has_selection,
        reason=NO_SELECTION,
        params=(Param("steps", "Pixels", 2, 1, 32),),
    )
)
register(
    Op(
        "shrink",
        "Shrink...",
        lambda ctx, tab, **params: tab.doc.shrink_selection(params["steps"]),
        menu="Select",
        enabled=has_selection,
        reason=NO_SELECTION,
        params=(Param("steps", "Pixels", 2, 1, 32),),
    )
)
register(
    Op(
        "border",
        "Border...",
        lambda ctx, tab, **params: tab.doc.border_selection(params["steps"]),
        menu="Select",
        enabled=has_selection,
        reason=NO_SELECTION,
        params=(Param("steps", "Pixels", 2, 1, 32),),
        hint=(
            "Replaces the selection with the band that many pixels either "
            "side of its edge -- fill it and you have stroked the outline."
        ),
    )
)


def _colour_range(ctx: Any, tab: Any) -> Any:
    state = ctx.state.inker
    # The *wand's* tolerance, by name: ``tool_options`` follows the tool in
    # hand, and this op is invocable whatever the tool is, so reading the
    # current tool's copy would give the pencil's number. One tolerance the
    # user set, one meaning of "similar" (``selection.colour_distance``).
    tolerance = state.options_for("wand")["wand_tolerance"]
    return tab.doc.select_colour_range(state.fg, tolerance=tolerance)


# --- View -------------------------------------------------------------------

register(
    Op(
        "fit_view",
        "Fit in window",
        lambda ctx, tab, **_: setattr(tab.view, "fitted", False),
        menu="View",
        key="Ctrl+0",
        enabled=has_doc,
        reason=NO_DOC,
    )
)
register(
    Op(
        "zoom_100",
        "Actual size",
        lambda ctx, tab, **_: setattr(tab.view, "pending_zoom", 1.0),
        menu="View",
        key="Ctrl+1",
        enabled=has_doc,
        reason=NO_DOC,
    )
)
register(
    Op(
        "rotate_view",
        "Rotate the view",
        _view("rotate_view", 1),
        menu="View",
        key="Ctrl+4",
        enabled=has_doc,
        reason=NO_DOC,
        separator_before=True,
    )
)
register(
    Op(
        "flip_view",
        "Mirror the view",
        _view("flip_view"),
        menu="View",
        key="Ctrl+5",
        enabled=has_doc,
        reason=NO_DOC,
    )
)
register(
    Op(
        "toggle_grid",
        "Grid",
        lambda ctx, tab, **_: _toggle(ctx, "grid"),
        menu="View",
        enabled=has_doc,
        reason=NO_DOC,
        separator_before=True,
    )
)
register(
    Op(
        "toggle_snap",
        "Snap to grid",
        lambda ctx, tab, **_: _toggle(ctx, "grid_snap"),
        menu="View",
        enabled=has_doc,
        reason=NO_DOC,
    )
)
register(
    Op(
        "toggle_rulers",
        "Rulers",
        lambda ctx, tab, **_: _toggle(ctx, "rulers"),
        menu="View",
        enabled=has_doc,
        reason=NO_DOC,
    )
)


def _tiles() -> Any:
    """The tile pane's own predicate module, imported lazily.

    The one place this registry reaches into ``panes/``, and only for a
    *question*: "is the active layer already a tilemap" is a tile fact, and
    duplicating it here is how the two would come to disagree. Lazy, because
    ``panes`` imports imgui and this module may not.
    """
    from .panes import inker_tiles

    return inker_tiles


def _toggle(ctx: Any, attr: str) -> None:
    """Flip a persisted view preference, and write it down.

    Persisted on the change rather than at quit: the grid and the rulers are
    how the user likes to *see*, and a preference that resets on the next
    launch is a control they have to rediscover.
    """
    from . import inker_mode

    state = ctx.state.inker
    setattr(state, attr, not getattr(state, attr))
    inker_mode.persist(ctx)
