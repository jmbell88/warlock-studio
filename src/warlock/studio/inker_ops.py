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

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MENUS",
    "OPS",
    "ACTION_MODIFIERS",
    "BINDINGS",
    "BINDING_CONTEXTS",
    "ActionModifier",
    "action_active",
    "Binding",
    "Op",
    "Param",
    "by_key",
    "binding_target",
    "bindings_for",
    "defaults_for",
    "get",
    "menu",
    "manifest",
    "parse_shortcuts",
    "register",
    "resolve_binding",
    "run",
    "set_shortcuts",
    "shortcut_for",
    "shortcuts_json",
]

#: The menu strip, in order -- which is Aseprite's order.
MENUS: tuple[str, ...] = (
    "File",
    "Edit",
    "Sprite",
    "Layer",
    "Frame",
    "Sheet",
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
    #: The sentence shown when ``enabled`` says no. A **callable** is allowed
    #: alongside a plain string for the ops that can be refused for more than
    #: one reason: an op gated on both "there is something to undo" and "the
    #: document is not being saved" has two answers, and a single string would
    #: have to pick the wrong one half the time. See ``reason_for``.
    reason: str | Callable[[Any, Any], str] = ""
    separator_before: bool = False
    params: tuple[Param, ...] = field(default=())
    hint: str = ""
    #: ``checked(state, tab)`` -> whether the row draws a tick, for the ops
    #: that are a document *state* rather than an action. None -- the default
    #: every other registration takes -- is "never ticked", which is what the
    #: Inker menu adapter used to hardcode for all of them.
    checked: Callable[[Any, Any], bool] | None = None


BINDING_KINDS = frozenset({"command", "tool", "action_modifier"})
BINDING_TRIGGERS = frozenset({"press", "hold", "wheel", "drag"})
BINDING_CONTEXTS = (
    *CONTEXTS,
    "TranslatingSelection",
    "ScalingSelection",
    "RotatingSelection",
)


def canonical_chord(chord: str) -> str:
    """One platform-neutral spelling, independent of modifier press order."""

    pieces = [piece.strip() for piece in str(chord).split("+") if piece.strip()]
    modifiers = [name for name in ("Ctrl", "Alt", "Shift") if name in pieces]
    keys = [piece for piece in pieces if piece not in {"Ctrl", "Alt", "Shift"}]
    return "+".join((*modifiers, *keys))


@dataclass(frozen=True)
class Binding:
    """One input gesture mapped to one registry target.

    Commands no longer own the binding relationship.  This separate record is
    what permits two chords for one command, the same chord in two contexts,
    and hold/wheel/drag gestures without inventing pseudo-commands.  ``Op.key``
    remains a compatibility view of the primary default while callers migrate.
    """

    target: str
    chord: str
    kind: str = "command"
    context: str = ""
    trigger: str = "press"
    priority: int = 0

    def __post_init__(self) -> None:
        if self.kind not in BINDING_KINDS:
            raise ValueError(f"unknown binding kind {self.kind!r}")
        if self.trigger not in BINDING_TRIGGERS:
            raise ValueError(f"unknown binding trigger {self.trigger!r}")
        if self.context not in BINDING_CONTEXTS:
            raise ValueError(f"unknown binding context {self.context!r}")
        if not self.target or not self.chord:
            raise ValueError("a binding needs a target and a chord")
        object.__setattr__(self, "chord", canonical_chord(self.chord))


@dataclass(frozen=True)
class ActionModifier:
    """A contextual held gesture which alters another action.

    These records are descriptive and dispatchable data rather than commands:
    a shape's Shift constraint only has meaning while that shape is in flight.
    Canvas gesture code can therefore consume the same registry that the
    shortcut editor and compatibility manifest expose.
    """

    name: str
    label: str
    context: str
    description: str


def binding_target(kind: str, target: str) -> str:
    """Stable JSON key for a command/tool/modifier target."""

    return f"{kind}:{target}"


def reason_for(op: Op, state: Any, tab: Any) -> str:
    """*op*'s refusal sentence right now, resolving the callable form."""
    reason = op.reason
    return reason(state, tab) if callable(reason) else reason


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


def _coerce_binding(value: Any, *, target_key: str = "") -> Binding | None:
    """Validate one user-authored binding without trusting settings JSON."""

    if isinstance(value, Binding):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        kind, target = target_key.split(":", 1)
    except ValueError:
        kind = str(value.get("kind", ""))
        target = str(value.get("target", ""))
    try:
        return Binding(
            target=target,
            chord=str(value.get("chord", "")),
            kind=kind,
            context=str(value.get("context", "")),
            trigger=str(value.get("trigger", "press")),
            priority=int(value.get("priority", 100 if target_key else 0)),
        )
    except (TypeError, ValueError):
        return None


def _normalise_overrides(raw: Any) -> dict[str, tuple[Binding, ...]]:
    """Return the valid part of a persisted shortcut override mapping.

    Presence of a target with an empty list deliberately means *unbound*.
    Unknown targets are discarded so a hand-edited settings file cannot create
    a command surface that the application has no way to execute.
    """

    if not isinstance(raw, Mapping):
        return {}
    known = {binding_target(item.kind, item.target) for item in BINDINGS}
    out: dict[str, tuple[Binding, ...]] = {}
    for target_key, values in raw.items():
        if target_key not in known or not isinstance(values, Sequence) or isinstance(values, str):
            continue
        parsed = tuple(
            binding
            for value in values
            if (binding := _coerce_binding(value, target_key=target_key)) is not None
        )
        out[str(target_key)] = parsed
    return out


def bindings_for(overrides: Any = None) -> tuple[Binding, ...]:
    """The effective binding table after target-scoped user overrides."""

    changed = _normalise_overrides(overrides)
    if not changed:
        return BINDINGS
    out = [
        binding
        for binding in BINDINGS
        if binding_target(binding.kind, binding.target) not in changed
    ]
    for target_key in sorted(changed):
        out.extend(changed[target_key])
    return tuple(out)


def resolve_binding(
    chord: str,
    context: str = "",
    overrides: Any = None,
    *,
    trigger: str = "press",
) -> Binding | None:
    """Resolve a gesture, with exact-context bindings ahead of global ones."""

    candidates = [
        binding
        for binding in bindings_for(overrides)
        if binding.chord == canonical_chord(chord) and binding.trigger == trigger
    ]
    candidates.sort(
        key=lambda binding: (
            binding.context == context and bool(context),
            not binding.context,
            binding.priority,
        ),
        reverse=True,
    )
    return next(
        (binding for binding in candidates if binding.context == context or not binding.context),
        None,
    )


def action_active(target: str, chord: str, context: str, overrides: Any = None) -> bool:
    """Whether a held action modifier is contained in the current chord.

    Shape modifiers compose (Ctrl+Shift means centre *and* square), while
    command resolution is exact.  Keeping that distinction here avoids every
    gesture reimplementing modifier-set arithmetic.
    """

    held = set(canonical_chord(chord).split("+"))
    return any(
        item.kind == "action_modifier"
        and item.target == target
        and item.context == context
        and item.trigger == "hold"
        and set(item.chord.split("+")).issubset(held)
        for item in bindings_for(overrides)
    )


def by_key(key: str, context: str = "", overrides: Any = None) -> Op | None:
    """Compatibility command lookup over the many-to-many binding registry."""

    binding = resolve_binding(key, context, overrides)
    if binding is None or binding.kind != "command":
        return None
    return get(binding.target)


def shortcut_for(
    kind: str,
    target: str,
    overrides: Any = None,
    *,
    context: str | None = None,
    trigger: str = "press",
) -> str:
    """Printable effective chords for a menu row or tooltip."""

    matches = [
        item.chord
        for item in bindings_for(overrides)
        if item.kind == kind
        and item.target == target
        and item.trigger == trigger
        and (context is None or item.context == context)
    ]
    return " or ".join(dict.fromkeys(matches))


def parse_shortcuts(payload: str | bytes | Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Read a versioned shortcut export and return validated JSON-ready overrides."""

    value = json.loads(payload) if isinstance(payload, str | bytes) else payload
    if not isinstance(value, Mapping) or value.get("version") != 1:
        raise ValueError("not an Inker shortcut file (version 1)")
    parsed = _normalise_overrides(value.get("overrides"))
    return {
        key: [
            {
                "chord": binding.chord,
                "context": binding.context,
                "trigger": binding.trigger,
                **({"priority": binding.priority} if binding.priority else {}),
            }
            for binding in bindings
        ]
        for key, bindings in parsed.items()
    }


def shortcuts_json(overrides: Any = None) -> str:
    """Portable, deterministic shortcut overrides for import/export."""

    parsed = _normalise_overrides(overrides)
    payload = {
        "version": 1,
        "target": "Aseprite 1.3.15.5 / Windows",
        "overrides": {
            key: [
                {
                    "chord": binding.chord,
                    "context": binding.context,
                    "trigger": binding.trigger,
                    **({"priority": binding.priority} if binding.priority else {}),
                }
                for binding in parsed[key]
            ]
            for key in sorted(parsed)
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def set_shortcuts(
    overrides: Any,
    kind: str,
    target: str,
    chords: Sequence[str],
    *,
    context: str = "",
    trigger: str = "press",
) -> dict[str, list[dict[str, Any]]]:
    """Replace one target's bindings and return a validated settings delta."""

    key = binding_target(kind, target)
    known = {binding_target(item.kind, item.target) for item in BINDINGS}
    if key not in known:
        raise KeyError(f"unknown binding target {key!r}")
    unique = tuple(dict.fromkeys(chord.strip() for chord in chords if chord.strip()))
    # Constructing records is the validation door for contexts and triggers.
    replacements = [Binding(target, chord, kind, context, trigger, 100) for chord in unique]
    current = {
        target_key: [
            {
                "chord": item.chord,
                "context": item.context,
                "trigger": item.trigger,
                **({"priority": item.priority} if item.priority else {}),
            }
            for item in values
        ]
        for target_key, values in _normalise_overrides(overrides).items()
    }
    preserved = [
        item
        for item in bindings_for(overrides)
        if item.kind == kind
        and item.target == target
        and (item.context != context or item.trigger != trigger)
    ]
    current[key] = [
        {
            "chord": item.chord,
            "context": item.context,
            "trigger": item.trigger,
            "priority": item.priority,
        }
        for item in (*preserved, *replacements)
    ]
    return current


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
        said = reason_for(op, state, tab)
        if said:
            state.say(said)
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


def when_ready(
    predicate: Callable[[Any, Any], bool], reason: str
) -> tuple[Callable[[Any, Any], bool], Callable[[Any, Any], str]]:
    """*predicate*, and only while the document can be restructured.

    Returns the ``enabled``/``reason`` pair together, because the two have to
    agree about which of the two refusals is being made: "there is nothing to
    undo" and "the document is being saved" are different sentences and a single
    string would say the wrong one half the time.

    This exists because the keyboard and the menu disagreed. ``handle_key``
    refuses ``_MUTATING_CTRL`` outright on a busy tab -- undo, redo, cut,
    select-all, deselect, invert and the two layer-from-selection verbs all
    restructure the stack or move the history head a save has already captured
    -- but the menu rows for those same verbs stayed live while ``ora.py`` was
    walking ``doc.stack`` on a task thread. The gate belongs on the op, where
    both doors read it.
    """

    def enabled(state: Any, tab: Any) -> bool:
        return ready(state, tab) and predicate(state, tab)

    def why(state: Any, tab: Any) -> str:
        return reason if ready(state, tab) else BUSY

    return enabled, why


def has_selection(state: Any, tab: Any) -> bool:
    return tab is not None and tab.doc.mask is not None


def _pattern(ctx: Any) -> dict[str, Any]:
    """The pattern keywords ``Edit > Fill`` and ``Edit > Stroke`` pass on.

    Read off the **bucket's** options rather than off whatever tool happens to
    be in hand, which is by definition a selection tool when these two are
    reached. "Fill with a pattern" is one setting in this app, so the menu item
    and a bucket click put down the same thing; a second switch on the menu
    would be a second answer to one question.
    """
    state = getattr(ctx.state, "inker", None)
    if state is None:
        return {}
    return {
        "pattern": state.pattern_for("fill"),
        "pattern_align": state.options_for("fill")["stamp_align"],
    }


def can_reselect(state: Any, tab: Any) -> bool:
    # Off the *memory* rather than off "there is no selection": the useful case
    # is re-selecting after something else was selected, and a mask the canvas
    # has outgrown is refused by the engine.
    return tab is not None and tab.doc._last_mask is not None


#: Said by every verb that needs frames to act on. One string, because six
#: rows saying it slightly differently is how one of them comes to be wrong.
NOT_ANIMATED = "This drawing has no frames yet -- Animate it first."


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
    return doc.group_of.get(_uid_at(doc, index)) == doc.group_of.get(_uid_at(doc, index - 1))


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


# ``_state_doc`` -- a document method taking one app-level tool setting as its
# argument -- was deleted on 2026-08-22 with zero callers, where its siblings
# ``_mode_ctx`` and ``_doc`` are used dozens of times.


def _own_palette(tab: Any) -> Any:
    """This frame's own table, or None. Guarded for a still document."""
    doc = getattr(tab, "doc", None)
    anim = None if doc is None else doc.anim
    if anim is None or not anim.frames:
        return None
    return anim.frame_palette(anim.frame.uid)


def _can_own_palette(state: Any, tab: Any) -> bool:
    """Offered on an indexed animated frame that does not already have one.

    Indexed only, because in palette-constrained RGB the table is a rule
    applied to writes rather than a lookup the pixels come through -- a
    per-frame one there would change what the next stroke snaps to and repaint
    nothing, which is not what the row promises.
    """
    doc = getattr(tab, "doc", None)
    if doc is None or not doc.is_indexed or doc.anim is None or not doc.anim.frames:
        return False
    return _own_palette(tab) is None


def _has_own_palette(state: Any, tab: Any) -> bool:
    return tab is not None and _own_palette(tab) is not None


def _can_split(state: Any, tab: Any) -> bool:
    return tab is not None and not tab.split


def _has_split(state: Any, tab: Any) -> bool:
    return tab is not None and tab.split


def _dup_view(tab: Any) -> Any:
    from . import inker_state

    return inker_state.duplicate_view(tab)


def _close_dup_view(tab: Any) -> Any:
    from . import inker_state

    return inker_state.close_duplicate_view(tab)


def _view(verb: str, *args: Any) -> Callable[..., Any]:
    def _run(ctx: Any, tab: Any, **_: Any) -> Any:
        from . import inker_state

        return getattr(inker_state, verb)(tab.view, *args)

    return _run


# --- File -------------------------------------------------------------------

register(Op("new", "New...", dialog("new-canvas"), menu="File", key="Ctrl+N"))
register(Op("open", "Open...", _mode_ctx("ask_open"), menu="File", key="Ctrl+O"))
#: The eight verbs ``inker_mode._MUTATING_CTRL`` refuses on a busy tab. The
#: keyboard has always refused them; the menu rows stayed live, so a click could
#: restructure the stack while ``ora.py`` walked it on a task thread.
_UNDO = when_ready(can_undo, "Nothing to undo yet.")
_REDO = when_ready(can_redo, "Nothing to redo: this is the newest step.")
_CUT = when_ready(has_selection, NO_SELECTION)
_SELECT_ALL = when_ready(has_doc, NO_DOC)
_DESELECT = when_ready(has_selection, NO_SELECTION)
_INVERT = when_ready(has_doc, NO_DOC)
_COPY_LAYER = when_ready(has_selection, NO_SELECTION)
_MOVE_LAYER = when_ready(has_selection, NO_SELECTION)


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
        "repeat_export",
        "Repeat last export",
        _mode("repeat_export"),
        menu="File",
        key="Ctrl+Shift+X",
        enabled=lambda state, tab: ready(state, tab) and bool(getattr(tab, "export_kind", "")),
        reason="Nothing to repeat yet -- export once and this runs it again.",
        hint=(
            "The hot-path escape valve: configure the export once, then one "
            "key forever. It writes where it wrote and asks nothing."
        ),
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
        enabled=_UNDO[0],
        reason=_UNDO[1],
    )
)
register(
    Op(
        "redo",
        "Redo",
        _doc("redo"),
        menu="Edit",
        key="Ctrl+Y",
        enabled=_REDO[0],
        reason=_REDO[1],
    )
)
register(
    Op(
        "undo_history",
        "Undo history...",
        dialog("inker-undo-history"),
        menu="Edit",
        enabled=has_doc,
        reason=NO_DOC,
        hint=(
            "Every step the stack is holding, with the head marked. Clicking "
            "one walks there through undo and redo -- it is the operation you "
            "already have, made easy."
        ),
    )
)
register(
    Op(
        "cut",
        "Cut",
        _doc("cut"),
        menu="Edit",
        key="Ctrl+X",
        enabled=_CUT[0],
        reason=_CUT[1],
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
        lambda ctx, tab, **_: tab.doc.fill_selection(
            ctx.state.inker.fg, **_pattern(ctx)
        ),
        menu="Edit",
        enabled=has_selection,
        reason=NO_SELECTION,
        separator_before=True,
        hint=(
            "The foreground colour -- or the captured tip, if the bucket is "
            "set to fill with a pattern. One setting, so this and a bucket "
            "click cannot put down two different things."
        ),
    )
)
register(
    Op(
        "stroke_selection",
        "Stroke selection...",
        lambda ctx, tab, **params: tab.doc.stroke_selection(
            ctx.state.inker.fg, int(params["width"]), **_pattern(ctx)
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
        lambda ctx, tab, **params: tab.doc.shift_selected(int(params["dx"]), int(params["dy"])),
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
register(
    Op(
        "regenerate_selection",
        "Regenerate selection...",
        dialog("inker-inpaint"),
        menu="Edit",
        enabled=_CUT[0],
        reason=_CUT[1],
    )
)


def _paste(ctx: Any, tab: Any, *, as_layer: bool) -> Any:
    from . import inker_mode

    inker_mode.paste_from_os(ctx, tab)
    if as_layer:
        return tab.doc.paste_as_layer()
    result = tab.doc.paste()
    if result is False:
        # ``enabled`` cannot see the clipboard, so this is the only place the
        # refusal can be made -- and a Ctrl+V that does nothing at all and says
        # nothing at all is the shape ``run``'s docstring calls out.
        ctx.state.inker.say("There is nothing on the clipboard.")
    if result is not False:
        # Only on success, ``inker_mode.stamp_text``'s rule and the Ctrl+V
        # precedent it cites. Switching unconditionally left a user holding the
        # Move tool with nothing pasted whenever the clipboard was empty or the
        # layer was locked -- and ``run`` discards the ``False``, so nothing was
        # said about that either.
        ctx.state.inker.set_tool("move")
    return result


def _filter(name: str) -> Callable[..., Any]:
    """An op that opens the filter popup with one filter already chosen.

    The popup has always been able to run any of ``filters.FILTERS``; what it
    had no route to was "the one I always use". Two of them get a key for the
    reason Aseprite gives them one -- hue/saturation and invert are the two a
    pixel artist reaches for by muscle memory.
    """

    def _run(ctx: Any, tab: Any, **_: Any) -> None:
        state = ctx.state.inker
        state.filter_name = name
        state.pending_dialog = "inker-filter"

    return _run


# --- Sprite -----------------------------------------------------------------

# Image size above Canvas size, which is Photoshop's order and Aseprite's:
# scaling the picture is the more common of the two and the one a reader looks
# for first.
register(
    Op(
        "scale_image",
        "Image size...",
        dialog("inker-scale"),
        menu="Sprite",
        # Aseprite's Sprite Size. It had nowhere to live for as long as "Scale
        # image" and "Resize canvas" were two buttons on one popup and one op
        # could not carry two bindings; two dialogs is what gives it a home.
        key="Ctrl+Alt+I",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "resize",
        "Canvas size...",
        dialog("inker-resize"),
        menu="Sprite",
        # Aseprite's Canvas Size key. The name, the key and the request id are
        # all unchanged by the 2026-08-29 split -- three tests pin them -- and
        # what changed is that this now opens a dialog about the canvas alone.
        key="Ctrl+Alt+C",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "filter_hue_saturation",
        "Hue / saturation...",
        _filter("hue / saturation"),
        menu="Sprite",
        key="Ctrl+U",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "filter_invert",
        "Invert colours...",
        _filter("invert"),
        menu="Sprite",
        key="Ctrl+I",
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
        reason=("This is not a tilemap layer, or it already updates its tileset as you draw."),
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
register(
    Op(
        "toggle_matte",
        "Flatten transparency onto white",
        _doc("toggle_matte"),
        menu="Sprite",
        enabled=lambda state, tab: ready(state, tab) and not tab.doc.has_background,
        reason="A document with a background layer has no transparency to flatten.",
        checked=lambda state, tab: tab is not None and tab.doc.matte is not None,
        hint=(
            "Puts white behind every erased area when this document is saved "
            "as a flat image -- on by default for a photo or a flat PNG opened "
            "here, so it is still a photo when it is saved. Turn it off and "
            "erased areas export transparent instead; the canvas backdrop "
            "follows the setting either way. Sheet, GIF and PNG-sequence "
            "exports ignore it and always keep transparency."
        ),
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
        lambda ctx, tab, **_: tab.doc.set_layer_props(tab.doc.stack.active_index, visible=True),
        menu="Layer",
        enabled=lambda state, tab: tab is not None and not tab.doc.stack.active.visible,
        reason="This layer is already visible.",
        separator_before=True,
    )
)
register(
    Op(
        "to_background",
        "Convert to background",
        _doc("to_background"),
        menu="Layer",
        enabled=lambda state, tab: (
            ready(state, tab) and len(tab.doc.stack) > 0 and not tab.doc.has_background
        ),
        reason="The bottom layer is already the background.",
        hint=(
            "Makes the bottom layer opaque, and folds the document's matte "
            "colour into its pixels -- so what was a flatten-time overlay "
            "becomes a layer you can paint on and every format can store."
        ),
        separator_before=True,
    )
)
register(
    Op(
        "from_background",
        "Layer from background",
        _doc("from_background"),
        menu="Layer",
        enabled=lambda state, tab: ready(state, tab) and tab.doc.has_background,
        reason="There is no background layer.",
    )
)
register(
    Op(
        "toggle_reference",
        "Reference layer",
        lambda ctx, tab, **_: tab.doc.set_reference(
            tab.doc.stack.active_index, not tab.doc.stack.active.reference
        ),
        menu="Layer",
        enabled=ready,
        reason=BUSY,
        hint="Drawn, never edited: an underlay to trace over.",
    )
)
register(
    Op(
        "solo_layer",
        "Solo this layer",
        lambda ctx, tab, **_: tab.doc.solo(tab.doc.stack.active_index),
        menu="Layer",
        key="Alt+S",
        enabled=ready,
        reason=BUSY,
        hint=(
            "Hides everything else -- and pressing it again on the layer that "
            "is already alone brings the rest back."
        ),
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
        # **Deliberately context-free.** It used to declare ``context="Normal"``,
        # which made the binding unreachable: ``by_key`` tries the requested
        # context and then ops with *no* context, and "Normal" is truthy -- so
        # with any paint tool in hand (``key_context`` answers "FreehandTool",
        # and brush is the default) Enter resolved to nothing and the op's
        # refusal never spoke. Enter still reached ``toggle_play`` through the
        # raw fallback in ``handle_key``, so the key worked and only the message
        # was missing: pressing Enter on a still document said nothing with a
        # brush in hand and toasted correctly with an eyedropper.
        #
        # Nothing is lost by dropping it. "Normal" is the last row of
        # ``KEY_CONTEXTS`` and means "no other context matched", and the two
        # contexts that must not see Enter -- ``Transformation`` and
        # ``Gesture`` -- are consumed by ``_modal`` before ``by_key`` is asked.
        enabled=animated,
        reason="This drawing has no frames yet -- Animate it first.",
    )
)
register(
    Op(
        "frame_palette",
        "Give this frame its own palette",
        lambda ctx, tab, **_: tab.doc.set_frame_palette(list(tab.doc.palette or ())),
        menu="Frame",
        enabled=_can_own_palette,
        reason=(
            "Only an indexed drawing can have a palette per frame -- its pixels"
            " are slot numbers, so a different table repaints them. Convert it"
            " to Indexed first."
        ),
        separator_before=True,
    )
)
register(
    Op(
        "clear_frame_palette",
        "Use the drawing's palette here",
        lambda ctx, tab, **_: tab.doc.clear_frame_palette(),
        menu="Frame",
        enabled=_has_own_palette,
        reason="This frame is already using the drawing's own palette.",
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
register(
    Op(
        "first_frame",
        "First frame",
        lambda ctx, tab, **_: tab.doc.set_current_frame(0),
        menu="Frame",
        key="Home",
        enabled=animated,
        reason=NOT_ANIMATED,
    )
)
register(
    Op(
        "last_frame",
        "Last frame",
        lambda ctx, tab, **_: tab.doc.set_current_frame(len(tab.doc.anim.frames) - 1),
        menu="Frame",
        key="End",
        enabled=animated,
        reason=NOT_ANIMATED,
    )
)
register(
    Op(
        "new_frame",
        "New frame",
        _doc("add_frame"),
        menu="Frame",
        key="Alt+N",
        # Offered on a *still* drawing too, and that is not an oversight:
        # ``add_frame`` folds the ``AnimateEdit`` into its own step, so this is
        # the verb that animates a drawing as well as the one that extends a
        # clip. Requiring "Animate this drawing" first would be a second name
        # for one operation.
        enabled=ready,
        reason=BUSY,
        separator_before=True,
    )
)
register(
    Op(
        "duplicate_frame",
        "Duplicate frame",
        _doc("add_frame", copy=True),
        menu="Frame",
        key="Alt+D",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "delete_frame",
        "Delete frame",
        _doc("remove_frame"),
        menu="Frame",
        # **No key, deliberately.** Aseprite has none either: every other verb
        # here is recoverable by pressing it again, and a one-key drop of the
        # frame under the playhead is the one worth reaching for a menu.
        enabled=lambda state, tab: (
            ready(state, tab) and animated(state, tab) and len(tab.doc.anim.frames) > 1
        ),
        reason="A clip keeps at least one frame.",
    )
)
register(
    Op(
        "toggle_onion",
        "Onion skin",
        lambda ctx, tab, **_: _toggle(ctx, "onion"),
        menu="Frame",
        key="F3",
        checked=lambda state, tab: bool(state.onion),
        enabled=has_doc,
        reason=NO_DOC,
        separator_before=True,
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
        enabled=_SELECT_ALL[0],
        reason=_SELECT_ALL[1],
    )
)
register(
    Op(
        "deselect",
        "Deselect",
        _doc("deselect"),
        menu="Select",
        key="Ctrl+D",
        enabled=_DESELECT[0],
        reason=_DESELECT[1],
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
        enabled=_INVERT[0],
        reason=_INVERT[1],
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
        enabled=_COPY_LAYER[0],
        reason=_COPY_LAYER[1],
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
        enabled=_MOVE_LAYER[0],
        reason=_MOVE_LAYER[1],
    )
)


def _select_slots(ctx: Any, tab: Any, *, used: bool) -> bool:
    """Select every pixel drawn in the used (or unused) palette slots.

    One function for Aseprite's two commands, because they are two readings of
    one histogram -- and the *selection* is the answer either way, so the two
    rows differ by a boolean rather than by a second walk of the document.
    """
    doc = tab.doc
    slots = doc.used_slots() if used else doc.unused_slots()
    if not slots:
        # Said rather than answered with a silent ``False``. ``enabled`` can
        # only ask whether there is a palette at all; whether any slot is drawn
        # in is a walk of the document, so this refusal can only be made here --
        # and ``run``'s promise that a refused op says why has to hold at the
        # runtime doors too, not only at the gate.
        ctx.state.inker.say(
            "Every slot is in use." if not used else "No slot in this palette is used."
        )
        return False
    return doc.select_slots(slots)


register(
    Op(
        "select_used_colours",
        "Used colours",
        lambda ctx, tab, **_: _select_slots(ctx, tab, used=True),
        menu="Select",
        enabled=lambda state, tab: tab is not None and bool(tab.doc.palette),
        reason="This document has no palette.",
        hint=(
            "Selects every pixel drawn in a palette slot that is in use. Its "
            "sibling below selects what the unused slots hold, which on a tidy "
            "drawing is nothing at all -- which is the answer."
        ),
        separator_before=True,
    )
)
register(
    Op(
        "select_unused_colours",
        "Unused colours",
        lambda ctx, tab, **_: _select_slots(ctx, tab, used=False),
        menu="Select",
        enabled=lambda state, tab: tab is not None and bool(tab.doc.palette),
        reason="This document has no palette.",
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
        "duplicate_view",
        "Duplicate View",
        lambda ctx, tab, **_: _dup_view(tab),
        menu="View",
        enabled=_can_split,
        reason="This tab is already showing two views.",
        separator_before=True,
    )
)
register(
    Op(
        "close_duplicate_view",
        "Close the second view",
        lambda ctx, tab, **_: _close_dup_view(tab),
        menu="View",
        enabled=_has_split,
        reason="This tab is showing one view.",
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
        "rotate_view_back",
        "Rotate the view back",
        _view("rotate_view", -1),
        menu="View",
        key="Ctrl+Shift+4",
        enabled=has_doc,
        reason=NO_DOC,
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



def _centre_view(ctx: Any, tab: Any, **_: Any) -> None:
    """Put the page back under the pane, keeping the zoom the user chose.

    ``pending_zoom`` and not ``fitted = False``, exactly as ``rotate_view`` and
    ``flip_view`` do it: clearing ``fitted`` re-*scales* as well as re-centring,
    which is the one thing this must not do. Distinct from *Fit in window*
    above for that reason -- panning far enough to lose the canvas entirely is
    easy, and before this the only way back threw the zoom away.
    """
    tab.view.pending_zoom = tab.view.zoom


register(
    Op(
        "center_view",
        "Center the page",
        _centre_view,
        menu="View",
        enabled=has_doc,
        reason=NO_DOC,
    )
)


def _set_tiled(mode: str) -> Callable[..., Any]:
    def run(ctx: Any, tab: Any, **_: Any) -> None:
        # One setting driving the view *and* the writes, deliberately: a canvas
        # that showed its neighbours while the brush went on clamping at the
        # edge would be a picture of a seamless tile you cannot paint.
        tab.tiled = mode

    return run


def _flag_is(attr: str) -> Callable[[Any, Any], bool]:
    """A view preference's own tick. ``tiled_*`` has had one since it became
    four checked rows; these six had none, so the View menu -- the only
    always-visible door to four of them -- could not say whether the aid it was
    offering was already on."""
    return lambda state, tab: state is not None and bool(getattr(state, attr, False))


def _tiled_is(mode: str) -> Callable[[Any, Any], bool]:
    return lambda state, tab: tab is not None and tab.tiled == mode


#: The tiling modes as four checked rows rather than a combo on a bar. They
#: were the trailing block of the canvas's view row until 2026-08-23, when that
#: row became the Aseprite context bar; a menu is where a four-way setting
#: nobody changes mid-stroke belongs, and a checked row says which one is on
#: without spending bar width saying so.
TILED_MODES: tuple[tuple[str, str], ...] = (
    ("off", "Tiled: off"),
    ("x", "Tiled: left and right"),
    ("y", "Tiled: top and bottom"),
    ("both", "Tiled: both ways"),
)

for _index, (_mode_key, _label) in enumerate(TILED_MODES):
    register(
        Op(
            f"tiled_{_mode_key}",
            _label,
            _set_tiled(_mode_key),
            menu="View",
            enabled=has_doc,
            reason=NO_DOC,
            checked=_tiled_is(_mode_key),
            separator_before=_index == 0,
        )
    )


def _wrap_half(ctx: Any, tab: Any, **_: Any) -> None:
    """Roll the layer half a canvas both ways, putting the seam in the middle.

    The classic put-the-seam-where-you-can-paint-it move, and the reason it is
    *half* rather than any amount: on even dimensions pressing it twice is the
    identity, so it is a look rather than an edit that has to be undone.

    In the View menu beside Tiled although it is a real edit -- it moves pixels
    and takes an undo step. Filed by what a user is *doing* rather than by what
    it touches: the only reason to reach for it is that the tiling preview has
    shown a seam, and it is the one verb in that group with somewhere else it
    could plausibly go.
    """
    width, height = tab.doc.size
    tab.doc.offset_layer(width // 2, height // 2)


register(
    Op(
        "wrap_half",
        "Roll the seam to the middle",
        _wrap_half,
        menu="View",
        enabled=lambda state, tab: ready(state, tab) and tab.tiled != "off",
        reason=lambda state, tab: (
            BUSY
            if tab is not None and tab.busy
            else "This document is not tiled -- there is no wrap seam to move."
        ),
    )
)
register(
    Op(
        "trim",
        "Trim",
        _doc("trim"),
        menu="Sprite",
        enabled=ready,
        reason=BUSY,
        hint=(
            "Crops away the fully transparent border. A document with nothing "
            "in it is left alone -- an empty canvas is a size you chose."
        ),
    )
)
register(
    Op(
        "duplicate_sprite",
        "Duplicate document",
        _mode("duplicate_document"),
        menu="Sprite",
        enabled=ready,
        reason=BUSY,
    )
)
register(
    Op(
        "toggle_pixel_grid",
        "Pixel grid",
        lambda ctx, tab, **_: _toggle(ctx, "pixel_grid"),
        menu="View",
        enabled=has_doc,
        reason=NO_DOC,
        separator_before=True,
        checked=_flag_is("pixel_grid"),
    )
)
register(
    Op(
        "toggle_layer_edges",
        "Layer edges",
        lambda ctx, tab, **_: _toggle(ctx, "layer_edges"),
        menu="View",
        enabled=has_doc,
        reason=NO_DOC,
        checked=_flag_is("layer_edges"),
    )
)
register(
    Op(
        "toggle_tile_numbers",
        "Tile numbers",
        lambda ctx, tab, **_: _toggle(ctx, "tile_numbers"),
        menu="View",
        enabled=has_doc,
        reason=NO_DOC,
        checked=_flag_is("tile_numbers"),
    )
)
register(
    Op(
        "grid_from_selection",
        "Grid from selection",
        lambda ctx, tab, **_: _grid_from_selection(ctx, tab),
        menu="View",
        enabled=has_selection,
        reason=NO_SELECTION,
        hint=(
            "Sets the grid to the selection's own size, which is how a tile "
            "size gets from a drawing into the grid without being measured."
        ),
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
        checked=_flag_is("grid"),
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
        checked=_flag_is("grid_snap"),
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
        checked=_flag_is("rulers"),
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


def _grid_from_selection(ctx: Any, tab: Any) -> bool:
    """Aseprite's *Selection as Grid*: the grid takes the marquee's size."""
    from . import inker_mode

    mask = tab.doc.mask
    bounds = None if mask is None else mask.bounds
    if bounds is None:
        return False
    x0, y0, x1, y1 = bounds
    state = ctx.state.inker
    # One number, because the grid is square here: the *shorter* side, so a
    # grid derived from a rectangle never claims cells the selection did not
    # cover.
    state.grid_size = max(2, min(512, min(x1 - x0, y1 - y0)))
    state.grid = True
    inker_mode.persist(ctx)
    return True


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


register(
    Op(
        "keyboard_shortcuts",
        "Keyboard Shortcuts...",
        dialog("inker-shortcuts"),
        menu="Edit",
        key="Ctrl+Alt+Shift+K",
        separator_before=True,
        hint=(
            "Search commands, tools and held action modifiers; assign multiple "
            "contextual bindings, import or export them, or restore defaults."
        ),
    )
)



# --- sheet corrections ------------------------------------------------------
#
# The Troupe phase-6 verbs. Every predicate and every sentence is
# ``inker_sheet``'s, so the strip under the transport, this menu and the probe
# census answer "why is this grey" with one voice. The strip is where these
# are pressed; the menu is where they are *found*.


def _sheet(verb: str) -> Callable[..., Any]:
    def _run(ctx: Any, tab: Any, **_: Any) -> Any:
        from . import inker_sheet

        return getattr(inker_sheet, verb)(ctx, tab)

    return _run


def _sheet_tab(verb: str) -> Callable[..., Any]:
    """A sheet verb of the tab alone -- no ctx, no toast."""

    def _run(ctx: Any, tab: Any, **_: Any) -> Any:
        from . import inker_sheet

        return getattr(inker_sheet, verb)(tab)

    return _run


def _sheet_merge(ctx: Any, tab: Any, **_: Any) -> Any:
    """Pick a re-rendered sheet of the same character and merge it in.

    The load is a decode of a whole atlas, so it goes through the task runner
    -- the frame loop never blocks. The document write lands back on the frame
    thread, which is where every other document write happens.
    """
    from . import inker_mode, inker_sheet

    doc = getattr(tab, "doc", None)
    base = getattr(doc, "sheet_base", None)
    if base is None:
        return False
    job_id = str(base.source.get("job") or "")
    sheet_id = str(base.source.get("sheet") or "")
    if not (job_id and sheet_id):
        ctx.toast(
            "This sheet does not record which job it came from, so the "
            "re-render cannot be found automatically.",
            "warn",
        )
        return False
    newest = inker_mode.newest_sheet_after(ctx.svc, job_id, sheet_id)
    if not newest:
        ctx.toast("No newer sheet of this character to merge in.", "info")
        return False

    def work() -> Any:
        return inker_mode.load_sheet_cells(ctx.svc, job_id, newest)

    def done(cells: Any) -> None:
        if inker_sheet.merge(ctx, tab, cells):
            base.source["sheet"] = newest

    return ctx.submit(f"inker-merge:{newest}", work, on_done=done)


def _sheet_conflict_next(ctx: Any, tab: Any, **_: Any) -> Any:
    from . import inker_sheet

    doc = getattr(tab, "doc", None)
    if doc is None or doc.anim is None:
        return False
    nxt = inker_sheet.next_conflict(tab, doc.anim.current)
    if nxt is None:
        ctx.toast(inker_sheet.NO_CONFLICTS, "info")
        return False
    doc.anim.current = nxt
    return True


def _sheet_keep_edit(ctx: Any, tab: Any, **_: Any) -> Any:
    from . import inker_sheet

    doc = getattr(tab, "doc", None)
    if doc is None or doc.anim is None:
        return False
    return inker_sheet.resolve_keep(ctx, tab, [doc.anim.current])


def _sheet_pred(name: str) -> Callable[[Any, Any], bool]:
    def _enabled(state: Any, tab: Any) -> bool:
        from . import inker_sheet

        return bool(getattr(inker_sheet, name)(state, tab))

    return _enabled


def _sheet_reason(name: str) -> Callable[[Any, Any], str]:
    def _reason(state: Any, tab: Any) -> str:
        from . import inker_sheet

        return str(getattr(inker_sheet, name)(state, tab))

    return _reason


register(
    Op(
        "sheet_merge",
        "Merge re-render...",
        _sheet_merge,
        menu="Sheet",
        separator_before=True,
        enabled=_sheet_pred("can_merge"),
        reason=_sheet_reason("merge_reason"),
        hint=(
            "Brings a re-rendered sheet into this document three ways: cells you "
            "have not touched take the new render, cells you painted keep your "
            "work, and cells where both changed are flagged for you to decide. "
            "Nothing you painted is overwritten without asking. One undo step."
        ),
    )
)
register(
    Op(
        "sheet_conflict_next",
        "Go to the next conflicted cell",
        _sheet_conflict_next,
        menu="Sheet",
        enabled=_sheet_pred("can_merge"),
        reason=_sheet_reason("merge_reason"),
        hint="Walks the cells a merge could not decide, wrapping at the end.",
    )
)
register(
    Op(
        "sheet_keep_edit",
        "Keep the hand edit on this cell",
        _sheet_keep_edit,
        menu="Sheet",
        enabled=_sheet_pred("can_merge"),
        reason=_sheet_reason("merge_reason"),
        hint=(
            "Clears this cell's conflict flag and leaves what you painted. "
            "Nothing is written -- your work is already what is on the canvas."
        ),
    )
)

register(
    Op(
        "sheet_propagate",
        "Propagate correction",
        _sheet("propagate"),
        menu="Sheet",
        enabled=_sheet_pred("can_propagate"),
        reason=_sheet_reason("propagate_reason"),
        hint=(
            "Copies what changed on this cell since it was marked onto every "
            "cell the scope names -- the same frame in every direction, say. "
            "One undo step for all of them."
        ),
    )
)
register(
    Op(
        "sheet_remark",
        "Mark this cell",
        _sheet_tab("remark"),
        menu="Sheet",
        enabled=_sheet_pred("is_sheet_tab"),
        reason=_sheet_reason("no_sheet_reason"),
        hint="Takes the cell as it is now as the point a correction is measured from.",
    )
)
register(
    Op(
        "sheet_replace",
        "Replace colour across scope",
        _sheet("replace_colour"),
        menu="Sheet",
        enabled=_sheet_pred("can_scope"),
        reason=_sheet_reason("scope_reason"),
        separator_before=True,
        hint="The strip's recolour pair, applied to this cell and every cell the scope names.",
    )
)
register(
    Op(
        "sheet_shift",
        "Shift selection across scope",
        _sheet("shift"),
        menu="Sheet",
        enabled=_sheet_pred("can_shift"),
        reason=_sheet_reason("shift_reason"),
        hint=(
            "Moves the selected pixels by the strip's offset on this cell and "
            "every cell the scope names."
        ),
    )
)
register(
    Op(
        "sheet_mirror",
        "Apply mirror to opposite direction",
        _sheet("mirror_to"),
        menu="Sheet",
        enabled=_sheet_pred("can_mirror"),
        reason=_sheet_reason("mirror_reason"),
        separator_before=True,
        hint=(
            "Writes this cell, flipped, onto the same frame of the mirror "
            "direction -- everything but the face, which is drawn on its own "
            "side."
        ),
    )
)
register(
    Op(
        "sheet_mirror_run",
        "Apply mirror to whole run",
        _sheet("mirror_run"),
        menu="Sheet",
        enabled=_sheet_pred("can_mirror"),
        reason=_sheet_reason("mirror_reason"),
        hint="Every frame of this direction mirrored onto its counterpart, as one undo step.",
    )
)

# --- input registry ---------------------------------------------------------
#
# Kept after command registration so the command half is derived rather than
# restated.  The first binding for a target is its Aseprite-compatible primary
# gesture; compatibility aliases follow it and remain visible in the shortcut
# editor instead of living as invisible branches in ``handle_key``.

ACTION_MODIFIERS: tuple[ActionModifier, ...] = (
    ActionModifier(
        "freehand_straight",
        "Straight line from last point",
        "FreehandTool",
        "Draw from the last painted pixel.",
    ),
    ActionModifier(
        "freehand_angle_snap",
        "Angle snap from last point",
        "FreehandTool",
        "Snap the straight-line angle.",
    ),
    ActionModifier(
        "move_auto_select", "Auto-select layer", "MoveTool", "Select the layer under the cursor."
    ),
    ActionModifier("selection_add", "Add selection", "Selection", "Combine the gesture by union."),
    ActionModifier(
        "selection_subtract",
        "Subtract selection",
        "Selection",
        "Remove the gesture from the selection.",
    ),
    ActionModifier(
        "selection_intersect",
        "Intersect selection",
        "Selection",
        "Keep only the overlap.",
    ),
    ActionModifier(
        "shape_square",
        "Square/circle constraint",
        "ShapeTool",
        "Constrain both axes to the same extent.",
    ),
    ActionModifier(
        "shape_center",
        "Draw from centre",
        "ShapeTool",
        "Use the press point as the shape centre.",
    ),
    ActionModifier("shape_rotate", "Rotate shape", "ShapeTool", "Rotate before committing."),
    ActionModifier(
        "shape_move_origin",
        "Move shape origin",
        "ShapeTool",
        "Reposition the whole uncommitted shape.",
    ),
    ActionModifier(
        "translate_snap_grid",
        "Snap to grid",
        "TranslatingSelection",
        "Snap translation to the grid.",
    ),
    ActionModifier(
        "translate_lock_axis", "Lock axis", "TranslatingSelection", "Move along one axis only."
    ),
    ActionModifier(
        "translate_copy",
        "Copy selection",
        "TranslatingSelection",
        "Duplicate when the move begins.",
    ),
    ActionModifier(
        "translate_fine",
        "Fine translation",
        "TranslatingSelection",
        "Adjust at subpixel precision.",
    ),
    ActionModifier(
        "scale_aspect",
        "Maintain aspect ratio",
        "ScalingSelection",
        "Keep the original aspect ratio.",
    ),
    ActionModifier(
        "scale_center", "Scale from centre", "ScalingSelection", "Scale around the transform pivot."
    ),
    ActionModifier(
        "scale_fine", "Fine scaling", "ScalingSelection", "Adjust at subpixel precision."
    ),
    ActionModifier("rotate_snap", "Angle snap", "RotatingSelection", "Snap the rotation angle."),
)

_TOOL_BINDINGS: tuple[Binding, ...] = (
    Binding("brush", "B", "tool", priority=10),
    Binding("spray", "Shift+B", "tool", priority=10),
    Binding("spray", "A", "tool"),
    Binding("eraser", "E", "tool", priority=10),
    Binding("eyedropper", "I", "tool", priority=10),
    Binding("move", "V", "tool", priority=10),
    Binding("slice", "Shift+C", "tool", priority=10),
    Binding("slice", "C", "tool"),
    Binding("fill", "G", "tool", priority=10),
    Binding("gradient", "Shift+G", "tool", priority=10),
    Binding("gradient", "K", "tool"),
    Binding("line", "L", "tool", priority=10),
    Binding("curve", "Shift+L", "tool", priority=10),
    Binding("curve", "F", "tool"),
    Binding("rect", "U", "tool", priority=10),
    Binding("ellipse", "Shift+U", "tool", priority=10),
    Binding("ellipse", "J", "tool"),
    Binding("polyline", "P", "tool"),
    Binding("polygon", "Shift+D", "tool", priority=10),
    Binding("polygon", "O", "tool"),
    Binding("blur", "R", "tool", priority=10),
    Binding("smudge", "N", "tool"),
    Binding("shade", "H", "tool"),
    Binding("select", "M", "tool", priority=10),
    Binding("select_ellipse", "Shift+M", "tool", priority=10),
    Binding("select_ellipse", "S", "tool"),
    Binding("lasso", "Q", "tool"),
    Binding("lasso_poly", "D", "tool"),
    Binding("wand", "W", "tool", priority=10),
    Binding("text", "T", "tool"),
    Binding("tile", "Y", "tool"),
)

_ACTION_BINDINGS: tuple[Binding, ...] = (
    Binding("freehand_straight", "Shift", "action_modifier", "FreehandTool", "hold", 10),
    Binding("freehand_angle_snap", "Ctrl+Shift", "action_modifier", "FreehandTool", "hold", 20),
    Binding("move_auto_select", "Ctrl", "action_modifier", "MoveTool", "hold", 10),
    Binding("selection_add", "Shift", "action_modifier", "Selection", "hold", 10),
    Binding("selection_subtract", "Shift+Alt", "action_modifier", "Selection", "hold", 10),
    Binding("selection_intersect", "Ctrl+Shift", "action_modifier", "Selection", "hold", 20),
    Binding("shape_square", "Shift", "action_modifier", "ShapeTool", "hold", 10),
    Binding("shape_center", "Ctrl", "action_modifier", "ShapeTool", "hold", 10),
    Binding("shape_rotate", "Alt", "action_modifier", "ShapeTool", "hold", 10),
    Binding("shape_move_origin", "Space", "action_modifier", "ShapeTool", "hold", 10),
    Binding("translate_snap_grid", "Alt", "action_modifier", "TranslatingSelection", "hold", 10),
    Binding("translate_lock_axis", "Shift", "action_modifier", "TranslatingSelection", "hold", 10),
    Binding("translate_copy", "Ctrl", "action_modifier", "TranslatingSelection", "hold", 10),
    Binding("translate_fine", "Ctrl", "action_modifier", "TranslatingSelection", "hold", 5),
    Binding("scale_aspect", "Shift", "action_modifier", "ScalingSelection", "hold", 10),
    Binding("scale_center", "Alt", "action_modifier", "ScalingSelection", "hold", 10),
    Binding("scale_fine", "Ctrl", "action_modifier", "ScalingSelection", "hold", 10),
    Binding("rotate_snap", "Shift", "action_modifier", "RotatingSelection", "hold", 10),
)

_QUICK_TOOL_BINDINGS: tuple[Binding, ...] = (
    Binding("eyedropper", "Alt", "tool", "FreehandTool", "hold", 30),
    Binding("move", "Ctrl", "tool", "FreehandTool", "hold", 30),
)

_CONTEXT_COMMAND_BINDINGS: tuple[Binding, ...] = (
    Binding("fill_selection", "F", context="Selection", priority=20),
    Binding("stroke_selection", "S", context="Selection", priority=20),
)

BINDINGS: tuple[Binding, ...] = (
    *(Binding(op.name, op.key, context=op.context) for op in OPS if op.key),
    *_CONTEXT_COMMAND_BINDINGS,
    *_TOOL_BINDINGS,
    *_QUICK_TOOL_BINDINGS,
    *_ACTION_BINDINGS,
)


def manifest() -> dict[str, Any]:
    """Machine-readable frozen parity contract consumed by tests and tooling."""

    effective = BINDINGS
    return {
        "schema": 1,
        "target": {"application": "Aseprite", "version": "1.3.15.5", "platform": "Windows"},
        "menus": list(MENUS),
        "commands": [
            {
                "id": op.name,
                "label": op.label,
                "menu": op.menu,
                "context": op.context,
                "parameters": [
                    {
                        "id": param.name,
                        "label": param.label,
                        "default": param.default,
                        "minimum": param.low,
                        "maximum": param.high,
                        "step": param.step,
                        "integer": param.integer,
                    }
                    for param in op.params
                ],
                "bindings": [
                    item.chord
                    for item in effective
                    if item.kind == "command" and item.target == op.name
                ],
            }
            for op in OPS
        ],
        "tools": [
            {
                "id": tool,
                "bindings": [
                    item.chord for item in effective if item.kind == "tool" and item.target == tool
                ],
            }
            for tool in dict.fromkeys(item.target for item in _TOOL_BINDINGS)
        ],
        "action_modifiers": [
            {
                "id": modifier.name,
                "label": modifier.label,
                "context": modifier.context,
                "description": modifier.description,
                "bindings": [
                    item.chord
                    for item in effective
                    if item.kind == "action_modifier" and item.target == modifier.name
                ],
            }
            for modifier in ACTION_MODIFIERS
        ],
        "document_fields": [
            "canvas",
            "color_mode",
            "palette",
            "layers",
            "groups",
            "cels",
            "frames",
            "tags",
            "slices",
            "tilesets",
            "tilemaps",
            "selection",
            "grid",
            "matte",
            "metadata",
        ],
    }
