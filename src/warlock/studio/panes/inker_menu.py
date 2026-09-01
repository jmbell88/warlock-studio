"""The Inker's op dialogs: the parameter sheet, layer properties and history.

**The menu strip that used to live here is gone.** Its seven names moved to the
application menu bar (:mod:`~warlock.studio.menus`, which builds its Inker rows
straight from ``inker_ops.MENUS`` and calls :func:`activate`), because the
editor shell has one menu bar and a per-pane strip underneath it was the same
verbs twice. What could not move is the dialogs: an imgui popup only renders in
the id stack of the window that opened it, and these three are opened from
inside the centre pane. So :func:`draw_popups` is called from
``inker_canvas.draw`` and hosts them there, while the thing that *raises* them
sits in the menu bar a window away -- bridged by ``state.pending_op`` rather
than by a call, for exactly that reason.

**A row is an ``inker_ops.Op`` and nothing else.** ``menus.py`` decides where a
row appears; the registry decides what exists, whether it can run and what to
say when it cannot. That is still the whole of why this file is short: there is
no list of verbs in it.

This pane replaced ``inker_bridge``'s five verb blocks. That module keeps its
four popups and their dialog machinery and is no longer drawn as a pane; see
its docstring.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, inker, inker_mode, inker_ops, inker_state, widgets
from ..tokens import sp

#: Opacity mid-drag, per layer uid. See :func:`header_controls`.
_opacity_drag: dict[int, float] = {}

PARAM_POPUP = "inker-op-params"
PROPERTIES_POPUP = "inker-layer-properties"
HISTORY_POPUP = "inker-undo-history"
SHORTCUTS_POPUP = "inker-shortcuts"

#: How wide an ``Op.hint`` wraps, in design pixels. ``clay_menu``'s number and
#: its reason: a popup that auto-sizes past its own panel reads as a window.
HINT_WRAP = 300


def activate(ctx: Any, op: Any) -> None:
    """Run an operation selected from any command surface.

    Parameter popups are requested by flag because the global menu and the
    editor child are different imgui windows; the editor opens its own popup
    on the next frame.
    """

    state = inker_mode.ensure(ctx)
    if op.params:
        state.pending_op = op.name
        state.op_params.setdefault(op.name, inker_ops.defaults_for(op))
        state.pending_dialog = PARAM_POPUP
    else:
        inker_ops.run(ctx, op)


def draw_popups(ctx: Any) -> None:
    """Host the dialogs after the menu strip moved to application scope."""

    state = inker_mode.ensure(ctx)
    tab = state.active
    _params_popup(ctx, state)
    _properties_popup(ctx, state, tab)
    _history_popup(ctx, state, tab)
    _shortcuts_popup(ctx, state)


def _params_popup(ctx: Any, state: Any) -> None:
    """The fields for a parameterised op, and its Apply button.

    ``clay_menu.params_popup``'s shape and its argument: a Grow with no amount
    is a grow by whatever the last one was, which is right four times and wrong
    on the fifth. The last values are remembered on the state, so the common
    case is still two clicks.
    """

    if state.pending_dialog == PARAM_POPUP:
        state.pending_dialog = ""
        imgui.open_popup(PARAM_POPUP)
    if not state.pending_op:
        return
    try:
        op = inker_ops.get(state.pending_op)
    except KeyError:  # pragma: no cover - a stale name from a removed op
        state.pending_op = ""
        return
    with controls.menu_popup(PARAM_POPUP) as opened:
        if not opened:
            return
        values = state.op_params.setdefault(op.name, inker_ops.defaults_for(op))
        imgui.text(op.label.rstrip("."))
        imgui.separator()
        if op.hint:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + sp(HINT_WRAP))
            widgets.muted(op.hint)
            imgui.pop_text_wrap_pos()
        for param in op.params:
            imgui.set_next_item_width(sp(120))
            current = float(values.get(param.name, param.default))
            if param.integer:
                changed, value = controls.input_int(
                    f"{param.label}##{PARAM_POPUP}/{param.name}", int(current)
                )
            else:
                changed, value = controls.input_float(
                    f"{param.label}##{PARAM_POPUP}/{param.name}", current
                )
            if changed:
                values[param.name] = min(max(float(value), param.low), param.high)
            if param.warn:
                widgets.muted(param.warn)
        if widgets.primary_button(f"Apply##{PARAM_POPUP}"):
            inker_ops.run(ctx, op, **values)
            state.pending_op = ""
            imgui.close_current_popup()
        imgui.same_line()
        # **A Cancel, the shape ``clay_menu`` already has.** Without one the
        # only way out was clicking outside, which imgui answers by closing the
        # popup without telling anybody -- so ``pending_op`` stayed set until
        # the next parameterised op happened to overwrite it.
        if controls.button(f"Cancel##{PARAM_POPUP}"):
            state.pending_op = ""
            imgui.close_current_popup()


def _properties_popup(ctx: Any, state: Any, tab: Any) -> None:
    """Layer ▸ Properties -- blend, opacity and the three locks.

    Where Aseprite keeps them (double-click a row), and where they stop being a
    fixed 110 px header on a panel whose whole job is picking a layer.
    """

    if state.pending_dialog == PROPERTIES_POPUP:
        state.pending_dialog = ""
        imgui.open_popup(PROPERTIES_POPUP)
    if tab is None:
        return
    with controls.menu_popup(PROPERTIES_POPUP) as opened:
        if not opened:
            return
        # Every control in here writes persisted layer state -- blend,
        # opacity, the two locks, continuous -- so it takes the gate the
        # canvas, the keyboard path and the tool panels already share
        # (``busy`` is ``saving or playing``). ``ora.py`` reads ``doc.stack``
        # twice on the task thread, and a blend mode changed between the two
        # passes writes an archive whose ``stack.xml`` disagrees with its own
        # PNG members.
        imgui.begin_disabled(tab.busy)
        header_controls(ctx, tab.doc)
        imgui.end_disabled()


def _history_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The Undo History panel (6.10), as a popover rather than a tenth pane.

    A pane would be a column of one list, on screen all session, for a thing
    reached when something has gone wrong -- and it would want a share, a
    floor, a help target and a place in every saved layout. What it *is* is a
    list of the stack's own steps with the head marked, and clicking one asks
    the stack to move: it holds no list of its own, which is how an undo panel
    goes wrong -- by drifting from the stack it claims to picture once the byte
    budget evicts something.
    """

    if state.pending_dialog == HISTORY_POPUP:
        state.pending_dialog = ""
        imgui.open_popup(HISTORY_POPUP)
    if tab is None:
        return
    with controls.menu_popup(HISTORY_POPUP) as opened:
        if not opened:
            return
        history = tab.doc.history
        steps = history.history()
        widgets.secondary(f"{len(steps)} step(s)")
        imgui.separator()
        if controls.selectable("(the document as opened)", not any(done for _label, done in steps))[
            0
        ]:
            history.step_to(tab.doc, 0)
        for index, (label, done) in enumerate(steps):
            # The *count of done steps* this row represents, which is the
            # number ``step_to`` takes: row 0 is "one step done".
            wanted = index + 1
            at_head = done and (index + 1 == sum(1 for _l, d in steps if d))
            row = f"{label}{'  <' if at_head else ''}"
            if not done:
                row = f"{label}  (undone)"
            if controls.selectable(f"{row}##undo{index}", at_head)[0]:
                history.step_to(tab.doc, wanted)
                tab.doc.invalidate_all()


def _shortcut_rows() -> list[tuple[str, str, str, str, str]]:
    """(kind, target, label, context, trigger), in stable display order."""

    contract = inker_ops.manifest()
    rows = [
        ("command", row["id"], row["label"], row["context"], "press")
        for row in contract["commands"]
    ]
    labels = {tool: label for tool, label, _key in inker_state.TOOLS}
    rows.extend(
        (
            "tool",
            row["id"],
            labels.get(row["id"], row["id"].replace("_", " ").title()),
            "",
            "press",
        )
        for row in contract["tools"]
    )
    rows.extend(
        (
            "tool",
            binding.target,
            f"{labels.get(binding.target, binding.target.title())} (quick)",
            binding.context,
            "hold",
        )
        for binding in inker_ops.BINDINGS
        if binding.kind == "tool" and binding.trigger == "hold"
    )
    rows.extend(
        ("action_modifier", row["id"], row["label"], row["context"], "hold")
        for row in contract["action_modifiers"]
    )
    return rows


def _select_shortcut(state: Any, kind: str, target: str, context: str, trigger: str) -> None:
    state.shortcut_target = inker_ops.binding_target(kind, target)
    current = [
        item
        for item in inker_ops.bindings_for(state.shortcut_overrides)
        if item.kind == kind
        and item.target == target
        and item.context == context
        and item.trigger == trigger
    ]
    state.shortcut_draft = "; ".join(item.chord for item in current)
    state.shortcut_context = context
    state.shortcut_trigger = trigger


def _shortcuts_popup(ctx: Any, state: Any) -> None:
    """Searchable editor for the shared command/tool/modifier registry."""

    if state.pending_dialog == SHORTCUTS_POPUP:
        state.pending_dialog = ""
        imgui.open_popup(SHORTCUTS_POPUP)
    with controls.menu_popup(SHORTCUTS_POPUP) as opened:
        if not opened:
            return
        imgui.text("Aseprite 1.3.15.5 / Windows bindings")
        widgets.muted("Separate multiple chords with semicolons.")
        state.shortcut_query = widgets.input_text(
            "##shortcut-search", state.shortcut_query, max_length=80, hint="Search shortcuts"
        )
        query = state.shortcut_query.casefold().strip()
        shown = 0
        for kind, target, label, context, trigger in _shortcut_rows():
            haystack = f"{kind} {target} {label}".casefold()
            if query and query not in haystack:
                continue
            key = inker_ops.binding_target(kind, target)
            chord = (
                inker_ops.shortcut_for(
                    kind,
                    target,
                    state.shortcut_overrides,
                    context=context,
                    trigger=trigger,
                )
                or "Unbound"
            )
            picked, _ = controls.selectable(
                f"{label}  —  {chord}##shortcut/{key}/{context}/{trigger}",
                state.shortcut_target == key
                and state.shortcut_context == context
                and state.shortcut_trigger == trigger,
            )
            if picked:
                _select_shortcut(state, kind, target, context, trigger)
            shown += 1
            if shown >= 30:
                widgets.muted("Refine the search to see more entries.")
                break
        if state.shortcut_target:
            imgui.separator()
            state.shortcut_draft = widgets.input_text(
                "Bindings##shortcut-bindings", state.shortcut_draft, max_length=160
            )
            state.shortcut_context = widgets.labeled_combo(
                "Context",
                state.shortcut_context,
                [(name, name or "Everywhere") for name in inker_ops.BINDING_CONTEXTS],
            )
            state.shortcut_trigger = widgets.labeled_combo(
                "Trigger",
                state.shortcut_trigger,
                [(name, name.title()) for name in ("press", "hold", "wheel", "drag")],
            )
            kind, target = state.shortcut_target.split(":", 1)
            if widgets.primary_button("Apply##shortcut-apply"):
                state.shortcut_overrides = inker_ops.set_shortcuts(
                    state.shortcut_overrides,
                    kind,
                    target,
                    [part.strip() for part in state.shortcut_draft.split(";")],
                    context=state.shortcut_context,
                    trigger=state.shortcut_trigger,
                )
                inker_mode.persist(ctx)
            imgui.same_line()
            if controls.button("Default##shortcut-default"):
                state.shortcut_overrides.pop(state.shortcut_target, None)
                _select_shortcut(
                    state, kind, target, state.shortcut_context, state.shortcut_trigger
                )
                inker_mode.persist(ctx)
        imgui.separator()
        if controls.button("Copy JSON##shortcut-export"):
            imgui.set_clipboard_text(inker_ops.shortcuts_json(state.shortcut_overrides))
            state.say("Shortcut overrides copied as JSON.")
        imgui.same_line()
        if controls.button("Paste JSON##shortcut-import"):
            try:
                state.shortcut_overrides = inker_ops.parse_shortcuts(
                    imgui.get_clipboard_text() or ""
                )
            except (TypeError, ValueError):
                state.say("The clipboard does not contain an Inker shortcut file.")
            else:
                inker_mode.persist(ctx)
                state.say("Shortcut overrides imported.")
        imgui.same_line()
        if controls.button("Reset all##shortcut-reset"):
            state.shortcut_overrides = {}
            state.shortcut_target = ""
            inker_mode.persist(ctx)


def header_controls(ctx: Any, doc: Any) -> None:
    """The active layer's blend, opacity and locks -- the Photoshop header.

    Blend first, then opacity, which is the order every layers panel since
    Photoshop has used and therefore the one a user's eye already knows.
    """
    layer = doc.stack.active
    blend = widgets.labeled_combo(
        "Blend",
        layer.blend,
        [(m, m) for m in inker.BLEND_MODES],
        help_text=(
            "How this layer combines with everything under it. Saved into the .ora "
            "so other editors read it the same way."
        ),
    )
    if blend != layer.blend:
        doc.set_layer_props(blend=blend)
    changed, value = widgets.labeled_slider_float(
        "Opacity",
        layer.opacity,
        0.0,
        1.0,
        help_text=(
            "The active layer's opacity. Dragging previews it live and records one "
            "undo step when you let go."
        ),
    )
    if changed:
        # Live while dragging, but only one undo step: set the value directly
        # for the preview and record the step when the drag is released. The
        # value it started from has to be remembered here -- by release the
        # layer already holds the new one, and asking the document to diff it
        # against itself records nothing at all.
        _opacity_drag.setdefault(layer.uid, layer.opacity)
        layer.opacity = value
        doc.invalidate_all()
    if imgui.is_item_deactivated_after_edit():
        was = _opacity_drag.pop(layer.uid, None)
        if was is not None:
            doc.set_layer_props(opacity=layer.opacity, was={"opacity": was})

    widgets.muted("Lock:")
    imgui.same_line()
    if _toggle_icon(
        f"{icons.SQUARE_DASHED}##lockalpha",
        layer.alpha_lock,
        "Lock alpha: paints inside what is already on this layer and never "
        "past its edge -- colours change, transparency does not. The eraser "
        "does nothing here, because erasing is changing transparency.",
    ):
        doc.set_layer_props(alpha_lock=not layer.alpha_lock)
    imgui.same_line()
    if _toggle_icon(
        f"{icons.LOCK}##locklayer",
        layer.locked,
        "Lock layer: refuses every tool -- no strokes, fills, gradients, "
        "filters, lifts or pastes land on it. Renaming, hiding, reordering "
        "and deleting still work, and so do whole-document changes like a "
        "rotate or a crop.",
    ):
        doc.set_layer_props(locked=not layer.locked)
    imgui.same_line()
    # Its own label, not a third button under "Lock:". It is a statement about
    # what a *new cel* starts as, and filing it with the two locks would have
    # the panel claiming it refuses something -- which is the one thing it does
    # not do. Disabled, never hidden, on a still document: the engine refuses
    # it there by name and a greyed button says the setting exists.
    widgets.muted("  Cels:")
    imgui.same_line()
    continuous = doc.anim is not None and doc.anim.tracks[doc.stack.active_index].continuous
    # Greyed *and* explained. A bare ``begin_disabled`` leaves a control that
    # does nothing and says nothing about why, which is the one reading of a
    # dead button that is worse than hiding it -- and this button's whole
    # subject is frames, on a document that has none.
    if _toggle_icon(
        f"{icons.COPY}##continuous",
        continuous,
        "Continuous: drawing on an empty frame of this layer starts from a "
        "copy of the last drawing on it rather than from nothing. A copy, not "
        "a link -- editing it does not change the frame it came from.",
        enabled=doc.anim is not None,
        reason="This drawing has no frames yet -- Animate it first.",
    ):
        doc.set_layer_props(continuous=not continuous)


def _toggle_icon(
    icon: str, engaged: bool, tooltip: str, *, enabled: bool = True, reason: str = ""
) -> bool:
    """An icon button that shows its on state -- the shared selection idiom.

    ``reason`` joins the tooltip rather than replacing it, because a glyph
    button's tooltip is its accessible name: a disabled one that showed only
    the refusal would say why you cannot press it without ever saying what it
    is. ``controls.menu_item_simple`` and ``controls.checkbox`` take the pair
    as fields; ``icon_button`` has only ``enabled``, so the joining is here.
    """
    if not enabled and reason:
        tooltip = tooltip + "\n\n" + reason
    return widgets.icon_button(icon, tooltip, selected=engaged, enabled=enabled)
