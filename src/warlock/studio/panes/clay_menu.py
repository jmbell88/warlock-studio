"""The right-mouse context menu over the Clay viewport.

The Wings3D idea, and the reason the ops registry exists: the operations that
apply to what is selected, under the cursor, at the moment the user wants them
-- rather than in a toolbar the eye has to leave the model to find. Every row
comes from :mod:`~warlock.studio.clay_ops`, so the menu cannot offer an op the
keyboard does not have or grey out one the tools pane would have run.

This is the only layer that knows imgui exists. The registry decides *what* is
invocable and *whether*; this decides where the popup opens and what a row looks
like, and nothing else.

**An op with parameters opens a second popup rather than running.** Bevel with
no width is a bevel of whatever the last one was, which is the sort of thing
that is right four times and destroys a model on the fifth; the popup follows
the raster editor's resize-dialog idiom -- fields plus Apply, with the last
values remembered on ``ClayState`` so the common case is two clicks.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, clay_ops, widgets

POPUP = "clay-context"
PARAM_POPUP = "clay-op-params"


def draw(ctx: Any, view: Any) -> None:
    """Open and render the menu. Called from the viewport pane, after the image."""
    tab = clay_mode.active(ctx)
    if tab is None:
        return
    state = clay_mode.ensure(ctx)
    if view.menu_request is not None:
        view.menu_request = None
        imgui.open_popup(POPUP)

    if imgui.begin_popup(POPUP):
        _rows(ctx, state, tab, tab.doc)
        imgui.end_popup()
    params_popup(ctx, state, tab)


def _rows(ctx: Any, state: Any, tab: Any, doc: Any) -> None:
    imgui.text_disabled(f"{doc.element_mode} mode")
    imgui.separator()
    if tab.saving:
        # The gate every other control in the app has, and the one this menu
        # did not: ``enabled`` never consulted it, so every row stayed
        # clickable during a save and the click was then swallowed by the
        # ``or tab.saving`` below -- a live-looking menu that did nothing and
        # said nothing. Told once, at the top, rather than as fifteen greyed
        # rows with no reason attached.
        imgui.text_disabled("Saving...")
        imgui.separator()
    for op in clay_ops.menu(doc.element_mode):
        if op.separator_before:
            imgui.separator()
        clicked, _ = imgui.menu_item(op.label, op.key, False, op.enabled(doc) and not tab.saving)
        if not clicked:
            continue
        if op.params:
            state.pending_op = op.name
            state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
            imgui.close_current_popup()
            # Here the id stack *is* a window's, so this opens directly rather
            # than going through open_op_popup.
            imgui.open_popup(PARAM_POPUP)
        else:
            clay_ops.run(ctx, doc, op)


def params_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The fields for a parameterised op, and its Apply button.

    Called from *both* the viewport (for a menu row) and the tools pane (for a
    button), because an imgui popup only renders inside the window whose id
    stack opened it -- a single call site would leave whichever half did not
    make it silently doing nothing when clicked.

    Opened by name rather than by holding the ``Op``: the popup survives across
    frames and the registry is the only thing allowed to own that object.
    """
    if not state.pending_op:
        return
    try:
        op = clay_ops.get(state.pending_op)
    except KeyError:  # pragma: no cover - a stale name from a removed op
        state.pending_op = ""
        state.open_op_popup = False
        return

    if state.open_op_popup:
        # A request from outside a window -- the keyboard path, which cannot
        # call open_popup itself. Cleared here whether or not the popup ends up
        # rendering, so a request can never outlive the frame that made it.
        state.open_op_popup = False
        imgui.open_popup(PARAM_POPUP)
    if not imgui.begin_popup(PARAM_POPUP):
        return
    values = state.op_params.setdefault(op.name, clay_ops.defaults_for(op))
    imgui.text(op.label.rstrip("."))
    imgui.separator()
    for param in op.params:
        label = f"{param.label}##{op.name}-{param.name}"
        if param.integer:
            # Honoured rather than declared. Smooth's "levels" is the only
            # integer parameter and it was drawn as a float field, so it
            # accepted 1.5 and the op then truncated it -- a number the user
            # typed, silently becoming a different one.
            changed, value = imgui.input_int(label, int(values.get(param.name, param.default)))
        else:
            changed, value = imgui.input_float(
                label, float(values.get(param.name, param.default)), param.step
            )
        if changed:
            clamped = min(max(float(value), param.low), param.high)
            values[param.name] = int(clamped) if param.integer else clamped
        if param.warn:
            imgui.text_disabled(param.warn)
    # Greyed rather than drawn live and ignored, which is what "and not
    # tab.saving" after the click amounted to.
    if widgets.disabled_button(f"Apply##{op.name}", not tab.saving):
        clay_ops.run(ctx, tab.doc, op, **values)
        state.pending_op = ""
        imgui.close_current_popup()
    imgui.same_line()
    if imgui.button(f"Cancel##{op.name}"):
        state.pending_op = ""
        imgui.close_current_popup()
    imgui.end_popup()
