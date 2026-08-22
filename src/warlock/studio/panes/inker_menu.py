"""The Inker's menu strip: seven names, and every verb the editor has.

Drawn from ``inker_canvas.draw`` before the tab bar, which is Aseprite's own
order -- menu, tabs, context bar, canvas, status -- and inside the centre
window because an imgui popup only renders in the id stack of the window that
opened it.

**A row is an ``inker_ops.Op`` and nothing else.** The strip decides where a
popup opens and what a row looks like; the registry decides what exists,
whether it can run and what to say when it cannot. That is the whole of why
this file is short: there is no list of verbs in it.

**Not ``imgui.begin_menu_bar``.** A menu bar belongs to a *window* via a flag,
and the centre pane's flags are ``layout.pane``'s; changing the child's content
region is exactly what the canvas's height reservation depends on, and a
negative child height silently kills the canvas. A row of ghost buttons that
each open an anchored popup is the same picture with none of that risk -- and
it goes through :mod:`~warlock.studio.toolbar`, so a strip too wide for the
pane collapses into an overflow with the menu names back rather than clipping
"View" off the end.

This pane replaced ``inker_bridge``'s five verb blocks. That module keeps its
four popups and their dialog machinery and is no longer drawn as a pane; see
its docstring.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, inker_mode, inker_ops, toolbar, widgets
from ..tokens import sp

BAR = "inker-menu"
PARAM_POPUP = "inker-op-params"
PROPERTIES_POPUP = "inker-layer-properties"

#: How wide an ``Op.hint`` wraps, in design pixels. ``clay_menu``'s number and
#: its reason: a popup that auto-sizes past its own panel reads as a window.
HINT_WRAP = 300


def draw(ctx: Any) -> None:
    """The strip, its popups, and the parameter dialog behind them."""

    state = inker_mode.ensure(ctx)
    tab = state.active
    items = [
        toolbar.Item(name, name, priority=index // 3, role=controls.ButtonRole.GHOST)
        for index, name in enumerate(inker_ops.MENUS)
    ]
    clicked = toolbar.toolbar(BAR, items)
    if clicked:
        imgui.open_popup(controls.menu_bar_id(BAR, clicked))
    for name in inker_ops.MENUS:
        with controls.menu_popup(controls.menu_bar_id(BAR, name)) as opened:
            if opened:
                _rows(ctx, state, tab, name)
    _params_popup(ctx, state)
    _properties_popup(ctx, state, tab)


def _rows(ctx: Any, state: Any, tab: Any, name: str) -> None:
    for op in inker_ops.menu(name):
        if op.separator_before:
            controls.menu_separator()
        enabled = bool(op.enabled(state, tab))
        hit = controls.menu_item(
            f"{op.label}##{BAR}/{op.name}",
            op.key,
            False,
            enabled,
            reason=op.reason,
            tooltip=op.hint,
        )
        if not (hit[0] if isinstance(hit, tuple) else hit):
            continue
        if op.params:
            # Opened by *name*: the popup survives across frames and the
            # registry is the only thing allowed to own the ``Op`` object.
            state.pending_op = op.name
            state.op_params.setdefault(op.name, inker_ops.defaults_for(op))
            imgui.close_current_popup()
            imgui.open_popup(PARAM_POPUP)
        else:
            inker_ops.run(ctx, op)


def _params_popup(ctx: Any, state: Any) -> None:
    """The fields for a parameterised op, and its Apply button.

    ``clay_menu.params_popup``'s shape and its argument: a Grow with no amount
    is a grow by whatever the last one was, which is right four times and wrong
    on the fifth. The last values are remembered on the state, so the common
    case is still two clicks.
    """

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
        from . import inker_layers

        inker_layers.header_controls(ctx, tab.doc)
