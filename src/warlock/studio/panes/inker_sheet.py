"""The sheet-correction strip: between the transport and the grid.

Drawn only on a document whose tags name a character sheet
(``inker_sheet.is_sheet``); an ordinary animation sees nothing here. Every
button is an ``inker_ops`` entry pressed through ``inker_ops.run``, so the
menu, this strip and the probe census share one ``enabled``/``reason`` --
the strip draws the controls and decides nothing.

Two rows. The first is *where* a correction goes and the three verbs that
send one (a marked patch, a recolour, a shift); the second is the mirror --
which direction it would land on, the live diff count, the face box, and the
two buttons that write it.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, inker_ops, inker_sheet, widgets
from ..inker import sheetscope
from ..manual import render as manual_render
from ..tokens import sp

#: Every op the strip presses, so a test can assert each is registered.
STRIP_OPS: tuple[str, ...] = (
    "sheet_propagate",
    "sheet_remark",
    "sheet_replace",
    "sheet_shift",
    "sheet_mirror",
    "sheet_mirror_run",
)

_SCOPE_OPTIONS: tuple[tuple[str, str], ...] = tuple(
    (key, sheetscope.SCOPE_LABELS[key]) for key in sheetscope.SCOPES
)


def _press(ctx: Any, name: str, label: str, *, tooltip: str = "") -> bool:
    """One registered op as a button, greyed with its own reason."""
    state = ctx.state.inker
    op = inker_ops.get(name)
    tab = state.active
    enabled = op.enabled(state, tab)
    reason = inker_ops.reason_for(op, state, tab) if not enabled else ""
    if controls.button(
        f"{label}##sheet-{name}", enabled=enabled, reason=reason, tooltip=tooltip or op.hint
    ):
        return inker_ops.run(ctx, op)
    return False


def draw_strip(ctx: Any, tab: Any) -> None:
    state = ctx.state.inker
    if state is None or tab is None:
        return
    inker_sheet.sync_mark(tab)
    if not inker_sheet.is_sheet(tab):
        return
    here = inker_sheet.run_of(tab)
    if here is None:
        widgets.muted("The playhead is outside the sheet's runs.")
        return
    run, offset = here
    widgets.muted(f"{run.animation} / {run.direction}, frame {offset + 1} of {run.frames}")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-sheet")

    # -- row 1: scope and the three verbs ------------------------------------
    imgui.set_next_item_width(sp(230))
    changed, picked = controls.combo(
        "##sheet-scope",
        state.sheet_scope,
        _SCOPE_OPTIONS,
        tooltip="Which other cells a correction is sent to.",
    )
    if changed:
        state.sheet_scope = picked
    imgui.same_line()
    reach = len(inker_sheet.targets(state, tab))
    widgets.muted(f"{reach} cell(s)")
    imgui.same_line()
    _press(ctx, "sheet_propagate", "Propagate patch")
    imgui.same_line()
    _press(ctx, "sheet_remark", "Re-mark")

    imgui.set_next_item_width(sp(120))
    changed, old = controls.color_edit4(
        "##sheet-old",
        [c / 255.0 for c in state.sheet_old],
        imgui.ColorEditFlags_.no_inputs.value,
        tooltip="The colour to replace.",
    )
    if changed:
        state.sheet_old = tuple(int(round(c * 255)) for c in old)
    imgui.same_line()
    widgets.muted("to")
    imgui.same_line()
    imgui.set_next_item_width(sp(120))
    changed, new = controls.color_edit4(
        "##sheet-new",
        [c / 255.0 for c in state.sheet_new],
        imgui.ColorEditFlags_.no_inputs.value,
        tooltip="What it becomes.",
    )
    if changed:
        state.sheet_new = tuple(int(round(c * 255)) for c in new)
    imgui.same_line()
    imgui.set_next_item_width(sp(110))
    changed, tolerance = controls.slider_float(
        "##sheet-tolerance",
        float(state.sheet_tolerance),
        0.0,
        128.0,
        "tol %.0f",
        tooltip="How far from the colour a pixel may be and still be replaced.",
    )
    if changed:
        state.sheet_tolerance = float(tolerance)
    imgui.same_line()
    _press(ctx, "sheet_replace", "Replace across scope")

    imgui.set_next_item_width(sp(90))
    changed, dx = controls.input_int("##sheet-dx", int(state.sheet_dx), 1, 4, tooltip="Right")
    if changed:
        state.sheet_dx = max(-4096, min(int(dx), 4096))
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed, dy = controls.input_int("##sheet-dy", int(state.sheet_dy), 1, 4, tooltip="Down")
    if changed:
        state.sheet_dy = max(-4096, min(int(dy), 4096))
    imgui.same_line()
    _press(ctx, "sheet_shift", "Shift selection across scope")

    # -- row 2: the mirror ---------------------------------------------------
    widgets.divider()
    opposite = sheetscope.opposite(run.direction)
    if opposite is None:
        widgets.muted(f"{run.direction} has no mirror direction.")
    else:
        widgets.muted(f"mirror -> {opposite}")
    imgui.same_line()
    changed, on = controls.switch(
        "Preview diff",
        bool(tab.mirror_preview),
        control_id="sheet-mirror-preview",
        enabled=opposite is not None,
        reason="" if opposite is not None else inker_sheet.mirror_reason(state, tab),
        tooltip="Show on the canvas which pixels the mirror direction would take.",
    )
    if changed:
        tab.mirror_preview = on
    imgui.same_line()
    imgui.set_next_item_width(sp(120))
    changed, fraction = controls.slider_float(
        "##sheet-face",
        float(tab.face_fraction),
        0.0,
        0.6,
        "face %.2f",
        tooltip="How much of the sprite, from the top, is face and stays unmirrored.",
    )
    if changed:
        tab.face_fraction = max(0.0, min(float(fraction), 1.0))
    report = inker_sheet.mirror_report(tab) if opposite is not None else None
    if report is not None:
        imgui.same_line()
        widgets.muted(f"{report[0]} px differ outside the face, {report[1]} inside")
    imgui.same_line()
    _press(ctx, "sheet_mirror", f"Apply to {opposite}" if opposite else "Apply to mirror")
    imgui.same_line()
    _press(ctx, "sheet_mirror_run", f"Apply whole {run.direction} run")
