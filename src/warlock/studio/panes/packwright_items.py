"""Packwright's right-top pane: what the last pack actually produced.

A pack that could not fit says so *here* rather than showing an empty list,
which is the one outcome that reads as success when it is the opposite. The
error text comes from the engine -- ``layout`` raises with the number and the
remedy -- and is carried through ``PackTab.pack_error`` by the mode's
``on_task_failed``.

The selection is the same ``PackwrightState.selected`` the sources list and the
preview use: one answer to "which sprite are we talking about", rather than
three that have to be kept in step.
"""

from __future__ import annotations

from typing import Any

from .. import icons, packwright_mode, theme, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("Packed")
    manual_render.help_button(ctx, "packwright-items")

    if tab is None:
        widgets.muted("Start or open an atlas first.")
        return

    if tab.pack_error:
        widgets.text_colored(theme.ERR, f"{icons.TRIANGLE_ALERT} {tab.pack_error}")
        imgui.dummy((0, 4))
        widgets.muted_wrapped(
            "Raise the max size, turn trimming on, or split this into two atlases."
        )
        return

    layout = tab.layout
    if layout is None:
        widgets.muted("Packing..." if tab.packing else "Add a sprite to pack.")
        return

    used = sum(frame.w * frame.h for frame in layout.frames)
    total = max(layout.width * layout.height, 1)
    widgets.muted(
        f"{len(layout.frames)} sprite(s) in {layout.width} x {layout.height} px "
        f"-- {100 * used // total}% covered"
    )
    if layout.is_grid:
        widgets.muted(
            f"grid: {layout.columns} x {layout.rows} cells of "
            f"{layout.cell_w} x {layout.cell_h}"
        )
    if tab.packing:
        widgets.muted("Repacking...")

    imgui.dummy((0, 6))
    by_key = {source.key: source for source in tab.doc.sources}
    for frame in layout.frames:
        source = by_key.get(frame.key)
        uid = source.uid if source is not None else None
        selected = uid is not None and state.selected == uid
        imgui.push_id(frame.key)
        if imgui.selectable(f"{frame.name}##item", selected)[0] and uid is not None:
            state.selected = None if selected else uid
        if imgui.is_item_hovered():
            trimmed = " (trimmed)" if frame.trimmed else ""
            imgui.set_tooltip(
                f"{frame.w} x {frame.h} at {frame.x}, {frame.y}{trimmed}\n"
                f"source {frame.source_w} x {frame.source_h}"
            )
        if frame.empty:
            imgui.same_line()
            widgets.muted("(blank)")
        imgui.pop_id()
