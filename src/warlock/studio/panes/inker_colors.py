"""Foreground, background and the swatch row.

Two colours rather than one because the gradient tool needs both ends and X
swapping them is universal muscle memory. The swatches are persisted (unlike
the old editor's fixed palette) because a project has a palette and retyping it
every session is the kind of small friction that makes a tool feel unfinished.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import inker_mode, widgets
from ..manual import render as manual_render
from ..tokens import sp

# One swatch, in *design* pixels -- see ``_swatches``.
SWATCH = 20.0
FLAGS = imgui.ColorEditFlags_.no_inputs.value | imgui.ColorEditFlags_.alpha_bar.value


def _to_rgba(value: Any) -> tuple[int, int, int, int]:
    """imgui's float colour -> the 8-bit tuple the engine writes with."""
    return (
        int(round(value.x * 255)),
        int(round(value.y * 255)),
        int(round(value.z * 255)),
        int(round(value.w * 255)),
    )


def _vec(colour: tuple[int, int, int, int]) -> Any:
    return imgui.ImVec4(*[c / 255.0 for c in colour])


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    widgets.section("colour")
    # After the heading, never before it: help_button is a same_line, and
    # same_line returns to the previous row unconditionally -- called first it
    # lands on whatever the pane above drew.
    manual_render.help_button(ctx, "inker-colors")

    changed, value = imgui.color_edit4("Foreground", _vec(state.fg), FLAGS)
    if changed:
        state.fg = _to_rgba(value)
    changed, value = imgui.color_edit4("Background", _vec(state.bg), FLAGS)
    if changed:
        state.bg = _to_rgba(value)

    if imgui.button("Swap (X)"):
        state.swap_colours()
    imgui.same_line()
    if imgui.button("+ swatch"):
        state.add_swatch(state.fg)
        inker_mode.persist(ctx)

    imgui.dummy((0, 4))
    _swatches(ctx, state)


def _swatches(ctx: Any, state: Any) -> None:
    avail = imgui.get_content_region_avail().x
    # The gap between two swatches is the style's, not 6: a row of n costs
    # n * SWATCH + (n - 1) * spacing, and pricing the gap two pixels under the
    # real one bought a swatch that did not fit and was clipped at the edge.
    gap = imgui.get_style().item_spacing.x
    # Through sp() (K97): a swatch that stayed 20 physical pixels while the
    # panel around it grew with the monitor was a grid of dots on a 4K display.
    side = sp(SWATCH)
    per_row = max(1, int((avail + gap) // (side + gap)))
    for index, colour in enumerate(list(state.swatches)):
        imgui.push_id(f"swatch{index}")
        if imgui.color_button("##swatch", _vec(colour), 0, (side, side)):
            state.fg = colour
        # Right-click removes: a swatch row with no way to prune it fills up
        # with mistakes and stops being useful within a session.
        if imgui.is_item_clicked(1):
            state.swatches.remove(colour)
            inker_mode.persist(ctx)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{colour}  -  right-click to remove")
        imgui.pop_id()
        if index % per_row != per_row - 1:
            imgui.same_line()
    imgui.new_line()
