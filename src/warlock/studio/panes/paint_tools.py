"""The tool grid and its per-tool options.

Options are shown for the tool that is selected and hidden otherwise, rather
than laid out as one long form. A brush's hardness means nothing while the wand
is active, and a panel that shows every control at once is how a paint program
becomes unreadable.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import paint, paint_mode, paint_state, theme, widgets
from ..paint_state import PAINT_TOOLS, SELECT_TOOLS, SHAPE_TOOLS

# Three across, so the grid stays inside the 340-pixel sidebar.
COLUMNS = 3

SYMMETRY_LABELS = (("none", "off"), ("x", "left / right"), ("y", "top / bottom"), ("xy", "both"))
BLEND_LABELS = tuple((mode, mode) for mode in paint.BLEND_MODES)


def draw(ctx: Any) -> None:
    state = paint_mode.ensure(ctx)
    tab = state.active
    widgets.section("tools")
    _grid(state)
    imgui.dummy((0, 6))
    if tab is None:
        widgets.muted("Open something to paint on.")
        return
    _options(ctx, state, tab)
    imgui.dummy((0, 6))
    _canvas_options(state)


def _grid(state: Any) -> None:
    width = (imgui.get_content_region_avail().x - 8 * (COLUMNS - 1)) / COLUMNS
    for index, (key, label, shortcut) in enumerate(paint_state.TOOLS):
        selected = state.tool == key
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        if imgui.button(f"{label}##tool{key}", (width, 26)):
            state.tool = key
        if selected:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label}  ({shortcut})")
        if index % COLUMNS != COLUMNS - 1:
            imgui.same_line()
    imgui.new_line()


def _options(ctx: Any, state: Any, tab: Any) -> None:
    tool = state.tool
    doc = tab.doc

    if tool in PAINT_TOOLS or tool in SHAPE_TOOLS:
        widgets.section("brush")
        imgui.set_next_item_width(-1)
        changed, size = imgui.slider_int("Size", state.brush_size, paint.MIN_BRUSH, paint.MAX_BRUSH)
        if changed:
            state.brush_size = paint.clamp_brush(size)
    if tool in PAINT_TOOLS:
        imgui.set_next_item_width(-1)
        changed, value = imgui.slider_float("Hardness", state.hardness, 0.0, 1.0)
        if changed:
            state.hardness = value
        imgui.set_next_item_width(-1)
        changed, value = imgui.slider_float("Opacity", state.opacity, 0.05, 1.0)
        if changed:
            state.opacity = value
        imgui.set_next_item_width(-1)
        changed, value = imgui.slider_float("Spacing", state.spacing, 0.02, 1.0)
        if changed:
            state.spacing = value
        if tool in ("blur", "smudge"):
            imgui.set_next_item_width(-1)
            changed, value = imgui.slider_float("Strength", state.strength, 0.05, 1.0)
            if changed:
                state.strength = value
    if tool in SHAPE_TOOLS and tool != "line":
        changed, filled = imgui.checkbox("Filled", state.shape_filled)
        if changed:
            state.shape_filled = filled

    if tool in ("fill", "wand"):
        widgets.section("tolerance")
        imgui.set_next_item_width(-1)
        changed, value = imgui.slider_int("Tolerance", state.wand_tolerance, 0, 255)
        if changed:
            state.wand_tolerance = value
        changed, value = imgui.checkbox("Contiguous", state.wand_contiguous)
        if changed:
            state.wand_contiguous = value
        widgets.help_marker(
            "Off selects every similar pixel in the image, not just the ones touching."
        )

    if tool == "gradient":
        widgets.section("gradient")
        state.gradient_kind = widgets.combo(
            "Shape",
            state.gradient_kind,
            [(k, k) for k in paint.GRADIENT_KINDS],
        )
        changed, value = imgui.checkbox("To transparent", state.gradient_to_transparent)
        if changed:
            state.gradient_to_transparent = value

    if tool in SELECT_TOOLS or doc.mask is not None:
        _selection_actions(state, doc)
    _transform_entry(ctx, state, doc)


def _transform_entry(ctx: Any, state: Any, doc: Any) -> None:
    """Free transform is a state rather than a tool, so it gets a button rather
    than a slot in the grid -- it takes the canvas over until it is applied."""
    widgets.section("transform")
    if state.transforming:
        widgets.text_colored(theme.ACCENT, "Transforming - Enter applies, Esc cancels.")
        return
    if imgui.button("Free transform (Ctrl+T)", (-1, 0)):
        paint_mode.begin_transform(ctx)
    widgets.muted("Rotates and scales the selection, or the whole layer.")


def _selection_actions(state: Any, doc: Any) -> None:
    widgets.section("selection")
    widgets.muted("Shift adds, Alt subtracts.")
    if imgui.button("All"):
        doc.select_all()
    imgui.same_line()
    if widgets.disabled_button("None", doc.mask is not None):
        doc.deselect()
    imgui.same_line()
    if imgui.button("Invert"):
        doc.invert_selection()
    imgui.set_next_item_width(-80)
    changed, value = imgui.slider_float("##feather", state.feather_radius, 0.0, 32.0, "%.1f px")
    if changed:
        state.feather_radius = value
    imgui.same_line()
    if widgets.disabled_button("Feather", doc.mask is not None):
        doc.feather_selection(state.feather_radius)
    if widgets.disabled_button("Crop to selection", doc.mask is not None):
        doc.crop_to_selection()


def _canvas_options(state: Any) -> None:
    widgets.section("canvas")
    state.symmetry = widgets.combo("Symmetry", state.symmetry, list(SYMMETRY_LABELS))
    changed, value = imgui.checkbox("Grid", state.grid)
    if changed:
        state.grid = value
    if state.grid:
        imgui.same_line()
        imgui.set_next_item_width(80)
        changed, size = imgui.input_int("##gridsize", state.grid_size, 0)
        if changed:
            state.grid_size = max(2, min(512, size))
