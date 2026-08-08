"""The tool grid and its per-tool options.

Options are shown for the tool that is selected and hidden otherwise, rather
than laid out as one long form. A brush's hardness means nothing while the wand
is active, and a panel that shows every control at once is how a paint program
becomes unreadable.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, inker, inker_mode, inker_state, theme, widgets
from ..inker_state import PAINT_TOOLS, SELECT_TOOLS, SHAPE_TOOLS
from ..manual import render as manual_render
from ..tokens import sp

# Icon-only, five across: what every paint program's toolbox looks like. The
# name and shortcut live in the tooltip -- three columns of bare labels
# truncated ("Ellipse select" in a 104px button) and anything wider is taller,
# which the canvas pays for.
COLUMNS = 5

TOOL_ICONS = {
    "brush": icons.BRUSH,
    "eraser": icons.ERASER,
    "fill": icons.PAINT_BUCKET,
    "gradient": icons.BLEND,
    "blur": icons.SPRAY_CAN,
    "smudge": icons.HAND,
    "line": icons.SLASH,
    "rect": icons.SQUARE,
    "ellipse": icons.CIRCLE,
    "select": icons.SQUARE_DASHED,
    "select_ellipse": icons.EGG,
    "lasso": icons.LASSO_SELECT,
    "wand": icons.WAND,
    "move": icons.MOVE,
    "eyedropper": icons.PIPETTE,
}

SYMMETRY_LABELS = (("none", "off"), ("x", "left / right"), ("y", "top / bottom"), ("xy", "both"))
BLEND_LABELS = tuple((mode, mode) for mode in inker.BLEND_MODES)


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    tab = state.active
    widgets.section("tools")
    manual_render.help_button(ctx, "inker-tools")
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
    for index, (key, label, shortcut) in enumerate(inker_state.TOOLS):
        selected = state.tool == key
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        icon = TOOL_ICONS.get(key) or label[:1]
        if imgui.button(f"{icon}##tool{key}", (width, sp(30))):
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
        _per_tool_note()
        imgui.set_next_item_width(-1)
        changed, size = imgui.slider_int("Size", state.brush_size, inker.MIN_BRUSH, inker.MAX_BRUSH)
        if changed:
            state.brush_size = inker.clamp_brush(size)
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
        _per_tool_note()
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

    if tool == "eyedropper":
        widgets.section("sample")
        changed, value = imgui.checkbox("This layer only", state.sample_layer)
        if changed:
            state.sample_layer = value
        widgets.help_marker(
            "Off reads the colour you can see, which is the blend of every "
            "visible layer. On reads the active layer's own pixels -- what was "
            "painted into it, before its opacity and blend mode."
        )

    if tool == "gradient":
        widgets.section("gradient")
        state.gradient_kind = widgets.combo(
            "Shape",
            state.gradient_kind,
            [(k, k) for k in inker.GRADIENT_KINDS],
        )
        changed, value = imgui.checkbox("To transparent", state.gradient_to_transparent)
        if changed:
            state.gradient_to_transparent = value
        _gradient_stops(state)

    if _has_options(tool) and imgui.small_button(f"Reset {tool.replace('_', ' ')}##inkreset"):
        state.reset_tool_options(tool)

    # Everything above this line adjusts the *tool*, which a save does not
    # read. Everything below changes the document -- the selection ops each
    # push a history step and Crop rebinds every layer's pixels -- so it waits
    # for the save the same way the canvas, the layers panel and the keyboard
    # shortcuts already do.
    imgui.begin_disabled(tab.busy)
    if tool in SELECT_TOOLS or doc.mask is not None:
        _selection_actions(state, doc)
    _transform_entry(ctx, state, doc)
    imgui.end_disabled()


def _gradient_stops(state: Any) -> None:
    """The stop list, or the two-colour preset when it is empty.

    Empty is the *preset* rather than "no stops": a materialised two-stop list
    would stop following the foreground and background colours, so swapping
    them with X would no longer change the next gradient. Adding a stop is
    therefore the moment the gradient stops being a preset, and the button says
    so by seeding the list from the two colours it was already using.
    """
    widgets.field_label("stops")
    if not state.gradient_stops:
        widgets.muted("foreground to background")
        if imgui.small_button("Add stops##gradstops"):
            state.gradient_stops = [(0.0, tuple(state.fg)), (1.0, tuple(state.bg))]
        return

    remove = -1
    for index, (position, colour) in enumerate(list(state.gradient_stops)):
        imgui.push_id(f"gradstop{index}")
        imgui.set_next_item_width(sp(70))
        changed, value = imgui.slider_float("##pos", float(position), 0.0, 1.0, "%.2f")
        if changed:
            state.gradient_stops[index] = (float(value), colour)
        imgui.same_line()
        edited, rgba = imgui.color_edit4(
            "##col",
            [c / 255.0 for c in colour],
            imgui.ColorEditFlags_.no_inputs.value | imgui.ColorEditFlags_.alpha_bar.value,
        )
        if edited:
            state.gradient_stops[index] = (
                float(position),
                tuple(int(round(c * 255.0)) for c in rgba),
            )
        imgui.same_line()
        # Never below one stop: ``sample`` treats a single stop as a flat
        # colour rather than raising, but a list with none in it has no
        # gradient to draw and no way back to the preset except this button.
        if widgets.disabled_button("x", len(state.gradient_stops) > 1):
            remove = index
        imgui.pop_id()
    if remove >= 0:
        del state.gradient_stops[remove]
    if imgui.small_button("Add##gradadd"):
        state.gradient_stops.append((0.5, tuple(state.fg)))
    imgui.same_line()
    if imgui.small_button("Use fg / bg##gradreset"):
        state.gradient_stops = []


def _has_options(tool: str) -> bool:
    """Whether this tool has anything of its own to reset.

    The move and eyedropper tools have no options at all, and a Reset button
    that clears nothing is a control that says the panel is confused about
    which tool is selected.
    """
    return tool in PAINT_TOOLS or tool in SHAPE_TOOLS or tool in ("fill", "wand", "eyedropper")


def _per_tool_note() -> None:
    """Says out loud that these belong to the tool.

    Without it the feature is invisible in the good case and looks like a bug
    in the bad one: a user who sizes the eraser to 60 and finds the brush still
    at 12 has either been given a convenience or lost a setting, and nothing on
    screen said which.
    """
    widgets.help_marker(
        "These belong to the tool in your hand. Sizing the eraser does not "
        "resize the brush, and switching back finds each one as you left it."
    )


def _transform_entry(ctx: Any, state: Any, doc: Any) -> None:
    """Free transform is a state rather than a tool, so it gets a button rather
    than a slot in the grid -- it takes the canvas over until it is applied."""
    widgets.section("transform")
    if state.transforming:
        widgets.text_colored(theme.ACCENT, "Transforming - Enter applies, Esc cancels.")
        return
    if imgui.button("Free transform (Ctrl+T)", (-1, 0)):
        inker_mode.begin_transform(ctx)
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
    if imgui.button("This layer"):
        doc.select_layer_alpha()
    widgets.help_marker(
        "Selects what is painted on the active layer, at the coverage it is "
        "painted at -- a soft edge becomes a soft selection."
    )

    imgui.set_next_item_width(-80)
    changed, value = imgui.slider_float("##feather", state.feather_radius, 0.0, 32.0, "%.1f px")
    if changed:
        state.feather_radius = value
    imgui.same_line()
    if widgets.disabled_button("Feather", doc.mask is not None):
        doc.feather_selection(state.feather_radius)

    # Whole pixels, and its own control: feather softens an edge where these
    # *move* it, and one slider serving both would have to pick a unit that is
    # wrong for one of them.
    imgui.set_next_item_width(-80)
    changed, steps = imgui.slider_int("##selgrow", int(state.select_steps), 1, 32, "%d px")
    if changed:
        state.select_steps = int(steps)
    imgui.same_line()
    widgets.muted("by")
    has = doc.mask is not None
    if widgets.disabled_button("Grow", has):
        doc.grow_selection(state.select_steps)
    imgui.same_line()
    if widgets.disabled_button("Shrink", has):
        doc.shrink_selection(state.select_steps)
    imgui.same_line()
    if widgets.disabled_button("Border", has):
        doc.border_selection(state.select_steps)
    widgets.help_marker(
        "Border replaces the selection with the band that many pixels either "
        "side of its edge -- fill it and you have stroked the outline."
    )

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
