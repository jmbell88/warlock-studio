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
from ..inker import brush, transform
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
    "slice": icons.CROP,
    # Not SPRAY_CAN, which the blur tool already carries: two tools drawn with
    # one glyph is a toolbox a user has to read the tooltips of.
    "spray": icons.SPARKLES,
    # Lucide's plain ``lasso`` beside the freehand tool's ``lasso-select``:
    # the same family, which is what the pair are, and distinct glyphs, which
    # the toolbox requires.
    "lasso_poly": icons.LASSO,
}

#: The brush's ink, and only the brush's: this app has brush modes and layer
#: locks where Aseprite has a per-tool ink selector, so offering it on every
#: tool would be four more controls saying the same thing. The keys are
#: ``blend`` (the composite every stroke has always done) and ``replace``,
#: which is a member of ``brush.MODES``.
INK_LABELS = (
    ("blend", "Blend"),
    ("replace", "Replace"),
)

# One entry per mode ``brush.SYMMETRY`` carries, and that is checked rather
# than trusted: the table used to stop at ``xy``, so the radial mode the engine
# implements, the "Ways" slider below and the manual chapter all described a
# setting the combo could never select.
# One entry per ``brush.NIBS`` member, checked against it the same way.
NIB_LABELS = (
    ("soft", "soft (antialiased)"),
    ("pixel", "pixel (round)"),
    ("square", "pixel (square)"),
)

SYMMETRY_LABELS = (
    ("none", "off"),
    ("x", "left / right"),
    ("y", "top / bottom"),
    ("xy", "both"),
    ("radial", "radial"),
)


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
    width = widgets.grid_width(COLUMNS)
    for index, (key, label, shortcut) in enumerate(inker_state.TOOLS):
        selected = state.tool == key
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        icon = TOOL_ICONS.get(key) or label[:1]
        if imgui.button(f"{icon}##tool{key}", (width, sp(30))):
            # Through ``set_tool``, like every other way of picking one: a
            # half-drawn multi-click gesture belongs to the tool that started it.
            state.set_tool(key)
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
        changed, size = widgets.labeled_slider_int(
            "Size", state.brush_size, inker.MIN_BRUSH, inker.MAX_BRUSH
        )
        if changed:
            state.brush_size = inker.clamp_brush(size)
    if tool in PAINT_TOOLS:
        state.nib = widgets.labeled_combo("Nib", state.nib, list(NIB_LABELS))
        widgets.help_marker(
            "Soft is the antialiased disc, which is what a painted reference "
            "wants. The two pixel nibs lay down whole pixels only -- no partial "
            "coverage anywhere -- which is what pixel art wants and what keeps "
            "a drawing's colour count from growing along every edge."
        )
        if state.nib in inker.PIXEL_NIBS:
            # Not for the spray: the corner filter is about a *line*, and the
            # canvas forces it off there -- a ticked box that does nothing is
            # worse than no box.
            if tool != "spray":
                changed, value = imgui.checkbox("Pixel perfect", state.pixel_perfect)
                if changed:
                    state.pixel_perfect = value
                widgets.help_marker(
                    "Drops the doubled corner pixel a freehand diagonal leaves at "
                    "every step, so the line is one pixel wide the whole way."
                )
        else:
            # Hidden rather than disabled: a pixel nib's coverage is 0 or 1 by
            # definition, so there is no falloff for this to shape and a greyed
            # slider would suggest there is one somewhere.
            changed, value = widgets.labeled_slider_float("Hardness", state.hardness, 0.0, 1.0)
            if changed:
                state.hardness = value
        if tool == "brush":
            _ink(state)
        changed, value = widgets.labeled_slider_float("Opacity", state.opacity, 0.05, 1.0)
        if changed:
            state.opacity = value
    if tool == "spray":
        # Spacing, smoothing and the corner filter are all about a *line*, and
        # a spray does not walk one -- the canvas forces the last two off, so
        # showing them would be controls that do nothing.
        changed, rate = widgets.labeled_slider_int("Rate", int(state.spray_rate), 5, 400)
        if changed:
            state.spray_rate = int(rate)
        widgets.help_marker(
            "Dabs a second while the button is held. Size is the width of the "
            "cloud rather than of one dab, so a wide spray is thin and a "
            "narrow one builds up fast."
        )
    elif tool in PAINT_TOOLS:
        changed, value = widgets.labeled_slider_float("Spacing", state.spacing, 0.02, 1.0)
        if changed:
            state.spacing = value
        if tool in ("blur", "smudge"):
            changed, value = widgets.labeled_slider_float("Strength", state.strength, 0.05, 1.0)
            if changed:
                state.strength = value
        changed, value = widgets.labeled_slider_float("Smoothing", state.stabilise, 0.0, 0.95)
        if changed:
            state.stabilise = value
        widgets.help_marker(
            "The brush follows the cursor at a distance instead of exactly, "
            "which turns a shaky line into a smooth one. It catches up when "
            "you stop moving."
        )
        changed, value = widgets.labeled_slider_float("Taper", state.speed_taper, 0.0, 1.0)
        if changed:
            state.speed_taper = value
        widgets.help_marker("How much a fast stroke thins, for a pen-like flick.")
    if tool in SHAPE_TOOLS and tool != "line":
        changed, filled = imgui.checkbox("Filled", state.shape_filled)
        if changed:
            state.shape_filled = filled

    if tool in ("fill", "wand"):
        widgets.section("tolerance")
        _per_tool_note()
        changed, value = widgets.labeled_slider_int("Tolerance", state.wand_tolerance, 0, 255)
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
        # Derived from the engine's own tuple rather than written out, so the
        # combo cannot offer a matrix ``dither`` does not have.
        state.gradient_dither = widgets.combo(
            "Dither",
            state.gradient_dither,
            [("none", "none"), *((k, k) for k in inker.DITHER_ORDERED)],
        )
        widgets.help_marker(
            "Throws away the blend between stops and thresholds each pixel onto "
            "one of them instead, so the ramp lands on exactly the colours you "
            "chose. A selection's soft edge is not dithered."
        )
        _gradient_stops(state)

    if tool == "slice":
        _slices(ctx, state, tab)

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


#: The pane's own ceiling on a slice name. A slice's name lands in a sprite
#: sheet's sidecar and in a TexturePacker atlas, so it is a name other programs
#: read -- ``packwright.document.MAX_NAME_LEN``'s reason, one editor over.
MAX_SLICE_NAME = 64


def _slices(ctx: Any, state: Any, tab: Any) -> None:
    """The slice list and what can be done to the selected one.

    A section in this panel rather than a pane of its own: slices are a *tool's*
    subject, they are drawn on the canvas beside the drawing, and a fourth
    sidebar would cost the canvas its width to hold a list that is usually two
    rows long. It also means no new help anchor -- the chapter this rides is the
    one about the toolbox.

    The overlay checkbox sits *outside* the busy gate on purpose, for the same
    reason the view-rotation buttons on the file row do: it changes nothing
    about the document, and refusing to let the user look at their slices while
    a file is being written would be an editor arguing with them.
    """
    doc = tab.doc
    widgets.section("slices")
    widgets.muted("Drag on the canvas to add one; drag a corner to resize.")
    changed, value = imgui.checkbox("Show with other tools", state.show_slices)
    if changed:
        state.show_slices = value

    imgui.begin_disabled(tab.busy)
    if not doc.slices:
        widgets.muted("No slices yet.")
    for entry in list(doc.slices):
        selected = entry.uid == state.slice_uid
        # The uid in the id, not the index: two slices may share a name, and an
        # index moves the moment one above it is deleted.
        if imgui.selectable(f"{entry.name}##slice{entry.uid}", selected)[0]:
            state.slice_uid = entry.uid
    chosen = doc.slice_by_uid(state.slice_uid)
    if chosen is not None:
        _slice_options(ctx, state, tab, chosen)
    imgui.end_disabled()


def _slice_options(ctx: Any, state: Any, tab: Any, entry: Any) -> None:
    doc = tab.doc
    imgui.set_next_item_width(sp(140))
    # Committed when the field is let go of, not on every keystroke -- the same
    # rule the layer opacity slider beside it follows, and for the same reason:
    # typing "hitbox" is one rename, not six undo steps.
    _changed, name = imgui.input_text(f"Name##slice{entry.uid}", entry.name)
    if imgui.is_item_deactivated_after_edit() and name.strip():
        doc.set_slice(entry.uid, name=name.strip()[:MAX_SLICE_NAME])

    frame_uid = tab.frame_uid
    key = entry.at(frame_uid)
    x0, y0, x1, y1 = key.bounds
    widgets.muted(f"{x0}, {y0}  {x1 - x0} x {y1 - y0}")

    changed, value = imgui.checkbox(f"Pivot##slice{entry.uid}", key.pivot is not None)
    if changed:
        # The centre of the slice when it is switched on, which is a defensible
        # answer a user can then drag -- rather than the origin, which looks
        # like the feature did nothing.
        doc.set_slice(
            entry.uid,
            pivot=None if not value else ((x1 - x0) / 2.0, float(y1 - y0)),
        )
    widgets.help_marker(
        "Where an engine places this sprite -- the point that stays put as the "
        "character turns. The first slice with one decides the sheet's pivot."
    )

    changed, value = imgui.checkbox(f"Nine-slice##slice{entry.uid}", key.center is not None)
    if changed:
        # A third in from each edge: the conventional starting nine-patch, and
        # the one shape that is obviously editable rather than degenerate.
        doc.set_slice(
            entry.uid,
            center=None
            if not value
            else (
                max(1, (x1 - x0) // 3),
                max(1, (y1 - y0) // 3),
                max(2, (x1 - x0) - (x1 - x0) // 3),
                max(2, (y1 - y0) - (y1 - y0) // 3),
            ),
        )
    widgets.help_marker(
        "The stretchable middle of a panel. The four corners stay their own "
        "size and the edges repeat, which is how a UI frame scales."
    )

    if frame_uid is not None:
        keyed = frame_uid in entry.keys
        if imgui.button("Unkey this frame" if keyed else "Key this frame", (-1, 0)):
            doc.set_slice_key(entry.uid, frame_uid, clear=keyed)
        widgets.help_marker(
            "Keys are always explicit. Dragging a slice moves it on every "
            "frame; a key is how one frame is allowed to differ."
        )
    if imgui.button(f"Delete##slice{entry.uid}", (-1, 0)):
        doc.remove_slice(entry.uid)


def _ink(state: Any) -> None:
    """Blend or replace, as a radio pair rather than a combo.

    Two options that a user switches between constantly want to be two clicks
    away from each other, not behind a dropdown -- and the pair is small enough
    to sit on one row, which a combo plus its label is not.
    """
    widgets.field_label("ink")
    for index, (key, label) in enumerate(INK_LABELS):
        if index:
            imgui.same_line()
        if imgui.radio_button(f"{label}##ink{key}", state.paint_ink == key):
            state.paint_ink = key
    widgets.help_marker(
        "Blend composites the colour over what is already there. Replace "
        "writes it exactly -- alpha included -- so it can paint transparency "
        "back down as well as up, which is what recolouring flat pixel art "
        "wants. A soft nib still feathers either way."
    )


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
        _transform_numbers(state, doc)
        return
    if imgui.button("Free transform (Ctrl+T)", (-1, 0)):
        inker_mode.begin_transform(ctx)
    widgets.muted("Rotates, scales and slants the selection, or the whole layer.")


def _transform_numbers(state: Any, doc: Any) -> None:
    """Typed values for what the handles do by feel.

    A drag cannot express "exactly 128 wide" or "exactly 15 degrees", and near
    misses are worse than either -- a sprite one pixel off the grid it was
    drawn on, an italic that is nearly the same slant as the last one. These
    are the same four numbers the handles produce, so nothing new can be
    expressed here; what is new is being able to say them exactly.

    Width and height are converted against ``base_size`` -- the *lifted*
    pixels' size -- rather than against the buffer's current size. Deriving a
    factor from the transformed result would compound every keystroke against
    the last one, which is the anti-compounding rule the drag handles follow
    from the other side.
    """
    buf = doc.floating
    if buf is None:
        return
    base_w, base_h = buf.base_size
    width, height = buf.size

    imgui.set_next_item_width(sp(70))
    changed_x, x = imgui.input_int("X##inkxfx", int(buf.offset[0]), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    changed_y, y = imgui.input_int("Y##inkxfy", int(buf.offset[1]), 0)
    if changed_x or changed_y:
        # Through ``move_floating``'s delta rather than by writing ``offset``:
        # one owner for where a buffer sits, and it bumps ``rev`` for the pane.
        doc.move_floating(int(x) - buf.offset[0], int(y) - buf.offset[1])

    imgui.set_next_item_width(sp(70))
    changed_w, new_w = imgui.input_int("W##inkxfw", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    changed_h, new_h = imgui.input_int("H##inkxfh", int(height), 0)
    if changed_w or changed_h:
        fx = max(1, int(new_w)) / base_w if changed_w else buf.scale[0]
        fy = max(1, int(new_h)) / base_h if changed_h else buf.scale[1]
        if state.transform_link:
            fx = fy = fx if changed_w else fy
        doc.transform_floating(scale=(fx, fy), resample=state.resample)

    imgui.set_next_item_width(sp(150))
    changed, angle = imgui.input_float("Angle##inkxfa", float(buf.angle), 0.0, 0.0, "%.2f")
    if changed:
        doc.transform_floating(angle=angle, resample=state.resample)

    limit = transform.SHEAR_MAX
    imgui.set_next_item_width(sp(150))
    changed, values = imgui.input_float2(
        "Slant##inkxfs", [float(buf.shear[0]), float(buf.shear[1])], "%.1f"
    )
    if changed:
        doc.transform_floating(
            shear=(
                max(-limit, min(float(values[0]), limit)),
                max(-limit, min(float(values[1]), limit)),
            ),
            resample=state.resample,
        )
    widgets.help_marker(
        "Slant in degrees, horizontal then vertical -- an italic, or a shadow "
        "cast along the ground. Numbers only for now; there are no slant "
        "handles on the box. Applied after the scale and before the rotation, "
        "so the two axes stay the page's. Two large slants the same way fight "
        "each other and would squash the picture to a sliver, so a pair that "
        "extreme comes back unslanted."
    )


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
    imgui.same_line()
    # Enabled off the *memory* rather than off "there is no selection": the
    # useful case is exactly re-selecting after something else was selected,
    # and a mask the canvas has outgrown is refused by the engine.
    if widgets.disabled_button("Reselect", doc._last_mask is not None):
        doc.reselect()
    widgets.help_marker(
        "Brings back the selection you last dismissed (Ctrl+Shift+D). A "
        "selection from before a resize or a crop cannot come back -- it "
        "describes a canvas that no longer exists."
    )
    if widgets.disabled_button("Copy to layer", doc.mask is not None):
        doc.layer_from_selection(cut=False)
    imgui.same_line()
    if widgets.disabled_button("Move to layer", doc.mask is not None):
        doc.layer_from_selection(cut=True)
    widgets.help_marker(
        "Ctrl+J copies the selection onto a layer of its own and leaves the "
        "original where it was; Ctrl+Shift+J moves it, cutting it out of the "
        "layer it came from. Either way it is one undo step, and the new layer "
        "lines up with what it came from."
    )
    if imgui.button("This layer"):
        doc.select_layer_alpha()
    widgets.help_marker(
        "Selects what is painted on the active layer, at the coverage it is "
        "painted at -- a soft edge becomes a soft selection."
    )

    imgui.set_next_item_width(-sp(80))
    changed, value = imgui.slider_float("##feather", state.feather_radius, 0.0, 32.0, "%.1f px")
    if changed:
        state.feather_radius = value
    imgui.same_line()
    if widgets.disabled_button("Feather", doc.mask is not None):
        doc.feather_selection(state.feather_radius)

    # Whole pixels, and its own control: feather softens an edge where these
    # *move* it, and one slider serving both would have to pick a unit that is
    # wrong for one of them.
    imgui.set_next_item_width(-sp(80))
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
    state.symmetry = widgets.labeled_combo("Symmetry", state.symmetry, list(SYMMETRY_LABELS))
    if state.symmetry == "radial":
        imgui.set_next_item_width(sp(90))
        changed, count = imgui.slider_int(
            "Ways", int(state.radial_count), brush.MIN_RADIAL, brush.MAX_RADIAL
        )
        if changed:
            state.radial_count = int(count)
    if state.symmetry != "none":
        _symmetry_axis(state)
    changed, value = imgui.checkbox("Grid", state.grid)
    if changed:
        state.grid = value
    if state.grid:
        imgui.same_line()
        imgui.set_next_item_width(sp(80))
        changed, size = imgui.input_int("##gridsize", state.grid_size, 0)
        if changed:
            state.grid_size = max(2, min(512, size))
        changed, value = imgui.checkbox("Snap to grid", state.grid_snap)
        if changed:
            state.grid_snap = value
        widgets.help_marker(
            "Shapes, lines and the marquee land on grid intersections. "
            "Freehand strokes never snap -- quantising a brush to a lattice is "
            "a different tool, not a drawing aid."
        )


def _symmetry_axis(state: Any) -> None:
    """Where the mirrors sit. Empty means the canvas centre.

    Shown only with a symmetry on, and offered as two numbers rather than a
    draggable handle because the useful values are exact ones -- the centre, a
    character's spine, a tile edge -- and a handle can only be dragged near
    them.
    """
    axis = state.symmetry_axis
    imgui.set_next_item_width(sp(120))
    changed, values = imgui.input_float2(
        "Axis##symaxis", list(axis or (0.0, 0.0)), "%.0f"
    )
    if changed:
        state.symmetry_axis = (float(values[0]), float(values[1]))
    imgui.same_line()
    if widgets.disabled_button("Centre##symcentre", axis is not None):
        # Back to None rather than to the middle of the current document: None
        # *is* the centre, and stays the centre across a resize.
        state.symmetry_axis = None
    if axis is None:
        widgets.muted("centred")
