"""The tool grid and its per-tool options.

Options are shown for the tool that is selected and hidden otherwise, rather
than laid out as one long form. A brush's hardness means nothing while the wand
is active, and a panel that shows every control at once is how a paint program
becomes unreadable.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, fonts, icons, inker, inker_mode, inker_state, theme, tokens, widgets
from ..inker import brush, transform
from ..inker_state import (
    OPEN_SHAPE_TOOLS,
    PAINT_TOOLS,
    SELECT_TOOLS,
    SHAPE_TOOLS,
    STAMP_TOOLS,
)
from ..manual import render as manual_render
from ..tokens import sp

# Icon-only, five across: what every paint program's toolbox looks like. The
# name and shortcut live in the tooltip -- three columns of bare labels
# truncated ("Ellipse select" in a 104px button) and anything wider is taller,
# which the canvas pays for.
COLUMNS = 5

#: One toolbox button, in design px. The same number ``_grid`` draws with, said
#: once so :func:`grid_height` cannot drift from it.
BUTTON_H = 30.0

#: How much of the options block has to be reachable without scrolling, in
#: design px, before the toolbox is allowed to claim the rest of the pane.
#: Three rows and a heading: the Brush section's own name, ``Size`` and its
#: slider. Not "all of it" -- the panel scrolls and is meant to -- but a
#: section heading with its first control cut in half is not a panel that
#: scrolls, it is one that looks broken.
OPTIONS_FLOOR = 132.0

#: The gap ``draw`` leaves under the grid, in design px. Named so the dummy
#: that draws it and the reservation that counts it are one number: the dummy
#: used to emit 6 *physical* px while :func:`grid_height` counted 6 design px,
#: which is a drift of ``6 * (SCALE - 1)`` at every UI scale but 1.0.
GRID_GAP = 6.0


def _reserve(rows: int, heading: float, spacing: float, scale: float) -> float:
    """The toolbox reservation in design px, from physically measured parts.

    Split out so the arithmetic is assertable without a GL context: ``heading``
    and ``spacing`` arrive in physical px exactly as the style hands them over,
    ``BUTTON_H`` and ``GRID_GAP`` are design px, and the answer is design px
    because ``main`` passes it back through ``sp``. The parts measured at one
    scale must therefore reserve the same design height at every scale.
    """
    scale = max(scale, 0.001)
    return heading / scale + rows * (BUTTON_H + spacing / scale) + GRID_GAP


def grid_height() -> float:
    """The toolbox block's height in design px: heading, rows, trailing gap.

    Deterministic, because the toolbox is a fixed table laid out at a fixed
    button height -- which is what lets the workspace reserve room for the
    options *below* it instead of guessing a share and hoping.

    The gap between rows is asked of the style for the reason ``grid_width``
    asks for the horizontal one: a literal is right at UI scale 1.0 and wrong
    at every other, and 1.0 is the scale the smoke suite runs at. The heading
    row is *measured* for the same reason 28.0 was wrong: ``section`` draws
    the label in ``fonts.label`` and the manual's help button rides the same
    row at frame height, and both of those scale with the font.
    """
    rows = -(-len(inker_state.TOOLS) // COLUMNS)
    style = imgui.get_style()
    with fonts.label(imgui):
        label_h = imgui.get_text_line_height()
    heading = max(label_h, imgui.get_frame_height()) + style.item_spacing.y
    return _reserve(rows, heading, style.item_spacing.y, tokens.SCALE)


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
    "text": icons.TYPE,
    # The palette, because that is literally what this tool paints with: the
    # colours it can produce are the swatches selected in the Colour panel and
    # nothing else.
    "shade": icons.PALETTE,
    # The three clicked shapes (Q-c). Lucide's ``waypoints`` is a run of points
    # joined by segments, which is the polyline exactly; ``pentagon`` is the
    # closed one; and ``spline`` is a curve drawn with its control points *on*
    # it, which is what a Catmull-Rom through clicked vertices is.
    "polyline": icons.WAYPOINTS,
    "polygon": icons.PENTAGON,
    "curve": icons.SPLINE,
    # The tile stamp. Lucide's ``grid-3x3`` is the lattice a tilemap layer
    # *is*, and it is the same glyph Plotter's tileset panel already carries --
    # deliberately, because the two panels are the same idea in two modes.
    "tile": icons.GRID,
}

#: The shading tool's direction, as a radio pair for ``_ink``'s reason: two
#: options a user switches between constantly want to be two clicks apart
#: rather than behind a dropdown. The keys are the ``shade_dir`` values the
#: engine takes.
SHADE_LABELS = (
    (1, "Forward"),
    (-1, "Back"),
)

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

#: How an image brush places its dabs. One entry per ``brush.STAMP_ALIGN``
#: member, checked against it the way the nib and symmetry tables are.
STAMP_ALIGN_LABELS = (
    ("free", "free"),
    ("aligned", "aligned to a grid"),
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
    widgets.section("Tools")
    manual_render.help_button(ctx, "inker-tools")
    _grid(state, None if tab is None else tab.doc)
    # Through ``sp`` because ``grid_height`` counts this gap in design px --
    # an unscaled 6 here is a reservation short by 6 * (SCALE - 1).
    imgui.dummy((0, sp(GRID_GAP)))
    if tab is None:
        # The greyed grid above already says the toolbox has nothing to act on.
        # The canvas's ``nothing_open`` is the one voice for the empty state.
        return
    _options(ctx, state, tab)
    imgui.dummy((0, 6))
    _canvas_options(ctx, state)


def _grid(state: Any, doc: Any = None) -> None:
    """The icon grid, with any tool this document cannot use greyed out.

    Greyed rather than hidden, and the tooltip carries the reason: a toolbox
    that quietly loses a button when a document is not indexed is one where the
    feature looks like it was imagined. ``allow_when_disabled`` is what makes
    the tooltip reachable at all -- imgui swallows hover on a disabled item,
    which is exactly the state whose explanation matters
    (``widgets.disabled_button`` has the long version of this note).
    """
    width = widgets.grid_width(COLUMNS)
    for index, (key, label, shortcut) in enumerate(inker_state.TOOLS):
        selected = state.tool == key
        reason = inker_state.tool_reason(key, doc)
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        icon = TOOL_ICONS.get(key) or label[:1]
        if reason:
            imgui.begin_disabled()
        clicked = controls.button(f"{icon}##tool{key}", (width, sp(30)))
        if reason:
            imgui.end_disabled()
        if clicked and not reason:
            # Through ``set_tool``, like every other way of picking one: a
            # half-drawn multi-click gesture belongs to the tool that started it.
            state.set_tool(key)
        if selected:
            imgui.pop_style_color()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            # Both bindings, because three tools have two: the tooltip is the
            # only place a glyph button says what it does, so it is also the
            # only place the second chord can be found.
            chords = shortcut
            alt = inker_mode.ALT_TOOL_CHORDS.get(key)
            if alt:
                chords = f"{shortcut} or {alt}"
            note = f"{label}  ({chords})"
            imgui.set_tooltip(f"{note}\n{reason}" if reason else note)
        if index % COLUMNS != COLUMNS - 1:
            imgui.same_line()
    imgui.new_line()


def _options(ctx: Any, state: Any, tab: Any) -> None:
    tool = state.tool
    doc = tab.doc

    if tool in PAINT_TOOLS or tool in SHAPE_TOOLS:
        widgets.section("Brush")
        _per_tool_note()
        changed, size = widgets.labeled_slider_int(
            "Size", state.brush_size, inker.MIN_BRUSH, inker.MAX_BRUSH
        )
        if changed:
            state.brush_size = inker.clamp_brush(size)
    if tool in PAINT_TOOLS:
        # **An indexed document is offered the pixel nibs only**, and the soft
        # one is taken off the list rather than greyed beside them. It is not a
        # restriction the engine needs -- a soft nib is legal there and the
        # commit funnel simply thresholds it -- which is exactly the problem: the
        # control would go on promising a feathered edge and deliver a hard one.
        # A menu that cannot lie is better than a slider that can. (Aseprite
        # reaches the same place from the other direction: it has no
        # antialiasing in indexed mode at all.)
        indexed = bool(getattr(tab.doc, "is_indexed", False)) if tab is not None else False
        labels = [pair for pair in NIB_LABELS if not indexed or pair[0] in inker.PIXEL_NIBS]
        if indexed and state.nib not in inker.PIXEL_NIBS:
            state.nib = labels[0][0]
        state.nib = widgets.labeled_combo(
            "Nib",
            state.nib,
            labels,
            help_text=(
                "Soft is the antialiased disc, which is what a painted reference "
                "wants. The two pixel nibs lay down whole pixels only -- no partial "
                "coverage anywhere -- which is what pixel art wants and what keeps "
                "a drawing's colour count from growing along every edge."
                + (
                    "\n\nThis document is indexed, so every pixel is a palette slot "
                    "and there is no partial coverage to be had: the soft nib is not "
                    "offered because it could not do what it says."
                    if indexed
                    else ""
                )
            ),
        )
        if state.nib in inker.PIXEL_NIBS:
            # Not for the spray: the corner filter is about a *line*, and the
            # canvas forces it off there -- a ticked box that does nothing is
            # worse than no box.
            if tool != "spray":
                changed, value = controls.checkbox("Pixel perfect", state.pixel_perfect)
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
        if tool == "brush" and state.tip_for(tool) is None:
            # Hidden while an image tip is loaded, for the reason hardness is
            # hidden on a pixel nib: a captured picture's alpha is both its
            # shape and its transparency, so a copy ink has nothing left to say
            # about it -- and a radio pair that changed nothing would read as
            # the tool ignoring it. (The engine agrees from the other side: a
            # tip handed to the replace mode is dropped.)
            _ink(state)
        if tool == "shade":
            _shading(state, doc)
        if tool != "shade":
            # Hidden for shading, for the reason hardness is hidden on a pixel
            # nib: a shift lands on the next swatch of the ramp exactly or it
            # does not happen, so there is no partial version of it for an
            # opacity to scale, and a slider that changed nothing would read as
            # the tool ignoring it.
            changed, value = widgets.labeled_slider_float(
                "Opacity", state.opacity, 0.05, 1.0, percent=True
            )
            if changed:
                state.opacity = value
    if tool == "spray":
        # Spacing, smoothing and the corner filter are all about a *line*, and
        # a spray does not walk one -- the canvas forces the last two off, so
        # showing them would be controls that do nothing.
        changed, rate = widgets.labeled_slider_int(
            "Rate",
            int(state.spray_rate),
            5,
            400,
            help_text=(
                "Dabs a second while the button is held. Size is the width of the "
                "cloud rather than of one dab, so a wide spray is thin and a "
                "narrow one builds up fast."
            ),
        )
        if changed:
            state.spray_rate = int(rate)
    elif tool in PAINT_TOOLS:
        changed, value = widgets.labeled_slider_float(
            "Spacing", state.spacing, 0.02, 1.0, percent=True
        )
        if changed:
            state.spacing = value
        if tool in ("blur", "smudge"):
            changed, value = widgets.labeled_slider_float(
                "Strength", state.strength, 0.05, 1.0, percent=True
            )
            if changed:
                state.strength = value
        changed, value = widgets.labeled_slider_float(
            "Smoothing",
            state.stabilise,
            0.0,
            0.95,
            percent=True,
            help_text=(
                "The brush follows the cursor at a distance instead of exactly, "
                "which turns a shaky line into a smooth one. It catches up when "
                "you stop moving."
            ),
        )
        if changed:
            state.stabilise = value
        changed, value = widgets.labeled_slider_float(
            "Taper",
            state.speed_taper,
            0.0,
            1.0,
            help_text="How much a fast stroke thins, for a pen-like flick.",
        )
        if changed:
            state.speed_taper = value
    if tool in STAMP_TOOLS:
        # Last, not in the middle of the brush options. ``section`` tints from
        # its heading to the next one, so opening this one early put the ink
        # radio, Opacity, Spacing, Smoothing and Taper inside a block headed
        # "Image brush" -- controls it does not configure, under a name that
        # says it does, on the three most-used tools in the box.
        _image_brush(ctx, state, tab)
    if tool in SHAPE_TOOLS and tool not in OPEN_SHAPE_TOOLS:
        changed, filled = controls.checkbox("Filled", state.shape_filled)
        if changed:
            state.shape_filled = filled

    if tool in ("fill", "wand"):
        widgets.section("Tolerance")
        _per_tool_note()
        changed, value = widgets.labeled_slider_int("Tolerance", state.wand_tolerance, 0, 255)
        if changed:
            state.wand_tolerance = value
        changed, value = controls.checkbox("Contiguous", state.wand_contiguous)
        if changed:
            state.wand_contiguous = value
        widgets.help_marker(
            "Off selects every similar pixel in the image, not just the ones touching."
        )

    if tool == "eyedropper":
        widgets.section("Sample")
        changed, value = controls.checkbox("This layer only", state.sample_layer)
        if changed:
            state.sample_layer = value
        widgets.help_marker(
            "Off reads the colour you can see, which is the blend of every "
            "visible layer. On reads the active layer's own pixels -- what was "
            "painted into it, before its opacity and blend mode."
        )

    if tool == "gradient":
        widgets.section("Gradient")
        state.gradient_kind = widgets.labeled_combo(
            "Shape",
            state.gradient_kind,
            [(k, k) for k in inker.GRADIENT_KINDS],
        )
        changed, value = controls.checkbox("To transparent", state.gradient_to_transparent)
        if changed:
            state.gradient_to_transparent = value
        # Derived from the engine's own tuple rather than written out, so the
        # combo cannot offer a matrix ``dither`` does not have.
        state.gradient_dither = widgets.labeled_combo(
            "Dither",
            state.gradient_dither,
            [("none", "none"), *((k, k) for k in inker.DITHER_ORDERED)],
            help_text=(
                "Throws away the blend between stops and thresholds each pixel onto "
                "one of them instead, so the ramp lands on exactly the colours you "
                "chose. A selection's soft edge is not dithered."
            ),
        )
        _gradient_stops(state)

    if tool == "tile":
        widgets.section("Tile stamp")
        widgets.muted_wrapped(
            "Click or drag on the canvas to put the picked tile down. The stamp "
            "writes cells, never pixels -- there is nothing here to size."
        )

    _tile_behavior(state, doc)

    if tool == "slice":
        _slices(ctx, state, tab)

    if tool == "text":
        # The controls themselves are on the canvas, in the popup the click
        # opens: a font, a size and an AA toggle sitting in this panel would be
        # a form the user fills in *before* choosing where the word goes, and
        # the whole gesture is "put this here". What is left for the panel is
        # saying so, and the Reset button ``_has_options`` gives every tool
        # that remembers something.
        widgets.section("Text")
        _per_tool_note()
        widgets.muted_wrapped(
            "Click on the canvas to type. What lands is pixels rather than a "
            "text object, so re-editing it is retyping it."
        )

    reset = f"Reset {inker_state.tool_label(tool)}##inkreset"
    if _has_options(tool) and controls.small_button(reset):
        state.reset_tool_options(tool)

    if _has_options(tool):
        _presets(ctx, state)

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


def _tile_behavior(state: Any, doc: Any) -> None:
    """Manual / Auto / Stack: what a *pixel* edit on a tilemap layer does to
    the tileset under it.

    Shown only while a tilemap layer is active, which is the one case where the
    three words mean anything -- hidden rather than greyed, unlike the toolbox's
    own rule, because this is not a control the user is looking for and failing
    to find: on an ordinary layer there is no tileset for it to describe, so a
    disabled radio row would be a promise about a feature the document does not
    have. (Aseprite does the same: the selector appears with the mode.)

    It writes ``doc.tile_behavior`` **directly**, and that is the whole point:
    the field is view state, never serialized, and a toggle here must not push
    a history step or move the document's saved head. A radio row rather than a
    combo, for ``_ink``'s reason -- three options a user switches between
    constantly want to be one click apart.
    """
    if doc is None or doc.active_tilemap_uid() is None:
        return
    imgui.dummy((0, 6))
    widgets.section("Tiles")
    current = str(doc.tile_behavior)
    width = widgets.grid_width(len(inker_state.TILE_BEHAVIORS))
    for index, (key, label, why) in enumerate(inker_state.TILE_BEHAVIORS):
        selected = current == key
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value,
                imgui.get_style().color_(imgui.Col_.button_active.value),
            )
        if controls.button(f"{label}##tilebehav{key}", (width, 0)):
            doc.tile_behavior = key
        if selected:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label}\n{why}")
        if index != len(inker_state.TILE_BEHAVIORS) - 1:
            imgui.same_line()
    imgui.new_line()
    note = next(
        (why for key, _label, why in inker_state.TILE_BEHAVIORS if key == current),
        "",
    )
    widgets.muted_wrapped(note)


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
    widgets.section("Slices")
    widgets.muted_wrapped("Drag on the canvas to add one; drag a corner to resize.")
    changed, value = controls.checkbox("Show with other tools", state.show_slices)
    if changed:
        state.show_slices = value

    imgui.begin_disabled(tab.busy)
    if not doc.slices:
        widgets.muted("No slices yet.")
    for entry in list(doc.slices):
        selected = entry.uid == state.slice_uid
        # The uid in the id, not the index: two slices may share a name, and an
        # index moves the moment one above it is deleted.
        if controls.selectable(f"{entry.name}##slice{entry.uid}", selected)[0]:
            state.slice_uid = entry.uid
    if doc.slices and controls.button("Export slices as PNGs", (-1, 0)):
        inker_mode.export_slices(ctx, tab)
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
    _changed, name = controls.input_text(f"Name##slice{entry.uid}", entry.name)
    if imgui.is_item_deactivated_after_edit() and name.strip():
        doc.set_slice(entry.uid, name=name.strip()[:MAX_SLICE_NAME])

    frame_uid = tab.frame_uid
    key = entry.at(frame_uid)
    x0, y0, x1, y1 = key.bounds
    widgets.muted(f"{x0}, {y0}  {x1 - x0} x {y1 - y0}")

    changed, value = controls.checkbox(f"Pivot##slice{entry.uid}", key.pivot is not None)
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

    changed, value = controls.checkbox(f"Nine-slice##slice{entry.uid}", key.center is not None)
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
        # ``tooltip=`` on the button, not a trailing ``help_marker``: a
        # ``(-1, 0)`` button leaves zero room on its line, so the marker
        # wrapped under it and read as the Delete button's.
        if controls.button(
            "Unkey this frame" if keyed else "Key this frame",
            (-1, 0),
            tooltip=(
                "Keys are always explicit. Dragging a slice moves it on every"
                " frame; a key is how one frame is allowed to differ."
            ),
        ):
            doc.set_slice_key(entry.uid, frame_uid, clear=keyed)
    if controls.button(f"Delete##slice{entry.uid}", (-1, 0)):
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
        if controls.radio_button(f"{label}##ink{key}", state.paint_ink == key):
            state.paint_ink = key
    widgets.help_marker(
        "Blend composites the colour over what is already there. Replace "
        "writes it exactly -- alpha included -- so it can paint transparency "
        "back down as well as up, which is what recolouring flat pixel art "
        "wants. A soft nib still feathers either way."
    )


def _shading(state: Any, doc: Any) -> None:
    """The shading ink's direction, and which ramp it is walking.

    The ramp is not edited here and deliberately has no control of its own: it
    *is* the palette selection, which the Colour panel already shows in the one
    place where the colours are visible. What this says is what that selection
    currently amounts to, because "select some swatches" is the half of the tool
    a user has to be told about.
    """
    reason = inker_state.tool_reason("shade", doc)
    if reason:
        widgets.muted_wrapped(reason)
        return
    widgets.field_label("direction")
    for index, (value, label) in enumerate(SHADE_LABELS):
        if index:
            imgui.same_line()
        if controls.radio_button(f"{label}##shadedir{value}", int(state.shade_dir) == value):
            state.shade_dir = value
    widgets.help_marker(
        "Forward moves each pixel one swatch toward the end of the ramp, Back "
        "toward its start. Either way it stops at the end rather than wrapping "
        "round to the other one."
    )
    ramp = inker.shade_ramp(doc.palette, state.palette_slots)
    if len(state.palette_slots) < 2:
        widgets.muted(f"ramp: the whole palette ({len(ramp)})")
    else:
        widgets.muted(f"ramp: {len(ramp)} selected swatches")
    widgets.help_marker(
        "Select the swatches to shade along in the Colour panel -- Ctrl+click "
        "for a few, Shift+click for a run. They are walked in palette order, "
        "and a pixel painted in a colour that is not one of them is left alone. "
        "With fewer than two selected the whole palette is the ramp."
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
        if controls.small_button("Add stops##gradstops"):
            state.gradient_stops = [(0.0, tuple(state.fg)), (1.0, tuple(state.bg))]
        return

    remove = -1
    for index, (position, colour) in enumerate(list(state.gradient_stops)):
        imgui.push_id(f"gradstop{index}")
        imgui.set_next_item_width(sp(70))
        changed, value = controls.slider_float("##pos", float(position), 0.0, 1.0, "%.2f")
        if changed:
            state.gradient_stops[index] = (float(value), colour)
        imgui.same_line()
        edited, rgba = controls.color_edit4(
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
        if widgets.disabled_button(
            "x",
            len(state.gradient_stops) > 1,
            reason="A gradient needs at least two stops.",
            tooltip="Remove this stop.",
        ):
            remove = index
        imgui.pop_id()
    if remove >= 0:
        del state.gradient_stops[remove]
    if controls.small_button("Add##gradadd"):
        state.gradient_stops.append((0.5, tuple(state.fg)))
    imgui.same_line()
    if controls.small_button("Use fg / bg##gradreset"):
        state.gradient_stops = []


def _image_brush(ctx: Any, state: Any, tab: Any) -> None:
    """Capture a tip out of the drawing, and the variants of it.

    A section on the three tools that can stamp one rather than a tool of its
    own: an image brush replaces the *tip* of the tool in your hand, so
    everything that tool already does comes with it. The variants are buttons
    rather than a rotation field because they cycle -- four presses of Rotate is
    where you started -- and because the useful values are the four quarter
    turns and the two mirrors, which is six clicks and no numbers.
    """
    imgui.dummy((0, 6))
    # Folded by default: an image brush is a property of the brush rather than
    # a step in using one, and eight rows of tip variants sat between the nib
    # and the opacity for every session that never captured a tip.
    if not widgets.header(
        "Image brush", default_open=False, persist_key="inker/image-brush"
    ):
        return
    if widgets.disabled_button(
        "Capture from selection##inkstamp",
        tab.doc.mask is not None,
        reason="Select something first -- the selection is what becomes the tip.",
    ):
        inker_mode.capture_brush(ctx)
    widgets.help_marker(
        "Turns what you have selected into the brush tip -- the drawing itself "
        "becomes the dab. It reads the active layer and cuts nothing, and a "
        "feathered or lasso selection makes a feathered tip rather than its "
        "bounding box."
    )
    stamp = state.stamp
    if stamp is None:
        widgets.muted("Nothing captured yet.")
        return

    changed, value = controls.checkbox("Use image brush", state.use_stamp)
    if changed:
        state.use_stamp = value
    width, height = stamp.size
    widgets.muted(f"{width} x {height} px, turned {stamp.rotation}")
    if controls.small_button("Rotate##inkstamprot"):
        state.stamp = stamp.rotated()
    imgui.same_line()
    if controls.small_button("Flip H##inkstampfx"):
        state.stamp = stamp.flipped("x")
    imgui.same_line()
    if controls.small_button("Flip V##inkstampfy"):
        state.stamp = stamp.flipped("y")
    imgui.same_line()
    if controls.small_button("Forget##inkstampclear"):
        inker_mode.clear_brush(ctx)
        return
    state.stamp_align = widgets.labeled_combo(
        "Placing",
        state.stamp_align,
        list(STAMP_ALIGN_LABELS),
        help_text=(
            "Free puts the picture under the cursor, which is what a brush does. "
            "Aligned snaps every dab to a grid of the tip's own size anchored on "
            "the canvas, so neighbouring stamps line up into a pattern and going "
            "over the same square twice changes nothing. Either way a stroke never "
            "builds up on itself: dragging slowly over one spot leaves exactly what "
            "one dab leaves."
        ),
    )


def _presets(ctx: Any, state: Any) -> None:
    """Named bundles of the current tool's options.

    One tool's options and nothing else -- not the colours, the symmetry or the
    grid, which are properties of the canvas or of the sitting rather than of a
    tool. Clicking one selects its tool as well as its settings, because a
    preset called "inking pen" that arrived on the eraser would be half applied.
    """
    imgui.dummy((0, 6))
    if not widgets.header("Presets", default_open=False, persist_key="inker/presets"):
        return
    imgui.set_next_item_width(-sp(56))
    _changed, name = controls.input_text("##inkpresetname", state.preset_name)
    state.preset_name = name[: inker_state.MAX_PRESET_NAME]
    imgui.same_line()
    if widgets.disabled_button(
        "Save##inkpresetsave",
        bool(state.preset_name.strip()),
        reason="Give the preset a name first.",
    ):
        state.save_preset(state.preset_name)
        state.preset_name = ""
        inker_mode.persist(ctx)
    widgets.help_marker(
        "Saves the options of the tool in your hand under a name, and clicking "
        "the name later picks that tool back up with them. The captured image "
        "tip is not part of one -- it is pixels rather than a setting."
    )
    if not state.presets:
        widgets.muted("No presets yet.")
        return
    remove = ""
    for saved_name, saved in list(state.presets.items()):
        imgui.push_id(f"inkpreset{saved_name}")
        if controls.small_button("x"):
            remove = saved_name
        imgui.same_line()
        label = inker_state.tool_label(saved["tool"])
        if controls.selectable(f"{saved_name}  ({label})", False)[0]:
            state.apply_preset(saved_name)
        imgui.pop_id()
    if remove:
        state.delete_preset(remove)
        inker_mode.persist(ctx)


def _has_options(tool: str) -> bool:
    """Whether this tool has anything of its own to reset.

    The move and eyedropper tools have no options at all, and a Reset button
    that clears nothing is a control that says the panel is confused about
    which tool is selected.
    """
    return tool in PAINT_TOOLS or tool in SHAPE_TOOLS or tool in (
        "fill",
        "wand",
        "eyedropper",
        # Its three live in the canvas popup rather than in this panel, but
        # they are per-tool options like every other row of this table and a
        # user who has scrolled the size to 300 needs the same way back.
        "text",
    )


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
    widgets.section("Transform")
    if state.transforming:
        widgets.wrapped(theme.ACCENT, "Transforming - Enter applies, Esc cancels.")
        _transform_numbers(state, doc)
        return
    if controls.button("Free transform (Ctrl+T)", (-1, 0)):
        inker_mode.begin_transform(ctx)
    widgets.muted_wrapped("Rotates, scales and slants the selection, or the whole layer.")


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
    changed_x, x = controls.input_int("X##inkxfx", int(buf.offset[0]), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    changed_y, y = controls.input_int("Y##inkxfy", int(buf.offset[1]), 0)
    if changed_x or changed_y:
        # Through ``move_floating``'s delta rather than by writing ``offset``:
        # one owner for where a buffer sits, and it bumps ``rev`` for the pane.
        doc.move_floating(int(x) - buf.offset[0], int(y) - buf.offset[1])

    imgui.set_next_item_width(sp(70))
    changed_w, new_w = controls.input_int("W##inkxfw", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    changed_h, new_h = controls.input_int("H##inkxfh", int(height), 0)
    if changed_w or changed_h:
        fx = max(1, int(new_w)) / base_w if changed_w else buf.scale[0]
        fy = max(1, int(new_h)) / base_h if changed_h else buf.scale[1]
        if state.transform_link:
            fx = fy = fx if changed_w else fy
        doc.transform_floating(scale=(fx, fy), resample=state.resample)

    imgui.set_next_item_width(sp(150))
    changed, angle = controls.input_float("Angle##inkxfa", float(buf.angle), 0.0, 0.0, "%.2f")
    if changed:
        doc.transform_floating(angle=angle, resample=state.resample)

    limit = transform.SHEAR_MAX
    imgui.set_next_item_width(sp(150))
    changed, values = controls.input_float2(
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
    widgets.section("Selection")
    widgets.muted("Shift adds, Alt subtracts.")
    if controls.button("All"):
        doc.select_all()
    imgui.same_line()
    if widgets.disabled_button("None", doc.mask is not None, reason="Nothing is selected."):
        doc.deselect()
    imgui.same_line()
    if controls.button("Invert"):
        doc.invert_selection()
    imgui.same_line()
    # Enabled off the *memory* rather than off "there is no selection": the
    # useful case is exactly re-selecting after something else was selected,
    # and a mask the canvas has outgrown is refused by the engine.
    if widgets.disabled_button(
        "Reselect",
        doc._last_mask is not None,
        reason="Nothing has been deselected yet.",
    ):
        doc.reselect()
    widgets.help_marker(
        "Brings back the selection you last dismissed (Ctrl+Shift+D). A "
        "selection from before a resize or a crop cannot come back -- it "
        "describes a canvas that no longer exists."
    )
    if widgets.disabled_button(
        "Copy to layer", doc.mask is not None, reason="Nothing is selected."
    ):
        doc.layer_from_selection(cut=False)
    imgui.same_line()
    if widgets.disabled_button(
        "Move to layer", doc.mask is not None, reason="Nothing is selected."
    ):
        doc.layer_from_selection(cut=True)
    widgets.help_marker(
        "Ctrl+J copies the selection onto a layer of its own and leaves the "
        "original where it was; Ctrl+Shift+J moves it, cutting it out of the "
        "layer it came from. Either way it is one undo step, and the new layer "
        "lines up with what it came from."
    )
    if controls.button("This layer"):
        doc.select_layer_alpha()
    widgets.help_marker(
        "Selects what is painted on the active layer, at the coverage it is "
        "painted at -- a soft edge becomes a soft selection."
    )

    # Colour-first where the wand is seed-first, and the *wand's* tolerance
    # rather than ``state.wand_tolerance``: that property follows the tool in
    # hand, and this button is in a section that is drawn whatever the tool is,
    # so it would otherwise read the pencil's copy. Reading the wand's is the
    # one-predicate rule kept honest -- there is one tolerance the user set and
    # one meaning of "similar" (see ``selection.colour_distance``).
    wand_options = state.options_for("wand")
    imgui.set_next_item_width(-sp(110))
    changed, value = controls.slider_int(
        "##range_tolerance", wand_options["wand_tolerance"], 0, 255, "tolerance %d"
    )
    if changed:
        wand_options["wand_tolerance"] = value
    imgui.same_line()
    if controls.button("Colour range"):
        doc.select_colour_range(state.fg, tolerance=wand_options["wand_tolerance"])
    widgets.help_marker(
        "Selects every pixel close to the *foreground* colour, anywhere on the "
        "canvas -- it is not contiguous, so one press selects a palette entry "
        "wherever it was used. The tolerance beside it is the magic wand's."
    )

    imgui.set_next_item_width(-sp(80))
    changed, value = controls.slider_float("##feather", state.feather_radius, 0.0, 32.0, "%.1f px")
    if changed:
        state.feather_radius = value
    imgui.same_line()
    if widgets.disabled_button("Feather", doc.mask is not None, reason="Nothing is selected."):
        doc.feather_selection(state.feather_radius)

    # Whole pixels, and its own control: feather softens an edge where these
    # *move* it, and one slider serving both would have to pick a unit that is
    # wrong for one of them.
    imgui.set_next_item_width(-sp(80))
    changed, steps = controls.slider_int("##selgrow", int(state.select_steps), 1, 32, "%d px")
    if changed:
        state.select_steps = int(steps)
    imgui.same_line()
    widgets.muted("by")
    has = doc.mask is not None
    if widgets.disabled_button("Grow", has, reason="Nothing is selected."):
        doc.grow_selection(state.select_steps)
    imgui.same_line()
    if widgets.disabled_button("Shrink", has, reason="Nothing is selected."):
        doc.shrink_selection(state.select_steps)
    imgui.same_line()
    if widgets.disabled_button("Border", has, reason="Nothing is selected."):
        doc.border_selection(state.select_steps)
    widgets.help_marker(
        "Border replaces the selection with the band that many pixels either "
        "side of its edge -- fill it and you have stroked the outline."
    )

    if widgets.disabled_button(
        "Crop to selection", doc.mask is not None, reason="Nothing is selected."
    ):
        doc.crop_to_selection()


def _canvas_options(ctx: Any, state: Any) -> None:
    imgui.dummy((0, 6))
    # Symmetry, the grid, snapping and the rulers: settings of the *sitting*
    # rather than of the gesture, reached once and then left for an hour.
    if not widgets.header("Canvas", default_open=False, persist_key="inker/canvas"):
        return
    state.symmetry = widgets.labeled_combo("Symmetry", state.symmetry, list(SYMMETRY_LABELS))
    if state.symmetry == "radial":
        imgui.set_next_item_width(sp(90))
        changed, count = controls.slider_int(
            "Ways", int(state.radial_count), brush.MIN_RADIAL, brush.MAX_RADIAL
        )
        if changed:
            state.radial_count = int(count)
    if state.symmetry != "none":
        _symmetry_axis(state)
    # Every write below goes back through ``persist``: the grid and the rulers
    # are how the user likes to see, and a preference that resets on the next
    # launch is a control they have to rediscover. Persisting on the change
    # rather than only at quit is what survives a crash.
    changed, value = controls.checkbox("Grid", state.grid)
    if changed:
        state.grid = value
        inker_mode.persist(ctx)
    if state.grid:
        imgui.same_line()
        imgui.set_next_item_width(sp(80))
        changed, size = controls.input_int("##gridsize", state.grid_size, 0)
        if changed:
            state.grid_size = max(2, min(512, size))
        if imgui.is_item_deactivated_after_edit():
            inker_mode.persist(ctx)
        changed, value = controls.checkbox("Snap to grid", state.grid_snap)
        if changed:
            state.grid_snap = value
            inker_mode.persist(ctx)
        widgets.help_marker(
            "Shapes, lines and the marquee land on grid intersections. "
            "Freehand strokes never snap -- quantising a brush to a lattice is "
            "a different tool, not a drawing aid."
        )
    changed, value = controls.checkbox("Rulers", state.rulers)
    if changed:
        state.rulers = value
        inker_mode.persist(ctx)
    widgets.help_marker(
        "Pixel rulers along the canvas's top and left edges, for a sense of "
        "size. Tick steps follow the decimal 1/2/5 ladder."
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
    changed, values = controls.input_float2(
        "Axis##symaxis", list(axis or (0.0, 0.0)), "%.0f"
    )
    if changed:
        state.symmetry_axis = (float(values[0]), float(values[1]))
    imgui.same_line()
    if widgets.disabled_button(
        "Centre##symcentre",
        axis is not None,
        reason="The axis is already centred.",
        tooltip="Put the symmetry axis back in the middle of the canvas.",
    ):
        # Back to None rather than to the middle of the current document: None
        # *is* the centre, and stays the centre across a resize.
        state.symmetry_axis = None
    if axis is None:
        widgets.muted("centred")
