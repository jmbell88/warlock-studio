"""The tool grid and its per-tool options.

Options are shown for the tool that is selected and hidden otherwise, rather
than laid out as one long form. A brush's hardness means nothing while the wand
is active, and a panel that shows every control at once is how a paint program
becomes unreadable.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, inker, inker_mode, inker_state, theme, widgets
from ..inker import brush, transform
from ..inker_state import (
    PAINT_TOOLS,
    SHAPE_TOOLS,
    STAMP_TOOLS,
)
from ..tokens import sp

# Icon-only, five across: what every paint program's toolbox looks like. The
# name and shortcut live in the tooltip -- three columns of bare labels
# truncated ("Ellipse select" in a 104px button) and anything wider is taller,
# which the canvas pays for.
COLUMNS = 2

#: One toolbox button, in design px. The same number ``_grid`` draws with, said
#: once so :func:`grid_height` cannot drift from it.
BUTTON_H = 30.0

#: One toolbox button's *width*, in design px, and the reason it is a constant
#: rather than ``grid_width(COLUMNS)``: the toolbox is a rail, not a column. A
#: button that took half of whatever pane it is in would be 150 px wide in a
#: 300 px sidebar and a different shape again at every drag of the handle, and
#: a square of icons is the picture a user recognises a toolbox by.
BUTTON_W = 34.0

#: The gap ``draw`` leaves under the grid, in design px. Named so the dummy
#: that draws it and the reservation that counts it are one number: the dummy
#: used to emit 6 *physical* px while :func:`grid_height` counted 6 design px,
#: which is a drift of ``6 * (SCALE - 1)`` at every UI scale but 1.0.
GRID_GAP = 6.0


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
    # The two 45-degree mirrors (6.8). Named for the line they reflect about
    # rather than "diagonal", which does not say *which* diagonal.
    ("diag", "diagonal, top-left to bottom-right"),
    ("anti", "diagonal, bottom-left to top-right"),
)


#: The rail's width in design px: two 34 px buttons, the gap between them and
#: the pane's own padding. Fixed, and the whole point of the wave: the column
#: it replaces was 300 px carrying the same twelve decisions plus fourteen
#: controls that are now a row above the canvas.
RAIL_W = 90.0

#: The colour chips' flags: no inputs, no label, alpha as a split square. The
#: full editor is the colour panel's; these two are a *picture* of what the
#: tool writes with, and a click opens the picker anyway. Zero would draw the
#: picker's own R/G/B strips beside a 34 px swatch, which is what the rail's
#: first screenshot showed.
_CHIP_FLAGS = (
    imgui.ColorEditFlags_.no_inputs.value
    | imgui.ColorEditFlags_.no_label.value
    | imgui.ColorEditFlags_.alpha_preview_half.value
)


def draw(ctx: Any) -> None:
    """The toolbox rail: twelve groups, three view toggles, the two colours.

    No heading and no help button -- at 90 px a section title does not fit and
    would push the first row of tools off the top. The pane is exempted in
    ``tests/manual/test_coverage.py`` for that reason, and the tools are
    documented in the Inker chapter as they always were.
    """
    state = inker_mode.ensure(ctx)
    tab = state.active
    _grid(state, None if tab is None else tab.doc)
    imgui.dummy((0, sp(GRID_GAP)))
    _view_toggles(ctx, state)
    imgui.dummy((0, sp(GRID_GAP)))
    _colour_chips(ctx, state)


def _view_toggles(ctx: Any, state: Any) -> None:
    """Grid, snap and rulers, as three icon toggles rather than three rows.

    ``clay_tools``' idiom. Each writes through ``inker_mode.persist``: these
    are how the user likes to *see*, and a preference that resets on the next
    launch is a control they have to rediscover.
    """
    for icon, attr, tip in (
        (icons.GRID, "grid", "Show the pixel grid"),
        (icons.MAGNET, "grid_snap", "Snap shapes and the marquee to the grid"),
        (icons.RULER, "rulers", "Rulers along the top and left edges"),
    ):
        engaged = bool(getattr(state, attr))
        if engaged:
            imgui.push_style_color(
                imgui.Col_.button.value,
                imgui.get_style().color_(imgui.Col_.button_active.value),
            )
        if widgets.icon_button(f"{icon}##inkview{attr}", tip):
            setattr(state, attr, not engaged)
            inker_mode.persist(ctx)
        if engaged:
            imgui.pop_style_color()
        if attr != "rulers":
            imgui.same_line()
    imgui.new_line()


def _colour_chips(ctx: Any, state: Any) -> None:
    """The foreground and background, at the foot of the rail.

    Aseprite's "Mirrored Default" shape: the two colours belong with the tools
    because they are what the tool in your hand writes with, and the *palette*
    -- swatches, ramps, the indexed table -- is a panel, which is why it sits
    in the right column instead (``inker_colors``).
    """
    # **Sized, and drawn as swatches rather than editors.** ``color_edit4``
    # defaults to a full-width item and to showing its inputs, so in a 90 px
    # rail the two chips drew as coloured slivers with the picker's own strips
    # either side -- which is what the screenshot pass at 1.5 caught. A chip is
    # a picture of the colour; the editor is the click.
    chip = sp(BUTTON_W)
    imgui.set_next_item_width(chip)
    changed, value = controls.color_edit4("##inkfg", _vec(state.fg), _CHIP_FLAGS)
    if changed:
        state.set_fg(_rgba(value))
    if imgui.is_item_hovered():
        imgui.set_tooltip("Foreground -- what the tool writes with")
    imgui.same_line()
    imgui.set_next_item_width(chip)
    changed, value = controls.color_edit4("##inkbg", _vec(state.bg), _CHIP_FLAGS)
    if changed:
        state.bg = _rgba(value)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Background -- what a right-drag writes with")
    if widgets.icon_button(f"{icons.SHUFFLE}##inkswap", "Swap the two colours (X)"):
        state.swap_colours()


def _vec(colour: tuple[int, int, int, int]) -> Any:
    return imgui.ImVec4(*(channel / 255.0 for channel in colour))


def _rgba(value: Any) -> tuple[int, int, int, int]:
    return tuple(max(0, min(255, round(channel * 255))) for channel in value)


def panels(ctx: Any, state: Any, tab: Any) -> bool:
    """The tool's remaining *panels*, drawn wherever the caller wants them.

    What is left once the fourteen simple options became a context bar: a list
    (the gradient's stops, the document's slices), a block of verbs (the image
    brush), a shading ramp, the named presets. Each is a panel rather than a
    control, so each is drawn in a popover off the context bar rather than in a
    column that would have to exist all session for the tools that have one.

    -> whether anything was drawn, so the bar can leave the button out.
    """
    if tab is None:
        return False
    before = _has_panels(state, tab)
    if before:
        _options(ctx, state, tab)
    return before


def _has_panels(state: Any, tab: Any) -> bool:
    tool = state.tool
    if tool in ("gradient", "shade", "slice") or tool in STAMP_TOOLS:
        return True
    if tab.doc.active_tilemap_uid() is not None:
        return True
    return _has_options(tool)


def _grid(state, doc=None) -> None:
    """Twelve groups, two across, and the flyout that picks inside one.

    Aseprite's toolbox, and its gesture exactly: **mouse-down selects, opens
    the strip and captures, all on one event**, so pressing a group and letting
    go on the spot is simply "pick this group's tool", and pressing and sliding
    is "pick that one". There is no click-then-click and no menu to dismiss.

    The strip draws into the *foreground* draw list rather than into an imgui
    popup, and that is the load-bearing choice: a popup is a window, and a
    window opening under the cursor steals the capture from the button being
    held -- which ends the slide the moment it crosses the strip's border. With
    the button still the active item, ``is_item_active()`` **is**
    ``captureMouse()``.

    Greyed rather than hidden, and the tooltip carries the reason: a toolbox
    that quietly loses a button when a document is not indexed is one where the
    feature looks like it was imagined. ``allow_when_disabled`` is what makes
    the tooltip reachable at all -- imgui swallows hover on a disabled item,
    which is exactly the state whose explanation matters
    (``widgets.disabled_button`` has the long version of this note).
    """
    width, height = sp(BUTTON_W), sp(BUTTON_H)
    rects: dict[str, tuple[float, float, float, float]] = {}
    for index, (group, label, members) in enumerate(inker_state.TOOL_GROUPS):
        shown = _group_tool(state, group, members)
        reason = inker_state.tool_reason(shown, doc)
        selected = state.tool in members
        origin = imgui.get_cursor_screen_pos()
        rects[group] = (origin.x, origin.y, width, height)
        if selected:
            imgui.push_style_color(
                imgui.Col_.button.value,
                imgui.get_style().color_(imgui.Col_.button_active.value),
            )
        icon = TOOL_ICONS.get(shown) or inker_state.tool_label(shown)[:1]
        if reason:
            imgui.begin_disabled()
        controls.button(f"{icon}##toolgroup{group}", (width, height))
        if reason:
            imgui.end_disabled()
        if not reason and imgui.is_item_activated():
            # The mouse-down frame, once. Selecting *here* rather than on
            # release is what makes a press and release in place mean "this
            # group", with no second decision to make.
            state.set_tool(shown)
            state.flyout = group if len(members) > 1 else ""
        if selected:
            imgui.pop_style_color()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(_group_tooltip(label, members, shown, reason))
        if index % COLUMNS != COLUMNS - 1:
            imgui.same_line()
    imgui.new_line()
    _flyout(state, doc, rects)


def _group_tool(state, group: str, members) -> str:
    """Which member of a group the rail shows.

    The tool in hand if it is in this group -- so the rail is a picture of what
    you are holding -- and otherwise the one last slid onto, which is the
    setting the flyout writes.
    """
    if state.tool in members:
        return state.tool
    remembered = state.group_tool.get(group)
    return remembered if remembered in members else members[0]


def _group_tooltip(label: str, members, shown: str, reason: str) -> str:
    """A group button's accessible name: the tool on it, and its chords.

    Both bindings, because three tools have two, and the other members' names
    when there are any -- the tooltip is the only place a glyph button says
    what it does, so it is also the only place "hold to reach the others" can
    be read.
    """
    chords = next((key for tool, _l, key in inker_state.TOOLS if tool == shown), "")
    alt = inker_mode.ALT_TOOL_CHORDS.get(shown)
    if alt:
        chords = f"{chords} or {alt}"
    note = f"{inker_state.tool_label(shown)}  ({chords})"
    if len(members) > 1:
        others = ", ".join(
            inker_state.tool_label(tool) for tool in members if tool != shown
        )
        note = note + "\n" + f"{label}: hold to reach {others}"
    return note + "\n" + reason if reason else note


def _flyout(state, doc, rects) -> None:
    """The open group's strip, drawn over everything and hit-tested by hand.

    Hit-testing is ``inker_state.flyout_hit``, which is pure, unit-tested with
    no imgui, and **the same function that places the cells** -- so the picture
    and the hit box cannot disagree.

    Sliding across to another *group* reassigns the strip, which is Aseprite's
    behaviour and falls out for free here: the button is still the active item,
    so every group's rect is known and the mouse simply picks one.
    """
    if not state.flyout:
        return
    if not imgui.is_mouse_down(0):
        state.flyout = ""
        return
    mouse = imgui.get_mouse_pos()
    point = (mouse.x, mouse.y)
    for group, (x, y, w, h) in rects.items():
        inside = x <= point[0] < x + w and y <= point[1] < y + h
        if inside and len(inker_state.group_members(group)) > 1:
            state.flyout = group
    rect = rects.get(state.flyout)
    members = inker_state.group_members(state.flyout)
    if rect is None or len(members) < 2:
        return
    cell = (sp(BUTTON_W), sp(BUTTON_H))
    hit = inker_state.flyout_hit(rect, cell, len(members), point)
    draw = imgui.get_foreground_draw_list()
    for index, (x, y, w, h) in enumerate(
        inker_state.flyout_cells(rect, cell, len(members))
    ):
        tool = members[index]
        blocked = bool(inker_state.tool_reason(tool, doc))
        colour = theme.PANEL if index != hit else theme.ACCENT
        draw.add_rect_filled(
            (x, y), (x + w, y + h), imgui.get_color_u32(theme.rgba(colour))
        )
        draw.add_rect(
            (x, y), (x + w, y + h), imgui.get_color_u32(theme.rgba(theme.DIVIDER))
        )
        glyph = TOOL_ICONS.get(tool) or inker_state.tool_label(tool)[:1]
        size = imgui.calc_text_size(glyph)
        draw.add_text(
            (x + (w - size.x) * 0.5, y + (h - size.y) * 0.5),
            imgui.get_color_u32(theme.rgba(theme.MUTED if blocked else theme.TEXT)),
            glyph,
        )
        if index == hit and not blocked:
            # Live while the mouse is held, so the canvas under the strip shows
            # the tool that would be picked. Through ``set_tool``, like every
            # other way of picking one.
            state.set_tool(tool)
            state.group_tool[state.flyout] = tool


def _options(ctx: Any, state: Any, tab: Any) -> None:
    """What is left in the sidebar once the context bar has the tool options.

    The fourteen controls that were here -- size, nib, hardness, opacity, ink,
    rate, spacing, strength, smoothing, taper, filled, tolerance, contiguous,
    sample, dither -- are one row above the canvas now (``inker_context``), at
    two rows each and ~812 px of column saved. Every one of them is still
    reachable, and ``tests/inker/test_context_bar.py`` asserts that against
    ``TOOL_OPTION_DEFAULTS`` rather than leaving it to reading.

    What could not go on a row stayed: a *list* (the gradient's stops, the
    document's slices), a **block of verbs** (the image brush's capture and its
    six variants), a radio row about the document rather than the tool (tile
    behaviour), and the named presets. Those are panels, and a 38 px row is not
    where a panel goes.
    """
    tool = state.tool
    doc = tab.doc
    # The whole block, not the two panels that happen to write: the slice list
    # renames and deletes, the image brush reads pixels off the stack a save is
    # walking, and the gate the canvas, the timeline and the keyboard already
    # share is one question with two answers behind it (``busy`` is ``saving or
    # playing``). Asking it once here is what stops a third reason being added
    # in one place and forgotten in nine.
    imgui.begin_disabled(tab.busy)

    if tool in STAMP_TOOLS:
        _image_brush(ctx, state, tab)
    if tool == "shade":
        _shading(state, doc)

    if tool == "gradient":
        widgets.section("Gradient")
        # The *shape* and the stops, not the dither -- that one is a per-tool
        # option and rides the context bar with the rest of them. These two are
        # app-level and one of them is a list, which is why they stayed.
        state.gradient_kind = widgets.labeled_combo(
            "Shape",
            state.gradient_kind,
            [(k, k) for k in inker.GRADIENT_KINDS],
        )
        changed, value = controls.checkbox("To transparent", state.gradient_to_transparent)
        if changed:
            state.gradient_to_transparent = value
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
    imgui.end_disabled()

    # Free transform is the Edit menu's row and Ctrl+T, and its live handles
    # are the context bar's Transformation state. The button that used to be
    # here was the only way *in* when the panel was the only surface; the
    # numbers it opened are on the bar beside the canvas they turn.
    #
    # The selection verbs that were here are the **Select menu** now, with
    # their amounts as parameter dialogs: eleven buttons and three sliders in a
    # section that drew whenever a mask existed, for operations reached once or
    # twice in a session.


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
    # ``segmented_choice``, not three hand-rolled buttons with a pushed style
    # colour: this is one mutually-exclusive choice and the shared control
    # already draws that -- including the selected pill, the compact height and
    # the per-option tooltips, which the hand-rolled version had to repeat.
    changed, picked = controls.segmented_choice(
        "inker-tilebehav",
        [(key, label) for key, label, _why in inker_state.TILE_BEHAVIORS],
        current,
        tooltips={key: why for key, _label, why in inker_state.TILE_BEHAVIORS},
        compact=True,
    )
    if changed:
        doc.tile_behavior = picked
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


def canvas_options(ctx: Any, state: Any) -> None:
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
