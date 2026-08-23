"""The layout rules the Inker UX pass fixed, pinned so they cannot come back.

Every test here answers a defect a *screenshot* found rather than a test: the
smoke suite asserts that a frame containing every panel can be built, and a
panel whose help marker has slid onto the wrong line builds perfectly. Three of
the four are AST scans for that reason -- what went wrong is a property of the
call site, and the call site is the only thing a headless test can read.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from warlock.studio import inker_mode, inker_state, layout, theme, tokens
from warlock.studio.panes import (
    inker_canvas,
    inker_colors,
    inker_textures,
    inker_tools,
)

STUDIO = Path(inker_tools.__file__).resolve().parent.parent
PANES = sorted((STUDIO / "panes").glob("*.py"))

#: The controls that draw at the full width of the content region. imgui puts a
#: slider's or a combo's own label *outside* the widget, to its right, so at
#: ``-1`` there is nothing left of the line for anything to follow on.
FULL_WIDTH = {"labeled_combo", "labeled_slider_int", "labeled_slider_float", "labeled_drag_int"}

#: Modules whose calls *draw*. A statement calling one of these ends the run
#: begun by a full-width control; a statement that only writes a value back
#: (``if changed: state.x = value``) does not, because it puts nothing on
#: screen and the marker still lands on the line below the control.
DRAW_MODULES = {"widgets", "controls", "imgui", "manual_render", "inker_bridge"}


def _name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", "")


def _draws(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in DRAW_MODULES
    return _name(call) in FULL_WIDTH


def _is_bare_marker(stmt: ast.stmt) -> bool:
    """A statement that is *only* a ``help_marker`` call, at this block level."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _name(stmt.value) == "help_marker"
    )


def _orphaned_markers(path: Path) -> list[int]:
    """Lines where a marker would be drawn on a line of its own.

    ``help_marker`` ends in ``same_line_or_wrap``, which refuses to draw past
    the content region and starts a new line instead. That is right -- the
    alternative is clipping it away -- but after a full-width control there is
    *never* room, so the glyph lands between the control it belongs to and the
    next field's name, where it reads as that one's.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[int] = []
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        pending: str | None = None
        for stmt in body:
            if _is_bare_marker(stmt):
                if pending is not None:
                    found.append(stmt.lineno)
                continue
            calls = [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]
            wide = [_name(c) for c in calls if _name(c) in FULL_WIDTH]
            if wide:
                pending = wide[-1]
            elif any(_draws(c) for c in calls):
                pending = None
    return found


@pytest.mark.parametrize("path", PANES, ids=lambda p: p.name)
def test_no_help_marker_is_left_on_a_line_of_its_own(path: Path) -> None:
    """A marker after a full-width control goes *on the label*, via ``help_text``.

    Found in a screenshot: the Layers panel drew BLEND, its combo, a bare
    circled-i, then OPACITY -- and the mark that explained the blend mode was
    sitting directly above the word "opacity", which is where a reader takes it
    to belong. Brush had the same thing between NIB and HARDNESS.
    """
    offenders = [f"{path.name}:{line}" for line in _orphaned_markers(path)]
    assert offenders == []


#: The button wrappers whose ``(-1, 0)`` size draws at full width -- the same
#: zero-room line as FULL_WIDTH, missed by the scan above because a button is
#: not a labelled control. A button has no label row to carry the mark, so the
#: help rides the button itself as ``tooltip=``.
BUTTONS = {"button", "disabled_button", "primary_button", "ghost_button"}

#: The panes the full-width-button sweep covers -- all of them, since the last
#: named site (packwright_sources' tile-set button) moved its help onto the
#: button's ``tooltip=``.
SWEPT = sorted(PANES)


def _full_width_button(call: ast.Call) -> bool:
    """A button call drawn at ``(-1, 0)`` -- args or ``size=``."""
    if _name(call) not in BUTTONS:
        return False
    sized = list(call.args) + [kw.value for kw in call.keywords if kw.arg == "size"]
    for node in sized:
        if not (isinstance(node, ast.Tuple) and node.elts):
            continue
        first = node.elts[0]
        if (
            isinstance(first, ast.UnaryOp)
            and isinstance(first.op, ast.USub)
            and isinstance(first.operand, ast.Constant)
            and first.operand.value == 1
        ):
            return True
    return False


def _markers_after_full_width_buttons(path: Path) -> list[int]:
    """`_orphaned_markers`, for the runs a ``(-1, 0)`` button begins."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[int] = []
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        pending = False
        for stmt in body:
            if _is_bare_marker(stmt):
                if pending:
                    found.append(stmt.lineno)
                continue
            calls = [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]
            if any(_full_width_button(c) for c in calls):
                pending = True
            elif any(_draws(c) for c in calls):
                pending = False
    return found


@pytest.mark.parametrize("path", SWEPT, ids=lambda p: p.name)
def test_no_marker_orphans_after_a_full_width_button(path: Path) -> None:
    """The bridge pane's Animate/Save/Make 3D/import rows all had one: the
    button took the whole line, ``same_line_or_wrap`` -- correctly -- refused
    to draw past it, and the glyph landed on the next control's row where it
    read as that one's. The help is the button's own ``tooltip=`` now."""
    offenders = [
        f"{path.name}:{line}" for line in _markers_after_full_width_buttons(path)
    ]
    assert offenders == []


def _named_combos(path: Path) -> list[int]:
    """``widgets.combo`` calls whose label would be drawn and then clipped.

    **Only the ones left at the default width.** imgui draws a combo's name to
    the right of the widget, so a *narrowed* combo has room for one on its own
    line and several form panes use exactly that -- ``combo("theme", ...,
    sp(160))`` reads fine. It is ``-1`` that leaves the cursor on the content
    region's right edge with the name still to come.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _name(node) != "combo":
            continue
        if not node.args:
            continue
        label = node.args[0]
        if not (isinstance(label, ast.Constant) and isinstance(label.value, str)):
            continue
        if label.value.startswith("##"):
            continue
        sized = len(node.args) > 3 or any(kw.arg == "width" for kw in node.keywords)
        if not sized:
            found.append(node.lineno)
    return found


@pytest.mark.parametrize("path", PANES, ids=lambda p: p.name)
def test_a_visible_combo_name_goes_through_labeled_combo(path: Path) -> None:
    """``widgets.combo``'s own docstring states this rule; five sites broke it.

    imgui draws a combo's name to its *right*, and the default width is ``-1``,
    so a named combo in a sidebar puts its name past the content region -- where
    ``same_line`` clips rather than wrapping, i.e. the name is simply not drawn.
    The gradient tool shipped two nameless dropdowns stacked on each other
    reading "linear" and "none". The rule was written down and not enforced,
    which is why it drifted back; this is the enforcement.
    """
    offenders = [f"{path.name}:{line}" for line in _named_combos(path)]
    assert offenders == []


# --- the transparency checker ------------------------------------------------


@pytest.mark.parametrize("palette", sorted(tokens.PALETTES))
def test_the_checker_is_a_palette_entry_in_both_themes(palette: str) -> None:
    """It used to be two module constants, and therefore dark in both themes.

    A near-black checkerboard under a white window is merely ugly; dark artwork
    drawn *onto* it cannot be seen at all, which is the half that made this a
    defect rather than a preference.
    """
    colours = tokens.PALETTES[palette]
    assert "CHECKER_A" in colours
    assert "CHECKER_B" in colours


@pytest.mark.parametrize("palette", sorted(tokens.PALETTES))
def test_the_two_checker_squares_are_far_enough_apart_to_read(palette: str) -> None:
    """The pair answers to each other rather than to a contrast bar.

    No WCAG rule covers it -- it carries neither copy nor a control boundary --
    but a checker whose squares are 14 levels apart, which is what shipped, is a
    flat field with a rumour of a pattern in it. Sixteen is about where the step
    survives a dim panel.
    """
    colours = tokens.PALETTES[palette]
    a, b = colours["CHECKER_A"], colours["CHECKER_B"]
    steps = [
        abs(((a >> shift) & 0xFF) - ((b >> shift) & 0xFF)) for shift in (16, 8, 0)
    ]
    assert max(steps) >= 16, f"{palette}: {max(steps)} levels between the squares"


def test_the_checker_squares_follow_the_theme_in_force() -> None:
    """Read through ``theme``, so the switch reaches them like everything else."""
    before = tokens.THEME
    try:
        tokens.set_theme("dark")
        dark = inker_textures.checker_squares()
        tokens.set_theme("light")
        light = inker_textures.checker_squares()
    finally:
        tokens.set_theme(before)
    assert dark != light
    # The light theme's checker is the pale one, which is the whole point of
    # having two: it is the ground the artwork is judged against.
    assert sum(light[0][:3]) > sum(dark[0][:3])


def test_the_checker_texture_is_keyed_on_the_theme() -> None:
    """A tile built under one palette must not outlive the switch away from it.

    ``theme.__getattr__`` resolves live precisely so a colour cannot be copied
    into a constant and go stale; a texture *is* a copy, so the cache carries
    the palette it was baked under.
    """
    source = Path(inker_textures.__file__).read_text(encoding="utf-8")
    assert "_CHECKER_KEY}:theme" in source


# --- one name for a tool -----------------------------------------------------


def test_tool_label_answers_for_every_tool_in_the_box() -> None:
    for key, label, _shortcut in inker_state.TOOLS:
        assert inker_state.tool_label(key) == label


def test_tool_label_falls_back_to_the_key_it_was_given() -> None:
    """A preset saved under a tool that has since been renamed still lists."""
    assert inker_state.tool_label("no-such-tool") == "no-such-tool"


#: The panes that draw Inker tool names. Scoped rather than repo-wide, and the
#: difference is the reason: Clay and the 2D settings turn *generator* and
#: *field* keys into labels the same way, and they are right to -- there is no
#: table of display names for those. Inker has one, in ``inker_state.TOOLS``,
#: which is exactly why spelling the key out here was a bug.
TOOL_NAMING_PANES = ("inker_tools.py", "inker_canvas.py", "inker_timeline.py")


@pytest.mark.parametrize("name", TOOL_NAMING_PANES)
def test_no_inker_pane_spells_a_tool_name_out_of_its_key(name: str) -> None:
    """``select_ellipse`` is "Ellipse select", not "select ellipse".

    Two call sites built the name by swapping underscores for spaces, so the
    Reset button under a tool the box called "Ellipse select" read "Reset select
    ellipse", and the preset list named tools nothing else in the app did.
    """
    source = (STUDIO / "panes" / name).read_text(encoding="utf-8")
    assert 'replace("_", " ")' not in source, name
    assert "replace('_', ' ')" not in source, name


# --- a section names what follows it ----------------------------------------


def test_the_brush_controls_left_the_panel_for_the_context_bar() -> None:
    """The ordering defect this replaces cannot recur, because the controls it
    was about are not in this file any more.

    "Image brush" opened in the middle of the brush options used to run its
    tint over the ink radio, Opacity, Spacing, Smoothing and Taper -- controls
    it does not configure, under a heading that says it does. Those are the
    context bar's now (W2.4), so what is left in ``_options`` is the blocks a
    38 px row cannot hold, and the image brush is the first of them.
    """
    source = Path(inker_tools.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    options = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_options"
    )
    calls = [n for n in ast.walk(options) if isinstance(n, ast.Call)]
    assert [n.lineno for n in calls if _name(n) == "_image_brush"], (
        "the image brush block left _options entirely"
    )
    labels = {
        n.args[0].value
        for n in calls
        if _name(n) in FULL_WIDTH and n.args and isinstance(n.args[0], ast.Constant)
    }
    assert not labels & {
        "Size", "Nib", "Opacity", "Spacing", "Rate", "Strength", "Smoothing", "Taper"
    }, "a per-tool brush control is drawn in both the sidebar and the context bar"


def test_a_live_transform_says_so_in_the_accent_colour() -> None:
    """The sidebar's wrapped accent line went with ``_transform_entry``, which
    had no caller left: the row is the canvas's own now
    (``inker_context._transform_bar`` is deliberately empty and says why),
    because the numeric handles read and write the floating buffer the canvas
    is already holding. What has to stay true is that a transform in flight is
    *announced*, and in the accent colour."""
    from warlock.studio.panes import inker_canvas

    source = Path(inker_canvas.__file__).read_text(encoding="utf-8")
    assert 'widgets.text_colored(theme.ACCENT, "Transform")' in source
    assert hasattr(theme, "ACCENT")


# --- the Aseprite letters ----------------------------------------------------


def test_the_letter_table_is_derived_from_the_toolbox() -> None:
    """Two hand-kept tables of twenty-three entries were one table too many.

    ``plotter_state`` had already replaced the arrangement with a derivation;
    this is the same one line, so a letter cannot be changed in the toolbox and
    left behind in the dispatcher.
    """
    derived = {letter.lower(): key for key, _label, letter in inker_state.TOOLS}
    assert derived == inker_mode.TOOL_KEYS


def test_every_tool_has_a_letter_and_no_two_share_one() -> None:
    letters = [letter.lower() for _key, _label, letter in inker_state.TOOLS]
    assert len(letters) == len(set(letters))
    assert len(inker_mode.TOOL_KEYS) == len(inker_state.TOOLS)


def test_aseprites_line_and_rectangle_letters_reach_the_line_and_rectangle() -> None:
    """The two that were being squatted.

    Letters were handed out in toolbox order, so ``L`` had gone to the polyline
    and ``U`` to the gradient before the line and the rectangle were reached. A
    hand trained on Aseprite pressing ``L`` for a line got a polyline -- not a
    near miss but a different *gesture*, click-click-Enter instead of a drag.
    """
    assert inker_mode.TOOL_KEYS["l"] == "line"
    assert inker_mode.TOOL_KEYS["u"] == "rect"


def test_the_displaced_tools_took_the_letters_that_were_vacated() -> None:
    assert inker_mode.TOOL_KEYS["p"] == "polyline"
    assert inker_mode.TOOL_KEYS["k"] == "gradient"


def test_every_shifted_letter_names_a_real_tool() -> None:
    """Aseprite's slot-mates, added beside the plain letters rather than
    instead of them: a hand trained there finds what it reaches for and a hand
    trained here keeps what it had."""
    tools = {key for key, _label, _letter in inker_state.TOOLS}
    for letter, tool in inker_mode.SHIFT_TOOL_KEYS.items():
        assert tool in tools, letter
        assert len(letter) == 1 and letter.islower()
    assert set(inker_mode.ALT_TOOL_CHORDS) == set(
        inker_mode.SHIFT_TOOL_KEYS.values()
    )


# --- the zoom ladder ---------------------------------------------------------


def test_every_rung_above_one_to_one_is_a_whole_scale() -> None:
    """The rule the keyboard zoom exists for.

    At 135% a source pixel is 1.35 screen pixels, so the renderer draws some of
    them one pixel wide and some two and a checkerboard dither comes out as
    bands. Every rung at or above 1:1 maps one source pixel onto a whole number
    of screen pixels.
    """
    for rung in inker_state.ZOOM_LADDER:
        if rung >= 1.0:
            assert rung == int(rung), rung
        else:
            assert (1.0 / rung) == int(1.0 / rung), rung


def test_the_ladder_is_sorted_and_inside_the_panes_bounds() -> None:
    ladder = inker_state.ZOOM_LADDER
    assert list(ladder) == sorted(ladder)
    assert ladder[0] >= inker_state.INKER_MIN_ZOOM
    assert ladder[-1] <= inker_state.INKER_MAX_ZOOM


def test_a_rung_step_goes_strictly_past_the_current_zoom() -> None:
    """Not nearest-then-step: a view sitting at 135% zooms *out* to 100% and
    *in* to 200%, rather than snapping sideways to 100% on a press labelled
    "in"."""
    assert inker_state.zoom_rung(1.35, +1) == 2.0
    assert inker_state.zoom_rung(1.35, -1) == 1.0
    assert inker_state.zoom_rung(1.0, +1) == 2.0
    assert inker_state.zoom_rung(1.0, -1) == 0.5


def test_the_ladder_holds_at_both_ends() -> None:
    ladder = inker_state.ZOOM_LADDER
    assert inker_state.zoom_rung(ladder[-1], +1) == ladder[-1]
    assert inker_state.zoom_rung(ladder[0], -1) == ladder[0]


def test_the_wheel_still_moves_in_five_percent_notches() -> None:
    """The keyboard took the ladder; the wheel is the *fine* control and keeps
    what it had. Aseprite splits the two the same way round."""
    assert inker_state.ZOOM_PERCENT_STEP == 5


# --- the status bar ----------------------------------------------------------


def test_the_status_bar_reads_the_four_things_it_used_to_leave_out() -> None:
    """Cursor colour, brush size, selection size, and the layer and frame.

    A source scan rather than a rendered frame: the bar is one ``muted`` line
    built from a list, so what is in the list is the whole assertion, and
    nothing headless has a cursor to hover with.
    """
    source = Path(inker_canvas.__file__).read_text(encoding="utf-8")
    body = source.split("def _status_bar")[1].split("\n# --- input")[0]
    assert "color_button" in body, "the colour under the cursor"
    assert "brush_size" in body, "the brush size [ and ] change"
    assert "mask.bounds" in body, "the size of the selection"
    assert "stack.active" in body, "which layer is being drawn into"
    assert "anim.current" in body, "which frame is being drawn into"


def test_the_cursor_readout_is_gated_on_being_over_the_page() -> None:
    """The invisible button covers the whole child, so "hovered" is not
    "over the drawing" -- and with tiling off ``canonical`` folds nothing, which
    is how a 256-wide document reported "-37, 412" as a position."""
    source = Path(inker_canvas.__file__).read_text(encoding="utf-8")
    body = source.split("def _cursor_pixel")[1].split("def _status_bar")[0]
    assert "in_bounds" in body


# --- what the pointer says ---------------------------------------------------


def test_the_ring_reaches_every_tool_the_brush_slider_sizes() -> None:
    """``doc.shape`` takes the same width the brush does, so a line has a
    stroke width -- and nothing on screen said what it was until it committed."""
    assert inker_canvas._RINGED_TOOLS == inker_state.PAINT_TOOLS | inker_state.SHAPE_TOOLS
    assert "line" in inker_canvas._RINGED_TOOLS
    assert "brush" in inker_canvas._RINGED_TOOLS


def test_the_single_cell_outline_waits_until_a_cell_is_bigger_than_its_line() -> None:
    assert inker_canvas.PIXEL_CELL_MIN_ZOOM >= 2.0


# --- the pane fits in the pane ----------------------------------------------

#: The blocks that fold, and the keys their open state is remembered under.
#: Named rather than counted, because *which* three is the decision: the tool
#: grid and the brush options are what the pane is for and stay open, and these
#: are the ones reached once and then left alone for an hour.
#: "Canvas" is deliberately absent: it was a folding block in a sidebar that
#: no longer draws it, and it is a popover off the rail's canvas glyph now --
#: which is a header of a different kind, with nothing to remember an open
#: state for. ``test_the_canvas_settings_are_reachable`` is what guards it.
FOLDING = {
    "Image brush": "inker/image-brush",
    "Presets": "inker/presets",
}


@pytest.mark.parametrize("label", sorted(FOLDING))
def test_the_occasional_blocks_fold_and_remember(label: str) -> None:
    """~45 rows of controls plus a five-row icon grid, in 55% of a 300 dp
    sidebar, with no folding anywhere: about half the pane was below the fold at
    UI scale 1.0 and the Brush section was almost entirely below it at 1.5.
    Plotter had already answered this half; the splitter answers the other."""
    source = Path(inker_tools.__file__).read_text(encoding="utf-8")
    assert f'"{label}", default_open=False, persist_key="{FOLDING[label]}"' in source


def test_the_blocks_that_are_the_panel_do_not_fold() -> None:
    """An asset opened with every section collapsed shows a column of headings
    and nothing to act on -- which is what ``widgets.header``'s own docstring
    says, and why the brush options are still a plain section."""
    source = Path(inker_tools.__file__).read_text(encoding="utf-8")
    # "Tools" went with the heading itself -- the toolbox is a 90 px rail and
    # a section title does not fit across it. "Brush" and "Selection" went with
    # their controls, to the context bar and to the Select menu. What is left
    # of the panel is still plain rather than folded.
    for label in ("Gradient", "Slices"):
        assert f'widgets.section("{label}")' in source


def test_the_canvas_settings_are_reachable() -> None:
    """``canvas_options`` had **zero callers**: the symmetry mode, its ways and
    its axis are the only writers of ``state.symmetry`` and friends, so
    symmetry was permanently off in the shipped app while the whole mirror
    engine behind it stayed live and tested. The grid's numeric step went the
    same way. Both are the toolbox's canvas popover now -- except the four
    mirrors, which are the context bar's toggles (2026-08-23), so what this
    checks is that *every* writer is reachable rather than that one of them is
    a combo.
    """
    from warlock.studio.panes import inker_context

    source = Path(inker_tools.__file__).read_text(encoding="utf-8")
    assert "canvas_options(ctx, state)" in source
    assert "imgui.open_popup(CANVAS_POPUP)" in source
    assert "state.symmetry = brush.toggled(" in source
    assert "state.radial_count = int(count)" in source
    assert "state.symmetry_axis = " in source
    bar = Path(inker_context.__file__).read_text(encoding="utf-8")
    assert "state.symmetry = brush.toggled(state.symmetry, axis)" in bar


def test_every_folding_key_is_namespaced_to_this_pane() -> None:
    """They share one settings dict with every other pane's."""
    for key in FOLDING.values():
        assert key.startswith("inker/")


def test_the_inker_workspace_has_drag_handles_of_its_own() -> None:
    """Both of its shares were ``lay.settings_share`` and it had no splitter,
    so the only way to change the split was to switch to Create and drag the
    handle there. One setting seen twice, not two settings to keep in step.

    Inker's columns are **data** now (``skeletons.inker``, wave 5) rather than
    a composition function, so what this pins is the same fact one layer down:
    the right column declares a share key of its own, and ``layout.column``
    derives the handle from it. ``tests/test_layout.py`` gates every key in
    both sources at once.
    """
    from warlock.studio import skeletons

    columns = skeletons.inker(None)
    keys = {slot.share_key for slot in columns["right"].slots if slot.share_key}
    assert keys == {"inker-tools", "inker-tiles"}
    # And the *left* column is a split now too: it was a fixed 90 px rail with
    # one pane in it and nothing to divide (W2.9), and it is the palette over
    # the picker after the columns swapped to Aseprite's default sides.
    assert {slot.share_key for slot in columns["left"].slots if slot.share_key} == {
        "inker-colors"
    }


def test_both_floors_are_named_where_the_panes_are() -> None:
    """The numbers belong to the panels that own them, not to the workspace
    that stacks them: the workspace does not know what a colour panel needs.

    The toolbox's pair went with the give-way split itself (W2.9): a fixed rail
    reserves nothing and gives way to nothing. What is here now is one floor
    per pane that can be squeezed, in both columns.
    """
    from warlock.studio import skeletons
    from warlock.studio.panes import inker_generate, inker_picker, inker_tiles

    assert inker_colors.PANEL_FLOOR > 0
    assert inker_tiles.PANEL_FLOOR > 0
    assert inker_picker.PICKER_FLOOR > 0
    assert inker_generate.GENERATE_FLOOR > 0
    # Read by the slot table rather than by the composition function: the
    # workspace is data now, and the floors travel with the panes they belong
    # to rather than with the code that stacks them.
    columns = skeletons.inker(None)
    floors = {
        slot.id: slot.floor
        for column in columns.values()
        for slot in column.slots
    }
    assert floors["inker-colors"] == inker_colors.PANEL_FLOOR
    assert floors["inker-picker"] == inker_picker.PICKER_FLOOR
    assert floors["inker-generate"] == inker_generate.GENERATE_FLOOR
    assert floors["inker-tiles"] == inker_tiles.PANEL_FLOOR


def test_a_pinned_give_way_handle_does_not_drive_the_share() -> None:
    """``give_way`` pins the tools pane at its own minimum, and the old drag
    arithmetic kept updating the stored share anyway: zero travel under the
    cursor while the right column -- keyed to the same share, no give_way --
    resized in real time. A drag the pane cannot follow leaves the share
    exactly where it was."""
    avail, wanted, floor = 1000.0, 500.0, 200.0
    share = 0.30
    assert layout.give_way(avail, share, wanted, floor) == pytest.approx(500.0)
    # Dragging up: the pane is already at its minimum, so nothing may move.
    assert layout.give_way_drag(avail, share, wanted, floor, -40.0) == share
    # Dragging down: the share moves from the height actually drawn, so the
    # pane under the handle follows the very first pixel of the drag.
    grown = layout.give_way_drag(avail, share, wanted, floor, 40.0)
    assert layout.give_way(avail, grown, wanted, floor) == pytest.approx(540.0)


def test_an_unpinned_give_way_drag_is_the_plain_arithmetic() -> None:
    """With the share honoured untouched the handle behaves as it always did."""
    avail, wanted, floor = 1000.0, 300.0, 200.0
    assert layout.give_way_drag(avail, 0.55, wanted, floor, 30.0) == pytest.approx(0.58)
    assert layout.give_way_drag(avail, 0.55, wanted, floor, -30.0) == pytest.approx(0.52)


def test_a_give_way_drag_respects_both_ends_and_a_collapsed_column() -> None:
    avail, wanted, floor = 1000.0, 300.0, 200.0
    # Down past everything: clamped at SHARE_MAX.
    assert layout.give_way_drag(avail, 0.6, wanted, floor, 900.0) == layout.SHARE_MAX
    # Up past everything: the pane stops at its own minimum (the ``wanted``
    # height here), and the share is re-derived from that height, not run to
    # SHARE_MIN while the pane sits still.
    pinched = layout.give_way_drag(avail, 0.6, wanted, floor, -900.0)
    assert layout.give_way(avail, pinched, wanted, floor) == pytest.approx(300.0)
    # A zero-height frame mid-resize changes nothing.
    assert layout.give_way_drag(0.0, 0.55, wanted, floor, 40.0) == 0.55


def test_the_toolbox_is_a_pane_with_a_floor_rather_than_a_rail() -> None:
    """``RAIL_W`` is gone, and so is the arithmetic it replaced.

    ``grid_height``/``_reserve`` went with the give-way split (W2.9): they
    mixed design and physical px and got it wrong twice, and the way to keep
    that arithmetic correct was not to have it. The 90 px rail that replaced
    them is gone in its turn -- 90 minus 16 px of pane padding left 74 px of
    content, which fitted two buttons exactly and clipped the two rows under
    them. What a shareable pane needs instead is a floor.
    """
    assert not hasattr(inker_tools, "grid_height")
    assert not hasattr(inker_tools, "_reserve")
    assert not hasattr(inker_tools, "OPTIONS_FLOOR")
    assert not hasattr(inker_tools, "RAIL_W")
    assert inker_tools.TOOLS_FLOOR > 0


def test_the_tool_grid_widens_with_the_pane_rather_than_staying_two_across()        -> None:
    """Two across was the rail's answer and it does not survive it: a 300 px
    column drawing a 68 px strip of buttons down one edge is the same defect
    the rail had, mirrored."""
    from warlock.studio import tokens

    step = inker_tools.BUTTON_W + inker_tools.GRID_GAP
    assert inker_tools.columns_for(0.0) == inker_tools.COLUMNS
    assert inker_tools.columns_for(step * tokens.SCALE) == inker_tools.COLUMNS
    assert inker_tools.columns_for(step * 4 * tokens.SCALE) == 4
    # And capped, so a very wide pane is still a square of icons rather than a
    # single row of twelve.
    assert inker_tools.columns_for(step * 40 * tokens.SCALE) == inker_tools.MAX_COLUMNS


def test_the_trailing_gap_is_drawn_and_counted_as_one_number() -> None:
    """The dummy under the grid goes through ``sp(GRID_GAP)`` so ``draw`` and
    ``grid_height`` cannot drift apart again."""
    source = Path(inker_tools.__file__).read_text(encoding="utf-8")
    assert "imgui.dummy((0, sp(GRID_GAP)))" in source


def test_a_zoom_rung_cannot_outlive_an_undrawn_canvas() -> None:
    """``pending_zoom_rung`` is banked by ``handle_key`` from outside the
    canvas and consumed inside the canvas child; on a frame the child (or the
    whole centre pane) does not draw, the flag used to survive and fire a zoom
    rung later, unprompted. ``pending_zoom`` cannot do this -- it is only ever
    set from inside a visible canvas -- so both invisible branches drop the
    rung rather than banking it."""
    canvas = Path(inker_canvas.__file__).read_text(encoding="utf-8")
    # Once consumed after stepping the ladder, once dropped by the invisible
    # child's branch.
    assert canvas.count("pending_zoom_rung = 0") >= 2
    main = Path(inker_tools.__file__).resolve().parent.parent / "main.py"
    assert "pending_zoom_rung = 0" in main.read_text(encoding="utf-8")


def _function(module, name: str) -> ast.FunctionDef:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} has moved out of {module.__name__}")


def test_the_palette_sources_share_one_row() -> None:
    """Four stacked buttons put three of them below the Colour pane's fold.

    Measured on 2026-08-23 at 1600x950: the pane's content ran out of height
    part-way through **Index to the swatches**, so the three palette-source
    buttons under it were clipped away entirely -- imgui reported them not
    visible, which means nobody could click them without scrolling first. One
    row of three short buttons is two rows shorter, and costs no height
    anywhere else in a column whose Picker pane is over its allotment too.
    """
    node = _function(inker_colors, "_not_indexed")
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and _name(call) in {"button", "disabled_button", "same_line"}
    ]
    kinds = [_name(call) for call in calls]
    # index-to-swatches on its own row, then three buttons joined by two
    # same_line calls: button, same_line, button, same_line, button.
    assert kinds == [
        "disabled_button",
        "button",
        "same_line",
        "button",
        "same_line",
        "button",
    ], kinds


def test_a_palette_source_button_is_short_enough_for_a_third_of_the_column() -> None:
    """Three across a ~300 px column, so the words have to be single ones.

    The sentence each one used to be lives on as its tooltip -- the label is
    what has to fit, not the explanation.
    """
    node = _function(inker_colors, "_not_indexed")
    labels = [
        call.args[0].value
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and _name(call) == "button"
        and call.args
        and isinstance(call.args[0], ast.Constant)
    ]
    assert len(labels) == 3, labels
    for label in labels:
        assert len(label) <= 12, label
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and _name(call) == "button":
            assert any(
                kw.arg == "tooltip" for kw in call.keywords
            ), f"{call.args[0].value} lost the sentence it used to be"


def test_the_palette_section_is_a_header_that_starts_closed_when_unindexed() -> None:
    """The one change that actually clears the Colour pane's fold.

    Compacting the palette-source buttons onto one row saved two rows and left
    them 50 px below the pane's bottom anyway -- measured at the app's own
    default 1600x950, where ``inker-colors`` is 470 px tall over 516 px of
    content and the Picker under it is 37 px over its own allotment, so there
    is no height to move between them. Collapsed, the block costs one row.

    Open when the document *has* a table, because then the section is the
    document's own storage rather than four ways to make one.
    """
    node = _function(inker_colors, "_indexed")
    calls = [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and _name(call) in {"header", "section"}
    ]
    assert [_name(call) for call in calls] == ["header"], (
        "the Palette heading must be collapsible, not a plain section"
    )
    (header,) = calls
    assert header.args[0].value == "Palette"
    keywords = {kw.arg for kw in header.keywords}
    assert "default_open" in keywords, "the header must state its own default"
    assert "persist_key" in keywords, "a collapse the user chose must survive a restart"


def test_a_collapsed_palette_draws_nothing_it_would_have_to_balance() -> None:
    """``begin_disabled`` without its ``end_disabled`` unbalances the frame.

    The body is guarded by the header, so every call that has a partner has to
    be inside the guard with it -- the failure otherwise is not a missing
    section but a wedged imgui two panes later.
    """
    node = _function(inker_colors, "_indexed")
    body = ast.unparse(node)
    assert body.count("begin_disabled") == body.count("end_disabled") == 1
    # ``if not header(...): return`` -- the pane's own idiom, one the Shades
    # strip above already uses. Everything with a partner has to sit *after*
    # that return, so a collapsed section leaves the frame balanced.
    guard = next(
        (
            index
            for index, stmt in enumerate(node.body)
            if isinstance(stmt, ast.If)
            and "header" in ast.unparse(stmt.test)
            and any(isinstance(inner, ast.Return) for inner in stmt.body)
        ),
        None,
    )
    assert guard is not None, "the header does not guard the body"
    before = ast.unparse(ast.Module(body=node.body[:guard], type_ignores=[]))
    assert "begin_disabled" not in before and "end_disabled" not in before


def test_the_picker_floor_is_the_height_its_own_content_needs() -> None:
    """A floor that does not cover the pane's content protects nothing.

    ``PICKER_FLOOR`` named the heading, the target row, the tab strip, the four
    sliders *and* the hex field, and put 200 px on them. Measured at the app's
    own default 1600x950 on 2026-08-23, that content runs 400 px: the pane was
    allotted 363 px, which is comfortably above a 200 px floor, so ``give_way``
    never intervened and the hex field was drawn at y=909 against a pane that
    ended at 902 -- clipped away, unclickable, the last such control in Inker.
    The floor has to be what the pane actually needs, or it is decoration.
    """
    from warlock.studio.panes import inker_picker

    assert inker_picker.PICKER_FLOOR >= 400.0


def test_the_colour_pane_gives_way_to_the_pickers_floor() -> None:
    """The left column at 1600x950, which is what a first run gets.

    833 px of column between the two panes' outer edges. The share alone would
    hand the Colour pane 458 px and leave the Picker 375 -- 25 px short of its
    content. ``give_way`` is the mechanism that stops it, and this is the
    arithmetic that says it does.
    """
    from warlock.studio import layout
    from warlock.studio.panes import inker_colors, inker_picker

    avail = 833.0
    share = layout.SHARE_DEFAULTS.get("inker-colors", 0.55)
    top = layout.give_way(
        avail, share, inker_colors.PANEL_FLOOR, inker_picker.PICKER_FLOOR
    )
    assert avail - top >= inker_picker.PICKER_FLOOR
