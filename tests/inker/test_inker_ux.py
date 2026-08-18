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

from warlock.studio import inker_state, theme, tokens
from warlock.studio.panes import inker_textures, inker_tools

STUDIO = Path(inker_tools.__file__).resolve().parent.parent
PANES = sorted((STUDIO / "panes").glob("*.py"))

#: The controls that draw at the full width of the content region. imgui puts a
#: slider's or a combo's own label *outside* the widget, to its right, so at
#: ``-1`` there is nothing left of the line for anything to follow on.
FULL_WIDTH = {"labeled_combo", "labeled_slider_int", "labeled_slider_float"}

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


def test_the_image_brush_section_is_opened_after_the_brush_options() -> None:
    """``section`` tints from its heading to the *next* one.

    Opened in the middle of the brush options, "Image brush" ran over the ink
    radio, Opacity, Spacing, Smoothing and Taper -- controls it does not
    configure, under a heading that says it does, on brush, eraser and spray,
    i.e. the three most-used tools in the box.
    """
    source = Path(inker_tools.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    options = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_options"
    )
    calls = [n for n in ast.walk(options) if isinstance(n, ast.Call)]
    stamp = [n.lineno for n in calls if _name(n) == "_image_brush"]
    # By label, not by "every full-width control in the function": the tools
    # after this one have their own sections (Tolerance, Gradient), they are
    # drawn under a different branch of the same ``if`` ladder, and they never
    # share a frame with the image brush.
    brush_controls = {
        "Size", "Nib", "Opacity", "Spacing", "Rate", "Strength", "Smoothing", "Taper"
    }
    sliders = [
        n.lineno
        for n in calls
        if _name(n) in FULL_WIDTH
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value in brush_controls
    ]
    assert stamp, "the image brush block left _options entirely"
    assert sliders, "the brush controls were renamed; this test needs the new names"
    assert max(sliders) < min(stamp), (
        "a full-width brush control is drawn after the Image brush heading, so "
        "it is tinted into that block and named by it"
    )


def test_the_accent_line_about_a_live_transform_wraps() -> None:
    """At 42 characters it was cut off at the sidebar edge, mid-sentence."""
    source = Path(inker_tools.__file__).read_text(encoding="utf-8")
    assert 'widgets.wrapped(theme.ACCENT, "Transforming' in source
    assert hasattr(theme, "ACCENT")
