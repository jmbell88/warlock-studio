"""Tinted section blocks: the grouping behind every panel heading.

``widgets.section`` used to draw a heading and a gap and nothing else, on the
argument that "a rule is a line drawn where the rhythm failed" (UX.md Phase 2).
The rhythm did fail, in the one place the argument did not account for: a
sidebar with six headings in it is not a page with six headings on it, and at
300 px wide the gap that separates two sections is the same gap that separates a
heading from its own first control. Reading order stopped being obvious --
which is the report this came from, for the Clay tools, outliner and properties
panes and Plotter's tools pane.

So a section is a *surface* now rather than a gap. The heading and everything
under it up to the next heading sit on one tinted block, and the blocks are what
the eye groups by. The rule is still not back: nothing here draws a line.

**The blocks are opt-in per pane**, which is the one thing that changed when the
editor shell landed. ``layout.pane`` used to open a scope for every pane it
drew, on the argument that the tint is a property of *being* a pane; the shell
overturned that, because a workspace canvas is a pane too and wants no tint.
What did not change is the report above -- the four narrow sidebars it names
still need the grouping, so they open a scope themselves and
``test_the_four_sidebars_the_report_named_still_ask_for_blocks`` is what keeps
them from quietly losing it again.

The mechanism is the interesting part and is what this file mostly pins. A
block's height is not known until its content has been drawn, and imgui has no
way to insert behind what is already in a draw list -- so the scope splits the
window's draw list into two channels, puts content on the upper one and the
fills on the lower, and merges on the way out. Getting that wrong does not
produce a wrong colour; it produces a frame that renders in the wrong order or
an unbalanced splitter that corrupts the list, which is why the balance is
asserted directly rather than inferred from a screenshot.
"""

from __future__ import annotations

import pytest

from warlock.studio import widgets


@pytest.fixture
def imgui_ctx(gl):
    """This file's own imgui context, built and torn down around it.

    Copied from ``test_poser_panes_smoke`` rather than shared with
    ``test_studio_smoke``'s session-scoped one, and the reason is worth writing
    down because it costs a duplicated fixture. Two imgui contexts, each with
    its own ``ImguiRenderer`` over the *one* GL context, crash the process with
    an access violation when they overlap -- so the rule is that at most one
    exists at a time. A session-scoped context shared across files does not
    overlap with anything only by accident of collection order, and this file
    sorts before ``test_studio_controls``, which builds its own. Building and
    destroying one here keeps the invariant true regardless of ordering.

    ``prev_ctx`` is restored for the same reason: ``destroy_context`` leaves no
    current context at all.
    """
    from imgui_bundle import imgui

    from warlock.studio import imgui_backend, theme

    prev_ctx = imgui.get_current_context()
    prev_screen = type(gl).__dict__.get("screen")
    fbo = gl.simple_framebuffer((1600, 950))
    fbo.use()
    type(gl).screen = property(lambda _self: fbo)

    ctx = imgui.create_context()
    io = imgui.get_io()
    # See ``test_studio_smoke``'s fixture: a persisted collapsed flag in
    # ``imgui.ini`` survives the process and silently empties every window.
    io.set_ini_filename(None)
    io.display_size = (1600, 950)
    io.delta_time = 1 / 60
    io.fonts.add_font_default()
    theme.apply(imgui)
    renderer = imgui_backend.ImguiRenderer(gl)
    yield imgui, renderer
    renderer.shutdown()
    imgui.destroy_context(ctx)
    if prev_screen is not None:
        type(gl).screen = prev_screen
    if prev_ctx is not None:
        imgui.set_current_context(prev_ctx)


@pytest.fixture
def frame(imgui_ctx):
    """Open a window, run ``build`` in it, close the frame. -> what build returned.

    Its own window per call, as ``_claimed_ids`` uses one: the scope keys off
    the *current* draw list, and a shared host would hide the one bug this file
    exists to catch (two scopes fighting over one list).
    """
    imgui, _renderer = imgui_ctx
    counter = [0]

    def run(build, *, size=(420.0, 900.0)):
        counter[0] += 1
        out = []
        imgui.new_frame()
        imgui.set_next_window_size(size)
        imgui.begin(f"##blocks-{counter[0]}")
        out.append(build())
        imgui.end()
        imgui.render()
        return out[0]

    return run


def test_a_section_outside_a_scope_draws_no_block(frame):
    """The graceful half. Not every surface that draws a heading is a pane --
    a popup, a tooltip, the manual's own body -- and a heading in one of those
    keeps exactly the look it had, rather than growing a block that spans
    whatever container it happens to be in."""

    def build():
        widgets.section("Alone")
        return widgets.section_block_count()

    assert frame(build) == 0


def test_each_section_in_a_scope_gets_one_block(frame):
    def build():
        with widgets.section_blocks() as scope:
            widgets.section("Tools")
            widgets.muted("a")
            widgets.section("Map")
            widgets.muted("b")
            widgets.section("Properties")
            widgets.muted("c")
        return list(scope.blocks)

    blocks = frame(build)
    assert len(blocks) == 3


def test_a_block_starts_above_its_heading_and_ends_below_its_last_row(frame):
    """The block is the group, not the heading. A fill that began *under* the
    heading would put the one word naming the group outside the surface the
    group is drawn on, which is the opposite of what it is for."""

    def build():
        with widgets.section_blocks() as scope:
            widgets.section("Tools")
            top = imgui.get_cursor_screen_pos().y
            widgets.muted("row one")
            widgets.muted("row two")
            bottom = imgui.get_cursor_screen_pos().y
        return list(scope.blocks), top, bottom

    from imgui_bundle import imgui

    blocks, top, bottom = frame(build)
    assert len(blocks) == 1
    x0, y0, x1, y1 = blocks[0]
    assert y0 < top, "the heading sits inside its own block"
    assert y1 >= bottom, "so does the last row under it"
    assert x1 > x0


def test_consecutive_blocks_do_not_overlap_and_keep_their_order(frame):
    """The gap between two blocks is what says "a new thing starts here" --
    the job the SP_6 gap used to do alone. Overlapping blocks would read as one
    surface with a seam through it."""

    def build():
        with widgets.section_blocks() as scope:
            widgets.section("One")
            widgets.muted("a")
            widgets.section("Two")
            widgets.muted("b")
        return list(scope.blocks)

    blocks = frame(build)
    assert len(blocks) == 2
    assert blocks[0][3] < blocks[1][1], "block one ends before block two starts"


def test_the_scope_merges_the_draw_list_even_when_the_body_raises(frame):
    """An unbalanced splitter is not a cosmetic failure: the list is left in a
    state the renderer walks wrongly, and the frame after it is the one that
    looks broken. The merge is in a ``finally`` for that reason."""

    def build():
        with pytest.raises(RuntimeError), widgets.section_blocks():
            widgets.section("One")
            raise RuntimeError("boom")
        return widgets.section_scope_depth()

    assert frame(build) == 0


def test_a_second_scope_on_the_same_draw_list_is_a_no_op(frame):
    """One list, one splitter -- imgui's own restriction. A pane that opened a
    scope and then called a helper that opened another would split an already
    split list, and the corruption shows up in a *different* pane."""

    def build():
        with widgets.section_blocks() as outer:
            widgets.section("One")
            widgets.muted("a")
            with widgets.section_blocks() as inner:
                widgets.section("Two")
                widgets.muted("b")
            same = inner is outer
        # Read after the scope: a block is painted when it *closes*, so the last
        # one does not exist until the scope that owns it has ended.
        return list(outer.blocks), same

    blocks, same = frame(build)
    assert same, "the inner scope hands back the outer one rather than splitting again"
    assert len(blocks) == 2, "and its sections still land on the outer scope's blocks"


def test_a_scope_that_draws_no_section_costs_nothing(frame):
    """Most panes have a branch that draws a sentence and returns -- "Open a map
    first". A block around nothing would be a coloured rectangle with one line
    of grey text in it."""

    def build():
        with widgets.section_blocks() as scope:
            widgets.muted("Open or start a map to draw on.")
        return list(scope.blocks)

    assert frame(build) == []


# --- the wiring ---------------------------------------------------------------
#
# Which containers open a scope is a decision per container, not a property of
# drawing a heading, and both answers are deliberate. These pin the ones that
# say yes; the ones that say no (the manual's prose, the library's date groups)
# carry their reasoning inline at the ``widgets.section`` call.


def test_a_pane_is_flat_until_it_asks_for_blocks(frame):
    """``layout.pane`` opens no scope of its own -- the editor shell's panes are
    flat, and the tint is a thing a pane *asks* for. What this pins is that the
    ask is all it takes: one ``with`` inside the pane body and every heading
    under it is on a block, with the scope closed on the way out."""
    from warlock.studio import layout

    def build():
        depths = []
        with layout.pane("probe-pane", (300.0, 400.0)) as visible:
            depths.append(widgets.section_scope_depth())
            if visible:
                with widgets.section_blocks():
                    depths.append(widgets.section_scope_depth())
                    widgets.section("Tools")
                    widgets.muted("a")
        depths.append(widgets.section_scope_depth())
        return depths

    bare, asked, after = frame(build)
    assert bare == 0, "the pane itself opens nothing"
    assert asked == 1, "the pane asked, so it has one"
    assert after == 0, "and it closed on the way out"


def test_two_sibling_panes_each_get_their_own_scope(frame):
    """Never one shared splitter over two panes. Each ``begin_child`` has its
    own draw list, which is exactly why per-pane is the safe granularity -- and
    why a *nested* scope over one list has to be the no-op it is."""
    from warlock.studio import layout

    def build():
        seen = []
        for name in ("probe-a", "probe-b"):
            with layout.pane(name, (200.0, 300.0)) as visible:
                if visible:
                    with widgets.section_blocks():
                        widgets.section("Heading")
                        widgets.muted("row")
                        seen.append(widgets.section_scope_depth())
        return seen, widgets.section_scope_depth()

    seen, after = frame(build)
    assert seen == [1, 1], "one at a time, not nested"
    assert after == 0


def test_the_scope_survives_a_pane_that_draws_nothing(frame):
    """A culled or zero-height pane yields ``False`` and its body is skipped.
    The scope still has to balance, because the split already happened."""
    from warlock.studio import layout

    def build():
        with layout.pane("probe-culled", (0.0, 0.0)) as visible:
            drew = visible
        return drew, widgets.section_scope_depth()

    _drew, after = frame(build)
    assert after == 0


def test_end_section_closes_a_block_without_starting_another(frame):
    """Home mixes headed groups with content that has no heading, and without
    this the "Unsaved work" block ran all the way down through the release
    note, the New button and the status line to the next heading."""

    def build():
        with widgets.section_blocks() as scope:
            widgets.section("Unsaved work")
            widgets.muted("a row")
            widgets.end_section()
            after = len(scope.blocks)
            widgets.muted("belongs to nothing")
            widgets.muted("nor this")
        return after, list(scope.blocks)

    after, blocks = frame(build)
    assert after == 1, "the block closed at the call, not at the end of the scope"
    assert len(blocks) == 1, "and the unheaded rows below started no new one"


def test_end_section_is_harmless_with_no_block_open(frame):
    """Called twice, or before any heading, or outside a scope entirely -- a
    pane should not have to track whether it has opened one."""

    def build():
        widgets.end_section()  # no scope at all
        with widgets.section_blocks() as scope:
            widgets.end_section()  # scope, no block
            widgets.section("One")
            widgets.muted("a")
            widgets.end_section()
            widgets.end_section()  # already closed
        return list(scope.blocks)

    assert len(frame(build)) == 1


def test_a_scope_inside_a_child_window_is_a_real_second_scope(frame):
    """The 2D pane's fix (2026-08-17). The pane opens a scope on its own draw
    list; the 2D form then opens a child window (``2d-form``) with its *own*
    list and an opaque background -- so fills painted on the parent's list are
    covered, and only the 8dp left overhang survived. The fix is a second scope
    opened inside the child, and it is only a fix because the dedup keys on the
    draw list's owner name (distinct per child window) rather than treating any
    nested scope as a no-op.

    Unchanged by the editor shell going flat: what moved is *who* opens the
    outer scope, not how a nested one behaves."""
    from imgui_bundle import imgui

    from warlock.studio import layout

    def build():
        depths = []
        blocks = []
        with layout.pane("probe-2d", (320.0, 500.0)) as visible:
            if visible:
                with widgets.section_blocks():
                    depths.append(widgets.section_scope_depth())
                    if imgui.begin_child("probe-2d-form", (0.0, 300.0)):
                        child_left = imgui.get_cursor_screen_pos().x
                        child_width = imgui.get_content_region_avail().x
                        with widgets.section_blocks() as scope:
                            depths.append(widgets.section_scope_depth())
                            widgets.section("Output")
                            widgets.muted("a row")
                        blocks = list(scope.blocks)
                    imgui.end_child()
        return depths, blocks, child_left, child_width

    depths, blocks, child_left, child_width = frame(build)
    assert depths == [1, 2], "the child's scope is real, not the pane's again"
    assert len(blocks) == 1
    x0, _y0, x1, _y1 = blocks[0]
    # The block spans the child's content width (give or take the overhang),
    # not a sliver at the left edge.
    assert x1 - x0 >= child_width * 0.9, (x0, x1, child_width)
    assert x0 <= child_left


def test_the_four_sidebars_the_report_named_still_ask_for_blocks():
    """The regression guard for the flat-pane change.

    ``layout.pane`` no longer opens a scope, so the grouping these four panes
    were given in the first place now depends on each of them asking. Nothing
    else in the app would notice if one stopped: the headings still draw, they
    just stop being grouped, which is precisely the failure the module docstring
    describes and precisely the kind a screenshot test would not catch either.

    Read off the source rather than rendered, because rendering four real panes
    needs four real documents -- and what is being pinned is the *ask*, which is
    a property of the code, not of a particular document.
    """
    import ast
    import inspect

    from warlock.studio.panes import clay_outliner, clay_props, clay_tools, plotter_tools

    for module in (clay_tools, clay_outliner, clay_props, plotter_tools):
        tree = ast.parse(inspect.getsource(module))
        draw = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "draw"
        )
        opens = [
            node
            for node in ast.walk(draw)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "section_blocks"
        ]
        assert opens, (
            f"{module.__name__}.draw no longer opens a section_blocks scope -- "
            "its headings are back to being a column of gaps"
        )
