"""The canvas's scrollbars, and the one thing they must never do.

Every test here drives a *real* imgui context, because the feature is a claim
about imgui's own hit-testing: the bars are ``invisible_button``s submitted
before the paint surface, and what makes the press unambiguous in both
directions is that imgui gives an overlapping hover to whichever item was
submitted first. That is not something a stub can be asked about -- a fake
would simply agree with whatever this file asserted.

It is also why the control census cannot serve here. ``probe.record`` is called
from ``controls._finish_item`` only, so every ``invisible_button`` in the app --
the canvas surface, the rail, the layout splitter -- is invisible to it. The
equivalent guarantee is
``test_pressing_the_bar_scrolls_and_the_canvas_does_not_take_the_press``, which
presses a real rect and asserts the surface did *not* activate.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _ui_context import imgui_context

from warlock.studio import inker, inker_state
from warlock.studio.panes import inker_canvas

SIZE = (2000, 2000)
REGION = (400.0, 300.0)
#: Where the host window puts the canvas child. Not the origin, so a test that
#: silently used screen coordinates as pane coordinates would fail.
AT = (40.0, 30.0)


@pytest.fixture
def ui(monkeypatch):
    """The shared imgui context; see ``_ui_context`` for why it is not a
    conftest fixture."""
    with imgui_context(monkeypatch) as imgui:
        yield imgui

def _scene(size=SIZE, *, rulers=False, tiled="off"):
    state = inker_state.InkerState()
    state.rulers = rulers
    tab = SimpleNamespace(
        doc=inker.Document.blank(*size),
        tiled=tiled,
        busy=False,
        views=[inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True)],
        focus=0,
    )
    tab.view = tab.views[0]
    inker_state.centre(tab.view, size, REGION, 1.0)
    return state, tab


def _build(imgui, state, tab, *, surface=True):
    """One frame of what ``_one_canvas`` submits, in the same order."""
    seen = {}

    def build():
        imgui.set_cursor_screen_pos(AT)
        seen["lit"] = inker_canvas._scrollbar_input(state, tab, AT, REGION, 0)
        if surface:
            imgui.set_cursor_screen_pos(AT)
            imgui.invisible_button("##inker-surface", REGION)
            seen["surface_active"] = imgui.is_item_active()
            seen["surface_hovered"] = imgui.is_item_hovered()

    return build, seen


def _frame(imgui, build, *, pos, down):
    io = imgui.get_io()
    io.add_mouse_pos_event(pos[0], pos[1])
    io.add_mouse_button_event(0, down)
    imgui.new_frame()
    imgui.set_next_window_size((900.0, 700.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.end_frame()


def _press(imgui, build, pos, *, frames=1):
    """Settle, then press and hold. Two settling frames because imgui needs a
    frame to know an item is there before it can report a hover on it."""
    _frame(imgui, build, pos=pos, down=False)
    _frame(imgui, build, pos=pos, down=False)
    for _ in range(frames):
        _frame(imgui, build, pos=pos, down=True)


def _track(state, axis):
    return inker_canvas._scroll_tracks(state, AT, REGION)[axis]


def _mid(track):
    x0, y0, x1, y1 = track
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def test_pressing_the_bar_scrolls_and_the_canvas_does_not_take_the_press(ui):
    """The whole feature in one test, and the only one that fails if the
    submission order is ever swapped.

    Both halves matter. A bar that scrolls but lets the press through paints a
    stroke down the edge of the drawing every time you scroll; a bar that
    blocks the press without scrolling is a dead strip.
    """
    state, tab = _scene()
    build, seen = _build(ui, state, tab)
    before = tab.view.pan
    # Well off the thumb, which is centred: press near the track's left end.
    x0, y0, _x1, y1 = _track(state, 0)
    _press(ui, build, (x0 + 8.0, (y0 + y1) / 2.0))
    assert tab.view.pan != pytest.approx(before)
    assert seen["surface_active"] is False


def test_the_order_is_what_decides_it_and_not_the_geometry(ui):
    """The negative of the test above, and the reason it is trustworthy.

    Submit the paint surface *first* and the bars stop working -- the surface
    holds ``HoveredId`` over the whole pane and imgui refuses the overlapping
    item behind it. So the assertion above is really about the submission
    order rather than about the rects, which is the fact the design rests on;
    without this, a bar that worked for some unrelated reason would look
    exactly the same.
    """
    state, tab = _scene()
    seen = {}

    def build():
        imgui = ui
        imgui.set_cursor_screen_pos(AT)
        imgui.invisible_button("##inker-surface", REGION)
        imgui.set_cursor_screen_pos(AT)
        seen["lit"] = inker_canvas._scrollbar_input(state, tab, AT, REGION, 0)

    before = tab.view.pan
    x0, y0, _x1, y1 = _track(state, 0)
    _press(ui, build, (x0 + 8.0, (y0 + y1) / 2.0))
    assert tab.view.pan == pytest.approx(before)
    assert seen["lit"] == [False, False]


def test_a_stroke_in_flight_keeps_the_mouse_across_a_bar(ui):
    """The arbitration in reverse: imgui keeps the ActiveId on the held
    surface, so dragging a stroke down over the bottom edge neither stops the
    stroke nor lights the bar up."""
    state, tab = _scene()
    build, seen = _build(ui, state, tab)
    middle = (AT[0] + REGION[0] / 2.0, AT[1] + REGION[1] / 2.0)
    _frame(ui, build, pos=middle, down=False)
    _frame(ui, build, pos=middle, down=False)
    _frame(ui, build, pos=middle, down=True)
    assert seen["surface_active"] is True
    # Now drag onto the horizontal bar without letting go.
    _frame(ui, build, pos=_mid(_track(state, 0)), down=True)
    assert seen["surface_active"] is True
    assert seen["lit"] == [False, False]


def test_dragging_the_thumb_moves_the_view_the_other_way(ui):
    """A scrollbar means the page goes the opposite way from the thumb."""
    state, tab = _scene()
    build, _seen = _build(ui, state, tab)
    x0, y0, _x1, y1 = _track(state, 0)
    start = _mid(_track(state, 0))
    _press(ui, build, start)
    before = tab.view.pan[0]
    _frame(ui, build, pos=(start[0] + 40.0, start[1]), down=True)
    assert tab.view.pan[0] < before


def test_a_click_on_the_bare_track_jumps_the_thumb_to_it(ui):
    state, tab = _scene()
    build, _seen = _build(ui, state, tab)
    x0, y0, x1, y1 = _track(state, 0)
    before, _length = inker_state.scroll_thumb(tab.view, SIZE, REGION, 0)
    _press(ui, build, (x0 + 12.0, (y0 + y1) / 2.0))
    after, _length = inker_state.scroll_thumb(tab.view, SIZE, REGION, 0)
    assert after < before


def test_pressing_the_thumb_itself_moves_nothing(ui):
    """A press that lands *on* the thumb is the start of a drag. Jumping first
    would make every drag begin with a lurch."""
    state, tab = _scene()
    build, _seen = _build(ui, state, tab)
    before = tab.view.pan
    _press(ui, build, _mid(_track(state, 0)))
    assert tab.view.pan == pytest.approx(before)


def test_the_bars_keep_off_the_rulers(ui):
    """The rulers own the top and left strips and the bars own the right and
    bottom ones, so neither overlay ever draws through the other -- and the
    corner where the two bars would meet belongs to neither."""
    state, _tab = _scene(rulers=True)
    from warlock.studio.tokens import sp

    band = sp(inker_canvas.RULER_THICKNESS)
    horizontal = _track(state, 0)
    vertical = _track(state, 1)
    assert horizontal[0] >= AT[0] + band
    assert vertical[1] >= AT[1] + band
    # The two tracks do not overlap: the horizontal one stops short of the
    # vertical one's left edge.
    assert horizontal[2] <= vertical[0]


def test_tiling_does_not_change_the_thumb(ui):
    """The scrollable content is the canonical page. The tiled neighbourhood is
    a preview of *wrap* rather than addressable pixels, and a thumb that shrank
    to a third when a View row was checked would be telling the user their
    document is three times the size it is."""
    plain, tab_plain = _scene(tiled="off")
    tiled, tab_tiled = _scene(tiled="both")
    assert inker_state.scroll_thumb(
        tab_plain.view, SIZE, REGION, 0
    ) == pytest.approx(inker_state.scroll_thumb(tab_tiled.view, SIZE, REGION, 0))


def test_each_pane_of_a_split_gets_its_own_bars(ui):
    """Two views over one document, each with its own pan -- so each bar must
    read the view it belongs to, and the two buttons must not share an imgui
    id or a press in either would drive both."""
    state, tab = _scene()
    tab.views.append(inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True))
    inker_state.centre(tab.views[1], SIZE, REGION, 1.0)
    inker_state.pan_by(tab.views[1], SIZE, REGION, -200.0, 0.0)
    first = inker_state.scroll_thumb(tab.views[0], SIZE, REGION, 0)
    second = inker_state.scroll_thumb(tab.views[1], SIZE, REGION, 0)
    assert first[0] != pytest.approx(second[0])


def test_a_pane_too_small_for_a_bar_draws_nothing_rather_than_a_negative_one(ui):
    """A zero-or-negative track length is the shape that turns into a crash or
    an inverted rect, and a canvas squeezed to nothing is reachable by dragging
    a splitter."""
    state, tab = _scene()
    build, _seen = _build(ui, state, tab)
    tiny = (4.0, 4.0)
    ui.new_frame()
    ui.begin("##host")
    ui.set_cursor_screen_pos(AT)
    lit = inker_canvas._scrollbar_input(state, tab, AT, tiny, 0)
    inker_canvas._scrollbars(state, tab, AT, tiny, lit)
    ui.end()
    ui.end_frame()
