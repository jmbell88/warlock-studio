"""The slice overlay and its gestures, without a window.

Two properties, and both are the ones this pane has already got wrong once for
another overlay.

**Everything is placed through the view's basis.** An overlay that computes
``origin + x * zoom`` is right at rotation 0 and silently a quarter turn out
everywhere else, which is what the grid and the marching ants both did before
``_corners``/``_box`` existed. So the rectangles below are checked against
*numbers*, on a turned and on a mirrored page, rather than against the helper
that produced them.

**One gesture is one Ctrl+Z.** A slice drag mutates the live object every frame
so the overlay follows the cursor, which is exactly the shape that pushes a step
per frame if the release is written carelessly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import inker_state
from warlock.studio.inker.document import Document
from warlock.studio.inker.slices import SliceKey
from warlock.studio.panes import inker_canvas

ORIGIN = (0.0, 0.0)


class _Lines:
    """A draw list that only takes notes.

    At an identity view and a zero origin a screen coordinate *is* the image
    coordinate, so under a turn the recorded numbers are exactly what the view
    transform did -- which is the thing being asserted.
    """

    def __init__(self) -> None:
        self.lines: list[tuple] = []
        self.rects: list[tuple] = []
        self.filled: list[tuple] = []
        self.circles: list[tuple] = []

    def add_line(self, a, b, colour, thickness=1.0) -> None:
        self.lines.append((a, b))

    def add_rect(self, a, b, colour) -> None:
        self.rects.append((a, b))

    def add_rect_filled(self, a, b, colour) -> None:
        self.filled.append((a, b))

    def add_circle(self, centre, radius, colour) -> None:
        self.circles.append((centre, radius))


@pytest.fixture(autouse=True)
def _no_context(monkeypatch):
    """``_u32`` reaches into a live imgui context and takes the process down
    without one, which is the whole reason this pane's drawing was untested."""
    monkeypatch.setattr(inker_canvas, "_u32", lambda colour, alpha=1.0: 0)


def _tab(width: int = 32, height: int = 16, **view) -> object:
    doc = Document.blank(width, height)
    return inker_state.InkerDoc(doc=doc, view=inker_state.PaintView(**view))


def _state(tab, tool: str = "slice") -> inker_state.InkerState:
    state = inker_state.InkerState(tool=tool)
    state.add(tab)
    return state


def _draw(state, tab) -> _Lines:
    lines = _Lines()
    inker_canvas._slices(state, tab, lines, ORIGIN)
    return lines


def _at(monkeypatch, x: float, y: float) -> None:
    monkeypatch.setattr(
        inker_canvas.imgui, "get_mouse_pos", lambda: SimpleNamespace(x=x, y=y)
    )
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=False, key_alt=False),
    )


# --- the overlay --------------------------------------------------------------


def test_a_slice_is_drawn_where_it_is():
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    tab.doc.add_slice((2, 3, 10, 7))
    assert _draw(_state(tab), tab).rects == [((2.0, 3.0), (10.0, 7.0))]


def test_a_slice_follows_a_quarter_turn_of_the_page():
    """The failure this guards is not a crash. ``origin + x * zoom`` would put
    the rectangle at ``(2, 3)-(10, 7)`` here, which is where the canvas *is not*
    once the page has been turned."""
    tab = _tab(zoom=1.0, pan=(0.0, 0.0), rotation=90)
    tab.doc.add_slice((2, 3, 10, 7))
    # One clockwise quarter turn on a downward-y screen is ``(x, y) -> (-y, x)``,
    # so the corners are (-3, 2) and (-7, 10) and the box is their min/max.
    assert _draw(_state(tab), tab).rects == [((-7.0, 2.0), (-3.0, 10.0))]


def test_a_slice_follows_a_mirrored_page():
    tab = _tab(zoom=1.0, pan=(0.0, 0.0), flipped=True)
    tab.doc.add_slice((2, 3, 10, 7))
    assert _draw(_state(tab), tab).rects == [((-10.0, 3.0), (-2.0, 7.0))]


def test_the_selected_slice_gets_four_handles_and_the_others_do_not():
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    first = tab.doc.add_slice((0, 0, 4, 4))
    tab.doc.add_slice((8, 8, 12, 12))
    state = _state(tab)

    assert _draw(state, tab).filled == []
    state.slice_uid = first.uid
    handles = _draw(state, tab).filled
    assert len(handles) == 4
    # Centred on the four corners, whichever way the page is turned.
    centres = {((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) for a, b in handles}
    assert centres == {(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)}


def test_a_pivot_draws_a_ring_and_a_crosshair_at_the_right_place():
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    tab.doc.add_slice((4, 4, 12, 12), pivot=(4.0, 8.0))
    drawn = _draw(_state(tab), tab)
    assert drawn.circles and drawn.circles[0][0] == (8.0, 12.0)
    assert len(drawn.lines) == 2


def test_a_nine_slice_centre_is_dashed_inside_its_slice():
    """Dashed, because a solid rectangle a few pixels inside the slice's own
    outline reads as one thick border rather than as two things."""
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    tab.doc.add_slice((0, 0, 30, 12), center=(10, 4, 20, 8))
    drawn = _draw(_state(tab), tab)
    assert len(drawn.rects) == 1, "the centre is not a second rect"
    assert drawn.lines, "it is drawn as dashes"
    xs = [p[0] for pair in drawn.lines for p in pair]
    ys = [p[1] for pair in drawn.lines for p in pair]
    assert (min(xs), max(xs)) == (10.0, 20.0)
    assert (min(ys), max(ys)) == (4.0, 8.0)


def test_a_per_frame_key_is_what_the_overlay_draws():
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    tab.doc.add_frame()
    entry = tab.doc.add_slice((0, 0, 4, 4))
    tab.doc.set_slice_key(
        entry.uid, tab.doc.anim.frames[1].uid, key=SliceKey(bounds=(6, 6, 10, 10))
    )
    tab.doc.set_current_frame(1)
    assert _draw(_state(tab), tab).rects == [((6.0, 6.0), (10.0, 10.0))]
    tab.doc.set_current_frame(0)
    assert _draw(_state(tab), tab).rects == [((0.0, 0.0), (4.0, 4.0))]


def test_the_overlay_is_on_with_the_tool_and_optional_otherwise():
    state = inker_state.InkerState(tool="slice")
    assert inker_canvas.slices_visible(state)
    state.tool = "brush"
    assert not inker_canvas.slices_visible(state)
    state.show_slices = True
    assert inker_canvas.slices_visible(state)


# --- gestures -----------------------------------------------------------------


def test_dragging_a_slice_is_exactly_one_step(monkeypatch):
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    entry = tab.doc.add_slice((2, 2, 26, 10))
    state = _state(tab)
    state.slice_uid = entry.uid
    head = tab.doc.history.head

    _at(monkeypatch, 14.0, 6.0)
    inker_canvas._press(None, state, tab, (14.0, 6.0), ORIGIN)
    assert state.drag_kind == "slice-move"
    # Four frames of drag, as a real one is: each moves the live object so the
    # overlay follows the cursor, and none of them pushes anything.
    for step in range(1, 5):
        inker_canvas._drag(state, tab, (14.0 + step, 6.0 + step))
    assert tab.doc.history.head == head, "a drag commits nothing"
    inker_canvas._release(None, state, tab, (18.0, 10.0))

    assert len(tab.doc.history) == 2
    assert tab.doc.slices[0].bounds == (6, 6, 30, 14)
    tab.doc.undo()
    assert tab.doc.slices[0].bounds == (2, 2, 26, 10)


def test_a_slice_dragged_past_the_edge_is_clamped_to_the_canvas(monkeypatch):
    """A slice names a region *of the canvas*, so it stops at it -- the same
    clamp ``add_slice`` applies, in the one place a user can reach it."""
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    entry = tab.doc.add_slice((2, 2, 26, 10))
    state = _state(tab)
    state.slice_uid = entry.uid

    _at(monkeypatch, 14.0, 6.0)
    inker_canvas._press(None, state, tab, (14.0, 6.0), ORIGIN)
    inker_canvas._drag(state, tab, (114.0, 106.0))
    inker_canvas._release(None, state, tab, (114.0, 106.0))
    x0, y0, x1, y1 = tab.doc.slices[0].bounds
    assert (x1, y1) == (32, 16)
    assert 0 <= x0 < x1 and 0 <= y0 < y1


def test_dragging_a_corner_resizes_from_that_corner(monkeypatch):
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    entry = tab.doc.add_slice((2, 2, 30, 14))
    state = _state(tab)
    state.slice_uid = entry.uid

    _at(monkeypatch, 2.0, 2.0)
    inker_canvas._press(None, state, tab, (2.0, 2.0), ORIGIN)
    assert state.drag_kind == "slice-resize"
    inker_canvas._drag(state, tab, (6.0, 5.0))
    inker_canvas._release(None, state, tab, (6.0, 5.0))
    assert tab.doc.slices[0].bounds == (6, 5, 30, 14)
    assert len(tab.doc.history) == 2


def test_dragging_on_empty_canvas_adds_one_slice(monkeypatch):
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    state = _state(tab)

    _at(monkeypatch, 2.0, 3.0)
    inker_canvas._press(None, state, tab, (2.0, 3.0), ORIGIN)
    assert state.drag_kind == "slice-new"
    inker_canvas._drag(state, tab, (10.0, 7.0))
    assert tab.doc.slices == [], "nothing exists until the release"
    inker_canvas._release(None, state, tab, (10.0, 7.0))

    assert len(tab.doc.slices) == 1
    assert tab.doc.slices[0].bounds == (2, 3, 10, 7)
    assert state.slice_uid == tab.doc.slices[0].uid
    assert len(tab.doc.history) == 1


def test_a_click_with_no_drag_adds_nothing(monkeypatch):
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    state = _state(tab)
    _at(monkeypatch, 5.0, 5.0)
    inker_canvas._press(None, state, tab, (5.0, 5.0), ORIGIN)
    inker_canvas._release(None, state, tab, (5.0, 5.0))
    assert tab.doc.slices == []
    assert tab.doc.history.head == 0


def test_dragging_the_pivot_moves_it_within_the_slice(monkeypatch):
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    entry = tab.doc.add_slice((2, 2, 30, 14), pivot=(10.0, 6.0))
    state = _state(tab)
    state.slice_uid = entry.uid

    _at(monkeypatch, 12.0, 8.0)  # where the pivot is, in screen space
    inker_canvas._press(None, state, tab, (12.0, 8.0), ORIGIN)
    assert state.drag_kind == "slice-pivot"
    inker_canvas._drag(state, tab, (10.0, 6.0))
    inker_canvas._release(None, state, tab, (10.0, 6.0))
    assert tab.doc.slices[0].pivot == (8.0, 4.0)
    assert tab.doc.slices[0].bounds == (2, 2, 30, 14)
    assert len(tab.doc.history) == 2


def test_a_press_that_moves_nothing_pushes_nothing(monkeypatch):
    """The rule every op in the engine follows, exercised through the gesture
    that is most likely to break it: a click on a slice selects it, and
    selecting is not an edit."""
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    tab.doc.add_slice((2, 2, 30, 14))
    state = _state(tab)
    head = tab.doc.history.head

    _at(monkeypatch, 16.0, 8.0)
    inker_canvas._press(None, state, tab, (16.0, 8.0), ORIGIN)
    inker_canvas._release(None, state, tab, (16.0, 8.0))
    assert tab.doc.history.head == head


def test_the_slice_tool_never_falls_through_to_paint(monkeypatch):
    """The failure a missing early-out looks like: a dab left on the layer the
    first time somebody drags a slice out."""
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    state = _state(tab)
    before = tab.doc.stack.active.pixels.copy()
    _at(monkeypatch, 5.0, 5.0)
    inker_canvas._press(None, state, tab, (5.0, 5.0), ORIGIN)
    inker_canvas._drag(state, tab, (9.0, 9.0))
    inker_canvas._release(None, state, tab, (9.0, 9.0))

    import numpy as np

    assert np.array_equal(tab.doc.stack.active.pixels, before)


def test_a_slice_drag_is_refused_while_a_transform_is_open(monkeypatch):
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    tab.doc.add_slice((2, 2, 30, 14))
    state = _state(tab)
    state.transforming = True
    _at(monkeypatch, 16.0, 8.0)
    inker_canvas._press(None, state, tab, (16.0, 8.0), ORIGIN)
    assert state.drag_kind == ""


def test_the_shortcut_letter_is_the_one_the_toolbox_shows():
    from warlock.studio import inker_mode

    letters = {tool: key for key, tool in inker_mode.TOOL_KEYS.items()}
    assert letters["slice"] == "c"
    assert dict(
        (tool, shortcut.lower()) for tool, _label, shortcut in inker_state.TOOLS
    )["slice"] == "c"
