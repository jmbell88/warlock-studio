"""The polygonal lasso, and the multi-click gesture underneath it (C4).

Three properties, and each of them has a failure that looks like a working
editor rather than a crash.

**A gesture is not a drag.** ``_input`` refuses a press while a gesture owns the
mouse -- the C12d guard -- so a tool whose vertices are *clicked* has to leave
``drag_kind`` empty or its own second click is eaten by its own first. That is
asserted by driving the real dispatch frame by frame, which is the only place
the guard exists at all.

**One gesture is one Ctrl+Z**, however many clicks it took, and the polygon that
lands is the one the freehand lasso would have landed from the same vertices --
which is what the shared ``polygon_select`` is for.

**A half-drawn polygon is dropped, never left behind.** Escape, a tool switch, a
tab switch and the tab going busy each have to cancel it; vertices that outlive
their tool would be committed by the next tool's closing click.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import inker, inker_mode, inker_state
from warlock.studio.panes import inker_canvas

SIZE = (32, 32)
ORIGIN = (0.0, 0.0)
#: The pane ``_input`` is driven in. Large enough that these documents sit
#: unclamped at ``pan=(0, 0)``, so the gestures under test are unaffected
#: by the canvas pan bound.
REGION = (400.0, 300.0)
TRIANGLE = [(4.0, 4.0), (24.0, 6.0), (10.0, 26.0)]


def _tab(**view):
    return inker_state.InkerDoc(
        doc=inker.Document.blank(*SIZE), views=[inker_state.PaintView(**view)]
    )


class _Mouse:
    """imgui's mouse, as much of it as the press dispatch reads.

    Its own copy rather than the one in ``test_canvas_input``: this gesture is
    the first thing in the pane to care about ``is_mouse_double_clicked``.
    """

    def __init__(self) -> None:
        self.at = (0.0, 0.0)
        self.down = {0: False, 1: False, 2: False}
        self.clicked = {0: False, 1: False, 2: False}
        self.double = {0: False, 1: False, 2: False}
        self.shift = False
        self.alt = False

    def module(self) -> SimpleNamespace:
        return SimpleNamespace(
            get_io=lambda: SimpleNamespace(
                mouse_wheel=0.0,
                mouse_wheel_h=0.0,
                key_shift=self.shift,
                key_ctrl=False,
                key_alt=self.alt,
                delta_time=1.0 / 60.0,
            ),
            get_mouse_pos=lambda: SimpleNamespace(x=self.at[0], y=self.at[1]),
            is_mouse_clicked=lambda button: self.clicked[button],
            is_mouse_double_clicked=lambda button: self.double[button],
            is_mouse_down=lambda button: self.down[button],
            is_mouse_dragging=lambda button: False,
        )


@pytest.fixture
def scene(monkeypatch, patch_canvas):
    """The poly lasso in hand, at an identity view so screen == image.

    ``click`` drives the real ``_input`` -- press frame then release frame --
    rather than calling ``_press``, because half of what this gesture has to get
    right lives in the dispatcher above it: the press guard, and the grid snap
    that is applied to the point *before* the tool ever sees it.
    """
    mouse = _Mouse()
    patch_canvas("imgui", mouse.module())
    tab = _tab(zoom=1.0, pan=(0.0, 0.0), fitted=True)
    state = inker_state.InkerState(tool="lasso_poly")
    state.add(tab)

    def frame(at, *, pressed=None, down=(), double=False):
        mouse.at = (float(at[0]), float(at[1]))
        mouse.clicked = {0: False, 1: False, 2: False}
        mouse.double = {0: False, 1: False, 2: False}
        if pressed is not None:
            mouse.clicked[pressed] = True
            mouse.double[pressed] = double
        mouse.down = {b: b in down for b in (0, 1, 2)}
        inker_canvas._input(None, state, tab, ORIGIN, REGION, active=True, hovered=True)

    def click(point, *, button: int = 0, double: bool = False, shift=False, alt=False):
        mouse.shift, mouse.alt = shift, alt
        frame(point, pressed=button, down=(button,), double=double)
        frame(point)

    return state, tab, click, mouse


# --- the extracted polygon -> mask helper ------------------------------------


def test_three_vertices_become_a_selection_and_one_undo_step():
    doc = inker.Document.blank(*SIZE)
    assert inker_mode.polygon_select(doc, TRIANGLE) is True
    assert doc.mask is not None and doc.mask.contains((12, 10))
    assert not doc.mask.contains((30, 30))
    assert len(doc.history) == 1


def test_fewer_than_three_vertices_select_nothing_and_push_nothing():
    """Two points are a line: the rasteriser returns an empty mask and
    ``select`` would move the history head for a selection of nothing."""
    doc = inker.Document.blank(*SIZE)
    for points in ([], [(4.0, 4.0)], [(4.0, 4.0), (20.0, 20.0)]):
        assert inker_mode.polygon_select(doc, points) is False
    assert doc.mask is None
    assert len(doc.history) == 0


def test_the_combine_op_reaches_the_document():
    doc = inker.Document.blank(*SIZE)
    doc.select(inker.SelectionMask.from_rect(SIZE, (0, 0, 32, 32)))
    inker_mode.polygon_select(doc, TRIANGLE, "subtract")
    assert doc.mask is not None
    assert not doc.mask.contains((12, 10))  # the triangle was cut out
    assert doc.mask.contains((30, 30))


def test_the_freehand_lasso_lands_through_the_same_helper(monkeypatch):
    """The point of extracting it: the two tools differ in how the vertices are
    collected and in nothing else, so a change to the landing cannot reach one
    of them only."""
    seen: list[tuple] = []
    monkeypatch.setattr(
        inker_mode,
        "polygon_select",
        lambda doc, points, op="replace": seen.append((list(points), op)) or True,
    )
    state = inker_state.InkerState(tool="lasso")
    tab = _tab(zoom=1.0)
    state.drag_kind = "lasso"
    state.lasso = list(TRIANGLE)
    state.combine = "add"
    inker_canvas._release(None, state, tab, TRIANGLE[-1])
    assert seen == [(TRIANGLE, "add")]


# --- close detection (pure) ---------------------------------------------------


def test_a_polygon_of_two_vertices_cannot_be_closed():
    assert inker_canvas.closes_gesture([(4.0, 4.0), (8.0, 4.0)], (4.0, 4.0), 1.0, 7.0) is False


def test_a_click_on_the_first_vertex_closes_the_polygon():
    assert inker_canvas.closes_gesture(TRIANGLE, (4.0, 4.0), 1.0, 7.0) is True
    assert inker_canvas.closes_gesture(TRIANGLE, (4.0, 14.0), 1.0, 7.0) is False


def test_the_radius_is_screen_pixels_and_not_image_pixels():
    """A fixed image-space radius would be unhittable at 100x zoom and cover
    half the canvas at 5%. Four image pixels from the first vertex: inside the
    ring at zoom 1, far outside it at zoom 8, comfortably inside at zoom 0.5."""
    near = (8.0, 4.0)
    assert inker_canvas.closes_gesture(TRIANGLE, near, 1.0, 7.0) is True
    assert inker_canvas.closes_gesture(TRIANGLE, near, 8.0, 7.0) is False
    assert inker_canvas.closes_gesture(TRIANGLE, near, 0.5, 7.0) is True


# --- the click sequence -------------------------------------------------------


def test_each_click_places_a_vertex_and_starts_no_drag(scene):
    """The load-bearing assertion: this tool must never take the freehand
    ``drag_kind="lasso"`` path, and an empty ``drag_kind`` is also what keeps
    the next click out of the press guard."""
    state, tab, click, _mouse = scene
    for point in TRIANGLE:
        click(point)
        assert state.drag_kind == ""
    assert state.gesture_pts == TRIANGLE
    assert state.lasso == []
    assert tab.doc.mask is None  # nothing lands until it is closed


def test_a_click_back_on_the_first_vertex_commits_one_step(scene):
    state, tab, click, _mouse = scene
    for point in TRIANGLE:
        click(point)
    click((4.0, 4.0))
    assert state.gesture_pts == []
    assert len(tab.doc.history) == 1
    expected = inker.SelectionMask.from_polygon(SIZE, TRIANGLE)
    assert tab.doc.mask is not None
    assert (tab.doc.mask.mask == expected.mask).all()


def test_a_double_click_commits_without_placing_its_own_vertex(scene):
    """The second press of a double-click lands on top of the first, so
    appending it would put a zero-length edge in the polygon."""
    state, tab, click, _mouse = scene
    for point in TRIANGLE:
        click(point)
    click((10.0, 26.0), double=True)
    assert state.gesture_pts == []
    expected = inker.SelectionMask.from_polygon(SIZE, TRIANGLE)
    assert tab.doc.mask is not None
    assert (tab.doc.mask.mask == expected.mask).all()


def test_the_combine_op_is_captured_at_the_first_click(scene):
    """Letting go of Shift halfway through must not turn an add into a replace
    that throws the selection away."""
    state, tab, click, _mouse = scene
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (26, 26, 32, 32)))
    click(TRIANGLE[0], shift=True)
    assert state.gesture_combine == "add"
    click(TRIANGLE[1])
    click(TRIANGLE[2])
    click(TRIANGLE[0])
    assert tab.doc.mask is not None
    assert tab.doc.mask.contains((12, 10)) and tab.doc.mask.contains((28, 28))


def test_a_polygon_of_two_vertices_closes_nothing_and_keeps_the_selection(scene):
    """Cancelling is the honest answer: deselecting would throw away a
    selection the user never asked to lose."""
    state, tab, click, _mouse = scene
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (2, 2, 8, 8)))
    depth = len(tab.doc.history)
    click((4.0, 4.0))
    click((20.0, 4.0), double=True)
    assert state.gesture_pts == []
    assert len(tab.doc.history) == depth
    assert tab.doc.mask is not None and tab.doc.mask.contains((4, 4))


def test_the_vertices_snap_to_the_grid_when_snapping_is_on(scene):
    """Unlike the freehand lasso, which is deliberately exempt: a clicked corner
    is a placed point, and placing points on the grid is what the setting is
    for. The snap happens above the tool, in ``_input``."""
    state, _tab_, click, _mouse = scene
    state.grid = True
    state.grid_snap = True
    state.grid_size = 8
    click((9.0, 15.0))
    assert state.gesture_pts == [(8.0, 16.0)]


# --- the press guard (C12d) ---------------------------------------------------


def test_the_press_guard_does_not_eat_the_second_click(monkeypatch, patch_canvas):
    """Driven through the real ``_input``, one frame per call: the guard refuses
    a press while a gesture owns the mouse, and a click sequence is not a held
    button -- so every click must reach ``_press``."""
    mouse = _Mouse()
    patch_canvas("imgui", mouse.module())
    tab = _tab(zoom=1.0, pan=(0.0, 0.0), fitted=True)
    state = inker_state.InkerState(tool="lasso_poly")
    state.add(tab)

    def frame(at, *, click=None, down=()):
        mouse.at = (float(at[0]), float(at[1]))
        mouse.clicked = {0: False, 1: False, 2: False}
        if click is not None:
            mouse.clicked[click] = True
        mouse.down = {b: b in down for b in (0, 1, 2)}
        inker_canvas._input(None, state, tab, ORIGIN, REGION, active=True, hovered=True)

    for point in TRIANGLE:
        frame(point, click=0, down=(0,))  # press
        frame(point, down=(0,))  # held
        frame(point, down=())  # released
    assert state.gesture_pts == TRIANGLE

    frame((4.0, 4.0), click=0, down=(0,))
    assert state.gesture_pts == []
    assert tab.doc.mask is not None
    assert len(tab.doc.history) == 1


def test_a_right_click_does_not_place_a_vertex(scene):
    """The right button is inert on every selection tool, and inert means the
    open polygon is neither extended nor dropped."""
    state, tab, click, _mouse = scene
    click(TRIANGLE[0])
    click((20.0, 20.0), button=1)
    assert state.gesture_pts == [TRIANGLE[0]]


# --- cancelling ---------------------------------------------------------------


def test_a_tool_switch_drops_the_polygon(scene):
    state, _tab_, click, _mouse = scene
    click(TRIANGLE[0])
    click(TRIANGLE[1])
    state.set_tool("brush")
    assert state.gesture_pts == []


def test_a_tab_switch_drops_the_polygon(scene):
    state, _tab_, click, _mouse = scene
    other = _tab(zoom=1.0)
    state.add(other)
    click(TRIANGLE[0])
    click(TRIANGLE[1])
    state.activate(state.docs[0].uid)
    assert state.gesture_pts == []


def test_the_tab_going_busy_drops_the_polygon(monkeypatch, patch_canvas):
    """Its next click cannot be delivered while ``_input`` returns early, so
    leaving the vertices up would finish the polygon against whatever the
    document looked like whenever the tab came back."""
    mouse = _Mouse()
    patch_canvas("imgui", mouse.module())
    tab = _tab(zoom=1.0, pan=(0.0, 0.0), fitted=True)
    state = inker_state.InkerState(tool="lasso_poly")
    state.add(tab)
    state.gesture_pts = list(TRIANGLE)

    tab.saving = True
    inker_canvas._input(None, state, tab, ORIGIN, REGION, active=True, hovered=True)
    assert state.gesture_pts == []


def test_escape_drops_the_polygon_and_keeps_the_selection(monkeypatch):
    """Esc means "drop the one thing I am doing". With a polygon open that is
    the polygon, not the selection underneath it."""
    import pygame

    tab = _tab(zoom=1.0)
    state = inker_state.InkerState(tool="lasso_poly")
    state.add(tab)
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (2, 2, 8, 8)))
    state.gesture_pts = list(TRIANGLE)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0)

    assert inker_mode.handle_key(ctx, event) is True
    assert state.gesture_pts == []
    assert tab.doc.mask is not None
    # And a second press goes on to do what Esc always did.
    assert inker_mode.handle_key(ctx, event) is True
    assert tab.doc.mask is None


def test_enter_commits_the_polygon_rather_than_starting_playback(monkeypatch):
    import pygame

    tab = _tab(zoom=1.0)
    state = inker_state.InkerState(tool="lasso_poly")
    state.add(tab)
    state.gesture_pts = list(TRIANGLE)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0)

    assert inker_mode.handle_key(ctx, event) is True
    assert state.gesture_pts == []
    assert tab.doc.mask is not None and tab.doc.mask.contains((12, 10))
    assert tab.playing is False


def test_committing_a_busy_tab_cancels_instead_of_selecting():
    tab = _tab(zoom=1.0)
    state = inker_state.InkerState(tool="lasso_poly")
    state.add(tab)
    state.gesture_pts = list(TRIANGLE)
    tab.saving = True
    assert inker_mode.commit_gesture(state, tab) is False
    assert state.gesture_pts == []
    assert tab.doc.mask is None


# --- the overlay --------------------------------------------------------------


class _Lines:
    """A draw list that only takes notes; ``test_slice_overlay``'s idiom."""

    def __init__(self) -> None:
        self.lines: list[tuple] = []
        self.circles: list[tuple] = []

    def add_line(self, a, b, colour, thickness=1.0) -> None:
        self.lines.append((a, b))

    def add_circle(self, centre, radius, colour) -> None:
        self.circles.append((centre, radius))


@pytest.fixture
def overlay(monkeypatch, patch_canvas):
    patch_canvas("_u32", lambda colour, alpha=1.0: 0)
    mouse = _Mouse()
    patch_canvas("imgui", mouse.module())
    return mouse


def test_the_open_polygon_is_drawn_with_a_rubber_band(overlay):
    overlay.at = (20.0, 20.0)
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    state = inker_state.InkerState(tool="lasso_poly")
    state.gesture_pts = list(TRIANGLE)
    lines = _Lines()
    inker_canvas._gesture_preview(state, tab, lines, ORIGIN)
    # Two placed edges, the band from the last vertex to the cursor, and the
    # edge a commit would close with.
    assert lines.lines[:2] == [(TRIANGLE[0], TRIANGLE[1]), (TRIANGLE[1], TRIANGLE[2])]
    assert lines.lines[2] == (TRIANGLE[2], (20.0, 20.0))
    assert lines.lines[3] == ((20.0, 20.0), TRIANGLE[0])
    assert lines.circles[0][0] == TRIANGLE[0]


def test_the_polygon_follows_a_quarter_turn_of_the_page(overlay):
    """Through ``to_screen`` like every other overlay: computing
    ``origin + x * zoom`` is right at rotation 0 and a quarter turn out
    everywhere else."""
    overlay.at = (-4.0, 4.0)
    tab = _tab(zoom=1.0, pan=(0.0, 0.0), rotation=90)
    state = inker_state.InkerState(tool="lasso_poly")
    state.gesture_pts = [(4.0, 4.0), (24.0, 6.0)]
    lines = _Lines()
    inker_canvas._gesture_preview(state, tab, lines, ORIGIN)
    # (x, y) -> (-y, x) is one clockwise quarter turn on a downward-y screen.
    assert lines.lines[0] == ((-4.0, 4.0), (-6.0, 24.0))


def test_nothing_is_drawn_with_no_gesture_open(overlay):
    tab = _tab(zoom=1.0)
    state = inker_state.InkerState(tool="lasso_poly")
    lines = _Lines()
    inker_canvas._gesture_preview(state, tab, lines, ORIGIN)
    assert lines.lines == [] and lines.circles == []


# --- the toolbox --------------------------------------------------------------


def test_the_tool_is_listed_with_its_own_letter():
    letters = {tool: key for key, tool in inker_mode.TOOL_KEYS.items()}
    assert letters["lasso_poly"] == "d"
    rows = {tool: shortcut.lower() for tool, _label, shortcut in inker_state.TOOLS}
    assert rows["lasso_poly"] == "d"
    assert "lasso_poly" in inker_state.SELECT_TOOLS
