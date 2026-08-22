"""The clicked shape tools: polyline, polygon and curve (Q-c).

Three claims, and each has a failure mode that looks like a working editor.

**The curve goes through the points you clicked.** That is what makes an
interpolating spline the only honest curve for a click-per-vertex gesture, and
it is asserted point by point rather than by eye. The parameterisation is
centripetal, and the test below shows what the uniform kind does instead: a
visible overshoot away from a close-spaced pair, which on a drawing reads as the
tool ignoring where you clicked.

**A path shape is a shape.** It rasterises through the same ``PaintOps``
machinery the dragged line, rectangle and ellipse go through, so it wraps on a
tiled document, clips to the selection, obeys the content lock and lands as
**one undo step** however many clicks it took. None of that is reimplemented, so
all of it is pinned here as composition rather than as behaviour of its own.

**The gesture is C4's, with a different landing.** The vertices are collected by
the same click cycle the poly lasso uses -- the same press guard, the same
double-click, the same cancel-on-busy -- and what changes at the end is that
these tools paint instead of selecting. The click sequences are driven through
the real ``_input`` for the reason C4's are: half of what has to be right lives
in the dispatcher above the tool.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import inker, inker_mode, inker_state
from warlock.studio.inker import _doc_paint
from warlock.studio.panes import inker_canvas, inker_tools

SIZE = (32, 32)
ORIGIN = (0.0, 0.0)
RED = (255, 0, 0, 255)
#: An open path with a bend in it, well inside the canvas.
BEND = [(4.0, 6.0), (16.0, 24.0), (28.0, 6.0)]


def _doc(size=SIZE) -> inker.Document:
    return inker.Document.blank(*size)


def _alpha(doc: inker.Document) -> np.ndarray:
    return doc.stack.active.pixels[..., 3]


def _painted(doc: inker.Document) -> int:
    return int((_alpha(doc) > 0).sum())


# --- the curve rasteriser (pure) ---------------------------------------------


def test_the_curve_passes_through_every_point_that_was_clicked():
    """The property the tool is sold on. A Bezier's handles are points the
    drawing never reaches, and a click-per-vertex gesture has nowhere to put
    them, so an interpolating spline is the only honest curve here."""
    path = _doc_paint.catmull_rom(BEND)
    for point in BEND:
        assert any(math.dist(point, sample) < 1e-9 for sample in path), point
    assert path[0] == BEND[0] and path[-1] == BEND[-1]


def test_too_few_points_to_bend_come_back_unchanged():
    """Two points are a straight line and one is a dot; neither has curvature
    to find, and inventing samples between two of them would be arithmetic that
    cannot change the answer."""
    assert _doc_paint.catmull_rom([]) == []
    assert _doc_paint.catmull_rom([(1.0, 1.0)]) == [(1.0, 1.0)]
    assert _doc_paint.catmull_rom([(1.0, 1.0), (5.0, 9.0)]) == [(1.0, 1.0), (5.0, 9.0)]


def test_a_repeated_point_is_collapsed_rather_than_dividing_by_zero():
    """A double-click's second press lands on top of the first, so a repeated
    vertex is a thing that happens rather than a thing that should not."""
    doubled = [(1.0, 1.0), (1.0, 1.0), (5.0, 9.0), (5.0, 9.0)]
    assert _doc_paint.catmull_rom(doubled) == [(1.0, 1.0), (5.0, 9.0)]
    assert _doc_paint.catmull_rom([(2.0, 2.0)] * 4) == [(2.0, 2.0)]


def test_collinear_points_stay_on_their_line():
    """Every sample is an affine combination of the four control points, so
    three points on a line can only produce a line -- which is what makes the
    curve tool usable for a straight run in the middle of a shape."""
    path = _doc_paint.catmull_rom([(0.0, 0.0), (2.0, 0.0), (30.0, 0.0)])
    assert max(abs(y) for _x, y in path) == 0.0


def _uniform(points, steps: int = 32):
    """Uniform Catmull-Rom, for the comparison below and nowhere else.

    The matrix form, which is the textbook one and the one this module would
    have had if the knots were spaced evenly instead of by ``dist ** 0.5``.
    """

    def at(p0, p1, p2, p3, t):
        return tuple(
            0.5
            * (
                2 * a1
                + (-a0 + a2) * t
                + (2 * a0 - 5 * a1 + 4 * a2 - a3) * t * t
                + (-a0 + 3 * a1 - 3 * a2 + a3) * t**3
            )
            for a0, a1, a2, a3 in zip(p0, p1, p2, p3, strict=True)
        )

    ext = [
        _doc_paint._mirrored(points[1], points[0]),
        *points,
        _doc_paint._mirrored(points[-2], points[-1]),
    ]
    out = []
    for index in range(len(points) - 1):
        p0, p1, p2, p3 = ext[index : index + 4]
        out += [at(p0, p1, p2, p3, s / steps) for s in range(steps + 1)]
    return out


def test_the_parameterisation_is_centripetal_and_does_not_overshoot():
    """A clicked path is unevenly spaced by nature -- three quick clicks in a
    corner and one across the canvas -- and that is exactly where uniform
    Catmull-Rom loops away from its own control points. Here the first three
    points sit on ``y == 0`` and the fourth is 30 pixels up: the uniform curve
    dives more than two pixels *below* the line before climbing, which on a
    drawing is the tool visibly ignoring where you clicked.
    """
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (2.0, 30.0)]
    ours = min(y for _x, y in _doc_paint.catmull_rom(points))
    theirs = min(y for _x, y in _uniform(points))
    assert theirs < -2.0
    assert ours > -0.5


def test_a_long_segment_is_sampled_more_finely_than_a_short_one():
    """A per-segment count that looks smooth on a 32px sprite is a visible
    chain of chords across a 2000px canvas."""
    short = _doc_paint.catmull_rom([(0.0, 0.0), (4.0, 1.0), (8.0, 0.0)])
    long = _doc_paint.catmull_rom([(0.0, 0.0), (400.0, 100.0), (800.0, 0.0)])
    assert len(long) > len(short)
    # And capped, so a curve corner to corner on a huge canvas cannot cost
    # thousands of points for a smoothness no pixel can show.
    assert len(long) <= 2 * _doc_paint.CURVE_MAX_SAMPLES + 1


def test_the_sample_spacing_is_the_declared_step():
    path = _doc_paint.catmull_rom([(0.0, 0.0), (20.0, 0.0), (40.0, 0.0)])
    gaps = [math.dist(a, b) for a, b in zip(path, path[1:], strict=False)]
    assert max(gaps) <= _doc_paint.CURVE_STEP + 1e-9


def test_the_spans_are_the_curve_cut_at_its_control_points():
    """``curve_spans`` is what makes the preview's memo possible, so the two
    must be one arithmetic rather than two: concatenating the spans *is*
    ``catmull_rom``, bit for bit."""
    pts, spans = _doc_paint.curve_spans(BEND)
    assert pts == [(float(x), float(y)) for x, y in BEND]
    assert len(spans) == len(pts) - 1
    flat = [pts[0]]
    for span in spans:
        flat.extend(span)
    assert flat == _doc_paint.catmull_rom(BEND)
    # Each span ends on the control point it runs to -- the property a caller
    # splitting the curve by segment index relies on.
    for index, span in enumerate(spans):
        assert math.dist(span[-1], pts[index + 1]) < 1e-9


def test_a_segment_s_knots_are_computed_once_rather_than_once_per_sample(monkeypatch):
    """The cost fix, asserted as the contract rather than as a timing: the knot
    vector is a property of the *segment*, so ``_curve_point`` is handed it.
    Three square roots per segment, not three per sample -- which was 192 of
    them across one 64-sample segment."""
    calls: list[tuple] = []
    real = _doc_paint._knot
    monkeypatch.setattr(
        _doc_paint, "_knot", lambda a, b: (calls.append((a, b)), real(a, b))[1]
    )
    _pts, spans = _doc_paint.curve_spans([(0.0, 0.0), (60.0, 40.0), (120.0, 0.0)])
    assert len(calls) == 3 * len(spans)
    assert sum(len(span) for span in spans) > 10 * len(spans)  # far more samples


def test_the_collapse_is_the_one_function_both_callers_count_with():
    """A caller splitting a curve by segment index has to be counting the same
    control points ``curve_spans`` counted."""
    assert _doc_paint.curve_points([(1, 1), (1, 1), (5, 9)]) == [(1.0, 1.0), (5.0, 9.0)]
    assert _doc_paint.curve_points([]) == []
    raw = [(1, 1), (1, 1), (5, 9), (2, 7)]
    pts, _spans = _doc_paint.curve_spans(raw)
    assert pts == _doc_paint.curve_points(raw)


# --- the engine: shape_path --------------------------------------------------


def test_the_kinds_are_in_shapes_and_named_as_a_group():
    assert set(_doc_paint.PATH_SHAPES) == {"polyline", "polygon", "curve"}
    assert set(_doc_paint.PATH_SHAPES) <= set(inker.SHAPES)
    assert inker.PATH_SHAPES == _doc_paint.PATH_SHAPES


def test_an_unknown_path_shape_is_a_programming_error():
    with pytest.raises(ValueError):
        _doc().shape_path("hexagon", BEND, RED, 1)


def test_a_two_point_polyline_is_exactly_the_line_tool_s_line():
    """The two entry points must not be able to disagree about what a polyline
    of two points is, so ``shape`` routes rather than reimplements -- and this
    is byte equality, not agreement to within a pixel."""
    line, poly = _doc(), _doc()
    line.shape("line", (4, 4), (24, 20), RED, 5)
    poly.shape_path("polyline", [(4, 4), (24, 20)], RED, 5)
    assert np.array_equal(line.stack.active.pixels, poly.stack.active.pixels)
    routed = _doc()
    routed.shape("polyline", (4, 4), (24, 20), RED, 5)
    assert np.array_equal(line.stack.active.pixels, routed.stack.active.pixels)


def test_a_path_shape_touches_only_its_own_bounding_box():
    for kind in _doc_paint.PATH_SHAPES:
        doc = _doc()
        assert doc.shape_path(kind, BEND, RED, 1) is True, kind
        alpha = _alpha(doc)
        assert int(alpha[:5, :].max()) == 0, kind  # above the topmost point
        assert int(alpha[:, 29:].max()) == 0, kind  # right of the rightmost


def test_however_many_points_it_is_one_undo_step():
    """The whole reason the gesture collects vertices instead of painting them:
    twenty clicks are one Ctrl+Z."""
    doc = _doc()
    many = [(2.0 + i, 4.0 + (i % 2) * 8.0) for i in range(20)]
    assert doc.shape_path("polyline", many, RED, 1) is True
    assert len(doc.history) == 1


def test_a_filled_polygon_is_solid_and_an_outlined_one_is_not():
    filled, hollow = _doc(), _doc()
    filled.shape_path("polygon", BEND, RED, 1, filled=True)
    hollow.shape_path("polygon", BEND, RED, 1, filled=False)
    assert _painted(filled) > _painted(hollow)
    # A point inside the triangle but off its edges.
    assert int(_alpha(filled)[12, 16]) > 0
    assert int(_alpha(hollow)[12, 16]) == 0


def test_a_polygon_is_closed_and_a_polyline_is_not():
    """The whole of the difference between the two, and it is visible on the
    edge from the last point back to the first."""
    closed, open_ = _doc(), _doc()
    closed.shape_path("polygon", BEND, RED, 1)
    open_.shape_path("polyline", BEND, RED, 1)
    # Midway along the closing edge, which only the polygon draws.
    assert int(_alpha(closed)[6, 16]) > 0
    assert int(_alpha(open_)[6, 16]) == 0


def test_filled_means_nothing_to_an_open_path():
    """An open path has no inside. Rather than invent one -- Aseprite's contour
    tool closes the shape to fill it, which is a *different shape* from the one
    previewed -- the flag is ignored, and that is pinned as byte equality."""
    for kind in ("polyline", "curve"):
        plain, flagged = _doc(), _doc()
        plain.shape_path(kind, BEND, RED, 3, filled=False)
        flagged.shape_path(kind, BEND, RED, 3, filled=True)
        assert np.array_equal(plain.stack.active.pixels, flagged.stack.active.pixels), kind


def test_a_path_with_nothing_to_draw_writes_nothing_and_pushes_nothing():
    """One point is a dot the brush draws better, and the coverage of a
    zero-length line is nothing at all."""
    doc = _doc()
    for points in ([], [(4.0, 4.0)], [(4.0, 4.0)] * 5):
        assert doc.shape_path("polyline", points, RED, 3) is False
    assert len(doc.history) == 0
    assert _painted(doc) == 0


def test_the_curve_paints_off_the_chord_between_its_points():
    """The difference between the curve and the polyline, at the pixel level:
    the curve bulges away from the straight run through the same clicks."""
    curve, poly = _doc(), _doc()
    points = [(4.0, 16.0), (16.0, 6.0), (28.0, 16.0)]
    curve.shape_path("curve", points, RED, 1)
    poly.shape_path("polyline", points, RED, 1)
    assert not np.array_equal(curve.stack.active.pixels, poly.stack.active.pixels)
    # The curve passes through every clicked point all the same.
    assert int(_alpha(curve)[6, 16]) > 0


# --- composition: the doors every other shape comes through ------------------


def test_a_polyline_at_the_edge_wraps_its_overhang():
    """``test_tiled_paint``'s claim for the dragged shapes, for the clicked
    ones: the plane is grown to hold the whole path and folded home once, so a
    stroke that leaves one side arrives on the other."""
    doc = _doc()
    doc.shape_path("polyline", [(0.0, 2.0), (0.0, 20.0)], RED, 7, wrap="x")
    alpha = _alpha(doc)
    assert int(alpha[10, -1]) > 0
    assert int(alpha[10, 0]) > 0


def test_the_same_polyline_untiled_loses_its_overhang():
    doc = _doc()
    doc.shape_path("polyline", [(0.0, 2.0), (0.0, 20.0)], RED, 7, wrap="off")
    assert int(_alpha(doc)[10, -1]) == 0


def test_an_interior_path_is_identical_however_the_flag_is_set():
    """The off-path parity the tiled work is held to: with nothing crossing a
    seam the plane is the canvas at the origin and the fold is the identity, so
    this is byte equality rather than approximate agreement."""
    plain, tiled = _doc(), _doc()
    plain.shape_path("curve", BEND, RED, 3)
    tiled.shape_path("curve", BEND, RED, 3, wrap="both")
    assert np.array_equal(plain.stack.active.pixels, tiled.stack.active.pixels)


def test_a_path_shape_crossing_a_seam_stays_one_connected_stroke():
    """One plane for the whole path rather than one per segment: folding
    segment by segment would rasterise each into its own plane and lose the
    joins between them."""
    doc = _doc()
    doc.shape_path("polyline", [(-6.0, 8.0), (8.0, 8.0), (8.0, 20.0)], RED, 3, wrap="x")
    alpha = _alpha(doc)
    assert int(alpha[8, 28]) > 0  # the part that wrapped
    assert int(alpha[8, 4]) > 0  # and the part that did not


def test_the_selection_clips_a_path_shape():
    doc = _doc()
    doc.select(inker.SelectionMask.from_rect(SIZE, (0, 0, SIZE[0], 12)))
    doc.shape_path("polygon", BEND, RED, 1, filled=True)
    alpha = _alpha(doc)
    assert int(alpha[:12].max()) > 0
    assert int(alpha[12:].max()) == 0


def test_a_locked_layer_refuses_a_path_shape():
    doc = _doc()
    doc.stack.active.locked = True
    assert doc.shape_path("polygon", BEND, RED, 1) is False
    assert len(doc.history) == 0


def test_the_alpha_lock_keeps_a_path_shape_invisible():
    """The lock is checked once, at ``write_colour``, which is exactly why this
    needs no code of its own to be true."""
    doc = _doc()
    doc.stack.active.alpha_lock = True
    doc.shape_path("polyline", BEND, RED, 3)
    assert int(_alpha(doc).max()) == 0


# --- the toolbox --------------------------------------------------------------


def test_the_three_tools_are_listed_with_their_own_letters():
    """A letter each, and not one another's.

    The literal letters were pinned here until the Aseprite pass moved the
    polyline off ``L`` -- which is Aseprite's *line* and had been handed out in
    toolbox order before the line was reached. What is worth pinning is what
    the test is named for: three tools, three distinct letters, each reachable.
    ``TOOL_KEYS`` is derived from ``TOOLS`` now, so "the two tables agree" is
    structural rather than something a test has to re-check -- but checking it
    costs nothing and says where the letters come from.
    """
    rows = {tool: shortcut.lower() for tool, _label, shortcut in inker_state.TOOLS}
    letters = {tool: key for key, tool in inker_mode.TOOL_KEYS.items()}
    three = ("polyline", "polygon", "curve")
    assert len({rows[tool] for tool in three}) == 3
    for tool in three:
        assert rows[tool] == letters[tool]
        assert tool in inker_state.SHAPE_TOOLS
        assert tool in inker_state.PATH_SHAPE_TOOLS
        assert tool in inker_tools.TOOL_ICONS


def test_the_open_paths_have_no_fill_and_the_polygon_does():
    """A Filled checkbox on a shape that encloses nothing is a control that
    does nothing."""
    assert sorted(inker_state.OPEN_SHAPE_TOOLS) == ["curve", "line", "polyline"]
    assert "polygon" not in inker_state.OPEN_SHAPE_TOOLS
    assert inker_state.OPEN_SHAPE_TOOLS < inker_state.SHAPE_TOOLS


def test_a_clicked_shape_is_not_a_selection_tool():
    """It paints, so it is past the lock check and inert on the right button
    like every other shape -- not a member of the group that reserves it."""
    assert not (inker_state.PATH_SHAPE_TOOLS & inker_state.SELECT_TOOLS)
    assert not (inker_state.PATH_SHAPE_TOOLS & inker_state.BG_BUTTON_TOOLS)


# --- the gesture, through the real dispatch ----------------------------------


class _Mouse:
    """imgui's mouse, as much of it as the press dispatch reads.

    ``test_poly_lasso``'s, because this gesture is that gesture: the double
    click is what commits both.
    """

    def __init__(self) -> None:
        self.at = (0.0, 0.0)
        self.down = {0: False, 1: False, 2: False}
        self.clicked = {0: False, 1: False, 2: False}
        self.double = {0: False, 1: False, 2: False}
        self.shift = False
        self.alt = False
        #: Every pointer shape ``_os_cursor`` asked for, newest last.
        self.cursors: list[str] = []

    def module(self) -> SimpleNamespace:
        return SimpleNamespace(
            get_io=lambda: SimpleNamespace(
                mouse_wheel=0.0,
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
            set_mouse_cursor=self.cursors.append,
            MouseCursor_=SimpleNamespace(
                **{
                    name: SimpleNamespace(value=name)
                    for name in ("hand", "not_allowed", "resize_all", "arrow")
                }
            ),
        )


def _tab(**view):
    return inker_state.InkerDoc(
        doc=inker.Document.blank(*SIZE), view=inker_state.PaintView(**view)
    )


@pytest.fixture
def scene(monkeypatch):
    """A clicked shape tool in hand at an identity view, driven through the
    real ``_input`` -- press frame then release frame -- because the press
    guard and the grid snap both live above the tool."""
    mouse = _Mouse()
    monkeypatch.setattr(inker_canvas, "imgui", mouse.module())
    tab = _tab(zoom=1.0, pan=(0.0, 0.0), fitted=True)
    state = inker_state.InkerState(tool="polyline")
    state.fg = RED
    state.brush_size = 1
    state.add(tab)
    toasts: list[tuple[str, str]] = []
    ctx = SimpleNamespace(toast=lambda text, kind="info": toasts.append((text, kind)))

    def frame(at, *, pressed=None, down=(), double=False):
        mouse.at = (float(at[0]), float(at[1]))
        mouse.clicked = {0: False, 1: False, 2: False}
        mouse.double = {0: False, 1: False, 2: False}
        if pressed is not None:
            mouse.clicked[pressed] = True
            mouse.double[pressed] = double
        mouse.down = {b: b in down for b in (0, 1, 2)}
        inker_canvas._input(ctx, state, tab, ORIGIN, active=True, hovered=True)

    def click(point, *, button: int = 0, double: bool = False):
        frame(point, pressed=button, down=(button,), double=double)
        frame(point)

    return SimpleNamespace(state=state, tab=tab, click=click, mouse=mouse, toasts=toasts)


def test_each_click_places_a_vertex_and_paints_nothing_yet(scene):
    """The load-bearing assertion, C4's: an empty ``drag_kind`` is what keeps
    the next click out of the press guard, and it is also what keeps this tool
    off the dragged-shape path."""
    for point in BEND:
        scene.click(point)
        assert scene.state.drag_kind == ""
    assert scene.state.gesture_pts == BEND
    assert _painted(scene.tab.doc) == 0
    assert len(scene.tab.doc.history) == 0


def test_a_double_click_commits_the_polyline_as_one_step(scene):
    for point in BEND:
        scene.click(point)
    scene.click(BEND[-1], double=True)
    assert scene.state.gesture_pts == []
    assert len(scene.tab.doc.history) == 1
    expected = _doc()
    expected.shape_path("polyline", BEND, RED, 1)
    assert np.array_equal(
        scene.tab.doc.stack.active.pixels, expected.stack.active.pixels
    )


def test_a_click_back_on_the_first_point_closes_a_polygon(scene):
    scene.state.set_tool("polygon")
    # After the switch, because the size is a *per-tool* option and the polygon
    # has its own -- which is the same reason the landing reads it at commit.
    scene.state.brush_size = 1
    for point in BEND:
        scene.click(point)
    scene.click(BEND[0])
    assert scene.state.gesture_pts == []
    assert len(scene.tab.doc.history) == 1
    # The closing click places no vertex of its own -- a click near vertex 0
    # *is* vertex 0, and appending it would put a zero-length edge in the shape.
    expected = _doc()
    expected.shape_path("polygon", BEND, RED, 1)
    assert np.array_equal(
        scene.tab.doc.stack.active.pixels, expected.stack.active.pixels
    )


def test_an_open_path_takes_a_click_on_its_first_point_as_a_vertex(scene):
    """The polyline and the curve have no closing click, because there is
    nothing to close: a click near their first point is a click like any other
    and placing a vertex is the only thing it can honestly mean."""
    for tool in ("polyline", "curve"):
        scene.state.set_tool(tool)
        for point in BEND:
            scene.click(point)
        scene.click(BEND[0])
        assert scene.state.gesture_pts == [*BEND, BEND[0]], tool
        assert len(scene.tab.doc.history) == 0, tool
        scene.state.clear_gesture()


def test_a_polygon_of_two_clicks_commits_nothing(scene):
    """The poly lasso's floor, for its reason: two clicks make a polygon that is
    a line, and a tool asked to close a shape that encloses nothing leaves no
    mark for the user to find and undo."""
    scene.state.set_tool("polygon")
    scene.click(BEND[0])
    scene.click(BEND[1], double=True)
    assert scene.state.gesture_pts == []
    assert len(scene.tab.doc.history) == 0
    assert _painted(scene.tab.doc) == 0


def test_two_clicks_are_enough_for_an_open_path(scene):
    """The double-click's second press places nothing, so this is the two
    vertices of the first two clicks -- and a polyline of two is a line."""
    scene.click(BEND[0])
    scene.click(BEND[1])
    scene.click(BEND[1], double=True)
    assert len(scene.tab.doc.history) == 1
    assert _painted(scene.tab.doc) > 0


def test_the_curve_lands_the_curve_that_was_previewed(scene):
    scene.state.set_tool("curve")
    scene.state.brush_size = 1
    points = [(4.0, 16.0), (16.0, 6.0), (28.0, 16.0)]
    for point in points:
        scene.click(point)
    scene.click(points[-1], double=True)
    expected = _doc()
    expected.shape_path("curve", points, RED, 1)
    assert np.array_equal(
        scene.tab.doc.stack.active.pixels, expected.stack.active.pixels
    )


def test_a_right_click_neither_places_a_vertex_nor_drops_the_path(scene):
    """A shape tool is inert on the right button; inert means the open path is
    neither extended nor thrown away."""
    scene.click(BEND[0])
    scene.click((20.0, 20.0), button=1)
    assert scene.state.gesture_pts == [BEND[0]]


def test_the_vertices_snap_to_the_grid_when_snapping_is_on(scene):
    """Above the tool, in ``_input``: a clicked corner is a placed point, and
    placing points on the grid is what the setting is for."""
    scene.state.grid = True
    scene.state.grid_snap = True
    scene.state.grid_size = 8
    scene.click((9.0, 15.0))
    assert scene.state.gesture_pts == [(8.0, 16.0)]


def test_a_locked_layer_opens_no_path_at_all(scene):
    """The refusal is at the press rather than at the commit: offering a rubber
    band, taking four clicks and only then saying the layer is locked is a form
    the app knew the answer to before it drew it."""
    scene.tab.doc.stack.active.locked = True
    scene.click(BEND[0])
    assert scene.state.gesture_pts == []
    # A tip, not a toast (W2.6): this answers the click, and it belongs under
    # the cursor that made it rather than over the window.
    assert scene.state.tip is not None and "locked" in scene.state.tip.text


def test_a_tool_switch_drops_the_path(scene):
    scene.click(BEND[0])
    scene.click(BEND[1])
    scene.state.set_tool("brush")
    assert scene.state.gesture_pts == []


def test_the_tab_going_busy_drops_the_path(scene):
    """A save is encoding the live document. The next click cannot be delivered
    while ``_input`` returns early, so vertices left up would finish a shape
    against whatever the document looked like when the tab came back."""
    scene.click(BEND[0])
    scene.click(BEND[1])
    scene.tab.saving = True
    scene.click(BEND[2])
    assert scene.state.gesture_pts == []
    assert _painted(scene.tab.doc) == 0


def test_committing_on_a_busy_tab_paints_nothing_and_closes_the_path():
    """C4's answer, and it is the one that cannot strand pixels: a commit that
    arrived mid-save would put half a polygon in the file."""
    tab = _tab(zoom=1.0)
    state = inker_state.InkerState(tool="polygon")
    state.fg = RED
    state.add(tab)
    state.gesture_pts = list(BEND)
    tab.saving = True
    assert inker_mode.commit_gesture(state, tab) is False
    assert state.gesture_pts == []
    assert len(tab.doc.history) == 0
    assert _painted(tab.doc) == 0


def test_enter_commits_the_path_and_escape_abandons_it(monkeypatch):
    import pygame

    for key, painted in ((pygame.K_RETURN, True), (pygame.K_ESCAPE, False)):
        tab = _tab(zoom=1.0)
        state = inker_state.InkerState(tool="polyline")
        state.fg = RED
        state.brush_size = 1
        state.add(tab)
        state.gesture_pts = list(BEND)
        ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
        event = pygame.event.Event(pygame.KEYDOWN, key=key, mod=0)
        assert inker_mode.handle_key(ctx, event) is True
        assert state.gesture_pts == []
        assert (_painted(tab.doc) > 0) is painted
        assert (len(tab.doc.history) == 1) is painted


def test_the_landing_is_chosen_by_the_tool_holding_the_vertices():
    """Safe precisely because ``set_tool`` clears the gesture: the tool at the
    commit is always the tool that placed the points."""
    for tool, paints in (("lasso_poly", False), ("polygon", True)):
        tab = _tab(zoom=1.0)
        state = inker_state.InkerState(tool=tool)
        state.fg = RED
        state.add(tab)
        state.gesture_pts = list(BEND)
        assert inker_mode.commit_gesture(state, tab) is True
        assert (_painted(tab.doc) > 0) is paints, tool
        assert (tab.doc.mask is not None) is not paints, tool


def test_the_options_the_landing_reads_are_the_tool_s_own():
    """Size, colour, fill and the document's tiling, exactly as the dragged
    shapes read them -- so a polygon drawn at size 9 on a tiled document is the
    same write ``_release`` would have made."""
    tab = _tab(zoom=1.0)
    tab.tiled = "x"
    state = inker_state.InkerState(tool="polygon")
    state.fg = RED
    state.brush_size = 5
    state.shape_filled = True
    assert inker_mode.paint_path(state, tab, "polygon", BEND) is True
    expected = _doc()
    expected.shape_path("polygon", BEND, RED, 5, filled=True, wrap="x")
    assert np.array_equal(tab.doc.stack.active.pixels, expected.stack.active.pixels)


# --- the overlay --------------------------------------------------------------


class _Lines:
    """A draw list that only takes notes; ``test_poly_lasso``'s idiom."""

    def __init__(self) -> None:
        self.lines: list[tuple] = []
        self.circles: list[tuple] = []

    def add_line(self, a, b, colour, thickness=1.0) -> None:
        self.lines.append((a, b))

    def add_circle(self, centre, radius, colour) -> None:
        self.circles.append((centre, radius))


@pytest.fixture
def overlay(monkeypatch):
    monkeypatch.setattr(inker_canvas, "_u32", lambda colour, alpha=1.0: 0)
    mouse = _Mouse()
    monkeypatch.setattr(inker_canvas, "imgui", mouse.module())
    return mouse


def _preview(tool: str, points) -> _Lines:
    tab = _tab(zoom=1.0, pan=(0.0, 0.0))
    state = inker_state.InkerState(tool=tool)
    state.gesture_pts = list(points)
    lines = _Lines()
    inker_canvas._gesture_preview(state, tab, lines, ORIGIN)
    return lines


def test_the_polygon_preview_shows_the_edge_it_would_close_with(overlay):
    overlay.at = (20.0, 20.0)
    lines = _preview("polygon", BEND)
    assert lines.lines[-1] == ((20.0, 20.0), BEND[0])
    assert lines.circles[0][0] == BEND[0]


def test_an_open_path_previews_no_closing_edge_and_no_ring(overlay):
    """Drawing a target that closes nothing would promise a click that does not
    exist."""
    overlay.at = (20.0, 20.0)
    lines = _preview("polyline", BEND)
    assert lines.circles == []
    assert lines.lines == [
        (BEND[0], BEND[1]),
        (BEND[1], BEND[2]),
        (BEND[2], (20.0, 20.0)),
    ]


def test_the_curve_preview_is_the_curve_rather_than_its_chords(overlay):
    """Segment by segment through the same ``catmull_rom`` the commit
    rasterises: a spline preview that draws a polygon is a preview of a
    different tool, and one that samples the curve its own way is a promise the
    rasteriser has not made."""
    overlay.at = (20.0, 20.0)
    lines = _preview("curve", BEND)
    expected = _doc_paint.catmull_rom([*BEND, (20.0, 20.0)])
    assert len(lines.lines) == len(expected) - 1
    assert lines.lines[0] == (expected[0], expected[1])
    assert lines.lines[-1][1] == expected[-1]
    assert lines.circles == []


def test_nothing_is_drawn_with_no_path_open(overlay):
    lines = _preview("curve", [])
    assert lines.lines == [] and lines.circles == []


# --- the preview's memo -------------------------------------------------------
#
# The overlay runs on the pygame frame loop, which never blocks. Resampling a
# fifty-vertex curve from scratch every frame is milliseconds of arithmetic for
# a picture that changed in two segments, so the settled segments are memoised
# -- and the whole of what that memo has to be worth is that nobody can tell.


@pytest.fixture(autouse=True)
def _cold_cache():
    """Every test in this file starts with the memo empty, and leaves it so."""
    inker_canvas._curve_settled = None
    yield
    inker_canvas._curve_settled = None


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 9, 20])
def test_the_memoised_path_is_the_full_resample_bit_for_bit(count: int):
    """The claim the memo lives or dies by, at every length: identical, not
    close. The same control points in the same order through the same
    arithmetic can only produce the same floats."""
    points = [(4.0 * i, 6.0 + 5.0 * (i % 3)) for i in range(count)]
    cursor = (4.0 * count + 3.0, 9.0)
    assert inker_canvas._curve_path(points, cursor) == _doc_paint.catmull_rom(
        [*points, cursor]
    )
    # And again with the memo warm, which is the case that actually skips work.
    assert inker_canvas._curve_path(points, cursor) == _doc_paint.catmull_rom(
        [*points, cursor]
    )


def test_a_moving_cursor_never_disturbs_the_settled_prefix():
    """The cursor reaches exactly the last two segments -- the one that ends at
    it and the one before, which has it as its far neighbour."""
    points = [(4.0, 6.0), (14.0, 22.0), (24.0, 4.0), (30.0, 20.0), (40.0, 8.0)]
    settled = len(_doc_paint.catmull_rom(points[:-2]))
    first = inker_canvas._curve_path(points, (50.0, 20.0))
    for cursor in ((51.0, 4.0), (12.0, 30.0), (44.0, 9.0)):
        moved = inker_canvas._curve_path(points, cursor)
        assert moved[:settled] == first[:settled]
        assert moved == _doc_paint.catmull_rom([*points, cursor])


def test_a_warm_memo_resamples_only_the_tail(monkeypatch):
    """What the memo is *for*, counted rather than timed: with the vertices
    unchanged, no frame resamples more than the four control points of the two
    segments the cursor can move."""
    points = [(3.0 * i, 6.0 + 7.0 * (i % 4)) for i in range(30)]
    inker_canvas._curve_path(points, (95.0, 10.0))  # warms it
    sizes: list[int] = []
    real = inker_canvas.curve_spans
    monkeypatch.setattr(
        inker_canvas,
        "curve_spans",
        lambda pts, **kw: (sizes.append(len(list(pts))), real(pts, **kw))[1],
    )
    for step in range(10):
        inker_canvas._curve_path(points, (95.0 + step, 10.0))
    assert sizes == [4] * 10


def test_adding_a_vertex_re_settles_the_prefix():
    """A click is a new key, so the memo misses once and is right afterwards --
    the only invalidation this needs, because a stale entry cannot be read."""
    points = [(4.0, 6.0), (14.0, 22.0), (24.0, 4.0)]
    cursor = (34.0, 18.0)
    inker_canvas._curve_path(points, cursor)
    grown = [*points, (34.0, 18.0)]
    assert inker_canvas._curve_path(grown, (44.0, 6.0)) == _doc_paint.catmull_rom(
        [*grown, (44.0, 6.0)]
    )
    # And back down again: undoing a click must not read the longer path's memo.
    assert inker_canvas._curve_path(points, cursor) == _doc_paint.catmull_rom(
        [*points, cursor]
    )


def test_a_repeated_vertex_or_a_cursor_on_the_last_point_still_agrees():
    """The collapse happens before the split, so a double-clicked vertex cannot
    put the memo's segment count out of step with the rasteriser's."""
    points = [(4.0, 6.0), (14.0, 22.0), (14.0, 22.0), (24.0, 4.0), (30.0, 20.0)]
    for cursor in ((30.0, 20.0), (40.0, 12.0), (4.0, 6.0)):
        assert inker_canvas._curve_path(points, cursor) == _doc_paint.catmull_rom(
            [*points, cursor]
        )
