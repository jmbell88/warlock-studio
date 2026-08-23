"""The canvas's half of tiled painting, the spray and the right button.

Everything here is a pure function of the pane's own state, which is the point:
the drawing is imgui's problem, but *which point the tool was handed* and *which
colour it was given* are decisions, and each of them has a failure that looks
like a working editor -- a brush that jumps a tile at the seam, a right-drag
that ends on the wrong frame, a spray whose dab is as wide as its cloud.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import imgui_backend, inker, inker_mode, inker_state
from warlock.studio.inker import brush, tiling
from warlock.studio.panes import inker_canvas, inker_tools

SIZE = (32, 32)
FG = (255, 0, 0, 255)
BG = (0, 0, 255, 255)


@pytest.fixture
def scene(monkeypatch):
    """A pane state, a tab and a document, with imgui's io stubbed.

    ``_press`` reads exactly one thing off imgui -- the Alt modifier -- and
    reaching for a live context takes the process down, which is what kept this
    dispatch untested.
    """
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=False, key_alt=False, key_ctrl=False),
    )
    state = inker_state.InkerState(fg=FG, bg=BG)
    tab = SimpleNamespace(doc=inker.Document.blank(*SIZE), tiled="off", busy=False)
    return state, tab


def _press(state, tab, point):
    inker_canvas._press(None, state, tab, point)


def _alt(monkeypatch, held=True):
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=False, key_alt=held, key_ctrl=False),
    )


# --- driving ``_input`` frame by frame --------------------------------------
#
# The two-button hazard only exists at the dispatch level: ``_press`` on its own
# cannot tell that another gesture owns the mouse. So these tests drive the real
# ``_input`` with a fake mouse, one frame per call, which is the only place the
# "a press is refused while a gesture owns the mouse" rule can be asserted.


class _Mouse:
    """imgui's mouse, as much of it as ``_input`` reads."""

    def __init__(self) -> None:
        self.at = (0.0, 0.0)
        self.down = {0: False, 1: False, 2: False}
        self.clicked = {0: False, 1: False, 2: False}
        #: Which buttons imgui reports as *dragging*, which is what arms
        #: the pan arm -- middle-drag, or space plus left-drag.
        self.dragging = {0: False, 1: False, 2: False}
        # As the backend delivers it -- already scaled by ``WHEEL_SCALE`` -- so
        # a test that sets it is exercising the same number the pane sees.
        self.wheel = 0.0
        #: Every pointer shape asked for, newest last. See ``module``.
        self.cursors: list[str] = []
        #: Held modifiers, for the gestures that read them at the press.
        self.shift = False
        self.ctrl = False

    def module(self) -> SimpleNamespace:
        return SimpleNamespace(
            get_io=lambda: SimpleNamespace(
                mouse_wheel=self.wheel,
                key_shift=self.shift,
                key_alt=False,
                key_ctrl=self.ctrl,
                delta_time=1.0 / 60.0,
            ),
            get_mouse_pos=lambda: SimpleNamespace(x=self.at[0], y=self.at[1]),
            is_mouse_clicked=lambda button: self.clicked[button],
            is_mouse_down=lambda button: self.down[button],
            is_mouse_dragging=lambda button: self.dragging[button],
            get_mouse_drag_delta=lambda button: SimpleNamespace(x=0.0, y=0.0),
            reset_mouse_drag_delta=lambda button: None,
            # The pointer shape ``_os_cursor`` sets. Recorded rather than
            # ignored: what the pointer says over a locked layer is the whole
            # point of that helper, so a fake that swallowed it would let the
            # feedback regress silently.
            set_mouse_cursor=self.cursors.append,
            MouseCursor_=SimpleNamespace(
                **{
                    name: SimpleNamespace(value=name)
                    for name in ("hand", "not_allowed", "resize_all", "arrow")
                }
            ),
        )


@pytest.fixture
def driven(monkeypatch):
    """``_input`` with a fake mouse, at identity view so screen == image."""
    mouse = _Mouse()
    monkeypatch.setattr(inker_canvas, "imgui", mouse.module())
    state = inker_state.InkerState(fg=FG, bg=BG)
    tab = SimpleNamespace(
        doc=inker.Document.blank(*SIZE),
        tiled="off",
        busy=False,
        view=inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True),
    )

    def frame(
        at,
        *,
        click=None,
        down=(),
        dragging=(),
        wheel=0.0,
        shift=False,
        ctrl=False,
        hovered=True,
    ):
        mouse.at = (float(at[0]), float(at[1]))
        mouse.clicked = {0: False, 1: False, 2: False}
        if click is not None:
            mouse.clicked[click] = True
        mouse.down = {b: b in down for b in (0, 1, 2)}
        mouse.dragging = {b: b in dragging for b in (0, 1, 2)}
        mouse.wheel = float(wheel)
        mouse.shift = shift
        mouse.ctrl = ctrl
        inker_canvas._input(None, state, tab, (0.0, 0.0), active=True, hovered=hovered)

    frame.mouse = mouse
    return state, tab, frame


# --- the tables -------------------------------------------------------------


def test_the_tiled_combo_offers_every_mode_the_engine_implements():
    assert tuple(key for key, _label in inker_canvas.TILED_LABELS) == tiling.TILED_AXES


def test_every_ink_names_a_real_brush_mode():
    """Five inks now (6.1), and each one *is* an engine mode: the table carries
    the mode rather than a label the press has to map, so an ink that named
    nothing would be a control that silently painted with the default."""
    for key, label, mode, _lock, hint in inker_state.INKS:
        assert mode in brush.MODES, key
        assert label and hint, key
    keys = [key for key, _l, _m, _lock, _h in inker_state.INKS]
    assert keys == ["simple", "alpha", "copy", "lock_alpha", "shading"]
    # Lock Alpha is the one whose *mode* is the ordinary composite: what makes
    # it an ink is the flag beside it.
    assert inker_state.ink_mode("lock_alpha") == "paint"
    assert inker_state.ink_locks_alpha("lock_alpha") is True


def test_every_tool_has_an_icon_of_its_own():
    """A toolbox where two buttons carry one glyph is one a user has to hover
    to read. Spray is the row that nearly took the blur tool's spray can."""
    icons = [inker_tools.TOOL_ICONS[key] for key, _l, _s in inker_state.TOOLS]
    assert len(set(icons)) == len(icons)


def test_the_right_button_tools_are_all_real_tools():
    tools = {key for key, _l, _s in inker_state.TOOLS}
    assert tools >= inker_state.BG_BUTTON_TOOLS
    assert not (inker_state.BG_BUTTON_TOOLS & inker_state.SELECT_TOOLS)


# --- the 3x3 view -----------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "extent"),
    [
        ("off", (0, 0, 32, 32)),
        ("x", (-32, 0, 64, 32)),
        ("y", (0, -32, 32, 64)),
        ("both", (-32, -32, 64, 64)),
    ],
)
def test_the_drawn_extent_grows_only_along_wrapped_axes(mode, extent):
    """X-only tiling shows a strip of three rather than a 3x3 block, which is
    the honest picture of what will actually wrap when you paint on it."""
    tab = SimpleNamespace(tiled=mode)
    assert inker_canvas._tiled_extent(tab, SIZE) == extent


def test_the_two_corner_uvs_become_the_four_the_quad_takes(monkeypatch):
    """``_blit``'s new spelling, checked on the values rather than on the call:
    UVs outside 0..1 with the sampler repeating is what draws the neighbourhood
    in one quad instead of nine images placed by hand."""
    recorded = {}

    class _List:
        def add_image_quad(self, ref, a, b, c, d, *uv):
            recorded["uv"] = uv

    view = inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0))
    # ``texture_ref`` reaches into a live GL renderer, which is the whole
    # reason this pane's drawing is otherwise untested; the reference itself is
    # not what this asserts.
    monkeypatch.setattr(inker_canvas.widgets, "texture_ref", lambda texture: texture)
    inker_canvas._blit(
        _List(), SimpleNamespace(), view, (0.0, 0.0), -32, -32, 64, 64,
        uv0=(-1.0, -1.0), uv1=(2.0, 2.0),
    )
    assert recorded["uv"] == ((-1.0, -1.0), (2.0, -1.0), (2.0, 2.0), (-1.0, 2.0))


# --- input folding ----------------------------------------------------------


def test_a_gesture_subtracts_the_tile_it_pressed_in_and_keeps_subtracting_it():
    """The whole reason ``tile_offset`` is separate from ``canonical``: folding
    each sample independently would jump the brush a full tile the moment the
    cursor crossed a seam mid-stroke."""
    state = inker_state.InkerState()
    state.tile_offset = tiling.tile_offset((40.0, 5.0), SIZE, (True, True))
    assert state.tile_offset == (32, 0)
    assert inker_canvas._local(state, (40.0, 5.0)) == (8.0, 5.0)
    # And past the seam, in the same gesture: 30 stays 30 rather than folding
    # back to 30 from the *other* tile.
    assert inker_canvas._local(state, (30.0, 5.0)) == (-2.0, 5.0)


def test_an_untiled_document_folds_nothing():
    state = inker_state.InkerState()
    state.tile_offset = tiling.tile_offset((40.0, 5.0), SIZE, (False, False))
    assert inker_canvas._local(state, (40.0, 5.0)) == (40.0, 5.0)


# --- the right button -------------------------------------------------------


def test_a_right_press_on_the_brush_paints_the_background_colour(scene):
    state, tab = scene
    state.tool = "brush"
    state.nib = "square"
    state.brush_size = 3
    state.drag_button = 1
    _press(state, tab, (8.0, 8.0))
    tab.doc.end_stroke()
    assert tuple(int(v) for v in tab.doc.stack.active.pixels[8, 8]) == BG


def test_a_left_press_still_paints_the_foreground_colour(scene):
    state, tab = scene
    state.tool = "brush"
    state.nib = "square"
    state.brush_size = 3
    _press(state, tab, (8.0, 8.0))
    tab.doc.end_stroke()
    assert tuple(int(v) for v in tab.doc.stack.active.pixels[8, 8]) == FG


def test_a_right_press_on_a_selection_tool_is_inert(scene):
    """Reserved rather than given a second meaning: an inert button is a
    promise that can be kept later where a wrong one cannot be taken back."""
    state, tab = scene
    state.tool = "lasso"
    state.drag_button = 1
    _press(state, tab, (8.0, 8.0))
    assert state.drag_kind == ""
    assert state.lasso == []


def test_a_right_press_fills_with_the_background_colour(scene):
    state, tab = scene
    state.tool = "fill"
    state.drag_button = 1
    _press(state, tab, (8.0, 8.0))
    assert tuple(int(v) for v in tab.doc.stack.active.pixels[8, 8]) == BG


def test_alt_right_click_picks_into_the_background(scene, monkeypatch):
    """The other half of the pair: the right button paints with bg, so it picks
    into bg -- one button, one meaning."""
    state, tab = scene
    tab.doc.stack.active.pixels[4, 4] = (9, 8, 7, 255)
    tab.doc.invalidate_all()
    state.tool = "brush"
    state.drag_button = 1
    _alt(monkeypatch)
    _press(state, tab, (4.0, 4.0))
    assert state.bg == (9, 8, 7, 255)
    assert state.fg == FG


def test_alt_left_click_still_picks_into_the_foreground(scene, monkeypatch):
    state, tab = scene
    tab.doc.stack.active.pixels[4, 4] = (9, 8, 7, 255)
    tab.doc.invalidate_all()
    state.tool = "brush"
    _alt(monkeypatch)
    _press(state, tab, (4.0, 4.0))
    assert state.fg == (9, 8, 7, 255)
    assert state.bg == BG


# --- the spray --------------------------------------------------------------


def test_a_spray_dab_is_a_fraction_of_the_cloud_it_scatters_in():
    """The size slider is the *cloud* for this one tool, so the dab has to be
    part of it -- a spray whose dabs are as wide as its own disc is a blob."""
    state = inker_state.InkerState()
    state.tool = "spray"
    state.brush_size = 40
    assert inker_canvas._dab_size(state, True) == 10
    assert inker_canvas._dab_size(state, False) == 40
    # Never zero: a stamp with no diameter is a stroke that stops.
    state.brush_size = 1
    assert inker_canvas._dab_size(state, True) == 1


def test_pressing_the_spray_opens_a_scattered_stroke(scene):
    state, tab = scene
    state.tool = "spray"
    state.brush_size = 16
    _press(state, tab, (16.0, 16.0))
    assert state.drag_kind == "spray"
    stroke = tab.doc._stroke
    assert stroke.scatter == 8.0
    assert stroke.mode == "paint"
    # Forced off, whatever the tool's own options say.
    assert stroke.pixel_perfect is False and stroke.stabilise == 0.0


def test_two_presses_of_the_spray_use_different_seeds(scene):
    state, tab = scene
    state.tool = "spray"
    _press(state, tab, (16.0, 16.0))
    first = tab.doc._stroke.seed
    tab.doc.end_stroke()
    _press(state, tab, (16.0, 16.0))
    assert tab.doc._stroke.seed != first


def test_the_brush_ink_option_reaches_the_stroke(scene):
    state, tab = scene
    state.tool = "brush"
    state.paint_ink = "replace"
    _press(state, tab, (8.0, 8.0))
    assert tab.doc._stroke.mode == "replace"
    tab.doc.end_stroke()
    # And it is the brush's alone -- the eraser still erases.
    state.tool = "eraser"
    state.paint_ink = "replace"
    _press(state, tab, (8.0, 8.0))
    assert tab.doc._stroke.mode == "erase"


def test_the_wrap_mode_reaches_the_stroke(scene):
    state, tab = scene
    state.tool = "brush"
    tab.tiled = "x"
    _press(state, tab, (8.0, 8.0))
    assert tab.doc._stroke.wrap_axes == (True, False)


# --- the move tool's third arm ----------------------------------------------


def test_pressing_the_move_tool_on_bare_pixels_opens_a_layer_move(scene):
    state, tab = scene
    tab.doc.stack.active.pixels[4:8, 4:8] = FG
    tab.doc.invalidate_all()
    state.tool = "move"
    _press(state, tab, (6.0, 6.0))
    assert state.drag_kind == "layer_move"
    assert tab.doc._move is not None


def test_a_selection_still_takes_priority_over_the_layer(scene):
    state, tab = scene
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (4, 4, 8, 8)))
    state.tool = "move"
    _press(state, tab, (6.0, 6.0))
    assert state.drag_kind == "move"
    assert tab.doc.floating is not None


def test_arrow_nudge_moves_the_layer_under_the_move_tool(scene):
    state, tab = scene
    tab.doc.stack.active.pixels[4:8, 4:8] = FG
    tab.doc.invalidate_all()
    state.tool = "move"
    assert inker_mode.nudge(state, tab, 1, 0)
    _ys, xs = tab.doc.stack.active.pixels[..., 3].nonzero()
    assert int(xs.min()) == 5
    # One step per press, so a nudge is one Ctrl+Z rather than a session left
    # open waiting for a release that never comes.
    assert tab.doc.history.can_undo
    tab.doc.undo()
    assert not tab.doc.history.can_undo


def test_arrow_nudge_moves_a_floating_buffer_first(scene):
    state, tab = scene
    tab.doc.stack.active.pixels[4:8, 4:8] = FG
    tab.doc.invalidate_all()
    tab.doc.select(inker.SelectionMask.from_rect(SIZE, (4, 4, 8, 8)))
    tab.doc.lift()
    state.tool = "brush"  # a float is nudged whatever tool is in hand
    assert inker_mode.nudge(state, tab, 0, 2)
    assert tab.doc.floating.offset == (4, 6)


def test_arrow_nudge_is_gated_on_the_move_tool(scene):
    """The arrows are the last keys a document pane has to give away, and
    quietly translating a layer because somebody pressed Right with the brush
    selected is not a trade worth making."""
    state, tab = scene
    tab.doc.stack.active.pixels[4:8, 4:8] = FG
    tab.doc.invalidate_all()
    state.tool = "brush"
    assert not inker_mode.nudge(state, tab, 1, 0)
    assert not tab.doc.history.can_undo


# --- two buttons at once (the C12d regression) ------------------------------


def test_a_right_click_mid_layer_move_does_not_abandon_the_move(driven):
    """The critical one. Before the guard, the right button coming down
    mid-drag reached ``_press``, whose inert arm cleared ``drag_kind`` and
    returned -- so the release never committed, the layer stayed at the
    previewed offset with no undo step and no dirty flag, and the *next*
    ``begin_layer_move`` would have rolled it back from a stale snapshot,
    taking every stroke painted in between with it."""
    state, tab, frame = driven
    tab.doc.stack.active.pixels[4:8, 4:8] = FG
    tab.doc.invalidate_all()
    state.tool = "move"

    frame((6, 6), click=0, down=(0,))
    assert state.drag_kind == "layer_move"
    frame((9, 6), down=(0,))
    assert tab.doc._move is not None

    frame((9, 6), click=1, down=(0, 1))
    # Ignored outright: the gesture is untouched and still owned by button 0.
    assert state.drag_kind == "layer_move"
    assert state.drag_button == 0
    assert tab.doc._move is not None

    frame((9, 6), down=())
    assert state.drag_kind == ""
    assert tab.doc._move is None
    assert tab.doc.history.can_undo  # committed, not orphaned
    _ys, xs = tab.doc.stack.active.pixels[..., 3].nonzero()
    assert int(xs.min()) == 7


def test_a_right_click_mid_blur_stroke_does_not_orphan_the_stroke(driven):
    """The other half of the same bug: blur, smudge and spray are paint tools
    but not right-button tools, so a right-click mid-stroke hit the inert gate
    and left ``doc._stroke`` open with its pixels already written -- pushed out
    of band by the *next* stroke's ``end_stroke``."""
    state, tab, frame = driven
    tab.doc.stack.active.pixels[:] = 255
    tab.doc.invalidate_all()
    state.tool = "blur"

    frame((10, 10), click=0, down=(0,))
    assert state.drag_kind == "paint"
    stroke = tab.doc._stroke
    assert stroke is not None

    frame((14, 10), click=1, down=(0, 1))
    assert state.drag_kind == "paint"
    assert tab.doc._stroke is stroke  # the same open stroke, untouched

    frame((14, 10), down=())
    assert tab.doc._stroke is None


def test_a_press_and_release_in_one_frame_closes_before_the_next_press(driven):
    """The narrow case the guard alone does not cover: the owning button is
    already up when the next press arrives, so nothing is "holding" -- and the
    gesture would be orphaned by the press rather than by the button. Closed
    explicitly, with the previous gesture's tile offset still in state."""
    state, tab, frame = driven
    tab.doc.stack.active.pixels[4:8, 4:8] = FG
    tab.doc.invalidate_all()
    state.tool = "move"

    frame((6, 6), click=0, down=(0,))
    frame((9, 6), down=(0,))
    # Left up and right down in the same frame.
    frame((9, 6), click=1, down=(1,))
    assert tab.doc._move is None  # the move was closed, not left open
    assert tab.doc.history.can_undo


def test_the_guard_does_not_stop_an_ordinary_second_gesture(driven):
    """The control: refusing a press while a gesture owns the mouse must not
    refuse the *next* one once the button is up."""
    state, tab, frame = driven
    state.tool = "brush"
    state.nib = "square"
    state.brush_size = 3

    frame((8, 8), click=0, down=(0,))
    frame((8, 8), down=())
    assert state.drag_kind == ""
    frame((20, 20), click=0, down=(0,))
    assert state.drag_kind == "paint"


def test_cancelling_a_move_that_is_not_open_falls_through(scene):
    """What the Escape chain leans on: no session means "not handled", so Esc
    goes on to cancel the float or drop the selection as it always did."""
    _state, tab = scene
    assert tab.doc.cancel_layer_move() is False


# --- the wheel --------------------------------------------------------------


def test_one_wheel_notch_moves_the_zoom_by_five_percent(driven):
    """The backend halves every wheel event; the pane divides that back out.

    Asserted through ``_input`` rather than against ``zoom_step`` directly,
    because the number under test is exactly the one that crosses the boundary
    between the two modules.
    """
    _state, tab, frame = driven
    tab.view.zoom = 1.0
    frame((16.0, 16.0), wheel=imgui_backend.WHEEL_SCALE)
    assert tab.view.zoom == pytest.approx(1.05)
    frame((16.0, 16.0), wheel=-imgui_backend.WHEEL_SCALE)
    assert tab.view.zoom == pytest.approx(1.0)


def test_the_wheel_stops_at_the_inker_bounds(driven):
    _state, tab, frame = driven
    for _ in range(400):
        frame((16.0, 16.0), wheel=imgui_backend.WHEEL_SCALE)
    assert tab.view.zoom == pytest.approx(inker_state.INKER_MAX_ZOOM)
    for _ in range(400):
        frame((16.0, 16.0), wheel=-imgui_backend.WHEEL_SCALE)
    assert tab.view.zoom == pytest.approx(inker_state.INKER_MIN_ZOOM)


# --- Shift paints a line from where the last stroke ended --------------------


def test_shift_click_opens_the_stroke_at_the_last_point(driven):
    """Aseprite's line-from-last-point, and the one gesture in the box faster
    than the line tool for what it does: click, Shift-click, Shift-click walks a
    polyline in the brush already in hand.

    Not a separate code path -- the stroke *opens* at the remembered point and
    is walked to the click -- so what is asserted is that pixels between the two
    points were painted, which only a real segment can do.
    """
    state, tab, frame = driven
    state.set_tool("brush")
    state.brush_size = 1
    state.nib = "pixel"
    frame((4.0, 4.0), click=0, down=(0,))
    frame((4.0, 4.0))
    assert tab.view.last_paint == (4.0, 4.0)

    frame((20.0, 4.0), click=0, down=(0,), shift=True)
    frame((20.0, 4.0))
    row = tab.doc.stack.active.pixels[4, 4:21, 3]
    assert int(row.min()) > 0, "every pixel of the segment was laid down"


def test_an_unmodified_click_paints_only_where_it_landed(driven):
    """The control for the test above: without Shift there is no segment."""
    state, tab, frame = driven
    state.set_tool("brush")
    state.brush_size = 1
    state.nib = "pixel"
    frame((4.0, 4.0), click=0, down=(0,))
    frame((4.0, 4.0))
    frame((20.0, 4.0), click=0, down=(0,))
    frame((20.0, 4.0))
    assert int(tab.doc.stack.active.pixels[4, 12, 3]) == 0, "nothing in between"


def test_the_first_shift_click_of_a_session_is_an_ordinary_press(driven):
    """Nothing to draw from, so nothing is drawn from it."""
    state, tab, frame = driven
    state.set_tool("brush")
    state.brush_size = 1
    state.nib = "pixel"
    assert tab.view.last_paint is None
    frame((20.0, 4.0), click=0, down=(0,), shift=True)
    frame((20.0, 4.0))
    assert int(tab.doc.stack.active.pixels[4, 12, 3]) == 0


def test_the_spray_is_left_out_of_it(driven):
    """Its advance is ``spray_at`` on a timer rather than a walk down a
    segment, so there is no line for this to draw."""
    state, tab, frame = driven
    state.set_tool("spray")
    frame((4.0, 4.0), click=0, down=(0,))
    frame((4.0, 4.0))
    before = tab.view.last_paint
    frame((20.0, 4.0), click=0, down=(0,), shift=True)
    assert state.drag_kind == "spray"
    assert before == (4.0, 4.0)


# --- what the pointer says ---------------------------------------------------


def test_the_pointer_refuses_over_a_locked_layer(driven):
    """The refusal was a toast raised *after* a press, so the way to find out a
    layer was locked was to try to draw on it."""
    state, tab, frame = driven
    state.set_tool("brush")
    tab.doc.stack.active.locked = True
    frame((8.0, 8.0))
    assert frame.mouse.cursors[-1] == "not_allowed"


def test_a_pick_over_a_locked_layer_is_not_shown_as_refused(driven):
    """Deliberately the same exemptions ``_locked_out`` uses: an eyedropper or
    a marquee over a locked layer is not refused, so it must not look it."""
    state, tab, frame = driven
    tab.doc.stack.active.locked = True
    for tool in ("eyedropper", "select"):
        state.set_tool(tool)
        frame.mouse.cursors.clear()
        frame((8.0, 8.0))
        assert "not_allowed" not in frame.mouse.cursors, tool


def test_the_pointer_grabs_while_space_is_held(driven):
    state, _tab, frame = driven
    state.space_held = True
    frame((8.0, 8.0))
    assert frame.mouse.cursors[-1] == "hand"


def test_the_move_tool_says_so(driven):
    state, _tab, frame = driven
    state.set_tool("move")
    frame((8.0, 8.0))
    assert frame.mouse.cursors[-1] == "resize_all"


def test_nothing_is_asked_for_while_the_canvas_is_not_hovered(driven):
    """imgui resets the cursor every frame, so a pane that set one while the
    pointer was elsewhere would be overriding the rest of the window."""
    state, _tab, frame = driven
    state.set_tool("move")
    frame((8.0, 8.0), hovered=False)
    assert frame.mouse.cursors == []


# --- Ctrl picks the layer under the cursor ----------------------------------


def _two_layers(tab):
    tab.doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    tab.doc.add_layer("upper")
    tab.doc.stack[1].pixels[2, 2] = (0, 0, 255, 255)
    tab.doc.invalidate_all()
    tab.doc.set_active_layer(0)


def test_ctrl_click_makes_the_layer_under_the_cursor_active(driven):
    state, tab, frame = driven
    _two_layers(tab)
    state.set_tool("brush")

    frame((2.5, 2.5), click=0, down=(0,), ctrl=True)

    assert tab.doc.stack.active_index == 1


def test_ctrl_click_leaves_no_dab_behind(driven):
    """The slice and text arms' rule: a modifier gesture must return before
    every paint branch, or picking a layer paints on the one you left."""
    state, tab, frame = driven
    _two_layers(tab)
    state.set_tool("brush")
    before = tab.doc.stack[1].pixels.copy()

    frame((2.5, 2.5), click=0, down=(0,), ctrl=True)

    assert np.array_equal(tab.doc.stack[1].pixels, before)
    assert state.drag_kind == ""


def test_ctrl_click_over_empty_canvas_changes_nothing(driven):
    state, tab, frame = driven
    _two_layers(tab)
    tab.doc.stack[0].pixels[:, :] = 0
    tab.doc.invalidate_all()
    state.set_tool("brush")

    frame((6.5, 6.5), click=0, down=(0,), ctrl=True)

    assert tab.doc.stack.active_index == 0


# --- a pan must not strand the gesture it interrupts -------------------------


def test_starting_a_pan_mid_stroke_closes_the_stroke(driven):
    """Taking ``drag_kind`` for the pan without dispatching a release left
    ``end_stroke`` un-run: the pixels were in the layer with no step behind
    them, so Ctrl+Z did nothing until the *next* stroke pushed them out of
    band. The C12d guard fixed this for the right button and not for the pan."""
    state, tab, frame = driven
    state.tool = "brush"
    head = tab.doc.history.head

    frame((4, 4), click=0, down=(0,))
    frame((12, 12), down=(0,))
    assert state.drag_kind == "paint"
    painted = int((tab.doc.stack.active.pixels[..., 3] > 0).sum())
    assert painted, "the stroke drew something"

    # Space goes down mid-drag and the pan takes the mouse.
    state.space_held = True
    frame((14, 14), down=(0,), dragging=(0,))
    assert state.drag_kind == "pan"

    assert tab.doc.history.head != head, "the stroke was committed, not stranded"
    assert tab.doc.history.can_undo
    tab.doc.undo()
    assert int((tab.doc.stack.active.pixels[..., 3] > 0).sum()) == 0


def test_a_middle_drag_mid_stroke_closes_the_stroke_too(driven):
    state, tab, frame = driven
    state.tool = "brush"
    head = tab.doc.history.head

    frame((4, 4), click=0, down=(0,))
    frame((10, 10), down=(0,))
    frame((11, 11), down=(0, 2), dragging=(2,))

    assert state.drag_kind == "pan"
    assert tab.doc.history.head != head
    tab.doc.undo()
    assert int((tab.doc.stack.active.pixels[..., 3] > 0).sum()) == 0


def test_a_tab_going_busy_mid_stroke_closes_the_stroke(driven):
    """The busy gate cleared the multi-click gesture and returned, but a live
    ``drag_kind`` survived with its stroke open -- the pan's failure by a third
    door."""
    state, tab, frame = driven
    state.tool = "brush"
    head = tab.doc.history.head

    frame((4, 4), click=0, down=(0,))
    frame((10, 10), down=(0,))
    assert state.drag_kind == "paint"

    tab.busy = True
    frame((11, 11), down=(0,))

    assert state.drag_kind == ""
    assert tab.doc.history.head != head
