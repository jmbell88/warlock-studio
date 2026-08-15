"""The canvas's half of tiled painting, the spray and the right button.

Everything here is a pure function of the pane's own state, which is the point:
the drawing is imgui's problem, but *which point the tool was handed* and *which
colour it was given* are decisions, and each of them has a failure that looks
like a working editor -- a brush that jumps a tile at the seam, a right-drag
that ends on the wrong frame, a spray whose dab is as wide as its cloud.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import inker, inker_state
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
        inker_canvas.imgui, "get_io", lambda: SimpleNamespace(key_shift=False, key_alt=False)
    )
    state = inker_state.InkerState(fg=FG, bg=BG)
    tab = SimpleNamespace(doc=inker.Document.blank(*SIZE), tiled="off", busy=False)
    return state, tab


def _press(state, tab, point):
    inker_canvas._press(None, state, tab, point)


def _alt(monkeypatch, held=True):
    monkeypatch.setattr(
        inker_canvas.imgui, "get_io", lambda: SimpleNamespace(key_shift=False, key_alt=held)
    )


# --- the tables -------------------------------------------------------------


def test_the_tiled_combo_offers_every_mode_the_engine_implements():
    assert tuple(key for key, _label in inker_canvas.TILED_LABELS) == tiling.TILED_AXES


def test_the_ink_radio_names_a_real_brush_mode():
    """``blend`` is deliberately *not* a mode -- it is the composite every
    stroke has always done -- so only the other key has to exist."""
    keys = [key for key, _label in inker_tools.INK_LABELS]
    assert keys == ["blend", "replace"]
    assert "replace" in brush.MODES and "blend" not in brush.MODES


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
