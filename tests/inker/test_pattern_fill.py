"""Pattern fills: the bucket pours the captured tip instead of the swatch.

**A pattern is a stamp used as a fill source**, and that framing is what this
file asserts as much as the pixels are. There is no second image-tiling
mechanism: the source is ``state.stamp`` -- the one captured tip the brush
already stamps -- the lattice option is ``brush.STAMP_ALIGN``, and the write
goes through ``write_colour``, the door the bucket and all six shapes already
came through. So the content lock, the alpha lock, the cel autovivification and
the single undo patch are the ones that already existed, and there is nothing
here for them to be asserted a second time about.

Two things this file does that the arithmetic alone would not catch:

* **The standing negative control.** A document that never uses the feature
  must be unchanged to the byte -- same pixels, same history, same ``.ora``.
  The new parameter is inert when it is ``None`` and that is pinned rather than
  assumed, because "an optional argument nobody passes" is exactly the shape of
  change that quietly moves a stored corpus.
* **The real input path.** A control that is drawn and wired to nothing is this
  codebase's most common historical defect, so the checkbox is *pressed* inside
  a real imgui frame through :mod:`.probe`, and the fill is made by handing the
  pane's own press dispatch a point -- not by calling ``Document.fill``.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from warlock.studio import inker_mode, inker_state, probe, widgets
from warlock.studio.inker import asein, aseout, ora
from warlock.studio.inker import composite as cp
from warlock.studio.inker.brush import STAMP_ALIGN, Stamp
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask
from warlock.studio.panes import inker_canvas, inker_context, inker_tools

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
YELLOW = (255, 255, 0, 255)
GREY = (10, 10, 10, 255)


def _tile() -> np.ndarray:
    """A 2x2 whose four cells are four different colours.

    Four rather than two, and asymmetric in both axes, because a two-colour
    checker cannot tell a lattice that is one cell out from one that is right.
    """
    tile = np.zeros((2, 2, 4), dtype=np.uint8)
    tile[0, 0] = RED
    tile[0, 1] = GREEN
    tile[1, 0] = BLUE
    tile[1, 1] = YELLOW
    return tile


def _stamp() -> Stamp:
    return Stamp(_tile())


def _doc(width: int = 8, height: int = 8) -> Document:
    doc = Document.blank(width, height)
    doc.stack[0].pixels[:, :] = GREY
    doc.invalidate_all()
    return doc


# --- the lattice ------------------------------------------------------------


def test_an_aligned_pattern_is_anchored_on_the_canvas_origin():
    """Which is what makes two fills in two corners one pattern."""
    doc = _doc()
    assert doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    art = doc.stack[0].pixels
    for y in range(8):
        for x in range(8):
            assert tuple(art[y, x]) == tuple(_tile()[y % 2, x % 2]), (x, y)


def test_an_aligned_fill_of_a_region_that_does_not_start_at_the_origin_still_lines_up():
    """The lattice belongs to the canvas, not to the thing being filled."""
    doc = _doc()
    # Two regions, filled separately, that between them cover the canvas.
    doc.stack[0].pixels[:, 4:] = BLUE
    doc.invalidate_all()
    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    doc.fill((5, 0), RED, pattern=_stamp(), pattern_align="aligned")
    art = doc.stack[0].pixels
    # Column 4 is a fresh region's left edge and still carries column 4's cell.
    assert tuple(art[0, 4]) == RED
    assert tuple(art[0, 5]) == GREEN


def test_a_free_pattern_starts_at_the_corner_of_what_was_filled():
    doc = _doc()
    doc.stack[0].pixels[:, :5] = BLUE
    doc.invalidate_all()
    # The region is x >= 5, so its own top-left is (5, 0) and the tile's first
    # cell lands there -- one column later than the canvas lattice would put it.
    doc.fill((6, 0), RED, pattern=_stamp(), pattern_align="free")
    art = doc.stack[0].pixels
    assert tuple(art[0, 5]) == RED
    assert tuple(art[0, 6]) == GREEN


def test_an_unknown_alignment_falls_back_to_free_rather_than_refusing():
    """``StrokeState.__post_init__``'s own rule: a stale settings value is not
    a reason to fail a click."""
    doc = _doc()
    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="pattern")
    free = doc.stack[0].pixels.copy()
    other = _doc()
    other.fill((0, 0), RED, pattern=_stamp(), pattern_align=STAMP_ALIGN[0])
    assert np.array_equal(free, other.stack[0].pixels)


def test_the_default_alignment_is_the_one_the_brush_defaults_to():
    """One option, one default -- or the bucket and the brush disagree about
    what the setting they share means."""
    doc, other = _doc(), _doc()
    doc.fill((0, 0), RED, pattern=_stamp())
    other.fill((0, 0), RED, pattern=_stamp(), pattern_align=STAMP_ALIGN[0])
    assert np.array_equal(doc.stack[0].pixels, other.stack[0].pixels)


def test_tiled_hands_back_a_copy_rather_than_a_view_of_the_read_only_image():
    stamp = _stamp()
    out = stamp.tiled((0, 0, 4, 4))
    out[0, 0] = (1, 2, 3, 4)
    assert tuple(stamp.image[0, 0]) == RED


# --- what a pattern fill is, and is not -------------------------------------


def test_a_pattern_fill_changes_the_colours_and_not_the_region():
    """The wand still decides which pixels; the pattern only decides what."""
    doc = _doc()
    doc.stack[0].pixels[0:4, 0:4] = BLUE
    doc.invalidate_all()
    flat = _doc()
    flat.stack[0].pixels[0:4, 0:4] = BLUE
    flat.invalidate_all()

    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    flat.fill((0, 0), RED)
    touched = np.any(doc.stack[0].pixels != GREY, axis=-1)
    flat_touched = np.any(flat.stack[0].pixels != GREY, axis=-1)
    assert np.array_equal(touched, flat_touched)
    assert not np.array_equal(doc.stack[0].pixels, flat.stack[0].pixels)


def test_a_transparent_cell_of_the_pattern_leaves_the_pixel_alone():
    """The tip's own alpha is its shape, exactly as it is for a dab."""
    tile = _tile()
    tile[1, 1, 3] = 0
    doc = _doc()
    doc.fill((0, 0), RED, pattern=Stamp(tile), pattern_align="aligned")
    assert tuple(doc.stack[0].pixels[1, 1]) == GREY
    assert tuple(doc.stack[0].pixels[0, 0]) == RED


def test_a_pattern_fill_is_one_undo_step():
    doc = _doc()
    before = doc.stack[0].pixels.copy()
    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    assert doc.history.can_undo
    doc.undo()
    assert np.array_equal(doc.stack[0].pixels, before)
    assert not doc.history.can_undo


def test_a_pattern_fill_is_refused_on_a_content_locked_layer():
    """Through the door that already checks, rather than a second check."""
    doc = _doc()
    doc.stack[0].locked = True
    assert not doc.fill((0, 0), RED, pattern=_stamp())
    assert tuple(doc.stack[0].pixels[0, 0]) == GREY


def test_the_alpha_lock_still_holds_against_a_pattern():
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (0, 0, 0, 0)
    doc.stack[0].pixels[0:2, 0:2] = GREY
    doc.stack[0].alpha_lock = True
    doc.invalidate_all()
    doc.fill((3, 3), RED, pattern=_stamp(), pattern_align="aligned")
    assert doc.stack[0].pixels[3, 3, 3] == 0


def test_a_feathered_selection_still_fades_a_pattern_fill_in():
    doc = _doc(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, :] = 128
    doc.select(SelectionMask(mask))
    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    # Half of red over grey, not red.
    assert tuple(doc.stack[0].pixels[0, 0]) != RED
    assert doc.stack[0].pixels[0, 0][0] > GREY[0]


# --- the two selection ops --------------------------------------------------


def test_fill_selection_takes_a_pattern():
    doc = _doc()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:4, 0:4] = 255
    doc.select(SelectionMask(mask))
    assert doc.fill_selection(RED, pattern=_stamp(), pattern_align="aligned")
    assert tuple(doc.stack[0].pixels[0, 1]) == GREEN
    assert tuple(doc.stack[0].pixels[0, 5]) == GREY


def test_stroke_selection_takes_a_pattern_on_the_canvas_lattice():
    """A patterned outline and a patterned fill of one selection are two cuts
    of one pattern, not two patterns meeting at the edge."""
    doc = _doc()
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:7, 1:7] = 255
    doc.select(SelectionMask(mask))
    assert doc.stroke_selection(RED, 1, pattern=_stamp(), pattern_align="aligned")
    art = doc.stack[0].pixels
    assert tuple(art[1, 1]) == tuple(_tile()[1, 1])
    assert tuple(art[1, 2]) == tuple(_tile()[1, 0])
    # Inside the band is untouched, which is what "inside the selection" means.
    assert tuple(art[3, 3]) == GREY


# --- the standing negative control ------------------------------------------


def _saved(doc: Document) -> bytes:
    return ora.ora_bytes(doc)


def _flat_filled(**kwargs) -> Document:
    doc = _doc()
    doc.stack[0].pixels[0:4, 0:4] = BLUE
    doc.invalidate_all()
    doc.fill((0, 0), RED, **kwargs)
    return doc


def test_a_document_that_never_uses_a_pattern_writes_the_same_ora_bytes():
    """The standing negative control for this track.

    Passing the new keywords explicitly with no pattern must be the same file
    as not passing them at all -- and both must be the same file as each other
    twice over, which is the determinism pin these bytes already carry.
    """
    plain = _saved(_flat_filled())
    explicit = _saved(_flat_filled(pattern=None, pattern_align="aligned"))
    assert plain == explicit
    assert plain == _saved(_flat_filled())


def test_a_captured_tip_that_is_not_poured_changes_no_bytes_either():
    """The tip is app state and never reaches the document: a session with one
    captured and the switch off writes what a session with none writes."""
    state = inker_state.InkerState()
    state.stamp = _stamp()
    assert state.pattern_for("fill") is None
    assert _saved(_flat_filled()) == _saved(_flat_filled())


def test_using_the_feature_does_change_the_bytes():
    """The control is worth nothing without its positive."""
    used = _flat_filled(pattern=_stamp(), pattern_align="aligned")
    assert _saved(used) != _saved(_flat_filled())


def test_the_flat_path_is_still_the_flat_arithmetic_to_the_bit():
    """``write_colour`` with no pattern must be the call it always was."""
    doc = _doc(4, 4)
    weight = np.full((4, 4), 0.4, dtype=np.float32)
    before = doc.stack[0].pixels.astype(np.float32)
    expected = cp.to_uint8_255(cp.paint_colour(before, RED, weight))
    doc.write_colour((0, 0, 4, 4), RED, weight)
    assert np.array_equal(doc.stack[0].pixels, expected)


def test_a_pattern_of_one_flat_opaque_colour_is_the_flat_fill():
    """The two formulas are the same formula, so the degenerate pattern lands
    exactly where the swatch does."""
    tile = np.zeros((2, 2, 4), dtype=np.uint8)
    tile[..., :] = RED
    patterned = _flat_filled(pattern=Stamp(tile), pattern_align="aligned")
    flat = _flat_filled()
    assert np.array_equal(patterned.stack[0].pixels, flat.stack[0].pixels)


# --- the record is the pixels -----------------------------------------------


def test_an_aseprite_write_carries_the_rasterised_result_and_no_pattern_chunk():
    """A pattern fill resolves to its pixels on write -- the flatten matte's
    posture, and the only one the format can hold: there is no pattern chunk in
    ``.aseprite`` and inventing one would be a file real Aseprite cannot read.
    """
    doc = _doc()
    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    expected = doc.stack[0].pixels.copy()
    back, warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert warnings == []
    assert np.array_equal(back.stack[0].pixels, expected)


def test_an_ora_round_trip_carries_the_rasterised_result_too(tmp_path):
    doc = _doc()
    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    expected = doc.stack[0].pixels.copy()
    path = tmp_path / "doc.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)
    assert np.array_equal(back.stack[0].pixels, expected)


def test_the_saved_file_holds_no_member_naming_a_pattern():
    """The pixels *are* the record: nothing new is stored beside them."""
    doc = _doc()
    doc.fill((0, 0), RED, pattern=_stamp(), pattern_align="aligned")
    with zipfile.ZipFile(BytesIO(_saved(doc))) as zf:
        assert not [name for name in zf.namelist() if "pattern" in name.lower()]


# --- the wiring -------------------------------------------------------------


def test_the_bucket_is_a_pattern_tool_and_not_a_stamp_tool():
    """``STAMP_TOOLS`` stays a derivation of the brush modes; the bucket has
    no brush mode at all, so it is named separately."""
    assert set(inker_state.PATTERN_TOOLS) == {"fill"}
    assert not inker_state.PATTERN_TOOLS & inker_state.STAMP_TOOLS


def test_the_bucket_answers_pattern_for_and_not_tip_for():
    """A bucket that answered ``tip_for`` would put a tip-shaped ring under a
    cursor that stamps nothing."""
    state = inker_state.InkerState()
    state.stamp = _stamp()
    state.options_for("fill")["use_stamp"] = True
    assert state.pattern_for("fill") is state.stamp
    assert state.tip_for("fill") is None
    assert state.pattern_for("brush") is None


def test_the_switch_is_off_until_it_is_turned_on():
    state = inker_state.InkerState()
    state.stamp = _stamp()
    assert state.pattern_for("fill") is None


def test_forgetting_the_tip_clears_the_buckets_switch_too():
    state = inker_state.InkerState()
    state.stamp = _stamp()
    state.options_for("fill")["use_stamp"] = True
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))
    inker_mode.clear_brush(ctx)
    assert state.options_for("fill")["use_stamp"] is False
    assert state.stamp is None


def test_capturing_with_the_bucket_in_hand_keeps_the_bucket():
    """Capturing from the fill tool is somebody building a pattern to pour."""
    doc = _doc()
    doc.select_all()
    state = inker_state.InkerState()
    state.set_tool("fill")
    tab = SimpleNamespace(doc=doc, busy=False, uid="t")
    state.docs.append(tab)
    state.active_uid = "t"
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state), toast=lambda *a, **k: None)
    assert inker_mode.capture_brush(ctx)
    assert state.tool == "fill"
    assert state.pattern_for("fill") is state.stamp


def test_capturing_from_a_selection_tool_still_picks_the_brush_up():
    doc = _doc()
    doc.select_all()
    state = inker_state.InkerState()
    state.set_tool("marquee")
    tab = SimpleNamespace(doc=doc, busy=False, uid="t")
    state.docs.append(tab)
    state.active_uid = "t"
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state), toast=lambda *a, **k: None)
    assert inker_mode.capture_brush(ctx)
    assert state.tool == "brush"


def test_the_context_bars_placement_combo_offers_the_two_that_exist():
    """It used to offer ``origin`` and ``tile``, which the engine snapped back
    to ``free`` -- two settings that did nothing."""
    assert tuple(key for key, _l in inker_context.ALIGN_LABELS) == STAMP_ALIGN
    assert tuple(k for k, _l in inker_tools.STAMP_ALIGN_LABELS) == STAMP_ALIGN


def test_the_bucket_shows_the_image_brush_panel():
    """Or the pattern has no door: capture, the variants and the switch all
    live in that section."""
    state = inker_state.InkerState()
    state.set_tool("fill")
    tab = SimpleNamespace(doc=_doc(), busy=False)
    assert inker_tools._has_panels(state, tab)


# --- through the real input path --------------------------------------------


@pytest.fixture
def ui(monkeypatch):
    """An imgui context with the control census on, torn down after.

    Built and destroyed here rather than shared, for ``test_pane_guard``'s
    reason: two imgui contexts over one GL context take the process down. No GL
    is needed -- nothing is rendered, only laid out.
    """
    from imgui_bundle import imgui

    from warlock.studio import theme

    previous = imgui.get_current_context()
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename(None)
    io.display_size = (1600.0, 950.0)
    io.delta_time = 1.0 / 60.0
    io.fonts.add_font_default()
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    theme.apply(imgui)
    monkeypatch.setattr(probe, "ENABLED", True)
    # The image-brush section is collapsed by default -- it is a property of the
    # brush rather than a step in using one -- and a census of a closed header
    # finds nothing. This is the smoke test's own switch for the same reason.
    monkeypatch.setattr(widgets, "FORCE_SECTIONS_OPEN", True)
    yield imgui
    imgui.destroy_context(ctx)
    if previous is not None:
        imgui.set_current_context(previous)


def _pane_state(doc: Document) -> tuple[Any, Any, Any]:
    state = inker_state.InkerState(fg=RED, bg=BLUE)
    state.set_tool("fill")
    state.stamp = _stamp()
    tab = SimpleNamespace(doc=doc, busy=False, tiled="off", range_sel=None, uid="t")
    state.docs.append(tab)
    state.active_uid = "t"
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=state), toast=lambda *a, **k: None, viewer=None
    )
    return ctx, state, tab


def _frame(imgui, build, *, pos=(400.0, 300.0), down=False):
    io = imgui.get_io()
    io.add_mouse_pos_event(pos[0], pos[1])
    io.add_mouse_button_event(0, down)
    probe.begin_frame()
    imgui.new_frame()
    imgui.set_next_window_size((520.0, 900.0))
    imgui.set_next_window_pos((0.0, 0.0))
    imgui.begin("##host")
    build()
    imgui.end()
    imgui.end_frame()
    return list(probe.FRAME_CONTROLS)


def _laid_out(imgui, build, frames=2):
    """Two frames, not one: a fresh popup is laid out once with its items
    hidden to *measure* it, and a rect read from that pass sits in a window
    that is not yet visible -- which is the state in which a press lands on
    nothing."""
    out = []
    for _ in range(frames):
        out = _frame(imgui, build)
    return out


def _panel(ui, ctx, state, tab):
    def build():
        if not ui.is_popup_open("panels"):
            ui.open_popup("panels")
        if ui.begin_popup("panels"):
            inker_tools._image_brush(ctx, state, tab)
            ui.end_popup()

    return build


def _click(imgui, build, pos):
    """Press and release, which is what a checkbox answers to.

    A slider moves under a button that is merely *down*; imgui's button
    behaviour fires a checkbox on the **release** inside it, so a single
    down-frame is a press that never happened.
    """
    _frame(imgui, build, pos=pos, down=True)
    _frame(imgui, build, pos=pos, down=False)


def _control(controls, label):
    found = [c for c in controls if c.label.startswith(label)]
    assert found, [c.label for c in controls]
    return found[0]


def test_the_pattern_checkbox_is_drawn_for_the_bucket(ui):
    ctx, state, tab = _pane_state(_doc())
    controls = _laid_out(ui, _panel(ui, ctx, state, tab))
    box = _control(controls, "Fill with this pattern")
    assert box.kind == "checkbox"
    assert box.enabled and box.visible


def test_pressing_the_checkbox_turns_the_buckets_pattern_on(ui):
    """The press, not the assignment: a checkbox wired to nothing looks exactly
    like this test's assertion passing."""
    ctx, state, tab = _pane_state(_doc())
    build = _panel(ui, ctx, state, tab)
    box = _control(_laid_out(ui, build), "Fill with this pattern")
    assert state.pattern_for("fill") is None

    x, y, w, h = box.hit
    _click(ui, build, (x + w * 0.5, y + h * 0.5))
    assert state.pattern_for("fill") is state.stamp, "the box drew but did nothing"


def test_a_bucket_press_on_the_canvas_pours_the_pattern(ui):
    """End to end: the pane's own press dispatch, with the switch turned on by
    the press above rather than by hand."""
    doc = _doc()
    ctx, state, tab = _pane_state(doc)
    build = _panel(ui, ctx, state, tab)
    box = _control(_laid_out(ui, build), "Fill with this pattern")
    x, y, w, h = box.hit
    _click(ui, build, (x + w * 0.5, y + h * 0.5))
    state.options_for("fill")["stamp_align"] = "aligned"

    inker_canvas._press(ctx, state, tab, (0.0, 0.0))
    art = doc.stack[0].pixels
    assert tuple(art[0, 0]) == RED
    assert tuple(art[0, 1]) == GREEN
    assert tuple(art[1, 0]) == BLUE
    assert doc.history.can_undo


def test_the_same_press_with_the_switch_off_still_fills_flat(ui):
    """The negative control at the input path: an untouched session's click is
    the click it always was."""
    doc = _doc()
    ctx, state, tab = _pane_state(doc)
    inker_canvas._press(ctx, state, tab, (0.0, 0.0))
    assert np.all(doc.stack[0].pixels == np.array(RED, dtype=np.uint8))
