"""The shading ink: one step along a ramp, per stroke.

The mode that does not paint the stroke's colour. What it writes is decided by
what is already on the layer, which makes its failure modes different from every
other brush mode's -- so what is asserted here is the four rules the engine's
docstring states, each of which is the difference between a usable tool and one
that eats a drawing:

*One step per stroke*, or scrubbing back and forth over a shoulder runs it off
the end of the ramp before the user can lift the button. *Clamped at the ends*,
or the deepest shadow becomes the brightest highlight in one dab. *Exact
matching*, or a shade turns into a whole-drawing conversion. And *the ramp is
the palette selection*, in palette order, or the same five swatches mean five
different things depending on the order they were clicked in.

The rest -- symmetry, one undo step, the indexed snap -- is asserted rather than
assumed for ``test_spray.py``'s reason: it comes free from going through
``_dab``/``_commit_patch``, and "free" is a claim about code that can change.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from warlock.studio import inker, inker_state
from warlock.studio.inker import brush, indexed
from warlock.studio.panes import inker_canvas, inker_tools

SIZE = (24, 24)

#: A four-step grey ramp, dark to light. Deliberately not evenly spaced in a way
#: any nearest-match could reproduce by accident: the assertions below are about
#: *which entry* a pixel landed on, not about how far it moved.
RAMP = [
    (20, 20, 20, 255),
    (80, 70, 60, 255),
    (150, 140, 120, 255),
    (240, 235, 220, 255),
]
#: A colour on the palette but deliberately *off* any selected ramp.
OTHER = (10, 90, 200, 255)


def _doc(palette=None, fill=None):
    """A document painted flat in one palette colour, with the palette set.

    ``set_palette`` would snap the pixels; the fill is written first and is
    already exactly on the table, so the snap is the no-op this file also
    asserts about a shade stroke.
    """
    doc = inker.Document.blank(*SIZE)
    if fill is not None:
        doc.stack.active.pixels[:, :] = np.asarray(fill, dtype=np.uint8)
    doc.set_palette(palette if palette is not None else [*RAMP, OTHER])
    doc.history.clear()
    return doc


def _shade(doc, *, slots=(), direction=1, points=((12.0, 12.0),), size=9, **kw):
    ramp = indexed.shade_ramp(doc.palette, slots)
    doc.begin_stroke(
        points[0],
        (0, 0, 0, 255),
        size=size,
        nib="square",
        mode="shade",
        ramp=ramp,
        shade_dir=direction,
        **kw,
    )
    for point in points[1:]:
        doc.stroke_to(point)
    doc.end_stroke()
    return doc.stack.active.pixels


def _at(pixels, xy=(12, 12)):
    return tuple(int(c) for c in pixels[xy[1], xy[0]])


# --- the ramp ---------------------------------------------------------------


def test_the_ramp_is_the_selection_in_palette_order_not_click_order():
    """Picking the light end first describes the same ramp as picking the dark
    end first -- the direction toggle is what reverses it, not the click order."""
    palette = [*RAMP, OTHER]
    assert indexed.shade_ramp(palette, [3, 0, 1]) == [RAMP[0], RAMP[1], RAMP[3]]
    assert indexed.shade_ramp(palette, [1, 3, 0]) == indexed.shade_ramp(palette, [0, 1, 3])


def test_the_selected_slots_are_adjacent_steps_however_far_apart_they_sit():
    """Slots 0 and 3 are one step apart *on the ramp*, which is what makes a
    ramp pickable out of a table that holds several."""
    ramp = indexed.shade_ramp([*RAMP, OTHER], [0, 3])
    assert ramp == [RAMP[0], RAMP[3]]


def test_fewer_than_two_slots_falls_back_to_the_whole_palette():
    palette = [*RAMP, OTHER]
    assert indexed.shade_ramp(palette, []) == palette
    assert indexed.shade_ramp(palette, [2]) == palette
    # And a slot the palette no longer has is dropped rather than raising, which
    # is what an undone Remove leaves behind in ``palette_slots``.
    assert indexed.shade_ramp(palette, [2, 99]) == palette


def test_a_document_with_no_palette_has_no_ramp():
    assert indexed.shade_ramp(None, [0, 1]) == []
    assert indexed.shade_ramp([], [0, 1]) == []


# --- one step ---------------------------------------------------------------


def test_a_dab_moves_a_covered_pixel_exactly_one_swatch():
    doc = _doc(fill=RAMP[1])
    pixels = _shade(doc, slots=[0, 1, 2, 3])
    assert _at(pixels) == RAMP[2]


def test_the_other_direction_moves_back_along_the_ramp():
    doc = _doc(fill=RAMP[2])
    pixels = _shade(doc, slots=[0, 1, 2, 3], direction=-1)
    assert _at(pixels) == RAMP[1]


def test_a_whole_dragged_stroke_is_still_one_step():
    """The rule the tool lives or dies by: dabs land on top of each other every
    few pixels, and without ``_shifted`` a slow drag would walk a pixel to the
    end of the ramp in the time it takes to notice."""
    doc = _doc(fill=RAMP[0])
    pixels = _shade(
        doc,
        slots=[0, 1, 2, 3],
        points=[(12.0, 12.0), (13.0, 12.0), (12.0, 12.0), (13.0, 12.0), (12.0, 12.0)],
    )
    assert _at(pixels) == RAMP[1]


def test_a_second_stroke_takes_a_second_step():
    """Per *stroke*, not per document: lifting the button and dragging again is
    how the user asks for another step."""
    doc = _doc(fill=RAMP[0])
    _shade(doc, slots=[0, 1, 2, 3])
    pixels = _shade(doc, slots=[0, 1, 2, 3])
    assert _at(pixels) == RAMP[2]


def test_the_shifted_plane_belongs_to_a_shade_stroke_alone():
    """Every other mode pays nothing for it."""
    blank = np.zeros((SIZE[1], SIZE[0], 4), np.uint8)
    plain = brush.StrokeState(layer_uid=1, size=SIZE, before=blank, colour=OTHER)
    assert plain._shifted is None
    shading = brush.StrokeState(
        layer_uid=1, size=SIZE, before=blank, colour=OTHER, mode="shade", ramp=tuple(RAMP)
    )
    assert shading._shifted is not None
    assert shading._shifted.shape == (SIZE[1], SIZE[0])
    assert shading._shifted.dtype == np.dtype(bool)


# --- clamping ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("fill", "direction"),
    [(RAMP[-1], 1), (RAMP[0], -1)],
)
def test_a_pixel_at_the_end_of_the_ramp_stays_there(fill, direction):
    """Clamped rather than wrapped: sending the deepest shadow to the brightest
    highlight in one dab reads as corruption, not as a tool."""
    doc = _doc(fill=fill)
    pixels = _shade(doc, slots=[0, 1, 2, 3], direction=direction)
    assert _at(pixels) == fill


def test_clamping_leaves_no_undo_step_because_nothing_changed():
    doc = _doc(fill=RAMP[-1])
    _shade(doc, slots=[0, 1, 2, 3])
    assert not doc.history.can_undo


# --- what is left alone -----------------------------------------------------


def test_a_colour_that_is_not_on_the_ramp_is_untouched():
    """Exact match, never nearest: nearest would drag every colour in the
    drawing onto a swatch, which is a conversion rather than a shade."""
    doc = _doc(fill=OTHER)
    pixels = _shade(doc, slots=[0, 1, 2, 3])
    assert _at(pixels) == OTHER


def test_a_palette_colour_outside_the_selection_is_untouched():
    """The selection gate. Slots 0 and 1 are the ramp; a pixel painted in slot 2
    is on the palette and still not on the ramp."""
    doc = _doc(fill=RAMP[2])
    pixels = _shade(doc, slots=[0, 1])
    assert _at(pixels) == RAMP[2]
    # And the same document with the full selection does move it, so the test
    # above is about the selection rather than about the colour.
    assert _at(_shade(_doc(fill=RAMP[2]), slots=[0, 1, 2, 3])) == RAMP[3]


def test_a_transparent_pixel_is_left_alone():
    """It has no colour to be on the ramp, and writing its dead RGB would make
    an empty layer look edited."""
    doc = _doc()
    pixels = _shade(doc, slots=[0, 1, 2, 3])
    assert int(pixels[..., 3].max()) == 0
    assert not doc.history.can_undo


def test_alpha_is_never_touched_so_the_alpha_lock_holds_trivially():
    doc = _doc(fill=(*RAMP[1][:3], 128))
    doc.stack.active.alpha_lock = True
    pixels = _shade(doc, slots=[0, 1, 2, 3])
    assert _at(pixels) == (*RAMP[2][:3], 128)


def test_a_pixel_the_brush_does_not_cover_is_untouched():
    doc = _doc(fill=RAMP[1])
    pixels = _shade(doc, slots=[0, 1, 2, 3], size=3)
    assert _at(pixels) == RAMP[2]
    assert _at(pixels, (0, 0)) == RAMP[1]


def test_an_empty_ramp_is_a_no_op_rather_than_a_crash():
    """Reachable only on a document with no palette, which the UI gates -- but
    the engine must not be the thing that discovers that."""
    doc = inker.Document.blank(*SIZE)
    doc.stack.active.pixels[:, :] = np.asarray(RAMP[1], dtype=np.uint8)
    before = doc.stack.active.pixels.copy()
    doc.begin_stroke((12.0, 12.0), OTHER, size=9, nib="square", mode="shade", ramp=())
    doc.end_stroke()
    assert np.array_equal(doc.stack.active.pixels, before)


# --- what comes free --------------------------------------------------------


def test_symmetry_mirrors_a_shade_dab_without_the_mode_knowing():
    """Mirroring happens in ``_dab``, above ``_stamp``, so every mode inherits
    it. Nothing in ``_shade`` mentions symmetry and this is why."""
    doc = _doc(fill=RAMP[1])
    pixels = _shade(doc, slots=[0, 1, 2, 3], size=5, points=[(4.0, 12.0)], symmetry="x")
    assert _at(pixels, (4, 12)) == RAMP[2]
    # ``2a - x`` about the canvas centre sends column 4 to column 19.
    assert _at(pixels, (SIZE[0] - 1 - 4, 12)) == RAMP[2]


def test_a_selection_clips_the_shade():
    doc = _doc(fill=RAMP[1])
    doc.select(inker.SelectionMask.from_rect(SIZE, (0, 0, SIZE[0], 12)))
    pixels = _shade(doc, slots=[0, 1, 2, 3], size=9)
    assert _at(pixels, (12, 11)) == RAMP[2]
    assert _at(pixels, (12, 13)) == RAMP[1]


def test_one_press_to_release_is_one_undo_step():
    doc = _doc(fill=RAMP[0])
    before = doc.stack.active.pixels.copy()
    _shade(
        doc,
        slots=[0, 1, 2, 3],
        points=[(6.0, 12.0), (10.0, 12.0), (14.0, 12.0), (18.0, 12.0)],
    )
    assert doc.history.can_undo
    doc.undo()
    assert np.array_equal(doc.stack.active.pixels, before)
    assert not doc.history.can_undo


def test_the_indexed_snap_is_a_no_op_on_what_a_shade_writes():
    """Every shade writes a colour that is already a palette entry, so the snap
    in ``_commit_patch`` has nothing to do -- which is what stops the ink and the
    palette constraint disagreeing about what a stroke produced."""
    doc = _doc(fill=RAMP[1])
    pixels = _shade(doc, slots=[0, 1, 2, 3], size=9)
    assert np.array_equal(indexed.snap(pixels, doc.palette), pixels)
    assert _at(pixels) == RAMP[2]


def test_a_shade_stroke_wraps_when_the_document_is_tiled():
    doc = _doc(fill=RAMP[1])
    _shade(doc, slots=[0, 1, 2, 3], size=7, points=[(0.0, 12.0)], wrap="x")
    assert _at(doc.stack.active.pixels, (SIZE[0] - 1, 12)) == RAMP[2]


# --- the tool row -----------------------------------------------------------


def test_the_shading_tool_asks_the_engine_for_a_mode_it_has():
    assert inker_state.BRUSH_MODES["shade"] == "shade"
    assert "shade" in brush.MODES
    assert "shade" in inker_state.PAINT_TOOLS


def test_the_tool_is_refused_with_a_reason_until_the_document_is_indexed():
    """A reason string rather than a silent no-op: a greyed button with nothing
    to say is indistinguishable from a broken one."""
    doc = inker.Document.blank(*SIZE)
    assert inker_state.tool_reason("shade", doc) == inker_state.SHADE_REASONS["none"]
    doc.set_palette([RAMP[0]])
    assert inker_state.tool_reason("shade", doc) == inker_state.SHADE_REASONS["one"]
    doc.set_palette(RAMP)
    assert inker_state.tool_reason("shade", doc) == ""


def test_no_other_tool_is_gated_and_no_document_gates_nothing():
    """The two gated tools are shading and the tile stamp; nothing else has a
    document-shaped reason to be out, and no document gates either of them."""
    doc = inker.Document.blank(*SIZE)
    for tool, _label, _key in inker_state.TOOLS:
        if tool not in ("shade", "tile"):
            assert inker_state.tool_reason(tool, doc) == ""
    assert inker_state.tool_reason("shade", None) == ""
    assert inker_state.tool_reason("tile", None) == ""


def test_the_direction_is_a_per_tool_option_at_the_declared_default():
    state = inker_state.InkerState()
    state.tool = "shade"
    assert state.shade_dir == inker_state.TOOL_OPTION_DEFAULTS["shade_dir"] == 1
    state.shade_dir = -1
    state.tool = "brush"
    state.tool = "shade"
    assert state.shade_dir == -1


def test_the_direction_radio_offers_the_values_the_engine_takes():
    """``test_ui_tables.py``'s rule: a control that offers something the engine
    does not take fails on the first click."""
    values = [value for value, _label in inker_tools.SHADE_LABELS]
    assert values == [1, -1]
    assert inker_state.TOOL_OPTION_DEFAULTS["shade_dir"] in values


# --- the press ---------------------------------------------------------------


class _Ctx:
    """Enough of Ctx for the one thing ``_press`` asks of it."""

    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def toast(self, text: str, kind: str = "info") -> None:
        self.toasts.append((text, kind))


@pytest.fixture
def scene(monkeypatch):
    """A pane state and a tab, with the one imgui call ``_press`` makes stubbed."""
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=False, key_alt=False, key_ctrl=False),
    )
    state = inker_state.InkerState()
    state.tool = "shade"
    tab = SimpleNamespace(doc=inker.Document.blank(*SIZE), tiled="off", busy=False)
    return state, tab, _Ctx()


def test_a_shade_press_on_a_document_with_no_palette_is_refused_out_loud(scene):
    """The panel greys the button, but a shortcut key selects a tool without
    asking the panel anything -- so the press is the door that has to refuse."""
    state, tab, ctx = scene
    inker_canvas._press(ctx, state, tab, (12.0, 12.0))
    assert state.drag_kind == ""
    assert tab.doc._stroke is None
    assert ctx.toasts == [(inker_state.SHADE_REASONS["none"], "warn")]


def test_the_press_hands_the_engine_the_selected_ramp_and_direction(scene):
    state, tab, ctx = scene
    tab.doc.set_palette([*RAMP, OTHER])
    state.palette_slots = [1, 0]
    state.shade_dir = -1

    inker_canvas._press(ctx, state, tab, (12.0, 12.0))
    stroke = tab.doc._stroke
    assert ctx.toasts == []
    assert state.drag_kind == "paint"
    assert stroke is not None
    assert stroke.mode == "shade"
    # Palette order, not the order the slots were clicked in.
    assert stroke.ramp == (RAMP[0], RAMP[1])
    assert stroke.shade_dir == -1


def test_every_other_tool_still_opens_the_stroke_it_always_did(scene):
    """The refusal is the shading tool's alone: the gate is at the top of the
    paint branch, which every painting tool goes through."""
    state, tab, ctx = scene
    state.tool = "brush"
    inker_canvas._press(ctx, state, tab, (12.0, 12.0))
    assert ctx.toasts == []
    assert state.drag_kind == "paint"
    assert tab.doc._stroke is not None and tab.doc._stroke.mode == "paint"
