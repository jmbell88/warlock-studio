"""Sorting the table, ramping between two slots, and selecting more than one.

None of it moves a pixel, which is why none of it pushes an undo step: order is
presentation in an indexed document -- the exported ``.gpl`` and the GIF colour
table are what it decides -- and a new swatch is a colour you *may* paint with.
``move_slot`` set that rule and these follow it.
"""

from __future__ import annotations

import pytest

from warlock.studio.inker import indexed as ix
from warlock.studio.inker.document import Document
from warlock.studio.inker_state import InkerState

BLACK = (0, 0, 0, 255)
GREY = (128, 128, 128, 255)
WHITE = (255, 255, 255, 255)
RED = (200, 30, 30, 255)
BLUE = (30, 30, 200, 255)


def _doc(palette) -> Document:
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:] = palette[0]
    doc.set_palette(list(palette))
    return doc


# --- the sort keys -----------------------------------------------------------


def test_every_key_the_engine_names_actually_sorts():
    """A key in the tuple with no branch behind it is a control that silently
    does nothing -- and the pane builds its combo straight off this tuple."""
    palette = [WHITE, RED, BLACK, BLUE, GREY]
    for key in ix.SORT_KEYS:
        order = ix.sort_order(palette, key, counts=[5, 4, 3, 2, 1])
        assert sorted(order) == list(range(len(palette))), key


def test_an_unknown_key_is_refused():
    with pytest.raises(ValueError):
        ix.sort_order([BLACK], "temperature")


def test_sorting_by_brightness_orders_a_ramp():
    doc = _doc([WHITE, BLACK, GREY])
    assert doc.sort_palette("luma") is True
    assert doc.palette == [BLACK, GREY, WHITE]


def test_descending_is_the_same_order_upside_down():
    doc = _doc([WHITE, BLACK, GREY])
    doc.sort_palette("luma", descending=True)
    assert doc.palette == [WHITE, GREY, BLACK]


def test_a_tie_keeps_its_order_in_both_directions():
    """Ties are broken by position, not by the sort's own reversal: sorting
    down and then up has to give the table back, or two identical swatches swap
    places for no reason the user can see."""
    same = [(10, 10, 10, 255), (10, 10, 10, 100), (10, 10, 10, 0)]
    up = ix.sort_order(same, "luma")
    down = ix.sort_order(same, "luma", descending=True)
    assert up == [0, 1, 2]
    assert down == [0, 1, 2]


def test_greys_sort_to_one_end_by_hue_rather_than_scattering():
    """A grey has no hue, and the useful answer is 0 rather than whatever falls
    out of a division by zero."""
    assert ix._hue_saturation(GREY) == (0.0, 0.0)
    orange = (200, 100, 30, 255)
    order = ix.sort_order([orange, GREY, BLUE], "hue")
    assert order[0] == 1


def test_sorting_by_usage_takes_the_counts_it_is_given():
    doc = _doc([BLACK, GREY, WHITE])
    doc.sort_palette("usage", counts=[9, 1, 5])
    assert doc.palette == [GREY, WHITE, BLACK]


def test_sorting_by_usage_with_no_counts_measures_the_document():
    """Asked for rather than kept live -- counting is a walk over every pixel of
    every cel -- but a sort that was handed none must still sort by the real
    figures rather than by zeros."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:] = WHITE
    doc.set_palette([BLACK, GREY, WHITE])
    doc.sort_palette("usage", descending=True)
    assert doc.palette[0] == WHITE


def test_sorting_pushes_no_undo_step_and_moves_no_pixel():
    doc = _doc([WHITE, BLACK, GREY])
    depth = len(doc.history)
    before = doc.stack.active.pixels.copy()

    doc.sort_palette("luma")

    assert len(doc.history) == depth
    assert (doc.stack.active.pixels == before).all()


def test_a_sort_that_changes_nothing_reports_nothing():
    doc = _doc([BLACK, GREY, WHITE])
    assert doc.sort_palette("luma") is False


def test_a_document_with_no_palette_cannot_be_sorted():
    assert Document.blank(4, 4).sort_palette("luma") is False


# --- sorting a subset in place ----------------------------------------------


def test_a_selection_sorts_within_its_own_positions():
    """What makes "sort these five" a thing you can do to the middle of a
    hand-arranged table without the rest of it moving."""
    doc = _doc([RED, WHITE, BLACK, GREY, BLUE])
    doc.sort_palette("luma", indices=[1, 2, 3])
    assert doc.palette == [RED, BLACK, GREY, WHITE, BLUE]


def test_a_non_contiguous_selection_sorts_into_the_slots_it_occupies():
    doc = _doc([WHITE, RED, BLACK, BLUE, GREY])
    doc.sort_palette("luma", indices=[0, 2, 4])
    assert doc.palette[1] == RED
    assert doc.palette[3] == BLUE
    assert [doc.palette[i] for i in (0, 2, 4)] == [BLACK, GREY, WHITE]


def test_one_selected_slot_is_not_a_sort():
    doc = _doc([WHITE, BLACK])
    assert doc.sort_palette("luma", indices=[1]) is False


def test_selected_slots_off_the_end_are_ignored():
    doc = _doc([WHITE, BLACK])
    assert doc.sort_palette("luma", indices=[0, 1, 7]) is True
    assert doc.palette == [BLACK, WHITE]


# --- the ramp ----------------------------------------------------------------


def test_a_ramp_is_inserted_between_the_two_slots():
    doc = _doc([BLACK, WHITE])
    assert doc.insert_ramp(0, 1, 1) is True
    assert doc.palette == [BLACK, (128, 128, 128, 255), WHITE]


def test_the_run_goes_from_the_lower_position_to_the_higher_one():
    """Direction is the table's, not the click order's -- so passing the two the
    other way round produces the same run."""
    forward = _doc([BLACK, WHITE])
    backward = _doc([BLACK, WHITE])
    forward.insert_ramp(0, 1, 3)
    backward.insert_ramp(1, 0, 3)
    assert forward.palette == backward.palette


def test_the_endpoints_are_not_duplicated():
    doc = _doc([BLACK, WHITE])
    doc.insert_ramp(0, 1, 4)
    assert doc.palette[0] == BLACK
    assert doc.palette[-1] == WHITE
    assert doc.palette.count(BLACK) == 1
    assert doc.palette.count(WHITE) == 1


def test_colours_already_in_the_table_are_skipped():
    doc = _doc([BLACK, GREY, WHITE])
    doc.insert_ramp(0, 2, 1)  # the midpoint is already there
    assert doc.palette == [BLACK, GREY, WHITE]


def test_a_ramp_of_more_steps_than_the_gap_does_not_repeat_itself():
    doc = _doc([(0, 0, 0, 255), (3, 3, 3, 255)])
    doc.insert_ramp(0, 1, 12)
    assert len(set(doc.palette)) == len(doc.palette)


def test_a_ramp_that_adds_nothing_reports_nothing():
    doc = _doc([BLACK, GREY, WHITE])
    assert doc.insert_ramp(0, 2, 1) is False


def test_a_ramp_pushes_no_undo_step():
    doc = _doc([BLACK, WHITE])
    depth = len(doc.history)
    doc.insert_ramp(0, 1, 3)
    assert len(doc.history) == depth


def test_ramping_a_slot_to_itself_is_refused():
    doc = _doc([BLACK, WHITE])
    assert doc.insert_ramp(1, 1, 3) is False


def test_a_ramp_of_no_steps_is_refused():
    doc = _doc([BLACK, WHITE])
    assert doc.insert_ramp(0, 1, 0) is False


def test_alpha_rides_the_ramp_with_the_colour():
    out = ix.ramp_between((0, 0, 0, 0), (0, 0, 0, 255), 1)
    assert out == [(0, 0, 0, 128)]


# --- multi-slot selection ----------------------------------------------------


def test_a_plain_click_replaces_the_selection_and_moves_the_anchor():
    state = InkerState()
    state.select_slot(3)
    assert state.palette_slot == 3
    assert state.palette_slots == [3]
    state.select_slot(1)
    assert state.palette_slots == [1]


def test_ctrl_click_toggles():
    state = InkerState()
    state.select_slot(0)
    state.select_slot(2, ctrl=True)
    state.select_slot(4, ctrl=True)
    assert state.palette_slots == [0, 2, 4]
    state.select_slot(2, ctrl=True)
    assert state.palette_slots == [0, 4]


def test_ctrl_click_deselecting_leaves_the_anchor_where_it_is():
    """The anchor is where the next range starts, not a member of the
    selection: moving it on a deselect would range from a slot the user had
    just taken out."""
    state = InkerState()
    state.select_slot(0)
    state.select_slot(5, ctrl=True)
    assert state.palette_slot == 5
    state.select_slot(5, ctrl=True)
    assert state.palette_slot == 5
    assert state.palette_slots == [0]


def test_shift_click_ranges_from_the_anchor():
    state = InkerState()
    state.select_slot(4)
    state.select_slot(1, shift=True)
    assert state.palette_slots == [1, 2, 3, 4]
    assert state.palette_slot == 4, "the anchor does not move on a range"


def test_shift_click_replaces_rather_than_adding():
    state = InkerState()
    state.select_slot(0)
    state.select_slot(2, shift=True)
    state.select_slot(1, shift=True)
    assert state.palette_slots == [0, 1]


def test_a_shrinking_palette_drops_the_slots_it_no_longer_has():
    state = InkerState()
    state.select_slot(1)
    state.select_slot(6, ctrl=True)
    state.clamp_slots(3)
    assert state.palette_slots == [1]
    assert state.palette_slot == 2


def test_selected_slots_falls_back_to_the_anchor():
    state = InkerState()
    state.palette_slot = 2
    assert state.selected_slots == [2]
    state.select_slot(0)
    state.select_slot(1, ctrl=True)
    assert state.selected_slots == [0, 1]


def test_a_usage_count_is_not_shared_between_documents():
    """The cache is app-level and was keyed on ``doc.rev`` alone, so two open
    documents at the same rev with palettes of the same length answered each
    other's counts. "0 px, safe to delete" is the one thing this number must
    never say wrongly."""

    from warlock.studio import inker, inker_state
    from warlock.studio.panes import inker_colors

    state = inker_state.InkerState()
    a = inker_state.InkerDoc(doc=inker.Document.blank(4, 4), uid="ta", title="a")
    b = inker_state.InkerDoc(doc=inker.Document.blank(4, 4), uid="tb", title="b")
    for tab in (a, b):
        tab.doc.set_palette([(1, 2, 3, 255), (4, 5, 6, 255)])
    assert a.doc.rev == b.doc.rev, "the fixture needs them level"

    counts = [7, 0]
    state.palette_usage = (a.uid, a.doc.rev, counts)
    assert inker_colors._usage(state, a, 2) == counts
    assert state.palette_usage is not None

    state.palette_usage = (a.uid, a.doc.rev, counts)
    assert inker_colors._usage(state, b, 2) is None, "the other document gets nothing"
    assert state.palette_usage is None, "and the stale entry is dropped"


def test_matte_for_answers_none_for_a_plane_with_no_alpha():
    """The guard was the crash it was guarding against."""
    import numpy as np

    from warlock.studio import inker

    assert inker.matte_for(np.zeros((4, 4), np.uint8)) is None
    assert inker.matte_for(np.zeros((4, 4, 3), np.uint8)) is None
