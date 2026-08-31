"""Discontiguous layer multi-select: Ctrl+click on a layer name.

The timeline could select a *span* of tracks -- one rectangle, two bounds --
and layers 1, 3 and 6 was not expressible. It is now, and the interesting part
is how the two ways a layer gets selected are kept from disagreeing.

They are not two selections. ``InkerDoc.track_sel`` is an **override** and the
cell marquee's track span is the **default**, resolved in one accessor
(``track_rows``) that every verb reads. Empty means no override, so there is
nothing to fall out of step with; not empty means it wins outright, and the
gesture that sets it clears the marquee because a discontiguous set is not a
rectangle and an outline claiming otherwise would be a lie about what the
verbs will touch.

The engine half is ``set_tracks_props``, which is ``set_range_props`` with a
set instead of a span -- the span form is kept and delegates, so every existing
caller and its tests are untouched.
"""

from __future__ import annotations

import pytest

from warlock.studio import inker_state
from warlock.studio.inker.document import Document
from warlock.studio.panes import inker_timeline as tl


def _tab(tracks: int = 5) -> inker_state.InkerDoc:
    doc = Document.blank(8, 8)
    for index in range(1, tracks):
        doc.add_layer(f"L{index}")
    doc.invalidate_all()
    doc.ensure_animation()
    return inker_state.InkerDoc(doc=doc, title="t")


# --- the accessor ------------------------------------------------------------


def test_nothing_selected_is_no_rows():
    tab = _tab()
    assert tab.track_sel == set()
    assert tl.track_rows(tab, tab.doc) == []


def test_the_marquee_span_is_the_default():
    tab = _tab()
    tab.range_sel = (1, 3, 0, 0)
    assert tl.track_rows(tab, tab.doc) == [1, 2, 3]


def test_an_explicit_set_wins_over_the_marquee():
    tab = _tab()
    tab.range_sel = (1, 3, 0, 0)
    tab.track_sel = {0, 4}
    # Outright, not merged: a precedence chain, not two selections unioned.
    assert tl.track_rows(tab, tab.doc) == [0, 4]


def test_the_rows_come_back_in_stack_order():
    tab = _tab()
    tab.track_sel = {4, 0, 2}
    assert tl.track_rows(tab, tab.doc) == [0, 2, 4]


def test_a_row_a_delete_took_away_is_dropped_at_use():
    """Stored unclamped on purpose -- ``range_sel``'s rule and its reason."""
    tab = _tab(tracks=3)
    tab.track_sel = {0, 2, 9}
    assert tl.track_rows(tab, tab.doc) == [0, 2]


# --- the gesture -------------------------------------------------------------


def test_ctrl_click_seeds_from_what_is_already_selected():
    """Widening a marquee rather than throwing it away is what makes a
    multi-select feel like every other list a user has met."""
    tab = _tab()
    tab.range_sel = (1, 2, 0, 0)
    assert tl.toggle_track(tab, tab.doc, 4) is True
    assert tl.track_rows(tab, tab.doc) == [1, 2, 4]


def test_ctrl_click_on_a_selected_row_takes_it_out():
    tab = _tab()
    tab.track_sel = {0, 2, 4}
    tl.toggle_track(tab, tab.doc, 2)
    assert tl.track_rows(tab, tab.doc) == [0, 4]


def test_ctrl_click_clears_the_cell_marquee():
    """The two cannot both be true, and a stale outline would claim cells the
    verbs are not going to touch."""
    tab = _tab()
    tab.range_sel = (1, 2, 0, 0)
    tl.toggle_track(tab, tab.doc, 4)
    assert tab.range_sel is None


def test_toggling_a_row_that_is_not_there_is_refused():
    tab = _tab(tracks=3)
    assert tl.toggle_track(tab, tab.doc, 9) is False
    assert tab.track_sel == set()


def test_clearing_drops_both_halves():
    tab = _tab()
    tab.range_sel = (1, 2, 0, 0)
    tab.track_sel = {0, 4}
    tl.clear_track_selection(tab)
    assert tab.track_sel == set()
    assert tab.range_sel is None


# --- what the verbs act on ---------------------------------------------------


def test_row_targets_is_the_selection_when_the_click_is_inside_it():
    tab = _tab()
    tab.track_sel = {0, 2, 4}
    assert tl.row_targets(tab, tab.doc, 2) == [0, 2, 4]


def test_row_targets_is_the_clicked_row_when_the_click_is_outside():
    tab = _tab()
    tab.track_sel = {0, 2}
    assert tl.row_targets(tab, tab.doc, 4) == [4]


def test_a_single_selected_row_targets_only_itself():
    """``len(rows) > 1`` rather than "is there a selection": a one-row
    selection and no selection must mean the same thing to a verb, or the
    menu's "Delete 1 layers" label appears."""
    tab = _tab()
    tab.track_sel = {2}
    assert tl.row_targets(tab, tab.doc, 2) == [2]


# --- the engine door ---------------------------------------------------------


def test_set_tracks_props_writes_a_discontiguous_set():
    tab = _tab()
    doc = tab.doc
    assert doc.set_tracks_props([0, 2, 4], visible=False) is True
    assert [t.visible for t in doc.anim.tracks] == [False, True, False, True, False]


def test_set_tracks_props_is_one_undo_step():
    tab = _tab()
    doc = tab.doc
    head = doc.history.head
    doc.set_tracks_props([0, 2, 4], visible=False)
    assert doc.history.head == head + 1
    doc.undo()
    assert [t.visible for t in doc.anim.tracks] == [True] * 5


def test_set_tracks_props_changing_nothing_pushes_nothing():
    tab = _tab()
    doc = tab.doc
    head = doc.history.head
    assert doc.set_tracks_props([0, 2], visible=True) is False
    assert doc.history.head == head


def test_out_of_range_and_duplicate_rows_are_dropped_not_refused():
    tab = _tab(tracks=3)
    doc = tab.doc
    assert doc.set_tracks_props([0, 0, 9, -1], visible=False) is True
    assert [t.visible for t in doc.anim.tracks] == [False, True, True]


def test_an_empty_set_is_refused():
    tab = _tab()
    assert tab.doc.set_tracks_props([], visible=False) is False


def test_an_unknown_property_still_raises():
    tab = _tab()
    with pytest.raises(ValueError, match="unknown track property"):
        tab.doc.set_tracks_props([0], nonsense=True)


def test_the_span_door_still_works_and_delegates():
    """``set_range_props`` is kept because a rectangle is what a cell marquee
    produces and what most callers already have."""
    tab = _tab()
    doc = tab.doc
    assert doc.set_range_props(1, 3, visible=False) is True
    assert [t.visible for t in doc.anim.tracks] == [True, False, False, False, True]


def test_the_span_door_still_refuses_a_still_document():
    doc = Document.blank(8, 8)
    assert doc.set_range_props(0, 0, visible=False) is False
    assert doc.set_tracks_props([0], visible=False) is False
