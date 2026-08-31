"""Duplicate View: two panes onto one drawing.

Aseprite's verb for working a few pixels while watching the whole sprite. The
model half is the interesting one and it is what this file is mostly about: a
tab used to *have* a view and now *has views and a focus*, with ``tab.view`` a
read-only accessor over the pair.

That shape is deliberate and is asserted here rather than left implicit,
because the alternative -- a mirrored ``view`` field kept in step beside the
list -- is the bug the Plotter's ``selected_object`` entry already names. The
two go out of step and the pane draws one object while every zoom lands on
another. There is no setter, and a test says so.

The other half is ``viewing``: two views are drawn in one frame, so "the
current view" has to be scoped to whichever pane's body is running or the
second pane renders its picture at the first pane's zoom.
"""

from __future__ import annotations

import pytest

from warlock.studio import inker_ops, inker_state
from warlock.studio.inker.document import Document


def _tab(**view) -> inker_state.InkerDoc:
    return inker_state.InkerDoc(
        doc=Document.blank(16, 16), views=[inker_state.PaintView(**view)]
    )


# --- the accessor ------------------------------------------------------------


def test_a_fresh_tab_has_one_view_and_is_not_split():
    tab = _tab()
    assert len(tab.views) == 1
    assert tab.split is False
    assert tab.view is tab.views[0]


def test_the_view_accessor_has_no_setter():
    """The whole reason the pair is the truth: an installed view the list does
    not hold is a pane drawing one object and every gesture moving another."""
    tab = _tab()
    with pytest.raises(AttributeError):
        tab.view = inker_state.PaintView()


def test_the_accessor_follows_the_focus():
    tab = _tab(zoom=1.0)
    inker_state.duplicate_view(tab)
    tab.views[1].zoom = 8.0

    tab.focus = 0
    assert tab.view.zoom == 1.0
    tab.focus = 1
    assert tab.view.zoom == 8.0


def test_a_focus_past_the_end_reads_the_last_view_rather_than_raising():
    """Closing a pane and drawing it are two different frames."""
    tab = _tab()
    tab.focus = 7
    assert tab.view is tab.views[0]


# --- the two commands --------------------------------------------------------


def test_duplicating_copies_the_view_you_were_looking_at():
    """A second pane arriving fitted-to-window would throw away the framing
    the user had and make the command read as a reset."""
    tab = _tab(zoom=4.0, pan=(12.0, -3.0), fitted=True)
    assert inker_state.duplicate_view(tab) is True

    assert tab.split is True
    assert (tab.views[1].zoom, tab.views[1].pan) == (4.0, (12.0, -3.0))
    assert tab.views[1].fitted is True
    # A copy, not the same object: zooming one pane must not zoom the other.
    assert tab.views[1] is not tab.views[0]


def test_duplicating_focuses_the_new_pane():
    tab = _tab()
    inker_state.duplicate_view(tab)
    assert tab.focus == 1
    assert tab.view is tab.views[1]


def test_a_third_view_is_refused():
    tab = _tab()
    assert inker_state.duplicate_view(tab) is True
    assert inker_state.duplicate_view(tab) is False
    assert len(tab.views) == 2


def test_closing_keeps_the_view_that_had_your_attention():
    tab = _tab(zoom=1.0)
    inker_state.duplicate_view(tab)
    tab.views[1].zoom = 8.0
    tab.focus = 1

    assert inker_state.close_duplicate_view(tab) is True
    assert tab.split is False
    assert tab.view.zoom == 8.0
    assert tab.focus == 0


def test_closing_from_the_first_pane_keeps_that_one():
    tab = _tab(zoom=1.0)
    inker_state.duplicate_view(tab)
    tab.views[1].zoom = 8.0
    tab.focus = 0

    assert inker_state.close_duplicate_view(tab) is True
    assert tab.view.zoom == 1.0


def test_closing_an_unsplit_tab_is_refused():
    tab = _tab()
    assert inker_state.close_duplicate_view(tab) is False
    assert len(tab.views) == 1


# --- the scoped current view -------------------------------------------------


def test_viewing_makes_that_pane_the_current_one():
    tab = _tab(zoom=1.0)
    inker_state.duplicate_view(tab)
    tab.views[1].zoom = 8.0
    tab.focus = 1

    with inker_state.viewing(tab, 0) as view:
        # Inside pane 0's body every ``tab.view`` means pane 0, whatever the
        # user's focus is -- which is what stops the second pane rendering its
        # picture at the first pane's zoom.
        assert view is tab.views[0]
        assert tab.view.zoom == 1.0
    assert tab.view.zoom == 8.0


def test_viewing_restores_what_it_found_rather_than_none():
    tab = _tab()
    inker_state.duplicate_view(tab)
    with inker_state.viewing(tab, 0):
        with inker_state.viewing(tab, 1):
            assert tab.view is tab.views[1]
        # The inner scope unwinds to the outer one, not to "follow focus".
        assert tab.view is tab.views[0]
    assert tab.active_view is None


def test_viewing_restores_after_a_raise():
    tab = _tab()
    with pytest.raises(RuntimeError), inker_state.viewing(tab, 0):
        raise RuntimeError("a draw went wrong")
    assert tab.active_view is None


# --- the ops -----------------------------------------------------------------


def test_the_two_ops_are_registered_under_view():
    ops = {op.name: op for op in inker_ops.OPS}
    assert ops["duplicate_view"].menu == "View"
    assert ops["close_duplicate_view"].menu == "View"


def test_each_op_is_offered_only_when_it_would_do_something():
    ops = {op.name: op for op in inker_ops.OPS}
    tab = _tab()
    state = inker_state.InkerState()

    assert ops["duplicate_view"].enabled(state, tab) is True
    assert ops["close_duplicate_view"].enabled(state, tab) is False

    inker_state.duplicate_view(tab)
    assert ops["duplicate_view"].enabled(state, tab) is False
    assert ops["close_duplicate_view"].enabled(state, tab) is True


def test_neither_op_is_offered_without_a_document():
    ops = {op.name: op for op in inker_ops.OPS}
    state = inker_state.InkerState()
    assert ops["duplicate_view"].enabled(state, None) is False
    assert ops["close_duplicate_view"].enabled(state, None) is False
