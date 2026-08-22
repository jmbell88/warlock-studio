"""The timeline absorbed the layers panel: one grid, and when it shows itself.

The riskiest single change of the Inker wave, because it deletes a pane every
session uses. What is testable without a frame is the part that decides whether
a user can find their layers at all -- the auto-show rule, the ``Tab`` toggle,
and the fact that a still document has a row per layer and exactly one frame
column -- so that is what is pinned here.
"""

from __future__ import annotations

from warlock.studio import inker, inker_state
from warlock.studio.panes import inker_timeline


def _tab(uid: str = "t1"):
    doc = inker.Document.blank(16, 16)
    return inker_state.InkerDoc(doc=doc, uid=uid, title="Untitled")


def _session():
    state = inker_state.InkerState()
    tab = _tab()
    state.add(tab)
    return state, tab


def test_a_fresh_single_layer_still_shows_no_strip():
    state, tab = _session()
    assert inker_timeline.autoshow(state, tab) is False
    assert state.timeline_open is False


def test_a_second_layer_raises_the_strip_by_itself():
    state, tab = _session()
    tab.doc.add_layer()
    assert inker_timeline.autoshow(state, tab) is True
    assert state.timeline_open is True


def test_a_second_frame_raises_it_too():
    state, tab = _session()
    tab.doc.ensure_animation()
    tab.doc.add_frame()
    assert inker_timeline.autoshow(state, tab) is True


def test_auto_show_fires_once_per_document():
    """Closing the strip on a five-layer drawing has to stick."""

    state, tab = _session()
    tab.doc.add_layer()
    inker_timeline.autoshow(state, tab)
    state.timeline_open = False
    tab.doc.add_layer()
    assert inker_timeline.autoshow(state, tab) is False
    assert state.timeline_open is False


def test_tab_toggles_it_and_stops_it_arguing_back():
    state, tab = _session()
    inker_timeline.toggle(state)
    assert state.timeline_open is True
    inker_timeline.toggle(state)
    assert state.timeline_open is False
    # And the close sticks: every open document counts as already shown.
    tab.doc.add_layer()
    assert inker_timeline.autoshow(state, tab) is False


def test_a_still_document_is_a_one_frame_sprite():
    """One column, and no model change: ``doc.anim`` is still None."""

    tab = _tab()
    assert tab.doc.anim is None
    assert inker_timeline.frame_uids(tab.doc) == [None]


def test_an_animated_document_has_a_column_per_frame():
    tab = _tab()
    tab.doc.ensure_animation()
    tab.doc.add_frame()
    assert len(inker_timeline.frame_uids(tab.doc)) == len(tab.doc.anim.frames) == 2


def test_the_rows_run_bottom_up():
    """Aseprite's order, Photoshop's order, and the order the grid's own frame
    columns already implied. Read off the source, because the walk is what the
    change was: the panel counted down and the grid now counts up."""

    import inspect

    source = inspect.getsource(inker_timeline._grid)
    assert "for index in range(len(doc.stack)):" in source
    assert "range(len(anim.tracks) - 1, -1, -1)" not in source


def test_the_range_readers_came_with_the_rows():
    assert callable(inker_timeline.track_range)
    assert callable(inker_timeline.extend_range)


def test_the_layers_pane_is_gone():
    from pathlib import Path

    panes = Path(inker_timeline.__file__).parent
    assert not (panes / "inker_layers.py").exists()


def test_an_eye_drag_paints_one_state_rather_than_flipping_each_row():
    """The value is the one the first row took; a flip-per-row drag would
    leave a striped stack behind whichever way the hand moved."""

    import inspect

    source = inspect.getsource(inker_timeline._drag_toggle)
    assert "state.eye_drag = not tab.doc.stack[index].visible" in source
    assert "!= state.eye_drag" in source


def test_toggle_all_shows_everything_when_anything_is_hidden():
    """With three of ten hidden, the button a user reaches for means "show
    everything" -- a strict-all rule would hide the other seven."""

    import inspect

    source = inspect.getsource(inker_timeline._toggle_all)
    assert "any(not layer.visible for layer in doc.stack)" in source


def test_the_timeline_state_starts_closed_and_remembers_nothing_per_document():
    state = inker_state.InkerState()
    assert state.timeline_open is False
    assert state.timeline_shown == set()
    assert state.eye_drag is None
