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


def test_hiding_the_strip_mid_playback_does_not_wedge_the_document():
    """``_tick`` was below ``draw``'s ``timeline_open`` early return, and it is
    the only caller of ``tick_playback`` -- the only thing that advances the
    playhead or ends a clip.

    So Tab (Aseprite's own binding for hiding the strip) left ``playing`` True
    forever and ``tab.busy`` with it, and the canvas then refused every paint
    gesture silently: the Stop button and the "playback is running" line that
    would have explained it are drawn inside the strip that was just hidden.
    """

    state, tab = _session()
    tab.doc.ensure_animation()
    tab.doc.add_frame()
    state.timeline_open = True
    tab.playing = True
    assert tab.busy is True
    inker_timeline.toggle(state)
    assert state.timeline_open is False
    assert tab.playing is False
    assert tab.busy is False


def test_showing_the_strip_again_leaves_playback_alone():
    state, tab = _session()
    tab.doc.ensure_animation()
    tab.doc.add_frame()
    state.timeline_open = False
    tab.playing = True
    inker_timeline.toggle(state)
    assert state.timeline_open is True
    assert tab.playing is True


def test_the_playback_tick_runs_with_the_strip_hidden():
    """Defence in depth for the same wedge, one layer down: any *other* route
    to a hidden strip has to keep the clip advancing rather than freezing it
    with ``busy`` held."""

    from pathlib import Path

    source = Path(inker_timeline.__file__).read_text(encoding="utf-8")
    body = source[source.index("def draw(ctx: Any) -> None:") :]
    body = body[: body.index("def autoshow")]
    assert body.index("_tick(tab)") < body.index("if not state.timeline_open:")


# -- one gesture, one undo step ----------------------------------------------


def test_hiding_every_layer_is_one_undo_step():
    """``set_layer_props`` pushes its own edit per call that changes something,
    so the header's loop cost a ten-layer document ten Ctrl+Z to reverse one
    click -- against the one-gesture-one-step rule the filters, the palette
    conversion and ``apply_matte`` all follow."""

    doc = inker.Document.blank(8, 8)
    for _ in range(4):
        doc.add_layer()
    head = doc.history.head
    assert doc.set_all_layer_props(visible=False) is True
    assert all(not layer.visible for layer in doc.stack)
    assert doc.undo() is True
    assert all(layer.visible for layer in doc.stack)
    assert doc.history.head == head


def test_locking_every_layer_of_an_animated_document_is_one_step_too():
    doc = inker.Document.blank(8, 8)
    doc.add_layer()
    doc.ensure_animation()
    assert doc.set_all_layer_props(locked=True) is True
    assert all(track.locked for track in doc.anim.tracks)
    assert doc.undo() is True
    assert not any(track.locked for track in doc.anim.tracks)


def test_a_stack_that_already_agrees_records_nothing():
    """A no-op must not make a saved document ask to be saved again."""
    doc = inker.Document.blank(8, 8)
    doc.add_layer()
    assert doc.set_all_layer_props(visible=True) is False


def test_only_the_rows_that_change_contribute_a_step():
    doc = inker.Document.blank(8, 8)
    for _ in range(3):
        doc.add_layer()
    doc.set_layer_props(1, visible=False)
    head = doc.history.head
    assert doc.set_all_layer_props(visible=False) is True
    assert doc.undo() is True
    assert doc.history.head == head
    # Row 1 was already hidden and must stay hidden, not be shown again.
    assert [layer.visible for layer in doc.stack] == [True, False, True, True]


def test_an_eye_drag_is_one_step_for_every_row_it_crossed():
    """The gesture writes the rows live so the column follows the cursor, and
    asks for its undo entry once, on release. Eight rows crossed used to cost
    eight Ctrl+Z to put back."""

    doc = inker.Document.blank(8, 8)
    for _ in range(3):
        doc.add_layer()
    head = doc.history.head
    # What ``_drag_toggle`` does across three rows, then what release does.
    was = {}
    for index in (0, 1, 2):
        was[index] = {"visible": doc.stack[index].visible}
        doc.stack[index].visible = False
    assert doc.set_layers_props([0, 1, 2], was=was, visible=False) is True
    assert doc.undo() is True
    assert [layer.visible for layer in doc.stack] == [True, True, True, True]
    assert doc.history.head == head
