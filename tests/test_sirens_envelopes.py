"""Sirens' envelope editor: what a drag means, and what a release point does.

Two halves, and neither of them draws anything. **The edits** -- painting a
column, filling the run a fast pointer skipped, moving and toggling the two
markers -- are pure functions of a ``Sequence``, which is what makes them
assertable on a box with no display; that is the same reason the engine under
``studio/sirens/`` has no pygame in it.

**And the sound.** The claim the editor exists to make legible is that
``release`` *splits* a sequence: everything before it is what a held note plays,
everything from it is the tail. That is asserted here through the rendered
samples rather than by reading the sequence back, because the sequence reading
correctly is not the thing that was wrong in Phase 1 -- the note running into
its own release tail while the key was down was.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from test_sirens_mode import FakeCtx, _tab

from warlock.studio import sirens_mode
from warlock.studio.panes import sirens_envelopes as env
from warlock.studio.sirens import document as D
from warlock.studio.sirens import instruments as inst
from warlock.studio.sirens import notes, synth

# --- the vocabulary -----------------------------------------------------------


def test_the_pane_and_the_mode_name_the_same_four_sequences():
    """A fifth sequence in the engine has to reach the editor and the gesture
    together: one without the other is a curve you can draw and not undo."""
    assert tuple(field for field, _ in env.FIELDS) == sirens_mode.ENVELOPE_FIELDS
    for field in sirens_mode.ENVELOPE_FIELDS:
        assert isinstance(getattr(inst.default(0), field), inst.Sequence)


# --- ranges -------------------------------------------------------------------


def test_the_bounded_ranges_are_the_engines_own():
    assert env.span("volume", inst.Sequence()) == (0, inst.MAX_VOLUME)
    assert env.span("duty", inst.Sequence()) == (0, inst.MAX_DUTY)


def test_the_signed_ranges_are_centred_on_zero():
    for field in ("arpeggio", "pitch"):
        low, high = env.span(field, inst.Sequence())
        assert low == -high and low < 0


def test_a_sequence_wider_than_the_reach_widens_its_own_graph():
    """Drawing a stored +30 semitones clipped at +12 would be the editor
    disagreeing with the file about what the instrument does."""
    wide = inst.Sequence(values=(0, 30))
    assert env.span("arpeggio", wide) == (-30, 30)


def test_an_empty_sequence_still_has_somewhere_to_draw():
    assert env.columns(inst.Sequence()) == env.MIN_STEPS
    assert env.columns(inst.Sequence(values=tuple(range(40)))) == 40


# --- painting -----------------------------------------------------------------


def test_painting_past_the_end_lengthens_the_sequence():
    after = env.painted(inst.Sequence(values=(15, 15)), 4, 3)
    assert after.values == (15, 15, 15, 15, 3)


def test_a_fast_drag_fills_the_columns_it_skipped():
    """A pointer crosses several columns in a frame. Without the run, a quick
    drag leaves a comb of untouched steps and reads as dropped input."""
    after = env.painted(inst.Sequence(values=(15,) * 6), 5, 2, previous=1)
    assert after.values == (15, 2, 2, 2, 2, 2)


def test_painting_leaves_the_markers_where_they_were():
    before = inst.Sequence(values=(15, 8, 0), loop=0, release=2)
    after = env.painted(before, 1, 4)
    assert (after.loop, after.release) == (0, 2)


def test_a_step_past_the_engines_ceiling_is_clamped_rather_than_refused():
    after = env.painted(inst.Sequence(values=(1,)), inst.MAX_SEQUENCE_LEN + 50, 7)
    assert len(after.values) == inst.MAX_SEQUENCE_LEN
    assert after.values[-1] == 7


# --- the markers --------------------------------------------------------------


def test_a_marker_stops_at_the_end_of_the_values_it_points_into():
    """An index past the end is one the engine ignores and the editor cannot
    draw -- a marker the user has lost is worse than one that will not move."""
    moved = env.moved(inst.Sequence(values=(1, 2, 3)), "release", 40)
    assert moved.release == 2


def test_a_marker_on_an_empty_sequence_does_not_appear():
    assert env.moved(inst.Sequence(), "loop", 3) == inst.Sequence()
    assert env.toggled(inst.Sequence(), "loop") == inst.Sequence()


def test_toggling_a_release_lands_it_where_a_held_note_still_sounds():
    """``release == 0`` makes every value tail material and a held note silent.
    A file can say that; a button should not produce it."""
    on = env.toggled(inst.Sequence(values=(15, 12, 8, 4)), "release")
    assert on.release >= 1
    assert env.toggled(on, "release").release == -1


def test_toggling_a_loop_lands_it_at_the_start_and_takes_it_off_again():
    on = env.toggled(inst.Sequence(values=(15, 12)), "loop")
    assert on.loop == 0
    assert env.toggled(on, "loop").loop == -1


def test_the_release_handle_wins_a_press_between_the_two():
    """They can sit on the same step, and the release is the one whose position
    changes what is heard while the key is down."""
    both = inst.Sequence(values=(1, 2, 3, 4), loop=2, release=2)
    assert env.grabbed(both, 20.0, 10.0, 5.0) == "release"
    assert env.grabbed(both, 5.0, 10.0, 5.0) == "paint"


def test_a_press_away_from_both_handles_paints():
    sequence = inst.Sequence(values=(1, 2, 3, 4), loop=0, release=3)
    assert env.grabbed(sequence, 18.0, 10.0, 2.0) == "paint"


def test_shortening_a_sequence_brings_its_markers_with_it():
    short = env.resized(inst.Sequence(values=tuple(range(20)), loop=15, release=18), 4)
    assert short.values == (0, 1, 2, 3)
    # The release lands on the last step; the loop stops one short of it,
    # because the loop repeats the *held* half and a loop point inside the
    # tail is a handle the graph does not draw and the engine never reaches.
    assert (short.loop, short.release) == (2, 3)


def test_lengthening_a_sequence_holds_its_last_value():
    long = env.resized(inst.Sequence(values=(9, 4)), 5)
    assert long.values == (9, 4, 4, 4, 4)


# --- reading the pointer ------------------------------------------------------


def test_the_top_of_a_graph_is_its_high_value_and_the_bottom_its_low():
    assert env.value_at(0.0, 50.0, 0, 15) == 15
    assert env.value_at(50.0, 50.0, 0, 15) == 0
    assert env.value_at(25.0, 50.0, -12, 12) == 0


def test_a_pointer_dragged_off_the_graph_stays_inside_it():
    assert env.value_at(-40.0, 50.0, 0, 15) == 15
    assert env.value_at(400.0, 50.0, 0, 15) == 0
    assert env.step_at(-9.0, 10.0, 8) == 0
    assert env.step_at(9000.0, 10.0, 8) == 7


# --- one drag, one undo step --------------------------------------------------


def _drag(ctx: FakeCtx, tab: Any, field: str, columns: list[tuple[int, int]]) -> None:
    """What the pane's three moments do, without an imgui frame between them."""
    state = sirens_mode.ensure(ctx)
    instrument = tab.doc.instruments[0]
    sirens_mode.begin_envelope_drag(ctx, tab, field, "paint")
    for step, value in columns:
        sequence = getattr(tab.doc.instrument(instrument.uid), field)
        after = env.painted(sequence, step, value, previous=state.env_step)
        state.env_step = step
        sirens_mode.set_sequence(ctx, tab, instrument.uid, field, after)
    sirens_mode.end_envelope_drag(ctx, tab)


def test_a_drag_across_a_curve_is_one_undo_step():
    """Without the collapse, painting a decay costs one Ctrl+Z per column the
    pointer crossed."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    _drag(ctx, tab, "volume", [(step, 15 - step) for step in range(12)])
    assert len(tab.doc.history) == depth + 1
    painted = tab.doc.instruments[0].volume.values
    tab.doc.undo()
    assert tab.doc.instruments[0].volume.values != painted
    assert len(tab.doc.history) == depth


def test_a_drag_that_changed_one_thing_is_still_one_step():
    ctx = FakeCtx()
    tab = _tab(ctx)
    depth = len(tab.doc.history)
    _drag(ctx, tab, "duty", [(0, 3)])
    assert len(tab.doc.history) == depth + 1


def test_a_drag_that_changed_nothing_pushes_nothing():
    """A pointer held still over a bar it has already painted is a stream of
    frames, and ``update_instrument`` refuses every one of them."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    held = tab.doc.instruments[0].volume
    depth = len(tab.doc.history)
    _drag(ctx, tab, "volume", [(0, held.values[0])] * 20)
    assert len(tab.doc.history) == depth


def test_a_drag_re_arms_the_renderer():
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.render_dirty = False
    _drag(ctx, tab, "arpeggio", [(0, 4), (1, 7)])
    assert tab.render_dirty


def test_a_gesture_on_a_busy_tab_never_opens():
    """Saving disables every editing control; a drag that recorded a depth
    against a document it may not touch would fold a run that never happened."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    tab.saving = True
    sirens_mode.begin_envelope_drag(ctx, tab, "volume", "paint")
    assert sirens_mode.ensure(ctx).env_depth == -1
    assert not sirens_mode.set_sequence(
        ctx, tab, tab.doc.instruments[0].uid, "volume", inst.Sequence(values=(1,))
    )


def test_an_unknown_sequence_name_is_refused_rather_than_set():
    ctx = FakeCtx()
    tab = _tab(ctx)
    assert not sirens_mode.set_sequence(
        ctx, tab, tab.doc.instruments[0].uid, "tremolo", inst.Sequence(values=(1,))
    )


# --- what release actually does -----------------------------------------------


def _one_note(*, released: bool) -> tuple[np.ndarray, int]:
    """A pulse note whose instrument is silent while held and loud once let go.

    ``volume`` is ``(0, 15)`` with the loop at 0 and the release at 1: held, the
    sequence never reaches index 1, because the held half stops at the release
    point and loops back to 0; released, it plays index 1 and finishes.

    Returns the samples **and where the pattern's own rows end**. Past that,
    ``_render`` releases every voice still sounding so a tail is not chopped off
    mid-decay, which is right and is exactly the tail this pair is about -- so
    the held case has to be read over the rows rather than over the file.
    """
    doc = D.new_song()
    pattern = doc.patterns[0]
    instrument = doc.instruments[0]
    doc.update_instrument(
        instrument.uid, volume=inst.Sequence(values=(0, 15), loop=0, release=1)
    )
    doc.set_cell(pattern.uid, 0, 0, D.NOTE, 48)
    doc.set_cell(pattern.uid, 0, 0, D.INSTRUMENT, instrument.uid)
    if released:
        doc.set_cell(pattern.uid, 8, 0, D.NOTE, notes.NOTE_RELEASE)
    rows_end = int(pattern.rows * doc.speed / doc.tick_rate * synth.SAMPLE_RATE)
    return synth.render_pattern(doc, pattern.uid), rows_end


def test_a_held_note_never_reaches_its_release_tail():
    """The Phase 1 bug, as a standing test: a note that ran into its own tail
    faded out while the key was still down."""
    samples, rows_end = _one_note(released=False)
    assert np.abs(samples[:rows_end]).max() < 1e-6


def test_a_released_note_plays_the_tail_the_held_one_did_not():
    samples, rows_end = _one_note(released=True)
    assert np.abs(samples[:rows_end]).max() > 0.01


def test_the_released_half_does_not_loop():
    """A release that loops is a note that never ends -- so the tail plays once
    and the rest of the render is silence."""
    doc = D.new_song()
    pattern = doc.patterns[0]
    instrument = doc.instruments[0]
    doc.update_instrument(
        instrument.uid, volume=inst.Sequence(values=(15, 15), loop=0, release=1)
    )
    doc.set_cell(pattern.uid, 0, 0, D.NOTE, 48)
    doc.set_cell(pattern.uid, 0, 0, D.INSTRUMENT, instrument.uid)
    doc.set_cell(pattern.uid, 1, 0, D.NOTE, notes.NOTE_RELEASE)
    samples = np.abs(synth.render_pattern(doc, pattern.uid)).max(axis=1)
    # One tick of tail at the default rate, then nothing for the rest of the
    # pattern and its two-second run-out.
    assert samples[: synth.SAMPLE_RATE // 30].max() > 0.01
    assert samples[synth.SAMPLE_RATE // 2 :].max() < 1e-6


@pytest.mark.parametrize("field", ["volume", "arpeggio", "pitch", "duty"])
def test_every_sequence_the_editor_draws_is_one_the_engine_ticks(field):
    """The four graphs are not a menu: each one has to reach the tick loop, or
    it is a control that does nothing."""
    import inspect

    assert f"instrument.{field}" in inspect.getsource(synth)
