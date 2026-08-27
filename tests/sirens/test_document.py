"""The song document: what a step addresses, and what refuses to be a step.

Two rules are asserted repeatedly here because they are the two that go wrong
silently. **A step addresses by uid**, so reordering a list cannot retarget an
undo; and **a call that changes nothing pushes nothing**, so a history panel is
a list of things that happened rather than of keys that were pressed.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.sirens import document as D
from warlock.studio.sirens import notes


def _song() -> D.SongDoc:
    return D.new_song()


def test_a_new_song_is_playable_and_unmodified():
    """An empty order is a song where Space does nothing and there is no way to
    find out why."""
    doc = _song()
    assert doc.order == [doc.patterns[0].uid]
    assert doc.instruments
    assert not doc.dirty


def test_the_tick_rate_falls_out_of_musical_units():
    """150 BPM at speed 6 is exactly 60 Hz, which is the rate the whole idiom
    was written against. That it is derived rather than declared is the point:
    a user changes the tempo and the engine follows."""
    doc = _song()
    assert doc.tick_rate == 60.0
    doc.set_song(tempo=75)
    assert doc.tick_rate == 30.0


def test_retyping_the_note_that_is_already_there_is_not_a_change():
    doc = _song()
    uid = doc.patterns[0].uid
    assert doc.set_cell(uid, 0, 0, D.NOTE, 48)
    depth = len(doc.history)
    assert not doc.set_cell(uid, 0, 0, D.NOTE, 48)
    assert len(doc.history) == depth


def test_a_block_that_runs_off_the_end_is_clipped_rather_than_refused():
    """Pasting four bars near the bottom of a pattern puts in what fits, which
    is what every tracker does and what dragging a selection implies."""
    doc = _song()
    pattern = doc.patterns[0]
    block = np.full((8, 1, 1), 60, dtype=np.int16)
    assert doc.set_cells(pattern.uid, pattern.rows - 3, 0, D.NOTE, block)
    assert list(pattern.cells[-3:, 0, D.NOTE]) == [60, 60, 60]


def test_a_block_entirely_outside_the_pattern_changes_nothing():
    doc = _song()
    pattern = doc.patterns[0]
    block = np.full((2, 1, 1), 60, dtype=np.int16)
    assert not doc.set_cells(pattern.uid, pattern.rows + 5, 0, D.NOTE, block)


def test_transposing_leaves_empty_cells_and_commands_alone():
    doc = _song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 0, 0, D.NOTE, 48)
    doc.set_cell(uid, 1, 0, D.NOTE, notes.NOTE_OFF)
    doc.transpose(uid, 0, 0, 4, 1, 12)
    cells = doc.patterns[0].cells
    assert cells[0, 0, D.NOTE] == 60
    assert cells[1, 0, D.NOTE] == notes.NOTE_OFF
    assert cells[2, 0, D.NOTE] == notes.EMPTY


def test_a_note_that_would_leave_the_range_stays_where_it_is():
    """Rather than clamping to the edge: clamping the top voice of a chord turns
    it into a different chord, silently, and a note that did not move is
    visible."""
    doc = _song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 0, 0, D.NOTE, notes.MAX_NOTE - 1)
    doc.set_cell(uid, 1, 0, D.NOTE, 48)
    doc.transpose(uid, 0, 0, 2, 1, 5)
    assert doc.patterns[0].cells[0, 0, D.NOTE] == notes.MAX_NOTE - 1
    assert doc.patterns[0].cells[1, 0, D.NOTE] == 53


def test_an_undo_after_a_reorder_still_lands_on_the_right_pattern():
    """The uid rule, stated as the failure it prevents."""
    doc = _song()
    first = doc.patterns[0].uid
    doc.set_cell(first, 0, 0, D.NOTE, 48)
    second = doc.add_pattern()
    doc.patterns.reverse()
    doc.undo()  # the add
    doc.undo()  # the note
    assert doc.pattern(first).cells[0, 0, D.NOTE] == notes.EMPTY
    assert doc.pattern(second.uid) is None


def test_deleting_a_pattern_takes_its_order_entries_with_it_in_one_step():
    """Two steps would let one Ctrl+Z reach a song that refers to a pattern
    which does not exist."""
    doc = _song()
    first = doc.patterns[0].uid
    second = doc.add_pattern().uid
    doc.set_order([first, second, first])
    depth = len(doc.history)
    doc.remove_pattern(second)
    assert doc.order == [first, first]
    assert len(doc.history) == depth + 1
    doc.undo()
    assert doc.order == [first, second, first]
    assert doc.pattern(second) is not None


def test_the_order_cannot_name_a_pattern_the_song_does_not_have():
    doc = _song()
    with pytest.raises(ValueError, match=D.MISSING_PATTERN):
        doc.set_order([doc.patterns[0].uid, 999999])


def test_a_loop_point_past_the_end_of_the_order_is_dropped():
    doc = _song()
    second = doc.add_pattern().uid
    doc.set_order([doc.patterns[0].uid, second])
    doc.set_song(loop_order=1)
    doc.set_order([doc.patterns[0].uid])
    assert doc.loop_order == -1


def test_adding_a_channel_widens_every_pattern_and_undoes_as_one_step():
    doc = _song()
    doc.add_pattern()
    before = [one.cells.shape for one in doc.patterns]
    channel = doc.add_channel(kind="pulse", name="Pulse 3")
    assert [one.cells.shape[1] for one in doc.patterns] == [s[1] + 1 for s in before]
    doc.undo()
    assert [one.cells.shape for one in doc.patterns] == before
    assert doc.channel(channel.uid) is None


def test_removing_a_channel_puts_its_notes_back_on_undo():
    """The only honest way to reverse a delete is to have kept what it deleted,
    which is what makes :class:`~.edits.ChannelsEdit` the expensive step."""
    doc = _song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 0, 1, D.NOTE, 60)
    target = doc.channels[1].uid
    doc.remove_channel(target)
    assert doc.patterns[0].cells.shape[1] == 4
    doc.undo()
    assert doc.patterns[0].cells[0, 1, D.NOTE] == 60


def test_a_song_keeps_at_least_one_channel():
    doc = _song()
    for one in list(doc.channels[1:]):
        doc.remove_channel(one.uid)
    with pytest.raises(ValueError, match="at least one channel"):
        doc.remove_channel(doc.channels[0].uid)


def test_removing_an_instrument_leaves_the_cells_that_named_it():
    """Rewriting every cell in the song is a far larger edit than the one that
    was asked for, and a cell holding an unknown uid plays silently and can be
    put back by an undo."""
    doc = _song()
    uid = doc.patterns[0].uid
    instrument = doc.instruments[0].uid
    doc.set_cell(uid, 0, 0, D.INSTRUMENT, instrument)
    doc.remove_instrument(instrument)
    assert doc.patterns[0].cells[0, 0, D.INSTRUMENT] == instrument
    assert doc.instrument(instrument) is None


def test_a_one_shot_and_its_pattern_are_one_gesture():
    doc = _song()
    patterns = len(doc.patterns)
    effect = doc.add_oneshot("jump")
    assert len(doc.patterns) == patterns + 1
    doc.undo()
    assert len(doc.patterns) == patterns
    assert doc.oneshot(effect.uid) is None


def test_resizing_a_pattern_keeps_what_fits_and_restores_what_did_not():
    doc = _song()
    uid = doc.patterns[0].uid
    doc.set_cell(uid, 40, 0, D.NOTE, 48)
    doc.resize_pattern(uid, 16)
    assert doc.patterns[0].rows == 16
    doc.undo()
    assert doc.patterns[0].cells[40, 0, D.NOTE] == 48


def test_the_sample_table_takes_add_replace_and_remove():
    doc = _song()
    assert doc.set_sample("kick", np.ones(4, dtype=np.float32))
    assert not doc.set_sample("kick", np.ones(4, dtype=np.float32))
    assert doc.set_sample("kick", np.zeros(4, dtype=np.float32))
    assert doc.set_sample("kick", None)
    assert "kick" not in doc.samples
    doc.undo()
    assert "kick" in doc.samples


def test_the_song_scalars_only_record_what_moved():
    doc = _song()
    doc.set_song(tempo=120, title="A")
    assert not doc.set_song(tempo=120)
    doc.set_song(title="B")
    doc.undo()
    assert doc.title == "A" and doc.tempo == 120


def test_an_unknown_song_field_is_refused():
    with pytest.raises(ValueError, match="has no"):
        _song().set_song(swing=3)


def test_dirty_is_a_comparison_rather_than_a_flag():
    """Undo back to where it was saved and the document is not unsaved, which a
    latching flag cannot express."""
    doc = _song()
    doc.set_cell(doc.patterns[0].uid, 0, 0, D.NOTE, 48)
    assert doc.dirty
    doc.mark_saved()
    doc.set_cell(doc.patterns[0].uid, 1, 0, D.NOTE, 48)
    assert doc.dirty
    doc.undo()
    assert not doc.dirty


def test_uids_handed_out_after_an_open_cannot_collide_with_the_file():
    D.reserve_uid(10_000)
    assert D.new_uid() > 10_000


def test_every_refusal_names_the_thing_that_is_missing():
    doc = _song()
    for call in (
        lambda: doc.set_cell(999999, 0, 0, D.NOTE, 48),
        lambda: doc.remove_pattern(999999),
        lambda: doc.resize_pattern(999999, 8),
    ):
        with pytest.raises(ValueError, match=D.MISSING_PATTERN):
            call()
    with pytest.raises(ValueError, match=D.MISSING_INSTRUMENT):
        doc.remove_instrument(999999)
    with pytest.raises(ValueError, match=D.MISSING_CHANNEL):
        doc.update_channel(999999, name="x")
    with pytest.raises(ValueError, match=D.MISSING_ONESHOT):
        doc.remove_oneshot(999999)


def test_an_edit_owns_its_data_rather_than_a_view_of_the_pattern():
    """A slice reports its own few hundred bytes to the undo budget while
    keeping the whole pattern alive, so a stack of one-cell edits could pin a
    hundred megabytes the budget never sees."""
    doc = _song()
    doc.set_cell(doc.patterns[0].uid, 0, 0, D.NOTE, 48)
    step = doc.history.top
    assert step.before.base is None and step.after.base is None
