"""The sequence evaluator, which is the whole of how a note evolves over time.

Eight lines of :meth:`Sequence.index_at` decide every envelope in the app, so
they are tested directly rather than through a rendered waveform: a failure here
would otherwise show up as "the decay sounds wrong", which is not a bug report
anybody can act on.
"""

from __future__ import annotations

import dataclasses

import pytest

from warlock.studio.sirens.instruments import (
    MAX_SEQUENCE_LEN,
    Instrument,
    Sequence,
    default,
)


def _held(sequence: Sequence, ticks: int = 8) -> list[int | None]:
    return [sequence.index_at(t) for t in range(ticks)]


def test_an_empty_sequence_is_over_before_it_starts():
    assert Sequence().index_at(0) is None
    assert not Sequence()


def test_a_sequence_with_no_loop_holds_its_last_value():
    """Rather than going silent. A three-step attack on an instrument with no
    loop is an attack followed by a sustain, which is what a user who typed
    three numbers and no loop point meant."""
    assert _held(Sequence((5, 4, 3)), 6) == [0, 1, 2, 2, 2, 2]


def test_a_loop_repeats_the_tail():
    assert _held(Sequence((1, 2, 3), loop=1), 7) == [0, 1, 2, 1, 2, 1, 2]


def test_a_release_point_is_where_the_held_half_stops():
    """The rule the whole model turns on: everything from ``release`` onwards is
    the tail, and a held note must never play into it. Before this was true, an
    instrument with a decay written into it faded out while the key was down."""
    sequence = Sequence((15, 12, 9, 6, 3, 0), loop=0, release=1)
    assert [sequence.value_at(t) for t in range(5)] == [15, 15, 15, 15, 15]


def test_the_released_half_plays_through_and_stops():
    sequence = Sequence((15, 12, 9, 6, 3, 0), loop=0, release=1)
    assert [sequence.value_at(t, 2) for t in range(9)] == [15, 15, 12, 9, 6, 3, 0, 0, 0]
    assert sequence.index_at(8, 2) is None


def test_the_released_half_never_loops():
    """A release that loops is a note that never ends, and a stuck voice is the
    one failure in a synthesiser a user cannot work around."""
    sequence = Sequence((9, 8, 7, 6), loop=0, release=2)
    assert sequence.index_at(20, 0) is None


def test_no_release_point_means_a_note_off_cuts_the_voice():
    assert Sequence((5, 4, 3)).index_at(1, 1) is None


def test_only_the_volume_sequence_ends_a_note():
    """``finished`` is asked of the volume sequence alone. An arpeggio running
    out does not silence anything, and a version of this that let it would cut
    every note using a three-step arpeggio after three ticks."""
    assert Sequence((5, 4, 3)).finished(1, 1)
    assert not Sequence().finished(1, 1)


def test_a_sequence_longer_than_the_ceiling_is_refused():
    with pytest.raises(ValueError, match="past the"):
        Sequence(values=tuple(range(MAX_SEQUENCE_LEN + 1)))


def test_a_sequence_is_frozen_so_an_undo_step_can_hold_one():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Sequence((1, 2)).values = (3,)  # type: ignore[misc]


def test_an_instrument_names_a_kind_that_exists():
    with pytest.raises(ValueError, match="not one of"):
        Instrument(uid=1, kind="theremin")


def test_a_new_instrument_makes_a_sound():
    """A silent default is indistinguishable from a broken app: the user adds an
    instrument, types a note, hears nothing and has no way to tell which of the
    two happened."""
    volume = default(1).volume
    assert volume.value_at(0) > 0
    assert volume.value_at(120) > 0, "a held note must not fade out on its own"
    assert volume.finished(20, 0), "and it must end when it is released"
