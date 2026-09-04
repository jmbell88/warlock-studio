"""Sirens' hint line: does it say the right thing, and is any of it a lie.

``tests/test_clay_hints.py``'s questions for the second hint line in the app.
The parity test at the foot is the one that matters -- a hint naming a binding
nothing implements is worse than no hint, because it is read as a promise.
"""

from __future__ import annotations

import pytest

from warlock.studio import sirens_hints, sirens_keys, sirens_state
from warlock.studio.sirens import document as D
from warlock.studio.sirens import synth


def test_a_line_for_every_column_the_document_has():
    """One per column, in the document's order, which is what makes indexing
    it by ``state.column`` correct rather than nearly correct."""
    assert len(sirens_hints._COLUMNS) == len(sirens_state.COLUMN_LABELS)
    assert len(sirens_hints._COLUMNS) == D.COLUMNS


@pytest.mark.parametrize("column", range(D.COLUMNS))
def test_every_column_says_something_and_something_different(column):
    lines = {sirens_hints.hint(one) for one in range(D.COLUMNS)}
    assert sirens_hints.hint(column)
    assert len(lines) == D.COLUMNS


def test_the_note_column_names_the_piano_and_the_others_do_not():
    """The confusion this line exists for: ``c`` is a note in one column and a
    hex digit in two others."""
    assert sirens_hints.LOWER_ROW in sirens_hints.hint(D.NOTE)
    for column in (D.INSTRUMENT, D.VOLUME, D.EFFECT, D.PARAM):
        assert sirens_hints.LOWER_ROW not in sirens_hints.hint(column)


def test_the_keys_that_work_everywhere_are_on_every_line():
    for column in range(D.COLUMNS):
        found = sirens_hints.keys_named(sirens_hints.hint(column))
        assert {"Arrows", "Space", "Delete"} <= found


def test_the_block_chords_appear_only_with_a_block():
    """Four chords that do nothing without a selection. Offered unconditionally
    they would be four promises the app refuses."""
    assert "Ctrl+C" not in sirens_hints.keys_named(sirens_hints.hint(D.NOTE))
    with_block = sirens_hints.keys_named(
        sirens_hints.hint(D.NOTE, has_selection=True)
    )
    assert {"Ctrl+C", "Ctrl+X", "Ctrl+V", "Ctrl+G"} <= with_block


def test_a_column_out_of_range_clamps_rather_than_raising():
    """It is a readout drawn every frame: a caret briefly out of range must
    cost a wrong line, never the frame."""
    assert sirens_hints.hint(-1) == sirens_hints.hint(D.NOTE)
    assert sirens_hints.hint(99) == sirens_hints.hint(D.COLUMNS - 1)


def test_keys_named_does_not_read_english_as_a_binding():
    """``clay_hints``' rule, and the reason the piano rows are lowercase: a
    single letter counts only when it is capital."""
    assert sirens_hints.keys_named("zsxdcvgbhnjm are the piano") == set()
    assert sirens_hints.keys_named("F loudest, 0 silent") == {"F"}


def test_the_effect_letters_come_from_the_engine():
    """Not a second copy of the table. A tenth effect appears here by itself."""
    assert set(sirens_hints.EFFECT_LETTERS) == {
        letter for letter, _what in synth.EFFECT_NAMES.values()
    }
    line = sirens_hints.hint(D.EFFECT)
    for letter in sirens_hints.EFFECT_LETTERS:
        assert letter in line


def test_every_letter_of_the_piano_rows_is_a_piano_key():
    """The rows are spelled out here rather than derived, because deriving them
    would cost this module its purity -- so the copy is checked instead."""
    for letter in sirens_hints.LOWER_ROW + sirens_hints.UPPER_ROW:
        assert letter in sirens_keys.PIANO_KEYS, letter


def test_every_key_the_line_names_is_a_key_the_mode_listens_to():
    """The parity that matters. Every chord and capital the five lines name is
    either a hex digit, an effect letter, or one of the keys ``handle_key``
    dispatches on before it reaches a column at all."""

    #: Written out, and deliberately: this is the second, independently
    #: authored list of what ``sirens_keys.handle_key`` answers, so a line
    #: promising a chord that was never implemented fails here rather than in
    #: front of a reader.
    elsewhere = {
        "Arrows", "Shift+Arrows", "Space", "Delete", "Insert", "Shift+Delete",
        "+", "Shift+`",
        "Ctrl+Up", "Ctrl+Down",
        "Ctrl+C", "Ctrl+X", "Ctrl+V", "Ctrl+G",
    }
    hexes = set("0123456789ABCDEF")
    letters = {letter.upper() for letter, _what in synth.EFFECT_NAMES.values()}
    known = elsewhere | hexes | letters
    named: set[str] = set()
    for column in range(D.COLUMNS):
        for selection in (False, True):
            named |= sirens_hints.keys_named(
                sirens_hints.hint(column, has_selection=selection)
            )
    assert named <= known, sorted(named - known)
