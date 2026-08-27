"""Every column of the pattern grid, typed into.

This file exists because of what its absence cost. Sirens landed across five
green landings with a grid that drew five columns, a caret that walked into all
five, and **keys for one of them** -- and nothing in the suite noticed, because
everything that typed anything typed a note. The engine's half was covered from
the first landing: ``synth`` renders ``Fxx``, ``Axy``, the slides and the
release note, and a stored corpus pins the samples byte for byte. What nobody
ever asserted was that a *keystroke* could produce any of it.

So the assertions here are shaped to fail on that class of gap rather than on
spelling:

- the write tests are parametrised over ``range(document.COLUMNS)`` and look
  their keys up in a table with ``[]``, so a sixth column added to the engine
  fails here with a ``KeyError`` until somebody has decided what typing into it
  means;
- the two that matter most go through ``synth.render`` and compare **audio**. A
  tempo effect typed through ``handle_key`` has to change the song's length, and
  a released note has to sound where a cut one is silent. A test that only read
  the cell back would pass just as happily against a grid whose keys wrote into
  a column the synthesiser never looks at.

Nothing here touches a sound device: ``test_sirens_mode``'s no-device fixtures
come with ``FakeCtx``, and rendering is arithmetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from test_sirens_mode import FakeCtx, _Event, _tab

from warlock.studio import sirens_mode
from warlock.studio.sirens import document as D
from warlock.studio.sirens import notes, synth

#: One column -> the keys that fill it, and the value they should leave behind.
#: Keyed by every column the document has and read with ``[]`` on purpose: a
#: sixth column is a decision about what its alphabet is, and the right way to
#: be told one is owed is a failure in here.
COLUMN_KEYS: dict[int, tuple[tuple[str, ...], int]] = {
    # ``z`` in octave 4 is C-4, which is 48 in this engine's numbering.
    D.NOTE: (("z",), 48),
    D.INSTRUMENT: (("0", "5"), 0x05),
    # ``c`` is a piano key in the note column and the hex digit twelve here,
    # which is the overlap the dispatch on ``state.column`` exists to resolve.
    D.VOLUME: (("c",), 0x0C),
    D.EFFECT: (("f",), synth.FX_TEMPO),
    D.PARAM: (("7", "8"), 0x78),
}


def _press(ctx: FakeCtx, key: str, mod: int = 0) -> bool:
    """One KEYDOWN through the real handler. ``key`` is a pygame key's name."""
    import pygame

    name = f"K_{key}" if len(key) == 1 else f"K_{key.upper()}"
    return sirens_mode.handle_key(ctx, _Event(getattr(pygame, name), mod))


def _cells(ctx: FakeCtx, tab: Any) -> Any:
    return tab.doc.pattern(sirens_mode.ensure(ctx).pattern).cells


def _at(ctx: FakeCtx, column: int, *, row: int = 0) -> None:
    sirens_mode.set_caret(ctx, row=row, channel=0, column=column)


# --- typing into every column -------------------------------------------------


@pytest.mark.parametrize("column", sorted(range(D.COLUMNS)))
def test_a_key_writes_into_every_column(column):
    """The gap this whole file is about, stated once per column."""
    keys, expected = COLUMN_KEYS[column]
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, column)
    for key in keys:
        assert _press(ctx, key), f"{key!r} was not consumed in column {column}"
    assert _cells(ctx, tab)[0, 0, column] == expected


@pytest.mark.parametrize("column", sorted(range(D.COLUMNS)))
def test_a_write_into_any_column_arms_the_renderer(column):
    """Every obligation ``write_cell`` carries is owed by every column, not by
    the note column alone: an edit that does not re-arm the render is an edit
    you cannot hear, which is what an edit that did not happen sounds like."""
    keys, _expected = COLUMN_KEYS[column]
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, column)
    tab.render_dirty = False
    for key in keys:
        _press(ctx, key)
    assert tab.render_dirty


@pytest.mark.parametrize("column", sorted(range(D.COLUMNS)))
def test_a_finished_entry_steps_the_caret_and_a_half_finished_one_does_not(column):
    """The low nibble has to land in the cell the high nibble went into."""
    keys, _expected = COLUMN_KEYS[column]
    ctx = FakeCtx()
    _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.step = 4
    _at(ctx, column)
    for index, key in enumerate(keys):
        _press(ctx, key)
        assert state.row == (4 if index == len(keys) - 1 else 0)


@pytest.mark.parametrize("column", sorted(range(D.COLUMNS)))
def test_a_busy_tab_takes_no_typing_in_any_column(column):
    """A tab mid-save is locked in every column, not only in the one that had
    keys first."""
    keys, _expected = COLUMN_KEYS[column]
    ctx = FakeCtx()
    tab = _tab(ctx)
    _at(ctx, column)
    tab.saving = True
    for key in keys:
        _press(ctx, key)
    assert (_cells(ctx, tab)[0, 0, :] == notes.EMPTY).all()


# --- two-digit entry ----------------------------------------------------------


def test_the_first_digit_is_the_high_nibble_and_the_second_the_low():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.step = 0
    _at(ctx, D.PARAM)
    _press(ctx, "4")
    assert _cells(ctx, tab)[0, 0, D.PARAM] == 0x40
    assert state.digit == 1
    _press(ctx, "a")
    assert _cells(ctx, tab)[0, 0, D.PARAM] == 0x4A
    assert state.digit == 0


def test_a_digit_replaces_one_nibble_in_place():
    """Correcting the low half of a byte is one keystroke on the second
    character, which is what makes the numeric columns editable rather than
    merely writable."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, D.PARAM)
    for key in ("4", "a"):
        _press(ctx, key)
    _at(ctx, D.PARAM)
    for key in ("4", "b"):
        _press(ctx, key)
    assert _cells(ctx, tab)[0, 0, D.PARAM] == 0x4B


def test_moving_the_caret_between_two_digits_resets_the_sub_position():
    """Otherwise the second nibble lands in a cell nobody was looking at, under
    an undo step nobody would recognise."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.step = 0
    _at(ctx, D.PARAM)
    _press(ctx, "4")
    assert _press(ctx, "DOWN")
    assert state.digit == 0
    _press(ctx, "7")
    cells = _cells(ctx, tab)
    assert cells[0, 0, D.PARAM] == 0x40
    assert cells[1, 0, D.PARAM] == 0x70


def test_a_click_between_two_digits_resets_the_sub_position():
    """``set_caret`` is the other way in, and the grid's own click uses it."""
    ctx = FakeCtx()
    _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.step = 0
    _at(ctx, D.INSTRUMENT)
    _press(ctx, "0")
    assert state.digit == 1
    sirens_mode.set_caret(ctx, row=8, channel=1)
    assert state.digit == 0


def test_an_instrument_number_past_the_id_space_writes_nothing():
    """Ids are minted out of a space bounded by ``MAX_INSTRUMENTS``, so ``80``
    names a slot no song can hold: silence in the grid, and a typo nobody can
    see."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, D.INSTRUMENT)
    assert not _press(ctx, "8")
    assert _cells(ctx, tab)[0, 0, D.INSTRUMENT] == notes.EMPTY


def test_the_volume_column_takes_one_digit_and_covers_the_engines_range():
    from warlock.studio.sirens import instruments as inst

    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, D.VOLUME)
    assert _press(ctx, "f")
    assert _cells(ctx, tab)[0, 0, D.VOLUME] == inst.MAX_VOLUME
    assert sirens_mode.ensure(ctx).digit == 0


# --- the effect column --------------------------------------------------------


def test_an_effect_letter_the_engine_does_not_have_writes_nothing():
    """``synth.EFFECT_NAMES`` is the authority. An id the tick loop has no
    handler for draws as ``?`` and plays as silence, which the person who typed
    it cannot tell from a broken synthesiser."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _at(ctx, D.EFFECT)
    assert "e" not in {name.lower() for name, _ in synth.EFFECT_NAMES.values()}
    assert not _press(ctx, "e")
    assert _cells(ctx, tab)[0, 0, D.EFFECT] == notes.EMPTY


@pytest.mark.parametrize("effect", sorted(synth.EFFECT_NAMES))
def test_every_effect_the_engine_implements_can_be_typed(effect):
    """The claim the manual's effect table makes, asserted rather than read."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, D.EFFECT)
    assert _press(ctx, synth.EFFECT_NAMES[effect][0].lower())
    assert _cells(ctx, tab)[0, 0, D.EFFECT] == effect


def test_the_piano_row_still_does_not_fire_outside_the_note_column():
    """The column-0 guard is correct and had to survive this: ``b`` is a note
    in the first column and the jump effect in the fourth."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _at(ctx, D.EFFECT)
    _press(ctx, "b")
    cells = _cells(ctx, tab)
    assert cells[0, 0, D.NOTE] == notes.EMPTY
    assert cells[0, 0, D.EFFECT] == synth.FX_JUMP


# --- clearing -----------------------------------------------------------------


@pytest.mark.parametrize("column", sorted(range(D.COLUMNS)))
def test_delete_blanks_the_cell_under_the_caret_in_every_column(column):
    """And only that column: a wrong instrument number is taken back without
    losing the note beside it."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    state.step = 0
    for index in range(D.COLUMNS):
        tab.doc.set_cell(state.pattern, 0, 0, index, 1)
    _at(ctx, column)
    assert _press(ctx, "DELETE")
    cells = _cells(ctx, tab)
    assert cells[0, 0, column] == notes.EMPTY
    others = [int(cells[0, 0, one]) for one in range(D.COLUMNS) if one != column]
    assert others == [1] * (D.COLUMNS - 1)


def test_delete_over_a_selection_still_clears_every_column():
    """A block is a rectangle over rows and channels and has never had a column
    axis to narrow along."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    for index in range(D.COLUMNS):
        tab.doc.set_cell(state.pattern, 0, 0, index, 1)
    _at(ctx, D.PARAM)
    state.anchor = (0, 0)
    assert _press(ctx, "DELETE")
    assert (_cells(ctx, tab)[0, 0, :] == notes.EMPTY).all()


def test_clearing_an_empty_cell_pushes_no_history():
    """``set_cells`` refuses a no-op, and Delete is not special."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _at(ctx, D.VOLUME)
    head = tab.doc.history.head
    _press(ctx, "DELETE")
    assert tab.doc.history.head == head


# --- what it sounds like ------------------------------------------------------
#
# The two assertions that make this file worth more than a spelling check.


def _row_samples(doc: Any) -> int:
    """How many samples one row lasts, from the document's own tempo."""
    return int(round(synth.SAMPLE_RATE * 60.0 / (doc.tempo * D.ROWS_PER_BEAT)))


def _note_then(ctx: FakeCtx, key: str, mod: int = 0) -> Any:
    """A C-4 at row 0 and one more note-column key at row 4, both through keys."""
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, D.NOTE)
    _press(ctx, "z")
    _at(ctx, D.NOTE, row=4)
    _press(ctx, key, mod)
    return tab


def test_a_released_note_sounds_where_a_cut_one_is_silent():
    """The whole point of ``Sequence.release``, asserted through the renderer.

    A cut is ``Voice.cut`` and stops the voice dead; a release runs the tail the
    envelope editor goes to the most trouble to draw. Reading the cell back
    would only prove the key wrote *a* number -- this proves it wrote the one
    the synthesiser acts on, which is the difference between this landing and
    the four before it.
    """
    import pygame

    cut_ctx, release_ctx = FakeCtx(), FakeCtx()
    cut_tab = _note_then(cut_ctx, "BACKQUOTE")
    release_tab = _note_then(release_ctx, "BACKQUOTE", pygame.KMOD_SHIFT)
    assert _cells(cut_ctx, cut_tab)[4, 0, D.NOTE] == notes.NOTE_OFF
    assert _cells(release_ctx, release_tab)[4, 0, D.NOTE] == notes.NOTE_RELEASE

    cut, _loop = synth.render(cut_tab.doc)
    released, _loop = synth.render(release_tab.doc)
    row = _row_samples(cut_tab.doc)
    # From just after the release row to well inside the default instrument's
    # five-tick decay. The margin at the front is the decimation filter's, which
    # smears an edge across a handful of samples.
    window = slice(4 * row + 441, 4 * row + 2646)
    assert np.abs(cut[window]).max() < 0.01
    assert np.abs(released[window]).max() > 0.05
    # And they are the same song right up to the row where they differ.
    assert np.allclose(cut[: 4 * row - 441], released[: 4 * row - 441])


def test_a_tempo_effect_typed_through_the_keyboard_changes_the_song():
    """The round trip whose absence let the gap ship: keys in, audio out.

    ``F3C`` is 60 BPM against a new song's 150, so the same pattern has to take
    appreciably longer to play. Nothing here reads a cell -- if the effect
    column took a keystroke and the synthesiser ignored it, the length would not
    move and this is the test that would say so.
    """
    ctx = FakeCtx()
    tab = _tab(ctx)
    sirens_mode.ensure(ctx).step = 0
    _at(ctx, D.NOTE)
    _press(ctx, "z")
    before, _loop = synth.render(tab.doc)

    _at(ctx, D.EFFECT)
    assert _press(ctx, "f")
    _at(ctx, D.PARAM)
    for key in ("3", "c"):
        assert _press(ctx, key)
    after, _loop = synth.render(tab.doc)

    assert len(after) > len(before) * 2
