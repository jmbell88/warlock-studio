"""The Sirens block clipboard: copy, cut and paste on Ctrl+C/X/V.

The last thing the mode's Experimental chip named, and the file
``modes.MATURITY``'s comment cites for its removal. Everything here runs the
real ``handle_key`` against ``FakeCtx`` from ``test_sirens_mode``; nothing
touches pygame's display or the mixer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from test_sirens_keys import _at, _cells, _press
from test_sirens_mode import FakeCtx, _tab

from warlock.studio import sirens_mode
from warlock.studio.sirens import document as D
from warlock.studio.sirens import notes


def _ctrl(ctx: FakeCtx, key: str) -> bool:
    import pygame

    return _press(ctx, key, pygame.KMOD_CTRL)


def _fill(ctx: FakeCtx, tab: Any, row: int, chan: int, rows: int, chans: int) -> np.ndarray:
    """Stamp a recognisable block and return a copy of what was written."""
    state = sirens_mode.ensure(ctx)
    block = np.arange(rows * chans * D.COLUMNS, dtype=np.int16).reshape(rows, chans, D.COLUMNS) + 1
    tab.doc.set_cells(state.pattern, row, chan, 0, block)
    return block.copy()


def _select(ctx: FakeCtx, row: int, chan: int, rows: int, chans: int) -> None:
    state = sirens_mode.ensure(ctx)
    state.anchor = (row, chan)
    sirens_mode.set_caret(ctx, row=row + rows - 1, channel=chan + chans - 1)


def test_copy_then_paste_round_trips_a_block():
    ctx = FakeCtx()
    tab = _tab(ctx)
    block = _fill(ctx, tab, 0, 0, 3, 2)
    _select(ctx, 0, 0, 3, 2)
    assert _ctrl(ctx, "c")
    state = sirens_mode.ensure(ctx)
    state.anchor = None
    _at(ctx, D.NOTE, row=8)
    assert _ctrl(ctx, "v")
    assert (_cells(ctx, tab)[8:11, 0:2, :] == block).all()
    # The source is untouched: a copy is not a move.
    assert (_cells(ctx, tab)[0:3, 0:2, :] == block).all()


def test_the_clipboard_is_a_private_copy_of_the_source():
    """Editing the source after a copy must not change what a paste puts down."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    block = _fill(ctx, tab, 0, 0, 2, 1)
    _select(ctx, 0, 0, 2, 1)
    _ctrl(ctx, "c")
    state = sirens_mode.ensure(ctx)
    tab.doc.clear_cells(state.pattern, 0, 0, 2, 1)
    state.anchor = None
    _at(ctx, D.NOTE, row=4)
    _ctrl(ctx, "v")
    assert (_cells(ctx, tab)[4:6, 0:1, :] == block).all()


def test_copy_with_no_selection_takes_the_cell_under_the_caret():
    ctx = FakeCtx()
    tab = _tab(ctx)
    block = _fill(ctx, tab, 5, 1, 1, 1)
    sirens_mode.set_caret(ctx, row=5, channel=1, column=D.NOTE)
    assert _ctrl(ctx, "c")
    assert sirens_mode.ensure(ctx).clip.shape == (1, 1, D.COLUMNS)
    _at(ctx, D.NOTE, row=0)
    _ctrl(ctx, "v")
    assert (_cells(ctx, tab)[0:1, 0:1, :] == block).all()


def test_cut_is_one_undo_step():
    ctx = FakeCtx()
    tab = _tab(ctx)
    block = _fill(ctx, tab, 0, 0, 4, 2)
    head = tab.doc.history.head
    _select(ctx, 0, 0, 4, 2)
    assert _ctrl(ctx, "x")
    assert (_cells(ctx, tab)[0:4, 0:2, :] == notes.EMPTY).all()
    assert (sirens_mode.ensure(ctx).clip == block).all()
    assert tab.doc.history.head == head + 1
    assert tab.doc.undo()
    assert (_cells(ctx, tab)[0:4, 0:2, :] == block).all()


def test_paste_clips_at_the_pattern_edge():
    ctx = FakeCtx()
    tab = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    pattern = tab.doc.pattern(state.pattern)
    block = _fill(ctx, tab, 0, 0, 4, 1)
    _select(ctx, 0, 0, 4, 1)
    _ctrl(ctx, "c")
    state.anchor = None
    last = pattern.rows - 2
    _at(ctx, D.NOTE, row=last)
    assert _ctrl(ctx, "v")
    assert (_cells(ctx, tab)[last:, 0:1, :] == block[:2]).all()


def test_paste_with_nothing_on_the_clipboard_pushes_no_history():
    """Mirrors ``test_clearing_an_empty_cell_pushes_no_history``: a no-op is
    not an edit, and must not cost an undo step or arm the renderer."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    assert sirens_mode.ensure(ctx).clip is None
    head = tab.doc.history.head
    tab.render_dirty = False
    _at(ctx, D.NOTE)
    assert _ctrl(ctx, "v")  # consumed -- the key is Sirens' -- but nothing done
    assert tab.doc.history.head == head
    assert tab.render_dirty is False


def test_pasting_what_is_already_there_pushes_no_history():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _fill(ctx, tab, 0, 0, 2, 1)
    _select(ctx, 0, 0, 2, 1)
    _ctrl(ctx, "c")
    sirens_mode.ensure(ctx).anchor = None
    _at(ctx, D.NOTE, row=0)
    head = tab.doc.history.head
    _ctrl(ctx, "v")
    assert tab.doc.history.head == head


def test_the_clipboard_crosses_tabs():
    """App-level, like Inker's ``cel_clip``: the one thing a clipboard does
    that a second order-list entry cannot."""
    ctx = FakeCtx()
    first = _tab(ctx)
    block = _fill(ctx, first, 0, 0, 2, 2)
    _select(ctx, 0, 0, 2, 2)
    _ctrl(ctx, "c")
    second = _tab(ctx)
    state = sirens_mode.ensure(ctx)
    assert state.active is second
    assert state.clip is not None, "activating another tab must not clear the clipboard"
    _at(ctx, D.NOTE, row=0)
    assert _ctrl(ctx, "v")
    assert (_cells(ctx, second)[0:2, 0:2, :] == block).all()


def test_moving_the_caret_keeps_the_clipboard():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _fill(ctx, tab, 0, 0, 1, 1)
    _at(ctx, D.NOTE)
    _ctrl(ctx, "c")
    _press(ctx, "DOWN")
    _press(ctx, "RIGHT")
    _press(ctx, "ESCAPE")
    assert sirens_mode.ensure(ctx).clip is not None


@pytest.mark.parametrize("key", ["x", "v"])
def test_cut_and_paste_are_refused_on_a_busy_tab(key):
    ctx = FakeCtx()
    tab = _tab(ctx)
    block = _fill(ctx, tab, 0, 0, 2, 1)
    _select(ctx, 0, 0, 2, 1)
    _ctrl(ctx, "c")
    sirens_mode.ensure(ctx).anchor = None
    _at(ctx, D.NOTE, row=0)
    tab.saving = True
    head = tab.doc.history.head
    assert _ctrl(ctx, key)  # consumed, so the app does not fall through
    assert tab.doc.history.head == head
    assert (_cells(ctx, tab)[0:2, 0:1, :] == block).all()


def test_copy_is_allowed_on_a_busy_tab():
    """Copy reads and pushes nothing, so a tab mid-save can still be copied from."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    _fill(ctx, tab, 0, 0, 1, 1)
    _at(ctx, D.NOTE)
    tab.saving = True
    assert _ctrl(ctx, "c")
    assert sirens_mode.ensure(ctx).clip is not None


def test_cut_and_paste_arm_the_renderer():
    ctx = FakeCtx()
    tab = _tab(ctx)
    _fill(ctx, tab, 0, 0, 1, 1)
    _at(ctx, D.NOTE)
    tab.render_dirty = False
    assert sirens_mode.cut_selection(ctx)
    assert tab.render_dirty
    tab.render_dirty = False
    _at(ctx, D.NOTE, row=3)
    assert sirens_mode.paste(ctx)
    assert tab.render_dirty
