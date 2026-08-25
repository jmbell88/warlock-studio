"""The frame-loop notice for a history that dropped steps to stay in memory.

``studio.undo`` bounds the stack by bytes and, past ``UNDO_HARD_BYTES``, stops
honouring the depth floor entirely -- so one rotate on a large document can
take most of the history with it. That is the right trade (the alternative is
an out-of-memory kill holding an unsaved painting) but it is invisible: the
undo the user reaches for a minute later is simply not there.

``pump_undo_trim`` is the half that says so. What is pinned here is that it
says it once per event rather than once per frame, that a second event still
gets through, and that neither a quiet session nor a missing Inker state can
make it speak.
"""

from __future__ import annotations

from typing import Any

from warlock.studio import inker, inker_mode
from warlock.studio.inker_state import InkerDoc, InkerState


class _Ctx:
    def __init__(self, state: Any) -> None:
        self.state = _AppState(state)
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, kind: str = "info", **_kw: Any) -> None:
        self.toasts.append((message, kind))


class _AppState:
    def __init__(self, inker_state: Any) -> None:
        self.inker = inker_state


def _tab(title: str = "Big painting") -> InkerDoc:
    return InkerDoc(doc=inker.Document.blank(4, 4), title=title)


def _state(*tabs: InkerDoc) -> InkerState:
    state = InkerState()
    state.docs = list(tabs)
    return state


def test_a_quiet_history_says_nothing():
    tab = _tab()
    ctx = _Ctx(_state(tab))
    inker_mode.pump_undo_trim(ctx)
    inker_mode.pump_undo_trim(ctx)
    assert ctx.toasts == []


def test_a_trim_is_reported_once_and_not_once_a_frame():
    """The pump runs every frame in every mode. Without the per-tab mark this
    would raise the same toast sixty times a second for as long as the count
    stayed different from zero."""
    tab = _tab()
    ctx = _Ctx(_state(tab))
    tab.doc.history.trimmed = 3

    inker_mode.pump_undo_trim(ctx)
    assert len(ctx.toasts) == 1
    message, level = ctx.toasts[0]
    assert "Big painting" in message
    assert level == "warn"

    for _ in range(5):
        inker_mode.pump_undo_trim(ctx)
    assert len(ctx.toasts) == 1


def test_a_second_trim_is_reported_again():
    tab = _tab()
    ctx = _Ctx(_state(tab))
    tab.doc.history.trimmed = 1
    inker_mode.pump_undo_trim(ctx)
    tab.doc.history.trimmed = 2
    inker_mode.pump_undo_trim(ctx)
    assert len(ctx.toasts) == 2


def test_clearing_the_history_re_arms_without_announcing_itself():
    """``UndoStack.clear`` puts the counter back to zero with the history, so
    the mark has to be compared with ``!=`` and not ``>`` -- and the zero
    itself is not news."""
    tab = _tab()
    ctx = _Ctx(_state(tab))
    tab.doc.history.trimmed = 4
    inker_mode.pump_undo_trim(ctx)
    assert len(ctx.toasts) == 1

    tab.doc.history.clear()
    assert tab.doc.history.trimmed == 0
    inker_mode.pump_undo_trim(ctx)
    assert len(ctx.toasts) == 1
    assert tab.trim_seen == 0

    tab.doc.history.trimmed = 1
    inker_mode.pump_undo_trim(ctx)
    assert len(ctx.toasts) == 2


def test_each_tab_is_marked_on_its_own():
    one, two = _tab("One"), _tab("Two")
    ctx = _Ctx(_state(one, two))
    one.doc.history.trimmed = 1
    two.doc.history.trimmed = 1
    inker_mode.pump_undo_trim(ctx)
    assert [t[0].split(":")[0] for t in ctx.toasts] == [
        "Undo history trimmed on One",
        "Undo history trimmed on Two",
    ]


def test_a_session_that_never_opened_inker_pays_nothing():
    """The pump is called in *every* mode, so the state it reads is routinely
    absent -- ``ensure`` builds it lazily on the first visit to Inker."""
    ctx = _Ctx(None)
    inker_mode.pump_undo_trim(ctx)
    assert ctx.toasts == []
