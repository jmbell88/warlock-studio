"""The inspector's name and tag fields, and the write they must not drop.

``TaskRunner.submit`` refuses a key already in flight, and ``widgets.input_text``
hands back the *passed* value on a frame with no keystroke -- so a refused
write left the next frame comparing the stale cached job against itself, with
nothing anywhere still holding what the user typed.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio.panes import inspector


class _Ctx:
    """Accepts a submit only when the key is not already in flight, which is
    ``TaskRunner.submit``'s whole contract."""

    def __init__(self, svc: Any = None) -> None:
        self.svc = svc
        self.inflight: set[str] = set()
        self.sent: list[tuple[str, dict]] = []

    def submit(self, key: str, _fn: Any, _svc: Any, _job_id: str, payload: dict) -> bool:
        if key in self.inflight:
            return False
        self.inflight.add(key)
        self.sent.append((key, payload))
        return True


@pytest.fixture(autouse=True)
def _clean():
    inspector._unsent.clear()
    yield
    inspector._unsent.clear()


def _write(ctx, typed: str, current: str) -> None:
    inspector._write_field(ctx, "name:abc", "abc", lambda v: {"name": v}, typed, current)


def test_an_edit_is_sent() -> None:
    ctx = _Ctx()
    _write(ctx, "chest", "")
    assert ctx.sent == [("name:abc", {"name": "chest"})]
    assert inspector._unsent == {}, "an accepted write clears what it sent"


def test_keystrokes_typed_during_an_in_flight_write_are_not_lost() -> None:
    """The defect: the refused submit dropped them, and the frame after -- with
    ``input_text`` echoing the stale cached value back -- had nothing left to
    resend. Fast typing committed a prefix."""
    ctx = _Ctx()
    _write(ctx, "che", "")  # accepted; now in flight
    _write(ctx, "chest", "")  # refused
    _write(ctx, "chest", "")  # still refused, still remembered

    assert ctx.sent == [("name:abc", {"name": "che"})]

    ctx.inflight.clear()  # the write lands
    _write(ctx, "chest", "")

    assert ctx.sent[-1] == ("name:abc", {"name": "chest"})
    assert inspector._unsent == {}, "an accepted write clears what it sent"


def test_nothing_is_held_for_a_field_nobody_edited() -> None:
    ctx = _Ctx()
    _write(ctx, "chest", "chest")
    assert ctx.sent == []
    assert inspector._unsent == {}
