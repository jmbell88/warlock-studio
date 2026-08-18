"""``conftest._selects_gpu``: which ``-m`` expressions count as asking for the
gpu lane.

The answer gates two things that must never fire on an ordinary run and must
always fire on a real one: the xdist refusal (``-m gpu`` under ``-n 8`` is N
simultaneous 7 GB loads onto one card) and the exemption from the
``WARLOCK_HOME`` / ``memlog.system_memory`` pins (pinned, the lane goes green
as skips that loaded nothing). A negated mention in *any* spelling therefore
must not count -- the old lookbehind regex could not see ``not`` through a
parenthesis, so ``-m "not (gpu or perf)"``, a legal spelling of the default
run, was refused under the default ``-n 8`` and silently unpinned under
``-n 0``.
"""

from __future__ import annotations

import pytest
from conftest import _selects_gpu


class _Config:
    """Just enough of a pytest config: ``getoption("-m")``."""

    def __init__(self, markexpr: str | None) -> None:
        self.markexpr = markexpr

    def getoption(self, name: str, default=None):
        assert name == "-m"
        return self.markexpr


@pytest.mark.parametrize(
    "expr",
    [
        "gpu",
        "gpu or perf",
        "perf or gpu",
        "gpu and not slow",
        "(gpu)",
    ],
)
def test_a_genuine_ask_selects_the_lane(expr):
    assert _selects_gpu(_Config(expr)) is True


@pytest.mark.parametrize(
    "expr",
    [
        # The default expression, and every legal respelling of it: a negated
        # mention is not an ask, however it is parenthesised.
        "not gpu and not perf",
        "not gpu",
        "not (gpu or perf)",
        "not (perf or gpu)",
        "not(gpu or perf)",
        "not ( gpu )",
        # An expression that merely fails to exclude gpu is not asking for it:
        # it admits an unmarked test too, so it is an ordinary filtered run.
        "not perf",
        # No expression at all.
        "",
        "   ",
        None,
        # Other lanes.
        "perf",
        # Nonsense pytest itself will refuse selects nothing here.
        "gpu or",
    ],
)
def test_anything_short_of_an_ask_does_not(expr):
    assert _selects_gpu(_Config(expr)) is False
