"""The overflow toolbar's arithmetic, and the value formatting beside it.

``toolbar.plan`` is pure on purpose (REDESIGN.md wave 2): the failure it exists
to prevent -- a button clipped off the end of a row -- is invisible at UI scale
1.0 in a wide window, which is the only configuration the GL smoke suite draws.
Testing the tiering as numbers is what lets every width be checked instead of
one.
"""

from __future__ import annotations

import pytest

from warlock.studio import toolbar, widgets
from warlock.studio.toolbar import FULL, ICON, MENU

# Four items: two important (priority 0) and two extras (priority 1). 100 px
# each with a label, 30 as a glyph; the overflow button is 30 too.
WIDE = [100.0, 100.0, 100.0, 100.0]
NARROW = [30.0, 30.0, 30.0, 30.0]
PRIORITIES = [0, 0, 1, 1]
UNPINNED = [False] * 4


def _plan(avail: float, *, pinned=None, priorities=None, gap: float = 0.0):
    return toolbar.plan(
        WIDE,
        NARROW,
        priorities or PRIORITIES,
        pinned or UNPINNED,
        avail,
        30.0,
        gap=gap,
    )


def test_a_row_with_room_keeps_every_label():
    assert _plan(400.0) == [FULL] * 4


def test_the_lowest_priority_group_is_the_first_to_lose_its_labels():
    """Collapse order is what makes the degradation legible: the row's reason
    for existing (priority 0) is the last thing to be abbreviated."""
    assert _plan(300.0) == [FULL, FULL, ICON, ICON]


def test_tiering_is_all_or_nothing_within_a_group():
    """``segmented_control``'s rule. A row where one of two equals shows a word
    and the other shows a picture is one control saying two kinds of thing --
    and which one is which would change as the window is dragged."""
    # 100+100+30+30 = 260 does not fit in 250; demoting *one* priority-0 item
    # would (30+100+30+30 = 190), and that is exactly what must not happen.
    tiers = _plan(250.0)
    assert tiers[0] == tiers[1]
    assert tiers == [ICON] * 4


def test_a_group_that_still_does_not_fit_moves_into_the_menu():
    """Two glyphs (60) plus the ... button (30) is 90, and the extras go."""
    assert _plan(90.0) == [ICON, ICON, MENU, MENU]


def test_the_overflow_button_is_measured_or_the_row_it_creates_overflows():
    """A bar that "fits" at exactly the width where the ... it just grew hangs
    off the end has measured everything except the thing it added.

    One pixel under, the two glyphs it was keeping go into the menu as well --
    which is the honest answer, since 60 px of buttons plus the ... they need
    is 90 and there are 89.
    """
    assert _plan(90.0) == [ICON, ICON, MENU, MENU]
    assert _plan(89.0) == [MENU, MENU, MENU, MENU]


def test_a_pinned_item_never_hides_in_the_menu():
    """Destructive actions and transports: a Delete behind ... is a Delete
    somebody finds by accident, and a play button that moves house when the
    window is resized is not a transport."""
    tiers = _plan(40.0, pinned=[True, False, False, False])
    assert tiers[0] == ICON
    assert tiers[1:] == [MENU] * 3


def test_a_row_of_pinned_items_reports_that_it_does_not_fit():
    """There is nothing left to give, and drawing the truth beats pretending:
    the caller gets glyphs and the clipping is real rather than hidden."""
    assert _plan(10.0, pinned=[True] * 4) == [ICON] * 4


def test_the_gap_between_items_is_part_of_the_width():
    """Four 30 px glyphs with 8 px between them need 144, not 120 -- the same
    unscaled-spacing error ``widgets.grid_width`` exists to end."""
    assert _plan(144.0, gap=8.0) == [ICON] * 4
    assert _plan(143.0, gap=8.0) != [ICON] * 4


def test_an_empty_row_plans_nothing():
    assert toolbar.plan([], [], [], [], 0.0, 30.0) == []


# --- value formatting --------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.0, 360.0, "%.0f"),
        (-180.0, 180.0, "%.0f"),
        (0.0, 32.0, "%.1f"),
        (0.0, 10.0, "%.1f"),
        (0.05, 8.0, "%.2f"),
        (0.0, 1.0, "%.2f"),
    ],
)
def test_the_format_narrows_as_the_range_does(low, high, expected):
    """imgui's ``%.3f`` for everything is how an angle came to read 45.000."""
    assert widgets.float_format(low, high) == expected


def test_a_step_outvotes_the_span():
    """A slider that moves in halves over 0..100 needs a decimal the span alone
    would talk it out of."""
    assert widgets.float_format(0.0, 100.0) == "%.0f"
    assert widgets.float_format(0.0, 100.0, 0.5) == "%.1f"


class _Slider:
    """Enough imgui for ``labeled_slider_float`` and nothing else."""

    def __init__(self, returns: float):
        self.returns = returns
        self.call: tuple = ()

    def slider_float(self, label, value, low, high, fmt="%.3f"):
        self.call = (label, value, low, high, fmt)
        return True, self.returns

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _slider(monkeypatch, returns, *args, **kwargs):
    stub = _Slider(returns)
    monkeypatch.setattr(widgets, "imgui", stub)
    result = widgets.labeled_slider_float(*args, **kwargs)
    return stub, result


def test_a_zero_to_one_slider_is_a_percentage_without_being_told(monkeypatch):
    """Every 0..1 slider in this app is one -- opacity, hardness, flow, weight
    -- and a rule that has to be remembered at eleven call sites is a rule that
    will be forgotten at the twelfth."""
    stub, (changed, value) = _slider(monkeypatch, 40.0, "Opacity", 0.25, 0.0, 1.0)
    assert stub.call[1:] == (25.0, 0.0, 100.0, "%.0f%%")
    assert changed and value == pytest.approx(0.4)


def test_a_range_that_does_not_start_at_zero_has_to_say_so(monkeypatch):
    """The inference cannot see 0.05..1.0, which is why the flag exists."""
    stub, _ = _slider(monkeypatch, 50.0, "Opacity", 0.5, 0.05, 1.0)
    assert stub.call[3] == 1.0
    stub, (_, value) = _slider(
        monkeypatch, 50.0, "Opacity", 0.5, 0.05, 1.0, percent=True
    )
    assert stub.call[1:] == (50.0, 5.0, 100.0, "%.0f%%")
    assert value == pytest.approx(0.5)


def test_an_explicit_format_carries_a_unit_the_range_cannot_know(monkeypatch):
    stub, _ = _slider(monkeypatch, 2.0, "X", 2.0, 0.05, 8.0, fmt="%.2fx")
    assert stub.call[4] == "%.2fx"
