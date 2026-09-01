"""The overflow toolbar's arithmetic, and the value formatting beside it.

``toolbar.plan`` is pure on purpose (the UI redesign, wave 2): the failure it exists
to prevent -- a button clipped off the end of a row -- is invisible at UI scale
1.0 in a wide window, which is the only configuration the GL smoke suite draws.
Testing the tiering as numbers is what lets every width be checked instead of
one.
"""

from __future__ import annotations

import pytest

from warlock.studio import controls, toolbar, widgets
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
    """Enough imgui for ``labeled_slider_float`` and nothing else.

    Patched onto ``controls`` as well as ``widgets``: the labelled helpers draw
    through the presentational layer now (that is what gets them the disabled
    treatment, the error ring, the probe census and the typed-entry clamp), so
    the widget call this stub is here to observe is made from there.
    ``flags`` is accepted because the clamp is injected as a keyword.
    """

    def __init__(self, returns: float):
        self.returns = returns
        self.call: tuple = ()

    def slider_float(self, label, value, low, high, fmt="%.3f", flags=0):
        self.call = (label, value, low, high, fmt)
        return True, self.returns

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _slider(monkeypatch, returns, *args, **kwargs):
    stub = _Slider(returns)
    monkeypatch.setattr(widgets, "imgui", stub)
    monkeypatch.setattr(controls, "imgui", stub)
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


# --- Fields (wave 1) --------------------------------------------------------
#
# A context bar is a row of settings, not of buttons, and the tiering has to
# treat the two alike or the collapse order becomes a rule about widget kinds
# rather than about importance. These pin the two halves of that: a Field
# reports both of its widths as numbers, and a Field's priority competes with
# an Item's directly inside ``plan``.


def test_a_field_with_no_compact_size_reports_one_width_twice():
    field = toolbar.Field("size", "Size", draw=lambda compact: None, width=120)
    wide, narrow = field.widths()
    assert wide == narrow > 0


def test_a_field_narrows_before_a_more_important_button_does():
    field = toolbar.Field(
        "size", "Size", draw=lambda compact: None, width=120, compact=48, priority=1
    )
    wide, narrow = field.widths()
    tiers = toolbar.plan(
        [100.0, wide], [30.0, narrow], [0, field.priority], [False, False],
        100.0 + narrow, 30.0,
    )
    assert tiers == [FULL, ICON]


# --- the trailing block's own tier ------------------------------------------
#
# ``trailing`` is measured and subtracted *before* the tiers are chosen, so
# whatever sits in it wins against the tool's own settings. What the two tiers
# add is a bound on that: a block that cannot be made to fit gives up its own
# controls rather than pushing the row off the end -- which matters because
# ``toolbar`` continues it with ``same_line_or_wrap``, and a wrapped trailing
# block above Inker's canvas is a second row, the one thing that bar forbids.
#
# The tier is decided from the row's *smallest* width, not its largest. See
# ``trailing_compact``: measured at the app's default size the other rule
# folded Inker's symmetry mirrors away on the commonest tool, which is the
# opposite of what putting them on the bar was for.

FULL_W, COMPACT_W, GAP = 217.0, 93.0, 8.0
#: Inker's context bar at 1600x950, measured rather than assumed: the mode rail
#: takes ~70 px that a two-sidebars sum misses.
BAR_AVAIL = 835.0
#: The brush's row -- two labelled buttons and five fields -- at each tier.
BRUSH_FULL, BRUSH_MIN = 689.0, 376.0


def test_the_block_keeps_its_controls_while_the_row_can_be_squeezed_to_fit():
    """The measured case this policy exists for. The row cannot fit *with its
    labels* beside the mirrors (689 + 8 + 217 = 914 > 835), and that is not the
    question: compacted it needs 376, so both survive."""
    assert (
        toolbar.trailing_compact(BRUSH_MIN, BAR_AVAIL, FULL_W, COMPACT_W, gap=GAP)
        is False
    )
    assert BRUSH_FULL + GAP + FULL_W > BAR_AVAIL


def test_the_block_folds_only_when_even_a_compacted_row_cannot_fit():
    assert (
        toolbar.trailing_compact(800.0, BAR_AVAIL, FULL_W, COMPACT_W, gap=GAP)
        is True
    )


def test_the_trailing_block_never_pushes_the_row_past_the_edge():
    """The property that matters, swept rather than sampled: wherever the full
    block would have overrun even a minimal row, the compact one is chosen."""
    for avail in range(200, 1300, 10):
        for row_min in (120.0, 376.0, 500.0):
            compact = toolbar.trailing_compact(
                row_min, float(avail), FULL_W, COMPACT_W, gap=GAP
            )
            if not compact:
                assert row_min + GAP + FULL_W <= avail


def test_a_row_that_cannot_fit_at_all_still_folds_the_block_first():
    """When nothing fits the row is allowed to overrun -- ``plan`` says so
    rather than pretending -- but it must not overrun by more than it has to."""
    assert (
        toolbar.trailing_compact(2000.0, 300.0, FULL_W, COMPACT_W, gap=GAP) is True
    )


def test_a_two_tuple_trailing_still_means_one_tier():
    """Every existing caller passes ``(width, draw)`` -- ``_float_bar`` and
    ``_gesture_bar`` among them -- and must keep working untouched."""
    full, compact, draw = toolbar.trailing_widths((120.0, lambda: None))
    assert (full, compact) == (120.0, 120.0)
    assert draw is not None


def test_a_trailing_object_carries_both_widths():
    block = toolbar.Trailing(FULL_W, COMPACT_W, lambda _compact: None)
    full, compact, draw = toolbar.trailing_widths(block)
    assert (full, compact) == (FULL_W, COMPACT_W)
    assert draw is not None
