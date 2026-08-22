"""The status-bar tip: the door a refused *gesture* goes through.

The distinction being pinned here is the one the toasts had lost, and it is
stated in :class:`inker_state.Tip`'s docstring rather than in this file:
a tip answers a gesture, a toast reports a job. These tests hold the two
properties that make a tip safe to reach for -- it expires, and its remedy is
a name rather than a function, so it can never offer something the menus and
the keyboard do not have.
"""

from __future__ import annotations

from warlock.studio import inker_state


def test_saying_something_puts_it_under_the_canvas():
    state = inker_state.InkerState()
    assert state.tip is None
    state.say("The layer is locked.")
    assert state.tip is not None and state.tip.text == "The layer is locked."
    assert state.tip.alive()


def test_a_tip_stops_being_drawn_once_it_is_stale():
    state = inker_state.InkerState()
    state.say("The layer is locked.")
    assert not state.tip.alive(state.tip.at + inker_state.TIP_SECONDS)
    assert state.tip.alive(state.tip.at + inker_state.TIP_SECONDS - 0.5)


def test_a_remedy_is_a_name_and_not_a_callable():
    state = inker_state.InkerState()
    state.say("Snapping is manual.", remedy="tile_auto", remedy_label="Switch to Auto")
    assert isinstance(state.tip.remedy, str)
    assert state.tip.remedy_label == "Switch to Auto"


def test_the_newest_tip_replaces_the_last_one():
    state = inker_state.InkerState()
    state.say("first")
    first = state.tip
    state.say("second")
    assert state.tip is not first and state.tip.text == "second"
