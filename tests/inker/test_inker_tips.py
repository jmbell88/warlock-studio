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
    """Asserted a hair *past* the boundary rather than on it: ``at`` is a
    ``monotonic`` reading, so ``(at + TIP_SECONDS) - at`` is not exactly
    ``TIP_SECONDS`` -- at an uptime of 2044.5 s it comes back
    5.999999999999773, and the test's verdict turned on how long the machine
    had been up."""
    state = inker_state.InkerState()
    state.say("The layer is locked.")
    assert not state.tip.alive(state.tip.at + inker_state.TIP_SECONDS + 1e-6)
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


def test_the_canvas_raises_no_toasts_at_all():
    """The rule, as a scan: **a tip answers a gesture; a toast reports a job.**

    Every refusal in ``inker_canvas`` is an answer to something the user just
    did with the mouse or a key, so every one of them is a tip. A toast belongs
    to work that finished while the user was looking somewhere else -- "Saved.",
    "Exported to ...", "Cannot export: ..." -- and none of that happens here.
    Stated as a scan because the failure mode is one line added in a hurry.
    """

    import inspect

    from warlock.studio.panes import inker_canvas

    source = inspect.getsource(inker_canvas)
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "ctx.toast(" not in body
    assert "toast_once(" not in body


def test_every_remedy_names_an_op_that_exists():
    """A tip may only offer what the menus and the keyboard also have."""

    import inspect

    from warlock.studio import inker_mode, inker_ops
    from warlock.studio.panes import inker_canvas

    names = {op.name for op in inker_ops.OPS}
    for module in (inker_canvas, inker_mode):
        source = inspect.getsource(module)
        for chunk in source.split("remedy=")[1:]:
            name = chunk.split(",")[0].strip().strip('"').strip("'")
            if name.startswith(("remedy", "op.")) or not name:
                continue
            assert name in names, f"{module.__name__} offers a remedy named {name!r}"
