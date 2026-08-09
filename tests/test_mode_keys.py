"""Section I: the global keyboard bindings and the dialog queues.

Two things are pinned here that are easy to undo. The mode-switch digits must
fire in *every* mode, including the three that consume every key they are
given; and ``ConfirmQueue`` must really queue, because the whole reason it did
not is the reason ``_request_quit`` had to nest three guards by hand.
"""

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace

import pygame
import pytest

from warlock.studio import dialogs, main, modes
from warlock.studio.state import AppState

# --- I75: Alt+1..8 -----------------------------------------------------------


def _app(mode: str = "3d") -> SimpleNamespace:
    """The smallest thing ``_shortcut`` needs to route a key.

    The mode helpers are bound off the real class rather than stubbed: they are
    the behaviour under test, and a stub here would pass whatever they did.
    """
    state = AppState()
    state.mode = state.mode_observed = state.previous_mode = mode
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(state=state, cache=SimpleNamespace(get=lambda _id: None)),
        viewer=SimpleNamespace(
            pose_mode=False,
            frame=lambda: None,
            set_wireframe=lambda v: None,
            set_turntable=lambda v: None,
            exit_compare=lambda: None,
        ),
    )
    for name in ("_note_mode", "_set_mode", "_escape_mode"):
        setattr(app, name, MethodType(getattr(main.App, name), app))
    return app


def _press(app: SimpleNamespace, key: int) -> None:
    main.App._shortcut(app, pygame.event.Event(pygame.KEYDOWN, key=key))


@pytest.fixture
def alt(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_ALT)


@pytest.fixture
def no_mods(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def test_the_digits_are_the_switch_order_and_nothing_else():
    """The binding is the picture on screen: the nth segment is the nth digit.

    Derived from ``MODES`` rather than written out, so a mode inserted into the
    switch cannot leave the shortcuts pointing one place to the left.
    """
    assert [modes.mode_for_digit(n) for n in range(1, len(modes.MODES) + 1)] == list(
        modes.KEYS
    )
    assert modes.mode_for_digit(len(modes.MODES) + 1) is None
    assert modes.mode_for_digit(0) is None
    for index, key in enumerate(modes.KEYS, start=1):
        assert modes.digit_for_mode(key) == index
    assert modes.digit_for_mode("quit") is None


@pytest.mark.parametrize("mode", sorted(modes.KEYS))
def test_alt_digit_switches_from_every_mode_including_the_workspaces(mode, alt):
    """Inker, Clay and Review consume every key they are handed -- deliberately,
    so F/W/S cannot act on a viewport they have replaced. The mode switch is
    therefore checked *above* them, or it would be the one binding that stops
    working exactly where the user is most likely to be stuck."""
    app = _app(mode)
    _press(app, pygame.K_4)  # the 4th segment
    assert app.app_ctx.state.mode == modes.KEYS[3]


def test_the_numpad_digits_switch_too(alt):
    app = _app("home")
    _press(app, pygame.K_KP3)
    assert app.app_ctx.state.mode == modes.KEYS[2]


def test_the_last_two_slots_are_alt_9_and_alt_0(alt):
    """Ten segments and nine digits above zero, so the tenth takes ``0`` -- the
    way every application with ten of anything does. That the *key* is 0 while
    the *slot* is 10 is a keyboard fact, which is why it lives in
    ``_digit_keys`` and in ``modes.digit_key_label`` and nowhere else."""
    app = _app("home")
    _press(app, pygame.K_9)
    assert app.app_ctx.state.mode == modes.KEYS[8]
    _press(app, pygame.K_0)
    assert app.app_ctx.state.mode == modes.KEYS[9]
    app = _app("home")
    _press(app, pygame.K_KP0)
    assert app.app_ctx.state.mode == modes.KEYS[9]


def test_a_digit_past_the_last_mode_still_maps_to_nothing():
    """Every digit key is now spoken for, so the out-of-range property is only
    assertable on the pure function -- which is where it was always the real
    guarantee. ``0`` is *not* the zeroth slot: there is no zeroth segment."""
    assert modes.mode_for_digit(len(modes.MODES) + 1) is None
    assert modes.mode_for_digit(0) is None


def test_a_bare_digit_is_not_a_mode_switch(no_mods):
    """Clay binds 1-4 to the element modes and Review binds 1-5 to reject
    reasons. Alt is what keeps those keys theirs."""
    app = _app("home")
    _press(app, pygame.K_4)
    assert app.app_ctx.state.mode == "home"


def test_the_mode_switch_is_checked_before_the_workspace_handlers():
    source = inspect.getsource(main.App._shortcut)
    assert source.index("_digit_keys()") < source.index("WORK_MODES")


def test_switching_to_home_resets_the_landing_view(alt):
    """The switch's own segment does this; a shortcut that did not would drop
    the user on whichever sub-view Home was last showing."""
    app = _app("3d")
    app.app_ctx.state.landing_view = "assets"
    _press(app, pygame.K_1)
    assert app.app_ctx.state.mode == "home"
    assert app.app_ctx.state.landing_view == "choose"


# --- I76: Esc out of the pass-through modes ----------------------------------


def test_esc_returns_to_the_mode_you_came_from(no_mods):
    app = _app("3d")
    state = app.app_ctx.state
    _press(app, pygame.K_F1)  # into the Manual
    assert state.mode == "manual"
    _press(app, pygame.K_ESCAPE)
    assert state.mode == "3d"


def test_esc_falls_back_to_home_when_there_is_nowhere_to_go(no_mods):
    """The app opens on Home, so ``previous_mode`` is routinely the mode you
    are already in. Bouncing off a stale value would be a switch nobody asked
    for."""
    app = _app("settings")
    app.app_ctx.state.previous_mode = "settings"
    _press(app, pygame.K_ESCAPE)
    assert app.app_ctx.state.mode == "home"


def test_esc_on_home_does_nothing(no_mods):
    """Home is the floor, not a place you escape from."""
    app = _app("home")
    app.app_ctx.state.previous_mode = "3d"
    _press(app, pygame.K_ESCAPE)
    assert app.app_ctx.state.mode == "home"


def test_previous_mode_is_sampled_per_key_event_not_per_frame(no_mods):
    """F1 changes the mode from inside ``_shortcut``, so a sample taken once at
    the top of the frame would still hold the mode from before it and Esc would
    go two steps back."""
    app = _app("2d")
    state = app.app_ctx.state
    state.mode = "3d"  # a landing tile clicked since the last keypress
    _press(app, pygame.K_F1)
    _press(app, pygame.K_ESCAPE)
    assert state.mode == "3d"


def test_esc_in_a_work_mode_is_still_the_pane_s(no_mods):
    """Esc means "drop what I am doing" in a mode that has something to drop --
    a comparison, a pose edit. It must not leave the mode as well."""
    app = _app("3d")
    app.app_ctx.state.comparing = "job-1"
    _press(app, pygame.K_ESCAPE)
    assert app.app_ctx.state.mode == "3d"
    assert app.app_ctx.state.comparing is None


def test_no_mode_shortcut_is_persisted():
    """``previous_mode``/``mode_observed`` join ``mode`` in never being
    written to settings: a remembered one would have no reader across
    launches, which is how the two halves drift."""
    source = inspect.getsource(main)
    for name in ("previous_mode", "mode_observed"):
        assert f'settings.set("{name}"' not in source


# --- I78: the queues actually queue ------------------------------------------


def test_a_second_confirm_is_queued_and_not_dropped():
    """The regression this section exists for. ``ask`` used to keep the first
    question and silently discard every later one, on the reasoning that the
    user cannot act twice between frames -- true of clicks, false of a finished
    task and a click landing in the same frame, and false of any code path that
    asks more than once."""
    queue = dialogs.ConfirmQueue()
    first = dialogs.Confirm(title="A", message="first")
    second = dialogs.Confirm(title="B", message="second")
    queue.ask(first)
    queue.ask(second)
    assert queue.pending is first
    assert queue.waiting == 1
    queue.dismiss()
    assert queue.pending is second
    assert queue.waiting == 0
    queue.dismiss()
    assert queue.pending is None


def test_a_prompt_queue_queues_too():
    queue = dialogs.PromptQueue()
    queue.ask(dialogs.Prompt(title="A", label="name"))
    queue.ask(dialogs.Prompt(title="B", label="name"))
    assert queue.waiting == 1
    queue.dismiss()
    assert queue.pending is not None and queue.pending.title == "B"


def test_answering_makes_room_for_a_question_the_answer_raises():
    """The quit chain's shape: ``on_confirm`` asks the next question. The head
    has to be gone before the callback runs, or the new question lands behind a
    corpse and is drawn a frame late."""
    queue = dialogs.ConfirmQueue()
    queue.ask(
        dialogs.Confirm(
            title="A",
            message="first",
            on_confirm=lambda: queue.ask(dialogs.Confirm(title="B", message="second")),
        )
    )
    head = queue.pending
    queue.dismiss()
    head.on_confirm()
    assert queue.pending is not None and queue.pending.title == "B"


def test_pending_is_read_only():
    """Assigning ``None`` to it used to be how a caller cancelled a question,
    and on a queue that silently discards everything behind it as well."""
    queue = dialogs.ConfirmQueue()
    with pytest.raises(AttributeError):
        queue.pending = None


def test_request_quit_no_longer_nests_its_guards_by_hand():
    """The nesting existed because the queue dropped questions. The chain
    stays -- cancelling the first must not leave two more to dismiss -- but it
    is a list walked by index rather than three lambdas."""
    source = inspect.getsource(main.App._request_quit)
    body = source.split('"""', 2)[2]
    assert "guards" in body
    assert body.count("lambda") == 1


def test_the_quit_chain_stops_at_the_first_cancel():
    from warlock.studio import clay_mode, inker_mode, packwright_mode, plotter_mode
    from warlock.studio.panes import pose_panel

    quit_calls: list[str] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(inker=None, clay=None, plotter=None, packwright=None),
        confirms=dialogs.ConfirmQueue(),
        viewer=None,
    )
    app = SimpleNamespace(
        app_ctx=ctx, _quit=lambda: quit_calls.append("quit")
    )
    # Nothing dirty anywhere: every guard proceeds, so the chain runs to the
    # end without a single question.
    main.App._request_quit(app)
    assert quit_calls == ["quit"]
    assert ctx.confirms.pending is None
    # And the guards are the ones this file names, in this order: painted
    # pixels, built geometry, a map, an atlas, then a pose.
    source = inspect.getsource(main.App._request_quit)
    order = [
        "inker_mode.guard",
        "clay_mode.guard",
        "plotter_mode.guard",
        "packwright_mode.guard",
        "pose_panel.guard",
    ]
    positions = [source.index(name) for name in order]
    assert positions == sorted(positions)
    assert all(
        (inker_mode.guard, clay_mode.guard, plotter_mode.guard, packwright_mode.guard,
         pose_panel.guard)
    )


# --- I77: the keyboard reaches the modal, and only the modal -----------------


def test_a_modal_takes_the_keyboard_away_from_the_shortcuts():
    """Esc cancels the dialog; letting the same press through would also leave
    the mode behind it, and Enter would submit the form the dialog is a
    question about."""
    source = inspect.getsource(main.App._events)
    assert "_modal_open()" in source
    guard = source.split("_modal_open()", 1)[0]
    assert "KEYUP" in guard  # releases still pass: space-to-pan is a hold


def test_modal_open_sees_both_queues():
    ctx = SimpleNamespace(
        confirms=dialogs.ConfirmQueue(), prompts=dialogs.PromptQueue()
    )
    app = SimpleNamespace(app_ctx=ctx)
    assert main.App._modal_open(app) is False
    ctx.prompts.ask(dialogs.Prompt(title="A", label="name"))
    assert main.App._modal_open(app) is True


def test_the_confirm_modal_binds_enter_and_escape_and_focuses_confirm():
    source = inspect.getsource(dialogs.ConfirmQueue.draw)
    assert "_escape_pressed()" in source
    assert "_enter_pressed()" in source
    assert "set_item_default_focus" in source
    # Focus is claimed once, not every frame, or Tab can never move off it.
    assert "_focused" in source
