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


def _app(mode: str = "create") -> SimpleNamespace:
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


def _press(app: SimpleNamespace, key: int, mod: int = 0) -> None:
    # ``mod`` is not optional on a real event: every pygame KEYDOWN carries the
    # modifier state at the moment of the press, and the shortcut handler reads
    # it from there rather than from ``pygame.key.get_mods()`` -- ``mod`` is the
    # state *then*, ``get_mods()`` is the state now, and events drain in a batch
    # after the frame (UX-12). Synthesising an event without it modelled a
    # pygame that does not exist.
    main.App._shortcut(app, pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod))


@pytest.fixture
def alt(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_ALT)


@pytest.fixture
def no_mods(monkeypatch):
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)


def test_no_digit_is_a_mode_switch(alt, no_mods):
    """The positional Alt+digit bindings are gone, and the negative is the whole
    point of the test: twelve modes against ten digits meant either two modes
    with no key or a second table saying which two -- and that table is exactly
    what the positional scheme existed to avoid. Clay binds 1-4 to the element
    modes and Review binds 1-5 to reject reasons; those keys are theirs again,
    with or without Alt."""
    for key in (pygame.K_1, pygame.K_4, pygame.K_9, pygame.K_0, pygame.K_KP3, pygame.K_KP0):
        app = _app("create")
        _press(app, key)
        assert app.app_ctx.state.mode == "create"


def test_the_digit_helpers_are_gone_rather_than_left_unused():
    """A helper with no caller is not free (the ``setup()`` lesson): it reads as
    a supported entry point, and the next thing to want a digit map would find
    one that nothing keeps honest."""
    for name in ("mode_for_digit", "digit_key_label", "digit_for_mode"):
        assert not hasattr(modes, name)
    assert "_digit_keys" not in inspect.getsource(main)


def test_the_palette_is_checked_before_the_workspace_handlers():
    """Ctrl+K is now the *only* keyboard route to a mode, so it has to be
    checked above Inker/Clay/Review, which consume every key they are handed."""
    source = inspect.getsource(main.App._shortcut)
    assert source.index("K_k") < source.index("WORK_MODES")


# --- I76: Esc out of the pass-through modes ----------------------------------


def test_esc_returns_to_the_mode_you_came_from(no_mods):
    app = _app("create")
    state = app.app_ctx.state
    app._set_mode("settings")
    assert state.mode == "settings"
    _press(app, pygame.K_ESCAPE)
    assert state.mode == "create"


# --- the UI redesign, wave 3: the Manual is an overlay, not a mode -------------------


def test_f1_raises_the_manual_over_the_mode_you_are_in(no_mods):
    """It used to *replace* it, which is the defect: the (?) beside a control
    answered the question by taking the control away."""
    app = _app("create")
    state = app.app_ctx.state
    _press(app, pygame.K_F1)
    assert state.manual.open
    assert state.mode == "create"


def test_f1_puts_it_away_again(no_mods):
    """The key that raises a reference is the key that closes it."""
    app = _app("create")
    _press(app, pygame.K_F1)
    _press(app, pygame.K_F1)
    assert not app.app_ctx.state.manual.open


def test_esc_closes_the_manual_before_the_mode_sees_it(no_mods):
    """Ordering, not tidiness: the workspace modes consume every key they are
    handed, so an Esc dispatched to Inker with the overlay up would drop a
    floating selection and leave the reference open on top of it."""
    app = _app("inker")
    state = app.app_ctx.state
    _press(app, pygame.K_F1)
    _press(app, pygame.K_ESCAPE)
    assert not state.manual.open
    assert state.mode == "inker"


def test_esc_still_leaves_a_mode_once_the_manual_is_closed(no_mods):
    app = _app("settings")
    state = app.app_ctx.state
    state.previous_mode = "create"
    _press(app, pygame.K_F1)
    _press(app, pygame.K_ESCAPE)
    assert state.mode == "settings"
    _press(app, pygame.K_ESCAPE)
    assert state.mode == "create"


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
    app.app_ctx.state.previous_mode = "create"
    _press(app, pygame.K_ESCAPE)
    assert app.app_ctx.state.mode == "home"


def test_previous_mode_is_sampled_per_key_event_not_per_frame(no_mods):
    """F1 changes the mode from inside ``_shortcut``, so a sample taken once at
    the top of the frame would still hold the mode from before it and Esc would
    go two steps back."""
    app = _app("create")
    state = app.app_ctx.state
    state.mode = "library"  # a Home row clicked since the last keypress
    _press(app, pygame.K_F1)
    _press(app, pygame.K_ESCAPE)
    assert state.mode == "library"


def test_esc_in_a_work_mode_is_still_the_pane_s(no_mods):
    """Esc means "drop what I am doing" in a mode that has something to drop --
    a comparison, a pose edit. It must not leave the mode as well."""
    app = _app("create")
    app.app_ctx.state.comparing = "job-1"
    _press(app, pygame.K_ESCAPE)
    assert app.app_ctx.state.mode == "create"
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
    from warlock.studio import (
        clay_mode,
        inker_mode,
        packwright_mode,
        plotter_mode,
        poser_mode,
    )
    from warlock.studio.panes import pose_panel

    quit_calls: list[str] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            inker=None, clay=None, plotter=None, packwright=None, sirens=None
        ),
        confirms=dialogs.ConfirmQueue(),
        viewer=None,
        poser_viewer=None,
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
    # pixels, built geometry, a map, an atlas, then the two pose editors --
    # the inspector's and the Poser's, which read different viewers and so can
    # never double-ask about one edit.
    source = inspect.getsource(main.App._request_quit)
    order = [
        "inker_mode.guard",
        "clay_mode.guard",
        "plotter_mode.guard",
        "packwright_mode.guard",
        "pose_panel.guard",
        "poser_mode.guard",
    ]
    positions = [source.index(name) for name in order]
    assert positions == sorted(positions)
    assert all(
        (inker_mode.guard, clay_mode.guard, plotter_mode.guard, packwright_mode.guard,
         pose_panel.guard, poser_mode.guard)
    )


# --- the library's arrows and Enter ------------------------------------------
#
# Home's Resume list set the precedent (M107): a list the user is looking at
# takes Up/Down and Enter, or the keyboard cannot reach what the mouse can.
# The library is a mode now, so its cards get the same three keys, routed to
# the same helpers the 2D/3D fall-through already uses -- one selection-move,
# not a second spelling of it.


def _library_app(jobs: list[dict], selected: str | None = None) -> SimpleNamespace:
    state = AppState()
    state.mode = state.mode_observed = state.previous_mode = "library"
    state.selected = selected
    by_id = {job["id"]: job for job in jobs}
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(
            state=state,
            cache=SimpleNamespace(
                visible=lambda _filters: jobs,
                get=lambda job_id: by_id.get(job_id),
            ),
        ),
    )
    for name in ("_note_mode", "_set_mode", "_escape_mode"):
        setattr(app, name, MethodType(getattr(main.App, name), app))
    return app


def test_the_library_arrows_move_the_selection(no_mods):
    jobs = [
        {"id": "aaa", "stage": "model", "status": "done"},
        {"id": "bbb", "stage": "model", "status": "done"},
    ]
    app = _library_app(jobs, selected="aaa")
    _press(app, pygame.K_DOWN)
    assert app.app_ctx.state.selected == "bbb"
    _press(app, pygame.K_UP)
    assert app.app_ctx.state.selected == "aaa"
    assert app.app_ctx.state.mode == "library", "the arrows never change mode"


def test_enter_opens_the_selected_asset_in_the_mode_that_shows_it(no_mods):
    """The same routing Home's Resume list applies to an asset row: a
    reference or a tile opens at the Reference stage, everything else at
    Mesh."""
    jobs = [
        {"id": "aaa", "stage": "reference", "status": "done"},
        {"id": "bbb", "stage": "model", "status": "done"},
    ]
    app = _library_app(jobs, selected="bbb")
    _press(app, pygame.K_RETURN)
    assert app.app_ctx.state.mode == "create"
    assert app.app_ctx.state.create_stage == "mesh"
    # Through set_mode, so Esc still knows it came from the library.
    assert app.app_ctx.state.previous_mode == "library"

    app = _library_app(jobs, selected="aaa")
    _press(app, pygame.K_RETURN)
    assert app.app_ctx.state.mode == "create"
    assert app.app_ctx.state.create_stage == "reference"


def test_enter_with_no_selection_stays_in_the_library(no_mods):
    """Enter with no cursor has nothing it could mean, and bouncing to an
    empty Create pane would read as the library losing the user's place."""
    app = _library_app([], selected=None)
    _press(app, pygame.K_RETURN)
    assert app.app_ctx.state.mode == "library"


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

    # And the fifth: the Send to Troupe question. It is a real modal with its
    # own Send and Cancel, and a shortcut leaking through it is UX-08 -- Esc
    # would cancel the dialog *and* leave the mode behind it. The partial ctx
    # is the point of ``troupe_send.is_open``'s ``getattr``: this caller has
    # never built a state object.
    from types import SimpleNamespace as _NS

    from warlock.studio.panes import troupe_send

    ctx.prompts.dismiss()
    assert main.App._modal_open(app) is False
    ctx.state = _NS(troupe_send=troupe_send.TroupeSend(job_id="abc"))
    assert main.App._modal_open(app) is True


def test_the_confirm_modal_binds_enter_and_escape_and_focuses_confirm():
    source = inspect.getsource(dialogs.ConfirmQueue.draw)
    assert "_escape_pressed()" in source
    assert "_enter_pressed()" in source
    assert "set_item_default_focus" in source
    # Focus is claimed once, not every frame, or Tab can never move off it.
    assert "_focused" in source


# --- C1: a workspace mode's arm must consume the key -------------------------
#
# Every workspace mode replaces the asset viewport and the form column with its
# own, so the shared 2D/3D block at the foot of ``_shortcut`` -- Delete, the
# arrows, Ctrl+Enter, F/W/S -- is addressed at panes that mode is not showing.
# Each arm therefore returns unconditionally, *whether or not* its handle_key
# consumed the key, and each says so in a comment.
#
# Packwright's return was lost when Troupe's branch was spliced in ahead of it
# (the giveaway was the unreachable second ``return`` left behind after
# Troupe's own), and pressing Delete in the atlas packer sent the selected
# *library* asset to the trash -- confirm-free, because the library binding it
# fell through to is deliberately confirm-free. Nothing caught it: the per-mode
# key tests call ``<mode>_mode.handle_key`` directly and never route through
# this dispatcher, so the fall-through is invisible from there. This pins it
# from the dispatcher's side, which is the only side that can see it.

_WORKSPACE_ARMS = {
    "clay": "clay_mode",
    "poser": "poser_mode",
    "review": "review_mode",
    "inker": "inker_mode",
    "plotter": "plotter_mode",
    "packwright": "packwright_mode",
    "troupe": "troupe_mode",
    "muse": "muse_mode",
    "sirens": "sirens_mode",
}

# The keys the shared block binds, each with the modifier that arms it. A key
# added there without being added here is a hole this test cannot see, which is
# why the two are checked against the source below.
_SHARED_KEYS = (
    (pygame.K_DELETE, 0),
    (pygame.K_RETURN, pygame.KMOD_CTRL),
    (pygame.K_UP, 0),
    (pygame.K_DOWN, 0),
    (pygame.K_f, 0),
    (pygame.K_w, 0),
    (pygame.K_s, 0),
)


def test_every_work_mode_is_either_create_or_named_here() -> None:
    """The list above is exhaustive, and stays that way.

    A tenth mode that reaches ``_shortcut`` without an arm of its own falls
    through by construction, so the drift this catches is a *new* mode rather
    than an edited one. ``create`` is the exception on purpose: the shared
    block is its own keyboard.
    """
    assert set(_WORKSPACE_ARMS) | {"create"} == set(modes.WORK_MODES)


@pytest.mark.parametrize("mode", sorted(_WORKSPACE_ARMS))
def test_a_workspace_mode_never_falls_through_to_the_shared_block(mode, monkeypatch):
    """No key pressed in a workspace mode reaches the 2D/3D bindings.

    ``handle_key`` is stubbed to answer False -- "I did not bind this" -- which
    is the case the arms exist for: a mode that consumed everything would pass
    this test even with its return missing.
    """
    fired: list[str] = []
    monkeypatch.setattr(
        f"warlock.studio.{_WORKSPACE_ARMS[mode]}.handle_key",
        lambda ctx, event: False,
    )
    for target, name in (
        ("warlock.studio.panes.library.delete_asset", "delete_asset"),
        ("warlock.studio.panes.library.select_relative", "select_relative"),
        ("warlock.studio.panes.settings_2d.generate", "generate"),
        ("warlock.studio.panes.settings_3d.promote", "promote"),
    ):
        monkeypatch.setattr(target, (lambda n: lambda *a, **k: fired.append(n))(name))

    app = _app(mode)
    app.app_ctx.state.selected = "job-1"
    # Clay's arm frames its own selection on F; it is part of the arm, not of
    # the block below it, so it is stubbed rather than counted.
    app._frame_clay_selection = lambda: None
    app.viewer.frame = lambda: fired.append("frame")
    app.viewer.set_wireframe = lambda v: fired.append("wireframe")
    app.viewer.set_turntable = lambda v: fired.append("turntable")

    for key, mod in _SHARED_KEYS:
        _press(app, key, mod)

    assert fired == [], f"{mode} fell through to the shared 2D/3D block"


def test_the_shared_block_binds_nothing_this_test_does_not_press() -> None:
    """``_SHARED_KEYS`` is the whole of the fall-through, not a sample.

    Read off the source rather than maintained by hand: the block is a chain of
    ``event.key ==``/``in`` tests, and a binding added to it without a line
    here would leave a key no arm is proved to consume.
    """
    source = inspect.getsource(main.App._shortcut)
    tail = source[source.index("mods = event.mod") :]
    pressed = {key for key, _mod in _SHARED_KEYS}
    for name in ("K_DELETE", "K_RETURN", "K_UP", "K_DOWN", "K_f", "K_w", "K_s"):
        assert f"pygame.{name}" in tail, f"{name} left the shared block"
        assert getattr(pygame, name) in pressed
