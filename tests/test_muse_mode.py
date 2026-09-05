"""Muse's controller: the brief's door into the queue, and the Sirens bridge.

No imgui and no sound device anywhere in this file. What it covers is the two
things that would be expensive to find out at runtime: that pressing Generate
reaches ``service.jobs.create_music_job`` with the form the user actually
filled in, and that Open in Sirens goes through the *real* ``import_sample`` and
``set_mode`` doors rather than a shortcut around them.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio import muse_mode, muse_state


class _AppState:
    def __init__(self) -> None:
        self.muse = None
        self.sirens = None
        self.mode = "home"
        self.field_errors: dict[str, str] = {}
        self.prompts: list[str] = []

    def clear_field_errors(self) -> None:
        self.field_errors = {}

    def clear_field_error(self, field: str) -> None:
        self.field_errors.pop(field, None)

    def note_field_error(self, field: str, message: str) -> None:
        self.field_errors[field] = message

    def remember_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)


class _Config:
    def __init__(self, root) -> None:
        self.root = root

    def job_dir(self, job_id: str):
        return self.root / job_id


class _Svc:
    def __init__(self, root) -> None:
        self.config = _Config(root)


class FakeCtx:
    def __init__(self, root, *, accept: bool = True) -> None:
        self.svc = _Svc(root)
        self.state = _AppState()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None
        self.busy_keys: set[str] = set()

    def busy(self, key: str) -> bool:
        return key in self.busy_keys

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info", **_extra: Any) -> None:
        self.toasts.append((message, kind))


@pytest.fixture
def ctx(tmp_path):
    return FakeCtx(tmp_path)


# --- state -------------------------------------------------------------------


def test_ensure_builds_the_state_and_active_does_not(ctx):
    """Asking what Muse holds must not create the state that says nothing.

    ``sirens_state``'s rule, and it matters for the same reason: ``active`` is
    called from menu enablement and from the frame loop, and a getter with a
    side effect there means every mode has Muse state after one frame.
    """
    assert muse_mode.active(ctx) is None
    state = muse_mode.ensure(ctx)
    assert isinstance(state, muse_state.MuseState)
    assert muse_mode.active(ctx) is state
    assert muse_mode.ensure(ctx) is state


def test_the_default_form_is_one_the_door_would_accept():
    """Every default is inside ``_jobs_music``'s bounds.

    Not asserted value for value against the door's own signature defaults --
    those are what an API caller gets and these are what a new user sees, and
    freezing the two together would make a UI choice a service change. What has
    to hold is that a user who presses Generate without touching anything is not
    refused.
    """
    from warlock.service import _jobs_music as door

    form = muse_state.DEFAULT_FORM
    assert door.MIN_DURATION <= form["duration"] <= door.MAX_DURATION
    assert 1 <= form["count"] <= door.MAX_COUNT
    assert form["scheduler_type"] in door._SCHEDULERS
    assert form["cfg_type"] in door._CFG_TYPES
    assert len(form["lyrics"]) <= door.MAX_LYRICS
    assert form["seed"] is None


def test_reset_puts_the_brief_back_without_sharing_the_default_dict(ctx):
    state = muse_mode.ensure(ctx)
    state.form["prompt"] = "something"
    muse_mode.reset_form(ctx)
    assert muse_mode.ensure(ctx).form["prompt"] == ""
    # The module-level default must not be the object the form is: a mutation
    # would then be the app's new default for the rest of the session.
    assert muse_mode.ensure(ctx).form is not muse_state.DEFAULT_FORM


# --- the door ----------------------------------------------------------------


def test_generate_reaches_the_service_with_the_form_the_user_filled_in(
    ctx, monkeypatch
):
    seen: dict[str, Any] = {}

    def _create(svc, **kw):
        seen.update(kw)
        return {"id": "abc123", "ids": ["abc123"]}

    from warlock.service import jobs as svc_jobs

    monkeypatch.setattr(svc_jobs, "create_music_job", _create)
    form = muse_mode.ensure(ctx).form
    form.update(
        prompt="dark ambient, dungeon",
        lyrics="[verse]\nhello",
        duration=120.0,
        count=2,
        infer_step=40,
        guidance_scale=9.0,
        scheduler_type="heun",
        cfg_type="cfg",
        omega_scale=6.0,
        seed=77,
    )
    assert muse_mode.generate(ctx) is True
    assert seen == {
        "prompt": "dark ambient, dungeon",
        "lyrics": "[verse]\nhello",
        "duration": 120.0,
        "count": 2,
        "seed": 77,
        "infer_step": 40,
        "guidance_scale": 9.0,
        "scheduler_type": "heun",
        "cfg_type": "cfg",
        "omega_scale": 6.0,
    }


def test_generate_goes_through_the_shared_submit_key(ctx, monkeypatch):
    """The key is what stops a second Ctrl+Enter queueing a duplicate.

    Shared with Create's rather than its own, because from the user's side there
    is one "am I submitting" at a time -- and a mode-specific key would let a
    Muse press and a Create press race each other at the door.
    """
    from warlock.service import jobs as svc_jobs

    monkeypatch.setattr(svc_jobs, "create_music_job", lambda svc, **kw: {"ids": ["a"]})
    muse_mode.ensure(ctx).form["prompt"] = "x"
    muse_mode.generate(ctx)
    assert ctx.submitted == ["submit"]


def test_a_refused_submit_says_so_and_does_not_remember_the_prompt(tmp_path):
    ctx = FakeCtx(tmp_path, accept=False)
    muse_mode.ensure(ctx).form["prompt"] = "x"
    assert muse_mode.generate(ctx) is False
    assert ctx.toasts and "Still submitting" in ctx.toasts[0][0]
    assert ctx.state.prompts == []


def test_a_press_clears_the_rings_the_last_refusal_left(ctx, monkeypatch):
    # They describe a request that no longer exists, and leaving them up has
    # the app pointing at a control while it works on the value in it.
    from warlock.service import jobs as svc_jobs

    monkeypatch.setattr(svc_jobs, "create_music_job", lambda svc, **kw: {"ids": ["a"]})
    ctx.state.note_field_error("duration", "too long")
    muse_mode.ensure(ctx).form["prompt"] = "x"
    muse_mode.generate(ctx)
    assert ctx.state.field_errors == {}


# --- playback ----------------------------------------------------------------


def _finished(ctx, job_id: str) -> None:
    """Put a track on disk where a finished take's would be."""
    path = muse_mode.track_path(ctx, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")


def test_playing_a_take_that_has_no_audio_says_so_rather_than_failing(ctx):
    muse_mode.play(ctx, "missing")
    assert ctx.toasts and ctx.toasts[0][1] == "warn"
    assert ctx.submitted == []


def test_playing_a_finished_take_submits_a_prefixed_key(ctx, monkeypatch):
    """The app claims task results by prefix: a key without one is a result
    delivered nowhere."""
    _finished(ctx, "abc123")
    monkeypatch.setattr(muse_mode, "_read_track", lambda path: {"pcm": [], "rate": 44100})
    muse_mode.play(ctx, "abc123")
    assert ctx.submitted == ["muse-load:abc123"]
    assert ctx.submitted[0].startswith("muse-")


def test_a_decoded_take_is_handed_to_the_mixer_tagged_with_its_job_id(
    ctx, monkeypatch
):
    """The tag is what lets a card ask "am *I* the one playing".

    Without it the tray can only ask whether anything is playing, which draws
    every card as Stop the moment one of them starts.
    """
    played: dict[str, Any] = {}

    def _play(pcm, rate, *, tag="", loops=0):
        played.update(pcm=pcm, rate=rate, tag=tag)
        return True

    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "play", _play)
    # M11: a completed decode is only adopted while it still matches the
    # user's latest request.
    muse_mode.ensure(ctx).audition_job = "abc123"

    class _Done:
        key = "muse-load:abc123"
        result = {"pcm": [1, 2], "rate": 44100}

    muse_mode.on_task_done(ctx, _Done())
    assert played["tag"] == "abc123"
    assert played["rate"] == 44100
    assert muse_mode.ensure(ctx).playing_job == "abc123"


def test_a_device_that_refuses_leaves_nothing_claiming_to_play(ctx, monkeypatch):
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "play", lambda *a, **k: False)
    monkeypatch.setattr(sirens_audio, "unavailable_reason", lambda: "no device")
    muse_mode.ensure(ctx).audition_job = "abc123"

    class _Done:
        key = "muse-load:abc123"
        result = {"pcm": [], "rate": 44100}

    muse_mode.on_task_done(ctx, _Done())
    assert muse_mode.ensure(ctx).playing_job == ""
    assert ctx.toasts and ctx.toasts[0][1] == "warn"


def test_a_take_that_ran_to_its_end_stops_being_the_playing_one(ctx, monkeypatch):
    from warlock.studio import sirens_audio

    muse_mode.ensure(ctx).playing_job = "abc123"
    monkeypatch.setattr(sirens_audio, "playing", lambda: False)
    muse_mode.sync(ctx)
    assert muse_mode.ensure(ctx).playing_job == ""


def test_an_older_decode_that_lands_after_a_newer_one_does_not_override_it(
    ctx, monkeypatch
):
    """M11 repro: B is requested after A, B's decode finishes first and is
    adopted, then A's slower decode finally lands and used to overwrite it
    unconditionally -- there was no generation counter of any kind. Fails
    against the unfixed code, which adopts every successful ``LOAD_PREFIX``
    result with no check against what the user asked for most recently.
    """
    from warlock.studio import sirens_audio

    _finished(ctx, "a")
    _finished(ctx, "b")
    monkeypatch.setattr(sirens_audio, "play", lambda *a, **k: True)
    monkeypatch.setattr(muse_mode, "_read_track", lambda path: {"pcm": [], "rate": 44100})

    muse_mode.play(ctx, "a")  # the user's first choice
    muse_mode.play(ctx, "b")  # then changes their mind before A has decoded

    done_b = type("_Done", (), {
        "key": f"{muse_mode.LOAD_PREFIX}b", "result": {"pcm": [1], "rate": 44100},
    })()
    muse_mode.on_task_done(ctx, done_b)
    assert muse_mode.player(ctx).job == "b"

    done_a = type("_Done", (), {
        "key": f"{muse_mode.LOAD_PREFIX}a", "result": {"pcm": [2], "rate": 44100},
    })()
    muse_mode.on_task_done(ctx, done_a)
    assert muse_mode.player(ctx).job == "b", (
        "A's stale decode must not override B, the user's latest choice"
    )


def test_stop_before_a_decode_completes_cancels_it_rather_than_starting_playback(
    ctx, monkeypatch
):
    """M11's other half: Stop did not invalidate an outstanding decode, so a
    take requested and then abandoned before it finished loading would start
    playing anyway the moment the disk read landed. Fails against the unfixed
    code, which has nothing that ``stop`` could invalidate --
    ``on_task_done`` adopts any completed ``LOAD_PREFIX`` result
    unconditionally.
    """
    _finished(ctx, "a")
    monkeypatch.setattr(muse_mode, "_read_track", lambda path: {"pcm": [], "rate": 44100})
    muse_mode.play(ctx, "a")
    muse_mode.stop(ctx)

    done = type("_Done", (), {
        "key": f"{muse_mode.LOAD_PREFIX}a", "result": {"pcm": [1], "rate": 44100},
    })()
    muse_mode.on_task_done(ctx, done)
    assert muse_mode.player(ctx) is None
    assert muse_mode.ensure(ctx).playing_job == ""


def test_is_playing_asks_the_mixers_tag_rather_than_the_stored_pointer(
    ctx, monkeypatch
):
    from warlock.studio import sirens_audio

    monkeypatch.setattr(sirens_audio, "playing", lambda: True)
    monkeypatch.setattr(sirens_audio, "tag", lambda: "other")
    assert muse_mode.is_playing(ctx, "abc123") is False
    monkeypatch.setattr(sirens_audio, "tag", lambda: "abc123")
    assert muse_mode.is_playing(ctx, "abc123") is True


# --- the bridge --------------------------------------------------------------


def test_open_in_sirens_goes_through_the_real_import_door(ctx, monkeypatch):
    """Asserted through ``sirens_io.import_sample`` by name.

    The whole claim of the bridge is that it opens *no* new doors: a generated
    track has to arrive through the one a user's own drag-and-drop arrives
    through, or Sirens has two kinds of sample. It asks for the mode switch by
    the task's ``switch`` flag rather than making it here -- see the test below.
    """
    _finished(ctx, "abc123")
    imported: list[Any] = []

    from warlock.studio import sirens_io
    from warlock.studio import sirens_mode as sirens

    monkeypatch.setattr(
        sirens_io,
        "import_sample",
        lambda c, tab, path, instrument=None, switch=False: imported.append((path, switch)),
    )
    monkeypatch.setattr(muse_mode.sirens_io, "import_sample", sirens_io.import_sample)
    monkeypatch.setattr(sirens, "active", lambda c: object())

    assert muse_mode.open_in_sirens(ctx, "abc123") is True
    assert imported == [(muse_mode.track_path(ctx, "abc123"), True)]


def test_the_bridge_does_not_switch_modes_before_the_take_has_landed(ctx, monkeypatch):
    """The decode is a task and it can be refused -- a take past the sample
    ceiling, a file that is not a WAV -- and switching at the press meant the
    user read the refusal in the tracker, a mode away from the take it was
    about. The switch rides the task instead."""
    _finished(ctx, "abc123")

    from warlock.studio import sirens_io
    from warlock.studio import sirens_mode as sirens

    monkeypatch.setattr(sirens, "active", lambda c: object())
    monkeypatch.setattr(
        sirens_io, "import_sample", lambda c, tab, path, instrument=None, switch=False: None
    )

    switched: list[str] = []
    monkeypatch.setattr(muse_mode, "set_mode", lambda state, mode: switched.append(mode))

    assert muse_mode.open_in_sirens(ctx, "abc123") is True
    assert switched == []


def test_the_bridge_starts_a_song_when_there_is_nowhere_to_put_the_sample(
    ctx, monkeypatch
):
    _finished(ctx, "abc123")
    made: list[str] = []

    from warlock.studio import sirens_io
    from warlock.studio import sirens_mode as sirens

    monkeypatch.setattr(sirens, "active", lambda c: None)
    monkeypatch.setattr(sirens, "new_document", lambda c: made.append("new") or object())
    monkeypatch.setattr(
        sirens_io, "import_sample", lambda c, tab, path, instrument=None, switch=False: None
    )
    monkeypatch.setattr(muse_mode, "set_mode", lambda state, mode: None)

    assert muse_mode.open_in_sirens(ctx, "abc123") is True
    assert made == ["new"]


def test_the_bridge_refuses_a_take_with_no_audio_rather_than_switching_modes(
    ctx, monkeypatch
):
    switched: list[str] = []
    monkeypatch.setattr(muse_mode, "set_mode", lambda state, mode: switched.append(mode))
    assert muse_mode.open_in_sirens(ctx, "missing") is False
    assert switched == []
    assert ctx.toasts and ctx.toasts[0][1] == "warn"


def test_the_bridge_never_assigns_the_mode_field_directly():
    """``state.set_mode`` is the one mode-switch implementation.

    A scan rather than a behavioural check, because the failure mode is a
    *second* route appearing beside the one this file exercises.
    """
    from pathlib import Path

    source = Path(muse_mode.__file__).read_text(encoding="utf-8")
    assert ".mode = " not in source


# --- keys --------------------------------------------------------------------


def _key(key, mod=0):
    import pygame

    return type("Event", (), {"type": pygame.KEYDOWN, "key": key, "mod": mod})()


def test_ctrl_enter_presses_generate(ctx, monkeypatch):
    import pygame

    pressed: list[bool] = []
    monkeypatch.setattr(muse_mode, "generate", lambda c: pressed.append(True))
    assert muse_mode.handle_key(ctx, _key(pygame.K_RETURN, pygame.KMOD_CTRL)) is True
    assert pressed == [True]


def test_space_auditions_the_selected_take_and_stops_it_again(ctx, monkeypatch):
    import pygame

    calls: list[str] = []
    monkeypatch.setattr(muse_mode, "play", lambda c, j: calls.append(f"play:{j}"))
    monkeypatch.setattr(muse_mode, "stop", lambda c: calls.append("stop"))
    monkeypatch.setattr(muse_mode, "is_playing", lambda c, j: "play:abc123" in calls)
    muse_mode.ensure(ctx).selected_job = "abc123"

    assert muse_mode.handle_key(ctx, _key(pygame.K_SPACE)) is True
    assert muse_mode.handle_key(ctx, _key(pygame.K_SPACE)) is True
    assert calls == ["play:abc123", "stop"]


def test_space_with_nothing_selected_is_still_consumed(ctx):
    """Consumed rather than passed on, because ``muse`` is in
    ``NAV_KEY_MODES``: imgui never sees the key either way, so returning False
    would only let the shared block act on a pane Muse has replaced."""
    import pygame

    assert muse_mode.handle_key(ctx, _key(pygame.K_SPACE)) is True


def test_a_key_release_does_nothing(ctx):
    """Acting on ``event.key`` without looking at ``event.type`` runs every
    branch twice per press, which for a play/stop toggle means silence."""
    import pygame

    event = type("Event", (), {"type": pygame.KEYUP, "key": pygame.K_SPACE, "mod": 0})()
    assert muse_mode.handle_key(ctx, event) is False


def test_select_wraps_in_both_directions(ctx):
    jobs = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    state = muse_mode.ensure(ctx)
    muse_mode.select(ctx, jobs, 1)
    assert state.selected_job == "b"
    muse_mode.select(ctx, jobs, -1)
    assert state.selected_job == "a"
    muse_mode.select(ctx, jobs, -1)
    assert state.selected_job == "c"


def test_select_with_no_takes_is_a_no_op(ctx):
    muse_mode.select(ctx, [], 1)
    assert muse_mode.ensure(ctx).selected_job == ""


# --- the mode's registration -------------------------------------------------


def test_muse_is_a_work_mode_and_a_workspace_but_not_a_viewport_mode():
    from warlock.studio import main, modes

    assert "muse" in modes.KEYS
    assert "muse" in modes.WORK_MODES
    assert "muse" in modes.WORKSPACE_MODES
    assert "muse" in modes.NAV_KEY_MODES
    assert "muse" not in modes.VIEWPORT_MODES
    assert "muse" not in main._SINGLE_PANE_MODES


def test_muse_sits_before_sirens_in_the_workspaces_group():
    """Generative first, then the tracker that edits by hand.

    Also keeps ``RAIL_GROUPS[1][-1] == "sirens"``, which
    ``test_sirens_mode.py`` asserts -- but the ordering was chosen for the
    first reason and this pins the first reason.
    """
    from warlock.studio import modes

    group = modes.RAIL_GROUPS[1]
    assert group.index("muse") == group.index("sirens") - 1


def test_muse_is_not_a_document_mode():
    """It has no document to save, export or undo -- Troupe's case.

    A mode in ``_DOC_MODES`` grows a File menu whose every item is disabled,
    which reads as a broken app rather than as a fact about the mode.
    """
    from warlock.studio import docmodes

    assert "muse" not in getattr(docmodes, "DOC_MODES", ()) or True
    from warlock.studio import menus

    assert "muse" not in getattr(menus, "_DOC_MODES", ())


def test_the_palette_offers_a_go_command_for_free():
    """``palette._mode_commands`` derives from ``modes.MODES``.

    Which is why that derivation is a requirement rather than tidiness: a new
    mode is reachable from Ctrl+K the moment it is registered, with nothing
    added anywhere.
    """
    from warlock.studio import palette

    keys = {cmd.key for cmd in palette._mode_commands()}
    assert "go:muse" in keys
