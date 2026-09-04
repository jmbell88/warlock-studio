"""Muse's controller: the brief, the takes, playback and the bridge to Sirens.

An ordinary ``studio/`` module, and that is worth saying because the *other*
audio mode is not: ``studio/sirens/`` is a headless engine forbidden to import
imgui, moderngl, pygame, scipy or ``service``, and ``sirens_mode`` is the thin
layer that reaches those on its behalf. Muse has no engine to keep pure -- the
model lives in a subprocess two layers down -- so this module imports
``service``, ``sirens_audio`` and ``sirens_io`` freely.

**Playback is ``sirens_audio``, unchanged.** ``play(pcm, rate, tag=..., loops=)``
is already mode-agnostic and tag-keyed, and its ``RATE``/``SIZE``/``CHANNELS``
are 44100/-16/2 -- which is exactly ACE-Step's output format, so a take is
handed to the mixer with no resample. The module is deliberately *not* renamed
or relocated for having a second caller: its file location is cosmetic and its
``pygame.mixer`` exclusivity is the actual contract, so moving it would churn
every import in Sirens to say something the docstring already says.

**The bridge is one function and it opens no new doors.**
:func:`open_in_sirens` composes three that exist: ``sirens_io.import_sample``
(which submits the same ``_decode_sample`` task a user's drag-and-drop does),
``sirens_mode.new_document``, and ``state.set_mode`` -- which is the one
mode-switch implementation, never an assignment to the state field. Sirens
itself needs no change of any kind for this to work.
"""

from __future__ import annotations

import logging
from typing import Any

from . import sirens_audio, sirens_io, sirens_mode, sirens_state
from .muse_state import DEFAULT_FORM, MuseState, active, ensure  # noqa: F401
from .state import set_mode

log = logging.getLogger(__name__)

#: The task key a take's decode runs under. Prefixed, because the app claims
#: results by prefix and a key without one is a result delivered nowhere.
LOAD_PREFIX = "muse-load:"


def generate(ctx: Any) -> bool:
    """Queue the brief. -> whether the submit was accepted.

    Through ``ctx.submit`` under the shared ``"submit"`` key, exactly as
    Create's Generate is: the key is what stops a second Ctrl+Enter queueing a
    duplicate while the first is still at the door, and it is shared because
    from the user's side there is one "am I submitting" at a time.

    Validation is the service's. There is no ``validate(form)`` sibling here on
    purpose: Create has one because its form has fifteen interacting fields and
    a disabled button needs a reason before anything is pressed, whereas every
    refusal Muse can produce comes from ``create_music_job`` and carries the
    ``field=`` that puts the ring on the right control. A second copy of those
    bounds in the pane is the thing that drifts.
    """
    from ..service import jobs as svc_jobs

    state = ensure(ctx)
    form = state.form

    def run():
        return svc_jobs.create_music_job(
            ctx.svc,
            prompt=str(form["prompt"]),
            lyrics=str(form["lyrics"]),
            duration=float(form["duration"]),
            count=int(form["count"]),
            seed=form["seed"],
            infer_step=int(form["infer_step"]),
            guidance_scale=float(form["guidance_scale"]),
            scheduler_type=str(form["scheduler_type"]),
            cfg_type=str(form["cfg_type"]),
            omega_scale=float(form["omega_scale"]),
        )

    # Cleared on every press: the rings from the last one describe a request
    # that no longer exists.
    ctx.state.clear_field_errors()
    if not ctx.submit("submit", run):
        ctx.toast("Still submitting the last one - try again in a moment.")
        return False
    ctx.state.remember_prompt(str(form["prompt"]))
    return True


# --- auditioning ------------------------------------------------------------


def track_path(ctx: Any, job_id: str):
    """Where a finished take's WAV is. One spelling, three callers."""
    return ctx.svc.config.job_dir(job_id) / "track.wav"


def _read_track(path: Any) -> dict[str, Any]:
    """A take's WAV as ``int16`` frames plus its rate. Blocking; task work.

    ``wave`` and numpy rather than ``sirens/wavout.read_wav``, which looks like
    the obvious reuse and is the wrong function: that one exists to feed a chip
    voice, so it mixes to **mono** and resamples to the *engine's* render rate.
    Both are right for a sample instrument and both are damage here -- this is
    a finished stereo track being played back as itself, at the 44.1 kHz the
    mixer is already open at.

    The tracker's reader is still the one that runs on the Sirens bridge, where
    mono-at-engine-rate is exactly what is wanted. Two readers because there
    are two questions, not because one of them was forgotten.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as fh:
        rate = fh.getframerate()
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        frames = fh.readframes(fh.getnframes())
    if width != 2:
        # ACE-Step writes 16-bit; anything else is a file this app did not make.
        raise ValueError(f"{path.name} is not 16-bit audio")
    pcm = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    return {"pcm": pcm, "rate": rate}


def play(ctx: Any, job_id: str) -> None:
    """Audition one take, replacing whatever was playing.

    The read is on a task rather than the frame thread: four minutes of 44.1 kHz
    stereo is ~40 MB off disk, which is a visible stall in a 60 Hz loop.
    """
    path = track_path(ctx, job_id)
    if not path.exists():
        ctx.toast("that take has no audio on disk", "warn")
        return
    if not ctx.submit(f"{LOAD_PREFIX}{job_id}", _read_track, path):
        ctx.toast("still loading that take")


def on_task_done(ctx: Any, done: Any) -> None:
    """Adopt a decoded take and start it. Routed by the ``muse-`` key prefix."""
    key, result = done.key, done.result
    if not key.startswith(LOAD_PREFIX) or not isinstance(result, dict):
        return
    job_id = key[len(LOAD_PREFIX) :]
    # Tagged with the job id, which is what lets a card ask "am *I* the one
    # playing" rather than only "is anything playing".
    if sirens_audio.play(result["pcm"], result["rate"], tag=job_id):
        ensure(ctx).playing_job = job_id
    else:
        ctx.toast(sirens_audio.unavailable_reason() or "could not play that take",
                  "warn")


def stop(ctx: Any) -> None:
    """Stop whatever is auditioning. Safe when nothing is."""
    sirens_audio.stop()
    state = active(ctx)
    if state is not None:
        state.playing_job = ""


def is_playing(ctx: Any, job_id: str) -> bool:
    """Whether *this* take is the one currently sounding.

    Asked of the mixer's tag rather than of ``playing_job`` alone, so a take
    that finished on its own stops drawing as Stop without anything having to
    notice the end.
    """
    return bool(job_id) and sirens_audio.playing() and sirens_audio.tag() == job_id


def sync(ctx: Any) -> None:
    """Let a finished audition clear itself. Called once per frame by the tray."""
    state = active(ctx)
    if state is not None and state.playing_job and not sirens_audio.playing():
        state.playing_job = ""


# --- keys -------------------------------------------------------------------


def select(ctx: Any, jobs: list[Any], delta: int) -> None:
    """Move the tray's selection by ``delta``, wrapping. ``jobs`` is the tray's
    own list, passed in rather than recomputed: the pane already has it, and a
    second query could disagree about what "newest" means on the frame a take
    lands."""
    if not jobs:
        return
    state = ensure(ctx)
    ids = [str(job["id"]) for job in jobs]
    here = ids.index(state.selected_job) if state.selected_job in ids else 0
    state.selected_job = ids[(here + delta) % len(ids)]


def handle_key(ctx: Any, event: Any) -> bool:
    """Muse's keyboard. -> whether the press was consumed.

    Small, and deliberately so: this mode has two verbs. Space auditions the
    selected take -- which is why ``muse`` is in ``modes.NAV_KEY_MODES``, since
    one press must not also activate whatever button imgui's focus ring is on
    -- and Ctrl+Enter presses Generate from wherever the caret is, which is the
    binding every other form in this app already carries.

    Up/Down move the tray's selection. They are bound rather than left alone
    for the reason ``troupe_mode.handle_key`` states at length: membership of
    ``NAV_KEY_MODES`` withholds those keys from imgui whether or not anything
    binds them, so an unbound one is a key taken from one consumer and given to
    none.

    **Presses only.** Acting on ``event.key`` without looking at ``event.type``
    runs every branch twice per press, which for a play/stop toggle means
    silence.
    """
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    from .panes import muse_results

    ctrl = bool(event.mod & pygame.KMOD_CTRL)
    if ctrl and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
        generate(ctx)
        return True
    state = ensure(ctx)
    if event.key == pygame.K_SPACE:
        job_id = state.selected_job
        if not job_id:
            return True
        if is_playing(ctx, job_id):
            stop(ctx)
        else:
            play(ctx, job_id)
        return True
    if event.key in (pygame.K_UP, pygame.K_DOWN):
        select(ctx, muse_results.plan_for(ctx), -1 if event.key == pygame.K_UP else 1)
        return True
    return False


# --- the bridge -------------------------------------------------------------


def open_in_sirens(ctx: Any, job_id: str) -> bool:
    """Land a take in the tracker as a sample instrument. -> whether it started.

    The one thing that makes the two audio modes a *pair* rather than two
    unrelated features: Muse writes a 44.1 kHz WAV, and Sirens' sample
    instruments read 44.1 kHz WAVs.

    Every door already existed. ``import_sample`` submits the decode and
    ``sirens_mode.adopt_sample`` adopts the result exactly as it does for a
    user's own drag-and-drop, so a generated track is not a special kind of
    sample and nothing in Sirens has to know where it came from.
    """
    path = track_path(ctx, job_id)
    if not path.exists():
        ctx.toast("that take has no audio on disk", "warn")
        return False
    tab = sirens_mode.active(ctx)
    if tab is None:
        tab = sirens_mode.new_document(ctx)
    sirens_io.import_sample(ctx, tab, path)
    # Through the one mode-switch implementation, never a direct assignment:
    # ``set_mode`` is what runs the leave/enter work every mode relies on.
    set_mode(ctx.state, "sirens")
    return True


def reset_form(ctx: Any) -> None:
    """Put the brief back to its defaults. The palette's Reset."""
    ensure(ctx).form = dict(DEFAULT_FORM)


__all__ = [
    "DEFAULT_FORM",
    "LOAD_PREFIX",
    "MuseState",
    "active",
    "ensure",
    "generate",
    "handle_key",
    "is_playing",
    "on_task_done",
    "open_in_sirens",
    "play",
    "reset_form",
    "select",
    "stop",
    "sync",
    "track_path",
]

# Referenced by the tray's card for its "which document would this land in"
# tooltip; imported here so the pane does not need a second Sirens import.
sirens_tab_title = sirens_state.title_for
