"""Muse's controller: the brief, the takes, playback and the bridge to Sirens.

An ordinary ``studio/`` module, and that is worth saying because the *other*
audio mode is not: ``studio/sirens/`` is a headless engine forbidden to import
imgui, moderngl, pygame, scipy or ``service``, and ``sirens_mode`` is the thin
layer that reaches those on its behalf. Muse has no engine to keep pure -- the
model lives in a subprocess two layers down -- so this module imports
``service``, ``sirens_audio`` and ``sirens_io`` freely.

**Playback is ``sirens_audio``, and what it gained is a volume, not an offset.**
``play(pcm, rate, tag=..., loops=)`` is already mode-agnostic and tag-keyed, and
it refuses anything but its ``RATE`` (44100) outright rather than resampling.
ACE-Step's vendored ``latents2audio`` *defaulted* to 48000, which is why
``read_track`` reconciles the two -- but ``WARLOCK 5/5`` now pins the writer to
44 100 Hz 16-bit at the call site, so that path is the fallback for a take made
by an older build rather than the normal one. See
``pipelines/acestep/ATTRIBUTION.md``.

The volume lives there because that module owns the device and its exclusivity
is the contract: one reserved channel means one ``set_volume``, so a per-mode
volume would be a control that disagrees with itself. No offset, for the
mirror-image reason -- it does not own the caller's buffer, and seeking is
slice-and-replay, so the base lives on ``MuseState.player.play_offset``.

The module is deliberately *not* renamed or relocated for having a second
caller: its file location is cosmetic and its ``pygame.mixer`` exclusivity is
the actual contract, so moving it would churn every import in Sirens to say
something the docstring already says.

**The bridge runs both ways, and only one leg opens a door.**
:func:`open_in_sirens` composes three functions that exist:
``sirens_io.import_sample`` (which submits the same ``_decode_sample`` task a
user's drag-and-drop does), ``sirens_mode.new_document``, and ``state.set_mode``
-- the one mode-switch implementation, never an assignment to the state field.
Sirens needs no change of any kind for it.

:func:`compose_from_sirens` is the mirror, and the Sirens half of it opens
nothing either -- ``wsng_bytes``, ``read_wsng``/``synth.render_marked`` and
``wavout.wav_bytes`` are already used in exactly that combination by
``sirens_play.request_render`` and ``sirens_io.export_plan``. What it does need
is one keyword on the *Muse* side, because ``create_music_job`` takes scalars
only: ``reference_wav: bytes``. Named here rather than left implicit, because
"the bridge opens no new doors" was a load-bearing claim and half of it has
stopped being true.
"""

from __future__ import annotations

import logging
from typing import Any

from . import muse_io, sirens_audio, sirens_io, sirens_mode, sirens_state
from .muse_state import (  # noqa: F401
    DEFAULT_DERIVE,
    DEFAULT_FORM,
    MuseState,
    active,
    ensure,
)
from .muse_state import Player as MusePlayer
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


# --- deriving ---------------------------------------------------------------


#: How far Left/Right move the playhead, in seconds, and what Shift multiplies
#: that by. One second is a beat or two at most tempos -- fine enough to place a
#: marker by ear -- and ten seconds is a phrase, which is the other scale a
#: listener works at. Two figures rather than a ladder because there are two
#: questions and no third.
NUDGE_SECONDS = 1.0
NUDGE_MULTIPLIER = 10


#: Which controls the popup draws for each task, and the only place that is
#: written down. Read as data by the pane so that adding a task is a row here
#: rather than a seventh ``elif`` in a draw function -- ``_jobs_music.TASKS``
#: is the door's half of the same list, and ``open_derive`` refuses anything
#: this table has no entry for.
DERIVE_CONTROLS: dict[str, tuple[str, ...]] = {
    "retake": ("retake_variance",),
    "extend": ("extend_left", "extend_right"),
    "repaint": ("repaint_start", "repaint_end"),
    "edit": ("edit_prompt", "edit_lyrics"),
    "loop": ("repaint_end",),
    "audio2audio": ("ref_audio_strength",),
}


def open_derive(ctx: Any, job_id: str, task: str) -> None:
    """Point the derive popup at one take. **Frame-thread only.**

    The form is rebuilt from :data:`DEFAULT_DERIVE` on every open rather than
    carried, for the reason it is a separate dict at all: a window left over
    from the last take is a request about a piece of music the user is no
    longer looking at.
    """
    state = ensure(ctx)
    state.derive_job = job_id
    state.derive_form = dict(DEFAULT_DERIVE)
    state.derive_form["task"] = task if task in DERIVE_CONTROLS else "retake"


def close_derive(ctx: Any) -> None:
    ensure(ctx).derive_job = ""


def derive(ctx: Any) -> bool:
    """Queue what the popup asks for. -> whether the submit was accepted.

    ``generate``'s shape and its reasoning verbatim, on the other door: the
    shared ``"submit"`` key, no ``validate`` sibling, and every refusal coming
    back from ``derive_music_job`` with the ``field=`` that rings the control.

    An ``edit`` sends ``None`` for a field the user left alone rather than the
    empty string, because those mean different things at that door: ``None`` is
    "keep the parent's" and ``""`` is "drop the words entirely".
    """
    from ..service import jobs as svc_jobs

    state = ensure(ctx)
    form = dict(state.derive_form)
    job_id = state.derive_job
    task = str(form["task"])
    if not job_id:
        return False

    kwargs: dict[str, Any] = {"task": task, "count": int(form["count"])}
    for name in DERIVE_CONTROLS[task]:
        value = form[name]
        if name in ("edit_prompt", "edit_lyrics"):
            kwargs[name] = str(value) if str(value).strip() else None
        else:
            kwargs[name] = float(value)
    if task == "loop":
        # The popup asks for one figure -- how much of the joint to rewrite --
        # and the door reads a window. Zero to that span *is* that figure; the
        # door then centres it on the roll. See ``derive_music_job``.
        kwargs["repaint_start"] = 0.0

    ctx.state.clear_field_errors()
    if not ctx.submit("submit", lambda: svc_jobs.derive_music_job(ctx.svc, job_id, **kwargs)):
        ctx.toast("Still submitting the last one - try again in a moment.")
        return False
    state.derive_job = ""
    return True


# --- auditioning ------------------------------------------------------------


def track_path(ctx: Any, job_id: str):
    """Where a finished take's WAV is. One spelling, three callers."""
    return ctx.svc.config.job_dir(job_id) / "track.wav"


#: ``muse_io.read_track``, under the name this module used to define.
#:
#: The function moved to ``muse_io`` with everything else that touches a take's
#: file; the alias stays because ``tests/test_muse_mode.py`` patches it by this
#: name, and a rename that breaks a test's patch point is a rename that hides
#: what it changed. There is one implementation.
_read_track = muse_io.read_track


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
    """Adopt a decoded take, or a set of loop points. Routed by key prefix."""
    key, result = done.key, done.result
    if key.startswith(muse_io.FIND_PREFIX):
        one = player(ctx)
        if one is not None and one.job == key[len(muse_io.FIND_PREFIX) :]:
            one.finding = False
            one.candidates = list(result or [])
            if one.candidates:
                # The best one adopted immediately: the finder's whole output
                # is a ranking, and making the user press a second time to hear
                # the answer it already has is a step with no decision in it.
                choose_candidate(ctx, 0)
            else:
                ctx.toast("No loop points stood out in this take.", "warn")
        return
    if not key.startswith(LOAD_PREFIX) or not isinstance(result, dict):
        return
    job_id = key[len(LOAD_PREFIX) :]
    state = ensure(ctx)
    # **One take at a time.** ~42 MB for four minutes, so replaced rather than
    # cached per job -- see ``MuseState.player``.
    state.player = MusePlayer(
        job=job_id,
        pcm=result["pcm"],
        rate=int(result["rate"]),
        env=result.get("env"),
        duration=float(result.get("duration", 0.0)),
    )
    # Tagged with the job id, which is what lets a card ask "am *I* the one
    # playing" rather than only "is anything playing".
    if sirens_audio.play(result["pcm"], result["rate"], tag=job_id):
        state.playing_job = job_id
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
    if state is None:
        return
    if state.playing_job and not sirens_audio.playing():
        state.playing_job = ""
    if state.player is not None and state.player.job:
        # Drop the ~42 MB when its take leaves the Library. Off the cached rows
        # rather than a stat, for the tray's reason: this runs every frame.
        known = {str(job["id"]) for job in getattr(ctx.cache, "jobs", []) or []}
        if known and state.player.job not in known:
            state.player = None


def separate(ctx: Any, job_id: str) -> bool:
    """Queue a split of one take into stems. -> whether the submit landed.

    A one-line controller over ``separate_job``, and it is here rather than
    inline in the pane for ``generate``'s reason: every refusal comes back from
    the door carrying the ``field=`` that rings a control, and a second copy of
    those rules in a pane is the thing that drifts.

    The refusal that matters is the missing model, and it arrives with
    ``rows=`` so the toast reaches the Download button rather than being a
    sentence about a file. Muse itself is unaffected: a take with no stems is a
    take, which is why ``check_weights`` refuses this job and never a
    generation.
    """
    from ..service import jobs as svc_jobs

    ctx.state.clear_field_errors()
    if not ctx.submit(f"muse-separate:{job_id}", svc_jobs.separate_job, ctx.svc, job_id):
        ctx.toast("Already splitting that take.")
        return False
    return True


def has_stems(ctx: Any, job: Any) -> bool:
    """Whether this take already has its stems. Off the cached row.

    ``files.LISTED`` carries them, so the answer is in ``job["files"]`` and no
    pane has to stat anything per frame -- the tray's rule for every other
    per-card question.
    """
    files = job.get("files") or []
    return any(name.startswith("stems/") for name in files)


# --- the player -------------------------------------------------------------


def player(ctx: Any):
    """The decoded take under the strip, or ``None``. Never builds one."""
    state = active(ctx)
    return None if state is None else state.player


def position(ctx: Any) -> float:
    """Where the playhead is, in seconds into the *take*.

    ``sirens_audio.position`` answers where it is in the *buffer*, and seeking
    is slice-and-replay -- so the offset the slice began at is what makes the
    two the same number. That offset lives on :class:`Player`, not in
    ``sirens_audio``: the mixer does not own the caller's buffer, and a second
    module tracking the same figure is how the two come to disagree.
    """
    one = player(ctx)
    if one is None or not is_playing(ctx, one.job):
        return one.play_offset if one is not None else 0.0
    return min(one.play_offset + sirens_audio.position(), one.duration)


def seek(ctx: Any, seconds: float) -> None:
    """Move the playhead, restarting from there if a take is sounding.

    Slice-and-replay, exactly ``sirens_play.play_from_caret``: there is no
    device-side seek in the mixer this app uses, so the buffer handed to it
    *is* the remainder. Cheap because the samples are already in memory -- the
    slice is a view and ``make_sound`` copies once.

    The caller is expected to fire this on release rather than per mouse-move:
    a ``make_sound`` every frame of a drag is a ~40 MB copy per frame.
    """
    one = player(ctx)
    if one is None or one.pcm is None:
        return
    seconds = min(max(float(seconds), 0.0), one.duration)
    one.play_offset = seconds
    if not sirens_audio.playing() or sirens_audio.tag() != one.job:
        # Not sounding: move the playhead and leave it there. A seek is not a
        # play, and starting one because the user clicked the waveform to look
        # at something would be the pane deciding to make a noise.
        return
    _play_from(ctx, one, seconds)


def _play_from(ctx: Any, one: Any, seconds: float) -> None:
    """Put the remainder of the take on the channel, from ``seconds``."""
    start = int(seconds * one.rate)
    end = int(one.loop_end * one.rate) if one.loop_end is not None else None
    tail = one.pcm[start:end] if end and end > start else one.pcm[start:]
    if len(tail) == 0:
        return
    # ``loops=-1`` inside a region: ``sirens_audio.position`` already wraps
    # modulo the buffer length when loops is non-zero, so a region's playhead
    # falls out with no arithmetic here at all.
    repeat = -1 if one.loop_start is not None and one.loop_end is not None else 0
    if sirens_audio.play(tail, one.rate, tag=one.job, loops=repeat):
        one.play_offset = seconds
        ensure(ctx).playing_job = one.job


def play_region(ctx: Any) -> None:
    """Play the loop region on repeat, from its start. The audition that
    matters: a seam is a thing you judge by hearing it come round again."""
    one = player(ctx)
    if one is None or one.loop_start is None:
        return
    _play_from(ctx, one, one.loop_start)


def set_region(ctx: Any, start: float | None, end: float | None) -> None:
    """Set or clear the loop markers, in seconds. Ordered and clamped here.

    One place, because the markers are set from four: the finder's answer, the
    two draggable grips, and the ``[``/``]`` keys. Four copies of "clamp, then
    swap if reversed" is four chances to leave a region the exporter refuses.
    """
    one = player(ctx)
    if one is None:
        return
    if start is None or end is None:
        one.loop_start = one.loop_end = None
        return
    low = min(max(float(start), 0.0), one.duration)
    high = min(max(float(end), 0.0), one.duration)
    one.loop_start, one.loop_end = min(low, high), max(low, high)


def find_loops(ctx: Any) -> None:
    """Ask for loop points. The answer lands in ``on_task_done``.

    On a task because a four-minute take is a full STFT and a Gram matrix; the
    strip draws a spinner meanwhile, which is what "being wrong costs a spinner
    rather than a frozen window" buys.
    """
    one = player(ctx)
    if one is None or one.pcm is None:
        return
    one.finding = True
    if not ctx.submit(f"{muse_io.FIND_PREFIX}{one.job}", muse_io.find_loops, one.pcm, one.rate):
        one.finding = False
        ctx.toast("Still looking for loop points.")


def choose_candidate(ctx: Any, index: int) -> None:
    """Adopt one of the finder's answers as the region.

    By index rather than by value because the strip offers them as a numbered
    list -- and the list is the point: one answer with no alternatives would
    claim a confidence the method does not have (``muse.loops``).
    """
    one = player(ctx)
    if one is None or not 0 <= index < len(one.candidates):
        return
    candidate = one.candidates[index]
    set_region(ctx, candidate.start / one.rate, candidate.end / one.rate)


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

    The player's six are the same bargain, and cheap for the same reason.
    Left/Right nudge the playhead (Shift, ten times as far), Home returns it to
    the start, ``[`` and ``]`` set the loop's two ends *at the playhead* -- which
    is what makes the keyboard a real alternative to dragging a grip rather than
    a shortcut for the buttons -- and ``L`` runs the finder.

    Every one of them is a no-op with no player, which is the honest floor: they
    are about a decoded take, and before the first audition there is none.

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

    one = player(ctx)
    if one is None:
        return False
    if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
        step = NUDGE_SECONDS * (NUDGE_MULTIPLIER if event.mod & pygame.KMOD_SHIFT else 1)
        seek(ctx, position(ctx) + (step if event.key == pygame.K_RIGHT else -step))
        return True
    if event.key == pygame.K_HOME:
        seek(ctx, 0.0)
        return True
    if event.key == pygame.K_LEFTBRACKET:
        # At the playhead, and against whichever end already exists.
        # ``set_region`` orders the pair, so setting a start past the end is a
        # region with its two markers swapped rather than an invalid one.
        set_region(ctx, position(ctx), one.loop_end if one.loop_end is not None else one.duration)
        return True
    if event.key == pygame.K_RIGHTBRACKET:
        set_region(ctx, one.loop_start if one.loop_start is not None else 0.0, position(ctx))
        return True
    if event.key == pygame.K_l:
        find_loops(ctx)
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

    **The window moves when the take lands, not when the button is pressed.**
    The decode is a task and it can be refused; switching first meant a user
    read "this sample could not be loaded" in the tracker, with the take they
    were looking at a mode away. ``switch=True`` rides the task instead, and
    ``sirens_mode.on_task_done`` calls the one ``set_mode`` after the adopt.
    """
    path = track_path(ctx, job_id)
    if not path.exists():
        ctx.toast("that take has no audio on disk", "warn")
        return False
    tab = sirens_mode.active(ctx)
    if tab is None:
        tab = sirens_mode.new_document(ctx)
    sirens_io.import_sample(ctx, tab, path, switch=True)
    return True


def compose_from_sirens(ctx: Any, tab: Any = None) -> bool:
    """Render the open song and use it as a reference for a new take.

    ``open_in_sirens``'s mirror, and the direction the manual called
    deliberately unbuilt. **The Sirens half opens no door at all**: the document
    is serialised on the frame thread (the only thread it is safe to read on),
    rendered and encoded on a task thread, and all three of ``wsng_bytes``,
    ``read_wsng``/``synth.render`` and ``wavout.wav_bytes`` are already used in
    exactly this combination by ``sirens_play.request_render`` and
    ``sirens_io.export_plan``.

    The reference therefore carries **the loop points the user authored**,
    because ``wav_bytes`` writes them into the ``smpl`` chunk and this passes
    them. That is the headline feature meeting the round trip rather than
    fighting it.

    The rates already agree -- ``synth.SAMPLE_RATE`` is 44100 and the model's
    own loader resamples anything -- and ``tests/test_muse_bridge.py`` asserts
    it rather than a comment claiming it.

    One door *is* opened, on the Muse side: ``create_music_job`` takes scalars
    only, so it gains ``reference_wav``. See that function for why bytes rather
    than a path.
    """
    from ..service import jobs as svc_jobs
    from .sirens import wsng

    state = ensure(ctx)
    tab = tab or sirens_mode.active(ctx)
    if tab is None or not tab.doc.order:
        ctx.toast("There is nothing in the order list to compose from.", "warn")
        return False
    if not str(state.form["prompt"]).strip():
        # Refused here rather than at the door, because the door's sentence
        # would arrive in Muse pointing at a field the user is not looking at.
        # The tags are what the model is being asked *for*; the song is only
        # what it is being asked to sound like.
        ctx.toast("Describe the music you want in Muse first, then compose.", "warn")
        set_mode(ctx.state, "muse")
        return False

    # The snapshot, on the frame thread. ``request_render``'s rule and its
    # reason: this is where the document is safe to read.
    data = wsng.wsng_bytes(tab.doc)
    form = dict(state.form)

    def run():
        from ..service.errors import invalid_from
        from .sirens import synth, wavout

        try:
            doc = wsng.read_wsng(data)
            samples, loop, _marks = synth.render_marked(doc)
        except ValueError as exc:
            raise invalid_from(exc, "That song did not render") from exc
        reference = wavout.wav_bytes(samples, synth.SAMPLE_RATE, loop=loop)
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
            reference_wav=reference,
        )

    ctx.state.clear_field_errors()
    if not ctx.submit("submit", run):
        ctx.toast("Still submitting the last one - try again in a moment.")
        return False
    ctx.state.remember_prompt(str(form["prompt"]))
    # Through the one mode-switch implementation, never a direct assignment.
    set_mode(ctx.state, "muse")
    return True


def reset_form(ctx: Any) -> None:
    """Put the brief back to its defaults. The palette's Reset."""
    ensure(ctx).form = dict(DEFAULT_FORM)


__all__ = [
    "DEFAULT_DERIVE",
    "DEFAULT_FORM",
    "LOAD_PREFIX",
    "MuseState",
    "DERIVE_CONTROLS",
    "active",
    "close_derive",
    "compose_from_sirens",
    "derive",
    "ensure",
    "generate",
    "handle_key",
    "is_playing",
    "choose_candidate",
    "find_loops",
    "has_stems",
    "on_task_done",
    "open_derive",
    "open_in_sirens",
    "play",
    "play_region",
    "player",
    "position",
    "reset_form",
    "seek",
    "set_region",
    "select",
    "separate",
    "stop",
    "sync",
    "track_path",
]

# Referenced by the tray's card for its "which document would this land in"
# tooltip; imported here so the pane does not need a second Sirens import.
sirens_tab_title = sirens_state.title_for
