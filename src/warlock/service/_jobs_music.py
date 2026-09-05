"""Queueing a music job -- Muse's one door into the queue.

**Two doors, and the second is a sibling of the first for the same reason the
first is a sibling of ``create_job``.** ``create_music_job`` is a brief fanned
into rows by a seed walk; ``derive_music_job`` walks ``retake_seed`` instead,
on a different key, in the middle of the one loop that function has -- and its
duration check, the refusal the whole admission story rests on, would grow an
``if task ==`` in front of it. Half the derived door's parameters are also
meaningless without a parent to measure them against.

**A sibling of ``create_job``, not a branch of it.** That function is hard-gated
to ``kind in ("text", "image")`` and its body is the reason why: an image
upload, a resolution, background removal, conditioning, an asset-type/intent
table, the promote gate and the sprite/tile follow-ups. Every one of those is a
statement about SDXL and trellis, and none has a meaning for a track. Adding a
third kind to that gate would have meant threading ``if kind == "music"`` past
each of them, which is the shape that makes a 900-line function.

What it does keep is the invariant, verbatim: **validate first, directory before
row, and ``rmtree`` if the row write raises.** A job directory with no row is
storage nothing will ever clean; a row with no directory is a job the worker
fails on its first read.
"""

from __future__ import annotations

import io
import shutil
import struct
import uuid
import wave
from typing import TYPE_CHECKING, Any

from .. import models
from .errors import Invalid
from .validation import (
    DERIVED_PARAMS,
    MAX_SEED,
    check_job_id,
    check_prompt,
    check_seed,
    check_vram,
    check_weights,
    not_done_message,
    random_seed,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import WarlockService

#: The longest track a single submit may ask for, in seconds.
#:
#: A bound and not a preference. Duration is the one parameter that is
#: *unbounded in cost*: it sets the latent length, so it drives both the
#: generation time and the figure ``vram.estimate`` has to price -- and an
#: unpriceable job is one admission cannot refuse before it OOMs. Four minutes
#: is longer than the loop any game needs and short enough that a mistyped
#: value is a wait rather than a wedged queue.
MAX_DURATION = 240.0

#: The shortest. Below this the model has no room to establish anything and
#: the output is an artefact rather than a piece of music.
MIN_DURATION = 10.0

#: Lyrics are markup plus words -- ``[verse]``/``[chorus]`` and the lines under
#: them. Capped far above any real song and far below "somebody pasted a
#: novel", which the lyric tokenizer would spend minutes on before the sampler
#: started.
MAX_LYRICS = 4000

#: How many takes one Generate may queue. The Create door's reason: several
#: cheap candidates to choose between is the point of a generative mode, and a
#: run away number of them is a queue nobody can get back.
MAX_COUNT = 4

#: How large a rendered reference may be, in bytes.
#:
#: A ceiling and not a preference, ``MAX_DURATION``'s argument on the other
#: input: this arrives as bytes over a function call rather than as a file the
#: user picked, so nothing else bounds it. Four minutes of 44.1 kHz 16-bit
#: stereo is ~42 MB, and this is comfortably above that -- the duration check
#: below is the one that actually decides, and this only stops a caller handing
#: over something absurd before ``wave`` is asked to parse it.
MAX_REFERENCE_BYTES = 128 * 1024 * 1024

#: The recipe knobs the mode exposes, each with its bound. Checked as a table
#: rather than as five hand-written ``if``s so that adding a sixth is a row --
#: and so the error sentence is one spelling, with ``field=`` naming the control
#: the UI must put it beside.
_RANGES: tuple[tuple[str, float, float], ...] = (
    ("infer_step", 1, 200),
    ("guidance_scale", 0.0, 30.0),
    ("omega_scale", 0.0, 100.0),
)

#: What ACE-Step's ``text2music_diffusion_process`` will dispatch on. Refused at
#: the door rather than passed through, because that dispatch is on a *string*:
#: an unknown one fails inside the sampler with the weights already resident,
#: which is the failure admission exists to move forward.
_SCHEDULERS = ("euler", "heun", "pingpong")
_CFG_TYPES = ("apg", "cfg", "cfg_star")

#: The registry rows Muse's two actions need, for the panes' pre-emptive gate.
#:
#: Derived from the same defaults ``validation.check_weights`` refuses on, so
#: the notice ahead of the button and the refusal at the door cannot name
#: different downloads. Siblings of ``sprites.SPRITE_ROWS`` and
#: ``sheets.PIXEL_SHEET_ROWS``.
MUSIC_ROWS: tuple[str, ...] = (f"music:{models.DEFAULT_MUSIC_MODEL}",)
SEPARATION_ROWS: tuple[str, ...] = (f"separation:{models.DEFAULT_SEPARATION}",)


def _check_ref_strength(strength: float, steps: int) -> float:
    """The two refusals an ``audio2audio`` strength owes. -> the value, as a float.

    Shared by both doors that mint the task (W1, 2026-09-05). The second is the
    interesting one and is **a refusal, not a runtime failure**: the sampler
    computes ``infer_steps = int((1 - strength) * infer_steps)``, so a strength
    close enough to 1.0 asks for zero sampling steps and decodes the noised
    reference unchanged -- a two-minute job that hands back its own input. Both
    controls are named in the sentence because either one fixes it.
    """
    value = float(strength)
    if not 0.0 <= value <= 1.0:
        raise Invalid("strength must be between 0 and 1", field="ref_audio_strength")
    if int((1.0 - value) * int(steps)) < 1:
        raise Invalid(
            "at that strength the model takes no sampling steps at all --"
            " lower the strength or raise the step count",
            field="ref_audio_strength",
        )
    return value


def create_music_job(
    svc: WarlockService,
    *,
    prompt: str,
    lyrics: str = "",
    duration: float = 60.0,
    count: int = 1,
    music_model: str | None = None,
    seed: int | None = None,
    reference_wav: bytes | None = None,
    ref_audio_strength: float = 0.5,
    infer_step: int = 60,
    guidance_scale: float = 15.0,
    scheduler_type: str = "euler",
    cfg_type: str = "apg",
    omega_scale: float = 10.0,
) -> dict[str, Any]:
    """Queue ``count`` takes of one brief. -> ``{"id": ..., "ids": [...]}``

    ``prompt`` is the comma-separated style tag string and ``lyrics`` is the
    marked-up lyric block -- the model's two inputs, under the model's own
    names, so nothing is translated between what the user types and what is
    asked for.

    Every take is an independent row with its own seed and its own directory,
    for the reason a reference candidate is: they are alternatives to choose
    between, and deleting one must not touch another.

    ``reference_wav`` is **the one door the reverse Sirens bridge opens**, and
    saying so is better than pretending otherwise: ``docs/INVARIANTS.md``'s "the
    bridge to Sirens opens no new doors" is about Muse -> Sirens and stays true,
    while the other direction genuinely needs this, because this function
    otherwise takes scalars only.

    ``ref_audio_strength`` goes with it, and is a parameter rather than the
    literal ``0.5`` this used to write into ``params`` unconditionally (W1,
    2026-09-05): it is the same knob the derive door exposes as *Closeness*, so
    a caller who can hand over a reference can say how near to stay to it. The
    same two refusals apply, through :func:`_check_ref_strength` -- one function
    rather than two hand-written copies that agree today.

    Deliberately the narrowest thing that works: **bytes, not a path.** A path
    means a temporary file with an owner, a lifetime and a cleanup story; bytes
    are written into the new job's own directory as ``reference.wav`` before the
    row exists, which is ``rerun_job``'s ``input.png`` precedent verbatim and
    which the existing ``made_dirs``/``rmtree`` block already covers with no new
    cleanup code. No temp directory, no config field, no orphan sweeper -- and
    the reference lives with the job that used it forever, which is provenance.
    """
    config = svc.config
    key = str(music_model or models.DEFAULT_MUSIC_MODEL)
    if key not in models.MUSIC_MODELS:
        # Refused at the door rather than falling back to the default: a silent
        # substitution records a recipe naming a model that never ran.
        raise Invalid("that music model is not one this build knows about",
                      field="music_model")

    prompt = (prompt or "").strip()
    if not prompt:
        raise Invalid("describe the music you want -- style tags, comma separated",
                      field="prompt")
    check_prompt(prompt)
    if len(lyrics) > MAX_LYRICS:
        raise Invalid(f"lyrics must be at most {MAX_LYRICS} characters",
                      field="lyrics")
    if not MIN_DURATION <= float(duration) <= MAX_DURATION:
        raise Invalid(
            f"duration must be between {MIN_DURATION:.0f} and {MAX_DURATION:.0f} "
            "seconds",
            field="duration",
        )
    if not isinstance(count, int) or isinstance(count, bool):
        raise Invalid("count must be a whole number", field="count")
    if not 1 <= count <= MAX_COUNT:
        raise Invalid(f"count must be between 1 and {MAX_COUNT}", field="count")
    if scheduler_type not in _SCHEDULERS:
        raise Invalid("that scheduler is not one the model has",
                      field="scheduler_type")
    if cfg_type not in _CFG_TYPES:
        raise Invalid("that guidance type is not one the model has", field="cfg_type")
    given = {
        "infer_step": infer_step,
        "guidance_scale": guidance_scale,
        "omega_scale": omega_scale,
    }
    for field, low, high in _RANGES:
        value = float(given[field])
        if not low <= value <= high:
            raise Invalid(f"{field} must be between {low:g} and {high:g}", field=field)
    check_seed("seed", seed)
    if reference_wav is not None:
        if not reference_wav:
            raise Invalid("that render produced no audio", field="reference_wav")
        if len(reference_wav) > MAX_REFERENCE_BYTES:
            raise Invalid(
                "that reference is larger than this build will read",
                field="reference_wav",
            )
        try:
            with wave.open(io.BytesIO(reference_wav)) as handle:
                seconds = handle.getnframes() / float(max(handle.getframerate(), 1))
        except (wave.Error, EOFError, struct.error) as exc:
            # **Three exception types, not one.** ``wave`` raises ``Error`` for
            # a file whose header it read and disliked, but a *truncated* one
            # dies inside ``Chunk`` with ``EOFError`` and a malformed size field
            # with ``struct.error`` -- and either escaping here is an unhandled
            # traceback where a refusal naming the control belongs.
            raise Invalid(
                "that reference is not a WAV file this build reads",
                field="reference_wav",
            ) from exc
        if not MIN_DURATION <= seconds <= MAX_DURATION:
            raise Invalid(
                f"a reference must be between {MIN_DURATION:.0f} and "
                f"{MAX_DURATION:.0f} seconds long",
                field="reference_wav",
            )

    params: dict[str, Any] = {
        "music_model": key,
        "lyrics": lyrics,
        "duration": float(duration),
        "infer_step": int(infer_step),
        "guidance_scale": float(guidance_scale),
        "scheduler_type": scheduler_type,
        "cfg_type": cfg_type,
        "omega_scale": float(omega_scale),
    }
    if reference_wav is not None:
        # ``audio2audio``, reached from Sirens rather than from a parent take.
        # The same task the derive door mints -- so the imported reference is
        # just another source, with no new worker key and no new queue branch.
        params["task"] = "audio2audio"
        params["ref_audio_strength"] = _check_ref_strength(
            ref_audio_strength, int(infer_step)
        )
    # Both refusals before anything is written, which is what "at the door"
    # means: check_weights names the download that fixes it, and check_vram
    # prices the job off the params above rather than a guess.
    check_weights(svc, "music", params)
    check_vram(svc, "music", "music", params)

    made_dirs = []
    ids: list[str] = []
    try:
        rows = []
        for index in range(count):
            take = dict(params)
            # An explicit seed applies to the *first* take and the rest walk
            # from it, so "count=4 with this seed" is four different takes with
            # one of them reproducible -- rather than four identical files.
            take["seed"] = (
                random_seed() if seed is None else (int(seed) + index) % (MAX_SEED + 1)
            )
            job_id = uuid.uuid4().hex[:12]
            job_dir = config.job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            # Recorded before anything is written into it, not after: a write
            # that dies mid-file must still remove its directory.
            made_dirs.append(job_dir)
            if reference_wav is not None:
                # ``source.wav``, the name ``_q_music._task_kwargs`` reads --
                # not ``reference.wav``, so that an imported reference and a
                # derived one are the same file to the queue and the worker.
                # Before the row, for the reason the directory is: ``next_queued``
                # can otherwise claim the job in the gap and find nothing.
                (job_dir / "source.wav").write_bytes(reference_wav)
            rows.append((job_id, take))
        with svc.store.transaction():
            for job_id, take in rows:
                svc.store.create("music", prompt, take, job_id, stage="music")
                ids.append(job_id)
    except Exception:
        # The DB savepoint rolled every row back, so every directory made by
        # this request is now unowned and must go with it.
        for job_dir in made_dirs:
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    svc.wake_worker()
    return {"id": ids[0], "ids": ids}


#: The Muse tasks ``derive_music_job`` will mint.
#:
#: ``loop`` is Muse's own name; ``_q_music._UPSTREAM_TASK`` is where it becomes
#: a repaint. The two tables are deliberately separate: this one is about what
#: the *door* accepts and that one about what the *sampler* is told, and folding
#: them would make a UI rename a wire change.
TASKS = ("retake", "extend", "repaint", "edit", "loop", "audio2audio")

#: Which params keys spell out a derived job's task block.
#:
#: **Deliberately not added to ``DERIVED_PARAMS``**, and the reason matters: the
#: task block is *the request normalised* -- ``duration``'s and ``lyrics``'
#: case, and ``subset``/``base_sheet``'s. A reroll of a repaint means "repaint
#: that window again with a new seed", so it must keep the window; a reroll that
#: stripped it would silently become a plain generation wearing a repaint's
#: provenance.
#:
#: It is stripped only by the door that re-specifies the whole block -- which is
#: what stops a repaint *of* an extend inheriting a pair of pads it has nothing
#: to do with.
TASK_PARAMS = (
    "task",
    "retake_variance",
    "extend_left",
    "extend_right",
    "parent_duration",
    "repaint_start",
    "repaint_end",
    "edit_prompt",
    "edit_lyrics",
    "edit_n_min",
    "edit_n_max",
    "ref_audio_strength",
    "roll",
)

#: How short a repaint or extend window may be, in seconds. Below this the
#: window is fewer latent frames than the model's own patch size, so the
#: request is arithmetic rather than music.
MIN_WINDOW = 1.0


def _parent_audio(svc: WarlockService, job_id: str) -> tuple[bytes, float]:
    """The parent take's bytes and its true duration. -> (data, seconds)

    Read off the file rather than out of ``actual_duration``: that key is what
    the worker was *asked* for, and every window this door validates is
    measured against what is actually on disk. A row whose params and file
    disagree is exactly the row a bad window would be minted from.
    """
    path = svc.job_dir(job_id) / "track.wav"
    try:
        data = path.read_bytes()
        with wave.open(io.BytesIO(data)) as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, wave.Error) as exc:
        # ``muse_mode.play``'s own sentence, so the two surfaces say the same
        # thing about the same missing file.
        raise Invalid("that take has no audio on disk", field="parent_id") from exc
    if rate <= 0 or frames <= 0:
        raise Invalid("that take has no audio on disk", field="parent_id")
    return data, frames / float(rate)


def derive_music_job(
    svc: WarlockService,
    job_id: str,
    *,
    task: str,
    count: int = 1,
    seed: int | None = None,
    retake_variance: float = 0.5,
    extend_left: float = 0.0,
    extend_right: float = 0.0,
    repaint_start: float = 0.0,
    repaint_end: float = 0.0,
    edit_prompt: str | None = None,
    edit_lyrics: str | None = None,
    edit_n_min: float = 0.0,
    edit_n_max: float = 1.0,
    ref_audio_strength: float = 0.5,
) -> dict[str, Any]:
    """Queue ``count`` derivations of a finished take. -> ``{"id", "ids"}``

    **Muse gains no document from this.** Every capability past the brief is a
    new job row derived from a parent -- the house model, the one
    ``promote_to_model`` and ``rerun_job`` already use -- rather than an undo
    stack and a file format. Document-shaped editing stays in Sirens.

    ``seed`` vs ``retake_seed``, which is the distinction the whole family
    turns on: **seed** is the take's own noise draw, and a derived row
    *inherits the parent's* so that it is genuinely a derivation of that take;
    **retake_seed** is the variation's, drawn fresh here. A child that re-ran
    with a new ``seed`` would be a different piece of music that happened to be
    filed under a parent.

    The parent's audio is *copied* to ``source.wav`` rather than pointed at,
    for ``rerun_job``'s ``input.png`` reason: the queue is serial, a music job
    can sit behind a two-minute reconstruction, and a parent trashed in that
    window would strand a job admission was supposed to have made safe. It
    costs ~21 MB a job and nothing in ordering -- the directory already exists
    before the row, and the existing ``except`` already removes it.

    Deliberately *not* in ``files.MEDIA``: that is the export allowlist, and a
    source is not this job's output.
    """
    config = svc.config
    check_job_id(job_id)
    if task not in TASKS:
        raise Invalid("that is not a kind of derivation this build makes", field="task")

    parent = svc.require_job(job_id)
    if parent.get("kind") != "music":
        raise Invalid("only a track can be derived from", field="parent_id")
    if parent.get("status") != "done":
        raise Invalid(
            not_done_message(
                "this take cannot be derived from yet: it", str(parent.get("status"))
            ),
            field="parent_id",
        )

    key = str(parent["params"].get("music_model") or models.DEFAULT_MUSIC_MODEL)
    if key not in models.MUSIC_MODELS:
        raise Invalid(
            "that take was made with a music model this build no longer has",
            field="music_model",
        )

    data, parent_duration = _parent_audio(svc, job_id)

    if not isinstance(count, int) or isinstance(count, bool):
        raise Invalid("count must be a whole number", field="count")
    if not 1 <= count <= MAX_COUNT:
        raise Invalid(f"count must be between 1 and {MAX_COUNT}", field="count")
    check_seed("seed", seed)

    # The task block, built and bounded one task at a time. Each refusal names
    # the control that fixes it, which is what ``tests/test_jobs_music.py``'s
    # parametrised table is over.
    block: dict[str, Any] = {"task": task, "parent_duration": parent_duration}
    duration = parent_duration

    if task == "retake":
        if not 0.0 <= float(retake_variance) <= 1.0:
            raise Invalid("variance must be between 0 and 1", field="retake_variance")
        block["retake_variance"] = float(retake_variance)

    elif task == "extend":
        left, right = float(extend_left), float(extend_right)
        if left < 0.0 or right < 0.0:
            raise Invalid("an extension cannot be negative", field="extend_right")
        if left + right < MIN_WINDOW:
            raise Invalid(
                f"extend by at least {MIN_WINDOW:g} second at one end or the other",
                field="extend_right",
            )
        # **An undocumented hard bound in the sampler, made a refusal here.**
        # The pads are sliced out of a tensor allocated at the *source's* frame
        # length, so a pad longer than the parent is silently zero-filled by
        # the shape patch downstream -- silence, not music. Naming it costs one
        # sentence; not naming it costs a two-minute generation of nothing.
        if left > parent_duration or right > parent_duration:
            raise Invalid(
                "a single extension cannot be longer than the take it extends"
                " -- extend twice to go further",
                field="extend_right" if right > parent_duration else "extend_left",
            )
        duration = parent_duration + left + right
        if duration > MAX_DURATION:
            raise Invalid(
                f"that would make a track longer than {MAX_DURATION:.0f} seconds",
                field="extend_right",
            )
        block["extend_left"] = left
        block["extend_right"] = right

    elif task in ("repaint", "loop"):
        if task == "loop":
            # The window is derived, not asked for: a loop is a repaint across
            # the head/tail joint, and the user chooses how much of the joint
            # to rewrite rather than where it is. Rolled by half the take so
            # the joint sits in the middle -- see ``_q_music._roll_wav``.
            span = float(repaint_end - repaint_start) or 8.0
            if not MIN_WINDOW <= span <= parent_duration / 2.0:
                raise Invalid(
                    "the joint to rewrite must be between"
                    f" {MIN_WINDOW:g} second and half the take",
                    field="repaint_end",
                )
            block["roll"] = parent_duration / 2.0
            middle = parent_duration / 2.0
            block["repaint_start"] = middle - span / 2.0
            block["repaint_end"] = middle + span / 2.0
        else:
            start, end = float(repaint_start), float(repaint_end)
            if start < 0.0 or end > parent_duration:
                raise Invalid("that window is not inside the take", field="repaint_start")
            if end - start < MIN_WINDOW:
                raise Invalid(
                    f"a repaint must cover at least {MIN_WINDOW:g} second",
                    field="repaint_end",
                )
            if start <= 0.0 and end >= parent_duration:
                # Refused by name rather than run: repainting the whole take is
                # what ``retake`` is, and the sampler agrees -- ``__call__``
                # sets exactly this window for that task. Two spellings of one
                # operation is the shape this repo writes tests against.
                raise Invalid(
                    "repainting the whole take is a retake -- ask for one of"
                    " those instead",
                    field="repaint_start",
                )
            block["repaint_start"] = start
            block["repaint_end"] = end

    elif task == "edit":
        was_prompt = (parent["prompt"] or "").strip()
        was_lyrics = parent["params"].get("lyrics") or ""
        new_prompt = (edit_prompt if edit_prompt is not None else was_prompt).strip()
        new_lyrics = edit_lyrics if edit_lyrics is not None else was_lyrics
        if not new_prompt:
            raise Invalid(
                "an edit still needs style tags -- clear the lyrics instead to"
                " drop the words",
                field="edit_prompt",
            )
        check_prompt(new_prompt)
        if len(new_lyrics) > MAX_LYRICS:
            raise Invalid(
                f"lyrics must be at most {MAX_LYRICS} characters", field="edit_lyrics"
            )
        if new_prompt == was_prompt and new_lyrics == was_lyrics:
            raise Invalid(
                "an edit needs something changed -- retake it to get another"
                " take of the same brief",
                field="edit_prompt",
            )
        if not 0.0 <= float(edit_n_min) < float(edit_n_max) <= 1.0:
            raise Invalid(
                "the edit window must run from a lower figure to a higher one,"
                " both between 0 and 1",
                field="edit_n_max",
            )
        block["edit_prompt"] = new_prompt
        block["edit_lyrics"] = new_lyrics
        block["edit_n_min"] = float(edit_n_min)
        block["edit_n_max"] = float(edit_n_max)

    else:  # audio2audio
        # The step count is the *parent's*, because a derivation inherits its
        # recipe; the reference door checks its own.
        block["ref_audio_strength"] = _check_ref_strength(
            ref_audio_strength, int(parent["params"].get("infer_step", 60))
        )

    # The parent's recipe, minus what the worker recorded about *its* artifacts
    # and minus the parent's own task block. ``DERIVED_PARAMS`` is the
    # ``_jobs_resubmit`` idiom verbatim; ``TASK_PARAMS`` is this door's own, and
    # stripping it here is what stops a repaint of an extend inheriting a pair
    # of pads -- see that tuple's docstring.
    params = {
        k: v
        for k, v in parent["params"].items()
        if k not in DERIVED_PARAMS and k not in TASK_PARAMS
    }
    params["duration"] = duration
    params.update(block)
    # The parent's noise draw, carried deliberately: see the docstring.
    params["seed"] = parent["params"].get("seed")
    prompt = str(block.get("edit_prompt") or parent["prompt"] or "")

    check_weights(svc, "music", params)
    check_vram(svc, "music", "music", params)

    if task == "loop":
        from .._q_music import _roll_wav

        # Rolled once here rather than once per take: the roll is a property of
        # the parent, and ``count`` derivations share it.
        data = _roll_wav(data, block["roll"])

    made_dirs = []
    ids: list[str] = []
    try:
        rows = []
        for index in range(count):
            take = dict(params)
            # ``retake_seed``, not ``seed``: the variation's draw. An explicit
            # one applies to the first and the rest walk from it, which is
            # ``create_music_job``'s rule on the key this door actually varies.
            take["retake_seed"] = (
                random_seed() if seed is None else (int(seed) + index) % (MAX_SEED + 1)
            )
            new_id = uuid.uuid4().hex[:12]
            job_dir = config.job_dir(new_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            made_dirs.append(job_dir)
            # Before the row, for ``create_music_job``'s reason: ``next_queued``
            # can otherwise claim the job in the gap and find no source on disk.
            (job_dir / "source.wav").write_bytes(data)
            rows.append((new_id, take))
        with svc.store.transaction():
            for new_id, take in rows:
                # ``parent_id`` is a *column*, never a params key -- restating
                # ``promote_to_model``'s rule, and it matters more here: this
                # door copies params, so a params key would be inherited by
                # every later derivation and the lineage would flatten.
                svc.store.create(
                    "music", prompt, take, new_id, stage="music", parent_id=job_id
                )
                ids.append(new_id)
    except Exception:
        for job_dir in made_dirs:
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    svc.wake_worker()
    return {"id": ids[0], "ids": ids}
