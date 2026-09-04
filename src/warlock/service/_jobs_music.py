"""Queueing a music job -- Muse's one door into the queue.

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

import shutil
import uuid
from typing import TYPE_CHECKING, Any

from .. import models
from .errors import Invalid
from .validation import (
    MAX_SEED,
    check_prompt,
    check_vram,
    check_weights,
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


def create_music_job(
    svc: WarlockService,
    *,
    prompt: str,
    lyrics: str = "",
    duration: float = 60.0,
    count: int = 1,
    music_model: str | None = None,
    seed: int | None = None,
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
    if seed is not None and not 0 <= int(seed) <= MAX_SEED:
        raise Invalid(f"seed must be between 0 and {MAX_SEED}", field="seed")

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
