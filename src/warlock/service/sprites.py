"""Sprite sheet drafts: what one may ask for, and queuing a synthesis.

The door for the 2D entry point. Everything a ``sprite_synthesis`` job can be
refused for is checked here, before the row exists, for the reason
``create_sheet`` and ``create_pixel_sheet`` both state: an unrunnable request
should cost the request, not a place in the queue and two SDXL generations.

Nothing here reads a draft's *pixels*. The listing and the reads are
``rigging``'s pure file helpers behind the id guards, so the pane can call them
on the frame thread behind a stamp.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .. import rigging
from .core import WarlockService
from .errors import Conflict, Invalid, NotFound
from .validation import (
    check_job_id,
    check_seed,
    check_sprite_draft_id,
    check_vram,
    not_done_message,
    random_seed,
)

SPRITE_SHEET_TYPES = ("turnaround", "walk")

# Its own tuple rather than ``sheets.PIXEL_LOGICAL_SIZES``, deliberately. That
# one is the set of sizes that divide a *rendered* sheet's frame exactly, which
# is a constraint this path does not have -- the atlas is reduced in one pass
# and any size lands on a cell boundary. Sharing the tuple would tie two
# unrelated menus together and offer 16px cells here, which is below the size a
# recognisable four-direction character fits in.
SPRITE_LOGICAL_SIZES = (32, 48, 64)

SPRITE_COLOR_CHOICES = (8, 16, 32, 64)

DEFAULT_SPRITE_SHEET_TYPE = "turnaround"
DEFAULT_SPRITE_LOGICAL_SIZE = 64
DEFAULT_SPRITE_COLORS = 32

# The base every draft generates on today. Named here, and written into params,
# for the reason ``create_pixel_sheet`` gives: params outlive today's UI, and
# the day this becomes a control the pairing refusal below is already at the
# door.
SPRITE_BASE_MODEL = "sdxl_cfg"


def sprite_options() -> dict[str, Any]:
    """What a sprite sheet request may ask for. One source for the form.

    The grid summary comes from ``spritesynth.GEOMETRY`` rather than being
    written out again here: the pane says "4 cells, 2x2" under the combo, and a
    second copy of that arithmetic is a label that would go stale the first
    time a sheet type was added.
    """
    from ..pipelines import spritesynth

    types = []
    for key in SPRITE_SHEET_TYPES:
        geom = spritesynth.geometry(key)
        types.append(
            {
                "key": key,
                "columns": geom.columns,
                "rows": geom.rows,
                "cells": len(geom.cells),
                "frames_per_direction": geom.frames_per_direction,
            }
        )
    return {
        "sheet_types": types,
        "logical_sizes": list(SPRITE_LOGICAL_SIZES),
        "colors": list(SPRITE_COLOR_CHOICES),
        "directions": list(spritesynth.DIRECTION_ORDER),
        "defaults": {
            "sheet_type": DEFAULT_SPRITE_SHEET_TYPE,
            "logical_size": DEFAULT_SPRITE_LOGICAL_SIZE,
            "colors": DEFAULT_SPRITE_COLORS,
        },
    }


def _check_weights(svc: WarlockService) -> None:
    """Everything a synthesis loads, refused by name with its download line.

    ``validation.check_weights`` stays text-only on purpose -- it is keyed on
    ``params`` a text job wrote -- so this kind brings its own. The four are not
    optional here the way they are for a text job: the pose guide *is* the
    ControlNet and the identity *is* the IP-Adapter, so a missing one is not a
    slightly plainer picture, it is the feature not happening.
    """
    from .. import fetch, models

    base = models.BASE_MODELS[SPRITE_BASE_MODEL]
    ok, missing_lora = fetch.base_model_state(svc.config, base)
    if not ok:
        what = (
            f"its step-distillation LoRA is missing at {missing_lora}"
            if missing_lora is not None
            else "its weights are not downloaded"
        )
        raise Invalid(
            f"The image model {base.label!r} cannot run: {what}. "
            f"Download it with:\n  {base.download}",
            field="base_model",
        )
    required = (
        ("adapter", models.IP_ADAPTERS["plus"], "ip_adapter"),
        ("control", models.CONTROLNETS["canny"], "control"),
        ("lora", models.STYLE_LORAS[models.PIXEL_SHEET_LORA], "style_lora"),
    )
    for kindname, spec, field in required:
        if fetch.present(svc.config, kindname, spec):
            continue
        raise Invalid(
            f"A sprite sheet needs {spec.label!r}, which is not downloaded. "
            f"Download it with:\n  {spec.download}",
            field=field,
        )


def create_sprite_synthesis(
    svc: WarlockService,
    job_id: str,
    *,
    sheet_type: str = DEFAULT_SPRITE_SHEET_TYPE,
    logical_size: int = DEFAULT_SPRITE_LOGICAL_SIZE,
    colors: int = DEFAULT_SPRITE_COLORS,
    seed_a: int | None = None,
    seed_b: int | None = None,
) -> dict[str, Any]:
    """Queue two candidate sprite atlases from one finished reference.

    Two, not one, and that is the whole shape of the feature: what the model
    imagines for the three views it has never seen is a guess, and a pair the
    user picks between is a far better use of the same minute than one draft
    they have to decide about in the abstract.
    """
    from .. import models

    check_job_id(job_id)
    source = svc.require_job(job_id)
    if source["stage"] != "reference":
        raise Invalid("only a 2D reference can be made into a sprite sheet")
    if source["status"] != "done":
        raise Invalid(not_done_message("That reference", source["status"]))
    job_dir = svc.job_dir(job_id)
    if not (job_dir / "input.png").exists():
        raise Invalid("reference has no image")

    if str(sheet_type) not in SPRITE_SHEET_TYPES:
        raise Invalid(
            f"sheet_type must be one of {list(SPRITE_SHEET_TYPES)}", field="sheet_type"
        )
    if int(logical_size) not in SPRITE_LOGICAL_SIZES:
        raise Invalid(
            f"logical_size must be one of {list(SPRITE_LOGICAL_SIZES)}",
            field="logical_size",
        )
    if int(colors) not in SPRITE_COLOR_CHOICES:
        raise Invalid(
            f"colors must be one of {list(SPRITE_COLOR_CHOICES)}", field="colors"
        )

    if seed_a is not None:
        check_seed("seed_a", seed_a)
    if seed_b is not None:
        check_seed("seed_b", seed_b)
    first = random_seed() if seed_a is None else int(seed_a)
    second = int(seed_b) if seed_b is not None else None
    if second is None:
        # Drawn until distinct rather than "first + 1": consecutive seeds are
        # not meaningfully more different than equal ones under the same
        # conditioning, and the point of the pair is two independent guesses.
        while second is None or second == first:
            second = random_seed()
    if first == second:
        raise Invalid(
            "the two candidates need different seeds, or they will be the "
            "same picture twice",
            field="seed_b",
        )

    # The pixel look is the style LoRA, so a base it does not fit is refused
    # here rather than queued -- the worker would drop the adapter and publish
    # two smooth illustrations chopped into a grid, which is two generations
    # spent on a result the request did not mean. Same refusal, same wording
    # and same field as ``create_pixel_sheet``.
    base = models.BASE_MODELS[SPRITE_BASE_MODEL]
    pixel_lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    if not models.lora_fits(base, pixel_lora):
        fitting = sorted(
            key
            for key, loras in models.loras_by_base().items()
            if models.PIXEL_SHEET_LORA in loras
        )
        raise Invalid(
            f"base_model {base.key!r} is {base.family} and the pixel-sheet "
            f"LoRA {pixel_lora.key!r} is fitted to {pixel_lora.family}; "
            f"pick one of {fitting}",
            field="base_model",
        )

    _check_weights(svc)

    if len(rigging.list_sprite_drafts(job_dir)) >= rigging.MAX_SPRITE_DRAFTS:
        raise Conflict(
            f"this reference already has {rigging.MAX_SPRITE_DRAFTS} sprite "
            "sheet drafts; delete one first"
        )

    draft_id = rigging.new_id()
    params = {
        # Inputs only, exactly as ``create_pixel_sheet`` says: what actually
        # ran is recorded in the draft's own sidecar recipe, so nothing here is
        # derived and a rerun copies it verbatim.
        "source_job": job_id,
        "sheet_type": str(sheet_type),
        "logical_size": int(logical_size),
        "colors": int(colors),
        "seed_a": first,
        "seed_b": second,
        # Minted at the door so the create result can name the draft the pane
        # is about to watch for, rather than the pane having to diff a listing.
        "draft_id": draft_id,
        "base_model": SPRITE_BASE_MODEL,
    }
    check_vram(svc, "sprite_synthesis", "model", params)
    new_id = svc.store.create(
        "sprite_synthesis", source["prompt"], params, uuid.uuid4().hex[:12]
    )
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "draft": draft_id}


def list_sprite_drafts(svc: WarlockService, job_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    return {"drafts": rigging.list_sprite_drafts(svc.job_dir(job_id))}


def get_sprite_draft(svc: WarlockService, job_id: str, draft_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    check_sprite_draft_id(draft_id)
    record = rigging.read_sprite_draft(svc.job_dir(job_id), draft_id)
    if record is None:
        raise NotFound("no such sprite draft")
    return record


def sprite_draft_png(
    svc: WarlockService, job_id: str, draft_id: str, candidate: str
) -> Path:
    """One candidate's atlas, once the draft has finished.

    Gated on the sidecar and not on the PNG's existence: both PNGs are written
    before it, so a file that is there may still be half-written -- the same
    completion-marker rule ``sheet_pixel_png`` follows.
    """
    check_job_id(job_id)
    check_sprite_draft_id(draft_id)
    if candidate not in rigging.SPRITE_CANDIDATES:
        raise NotFound("no such sprite candidate")
    job_dir = svc.job_dir(job_id)
    path = rigging.sprite_draft_png_path(job_dir, draft_id, candidate)
    if not path.exists() or not rigging.sprite_draft_path(job_dir, draft_id).exists():
        raise NotFound("no such sprite draft")
    return path


def delete_sprite_draft(
    svc: WarlockService, job_id: str, draft_id: str
) -> dict[str, Any]:
    check_job_id(job_id)
    check_sprite_draft_id(draft_id)
    if not rigging.delete_sprite_draft(svc.job_dir(job_id), draft_id):
        raise NotFound("no such sprite draft")
    return {"deleted": draft_id}
