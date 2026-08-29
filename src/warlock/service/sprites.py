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

from .. import models, rigging
from ..pipelines import spritesynth
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

#: The two fixed-atlas kinds, which are always on offer: their guides ship and
#: every draft on disk is one of them. Deliberately *not* the whole menu any
#: more -- :func:`sprite_sheet_types` adds the planned kinds that have a pose
#: guide behind them, discovered on disk rather than listed here.
SPRITE_LEGACY_SHEET_TYPES = ("turnaround", "walk")

#: Historical spelling, kept because it is what several callers and tests import
#: and because it still names exactly what it always did.
SPRITE_SHEET_TYPES = SPRITE_LEGACY_SHEET_TYPES

#: What one band of one direction costs, in seconds, for the "about N minutes"
#: line a form draws before anything is queued. The worker's own ``nominal`` for
#: one SDXL generation on this path, restated rather than guessed at, and it is
#: an over-estimate for a band smaller than a full frame -- which is the right
#: direction for a promise about how long somebody will be waiting.
SECONDS_PER_GENERATION = 20.0

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

# Taken from the pipeline rather than restated, the rule this module already
# follows for the grids: a synthesised cell has no guaranteed margin and
# ``outer`` clips at a cell edge, and the whole argument lives beside the code
# that has to live with it. A second copy here would be one edit away from a
# door offering a look the assembler does not draw.
DEFAULT_SPRITE_OUTLINE = spritesynth.DEFAULT_SPRITE_OUTLINE

# The base every draft generates on today. Named here, and written into params,
# for the reason ``create_pixel_sheet`` gives: params outlive today's UI, and
# the day this becomes a control the pairing refusal below is already at the
# door.
SPRITE_BASE_MODEL = "sdxl_cfg"

# The three non-base weights a synthesis loads: (registry kind, registry key,
# the form field a refusal about it names). One list rather than three, and the
# single source for both ``SPRITE_ROWS`` below and ``_check_weights``'s loop --
# they were about to be two hand-copies of the same fact, which is the drift
# ``fetch.KINDS`` was written to end everywhere else.
_SPRITE_REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("adapter", "plus", "ip_adapter"),
    ("control", "canny", "control"),
    ("lora", models.PIXEL_SHEET_LORA, "style_lora"),
)

# Every registry row a sprite synthesis needs on this host, in
# ``fetch.Entry.row_key`` spelling and in the order they are checked. Exported
# so a pane can offer "install what this needs" without knowing what a sprite
# sheet is made of. Sizes are *not* summed from here: three of these four rows
# resolve to weights the others may already imply, so a caller wanting a figure
# composes this with ``downloads.plan_for``, which dedupes.
SPRITE_ROWS: tuple[str, ...] = (
    f"base:{SPRITE_BASE_MODEL}",
    *(f"{kind}:{key}" for kind, key, _field in _SPRITE_REQUIRED),
)


def sprite_sheet_types() -> tuple[str, ...]:
    """Every sheet kind this host can actually draw, in menu order.

    The two legacy atlases, then every planned kind with a pose guide *on this
    installation's disk*. Discovered rather than listed, and that is the whole
    point: the remaining poses are art and they land one file at a time, so a
    hardcoded menu is wrong on both sides -- it offers an action with no guide
    behind it, which is eight bands of an unposed character conditioned on
    nothing, and it hides one somebody has just authored.
    """
    return (*SPRITE_LEGACY_SHEET_TYPES, *spritesynth.available_kinds())


def sprite_kind(action: str, directions: int) -> str:
    """The sheet kind two form controls name between them."""
    return f"{action}{int(directions)}"


def kind_logical_sizes(kind: str) -> tuple[int, ...]:
    """Which of this door's cell sizes ``kind`` can actually be drawn at.

    The whole ladder for the two fixed atlases: they are one 1024px generation
    however small the published cell is, so the band arithmetic has nothing to
    say about them. For a planned kind it is
    ``spritesynth.sizes_for_action`` -- one direction is one generation, so an
    eight-frame action at 64px would want a 2048px-wide band and there is no
    such thing.
    """
    if kind in SPRITE_LEGACY_SHEET_TYPES:
        return tuple(SPRITE_LOGICAL_SIZES)
    action = spritesynth.KIND_ACTIONS.get(kind, "")
    if not action:
        return ()
    return spritesynth.sizes_for_action(action, SPRITE_LOGICAL_SIZES)


def _action_entry(action: str, counts: tuple[int, ...]) -> dict[str, Any]:
    """One action's row of :func:`sprite_options`, with the arithmetic done.

    The counts, the cells, the bands and the size ladder come from
    ``spritesynth`` and are *not* recomputed by the pane: the form draws a line
    saying "8 directions x 4 frames = 32 cells, 8 generations, about 3 minutes"
    before anything is queued, and a second copy of that arithmetic is a promise
    that goes stale the first time a frame count moves.
    """
    frames = spritesynth.ACTION_FRAMES[action]
    sizes = spritesynth.sizes_for_action(action, SPRITE_LOGICAL_SIZES)
    return {
        "key": action,
        "label": action.title(),
        "frames": frames,
        # The sizes this action fits a band at, out of the ones this door
        # offers. Empty is possible in principle (an action with more frames
        # than any rung takes) and is the honest rendering of "this action
        # cannot be drawn at any size on the menu" rather than a crash later.
        "logical_sizes": list(sizes),
        "directions": [
            {
                "count": count,
                "kind": sprite_kind(action, count),
                "cells": count * frames,
                "columns": frames,
                "rows": count,
                # One band is one whole direction, so this is also how many
                # generations one candidate costs.
                "bands": count,
                "seconds_per_candidate": count * SECONDS_PER_GENERATION,
                "candidates": spritesynth.default_candidates(count * frames),
            }
            for count in counts
        ],
    }


def sprite_options() -> dict[str, Any]:
    """What a sprite sheet request may ask for. One source for the form.

    The grid summary comes from ``spritesynth`` rather than being written out
    again here: the pane says "4 cells, 2x2" under the combo, and a second copy
    of that arithmetic is a label that would go stale the first time a sheet
    type was added.

    ``actions`` carries only the actions with a guide template on disk -- see
    :func:`sprite_sheet_types` -- with, per action, the direction counts it can
    be drawn as and the *size ladder it allows*. A six- or eight-frame action
    does not fit one band at 48 or 64px (``spritesynth.plan_sheet`` names both
    numbers and refuses), so a size picker that did not gate on the action would
    be a control whose only outcome is a refusal after the press.
    """
    from ..pipelines import pixelize

    types = []
    for key in sprite_sheet_types():
        sizes = kind_logical_sizes(key)
        if not sizes:
            # An action with more frames than the smallest rung on this door's
            # ladder takes. Not offered rather than offered-and-refused, which
            # is the same rule the missing-guide filter follows: a menu entry
            # whose only outcome is a refusal is not a menu entry.
            continue
        geom = spritesynth.sheet_geometry(key, max(sizes))
        types.append(
            {
                "key": key,
                "columns": geom.columns,
                "rows": geom.rows,
                "cells": len(geom.cells),
                "frames_per_direction": geom.frames_per_direction,
                "directions": list(geom.directions),
                "logical_sizes": list(sizes),
            }
        )
    available = spritesynth.available_actions()
    return {
        "sheet_types": types,
        "actions": [
            _action_entry(action, counts)
            for action, counts in sorted(
                available.items(),
                key=lambda item: list(spritesynth.ACTION_FRAMES).index(item[0]),
            )
        ],
        "logical_sizes": list(SPRITE_LOGICAL_SIZES),
        "colors": list(SPRITE_COLOR_CHOICES),
        "directions": list(spritesynth.DIRECTION_ORDER),
        "seconds_per_generation": SECONDS_PER_GENERATION,
        # From ``pixelize`` for the reason the grids come from ``spritesynth``:
        # a form offering a mode the assembler refuses is a control that fails
        # at the door it was drawn from.
        "outlines": list(pixelize.OUTLINE_MODES),
        "defaults": {
            "sheet_type": DEFAULT_SPRITE_SHEET_TYPE,
            "logical_size": DEFAULT_SPRITE_LOGICAL_SIZE,
            "colors": DEFAULT_SPRITE_COLORS,
            "outline": DEFAULT_SPRITE_OUTLINE,
            "dither": False,
            "palette": "",
        },
    }


def sprite_palettes(svc: WarlockService) -> list[str]:
    """Every authored palette a sprite sheet may be drawn on, by stem, sorted.

    ``tilesheets.tile_sheet_palettes``' twin, and its docstring's argument
    applies here unchanged: this is the one answer on this door that can change
    without the process changing, so it is a call taking ``svc`` rather than a
    key in :func:`sprite_options` -- which reads no disk and which a pane may
    therefore cache for its whole life. A palette is a *file the user drops in
    a directory*; folded into the options blob it would be a directory listing
    cached until the app restarts, and a palette installed five minutes ago
    would never appear.

    Empty on a host with no palette directory, which is not an error: the whole
    feature is optional and a form that offers nothing is the correct rendering
    of "none installed".
    """
    from . import palettes

    return palettes.available(svc.config)


def _check_options(svc: WarlockService, entries: dict[str, Any]) -> dict[str, Any]:
    """The pixelisation options a sprite request may carry, validated once.

    ``troupe._check_options``' shape and its reason: this is where the sprite
    path's ladders and its ``inner`` default live, so neither of the two doors
    below has to restate them -- and there are exactly two, which is why a
    shared function rather than a copy is the difference between "both refuse
    the same value" and "both happened to today".

    ``allow_reduce_mode=False`` because this path does not *have* a reduce mode:
    ``_sprite_assemble`` reduces the whole atlas with the alpha-weighted box and
    nothing offers the alternative. That is deliberately stronger than
    validating one and dropping it, for the reason ``check_pixel_options``
    gives -- a params blob quietly carrying a setting nothing reads is a dead
    field.
    """
    from .pixelopts import check_pixel_options

    return check_pixel_options(
        svc,
        entries,
        sizes=SPRITE_LOGICAL_SIZES,
        size_default=DEFAULT_SPRITE_LOGICAL_SIZE,
        colors=SPRITE_COLOR_CHOICES,
        colors_default=DEFAULT_SPRITE_COLORS,
        outline_default=DEFAULT_SPRITE_OUTLINE,
        allow_reduce_mode=False,
    )


def resolve_sheet_kind(
    sheet_type: Any = None, action: Any = None, directions: Any = None
) -> str:
    """The one kind name behind the two vocabularies a request may use.

    Two callers speak two languages and neither is wrong. A form has an Action
    combo and a Directions control, because those are the two things a person
    chooses; a stored params blob has ``sheet_type``, because that is what every
    draft on disk carries and what ``spritesynth.sheet_geometry`` is addressed
    by. This is the only place they are reconciled, so that exactly one spelling
    -- the kind -- is ever written down.

    An explicit ``action`` wins over a ``sheet_type`` sent beside it: a request
    carrying both came from a form whose controls are the action pair, and the
    ``sheet_type`` in it is the stale half.
    """
    if action:
        count = int(directions) if directions else spritesynth.DIRECTION_COUNTS[-1]
        return sprite_kind(str(action), count)
    return str(sheet_type or DEFAULT_SPRITE_SHEET_TYPE)


def check_sheet_kind(kind: str, logical: int) -> None:
    """Refuse a sheet nothing can draw, before the row exists.

    Three refusals and each one costs the request rather than an hour:

    * a kind this module does not name at all;
    * a planned kind with **no pose guide on disk**. The guide *is* the pose --
      the ControlNet hint is the only thing that decides where the limbs go --
      so without one the request is eight generations of an unposed character
      described by the right sentence, which is the failure that is invisible
      until somebody looks at the sheet;
    * a direction whose frames do not fit one band at this size.
      ``spritesynth.plan_sheet`` owns that arithmetic and names both numbers,
      so its sentence is re-raised rather than re-written: the door and the
      pipeline refusing in two wordings is how they come to refuse two
      different sets.
    """
    if kind in SPRITE_LEGACY_SHEET_TYPES:
        return
    if kind not in spritesynth.PLANNED_KINDS:
        raise Invalid(
            f"sheet_type must be one of {list(sprite_sheet_types())}",
            field="sheet_type",
        )
    if not spritesynth.has_guide_template(kind):
        offered = ", ".join(sprite_sheet_types())
        raise Invalid(
            f"there is no pose guide for a {kind!r} sheet, and the guide is what "
            f"decides where the limbs go -- without one the sheet would be an "
            f"unposed character in every cell. This build draws {offered}.",
            field="sheet_type",
        )
    try:
        spritesynth.plan_kind(kind, int(logical))
    except ValueError as exc:
        raise Invalid(str(exc), field="logical_size") from exc


def check_candidates(count: Any, cells: int) -> int:
    """How many candidates to draw, refused if it is not one or two.

    ``None`` is not a value: it is "nobody said", and what the path does then is
    ``spritesynth.default_candidates``' decision rather than this door's -- a
    pair for a small sheet, one draft for a big one.
    """
    if count is None:
        return spritesynth.default_candidates(cells)
    try:
        wanted = int(count)
    except (TypeError, ValueError):
        wanted = 0
    if wanted not in (1, 2):
        raise Invalid(
            "a sprite draft is one candidate or two; two of a sheet this size is "
            f"{2 * cells} generated cells",
            field="candidates",
        )
    return wanted


def _check_weights(svc: WarlockService) -> None:
    """Everything a synthesis loads, refused by name with its download line.

    ``validation.check_weights`` stays text-only on purpose -- it is keyed on
    ``params`` a text job wrote -- so this kind brings its own. The four are not
    optional here the way they are for a text job: the pose guide *is* the
    ControlNet and the identity *is* the IP-Adapter, so a missing one is not a
    slightly plainer picture, it is the feature not happening.
    """
    from .. import fetch
    from .downloads import needed_keys
    from .validation import check_base_model_weights, install_remedy

    check_base_model_weights(
        svc,
        models.BASE_MODELS[SPRITE_BASE_MODEL],
        rows=needed_keys(svc, SPRITE_ROWS),
    )
    for kindname, key, field in _SPRITE_REQUIRED:
        entry = fetch.find(f"{kindname}:{key}")
        assert entry is not None, f"{kindname}:{key} is not a registry row"
        spec = entry.spec
        if fetch.present(svc.config, kindname, spec):
            continue
        raise Invalid(
            f"A sprite sheet needs {spec.label!r}, which is not downloaded. "
            f"{install_remedy(spec.label, fetch.download_text(svc.config, kindname, spec))}",
            field=field,
            # Every row the feature is short of, not merely the one that
            # tripped: a user offered "install what this needs" wants one
            # download, not three refusals in a row.
            rows=needed_keys(svc, SPRITE_ROWS),
        )


def create_sprite_synthesis(
    svc: WarlockService,
    job_id: str,
    *,
    sheet_type: str = DEFAULT_SPRITE_SHEET_TYPE,
    action: str | None = None,
    directions: int | None = None,
    candidates: int | None = None,
    logical_size: int = DEFAULT_SPRITE_LOGICAL_SIZE,
    colors: int = DEFAULT_SPRITE_COLORS,
    palette: str | None = None,
    dither: bool = False,
    outline: str | None = None,
    seed_a: int | None = None,
    seed_b: int | None = None,
) -> dict[str, Any]:
    """Queue candidate sprite atlases from one finished reference.

    Two candidates, not one, is the shape the feature was written in: what the
    model imagines for the views it has never seen is a guess, and a pair the
    user picks between is a far better use of the same minute than one draft
    they have to decide about in the abstract. It stops being the default once a
    sheet is bigger than the atlas kinds -- see
    ``spritesynth.default_candidates`` -- because a pair of an eight-direction
    sheet is sixteen generations, and ``candidates`` may pin either.

    ``action`` and ``directions`` are the pair a form asks with;
    :func:`resolve_sheet_kind` turns them into the one ``sheet_type`` that is
    stored, so a params blob has one spelling of what it is.
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

    kind = resolve_sheet_kind(sheet_type, action, directions)
    # Through the shared checker, which is also what ``_check_sprite_sheet``
    # calls: the two doors have to refuse the same values in the same sentences
    # on the same fields, and the only way to be sure of that is for there to be
    # one of them.
    options = _check_options(
        svc,
        {
            "logical_size": logical_size,
            "colors": colors,
            "palette": palette,
            "dither": dither,
            "outline": outline,
        },
    )
    # After the size is on the ladder and before anything is queued: the band
    # refusal is *about* the size, so refusing an off-ladder one first keeps the
    # two sentences from arriving in the wrong order.
    check_sheet_kind(kind, options["logical_size"])
    geom = spritesynth.sheet_geometry(kind, options["logical_size"])
    wanted = check_candidates(candidates, len(geom.cells))

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

    draft_id = rigging.new_id()
    params = {
        # Inputs only, exactly as ``create_pixel_sheet`` says: what actually
        # ran is recorded in the draft's own sidecar recipe, so nothing here is
        # derived and a rerun copies it verbatim.
        "source_job": job_id,
        "sheet_type": kind,
        # How many of the pair to draw. Written always rather than only when
        # asked for, because absent means "whatever the path defaults to today"
        # and a stored request should say what it was admitted as.
        "candidates": wanted,
        # Normalised by the checker, not by this function: the palette name is
        # stripped and the outline defaulted there, and writing the raw
        # arguments back would put an unstripped name in a params blob the
        # worker re-resolves against the filesystem.
        **options,
        "seed_a": first,
        "seed_b": second,
        # Minted at the door so the create result can name the draft the pane
        # is about to watch for, rather than the pane having to diff a listing.
        "draft_id": draft_id,
        "base_model": SPRITE_BASE_MODEL,
    }
    source_params = source.get("params") if isinstance(source.get("params"), dict) else {}
    for identity_key in ("asset_type", "asset_intent"):
        if source_params.get(identity_key):
            params[identity_key] = source_params[identity_key]
    check_vram(svc, "sprite_synthesis", "model", params)
    # On disk *plus* every unfinished row that will land a draft here: the
    # trio is written by the worker minutes after this row is minted, so a
    # file count alone let N rapid submits all pass. Count and create under
    # one job-wide hold -- the pose cap's CON-03 rule, taken at this door.
    with svc.convert_lock(job_id, "sprite_drafts"):
        queued = sum(
            1
            for j in svc.store.active_jobs()
            if j["kind"] == "sprite_synthesis"
            and (j.get("params") or {}).get("source_job") == job_id
        )
        if len(rigging.list_sprite_drafts(job_dir)) + queued >= rigging.MAX_SPRITE_DRAFTS:
            raise Conflict(
                f"this reference already has {rigging.MAX_SPRITE_DRAFTS} sprite "
                "sheet drafts; delete one first"
            )
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
