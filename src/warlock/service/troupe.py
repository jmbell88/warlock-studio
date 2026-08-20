"""The door for Troupe: the T-pose reference block, and the character sheet.

Two entry points, because the program is deliberately two steps with a human
gate between them, and each step is a row the user can keep on its own:

* ``check_troupe`` validates the *request for a follow-up* that rides on a
  reference job, exactly as ``_jobs_create._check_sprite_sheet`` does for the
  sprite path -- and for the same reason, spelled out there: the worker mints
  the follow-up row itself, so a bad option discovered at that point would be a
  refusal an hour later on a row the user never submitted.
* ``create_charsheet`` is the direct door, for a mesh that already exists --
  a supplied base mesh, or a second sheet at a different size from the same
  character. Every refusal a character sheet has lives here rather than in the
  worker: an unrenderable request should cost the request, not a place in the
  queue and 256 EEVEE frames.

``expand_clips`` is re-exported from ``warlock.clips``: the worker needs the
same function and may not import ``service``, which is why that module exists.

The numbers this module offers come from ``pipelines.charsheet`` and
``pipelines.pixelize`` rather than being restated: those are the modules the
worker actually plans and reduces with, and a second copy here would be one
edit away from a form that offers a size the renderer refuses.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from .. import rigging
from ..clips import expand_clips
from ..pipelines import charsheet, pixelize, spritesynth
from .errors import Conflict, Invalid, NotFound, invalid_from
from .validation import check_job_id, check_vram

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import WarlockService

#: Which guide the reference stage conditions on. ``spritesynth`` owns the
#: list because it owns the templates.
TROUPE_VARIANTS: tuple[str, ...] = spritesynth.TPOSE_VARIANTS

#: The sizes a sheet may be laid out at. ``charsheet.SIZES``, which is also
#: what ``charsheet.plan`` validates against.
TROUPE_LOGICAL_SIZES: tuple[int, ...] = charsheet.SIZES

#: Palette budgets, when no designed palette is named. The same ladder the
#: sprite path offers, and for the same reason: these are the counts a median
#: cut produces a usable sprite palette at.
TROUPE_COLOR_CHOICES: tuple[int, ...] = (8, 16, 32, 64)

TROUPE_OUTLINE_MODES: tuple[str, ...] = pixelize.OUTLINE_MODES
TROUPE_REDUCE_MODES: tuple[str, ...] = pixelize.REDUCE_MODES

DEFAULT_TROUPE_VARIANT = "male"
DEFAULT_TROUPE_LOGICAL_SIZE = 32
DEFAULT_TROUPE_COLORS = 64
DEFAULT_TROUPE_OUTLINE = "outer"

#: The only rig template a Troupe sheet can be built on, because it is the only
#: one the clip library carries clips for -- and ``rigging.clip_library``
#: answers "no clips" for the rest rather than failing, so without this the
#: refusal would land in the worker as a frame-count mismatch.
TROUPE_TEMPLATE = "humanoid"

#: Pinned rather than inherited from the character's row, the rule
#: ``_maybe_queue_sprite_sheet`` states at length: the guide is a ControlNet
#: hint, so the reference stage needs a base that can run one.
TROUPE_BASE_MODEL = "sdxl_cfg"


def troupe_options(svc: WarlockService) -> dict[str, Any]:
    """What a Troupe request may ask for. One source for the form."""
    from . import palettes

    return {
        "variants": list(TROUPE_VARIANTS),
        "logical_sizes": list(TROUPE_LOGICAL_SIZES),
        "colors": list(TROUPE_COLOR_CHOICES),
        "outline_modes": list(TROUPE_OUTLINE_MODES),
        "reduce_modes": list(TROUPE_REDUCE_MODES),
        "palettes": palettes.available(svc.config),
        "animations": [
            {"name": name, "frames": frames, "loop": loop, "duration_ms": ms}
            for name, frames, loop, ms in charsheet.ANIMATIONS
        ],
        "directions": [name for name, _yaw in charsheet.DIRECTIONS],
        "cells": len(charsheet.frame_table()),
        "render_size": charsheet.RENDER_SIZE,
        "defaults": {
            "variant": DEFAULT_TROUPE_VARIANT,
            "logical_size": DEFAULT_TROUPE_LOGICAL_SIZE,
            "colors": DEFAULT_TROUPE_COLORS,
            "outline": DEFAULT_TROUPE_OUTLINE,
        },
    }


def _check_options(svc: WarlockService, entries: dict[str, Any]) -> dict[str, Any]:
    """The pixelisation options every Troupe path shares, validated once."""
    from . import palettes

    try:
        logical = int(entries.get("logical_size") or DEFAULT_TROUPE_LOGICAL_SIZE)
    except (TypeError, ValueError):
        raise Invalid("logical_size must be a whole number", field="logical_size") from None
    if logical not in TROUPE_LOGICAL_SIZES:
        raise Invalid(
            f"logical_size must be one of {list(TROUPE_LOGICAL_SIZES)}",
            field="logical_size",
        )
    try:
        colors = int(entries.get("colors") or DEFAULT_TROUPE_COLORS)
    except (TypeError, ValueError):
        raise Invalid("colors must be a whole number", field="colors") from None
    if colors not in TROUPE_COLOR_CHOICES:
        raise Invalid(
            f"colors must be one of {list(TROUPE_COLOR_CHOICES)}", field="colors"
        )
    outline = str(entries.get("outline") or DEFAULT_TROUPE_OUTLINE)
    if outline not in TROUPE_OUTLINE_MODES:
        raise Invalid(
            f"outline must be one of {list(TROUPE_OUTLINE_MODES)}", field="outline"
        )
    reduce_mode = str(entries.get("reduce_mode") or TROUPE_REDUCE_MODES[0])
    if reduce_mode not in TROUPE_REDUCE_MODES:
        raise Invalid(
            f"reduce_mode must be one of {list(TROUPE_REDUCE_MODES)}",
            field="reduce_mode",
        )
    palette = str(entries.get("palette") or "").strip()
    if palette:
        # Loaded and thrown away, so an unreadable palette costs the request
        # rather than 256 rendered frames. ``palettes.load`` raises Invalid
        # with the field already set.
        palettes.load(svc.config, palette)
    return {
        "logical_size": logical,
        "colors": colors,
        "outline": outline,
        "reduce_mode": reduce_mode,
        "dither": bool(entries.get("dither")),
        "palette": palette,
    }


def check_troupe(svc: WarlockService, block: Any) -> dict[str, Any]:
    """The Troupe follow-up's options, validated at the *reference* door.

    ``_jobs_create._check_sprite_sheet``'s shape and its argument. The chain a
    Troupe reference starts is reference -> gate -> mesh -> rig -> sheet, and
    only the first link exists when this runs; everything the later links will
    refuse that is knowable now is refused now.
    """
    entries = dict(block or {})
    variant = str(entries.get("variant") or DEFAULT_TROUPE_VARIANT)
    if variant not in TROUPE_VARIANTS:
        raise Invalid(
            f"variant must be one of {list(TROUPE_VARIANTS)}", field="variant"
        )
    options = _check_options(svc, entries)
    # The VRAM the *sheet* half needs is nothing -- EEVEE and CPU -- but the
    # mesh the gate promotes to is an ordinary image job and is admitted by
    # its own door. What is checked here is the reference stage's own base,
    # because this door pins it rather than inheriting it.
    check_vram(svc, "text", "reference", {"base_model": TROUPE_BASE_MODEL})
    return {"variant": variant, **options}


def create_charsheet(
    svc: WarlockService,
    job_id: str,
    *,
    logical_size: int | None = None,
    colors: int | None = None,
    outline: str | None = None,
    reduce_mode: str | None = None,
    dither: bool = False,
    palette: str | None = None,
    elevation: float | None = None,
    lighting: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Queue the full 256-cell character sheet for a finished, rigged mesh.

    The output is an ordinary sheet -- ``sheets/<id>.png`` plus its sidecar, in
    the *source* job's directory -- and that is the whole reason it is not a
    format of its own: "Open in Inker", the library, the exporters and the
    Aseprite writer all already read that pair. What makes it a Troupe sheet is
    the frame table it was laid out on and the ``animation`` block in the
    sidecar, both of which ``pipelines.charsheet`` owns.
    """
    from ..pipelines import sheet as sheetlib

    check_job_id(job_id)
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if source["status"] != "done" or not (job_dir / "model.glb").exists():
        raise Invalid("job has no finished mesh to render")
    if not (job_dir / "rig.glb").exists():
        # Every Troupe cell is a posed frame, so an unrigged mesh would render
        # 256 copies of one T-pose. Named as the missing step rather than as a
        # layout failure, because rigging it is what the user has to do next.
        raise Invalid("a character sheet needs a rigged mesh")

    rig_meta = rigging.read_rig(job_dir) or {}
    template = str(rig_meta.get("template") or "")
    if template != TROUPE_TEMPLATE:
        raise Invalid(
            f"a character sheet is animated from the {TROUPE_TEMPLATE} clip "
            f"library, and this mesh is rigged as {template or 'something else'}"
        )

    options = _check_options(
        svc,
        {
            "logical_size": logical_size,
            "colors": colors,
            "outline": outline,
            "reduce_mode": reduce_mode,
            "dither": dither,
            "palette": palette,
        },
    )

    try:
        # Expanded and thrown away, exactly as ``create_sheet`` plans and
        # throws away: a clip library that does not fill the frame table, or a
        # size whose atlas is over the texture limit, is refused now instead of
        # failing a job that has already rendered.
        records = expand_clips(TROUPE_TEMPLATE)
        charsheet.plan(
            records,
            frame_size=options["logical_size"],
            elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            lighting=lighting or "flat",
        )
    except KeyError as exc:
        raise Invalid(f"the {TROUPE_TEMPLATE} clip library is missing {exc}") from exc
    except ValueError as exc:
        raise invalid_from(exc, "That character sheet cannot be laid out") from exc

    sheet_name = (name or "").strip()
    if len(sheet_name) > rigging.MAX_SHEET_NAME:
        raise Invalid(
            f"sheet name must be at most {rigging.MAX_SHEET_NAME} characters", field="name"
        )

    params = {
        "source_job": job_id,
        "sheet_id": rigging.new_id(),
        "template": TROUPE_TEMPLATE,
        "elevation": sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
        "lighting": lighting or "flat",
        "name": sheet_name,
        **options,
    }
    # The sheet cap, counted the way ``create_sheet`` counts it and under the
    # same job-wide hold: the artifact lands minutes after the row is minted,
    # so counting files alone lets N rapid submits all read the same count.
    with svc.convert_lock(job_id, "sheets"):
        queued = sum(
            1
            for j in svc.store.active_jobs()
            if j["kind"] in ("sheet", "charsheet")
            and (j.get("params") or {}).get("source_job") == job_id
        )
        if len(rigging.list_sheets(job_dir)) + queued >= rigging.MAX_SHEETS:
            raise Conflict(f"a job may hold at most {rigging.MAX_SHEETS} sheets")
        new_id = svc.store.create(
            "charsheet", source["prompt"], params, uuid.uuid4().hex[:12]
        )
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "sheet_id": params["sheet_id"]}


def get_charsheet(svc: WarlockService, job_id: str, sheet_id: str) -> dict[str, Any]:
    """One rendered character sheet's sidecar.

    Delegated to ``sheets.get_sheet`` -- a Troupe sheet *is* a sheet -- and
    kept here as a name so a Troupe pane never has to know that.
    """
    from . import sheets

    record = sheets.get_sheet(svc, job_id, sheet_id)
    if not (record.get("animation") or {}).get("tags"):
        raise NotFound("that sheet is not a character sheet")
    return record
