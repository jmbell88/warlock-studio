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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .. import rigging
from ..clips import expand_clips
from ..pipelines import charsheet, pixelize, spritesynth
from .errors import Invalid, NotFound, invalid_from
from .sheets import check_sheet_cap
from .validation import check_job_id, check_vram

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import WarlockService

#: Which guide the reference stage conditions on. ``spritesynth`` owns the
#: list because it owns the templates.
TROUPE_VARIANTS: tuple[str, ...] = spritesynth.TPOSE_VARIANTS

#: And in which pose. The second axis of the same choice, crossed with the
#: variant rather than folded into it -- see ``spritesynth.REFERENCE_POSES``.
TROUPE_POSES: tuple[str, ...] = spritesynth.REFERENCE_POSES

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

#: **A-pose, chosen 2026-08-23 over the T-pose that shipped before it.** The
#: shipped humanoid rig template is itself an A-pose, so this is the pose whose
#: mesh the template fits directly -- no joints measured off its vertices, and
#: so no dependency on the ViTPose weights a bare install does not have. The
#: T-pose is still on offer, and still the better reconstruction: it separates
#: the limbs more, which is the one thing a single view most needs.
DEFAULT_TROUPE_POSE = "apose"
DEFAULT_TROUPE_LOGICAL_SIZE = 32
DEFAULT_TROUPE_COLORS = 64
DEFAULT_TROUPE_OUTLINE = "outer"

#: The only rig template a Troupe sheet can be built on, because it is the only
#: one the clip library carries clips for -- and ``rigging.clip_library``
#: answers "no clips" for the rest rather than failing, so without this the
#: refusal would land in the worker as a frame-count mismatch.
#: How many ``charsheet`` rows deep the settings lookup looks. The library's
#: own page size, and ``studio.troupe_mode``'s: a sheet older than this is one
#: whose settings the door reports as no longer on record, by name.
_SCAN_LIMIT = 400

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
        "poses": list(TROUPE_POSES),
        "logical_sizes": list(TROUPE_LOGICAL_SIZES),
        "colors": list(TROUPE_COLOR_CHOICES),
        "outline_modes": list(TROUPE_OUTLINE_MODES),
        "reduce_modes": list(TROUPE_REDUCE_MODES),
        "palettes": palettes.available(svc.config),
        "animations": [
            {
                "name": name,
                "frames": frames,
                "min_frames": charsheet.MOVEMENT_MIN_FRAMES[name],
                "max_frames": charsheet.MAX_FRAMES,
                "loop": loop,
                "duration_ms": ms,
            }
            for name, frames, loop, ms in charsheet.ANIMATIONS
        ],
        "directions": [name for name, _yaw in charsheet.DIRECTIONS],
        "direction_presets": list(charsheet.DIRECTION_PRESETS),
        "cells": len(charsheet.frame_table()),
        "warn_cells": charsheet.WARN_CELLS,
        "max_cells": charsheet.MAX_CELLS,
        "render_size": charsheet.RENDER_SIZE,
        "defaults": {
            "variant": DEFAULT_TROUPE_VARIANT,
            "pose": DEFAULT_TROUPE_POSE,
            "logical_size": DEFAULT_TROUPE_LOGICAL_SIZE,
            "colors": DEFAULT_TROUPE_COLORS,
            "outline": DEFAULT_TROUPE_OUTLINE,
            "reduce_mode": TROUPE_REDUCE_MODES[0],
            "layout": charsheet.resolve_layout().as_dict(),
        },
    }


def _check_options(svc: WarlockService, entries: dict[str, Any]) -> dict[str, Any]:
    """The pixelisation options every Troupe path shares, validated once.

    A thin wrapper over ``pixelopts.check_pixel_options`` since 2026-08-29:
    the body was the same four refusals every other pixel path needs, and the
    only Troupe-shaped things in it were the two ladders and the default
    outline, which are the parameters. Kept as a name here rather than having
    the three doors below call the shared function directly, because *this* is
    where the Troupe defaults live and a door should not have to restate them
    -- the delegation rule this module's docstring states, applied to itself.
    """
    from .pixelopts import check_pixel_options

    return check_pixel_options(
        svc,
        entries,
        sizes=TROUPE_LOGICAL_SIZES,
        size_default=DEFAULT_TROUPE_LOGICAL_SIZE,
        colors=TROUPE_COLOR_CHOICES,
        colors_default=DEFAULT_TROUPE_COLORS,
        outline_default=DEFAULT_TROUPE_OUTLINE,
    )


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
    pose = str(entries.get("pose") or DEFAULT_TROUPE_POSE)
    if pose not in TROUPE_POSES:
        raise Invalid(f"pose must be one of {list(TROUPE_POSES)}", field="pose")
    options = _check_options(svc, entries)
    try:
        layout = charsheet.resolve_layout(entries.get("layout"))
    except (TypeError, ValueError) as exc:
        raise Invalid(str(exc), field="layout") from exc
    # The VRAM the *sheet* half needs is nothing -- EEVEE and CPU -- but the
    # mesh the gate promotes to is an ordinary image job and is admitted by
    # its own door. What is checked here is the reference stage's own base,
    # because this door pins it rather than inheriting it.
    check_vram(svc, "text", "reference", {"base_model": TROUPE_BASE_MODEL})
    return {"variant": variant, "pose": pose, "layout": layout.as_dict(), **options}


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
    layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Queue a configured character sheet for a finished, rigged mesh.

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
        resolved_layout = charsheet.resolve_layout(layout)
        # Expanded and thrown away, exactly as ``create_sheet`` plans and
        # throws away: a clip library that does not fill the frame table, or a
        # size whose atlas is over the texture limit, is refused now instead of
        # failing a job that has already rendered.
        records = expand_clips(TROUPE_TEMPLATE, resolved_layout)
        charsheet.plan(
            records,
            frame_size=options["logical_size"],
            elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            lighting=lighting or "flat",
            layout=resolved_layout,
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
        "layout": resolved_layout.as_dict(),
        **options,
    }
    # The sheet cap, counted the way ``create_sheet`` counts it and under the
    # same job-wide hold: the artifact lands minutes after the row is minted,
    # so counting files alone lets N rapid submits all read the same count.
    with svc.convert_lock(job_id, "sheets"):
        check_sheet_cap(svc, job_id, job_dir)
        new_id = svc.store.create(
            "charsheet", source["prompt"], params, uuid.uuid4().hex[:12]
        )
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "sheet_id": params["sheet_id"]}


def rerender_charsheet(
    svc: WarlockService,
    job_id: str,
    *,
    sheet_id: str,
    subset: Sequence[Mapping[str, Any]],
    name: str | None = None,
) -> dict[str, Any]:
    """Re-render some of a sheet's runs, copying the rest from the sheet itself.

    **It takes no pixel options, and that is the design.** The settings are
    copied verbatim from the row that produced ``sheet_id``, because the new
    cells have to be reduced, quantised and outlined exactly as the ones they
    will sit beside were -- and any option this door accepted separately would
    be an option a user could set to something else. That turns a whole class
    of "these twelve cells look wrong" into a lookup.

    The output is a **new sheet**, not a rewrite of the old one: sheets are
    write-once under a fresh id, the old one stays openable, and Inker's merge
    is what brings the two together over a document that has hand edits on it.

    -> ``{"id", "source_job", "sheet_id", "runs"}``
    """
    check_job_id(job_id)
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if not rigging.is_valid_id(str(sheet_id or "")):
        raise Invalid("that is not a sheet id", field="sheet_id")
    record = rigging.read_sheet(job_dir, str(sheet_id))
    if not record:
        raise NotFound("that sheet is no longer on disk", field="sheet_id")
    snapshot = record.get("troupe")
    if not isinstance(snapshot, Mapping):
        raise Invalid(
            "that is not a character sheet, so it has no runs to re-render",
            field="sheet_id",
        )

    row = _charsheet_row(svc, job_id, str(sheet_id))
    if row is None:
        # Honest, and it names what to do instead. The settings are the whole
        # point of this door; without them the new cells could not be made to
        # match the ones they are landing beside.
        raise Invalid(
            "the settings that produced that sheet are no longer on record, so it "
            "cannot be re-rendered a run at a time -- build a new sheet instead",
            field="sheet_id",
        )

    try:
        resolved_layout = charsheet.resolve_layout(snapshot)
        runs = charsheet.check_subset(subset, resolved_layout)
    except ValueError as exc:
        raise invalid_from(exc, "Those runs cannot be re-rendered", field="subset") from exc

    sheet_name = (name or "").strip()
    if len(sheet_name) > rigging.MAX_SHEET_NAME:
        raise Invalid(
            f"sheet name must be at most {rigging.MAX_SHEET_NAME} characters", field="name"
        )

    params = dict(row.get("params") or {})
    params.update(
        {
            "source_job": job_id,
            "sheet_id": rigging.new_id(),
            "base_sheet": str(sheet_id),
            "subset": [{"animation": a, "direction": d} for a, d in runs],
            "layout": resolved_layout.as_dict(),
            "name": sheet_name or str(params.get("name") or ""),
        }
    )
    # Not inherited: they are the *previous* run's answers about its own output
    # and a fresh row must not wear them. ``DERIVED_PARAMS`` says the same thing
    # for a rerun; this door mints a new row, so it strips them itself.
    for derived in ("cells", "rendered_cells", "pixel_report"):
        params.pop(derived, None)

    # A re-render is a new sheet and draws on the same pool -- ``create_charsheet``'s
    # arrangement verbatim, under the same job-wide hold.
    with svc.convert_lock(job_id, "sheets"):
        check_sheet_cap(svc, job_id, job_dir)
        new_id = svc.store.create(
            "charsheet", source["prompt"], params, uuid.uuid4().hex[:12]
        )
    svc.wake_worker()
    return {
        "id": new_id,
        "source_job": job_id,
        "sheet_id": params["sheet_id"],
        "runs": [{"animation": a, "direction": d} for a, d in runs],
    }


def _charsheet_row(
    svc: WarlockService, job_id: str, sheet_id: str
) -> dict[str, Any] | None:
    """The ``charsheet`` row that produced one sheet, or None.

    Narrowed by ``kind`` in SQL rather than walked unfiltered: the answer is one
    row and the page this searches is the same one the mode's own sheet list is
    built from.
    """
    for row in svc.store.list(limit=_SCAN_LIMIT, kind="charsheet"):
        params = row.get("params") or {}
        if str(params.get("source_job") or "") != job_id:
            continue
        if str(params.get("sheet_id") or "") == sheet_id:
            return row
    return None


def send_to_troupe(
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
    layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Take a mesh the user already has into Troupe, rigging it first if needed.

    The third door, and the one the *library* uses. ``create_charsheet`` above
    is the direct one and refuses an unrigged mesh by design -- every Troupe
    cell is a posed frame -- which left the common case ("I have a character;
    make me a sheet") with no route at all: the user had to know to rig it
    first, with a humanoid template, from a different pane.

    Two shapes, one press:

    * **Already rigged** -- delegate to ``create_charsheet`` verbatim. One row,
      the existing path, including the humanoid refusal and the sheet cap.
    * **Not rigged** -- mint a rig row carrying a nested ``troupe_sheet``
      block, and let the worker mint the sheet on the finished rig
      (``_maybe_queue_sheet_after_rig``). That keeps the "four ordinary jobs,
      not an orchestrator" property: one press mints one row, the second is
      minted by the same mechanism that already mints rigs and sheets, and
      both cancel independently.

    **Everything knowable is refused here**, before either row exists --
    the options, the layout, the clip expansion and the plan -- because an
    unrenderable request should cost the request and not a rig plus 256 EEVEE
    frames. That is ``create_charsheet``'s own argument, applied one link
    earlier.

    **The marker is ``troupe_sheet`` and not ``troupe``.** They are different
    claims: ``troupe`` on a reference means "run the whole chain, human gate
    included", and this means "render this sheet once the rig lands". It is
    also *nested*, so ``VECTOR_PARAMS`` -- an allowlist of flat settings --
    cannot pick it up.

    **The mesh row is not stamped.** Marking it would work, and it would
    change what a reroll means: ``rerun_job``/``promote_to_model`` copy
    everything that is not derived, so the next "Remesh" would silently spend
    a rig and 256 rendered cells nobody asked for.

    The rig template is pinned to ``humanoid`` rather than taken from the
    user's Rig-stage preference, because the clip library this animates from
    is humanoid; ``joints="measured"`` for the reason ``_jobs_create`` gives
    where it sets the same flag -- the shipped template is an A-pose and
    mis-fits a T-posed mesh badly enough to skin the arms to the chest.
    """
    check_job_id(job_id)
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if source["status"] != "done" or not (job_dir / "model.glb").exists():
        # ``create_charsheet``'s sentence, verbatim: one refusal, one wording.
        raise Invalid("job has no finished mesh to render")

    if (job_dir / "rig.glb").exists():
        return create_charsheet(
            svc,
            job_id,
            logical_size=logical_size,
            colors=colors,
            outline=outline,
            reduce_mode=reduce_mode,
            dither=dither,
            palette=palette,
            elevation=elevation,
            lighting=lighting,
            name=name,
            layout=layout,
        )

    spec = _charsheet_spec(
        svc,
        logical_size=logical_size,
        colors=colors,
        outline=outline,
        reduce_mode=reduce_mode,
        dither=dither,
        palette=palette,
        elevation=elevation,
        lighting=lighting,
        name=name,
        layout=layout,
    )
    from .. import doctor

    if not doctor.blender_check().ok:
        # The Rig segment's own sentence, verbatim, so the app has one wording
        # for "this needs Blender" wherever it is met. Checked through the
        # same probe ``rig_templates`` answers the pane with, rather than a
        # second test of the same thing.
        raise Invalid("Rigging needs Blender, which is not installed.")

    params = {
        "source_job": job_id,
        # Pinned rather than taken from ``config.rig_template``: the sheet is
        # animated from the humanoid clip library, so a quadruped rig here
        # would produce a rig that the sheet then refuses.
        "template": TROUPE_TEMPLATE,
        "auto": True,
        "joints": "measured",
        "troupe_sheet": spec,
    }
    # The rig row *is* a sheet reservation -- ``_maybe_queue_sheet_after_rig``
    # mints the charsheet from it with no door in between -- so the cap is
    # taken here, under the same hold the direct door takes it.
    with svc.convert_lock(job_id, "sheets"):
        check_sheet_cap(svc, job_id, job_dir)
        new_id = svc.store.create("rig", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "rigged": False}


def _charsheet_spec(
    svc: WarlockService,
    *,
    logical_size: int | None,
    colors: int | None,
    outline: str | None,
    reduce_mode: str | None,
    dither: bool,
    palette: str | None,
    elevation: float | None,
    lighting: str | None,
    name: str | None,
    layout: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate a sheet request and freeze it as the params the worker will use.

    Everything ``create_charsheet`` refuses that does not depend on the rig,
    checked here and then *snapshotted*: the row the worker mints later must
    describe the settings that were on screen when the button was pressed, not
    whatever the pane holds by the time the rig finishes.
    """
    from ..pipelines import sheet as sheetlib

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
        resolved_layout = charsheet.resolve_layout(layout)
        records = expand_clips(TROUPE_TEMPLATE, resolved_layout)
        charsheet.plan(
            records,
            frame_size=options["logical_size"],
            elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            lighting=lighting or "flat",
            layout=resolved_layout,
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
    return {
        "template": TROUPE_TEMPLATE,
        "elevation": sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
        "lighting": lighting or "flat",
        "name": sheet_name,
        "layout": resolved_layout.as_dict(),
        **options,
    }

# ``get_charsheet`` was deleted on 2026-08-22. It delegated to
# ``sheets.get_sheet`` so a Troupe pane would not have to know that a Troupe
# sheet *is* a sheet -- and all three of its would-be callers
# (``packwright_mode``, ``inker_mode``, ``panes.sheet_panel``) called
# ``sheets.get_sheet`` directly anyway, which is the thing it existed to spare
# them. A wrapper nobody reaches for is a second answer to drift from.
