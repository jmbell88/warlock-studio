"""Sprite sheets: what one may ask for, and queuing a render."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .. import models, rigging
from .core import WarlockService
from .errors import Conflict, Invalid, NotFound, invalid_from
from .validation import (
    check_base_model_weights,
    check_job_id,
    check_pose_id,
    check_seed,
    check_sheet_id,
    check_vram,
    install_remedy,
    random_seed,
)

# The base a pixel-sheet restyle is pinned to, hoisted out of
# ``create_pixel_sheet`` so the row list below cannot name a different one. Not
# ``models.DEFAULT_BASE_MODEL``: the restyle wants full CFG and a ControlNet
# specifically, and it has to keep wanting them if the default ever moves again.
PIXEL_SHEET_BASE_MODEL = "sdxl_cfg"

# Every registry row a pixel sheet needs on this host, in ``fetch.Entry``'s
# ``row_key`` spelling. Exported so a pane can offer "install what this needs"
# without knowing what a pixel sheet is made of, and derived from the two
# constants the refusals below check so the two cannot drift.
PIXEL_SHEET_ROWS: tuple[str, ...] = (
    f"base:{PIXEL_SHEET_BASE_MODEL}",
    f"lora:{models.PIXEL_SHEET_LORA}",
)


def sheet_options() -> dict[str, Any]:
    """What a sheet request may ask for. One source for the form, as with the
    rig templates -- the UI never hardcodes a frame size."""
    from ..pipelines import sheet as sheetlib

    return {
        "frame_sizes": list(sheetlib.FRAME_SIZES),
        "lighting": list(sheetlib.LIGHTING),
        "yaws": list(sheetlib.yaw_angles()),
        "defaults": {
            "frame_size": sheetlib.DEFAULT_FRAME_SIZE,
            "elevation": sheetlib.DEFAULT_ELEVATION,
            "lighting": "flat",
        },
    }


def list_sheets(svc: WarlockService, job_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    return {"sheets": rigging.list_sheets(svc.job_dir(job_id))}



def queued_sheets(svc: WarlockService, job_id: str) -> int:
    """Rows not yet finished that will each end as one sheet of *job_id*.

    Three doors mint them and they draw on one ``MAX_SHEETS`` pool:
    ``create_sheet`` (a ``sheet`` row), ``troupe.create_charsheet`` (a
    ``charsheet`` row) and ``troupe.send_to_troupe`` on an unrigged mesh (a
    ``rig`` row carrying ``troupe_sheet``, which ``_maybe_queue_sheet_after_rig``
    turns into the charsheet). Each door used to count a different subset of
    the other two, so the pool could be reserved one past the cap through
    whichever door was not counting -- one function, the same count at all
    three.
    """
    total = 0
    for j in svc.store.active_jobs():
        params = j.get("params") or {}
        if params.get("source_job") != job_id:
            continue
        kind = j["kind"]
        if kind in ("sheet", "charsheet") or (kind == "rig" and params.get("troupe_sheet")):
            total += 1
    return total


def check_sheet_cap(svc: WarlockService, job_id: str, job_dir: Path) -> None:
    """Refuse the sheet that would take *job_id* past ``MAX_SHEETS``.

    Called under ``svc.convert_lock(job_id, "sheets")`` by every door that
    reserves a slot: the artifact lands minutes after the row is minted, so
    counting files alone let N rapid submits all read the same count.
    """
    if len(rigging.list_sheets(job_dir)) + queued_sheets(svc, job_id) >= rigging.MAX_SHEETS:
        raise Conflict(f"a job may hold at most {rigging.MAX_SHEETS} sheets")

def create_sheet(
    svc: WarlockService,
    job_id: str,
    *,
    poses: list[str] | None = None,
    elevation: float | None = None,
    frame_size: int | None = None,
    lighting: str | None = None,
    name: str | None = None,
    clip_from: str | None = None,
    clip_to: str | None = None,
    clip_frames: int = 8,
    yaws: int | None = None,
) -> dict[str, Any]:
    """Queue a pose x direction sprite sheet for a finished job's mesh.

    Every reason this can be refused is checked here rather than in the worker:
    an unrenderable request should cost the request, not a place in the queue
    and two minutes of EEVEE.
    """
    from ..pipelines import sheet as sheetlib

    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    if source["status"] != "done" or not (job_dir / "model.glb").exists():
        raise Invalid("job has no finished mesh to render")

    pose_ids = [p for p in (poses or []) if p]
    records = []
    for pose_id in pose_ids:
        check_pose_id(pose_id)
        record = rigging.read_pose(job_dir, pose_id)
        if record is None:
            raise NotFound(f"no such pose {pose_id}")
        records.append(record)
    if records and not (job_dir / "rig.glb").exists():
        raise Invalid("posed sheets need a rigged mesh")

    # A clip replaces the pose rows rather than adding to them: its rows *are*
    # the animation, and mixing static poses into the same sheet would give an
    # importer no way to tell which rows loop.
    if clip_from or clip_to:
        if not (clip_from and clip_to):
            # The empty one, not the pair: a refusal names the control it came
            # from, and highlighting the end the user *did* fill in would send
            # them to change the wrong select.
            raise Invalid(
                "a clip needs both clip_from and clip_to",
                field="clip_to" if clip_from else "clip_from",
            )
        for pose_id in (clip_from, clip_to):
            check_pose_id(pose_id)
        ends = [rigging.read_pose(job_dir, pid) for pid in (clip_from, clip_to)]
        if any(e is None for e in ends):
            raise NotFound("no such pose")
        if not (job_dir / "rig.glb").exists():
            raise Invalid("an animated clip needs a rigged mesh")
        try:
            records = sheetlib.interpolate(ends[0], ends[1], clip_frames)
        except ValueError as exc:
            # One refusal left, and it is the frame slider's: ``interpolate``
            # used to also refuse an endpoint carrying a root offset, and this
            # door had to work out which of three controls to ring. It now
            # interpolates the offset instead, so the only way to get here is a
            # frame count outside 1..MAX_CLIP_FRAMES.
            raise invalid_from(
                exc, "That clip cannot be built", field="clip_frames"
            ) from exc

    try:
        # Built and thrown away: the worker plans it again from the same
        # inputs. This call is here purely so a bad frame size or an atlas over
        # the texture limit is refused now instead of failing a job later.
        sheetlib.plan(
            records,
            frame_size=frame_size or sheetlib.DEFAULT_FRAME_SIZE,
            elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            lighting=lighting or "flat",
            yaws=yaws or sheetlib.DEFAULT_YAWS,
        )
    except ValueError as exc:
        raise invalid_from(exc, "That sprite sheet cannot be laid out") from exc

    sheet_name = (name or "").strip()
    if len(sheet_name) > rigging.MAX_SHEET_NAME:
        raise Invalid(
            f"sheet name must be at most {rigging.MAX_SHEET_NAME} characters", field="name"
        )

    params = {
        "source_job": job_id,
        "sheet_id": rigging.new_id(),
        "poses": pose_ids,
        "elevation": sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
        "frame_size": frame_size or sheetlib.DEFAULT_FRAME_SIZE,
        "lighting": lighting or "flat",
        "name": sheet_name,
        "yaws": yaws or sheetlib.DEFAULT_YAWS,
        # The two ends, not the expanded frames: the host is the single place a
        # clip is decided, and storing the frames would be a second copy that
        # could disagree with sheet.interpolate.
        "clip": ({"from": clip_from, "to": clip_to, "frames": clip_frames} if clip_from else None),
    }
    # The cap counts what is on disk *plus* every unfinished row that will
    # write a sheet into this directory: the artifact lands minutes after the
    # row is minted, so counting files alone let N rapid submits all read the
    # same count and all pass. Count and create under one job-wide hold --
    # the pose cap's CON-03 rule, taken at this door.
    with svc.convert_lock(job_id, "sheets"):
        check_sheet_cap(svc, job_id, job_dir)
        new_id = svc.store.create("sheet", source["prompt"], params, uuid.uuid4().hex[:12])
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "sheet_id": params["sheet_id"]}


def get_sheet(svc: WarlockService, job_id: str, sheet_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    check_sheet_id(sheet_id)
    record = rigging.read_sheet(svc.job_dir(job_id), sheet_id)
    if record is None:
        raise NotFound("no such sheet")
    return record


def sheet_png(svc: WarlockService, job_id: str, sheet_id: str) -> Path:
    check_job_id(job_id)
    check_sheet_id(sheet_id)
    job_dir = svc.job_dir(job_id)
    path = rigging.sheet_png_path(job_dir, sheet_id)
    # The sidecar is the completion marker (the worker writes the PNG first),
    # so PNG existence alone can serve a partial file mid-save.
    if not path.exists() or not rigging.sheet_path(job_dir, sheet_id).exists():
        raise NotFound("no such sheet")
    return path


def _restyle_in_flight(svc: WarlockService, sheet_id: str) -> bool:
    """Is a ``pixel_sheet`` row for *sheet_id* still queued or running.

    The 2026-09-06 audit (create2-02): unlike ``create_sheet``, this door used
    to take no lock and check nothing, so deleting a sheet while its own
    restyle job was mid-flight let the job's later ``os.replace`` resurrect
    the pixel PNG/sidecar pair the confirmed delete had just removed -- an
    orphan ``list_sheets`` would never show again, because it keys off the
    base sidecar this function's caller is about to unlink.
    """
    for j in svc.store.active_jobs():
        if j["kind"] != "pixel_sheet":
            continue
        if (j.get("params") or {}).get("sheet_id") == sheet_id:
            return True
    return False


def delete_sheet(svc: WarlockService, job_id: str, sheet_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    check_sheet_id(sheet_id)
    # Same per-asset hold ``create_sheet`` takes, so a restyle cannot be
    # queued in the gap between the check above and the unlink below.
    with svc.convert_lock(job_id, "sheets"):
        if _restyle_in_flight(svc, sheet_id):
            raise Conflict(
                "this sheet's pixel restyle is still running; wait for it to"
                " finish before deleting the sheet",
                field="sheet_id",
            )
        if not rigging.delete_sheet(svc.job_dir(job_id), sheet_id):
            raise NotFound("no such sheet")
    return {"ok": True}


# What a pixel restyle may be reduced to. Its own tuple rather than the 2D
# export's PIXEL_SIZES: those are artifact *names* in an allowlist, while these
# are logical cell sizes that have to divide a sheet's frame size.
# 128 is Troupe's top rung: the spec commits to 16px minimum and 128px maximum,
# and a size the sheet spec offers but this tuple refuses would be a form that
# cannot ask for the sheet it just rendered. ``check_restylable`` still refuses
# a restyle whose atlas would exceed 1024px across, so adding it here widens the
# ladder without widening what the AI path will accept.
PIXEL_LOGICAL_SIZES = (16, 24, 32, 48, 64, 128)
PIXEL_COLOR_CHOICES = (8, 16, 32, 64)


# ``"none"``, and not the sprite path's ``inner``. Not a taste difference: the
# default restyle has to keep producing the bytes ``PIXEL_SHEET_VERSION``
# already claims for every pixel sheet on disk, and an outline is a pixel this
# path did not draw before. Somebody who wants one asks for one.
DEFAULT_PIXEL_SHEET_OUTLINE = "none"


def _check_options(svc: WarlockService, entries: dict[str, Any]) -> dict[str, Any]:
    """The restyle's pixelisation options, validated once for its one door.

    ``troupe._check_options``' shape: the shared refusals from ``pixelopts``,
    with this path's two ladders and its ``none`` default as the parameters.

    ``allow_reduce_mode=False`` because this path has no reduce mode to offer.
    The reduction is ``pixelsheet.downscale`` -- an integer NEAREST stride,
    fixed by the requirement that a cell boundary stay on an output pixel
    boundary -- and by the time the worker's new branch runs, the atlas is
    already at the logical size. Offering the choice would be the dead field
    ``check_pixel_options`` refuses to create.
    """
    from .pixelopts import check_pixel_options

    return check_pixel_options(
        svc,
        entries,
        sizes=PIXEL_LOGICAL_SIZES,
        size_default=32,
        colors=PIXEL_COLOR_CHOICES,
        colors_default=32,
        outline_default=DEFAULT_PIXEL_SHEET_OUTLINE,
        allow_reduce_mode=False,
    )


def pixel_sheet_options() -> dict[str, Any]:
    """What a restyle request may ask for, from one source, as with the render
    above -- the pane never hardcodes a size or a strength range."""
    from ..pipelines import pixelize, pixelsheet

    return {
        "logical_sizes": list(PIXEL_LOGICAL_SIZES),
        "colors": list(PIXEL_COLOR_CHOICES),
        "outlines": list(pixelize.OUTLINE_MODES),
        "strength_range": [
            models.IMG2IMG_STRENGTH_MIN,
            models.IMG2IMG_STRENGTH_MAX,
        ],
        "defaults": {
            "logical_size": 32,
            "colors": 32,
            "outline": DEFAULT_PIXEL_SHEET_OUTLINE,
            "dither": False,
            "palette": "",
            "strength": models.DEFAULT_IMG2IMG_STRENGTH,
            "structure_lock": True,
        },
        "max_width": pixelsheet.BAND_PX,
    }


def _check_weights(svc: WarlockService) -> None:
    """Presence of the two weights a pixel-sheet restyle cannot run without.

    Presence as well as fit: the family check at the door only asks whether
    base and LoRA *belong* together; it says nothing about whether either is on
    this host. Missing, the worker's own tolerance takes over -- it logs and
    restyles bare -- so the job would finish, look like an ordinary img2img
    pass rather than pixel art, and write a sidecar naming a LoRA that never
    loaded. The ControlNet is deliberately not checked: ``structure_lock`` is
    a preference the worker already degrades gracefully, and refusing the
    whole request over an optional hint would be the wrong trade.

    Shared by ``create_pixel_sheet``'s door and ``rerun_job``'s re-admission,
    like ``tilesheets._check_weights`` and ``sprites._check_weights``: a reroll
    can be days later, against a models directory the user has since pruned.
    """
    from .. import fetch
    from .downloads import needed_keys

    base = models.BASE_MODELS[PIXEL_SHEET_BASE_MODEL]
    pixel_lora = models.STYLE_LORAS[models.PIXEL_SHEET_LORA]
    check_base_model_weights(svc, base, rows=needed_keys(svc, PIXEL_SHEET_ROWS))
    if not fetch.present(svc.config, "lora", pixel_lora):
        remedy = fetch.download_text(svc.config, "lora", pixel_lora)
        raise Invalid(
            f"A pixel sheet needs {pixel_lora.label!r}, which is not downloaded. "
            f"{install_remedy(pixel_lora.label, remedy)}",
            # ``style_lora``, like both siblings: ``PIXEL_SHEET_BASE_MODEL`` is
            # a pinned constant with no form control at all, so a refusal
            # naming it leaves ``main``'s field-driven error ring with nothing
            # to attach to and the user gets a bare toast -- exactly what
            # ``ServiceError.field`` exists to prevent.
            field="style_lora",
            rows=needed_keys(svc, PIXEL_SHEET_ROWS),
        )


def create_pixel_sheet(
    svc: WarlockService,
    job_id: str,
    sheet_id: str,
    *,
    logical_size: int = 32,
    colors: int = 32,
    palette: str | None = None,
    dither: bool = False,
    outline: str | None = None,
    strength: float | None = None,
    seed: int | None = None,
    structure_lock: bool = True,
) -> dict[str, Any]:
    """Queue a pixel-art restyle of one finished sprite sheet.

    Every refusal is here rather than in the worker, exactly as ``create_sheet``
    states: a sheet too wide to denoise in one frame, or a logical size that
    does not divide its cells, costs the request rather than a place in the
    queue and a minute of GPU.
    """
    from ..pipelines import pixelsheet

    check_job_id(job_id)
    check_sheet_id(sheet_id)
    source = svc.require_job(job_id)
    job_dir = svc.job_dir(job_id)
    meta = rigging.read_sheet(job_dir, sheet_id)
    if meta is None or not rigging.sheet_png_path(job_dir, sheet_id).exists():
        raise NotFound("no such sheet")

    # Through the shared checker, in the same sentences on the same fields as
    # every other pixel path -- ``pixelopts``' whole reason. Two of its options
    # are new here and one is deliberately absent: this path has no reduce mode
    # to offer, because the reduction is ``pixelsheet.downscale``'s integer
    # NEAREST stride and the worker's new branch leaves it already done.
    # ``outline_default="none"`` and not the sprite path's ``inner``: the
    # default request has to keep producing the bytes every restyled sheet
    # already on disk claims.
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
    logical_size = options["logical_size"]
    try:
        pixelsheet.check_restylable(meta, int(logical_size))
    except pixelsheet.NotRestylable as exc:
        raise invalid_from(
            exc, "This sheet cannot be restyled as pixel art", field="logical_size"
        ) from exc

    value = models.DEFAULT_IMG2IMG_STRENGTH if strength is None else float(strength)
    if not models.IMG2IMG_STRENGTH_MIN <= value <= models.IMG2IMG_STRENGTH_MAX:
        raise Invalid(
            f"strength must be between {models.IMG2IMG_STRENGTH_MIN} "
            f"and {models.IMG2IMG_STRENGTH_MAX}",
            field="strength",
        )
    if seed is not None:
        check_seed("seed", seed)

    # The restyle's identity is the pixel-art LoRA, so a base that adapter
    # does not fit is refused here rather than queued: the worker would drop
    # the style and ship a sheet that merely got smaller, which is a minute of
    # GPU for a result the request did not mean. The base below is this
    # function's own constant today, which makes this read like a guard on a
    # pair of constants -- it is, and deliberately: params outlive today's UI,
    # and the day base_model becomes a parameter this refusal is already at
    # the door. Worded as guidance.normalize words the same pairing refusal,
    # field naming the control the sentence mentions.
    base_key = PIXEL_SHEET_BASE_MODEL
    base = models.BASE_MODELS[base_key]
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

    params = {
        # Every one of these is an *input*: the restyle's own recipe is
        # recorded in the pixel sidecar, not here, so nothing in this dict is
        # derived. A rerun copies it, with one carve-out: ``sheet_id`` is on
        # DERIVED_PARAMS for the *sheet* render kind's sake, so ``rerun_job``
        # re-seeds it from the source row rather than letting the strip cost
        # this kind its input.
        "source_job": job_id,
        "sheet_id": sheet_id,
        # Normalised by the checker rather than by this function: the palette
        # name is stripped there, and the worker re-resolves it against the
        # filesystem exactly as written.
        **options,
        "strength": value,
        "structure_lock": bool(structure_lock),
        "seed": random_seed() if seed is None else int(seed),
        "base_model": base_key,
    }
    # At the door, exactly as create_job does, and before the row exists: an
    # img2img restyle wants SDXL plus a ControlNet, which is the shape of
    # request a smaller card has to refuse. The presence half lives in
    # ``_check_weights`` so ``rerun_job`` can hold the same door on the way
    # back in.
    _check_weights(svc)
    check_vram(svc, "pixel_sheet", "model", params)
    new_id = svc.store.create(
        "pixel_sheet", source["prompt"], params, uuid.uuid4().hex[:12]
    )
    svc.wake_worker()
    return {"id": new_id, "source_job": job_id, "sheet_id": sheet_id}


def get_pixel_sheet(svc: WarlockService, job_id: str, sheet_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    check_sheet_id(sheet_id)
    record = rigging.read_sheet_pixel(svc.job_dir(job_id), sheet_id)
    if record is None:
        raise NotFound("no such pixel sheet")
    return record


def sheet_pixel_png(svc: WarlockService, job_id: str, sheet_id: str) -> Path:
    check_job_id(job_id)
    check_sheet_id(sheet_id)
    job_dir = svc.job_dir(job_id)
    path = rigging.sheet_pixel_png_path(job_dir, sheet_id)
    # The sidecar is the completion marker here too: the worker writes the PNG
    # first, so existence alone can serve a partial file mid-save.
    if not path.exists() or not rigging.sheet_pixel_path(job_dir, sheet_id).exists():
        raise NotFound("no such pixel sheet")
    return path
