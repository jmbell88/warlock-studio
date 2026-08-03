"""Sprite sheets: what one may ask for, and queuing a render."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .. import rigging
from .core import WarlockService
from .errors import Conflict, Invalid, NotFound
from .validation import check_job_id, check_pose_id, check_sheet_id


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
            raise Invalid("a clip needs both clip_from and clip_to")
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
            raise Invalid(str(exc)) from exc

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
        raise Invalid(str(exc)) from exc

    sheet_name = (name or "").strip()
    if len(sheet_name) > rigging.MAX_SHEET_NAME:
        raise Invalid(
            f"sheet name must be at most {rigging.MAX_SHEET_NAME} characters", field="name"
        )

    if len(rigging.list_sheets(job_dir)) >= rigging.MAX_SHEETS:
        raise Conflict(f"a job may hold at most {rigging.MAX_SHEETS} sheets")

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


def delete_sheet(svc: WarlockService, job_id: str, sheet_id: str) -> dict[str, Any]:
    check_job_id(job_id)
    check_sheet_id(sheet_id)
    if not rigging.delete_sheet(svc.job_dir(job_id), sheet_id):
        raise NotFound("no such sheet")
    return {"ok": True}
