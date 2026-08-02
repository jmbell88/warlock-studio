"""FastAPI app: job API + static web UI."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import re
import secrets
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import doctor, guidance, rigging
from .config import get_config
from .db import JobStore
from .queue import Worker

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_RESOLUTIONS = {512, 1024, 1536}

# Ceiling for /api/jobs?limit= and the internal full-history reads (prune).
# Bounded so a caller can't ask the single sqlite connection for everything at
# once and stall every other request behind it.
MAX_LIST_LIMIT = 5000

# Exactly what create_job generates: uuid4().hex[:12].
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _random_seed() -> int:
    """A fresh seed for a re-roll. 31-bit so it round-trips through JS numbers
    and an sqlite INTEGER without surprises."""
    return secrets.randbelow(2**31)


def _check_job_id(job_id: str) -> None:
    """Reject anything that isn't a generated job id, before it reaches the FS.

    config.job_dir() is a bare data_dir / job_id join with no sanitisation, so
    every route that builds a path from a caller-supplied id needs this. Only
    get_file could actually be steered today -- the others gate on a DB lookup
    first -- but the check costs nothing and removes the class rather than the
    instance. 404 rather than 400: a malformed id and a missing one are
    indistinguishable to the caller, and saying so leaks less.
    """
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(404, "no such job")


def _check_pose_id(pose_id: str) -> None:
    """Same guard, same reasoning, for the ids that name files inside a job dir."""
    if not rigging.is_valid_id(pose_id):
        raise HTTPException(404, "no such pose")


def _check_sheet_id(sheet_id: str) -> None:
    if not rigging.is_valid_id(sheet_id):
        raise HTTPException(404, "no such sheet")


def create_app() -> FastAPI:
    config = get_config()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        store = JobStore(config.db_path)
        # A job still 'running' at process start was orphaned by a crash or
        # an unclean shutdown -- surface it instead of silently re-running
        # a 2-minute GPU job on every restart.
        await asyncio.to_thread(store.reconcile_startup)
        for check in await asyncio.to_thread(doctor.run_checks, config):
            if not check.ok:
                level = log.critical if check.fatal else log.warning
                level("doctor: %s -- %s", check.name, check.detail)
        worker = Worker(config, store)
        worker.start()
        app.state.store = store
        app.state.worker = worker
        try:
            yield
        finally:
            await worker.shutdown()
            store.close()

    app = FastAPI(title="Warlock Studio", lifespan=lifespan)

    def store() -> JobStore:
        return app.state.store

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        worker: Worker = app.state.worker
        checks = await asyncio.to_thread(
            functools.partial(doctor.run_checks, config, trellis_running=worker.trellis.running)
        )
        return {
            "ok": worker.alive and worker.fatal is None,
            "worker_alive": worker.alive,
            "fatal": str(worker.fatal) if worker.fatal else None,
            "trellis_running": worker.trellis.running,
            "checks": [asdict(c) for c in checks],
        }

    @app.get("/api/progress")
    async def progress() -> dict[str, Any]:
        """Live progress for the running job. Cheap by design: no DB, no disk.

        The UI polls this every ~600 ms while a job is active, and the full job
        list only every few seconds.
        """
        worker = app.state.worker
        return {
            "job_id": worker.current_job_id,
            "progress": worker.progress.snapshot(),
            # Lets the client correct for clock skew when rendering elapsed time.
            "server_time": time.time(),
        }

    @app.get("/api/guidance")
    async def guidance_catalog() -> dict[str, Any]:
        """Taxonomy for the design-guidance selects, so the UI has one source."""
        return guidance.catalog()

    @app.post("/api/jobs")
    async def create_job(
        kind: Annotated[str, Form()],
        prompt: Annotated[str | None, Form()] = None,
        seed: Annotated[int, Form()] = 42,
        resolution: Annotated[int | None, Form()] = None,
        genre: Annotated[str | None, Form()] = None,
        art_style: Annotated[str | None, Form()] = None,
        category: Annotated[str | None, Form()] = None,
        platform: Annotated[str | None, Form()] = None,
        size_m: Annotated[float | None, Form()] = None,
        base_model: Annotated[str | None, Form()] = None,
        style_lora: Annotated[str | None, Form()] = None,
        lora_weight: Annotated[float | None, Form()] = None,
        bg_removal: Annotated[str | None, Form()] = None,
        rig: Annotated[bool, Form()] = False,
        rig_template: Annotated[str | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        if kind not in ("text", "image"):
            raise HTTPException(400, "kind must be 'text' or 'image'")
        # An explicit resolution overrides the platform preset; the UI no longer
        # sends one, but the API keeps accepting it.
        if resolution is not None and resolution not in ALLOWED_RESOLUTIONS:
            raise HTTPException(400, f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
        if kind == "text" and not (prompt and prompt.strip()):
            raise HTTPException(400, "text jobs require a prompt")
        if kind == "image" and image is None:
            raise HTTPException(400, "image jobs require an image upload")

        # Validated up front: a rejected request must not leave an input.png behind.
        try:
            params = guidance.normalize(
                {
                    "genre": genre,
                    "art_style": art_style,
                    "category": category,
                    "platform": platform,
                    "size_m": size_m,
                    "resolution": resolution,
                    "base_model": base_model,
                    "style_lora": style_lora,
                    "lora_weight": lora_weight,
                    "bg_removal": bg_removal,
                }
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        params["seed"] = seed
        if rig:
            # Validated now rather than 90 seconds later: an unusable template
            # should cost the request, not the whole generation that precedes
            # the rig. The worker queues the follow-up job when the mesh lands.
            params["rig"] = True
            params["rig_template"] = _valid_template(rig_template)

        normalized: bytes | None = None
        if image is not None:
            data = await image.read()
            try:
                normalized = await asyncio.to_thread(_to_png, data)
            except Exception as exc:
                raise HTTPException(400, "could not decode uploaded image") from exc

        # Write the file before the row exists: the worker's next_queued()
        # poll can otherwise claim an image job in the gap and find no
        # input.png on disk yet.
        job_id = uuid.uuid4().hex[:12]
        if normalized is not None:
            job_dir = config.job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "input.png").write_bytes(normalized)
        await asyncio.to_thread(store().create, kind, prompt, params, job_id)
        return {"id": job_id}

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(limit, MAX_LIST_LIMIT))
        jobs = await asyncio.to_thread(store().list, limit)
        for job in jobs:
            _attach_files(job, config.job_dir(job["id"]))
            _attach_progress(job, app.state.worker)
        return jobs

    @app.get("/api/storage")
    async def storage() -> dict[str, Any]:
        """How much disk the generated assets are using.

        Jobs and their artifacts accumulate forever otherwise -- at 5-20 MB per
        GLB that is real disk within weeks of regular use, and nothing in the
        UI ever said so.
        """
        return await asyncio.to_thread(_measure_storage, config.data_dir)

    @app.post("/api/jobs/prune")
    async def prune_jobs(keep: Annotated[int, Form()] = 20) -> dict[str, Any]:
        """Delete everything but the newest ``keep`` jobs. Never touches a running one."""
        if keep < 0:
            raise HTTPException(400, "keep must be >= 0")
        jobs = await asyncio.to_thread(store().list, MAX_LIST_LIMIT)
        deleted = 0
        for job in jobs[keep:]:
            if job["status"] == "running":
                continue
            await asyncio.to_thread(store().delete, job["id"])
            await asyncio.to_thread(
                shutil.rmtree, config.job_dir(job["id"]), ignore_errors=True
            )
            deleted += 1
        return {"deleted": deleted}

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        _attach_files(job, config.job_dir(job_id))
        _attach_progress(job, app.state.worker)
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] == "cancelled":
            # Idempotent success: some earlier request (possibly this exact
            # race) already cancelled it. Only a genuinely terminal
            # done/error status below is "too late" and worth a 409.
            return {"ok": True}
        if job["status"] not in ("queued", "running"):
            raise HTTPException(409, f"job is {job['status']}")
        if job["status"] == "running":
            await app.state.worker.request_cancel(job_id)
        # Atomic: if the worker's own terminal write (done/error) landed
        # first, this is a no-op and the job's real outcome stands instead
        # of being retroactively overwritten to "cancelled". The DB-level
        # JobStore.finish() conditional write (queue.py) is what actually
        # closes the lost-cancel race for a job that was 'queued' here but
        # got claimed before this request reached the DB -- not this call
        # to request_cancel, which only matters for a job already running.
        cancelled = await asyncio.to_thread(store().cancel, job_id)
        if not cancelled:
            # Could be "already cancelled" (idempotent success -- this
            # request's own effect already landed, e.g. via the race above)
            # or "already done/error" (genuinely too late).
            current = await asyncio.to_thread(store().get, job_id)
            if current and current["status"] == "cancelled":
                return {"ok": True}
            raise HTTPException(409, "job already finished")
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/rerun")
    async def rerun_job(
        job_id: str,
        mode: Annotated[str, Form()] = "reroll",
        seed: Annotated[int | None, Form()] = None,
    ) -> dict[str, Any]:
        """Resubmit a finished job, either from scratch or from its reference image.

        The two loops this closes:

        * ``reroll`` -- same prompt and guidance, new seed. Generation is
          deterministic in the seed, so pressing Generate twice on an unchanged
          form used to produce the identical mesh; this is the "that's close,
          give me another" button.
        * ``remesh`` -- reuse the existing input.png and rerun only the 3D
          stage. When SDXL drew a good reference but trellis made a poor mesh
          there was no way to retry the second half without paying for the
          first, including the VRAM handoff in exclusive mode.

        Both reduce to creating an ordinary job -- remesh is just an image job
        whose input.png is copied across -- so the worker, the queue and the
        progress model need no special case.
        """
        _check_job_id(job_id)
        if mode not in ("reroll", "remesh"):
            raise HTTPException(400, "mode must be 'reroll' or 'remesh'")
        source = await asyncio.to_thread(store().get, job_id)
        if source is None:
            raise HTTPException(404, "no such job")

        kind = "image" if mode == "remesh" else source["kind"]
        src_png = config.job_dir(job_id) / "input.png"
        if kind == "image" and not src_png.exists():
            raise HTTPException(400, "source job has no reference image to reuse")

        # Derived values describe the *source* run, not this one: keeping them
        # would make the new job claim a composed prompt it never used and a
        # quality score for a mesh that doesn't exist yet.
        params = {
            k: v
            for k, v in source["params"].items()
            if k not in ("composed_prompt", "scale_factor", "mesh_audit")
        }
        params["seed"] = seed if seed is not None else _random_seed()
        params["rerun_of"] = job_id

        new_id = uuid.uuid4().hex[:12]
        if kind == "image":
            # Before the row exists, for the same reason create_job does it:
            # next_queued can otherwise claim the job in the gap and find no
            # input.png on disk.
            new_dir = config.job_dir(new_id)
            new_dir.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(shutil.copyfile, src_png, new_dir / "input.png")
        await asyncio.to_thread(store().create, kind, source["prompt"], params, new_id)
        return {"id": new_id, "seed": params["seed"]}

    @app.get("/api/rig/templates")
    async def rig_templates() -> dict[str, Any]:
        """Which skeletons a rig request may ask for, plus whether rigging
        works at all. The UI hides the rig controls when available is false
        rather than offering a button that can only fail."""
        check = await asyncio.to_thread(doctor.blender_check)
        return {
            "available": check.ok,
            "detail": check.detail,
            "default": config.rig_template,
            "templates": rigging.catalog(),
        }

    @app.post("/api/jobs/{job_id}/rig")
    async def create_rig(
        job_id: str, template: Annotated[str | None, Form()] = None
    ) -> dict[str, Any]:
        """Queue a rig for a finished job's mesh.

        A queue job rather than an inline call: automatic weights on a 300k-face
        mesh is minutes of CPU, and going through the queue buys cancellation,
        progress and history for free -- and guarantees it never overlaps a
        trellis or SDXL run.
        """
        _check_job_id(job_id)
        source = await asyncio.to_thread(store().get, job_id)
        if source is None:
            raise HTTPException(404, "no such job")
        if source["kind"] == "rig":
            raise HTTPException(400, "cannot rig a rig job; rig its source mesh")
        if source["status"] != "done" or not (config.job_dir(job_id) / "model.glb").exists():
            raise HTTPException(400, "job has no finished mesh to rig")
        params = {"source_job": job_id, "template": _valid_template(template)}
        new_id = await asyncio.to_thread(
            store().create, "rig", source["prompt"], params, uuid.uuid4().hex[:12]
        )
        return {"id": new_id, "source_job": job_id, "template": params["template"]}

    @app.get("/api/jobs/{job_id}/rig")
    async def get_rig(job_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        rig = await asyncio.to_thread(rigging.read_rig, config.job_dir(job_id))
        if rig is None:
            raise HTTPException(404, "job is not rigged")
        return rig

    # One lock per derived artifact, so the same STL/OBJ/posed GLB is never
    # converted twice concurrently. Created lazily and never evicted: an idle
    # asyncio.Lock is a few dozen bytes and the key space is jobs x a handful.
    _convert_locks: dict[tuple[str, str], asyncio.Lock] = {}

    # --- poses ---------------------------------------------------------------
    # A pose is a bone -> local rotation map saved against a job's rig, stored
    # as a file in that job's directory. No job kind and no DB row: a pose is
    # small, instant to write, and belongs to the mesh the same way its rig
    # does. Baking one into a GLB is the only slow part, and that is derived on
    # demand and cached, exactly like the STL and OBJ exports above.

    def _rig_bones(job_id: str) -> list[str]:
        bones = rigging.rig_bone_names(config.job_dir(job_id))
        if bones is None:
            raise HTTPException(404, "job is not rigged")
        return bones

    @app.get("/api/jobs/{job_id}/poses")
    async def list_poses(job_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        bones = await asyncio.to_thread(_rig_bones, job_id)
        poses = await asyncio.to_thread(rigging.list_poses, config.job_dir(job_id))
        return {"bones": bones, "poses": poses}

    @app.post("/api/jobs/{job_id}/poses")
    async def save_pose(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a pose, or overwrite one by passing its id.

        JSON rather than a form like the rest of the API: the body is a bone
        map of quaternions, which multipart can only carry as a re-encoded
        string.
        """
        _check_job_id(job_id)
        known = await asyncio.to_thread(_rig_bones, job_id)
        job_dir = config.job_dir(job_id)
        try:
            pose = rigging.validate_pose(payload, known)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        pose_id = str(payload["id"]) if payload.get("id") else None
        if pose_id is not None:
            _check_pose_id(pose_id)
            if not rigging.pose_path(job_dir, pose_id).exists():
                raise HTTPException(404, "no such pose")
        else:
            existing = await asyncio.to_thread(rigging.list_poses, job_dir)
            if len(existing) >= rigging.MAX_POSES:
                raise HTTPException(409, f"a job may hold at most {rigging.MAX_POSES} poses")
        return await asyncio.to_thread(rigging.save_pose, job_dir, pose, pose_id)

    @app.delete("/api/jobs/{job_id}/poses/{pose_id}")
    async def delete_pose(job_id: str, pose_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        _check_pose_id(pose_id)
        if not await asyncio.to_thread(rigging.delete_pose, config.job_dir(job_id), pose_id):
            raise HTTPException(404, "no such pose")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/poses/{pose_id}/model.glb")
    async def get_posed_model(job_id: str, pose_id: str) -> FileResponse:
        """The rig with one saved pose baked into it.

        Derived on first request and cached beside the pose, under the same
        per-artifact lock the STL/OBJ exports use -- posing runs Blender, and
        two tabs asking at once should wait for one subprocess, not start two.
        """
        _check_job_id(job_id)
        _check_pose_id(pose_id)
        job_dir = config.job_dir(job_id)
        pose = await asyncio.to_thread(rigging.read_pose, job_dir, pose_id)
        if pose is None:
            raise HTTPException(404, "no such pose")
        path = rigging.pose_glb_path(job_dir, pose_id)
        if not path.exists():
            lock = _convert_locks.setdefault((job_id, f"pose:{pose_id}"), asyncio.Lock())
            async with lock:
                if not path.exists():
                    if not (job_dir / "rig.glb").exists():
                        raise HTTPException(404, "job is not rigged")
                    spec = rigging.pose_spec(job_dir, pose_id, pose["bones"])
                    try:
                        await asyncio.to_thread(
                            functools.partial(
                                rigging.run_worker, spec, timeout=config.pose_timeout
                            )
                        )
                    except rigging.BlenderError as exc:
                        log.error("posing %s/%s failed: %s", job_id, pose_id, exc)
                        raise HTTPException(500, "could not bake this pose") from exc
        return FileResponse(
            path, media_type="model/gltf-binary", filename=f"{job_id}_{pose_id}.glb"
        )

    # --- sprite sheets -------------------------------------------------------

    @app.get("/api/sheets/options")
    async def sheet_options() -> dict[str, Any]:
        """What a sheet request may ask for. One source for the form, as with
        /api/rig/templates -- the UI never hardcodes a frame size."""
        from .pipelines import sheet as sheetlib

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

    @app.get("/api/jobs/{job_id}/sheets")
    async def list_sheets(job_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        return {"sheets": await asyncio.to_thread(rigging.list_sheets, config.job_dir(job_id))}

    @app.post("/api/jobs/{job_id}/sheets")
    async def create_sheet(
        job_id: str,
        poses: Annotated[list[str] | None, Form()] = None,
        elevation: Annotated[float | None, Form()] = None,
        frame_size: Annotated[int | None, Form()] = None,
        lighting: Annotated[str | None, Form()] = None,
        name: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        """Queue a pose x direction sprite sheet for a finished job's mesh.

        Every reason this can be refused is checked here rather than in the
        worker: an unrenderable request should cost the request, not a place in
        the queue and two minutes of EEVEE.
        """
        from .pipelines import sheet as sheetlib

        _check_job_id(job_id)
        source = await asyncio.to_thread(store().get, job_id)
        if source is None:
            raise HTTPException(404, "no such job")
        job_dir = config.job_dir(job_id)
        if source["status"] != "done" or not (job_dir / "model.glb").exists():
            raise HTTPException(400, "job has no finished mesh to render")

        pose_ids = [p for p in (poses or []) if p]
        records = []
        for pose_id in pose_ids:
            _check_pose_id(pose_id)
            record = await asyncio.to_thread(rigging.read_pose, job_dir, pose_id)
            if record is None:
                raise HTTPException(404, f"no such pose {pose_id}")
            records.append(record)
        if records and not (job_dir / "rig.glb").exists():
            raise HTTPException(400, "posed sheets need a rigged mesh")

        try:
            # Built and thrown away: the worker plans it again from the same
            # inputs. This call is here purely so a bad frame size or an atlas
            # over the texture limit is a 400 now instead of a failed job later.
            sheetlib.plan(
                records,
                frame_size=frame_size or sheetlib.DEFAULT_FRAME_SIZE,
                elevation=sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
                lighting=lighting or "flat",
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        existing = await asyncio.to_thread(rigging.list_sheets, job_dir)
        if len(existing) >= rigging.MAX_SHEETS:
            raise HTTPException(409, f"a job may hold at most {rigging.MAX_SHEETS} sheets")

        params = {
            "source_job": job_id,
            "sheet_id": rigging.new_id(),
            "poses": pose_ids,
            "elevation": sheetlib.DEFAULT_ELEVATION if elevation is None else elevation,
            "frame_size": frame_size or sheetlib.DEFAULT_FRAME_SIZE,
            "lighting": lighting or "flat",
            "name": (name or "").strip(),
        }
        new_id = await asyncio.to_thread(
            store().create, "sheet", source["prompt"], params, uuid.uuid4().hex[:12]
        )
        return {"id": new_id, "source_job": job_id, "sheet_id": params["sheet_id"]}

    @app.get("/api/jobs/{job_id}/sheets/{sheet_id}")
    async def get_sheet(job_id: str, sheet_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        _check_sheet_id(sheet_id)
        record = await asyncio.to_thread(rigging.read_sheet, config.job_dir(job_id), sheet_id)
        if record is None:
            raise HTTPException(404, "no such sheet")
        return record

    @app.get("/api/jobs/{job_id}/sheets/{sheet_id}/sheet.png")
    async def get_sheet_png(job_id: str, sheet_id: str) -> FileResponse:
        _check_job_id(job_id)
        _check_sheet_id(sheet_id)
        path = rigging.sheet_png_path(config.job_dir(job_id), sheet_id)
        if not path.exists():
            raise HTTPException(404, "no such sheet")
        return FileResponse(path, media_type="image/png", filename=f"{job_id}_{sheet_id}.png")

    @app.delete("/api/jobs/{job_id}/sheets/{sheet_id}")
    async def delete_sheet(job_id: str, sheet_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        _check_sheet_id(sheet_id)
        if not await asyncio.to_thread(rigging.delete_sheet, config.job_dir(job_id), sheet_id):
            raise HTTPException(404, "no such sheet")
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, Any]:
        _check_job_id(job_id)
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] == "running":
            raise HTTPException(409, "cancel the job before deleting it")
        await asyncio.to_thread(store().delete, job_id)
        # Threaded: a job dir is a GLB plus its textures, and rmtree on the
        # event loop stalls every other route for the duration.
        await asyncio.to_thread(shutil.rmtree, config.job_dir(job_id), ignore_errors=True)
        return {"ok": True}

    _MEDIA = {
        "model.glb": "model/gltf-binary",
        "input.png": "image/png",
        "model.stl": "model/stl",
        "model_obj.zip": "application/zip",
        "rig.glb": "model/gltf-binary",
    }

    @app.get("/api/jobs/{job_id}/files/{name}")
    async def get_file(job_id: str, name: str) -> FileResponse:
        _check_job_id(job_id)
        if name not in _MEDIA:
            raise HTTPException(404, "unknown file")
        job_dir = config.job_dir(job_id)
        path = job_dir / name
        glb = job_dir / "model.glb"
        # Same gate as _attach_files: the Blender export of rig.glb is not
        # atomic and the worker writes rig.json last, so serving rig.glb on
        # mere existence can hand back a truncated GLB to a direct GET.
        if name == "rig.glb" and not (job_dir / "rig.json").exists():
            raise HTTPException(404, "file not ready")
        if not path.exists() and name in ("model.stl", "model_obj.zip") and glb.exists():
            from .pipelines import postprocess

            convert = (
                postprocess.glb_to_stl if name == "model.stl" else postprocess.glb_to_obj_zip
            )
            # setdefault, not "if key not in": both are safe here because this
            # runs on the single event loop with no await between them, but
            # setdefault says so without relying on the reader to check.
            lock = _convert_locks.setdefault((job_id, name), asyncio.Lock())
            async with lock:
                # Re-checked inside the lock: whoever waited here was waiting
                # for exactly this file, so the second request serves the
                # finished artifact instead of exporting a 300K-face mesh again.
                if not path.exists():
                    await asyncio.to_thread(convert, glb, path)
        if not path.exists():
            raise HTTPException(404, "file not ready")
        return FileResponse(path, media_type=_MEDIA[name], filename=f"{job_id}_{name}")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _valid_template(key: str | None) -> str:
    """A known skeleton template key, or 400. None falls back to the config default."""
    config = get_config()
    try:
        return rigging.get_template(key or config.rig_template).key
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _to_png(data: bytes) -> bytes:
    """Re-encode any uploaded image as PNG; trellis.cpp only decodes PNG/JPEG.

    Alpha is preserved only when the source already had it, so a pre-matted
    upload (RGBA/LA/PA, or a palette image with a transparency entry) keeps
    its alpha for the server's bg-removal auto-detection, without forcing an
    opaque photo through the same path.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        out = io.BytesIO()
        im.convert("RGBA" if has_alpha else "RGB").save(out, "PNG")
        return out.getvalue()


def _measure_storage(data_dir: Path) -> dict[str, Any]:
    """Total bytes and directory count under data_dir. Runs in a thread."""
    total = 0
    dirs = 0
    if data_dir.exists():
        for entry in data_dir.iterdir():
            if not entry.is_dir():
                continue
            dirs += 1
            for f in entry.rglob("*"):
                # Symlinks and vanishing files (a concurrent delete) shouldn't
                # abort the whole measurement.
                with contextlib.suppress(OSError):
                    if f.is_file():
                        total += f.stat().st_size
    return {"job_dirs": dirs, "bytes": total}


def _attach_files(job: dict[str, Any], job_dir: Path) -> None:
    # model.glb is gated on status, not just existence: the worker still
    # writes to it after the file first appears on disk (see
    # queue.py:_apply_scale), so "exists" alone can expose a half-written
    # file to a concurrent GET and blow up FileResponse's Content-Length.
    files = []
    if (job_dir / "input.png").exists():
        files.append("input.png")
    if job["status"] == "done" and (job_dir / "model.glb").exists():
        files.append("model.glb")
    # rig.glb is gated on rig.json, not on its own existence and not on this
    # job's status. The rig lands in the *source* job's directory, so this job
    # is usually already 'done' while a separate rig job is still writing --
    # and the worker writes rig.json last, which makes it the completion
    # marker for the pair.
    if (job_dir / "rig.json").exists() and (job_dir / "rig.glb").exists():
        files.append("rig.glb")
    job["files"] = files


def _attach_progress(job: dict[str, Any], worker: Any) -> None:
    """Live progress, only ever for the job actually running."""
    job["progress"] = worker.progress.snapshot(job["id"])
