"""FastAPI app: job API + static web UI.

Transitional. Every route body is now a call into :mod:`warlock.service` plus a
mapping from its exceptions back to HTTP status codes -- the logic itself lives
there, where the desktop app can reach it too. This module (and the browser
frontend it mounts) goes away once the desktop UI is at parity.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from .config import get_config
from .service import WarlockService, errors
from .service import derive as svc_derive
from .service import export as svc_export
from .service import files as svc_files
from .service import jobs as svc_jobs
from .service import rig as svc_rig
from .service import sheets as svc_sheets
from .service import system as svc_system
from .service.validation import MAX_UPLOAD_BYTES

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Transitional aliases: these moved into the service layer, and this module's
# own tests still reach for them here.
_attach_files = svc_files.attach_files
_to_png = svc_files.to_png


def _to_http(exc: errors.ServiceError) -> HTTPException:
    """The service's exception hierarchy carries the status it used to return."""
    return HTTPException(exc.status, exc.message)


async def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a service function in a thread, translating its errors.

    Threaded because every one of them is blocking by design -- the sqlite
    connection, an rmtree, a trimesh export -- and this is the one place that
    knows the loop must not be the thread doing it.
    """
    try:
        return await asyncio.to_thread(functools.partial(fn, *args, **kwargs))
    except errors.ServiceError as exc:
        raise _to_http(exc) from exc


def create_app() -> FastAPI:
    config = get_config()

    import contextlib

    from .db import JobStore
    from .queue import Worker

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        from . import doctor

        store = JobStore(config.db_path)
        # A job still 'running' at process start was orphaned by a crash or an
        # unclean shutdown -- surface it instead of silently re-running a
        # 2-minute GPU job on every restart.
        await asyncio.to_thread(store.reconcile_startup)
        for check in await asyncio.to_thread(doctor.run_checks, config):
            if not check.ok:
                level = log.critical if check.fatal else log.warning
                level("doctor: %s -- %s", check.name, check.detail)
        worker = Worker(config, store)
        worker.start()
        app.state.store = store
        app.state.worker = worker
        app.state.svc = WarlockService(config, store, worker, asyncio.get_running_loop())
        try:
            yield
        finally:
            await worker.shutdown()
            store.close()

    app = FastAPI(title="Warlock Studio", lifespan=lifespan)

    def svc() -> WarlockService:
        return app.state.svc

    # --- system --------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return await _call(svc_system.health, svc())

    @app.get("/api/progress")
    async def progress() -> dict[str, Any]:
        """Live progress for the running job. Cheap by design: no DB, no disk."""
        import time

        worker = app.state.worker
        return {
            "job_id": worker.current_job_id,
            "progress": worker.progress.snapshot(),
            # Lets the client correct for clock skew when rendering elapsed time.
            "server_time": time.time(),
        }

    @app.get("/api/guidance")
    async def guidance_catalog() -> dict[str, Any]:
        return svc_system.guidance_catalog()

    @app.get("/api/prompt-preview")
    async def prompt_preview(request: Request) -> dict[str, Any]:
        q = request.query_params
        raw: dict[str, Any] = dict(svc_system.preview_fields(q))
        raw["size_m"] = q.get("size_m")
        raw["resolution"] = q.get("resolution")
        raw["lora_weight"] = q.get("lora_weight")
        raw["bg_removal"] = q.get("bg_removal")
        if "negative_prompt" in q:
            raw["negative_prompt"] = q["negative_prompt"]
        return await _call(svc_system.prompt_preview, svc(), raw, q.get("prompt") or "")

    @app.get("/api/logs/trellis")
    async def trellis_log() -> dict[str, Any]:
        return await _call(svc_system.trellis_log, svc())

    # --- jobs ----------------------------------------------------------------

    @app.post("/api/jobs")
    async def create_job(
        request: Request,
        kind: Annotated[str, Form()],
        prompt: Annotated[str | None, Form()] = None,
        seed: Annotated[int, Form()] = 42,
        reference_seed: Annotated[int | None, Form()] = None,
        mesh_seed: Annotated[int | None, Form()] = None,
        resolution: Annotated[int | None, Form()] = None,
        size_m: Annotated[float | None, Form()] = None,
        lora_weight: Annotated[float | None, Form()] = None,
        bg_removal: Annotated[str | None, Form()] = None,
        negative_prompt: Annotated[str | None, Form()] = None,
        rig: Annotated[bool, Form()] = False,
        rig_template: Annotated[str | None, Form()] = None,
        profile: Annotated[str | None, Form()] = None,
        custom_triangles: Annotated[int | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
        output: Annotated[str, Form()] = "model",
        count: Annotated[int, Form()] = 1,
    ) -> dict[str, Any]:
        guidance_fields = svc_system.preview_fields(await request.form())
        # One byte past the cap is all that's needed to know it's over, without
        # buffering however much the client actually sent.
        data = await image.read(MAX_UPLOAD_BYTES + 1) if image is not None else None
        return await _call(
            svc_jobs.create_job,
            svc(),
            kind=kind,
            prompt=prompt,
            seed=seed,
            reference_seed=reference_seed,
            mesh_seed=mesh_seed,
            resolution=resolution,
            size_m=size_m,
            lora_weight=lora_weight,
            bg_removal=bg_removal,
            negative_prompt=negative_prompt,
            rig=rig,
            rig_template=rig_template,
            profile=profile,
            custom_triangles=custom_triangles,
            image=data,
            output=output,
            count=count,
            guidance_fields=guidance_fields,
        )

    @app.get("/api/jobs")
    async def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        return await _call(svc_jobs.list_jobs, svc(), limit)

    @app.get("/api/storage")
    async def storage() -> dict[str, Any]:
        return await _call(svc_jobs.storage, svc())

    @app.post("/api/jobs/prune")
    async def prune_jobs(keep: Annotated[int, Form()] = 20) -> dict[str, Any]:
        return await _call(svc_jobs.prune_jobs, svc(), keep)

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        return await _call(svc_jobs.get_job, svc(), job_id)

    @app.patch("/api/jobs/{job_id}")
    async def update_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await _call(svc_jobs.update_job, svc(), job_id, payload)

    @app.post("/api/jobs/{job_id}/thumb.png")
    async def put_thumbnail(job_id: str, request: Request) -> dict[str, Any]:
        return await _call(svc_files.save_thumbnail, svc(), job_id, await request.body())

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        return await _call(svc_jobs.cancel_job, svc(), job_id)

    @app.post("/api/jobs/{job_id}/rerun")
    async def rerun_job(
        job_id: str,
        mode: Annotated[str, Form()] = "reroll",
        seed: Annotated[int | None, Form()] = None,
    ) -> dict[str, Any]:
        return await _call(svc_jobs.rerun_job, svc(), job_id, mode=mode, seed=seed)

    @app.post("/api/jobs/{job_id}/model")
    async def promote_to_model(
        job_id: str,
        mesh_seed: Annotated[int | None, Form()] = None,
        platform: Annotated[str | None, Form()] = None,
        size_m: Annotated[float | None, Form()] = None,
        bg_removal: Annotated[str | None, Form()] = None,
        profile: Annotated[str | None, Form()] = None,
        custom_triangles: Annotated[int | None, Form()] = None,
        rig: Annotated[bool | None, Form()] = None,
        rig_template: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        return await _call(
            svc_jobs.promote_to_model,
            svc(),
            job_id,
            mesh_seed=mesh_seed,
            platform=platform,
            size_m=size_m,
            bg_removal=bg_removal,
            profile=profile,
            custom_triangles=custom_triangles,
            rig=rig,
            rig_template=rig_template,
        )

    @app.post("/api/jobs/{job_id}/optimize")
    async def optimize_job(
        job_id: str,
        profile: Annotated[str | None, Form()] = None,
        custom_triangles: Annotated[int | None, Form()] = None,
    ) -> dict[str, Any]:
        return await _call(
            svc_jobs.optimize_job,
            svc(),
            job_id,
            profile=profile,
            custom_triangles=custom_triangles,
        )

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, Any]:
        return await _call(svc_jobs.delete_job, svc(), job_id)

    # --- rigs and poses ------------------------------------------------------

    @app.get("/api/rig/templates")
    async def rig_templates() -> dict[str, Any]:
        return await _call(svc_rig.rig_templates, svc())

    @app.get("/api/rig/templates/{key}/poses")
    async def template_presets(key: str) -> dict[str, Any]:
        return await _call(svc_rig.template_presets, key)

    @app.post("/api/jobs/{job_id}/rig")
    async def create_rig(
        job_id: str, template: Annotated[str | None, Form()] = None
    ) -> dict[str, Any]:
        return await _call(svc_rig.create_rig, svc(), job_id, template=template)

    @app.post("/api/jobs/{job_id}/rig/joints")
    async def adjust_joints(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await _call(svc_rig.adjust_joints, svc(), job_id, payload)

    @app.get("/api/jobs/{job_id}/rig")
    async def get_rig(job_id: str) -> dict[str, Any]:
        return await _call(svc_rig.get_rig, svc(), job_id)

    @app.get("/api/jobs/{job_id}/poses")
    async def list_poses(job_id: str) -> dict[str, Any]:
        return await _call(svc_rig.list_poses, svc(), job_id)

    @app.post("/api/jobs/{job_id}/poses")
    async def save_pose(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await _call(svc_rig.save_pose, svc(), job_id, payload)

    @app.delete("/api/jobs/{job_id}/poses/{pose_id}")
    async def delete_pose(job_id: str, pose_id: str) -> dict[str, Any]:
        return await _call(svc_rig.delete_pose, svc(), job_id, pose_id)

    @app.get("/api/jobs/{job_id}/poses/{pose_id}/model.glb")
    async def get_posed_model(job_id: str, pose_id: str) -> FileResponse:
        path = await _call(svc_rig.posed_model, svc(), job_id, pose_id)
        return FileResponse(
            path, media_type="model/gltf-binary", filename=f"{job_id}_{pose_id}.glb"
        )

    # --- sprite sheets -------------------------------------------------------

    @app.get("/api/sheets/options")
    async def sheet_options() -> dict[str, Any]:
        return svc_sheets.sheet_options()

    @app.get("/api/jobs/{job_id}/sheets")
    async def list_sheets(job_id: str) -> dict[str, Any]:
        return await _call(svc_sheets.list_sheets, svc(), job_id)

    @app.post("/api/jobs/{job_id}/sheets")
    async def create_sheet(
        job_id: str,
        poses: Annotated[list[str] | None, Form()] = None,
        elevation: Annotated[float | None, Form()] = None,
        frame_size: Annotated[int | None, Form()] = None,
        lighting: Annotated[str | None, Form()] = None,
        name: Annotated[str | None, Form()] = None,
        clip_from: Annotated[str | None, Form()] = None,
        clip_to: Annotated[str | None, Form()] = None,
        clip_frames: Annotated[int, Form()] = 8,
        yaws: Annotated[int | None, Form()] = None,
    ) -> dict[str, Any]:
        return await _call(
            svc_sheets.create_sheet,
            svc(),
            job_id,
            poses=poses,
            elevation=elevation,
            frame_size=frame_size,
            lighting=lighting,
            name=name,
            clip_from=clip_from,
            clip_to=clip_to,
            clip_frames=clip_frames,
            yaws=yaws,
        )

    @app.get("/api/jobs/{job_id}/sheets/{sheet_id}")
    async def get_sheet(job_id: str, sheet_id: str) -> dict[str, Any]:
        return await _call(svc_sheets.get_sheet, svc(), job_id, sheet_id)

    @app.get("/api/jobs/{job_id}/sheets/{sheet_id}/sheet.png")
    async def get_sheet_png(job_id: str, sheet_id: str) -> FileResponse:
        path = await _call(svc_sheets.sheet_png, svc(), job_id, sheet_id)
        return FileResponse(path, media_type="image/png", filename=f"{job_id}_{sheet_id}.png")

    @app.delete("/api/jobs/{job_id}/sheets/{sheet_id}")
    async def delete_sheet(job_id: str, sheet_id: str) -> dict[str, Any]:
        return await _call(svc_sheets.delete_sheet, svc(), job_id, sheet_id)

    # --- files and export ----------------------------------------------------

    @app.get("/api/jobs/{job_id}/files/{name}")
    async def get_file(job_id: str, name: str) -> FileResponse:
        path = await _call(svc_derive.get_file, svc(), job_id, name)
        return FileResponse(path, media_type=svc_files.MEDIA[name], filename=f"{job_id}_{name}")

    @app.post("/api/export")
    async def bulk_export(
        ids: Annotated[list[str], Form()],
        files: Annotated[list[str] | None, Form()] = None,
    ) -> FileResponse:
        """Zip the named artifacts of several jobs into one download.

        Built into a temp file rather than streamed from memory: a selection of
        twenty 22 MB GLBs is not something to hold in the event loop's heap.
        """
        fd, raw = tempfile.mkstemp(dir=config.data_dir, prefix=".export.", suffix=".zip")
        os.close(fd)
        archive = Path(raw)
        try:
            await _call(svc_export.bulk_export, svc(), ids, files, archive)
        except Exception:
            archive.unlink(missing_ok=True)
            raise
        return FileResponse(
            archive,
            media_type="application/zip",
            filename="warlock_export.zip",
            background=BackgroundTask(archive.unlink, missing_ok=True),
        )

    @app.post("/api/export/folder")
    async def export_to_folder(
        ids: Annotated[list[str], Form()],
        files: Annotated[list[str] | None, Form()] = None,
    ) -> dict[str, Any]:
        return await _call(svc_export.export_to_folder, svc(), ids, files)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
