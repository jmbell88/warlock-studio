"""FastAPI app: job API + static web UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import get_config
from .db import JobStore
from .queue import Worker

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_RESOLUTIONS = {512, 1024, 1536}


def create_app() -> FastAPI:
    config = get_config()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        store = JobStore(config.db_path)
        worker = Worker(config, store)
        worker.start()
        app.state.store = store
        app.state.worker = worker
        try:
            yield
        finally:
            await worker.shutdown()
            store.close()

    app = FastAPI(title="Animancer3D", lifespan=lifespan)

    def store() -> JobStore:
        return app.state.store

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "trellis_running": app.state.worker.trellis.running}

    @app.post("/api/jobs")
    async def create_job(
        kind: Annotated[str, Form()],
        prompt: Annotated[str | None, Form()] = None,
        seed: Annotated[int, Form()] = 42,
        resolution: Annotated[int, Form()] = 1024,
        image: Annotated[UploadFile | None, File()] = None,
    ) -> dict[str, Any]:
        if kind not in ("text", "image"):
            raise HTTPException(400, "kind must be 'text' or 'image'")
        if resolution not in ALLOWED_RESOLUTIONS:
            raise HTTPException(400, f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}")
        if kind == "text" and not (prompt and prompt.strip()):
            raise HTTPException(400, "text jobs require a prompt")
        if kind == "image" and image is None:
            raise HTTPException(400, "image jobs require an image upload")

        normalized: bytes | None = None
        if image is not None:
            data = await image.read()
            try:
                normalized = await asyncio.to_thread(_to_png, data)
            except Exception as exc:
                raise HTTPException(400, "could not decode uploaded image") from exc

        params = {"seed": seed, "resolution": resolution}
        job_id = await asyncio.to_thread(store().create, kind, prompt, params)
        if normalized is not None:
            job_dir = config.job_dir(job_id)
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "input.png").write_bytes(normalized)
        return {"id": job_id}

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        jobs = await asyncio.to_thread(store().list)
        for job in jobs:
            _attach_files(job, config.job_dir(job["id"]))
        return jobs

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        _attach_files(job, config.job_dir(job_id))
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] not in ("queued", "running"):
            raise HTTPException(409, f"job is {job['status']}")
        # A running job finishes its current GPU stage; the worker preserves the
        # cancelled status instead of marking it done.
        await asyncio.to_thread(store().set_status, job_id, "cancelled")
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job["status"] == "running":
            raise HTTPException(409, "cancel the job before deleting it")
        await asyncio.to_thread(store().delete, job_id)
        shutil.rmtree(config.job_dir(job_id), ignore_errors=True)
        return {"ok": True}

    _MEDIA = {
        "model.glb": "model/gltf-binary",
        "input.png": "image/png",
        "model.stl": "model/stl",
        "model_obj.zip": "application/zip",
    }

    @app.get("/api/jobs/{job_id}/files/{name}")
    async def get_file(job_id: str, name: str) -> FileResponse:
        if name not in _MEDIA:
            raise HTTPException(404, "unknown file")
        job_dir = config.job_dir(job_id)
        path = job_dir / name
        glb = job_dir / "model.glb"
        if not path.exists() and name in ("model.stl", "model_obj.zip") and glb.exists():
            from .pipelines import postprocess

            convert = (
                postprocess.glb_to_stl if name == "model.stl" else postprocess.glb_to_obj_zip
            )
            await asyncio.to_thread(convert, glb, path)
        if not path.exists():
            raise HTTPException(404, "file not ready")
        return FileResponse(path, media_type=_MEDIA[name], filename=f"{job_id}_{name}")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _to_png(data: bytes) -> bytes:
    """Re-encode any uploaded image as PNG; trellis.cpp only decodes PNG/JPEG.

    RGBA is preserved so pre-matted images keep their alpha for background removal.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        out = io.BytesIO()
        im.convert("RGBA").save(out, "PNG")
        return out.getvalue()


def _attach_files(job: dict[str, Any], job_dir: Path) -> None:
    job["files"] = [
        name for name in ("input.png", "model.glb") if (job_dir / name).exists()
    ]
