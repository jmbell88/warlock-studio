"""Single-worker GPU job queue.

One job runs at a time. VRAM handoff for text jobs: the trellis server is stopped
before Flux loads (both don't fit alongside each other), and Flux is unloaded
before the trellis server starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from .config import Config
from .db import JobStore
from .pipelines.trellis import TrellisServer

log = logging.getLogger(__name__)

POLL_INTERVAL = 1.0


class Worker:
    def __init__(self, config: Config, store: JobStore) -> None:
        self.config = config
        self.store = store
        self.trellis = TrellisServer(
            config.trellis_server_exe,
            config.trellis_models_dir,
            config.trellis_port,
            log_path=config.data_dir / "trellis.log",
        )
        self._text2image = None  # lazy: torch/diffusers may not be installed
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.current_job_id: str | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="gpu-worker")

    async def shutdown(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
        self.trellis.stop()

    async def _run(self) -> None:
        while not self._stop.is_set():
            job = await asyncio.to_thread(self.store.next_queued)
            if job is None:
                self._maybe_evict_idle()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=POLL_INTERVAL)
                continue
            await self._process(job)

    def _maybe_evict_idle(self) -> None:
        if (
            self.trellis.running
            and time.monotonic() - self.trellis.last_used > self.config.trellis_idle_timeout
        ):
            log.info("evicting idle trellis-server")
            self.trellis.stop()

    async def _process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        self.current_job_id = job_id
        await asyncio.to_thread(self.store.set_status, job_id, "running")
        try:
            await self._generate(job)
        except Exception as exc:
            log.exception("job %s failed", job_id)
            await asyncio.to_thread(self.store.set_status, job_id, "error", str(exc))
        else:
            # A cancel request may have landed mid-run; don't overwrite it.
            current = await asyncio.to_thread(self.store.get, job_id)
            if current and current["status"] == "running":
                await asyncio.to_thread(self.store.set_status, job_id, "done")
        finally:
            self.current_job_id = None

    async def _generate(self, job: dict[str, Any]) -> None:
        job_dir = self.config.job_dir(job["id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        params = job["params"]
        seed = int(params.get("seed", 42))
        resolution = int(params.get("resolution", 1024))
        image_path = job_dir / "input.png"

        if job["kind"] == "text":
            # Free VRAM held by the 3D server, run Flux, then free Flux.
            self.trellis.stop()
            t2i = self._get_text2image()
            try:
                await asyncio.to_thread(t2i.generate, job["prompt"], image_path, seed=seed)
            finally:
                await asyncio.to_thread(t2i.unload)
        elif not image_path.exists():
            raise RuntimeError("image job has no uploaded input.png")

        await self.trellis.generate(
            image_path, job_dir / "model.glb", seed=seed, resolution=resolution
        )

    def _get_text2image(self):
        if self._text2image is None:
            try:
                from .pipelines.text2image import Text2Image
            except ImportError as exc:
                raise RuntimeError(
                    "text-to-3D requires the text2image extra: uv sync --extra text2image"
                ) from exc
            self._text2image = Text2Image(
                self.config.t2i_model_id, self.config.t2i_image_size
            )
        return self._text2image
