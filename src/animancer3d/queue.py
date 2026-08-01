"""Single-worker GPU job queue.

One job runs at a time. VRAM handoff for text jobs: the trellis server is stopped
before Flux loads (both don't fit alongside each other), and Flux is unloaded
before the trellis server starts.

Cancellation has no HTTP counterpart on trellis-server.exe (it exposes exactly
/generate and /health) and aborting the client request does not stop the GPU.
The only mechanism that actually frees VRAM mid-run is killing the subprocess
(TrellisServer.stop()), which this module already does for the VRAM handoff.
A cancel during the text2image stage instead sets a threading.Event that the
diffusers step callback checks every step (see pipelines/text2image.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import errors
from .config import Config
from .db import JobStore
from .pipelines.trellis import TrellisServer
from .progress import ProgressBus, TrellisProgressParser

log = logging.getLogger(__name__)

POLL_INTERVAL = 1.0
SHUTDOWN_TIMEOUT = 20.0

# Labels for the text-to-image phases, which have no trace of their own.
T2I_PHASES = {
    "download": ("t2i_download", "Downloading SDXL-Turbo (~7 GB, first run only)"),
    "load": ("t2i_load", "Loading image model"),
    "sample": ("t2i_sample", "Drawing reference image"),
}


@dataclass
class _Cancel:
    job_id: str
    event: threading.Event = field(default_factory=threading.Event)


class Worker:
    def __init__(self, config: Config, store: JobStore) -> None:
        self.config = config
        self.store = store
        self.trellis = TrellisServer(
            config.trellis_server_exe,
            config.trellis_models_dir,
            config.trellis_port,
            log_path=config.data_dir / "trellis.log",
            webp=config.trellis_webp,
        )
        self._text2image = None  # lazy: torch/diffusers may not be installed
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.current_job_id: str | None = None
        self._cancel: _Cancel | None = None
        self.fatal: BaseException | None = None
        self.progress = ProgressBus()
        self._parser = TrellisProgressParser(self._emit_progress)
        self.trellis.on_line = self._parser.feed

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="gpu-worker")
        self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.fatal = exc
            log.critical("gpu worker task died", exc_info=exc)

    async def request_cancel(self, job_id: str) -> None:
        """No-op unless job_id is the job currently running."""
        if job_id != self.current_job_id or self._cancel is None:
            return
        self._cancel.event.set()
        snapshot = self.progress.snapshot()
        phase = snapshot["phase"] if snapshot else None
        if phase == "trellis":
            # The only real "abort" trellis-server.exe has: kill it. The
            # in-flight client.post then dies with a TransportError, which
            # _process below turns into a cancelled status because the
            # cancel event is already set.
            await asyncio.to_thread(self.trellis.stop)
        # t2i_sample: the diffusers step callback checks the event itself.
        # t2i_load / t2i_download: not interruptible; the event is checked
        # once between load() and sampling in Text2Image.generate().

    async def shutdown(self) -> None:
        self._stop.set()
        if self.current_job_id is not None:
            await self.request_cancel(self.current_job_id)
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=SHUTDOWN_TIMEOUT)
            except TimeoutError:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        await asyncio.to_thread(self.trellis.stop)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = await asyncio.to_thread(self.store.next_queued)
                if job is None:
                    self._maybe_evict_idle()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._stop.wait(), timeout=POLL_INTERVAL)
                    continue
                await self._process(job)
            except Exception:
                # A crash here used to kill the worker permanently and
                # silently -- next_queued or a DB hiccup would strand every
                # future job in 'queued' forever with no error surfaced.
                log.exception("worker loop iteration failed")
            if self._stop.is_set():
                break

    def _maybe_evict_idle(self) -> None:
        if (
            self.trellis.running
            and time.monotonic() - self.trellis.last_used > self.config.trellis_idle_timeout
        ):
            log.info("evicting idle trellis-server")
            self.trellis.stop()

    # --- progress plumbing ---

    def _emit_progress(
        self,
        phase: str,
        label: str,
        inner: float,
        inner_next: float | None,
        nominal: float,
        fields: dict[str, Any],
    ) -> None:
        """Sink for the trellis parser. Runs on the stdout reader thread."""
        job_id = self.current_job_id
        if job_id is None:
            return  # server chatter outside a job (startup banner, idle logs)
        self.progress.update(
            job_id,
            phase=phase,
            label=label,
            inner=inner,
            inner_next=inner_next,
            nominal=nominal,
            **fields,
        )

    def _t2i_state(self, job_id: str, state: str) -> None:
        phase, label = T2I_PHASES.get(state, ("t2i_load", "Preparing"))
        # Download and load have no measurable inner progress; creep across the
        # whole phase so the bar still moves.
        self.progress.update(
            job_id, phase=phase, label=label, inner=0.0, inner_next=1.0,
            nominal=90.0 if state == "download" else 20.0, detail="",
        )

    def _t2i_step(self, job_id: str, step: int, total: int) -> None:
        self.progress.update(
            job_id,
            phase="t2i_sample",
            label="Drawing reference image",
            inner=step / max(total, 1),
            inner_next=(step + 1) / max(total, 1),
            nominal=1.0,
            detail=f"step {step}/{total}",
            step=step,
            step_total=total,
        )

    async def _process(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        claimed = await asyncio.to_thread(self.store.claim, job_id)
        if not claimed:
            # Cancelled or deleted between next_queued() and here.
            return
        self.current_job_id = job_id
        self._cancel = _Cancel(job_id)
        # A cold trellis server loads ~8 GB inside its first stage; text jobs stop
        # the server outright, so they are always cold.
        self.progress.begin(
            job_id, job["kind"], cold=job["kind"] == "text" or not self.trellis.running
        )
        error: str | None = None
        try:
            await self._generate(job)
        except Exception as exc:
            if not self._cancel.event.is_set():
                log.exception("job %s failed", job_id)
                errors.write_error_log(self.config.job_dir(job_id), exc)
                error = errors.friendly(exc)
        finally:
            if self._cancel.event.is_set():
                await asyncio.to_thread(self.store.set_status, job_id, "cancelled")
                glb = self.config.job_dir(job_id) / "model.glb"
                with contextlib.suppress(OSError):
                    glb.unlink()
            elif error is not None:
                await asyncio.to_thread(self.store.set_status, job_id, "error", error)
            else:
                await asyncio.to_thread(self.store.set_status, job_id, "done")
            self.current_job_id = None
            self._cancel = None
            self.progress.end(job_id)

    async def _generate(self, job: dict[str, Any]) -> None:
        job_dir = self.config.job_dir(job["id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        params = job["params"]
        seed = int(params.get("seed", 42))
        resolution = int(params.get("resolution", 1024))
        image_path = job_dir / "input.png"

        job_id = job["id"]
        assert self._cancel is not None
        if job["kind"] == "text":
            # Free VRAM held by the 3D server, run Flux, then free Flux.
            self.trellis.stop()
            t2i = self._get_text2image()
            try:
                await asyncio.to_thread(
                    functools.partial(
                        t2i.generate,
                        job["prompt"],
                        image_path,
                        seed=seed,
                        on_state=lambda s: self._t2i_state(job_id, s),
                        on_step=lambda i, n: self._t2i_step(job_id, i, n),
                        cancel_event=self._cancel.event,
                    )
                )
            finally:
                await asyncio.to_thread(t2i.unload)
        elif not image_path.exists():
            raise RuntimeError("image job has no uploaded input.png")

        if self._cancel.event.is_set():
            return

        self.progress.update(
            job_id,
            phase="trellis",
            label="Starting 3D engine" if not self.trellis.running else "Sending image",
            inner=0.0,
            inner_next=0.02,
            nominal=6.0,
            detail="",
        )
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
