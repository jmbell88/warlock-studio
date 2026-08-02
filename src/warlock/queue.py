"""Single-worker GPU job queue.

One job runs at a time. By default SDXL-Turbo (~7 GB) and the trellis server
(~16 GB) coexist in VRAM — neither is stopped for the other, and both are
evicted after the idle timeout. With Config.vram_exclusive set
(WARLOCK_VRAM_EXCLUSIVE=1, for small cards or a resident Flux), text jobs
instead use the sequential handoff: the trellis server is stopped before the
image model loads, and the image model is unloaded before trellis restarts.

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
import json
import logging
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import errors, guidance, models, rigging
from .config import Config
from .db import JobStore
from .pipelines.trellis import TrellisServer
from .progress import ProgressBus, TrellisProgressParser

log = logging.getLogger(__name__)

POLL_INTERVAL = 1.0
SHUTDOWN_TIMEOUT = 20.0

# Labels for the text-to-image phases, which have no trace of their own.
T2I_PHASES = {
    "load": ("t2i_load", "Loading image model"),
    "sample": ("t2i_sample", "Drawing reference image"),
}


@dataclass
class _Cancel:
    job_id: str
    event: threading.Event = field(default_factory=threading.Event)


def vram_gib() -> tuple[float, float] | None:
    """-> (allocated, reserved) GiB held by *this process's* torch, or None.

    Looked up through sys.modules rather than imported. Two reasons, both
    load-bearing:

    * Importing torch takes seconds, and this is called from the event loop.
      An image job on a machine with torch installed but not yet loaded would
      stall the whole app just to log a number.
    * The number would be zero anyway. memory_allocated() only sees this
      process's allocations, so it is meaningful exactly when the SDXL
      pipeline is loaded -- which is precisely when torch is already imported.
      trellis-server's ~16 GB lives in a separate process and never appears
      here regardless.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    gib = 1024**3
    return (torch.cuda.memory_allocated() / gib, torch.cuda.memory_reserved() / gib)


def _log_vram(when: str) -> None:
    """Log VRAM at a handoff boundary.

    The VRAM invariant (stop-before-load, unload-before-next-start) has one
    failure mode: an OOM that only reproduces under load. Nothing used to
    record whether the memory actually came back, so a regression was
    invisible until a user hit it. These lines are the record.
    """
    mem = vram_gib()
    if mem is not None:
        log.info("vram %s: %.2f GiB allocated, %.2f GiB reserved", when, *mem)


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
            tex_res=config.trellis_tex_res,
            band=config.trellis_band,
        )
        self._text2image = None  # lazy: torch/diffusers may not be installed
        # Which base model the resident pipe is, so _get_text2image can tell a
        # cache hit from a swap.
        self._t2i_key: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self.current_job_id: str | None = None
        self._cancel: _Cancel | None = None
        # The live Blender subprocess, if a rig job is running. Set from the
        # worker thread inside run_worker's on_start and read from the event
        # loop in request_cancel -- a plain attribute assignment either way,
        # and only ever non-None for the one job the queue is running.
        self._blender: Any = None
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
        elif phase in ("rig", "sheet"):
            # Same story as trellis: bpy is inside a C weighting solve (or an
            # EEVEE render) and checks nothing, so killing the subprocess is
            # the only abort.
            proc = self._blender
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(OSError):
                    proc.kill()
        # t2i_sample: the diffusers step callback checks the event itself.
        # t2i_load: not interruptible; the event is checked once between
        # load() and sampling in Text2Image.generate().

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
                    await self._maybe_evict_idle()
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

    async def _maybe_evict_idle(self) -> None:
        # Both evictions go through a thread. The queue being idle does not
        # make them cheap: stop() blocks for up to ~20 s if the server ignores
        # SIGTERM, and unload() pays a gc.collect() plus empty_cache(). On the
        # event loop either one freezes /api/progress and every other route.
        if (
            self.trellis.running
            and time.monotonic() - self.trellis.last_used > self.config.trellis_idle_timeout
        ):
            log.info("evicting idle trellis-server")
            await asyncio.to_thread(self.trellis.stop)
        # In exclusive mode the per-job finally already unloaded it, so
        # loaded is never True here and this branch is inert.
        if (
            self._text2image is not None
            and self._text2image.loaded
            and time.monotonic() - self._text2image.last_used > self.config.trellis_idle_timeout
        ):
            log.info("evicting idle SDXL pipeline")
            await asyncio.to_thread(self._text2image.unload)

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
        # Loading has no measurable inner progress; creep across the whole
        # phase so the bar still moves.
        self.progress.update(
            job_id, phase=phase, label=label, inner=0.0, inner_next=1.0,
            nominal=20.0, detail="",
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
        # A cold trellis server loads ~8 GB inside its first stage. Only
        # exclusive-mode text jobs stop the server outright; coexist-mode
        # text jobs leave a warm server warm. A rig job never touches trellis,
        # so it is never cold regardless of the server's state.
        cold = job["kind"] in ("text", "image") and (
            not self.trellis.running
            or (self.config.vram_exclusive and job["kind"] == "text")
        )
        self.progress.begin(job_id, job["kind"], cold=cold)
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
                self._discard_artifacts(job)
            else:
                status = "error" if error is not None else "done"
                finished = await asyncio.to_thread(self.store.finish, job_id, status, error)
                if not finished:
                    # A cancel landed between claim() succeeding and this
                    # write (before self._cancel existed to observe it, or
                    # the API route's atomic cancel() won the DB race) --
                    # the DB already says cancelled; don't overwrite it and
                    # don't leave a viewable artifact behind.
                    self._discard_artifacts(job)
                elif status == "done":
                    await self._maybe_queue_rig(job)
            self.current_job_id = None
            self._cancel = None
            self._blender = None
            self.progress.end(job_id)

    async def _maybe_queue_rig(self, job: dict[str, Any]) -> None:
        """Honour the generate form's "rig this" checkbox, once the mesh exists.

        Queued as an ordinary follow-up job rather than run inline: rigging is
        minutes of CPU that the user should be able to cancel independently,
        and chaining it inside the generate job would hide it from the history
        and make one cancel ambiguous between two very different operations.
        """
        params = job["params"]
        if job["kind"] == "rig" or not params.get("rig"):
            return
        if not (self.config.job_dir(job["id"]) / "model.glb").exists():
            return
        rig_params = {
            "source_job": job["id"],
            "template": params.get("rig_template") or self.config.rig_template,
            "auto": True,
        }
        try:
            rig_id = await asyncio.to_thread(
                self.store.create, "rig", job["prompt"], rig_params, None
            )
        except Exception:
            # The mesh is finished and on disk. A failure to queue the optional
            # follow-up must not retroactively fail the job that produced it.
            log.exception("could not queue follow-up rig for job %s", job["id"])
            return
        log.info("queued follow-up rig %s for job %s", rig_id, job["id"])

    def _discard_artifacts(self, job: dict[str, Any]) -> None:
        """Remove what a cancelled job half-wrote -- and only that.

        Keyed on the job's kind rather than a fixed filename: a rig job's
        output is rig.glb sitting *next to* the model.glb it read, and deleting
        model.glb there would destroy the finished mesh of a different,
        successful job because the user cancelled a rig.
        """
        params = job["params"]
        if job["kind"] in ("rig", "sheet"):
            # Both write into the *source* job's directory, not their own --
            # see _rig and _sheet. Without a source_job there is nothing they
            # could have written, so there is nothing to undo.
            source = params.get("source_job")
            if not source:
                return
            job_dir = self.config.job_dir(str(source))
            if job["kind"] == "rig":
                paths = [job_dir / "rig.glb", job_dir / "rig.json"]
            else:
                sheet_id = str(params.get("sheet_id") or "")
                if not rigging.is_valid_id(sheet_id):
                    return
                paths = [
                    rigging.sheet_path(job_dir, sheet_id),
                    rigging.sheet_png_path(job_dir, sheet_id),
                ]
        else:
            paths = [self.config.job_dir(job["id"]) / "model.glb"]
        for path in paths:
            with contextlib.suppress(OSError):
                path.unlink()

    async def _generate(self, job: dict[str, Any]) -> None:
        if job["kind"] == "rig":
            await self._rig(job)
            return
        if job["kind"] == "sheet":
            await self._sheet(job)
            return
        job_dir = self.config.job_dir(job["id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        params = job["params"]
        seed = int(params.get("seed", 42))
        resolution = int(params.get("resolution", 1024))
        image_path = job_dir / "input.png"

        job_id = job["id"]
        assert self._cancel is not None
        if job["kind"] == "text":
            if self.config.vram_exclusive:
                # Sequential handoff: both models can't fit -- free the VRAM
                # held by the 3D server before the image model loads. Threaded
                # because stop() can block for up to ~20 s, and this fires at
                # the exact moment the user starts watching the progress bar.
                await asyncio.to_thread(self.trellis.stop)
                _log_vram("after trellis stop")
            base_key = str(params.get("base_model") or self.config.t2i_model)
            if base_key not in models.BASE_MODELS:
                # Params can predate a registry entry being renamed or removed;
                # falling back beats failing a job the user can't fix.
                log.warning("unknown base_model %r; using %s", base_key, self.config.t2i_model)
                base_key = self.config.t2i_model
            style_lora = params.get("style_lora") or None
            lora_weight = float(params.get("lora_weight", models.DEFAULT_LORA_WEIGHT))
            # Before trellis restarts in exclusive mode and before SDXL loads:
            # a base switch frees the previous 7 GB pipe, and doing it here
            # keeps the stop-before-load ordering intact either way.
            t2i = await self._get_text2image(base_key)
            # The guidance fragments steer the subject; text2image's own
            # PROMPT_TEMPLATE still wraps this with the TRELLIS-friendly
            # single-object framing.
            composed = guidance.compose_prompt(job["prompt"] or "", params)
            try:
                await asyncio.to_thread(
                    functools.partial(
                        t2i.generate,
                        composed,
                        image_path,
                        seed=seed,
                        lora=style_lora,
                        lora_weight=lora_weight,
                        on_state=lambda s: self._t2i_state(job_id, s),
                        on_step=lambda i, n: self._t2i_step(job_id, i, n),
                        cancel_event=self._cancel.event,
                    )
                )
                params["composed_prompt"] = composed
                await asyncio.to_thread(self.store.set_params, job_id, params)
            finally:
                if self.config.vram_exclusive:
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
        glb_path = job_dir / "model.glb"
        _log_vram("before trellis generate")
        await self.trellis.generate(
            image_path,
            glb_path,
            seed=seed,
            resolution=resolution,
            bg_removal=str(params.get("bg_removal") or "auto"),
        )
        await self._apply_scale(job_id, glb_path, params)
        await self._audit_mesh(job_id, glb_path, params)

    async def _rig(self, job: dict[str, Any]) -> None:
        """Fit a skeleton to a finished job's mesh, out-of-process in Blender.

        A rig job is a job in its own right, but its artifacts land in the
        *source* job's directory alongside model.glb -- the rig belongs to the
        mesh, not to the request that produced it, and a UI that had to chase a
        second job id to find rig.glb would be the wrong shape. params carries
        source_job so the history list can attach it to the parent card.

        Nothing here touches the GPU or either resident model, so the VRAM
        handoff in _generate is not involved. The queue being serial is what
        keeps a multi-minute weighting solve from overlapping a trellis run.
        """
        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        source_dir = self.config.job_dir(source_id)
        template = str(params.get("template") or self.config.rig_template)

        self.progress.update(
            job_id, phase="rig", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=12.0, detail="",
        )

        def on_progress(frac: float, label: str) -> None:
            # The worker's fractions are already the whole bar (PHASES_RIG),
            # and it emits them from its own thread-free subprocess; this
            # runs on the to_thread worker, which ProgressBus locks for.
            self.progress.update(
                job_id, phase="rig", label=label, inner=frac,
                inner_next=min(frac + 0.1, 1.0), nominal=30.0, detail="",
            )

        def on_start(proc: Any) -> None:
            self._blender = proc

        spec = rigging.rig_spec(source_dir, template)
        result = await asyncio.to_thread(
            functools.partial(
                rigging.run_worker,
                spec,
                on_progress=on_progress,
                on_start=on_start,
                timeout=self.config.rig_timeout,
            )
        )
        # Recorded on the rig job so the history row can say "envelope
        # weights" without the UI having to fetch rig.json for every card.
        params["weighting"] = result.get("weighting")
        params["bone_count"] = result.get("bones")
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info(
            "rigged job %s from %s: %s weights, %s bones",
            job_id, source_id, result.get("weighting"), result.get("bones"),
        )

    async def _sheet(self, job: dict[str, Any]) -> None:
        """Render a pose x direction sprite sheet for a finished job's mesh.

        Like a rig, the output belongs to the mesh and lands in the *source*
        job's directory (``sheets/<sheet_id>.png`` plus its sidecar). The grid
        and the packing are pure host-side code in ``pipelines/sheet.py``;
        Blender only ever renders one square transparent frame per cell into a
        scratch directory that goes away either way.

        CPU and an EEVEE render, so nothing here touches the resident models or
        the VRAM handoff -- the serial queue is what keeps it from overlapping
        a trellis run.
        """
        from .pipelines import sheet as sheetlib

        job_id = job["id"]
        params = job["params"]
        source_id = str(params.get("source_job") or "")
        source_dir = self.config.job_dir(source_id)
        sheet_id = str(params.get("sheet_id") or rigging.new_id())

        records = []
        for pose_id in params.get("poses") or []:
            record = await asyncio.to_thread(rigging.read_pose, source_dir, str(pose_id))
            if record is None:
                # Deleted between queueing and running. Failing beats quietly
                # rendering a sheet with a row missing that the user asked for.
                raise RuntimeError(f"pose {pose_id} no longer exists")
            records.append(record)

        layout = sheetlib.plan(
            records,
            frame_size=int(params.get("frame_size", sheetlib.DEFAULT_FRAME_SIZE)),
            elevation=float(params.get("elevation", sheetlib.DEFAULT_ELEVATION)),
            lighting=str(params.get("lighting", "flat")),
        )
        # A rig if there is one, so poses can apply; otherwise the plain mesh,
        # which is all an unrigged prop's turnaround needs.
        source_glb = source_dir / "rig.glb"
        if not source_glb.exists():
            source_glb = source_dir / "model.glb"
        if not source_glb.exists():
            raise RuntimeError("source job has no mesh to render")

        bones = {r["id"]: r["bones"] for r in records}
        cells = [
            {"index": c.index, "yaw": c.yaw, "pose": c.pose, "bones": bones.get(c.pose) or {}}
            for c in layout.cells
        ]

        self.progress.update(
            job_id, phase="sheet", label="Starting Blender", inner=0.0,
            inner_next=0.05, nominal=12.0, detail=f"{len(cells)} frames",
        )

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="sheet", label=label, inner=frac,
                inner_next=min(frac + 0.05, 1.0), nominal=20.0, detail="",
            )

        def on_start(proc: Any) -> None:
            self._blender = proc

        png = rigging.sheet_png_path(source_dir, sheet_id)
        with tempfile.TemporaryDirectory(prefix="a3d-sheet-") as tmp:
            frames_dir = Path(tmp)
            spec = rigging.sheet_spec(
                source_glb,
                frames_dir,
                cells,
                frame_size=layout.frame_size,
                elevation=layout.elevation,
                lighting=layout.lighting,
            )
            await asyncio.to_thread(
                functools.partial(
                    rigging.run_worker,
                    spec,
                    on_progress=on_progress,
                    on_start=on_start,
                    timeout=self.config.sheet_timeout,
                )
            )
            self.progress.update(
                job_id, phase="pack", label="Packing sheet", inner=0.0,
                inner_next=1.0, nominal=3.0, detail="",
            )
            frames = {c.index: frames_dir / f"{c.index:04d}.png" for c in layout.cells}
            await asyncio.to_thread(sheetlib.pack, layout, frames, png)

        # The sidecar is written last and is what list_sheets treats as the
        # completion marker, the same way rig.json is for a rig.
        meta = sheetlib.sidecar(
            layout,
            sheet_id=sheet_id,
            source_job=source_id,
            image=png.name,
            created=time.time(),
            name=str(params.get("name") or ""),
        )
        await asyncio.to_thread(
            rigging.sheet_path(source_dir, sheet_id).write_text,
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        params["sheet_id"] = sheet_id
        params["cells"] = len(cells)
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info(
            "rendered sheet %s for job %s: %dx%d cells at %dpx",
            sheet_id, source_id, layout.columns, layout.rows, layout.frame_size,
        )

    async def _apply_scale(
        self, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> None:
        """Resize the finished mesh to the requested real-world size.

        Runs after the GLB is on disk and holds no GPU memory, so it sits
        outside the VRAM handoff entirely. A cancel that landed during the
        trellis stage skips it -- the artifact is about to be deleted anyway.
        """
        size_m = params.get("size_m")
        if not size_m or (self._cancel is not None and self._cancel.event.is_set()):
            return
        self.progress.update(
            job_id,
            phase="scale",
            label="Scaling to target size",
            inner=0.0,
            inner_next=1.0,
            nominal=2.0,
            detail=f"{size_m} m",
        )
        # Imported here, not at module scope: postprocess pulls in trimesh,
        # which app startup should not pay for.
        from .pipelines import postprocess

        factor = await asyncio.to_thread(postprocess.scale_glb, glb_path, float(size_m))
        params["scale_factor"] = factor
        await asyncio.to_thread(self.store.set_params, job_id, params)

    async def _audit_mesh(
        self, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> None:
        """Measure how see-through the finished mesh is and record it.

        trellis-server's narrow-band remesh can emit a crust of disconnected
        plates that passes every integrity check while being visibly
        perforated, and the user currently finds that out in Blender. Measuring
        it here turns it into something they see on the job the moment it
        finishes. Runs after _apply_scale so it measures the mesh that will
        actually be downloaded; like scaling it is CPU-only and holds no GPU
        memory, so it sits outside the VRAM handoff.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        self.progress.update(
            job_id,
            phase="audit",
            label="Checking mesh",
            inner=0.0,
            inner_next=1.0,
            nominal=6.0,
            detail="",
        )
        try:
            # Imported here for the same reason as postprocess above: numpy and
            # trimesh at module scope would land in app startup.
            from . import meshaudit

            report = await asyncio.to_thread(
                meshaudit.hole_fraction,
                glb_path,
                meshaudit.DEFAULT_VIEWS,
                meshaudit.REQUEST_PATH_RESOLUTION,
            )
        except Exception:
            # A diagnostic must never be able to fail a job whose mesh is
            # already on disk and fine. Log it and leave mesh_audit unset --
            # the UI renders no badge rather than a wrong one.
            log.exception("mesh audit failed for job %s", job_id)
            return
        # Only the summary is stored: the per-view detail would ride along on
        # every row of the 100-job list for no one to read.
        params["mesh_audit"] = {
            "worst": report["worst"],
            "mean": report["mean"],
            "faces": report["faces"],
            "resolution": report["resolution"],
        }
        log.info(
            "job %s mesh audit: worst %.3f, mean %.3f over %d faces",
            job_id, report["worst"], report["mean"], report["faces"],
        )
        await asyncio.to_thread(self.store.set_params, job_id, params)

    async def _get_text2image(self, base_key: str):
        """The resident image pipeline, swapped if the job wants a different base.

        One base model at a time, not one per key: a 32 GB card holds
        trellis-server (~16 GB) plus a single SDXL-class pipe (~7 GB) and not
        two of the latter, so a switch has to free the old one first. Style
        LoRAs are the opposite -- they are adapters on the resident pipe and
        switch per job with no reload at all (text2image._apply_adapters).

        The unload goes through a thread for the same reason every other
        unload() call site does: gc.collect() plus empty_cache() on the event
        loop stalls /api/progress for every other client.
        """
        if self._text2image is not None and self._t2i_key != base_key:
            log.info("switching image model %s -> %s", self._t2i_key, base_key)
            await asyncio.to_thread(self._text2image.unload)
            self._text2image = None
        if self._text2image is None:
            try:
                from .pipelines.text2image import Text2Image
            except ImportError as exc:
                raise RuntimeError(
                    "text-to-3D requires the text2image extra: uv sync --extra text2image"
                ) from exc
            spec = models.BASE_MODELS[base_key]
            self._text2image = Text2Image(
                spec,
                self.config.t2i_model_root,
                # Honoured for turbo only -- see Config.t2i_turbo_dir.
                self.config.t2i_turbo_dir if base_key == models.DEFAULT_BASE_MODEL else None,
            )
            self._t2i_key = base_key
        return self._text2image
