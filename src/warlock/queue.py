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

from . import errors, guidance, memlog, models, provenance, rigging, vram
from .config import Config
from .db import JobStore
from .pipelines import control, reference
from .pipelines.trellis import TrellisServer
from .progress import ProgressBus, TrellisProgressParser

log = logging.getLogger(__name__)

POLL_INTERVAL = 1.0
SHUTDOWN_TIMEOUT = 20.0

# Labels for the text-to-image phases, which have no trace of their own.
T2I_PHASES = {
    "load": ("t2i_load", "Loading image model"),
    # Attaching a ControlNet/IP-Adapter is a few seconds of loading, so it maps
    # onto the existing load phase rather than adding one -- the progress model
    # and its ETA need no change.
    "condition": ("t2i_load", "Loading conditioning"),
    "sample": ("t2i_sample", "Drawing reference image"),
}

# Whether reference.prepare recentres and rescales the subject, or merely
# measures it. False on purpose: upstream TRELLIS crops and pads in its own
# preprocessing and the vendored exe very likely does too, in which case
# normalising host-side over-zooms. This phase ships the report, the rejection
# rules and reference.png as an audit artifact *before* it ships the
# transform. Flip the default only after a sweep over 3 occupancy values x 3
# subjects scored with meshaudit.hole_fraction -- the same way
# Config.trellis_band was settled.
DEFAULT_REFERENCE_PREP = False


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


def _log_mem(when: str) -> None:
    """Log VRAM *and* host memory at a stage boundary.

    The VRAM invariant (stop-before-load, unload-before-next-start) has one
    failure mode: an OOM that only reproduces under load. Nothing used to
    record whether the memory actually came back, so a regression was
    invisible until a user hit it. These lines are the record.

    The host half was added after the 2026-08-03 crash, which was commit
    exhaustion with the GPU nearly empty -- a failure the VRAM line alone
    could not have shown. Paired at the same boundaries, the two answer the
    question that matters: does memory come back after a job, or only grow.
    """
    mem = vram_gib()
    if mem is not None:
        log.info("vram %s: %.2f GiB allocated, %.2f GiB reserved", when, *mem)
    host = memlog.summary()
    if host is not None:
        log.info("host %s: %s", when, host)
    pressure = commit_fraction()
    if pressure is not None and pressure >= COMMIT_CEILING:
        log.critical(
            "host commit at %.0f%% %s -- at or past the ceiling the "
            "2026-08-03 crash hit; further jobs will be refused",
            pressure * 100,
            when,
        )


COMMIT_CEILING = 0.90
"""System commit fraction past which the worker stops taking jobs.

Windows kills the process (Resource-Exhaustion event 2004) rather than raising
something a job can fail on, so the only useful place to stand is *before* the
next allocation. It needs no watchdog thread: the stage boundaries _log_mem
already runs at are exactly the moments the number can be acted on.
"""


def commit_fraction() -> float | None:
    """System commit charge as a fraction of the limit, or None off Windows."""
    sysmem = memlog.system_memory()
    return None if sysmem is None else sysmem.commit_fraction


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
        # Set by wake() when something enqueues a job, so an idle dispatch loop
        # starts it immediately instead of waiting out the poll interval.
        self._wake = asyncio.Event()
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
            # Log with the traceback, store without it. A stored exception
            # keeps its __traceback__, which keeps every frame of the dying
            # call stack alive -- and those frames hold the pipeline and its
            # tensors. self.fatal is never cleared, so that retention lasts for
            # process life. Only the exception itself is ever read (main.py
            # renders str(fatal)); the traceback belongs in the log.
            #
            # The order is load-bearing and used to be inverted.
            # ``with_traceback`` mutates in place and returns *self*, so
            # stripping first left ``exc_info=exc`` with nothing to format --
            # the single log line for a dead GPU worker carried no stack at all.
            log.critical("gpu worker task died", exc_info=exc)
            self.fatal = exc.with_traceback(None)

    def wake(self) -> None:
        """Tell an idle dispatch loop there is work now.

        Called from the routes that insert a row, on the event loop the worker
        runs on -- so this is a plain Event.set(), not a threadsafe hop. The
        POLL_INTERVAL timeout in _run stays as the backstop: a caller that
        forgets to wake costs a second of latency, not a stuck queue.
        """
        self._wake.set()

    async def _wait_for_work(self) -> None:
        """Sleep until a job is enqueued, shutdown is asked for, or the poll
        interval elapses -- whichever comes first."""
        waiters = [
            asyncio.ensure_future(self._stop.wait()),
            asyncio.ensure_future(self._wake.wait()),
        ]
        try:
            await asyncio.wait(
                waiters, timeout=POLL_INTERVAL, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for waiter in waiters:
                waiter.cancel()
        # Cleared only after the wait, never before it: a wake() that landed
        # while next_queued() was in its thread must still be observed here
        # rather than swallowed into a full poll interval of sleep.
        self._wake.clear()

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
        # Shutdown used to stop trellis and leave SDXL loaded. Harmless when
        # the process exits immediately after -- but shutdown() is also reached
        # on paths that keep the interpreter alive, and the pipeline's several
        # GB of host-pinned staging memory has no other release point.
        if self._text2image is not None and self._text2image.loaded:
            await asyncio.to_thread(self._text2image.unload)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = await asyncio.to_thread(self.store.next_queued)
                if job is None:
                    await self._maybe_evict_idle()
                    await self._wait_for_work()
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

    def _check_resources(self, job: dict[str, Any]) -> None:
        """The dispatch-time half of admission control.

        service.validation.check_vram refuses a job the *card* cannot hold at
        submit time. This catches what has changed since: another application
        that took VRAM in the meantime, or a host whose commit charge has
        climbed to the wall while the job sat in the queue. Failing here costs
        a job; not failing here has cost the machine.
        """
        pressure = commit_fraction()
        if pressure is not None and pressure >= COMMIT_CEILING:
            raise RuntimeError(
                f"host memory is {pressure * 100:.0f}% committed, at or past the "
                f"{COMMIT_CEILING * 100:.0f}% ceiling. Close other applications "
                "or restart Warlock before running this job."
            )
        need = vram.estimate_job(job, exclusive=bool(self.config.vram_exclusive))
        if need <= 0:
            return
        device = vram.device_memory()
        if device is None:
            return
        # Against *free*, not total: whatever we already hold is part of the
        # budget this job runs inside. The WDDM caveat in vram.py applies --
        # this figure is the driver's virtualized view, so it is a secondary
        # check behind the submit-time one, not a replacement for it.
        already = self.trellis.running and job["kind"] in ("text", "image")
        headroom = device.free_gib + (vram.TRELLIS_GIB if already else 0.0)
        if need > headroom:
            raise RuntimeError(
                f"this job needs about {need:.1f} GiB of VRAM and only "
                f"{headroom:.1f} GiB is available. Close other GPU applications "
                "and try again."
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
            self._check_resources(job)
            await self._generate(job)
        except Exception as exc:
            if not self._cancel.event.is_set():
                log.exception("job %s failed", job_id)
                # The verdict first, the log file second. Writing the log can
                # itself raise -- a full or read-only disk is one of the ways
                # a job fails in the first place -- and doing it first left
                # `error` None, so the finally below recorded the job as
                # *done*: a successful-looking job with no model.glb and no
                # message anywhere.
                error = errors.friendly(exc)
                with contextlib.suppress(OSError):
                    errors.write_error_log(self.config.job_dir(job_id), exc)
        finally:
            try:
                if self._cancel.event.is_set():
                    await asyncio.to_thread(self.store.set_status, job_id, "cancelled")
                    self._discard_artifacts(job)
                else:
                    status = "error" if error is not None else "done"
                    finished = await asyncio.to_thread(
                        self.store.finish, job_id, status, error
                    )
                    if not finished:
                        # A cancel landed between claim() succeeding and this
                        # write (before self._cancel existed to observe it, or
                        # the API route's atomic cancel() won the DB race) --
                        # the DB already says cancelled; don't overwrite it and
                        # don't leave a viewable artifact behind.
                        self._discard_artifacts(job)
                    elif status == "done":
                        await self._maybe_queue_rig(job)
            finally:
                # Unconditionally, and in a nest of its own: the terminal write
                # can raise (`database is locked`, a full disk) and these four
                # lines are how the worker lets go of the job. Skipping them
                # left current_job_id naming a job that was no longer running,
                # a stale _Cancel event, and a ProgressBus entry that never
                # ended -- so every later job's trellis output was reported
                # against the dead one, and prune/delete refused to touch it.
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
            # Validated, not merely non-empty: this string becomes a path, and
            # the sibling sheet branch below has always checked its own id the
            # same way. An empty or malformed source_job makes job_dir() return
            # the assets root, so the cleanup would go looking for temps in the
            # directory that holds every job.
            source = str(params.get("source_job") or "")
            if not rigging.is_valid_id(source):
                return
            job_dir = self.config.job_dir(source)
            if job["kind"] == "rig":
                # Only the temps: the worker writes those, and finalize_rig
                # only renames them into rig.glb/rig.json on success. The
                # served names may belong to an earlier, successful rig job
                # (a cancelled re-rig must not destroy the rig it corrects).
                paths = [job_dir / rigging.RIG_GLB_TMP, job_dir / rigging.RIG_JSON_TMP]
            else:
                sheet_id = str(params.get("sheet_id") or "")
                if not rigging.is_valid_id(sheet_id):
                    return
                paths = [
                    rigging.sheet_path(job_dir, sheet_id),
                    rigging.sheet_png_path(job_dir, sheet_id),
                ]
        else:
            # Both halves of the contract: model.glb is what the user would
            # see, source.glb is what it was derived from. Leaving the source
            # behind would let a cancelled job be re-optimized back into
            # existence.
            job_dir = self.config.job_dir(job["id"])
            paths = [
                job_dir / "model.glb",
                job_dir / "source.glb",
                # Both derived from ref.png/input.png by this run. ref.png
                # itself stays: it is a user-supplied input, and keeping it is
                # what makes "cancel, tweak, resubmit" work.
                job_dir / "reference.png",
                job_dir / "control.png",
            ]
        for path in paths:
            with contextlib.suppress(OSError):
                path.unlink()

    async def _conditioning(self, job_dir: Path, params: dict[str, Any], spec):
        """Resolve a job's conditioning selection into a Conditioning, or None.

        Returns None whenever there is nothing to attach, which is what keeps
        the unconditioned path bit-identical: the pipeline is handed None, not
        an empty object it would have to interpret.

        An unknown registry key is dropped with a warning rather than failing
        the job -- the same tolerance base_model and the style LoRAs already
        get, for the same reason: params can predate a rename and the user
        cannot fix a row that already exists.
        """
        from .pipelines.conditioning import Conditioning

        ref = job_dir / "ref.png"
        if not ref.exists():
            return None

        ip_key = params.get("ip_adapter") or None
        if ip_key is not None and ip_key not in models.IP_ADAPTERS:
            log.warning("unknown ip_adapter %r; generating without it", ip_key)
            ip_key = None

        control_key = params.get("control") or None
        if control_key is not None and control_key not in models.CONTROLNETS:
            log.warning("unknown control %r; generating without it", control_key)
            control_key = None
        if control_key is not None and not spec.controlnet:
            # Reachable when a stored base_model fell back to the registry
            # default above, or when a base loses its controlnet flag.
            log.warning(
                "base model %s cannot run a ControlNet; generating without it", spec.key
            )
            control_key = None

        if ip_key is None and control_key is None:
            return None

        control_image = None
        if control_key is not None:
            cn = models.CONTROLNETS[control_key]
            control_image = job_dir / "control.png"
            params["control_hint"] = await asyncio.to_thread(
                functools.partial(
                    control.write_hint,
                    ref,
                    control_image,
                    kind=cn.preprocessor,
                    size=spec.image_size,
                )
            )

        return Conditioning(
            ip_adapter=ip_key,
            ip_image=ref if ip_key else None,
            ip_scale=float(params.get("ip_scale", models.DEFAULT_IP_SCALE)),
            control=control_key,
            control_image=control_image,
            control_scale=float(params.get("control_scale", models.DEFAULT_CONTROL_SCALE)),
            control_end=float(params.get("control_end", models.DEFAULT_CONTROL_END)),
        )

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
        # Two seeds, one per stage, falling back to the single legacy seed so a
        # job row written before the split still reproduces exactly.
        legacy_seed = int(params.get("seed", 42))
        reference_seed = int(params.get("reference_seed", legacy_seed))
        mesh_seed = int(params.get("mesh_seed", legacy_seed))
        resolution = int(params.get("resolution", 1024))
        image_path = job_dir / "input.png"

        job_id = job["id"]
        assert self._cancel is not None
        if job["kind"] == "text":
            base_key = str(params.get("base_model") or self.config.t2i_model)
            if base_key not in models.BASE_MODELS:
                # Params can predate a registry entry being renamed or removed;
                # falling back beats failing a job the user can't fix.
                log.warning("unknown base_model %r; using %s", base_key, self.config.t2i_model)
                base_key = self.config.t2i_model
            if base_key not in models.BASE_MODELS:
                # WARLOCK_T2I_MODEL itself can name a key that doesn't exist;
                # the registry default beats a bare KeyError in the worker.
                log.warning("unknown t2i_model %r; using %s", base_key, models.DEFAULT_BASE_MODEL)
                base_key = models.DEFAULT_BASE_MODEL
            style_lora = params.get("style_lora") or None
            lora_weight = float(params.get("lora_weight", models.DEFAULT_LORA_WEIGHT))
            spec = models.BASE_MODELS[base_key]
            cond = await self._conditioning(job_dir, params, spec)
            # A fully conditioned CFG job wants ~6 GB over the unconditioned
            # budget (a ControlNet plus the CLIP-ViT-H encoder), which does not
            # fit beside a resident trellis on a 32 GB card. A reference-stage
            # job -- where the UI actually offers conditioning -- is unaffected,
            # because trellis is not involved in it at all.
            # The old form of this test exempted stage == "reference", on the
            # reasoning that trellis is not involved in a reference job. But
            # trellis stays *resident* for trellis_idle_timeout (600 s) after
            # the previous model job, and the reference stage is the only path
            # the UI offers conditioning from (studio/panes/settings_2d.py) --
            # so the exemption fired on exactly the jobs it was meant to
            # protect. Ask whether trellis is actually holding VRAM instead.
            handoff = self.config.vram_exclusive or (
                cond is not None and self.trellis.running
            )
            if handoff:
                # Sequential handoff: both models can't fit -- free the VRAM
                # held by the 3D server before the image model loads. Threaded
                # because stop() can block for up to ~20 s, and this fires at
                # the exact moment the user starts watching the progress bar.
                await asyncio.to_thread(self.trellis.stop)
                _log_mem("after trellis stop")
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
                        seed=reference_seed,
                        lora=style_lora,
                        lora_weight=lora_weight,
                        negative_prompt=str(params.get("negative_prompt") or ""),
                        conditioning=cond,
                        on_state=lambda s: self._t2i_state(job_id, s),
                        on_step=lambda i, n: self._t2i_step(job_id, i, n),
                        cancel_event=self._cancel.event,
                    )
                )
                params["composed_prompt"] = t2i.last_prompt or composed
                if job.get("stage") == "reference":
                    # Measure only, and never a rejection: the user is judging
                    # the image, and the mesh stage is where the cost is. This
                    # is what promote_to_model's soft check reads.
                    params["reference_report"] = (
                        await asyncio.to_thread(reference.measure_file, image_path)
                    ).as_dict()
                if t2i.last_recipe:
                    params.setdefault("recipe", {})["reference"] = t2i.last_recipe
                await asyncio.to_thread(self.store.set_params, job_id, params)
            finally:
                if handoff:
                    await asyncio.to_thread(t2i.unload)
                else:
                    # Coexist mode keeps the pipeline resident by design, which
                    # is why nothing here used to run at all -- the teardown is
                    # a no-op closure. But "stays loaded" was silently also
                    # meaning "never returns its caching allocator pool", and
                    # under WDDM those device blocks are charged against system
                    # commit. trim() gives the pool back without unloading
                    # anything, so the VRAM-modes invariant is untouched.
                    await asyncio.to_thread(t2i.trim)

            if job.get("stage") == "reference":
                # The whole point of the split: the user judges the image before
                # anything pays for a trellis run. The job is finished here --
                # promotion creates a separate child job (app.promote_to_model).
                _log_mem("after reference-only job")
                return
        elif not image_path.exists():
            raise RuntimeError("image job has no uploaded input.png")

        if job.get("stage") == "reference":
            # The text branch returns here itself, having just generated the
            # image. This catches the other kind: a reference whose pixels came
            # from an upload or the paint editor. Its stage says the job stops
            # before the mesh, and honouring that is a property of the *stage*,
            # not of how the image was made -- reading it off the kind is how a
            # painted reference used to buy an unasked-for trellis run.
            _log_mem("after reference-only job")
            return

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
        # The reconstruction is kept verbatim as source.glb and never
        # overwritten: model.glb is derived from it, so re-targeting a triangle
        # budget later (POST /api/jobs/{id}/optimize) never has to pay for
        # another trellis run.
        source_glb = job_dir / "source.glb"
        glb_path = job_dir / "model.glb"

        # What trellis actually sees, measured and recorded before it is
        # uploaded. The rejection happens *here*, not after: a reference that
        # cannot reconstruct should cost the request, not two minutes of GPU.
        trellis_input = job_dir / "reference.png"
        report = await asyncio.to_thread(
            functools.partial(
                reference.prepare,
                image_path,
                trellis_input,
                enabled=bool(params.get("reference_prep", DEFAULT_REFERENCE_PREP)),
            )
        )
        params["reference_report"] = report.as_dict()
        params.setdefault("recipe", {})["trellis"] = provenance.trellis_recipe(
            self.config, params, mesh_seed=mesh_seed
        )
        await asyncio.to_thread(self.store.set_params, job_id, params)
        if not report.ok:
            raise RuntimeError("; ".join(report.reasons))

        _log_mem("before trellis generate")
        await self.trellis.generate(
            trellis_input,
            source_glb,
            seed=mesh_seed,
            resolution=resolution,
            bg_removal=str(params.get("bg_removal") or "auto"),
        )
        await self._optimize(job_id, source_glb, glb_path, params)
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
        # Validated before it becomes a path, the same way sheet_id already is.
        # This is the directory the Blender worker is pointed at and writes its
        # temps into, and job_dir("") is the assets root -- the whole reason
        # check_job_id and pose_path exist is to keep params off the filesystem
        # unchecked.
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
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

        spec = rigging.rig_spec(source_dir, template, params.get("bones"))
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    rigging.run_worker,
                    spec,
                    on_progress=on_progress,
                    on_start=on_start,
                    timeout=self.config.rig_timeout,
                )
            )
            # A cancel that landed after the solve finished must not publish
            # the artifacts of a job about to be recorded as cancelled -- the
            # finally below throws the temps away instead.
            if self._cancel is not None and self._cancel.event.is_set():
                # Nothing was published, so nothing gets recorded either: the
                # weighting/bone_count of a discarded rig must not end up in
                # the params of a job recorded as cancelled.
                return
            await asyncio.to_thread(rigging.finalize_rig, source_dir)
        finally:
            # No-op on success (finalize renamed them away); on failure or
            # cancel it removes the half-written temps and never touches the
            # served rig.glb/rig.json, which may belong to an earlier,
            # successful rig job.
            await asyncio.to_thread(rigging.discard_rig_temps, source_dir)
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
        # Validated before it becomes a path, the same way sheet_id already is.
        # This is the directory the Blender worker is pointed at and writes its
        # temps into, and job_dir("") is the assets root -- the whole reason
        # check_job_id and pose_path exist is to keep params off the filesystem
        # unchecked.
        if not rigging.is_valid_id(source_id):
            raise ValueError(f"source_job is not a job id: {source_id!r}")
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

        clip = params.get("clip")
        if clip:
            # Rebuilt from the same two poses rather than shipped in params: the
            # host is the single place a grid or a clip is decided (see
            # pipelines/sheet.py), and storing the expanded frames would be a
            # second copy that could disagree with it.
            ends = [
                await asyncio.to_thread(rigging.read_pose, source_dir, str(clip[k]))
                for k in ("from", "to")
            ]
            if any(e is None for e in ends):
                raise RuntimeError("a pose in this clip no longer exists")
            records = sheetlib.interpolate(ends[0], ends[1], int(clip["frames"]))

        layout = sheetlib.plan(
            records,
            frame_size=int(params.get("frame_size", sheetlib.DEFAULT_FRAME_SIZE)),
            elevation=float(params.get("elevation", sheetlib.DEFAULT_ELEVATION)),
            lighting=str(params.get("lighting", "flat")),
            yaws=int(params.get("yaws", sheetlib.DEFAULT_YAWS)),
        )
        # A rig if there is one, so poses can apply; otherwise the plain mesh,
        # which is all an unrigged prop's turnaround needs.
        source_glb = source_dir / "rig.glb"
        if not source_glb.exists():
            source_glb = source_dir / "model.glb"
        if not source_glb.exists():
            raise RuntimeError("source job has no mesh to render")

        # Keyed by (pose id, frame) rather than pose id: every frame of a clip
        # shares an id by construction, and keying on the id alone would render
        # frame 0 in every row of the clip.
        bones = {(r.get("id"), r.get("frame", 0)): r["bones"] for r in records}
        cells = [
            {
                "index": c.index,
                "yaw": c.yaw,
                "pose": c.pose,
                "frame": c.frame,
                "bones": bones.get((c.pose, c.frame)) or {},
            }
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
            result = await asyncio.to_thread(
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
            trims = await asyncio.to_thread(sheetlib.pack, layout, frames, png)

        # The sidecar is written last and is what list_sheets treats as the
        # completion marker, the same way rig.json is for a rig.
        #
        # A worker that reported no pivot (an older result, or a fake in a test)
        # falls back to the cell's centre-bottom rather than failing the sheet:
        # the atlas is already on disk and correct, and the pivot is metadata.
        pivot = result.get("pivot") if isinstance(result, dict) else None
        meta = sheetlib.sidecar(
            layout,
            sheet_id=sheet_id,
            source_job=source_id,
            image=png.name,
            created=time.time(),
            name=str(params.get("name") or ""),
            pivot=(float(pivot[0]), float(pivot[1])) if pivot else None,
            trims=trims,
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

    async def _optimize(
        self, job_id: str, source: Path, dest: Path, params: dict[str, Any]
    ) -> None:
        """Retarget the reconstruction to the job's triangle budget.

        Before the transform, not after: gltfpack rewrites the node graph, and
        running it over an already-grounded model would discard the transform
        node normalize_glb inserted. Optimizing first and transforming second is
        the only ordering where both survive.

        A failure here is not fatal. The reconstruction is on disk and usable;
        losing the budget costs the user file size, and failing the job would
        cost them the mesh.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        from .pipelines import optimize

        try:
            budget = optimize.resolve(
                str(params.get("profile") or self.config.mesh_profile),
                params.get("custom_triangles"),
            )
        except ValueError:
            log.warning("job %s has an unusable profile; shipping raw", job_id)
            budget = None
        self.progress.update(
            job_id, phase="optimize", label="Optimizing mesh", inner=0.0,
            inner_next=1.0, nominal=4.0, detail=f"{budget:,} tris" if budget else "raw",
        )
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    optimize.run,
                    source,
                    dest,
                    target_triangles=budget,
                    exe=self.config.gltfpack_exe,
                )
            )
        except Exception:
            log.exception("optimize failed for job %s; shipping the reconstruction", job_id)
            # Staged, not copyfile: on the /optimize route dest is a done
            # job's model.glb that a concurrent GET may be serving.
            await asyncio.to_thread(optimize.staged_copy, source, dest)
            return
        params["optimize"] = result
        await asyncio.to_thread(self.store.set_params, job_id, params)

    async def _apply_scale(
        self, job_id: str, glb_path: Path, params: dict[str, Any]
    ) -> None:
        """Resize the finished mesh, centre it in X/Z and sit it on the floor.

        The grounding half runs even when no ``size_m`` was requested -- a
        pivot at the reconstruction volume's centre is a manual fixup on every
        import regardless of whether the asset also needed rescaling.

        Runs after the GLB is on disk and holds no GPU memory, so it sits
        outside the VRAM handoff entirely. A cancel that landed during the
        trellis stage skips it -- the artifact is about to be deleted anyway.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        size_m = params.get("size_m")
        self.progress.update(
            job_id,
            phase="scale",
            label="Scaling and grounding",
            inner=0.0,
            inner_next=1.0,
            nominal=2.0,
            detail=f"{size_m} m" if size_m else "grounding",
        )
        # Imported here, not at module scope: postprocess pulls in trimesh,
        # which app startup should not pay for.
        from .pipelines import postprocess

        try:
            transform = await asyncio.to_thread(
                postprocess.normalize_glb, glb_path, float(size_m) if size_m else None
            )
        except Exception:
            # Grounding runs on every job now, not just sized ones, so a mesh
            # trimesh cannot parse must not fail a job whose GLB is already on
            # disk -- the same rule _audit_mesh and the report follow. The
            # report's achieved_size_m is what tells the user the size did not
            # land, rather than a job that errored after producing a model.
            log.exception("normalize failed for job %s; leaving the mesh as-is", job_id)
            return
        params["scale_factor"] = transform["scale"]
        params["transform"] = transform
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
        # The silhouette number stays exactly as it was -- it is the only thing
        # that catches trellis's disconnected-plate crust. The report adds what
        # the silhouette cannot see: topology, materials, budget, and whether
        # the thing will sit on an engine's floor.
        try:
            from . import meshreport

            params["mesh_report"] = await asyncio.to_thread(
                functools.partial(
                    meshreport.build,
                    glb_path,
                    target_size_m=params.get("size_m"),
                    silhouette=params["mesh_audit"],
                )
            )
        except Exception:
            log.exception("mesh report failed for job %s", job_id)
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
