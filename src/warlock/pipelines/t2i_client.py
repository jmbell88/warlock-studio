"""The app-process handle on the image pipeline that now lives in a child.

``Text2ImageClient`` presents the surface ``Text2Image`` presented -- ``load``,
``generate``, ``trim``, ``unload``, ``close``, ``loaded``, ``last_used``,
``last_prompt``, ``last_recipe``, ``spec``, ``model_dir`` -- so ``queue.Worker``
and everything that reads a resident pipe carry on unchanged. What changes is
where the weights live and, decisively, what ``unload()`` costs: a process kill
rather than a ``gc.collect()`` that measurement showed cannot work.

**The number this exists for.** ``flux_klein_distilled`` charged +21.1 GiB of
host commit on 2026-08-22 and ``unload()`` returned 0.1 GiB of it, because what
holds the rest is the allocator's arenas rather than a live object. The session
ended with the app refusing its own follow-up job at 94% commit. Killing the
process returns all of it, which is the trade ``matting_worker`` and
``loadprobe`` already made; the measurement carries the two options not
taken.

**Silence, not duration, is the timeout.** A 50-step CFG sample legitimately
takes minutes and a cold klein load takes tens of seconds, so a total-duration
cap would either be uselessly large or would kill working jobs. The child emits
a progress line per diffusion step and a state line per stage, so the question
that can actually be asked is "has it said anything lately" -- and the deadline
is reset by every line, including ones this method only forwards.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue as _queue
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import models, vram, winjob

log = logging.getLogger(__name__)

CHILD_ARGV = [sys.executable, "-m", "warlock.pipelines.text2image_worker"]

SILENCE_TIMEOUT = 900.0
"""Seconds of *no output at all* before the child is presumed hung.

Deliberately generous. The failure this guards against is a child that has
stopped talking -- a hard CUDA fault, a driver reset -- and not a slow job,
because a slow job is still emitting steps. Fifteen minutes is longer than any
single stage the pipeline has, and short enough that a wedged child does not
hold the queue for a session.
"""

STOP_TIMEOUT = 15.0
"""How long a killed child is waited for.

Longer than matting's 5 s: this one may be inside a CUDA teardown, and the
whole point of the kill is that its memory is back when the wait returns. In
exclusive mode trellis-server restarts immediately afterwards.
"""

READY_TIMEOUT = 120.0
"""How long the child has to say ``ready``.

It is a Python import and no weight read, but it is *this* package's import on a
cold file cache. A child that never says it is failing at startup -- a missing
extra, a broken venv -- and must be reported as that rather than as a hang in
the first generate.
"""


class ChildFailed(RuntimeError):
    """The child could not serve a request. Carries the child's own message."""


class Text2ImageClient:
    """One resident child, and the requests that keep it warm."""

    def __init__(
        self, spec: models.BaseModel, model_root: Path, model_dir: Path | None = None
    ) -> None:
        self.spec = spec
        self._model_root = model_root
        self._model_dir = model_dir or (model_root / spec.dir_name)
        self._proc: subprocess.Popen[str] | None = None
        self._lines: Any = None
        # Serialises the whole exchange, not just the write: the protocol is one
        # terminal response per request with nothing to correlate them by, so
        # two callers interleaving would each read the other's answer. Matting's
        # reasoning, and it applies here with a longer exposure.
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._loaded = False
        self.last_used: float = 0.0
        self.last_prompt: str = ""
        self.last_recipe: dict[str, Any] = {}

    # --- the surface Text2Image had ------------------------------------------

    @property
    def loaded(self) -> bool:
        """Whether the child holds a loaded pipe.

        A parent-side mirror rather than a question asked over the wire: it is
        read from the event loop (``_evict_stale_t2i``, ``_check_resources``,
        the dispatch credit), where a round trip to a child that is mid-sample
        would block the frame the app draws.
        """
        return self._loaded and self._proc is not None and self._proc.poll() is None

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def load(self, on_state: Callable[[str], None] | None = None) -> None:
        with self._lock:
            self._request({"op": "load"}, on_state=on_state)

    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        seed: int = 42,
        lora: str | None = None,
        lora_weight: float = models.DEFAULT_LORA_WEIGHT,
        negative_prompt: str | None = None,
        conditioning: Any | None = None,
        reference_images: list[Any] | tuple[Any, ...] | None = None,
        on_state: Callable[[str], None] | None = None,
        on_step: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        tile: bool = False,
        sheet: bool = False,
        tilesheet: bool = False,
        size: tuple[int, int] | None = None,
    ) -> Path:
        """Generate a reference image and save it to ``output_path``.

        Signature-for-signature what ``Text2Image.generate`` takes, because
        every caller of it is a caller of this -- which is why ``scene`` is
        gone: it was accepted here, put on the wire and dropped by
        ``text2image_worker.op_generate``, and the in-process backend it claimed
        to mirror would have raised ``TypeError`` on it. Nothing ever passed
        one. The image itself never crosses
        the pipe: the child writes ``output_path`` exactly as the in-process
        pipeline did, which is why this substitution needed no change at any
        call site.
        """
        from .text2image import JobCancelled

        payload = {
            "op": "generate",
            "prompt": prompt,
            "output": str(output_path),
            "seed": int(seed),
            "lora": lora,
            "lora_weight": float(lora_weight),
            "negative_prompt": negative_prompt,
            "conditioning": _conditioning_payload(conditioning),
            "tile": bool(tile),
            "sheet": bool(sheet),
            "tilesheet": bool(tilesheet),
            "size": [int(size[0]), int(size[1])] if size is not None else None,
        }
        # PIL images cannot cross the JSON process boundary.  Keep the files
        # alive for the duration of the request; the child opens them before it
        # answers, while path-like references can be passed through directly.
        references = list(reference_images or ())
        staging = (
            tempfile.TemporaryDirectory(prefix="warlock-t2i-ref-")
            if references
            else contextlib.nullcontext()
        )
        with staging as temp:
            if references:
                paths = []
                for index, reference in enumerate(references):
                    if isinstance(reference, (str, Path)):
                        paths.append(str(reference))
                        continue
                    path = Path(temp) / f"reference_{index}.png"
                    reference.save(path, format="PNG")
                    paths.append(str(path))
                payload["reference_images"] = paths
            with self._lock:
                resp = self._request(
                    payload, on_state=on_state, on_step=on_step, cancel_event=cancel_event
                )
        if resp.get("cancelled"):
            # The same exception the in-process pipeline raised, so the queue's
            # cancel handling is untouched by where the sampling happened.
            raise JobCancelled
        self.last_prompt = str(resp.get("prompt") or "")
        self.last_recipe = dict(resp.get("recipe") or {})
        self.last_used = time.monotonic()
        return Path(resp.get("path") or output_path)

    def trim(self) -> None:
        """Return cached-but-unused device memory. Keeps the child alive.

        A no-op when there is no child: "nothing loaded" is what trim leaves
        behind anyway, and spawning a process in order to tell it to release
        nothing would be the opposite of this method's purpose.
        """
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            try:
                self._request({"op": "trim"})
            except ChildFailed:
                # Advisory, exactly as it was in-process: a trim that could not
                # be delivered must not fail the job that asked for it.
                log.debug("trimming the t2i child failed", exc_info=True)

    def unload(self) -> None:
        """Drop the pipeline and give back *both* kinds of memory.

        This is the method the whole child exists for. In-process it returned
        the VRAM and left up to 21 GiB of host commit behind; here the process
        ends, so the arenas end with it and the commit limit gets it back.
        """
        with self._lock:
            self._stop_child()

    def close(self) -> None:
        """Forbid any further load, then stop the child. Sticky.

        Separate from ``unload`` for the reason it was separate before: unload
        drops what *is* resident, this forbids what is about to become resident,
        and shutdown needs this one first or a load that starts after the unload
        is a child nothing will reap.

        Killed *before* the lock is taken, not under it. ``_request`` holds the
        lock for the whole of a sample, and ``Worker.shutdown`` calls this on
        the loop thread before it cancels the running job -- so waiting for the
        lock meant waiting for the sample (or the silence timeout) with every
        other teardown queued behind it. The kill is what makes the in-flight
        request return: its reader sees the pipe close and raises
        ``ChildFailed``, the lock frees, and ``_stop_child`` under it then finds
        nothing left to do. ``proc.kill`` is safe from a second thread; the
        reference is read once so a concurrent restart cannot swap it.
        """
        self._closed.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()
        with self._lock:
            self._stop_child()

    # --- the child ------------------------------------------------------------

    def _start_child(self) -> None:
        """Spawn the worker. Caller holds the lock."""
        if self._closed.is_set():
            raise ChildFailed("the app is shutting down")
        argv = [
            *CHILD_ARGV,
            self.spec.key,
            str(self._model_root),
            str(self._model_dir),
        ]
        # stderr merged into stdout, drained by the reader thread below:
        # diffusers and transformers both print, an undrained second pipe
        # deadlocks the child once it fills, and a failed import is otherwise
        # invisible. The marker is what tells an answer from the chatter.
        #
        # ``winjob.assign`` follows immediately, as it must: this child is about
        # to hold more memory than anything else the app spawns, and the window
        # between Popen and that call is the only one in which a parent crash
        # can still orphan it. It is also what puts the *real* interpreter in
        # the job, since the pid Popen returned is a trampoline under a uv venv
        # (docs/measurements/2026-08-22-trampoline-child-pids.md).
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            raise ChildFailed(f"could not start the image worker: {exc}") from exc
        winjob.assign(proc.pid)
        winjob.track(proc.pid, f"text2image {self.spec.key}")

        lines: Any = _queue.Queue()

        def _pump(stream: Any) -> None:
            try:
                for raw in stream:
                    lines.put(raw)
            finally:
                lines.put(None)

        reader = threading.Thread(
            target=_pump, args=(proc.stdout,), name="t2i-worker", daemon=True
        )
        reader.start()
        self._proc, self._lines = proc, lines

        from .text2image_worker import MARKER

        deadline = time.monotonic() + READY_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_child()
                raise ChildFailed("the image worker did not start")
            try:
                raw = lines.get(timeout=remaining)
            except _queue.Empty:
                continue
            if raw is None:
                self._stop_child()
                raise ChildFailed("the image worker exited during startup")
            if MARKER not in raw:
                log.debug("t2i worker: %s", raw.rstrip())
                continue
            return

    def _stop_child(self) -> None:
        """Kill the child and forget it. Caller holds the lock. Never raises."""
        proc, self._proc, self._lines = self._proc, None, None
        self._loaded = False
        if proc is None:
            return
        if proc.poll() is None:
            log.info("stopping the image worker (pid %s)", proc.pid)
            proc.kill()
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Nothing useful left to do: it is in the kill-on-close job, so
                # it cannot outlive the app. Worth a line, because it means the
                # memory this call exists to reclaim has not come back yet.
                log.warning("the image worker (pid %s) did not exit", proc.pid)
        winjob.untrack(proc.pid)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                # ``ValueError`` as well as ``OSError``: a reader may still be
                # iterating ``stdout`` when this runs, and closing a file under
                # an in-progress read raises "I/O operation on closed file",
                # which is a ``ValueError``. It escaped into ``_thread_hook`` as
                # a logged traceback for a child that was being killed anyway.
                with contextlib.suppress(OSError, ValueError):
                    stream.close()

    def _request(
        self,
        payload: dict[str, Any],
        *,
        on_state: Callable[[str], None] | None = None,
        on_step: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Send one request and read to its terminal response. Caller holds the
        lock. Raises ``ChildFailed``.
        """
        from .text2image_worker import MARKER

        if self._proc is None or self._proc.poll() is not None:
            self._stop_child()
            self._start_child()
        proc, lines = self._proc, self._lines
        assert proc is not None and proc.stdin is not None
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self._stop_child()
            raise ChildFailed(f"the image worker went away: {exc}") from exc

        cancelled_sent = False
        deadline = time.monotonic() + SILENCE_TIMEOUT
        while True:
            if cancel_event is not None and cancel_event.is_set() and not cancelled_sent:
                # Written from this thread rather than watched from another: the
                # only thing this thread does between lines is wait, and a
                # dedicated watcher would be a second writer on one pipe.
                cancelled_sent = True
                try:
                    proc.stdin.write(json.dumps({"op": "cancel"}) + "\n")
                    proc.stdin.flush()
                except (OSError, ValueError):
                    # The child is already gone; the read below will say so.
                    pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_child()
                raise ChildFailed(
                    f"the image worker said nothing for {SILENCE_TIMEOUT:.0f}s"
                )
            try:
                # Capped so a pending cancel is noticed promptly even while the
                # child is quiet mid-step; the cap is not the timeout.
                raw = lines.get(timeout=min(remaining, 0.2))
            except _queue.Empty:
                continue
            if raw is None:
                self._stop_child()
                raise ChildFailed("the image worker exited without answering")
            # Any line at all is a sign of life, chatter included: a checkpoint
            # read prints for a long time without the worker emitting a state.
            deadline = time.monotonic() + SILENCE_TIMEOUT
            at = raw.find(MARKER)
            if at < 0:
                log.debug("t2i worker: %s", raw.rstrip())
                continue
            if at:
                # A response can share a physical line with the progress bar
                # that was mid-update when it was written; the prefix is that
                # bar, and it is chatter like any other.
                log.debug("t2i worker: %s", raw[:at].rstrip())
            try:
                msg = json.loads(raw[at + len(MARKER) :])
            except ValueError as exc:
                self._stop_child()
                raise ChildFailed(f"unreadable answer from the worker: {exc}") from exc
            kind = msg.get("kind")
            if kind == "state":
                if on_state is not None:
                    on_state(str(msg.get("text") or ""))
                continue
            if kind == "step":
                if on_step is not None:
                    on_step(int(msg.get("step", 0)), int(msg.get("total", 0)))
                continue
            if kind == "ready":
                # A restart this request did not ask for; the child is fresh and
                # the request that produced it was written before it died.
                continue
            self._publish(msg)
            if kind == "error" and not msg.get("cancelled"):
                raise ChildFailed(str(msg.get("error") or "the image worker failed"))
            return msg

    def _publish(self, msg: dict[str, Any]) -> None:
        """Take the readings the parent can no longer take for itself.

        ``vram.device_memory`` reaches torch through ``sys.modules``, and once
        the pipe lives in a child the app process never imports it -- so the
        figure admission reads would be None forever. The child reports it and
        this is where it is handed on.
        """
        self._loaded = bool(msg.get("loaded"))
        free = msg.get("device_free_gib")
        total = msg.get("device_total_gib")
        if free is not None and total is not None:
            vram.publish(
                float(free), float(total), str(msg.get("device_name") or "")
            )


def _conditioning_payload(conditioning: Any | None) -> dict[str, Any] | None:
    """A ``Conditioning`` in wire form, or None.

    Every field by name rather than ``as_dict()``: that method renders the
    *recipe* and drops halves that are not in play, so it cannot round-trip.
    """
    if conditioning is None:
        return None

    def _str(value: Any) -> str | None:
        return str(value) if value else None

    return {
        "ip_adapter": conditioning.ip_adapter,
        "ip_image": _str(conditioning.ip_image),
        "ip_scale": float(conditioning.ip_scale),
        "control": conditioning.control,
        "control_image": _str(conditioning.control_image),
        "control_scale": float(conditioning.control_scale),
        "control_end": float(conditioning.control_end),
        "init_image": _str(conditioning.init_image),
        "strength": float(conditioning.strength),
        "mask_image": _str(conditioning.mask_image),
    }
