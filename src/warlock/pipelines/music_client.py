"""The app-process handle on the music pipeline that lives in a child.

``MusicClient`` presents ``load`` / ``generate`` / ``trim`` / ``unload`` /
``close`` / ``loaded`` / ``last_used`` -- deliberately the same surface
``Text2ImageClient`` does, minus the parts that are about images. That is what
lets the queue's idle-eviction and dispatch-credit paths work against a parallel
``_music`` attribute rather than needing a second lifecycle taught to them.

**Why a child at all** is ``t2i_client``'s argument, unchanged: ``unload()``
in-process returns the VRAM and leaves the allocator's arenas holding host
commit that only exit gives back, and the app process is the one that has to
keep running.

**Silence, not duration, is the timeout.** A two-minute track at 60 steps
legitimately takes minutes and a cold 8.3 GiB load takes tens of seconds, so a
total-duration cap would either be uselessly large or kill working jobs. The
child emits a step line per sampling step, so the question that can be asked is
"has it said anything lately".

Simpler than its sibling in two ways, both real rather than cosmetic: there is
no ``_conditioning_payload`` equivalent and no ``TemporaryDirectory`` staging,
because ACE-Step's payload is flat scalars and lists with no PIL image in it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue as _queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import models, vram, winjob

log = logging.getLogger(__name__)

CHILD_ARGV = [sys.executable, "-m", "warlock.pipelines.music_worker"]

SILENCE_TIMEOUT = 900.0
"""Seconds of *no output at all* before the child is presumed hung.

Its own constant rather than a reuse of ``t2i_client``'s, and the same value
today only by coincidence: the two are answers to different questions -- how
long a diffusion step takes on a latent image versus on an audio latent -- and
measurement is expected to move one of them. Sharing them would make the first
such measurement a change to both.
"""

STOP_TIMEOUT = 15.0
"""How long a killed child is waited for.

It may be inside a CUDA teardown, and the whole point of the kill is that its
memory is back when the wait returns.
"""

READY_TIMEOUT = 120.0
"""How long the child has to say ``ready``.

A Python import and no weight read, but it is torch's import on a cold file
cache. A child that never says it is failing at startup -- a missing ``music``
extra, a broken venv -- and must be reported as that rather than as a hang in
the first generate.
"""


class ChildFailed(RuntimeError):
    """The child could not serve a request. Carries the child's own message."""


class MusicCancelled(RuntimeError):
    """A generation the user stopped.

    The app-process counterpart of the vendored pipeline's ``WarlockCancelled``,
    which is raised in a process this one cannot catch across. Named separately
    from the image pipeline's ``JobCancelled`` so nothing in the queue can
    handle one believing it is the other.
    """


class MusicClient:
    """One resident child, and the requests that keep it warm."""

    def __init__(self, spec: models.MusicModel, model_dir: Path) -> None:
        # One directory and no model root: ACE-Step ships a single weight set,
        # so there is no ``turbo``-style variant for a root plus a dir_name to
        # resolve between. Path resolution stays in ``fetch``/``config``, where
        # every other model's lives.
        self.spec = spec
        self._model_dir = model_dir
        self._proc: subprocess.Popen[str] | None = None
        self._lines: Any = None
        # Serialises the whole exchange, not just the write: the protocol is one
        # terminal response per request with nothing to correlate them by, so
        # two callers interleaving would each read the other's answer.
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._loaded = False
        self.last_used: float = 0.0
        self.last_recipe: dict[str, Any] = {}

    # --- the surface the queue holds it by ------------------------------------

    @property
    def loaded(self) -> bool:
        """Whether the child holds a loaded pipeline.

        A parent-side mirror rather than a question asked over the wire: it is
        read from the event loop, where a round trip to a child that is
        mid-sample would block the frame the app draws.
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
        lyrics: str = "",
        audio_duration: float = 60.0,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: float = 10.0,
        seed: int | None = None,
        on_state: Callable[[str], None] | None = None,
        on_step: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        **extra: Any,
    ) -> Path:
        """Generate a track and save it to ``output_path``.

        ``extra`` is passed through to the worker verbatim, which is where the
        retake / repaint / edit kwargs travel: they are arguments to the same
        sampler call rather than modes of their own, so naming each of them here
        would be a second copy of a table the worker already keeps.

        The audio never crosses the pipe -- the child writes ``output_path``,
        for the reason a 1024x1024 PNG does.
        """
        payload: dict[str, Any] = {
            "op": "generate",
            "prompt": prompt,
            "lyrics": lyrics,
            "output": str(output_path),
            "audio_duration": float(audio_duration),
            "infer_step": int(infer_step),
            "guidance_scale": float(guidance_scale),
            "scheduler_type": str(scheduler_type),
            "cfg_type": str(cfg_type),
            "omega_scale": float(omega_scale),
            "manual_seeds": [int(seed)] if seed is not None else None,
            **extra,
        }
        with self._lock:
            resp = self._request(
                payload, on_state=on_state, on_step=on_step, cancel_event=cancel_event
            )
        if resp.get("cancelled"):
            raise MusicCancelled
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
                # Advisory: a trim that could not be delivered must not fail the
                # job that asked for it.
                log.debug("trimming the music child failed", exc_info=True)

    def unload(self) -> None:
        """Drop the pipeline and give back *both* kinds of memory.

        This is the method the whole child exists for: the process ends, so the
        allocator's arenas end with it and the commit limit gets them back.
        """
        with self._lock:
            self._stop_child()

    def close(self) -> None:
        """Forbid any further load, then stop the child. Sticky.

        Separate from ``unload`` for the reason it is separate there: unload
        drops what *is* resident, this forbids what is about to become resident,
        and shutdown needs this one first or a load that starts after the unload
        is a child nothing will reap.

        Killed *before* the lock is taken, not under it. ``_request`` holds the
        lock for the whole of a sample, and ``Worker.shutdown`` calls this on
        the loop thread before it cancels the running job -- so waiting for the
        lock would mean waiting for the sample (or the silence timeout) with
        every other teardown queued behind it. The kill is what makes the
        in-flight request return.
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
        argv = [*CHILD_ARGV, self.spec.key, str(self._model_dir)]
        # stderr merged into stdout, drained by the reader thread below: torch,
        # transformers and loguru all print, an undrained second pipe deadlocks
        # the child once it fills, and a failed import is otherwise invisible.
        # The marker is what tells an answer from the chatter.
        #
        # ``winjob.assign`` follows immediately, as it must: this child holds
        # 8+ GiB, and the window between Popen and that call is the only one in
        # which a parent crash can still orphan it. It is also what puts the
        # *real* interpreter in the job, since the pid Popen returned is a
        # trampoline under a uv venv
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
            raise ChildFailed(f"could not start the music worker: {exc}") from exc
        winjob.assign(proc.pid)
        winjob.track(proc.pid, f"music {self.spec.key}")

        lines: Any = _queue.Queue()

        def _pump(stream: Any) -> None:
            try:
                for raw in stream:
                    lines.put(raw)
            except (OSError, ValueError):
                # _stop_child()/stop() can close proc.stdout from the main
                # thread mid-iteration; on Windows that raises inside this
                # generator. The sentinel below still unblocks any waiter.
                pass
            finally:
                lines.put(None)

        reader = threading.Thread(
            target=_pump, args=(proc.stdout,), name="music-worker", daemon=True
        )
        reader.start()
        self._proc, self._lines = proc, lines

        from .music_worker import MARKER

        deadline = time.monotonic() + READY_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop_child()
                raise ChildFailed("the music worker did not start")
            try:
                raw = lines.get(timeout=remaining)
            except _queue.Empty:
                continue
            if raw is None:
                self._stop_child()
                raise ChildFailed("the music worker exited during startup")
            if MARKER not in raw:
                log.debug("music worker: %s", raw.rstrip())
                continue
            return

    def _stop_child(self) -> None:
        """Kill the child and forget it. Caller holds the lock. Never raises."""
        proc, self._proc, self._lines = self._proc, None, None
        self._loaded = False
        if proc is None:
            return
        if proc.poll() is None:
            log.info("stopping the music worker (pid %s)", proc.pid)
            proc.kill()
            try:
                proc.wait(timeout=STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                # Nothing useful left to do: it is in the kill-on-close job, so
                # it cannot outlive the app. Worth a line, because it means the
                # memory this call exists to reclaim has not come back yet.
                log.warning("the music worker (pid %s) did not exit", proc.pid)
        winjob.untrack(proc.pid)
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                # ``ValueError`` as well as ``OSError``: a reader may still be
                # iterating ``stdout`` when this runs, and closing a file under
                # an in-progress read raises "I/O operation on closed file".
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
        from .music_worker import MARKER

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
            raise ChildFailed(f"the music worker went away: {exc}") from exc

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
                    f"the music worker said nothing for {SILENCE_TIMEOUT:.0f}s"
                )
            try:
                # Capped so a pending cancel is noticed promptly even while the
                # child is quiet mid-step; the cap is not the timeout.
                raw = lines.get(timeout=min(remaining, 0.2))
            except _queue.Empty:
                continue
            if raw is None:
                self._stop_child()
                raise ChildFailed("the music worker exited without answering")
            # Any line at all is a sign of life, chatter included: a checkpoint
            # read prints for a long time without the worker emitting a state.
            deadline = time.monotonic() + SILENCE_TIMEOUT
            at = raw.find(MARKER)
            if at < 0:
                log.debug("music worker: %s", raw.rstrip())
                continue
            if at:
                # A response can share a physical line with the progress bar
                # that was mid-update when it was written; the prefix is that
                # bar, and it is chatter like any other.
                log.debug("music worker: %s", raw[:at].rstrip())
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
                raise ChildFailed(str(msg.get("error") or "the music worker failed"))
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
            vram.publish(float(free), float(total), str(msg.get("device_name") or ""))
