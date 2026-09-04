"""Manages a resident trellis-server.exe subprocess and converts images to GLB via its HTTP API.

The server holds the TRELLIS.2 GGUF weights in VRAM; we start it on first use,
keep it warm between jobs, and stop it when idle or when another pipeline needs the GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import struct
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .. import winjob
from ..glbio import split_glb
from ..progress import pump

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 300.0  # health endpoint answers in ~1 s; weights load on first generate
GENERATE_TIMEOUT = 1800.0
# The largest response body ``generate`` will buffer. A GLB at the resolutions
# this project asks for is single-digit megabytes; half a gigabyte is far past
# any legitimate answer and well short of what a runaway (or a misaddressed
# server streaming something else entirely) can do to a process that reads the
# whole body into RAM before looking at it. Read from module globals at call
# time so a test can lower it.
MAX_GLB_BYTES = 512 * 1024 * 1024
# The largest *error* body. Two orders of magnitude smaller than the GLB
# ceiling, because this one is a message rather than an artifact: only ~500
# characters are ever shown, and the rest of the allowance is so a multi-byte
# character at the boundary still decodes and so a stack trace in the body is
# not truncated mid-word before its first line. The error path used to call
# ``await r.aread()`` -- unbounded -- so a wedged server could answer a failing
# request with gigabytes and take the host's memory with it (MDL-13).
MAX_ERROR_BYTES = 64 * 1024
LOG_MAX_BYTES = 5 * 1024 * 1024  # the log used to grow forever; roll it instead
# Respawn backoff: 5 s doubling to a 5-minute ceiling, and one give-up line
# after five consecutive failures. Jobs still fail fast with the reason -- the
# backoff stops the respawn storm, not the error reaching the inspector.
BACKOFF_BASE = 5.0
BACKOFF_MAX = 300.0
BACKOFF_GIVE_UP = 5
RECLAIM_TIMEOUT = 5.0  # how long to wait for a killed orphan to release the port
KILL_TIMEOUT = 5.0  # how long a killed server gets to actually be reaped


class TrellisStopFailed(RuntimeError):
    """``stop()`` could not confirm the server was dead.

    A stop that reports success is a *precondition* elsewhere: the handoff in
    ``queue._generate`` loads an image model into the VRAM this call is supposed
    to have released, and "released" means the OS has reaped the process, not
    that ``kill()`` returned. Windows reaps asynchronously, so ``kill()``
    followed by clearing the handle could hand back a green answer while ~16 GiB
    of device memory was still charged to a live process -- and the next thing
    to happen was a 16 GiB allocation.
    """


def _pid_alive(pid: int) -> bool:
    """Is that pid still around? Best-effort, and False on any doubt.

    Used only to decide whether a *recorded* port owner is still running, so a
    false negative costs a refusal a retry clears, and a false positive costs a
    refusal too -- both far cheaper than terminating somebody's live server.
    """
    if pid <= 0:
        return False
    return winjob.image_path(pid) is not None


def _port_in_use(port: int) -> bool:
    """True if something already listens on 127.0.0.1:`port`.

    Deliberately a bind attempt and not a connect: SO_REUSEADDR is not set, so
    this asks the exact question the child is about to ask -- "can a listener
    be created here" -- rather than "does something answer", which an orphan in
    a wedged state might not.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


class TrellisServer:
    def __init__(
        self,
        exe: Path,
        models_dir: Path,
        port: int,
        log_path: Path | None = None,
        webp: bool = False,
        tex_res: int = 512,
        band: int | None = None,
        gss: float | None = None,
        gsh: float | None = None,
        max_tokens: int | None = None,
        decim: int | None = None,
        atlas: int | None = None,
    ) -> None:
        self._exe = exe
        self._models_dir = models_dir
        self._port = port
        self._log_path = log_path
        self._webp = webp
        self._tex_res = tex_res
        self._band = band
        self._gss = gss
        self._gsh = gsh
        self._max_tokens = max_tokens
        self._decim = decim
        self._atlas = atlas
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = asyncio.Lock()
        # stop() is called from the event loop (ensure_started's own reap and
        # startup-timeout paths) *and* from worker threads (queue.py dispatches
        # it via asyncio.to_thread for cancel, eviction, the VRAM handoff and
        # shutdown). asyncio.Lock cannot guard across that boundary, so the
        # check-then-act sequence in stop() needs a real threading lock or two
        # callers can both join the reader and one can null _proc between
        # another's `is not None` and `.poll()`.
        self._stop_lock = threading.Lock()
        self.last_used = 0.0
        # Called for every decoded stdout line, on the reader thread.
        self.on_line: Callable[[str], None] | None = None
        self._reader: threading.Thread | None = None
        self._logfh = None
        # When the live child was spawned, so its lifetime can be logged at
        # reap time. A short lifetime is the signature of a bind failure.
        self._spawned_at: float | None = None
        # Consecutive startup failures, and the monotonic time before which
        # ensure_started refuses to spawn again. Both reset on "ready".
        self._start_failures = 0
        self._backoff_until = 0.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def running(self) -> bool:
        # One read: stop() nulls _proc from another thread, and re-reading
        # self._proc after the None test races an AttributeError with it.
        proc = self._proc
        return proc is not None and proc.poll() is None

    def _argv(self) -> list[str]:
        argv = [
            str(self._exe),
            "--models", str(self._models_dir),
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "--require-gpu",
            "--webp", "on" if self._webp else "off",
            "--tex-res", str(self._tex_res),
        ]
        # Omitted rather than passed as "auto": the exe has no auto keyword, the
        # heuristic is what runs when the flag is absent.
        if self._band is not None:
            argv += ["--band", str(self._band)]
        # Same rule for the guidance strengths and the token budget: absent
        # means the exe's default, which --help does not print.
        if self._gss is not None:
            argv += ["--gss", str(self._gss)]
        if self._gsh is not None:
            argv += ["--gsh", str(self._gsh)]
        if self._max_tokens is not None:
            argv += ["--max-tokens", str(self._max_tokens)]
        # ``is not None`` and never truthiness: --decim 0 is the rung that
        # matters (it turns the exe's own 300K-face simplify off), and a
        # truthy test would silently drop it.
        if self._decim is not None:
            argv += ["--decim", str(self._decim)]
        if self._atlas is not None:
            argv += ["--atlas", str(self._atlas)]
        return argv

    def _launch_config(self) -> tuple[Any, ...]:
        return (
            self._tex_res, self._band, self._gss, self._gsh, self._max_tokens,
            self._decim, self._atlas,
        )

    def ensure_config(
        self,
        *,
        tex_res: int,
        band: int | None,
        gss: float | None = None,
        gsh: float | None = None,
        max_tokens: int | None = None,
        decim: int | None = None,
        atlas: int | None = None,
    ) -> bool:
        """Adopt a launch config, stopping a server running with a different one.

        -> whether a running server was stopped. There is no new spawn site: the
        next ``ensure_started`` builds ``_argv`` from the fields written here, so
        "restart with a different band" is exactly "stop, and let the existing
        lazy start do its job".

        The check is against the *running* server rather than against any
        bookkeeping, which is what makes it free of state to get wrong: an idle
        eviction, a crash or a cancel all leave nothing running, and nothing is
        then stopped. Called on every model-stage job with fully resolved
        values, so an ordinary job following a sweep unit restores the config's
        own settings without anyone having to remember that the sweep changed
        them.

        Blocking (``stop`` is), so every caller dispatches it through
        ``asyncio.to_thread`` exactly as they dispatch ``stop``.
        """
        with self._stop_lock:
            wanted = (tex_res, band, gss, gsh, max_tokens, decim, atlas)
            changed = self._launch_config() != wanted
            (
                self._tex_res, self._band, self._gss, self._gsh, self._max_tokens,
                self._decim, self._atlas,
            ) = wanted
            # Read under the same lock that guards stop()'s check-then-act, and
            # released before calling it: _stop_lock is a plain Lock.
            restart = changed and self._proc is not None and self._proc.poll() is None
        if restart:
            log.info(
                "trellis-server config changed (tex_res=%s band=%s gss=%s gsh=%s "
                "max_tokens=%s decim=%s atlas=%s); restarting",
                tex_res, band, gss, gsh, max_tokens, decim, atlas,
            )
            self.stop()
        return restart

    def _reap_if_dead(self) -> None:
        """A self-crashed server otherwise leaks the old log handle and
        reader thread the next time _proc/_reader/_logfh are overwritten."""
        proc = self._proc  # one read; stop() nulls it from another thread
        if proc is not None and proc.poll() is not None:
            lifetime = (
                time.monotonic() - self._spawned_at
                if self._spawned_at is not None
                else float("nan")
            )
            log.warning(
                "trellis-server pid %s exited with code %s after %.1f s; reaping",
                proc.pid,
                proc.returncode,
                lifetime,
            )
            self.stop()

    async def ensure_started(self) -> None:
        async with self._lock:
            self._reap_if_dead()
            if self.running:
                return
            self._check_backoff()
            if not self._exe.exists():
                raise RuntimeError(f"trellis-server not found at {self._exe}")
            if not self._models_dir.exists():
                raise RuntimeError(f"TRELLIS GGUF models not found at {self._models_dir}")
            # Bind-precheck. /health returns a bare ok with no identity field,
            # so a stale orphan holding the port answers the poll below exactly
            # like a healthy fresh server would -- while the server we are
            # about to spawn dies on bind. Every generate then goes to the
            # orphan. Fail loudly here instead.
            if _port_in_use(self._port):
                await self._reclaim_port()
            log.info("starting trellis-server on port %d", self._port)
            self._open_log()
            # stdout is piped rather than redirected so we can parse the stage
            # trace for progress; the reader thread mirrors it into the log file
            # so trellis.log keeps receiving byte-identical output.
            # bufsize=0 is load-bearing: with the default buffering, read(65536)
            # blocks until 65536 bytes arrive and progress would arrive in bursts.
            self._proc = subprocess.Popen(
                self._argv(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            # Kill-on-close job object, assigned as early as possible: the
            # window between Popen and this call is the only one in which a
            # parent crash can still orphan the child.
            winjob.assign(self._proc.pid)
            # Tracked as well as assigned, unlike a child that only needs the
            # job object's guarantee: the registry is what measured_pids()
            # falls back to when CreateJobObject/SetInformationJobObject
            # failed, and this child can hold up to ~16 GiB of host commit --
            # exactly what that fallback exists to not lose. Paired with
            # stop()'s untrack, called only once death is confirmed (not on
            # the TrellisStopFailed path below, where the process is still
            # alive and would otherwise vanish from the accounting first).
            winjob.track(self._proc.pid, "trellis-server")
            # Recorded before the health poll, not after: a server that dies
            # during startup still held the port for a moment, and the claim is
            # what tells the *next* start that the corpse on that port is ours.
            self._claim_port(self._proc.pid)
            self._spawned_at = time.monotonic()
            log.info("trellis-server spawned as pid %d", self._proc.pid)
            self._reader = threading.Thread(
                target=self._pump, name="trellis-stdout", daemon=True
            )
            self._reader.start()
            deadline = time.monotonic() + STARTUP_TIMEOUT
            async with httpx.AsyncClient() as client:
                while time.monotonic() < deadline:
                    proc = self._proc
                    if proc is None:
                        raise RuntimeError("trellis-server was stopped during startup")
                    if proc.poll() is not None:
                        self._note_start_failure()
                        raise RuntimeError(
                            f"trellis-server exited during startup (code {proc.returncode})"
                        )
                    with contextlib.suppress(httpx.TransportError):
                        r = await client.get(f"{self.base_url}/health", timeout=2.0)
                        if r.status_code == 200:
                            log.info("trellis-server ready")
                            self._start_failures = 0
                            self._backoff_until = 0.0
                            return
                    # 0.1 s, not 1.0 (C35): the server flips healthy between
                    # polls, and every startup used to donate up to a second
                    # of dead air to the first job. A refused connection costs
                    # microseconds, so the tighter loop is nearly free.
                    await asyncio.sleep(0.1)
            # Suppressed, and for fetch._move_into's reason: this is the
            # cleanup on a failure path, so a stop that cannot confirm death
            # must not mask the error that brought us here -- "did not become
            # healthy in time" is the diagnosis, and the critical log inside
            # stop() has already recorded the second problem. Through
            # asyncio.to_thread like every other caller: stop() blocks for up
            # to ~25 s, and inline it parked the whole event loop on a process
            # that is refusing to die. Holding _lock across the await is
            # deliberate -- it is what keeps a concurrent ensure_started out
            # until the teardown has finished, exactly as the inline call did.
            with contextlib.suppress(TrellisStopFailed):
                await asyncio.to_thread(self.stop)
            self._note_start_failure()
            raise RuntimeError("trellis-server did not become healthy in time")

    # --- respawn control ---

    def _check_backoff(self) -> None:
        """Refuse to respawn inside the backoff window.

        The 2026-08-03 trellis.log holds five startup banners in one minute,
        each dying on bind: every generate() called ensure_started, and nothing
        between them ever paused. A storm like that buries the first failure --
        the only one that says what went wrong -- under identical repeats.
        """
        remaining = self._backoff_until - time.monotonic()
        if remaining > 0:
            raise RuntimeError(
                f"trellis-server failed to start {self._start_failures} time(s); "
                f"refusing to respawn for another {remaining:.0f} s -- see trellis.log"
            )

    def _note_start_failure(self) -> None:
        """Count a *startup* failure and widen the window.

        Only failures a retry could plausibly fix. A missing exe or missing
        weights is a configuration error: it raises before this, because
        backing off from it would delay the fix rather than the retry.
        """
        self._start_failures += 1
        delay = min(BACKOFF_BASE * 2 ** (self._start_failures - 1), BACKOFF_MAX)
        self._backoff_until = time.monotonic() + delay
        if self._start_failures >= BACKOFF_GIVE_UP:
            log.critical(
                "trellis-server has failed to start %d times in a row; 3D jobs will "
                "keep failing until it is fixed -- see trellis.log",
                self._start_failures,
            )
        else:
            log.warning(
                "trellis-server start failed (%d in a row); next attempt in %.0f s",
                self._start_failures, delay,
            )

    @property
    def _owner_path(self) -> Path | None:
        """Where this instance records that it owns the port. None when there is
        nowhere to write (a test constructing a server with no log path)."""
        if self._log_path is None:
            return None
        return self._log_path.parent / f"trellis-{self._port}.owner"

    def _claim_port(self, pid: int) -> None:
        """Record that *this* Warlock spawned the listener on this port.

        The file is deliberately not a lock and is never trusted to say a server
        is running -- ``_port_in_use`` answers that. Its one job is to say who
        started the thing that is listening, so a reclaim can tell our own
        orphan from somebody else's live server (RUN-01).
        """
        path = self._owner_path
        if path is None:
            return
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"owner_pid": os.getpid(), "server_pid": pid}),
                encoding="utf-8",
            )

    def _release_port_claim(self) -> None:
        path = self._owner_path
        if path is not None:
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    def _recorded_owner(self) -> int | None:
        """The pid of the Warlock that last claimed this port, if any."""
        path = self._owner_path
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return int(data.get("owner_pid") or 0) or None
        except (OSError, ValueError, TypeError):
            return None

    async def _reclaim_port(self) -> None:
        """Kill a provably-ours orphan holding the port, or refuse to guess.

        Two conditions, and the second is the one RUN-01 added. Executable
        identity alone was treated as proof the listener was *our* orphan --
        but the exe path is a property of the install, not of the instance, so
        two Warlocks configured against the same binary and the same port share
        it. The single-instance lock stops that for one home; it does not stop
        it for two homes (``WARLOCK_HOME`` exists precisely so a second library
        is possible, and ``WARLOCK_TRELLIS_PORT`` still defaults to 17971 in
        both). A second instance could therefore terminate the first's *live*
        server and read the kill as a successful orphan cleanup.

        So the listener must also be one *this* home claimed, and that claim's
        owner must be gone. A crash leaves the claim file naming a dead pid,
        which is exactly the orphan case; a live owner that is not us means the
        server belongs to somebody still running, and the answer is to say so.
        """
        pid = winjob.listener_pid(self._port)
        if pid is None:
            raise RuntimeError(
                f"port {self._port} is already in use, probably by an orphaned "
                "trellis-server.exe left behind by a previous crash. Run "
                "`Get-Process trellis-server` and stop it before retrying."
            )
        path = winjob.image_path(pid)
        ours = self._exe.resolve()
        if path is None or os.path.normcase(path) != os.path.normcase(str(ours)):
            raise RuntimeError(
                f"port {self._port} is held by pid {pid} ({path or 'unknown program'}), "
                f"which is not this Warlock's trellis-server ({ours}). Stop it or "
                "change WARLOCK_TRELLIS_PORT before retrying."
            )
        owner = self._recorded_owner()
        if owner is None:
            raise RuntimeError(
                f"port {self._port} is held by a trellis-server (pid {pid}) that this "
                f"Warlock did not start -- there is no record of this home claiming "
                f"it. It may belong to another Warlock using a different "
                f"WARLOCK_HOME. Stop it, or change WARLOCK_TRELLIS_PORT, before "
                f"retrying."
            )
        if owner != os.getpid() and _pid_alive(owner):
            raise RuntimeError(
                f"port {self._port} is held by a trellis-server started by a Warlock "
                f"that is still running (pid {owner}). Close it, or give this one its "
                f"own WARLOCK_TRELLIS_PORT, before retrying."
            )
        log.warning(
            "port %d is held by an orphaned trellis-server (pid %d) from a previous "
            "crash; terminating it", self._port, pid,
        )
        winjob.terminate(pid)
        # Polled, not slept through: the port is released when the kernel tears
        # the socket down, which is after TerminateProcess returns. asyncio.sleep
        # because this runs on the worker loop, inside self._lock.
        deadline = time.monotonic() + RECLAIM_TIMEOUT
        while time.monotonic() < deadline:
            if not _port_in_use(self._port):
                return
            # 50 ms (C36): the kernel tears the socket down almost immediately
            # after TerminateProcess, and the whole respawn waits on this.
            await asyncio.sleep(0.05)
        self._note_start_failure()
        raise RuntimeError(
            f"port {self._port} is still held after terminating pid {pid}"
        )

    # --- stdout plumbing ---

    def _open_log(self) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            if self._log_path.stat().st_size > LOG_MAX_BYTES:
                self._log_path.unlink()
        self._logfh = self._log_path.open("ab")

    def _write_log(self, chunk: bytes) -> None:
        if self._logfh is not None:
            self._logfh.write(chunk)
            self._logfh.flush()

    def _dispatch(self, line: str) -> None:
        if self.on_line is not None:
            self.on_line(line)

    def _pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        pump(proc.stdout, self._write_log, self._dispatch)

    def stop(self) -> None:
        """Kill the server and release its handles. Idempotent and thread-safe.

        Blocks for up to ~25 s in the worst case (terminate, wait(15),
        kill, wait(5), the reader join(5)), so every caller outside a failure
        path dispatches it through asyncio.to_thread rather than running it on
        the event loop.

        Raises ``TrellisStopFailed`` if the process is still alive after the
        kill. That is not a formality: three call sites treat a returned stop()
        as "the VRAM is back" and immediately allocate against it.
        """
        with self._stop_lock:
            # A second caller arriving after the first finished has nothing to
            # do; returning early keeps the log quiet and the joins single.
            if self._proc is None and self._reader is None and self._logfh is None:
                return
            # Read _proc into a local once: no other thread can clear it while
            # we hold the lock, but this also keeps the terminate/wait/kill
            # sequence operating on one object rather than re-reading a field.
            proc = self._proc
            if proc is not None and proc.poll() is None:
                log.info("stopping trellis-server pid %d", proc.pid)
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=15)
                if proc.poll() is None:
                    proc.kill()
                    # wait() after kill(), because kill() only *asks*. Without
                    # this the handle was cleared below regardless, so `running`
                    # went false and the caller loaded an image model into
                    # memory a live process still held.
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=KILL_TIMEOUT)
                if proc.poll() is None:
                    # Leave _proc in place: it may still own its VRAM, and
                    # keeping the handle is what lets `running` stay true and
                    # _reap_if_dead collect it later. Clearing it here would
                    # lose the only reference to a process nothing else tracks.
                    log.critical(
                        "trellis-server pid %d survived kill(); VRAM is still held",
                        proc.pid,
                    )
                    raise TrellisStopFailed(
                        f"trellis-server pid {proc.pid} did not exit after "
                        f"{KILL_TIMEOUT:.0f}s; its GPU memory is still held"
                    )
            # Join only after the process is dead, so the pending read hits EOF.
            if self._reader is not None:
                self._reader.join(timeout=5)
                self._reader = None
            if proc is not None and proc.stdout is not None:
                with contextlib.suppress(OSError):
                    proc.stdout.close()
            if self._logfh is not None:
                with contextlib.suppress(OSError):
                    self._logfh.close()
                self._logfh = None
            if proc is not None:
                # Confirmed dead (the TrellisStopFailed raise above returns
                # before here) -- forgotten from winjob's registry the same
                # way it was remembered, so measured_pids()'s fallback never
                # counts a pid that no longer exists.
                winjob.untrack(proc.pid)
            self._proc = None
            self._spawned_at = None
            # Only once the process is confirmed dead -- the raise above skips
            # this deliberately. A claim outliving a server that survived its
            # kill would invite the next start to "reclaim" a port whose holder
            # is very much alive.
            self._release_port_claim()

    async def generate(
        self,
        image_path: Path,
        output_path: Path,
        *,
        seed: int = 42,
        resolution: int = 1024,
        bg_removal: str | None = None,
    ) -> Path:
        """Run image -> 3D and write the returned GLB to output_path.

        ``bg_removal`` picks how the server mattes the input: birefnet is the
        learned matte (needs birefnet.gguf, see doctor), threshold is the cheap
        cutout, auto lets the server decide. Omitted entirely when None so the
        exe applies its own default rather than being handed a keyword it may
        not know.
        """
        await self.ensure_started()
        self.last_used = time.monotonic()
        data = {"seed": str(seed), "resolution": str(resolution)}
        if bg_removal is not None:
            data["bg_removal"] = bg_removal
        # Streamed rather than buffered by httpx, so the ceiling below can be
        # applied while the body is still arriving: `r.content` on a plain post
        # has already read the whole thing by the time anything could refuse it.
        limit = MAX_GLB_BYTES
        chunks: list[bytes] = []
        received = 0
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            with image_path.open("rb") as fh:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/generate",
                    files={"image": (image_path.name, fh)},
                    data=data,
                ) as r:
                    if r.status_code >= 400:
                        # Streamed and capped, exactly like the success body
                        # below. ``await r.aread()`` is unbounded: only ~500
                        # characters of the result are ever shown, but the whole
                        # thing was pulled into memory first, so a wedged or
                        # compromised local server could exhaust the host by
                        # answering an error with gigabytes (MDL-13).
                        #
                        # A far smaller ceiling than the GLB one, because this
                        # is a *message*: nothing legitimate needs more than the
                        # 500 characters that get displayed, and the extra room
                        # is only so a multi-byte character at the boundary
                        # still decodes.
                        error_bytes = bytearray()
                        async for chunk in r.aiter_bytes():
                            if len(error_bytes) >= MAX_ERROR_BYTES:
                                break
                            error_bytes.extend(chunk)
                        text = bytes(error_bytes).decode("utf-8", "replace")
                        detail = text[:500] if text else "(no body; see trellis.log)"
                        raise RuntimeError(f"trellis-server {r.status_code}: {detail}")
                    async for chunk in r.aiter_bytes():
                        received += len(chunk)
                        if received > limit:
                            raise RuntimeError(
                                f"trellis-server returned more than {limit} bytes; "
                                "refusing to buffer it"
                            )
                        chunks.append(chunk)
        content = b"".join(chunks)
        # Off the loop thread: parsing and flushing a multi-megabyte GLB
        # inline would block cancellation and every call_on_loop for its
        # duration.
        def _finish(content: bytes = content) -> None:
            _validate_glb(content)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(output_path, content)

        await asyncio.to_thread(_finish)
        self.last_used = time.monotonic()
        return output_path


def _validate_glb(data: bytes) -> None:
    """Refuse anything that isn't a GLB carrying at least one mesh.

    A 200 with an HTML error page, or a body truncated by a dying server, is
    otherwise written straight onto ``source.glb`` -- and because the queue
    deliberately swallows optimize/normalize/audit failures, the job would go
    *done* with garbage. Raising here is what turns it into a failed job.
    """
    detail = f"{len(data)} bytes, starts with {data[:16]!r}"
    try:
        _header, gltf, _rest = split_glb(data)
    except (ValueError, struct.error, UnicodeDecodeError) as exc:
        raise RuntimeError(f"trellis-server returned an invalid GLB: {exc} ({detail})") from exc
    if not gltf.get("meshes"):
        raise RuntimeError(f"trellis-server returned a GLB with no meshes ({detail})")


def _atomic_write(path: Path, data: bytes) -> None:
    """Stage beside the destination and rename, so a failed write never leaves
    a partial ``source.glb`` behind for the rest of the pipeline to read."""
    # A dotfile, per the staged-writes rule: a visible ``source.glb...tmp``
    # sibling would sit beside the served name in every directory listing.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
