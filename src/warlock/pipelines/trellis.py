"""Manages a resident trellis-server.exe subprocess and converts images to GLB via its HTTP API.

The server holds the TRELLIS.2 GGUF weights in VRAM; we start it on first use,
keep it warm between jobs, and stop it when idle or when another pipeline needs the GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
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

import httpx

from .. import winjob
from ..glbio import split_glb
from ..progress import pump

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 300.0  # health endpoint answers in ~1 s; weights load on first generate
GENERATE_TIMEOUT = 1800.0
LOG_MAX_BYTES = 5 * 1024 * 1024  # the log used to grow forever; roll it instead
# Respawn backoff: 5 s doubling to a 5-minute ceiling, and one give-up line
# after five consecutive failures. Jobs still fail fast with the reason -- the
# backoff stops the respawn storm, not the error reaching the inspector.
BACKOFF_BASE = 5.0
BACKOFF_MAX = 300.0
BACKOFF_GIVE_UP = 5
RECLAIM_TIMEOUT = 5.0  # how long to wait for a killed orphan to release the port


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
    ) -> None:
        self._exe = exe
        self._models_dir = models_dir
        self._port = port
        self._log_path = log_path
        self._webp = webp
        self._tex_res = tex_res
        self._band = band
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
        return self._proc is not None and self._proc.poll() is None

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
        return argv

    def _reap_if_dead(self) -> None:
        """A self-crashed server otherwise leaks the old log handle and
        reader thread the next time _proc/_reader/_logfh are overwritten."""
        if self._proc is not None and self._proc.poll() is not None:
            lifetime = (
                time.monotonic() - self._spawned_at
                if self._spawned_at is not None
                else float("nan")
            )
            log.warning(
                "trellis-server pid %s exited with code %s after %.1f s; reaping",
                self._proc.pid,
                self._proc.returncode,
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
                    await asyncio.sleep(1.0)
            self.stop()
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

    async def _reclaim_port(self) -> None:
        """Kill a provably-ours orphan holding the port, or refuse to guess.

        "Provably ours" is the whole safety argument: only a process whose
        image is the very exe this instance is configured to spawn gets killed.
        Anything else -- another app, or a path we cannot read -- is named in
        the error and left alone.
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
            await asyncio.sleep(0.25)
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

        Blocks for up to ~20 s in the worst case (terminate, wait(15), the
        reader join(5)), so every caller outside a failure path dispatches it
        through asyncio.to_thread rather than running it on the event loop.
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
            self._proc = None
            self._spawned_at = None

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
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            with image_path.open("rb") as fh:
                r = await client.post(
                    f"{self.base_url}/generate",
                    files={"image": (image_path.name, fh)},
                    data=data,
                )
        if r.status_code >= 400:
            detail = r.text[:500] if r.text else "(no body; see trellis.log)"
            raise RuntimeError(f"trellis-server {r.status_code}: {detail}")
        _validate_glb(r.content)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(output_path, r.content)
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
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
