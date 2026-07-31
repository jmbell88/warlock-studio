"""Manages a resident trellis-server.exe subprocess and converts images to GLB via its HTTP API.

The server holds the TRELLIS.2 GGUF weights in VRAM; we start it on first use,
keep it warm between jobs, and stop it when idle or when another pipeline needs the GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from ..progress import pump

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 300.0  # health endpoint answers in ~1 s; weights load on first generate
GENERATE_TIMEOUT = 1800.0
LOG_MAX_BYTES = 5 * 1024 * 1024  # the log used to grow forever; roll it instead


class TrellisServer:
    def __init__(
        self, exe: Path, models_dir: Path, port: int, log_path: Path | None = None
    ) -> None:
        self._exe = exe
        self._models_dir = models_dir
        self._port = port
        self._log_path = log_path
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = asyncio.Lock()
        self.last_used = 0.0
        # Called for every decoded stdout line, on the reader thread.
        self.on_line: Callable[[str], None] | None = None
        self._reader: threading.Thread | None = None
        self._logfh = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    async def ensure_started(self) -> None:
        async with self._lock:
            if self.running:
                return
            if not self._exe.exists():
                raise RuntimeError(f"trellis-server not found at {self._exe}")
            if not self._models_dir.exists():
                raise RuntimeError(f"TRELLIS GGUF models not found at {self._models_dir}")
            log.info("starting trellis-server on port %d", self._port)
            self._open_log()
            # stdout is piped rather than redirected so we can parse the stage
            # trace for progress; the reader thread mirrors it into the log file
            # so trellis.log keeps receiving byte-identical output.
            # bufsize=0 is load-bearing: with the default buffering, read(65536)
            # blocks until 65536 bytes arrive and progress would arrive in bursts.
            self._proc = subprocess.Popen(
                [
                    str(self._exe),
                    "--models", str(self._models_dir),
                    "--host", "127.0.0.1",
                    "--port", str(self._port),
                    "--require-gpu",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            self._reader = threading.Thread(
                target=self._pump, name="trellis-stdout", daemon=True
            )
            self._reader.start()
            deadline = time.monotonic() + STARTUP_TIMEOUT
            async with httpx.AsyncClient() as client:
                while time.monotonic() < deadline:
                    if self._proc.poll() is not None:
                        raise RuntimeError(
                            f"trellis-server exited during startup (code {self._proc.returncode})"
                        )
                    with contextlib.suppress(httpx.TransportError):
                        r = await client.get(f"{self.base_url}/health", timeout=2.0)
                        if r.status_code == 200:
                            log.info("trellis-server ready")
                            return
                    await asyncio.sleep(1.0)
            self.stop()
            raise RuntimeError("trellis-server did not become healthy in time")

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
        if self._proc is not None and self._proc.poll() is None:
            log.info("stopping trellis-server")
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=15)
            if self._proc.poll() is None:
                self._proc.kill()
        # Join only after the process is dead, so the pending read hits EOF.
        if self._reader is not None:
            self._reader.join(timeout=5)
            self._reader = None
        if self._proc is not None and self._proc.stdout is not None:
            with contextlib.suppress(OSError):
                self._proc.stdout.close()
        if self._logfh is not None:
            with contextlib.suppress(OSError):
                self._logfh.close()
            self._logfh = None
        self._proc = None

    async def generate(
        self,
        image_path: Path,
        output_path: Path,
        *,
        seed: int = 42,
        resolution: int = 1024,
    ) -> Path:
        """Run image -> 3D and write the returned GLB to output_path."""
        await self.ensure_started()
        self.last_used = time.monotonic()
        async with httpx.AsyncClient(timeout=GENERATE_TIMEOUT) as client:
            with image_path.open("rb") as fh:
                r = await client.post(
                    f"{self.base_url}/generate",
                    files={"image": (image_path.name, fh)},
                    data={"seed": str(seed), "resolution": str(resolution)},
                )
        if r.status_code >= 400:
            detail = r.text[:500] if r.text else "(no body; see trellis.log)"
            raise RuntimeError(f"trellis-server {r.status_code}: {detail}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(r.content)
        self.last_used = time.monotonic()
        return output_path
