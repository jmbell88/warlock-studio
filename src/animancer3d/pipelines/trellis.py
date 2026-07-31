"""Manages a resident trellis-server.exe subprocess and converts images to GLB via its HTTP API.

The server holds the TRELLIS.2 GGUF weights in VRAM; we start it on first use,
keep it warm between jobs, and stop it when idle or when another pipeline needs the GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 300.0  # first start loads ~8 GB of weights from disk
GENERATE_TIMEOUT = 1800.0


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
            if self._log_path is not None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                sink = self._log_path.open("ab")
            else:
                sink = subprocess.DEVNULL
            try:
                self._proc = subprocess.Popen(
                    [
                        str(self._exe),
                        "--models", str(self._models_dir),
                        "--host", "127.0.0.1",
                        "--port", str(self._port),
                        "--require-gpu",
                    ],
                    stdout=sink,
                    stderr=subprocess.STDOUT if sink is not subprocess.DEVNULL else sink,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            finally:
                if sink is not subprocess.DEVNULL:
                    sink.close()
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

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            log.info("stopping trellis-server")
            self._proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=15)
            if self._proc.poll() is None:
                self._proc.kill()
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
