from __future__ import annotations

import asyncio
import subprocess
import sys

import httpx
import pytest

from animancer3d.pipelines import trellis as trellis_mod
from animancer3d.pipelines.trellis import TrellisServer


def test_argv_defaults_webp_off(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    assert srv._argv()[-2:] == ["--webp", "off"]


def test_argv_webp_on_when_configured(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971, webp=True)
    assert srv._argv()[-2:] == ["--webp", "on"]


def test_argv_includes_models_host_port(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    argv = srv._argv()
    assert argv[0] == str(tmp_path / "exe")
    assert "--models" in argv and str(tmp_path / "models") in argv
    assert "--port" in argv and "17971" in argv


def test_reap_if_dead_clears_an_already_exited_process(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    srv._proc = dead
    srv._reap_if_dead()
    assert srv._proc is None


def test_reap_if_dead_leaves_a_running_process_alone(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    running = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    try:
        srv._proc = running
        srv._reap_if_dead()
        assert srv._proc is running
    finally:
        running.kill()
        running.wait()


class _FakeProc:
    """Never exits on its own; stdout is None so the reader thread returns
    immediately instead of blocking on a real pipe."""

    def __init__(self) -> None:
        self.stdout = None
        self.returncode = None

    def poll(self):
        return None


class _FakeAsyncClient:
    """Simulates a server that never answers /health -- every request fails
    with a TransportError, which ensure_started already suppresses."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, *_args, **_kwargs):
        raise httpx.TransportError("simulated: server not up yet")


@pytest.mark.asyncio
async def test_ensure_started_raises_cleanly_if_proc_cleared_mid_poll(tmp_path, monkeypatch):
    """Guards the cross-thread race (Finding 2): stop() running concurrently
    (e.g. via request_cancel's asyncio.to_thread) can set self._proc = None
    between poll iterations. Before the fix this raised AttributeError from
    None.poll(); it must now raise a clean RuntimeError instead."""
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    srv = TrellisServer(exe, models, 17971)

    monkeypatch.setattr(trellis_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", _FakeAsyncClient)

    real_sleep = asyncio.sleep
    calls = {"n": 0}

    async def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate a concurrent stop() clearing _proc between iterations.
            srv._proc = None
        await real_sleep(0)

    monkeypatch.setattr(trellis_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="stopped during startup"):
        await srv.ensure_started()
