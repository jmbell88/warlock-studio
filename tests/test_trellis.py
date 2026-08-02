from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import httpx
import pytest

from warlock.pipelines import trellis as trellis_mod
from warlock.pipelines.trellis import TrellisServer


def test_argv_defaults_webp_off(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    argv = srv._argv()
    assert argv[argv.index("--webp") + 1] == "off"


def test_argv_webp_on_when_configured(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971, webp=True)
    argv = srv._argv()
    assert argv[argv.index("--webp") + 1] == "on"


def test_argv_tex_res_defaults_to_512(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    argv = srv._argv()
    assert argv[argv.index("--tex-res") + 1] == "512"


def test_argv_tex_res_configurable(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971, tex_res=1024)
    argv = srv._argv()
    assert argv[argv.index("--tex-res") + 1] == "1024"


def test_argv_omits_band_when_unset(tmp_path):
    """No band flag at all -- the exe's res/512 heuristic is what runs then."""
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    assert "--band" not in srv._argv()


def test_argv_band_configurable(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971, band=4)
    argv = srv._argv()
    assert argv[argv.index("--band") + 1] == "4"


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


# --- stop() concurrency -----------------------------------------------------
#
# stop() is reachable from several threads at once: queue.py dispatches it via
# asyncio.to_thread for cancel, idle eviction, the VRAM handoff and shutdown,
# and shutdown() can overlap with request_cancel's own call. The whole body is
# check-then-act on _proc/_reader/_logfh, so without a lock two callers can
# both join the reader, both close the log handle, and one can null _proc
# between another's `is not None` and its `.poll()`.


class _CountingProc:
    """A process that is already dead. Records how often it is torn down."""

    def __init__(self) -> None:
        self.stdout = None
        self.returncode = 0
        self.terminates = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminates += 1


def test_stop_is_idempotent(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    srv._proc = _CountingProc()
    srv.stop()
    assert srv._proc is None
    # A second call has nothing left to tear down and must not raise.
    srv.stop()
    srv.stop()
    assert srv._proc is None


def test_stop_on_a_never_started_server_is_a_noop(tmp_path):
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    srv.stop()
    assert srv._proc is None
    assert srv._reader is None


def test_concurrent_stops_do_not_double_tear_down(tmp_path):
    """Eight threads racing stop(): exactly one teardown, no exceptions."""
    import threading

    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    closed = {"n": 0}

    class _Log:
        def close(self):
            closed["n"] += 1

    joins = {"n": 0}

    class _Reader:
        def join(self, timeout=None):
            joins["n"] += 1
            # Widen the window the unlocked version would have raced in.
            time.sleep(0.01)

    srv._proc = _CountingProc()
    srv._reader = _Reader()
    srv._logfh = _Log()

    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def run():
        start.wait()
        try:
            srv.stop()
        except BaseException as exc:  # noqa: BLE001 - the point is to see any
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert joins["n"] == 1
    assert closed["n"] == 1
    assert srv._proc is None
    assert srv._reader is None
    assert srv._logfh is None
