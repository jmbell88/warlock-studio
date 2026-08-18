from __future__ import annotations

import asyncio
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from warlock.glbio import rebuild_glb
from warlock.pipelines import trellis as trellis_mod
from warlock.pipelines.trellis import TrellisServer


def _valid_glb(meshes: list | None = None) -> bytes:
    """The smallest thing the server could legitimately return: a GLB whose
    JSON chunk declares one mesh, plus an empty BIN chunk."""
    binary = b"\x00" * 4
    chunk = struct.pack("<II", len(binary), 0x004E4942) + binary
    header = struct.pack("<III", 0x46546C67, 2, 0)
    gltf = {"asset": {"version": "2.0"}}
    gltf["meshes"] = [{"primitives": []}] if meshes is None else meshes
    return rebuild_glb(header, gltf, chunk)


def _run_generate(tmp_path, monkeypatch, body, sent=None, *, status=200):
    class FakeResponse:
        status_code = status
        content = body
        text = body.decode("utf-8", "replace") if status >= 400 else ""

        async def aread(self):
            return body

        async def aiter_bytes(self):
            # One chunk, which is what a small GLB arrives as anyway. The
            # ceiling is a running total, so a single oversized chunk trips it
            # exactly as a hundred small ones would.
            yield body

    class _Stream:
        """``client.stream(...)``'s async context manager."""

        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *exc):
            return False

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, files=None, data=None):
            if sent is not None:
                sent.update(data)
            return _Stream()

    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", FakeClient)
    server = trellis_mod.TrellisServer(tmp_path / "x.exe", tmp_path, 1234)
    monkeypatch.setattr(server, "ensure_started", _noop_async)
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    out = tmp_path / "out.glb"
    asyncio.run(server.generate(image, out, seed=7, bg_removal="birefnet"))
    return out


def test_generate_forwards_bg_removal(tmp_path, monkeypatch):
    """The exe accepts a bg_removal form field; we never used to send it."""
    sent = {}
    out = _run_generate(tmp_path, monkeypatch, _valid_glb(), sent)
    assert sent["bg_removal"] == "birefnet"
    assert sent["seed"] == "7"
    assert out.read_bytes() == _valid_glb()


def test_generate_rejects_a_200_that_is_not_a_glb(tmp_path, monkeypatch):
    """A proxy or an error page arriving with status 200 used to be written
    onto source.glb and carried all the way to a "done" job."""
    with pytest.raises(RuntimeError, match="invalid GLB"):
        _run_generate(tmp_path, monkeypatch, b"<html>500 oops</html>" * 4)
    assert not (tmp_path / "out.glb").exists()
    assert list(tmp_path.glob("out.glb*")) == []


def test_generate_rejects_a_truncated_body(tmp_path, monkeypatch):
    """Shorter than a GLB header: struct.error, not ValueError."""
    with pytest.raises(RuntimeError, match="invalid GLB"):
        _run_generate(tmp_path, monkeypatch, b"glTF")
    assert not (tmp_path / "out.glb").exists()


def test_generate_rejects_a_glb_with_no_meshes(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="no meshes"):
        _run_generate(tmp_path, monkeypatch, _valid_glb(meshes=[]))
    assert not (tmp_path / "out.glb").exists()


def test_generate_refuses_a_body_past_the_ceiling(tmp_path, monkeypatch):
    """An unbounded read of whatever the server sends is a way for one bad (or
    misaddressed) response to take the whole process out of memory. The refusal
    happens while the body is still arriving, so nothing is written."""
    monkeypatch.setattr(trellis_mod, "MAX_GLB_BYTES", 16)
    with pytest.raises(RuntimeError, match="more than 16 bytes"):
        _run_generate(tmp_path, monkeypatch, _valid_glb())
    assert not (tmp_path / "out.glb").exists()
    assert list(tmp_path.glob("out.glb*")) == []


def test_generate_still_reports_an_error_status_in_the_servers_words(tmp_path, monkeypatch):
    """A streamed response has to be read before ``.text`` exists; the error
    shape a user sees is unchanged."""
    with pytest.raises(RuntimeError, match="trellis-server 503: out of memory"):
        _run_generate(tmp_path, monkeypatch, b"out of memory", status=503)


async def _noop_async(*_a, **_k):
    return None


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

    def __init__(self, returncode: int | None = None) -> None:
        self.stdout = None
        # A code makes it a process that died on its own -- a bind failure, in
        # every case that matters here.
        self.returncode = returncode
        # winjob.assign reads .pid right after Popen; 0 is never a real pid, so
        # OpenProcess fails and assign logs and returns False, as designed.
        self.pid = 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -1

    def wait(self, timeout=None):
        return self.returncode

    def kill(self) -> None:
        self.returncode = -1


class _FakeHealthyClient:
    """A server that answers /health at once."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, *_args, **_kwargs):
        return SimpleNamespace(status_code=200)


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
    # The bind-precheck runs before the poll loop this test is about, and the
    # port is a real one -- an orphan (or anything else) holding it would fail
    # the run for an unrelated reason.
    monkeypatch.setattr(trellis_mod, "_port_in_use", lambda _port: False)

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


@pytest.mark.asyncio
async def test_ensure_started_refuses_a_port_an_orphan_already_holds(tmp_path, monkeypatch):
    """/health carries no identity field, so an orphaned server left by a crash
    answers the readiness poll exactly like the one we are about to spawn --
    which dies on bind. Every generate would then silently go to the orphan."""
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    srv = TrellisServer(exe, models, 17971)

    spawned = []
    monkeypatch.setattr(
        trellis_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a) or _FakeProc()
    )
    monkeypatch.setattr(trellis_mod, "_port_in_use", lambda _port: True)
    # Nothing owns the port as far as the TCP table is concerned (off Windows,
    # always) -- the holder cannot be identified, so it cannot be reclaimed.
    monkeypatch.setattr(trellis_mod.winjob, "listener_pid", lambda _port: None)

    with pytest.raises(RuntimeError, match="already in use"):
        await srv.ensure_started()
    assert spawned == []


@pytest.mark.asyncio
async def test_an_orphan_of_our_own_exe_is_reclaimed(tmp_path, monkeypatch):
    """The job object stops new orphans; it cannot clear one a pre-fix crash
    already stranded.

    Killing it takes *two* proofs, not one. The holder's image must be the very
    exe this instance is configured to spawn -- and this home must have claimed
    the port, with the claim naming an owner that is gone. Executable identity
    alone is a property of the install rather than of the instance, so on its
    own it would let a second Warlock (different WARLOCK_HOME, same binary, same
    default port) terminate the first one's live server (RUN-01).
    """
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    srv = TrellisServer(exe, models, 17971, log_path=tmp_path / "trellis.log")
    # The claim a crashed previous run left behind: an owner pid that is gone.
    (tmp_path / "trellis-17971.owner").write_text(
        '{"owner_pid": 999999001, "server_pid": 4321}', encoding="utf-8"
    )

    spawned: list = []
    killed: list[int] = []
    held = {"port": True}

    monkeypatch.setattr(
        trellis_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a) or _FakeProc()
    )
    monkeypatch.setattr(trellis_mod, "_port_in_use", lambda _port: held["port"])
    monkeypatch.setattr(trellis_mod.winjob, "listener_pid", lambda _port: 4321)
    monkeypatch.setattr(
        trellis_mod.winjob,
        "image_path",
        lambda pid: None if pid == 999999001 else str(exe.resolve()),
    )
    monkeypatch.setattr(trellis_mod.winjob, "assign", lambda _pid: True)

    def _kill(pid):
        killed.append(pid)
        held["port"] = False
        return True

    monkeypatch.setattr(trellis_mod.winjob, "terminate", _kill)
    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", _FakeAsyncClient)

    # The health poll never succeeds against the fake client, so the run ends
    # at the startup timeout -- what is under test is that it got as far as
    # spawning at all, having killed the orphan first.
    monkeypatch.setattr(trellis_mod, "STARTUP_TIMEOUT", 0.0)
    with pytest.raises(RuntimeError, match="healthy"):
        await srv.ensure_started()

    assert killed == [4321]
    assert len(spawned) == 1


@pytest.mark.asyncio
async def test_a_port_held_by_a_stranger_is_never_killed(tmp_path, monkeypatch):
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    srv = TrellisServer(exe, models, 17971)

    spawned: list = []
    killed: list[int] = []
    monkeypatch.setattr(
        trellis_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a) or _FakeProc()
    )
    monkeypatch.setattr(trellis_mod, "_port_in_use", lambda _port: True)
    monkeypatch.setattr(trellis_mod.winjob, "listener_pid", lambda _port: 4321)
    monkeypatch.setattr(
        trellis_mod.winjob, "image_path", lambda _pid: r"C:\Program Files\Somebody\server.exe"
    )
    monkeypatch.setattr(trellis_mod.winjob, "terminate", lambda pid: killed.append(pid) or True)

    with pytest.raises(RuntimeError, match="not this Warlock's trellis-server"):
        await srv.ensure_started()
    assert killed == [] and spawned == []


@pytest.mark.asyncio
async def test_a_failed_startup_stops_off_the_event_loop(tmp_path, monkeypatch):
    """stop() blocks for up to ~25 s, and the not-healthy-in-time path used to
    be the one caller that ran it inline on the warlock-loop thread. It must
    dispatch through asyncio.to_thread like every other caller -- and a
    TrellisStopFailed from that teardown must stay suppressed, so it cannot
    mask the diagnosis that brought us here."""
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    srv = TrellisServer(exe, models, 17971)

    monkeypatch.setattr(trellis_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(trellis_mod, "_port_in_use", lambda _port: False)
    monkeypatch.setattr(trellis_mod.winjob, "assign", lambda _pid: True)
    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(trellis_mod, "STARTUP_TIMEOUT", 0.0)

    loop_thread = threading.current_thread()
    stopped_on: list[threading.Thread] = []

    def fake_stop() -> None:
        stopped_on.append(threading.current_thread())
        raise trellis_mod.TrellisStopFailed("simulated: still alive")

    monkeypatch.setattr(srv, "stop", fake_stop)

    with pytest.raises(RuntimeError, match="did not become healthy in time"):
        await srv.ensure_started()

    assert stopped_on, "the failure path must still tear the spawn down"
    assert all(t is not loop_thread for t in stopped_on), (
        "stop() ran inline on the event-loop thread"
    )


@pytest.mark.asyncio
async def test_a_failed_start_is_not_retried_immediately(tmp_path, monkeypatch):
    """Five startup banners in one minute, each dying on bind, is what the
    2026-08-03 log actually contains: every generate() respawned at once and
    buried the first failure under identical repeats."""
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    srv = TrellisServer(exe, models, 17971)

    spawned: list = []
    monkeypatch.setattr(
        trellis_mod.subprocess, "Popen", lambda *a, **k: spawned.append(a) or _FakeProc(1)
    )
    monkeypatch.setattr(trellis_mod, "_port_in_use", lambda _port: False)
    monkeypatch.setattr(trellis_mod.winjob, "assign", lambda _pid: True)
    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", _FakeAsyncClient)

    with pytest.raises(RuntimeError, match="exited during startup"):
        await srv.ensure_started()
    assert len(spawned) == 1

    # Immediately again: refused without touching Popen.
    with pytest.raises(RuntimeError, match="refusing to respawn"):
        await srv.ensure_started()
    assert len(spawned) == 1

    # Once the window has passed it tries again, and a healthy server resets
    # the counters so the next failure starts from the shortest delay.
    srv._backoff_until = 0.0
    monkeypatch.setattr(trellis_mod.subprocess, "Popen", lambda *a, **k: _FakeProc())
    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", _FakeHealthyClient)
    await srv.ensure_started()
    assert srv._start_failures == 0 and srv._backoff_until == 0.0


def test_port_in_use_sees_a_bound_socket():
    import socket

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert trellis_mod._port_in_use(port) is True
    finally:
        sock.close()
    assert trellis_mod._port_in_use(port) is False


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


# --- stop() has to confirm the death it reports -----------------------------


class _StubbornProc:
    """A process that ignores terminate() and, optionally, kill() too."""

    def __init__(self, *, dies_on_kill: bool = True) -> None:
        self.stdout = None
        self.stdin = None
        self.returncode: int | None = None
        self.pid = 4242
        self.terminates = 0
        self.kills = 0
        self.waits: list[float | None] = []
        self._dies_on_kill = dies_on_kill

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminates += 1

    def kill(self):
        self.kills += 1
        if self._dies_on_kill:
            self.returncode = -9

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="trellis-server", timeout=timeout or 0)
        return self.returncode


def test_stop_waits_after_kill_before_reporting_success(tmp_path):
    # kill() only *asks*. Windows reaps asynchronously, so without the wait the
    # handle was cleared regardless -- `running` went false and the caller (the
    # VRAM handoff, three call sites) loaded an image model into memory a live
    # process still held.
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    proc = _StubbornProc()
    srv._proc = proc
    srv.stop()
    assert proc.terminates == 1 and proc.kills == 1
    # 15 for the polite wait, then KILL_TIMEOUT for the reap.
    assert proc.waits == [15, trellis_mod.KILL_TIMEOUT]
    assert srv._proc is None


def test_a_stop_that_cannot_confirm_death_raises_and_keeps_the_handle(tmp_path):
    # The handle must survive: it may still own ~16 GiB, and it is the only
    # reference to a process nothing else tracks -- clearing it would leave
    # `running` false and _reap_if_dead with nothing to collect.
    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    proc = _StubbornProc(dies_on_kill=False)
    srv._proc = proc
    with pytest.raises(trellis_mod.TrellisStopFailed, match="4242"):
        srv.stop()
    assert srv._proc is proc
    assert srv.running is True


def test_a_server_that_terminates_politely_is_never_killed(tmp_path):
    class _Polite(_StubbornProc):
        def terminate(self):
            self.terminates += 1
            self.returncode = 0

    srv = TrellisServer(tmp_path / "exe", tmp_path / "models", 17971)
    proc = _Polite()
    srv._proc = proc
    srv.stop()
    assert proc.terminates == 1 and proc.kills == 0
    assert srv._proc is None


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


@pytest.mark.asyncio
async def test_a_live_second_warlocks_server_is_never_terminated(tmp_path, monkeypatch):
    """RUN-01's sharp edge: the exe path is a property of the *install*.

    Two Warlocks with different WARLOCK_HOMEs share the same binary and, by
    default, the same port 17971. Executable identity alone therefore "proved"
    the other one's *live* server was our orphan, and reclaiming it terminated
    it -- with the kill logged as a successful cleanup.
    """
    exe = tmp_path / "trellis-server.exe"
    exe.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    srv = TrellisServer(exe, models, 17971, log_path=tmp_path / "trellis.log")

    killed: list[int] = []
    monkeypatch.setattr(trellis_mod, "_port_in_use", lambda _port: True)
    monkeypatch.setattr(trellis_mod.winjob, "listener_pid", lambda _port: 4321)
    monkeypatch.setattr(trellis_mod.winjob, "image_path", lambda _pid: str(exe.resolve()))
    monkeypatch.setattr(trellis_mod.winjob, "terminate", lambda pid: killed.append(pid))

    # No claim file at all: this home never started a server on this port, so
    # whoever is there belongs to somebody else.
    with pytest.raises(RuntimeError, match="did not start"):
        await srv.ensure_started()
    assert killed == []

    # And with a claim naming an owner that is still alive -- the other Warlock
    # running right now -- it is still refused, in different words.
    (tmp_path / "trellis-17971.owner").write_text(
        '{"owner_pid": 4242, "server_pid": 4321}', encoding="utf-8"
    )
    monkeypatch.setattr(trellis_mod.winjob, "image_path", lambda pid: str(exe.resolve()))
    with pytest.raises(RuntimeError, match="still running"):
        await srv.ensure_started()
    assert killed == []


def test_an_error_body_is_streamed_and_capped_like_a_success_body(tmp_path, monkeypatch):
    """MDL-13: the error path called ``await r.aread()`` -- unbounded.

    Only ~500 characters are ever displayed, but the whole body was pulled into
    memory first, so a wedged or compromised local server could answer a failing
    request with gigabytes. The success path has streamed under a ceiling for
    exactly this reason; the error path simply was not given one.
    """
    monkeypatch.setattr(trellis_mod, "MAX_ERROR_BYTES", 32)
    consumed = {"bytes": 0}

    class FakeResponse:
        status_code = 500

        async def aiter_bytes(self):
            # A server that keeps talking. If the ceiling is not honoured this
            # never ends, which is the failure in its purest form -- so the
            # generator is bounded far above the cap and the assertion below is
            # about how much was actually taken.
            for _ in range(1000):
                consumed["bytes"] += 16
                yield b"x" * 16

    class _Stream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *exc):
            return False

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *a, **k):
            return _Stream()

    monkeypatch.setattr(trellis_mod.httpx, "AsyncClient", FakeClient)
    server = trellis_mod.TrellisServer(tmp_path / "x.exe", tmp_path, 1234)
    monkeypatch.setattr(server, "ensure_started", _noop_async)
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    with pytest.raises(RuntimeError, match="trellis-server 500"):
        asyncio.run(server.generate(image, tmp_path / "out.glb"))
    assert consumed["bytes"] <= 64, "the whole body was read despite the cap"


def test_atomic_write_stages_through_a_dotfile(tmp_path, monkeypatch):
    """The staged-writes rule: the staging sibling is a dotfile, not a visible
    ``source.glbXXXX.tmp`` beside the served name."""
    import os
    from pathlib import Path

    staged: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        staged.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(trellis_mod.os, "replace", spy)
    dest = tmp_path / "source.glb"
    trellis_mod._atomic_write(dest, b"payload")

    assert dest.read_bytes() == b"payload"
    assert staged and staged[0].startswith(".source.glb.")
    assert staged[0].endswith(".tmp")
    assert [p.name for p in tmp_path.iterdir()] == ["source.glb"]
