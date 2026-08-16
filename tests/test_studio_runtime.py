"""Startup, shutdown and the task pool.

Real threads and a real event loop -- the point of these is the handoff
between them, which a mock would define away.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time

import pytest

from warlock.service.errors import Invalid
from warlock.studio.fps import FpsMeter
from warlock.studio.runtime import Runtime
from warlock.studio.tasks import TaskRunner, _threads_queues


@pytest.fixture
def runtime(tmp_path, monkeypatch, fake_pipelines):
    import warlock.config as config_mod
    from warlock.config import get_config

    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("WARLOCK_DB", str(tmp_path / "assets" / "jobs.sqlite"))
    monkeypatch.setenv("WARLOCK_TRELLIS_EXE", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(config_mod, "_config", None)
    rt = Runtime(get_config())
    yield rt
    rt.shutdown()


# --- runtime ----------------------------------------------------------------


def test_starting_gives_a_service_with_a_live_worker(runtime):
    svc = runtime.start()
    assert svc.store is not None
    assert runtime.alive is True
    assert runtime.fatal is None


def test_the_vram_mode_is_resolved_before_the_worker_exists(runtime, monkeypatch):
    """queue.py reads a plain bool once, at startup. Resolving it any later --
    or leaving the tri-state None on the config -- breaks that contract."""
    seen: list[object] = []
    original = runtime._make_worker

    async def spy():
        seen.append(runtime.config.vram_exclusive)
        return await original()

    monkeypatch.setattr(runtime, "_make_worker", spy)
    monkeypatch.setattr(runtime.config, "vram_exclusive", None)
    monkeypatch.setattr(runtime.config, "vram_total_gib", 8.0)
    svc = runtime.start()
    assert seen == [True]  # 8 GiB cannot coexist, so auto picks exclusive
    assert runtime.vram_plan is not None
    assert svc.vram_plan is runtime.vram_plan


def test_the_worker_runs_on_its_own_thread(runtime):
    runtime.start()
    assert threading.current_thread().name != "warlock-loop"
    assert any(t.name == "warlock-loop" for t in threading.enumerate())


def test_waking_the_worker_from_the_main_thread_does_not_explode(runtime):
    """wake() is loop-affine, so every caller outside the loop needs the hop."""
    svc = runtime.start()
    svc.wake_worker()  # a plain call from the test's own thread
    time.sleep(0.05)
    assert runtime.alive is True


def test_a_job_still_running_at_startup_is_reconciled(runtime, tmp_path):
    from warlock.db import JobStore

    store = JobStore(runtime.config.db_path)
    job_id = store.create("text", "x", {}, "aaaaaaaaaaaa")
    store.set_status(job_id, "running")
    store.close()

    svc = runtime.start()
    # Surfaced, not silently re-run: a crash mid-render should not cost two
    # minutes of GPU on every subsequent launch.
    assert svc.store.get(job_id)["status"] != "running"


def test_an_error_on_the_worker_loop_is_logged_not_printed(runtime, caplog):
    """asyncio's default handler writes to a stderr nobody keeps."""
    from warlock.studio.runtime import _loop_error

    runtime.start()
    assert runtime._loop.get_exception_handler() is _loop_error
    with caplog.at_level(logging.CRITICAL):
        _loop_error(runtime._loop, {"message": "Task exception was never retrieved",
                                    "exception": RuntimeError("boom")})
    assert any("worker loop" in r.message for r in caplog.records)


def test_shutdown_is_idempotent(runtime):
    runtime.start()
    runtime.shutdown()
    runtime.shutdown()
    assert runtime.worker is None


def test_progress_is_none_when_nothing_is_running(runtime):
    runtime.start()
    assert runtime.progress() is None
    assert runtime.current_job_id is None


def test_the_runtime_works_as_a_context_manager(runtime):
    with runtime as svc:
        assert svc is not None
    assert runtime.store is None


def test_a_failed_start_leaves_nothing_running(runtime):
    """The store connection and the loop thread are both open before the worker
    is built, so a constructor that raises used to leak both."""
    store = {}

    def explode(self):
        store["conn"] = self.store
        raise RuntimeError("no GPU today")

    runtime._make_worker = lambda: explode(runtime)
    with pytest.raises(RuntimeError, match="no GPU today"):
        runtime.start()

    assert runtime.store is None and runtime.worker is None
    assert runtime._loop is None and runtime._thread is None
    # The connection really is closed, not just dropped.
    with pytest.raises(sqlite3.ProgrammingError):
        store["conn"]._conn.execute("select 1")
    assert not any(t.name == "warlock-loop" and t.is_alive() for t in threading.enumerate())


def test_run_tears_down_when_setup_fails(runtime, monkeypatch):
    """setup() starts the runtime before it touches pygame or GL, so a failure
    past that point must still unwind it."""
    from warlock.studio.main import App

    app = App.__new__(App)
    app.runtime = runtime
    app.window = app.imgui_renderer = app.viewer = app.app_ctx = app.eta = None
    app._running = False
    app._last_frame = app._started_at = 0.0
    app.fps = FpsMeter()

    def setup():
        runtime.start()
        raise RuntimeError("no OpenGL 3.3")

    # setup_window rather than setup: run() calls the three phases directly so
    # the splash can be drawn over the middle one, which makes an `app.setup`
    # stub a silent no-op that lets run() reach the real pygame init.
    app.setup_window = setup
    # Reported, not re-raised: the traceback goes to warlock.log, and the exit
    # code is what the caller acts on.
    assert app.run() == 1
    assert runtime.store is None and runtime.worker is None


# --- tasks ------------------------------------------------------------------


@pytest.fixture
def tasks():
    runner = TaskRunner(workers=2)
    yield runner
    runner.shutdown()


def _wait(runner, key, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for done in runner.poll():
            if done.key == key:
                return done
        time.sleep(0.01)
    raise AssertionError(f"{key} never finished")


def test_a_result_comes_back_on_a_later_poll(tasks):
    assert tasks.submit("work", lambda: 6 * 7) is True
    assert _wait(tasks, "work").result == 42


def test_a_key_already_in_flight_is_refused_not_queued(tasks):
    """A double-clicked download must be one export, not two fighting over the
    same file."""
    gate = threading.Event()
    tasks.submit("slow", gate.wait)
    assert tasks.submit("slow", gate.wait) is False
    assert tasks.is_busy("slow") is True
    gate.set()
    _wait(tasks, "slow")
    assert tasks.is_busy("slow") is False


def test_the_same_key_can_be_submitted_again_once_it_has_finished(tasks):
    tasks.submit("once", lambda: 1)
    _wait(tasks, "once")
    assert tasks.submit("once", lambda: 2) is True
    assert _wait(tasks, "once").result == 2


def test_a_service_error_arrives_as_its_own_message(tasks):
    def boom():
        raise Invalid("size must be between 0.01 and 100 m", field="size_m")

    tasks.submit("bad", boom)
    done = _wait(tasks, "bad")
    assert done.ok is False
    assert done.message == "size must be between 0.01 and 100 m"


def test_an_unexpected_error_gets_a_generic_message_not_a_traceback(tasks):
    def boom():
        raise ZeroDivisionError("division by zero")

    tasks.submit("crash", boom)
    done = _wait(tasks, "crash")
    assert done.ok is False
    assert "division by zero" not in (done.message or "")
    assert isinstance(done.error, ZeroDivisionError)


def test_a_tag_rides_along_to_identify_what_finished(tasks):
    tasks.submit("save", lambda: None, tag={"job": "abc", "name": "model.stl"})
    assert _wait(tasks, "save").tag["name"] == "model.stl"


def test_polling_never_blocks_on_a_task_that_is_still_running(tasks):
    gate = threading.Event()
    tasks.submit("held", gate.wait)
    start = time.monotonic()
    assert tasks.poll() == []
    assert time.monotonic() - start < 0.05
    gate.set()


def test_busy_keys_can_be_asked_about_by_prefix(tasks):
    gate = threading.Event()
    tasks.submit("derive:model.stl", gate.wait)
    assert tasks.any_busy("derive:") is True
    assert tasks.any_busy("export:") is False
    gate.set()
    _wait(tasks, "derive:model.stl")


@pytest.mark.perf
def test_a_bounded_shutdown_returns_while_a_task_is_still_parked():
    """A native save dialog blocks until the user dismisses it, and by shutdown
    the window it belongs to is already destroyed. An unbounded wait there is a
    process that never exits."""
    runner = TaskRunner(workers=2)
    gate = threading.Event()
    runner.submit("stuck", gate.wait)
    try:
        start = time.monotonic()
        drained = runner.shutdown(timeout=0.2)
        elapsed = time.monotonic() - start
        assert 0.15 < elapsed < 2.0
        assert runner.busy_keys == set()
        # And it says so, because the caller's next move depends on it:
        # Runtime.shutdown closes the store immediately afterwards, on the
        # grounds that an in-flight task finds it open -- true only if the pool
        # actually drained. It did not.
        assert drained is False
        # And the stuck thread is no longer something interpreter exit joins on.
        assert not any(t in _threads_queues for t in runner._pool._threads)
    finally:
        gate.set()


def test_a_bounded_shutdown_of_an_idle_pool_returns_at_once():
    runner = TaskRunner(workers=2)
    runner.submit("quick", lambda: 1)
    start = time.monotonic()
    drained = runner.shutdown(timeout=5.0)
    assert time.monotonic() - start < 1.0
    assert drained is True


def test_the_store_is_left_open_when_the_task_pool_did_not_drain(tmp_path):
    """The regression: ``Runtime.shutdown`` closed the store unconditionally,
    on the stated grounds that a task still calling into the service would find
    it open -- which holds only while the pool drains. The very next comment
    says the wait is *bounded*. A task outliving the grace period (the
    documented case: parked in a never-dismissed native dialog) then resumed
    against a closed connection and got a ProgrammingError captured into a
    future nobody will ever poll."""
    from types import SimpleNamespace

    from warlock.studio import runtime as runtime_mod

    closed: list[str] = []
    rt = runtime_mod.Runtime.__new__(runtime_mod.Runtime)
    rt.worker = None
    rt._loop = None
    rt._thread = None
    rt.svc = None
    rt.store = SimpleNamespace(close=lambda: closed.append("closed"))

    rt.tasks = SimpleNamespace(shutdown=lambda timeout=None: False)
    runtime_mod.Runtime.shutdown(rt)
    assert closed == [], "something is still running; the connection stays up"

    rt.store = SimpleNamespace(close=lambda: closed.append("closed"))
    rt.tasks = SimpleNamespace(shutdown=lambda timeout=None: True)
    runtime_mod.Runtime.shutdown(rt)
    assert closed == ["closed"]


def test_an_unexpected_failure_offers_the_log_it_was_written_to(tasks):
    """The generic toast says "see the log for details" and, before this,
    offered no way to reach it -- the only Open-the-log button in the app is
    inside the diagnostics popup, three clicks from a toast that lasts eight
    seconds."""

    def boom():
        raise ValueError("weird")

    tasks.submit("odd", boom)
    done = _wait(tasks, "odd")
    assert done.action == "log"


def test_a_service_error_needs_no_log_route(tasks):
    """Its message was written for a person and names the remedy; a button to
    a log file is noise beside it, and the log has nothing extra to say."""

    def boom():
        raise Invalid("size must be between 0.01 and 100 m", field="size_m")

    tasks.submit("bad-size", boom)
    assert _wait(tasks, "bad-size").action is None


def test_a_successful_task_carries_no_action(tasks):
    tasks.submit("fine", lambda: 1)
    assert _wait(tasks, "fine").action is None
