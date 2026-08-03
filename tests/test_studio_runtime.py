"""Startup, shutdown and the task pool.

Real threads and a real event loop -- the point of these is the handoff
between them, which a mock would define away.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from warlock.service.errors import Invalid
from warlock.studio.runtime import Runtime
from warlock.studio.tasks import TaskRunner


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
    app._last_frame = 0.0

    def setup():
        runtime.start()
        raise RuntimeError("no OpenGL 3.3")

    app.setup = setup
    with pytest.raises(RuntimeError, match="no OpenGL"):
        app.run()
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
