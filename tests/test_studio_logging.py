"""What a crash leaves behind.

Every test here is about evidence rather than behaviour. The app died silently
twice -- 2026-08-03 to commit exhaustion, 2026-08-04 to something still unknown
-- and both investigations had to be run off Windows event logs because the
app's own warlock.log was created on every launch and never written to. These
pin the repairs: the file log actually receives records on the launch path the
console script uses, uncaught exceptions reach it, teardown cannot be skipped
by a failing stage, and an unclean exit says so on the next launch.
"""

from __future__ import annotations

import contextlib
import faulthandler
import json
import logging
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from warlock.studio import main
from warlock.studio.fps import FpsMeter


@pytest.fixture
def clean_root_logging():
    """Restore root logging, since _setup_logging deliberately reconfigures it.

    Handlers added by the test are closed, not merely dropped: on Windows an
    open RotatingFileHandler keeps a lock on warlock.log and tmp_path cleanup
    fails behind it.
    """
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()
    root.handlers[:] = before
    root.setLevel(level)
    faulthandler.disable()
    if main._crash_log is not None:
        main._crash_log.close()
        main._crash_log = None


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    import warlock.config as config_mod

    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(assets))
    monkeypatch.setenv("WARLOCK_DB", str(assets / "jobs.sqlite"))
    monkeypatch.setattr(config_mod, "_config", None)
    yield assets
    monkeypatch.setattr(config_mod, "_config", None)


# --- the file log ------------------------------------------------------------


def test_the_file_log_survives_an_earlier_basicconfig(clean_root_logging, data_dir):
    """The bug that cost us the 2026-08-04 crash record.

    cli.main() configured the root logger before dispatching to the app, so
    _setup_logging's own basicConfig was a no-op against an already-configured
    root: the file handler was built, opened, and then dropped on the floor.
    Replicated verbatim here rather than by calling cli.main, which would open
    a window.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    main._setup_logging()
    logging.getLogger("warlock.test").info("probe-record")

    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "probe-record" in (data_dir / "warlock.log").read_text(encoding="utf-8")


def test_the_cli_does_not_configure_logging_on_the_app_path():
    """The other half: main.py's force=True fixes the symptom, but the app path
    should not be pre-configuring the root logger at all."""
    import ast
    import inspect

    from warlock import cli

    tree = ast.parse(inspect.getsource(cli.main))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "basicConfig"
    ]
    # One per sub-command branch, and none at the top level of main().
    assert len(calls) == 2
    top_level = [n for n in tree.body[0].body if isinstance(n, ast.Expr)]
    assert not [
        n for n in top_level
        if isinstance(n.value, ast.Call) and getattr(n.value.func, "attr", "") == "basicConfig"
    ]


def test_the_crash_log_carries_a_session_header(clean_root_logging, data_dir):
    """faulthandler appends across runs; a dump with no session line above it
    belongs to no process anybody can identify."""
    main._setup_logging()
    main._crash_log.flush()

    header = (data_dir / "crash.log").read_text(encoding="utf-8")
    assert "=== session " in header
    assert f"pid={os.getpid()}" in header


# --- the session marker ------------------------------------------------------


def test_a_marker_from_a_dead_pid_reports_an_unclean_shutdown(data_dir, caplog):
    (data_dir / main.SESSION_MARKER).write_text(
        json.dumps({"pid": 999_999_999, "started_at": "2026-08-04T15:37:00+00:00",
                    "version": "0.0.6"}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        main._note_previous_session()
    assert any("did not shut down cleanly" in r.getMessage() for r in caplog.records)


def test_a_marker_from_a_live_pid_reports_a_second_instance(data_dir, caplog, monkeypatch):
    """Two Warlocks share one sqlite DB and one trellis port, which is worth
    saying out loud before either misbehaves."""
    monkeypatch.setattr(main, "_pid_alive", lambda pid: True)
    (data_dir / main.SESSION_MARKER).write_text(
        json.dumps({"pid": os.getpid() + 1, "started_at": "now", "version": "0.0.6"}),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        main._note_previous_session()
    messages = [r.getMessage() for r in caplog.records]
    assert any("appears to be running" in m for m in messages)
    assert not any("did not shut down cleanly" in m for m in messages)


def test_no_marker_says_nothing(data_dir, caplog):
    with caplog.at_level(logging.WARNING):
        main._note_previous_session()
    assert caplog.records == []


def test_the_marker_round_trips_and_clearing_is_idempotent(data_dir):
    main._write_session_marker()
    written = json.loads((data_dir / main.SESSION_MARKER).read_text(encoding="utf-8"))
    assert written["pid"] == os.getpid()
    assert written["version"] == main._version()

    main._clear_session_marker()
    main._clear_session_marker()
    assert not (data_dir / main.SESSION_MARKER).exists()


def test_pid_alive_knows_this_process_and_not_a_dead_one():
    assert main._pid_alive(os.getpid()) is True
    assert main._pid_alive(999_999_999) is False
    assert main._pid_alive(0) is False


# --- uncaught exceptions -----------------------------------------------------


def test_an_uncaught_exception_reaches_the_log(caplog, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(sys, "unraisablehook", sys.unraisablehook)
    main._install_excepthooks()

    try:
        raise ValueError("frame loop said no")
    except ValueError:
        with caplog.at_level(logging.CRITICAL):
            sys.excepthook(*sys.exc_info())

    record = next(r for r in caplog.records if r.levelno == logging.CRITICAL)
    assert record.exc_info is not None


def test_a_dying_daemon_thread_is_not_silent(caplog, monkeypatch):
    """warlock-loop and trellis' stdout reader are daemons: nothing at all is
    printed when one of them dies."""
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    main._install_excepthooks()

    args = SimpleNamespace(
        exc_type=RuntimeError, exc_value=RuntimeError("gone"), exc_traceback=None,
        thread=SimpleNamespace(name="warlock-loop"),
    )
    with caplog.at_level(logging.CRITICAL):
        threading.excepthook(args)
    assert any("warlock-loop" in (r.getMessage()) for r in caplog.records)

    caplog.clear()
    args.exc_type = SystemExit
    with caplog.at_level(logging.CRITICAL):
        threading.excepthook(args)
    assert caplog.records == []


# --- run() and teardown ------------------------------------------------------


def _bare_app(**runtime_kwargs):
    app = main.App.__new__(main.App)
    app.runtime = SimpleNamespace(shutdown=lambda: None, **runtime_kwargs)
    app.window = app.imgui_renderer = app.viewer = app.app_ctx = app.eta = None
    app._running = False
    app._last_frame = app._started_at = 0.0
    app.fps = FpsMeter()
    return app


def test_the_two_crash_phases_get_different_lines(caplog, monkeypatch):
    """A window that never opened and a window that vanished after twenty
    minutes are different bugs; the log line was the only thing telling them
    apart, when there was one at all."""
    shutdowns = []
    app = _bare_app()
    app.runtime = SimpleNamespace(shutdown=lambda: shutdowns.append(1))

    def bad_setup():
        raise RuntimeError("no OpenGL 3.3")

    # The phases, not App.setup: run() calls setup_window / _startup_with_splash
    # / setup_context directly so the splash can be drawn over the middle one.
    # Stubbing `setup` here used to isolate run() and silently stopped doing so
    # -- run() then fell through into the *real* setup_window, which initialises
    # pygame for real and left a live display behind for every GL test after it.
    app.setup_window = bad_setup
    with caplog.at_level(logging.ERROR):
        assert app.run() == 1
    assert any("could not start" in (r.getMessage()) for r in caplog.records)

    caplog.clear()
    app = _bare_app()
    app.runtime = SimpleNamespace(shutdown=lambda: shutdowns.append(1))
    app.setup_window = lambda: None
    app._startup_with_splash = lambda: True
    app.setup_context = lambda: None

    def bad_frame(dt):
        raise ZeroDivisionError("nan in the pose")

    app.frame = bad_frame
    with caplog.at_level(logging.ERROR):
        assert app.run() == 1
    assert any("crashed mid-session" in (r.getMessage()) for r in caplog.records)
    assert len(shutdowns) == 2


def test_a_failing_teardown_stage_does_not_skip_the_runtime(caplog):
    """runtime.shutdown is last and stops the worker loop and the trellis
    child; a GL release raising on a lost context used to skip it, stranding a
    16 GB server and a process that would not exit."""
    stopped = []
    app = _bare_app()
    app.runtime = SimpleNamespace(shutdown=lambda: stopped.append("runtime"))

    def boom():
        raise RuntimeError("context lost")

    app.viewer = SimpleNamespace(release=boom)
    app.imgui_renderer = SimpleNamespace(shutdown=boom)

    with caplog.at_level(logging.INFO):
        app.teardown()

    assert stopped == ["runtime"]
    failures = [r for r in caplog.records if "teardown:" in r.getMessage()]
    assert len(failures) == 2
    # The line whose absence is the evidence of a hard death.
    assert any("teardown complete" in r.getMessage() for r in caplog.records)
