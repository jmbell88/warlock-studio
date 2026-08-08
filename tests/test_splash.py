"""The splash's lifecycle, which is the whole hazard of showing one.

Not the pixels. The window now exists while the runtime starts, which means
the X button is live during the one stretch of the session where there is no
``Ctx`` to tear down and a half-constructed ``Runtime`` that must still be
unwound. These pin that: a quit mid-load, a teardown with no Ctx, the minimum
hold in both directions, and both halves of the failure report.

The clock is injected everywhere; nothing here sleeps.
"""

from __future__ import annotations

import logging
import sys
import threading
from types import SimpleNamespace

import pytest

from warlock.studio import main, splash


class Clock:
    """A hand-wound perf_counter."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _run(load, clock, **kwargs) -> splash.Startup:
    started = splash.Startup(load, clock=clock, **kwargs)
    started.start()
    return started


# -- the hold --------------------------------------------------------------


def test_a_fast_load_still_holds_the_splash_for_the_minimum():
    clock = Clock()
    started = _run(lambda: None, clock)
    started.join(5.0)
    assert started.loaded
    assert not started.finished()
    clock.advance(splash.MIN_SECONDS - 0.01)
    assert not started.finished()
    clock.advance(0.01)
    assert started.finished()


def test_a_slow_load_outlasts_the_minimum_and_the_splash_waits_for_it():
    clock = Clock()
    release = threading.Event()
    started = _run(release.wait, clock)
    clock.advance(splash.MIN_SECONDS * 10)
    # The minimum is long past and the load has not landed: the splash stays.
    assert not started.loaded
    assert not started.finished()
    release.set()
    started.join(5.0)
    assert started.finished()


def test_a_quit_during_the_splash_waits_for_the_load_then_drops_the_minimum():
    """The X button may not abandon a half-constructed Runtime.

    Nothing else could unwind a store, a loop thread and a worker opened by a
    ``runtime.start()`` still in flight -- so a quit keeps the splash up until
    the load lands, and only then skips the remaining hold.
    """
    clock = Clock()
    release = threading.Event()
    started = _run(release.wait, clock)
    started.request_quit()
    assert started.quit_requested
    assert not started.finished()  # the load is still running
    release.set()
    started.join(5.0)
    # No time has passed at all, and it is finished anyway.
    assert started.elapsed() == 0.0
    assert started.finished()


def test_a_failed_load_ends_the_splash_at_once_and_re_raises_on_the_frame_thread():
    clock = Clock()
    boom = RuntimeError("doctor exploded")

    def load():
        raise boom

    started = _run(load, clock)
    started.join(5.0)
    assert started.finished()  # no 3-second pause over a startup that failed
    with pytest.raises(RuntimeError) as caught:
        started.raise_if_failed()
    assert caught.value is boom


def test_a_load_that_raised_still_reports_itself_loaded():
    """``loaded`` is what lets a quit stop pumping; an exception must set it."""
    started = _run(lambda: 1 / 0, Clock())
    started.join(5.0)
    assert started.loaded


def test_the_logo_is_fitted_down_but_never_up():
    assert splash._fit((100, 100), (1000.0, 1000.0)) == (100.0, 100.0)
    assert splash._fit((2000, 1000), (1000.0, 1000.0)) == (600.0, 300.0)
    assert splash._fit((0, 0), (1000.0, 1000.0)) == (0.0, 0.0)


def test_the_logo_ships_beside_the_icon():
    assert splash.LOGO_PATH.is_file()


def test_the_splash_message_is_drawable_in_the_default_atlas():
    """Basic-Latin + Latin-1 only: a real ellipsis renders as a hollow box."""
    assert all(ord(ch) < 0x100 for ch in splash.MESSAGE)


# -- App lifecycle ---------------------------------------------------------


class FakeRuntime:
    def __init__(self) -> None:
        self.shutdowns = 0
        self.checks: list[object] = []
        self.config = SimpleNamespace(data_dir=None)

    def shutdown(self) -> None:
        self.shutdowns += 1


@pytest.fixture(autouse=True)
def fake_pygame(monkeypatch):
    """A stand-in for the window layer these tests never open.

    ``run`` and ``teardown`` both ``import pygame`` at call time, and what is
    under test here is which branches they take -- not anything pygame does.
    Stubbing it keeps these runnable on a checkout without the studio extra,
    where the real one is exactly the module that is missing.
    """
    stub = SimpleNamespace(
        quit=lambda: None,
        display=SimpleNamespace(flip=lambda: None),
        time=SimpleNamespace(Clock=lambda: SimpleNamespace(tick=lambda _fps: 0)),
    )
    monkeypatch.setitem(sys.modules, "pygame", stub)
    return stub


def test_teardown_is_safe_when_the_context_was_never_built(fake_pygame):
    """The splash's own window: no Ctx, no viewer, no renderer, a live runtime.

    ``runtime.shutdown`` is the step that stops the worker loop and the trellis
    child, so an AttributeError anywhere above it is a stranded subprocess.
    """
    runtime = FakeRuntime()
    app = main.App(runtime)
    app.teardown()
    assert runtime.shutdowns == 1


def test_a_quit_during_the_splash_exits_cleanly_without_building_the_context(monkeypatch):
    runtime = FakeRuntime()
    app = main.App(runtime)
    calls: list[str] = []
    monkeypatch.setattr(app, "setup_window", lambda: calls.append("window"))
    monkeypatch.setattr(app, "_startup_with_splash", lambda: False)
    monkeypatch.setattr(app, "setup_context", lambda: calls.append("context"))

    assert app.run() == 0
    assert calls == ["window"]
    assert app._running is False
    assert runtime.shutdowns == 1


def test_a_failure_during_startup_reaches_the_could_not_start_branch(monkeypatch, caplog):
    runtime = FakeRuntime()
    app = main.App(runtime)

    def boom() -> None:
        raise RuntimeError("no GL")

    monkeypatch.setattr(app, "setup_window", boom)
    with caplog.at_level(logging.ERROR, logger="warlock.studio.main"):
        assert app.run() == 1
    assert "could not start" in caplog.text
    assert "mid-session" not in caplog.text
    assert runtime.shutdowns == 1


def test_a_failure_after_the_splash_is_still_a_startup_failure(monkeypatch, caplog):
    """``setup_context`` is startup too -- the window is up but the app is not."""
    runtime = FakeRuntime()
    app = main.App(runtime)
    monkeypatch.setattr(app, "setup_window", lambda: None)
    monkeypatch.setattr(app, "_startup_with_splash", lambda: True)

    def boom() -> None:
        raise RuntimeError("no textures")

    monkeypatch.setattr(app, "setup_context", boom)
    with caplog.at_level(logging.ERROR, logger="warlock.studio.main"):
        assert app.run() == 1
    assert "could not start" in caplog.text
    assert "mid-session" not in caplog.text


def test_a_failure_in_the_frame_loop_is_still_reported_as_mid_session(monkeypatch, caplog):
    runtime = FakeRuntime()
    app = main.App(runtime)
    monkeypatch.setattr(app, "setup_window", lambda: None)
    monkeypatch.setattr(app, "_startup_with_splash", lambda: True)
    monkeypatch.setattr(app, "setup_context", lambda: None)
    monkeypatch.setattr(app, "_tick", lambda: 1 / 60)

    def boom(_dt: float) -> None:
        raise RuntimeError("a pane exploded")

    monkeypatch.setattr(app, "frame", boom)
    with caplog.at_level(logging.ERROR, logger="warlock.studio.main"):
        assert app.run() == 1
    assert "mid-session" in caplog.text
    assert "could not start" not in caplog.text
