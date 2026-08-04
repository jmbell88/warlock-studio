"""Work that must not happen on the frame thread, and GL objects that must not
be freed while a draw list still points at them.

Neither of these shows up as a test failure in the ordinary way -- one is a
stutter and the other is an intermittently wrong image -- so they are pinned
against the structures that cause them rather than against a rendered frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warlock.studio import textures

# --- storage measurement -----------------------------------------------------


def test_measuring_storage_is_reachable_without_touching_the_cache(svc):
    """``measure`` is the half that goes to a task thread; it returns the
    reading rather than assigning it, so the frame thread does the assignment
    when the result comes back."""
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(svc)
    before = cache.storage
    result = cache.measure()
    assert result is not None
    assert cache.storage is before  # unchanged: measuring does not publish


def test_a_finished_job_asks_for_storage_off_the_frame_thread(svc):
    """The regression: ``_refresh`` called the blocking walk inline, freezing
    the frame that should have shown the job finishing."""
    from types import SimpleNamespace

    from warlock.studio import main
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(svc)
    measured: list[str] = []
    cache.refresh_storage = lambda: measured.append("blocking walk")  # type: ignore[method-assign]

    class FakeApp:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.app_ctx = SimpleNamespace(cache=cache, toast=lambda *a, **k: None)

        def _request_storage(self) -> None:
            self.calls.append("_request_storage")

        def _sync_viewer(self) -> None:
            pass

        def _check_worker(self) -> None:
            pass

    app = FakeApp()
    # tick() drives the announce callback; a job reaching "done" is the moment
    # the old code did the walk inline.
    cache.tick = lambda announce: bool(  # type: ignore[method-assign]
        announce({"id": "a" * 12, "status": "done"}, "running")
    )

    main.App._refresh(app)

    assert app.calls == ["_request_storage"]
    assert measured == [], "the blocking walk must not run on the frame thread"


def test_requesting_storage_submits_the_non_publishing_measurement(svc):
    from types import SimpleNamespace

    from warlock.studio import main
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(svc)
    submitted: list[tuple[str, object]] = []
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(
            cache=cache,
            submit=lambda key, run: submitted.append((key, run)) or True,
        )
    )

    main.App._request_storage(app)

    # The same key every time, so a burst of jobs finishing coalesces into one
    # walk -- submit refuses a key already in flight.
    assert submitted == [("storage", cache.measure)]


def test_the_blocking_measurement_says_so():
    from warlock.studio.jobs_cache import JobsCache

    assert "Blocking" in (JobsCache.refresh_storage.__doc__ or "")


# --- the thumbnail cache -----------------------------------------------------


class FakeTexture:
    def __init__(self, size) -> None:
        self.size = size
        self.released = False
        self.filter = None
        self.repeat_x = self.repeat_y = True

    def release(self) -> None:
        self.released = True


class FakeGL:
    LINEAR = 1

    def __init__(self) -> None:
        self.made: list[FakeTexture] = []

    def texture(self, size, components, data) -> FakeTexture:
        tex = FakeTexture(size)
        self.made.append(tex)
        return tex


@pytest.fixture
def cache(monkeypatch, tmp_path):
    gl = FakeGL()
    cache = textures.ThumbnailCache(gl, limit=2)
    monkeypatch.setattr(cache, "_load", lambda path: gl.texture((8, 8), 4, b""))
    return cache


def _thumb(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.png"
    path.write_bytes(b"x")
    return path


def test_a_texture_still_on_screen_is_not_evicted_mid_frame(cache, tmp_path):
    """Every card asks for its thumbnail during the UI build; the pixels are
    not read until the backend draws. Evicting one card to make room for
    another frees a texture the draw list already points at."""
    cache.begin_frame()
    kept = [cache.get(f"j{i}", _thumb(tmp_path, f"j{i}")) for i in range(4)]

    assert all(t is not None for t in kept)
    assert not any(t.released for t in kept)
    assert len(cache._entries) == 4  # over the limit, deliberately, for one frame


def test_the_overshoot_drains_once_the_frame_is_over(cache, tmp_path):
    cache.begin_frame()
    for i in range(4):
        cache.get(f"j{i}", _thumb(tmp_path, f"j{i}"))

    cache.begin_frame()  # nothing asked for yet this frame
    cache.get("j9", _thumb(tmp_path, "j9"))
    assert len(cache._entries) <= cache.limit


def test_an_evicted_texture_is_not_freed_until_the_next_frame(cache, tmp_path, monkeypatch):
    # Isolation, not behaviour: another module's session fixture may have left
    # a real renderer registered, and a real forget_texture chokes on the fakes.
    class FakeRenderer:
        def forget_texture(self, texture):
            pass

    monkeypatch.setattr("warlock.studio.imgui_backend.current", lambda: FakeRenderer())
    cache.begin_frame()
    first = cache.get("a", _thumb(tmp_path, "a"))

    cache.begin_frame()
    for i in range(3):
        cache.get(f"b{i}", _thumb(tmp_path, f"b{i}"))
    assert not first.released  # retired, but this frame may still draw it

    cache.begin_frame()
    assert first.released


def test_eviction_forgets_the_backend_registration(cache, tmp_path, monkeypatch):
    """Releasing a registered texture without forgetting it leaves the backend
    holding a dead object under a GL name the driver will reuse."""
    forgotten: list[Any] = []

    class FakeRenderer:
        def forget_texture(self, texture):
            forgotten.append(texture)

    monkeypatch.setattr(
        "warlock.studio.imgui_backend.current", lambda: FakeRenderer()
    )

    cache.begin_frame()
    doomed = cache.get("a", _thumb(tmp_path, "a"))
    cache.begin_frame()
    for i in range(3):
        cache.get(f"b{i}", _thumb(tmp_path, f"b{i}"))
    cache.begin_frame()

    assert doomed in forgotten
    assert doomed.released


def test_releasing_everything_forgets_everything(cache, tmp_path, monkeypatch):
    forgotten: list[Any] = []

    class FakeRenderer:
        def forget_texture(self, texture):
            forgotten.append(texture)

    monkeypatch.setattr(
        "warlock.studio.imgui_backend.current", lambda: FakeRenderer()
    )
    cache.begin_frame()
    made = [cache.get(f"j{i}", _thumb(tmp_path, f"j{i}")) for i in range(3)]

    cache.release()
    assert all(t.released for t in made)
    assert set(forgotten) == set(made)
    assert not cache._entries


def test_release_survives_no_renderer(cache, tmp_path, monkeypatch):
    """Teardown can run after the backend is gone."""
    monkeypatch.setattr("warlock.studio.imgui_backend.current", lambda: None)
    cache.begin_frame()
    tex = cache.get("a", _thumb(tmp_path, "a"))
    cache.release()
    assert tex.released


# --- keyboard shortcuts ------------------------------------------------------


def test_shortcuts_act_on_the_press_and_not_on_the_release():
    """Both edges reach ``_shortcut`` so paint's space-to-pan can see the
    release. Everything below that point is a toggle or an action, so acting on
    the release as well cancels the toggle and double-submits the action."""
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._shortcut)
    body = source.split("inker_mode.handle_key", 1)[1]
    guard = body.split("K_RETURN", 1)[0]
    assert "event.type != pygame.KEYDOWN" in guard


def test_paint_still_sees_both_edges():
    """The guard has to sit *after* the paint delegation, or space-to-pan
    latches on and never releases."""
    import inspect

    from warlock.studio import inker_mode, main

    source = inspect.getsource(main.App._shortcut)
    assert source.index("inker_mode.handle_key") < source.index(
        "event.type != pygame.KEYDOWN"
    )
    assert "K_SPACE" in inspect.getsource(inker_mode.handle_key)


def test_inker_mode_never_leaks_a_key_to_the_viewport(monkeypatch):
    """Inker returns whether or not ``handle_key`` consumed the key.

    It returns False when no document is open, and the fall-through meant
    F/W/S framed the model and toggled wireframe and turntable while
    Ctrl+Enter submitted a mesh job -- all against a viewport Inker has
    replaced. Reachable the moment you enter Inker from Home with nothing
    loaded.
    """
    from types import SimpleNamespace

    import pygame

    from warlock.studio import main
    from warlock.studio.panes import settings_2d, settings_3d
    from warlock.studio.state import AppState

    submitted: list[str] = []
    monkeypatch.setattr(settings_2d, "generate", lambda *a, **k: submitted.append("2d"))
    monkeypatch.setattr(settings_3d, "promote", lambda *a, **k: submitted.append("3d"))
    monkeypatch.setattr(pygame.key, "get_mods", lambda: pygame.KMOD_CTRL)

    viewer_calls: list[str] = []
    viewer = SimpleNamespace(
        frame=lambda: viewer_calls.append("frame"),
        set_wireframe=lambda v: viewer_calls.append("wireframe"),
        set_turntable=lambda v: viewer_calls.append("turntable"),
        pose_mode=False,
        exit_compare=lambda: viewer_calls.append("exit_compare"),
    )
    state = AppState()
    state.mode = "inker"
    state.inker = None  # entered Inker with nothing open
    app = SimpleNamespace(
        app_ctx=SimpleNamespace(state=state, cache=SimpleNamespace(get=lambda _id: None)),
        viewer=viewer,
    )

    for key in (pygame.K_f, pygame.K_w, pygame.K_s, pygame.K_RETURN):
        main.App._shortcut(app, pygame.event.Event(pygame.KEYDOWN, key=key))

    assert viewer_calls == []
    assert submitted == []
    assert state.wireframe is False
    assert state.turntable is False


# --- a dead GPU worker -------------------------------------------------------


def test_a_dead_worker_is_reported_to_the_user():
    """``worker.fatal`` was surfaced only through /api/health, which the
    browser build polled. With no HTTP layer a mid-session crash became
    invisible outside the log file, and every job queued after it just sat."""
    import inspect

    from warlock.studio import main

    source = inspect.getsource(main.App._check_worker)
    assert "runtime.fatal" in source
    assert "runtime.alive" in source
    assert "last_error" in source
    assert "_fatal_reported" in source
    # And something actually calls it every frame.
    assert "_check_worker" in inspect.getsource(main.App._refresh)


def test_the_worker_traceback_is_logged_before_it_is_stripped():
    """``with_traceback`` mutates in place and returns self, so stripping first
    left ``exc_info=exc`` with no stack to format -- the one log line for a
    dead GPU worker carried nothing."""
    import inspect

    from warlock import queue

    source = inspect.getsource(queue.Worker._on_task_done)
    assert source.index("log.critical") < source.index("with_traceback(None)")
