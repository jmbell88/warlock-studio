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


def _fake_app(svc, cache, *, accept_submits: bool = True):
    """An App stand-in with just enough of ``app_ctx`` for ``_refresh``."""
    from types import SimpleNamespace

    from warlock.studio.state import AppState

    submitted: list[str] = []

    def submit(key: str, _run: Any, *_args: Any) -> bool:
        submitted.append(key)
        return accept_submits

    class FakeApp:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.submitted = submitted
            self.app_ctx = SimpleNamespace(
                svc=svc,
                cache=cache,
                state=AppState(),
                submit=submit,
                toast=lambda *a, **k: None,
            )

        def _request_storage(self) -> None:
            self.calls.append("_request_storage")

        def _sync_viewer(self) -> None:
            pass

        def _check_worker(self) -> None:
            pass

    return FakeApp()


def _tick_with(cache, job: dict[str, Any], previous: str = "running") -> None:
    """Drive ``tick``'s announce callback with one transition."""
    cache.tick = lambda announce: bool(announce(job, previous))  # type: ignore[method-assign]


def test_a_finished_job_asks_for_storage_off_the_frame_thread(svc):
    """The regression: ``_refresh`` called the blocking walk inline, freezing
    the frame that should have shown the job finishing."""
    from warlock.studio import main
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(svc)
    measured: list[str] = []
    cache.refresh_storage = lambda: measured.append("blocking walk")  # type: ignore[method-assign]

    app = _fake_app(svc, cache)
    # tick() drives the announce callback; a job reaching "done" is the moment
    # the old code did the walk inline.
    _tick_with(cache, {"id": "a" * 12, "status": "done"})

    main.App._refresh(app)

    assert app.calls == ["_request_storage"]
    assert measured == [], "the blocking walk must not run on the frame thread"


# --- the findings recompute --------------------------------------------------
#
# The worker appends an observation for every finished model job, but it runs on
# the asyncio thread and cannot submit a task. The frame loop noticing the job
# finish is the only place that evidence can enter findings.json, so these pin
# the seam rather than the aggregation (which test_findings_comparisons covers).


def test_a_finished_mesh_recomputes_the_findings_without_any_verdict(svc):
    """The regression: observations were recorded on every generation and
    reached the file only when somebody next filed a verdict."""
    from warlock.studio import main, review_mode
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(svc)
    app = _fake_app(svc, cache)
    _tick_with(cache, {"id": "a" * 12, "status": "done", "stage": "model", "kind": "text"})

    main.App._refresh(app)

    assert review_mode.FINDINGS_KEY in app.submitted
    assert app.app_ctx.state.findings_dirty is False  # accepted, so cleared


def test_a_finished_reference_recomputes_nothing(svc):
    """``_observe_finished`` writes no row for a reference, so there is nothing
    new to aggregate -- the condition is mirrored rather than approximated."""
    from warlock.studio import main, review_mode
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(svc)
    app = _fake_app(svc, cache)
    _tick_with(cache, {"id": "a" * 12, "status": "done", "stage": "reference", "kind": "text"})

    main.App._refresh(app)

    assert review_mode.FINDINGS_KEY not in app.submitted


def test_a_refused_recompute_is_retried_on_the_next_frame(svc):
    """The re-arm. ``submit`` refuses a key already in flight and nothing used
    to reschedule, so the last unit of a sweep -- with no verdict after it to
    pick anything up -- left the file behind for good."""
    from warlock.studio import main, review_mode
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(svc)
    app = _fake_app(svc, cache, accept_submits=False)
    _tick_with(cache, {"id": "a" * 12, "status": "done", "stage": "model", "kind": "text"})

    main.App._refresh(app)
    assert app.app_ctx.state.findings_dirty is True, "a refused submit must stay pending"

    # Nothing finishes on this frame; the pending request is what drives it.
    cache.tick = lambda announce: False  # type: ignore[method-assign]
    main.App._refresh(app)

    assert app.submitted.count(review_mode.FINDINGS_KEY) == 2


def test_requesting_a_recompute_does_no_work_on_the_frame_thread(svc):
    """``refresh_findings`` is called from the verdict path *and* from the
    frame loop's announce callback, so it must be a flag rather than a read of
    every verdict in the store."""
    from types import SimpleNamespace

    from warlock.studio import review_mode
    from warlock.studio.state import AppState

    ctx = SimpleNamespace(state=AppState())
    review_mode.refresh_findings(ctx)  # no svc, no submit: it must need neither

    assert ctx.state.findings_dirty is True


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


def test_the_job_list_does_not_re_stat_every_artifact_on_every_tick(svc, monkeypatch):
    """The regression: ``LISTED`` is nine names and ``ready`` stats one or two
    files for each, so a two-hundred row page cost upwards of two thousand
    ``stat`` calls -- twice a second, on the frame thread, growing without limit
    as "load more" widened the window."""
    from pathlib import Path

    from warlock.service import jobs as svc_jobs
    from warlock.studio.jobs_cache import JobsCache

    for i in range(5):
        svc_jobs.create_job(svc, kind="text", prompt=str(i))

    calls: list[Path] = []
    real = Path.exists

    def counting_exists(self):
        calls.append(self)
        return real(self)

    monkeypatch.setattr(Path, "exists", counting_exists)

    cache = JobsCache(svc)
    assert cache.tick() is True
    cold = len(calls)
    assert cold >= 5, "the first pass really does look at the disk"

    calls.clear()
    cache.invalidate()
    assert cache.tick() is True

    assert calls == [], "an unchanged page must answer from the cache"
    # And it still answers correctly.
    assert all("files" in job for job in cache.jobs)


def test_a_new_artifact_on_disk_is_noticed_without_a_status_change(svc):
    """The stamp is (status, the job directory's mtime), and the second half is
    load-bearing: a rig lands in the *source* job's directory while that job
    stays ``done``, so a status-only key would hide it forever."""
    from warlock.service import jobs as svc_jobs
    from warlock.studio.jobs_cache import JobsCache

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    cache = JobsCache(svc)
    cache.tick()
    assert "thumb.png" not in cache.by_id[job_id]["files"]

    (job_dir / "thumb.png").write_bytes(b"x")
    cache.invalidate()
    cache.tick()

    assert "thumb.png" in cache.by_id[job_id]["files"]


# --- the racily-clean guard -------------------------------------------------
#
# A directory's mtime has a resolution, and on Windows it is the system clock's
# 15.6 ms tick: measured, adding a file left the mtime unchanged 155 times in
# 200 (docs/measurements/2026-08-07-directory-mtime-granularity.md). The test
# above used to fail about one run in ten for exactly that reason, and the
# product bug behind it was worse than a flake -- a stamp that matches a stale
# answer matches it *forever*. ``os.utime`` is what makes the timing a decision
# here rather than a coin toss.


def _dir_job(svc):
    from warlock.service import jobs as svc_jobs

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    return {"id": job_id, "status": "done"}, job_dir


def test_a_directory_touched_moments_ago_is_answered_but_not_remembered(svc):
    import os
    import time

    from warlock.service import files as svc_files

    job, job_dir = _dir_job(svc)
    cache: dict = {}

    now = time.time_ns()
    os.utime(job_dir, ns=(now, now))
    svc_files.attach_files(job, job_dir, cache=cache)
    assert job["files"] == [], "the answer itself is still correct"
    assert cache == {}, "a directory this fresh may still be written in its own mtime tick"

    settled = now - 10 * svc_files.MTIME_RACE_NS
    os.utime(job_dir, ns=(settled, settled))
    svc_files.attach_files(job, job_dir, cache=cache)
    assert job["id"] in cache, "an mtime safely in the past is a sound key"


def test_a_write_that_never_moved_the_mtime_is_still_noticed(svc):
    """The real failure, made deterministic. Both calls see the same mtime --
    which is what Windows does three times in four -- so the only thing that
    can save the second one is the first having declined to cache."""
    import os
    import time

    from warlock.service import files as svc_files

    job, job_dir = _dir_job(svc)
    cache: dict = {}
    now = time.time_ns()

    os.utime(job_dir, ns=(now, now))
    svc_files.attach_files(job, job_dir, cache=cache)
    assert "thumb.png" not in job["files"]

    (job_dir / "thumb.png").write_bytes(b"x")
    os.utime(job_dir, ns=(now, now))
    svc_files.attach_files(job, job_dir, cache=cache)
    assert "thumb.png" in job["files"]


def test_the_files_cache_never_outgrows_the_page(svc):
    from warlock.service import jobs as svc_jobs
    from warlock.studio.jobs_cache import JobsCache

    ids = [svc_jobs.create_job(svc, kind="text", prompt=str(i))["id"] for i in range(4)]
    cache = JobsCache(svc)
    cache.tick()
    assert len(cache._files) == 4

    for job_id in ids[:2]:
        svc_jobs.delete_job(svc, job_id)
    cache.invalidate()
    cache.tick()

    assert set(cache._files) == set(ids[2:])


# --- loading the selected mesh ------------------------------------------------


def _viewer_app(svc, *, mode="3d", job=None, accept=True):
    """An App stand-in with just enough for ``_sync_viewer``/``_adopt_model``."""
    from types import SimpleNamespace

    from warlock.studio.state import AppState

    submitted: list[tuple[str, Any]] = []

    class FakeViewer:
        def __init__(self) -> None:
            self.path = None
            self.pending = None
            self.pose_mode = False
            self.adopted: list[Any] = []
            self.parsed: list[Any] = []

        def parse_model(self, path):
            self.parsed.append(path)
            return f"model:{path.name}"

        def adopt_model(self, model, path):
            self.adopted.append((model, path))
            self.path = path

    state = AppState()
    state.mode = mode

    class FakeApp:
        def __init__(self) -> None:
            self.viewer = FakeViewer()
            self.submitted = submitted
            self.app_ctx = SimpleNamespace(
                state=state,
                job=lambda: job,
                job_dir=lambda job_id: svc.job_dir(job_id),
                submit=lambda key, fn, *a, tag=None: (
                    submitted.append((key, tag)) or accept
                ),
                capture_thumbnail=lambda job_id: None,
                toast=lambda *a, **k: None,
            )

        def _refresh_rig_side_data(self) -> None:
            pass

    return FakeApp()


def test_the_selected_mesh_is_parsed_off_the_frame_thread(svc):
    """The regression: ``_sync_viewer`` ran the whole load -- a full accessor
    decode plus a PNG per texture slot -- inline, on the frame a job
    transitioned to done, which is when the file is largest and coldest."""
    from warlock.studio import main

    job = {"id": "a" * 12, "files": ["model.glb", "thumb.png"]}
    app = _viewer_app(svc, job=job)

    main.App._sync_viewer(app)

    assert app.viewer.parsed == [], "no parse may happen on the frame thread"
    assert [key for key, _ in app.submitted] == [main.VIEWER_KEY]
    assert app.viewer.pending == svc.job_dir(job["id"]) / "model.glb"


def test_a_load_in_flight_is_not_dispatched_again_every_tick(svc):
    from warlock.studio import main

    job = {"id": "a" * 12, "files": ["model.glb"]}
    app = _viewer_app(svc, job=job)

    main.App._sync_viewer(app)
    main.App._sync_viewer(app)
    main.App._sync_viewer(app)

    assert len(app.submitted) == 1


def test_a_parsed_model_is_adopted_on_the_frame_thread(svc):
    from types import SimpleNamespace

    from warlock.studio import main

    job = {"id": "a" * 12, "files": ["model.glb", "thumb.png"]}
    app = _viewer_app(svc, job=job)
    main.App._sync_viewer(app)
    wanted = app.viewer.pending

    main.App._adopt_model(app, SimpleNamespace(tag=wanted, result="parsed"))

    assert app.viewer.adopted == [("parsed", wanted)]
    assert app.viewer.pending is None


def test_a_result_the_selection_moved_past_is_dropped(svc):
    """The selection can move while a parse is in flight, and adopting a result
    nobody is waiting for would put the previous asset back on screen."""
    from types import SimpleNamespace

    from warlock.studio import main

    job = {"id": "a" * 12, "files": ["model.glb"]}
    app = _viewer_app(svc, job=job)
    main.App._sync_viewer(app)
    stale = app.viewer.pending
    app.viewer.pending = svc.job_dir("b" * 12) / "model.glb"  # the user clicked away

    main.App._adopt_model(app, SimpleNamespace(tag=stale, result="parsed"))

    assert app.viewer.adopted == []


def test_a_refused_dispatch_leaves_nothing_pending(svc):
    """One key, so a selection moving faster than the disk cannot pile up
    loads -- but a refused submit must not leave the viewer believing a load is
    coming, or it would never retry."""
    from warlock.studio import main

    job = {"id": "a" * 12, "files": ["model.glb"]}
    app = _viewer_app(svc, job=job, accept=False)

    main.App._sync_viewer(app)

    assert app.viewer.pending is None


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
    monkeypatch.setattr(cache, "_load", lambda path, nearest=False: gl.texture((8, 8), 4, b""))
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
    assert "note_error" in source
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
