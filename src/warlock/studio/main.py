"""The window, the frame loop, and everything wired together.

One pygame window with a GL 3.3 core context, one moderngl context over it, and
imgui drawing through that same context. The viewport is a texture the panels
show with ``imgui.image`` -- not a separate surface -- which is what makes
"panels over 3D" a layout question rather than a compositing one.

The frame is always the same six steps: collect finished tasks, refresh the job
cache, pump events, build the UI, render the viewport, present. Nothing in
those six may block.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import memlog
from . import fps as fps_mod

log = logging.getLogger(__name__)

WINDOW_TITLE = "Warlock Studio"
# How often the frame loop samples host memory. Long enough to be free, short
# enough that a 30-minute idle session yields 60 points to fit a slope through.
MEMORY_TICK_SECONDS = 30.0
# The task key the selection's GLB is parsed under. One key, so a selection
# moving faster than the disk cannot pile up loads: a refused submit is simply
# retried on the next tick, and a landed result is checked against
# ``viewer.pending`` before it is adopted.
VIEWER_KEY = "viewer-load"
DEFAULT_SIZE = (1600, 950)
MIN_SIZE = (1100, 700)
# Pane widths and the sidebar split now live in layout.Layout, persisted and
# draggable; the defaults there are the 340 / 0.55 this file used to hard-code.
TARGET_FPS = 60
# The modes that fill the host window with one pane. Inker and Clay are not
# here: each fills it with a three-column *workspace* instead, which is
# ``modes.WORKSPACE_MODES``. Those three categories partition ``modes.KEYS``
# exactly, and the partition is the guard on ``_build_ui``'s dispatch.
_SINGLE_PANE_MODES = ("home", "manual", "settings")


def _min_window_size(monitor_scale: float) -> tuple[int, int]:
    """The resize floor, in physical pixels.

    The *monitor's* scale and nothing else. ``tokens.SCALE`` also carries the
    user's UI-scale preference, and a zoom says nothing about how many pixels
    the screen has -- multiplying it in made a 2x preference demand a window
    larger than a 1080p display and refuse to be shrunk.
    """
    return (int(MIN_SIZE[0] * monitor_scale), int(MIN_SIZE[1] * monitor_scale))


def _ui_scale(settings: Any) -> float:
    """The stored multiplier, clamped. A junk value must not brick the window."""
    from . import tokens

    lo, hi = tokens.UI_SCALE_RANGE
    try:
        value = float(settings.get("ui_scale") or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return min(max(value, lo), hi)


def _right_column(
    ctx: Any,
    lay: Any,
    sidebar_w: float,
    *,
    inspector_draw: Callable[[Any], None],
    library_draw: Callable[[Any], None],
) -> None:
    """The right sidebar: inspector on top, library on bottom.

    Split by ``lay.settings_share`` -- the same split the left sidebar used to
    make between settings and library, before the library moved to share this
    column with the inspector instead. Pulled out of ``App._build_ui`` as a
    module-level function (no ``self`` needed) so a test can call the exact
    geometry the frame draws rather than a hand-copied reimplementation of it.
    """
    from imgui_bundle import imgui

    from . import layout as layout_mod
    from . import tokens

    imgui.begin_group()
    avail_y = imgui.get_content_region_avail().y
    inspector_height = avail_y * lay.settings_share
    if layout_mod.pane_child("inspector", (0, inspector_height)):
        inspector_draw(ctx)
    imgui.end_child()
    drag = layout_mod.splitter("sidebar-share", vertical=False, length=sidebar_w)
    if drag and avail_y > 0:
        lay.settings_share = min(
            max(lay.settings_share + drag * tokens.SCALE / avail_y, layout_mod.SHARE_MIN),
            layout_mod.SHARE_MAX,
        )
        lay.save()
    if layout_mod.pane_child("library", (0, 0)):
        library_draw(ctx)
    imgui.end_child()
    imgui.end_group()


class App:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.svc = None
        self.ctx = None
        self.window = None
        self.imgui_renderer = None
        self.viewer = None
        self.app_ctx = None
        self.eta = None
        self._running = False
        self._min_size = MIN_SIZE
        self._last_frame = time.perf_counter()
        self._started_at = self._last_frame
        # Seeded to 0.0, not to now: the first tick fires immediately and puts
        # a startup baseline in the log to measure every later sample against.
        self._last_memory_log = 0.0
        self._last_health_poll = 0.0
        # A dead worker is reported once. The banner is dismissible, and
        # re-raising it every frame would make it impossible to dismiss.
        self._fatal_reported = False
        # The mode the last frame was built in, so a change into a viewport
        # mode can resync the viewer -- a mode change is not something the
        # job cache announces.
        self._last_mode: str | None = None
        # Measured every frame, drawn only when state.show_fps is on (F10), and
        # logged once at teardown regardless -- the overlay answers "is it
        # smooth now", the log line is the evidence for "it ran at 60".
        self.fps = fps_mod.FpsMeter()
        # Set by _draw_viewport_image, read one frame later by _events. The
        # host window is fullscreen, so io.want_capture_mouse is always true
        # and cannot be the gate; imgui's own hover test on the viewport image
        # is, and it correctly goes false under popups and active widgets.
        self._viewport_hovered = False
        # Clay's own viewport, built on first use for the reason its
        # state is: a session that never opens Clay should not pay for a
        # renderer, a framebuffer and three gizmos.
        self.clay_view = None
        # Clay's own hover flag, set by the pane that draws its image, for the
        # reason _viewport_hovered exists: the host window is fullscreen, so
        # io.want_capture_mouse is always true and cannot be the gate.
        self._build_hovered = False

    # -- setup -------------------------------------------------------------

    def setup(self) -> None:
        import moderngl
        import pygame
        from imgui_bundle import imgui

        from . import dpi, fonts, imgui_backend, textures, theme, tokens, widgets
        from .app_ctx import Ctx
        from .jobs_cache import JobsCache
        from .layout import Layout
        from .settings import Settings, restore_form
        from .state import DEFAULT_FORM_3D, AppState, Eta, Filters, default_form_2d
        from .viewer_embed import Viewer

        self.svc = self.runtime.start()
        settings = Settings.load(self.runtime.config.data_dir)

        # Before the window exists: awareness is frozen at window creation.
        dpi.make_process_dpi_aware()

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "icon.ico"
        if icon_path.is_file():
            pygame.display.set_icon(pygame.image.load(str(icon_path)))
        # The window is in physical pixels (Per-Monitor-V2): a persisted size
        # is already physical, and the first-run default scales by the primary
        # monitor so 1600x950 means the same amount of screen everywhere.
        first_run_scale = dpi.system_scale()
        size = tuple(
            settings.get("window_size")
            or (int(DEFAULT_SIZE[0] * first_run_scale), int(DEFAULT_SIZE[1] * first_run_scale))
        )
        self.window = pygame.display.set_mode(
            size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption(WINDOW_TITLE)
        # Dropped files are how a reference image gets in without a dialog.
        pygame.event.set_allowed(None)

        # The scale everything is drawn at, before any font or style is built:
        # the monitor's own scale, and the user's multiplier on top of it. The
        # multiplier is folded in *here* rather than applied later so the font
        # atlas is baked at the size it will be drawn at; changing it in the
        # settings pane rescales everything immediately but only sharpens the
        # glyphs after a restart.
        monitor_scale = dpi.window_scale(pygame)
        tokens.set_scale(monitor_scale * _ui_scale(settings))
        self._min_size = _min_window_size(monitor_scale)

        self.ctx = moderngl.create_context()
        imgui.create_context()
        imgui.get_io().set_ini_filename("")  # imgui's own layout file is not ours to keep
        fonts.load(imgui)
        theme.apply(imgui)
        self.layout = Layout(settings)
        widgets.attach_settings(settings)
        self.imgui_renderer = imgui_backend.ImguiRenderer(self.ctx)
        self.viewer = Viewer(self.ctx)

        state = AppState()
        # No mode restore, and nothing writes one either: the app opens on Home
        # every launch (AppState's default), so a stored mode would be a key
        # with no reader that four call sites kept half-updated.
        state.show_fps = bool(settings.get("show_fps"))
        state.form_2d = restore_form(default_form_2d(), settings.get("form_2d"))
        state.form_3d = restore_form(DEFAULT_FORM_3D, settings.get("form_3d"))
        state.history = list(settings.get("history") or [])
        stored_filters = settings.get("filters") or {}
        state.filters = Filters(
            **{k: v for k, v in stored_filters.items() if k in Filters.__annotations__}
        )

        self.app_ctx = Ctx(
            svc=self.svc,
            runtime=self.runtime,
            state=state,
            cache=JobsCache(self.svc),
            tasks=self.runtime.tasks,
            settings=settings,
            viewer=self.viewer,
            textures=textures.ThumbnailCache(self.ctx),
        )
        self.app_ctx.dpi_scale = monitor_scale
        self.app_ctx.layout = self.layout
        self.app_ctx.load_presets = self.load_presets
        self.app_ctx.refresh_rig_data = self._refresh_rig_side_data
        self.eta = Eta()
        self._load_static_answers()
        self.app_ctx.cache.refresh_storage()
        self.viewer.on_pose_dirty = lambda _dirty: None

    def _load_static_answers(self) -> None:
        """Read the things that cannot change without a restart, once."""
        from .. import models
        from ..service import rig as svc_rig
        from ..service import sheets as svc_sheets
        from ..service import system as svc_system

        ctx = self.app_ctx
        # Clay's bridge asks the ctx for this rather than importing App: the
        # render it needs is an offscreen GL draw on the frame thread, which is
        # the App's business and not a pane's. Attached here so the button has a
        # handler from the first frame rather than toasting "not wired up yet".
        ctx.clay_send_to_3d = self._clay_send_to_3d
        ctx.guidance = svc_system.guidance_catalog(self.svc)
        ctx.sheet_options = svc_sheets.sheet_options()
        # Marked rather than hidden when weights are absent: the combo listing
        # every registered model regardless meant picking one whose weights
        # were never downloaded and learning at job-failure time, despite
        # doctor knowing at startup.
        missing = {
            check.name.removeprefix("image model: ")
            for check in (self.runtime.checks or [])
            if check.name.startswith("image model: ") and not check.ok
        }
        ctx.base_models = [
            (k, f"{spec.label} - weights missing" if spec.label in missing else spec.label)
            for k, spec in models.BASE_MODELS.items()
        ]
        ctx.style_loras = [("", "no style LoRA")] + [
            (k, spec.label) for k, spec in models.STYLE_LORAS.items()
        ]
        try:
            templates = svc_rig.rig_templates(self.svc)
        except Exception:
            log.exception("could not probe rigging")
            templates = {"available": False, "templates": [], "default": ""}
        ctx.rigging_available = bool(templates.get("available"))
        ctx.rig_templates = list(templates.get("templates") or [])
        ctx.rig_default = templates.get("default") or ""
        ctx.export_dir = str(self.runtime.config.export_dir or "") or None
        # The trellis port check is non-fatal -- the app is perfectly usable
        # without ever running trellis -- but a port already held at startup
        # means an orphaned server from a previous crash, and every 3D job will
        # fail (or, worse, be served by the orphan) until it is stopped. That
        # is worth the same banner a fatal check gets, so it joins them here
        # rather than being promoted to fatal in doctor.
        failed = [
            c
            for c in self.runtime.checks
            if not c.ok and (c.fatal or c.name == "trellis port")
        ]
        for check in failed:
            ctx.state.note_error(f"{check.name}: {check.detail}")

    # -- the loop ----------------------------------------------------------

    def run(self) -> int:
        import pygame

        # setup() is inside the try: it starts the runtime before it touches
        # pygame or GL, so a failure past that point used to skip teardown and
        # leave the store, the loop thread and the worker running.
        #
        # The two phases are reported differently on purpose. A window that
        # never appeared and a window that vanished after twenty minutes are
        # different bugs, and the log line was the only thing that could tell
        # them apart -- when there was one at all.
        in_setup = True
        rc = 0
        try:
            self.setup()
            in_setup = False
            self._running = True
            clock = pygame.time.Clock()
            while self._running:
                dt = self._tick()
                self.frame(dt)
                pygame.display.flip()
                clock.tick(TARGET_FPS)
        except Exception:
            rc = 1
            if in_setup:
                log.exception("Warlock Studio could not start")
            else:
                log.exception(
                    "the frame loop crashed mid-session after %d frames (%.1f s up)",
                    self.fps.frames, time.perf_counter() - self._started_at,
                )
        finally:
            self.teardown()
        return rc

    def _tick(self) -> float:
        now = time.perf_counter()
        dt = min(now - self._last_frame, 0.25)
        self._last_frame = now
        self.fps.record(dt)
        self._memory_ticker(now)
        self._health_ticker(now)
        return dt

    def _memory_ticker(self, now: float) -> None:
        """Log host memory every MEMORY_TICK_SECONDS.

        The single line that discriminates the two candidate causes of the
        2026-08-03 commit exhaustion. Stage-boundary logging (queue._log_mem)
        only fires when a job runs, so it cannot distinguish "each job leaks a
        little" from "the process grows while sitting idle". This samples
        regardless, so the shape of the curve is in the log either way.

        Cheap enough for the frame loop: two ctypes calls once per 30 s.
        """
        if now - self._last_memory_log < MEMORY_TICK_SECONDS:
            return
        self._last_memory_log = now
        summary = memlog.summary()
        if summary is not None:
            # The frame rate rides along on the same line: a session that dies
            # without unwinding leaves no teardown summary, and memory and
            # smoothness are the two things worth reading against each other.
            rate = f" | {self.fps.fps:.1f} fps" if self.fps.frames else ""
            log.info("host idle-tick: %s%s", summary, rate)

    def _health_ticker(self, now: float) -> None:
        """Keep the header health dot honest after startup.

        The dot reads ``runtime.checks``; before this poller it showed the
        startup snapshot forever -- unplug the disk or orphan the trellis port
        mid-session and the dot stayed green. The probe runs on a task thread
        (it binds a socket and stats a disk), and the submit is paced here so
        a task is not queued sixty times a second; ``cached_checks``' own TTL
        makes a stray extra call cheap rather than harmful.
        """
        from warlock.service import system as svc_system

        if now - self._last_health_poll < svc_system.HEALTH_TTL:
            return
        self._last_health_poll = now
        self.app_ctx.submit("health", svc_system.current_checks, self.app_ctx.svc)

    def frame(self, dt: float) -> None:
        from imgui_bundle import imgui

        self.app_ctx.textures.begin_frame()
        self._collect_tasks()
        self._refresh()
        self._events()

        import pygame

        io = imgui.get_io()
        io.delta_time = max(dt, 1e-4)
        # Set every frame rather than only on resize: a window that starts
        # minimised, or a display scale change, reaches imgui no other way, and
        # a zero display size is an assertion rather than a blank frame.
        io.display_size = pygame.display.get_window_size()
        io.display_framebuffer_scale = (1.0, 1.0)
        imgui.new_frame()
        self._build_ui()
        imgui.render()

        self.ctx.screen.use()
        self.ctx.clear(*_background())
        self.imgui_renderer.render(imgui.get_draw_data())
        self.app_ctx.settings.tick()

    # -- frame steps -------------------------------------------------------

    def _collect_tasks(self) -> None:
        ctx = self.app_ctx
        for done in ctx.tasks.poll():
            if not done.ok:
                ctx.toast(done.message or "That did not work.", "error", done.action)
                # A failed save must not leave the document locked: saving
                # disables every editing control, so without this one bad
                # write makes the tab read-only until it is closed. Each
                # editor claims its own key prefix.
                if done.key.startswith("inker-"):
                    from . import inker_mode

                    inker_mode.on_task_failed(ctx, done)
                elif done.key.startswith("clay-"):
                    from . import clay_mode

                    clay_mode.on_task_failed(ctx, done)
                elif done.key.startswith("review-"):
                    from . import review_mode

                    # Same rule: ``scanning`` gates every button and key, so a
                    # failed scan that left it set would make the mode inert.
                    review_mode.on_task_failed(ctx, done)
                continue
            self._on_task_done(done)

    def _on_task_done(self, done: Any) -> None:
        ctx = self.app_ctx
        key = done.key
        if key == "preview" and isinstance(done.result, dict):
            ctx.state.preview.update(done.result)
            return
        if key == "health":
            # The dot and the diagnostics popup read runtime.checks each
            # frame; replacing the list wholesale is atomic enough for both.
            if isinstance(done.result, list):
                self.runtime.checks = done.result
            return
        # The side data the pose and sheet panels read. Keyed by job so a
        # result that arrives after the selection moved on can be dropped
        # rather than shown against the wrong asset.
        if key.startswith(("poses:", "sheets:", "presets:")):
            name, _, job_id = key.partition(":")
            if name != "presets" and job_id != ctx.state.selected:
                return
            if name == "poses" and isinstance(done.result, dict):
                ctx.state.preview["poses"] = done.result.get("poses") or []
                ctx.state.preview["bones"] = done.result.get("bones") or []
            elif name == "sheets" and isinstance(done.result, dict):
                ctx.state.preview["sheets"] = done.result.get("sheets") or []
            elif name == "presets" and isinstance(done.result, dict):
                ctx.state.preview["presets"] = done.result.get("poses") or []
            return
        if key == "upload" and done.result is not None:
            from .panes import settings_3d

            settings_3d.upload(ctx, Path(done.result))
            return
        if key == "anchor-pick":
            # The picker and the file read happened on the task thread; the
            # settings write and the toast are frame-thread work.
            from .panes import profiles_panel

            profiles_panel.adopt_anchor(ctx, done.result)
            return
        if key == "ref-upload" and done.result is not None:
            # Only the path is kept here; the bytes are read in the submit
            # task, so picking a 20 MB image never touches the frame thread.
            ctx.state.form_2d["ref_path"] = str(done.result)
            return
        if key.startswith("clay-"):
            from . import clay_mode

            clay_mode.on_task_done(ctx, done)
            if isinstance(done.result, dict) and done.result.get("exported"):
                # The card appears in the library like any other asset, so it
                # needs the thumbnail every other asset gets -- and that is an
                # offscreen GL draw, which belongs on the frame thread rather
                # than in the task that minted the row.
                self._capture_clay_thumbnail(done.result["job_id"])
            return
        if key.startswith("inker-"):
            from . import inker_mode

            inker_mode.on_task_done(ctx, done)
            return
        if key.startswith("review-"):
            from . import review_mode

            review_mode.on_task_done(ctx, done)
            return
        if key == "submit":
            ctx.cache.invalidate()
            # Say where in line it landed: five rapid submits used to produce
            # five identical "Queued." toasts and no sense of depth.
            waiting = sum(1 for j in ctx.cache.jobs if j.get("status") == "queued")
            ctx.toast("Queued." if waiting <= 1 else f"Queued - {waiting} jobs in line.")
            return
        if key.startswith("save:") or key.startswith("bake:") or key.startswith("sheet-save:"):
            if done.result is not None:
                ctx.toast(f"Saved to {done.result}")
            return
        if key == "trellis-log":
            # The one diagnostic for "the 3D engine stopped unexpectedly". The
            # button submitted this and nothing ever stored the answer, so the
            # box under it stayed empty forever.
            if isinstance(done.result, dict):
                ctx.state.preview["trellis_log"] = done.result.get("text") or ""
            return
        if key.startswith("export-"):
            # A bulk export finishing with no visible outcome reads as a
            # failure; single-artifact saves have always toasted.
            if done.result is not None:
                ctx.toast(f"Exported to {done.result}")
            return
        if key == "storage":
            if done.result is not None:
                ctx.cache.storage = done.result
            return
        if key.startswith(("delete:", "prune", "rename:", "name:", "tags:", "fav:")):
            ctx.cache.invalidate()
            if key.startswith(("delete:", "prune")):
                self._request_storage()
            return
        if key.startswith("retarget:"):
            # model.glb was rewritten under the viewer, and the params it is
            # described by changed with it: drop the mesh verdicts on screen and
            # reload what is now on disk.
            ctx.cache.invalidate()
            self._reload_viewer()
            stale = (done.result or {}).get("stale") or []
            ctx.toast(
                f"Mesh rebuilt. {len(stale)} rig artifact(s) now describe the old mesh."
                if stale
                else "Mesh rebuilt."
            )
            return
        if key.startswith("sheet-del:"):
            # Not covered by the "sheet:" prefix below, and _sync_viewer's
            # early-return means nothing else refetches the list: a deleted
            # sheet stayed on screen with live-looking buttons.
            self._refresh_rig_side_data()
            return
        if key.startswith(("cancel:", "rerun:", "remesh:", "retry:", "rig:", "joints:", "sheet:")):
            ctx.cache.invalidate()
            if key.startswith("sheet:"):
                # A rendered sheet is side data, not a job-row change, so the
                # cache invalidation above does not bring it back.
                self._refresh_rig_side_data()
            return
        if key == VIEWER_KEY:
            self._adopt_model(done)
            return
        if key.startswith("pose-"):
            if key.startswith("pose-save:") and self.viewer.pose_mode:
                # Only now is the pose actually on disk. A failed save leaves
                # the flag set, so the guard still stops the user walking away
                # from work that was never written.
                self.viewer.editor.dirty = False
            self._refresh_rig_side_data()
            ctx.cache.invalidate()

    def _refresh(self) -> None:
        from . import review_mode
        from .jobs_cache import transition_message

        ctx = self.app_ctx

        def announce(job: Any, previous: str | None) -> None:
            message = transition_message(job, previous)
            if message is not None:
                ctx.toast(*message)
            if job["status"] == "done":
                self._request_storage()
                # The worker has just appended an observation for this job
                # (queue._observe_finished, same condition), and it has no way
                # to ask for the recompute itself -- it runs on the asyncio
                # thread and knows nothing about tasks or panes. This is the
                # only place a finished generation is noticed, so it is where
                # the machine half of the findings corpus enters the file:
                # without it, evidence recorded on every run would reach
                # findings.json only when somebody next filed a verdict.
                if job.get("stage") == "model" and job.get("kind") in ("text", "image"):
                    review_mode.refresh_findings(ctx)

        if ctx.cache.tick(announce):
            self._sync_viewer()
        # Outside the tick: the request may have been made by a verdict on a
        # frame the list did not re-read, and a refused submit has to be
        # retried on some later frame rather than on the next list refresh.
        review_mode.pump_findings(ctx)
        self._check_worker()

    def _check_worker(self) -> None:
        """Say so, once, when the GPU worker dies.

        Two plain attribute reads, so this is frame-loop safe. It used to be
        reported only through ``/api/health``, which the browser build polled;
        with the HTTP layer gone a mid-session worker crash became invisible
        outside the log file, and every job queued afterwards simply sat there.
        """
        if self._fatal_reported:
            return
        ctx = self.app_ctx
        fatal = self.runtime.fatal
        if fatal is not None:
            self._fatal_reported = True
            ctx.state.note_error(f"The GPU worker stopped: {fatal}. Restart Warlock.")
            ctx.toast("The GPU worker stopped. Nothing new will run.", "error")
        elif not self.runtime.alive:
            self._fatal_reported = True
            ctx.state.note_error("The GPU worker is not running. Restart Warlock.")
            ctx.toast("The GPU worker is not running.", "error")

    def _request_storage(self) -> None:
        """Re-measure the data directory off the frame thread.

        A recursive stat walk of every job directory is not something to do
        between ``new_frame`` and ``render`` -- and the moment it was being
        asked for is the worst one: the frame that should be showing a job
        finishing. ``submit`` refuses a duplicate key, so a burst of jobs
        completing coalesces into one walk rather than queuing several.
        """
        ctx = self.app_ctx
        ctx.submit("storage", ctx.cache.measure)

    def _reload_viewer(self) -> None:
        """Re-read whatever the viewport is showing, in place.

        ``_sync_viewer`` short-circuits when the path it wants is the path
        already loaded, which is right for a selection change and wrong after a
        retarget: model.glb was rewritten under the same name, so the file to
        reload is the one it is convinced is current.

        ``pending`` is cleared for the same reason and one more: a parse of the
        *pre*-retarget bytes may be in flight, and adopting it afterwards would
        put the old mesh back. Clearing it makes that result unwanted.
        """
        self.viewer.path = None
        self.viewer.pending = None
        self._sync_viewer()

    def _sync_viewer(self) -> None:
        """Show whatever the selection implies, when it changes.

        Driven off the cache rather than off the click so a job that finishes
        while it is selected starts showing its mesh without another click.
        """
        from . import modes

        ctx = self.app_ctx
        if ctx.state.mode not in modes.VIEWPORT_MODES:
            # Only two modes draw a viewport. Everywhere else there is
            # nothing to sync, and loading a mesh for the selection would
            # be work nothing shows.
            return
        job = ctx.job()
        if job is None:
            return
        if self.viewer.pose_mode:
            # The pose editor is showing rig.glb on purpose. Without this the
            # next cache tick decides the selection "should" be showing
            # model.glb and reloads it, which drops the editor -- half a second
            # after it was opened.
            return
        job_dir = ctx.job_dir(job["id"])
        files = job.get("files") or []
        wanted = None
        if ctx.state.mode == "3d" and "model.glb" in files:
            wanted = job_dir / "model.glb"
        elif ctx.state.mode == "2d" and "input.png" in files:
            wanted = job_dir / "input.png"
        if wanted is None or self.viewer.path == wanted or self.viewer.pending == wanted:
            return
        if wanted.suffix == ".png":
            try:
                self.viewer.clear()
                self.viewer.load_reference(wanted)
                self.viewer.path = wanted
            except Exception:
                log.exception("could not open %s", wanted)
                ctx.toast("Could not open that asset.", "error")
            self._refresh_rig_side_data()
            return
        # A GLB is parsed off-thread and adopted when it lands. This runs on a
        # *timer*, on the frame a job transitions to done -- which is when the
        # file is largest and coldest -- so doing the parse and the texture
        # decode here froze the frame that was meant to show the job finishing.
        # The GPU upload stays on the frame thread; see ``_adopt_model``.
        self.viewer.pending = wanted
        if not ctx.submit(VIEWER_KEY, self.viewer.parse_model, wanted, tag=wanted):
            # Another load is already in flight. Its result is checked against
            # ``pending`` before it is adopted, so this one is simply retried
            # on the next tick rather than queued.
            self.viewer.pending = None

    def _adopt_model(self, done: Any) -> None:
        """Take a parsed GLB as the viewer's current model. Frame thread only.

        The upload is what has to be here -- ``GpuModel`` creates buffers and
        textures on the one GL context, and releasing the old one does too.

        Checked against ``pending`` first: the selection can move while a parse
        is in flight, and adopting a result nobody is waiting for any more
        would put the previous asset back on screen.
        """
        ctx = self.app_ctx
        wanted = done.tag
        if wanted is None or self.viewer.pending != wanted:
            self.viewer.pending = None
            return
        self.viewer.pending = None
        try:
            self.viewer.adopt_model(done.result, wanted)
        except Exception:
            log.exception("could not open %s", wanted)
            ctx.toast("Could not open that asset.", "error")
            return
        job = ctx.job()
        # The thumbnail is free here: the model is loaded and framed, and a
        # server-side render would need the serial GPU queue for something
        # purely cosmetic.
        if job is not None and "thumb.png" not in (job.get("files") or []):
            ctx.capture_thumbnail(job["id"])
        self._refresh_rig_side_data()

    def _refresh_rig_side_data(self) -> None:
        """Poses, sheets and the shipped preset library, off-thread.

        Cleared first: these belong to a job, and leaving the previous one's
        poses on screen while the new one's are read would offer a list that
        applies to nothing.
        """
        from ..service import rig as svc_rig
        from ..service import sheets as svc_sheets

        ctx = self.app_ctx
        job = ctx.job()
        for key in ("poses", "sheets", "bones"):
            ctx.state.preview.pop(key, None)
        if job is None:
            return
        job_id = job["id"]
        if "rig.glb" in (job.get("files") or []):
            ctx.submit(f"poses:{job_id}", svc_rig.list_poses, ctx.svc, job_id)
        ctx.submit(f"sheets:{job_id}", svc_sheets.list_sheets, ctx.svc, job_id)

    def load_presets(self, template: str | None) -> None:
        """The shipped pose library for a skeleton.

        Read once per template rather than per job: it is a property of the
        rig, not of the mesh, and applying one saves an ordinary pose through
        the same path a hand-made one does.
        """
        from ..service import rig as svc_rig

        if not template:
            self.app_ctx.state.preview["presets"] = []
            return
        self.app_ctx.submit(f"presets:{template}", svc_rig.template_presets, template)

    def _events(self) -> None:
        import pygame
        from imgui_bundle import imgui

        from . import imgui_backend

        ctx = self.app_ctx
        io = imgui.get_io()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_quit()
                continue
            if event.type == pygame.VIDEORESIZE:
                # The *clamped* size is persisted, not the requested one: the
                # window that comes back is the clamped one, so storing the
                # raw event meant next launch opened below the resize floor
                # with no event to correct it.
                sized = (max(event.w, self._min_size[0]), max(event.h, self._min_size[1]))
                pygame.display.set_mode(
                    sized, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
                )
                ctx.settings.set("window_size", list(sized))
                continue
            if event.type == pygame.DROPFILE:
                self._on_drop(Path(event.file))
                continue
            imgui_backend.process_event(event)
            if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                if not io.want_text_input:
                    self._shortcut(event)
                continue
            # Clay owns its own centre pane, so its viewport takes the mouse
            # in that mode and the asset viewer never sees it -- the two would
            # otherwise both orbit on one drag.
            if ctx.state.mode == "clay":
                self._build_event(event)
                continue
            # The viewer sees the mouse when it is over the viewport image, and
            # a drag already in progress keeps it wherever the cursor goes.
            if self._viewport_hovered or self.viewer._grab is not None:
                self.viewer.handle_event(event, hovered=self._viewport_hovered)

    def _build_event(self, event: Any) -> None:
        """Route the mouse to Clay's viewport, on the same hover rule.

        A drag already in progress ignores the hover, so crossing onto a panel
        mid-orbit does not drop it -- which is exactly what ``_grab`` is for in
        the asset viewer.
        """
        from . import clay_mode

        tab = clay_mode.active(self.app_ctx)
        if tab is None or self.clay_view is None:
            return
        # Every panel refuses edits while a save is in flight; a gizmo or
        # element drag pushes history steps too, so the viewport must as well.
        # Only new presses are refused: a drag already in progress keeps its
        # release (the bytes were captured before the save started), and
        # swallowing it would strand _grab.
        import pygame

        if tab.saving and event.type == pygame.MOUSEBUTTONDOWN:
            return
        hovered = self._build_hovered
        if hovered or self.clay_view._grab is not None:
            self.clay_view.handle_event(tab.doc, event, hovered)

    def _shortcut(self, event: Any) -> None:
        import pygame

        ctx = self.app_ctx
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
            ctx.state.mode = "manual"
            return
        # Above the landing and Inker returns below: the frame rate is a
        # property of the loop, not of whichever pane happens to be on screen,
        # and the chooser is exactly where a slow startup would show.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F10:
            ctx.state.show_fps = not ctx.state.show_fps
            return
        from . import modes

        if ctx.state.mode not in modes.WORK_MODES:
            # Home, the Manual and Settings have no form to submit and no
            # viewport to frame; every one of these would act on a pane that is
            # not on screen.
            return
        if ctx.state.mode == "clay":
            from . import clay_mode

            # First refusal, and unconditional for the reason Inker's is:
            # handle_key returns False with no document open, and letting that
            # fall through meant F/W/S acted on a viewport Clay has replaced.
            clay_mode.handle_key(ctx, event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self._frame_clay_selection()
            return
        if ctx.state.mode == "review":
            from . import review_mode

            # Unconditional for the reason Clay's and Inker's are: handle_key
            # returns False with no sweep run open, and letting that fall
            # through would let A/S/R act on a viewport and forms Review has
            # replaced. Nothing below this line belongs to Review.
            review_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "inker":
            from . import inker_mode

            # Unconditionally, whether or not handle_key consumed it: it
            # returns False when no document is open, and letting that fall
            # through meant F/W/S toggled wireframe and turntable and
            # Ctrl+Enter submitted a mesh job -- all against a viewport Inker
            # has replaced. Nothing below this line belongs to Inker.
            inker_mode.handle_key(ctx, event)
            return
        # Both edges reach this function, because Inker's space-to-pan is a
        # hold and needs the release. Nothing below is a hold: every one of
        # these is a toggle or an action, so acting on the release too undoes
        # the toggle the press just made and submits a second job for one
        # Ctrl+Enter.
        if event.type != pygame.KEYDOWN:
            return
        mods = pygame.key.get_mods()
        if event.key == pygame.K_RETURN and mods & pygame.KMOD_CTRL:
            from .panes import settings_2d, settings_3d

            if ctx.state.mode == "2d":
                settings_2d.generate(ctx, ctx.state.form_2d)
            else:
                settings_3d.promote(ctx, ctx.cache.get(ctx.state.source_job), ctx.state.form_3d)
        elif event.key == pygame.K_ESCAPE:
            from .panes import pose_panel

            if ctx.state.comparing:
                ctx.state.comparing = None
                self.viewer.exit_compare()
            elif self.viewer.pose_mode:
                pose_panel.guard(ctx, "leave edit mode", lambda: pose_panel.leave(ctx))
        elif event.key == pygame.K_f:
            self.viewer.frame()
        elif event.key == pygame.K_w:
            ctx.state.wireframe = not ctx.state.wireframe
            self.viewer.set_wireframe(ctx.state.wireframe)
        elif event.key == pygame.K_s:
            ctx.state.turntable = not ctx.state.turntable
            self.viewer.set_turntable(ctx.state.turntable)

    def _on_drop(self, path: Path) -> None:
        from .panes import settings_3d

        ctx = self.app_ctx
        if ctx.state.mode == "inker":
            from . import inker_mode

            inker_mode.open_path(ctx, path)
            return
        if ctx.state.mode == "clay":
            from . import clay_mode, clay_state

            if path.suffix.lower() == clay_state.WBLK_SUFFIX:
                clay_mode.open_path(ctx, path)
            elif path.suffix.lower() == ".glb":
                clay_mode.import_glb_path(ctx, path)
            else:
                ctx.toast("Clay opens .wblk documents and .glb meshes.", "error")
            return
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            ctx.toast("Drop an image to start a mesh from it.", "error")
            return
        if ctx.state.mode == "2d":
            # In the 2D pane a dropped image is a *conditioning reference*, not
            # a mesh to build -- forcing the mode switch here would throw away
            # the prompt the user is composing. One branch, and it is what
            # makes the feature discoverable at all.
            ctx.state.form_2d["ref_path"] = str(path)
            ctx.toast(f"Using {path.name} as the reference.")
            return
        # A drop is a start: it would otherwise land behind the chooser, with
        # nothing on screen saying anything had happened.
        ctx.state.mode = "3d"
        settings_3d.upload(ctx, path)

    def _request_quit(self) -> None:
        """One chain, in order: painted pixels, then built geometry, then a pose.

        Nested rather than asked side by side, because ``ConfirmQueue`` holds a
        single pending question: three at once would silently drop two, and the
        user would lose whichever they were not shown.
        """
        from . import clay_mode, inker_mode
        from .panes import pose_panel

        ctx = self.app_ctx
        inker_mode.guard(
            ctx,
            "quit",
            lambda: clay_mode.guard(
                ctx, "quit", lambda: pose_panel.guard(ctx, "quit", self._quit)
            ),
        )

    def _quit(self) -> None:
        self._running = False

    # -- the UI ------------------------------------------------------------

    def _build_ui(self) -> None:
        from imgui_bundle import imgui

        from . import modes
        from .panes import (
            app_settings,
            inspector,
            landing,
            library,
            settings_2d,
            settings_3d,
        )

        ctx = self.app_ctx
        # Recomputed every frame by whoever draws the viewport image. Every
        # mode but 3D returns without drawing it, so it stays false there and
        # the viewer gets no events at all.
        self._viewport_hovered = False
        # Arriving in a viewport mode is a change the cache will not announce:
        # the job list has not changed, so nothing else would ask the viewer
        # to show what was just picked.
        if ctx.state.mode != self._last_mode and ctx.state.mode in modes.VIEWPORT_MODES:
            self._sync_viewer()
        if ctx.state.mode != self._last_mode and ctx.state.mode == "review":
            # Arriving is the one moment a rescan is certainly wanted, and it
            # is a mode change rather than a job-cache tick, so nothing else
            # would ask. Driven off the change and not off "the list is empty",
            # which would submit a walk of the bench directory every frame on a
            # machine that has never run a sweep.
            from . import review_mode

            review_mode.scan(ctx)
        self._last_mode = ctx.state.mode

        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.work_pos)
        imgui.set_next_window_size(viewport.work_size)
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
            | imgui.WindowFlags_.no_saved_settings.value
        )
        imgui.begin("##host", None, flags)
        # The switch is drawn in every mode, Home included: it is how you
        # leave wherever you are, so a mode that hides it is a dead end.
        self._mode_switch()
        from .panes import overlay

        overlay.doctor_banner(ctx)
        mode = ctx.state.mode
        if mode in _SINGLE_PANE_MODES or mode in modes.WORKSPACE_MODES:
            if mode == "home":
                landing.draw(ctx)
            elif mode == "manual":
                from .manual import render as manual_render

                manual_render.draw_body(ctx)
            elif mode == "settings":
                app_settings.draw(ctx)
            elif mode == "clay":
                self._clay_workspace()
            elif mode == "review":
                self._review_workspace()
            else:
                self._inker_workspace()
            imgui.end()
            self._overlays(viewport)
            return

        # The library used to share the left sidebar with settings, split by
        # settings_share; it shares the right sidebar with the inspector now
        # instead, so the left column is settings alone (nothing left to split
        # against) and the right column is the two-scroller stack that used to
        # live on the left.
        from . import layout as layout_mod
        from .tokens import sp

        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)
        if layout_mod.pane_child("settings", (sidebar_w, 0)):
            if ctx.state.mode == "2d":
                settings_2d.draw(ctx)
            else:
                settings_3d.draw(ctx)
        imgui.end_child()

        imgui.same_line()
        self._viewport_pane()
        imgui.same_line()

        _right_column(
            ctx, lay, sidebar_w, inspector_draw=inspector.draw, library_draw=library.draw
        )

        imgui.end()
        self._overlays(viewport)

    def _ensure_build_view(self) -> Any:
        from .clay_view import ClayView

        if self.clay_view is None:
            self.clay_view = ClayView(self.ctx, self.app_ctx)
        return self.clay_view

    def _frame_clay_selection(self) -> None:
        """F, in Clay. Frames the selection, or the whole document."""
        from . import clay_mode

        tab = clay_mode.active(self.app_ctx)
        if tab is not None and self.clay_view is not None:
            self.clay_view.frame_selection(tab.doc)

    def _capture_clay_thumbnail(self, job_id: str) -> None:
        """The library card's picture, from the viewport that is already drawn.

        On the frame thread because it reads a framebuffer, which is the same
        reason ``ctx.capture_thumbnail`` is -- and it is the one deliberate
        exception to "the frame loop never blocks", being a single offscreen
        read rather than work.
        """
        from ..service import files as svc_files

        ctx = self.app_ctx
        if self.clay_view is None:
            return
        try:
            data = self.clay_view.thumbnail_png()
        except Exception:
            log.exception("could not capture a thumbnail for built asset %s", job_id)
            return
        ctx.submit(f"thumb:{job_id}", svc_files.save_thumbnail, ctx.svc, job_id, data)

    def _clay_send_to_3d(self, tab: Any) -> None:
        """Render the document flat and hand the picture to trellis.

        The render is **synchronous on the frame thread** because it needs the
        GL context -- one offscreen draw, exactly what ``capture_thumbnail``
        already is. Only the service call goes to a task thread, which is the
        shape ``inker_mode.send_to_3d`` already has.

        Flat-shaded, on a plain background, with no grid, no gizmos and no
        overlays: trellis is being given a *subject*, and a grid line in the
        picture is a subject too.
        """
        from .panes import settings_3d

        ctx = self.app_ctx
        try:
            png = self._render_clay_reference(tab)
        except Exception:
            log.exception("could not render the build reference")
            ctx.toast("That document could not be rendered.", "error")
            return
        settings_3d.upload_bytes(ctx, png)

    def _render_clay_reference(self, tab: Any, size: int = 1024) -> bytes:
        """One offscreen 1024-square draw of the document, as PNG bytes."""
        from .viewer import capture, glctx

        view = self._ensure_build_view()
        view.sync(tab.doc)
        target = glctx.Viewport(self.ctx, (size, size))
        try:
            view.renderer.draw(
                target,
                view.camera,
                view._composite(tab.doc),
                flat=True,
                show_grid=False,
                background=(1.0, 1.0, 1.0, 1.0),
                overlays=[],
            )
            return capture.png_bytes(target)
        finally:
            target.release()

    def _clay_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses.

        Mirrors ``_inker_workspace`` line for line, including ``settings_share``
        for the vertical split, so the two editors do not drift into looking
        like different applications:

            [ clay_tools ]            [ clay_outliner ]
            [ clay_props ]  viewport  [ clay_bridge   ]
        """
        from imgui_bundle import imgui

        from . import clay_mode, widgets
        from . import layout as layout_mod
        from .panes import clay_bridge, clay_outliner, clay_props, clay_tools
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)

        imgui.begin_group()
        tools_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("clay-tools", (sidebar_w, tools_height)):
            clay_tools.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("clay-props", (sidebar_w, 0)):
            clay_props.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        if layout_mod.pane_child("clay-centre", (width, 0), flags):
            self._clay_viewport(ctx, clay_mode, widgets)
        imgui.end_child()

        imgui.same_line()
        imgui.begin_group()
        outliner_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("clay-outliner", (0, outliner_height)):
            clay_outliner.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("clay-bridge", (0, 0)):
            clay_bridge.draw(ctx)
        imgui.end_child()
        imgui.end_group()

    def _clay_viewport(self, ctx: Any, clay_mode: Any, widgets: Any) -> None:
        from imgui_bundle import imgui

        from .panes import clay_menu

        tab = clay_mode.active(ctx)
        if tab is None:
            self._clay_empty(ctx, clay_mode)
            return
        avail = imgui.get_content_region_avail()
        rect = (
            imgui.get_cursor_screen_pos().x,
            imgui.get_cursor_screen_pos().y,
            max(avail.x, 1.0),
            max(avail.y, 1.0),
        )
        state = clay_mode.ensure(ctx)
        view = self._ensure_build_view()
        view.wireframe = state.wireframe
        view.show_grid = state.grid
        texture = view.draw(tab.doc, rect, 1.0 / TARGET_FPS)
        imgui.image(widgets.texture_ref(texture), (rect[2], rect[3]), (0, 1), (1, 0))
        self._build_hovered = imgui.is_item_hovered()
        self._clay_marquee(imgui, view, rect)
        clay_menu.draw(ctx, view)

    def _clay_empty(self, ctx: Any, clay_mode: Any) -> None:
        """What Clay shows with nothing open, mirroring the raster editor's.

        Buttons rather than a sentence: ``new_document`` was reachable only
        through Ctrl+N, so the empty state told the user to "start a document"
        and offered no way to.
        """
        from pathlib import Path

        from imgui_bundle import imgui

        from . import widgets

        imgui.dummy((0, 40))
        imgui.text("Nothing open")
        widgets.muted("Start a model, open a document, or drop a .wblk on the window.")
        imgui.dummy((0, 16))
        if imgui.button("New model", (240, 0)):
            clay_mode.new_document(ctx)
        imgui.dummy((0, 8))
        if imgui.button("Open a file...", (240, 0)):
            clay_mode.ask_open(ctx)
        state = clay_mode.ensure(ctx)
        if state.recent:
            imgui.dummy((0, 16))
            widgets.section("recent")
            for path in list(state.recent)[:6]:
                # The path is in the id, not just the label: two documents can
                # share a basename and one imgui id between them is one row.
                if imgui.selectable(f"{Path(path).name}##{path}", False)[0]:
                    clay_mode.open_path(ctx, Path(path))
                if imgui.is_item_hovered():
                    imgui.set_tooltip(path)

    def _clay_marquee(self, imgui: Any, view: Any, rect: Any) -> None:
        """The selection rectangle, drawn in imgui rather than in GL.

        It is a two-dimensional screen decoration with no depth and no place in
        the scene, so putting it through the renderer would mean a vertex
        buffer rebuilt every mouse-move for four corners. The draw list is
        already there and already clipped to this window.
        """
        box = getattr(view, "marquee", None)
        if box is None:
            return
        draw = imgui.get_window_draw_list()
        x0, y0 = rect[0] + min(box[0], box[2]), rect[1] + min(box[1], box[3])
        x1, y1 = rect[0] + max(box[0], box[2]), rect[1] + max(box[1], box[3])
        draw.add_rect_filled((x0, y0), (x1, y1), imgui.get_color_u32((1, 1, 1, 0.08)))
        draw.add_rect((x0, y0), (x1, y1), imgui.get_color_u32((1, 1, 1, 0.55)))

    def _inker_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton the other modes use.

        Deliberately not a takeover of the whole window: the progress card
        floats over every mode, so a trellis run started before switching here
        is still visible while painting.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import (
            inker_bridge,
            inker_canvas,
            inker_colors,
            inker_layers,
            inker_tools,
        )
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)
        imgui.begin_group()
        tools_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("inker-tools", (sidebar_w, tools_height)):
            inker_tools.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("inker-colors", (sidebar_w, 0)):
            inker_colors.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        if layout_mod.pane_child("inker-centre", (width, 0), flags):
            inker_canvas.draw(ctx)
        imgui.end_child()

        imgui.same_line()
        imgui.begin_group()
        layers_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("inker-layers", (0, layers_height)):
            inker_layers.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("inker-bridge", (0, 0)):
            inker_bridge.draw(ctx)
        imgui.end_child()
        imgui.end_group()

    def _review_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses:

            [ review-runs  ]              [ review-verdict ]
            [ review-units ]  the mesh    [ the reference  ]

        The centre borrows the *shared* asset viewer rather than a second one:
        one GL context, one framebuffer, and a sweep unit's model.glb is an
        ordinary GLB. Leaving Review needs no cleanup because ``_sync_viewer``
        compares ``viewer.path`` against what the selection implies and reloads
        the moment 3D is on screen again.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import review_mode
        from .tokens import sp

        ctx = self.app_ctx
        state = review_mode.ensure(ctx)
        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)

        imgui.begin_group()
        runs_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("review-runs", (sidebar_w, runs_height)):
            self._review_runs(ctx, state, review_mode)
        imgui.end_child()
        if layout_mod.pane_child("review-units", (sidebar_w, 0)):
            self._review_units(state, review_mode)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        if layout_mod.pane_child("review-centre", (width, 0), flags):
            self._review_viewport(state, review_mode, width)
        imgui.end_child()

        imgui.same_line()
        if layout_mod.pane_child("review-verdict", (0, 0)):
            self._review_verdict(ctx, state, review_mode)
        imgui.end_child()

    def _review_runs(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """The sweep list, and the form that launches a new one."""
        from imgui_bundle import imgui

        from . import icons, widgets
        from .manual import render as manual_render

        widgets.section("Sweeps")
        manual_render.help_button(ctx, "review")
        if widgets.disabled_button(f"{icons.REFRESH} Rescan", not state.scanning):
            review_mode.scan(ctx)
        if state.scanning:
            imgui.same_line()
            widgets.muted("Reading...")
        for sweep in state.sweeps:
            todo = sweep["todo"]
            total = len(sweep["units"])
            selected = sweep["id"] == state.sweep_id
            if imgui.selectable(f"{sweep['label']}##sweep-{sweep['id']}", selected)[0]:
                review_mode.open_sweep(ctx, sweep["id"])
            widgets.muted(f"   {total - todo}/{total} reviewed")
            if selected and sweep["id"] != review_mode.RECENT_ID:
                self._review_delete_button(ctx, state, review_mode, sweep)
        imgui.separator()
        self._review_form(ctx, state, review_mode)

    def _review_delete_button(self, ctx: Any, state: Any, review_mode: Any, sweep: Any) -> None:
        """Delete a sweep's jobs and meshes, keeping what they taught.

        Behind the same confirm an asset delete goes through
        (``panes/library.py``), because it is the same kind of act. What the
        message has to say is the part that is *not* obvious: the verdicts and
        the findings they feed survive, because each verdict carries its own
        snapshot of the settings it was filed against.
        """
        from imgui_bundle import imgui

        from . import dialogs, icons, widgets

        sweep_id = sweep["id"]
        if widgets.icon_button(
            f"{icons.TRASH}##delete-{sweep_id}",
            "Delete this sweep's jobs and meshes",
            danger=True,
            enabled=not state.scanning,
        ):
            ctx.confirms.ask(
                dialogs.Confirm(
                    title="Delete this sweep?",
                    message=(
                        f"{sweep['label']}: its {len(sweep['units'])} job(s), their meshes "
                        "and their reference images are deleted.\n\n"
                        "The verdicts you recorded are kept, and so are the findings "
                        "they feed -- each one carries its own copy of the settings it "
                        "was filed against."
                    ),
                    confirm_label="Delete",
                    cancel_label="Keep",
                    on_confirm=lambda: review_mode.delete(ctx, sweep_id),
                )
            )
        imgui.dummy((0, 0))

    def _review_form(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """New sweep: a prompt, a baseline captured from the generate forms,
        seeds, and the axes to vary."""
        from imgui_bundle import imgui

        from ..service import sweeps as sweeps_mod
        from . import widgets

        if not widgets.header("New sweep", default_open=False):
            return
        form = state.form
        widgets.field_label("prompt")
        form.prompt = widgets.multiline("##sweep-prompt", form.prompt, 60, 1000)
        widgets.field_label("name")
        form.label = widgets.input_text("##sweep-label", form.label, max_length=120)
        widgets.field_label("seeds")
        form.seeds = widgets.input_text("##sweep-seeds", form.seeds, max_length=120)

        if imgui.button("Start from current 2D/3D settings"):
            form.base = review_mode.capture_base(ctx)
            form.base_note = f"{len(form.base)} setting(s) captured"
            ctx.toast("Captured the current settings as this sweep's baseline.")
        widgets.muted(form.base_note or "No baseline captured; units use the defaults.")

        widgets.field_label("vary")
        options = [("", "-")] + [(p, p) for p in sweeps_mod.axis_params()]
        for i, row in enumerate(form.axes):
            imgui.push_id(f"axis-{i}")
            row["param"] = widgets.combo("##param", row.get("param", ""), options, width=-1)
            row["values"] = widgets.input_text(
                "##values", row.get("values", ""), max_length=200, hint="comma-separated"
            )
            imgui.pop_id()
        if imgui.button("Add axis"):
            form.axes.append({"param": "", "values": ""})
        if len(form.axes) > 1:
            imgui.same_line()
            if imgui.button("Remove axis"):
                form.axes.pop()

        planned = review_mode.preview_units(state)
        widgets.muted(
            "Fill in the prompt and one axis." if planned < 0
            else f"{planned} job(s) - roughly two minutes of GPU each."
        )
        enabled = planned > 0 and not form.submitting and not state.scanning
        if widgets.primary_button("Launch sweep", (-1, 0), enabled=enabled):
            review_mode.launch(ctx)

    def _review_units(self, state: Any, review_mode: Any) -> None:
        from imgui_bundle import imgui

        from . import icons, widgets

        widgets.section("Units")
        if not state.units:
            widgets.muted("Nothing to review here.")
            return
        for i, unit in enumerate(state.units):
            mark = {"accept": icons.CHECK, "reject": icons.X}.get(unit["verdict"] or "", " ")
            if imgui.selectable(
                f"{mark} {review_mode.label(unit)}##unit-{unit['job_id']}", i == state.index
            )[0]:
                review_mode.step(state, i - state.index)

    def _review_viewport(self, state: Any, review_mode: Any, width: float) -> None:
        """The unit's mesh, in the shared viewer.

        **What decides whether to load is ``viewer.path``, not a remembered
        unit key** -- the same comparison ``_sync_viewer`` makes, and for a
        stronger reason here. Unit keys repeat across runs of one sweep spec,
        so a key-keyed marker said "already showing that" when the mesh on
        screen belonged to a *different run*, and a verdict was then filed
        against a mesh nobody had looked at. The same marker also survived a
        trip through 3D, which loads a library asset into this same viewer, so
        coming back drew that asset under Review's verdict buttons. Comparing
        paths fixes both, structurally, and needs no reset anywhere.
        """
        from imgui_bundle import imgui

        from . import widgets
        from .panes import overlay

        ctx = self.app_ctx
        if ctx.state.comparing:
            # 3D's Escape handler does exactly this pair (main.py's
            # ``_shortcut``), but Review draws no compare UI of its own and
            # its Escape branch returns before that handler runs -- so a
            # split entered in 3D and never exited stays armed forever once
            # the mode switches. ``_draw_viewport_image`` halves the width
            # for any mode whenever ``comparing`` is set, so without this a
            # sweep unit's mesh renders next to a stale compare texture.
            # Checked every frame Review draws (not just on entry), so it
            # also covers 3D -> Review -> 3D -> Review re-entry.
            ctx.state.comparing = None
            self.viewer.exit_compare()

        unit = review_mode.current(state)
        if self.viewer.pose_mode:
            # The pose editor owns the viewer and holds unsaved rotations;
            # loading over it would discard them without the confirm every
            # other exit goes through (``pose_panel.guard``). ``_sync_viewer``
            # refuses on exactly this condition -- this is the same refusal.
            widgets.muted("Finish or close the pose editor to review a mesh.")
            return

        self._review_load(unit, review_mode)

        image_pos = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        height = max(avail.y, 64)
        if unit is None:
            # Before ``has_model``: arriving from 3D leaves an asset loaded,
            # and asking the viewer first drew that asset with no unit selected
            # -- a mesh on screen that no button on the right refers to.
            overlay.placeholder(self.app_ctx)
        elif self.viewer.has_model:
            self._draw_viewport_image(image_pos, width, height)
        else:
            widgets.muted(f"No mesh for this unit (status: {unit['status']}).")

    def _review_load(self, unit: Any, review_mode: Any) -> None:
        """Show the unit's mesh if the viewer is not already showing it.

        ``viewer.path`` is set even when there is nothing to show, so a unit
        whose job errored (or whose GLB will not open) is tried once rather
        than re-attempted -- and re-toasted -- on every frame.
        """
        wanted = None if unit is None else review_mode.model_path(unit)
        if self.viewer.path == wanted:
            return
        if wanted is None or not wanted.exists():
            self.viewer.clear()
            self.viewer.path = wanted
            return
        try:
            self.viewer.load_model(wanted)
        except Exception:
            log.exception("could not open %s", wanted)
            self.viewer.clear()
            self.viewer.path = wanted
            self.app_ctx.toast("Could not open that sweep unit's mesh.", "error")

    def _review_verdict(self, ctx: Any, state: Any, review_mode: Any) -> None:
        from imgui_bundle import imgui

        from . import widgets

        unit = review_mode.current(state)
        if unit is None:
            widgets.muted("Pick a sweep on the left.")
            self._review_findings(ctx)
            return

        widgets.section(review_mode.label(unit))
        widgets.muted(f"{state.index + 1} of {len(state.units)}  -  {unit['job_id']}")

        reference = review_mode.reference_path(unit)
        if reference is not None:
            texture = ctx.textures.get(review_mode.cache_id(unit), reference)
            if texture is not None:
                side = min(imgui.get_content_region_avail().x, 220.0)
                imgui.image(widgets.texture_ref(texture), (side, side))

        for line in review_mode.mesh_lines(unit):
            widgets.muted(line)

        imgui.separator()
        enabled = not state.scanning
        if widgets.primary_button("Accept (A)", enabled=enabled):
            review_mode.record(ctx, "accept")
        imgui.same_line()
        if widgets.disabled_button("Reject (R)", enabled):
            state.pending_reject = True
        imgui.same_line()
        if widgets.disabled_button("Skip (S)", enabled):
            review_mode.advance(state)

        if state.pending_reject:
            widgets.muted("Why? (1-5, Esc to cancel)")
            for number, reason in review_mode.REASON_KEYS.items():
                if widgets.disabled_button(f"{number}  {reason}", enabled):
                    review_mode.record(ctx, "reject", (reason,))
        elif unit["verdict"]:
            recorded = unit["verdict"]
            if unit["reasons"]:
                recorded += " - " + ", ".join(unit["reasons"])
            widgets.muted(f"Recorded: {recorded}")

        self._review_findings(ctx)

    def _review_findings(self, ctx: Any) -> None:
        """What the verdicts add up to, and the one-click way to reuse it.

        Two answers, most conclusive first. Axis verdicts are matched pairs
        recovered from sweep structure -- same prompt, same seed, one param
        differing -- the only all-else-equal comparison in the pool. The
        ranked vectors are whole configurations ordered by their Wilson lower
        bound (the "floor" percentage), because the per-parameter marginals
        are confounded and a raw rate lets a lucky 5/5 outrank a 19/20.
        """
        from imgui_bundle import imgui

        from ..bench import findings as findings_lib
        from ..service import findings as svc_findings
        from . import dialogs, vector_presets, widgets

        doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
        imgui.separator()
        if not widgets.header("What works", default_open=False):
            return
        # Built after the header, not before it. ``load`` is mtime-cached but
        # these are not: one line per contrast plus one per metric, formatted
        # from scratch every frame, for a section that is closed by default.
        top = svc_findings.presets(doc or {})
        axis_lines = findings_lib.comparison_lines(doc)
        if axis_lines:
            widgets.muted("Axis verdicts (matched pairs, all else equal):")
            for line in axis_lines:
                if line.startswith("    "):
                    widgets.muted(line)
                else:
                    imgui.text_wrapped(line)
            imgui.separator()
        if not top:
            widgets.muted(
                f"No whole configuration has {svc_findings.PRESET_MIN_N} "
                "verdicts yet."
                if axis_lines
                else (
                    f"Nothing yet: a configuration needs "
                    f"{svc_findings.PRESET_MIN_N} verdicts to rank, and axis "
                    "verdicts need matched pairs from sweeps sharing seeds."
                )
            )
            return
        for entry in top[:5]:
            summary = vector_presets.describe(entry["vector"])
            imgui.text_wrapped(f"{findings_lib.vector_line(entry)}  -  {summary}")
            measured = findings_lib.metrics_line(entry.get("metrics"))
            if measured:
                widgets.muted(measured)
            vector = entry["vector"]
            if widgets.disabled_button(f"Apply to forms##apply-{entry['key']}", True):
                vector_presets.apply(ctx.state, vector)
                ctx.toast("Applied those settings to the 2D and 3D forms.")
            imgui.same_line()
            if widgets.disabled_button(f"Save as preset...##save-{entry['key']}", True):
                ctx.prompts.ask(
                    dialogs.Prompt(
                        title="Save settings preset",
                        label="Name",
                        value="",
                        on_accept=lambda name, v=vector: self._save_vector_preset(ctx, name, v),
                    )
                )
            imgui.separator()

    @staticmethod
    def _save_vector_preset(ctx: Any, name: str, vector: dict) -> None:
        from . import vector_presets

        if vector_presets.save_preset(ctx.settings, name, vector):
            ctx.toast(f"Saved the preset {name}.")
            return
        # A name that is empty or only spaces is the one refusal, and it used to
        # be silent: the modal closed and nothing was saved or said.
        ctx.toast("A preset needs a name.", "error")

    def _overlays(self, viewport: Any) -> None:
        """Toasts and modals, drawn over whichever layout ran.

        Outside the host window and after it ends, because a modal is its own
        window: the landing screen needs them as much as the workspace does,
        which is why this is not inline in either.
        """
        from . import widgets
        from .panes import overlay

        ctx = self.app_ctx
        overlay.fps_meter(ctx, self.fps)
        if ctx.state.mode != "home":
            overlay.progress_card(ctx, self.eta)
        widgets.toasts(
            ctx.state,
            (viewport.work_size.x, viewport.work_size.y),
            on_action=self._toast_action,
        )
        ctx.confirms.draw()
        ctx.prompts.draw()

    def _toast_action(self, name: str) -> None:
        """What a toast's action button does, kept out of the widget.

        ``widgets.toasts`` knows what to *draw* for an action and nothing about
        what it means, which is what lets state.py carry the name with no
        import of the App and lets a pane raise a toast without either.
        """
        if name == "log":
            self.app_ctx.open_log()

    def _mode_switch(self) -> None:
        from imgui_bundle import imgui

        from . import modes, widgets

        ctx = self.app_ctx
        state = ctx.state
        # No mode switch is destructive: Inker's documents are still open when
        # you come back, because it is a mode rather than a takeover. Only
        # quitting and closing a tab can lose pixels, and both ask.
        selected = widgets.segmented_control(
            "mode-seg",
            [
                (key, f"{icon} {label}")
                for key, label, icon in [*modes.MODES, modes.QUIT]
            ],
            state.mode,
        )
        if selected == modes.QUIT[0]:
            # An action rather than a mode, so ``state.mode`` is never assigned
            # here -- cancelling leaves the switch exactly where it was. The
            # window's X keeps today's behaviour (the unsaved-work chain, no
            # extra question); only this button always asks, because a switch
            # segment is a click away from every other mode.
            from . import dialogs

            ctx.confirms.ask(
                dialogs.Confirm(
                    title="Quit Warlock Studio?",
                    message="Anything still generating is cancelled.",
                    confirm_label="Quit",
                    cancel_label="Stay",
                    on_confirm=self._request_quit,
                )
            )
        elif selected != state.mode:
            state.mode = selected
            if selected == "home":
                state.landing_view = "choose"

        # Right-aligned health dot: green when everything passed, amber when a
        # non-fatal check failed, red for a fatal one or a dead worker. Click
        # for the full diagnostics list -- the non-fatal checks (missing
        # weights, gltfpack, CUDA) used to be visible only in the log file.
        from . import theme
        from .tokens import sp

        checks = list(getattr(ctx.runtime, "checks", []) or [])
        if state.errors or any(c.fatal and not c.ok for c in checks):
            colour = theme.ERR
        elif any(not c.ok for c in checks):
            colour = theme.WARN
        else:
            colour = theme.OK
        from . import widgets

        imgui.same_line(max(imgui.get_window_width() - sp(64), 0))
        if widgets.icon_button("?", "Keyboard shortcuts"):
            imgui.open_popup("shortcuts")
        self._shortcuts_popup()
        imgui.same_line()
        pos = imgui.get_cursor_screen_pos()
        centre_y = pos.y + imgui.get_frame_height() * 0.5
        imgui.get_window_draw_list().add_circle_filled(
            (pos.x + sp(8), centre_y), sp(4.5), imgui.get_color_u32(theme.rgba(colour)), 16
        )
        if imgui.invisible_button("##health", (sp(16), imgui.get_frame_height())):
            imgui.open_popup("diagnostics")
        if imgui.is_item_hovered():
            imgui.set_tooltip("System status - click for details")
        self._diagnostics_popup(checks)

    # Every binding the app answers to, in one place the user can find. The
    # tuples are (keys, what), grouped; Inker's letters come from TOOL_KEYS so
    # this list cannot drift from the handler.
    def _shortcuts_popup(self) -> None:
        from imgui_bundle import imgui

        from . import widgets
        from .tokens import sp

        if not imgui.begin_popup("shortcuts"):
            return

        def table(title: str, rows: list[tuple[str, str]]) -> None:
            widgets.section(title)
            if imgui.begin_table(f"keys/{title}", 2):
                for keys, what in rows:
                    imgui.table_next_column()
                    widgets.muted(keys)
                    imgui.table_next_column()
                    imgui.text(what)
                imgui.end_table()

        imgui.dummy((sp(420), 0))
        table(
            "Everywhere",
            [
                ("F1", "Switch to the Manual"),
                ("F10", "Toggle the frame-rate readout"),
            ],
        )
        table(
            "2D / 3D",
            [
                ("Ctrl+Enter", "Generate / Make 3D"),
                ("F", "Frame the model"),
                ("W", "Toggle wireframe"),
                ("S", "Toggle turntable"),
                ("Esc", "Exit comparison / pose edit"),
            ],
        )
        table(
            "Review",
            [
                ("A", "Accept the unit on screen"),
                ("R", "Reject (then 1-5 picks the reason)"),
                ("S", "Skip to the next unverdicted unit"),
                ("Left / Right", "Previous / next unit"),
                ("Esc", "Cancel a pending reject"),
            ],
        )
        from .clay_mode import TOOL_KEYS as CLAY_KEYS
        from .inker_mode import TOOL_KEYS

        table(
            "Clay",
            [
                (
                    " / ".join(k.upper() for k in CLAY_KEYS),
                    " / ".join(CLAY_KEYS.values()),
                ),
                ("1 / 2 / 3 / 4", "Vertex / edge / face / object mode"),
                ("E", "Extrude (with faces selected)"),
                ("F", "Frame the selection"),
                ("Delete", "Delete -- faces in an element mode, objects otherwise"),
                ("Ctrl+D", "Duplicate (object mode)"),
                ("Ctrl+J", "Merge the selected objects (object mode)"),
                ("Ctrl+A", "Select all, in the current mode"),
                ("Ctrl+Shift+I", "Invert the selection"),
                ("Right-click", "Context menu"),
                ("Alt+drag", "Orbit, in any mode"),
                ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
                ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
                ("Ctrl+N / O", "New / open"),
                ("Ctrl+E", "Export to the library"),
                ("Ctrl+Tab", "Next document"),
            ],
        )
        tools = ", ".join(f"{k.upper()}" for k in sorted(TOOL_KEYS))
        table(
            "Inker",
            [
                (tools, "Pick a tool (hover a tool for its letter)"),
                ("X", "Swap colours"),
                ("[ / ]", "Brush size (Shift: hardness)"),
                ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
                ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
                ("Ctrl+Shift+E", "Export PNG"),
                ("Ctrl+N / O / W", "New / open / close"),
                ("Ctrl+A / D", "Select all / deselect"),
                ("Ctrl+C / X / V", "Copy / cut / paste"),
                ("Ctrl+Shift+V", "Paste as a layer"),
                ("Ctrl+Shift+I", "Invert the selection"),
                ("Ctrl+T", "Free transform"),
                ("Ctrl+Tab", "Next tab"),
                ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
                ("Space / middle drag", "Pan (wheel zooms)"),
            ],
        )
        imgui.end_popup()

    def _diagnostics_popup(self, checks: list[Any]) -> None:
        from imgui_bundle import imgui

        from . import theme, widgets
        from .tokens import sp

        ctx = self.app_ctx
        imgui.set_next_window_size((sp(460), 0))
        if not imgui.begin_popup("diagnostics"):
            return
        widgets.section("Diagnostics")
        for check in checks:
            colour = theme.OK if check.ok else (theme.ERR if check.fatal else theme.WARN)
            widgets.text_colored(colour, "o" if check.ok else "x")
            imgui.same_line()
            imgui.text(check.name)
            imgui.same_line()
            widgets.muted("-")
            imgui.same_line()
            imgui.push_style_color(
                imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.MUTED))
            )
            imgui.text_wrapped(str(check.detail))
            imgui.pop_style_color()
        if not checks:
            widgets.muted("No checks ran.")
        imgui.separator()
        if imgui.button("Copy details"):
            imgui.set_clipboard_text(
                "\n".join(
                    f"{'ok' if c.ok else 'FAIL'} {c.name}: {c.detail}" for c in checks
                )
            )
        imgui.same_line()
        log_path = Path(ctx.runtime.config.data_dir) / "warlock.log"
        if widgets.disabled_button("Open the log", log_path.exists()):
            ctx.open_log()
        imgui.end_popup()

    def _viewport_pane(self) -> None:
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import overlay

        ctx = self.app_ctx
        # Leave room for the inspector; the progress card floats over the image
        # now, so the full height is the image's.
        width = layout_mod.centre_width()
        # no_scroll_with_mouse: over the viewport the wheel can only mean dolly.
        if layout_mod.pane_child(
            "viewport", (width, 0), imgui.WindowFlags_.no_scroll_with_mouse.value
        ):
            overlay.toolbar(ctx)
            image_pos = imgui.get_cursor_screen_pos()
            avail = imgui.get_content_region_avail()
            height = max(avail.y, 64)
            if ctx.state.mode == "3d" and self.viewer.has_model:
                self._draw_viewport_image(image_pos, width, height)
            elif ctx.state.mode == "2d" and self.viewer.reference is not None:
                self._draw_reference(width, height)
            else:
                overlay.placeholder(ctx)
        imgui.end_child()

    def _draw_viewport_image(self, pos: Any, width: float, height: float) -> None:
        from imgui_bundle import imgui

        from . import widgets

        ctx = self.app_ctx
        # AppState.select clears the flag but cannot reach the viewer, so the
        # split's GPU half is reconciled here -- otherwise a selection change
        # mid-compare leaves the stale mesh rendering a full second scene draw
        # every frame with nothing on screen showing it.
        if not ctx.state.comparing and self.viewer.comparing:
            self.viewer.exit_compare()
        halves = 2 if ctx.state.comparing else 1
        cell = (width - (8 if halves == 2 else 0)) / halves
        texture = self.viewer.render((pos.x, pos.y, cell, height), imgui.get_io().delta_time)
        # UV flipped: GL's origin is bottom-left and imgui's is top-left.
        imgui.image(widgets.texture_ref(texture), (cell, height), (0, 1), (1, 0))
        self._viewport_hovered |= imgui.is_item_hovered()
        if halves == 2 and self.viewer.compare_viewport is not None:
            imgui.same_line()
            imgui.image(
                widgets.texture_ref(self.viewer.compare_viewport.texture),
                (cell, height), (0, 1), (1, 0),
            )
            self._viewport_hovered |= imgui.is_item_hovered()

    def _draw_reference(self, width: float, height: float) -> None:
        from imgui_bundle import imgui

        from . import widgets

        texture = self.viewer.reference
        scale = min(width / texture.size[0], height / texture.size[1])
        imgui.image(
            widgets.texture_ref(texture),
            (texture.size[0] * scale, texture.size[1] * scale),
        )

    # -- teardown ----------------------------------------------------------

    def teardown(self) -> None:
        """Unwind everything, and let no step stop a later one.

        Each stage is independent, and the last of them -- runtime.shutdown --
        is the one that stops the worker loop and the trellis child. A GL
        release raising on a lost context used to skip it, which is a stranded
        trellis-server and a process that will not exit.
        """
        import pygame

        if self.fps.frames:
            log.info("frame loop: %s", self.fps.summary())
        ctx = self.app_ctx
        if ctx is not None:
            _step("persist settings", lambda: self._persist(ctx))
            _step("persist inker", lambda: self._persist_inker(ctx))
            _step("persist build", lambda: self._persist_clay(ctx))
            if ctx.textures is not None:
                _step("release textures", ctx.textures.release)
            from .panes import sheet_panel

            _step("release sheet strip", lambda: sheet_panel.release_strip_texture(ctx))
        if self.viewer is not None:
            _step("release viewer", self.viewer.release)
        # ``getattr``, not an attribute access: teardown runs after a *failed*
        # setup too, and Clay's viewport is one of the last things constructed
        # -- an AttributeError here would skip runtime.shutdown, which is the
        # step that stops the worker loop and the trellis child.
        clay_view = getattr(self, "clay_view", None)
        if clay_view is not None:
            _step("release clay view", clay_view.release)
        if self.imgui_renderer is not None:
            _step("shutdown imgui", self.imgui_renderer.shutdown)
        _step("pygame.quit", pygame.quit)
        _step("runtime shutdown", self.runtime.shutdown)
        # The line whose *absence* is evidence: a session that ends without it
        # died somewhere no `except` could see.
        log.info("teardown complete")

    def _persist(self, ctx: Any) -> None:
        from .settings import sanitise_form

        # No mode: the app opens on Home every launch, so storing the one it
        # happened to quit in would have no reader -- and quitting from the
        # Manual or Settings would store a mode nothing would want restored.
        ctx.settings.set("show_fps", ctx.state.show_fps)
        ctx.settings.set("form_2d", sanitise_form(ctx.state.form_2d))
        ctx.settings.set("form_3d", sanitise_form(ctx.state.form_3d))
        ctx.settings.set("history", ctx.state.history)
        ctx.settings.set("filters", vars(ctx.state.filters))
        ctx.settings.flush()

    def _persist_inker(self, ctx: Any) -> None:
        from . import inker_mode

        inker_mode.persist(ctx)
        ctx.settings.flush()

    def _persist_clay(self, ctx: Any) -> None:
        from . import clay_mode

        clay_mode.persist(ctx)
        ctx.settings.flush()


def _step(label: str, fn: Any) -> None:
    """Run one teardown stage; a failure is logged and the unwind continues."""
    try:
        fn()
    except Exception:
        log.exception("teardown: %s failed; continuing", label)


def _background() -> tuple[float, float, float, float]:
    from .theme import BG, rgba

    return rgba(BG)


def _setup_logging() -> None:
    """Console logging plus a rotating file log and a native crash log.

    Until this existed the app wrote no log file at all: basicConfig had only
    the default stream handler, so the VRAM instrumentation in queue.py went to
    a console nobody was watching. The 2026-08-03 memory-exhaustion crash left
    no in-app record whatsoever and had to be reconstructed from Windows event
    logs. Both files below exist so the next occurrence is attributable.

    faulthandler covers what logging cannot: a hard crash in native code
    (torch, CUDA, or the allocator giving up under commit exhaustion) never
    unwinds to a Python `except`, but faulthandler's signal handlers still get
    a traceback out to the fd.
    """
    import faulthandler
    from logging.handlers import RotatingFileHandler

    from ..config import get_config

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        data_dir = get_config().data_dir
        handlers.append(
            RotatingFileHandler(
                data_dir / "warlock.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
        )
        # Held open for process life on purpose -- faulthandler writes to this
        # fd from a signal handler at crash time, so it must not be a file
        # object that could be closed or garbage-collected first.
        global _crash_log
        _crash_log = (data_dir / "crash.log").open("a", encoding="utf-8")
        # Written before faulthandler is armed, so any dump below it is
        # attributable: crash.log is appended to across runs, and a bare
        # traceback with no session line above it belongs to nobody.
        _crash_log.write(
            f"=== session {_utc_now()} pid={os.getpid()} warlock={_version()} ===\n"
        )
        _crash_log.flush()
        faulthandler.enable(file=_crash_log)
    except OSError:
        # A read-only or missing data_dir is not a reason to refuse to start;
        # console logging alone is what we had before.
        logging.getLogger(__name__).warning("file logging unavailable", exc_info=True)

    # force=True is load-bearing, not defensive: cli.main() used to call
    # basicConfig() before dispatching here, which left the root logger with a
    # handler and made this call a silent no-op -- warlock.log was created on
    # every launch and never written to on the one path anybody actually uses.
    logging.basicConfig(
        level=os.environ.get("WARLOCK_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


_crash_log: Any = None

SESSION_MARKER = "session.marker"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _version() -> str:
    """The installed version, falling back to the packaged constant."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("warlock")
    except PackageNotFoundError:
        from .. import __version__

        return __version__


def _install_excepthooks() -> None:
    """Route every uncaught exception through logging before the default hook.

    The app path only. An exception escaping the frame loop went to stderr and
    nowhere else, which is precisely why the 2026-08-04 crash left an empty
    warlock.log; a daemon thread dying (``warlock-loop``, trellis' stdout
    reader) was even quieter, since nothing prints for those at all.
    """

    def _hook(exc_type, exc, tb):  # type: ignore[no-untyped-def]
        log.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    def _thread_hook(args):  # type: ignore[no-untyped-def]
        if issubclass(args.exc_type, SystemExit):
            return
        log.critical(
            "uncaught exception on thread %s",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def _unraisable(args):  # type: ignore[no-untyped-def]
        log.critical(
            "unraisable exception in %r",
            args.object,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _hook
    threading.excepthook = _thread_hook
    sys.unraisablehook = _unraisable


def _pid_alive(pid: int) -> bool:
    """Whether `pid` names a live process.

    Never ``os.kill(pid, 0)``: on Windows that signature terminates the target
    rather than probing it.
    """
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    from ctypes import wintypes

    _QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        return False


def _marker_path() -> Path:
    from ..config import get_config

    return get_config().data_dir / SESSION_MARKER


def _note_previous_session() -> None:
    """Say so in the log when the last session did not reach teardown.

    A crash that kills the process outright leaves no record of itself; it
    leaves this file behind instead. Never raises -- an unreadable data_dir is
    already handled by _setup_logging and must not block startup either.
    """
    try:
        raw = _marker_path().read_text(encoding="utf-8")
    except (OSError, ValueError):
        return
    try:
        data = json.loads(raw)
        pid = int(data.get("pid", 0))
    except (ValueError, TypeError, AttributeError):
        log.warning("previous session marker is unreadable: %r", raw[:200])
        return
    if _pid_alive(pid) and pid != os.getpid():
        log.warning(
            "another Warlock instance (pid %d, started %s) appears to be running; "
            "they share the job database and the trellis port",
            pid, data.get("started_at", "?"),
        )
        return
    log.warning(
        "the previous session (pid %d, started %s, warlock %s) did not shut down "
        "cleanly -- check crash.log and Windows event 2004",
        pid, data.get("started_at", "?"), data.get("version", "?"),
    )


def _write_session_marker() -> None:
    try:
        _marker_path().write_text(
            json.dumps({
                "pid": os.getpid(),
                "started_at": _utc_now(),
                "version": _version(),
            }),
            encoding="utf-8",
        )
    except OSError:
        log.warning("could not write the session marker", exc_info=True)


def _clear_session_marker() -> None:
    try:
        _marker_path().unlink(missing_ok=True)
    except OSError:
        log.warning("could not clear the session marker", exc_info=True)


def run() -> int:
    """The ``warlock`` entry point."""
    from .runtime import Runtime

    _setup_logging()
    _install_excepthooks()
    log.info(
        "Warlock Studio %s starting: pid=%d python=%s argv=%s",
        _version(), os.getpid(), sys.version.split()[0], sys.argv[1:],
    )
    _note_previous_session()
    _write_session_marker()
    try:
        return App(Runtime()).run()
    except Exception:
        # App.run reports and swallows its own failures, so anything arriving
        # here happened before the loop existed -- constructing the Runtime.
        log.exception("Warlock Studio could not start")
        return 1
    finally:
        _clear_session_marker()


if __name__ == "__main__":
    sys.exit(run())
