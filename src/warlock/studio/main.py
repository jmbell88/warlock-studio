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
DEFAULT_SIZE = (1600, 950)
MIN_SIZE = (1100, 700)
# Pane widths and the sidebar split now live in layout.Layout, persisted and
# draggable; the defaults there are the 340 / 0.55 this file used to hard-code.
TARGET_FPS = 60


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
        # A dead worker is reported once. The banner is dismissible, and
        # re-raising it every frame would make it impossible to dismiss.
        self._fatal_reported = False
        self._was_landing = True
        # Measured every frame, drawn only when state.show_fps is on (F10), and
        # logged once at teardown regardless -- the overlay answers "is it
        # smooth now", the log line is the evidence for "it ran at 60".
        self.fps = fps_mod.FpsMeter()
        # Set by _draw_viewport_image, read one frame later by _events. The
        # host window is fullscreen, so io.want_capture_mouse is always true
        # and cannot be the gate; imgui's own hover test on the viewport image
        # is, and it correctly goes false under popups and active widgets.
        self._viewport_hovered = False

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

        # The scale everything is drawn at: sampled from the monitor the
        # window actually opened on, before any font or style is built.
        tokens.set_scale(dpi.window_scale(pygame))
        self._min_size = (
            int(MIN_SIZE[0] * tokens.SCALE),
            int(MIN_SIZE[1] * tokens.SCALE),
        )

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
        state.mode = settings.get("mode") or "2d"
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
        ctx.guidance = svc_system.guidance_catalog()
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
        if failed:
            ctx.state.last_error = "; ".join(f"{c.name}: {c.detail}" for c in failed)

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
                ctx.toast(done.message or "That did not work.", "error")
                if done.key.startswith("inker-"):
                    from . import inker_mode

                    # A failed save must not leave the document locked: saving
                    # disables every editing control, so without this one bad
                    # write makes the tab read-only until it is closed.
                    inker_mode.on_task_failed(ctx, done)
                continue
            self._on_task_done(done)

    def _on_task_done(self, done: Any) -> None:
        ctx = self.app_ctx
        key = done.key
        if key == "preview" and isinstance(done.result, dict):
            ctx.state.preview.update(done.result)
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
        if key == "ref-upload" and done.result is not None:
            # Only the path is kept here; the bytes are read in the submit
            # task, so picking a 20 MB image never touches the frame thread.
            ctx.state.form_2d["ref_path"] = str(done.result)
            return
        if key.startswith("inker-"):
            from . import inker_mode

            inker_mode.on_task_done(ctx, done)
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
        if key.startswith("pose-"):
            if key.startswith("pose-save:") and self.viewer.pose_mode:
                # Only now is the pose actually on disk. A failed save leaves
                # the flag set, so the guard still stops the user walking away
                # from work that was never written.
                self.viewer.editor.dirty = False
            self._refresh_rig_side_data()
            ctx.cache.invalidate()

    def _refresh(self) -> None:
        from .jobs_cache import transition_message

        ctx = self.app_ctx

        def announce(job: Any, previous: str | None) -> None:
            message = transition_message(job, previous)
            if message is not None:
                ctx.toast(*message)
            if job["status"] == "done":
                self._request_storage()

        if ctx.cache.tick(announce):
            self._sync_viewer()
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
            ctx.state.last_error = f"The GPU worker stopped: {fatal}. Restart Warlock."
            ctx.toast("The GPU worker stopped. Nothing new will run.", "error")
        elif not self.runtime.alive:
            self._fatal_reported = True
            ctx.state.last_error = "The GPU worker is not running. Restart Warlock."
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
        """
        self.viewer.path = None
        self._sync_viewer()

    def _sync_viewer(self) -> None:
        """Show whatever the selection implies, when it changes.

        Driven off the cache rather than off the click so a job that finishes
        while it is selected starts showing its mesh without another click.
        """
        ctx = self.app_ctx
        if ctx.state.mode == "paint":
            # Paint owns the centre pane; there is no viewport to sync, and
            # loading a mesh for the selection would be work nothing shows.
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
        if wanted is None or self.viewer.path == wanted:
            return
        try:
            if wanted.suffix == ".png":
                self.viewer.clear()
                self.viewer.load_reference(wanted)
                self.viewer.path = wanted
            else:
                self.viewer.load_model(wanted)
                # The thumbnail is free here: the model is loaded and framed,
                # and a server-side render would need the serial GPU queue for
                # something purely cosmetic.
                if "thumb.png" not in files:
                    ctx.capture_thumbnail(job["id"])
        except Exception:
            log.exception("could not open %s", wanted)
            ctx.toast("Could not open that asset.", "error")
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
                pygame.display.set_mode(
                    (max(event.w, self._min_size[0]), max(event.h, self._min_size[1])),
                    pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
                )
                ctx.settings.set("window_size", [event.w, event.h])
                continue
            if event.type == pygame.DROPFILE:
                self._on_drop(Path(event.file))
                continue
            imgui_backend.process_event(event)
            if event.type in (pygame.KEYDOWN, pygame.KEYUP):
                if not io.want_text_input:
                    self._shortcut(event)
                continue
            # The viewer sees the mouse when it is over the viewport image, and
            # a drag already in progress keeps it wherever the cursor goes.
            if self._viewport_hovered or self.viewer._grab is not None:
                self.viewer.handle_event(event, hovered=self._viewport_hovered)

    def _shortcut(self, event: Any) -> None:
        import pygame

        ctx = self.app_ctx
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
            ctx.state.manual.open = True
            return
        # Above the landing and paint returns below: the frame rate is a
        # property of the loop, not of whichever pane happens to be on screen,
        # and the chooser is exactly where a slow startup would show.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F10:
            ctx.state.show_fps = not ctx.state.show_fps
            return
        if ctx.state.landing:
            # The chooser has no form to submit and no viewport to frame; every
            # one of these would act on a pane that is not on screen.
            return
        if ctx.state.mode == "paint":
            from . import inker_mode

            # Consumes every key while a painting is open, so F/W/S/Ctrl+Enter
            # cannot act on the panes Paint has replaced.
            if inker_mode.handle_key(ctx, event):
                return
        # Both edges are dispatched, because paint's space-to-pan is a hold and
        # needs the release. Nothing below is a hold: every one of these is a
        # toggle or an action, so acting on the release too undoes the toggle
        # the press just made and submits a second job for one Ctrl+Enter.
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
        if ctx.state.mode == "paint":
            from . import inker_mode

            ctx.state.landing = False
            inker_mode.open_path(ctx, path)
            return
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            ctx.toast("Drop an image to start a mesh from it.", "error")
            return
        if ctx.state.mode == "2d":
            # In the 2D pane a dropped image is a *conditioning reference*, not
            # a mesh to build -- forcing the mode switch here would throw away
            # the prompt the user is composing. One branch, and it is what
            # makes the feature discoverable at all.
            ctx.state.landing = False
            ctx.state.form_2d["ref_path"] = str(path)
            ctx.toast(f"Using {path.name} as the reference.")
            return
        ctx.state.mode = "3d"
        # A drop is a start: it would otherwise land behind the chooser, with
        # nothing on screen saying anything had happened.
        ctx.state.landing = False
        settings_3d.upload(ctx, path)

    def _request_quit(self) -> None:
        from . import inker_mode
        from .panes import pose_panel

        ctx = self.app_ctx
        inker_mode.guard(
            ctx, "quit", lambda: pose_panel.guard(ctx, "quit", self._quit)
        )

    def _quit(self) -> None:
        self._running = False

    # -- the UI ------------------------------------------------------------

    def _build_ui(self) -> None:
        from imgui_bundle import imgui

        from .panes import inspector, landing, library, settings_2d, settings_3d

        ctx = self.app_ctx
        # Recomputed every frame by whoever draws the viewport image. Landing,
        # 2D mode and the editor all return without drawing it, so it stays
        # false there and the viewer gets no events at all.
        self._viewport_hovered = False
        # Leaving the chooser is a mode change the cache will not announce: the
        # job list has not changed, so nothing else would ask the viewer to
        # show what was just picked.
        if self._was_landing and not ctx.state.landing:
            self._sync_viewer()
        self._was_landing = ctx.state.landing

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
        if ctx.state.landing:
            landing.draw(ctx)
            imgui.end()
            self._overlays(viewport)
            return
        self._mode_switch()
        from .panes import overlay

        overlay.doctor_banner(ctx)
        if ctx.state.mode == "paint":
            self._paint_workspace()
            imgui.end()
            self._overlays(viewport)
            return

        # The sidebar is two scrollers, not one: sharing a single scroll region
        # meant the settings form pushed the library off the bottom of a
        # 950-pixel window, which made the whole asset list unreachable.
        from . import layout as layout_mod
        from . import tokens
        from .tokens import sp

        lay = self.layout
        sidebar_w = sp(lay.sidebar_w)
        imgui.begin_group()
        avail_y = imgui.get_content_region_avail().y
        form_height = avail_y * lay.settings_share
        borders = imgui.ChildFlags_.borders.value
        if imgui.begin_child("settings", (sidebar_w, form_height), borders):
            if ctx.state.mode == "2d":
                settings_2d.draw(ctx)
            else:
                settings_3d.draw(ctx)
        imgui.end_child()
        drag = layout_mod.splitter("sidebar-share", vertical=False, length=sidebar_w)
        if drag and avail_y > 0:
            lay.settings_share = min(
                max(lay.settings_share + drag * tokens.SCALE / avail_y, layout_mod.SHARE_MIN),
                layout_mod.SHARE_MAX,
            )
            lay.save()
        if imgui.begin_child("library", (sidebar_w, 0), borders):
            library.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        drag = layout_mod.splitter("left-split")
        if drag:
            lay.sidebar_w = min(
                max(lay.sidebar_w + drag, layout_mod.SIDEBAR_MIN), layout_mod.SIDEBAR_MAX
            )
            lay.save()
        imgui.same_line()
        self._viewport_pane()
        imgui.same_line()
        drag = layout_mod.splitter("right-split")
        if drag:
            lay.inspector_w = min(
                max(lay.inspector_w - drag, layout_mod.SIDEBAR_MIN), layout_mod.SIDEBAR_MAX
            )
            lay.save()
        imgui.same_line()

        if imgui.begin_child("inspector", (0, 0), borders):
            inspector.draw(ctx)
        imgui.end_child()
        imgui.end()
        self._overlays(viewport)

    def _paint_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton the other modes use.

        Deliberately not a takeover of the whole window: the progress card
        floats over every mode, so a trellis run started before switching here
        is still visible while painting.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import (
            paint_bridge,
            paint_canvas,
            paint_colors,
            paint_layers,
            paint_tools,
        )
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = sp(lay.sidebar_w)
        inspector_w = sp(lay.inspector_w)
        style = imgui.get_style()
        borders = imgui.ChildFlags_.borders.value
        imgui.begin_group()
        tools_height = imgui.get_content_region_avail().y * lay.settings_share
        if imgui.begin_child("paint-tools", (sidebar_w, tools_height), borders):
            paint_tools.draw(ctx)
        imgui.end_child()
        if imgui.begin_child("paint-colors", (sidebar_w, 0), borders):
            paint_colors.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        drag = layout_mod.splitter("paint-left-split")
        if drag:
            lay.sidebar_w = min(
                max(lay.sidebar_w + drag, layout_mod.SIDEBAR_MIN), layout_mod.SIDEBAR_MAX
            )
            lay.save()
        imgui.same_line()
        reserved = inspector_w + sp(layout_mod.GRIP) + style.item_spacing.x * 2
        width = max(imgui.get_content_region_avail().x - reserved, sp(300))
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        if imgui.begin_child("paint-centre", (width, 0), borders, flags):
            paint_canvas.draw(ctx)
        imgui.end_child()

        imgui.same_line()
        drag = layout_mod.splitter("paint-right-split")
        if drag:
            lay.inspector_w = min(
                max(lay.inspector_w - drag, layout_mod.SIDEBAR_MIN), layout_mod.SIDEBAR_MAX
            )
            lay.save()
        imgui.same_line()
        imgui.begin_group()
        layers_height = imgui.get_content_region_avail().y * lay.settings_share
        if imgui.begin_child("paint-layers", (0, layers_height), borders):
            paint_layers.draw(ctx)
        imgui.end_child()
        if imgui.begin_child("paint-bridge", (0, 0), borders):
            paint_bridge.draw(ctx)
        imgui.end_child()
        imgui.end_group()

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
        if not ctx.state.landing:
            overlay.progress_card(ctx, self.eta)
        from .manual import render as manual_render

        manual_render.draw_window(ctx)
        widgets.toasts(ctx.state, (viewport.work_size.x, viewport.work_size.y))
        ctx.confirms.draw()
        ctx.prompts.draw()

    def _mode_switch(self) -> None:
        from imgui_bundle import imgui

        from . import icons, widgets

        ctx = self.app_ctx
        state = ctx.state
        # Neither Home nor a mode switch is destructive: Paint's documents are
        # still open when you come back, because it is a mode rather than a
        # takeover. Only quitting and closing a tab can lose pixels, and both
        # ask.
        if imgui.button(f"{icons.HOUSE} Home"):
            state.landing = True
            state.landing_view = "choose"
        imgui.same_line()
        selected = widgets.segmented_control(
            "mode-seg",
            [("2d", "2D reference"), ("3d", "3D asset"), ("paint", "Paint")],
            state.mode,
        )
        if selected != state.mode:
            state.mode = selected
            ctx.settings.set("mode", selected)
            self._sync_viewer()

        # Right-aligned health dot: green when everything passed, amber when a
        # non-fatal check failed, red for a fatal one or a dead worker. Click
        # for the full diagnostics list -- the non-fatal checks (missing
        # weights, gltfpack, CUDA) used to be visible only in the log file.
        from . import theme
        from .tokens import sp

        checks = list(getattr(ctx.runtime, "checks", []) or [])
        if state.last_error is not None or any(c.fatal and not c.ok for c in checks):
            colour = theme.ERR
        elif any(not c.ok for c in checks):
            colour = theme.WARN
        else:
            colour = theme.OK
        from . import widgets

        imgui.same_line(max(imgui.get_window_width() - sp(100), 0))
        if widgets.icon_button(icons.INFO, "Manual (F1)"):
            ctx.state.manual.open = True
        imgui.same_line()
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
    # tuples are (keys, what), grouped; paint's letters come from TOOL_KEYS so
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
                ("F1", "Open the manual"),
                ("F10", "Toggle the frame-rate readout"),
                ("Ctrl+Enter", "Generate / Make 3D"),
                ("F", "Frame the model"),
                ("W", "Toggle wireframe"),
                ("S", "Toggle turntable"),
                ("Esc", "Exit comparison / pose edit"),
            ],
        )
        from .inker_mode import TOOL_KEYS

        tools = ", ".join(f"{k.upper()}" for k in sorted(TOOL_KEYS))
        table(
            "Paint",
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
            ctx.submit("open-log", os.startfile, str(log_path))
        imgui.end_popup()

    def _viewport_pane(self) -> None:
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import overlay
        from .tokens import sp

        ctx = self.app_ctx
        style = imgui.get_style()
        # Leave room for the right splitter and the inspector; the progress
        # card floats over the image now, so the full height is the image's.
        reserved = sp(self.layout.inspector_w) + sp(layout_mod.GRIP) + style.item_spacing.x * 2
        width = max(imgui.get_content_region_avail().x - reserved, sp(300))
        # no_scroll_with_mouse: over the viewport the wheel can only mean dolly.
        if imgui.begin_child(
            "viewport",
            (width, 0),
            imgui.ChildFlags_.borders.value,
            imgui.WindowFlags_.no_scroll_with_mouse.value,
        ):
            overlay.toolbar(ctx)
            image_pos = imgui.get_cursor_screen_pos()
            avail = imgui.get_content_region_avail()
            height = max(avail.y, 64)
            if ctx.state.mode == "3d" and self.viewer.has_model:
                self._draw_viewport_image(image_pos, width, height)
            elif self.viewer.reference is not None:
                self._draw_reference(width, height)
            else:
                overlay.placeholder(ctx)
        imgui.end_child()

    def _draw_viewport_image(self, pos: Any, width: float, height: float) -> None:
        from imgui_bundle import imgui

        from . import widgets

        ctx = self.app_ctx
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
            _step("persist paint", lambda: self._persist_paint(ctx))
            if ctx.textures is not None:
                _step("release textures", ctx.textures.release)
        if self.viewer is not None:
            _step("release viewer", self.viewer.release)
        if self.imgui_renderer is not None:
            _step("shutdown imgui", self.imgui_renderer.shutdown)
        _step("pygame.quit", pygame.quit)
        _step("runtime shutdown", self.runtime.shutdown)
        # The line whose *absence* is evidence: a session that ends without it
        # died somewhere no `except` could see.
        log.info("teardown complete")

    def _persist(self, ctx: Any) -> None:
        from .settings import sanitise_form

        ctx.settings.set("mode", ctx.state.mode)
        ctx.settings.set("show_fps", ctx.state.show_fps)
        ctx.settings.set("form_2d", sanitise_form(ctx.state.form_2d))
        ctx.settings.set("form_3d", sanitise_form(ctx.state.form_3d))
        ctx.settings.set("history", ctx.state.history)
        ctx.settings.set("filters", vars(ctx.state.filters))
        ctx.settings.flush()

    def _persist_paint(self, ctx: Any) -> None:
        from . import inker_mode

        inker_mode.persist(ctx)
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
