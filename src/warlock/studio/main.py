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

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

WINDOW_TITLE = "Warlock Studio"
DEFAULT_SIZE = (1600, 950)
MIN_SIZE = (1100, 700)
SIDEBAR_WIDTH = 340.0
# How much of the sidebar the settings form gets before the library starts.
# The library is the thing a user scrolls, so it keeps the larger share of a
# short window rather than the form.
SETTINGS_SHARE = 0.55
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
        self._last_frame = time.perf_counter()

    # -- setup -------------------------------------------------------------

    def setup(self) -> None:
        import moderngl
        import pygame
        from imgui_bundle import imgui

        from . import imgui_backend, textures, theme
        from .app_ctx import Ctx
        from .jobs_cache import JobsCache
        from .settings import Settings, restore_form
        from .state import DEFAULT_FORM_3D, AppState, Eta, Filters, default_form_2d
        from .viewer_embed import Viewer

        self.svc = self.runtime.start()
        settings = Settings.load(self.runtime.config.data_dir)

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        size = tuple(settings.get("window_size") or DEFAULT_SIZE)
        self.window = pygame.display.set_mode(
            size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption(WINDOW_TITLE)
        # Dropped files are how a reference image gets in without a dialog.
        pygame.event.set_allowed(None)

        self.ctx = moderngl.create_context()
        imgui.create_context()
        imgui.get_io().set_ini_filename("")  # imgui's own layout file is not ours to keep
        theme.apply(imgui)
        self.imgui_renderer = imgui_backend.ImguiRenderer(self.ctx)
        self.viewer = Viewer(self.ctx)

        state = AppState()
        state.mode = settings.get("mode") or "2d"
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
        ctx.base_models = [(k, spec.label) for k, spec in models.BASE_MODELS.items()]
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
        failed = [c for c in self.runtime.checks if not c.ok and c.fatal]
        if failed:
            ctx.state.last_error = "; ".join(f"{c.name}: {c.detail}" for c in failed)

    # -- the loop ----------------------------------------------------------

    def run(self) -> int:
        import pygame

        self.setup()
        self._running = True
        clock = pygame.time.Clock()
        try:
            while self._running:
                dt = self._tick()
                self.frame(dt)
                pygame.display.flip()
                clock.tick(TARGET_FPS)
        finally:
            self.teardown()
        return 0

    def _tick(self) -> float:
        now = time.perf_counter()
        dt = min(now - self._last_frame, 0.25)
        self._last_frame = now
        return dt

    def frame(self, dt: float) -> None:
        from imgui_bundle import imgui

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
                continue
            self._on_task_done(done)

    def _on_task_done(self, done: Any) -> None:
        ctx = self.app_ctx
        key = done.key
        if key == "preview" and isinstance(done.result, dict):
            ctx.state.preview.update(done.result)
            return
        if key == "upload" and done.result is not None:
            from .panes import settings_3d

            settings_3d.upload(ctx, Path(done.result))
            return
        if key == "submit":
            ctx.cache.invalidate()
            ctx.toast("Queued.")
            return
        if key.startswith("save:") or key.startswith("bake:") or key.startswith("sheet-save:"):
            if done.result is not None:
                ctx.toast(f"Saved to {done.result}")
            return
        if key.startswith(("delete:", "prune", "rename:", "name:", "tags:", "fav:")):
            ctx.cache.invalidate()
            if key.startswith(("delete:", "prune")):
                ctx.cache.refresh_storage()
            return
        if key.startswith(("cancel:", "rerun:", "remesh:", "retry:", "rig:", "joints:", "sheet:")):
            ctx.cache.invalidate()
            return
        if key.startswith("pose-"):
            self._refresh_rig_side_data(force=True)
            ctx.cache.invalidate()

    def _refresh(self) -> None:
        from .jobs_cache import transition_message

        ctx = self.app_ctx

        def announce(job: Any, previous: str | None) -> None:
            message = transition_message(job, previous)
            if message is not None:
                ctx.toast(*message)
            if job["status"] == "done":
                ctx.cache.refresh_storage()

        if ctx.cache.tick(announce):
            self._sync_viewer()

    def _sync_viewer(self) -> None:
        """Show whatever the selection implies, when it changes.

        Driven off the cache rather than off the click so a job that finishes
        while it is selected starts showing its mesh without another click.
        """
        ctx = self.app_ctx
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

    def _refresh_rig_side_data(self, force: bool = False) -> None:
        """Poses and sheets for the selected job, off-thread."""
        from ..service import rig as svc_rig
        from ..service import sheets as svc_sheets

        ctx = self.app_ctx
        job = ctx.job()
        if job is None:
            return
        job_id = job["id"]
        if "rig.glb" in (job.get("files") or []):
            ctx.submit(f"poses:{job_id}", svc_rig.list_poses, ctx.svc, job_id)
        ctx.submit(f"sheets:{job_id}", svc_sheets.list_sheets, ctx.svc, job_id)
        del force

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
                    (max(event.w, MIN_SIZE[0]), max(event.h, MIN_SIZE[1])),
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
            # The viewer only sees the mouse when imgui does not want it, and a
            # drag already in progress keeps it whatever imgui says.
            if not io.want_capture_mouse or self.viewer._grab is not None:
                self.viewer.handle_event(event, hovered=self._over_viewport())

    def _over_viewport(self) -> bool:
        import pygame

        x, y = pygame.mouse.get_pos()
        rx, ry, rw, rh = self.viewer._rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def _shortcut(self, event: Any) -> None:
        import pygame

        if event.type != pygame.KEYDOWN:
            return
        ctx = self.app_ctx
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
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            ctx.toast("Drop an image to start a mesh from it.", "error")
            return
        ctx.state.mode = "3d"
        settings_3d.upload(ctx, path)

    def _request_quit(self) -> None:
        from .panes import pose_panel

        pose_panel.guard(self.app_ctx, "quit", self._quit)

    def _quit(self) -> None:
        self._running = False

    # -- the UI ------------------------------------------------------------

    def _build_ui(self) -> None:
        from imgui_bundle import imgui

        from .panes import inspector, library, settings_2d, settings_3d

        ctx = self.app_ctx
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
        self._mode_switch()

        # The sidebar is two scrollers, not one: sharing a single scroll region
        # meant the settings form pushed the library off the bottom of a
        # 950-pixel window, which made the whole asset list unreachable.
        imgui.begin_group()
        form_height = imgui.get_content_region_avail().y * SETTINGS_SHARE
        borders = imgui.ChildFlags_.borders.value
        if imgui.begin_child("settings", (SIDEBAR_WIDTH, form_height), borders):
            if ctx.state.mode == "2d":
                settings_2d.draw(ctx)
            else:
                settings_3d.draw(ctx)
        imgui.end_child()
        if imgui.begin_child("library", (SIDEBAR_WIDTH, 0), borders):
            library.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        self._viewport_pane()
        imgui.same_line()

        if imgui.begin_child("inspector", (SIDEBAR_WIDTH, 0), borders):
            inspector.draw(ctx)
        imgui.end_child()
        imgui.end()

        overlay_size = (viewport.work_size.x, viewport.work_size.y)
        from . import widgets

        widgets.toasts(ctx.state, overlay_size)
        ctx.confirms.draw()
        ctx.prompts.draw()

    def _mode_switch(self) -> None:
        from imgui_bundle import imgui

        state = self.app_ctx.state
        for mode, label in (("2d", "2D reference"), ("3d", "3D asset")):
            if imgui.radio_button(label, state.mode == mode):
                state.mode = mode
                self.app_ctx.settings.set("mode", mode)
                self._sync_viewer()
            imgui.same_line()
        imgui.new_line()

    def _viewport_pane(self) -> None:
        from imgui_bundle import imgui

        from .panes import overlay

        ctx = self.app_ctx
        width = imgui.get_content_region_avail().x - SIDEBAR_WIDTH - 16
        if imgui.begin_child("viewport", (width, 0), imgui.ChildFlags_.borders.value):
            overlay.doctor_banner(ctx)
            overlay.toolbar(ctx)
            image_pos = imgui.get_cursor_screen_pos()
            avail = imgui.get_content_region_avail()
            height = max(avail.y - 100, 64)
            if ctx.state.mode == "3d" and self.viewer.has_model:
                self._draw_viewport_image(image_pos, width, height)
            elif self.viewer.reference is not None:
                self._draw_reference(width, height)
            else:
                overlay.placeholder(ctx)
            overlay.progress_card(ctx, self.eta)
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
        if halves == 2 and self.viewer.compare_viewport is not None:
            imgui.same_line()
            imgui.image(
                widgets.texture_ref(self.viewer.compare_viewport.texture),
                (cell, height), (0, 1), (1, 0),
            )

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
        import pygame

        ctx = self.app_ctx
        if ctx is not None:
            from .settings import sanitise_form

            ctx.settings.set("mode", ctx.state.mode)
            ctx.settings.set("form_2d", sanitise_form(ctx.state.form_2d))
            ctx.settings.set("form_3d", sanitise_form(ctx.state.form_3d))
            ctx.settings.set("history", ctx.state.history)
            ctx.settings.set("filters", vars(ctx.state.filters))
            ctx.settings.flush()
            if ctx.textures is not None:
                ctx.textures.release()
        if self.viewer is not None:
            self.viewer.release()
        if self.imgui_renderer is not None:
            self.imgui_renderer.shutdown()
        pygame.quit()
        self.runtime.shutdown()


def _background() -> tuple[float, float, float, float]:
    from .theme import BG, rgba

    return rgba(BG)


def run() -> int:
    """The ``warlock`` entry point."""
    from .runtime import Runtime

    logging.basicConfig(
        level=os.environ.get("WARLOCK_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return App(Runtime()).run()
    except Exception:
        log.exception("Warlock Studio could not start")
        return 1


if __name__ == "__main__":
    sys.exit(run())
