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
from . import filetypes
from . import fps as fps_mod

log = logging.getLogger(__name__)

WINDOW_TITLE = "Warlock Studio"
# How often the frame loop samples host memory. Long enough to be free, short
# enough that a 30-minute idle session yields 60 points to fit a slope through.
MEMORY_TICK_SECONDS = 30.0
# How often the diagnostics popup re-stats the log file. Long enough that the
# popup costs no syscall per frame, short enough that the button ungreys within
# a second of the first line being written.
LOG_STAT_SECONDS = 1.0
# The task key the selection's GLB is parsed under. One key, so a selection
# moving faster than the disk cannot pile up loads: a refused submit is simply
# retried on the next tick, and a landed result is checked against
# ``viewer.pending`` before it is adopted.
VIEWER_KEY = "viewer-load"
DEFAULT_SIZE = (1600, 950)
MIN_SIZE = (1100, 700)
# Pane widths live in layout.py and are *fixed*, not draggable: three named
# sidebar widths (260 / 300 / 360 design px, ``layout.SIDEBAR_WIDTHS``) with
# ``default`` in force unless the settings file says otherwise. The one
# proportion the user still drags is the sidebar's internal split,
# ``Layout.settings_share``, which defaults to 0.55.
TARGET_FPS = 60
# The Ctrl chords a focused text field owns, spelled as ``pygame.key.name``
# gives them. imgui's own input-text widget binds these to editing the text, so
# they are the one class of modifier chord that must *not* reach the global
# shortcuts while a field has focus. See ``Studio._passes_text_field``.
_TEXT_FIELD_CTRL = frozenset({"z", "y", "x", "c", "v", "a"})
# Named rather than spelled as ``pygame.K_F*`` because this module imports
# pygame lazily, inside the loop, and a module-level constant table would drag
# the window library into every import of it.
_FUNCTION_KEYS = frozenset(f"f{n}" for n in range(1, 13))
# The redraw rate while nothing on screen can change (B11): no pending input,
# no job, no toast, no task, cameras settled, nothing playing. Fast enough
# that the first frame after a wake-up condition is never far away, slow
# enough that an idle session stops burning a core and the GPU.
IDLE_FPS = 12
# The modes that fill the host window with one pane. Inker, Clay, Review,
# Plotter and Packwright are not here: each fills it with a three-column
# *workspace* instead, which is ``modes.WORKSPACE_MODES``. Those three
# categories partition ``modes.KEYS`` exactly (thirteen modes today), and the
# partition is the guard on ``_build_ui``'s dispatch.
_SINGLE_PANE_MODES = ("home", "manual", "settings", "library", "profiles")


# What a drop onto the window is allowed to be. The refusal message and every
# accept path have to agree about it (H71) -- and so do the file pickers, which
# is why the list itself lives in ``filetypes`` and this is a name for it
# rather than a copy of it.
DROPPABLE_IMAGES = filetypes.IMAGE_SUFFIXES


def _ago(seconds: float) -> str:
    """``12s`` / ``4m`` / ``2h``, for the notification history (H67).

    Coarse on purpose: the question it answers is "was that the one from just
    now", and a figure with more precision than that invites reading it as a
    measurement.
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{int(seconds // 3600)}h"


# The two image-labelling passes, named as the questions they are. Wording is the
# feature here: the same PNG is a *product* in 2D mode and a *blank* on the way to
# trellis, and "good" means opposite things -- a dramatic plate with pillars and a
# cast shadow is a better asset and a worse blank. A reviewer who cannot tell
# which question is on screen labels the average of the two.
_LABEL_TITLES = {
    "reference": "Label: good 2D asset?",
    "blank": "Label: good to reconstruct?",
}
_LABEL_QUESTIONS = {
    "reference": "Judge it as the finished picture: composition, style, drama.",
    "blank": "Judge it as input for the mesh: one subject, plain background, neutral pose.",
}


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


def filter_shortcuts(
    sections: list[tuple[str, list[tuple[str, str]]]], query: str
) -> list[tuple[str, list[tuple[str, str]]]]:
    """The shortcut list narrowed by ``query`` (UX.md Phase 4).

    Pure, and through ``palette.match`` rather than a substring test, because
    the popup and the command palette are two lists of the same kind of thing
    and a second matcher is a second answer to "does 'ctz' find Ctrl+Z".

    A row matches on its keys, its description **or its group's name**, which is
    the rule that makes "clay" list Clay's fifteen bindings rather than the two
    whose text happens to say the word. A group whose heading matched keeps all
    of its rows for the same reason; a group with no surviving row is dropped
    entirely, because a heading over nothing reads as a section that broke.

    Rows are deliberately *not* re-ordered by score: within a group they are in
    a hand-chosen order, and a filter that also reshuffles is two changes to
    read at once.
    """
    from . import palette

    if not query.strip():
        return list(sections)
    out = []
    for title, rows in sections:
        if palette.match(query, title) is not None:
            out.append((title, list(rows)))
            continue
        kept = [row for row in rows if palette.match(query, f"{row[0]} {row[1]}") is not None]
        if kept:
            out.append((title, kept))
    return out


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
        # Read in setup_window (the window size is in it) and consumed in
        # setup_context, which the splash now runs between.
        self.settings = None
        self._monitor_scale = 1.0
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
        # How long the veil over the whole viewport takes to clear, or 0.0 for
        # "there is no transition running". A duration rather than a bool
        # because the two things that raise one want different lengths: a mode
        # switch is DUR_BASE, the splash dissolving into the app is DUR_SLOW.
        self._transition_duration = 0.0
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
        # Which Clay tab the one viewport camera currently belongs to. See
        # ``_clay_viewport``: the camera is per document and the viewport is not.
        self._clay_camera_tab = ""
        # Clay's own hover flag, set by the pane that draws its image, for the
        # reason _viewport_hovered exists: the host window is fullscreen, so
        # io.want_capture_mouse is always true and cannot be the gate.
        self._build_hovered = False
        # Poser's own Viewer and hover flag, for Clay's reasons: built on first
        # entry (a session that never poses pays for no second renderer), and
        # a separate instance so loading the armature preview can never call
        # ``adopt_model``'s unconditional ``exit_pose_mode`` over an inspector
        # pose session on the shared viewer.
        self.poser_viewer = None
        self._poser_hovered = False
        # What is typed into the shortcuts popup's own filter box (UX.md Phase
        # 4). Not persisted and cleared on every open: it is a way through
        # sixty rows, not a preference about them.
        self._shortcuts_query = ""

    # -- setup -------------------------------------------------------------

    # There is deliberately no ``setup()`` composing the three phases. It
    # existed briefly and had no caller: ``run`` drives the phases itself so it
    # can draw a splash over the middle one. What it did have was users in the
    # tests, which stubbed ``app.setup`` to isolate ``run`` -- and once ``run``
    # stopped calling it those stubs silently became no-ops, so ``run`` fell
    # through into the real ``setup_window``, initialised pygame for real, and
    # left a live display behind that broke every GL test that ran after it (74
    # errors, none of them in the code that changed). A convenience method with
    # no caller is not free; this one cost a seam the tests were relying on.

    def setup_window(self) -> None:
        """Everything that needs the main thread and the one GL context.

        Fast, and first: this used to run *after* ``runtime.start()``, so the
        slow half of startup -- doctor, the worker, the out-of-process bpy
        probe -- happened with no window on screen at all. Splitting it out is
        what lets the splash be drawn over the rest.
        """
        import moderngl
        import pygame
        from imgui_bundle import imgui

        from . import dpi, fonts, imgui_backend, theme, tokens, widgets
        from .layout import Layout
        from .settings import Settings
        from .viewer_embed import Viewer

        # Read before the runtime exists: it is a file under the configured
        # data directory, which the Config already knows, and the window size
        # it carries is needed by set_mode below.
        settings = Settings.load(self.runtime.config.data_dir)
        self.settings = settings

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
        # settings pane rescales everything immediately and re-bakes the atlas
        # between frames (K99).
        monitor_scale = dpi.window_scale(pygame)
        tokens.set_scale(monitor_scale * _ui_scale(settings))
        # Before ``theme.apply`` below, which copies the palette into imgui's
        # style (M105). An unknown stored name falls back to dark rather than
        # raising: a settings file written by a build with a third palette must
        # not stop the window opening.
        tokens.set_theme(str(settings.get("theme") or "dark"))
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
        self._monitor_scale = monitor_scale

    def setup_runtime(self, note: Any = None) -> None:
        """The slow half, run on a plain worker thread behind the splash.

        Nothing here touches GL or imgui: it opens the store, runs the doctor
        checks, starts the worker's loop thread and probes bpy in a
        subprocess. ``Ctx`` is deliberately *not* built here -- it constructs
        textures, and textures belong to the frame thread's one context.

        ``note`` is the splash's line, and it is optional because this is also
        called with nothing listening -- by the tests, and by any caller that
        starts a runtime without a window.
        """
        self.svc = self.runtime.start(note)

    def setup_context(self) -> None:
        """The Ctx and the state it carries. Frame thread only, after both."""
        from . import effects, motion, textures
        from .app_ctx import Ctx
        from .jobs_cache import JobsCache
        from .settings import restore_form
        from .state import (
            DEFAULT_FORM_3D,
            AppState,
            Eta,
            default_form_2d,
            filters_from_stored,
        )

        settings = self.settings
        monitor_scale = self._monitor_scale

        state = AppState()
        # No mode restore, and nothing writes one either: the app opens on Home
        # every launch (AppState's default), so a stored mode would be a key
        # with no reader that four call sites kept half-updated.
        state.show_fps = bool(settings.get("show_fps"))
        # Both halves, here rather than at the checkbox: the stored value has to
        # reach ``motion.REDUCED`` before the first frame is built, or the app
        # animates its own startup at somebody who asked it not to.
        state.reduce_motion = bool(settings.get("reduce_motion"))
        motion.set_reduced(state.reduce_motion)
        # The four Phase 5 switches, for the same reason and at the same
        # moment: they are read on the first frame, and a flag applied after it
        # would show one frame of an effect somebody turned off.
        effects.load(settings)
        state.form_2d = restore_form(default_form_2d(), settings.get("form_2d"))
        state.form_3d = restore_form(DEFAULT_FORM_3D, settings.get("form_3d"))
        state.history = list(settings.get("history") or [])
        # The filter bar, minus the fields that are views rather than filters:
        # the app always opens on the library, never in the trash. See
        # ``state.VOLATILE_FILTERS``.
        state.filters = filters_from_stored(settings.get("filters"))

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
        # Off the frame thread (C32): the walk stats every file under every
        # job directory, and nothing on the first frame needs the number --
        # the library's storage line simply appears when the task lands.
        self._request_storage()
        self.viewer.on_pose_dirty = self._on_pose_dirty

    def _load_static_answers(self) -> None:
        """Read the things that cannot change without a restart, once."""
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
        self._refresh_model_answers()
        # Off the frame thread: rig_templates asks doctor.blender_check, whose
        # first answer is a seconds-long bpy subprocess probe that no longer
        # runs during startup (C30). The rig controls appear when it lands.
        ctx.rig_default = self.runtime.config.rig_template or ""
        ctx.submit("rig-templates", svc_rig.rig_templates, self.svc)
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

    def _refresh_model_answers(self) -> None:
        """What the app knows about the weights on disk, recomputed from doctor.

        Called at startup and again whenever a download finishes. It used to run
        only once, and the ctx field it writes was documented as immutable --
        which was true only while nothing in the app could make weights appear.
        The generate combos read ``base_models`` every frame, so a model
        downloaded from the Settings pane has to stop saying "weights missing"
        without a restart.
        """
        from .. import fetch, models, vram
        from ..service import downloads as svc_downloads

        ctx = self.app_ctx
        # Marked rather than hidden when weights are absent: the combo listing
        # every registered model regardless meant picking one whose weights
        # were never downloaded and learning at job-failure time, despite
        # doctor knowing at startup.
        # The prefix comes from fetch.CHECK_PREFIXES rather than a literal:
        # doctor composes the row name through the same table, so a prefix
        # spelled twice would go on matching nothing and mark every downloaded
        # base model present forever.
        prefix = fetch.CHECK_PREFIXES["base"]
        missing = {
            check.name.removeprefix(prefix)
            for check in (self.runtime.checks or [])
            if check.name.startswith(prefix) and not check.ok
        }
        # And a second suffix, for the other reason a listed model cannot run.
        # Computed from the spec through ``vram.fits`` rather than by matching
        # more doctor strings: doctor has nothing to say about VRAM per model,
        # and growing the string-matching above into a second question is how
        # the prefix bug this block's comment describes happened the first time.
        # Guarded on the plan, so a host with no resolved budget sees no badge.
        plan_ = getattr(self.svc, "vram_plan", None)

        def _suffix(spec: Any) -> str:
            if spec.label in missing:
                return " - weights missing"
            if plan_ is not None and vram.fits(plan_, spec) == vram.FIT_NO:
                return " - won't fit this GPU"
            return ""

        # The 2-tuple shape is pinned by the smoke tests and read by every
        # combo: label decoration only, never a third element.
        ctx.base_models = [
            (k, f"{spec.label}{_suffix(spec)}") for k, spec in models.BASE_MODELS.items()
        ]
        ctx.style_loras = [("", "no style LoRA")] + [
            (k, spec.label) for k, spec in models.STYLE_LORAS.items()
        ]
        # The Settings pane draws this and may not ask the service itself: it
        # is a pane, and ``recommended_base`` needs a resolved Plan. Empty when
        # there is no plan, which is the pane's "say nothing" value.
        ctx.recommended_base_label = (
            models.BASE_MODELS[vram.recommended_base(plan_)].label
            if plan_ is not None
            else ""
        )
        try:
            ctx.model_rows = svc_downloads.rows(self.svc)
        except Exception:
            # A settings pane that cannot list its rows is not a reason to fail
            # startup, the same posture the rig-template probe below takes.
            log.exception("could not list the downloadable models")
            ctx.model_rows = []

    # -- the loop ----------------------------------------------------------

    def run(self) -> int:
        import pygame

        # Setup is inside the try: it starts the runtime, so a failure past
        # that point used to skip teardown and leave the store, the loop
        # thread and the worker running.
        #
        # The two phases are reported differently on purpose. A window that
        # never appeared and a window that vanished after twenty minutes are
        # different bugs, and the log line was the only thing that could tell
        # them apart -- when there was one at all. All three setup phases are
        # the *first* of those, splash or no splash: the window being up is
        # not the app being up, and a failure to build the Ctx is still a
        # startup failure.
        in_setup = True
        rc = 0
        try:
            self.setup_window()
            if self._startup_with_splash():
                self.setup_context()
                in_setup = False
                # The splash dissolves into the app rather than cutting to it
                # (UX.md Phase 1). The same veil a mode switch uses, at the
                # longer duration: the splash's last frame and the app's first
                # are both on the window background, so a veil clearing over
                # the app *is* the crossfade between them -- and the cheap half
                # of one is indistinguishable from the whole against near-black.
                from . import tokens as tokens_mod

                self._start_transition(tokens_mod.DUR_SLOW)
                self._running = True
                clock = pygame.time.Clock()
                while self._running:
                    if self._skip_idle_frame():
                        # Nothing can change on screen and the idle cadence is
                        # not due yet: sleep one 60 Hz tick without drawing.
                        # Events are only *peeked* here, so none is lost -- the
                        # frame that consumes them runs the moment one arrives.
                        clock.tick(TARGET_FPS)
                        continue
                    dt = self._tick()
                    self.frame(dt)
                    pygame.display.flip()
                    clock.tick(TARGET_FPS)
            else:
                # Closed during the splash. The load was waited out rather
                # than abandoned, so teardown unwinds a whole runtime.
                log.info("closed during startup")
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

    def _startup_with_splash(self) -> bool:
        """Draw the logo while ``setup_runtime`` runs. -> keep going?

        The window is up by now, which is the point and also the new risk: the
        X button is live, so a quit has to be handled here, before there is a
        ``Ctx``, a job cache or anything else the ordinary quit path talks to.
        It is honoured by *waiting* -- see ``splash.Startup`` -- because
        abandoning a half-started runtime strands whatever it had already
        opened, and then returning False so ``run`` skips straight to teardown.

        A load that raised is re-raised here rather than reported, so it lands
        in ``run``'s existing "could not start" branch with its own traceback.
        """
        import pygame
        from imgui_bundle import imgui

        from . import imgui_backend, splash

        started = splash.Startup(lambda: self.setup_runtime(started.note))
        started.start()
        splash.begin_fade()
        logo = splash.load_logo(self.ctx)
        clock = pygame.time.Clock()
        try:
            while not started.finished():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        started.request_quit()
                        continue
                    if event.type == pygame.VIDEORESIZE:
                        # Not persisted: settings are written at teardown from
                        # the Ctx that does not exist yet, and a resize during
                        # a three-second splash is not a preference.
                        sized = (
                            max(event.w, self._min_size[0]),
                            max(event.h, self._min_size[1]),
                        )
                        pygame.display.set_mode(
                            sized, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
                        )
                        continue
                    imgui_backend.process_event(event)
                io = imgui.get_io()
                io.delta_time = 1.0 / TARGET_FPS
                size = pygame.display.get_window_size()
                io.display_size = size
                io.display_framebuffer_scale = (1.0, 1.0)
                imgui.new_frame()
                # The load's own words rather than one fixed sentence (UX.md
                # Phase 4): the hold is at least three seconds and up to ten on
                # a cold start, and "Starting Warlock Studio..." spends all of
                # it saying nothing that was not already obvious from the logo.
                splash.draw(logo, size, started.message)
                imgui.render()
                self.ctx.screen.use()
                self.ctx.clear(*_background())
                self.imgui_renderer.render(imgui.get_draw_data())
                pygame.display.flip()
                clock.tick(TARGET_FPS)
        finally:
            # Two megabytes of decoded pixels, and the backend still holds the
            # object under its GL name -- forget it before the release, or the
            # next texture to be handed that name renders as this logo.
            splash.release_logo(logo, self.imgui_renderer)
        started.raise_if_failed()
        return not started.quit_requested

    def _skip_idle_frame(self) -> bool:
        """Whether this loop pass may go by without a redraw (B11).

        Gates the *whole* frame -- events, cache tick, UI build, render -- on
        whether anything could visibly change. Any pending input renders
        immediately (the events are peeked, never consumed); otherwise a frame
        is due at IDLE_FPS whenever something live is on screen, and the rest
        of the time only at IDLE_FPS anyway -- the conservative shape: being
        wrong about "idle" costs at most 1/IDLE_FPS of latency, never an
        event.
        """
        if time.perf_counter() - self._last_frame >= 1.0 / IDLE_FPS:
            return False
        return not self._frame_active()

    def _frame_active(self) -> bool:
        """Anything that wants the full TARGET_FPS cadence right now."""
        import pygame
        from imgui_bundle import imgui

        if pygame.event.peek():
            return True
        ctx = self.app_ctx
        if ctx is None:
            return True
        io = imgui.get_io()
        if io.want_text_input:
            return True  # the caret blinks
        # An animation in flight is a reason the screen can change with no
        # input at all -- which is exactly what the rest of this list
        # enumerates. It counts only keys the last frame actually touched, so a
        # widget that left the screen mid-move cannot hold the app at 60 fps
        # (``motion.animating``), and it is constantly false under
        # reduce-motion, where nothing is ever mid-move.
        from . import motion

        if motion.animating():
            return True
        state = ctx.state
        if state.toasts:
            return True  # TTL fade
        # A job running or queued animates the progress card and its easing.
        if self.runtime.current_job_id is not None or ctx.cache.active is not None:
            return True
        # Any task in flight draws spinners/progress somewhere.
        if ctx.tasks.busy_keys:
            return True
        viewer = self.viewer
        if viewer is not None and (
            viewer.pending is not None
            or viewer.stripping
            or viewer.camera.auto_rotate
            or not viewer.camera.settled()
        ):
            return True
        clay = self.clay_view
        if clay is not None and state.mode == "clay" and not clay.camera.settled():
            return True
        poser = self.poser_viewer
        if poser is not None and state.mode == "poser" and not poser.camera.settled():
            return True
        inker = state.inker
        tab = None if inker is None else inker.active
        return tab is not None and bool(getattr(tab, "playing", False))

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
        # K99, and the position is the whole of it: rebuilding the atlas
        # invalidates every ImFont handle, and those are pushed and popped all
        # through ``_build_ui``. Between frames is the only safe moment, so the
        # scale slider sets a flag and this consumes it.
        if self.app_ctx.state.fonts_dirty:
            from . import fonts

            self.app_ctx.state.fonts_dirty = False
            fonts.reload(imgui)
        imgui.new_frame()
        self._build_ui()
        imgui.render()

        self.ctx.screen.use()
        self.ctx.clear(*_background())
        self.imgui_renderer.render(imgui.get_draw_data())
        # After the render and before the flip: what is on the default
        # framebuffer now *is* the composed frame, which is the whole reason
        # the translucent surfaces sample a captured frame rather than asking
        # for the draw list to be split in two (UX.md Phase 5). It captures
        # nothing on a frame where a floating surface sampled it, so a panel
        # never blurs itself.
        from . import vibrancy

        vibrancy.capture(self.ctx, io.display_size)
        self.app_ctx.settings.tick()

    # -- frame steps -------------------------------------------------------

    def _collect_tasks(self) -> None:
        ctx = self.app_ctx
        for done in ctx.tasks.poll():
            if not done.ok:
                # The refusal's *address*, where it has one (UX.md Phase 3).
                # ``ServiceError.field`` has been carried since the class was
                # written and read by nothing, so a refusal about the seed and
                # one about the style LoRA arrived as the same red toast in the
                # corner. Recorded here rather than in each pane because this is
                # the one place every task failure passes through -- and the
                # toast still goes up either way: the ring says *which control*,
                # not *that something happened*, and a pane the user has since
                # navigated away from can draw no ring at all.
                named = getattr(done.error, "field", None)
                if isinstance(named, str):
                    ctx.state.note_field_error(named, done.message or "")
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
                elif done.key.startswith("plotter-"):
                    from . import plotter_mode

                    plotter_mode.on_task_failed(ctx, done)
                elif done.key.startswith("packwright-"):
                    from . import packwright_mode

                    # Same rule, plus one of its own: a failed *pack* has
                    # to clear ``packing`` and record why, or the items
                    # pane shows an empty list that reads as success.
                    packwright_mode.on_task_failed(ctx, done)
                elif done.key.startswith(("download:", "remove:")):
                    # A failed fetch has to be *routed* somewhere, not merely
                    # toasted: the rows carry a presence flag, and a fetch that
                    # got partway before failing has changed what is on disk.
                    # Re-probing costs a few stats and is the only thing that
                    # stops the pane staying optimistic about a download that
                    # did not happen. (Only the rows, not doctor's whole
                    # suite -- nothing succeeded, so the diagnostics have
                    # nothing new to say.)
                    #
                    # A failed *removal* is the same fact from the other side,
                    # and more sharply so: ``uninstall`` renames a directory
                    # out of the way before it deletes it, so a failure part
                    # way through has already made the model absent.
                    self._refresh_model_answers()
                elif done.key.startswith("review-"):
                    from . import review_mode

                    # Same rule: ``scanning`` gates every button and key, so a
                    # failed scan that left it set would make the mode inert.
                    review_mode.on_task_failed(ctx, done)
                elif done.key.startswith("poser-"):
                    from . import poser_mode

                    # Same rule again: ``loading``/``building`` gate the pane
                    # and the viewport's progress row.
                    poser_mode.on_task_failed(ctx, done)
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
                # The first poll is also what pays for the deferred bpy probe
                # (C30). If it says rigging works and the ctx does not yet,
                # re-ask for the templates -- the probe's answer is cached, so
                # the re-ask costs a directory read.
                blender_ok = any(
                    c.name == "Blender (rigging)" and c.ok for c in done.result
                )
                if blender_ok and not ctx.rigging_available:
                    from ..service import rig as svc_rig

                    ctx.submit("rig-templates", svc_rig.rig_templates, self.svc)
            return
        if key == "rig-templates":
            templates = done.result if isinstance(done.result, dict) else {}
            ctx.rigging_available = bool(templates.get("available"))
            ctx.rig_templates = list(templates.get("templates") or [])
            ctx.rig_default = templates.get("default") or ctx.rig_default
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
        if key.startswith("sprite:") and isinstance(done.result, dict):
            # Seeded from the create result so the panel can show *this* job's
            # bar. Keyed by the source reference, because the panel is drawn
            # against that row and not against the synthesis job.
            ctx.state.preview["sprite_active"] = dict(done.result)
            return
        if key.startswith("sprite-del:"):
            # The listing is stamped on the directory's mtime, so the delete
            # shows up on its own -- but the cached textures are keyed by draft
            # id and would otherwise outlive the files they decoded.
            ctx.cache.invalidate()
            return
        if key.startswith(("download:", "remove:")):
            # Re-probe wholesale, exactly as the "health" task above replaces
            # runtime.checks: the fetch wrote files doctor has never looked at,
            # and every model answer in the ctx is derived from that list. A
            # removal is the same wholesale change with the sign flipped, so it
            # takes the same body rather than a second one that could drift.
            from ..service import system as svc_system

            try:
                self.runtime.checks = svc_system.current_checks(self.svc, force=True)
            except Exception:
                log.exception("could not re-run the diagnostics after a download")
            self._refresh_model_answers()
            # Untick whatever is now on disk, rather than the rows this task
            # named: the plan is deduped, so fetching one SDXL 1.0 recipe
            # satisfies the other three and leaving them ticked would offer to
            # download 7 GB that is already there.
            present = {row["row_key"] for row in ctx.model_rows if row.get("present")}
            ctx.model_picks -= present
            ctx.toast(
                "Model removed." if key.startswith("remove:") else "Download finished."
            )
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
        if key.startswith("plotter-"):
            from . import plotter_mode

            plotter_mode.on_task_done(ctx, done)
            if isinstance(done.result, dict) and done.result.get("exported_asset"):
                # The card appears in the library like any other asset, so
                # it needs the thumbnail every other asset gets -- and that
                # is an offscreen GL draw, which belongs on the frame thread
                # rather than in the task that minted the row.
                self._capture_clay_thumbnail(done.result["job_id"])
            return
        if key.startswith("packwright-"):
            from . import packwright_mode

            packwright_mode.on_task_done(ctx, done)
            if isinstance(done.result, dict) and done.result.get("exported_asset"):
                self._capture_clay_thumbnail(done.result["job_id"])
            return
        if key.startswith("review-"):
            from . import review_mode

            review_mode.on_task_done(ctx, done)
            return
        if key.startswith("poser-"):
            from . import poser_mode

            poser_mode.on_task_done(ctx, done)
            return
        if key.startswith("matte-"):
            from . import matte_preview

            matte_preview.on_task_done(ctx, done)
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
        if key == "home-unreviewed":
            # Home's status block. A count only, and the last one stands until
            # a newer one lands -- a failed read leaves the previous figure up
            # rather than blanking a row somebody is reading.
            if isinstance(done.result, int):
                ctx.state.home_unreviewed = done.result
            return
        if key == "storage" or key.startswith("storage:"):
            # Both the full walk and the per-job incremental re-measure (C33)
            # land here; each returns the whole storage dict.
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
        if key.startswith("pose-library:"):
            # The global pose library rows the asset Pose panel offers, keyed
            # by job like poses:/sheets: so an answer that lands after the
            # selection moved on is dropped rather than shown against the
            # wrong asset.
            job_id = key.partition(":")[2]
            if job_id == ctx.state.selected and isinstance(done.result, dict):
                ctx.state.preview["library_poses"] = done.result.get("poses") or []
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
        from .jobs_cache import sweep_summary, transition_message

        ctx = self.app_ctx

        def announce(job: Any, previous: str | None) -> None:
            sweep_id = job.get("sweep_id")
            if sweep_id:
                # One toast per sweep, not one per unit (N109). A twenty-unit
                # sweep otherwise raises twenty notices, which is exactly the
                # burst the "+N more" line exists to count -- and the useful
                # message ("how did it go") is the one nothing was raising.
                summary = sweep_summary(ctx.cache.jobs, sweep_id)
                if summary is not None:
                    ctx.toast(*summary, action="review", action_arg=sweep_id)
            else:
                message = transition_message(job, previous)
                if message is not None:
                    # "Show" selects it (N108): a toast that names a job and
                    # offers no way to it makes the user find it by hand,
                    # which after an overnight batch is the whole problem.
                    ctx.toast(*message, action="show", action_arg=job["id"])
            if job["status"] == "done":
                # Incremental (C33): only this job's directory changed, so only
                # it is re-walked; delete and prune still trigger the full one.
                self._request_storage(job["id"])
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
        # Same shape, same reason: a burst of image labels must not train once on
        # the set as it stood at the first keypress.
        review_mode.pump_judge(ctx)
        # And once more: scoring is a DINOv2 pass per unit, so it is a task, and
        # the request following a retrain is the one with nothing after it.
        review_mode.pump_scores(ctx)
        self._check_worker()
        # Every mode, not only Inker: a crash while the user is looking at the
        # library still loses the painting. ``submit`` refuses a key already in
        # flight, so a slow encode skips a beat rather than queuing. Imported
        # here because ``inker_mode`` pulls the raster engine in and a session
        # that never opens Inker should not pay for it -- the same reason
        # ``ensure`` builds its state lazily.
        from . import inker_mode

        inker_mode.pump_autosave(self.app_ctx)
        if not ctx.state.recovery_offered:
            # Once, on the first frame that has a Ctx: the autosave directory
            # is also where *this* session's copies land, so a second offer
            # would hand the user their own open documents back.
            ctx.state.recovery_offered = True
            inker_mode.offer_recovery(ctx)

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

    def _request_storage(self, job_id: str | None = None) -> None:
        """Re-measure the data directory off the frame thread.

        A recursive stat walk of every job directory is not something to do
        between ``new_frame`` and ``render`` -- and the moment it was being
        asked for is the worst one: the frame that should be showing a job
        finishing. ``submit`` refuses a duplicate key, so a burst of jobs
        completing coalesces into one walk rather than queuing several.

        With ``job_id`` only that job's directory is re-measured and folded
        into the running totals (C33); a delete or prune, which can touch any
        number of directories, still asks for the full walk.
        """
        ctx = self.app_ctx
        if job_id is not None:
            ctx.submit(f"storage:{job_id}", ctx.cache.measure_one, job_id)
            return
        ctx.submit("storage", ctx.cache.measure)

    def _on_pose_dirty(self, dirty: bool) -> None:
        """The one reader of ``Viewer.on_pose_dirty``.

        The viewer reports on every pose edit, every gizmo release and both
        ends of the editor's life; this mirrors it onto ``AppState`` and marks
        the window. Cheap by construction: the callback fires on a *change*
        rather than per frame, and the caption is only touched when the answer
        actually moves -- ``set_caption`` is an OS call and the pose editor's
        rotate gizmo would otherwise make one per mouse-motion event.

        A mirror, not the authority. ``pose_panel.guard`` -- which is what
        stands between unsaved rotations and losing them, on the Done button,
        on a mode switch and in the quit chain -- goes on asking the editor
        itself, so the worst a missed notification can do is leave a marker on
        a title bar. The indicator exists because that guard is the only sign
        the app gives, and it appears *after* the user has asked to leave: the
        banner saying so is inside the very pane you have to be looking at.
        """
        ctx = self.app_ctx
        if ctx is None or bool(dirty) == ctx.state.pose_dirty:
            return
        ctx.state.pose_dirty = bool(dirty)
        self._sync_title()

    def _sync_title(self) -> None:
        """The window caption, marked while something is unsaved.

        Swallows its own failure: a caption is not worth taking a frame down
        for, and this is reachable from a viewer callback that knows nothing
        about whether a display still exists (teardown releases the viewer
        after pygame has quit).
        """
        state = self.app_ctx.state if self.app_ctx is not None else None
        marked = bool(state is not None and state.pose_dirty)
        try:
            import pygame

            pygame.display.set_caption(f"{WINDOW_TITLE} *" if marked else WINDOW_TITLE)
        except Exception:  # a lost display, a headless run
            log.debug("could not set the window caption", exc_info=True)

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
        from ..service import poses as svc_poses
        from ..service import rig as svc_rig
        from ..service import sheets as svc_sheets

        ctx = self.app_ctx
        job = ctx.job()
        for key in (
            "poses",
            "sheets",
            "bones",
            "library_poses",
            # The sprite panel's three, for the same reason: a form
            # holds this attempt's seeds, a draft listing holds one
            # reference's drafts, and the running bar names one job.
            "sprite_active",
            "sprite_drafts",
            "sprite_form",
        ):
            ctx.state.preview.pop(key, None)
        if job is None:
            return
        job_id = job["id"]
        if "rig.glb" in (job.get("files") or []):
            ctx.submit(f"poses:{job_id}", svc_rig.list_poses, ctx.svc, job_id)
            # The global pose library, filtered to this rig's own skeleton --
            # what the Pose panel's "Library poses" section offers.
            ctx.submit(f"pose-library:{job_id}", svc_poses.library_for_job, ctx.svc, job_id)
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
                # A modal owns the keyboard while it is up (I77): Esc cancels
                # it and Enter confirms it, and letting the same press through
                # here would also leave the mode behind the dialog, or submit
                # the form the dialog is a question about. Releases still pass,
                # because Inker's space-to-pan is a hold and would otherwise
                # latch on whenever a dialog opened mid-drag.
                #
                # A focused text field takes the *plain* keys only, so letters
                # still reach it. Modifier chords and the F-keys pass through:
                # the manual and the settings pane both promise Ctrl+K works
                # everywhere, and it used to die the moment the 2D prompt box
                # had focus -- which is exactly where you are when you want it.
                if not (event.type == pygame.KEYDOWN and self._modal_open()) and (
                    not io.want_text_input or self._passes_text_field(event)
                ):
                    self._shortcut(event)
                continue
            # Clay owns its own centre pane, so its viewport takes the mouse
            # in that mode and the asset viewer never sees it -- the two would
            # otherwise both orbit on one drag.
            if ctx.state.mode == "clay":
                self._build_event(event)
                continue
            # Poser too, and for a stronger reason: it has its own Viewer
            # instance, so the shared-viewer path below must never see its
            # events or one drag would orbit both cameras.
            if ctx.state.mode == "poser":
                self._poser_event(event)
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

    def _poser_event(self, event: Any) -> None:
        """Route the mouse to Poser's viewer, on the same hover rule as Clay's.

        A drag already in progress ignores the hover, so crossing onto a panel
        mid-orbit does not drop it.
        """
        viewer = self.poser_viewer
        if viewer is None:
            return
        if self._poser_hovered or viewer._grab is not None:
            viewer.handle_event(event, hovered=self._poser_hovered)

    @staticmethod
    def _passes_text_field(event: Any) -> bool:
        """Whether a key still reaches the shortcuts while a field has focus.

        Modifier chords and the F-keys do; plain keys do not, so typing stays
        typing. The exception list is the one imgui itself owns inside a text
        field -- Ctrl+Z/Y/X/C/V/A are edit-the-text bindings there, and letting
        them through would undo the *document* while you renamed a layer.

        Modifiers come off ``event.mod`` rather than ``pygame.key.get_mods()``,
        which is ``review_mode.handle_key``'s rule and for its reason: ``mod``
        is the state at the moment this key was *pressed*, and ``get_mods()``
        is the state now. Events are drained in a batch after the frame, so a
        modifier released between the press and this call was already read as
        never held -- the shortcut was silently dropped, and only when the
        typist was fast.

        The exception list is Ctrl's alone. Alt and Meta chords were being
        tested against it too, so Alt+C and Alt+V were blocked from the global
        shortcuts on a rationale -- "imgui binds this inside a text field" --
        that is true of neither.
        """
        import pygame

        name = pygame.key.name(event.key).lower()
        if name in _FUNCTION_KEYS:
            return True
        mods = event.mod
        if not mods & (pygame.KMOD_CTRL | pygame.KMOD_ALT | pygame.KMOD_META):
            return False
        if mods & pygame.KMOD_CTRL:
            return name not in _TEXT_FIELD_CTRL
        return True

    def _modal_open(self) -> bool:
        """Whether a confirm or a prompt is on screen and owns the keyboard."""
        ctx = self.app_ctx
        return ctx.confirms.pending is not None or ctx.prompts.pending is not None

    def _note_mode(self, state: Any) -> None:
        """Sample ``mode`` so Esc knows where it came from.

        Once per key event rather than once per frame, and that is the whole
        reason it works: F1 changes the mode from inside this very function, so
        a frame-start sample would still be holding the mode from before it and
        Esc would go two steps back. Sampling here means every change made
        since the previous keypress -- by a landing tile, a library card, a
        drop, or F1 a moment ago -- has already landed.
        """
        if state.mode != state.mode_observed:
            state.previous_mode = state.mode_observed
            state.mode_observed = state.mode

    def _set_mode(self, key: str) -> None:
        """The one way a *shortcut* changes mode, so Home's reset is not a
        second spelling of the switch's.

        The switch itself is :func:`state.set_mode`, which the command palette
        also calls -- the palette used to carry its own copy of these four
        lines, and the copy had already lost the early return.
        """
        from .state import set_mode

        set_mode(self.app_ctx.state, key)

    def _escape_mode(self) -> None:
        """Esc out of a mode you only pass through, back to the work you left.

        Home is the floor rather than a place you escape from: the app opens on
        it, so there is routinely nothing behind it, and bouncing to a stale
        ``previous_mode`` would be a mode switch nobody asked for.
        """
        from . import modes

        state = self.app_ctx.state
        if state.mode == "home":
            return
        target = state.previous_mode
        if target == state.mode or target not in modes.KEYS:
            target = "home"
        self._set_mode(target)

    def _shortcut(self, event: Any) -> None:
        import pygame

        from . import modes

        ctx = self.app_ctx
        self._note_mode(ctx.state)
        # Ctrl+K, before everything, because it is the only binding that must
        # work in *every* mode and the workspace modes each consume whatever
        # reaches them. It is also, since the positional Alt+digit bindings went
        # away with the tenth-and-eleventh mode, the only keyboard route to a
        # mode at all. K is bound by neither Inker nor Clay, so no workspace
        # binding is displaced.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_k
            and pygame.key.get_mods() & pygame.KMOD_CTRL
        ):
            from .panes import palette

            palette.toggle(ctx)
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
            self._set_mode("manual")
            return
        # Above the landing and Inker returns below: the frame rate is a
        # property of the loop, not of whichever pane happens to be on screen,
        # and the chooser is exactly where a slow startup would show.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F10:
            ctx.state.show_fps = not ctx.state.show_fps
            return

        if ctx.state.mode not in modes.WORK_MODES:
            # The Manual, Settings and Profiles have no form to submit and no
            # viewport to frame; every one of these would act on a pane that is
            # not on screen. Esc is the one exception, and it is about the mode
            # rather than about anything in it. Home and the Library are lists,
            # and a list the user is looking at takes the arrows and Enter.
            if event.type != pygame.KEYDOWN:
                return
            if event.key == pygame.K_ESCAPE:
                self._escape_mode()
                return
            # Home's Resume list takes the arrows and Enter (M107). Library and
            # Profiles are their own modes now, so there is no sub-view behind
            # which a cursor could move invisibly and then fire on Enter.
            if ctx.state.mode == "home":
                from .panes import landing

                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    landing.move(ctx, -1 if event.key == pygame.K_UP else 1)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    landing.activate(ctx, ctx.state.home_index)
            elif ctx.state.mode == "library":
                # The Home idiom exactly: the same selection-move the 2D/3D
                # fall-through routes the arrows to, so the library pane has
                # one keyboard whichever mode it is drawn in.
                from .panes import library

                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    library.select_relative(ctx, -1 if event.key == pygame.K_UP else 1)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    library.open_selected(ctx)
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
        if ctx.state.mode == "poser":
            from . import poser_mode

            # Unconditional for the workspace-mode reason: handle_key returns
            # False with nothing selected, and letting that fall through would
            # let F/W/S act on the asset viewport Poser has replaced.
            poser_mode.handle_key(ctx, event)
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
        if ctx.state.mode == "plotter":
            from . import plotter_mode

            # Unconditional for the reason the three above are: handle_key
            # returns False with no map open, and letting that fall through
            # would let F/W/S act on a viewport Plotter has replaced.
            plotter_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "packwright":
            from . import packwright_mode

            packwright_mode.handle_key(ctx, event)
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
        elif event.key in (pygame.K_UP, pygame.K_DOWN):
            # The library is the sidebar in both generate modes, so the arrows
            # are unambiguous here; Review owns Left/Right for its own list and
            # is returned above. Nothing else in 2D/3D reads an arrow key.
            from .panes import library

            library.select_relative(ctx, -1 if event.key == pygame.K_UP else 1)
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
        if ctx.state.mode == "plotter":
            from . import plotter_mode, plotter_state

            suffix = path.suffix.lower()
            if suffix in plotter_state.MAP_SUFFIXES:
                plotter_mode.open_path(ctx, path)
            elif suffix == ".tsx" or suffix in DROPPABLE_IMAGES:
                # An image dropped in Plotter is a *tileset*, not a map -- the
                # 2D pane's reasoning applied here: the refusal and the accept
                # have to say what a drop would have done in this mode.
                plotter_mode.add_tileset_path(ctx, path)
            else:
                ctx.toast(
                    "Plotter opens .wmap, .tmx and .tmj maps, and adds .tsx or "
                    "image files as tilesets.",
                    "error",
                )
            return
        if ctx.state.mode == "packwright":
            from . import packwright_mode, packwright_state

            suffix = path.suffix.lower()
            if suffix == packwright_state.WPACK_SUFFIX:
                packwright_mode.open_path(ctx, path)
            elif suffix in DROPPABLE_IMAGES:
                packwright_mode.add_source_paths(ctx, [path])
            else:
                ctx.toast(
                    "Packwright opens .wpack documents and packs image files.", "error"
                )
            return
        if path.suffix.lower() not in DROPPABLE_IMAGES:
            # The refusal says what a drop would have *done here* (H71). One
            # sentence for both modes was wrong in 2D, where a dropped image is
            # a conditioning reference and never a mesh -- and the sentence is
            # the only thing that teaches the difference, since the two modes
            # accept the same file types.
            ctx.toast(
                "Drop an image to condition this generation on it."
                if ctx.state.mode == "2d"
                else "Drop an image to start a mesh from it.",
                "error",
            )
            return
        if ctx.state.mode == "2d":
            # In the 2D pane a dropped image is a *conditioning reference*, not
            # a mesh to build -- forcing the mode switch here would throw away
            # the prompt the user is composing. One branch, and it is what
            # makes the feature discoverable at all.
            ctx.state.form_2d["ref_path"] = str(path)
            # The Reference section is collapsed by default, so without these
            # two lines the drop is accepted and nothing on screen moves (H70).
            from . import widgets

            widgets.request_open("2d/reference")
            self._flash_drop("2d-ref")
            ctx.toast(f"Using {path.name} as the reference.", "success")
            return
        # A drop is a start: it would otherwise land behind the chooser, with
        # nothing on screen saying anything had happened.
        self._set_mode("3d")
        self._flash_drop("3d-source")
        settings_3d.upload(ctx, path)

    def _flash_drop(self, slot: str) -> None:
        """Mark a slot as having just received a drop, for ``widgets.ring``."""
        state = self.app_ctx.state
        state.drop_flash_slot = slot
        state.drop_flash_at = time.monotonic()

    def _request_quit(self) -> None:
        """One chain, in order: painted pixels, then built geometry, then a pose.

        A list walked by index rather than three lambdas nested by hand (I78).
        ``ConfirmQueue`` is a real queue now, so the old reason for the nesting
        -- three questions at once would have dropped two -- is gone; the chain
        stays because it is the *semantics*, not the workaround. Asking all
        three side by side and quitting once all three said yes would mean
        clicking "Keep editing" on the first still left two more questions to
        dismiss, after the user has already said they are not quitting.
        """
        from . import clay_mode, inker_mode, packwright_mode, plotter_mode, poser_mode
        from .panes import pose_panel

        ctx = self.app_ctx
        # The two pose guards are mutually exclusive by construction: the
        # inspector's asks about the shared viewer's editor, the Poser's about
        # its own instance, so no press ever answers one question twice.
        guards = (
            inker_mode.guard,
            clay_mode.guard,
            plotter_mode.guard,
            packwright_mode.guard,
            pose_panel.guard,
            poser_mode.guard,
        )

        def step(index: int) -> None:
            if index == len(guards):
                self._quit()
                return
            guards[index](ctx, "quit", lambda: step(index + 1))

        step(0)

    def _quit(self) -> None:
        self._running = False

    # -- the UI ------------------------------------------------------------

    def _build_ui(self) -> None:
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import modes, tokens
        from .panes import (
            app_settings,
            inspector,
            landing,
            library,
            settings_2d,
            settings_3d,
        )

        ctx = self.app_ctx
        # Before any pane reads ``layout.SIDEBAR_W``: a width change eases, and
        # a half-eased width read by the left sidebar and the settled one read
        # by the right would be two columns disagreeing about the same frame.
        layout_mod.tick()
        # Recomputed every frame by whoever draws the viewport image. Every
        # mode but 3D returns without drawing it, so it stays false there and
        # the viewer gets no events at all.
        self._viewport_hovered = False
        # The dragged asset (I83) mirrors imgui's own drag state rather than
        # being cleared by whoever accepts it: a drag released over nothing
        # accepts nowhere, and a flag only the drop target clears would leave
        # every slot outlined for the rest of the session.
        if imgui.get_drag_drop_payload_py_id() is None:
            ctx.state.dragging_job = None
        # Arriving in a viewport mode is a change the cache will not announce:
        # the job list has not changed, so nothing else would ask the viewer
        # to show what was just picked.
        if ctx.state.mode != self._last_mode and ctx.state.mode in modes.VIEWPORT_MODES:
            self._sync_viewer()
        if self._last_mode is not None and ctx.state.mode != self._last_mode:
            # The content crossfade (UX.md Phase 1). One place, zero per-pane
            # work: the mode switch's pill already slides, and before this the
            # screen under it teleported. Not on the *first* frame -- there is
            # no previous screen to have come from, and the splash's own fade
            # already owns that moment.
            self._start_transition(tokens.DUR_BASE)
        if ctx.state.mode != self._last_mode and ctx.state.mode == "review":
            # Arriving is the one moment a rescan is certainly wanted, and it
            # is a mode change rather than a job-cache tick, so nothing else
            # would ask. Driven off the change and not off "the list is empty",
            # which would submit a walk of the bench directory every frame on a
            # machine that has never run a sweep.
            from . import review_mode

            review_mode.scan(ctx)
        if ctx.state.mode != self._last_mode and ctx.state.mode == "poser":
            # Review's rule: arriving refreshes the library and asks for the
            # armature preview, both cheap on a warm cache.
            from . import poser_mode

            poser_mode.enter(ctx)
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
            elif mode == "library":
                # The library verbatim rather than a second card list: the
                # filters, the cards and the primary-action ladder are the same
                # question here as in the sidebar, and two implementations of
                # it would drift. This is the call Home's "Open Existing" tile
                # used to make behind a sub-view enum.
                from .panes import library as library_pane

                library_pane.draw(ctx)
            elif mode == "profiles":
                from .panes import profiles_panel

                profiles_panel.draw(ctx)
            elif mode == "clay":
                self._clay_workspace()
            elif mode == "poser":
                self._poser_workspace()
            elif mode == "review":
                self._review_workspace()
            elif mode == "plotter":
                self._plotter_workspace()
            elif mode == "packwright":
                self._packwright_workspace()
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
        """Clay's viewport, built on first use -- and mirrored onto the ctx,
        ``_ensure_poser_viewer``'s way: clay_mode's drag keyboard, the axis
        views and ``camera_of`` all read ``ctx.clay_view``, and without the
        mirror every one of them found None forever."""
        from .clay_view import ClayView

        if self.clay_view is None:
            self.clay_view = ClayView(self.ctx, self.app_ctx)
            self.app_ctx.clay_view = self.clay_view
        return self.clay_view

    def _poser_workspace(self) -> None:
        """The sidebar / centre / sidebar skeleton, Poser's way:

            [ poser_library ]  viewport  [ poser_controls ]

        One pane per side rather than Clay's stacked pairs: the library and the
        controls are each one scroller, and an empty half-pane would be chrome.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import poser_controls, poser_library
        from .tokens import sp

        ctx = self.app_ctx
        sidebar_w = sp(layout_mod.SIDEBAR_W)
        if layout_mod.pane_child("poser-library", (sidebar_w, 0)):
            poser_library.draw(ctx)
        imgui.end_child()

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        if layout_mod.pane_child("poser-centre", (width, 0), flags):
            self._poser_viewport(ctx)
        imgui.end_child()

        imgui.same_line()
        if layout_mod.pane_child("poser-controls", (0, 0)):
            poser_controls.draw(ctx)
        imgui.end_child()

    def _poser_viewport(self, ctx: Any) -> None:
        from imgui_bundle import imgui

        from . import poser_mode, widgets
        from .panes import overlay

        self._poser_hovered = False
        state = poser_mode.ensure(ctx)
        if not ctx.rigging_available:
            overlay.placeholder(ctx)
            return
        viewer = self._ensure_poser_viewer()
        showing = poser_mode.sync_preview(ctx, viewer)
        if not showing:
            if state.building:
                imgui.dummy((0, 40))
                widgets.muted("Building the skeleton preview...")
                widgets.cost_note(
                    "The armature is built by Blender once per skeleton and "
                    "cached; the first open of a template takes a moment."
                )
            elif state.error:
                imgui.dummy((0, 40))
                widgets.muted(state.error)
            else:
                overlay.placeholder(ctx)
            return
        avail = imgui.get_content_region_avail()
        rect = (
            imgui.get_cursor_screen_pos().x,
            imgui.get_cursor_screen_pos().y,
            max(avail.x, 1.0),
            max(avail.y, 1.0),
        )
        texture = viewer.render(rect, 1.0 / TARGET_FPS)
        imgui.image(widgets.texture_ref(texture), (rect[2], rect[3]), (0, 1), (1, 0))
        self._poser_hovered = imgui.is_item_hovered()

    def _ensure_poser_viewer(self) -> Any:
        """Poser's own Viewer, built on first use for ClayView's reason -- and
        mirrored onto the ctx so poser_mode's guard can reach the editor."""
        from .viewer_embed import Viewer

        if self.poser_viewer is None:
            self.poser_viewer = Viewer(self.ctx)
            self.app_ctx.poser_viewer = self.poser_viewer
        return self.poser_viewer

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
            # The GL readback only; the PNG encode joins the save on the task
            # thread (D41), exactly as ctx.capture_thumbnail does.
            image = self.clay_view.screenshot()
        except Exception:
            # A warning rather than an error (E48): the export itself succeeded
            # and the asset is in the library -- what failed is its picture. The
            # card falls back to the placeholder, which on its own reads as "the
            # build produced nothing".
            log.exception("could not capture a thumbnail for built asset %s", job_id)
            # Level ``info``, not ``error``: the export succeeded and only the
            # picture did not, and there is exactly one level above info until
            # H68 adds the middle one -- claiming ``error`` here would put a red
            # dismissable card over a build that worked.
            ctx.toast("The asset was built, but its thumbnail could not be made.", "info", "log")
            return

        def run() -> Any:
            import io

            buf = io.BytesIO()
            image.convert("RGB").save(buf, "PNG")
            return svc_files.save_thumbnail(ctx.svc, job_id, buf.getvalue())

        ctx.submit(f"thumb:{job_id}", run)

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
            # The remedy is in the log, so say so (E48): the causes are a lost
            # GL context and a document the renderer choked on, and the message
            # cannot tell the user which without reading it.
            ctx.toast("That document could not be rendered.", "error", "log")
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

        self._clay_tabs(ctx, clay_mode)
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
        # One viewport, many tabs: the camera belongs to the *document*, so it
        # is snapshotted off the live one on the way out of a tab and put back
        # on the way in. Done here rather than in ``ClayState.activate`` because
        # this is the only place that has the viewport -- and it is keyed on
        # what is being drawn rather than on the switch, so a tab restored from
        # a ``.wblk`` or closed out from under the pointer lands correctly too.
        if self._clay_camera_tab != tab.uid:
            clay_mode.remember_camera(ctx, state.get(self._clay_camera_tab))
            clay_mode.apply_camera(ctx, tab)
            self._clay_camera_tab = tab.uid
        view.wireframe = state.wireframe
        view.show_grid = state.grid
        texture = view.draw(tab.doc, rect, 1.0 / TARGET_FPS)
        imgui.image(widgets.texture_ref(texture), (rect[2], rect[3]), (0, 1), (1, 0))
        self._build_hovered = imgui.is_item_hovered()
        self._clay_marquee(imgui, view, rect)
        self._clay_drag_hud(imgui, widgets, view, rect)
        clay_menu.draw(ctx, view)

    def _clay_tabs(self, ctx: Any, clay_mode: Any) -> None:
        """Clay's open documents, which nothing has ever drawn.

        The document model was all there -- ``docs``, ``active``, ``activate``,
        ``cycle``, ``close`` -- and the only thing missing was the bar: Ctrl+Tab
        switched between documents with nothing on screen to say there was more
        than one, and ``close`` had no caller at all.

        Drawn above ``_clay_empty`` as well as above the viewport, because the
        last tab closing is exactly when the bar disappears and the empty state
        has to be what is underneath it.

        ``unsaved_document`` rather than a ``"* "`` prefix, which is Inker's
        rule and the right one: the title is half of the tab's identity.
        """
        from imgui_bundle import imgui

        state = clay_mode.ensure(ctx)
        if not state.docs:
            return
        # ``auto_select_new_tabs`` for ``inker_canvas``'s reason: without it, a
        # second opened document lands behind the first and "Open" looks inert.
        flags = (
            imgui.TabBarFlags_.reorderable.value
            | imgui.TabBarFlags_.auto_select_new_tabs.value
        )
        if not imgui.begin_tab_bar("clay-tabs", flags):
            return
        for tab in list(state.docs):
            item_flags = imgui.TabItemFlags_.unsaved_document.value if tab.dirty else 0
            opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
            if opened:
                state.activate(tab.uid)
                imgui.end_tab_item()
            if not keep:
                clay_mode.close_tab(ctx, tab.uid)
        imgui.end_tab_bar()

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
        found = clay_mode.recent_paths(ctx)
        if found:
            imgui.dummy((0, 16))
            widgets.section("recent")
            for path in found[:6]:
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

    def _clay_drag_hud(self, imgui: Any, widgets: Any, view: Any, rect: Any) -> None:
        """What the live drag currently amounts to, above the cursor.

        In the draw list for ``_clay_marquee``'s reason, and *near the cursor*
        rather than in a corner: the number answers a question the user is
        asking with their hand, and a readout they have to look away to find is
        one they stop looking at. It draws only while a drag is live, so an idle
        viewport is unchanged.
        """
        text = getattr(view, "drag_hud", "")
        if not text or not getattr(view, "dragging", False):
            return
        from . import theme
        from .tokens import sp

        mouse = imgui.get_mouse_pos()
        x, y = mouse.x + sp(18), mouse.y - sp(28)
        draw = imgui.get_window_draw_list()
        size = imgui.calc_text_size(text)
        pad = sp(6)
        draw.add_rect_filled(
            (x - pad, y - pad),
            (x + size.x + pad, y + size.y + pad),
            imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.92)),
            sp(4),
        )
        draw.add_text((x, y), imgui.get_color_u32(theme.rgba(theme.TEXT)), text)

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
            inker_timeline,
            inker_tools,
        )
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)
        tab = None if ctx.state.inker is None else ctx.state.inker.active
        animated = tab is not None and tab.doc.anim is not None
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
        imgui.begin_group()
        # A *positive* height, never a bottom offset, and only when there is a
        # timeline to reserve for: with little room left a negative height
        # collapses the canvas child to nothing and the canvas -- and its
        # texture uploads -- silently stops being drawn. Same rule the status
        # bar inside ``inker_canvas`` already follows. A still document takes
        # the untouched branch, so its layout is what it always was.
        centre_h = 0.0
        if animated:
            strip = sp(inker_timeline.STRIP_H)
            centre_h = max(imgui.get_content_region_avail().y - strip, sp(120))
        if layout_mod.pane_child("inker-centre", (width, centre_h), flags):
            inker_canvas.draw(ctx)
        imgui.end_child()
        if animated:
            if layout_mod.pane_child("inker-timeline", (width, 0)):
                inker_timeline.draw(ctx)
            imgui.end_child()
        imgui.end_group()

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

    def _plotter_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses:

            [ plotter-tools   ]           [ plotter-layers ]
            [ plotter-tileset ]  the map  [ plotter-bridge ]

        Mirrors ``_clay_workspace`` line for line, ``settings_share`` included,
        so the editors do not drift into looking like different applications.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import (
            plotter_bridge,
            plotter_canvas,
            plotter_layers,
            plotter_tileset,
            plotter_tools,
        )
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)

        imgui.begin_group()
        tools_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("plotter-tools", (sidebar_w, tools_height)):
            plotter_tools.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("plotter-tileset", (sidebar_w, 0)):
            plotter_tileset.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        if layout_mod.pane_child("plotter-centre", (width, 0), flags):
            plotter_canvas.draw(ctx)
        imgui.end_child()

        imgui.same_line()
        imgui.begin_group()
        layers_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("plotter-layers", (0, layers_height)):
            plotter_layers.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("plotter-bridge", (0, 0)):
            plotter_bridge.draw(ctx)
        imgui.end_child()
        imgui.end_group()

    def _packwright_workspace(self) -> None:
        """The same skeleton again:

            [ packwright-sources  ]              [ packwright-items  ]
            [ packwright-settings ]  the atlas   [ packwright-bridge ]

        The centre pane is also the mode's heartbeat -- there is no per-mode
        update hook, so the pane that draws is what pumps the repack request.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import (
            packwright_bridge,
            packwright_items,
            packwright_preview,
            packwright_settings,
            packwright_sources,
        )
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = sp(layout_mod.SIDEBAR_W)

        imgui.begin_group()
        sources_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("packwright-sources", (sidebar_w, sources_height)):
            packwright_sources.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("packwright-settings", (sidebar_w, 0)):
            packwright_settings.draw(ctx)
        imgui.end_child()
        imgui.end_group()

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        if layout_mod.pane_child("packwright-centre", (width, 0), flags):
            packwright_preview.draw(ctx)
        imgui.end_child()

        imgui.same_line()
        imgui.begin_group()
        items_height = imgui.get_content_region_avail().y * lay.settings_share
        if layout_mod.pane_child("packwright-items", (0, items_height)):
            packwright_items.draw(ctx)
        imgui.end_child()
        if layout_mod.pane_child("packwright-bridge", (0, 0)):
            packwright_bridge.draw(ctx)
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
        # The labelling grid replaces the viewport rather than sitting beside it:
        # a mesh on screen under a question about a *picture* is the mismatch that
        # files an accept about the wrong artifact. It also scrolls, so it must
        # not inherit the viewport's no-scroll flag.
        labelling = state.labels is not None
        flags = 0 if labelling else imgui.WindowFlags_.no_scroll_with_mouse.value
        if layout_mod.pane_child("review-centre", (width, 0), flags):
            if labelling:
                self._review_labels(ctx, state, review_mode)
            else:
                self._review_viewport(state, review_mode, width)
        imgui.end_child()

        imgui.same_line()
        if layout_mod.pane_child("review-verdict", (0, 0)):
            if labelling:
                self._review_label_panel(ctx, state, review_mode)
            else:
                self._review_verdict(ctx, state, review_mode)
        imgui.end_child()

    # How wide a labelling cell is, in design px. Big enough to judge a
    # composition by -- which is the whole question -- and small enough that a
    # sidebar-width column still fits three across at 100% scale.
    _LABEL_CELL = 132

    def _review_labels(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """The labelling grid: images, two keys, no reason step.

        **One thumbnail upload per frame.** ``review_mode.next_thumbnail`` hands
        back at most one row per call, and the rest draw a placeholder until
        their turn comes -- ``viewer/sheet.StripRender``'s rule at a larger scale,
        because a synchronous upload per cell over a hundred cells is a freeze
        measured in seconds rather than frames.
        """
        from imgui_bundle import imgui

        from . import icons, theme, widgets
        from .tokens import sp

        labels = state.labels
        widgets.section(_LABEL_TITLES.get(labels.stage, labels.stage))
        widgets.hint_text(_LABEL_QUESTIONS.get(labels.stage, ""))
        if labels.loading:
            widgets.muted("Reading...")
            return
        if not labels.rows:
            widgets.empty_state(
                icons.CHECK,
                "Nothing left to label",
                "Every image has an answer for this question.",
            )
            return

        # Exactly one upload admitted per frame, claimed before the loop so which
        # cell gets it does not depend on where the scroll happens to be.
        review_mode.next_thumbnail(labels)
        side = float(sp(self._LABEL_CELL))
        per_row = max(int(imgui.get_content_region_avail().x // (side + sp(8))), 1)
        for i, row in enumerate(labels.rows):
            if i % per_row:
                imgui.same_line()
            imgui.begin_group()
            texture = None
            # ``ctx.textures`` is None until a GL context exists (app_ctx
            # defaults it), which is the state a headless or pre-init draw is
            # in -- every pane guards it and these three Review sites did not.
            if i < labels.uploaded and ctx.textures is not None:
                texture = ctx.textures.get(
                    review_mode.cache_id_for_label(row), row["image"]
                )
            if texture is not None:
                imgui.image(widgets.texture_ref(texture), (side, side))
            else:
                # A placeholder rather than nothing: the grid must not reflow as
                # the uploads land, or a click lands on a cell that moved.
                imgui.dummy((side, side))
            if imgui.is_item_clicked():
                labels.index = i
            mark = {"accept": icons.CHECK, "reject": icons.X}.get(row["verdict"] or "", "")
            colour = theme.ACCENT if i == labels.index else theme.MUTED
            widgets.text_colored(colour, f"{mark} {i + 1}")
            imgui.end_group()

    def _review_label_panel(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """What is being labelled, and what the probe knows so far."""
        from imgui_bundle import imgui

        from . import widgets

        labels = state.labels
        row = review_mode.current_label(state)
        widgets.section("Label")
        if row is None:
            widgets.muted("Nothing selected.")
        else:
            widgets.muted(str(row["prompt"])[:120])
            if row.get("status") == "error":
                # The most informative negatives in the corpus, and worth saying
                # so: this image was refused at the composition gate.
                widgets.hint_text("This job was refused; the picture is still judgeable.")
            texture = (
                None
                if ctx.textures is None
                else ctx.textures.get(review_mode.cache_id_for_label(row), row["image"])
            )
            if texture is not None:
                side = min(imgui.get_content_region_avail().x, 220.0)
                imgui.image(widgets.texture_ref(texture), (side, side))
        imgui.separator()
        if widgets.primary_button("Good (A)", enabled=row is not None):
            review_mode.record_label(ctx, "accept")
        imgui.same_line()
        if widgets.disabled_button("Bad (R)", row is not None):
            review_mode.record_label(ctx, "reject")
        imgui.same_line()
        if widgets.disabled_button("Skip (S)", row is not None):
            review_mode.advance_labels(labels)
        if imgui.button("Done"):
            review_mode.close_labels(ctx)

        # The snapshot the listing task read, kept current by ``record_label``.
        # Never a live ``judge.status`` call: that is a whole-table scan plus a
        # stat, and this panel draws every frame.
        status = labels.status
        imgui.separator()
        widgets.section("The probe")
        answered = sum(1 for r in labels.rows if r["verdict"])
        widgets.muted(f"{answered} labelled this session")
        widgets.muted(
            f"{status.get('positives', 0)} good / {status.get('negatives', 0)} bad, "
            f"{status.get('needed', 0)} of each needed"
        )
        if status.get("trained"):
            widgets.muted(f"trained on {status.get('trained_labels', 0)} label(s)")
        else:
            widgets.muted("no probe yet")
        widgets.hint_text(
            "Advisory only. A trained probe scores each unit and sorts the "
            "review best-first; it never hides, refuses or deletes anything, "
            "and it files no verdict of its own yet."
        )

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
        # Blinding is a session control rather than a per-sweep one, and it is
        # here because it belongs beside the list it re-presents. It renames and
        # *reorders*: see review_mode's docstring on why hiding the label alone
        # blinds nothing.
        changed, blind = widgets.toggle("Blind", state.blind, tag="review-blind")
        if changed:
            review_mode.set_blind(ctx, blind)
        widgets.hint_text("Hides which settings each unit ran, and the order.")
        if not state.sweeps and not state.scanning:
            # H73. An empty Sweeps heading with a Rescan button under it says
            # nothing about *why* -- and the two reasons (no sweep has ever run
            # here, versus the bench directory is somewhere else) want different
            # responses.
            widgets.empty_state(
                icons.LIST,
                "No sweep runs found",
                "Launch one below, or check that the bench directory is where "
                "you expect.",
            )
        # J86: a bench directory accumulates a run per experiment and nothing
        # ever removes one, so this is the panel list that grows fastest.
        needle = widgets.list_filter(ctx, "sweeps", len(state.sweeps))
        shown = 0
        for sweep in state.sweeps:
            if needle and needle not in str(sweep["label"]).lower():
                continue
            shown += 1
            todo = sweep["todo"]
            total = len(sweep["units"])
            selected = sweep["id"] == state.sweep_id
            if imgui.selectable(f"{sweep['label']}##sweep-{sweep['id']}", selected)[0]:
                review_mode.open_sweep(ctx, sweep["id"])
            widgets.muted(f"   {total - todo}/{total} reviewed")
            if selected and sweep["id"] != review_mode.RECENT_ID:
                self._review_delete_button(ctx, state, review_mode, sweep)
        widgets.no_matches(needle, shown)
        imgui.separator()
        # The labelling passes, beside the sweep list rather than in a mode of
        # their own: the judge is meant to improve as the corpus is reviewed,
        # which is the whole reason the loop lives here.
        widgets.section("Teach the judge")
        for stage, title in _LABEL_TITLES.items():
            open_here = state.labels is not None and state.labels.stage == stage
            if imgui.selectable(f"{title}##label-{stage}", open_here)[0]:
                if open_here:
                    review_mode.close_labels(ctx)
                else:
                    review_mode.open_labels(ctx, stage)
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
            dialogs.ask_delete(
                ctx,
                title="Delete this sweep?",
                message=(
                    f"{sweep['label']}: its {len(sweep['units'])} job(s), their meshes "
                    "and their reference images are deleted.\n\n"
                    "The verdicts you recorded are kept, and so are the findings "
                    "they feed -- each one carries its own copy of the settings it "
                    "was filed against.\n\n"
                    "Units you accepted, and any image you labelled, are kept with "
                    "their files: a verdict's copy of the settings cannot stand in "
                    "for the picture it was filed against."
                ),
                on_confirm=lambda: review_mode.delete(ctx, sweep_id),
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
            # The grade, which says more than the tick it replaces -- but a
            # unit judged before migration 10, or by an older build, has a
            # verdict and no grade, so the icons stay as the fallback rather
            # than that row going blank.
            mark = review_mode.grade_text(unit.get("grade")) or {
                "accept": icons.CHECK, "reject": icons.X
            }.get(unit["verdict"] or "", " ")
            if imgui.selectable(
                f"{mark} {review_mode.label(state, unit)}##unit-{unit['job_id']}",
                i == state.index,
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

        widgets.section(review_mode.label(state, unit))
        widgets.muted(f"{state.index + 1} of {len(state.units)}  -  {unit['job_id']}")

        reference = review_mode.reference_path(unit)
        if reference is not None and ctx.textures is not None:
            texture = ctx.textures.get(review_mode.cache_id(unit), reference)
            if texture is not None:
                side = min(imgui.get_content_region_avail().x, 220.0)
                imgui.image(widgets.texture_ref(texture), (side, side))

        for line in review_mode.mesh_lines(unit):
            widgets.muted(line)

        # Below the measurements and named as a judgement, because it is one and
        # the measurements are not. Empty when there is no probe, when the judge
        # had nothing to say about this row, and always under blinding.
        judged = review_mode.score_line(state, unit)
        if judged:
            widgets.muted(judged)

        imgui.separator()
        enabled = not state.scanning

        if state.pending_negative:
            # R is a *sign*, held until the next digit, and nothing on screen
            # said it was held: the reviewer who pressed R and then walked
            # away came back and pressed 4 expecting +4. Warn-coloured because
            # the consequence of not noticing is the opposite verdict, and it
            # says how to drop it -- Esc, which ``_disarm`` already answers.
            from . import theme

            widgets.text_colored(
                theme.WARN, "Negative armed: the next digit files a minus. Esc drops it."
            )

        grade = widgets.grade_buttons("review", enabled)
        if grade is not None:
            review_mode.record(ctx, grade, state.pending_tags)
        widgets.muted_wrapped(
            "+5 ships as-is, +3 usable, -5 unusable. Tags are optional. "
            "Keys: a digit grades, R first makes it negative, S skips."
        )

        tag = widgets.tag_toggles("review", state.pending_tags, enabled)
        if tag is not None:
            review_mode.toggle_tag(state, tag)

        if widgets.disabled_button("Skip (S)", enabled):
            review_mode.advance(state)

        if unit["verdict"]:
            # ``grade_text`` rather than the verdict word: the word is the
            # derived cut and the grade is what was actually said, so showing
            # the word here would answer a coarser question than the one the
            # buttons above ask.
            recorded = review_mode.grade_text(unit.get("grade")) or unit["verdict"]
            if unit.get("tags"):
                recorded += " - " + ", ".join(unit["tags"])
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

        imgui.separator()
        if not widgets.header("What works", default_open=False):
            return
        # Everything below the header guard (B21), the load included: it is
        # mtime-cached but still a stat per frame, for a section that is
        # closed by default -- and the lines are formatted from scratch.
        doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
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
            tagged = findings_lib.tag_line(entry)
            if tagged:
                widgets.muted(tagged)
            vector = entry["vector"]
            if imgui.button(f"Apply to forms##apply-{entry['key']}"):
                vector_presets.apply(ctx.state, vector)
                ctx.toast("Applied those settings to the 2D and 3D forms.")
            widgets.same_line_or_wrap(widgets.button_width("Save as preset..."))
            if imgui.button(f"Save as preset...##save-{entry['key']}"):
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
        from .panes import overlay, palette, settings_3d

        ctx = self.app_ctx
        overlay.fps_meter(ctx, self.fps)
        if ctx.state.mode != "home":
            overlay.progress_card(ctx, self.eta)
        widgets.toasts(
            ctx.state,
            (viewport.work_size.x, viewport.work_size.y),
            on_action=self._toast_action,
        )
        # Before the confirms, because it is the same kind of thing and the
        # earlier one wins the single modal slot imgui gives a frame.
        settings_3d.matte_modal(ctx)
        # Above the confirms it can raise (Delete asks): the palette closes
        # itself in the same frame it runs a command, so the question it asks
        # takes the modal slot on the frame after, with nothing to contend
        # with.
        palette.draw(ctx)
        ctx.confirms.draw()
        ctx.prompts.draw()
        # Last, and on the foreground list, so it covers everything above --
        # including the modals, which are part of the screen being crossfaded.
        self._transition_overlay(viewport)

    # -- transitions -------------------------------------------------------
    #
    # One full-viewport veil in the window's own background colour, fading
    # out. It is not a crossfade between two rendered screens -- imgui has one
    # framebuffer and keeping the previous frame's would be Phase 5's offscreen
    # copy -- it is the cheap half of one, and against a near-black ground the
    # difference is not visible at 200 ms. It paints only; the UI underneath
    # stays live, which is why a transition can never eat a click.

    TRANSITION_KEY = "app/transition"

    def _start_transition(self, duration: float) -> None:
        from . import motion

        self._transition_duration = duration
        motion.restart(self.TRANSITION_KEY)

    def _transition_overlay(self, viewport: Any) -> None:
        from imgui_bundle import imgui

        from . import motion, theme

        duration = self._transition_duration
        if duration <= 0.0:
            return
        t = motion.ease(self.TRANSITION_KEY, duration)
        if t >= 1.0:
            # Latched off rather than re-eased every frame for the life of the
            # session: ``ease`` on a finished key is cheap but not free, and a
            # veil at alpha 0 is still a full-viewport quad in the draw list.
            self._transition_duration = 0.0
            return
        low = viewport.pos
        high = (low.x + viewport.size.x, low.y + viewport.size.y)
        imgui.get_foreground_draw_list().add_rect_filled(
            (low.x, low.y), high, imgui.get_color_u32(theme.rgba(theme.BG, 1.0 - t))
        )

    def _toast_action(self, name: str, arg: str | None = None) -> None:
        """What a toast's action button does, kept out of the widget.

        ``widgets.toasts`` knows what to *draw* for an action and nothing about
        what it means, which is what lets state.py carry the name with no
        import of the App and lets a pane raise a toast without either.
        """
        ctx = self.app_ctx
        if name == "log":
            ctx.open_log()
        elif name == "show" and arg:
            from .panes import library

            # Through the library's own selector, so the promotion source
            # follows the selection exactly as a click on the card would.
            library.select(ctx, arg)
            job = ctx.cache.get(arg)
            if job is not None:
                self._set_mode("2d" if job.get("stage") in ("reference", "tile") else "3d")
            ctx.state.library_scroll_to = arg
        elif name == "undo" and arg:
            from .panes import library

            # Through the library's own restore, so the tick set and the
            # selection are handled exactly as they are when the trash view's
            # own Restore button is pressed.
            library.restore_asset(ctx, arg)
        elif name == "review":
            # The sweep is not named here: Review rescans on arrival and its
            # run list is the thing that knows which directories exist. Landing
            # on the mode is the whole of what the button promises.
            self._set_mode("review")

    def _mode_switch(self) -> None:
        from imgui_bundle import imgui

        from . import modes, widgets

        ctx = self.app_ctx
        state = ctx.state
        # No mode switch is destructive: Inker's documents are still open when
        # you come back, because it is a mode rather than a takeover. Only
        # quitting and closing a tab can lose pixels, and both ask.
        # Thirteen segments do not fit the resize floor at 1.5 UI scale, and a
        # clipped segment is an unreachable mode. So the switch has a compact
        # form -- the glyph alone, with the label in a tooltip -- taken when the
        # full one will not fit the width actually available. Both labellings
        # come from ``modes.MODES``, so the two can never name different things.
        selected = widgets.segmented_control(
            "mode-seg",
            [(key, f"{icon} {label}") for key, label, icon in modes.MODES],
            state.mode,
            breaks=modes.GROUP_BREAKS,
            compact=[(key, icon) for key, _label, icon in modes.MODES],
            max_width=imgui.get_content_region_avail().x,
        )
        if selected != state.mode:
            self._set_mode(selected)

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
        from .panes import overlay

        # The right-hand strip: the resource readout, the shortcuts button and
        # the health dot. Its width is *measured* rather than reserved as a
        # constant, because ``same_line`` past the content region clips instead
        # of wrapping -- a control drawn out there is simply gone, which is the
        # bug that once hid seven of them -- and the text's width is a function
        # of the DPI scale, the font and how many digits the readings have.
        # What the health control says and how wide it is (F58). A 16 px
        # invisible button with a 9 px circle drawn under it is not a control:
        # it is under the 24 px minimum anything is comfortably clickable at,
        # it has no label, and on a host where something is actually wrong it
        # looks exactly like a decoration. So when a check is failing the dot
        # gains a word and a real button-sized hit area; when everything passes
        # it stays a dot, because a permanent "OK" badge is noise.
        failing = [c for c in checks if not c.ok]
        health_label = "" if not failing else ("Issue" if len(failing) == 1 else "Issues")
        health_w = sp(24) if not health_label else (
            sp(22) + imgui.calc_text_size(health_label).x + imgui.get_style().frame_padding.x * 2
        )

        line, tip = overlay.status_text(ctx, self.fps)
        spacing = imgui.get_style().item_spacing.x
        strip = (
            imgui.calc_text_size(line).x
            + spacing
            + imgui.get_frame_height()  # the ? button is square
            + spacing
            + health_w
            + spacing
            + imgui.get_frame_height()  # and so is Quit
        )
        widgets.same_line_or_wrap(strip)
        avail = imgui.get_content_region_avail().x
        if avail >= strip:
            # Right-aligned when there is room for the whole strip, and left
            # exactly where the switch ended when there is not: never past the
            # edge, and never on top of the switch either.
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + avail - strip)
        overlay.status_readout(line, tip)
        imgui.same_line()
        # The tooltip carries F1 (I81): the popup lists it, but the popup is
        # the thing you have to already know about. A "?" that says only
        # "Keyboard shortcuts" leaves the one binding that opens the
        # documentation reachable exclusively from the documentation.
        if widgets.icon_button("?", "Keyboard shortcuts - F1 opens the Manual"):
            # Cleared on the way in rather than on the way out: a popup can be
            # dismissed by clicking anywhere, which is not a moment this has a
            # hook in, and reopening onto last time's query would look like a
            # list that had lost most of its rows.
            self._shortcuts_query = ""
            imgui.open_popup("shortcuts")
        self._shortcuts_popup()
        imgui.same_line()
        pos = imgui.get_cursor_screen_pos()
        centre_y = pos.y + imgui.get_frame_height() * 0.5
        imgui.get_window_draw_list().add_circle_filled(
            (pos.x + sp(11), centre_y), sp(4.5), imgui.get_color_u32(theme.rgba(colour)), 16
        )
        if imgui.invisible_button("##health", (health_w, imgui.get_frame_height())):
            imgui.open_popup("diagnostics")
        hovered = imgui.is_item_hovered()
        if health_label:
            # Drawn over the button rather than beside it, so the word is part
            # of the same hit area instead of a second thing to aim at.
            imgui.get_window_draw_list().add_text(
                (pos.x + sp(20), pos.y + imgui.get_style().frame_padding.y),
                imgui.get_color_u32(theme.rgba(colour)),
                health_label,
            )
        if hovered:
            # The failing rows by name (N110). "System status - click for
            # details" made the click mandatory to learn whether the amber dot
            # meant a missing style LoRA or a held trellis port -- one is worth
            # ignoring for the rest of the session and the other is not.
            imgui.set_tooltip(
                "Everything checks out - click for details"
                if not failing
                else "\n".join(
                    ["Click for details:"]
                    + [f"  {'!' if c.fatal else '-'} {c.name}" for c in failing[:8]]
                    + ([f"  ... and {len(failing) - 8} more"] if len(failing) > 8 else [])
                )
            )
        self._diagnostics_popup(checks)

        # Quit, out of the navigation control at last (UX.md Phase 2). It was
        # the eleventh segment of the mode switch: a destructive action inside
        # the thing you use to move around, one click from every mode, mitigated
        # by an unconditional confirm -- which is a mitigation and not a fix.
        # Here it sits with the other two app-level controls, the ones that are
        # about the *program* rather than about the work.
        #
        # It is still an action rather than a mode, so nothing is ever assigned
        # to ``state.mode`` and cancelling leaves the switch where it was; and it
        # still always asks, because the window's own X is the path that carries
        # the unsaved-work chain instead.
        imgui.same_line()
        if widgets.icon_button(modes.QUIT[2], f"{modes.QUIT[1]} Warlock Studio", danger=True):
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

    # Every binding the app answers to, in one place the user can find. The
    # tuples are (keys, what), grouped; Inker's letters come from TOOL_KEYS so
    # this list cannot drift from the handler.
    def _shortcuts_popup(self) -> None:
        from imgui_bundle import imgui

        from . import widgets
        from .tokens import sp

        alpha, rise = widgets.popup_enter("shortcuts")
        # Translucent (UX.md Phase 5): cleared before ``begin`` paints it,
        # painted back below as a blur of the app or as the solid fill.
        frosted = widgets.frosted()
        if frosted:
            imgui.set_next_window_bg_alpha(0.0)
        imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
        if not imgui.begin_popup("shortcuts"):
            imgui.pop_style_var()
            return
        rounding = imgui.get_style().popup_rounding
        widgets.window_shadow("raised", radius=rounding)
        if frosted:
            widgets.window_backdrop(radius=rounding)
        if rise > 0.0:
            imgui.dummy((0, rise))

        # Collected first, drawn after the box (UX.md Phase 4). The list is ~60
        # rows over eight groups, which is a scroll and a read rather than a
        # lookup -- and the subsequence matcher the command palette already
        # carries is the right instrument, so it is reused rather than
        # reimplemented. Every group's rows are gathered before anything is
        # drawn because the query decides which *groups* survive: a heading over
        # nothing is a section that looks broken.
        sections: list[tuple[str, list[tuple[str, str]]]] = []

        def table(title: str, rows: list[tuple[str, str]]) -> None:
            sections.append((title, rows))

        imgui.dummy((sp(420), 0))
        table(
            "Everywhere",
            [
                # No per-mode digit: thirteen modes against ten digits is either
                # two modes with no key or a table saying which two, and the
                # palette is the keyboard route to all of them.
                ("Ctrl+K", "Command palette -- switch mode, or open an asset"),
                ("F1", "Switch to the Manual"),
                ("Esc", "Leave Home, the Manual or Settings"),
                ("F10", "Toggle the frame-rate readout"),
            ],
        )
        table(
            "2D / 3D",
            [
                ("Ctrl+Enter", "Generate / Make 3D"),
                ("Tab / Shift+Tab", "Move between the form's controls"),
                ("Enter", "Press Generate / Make 3D when it is the one focused"),
                ("Up / Down", "Previous / next asset in the library"),
                ("Right-click a card", "Its actions menu"),
                ("F", "Frame the model"),
                ("W", "Toggle wireframe"),
                ("S", "Toggle turntable"),
                ("Esc", "Exit comparison / pose edit"),
            ],
        )
        table(
            "Review",
            [
                ("1 - 5", "Grade the mesh +1 to +5 (+3 is usable)"),
                ("R then 1 - 5", "Grade it -1 to -5"),
                ("0", "Grade it 0 - no opinion either way"),
                ("Ctrl + 1-5", "Toggle a good tag for the next grade"),
                ("Shift + 1-5", "Toggle a bad tag for the next grade"),
                ("S", "Skip to the next unverdicted unit"),
                ("Left / Right", "Previous / next unit"),
                ("Esc", "Clear the pending sign and tags"),
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
        from .plotter_state import TOOLS as PLOTTER_TOOLS

        table(
            "Plotter",
            [
                (
                    " / ".join(letter for _k, _l, letter in PLOTTER_TOOLS),
                    " / ".join(label for _k, label, _letter in PLOTTER_TOOLS),
                ),
                ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
                ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
                ("Ctrl+E", "Export to the library"),
                ("Ctrl+Shift+E", "Export a Tiled .tmx"),
                ("Ctrl+N / O / W", "New / open / close"),
                ("Ctrl+G", "Toggle the grid"),
                ("Ctrl+Tab", "Next map"),
                ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
                ("Space / middle drag", "Pan (wheel zooms)"),
            ],
        )
        table(
            "Packwright",
            [
                ("R", "Repack now"),
                ("Delete", "Remove the selected source"),
                ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
                ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
                ("Ctrl+E", "Export to the library"),
                ("Ctrl+Shift+E", "Export the atlas and its JSON"),
                ("Ctrl+N / O / W", "New / open / close"),
                ("Ctrl+Tab", "Next atlas"),
                ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
                # Middle drag alone, not "Space / middle drag" as Plotter's row
                # says: there is no space-pan in this mode to advertise.
                ("Middle drag", "Pan (wheel zooms)"),
            ],
        )
        self._draw_shortcut_rows(sections)
        imgui.end_popup()
        imgui.pop_style_var()

    def _draw_shortcut_rows(self, sections: list[tuple[str, list[tuple[str, str]]]]) -> None:
        """The filter box and whatever survives it."""
        from imgui_bundle import imgui

        from . import widgets
        from .tokens import sp

        imgui.set_next_item_width(sp(420))
        self._shortcuts_query = widgets.input_text(
            "##shortcuts-filter",
            self._shortcuts_query,
            max_length=60,
            hint="Filter shortcuts...",
        )
        kept = filter_shortcuts(sections, self._shortcuts_query)
        if not kept:
            widgets.muted("No shortcut matches that.")
            return
        for title, rows in kept:
            widgets.section(title)
            if imgui.begin_table(f"keys/{title}", 2):
                for keys, what in rows:
                    imgui.table_next_column()
                    widgets.muted(keys)
                    imgui.table_next_column()
                    imgui.text(what)
                imgui.end_table()

    def _diagnostics_popup(self, checks: list[Any]) -> None:
        from imgui_bundle import imgui

        from . import icons, theme, widgets
        from .tokens import sp

        ctx = self.app_ctx
        imgui.set_next_window_size((sp(460), 0))
        alpha, rise = widgets.popup_enter("diagnostics")
        # Translucent (UX.md Phase 5): cleared before ``begin`` paints it,
        # painted back below as a blur of the app or as the solid fill.
        frosted = widgets.frosted()
        if frosted:
            imgui.set_next_window_bg_alpha(0.0)
        imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
        if not imgui.begin_popup("diagnostics"):
            imgui.pop_style_var()
            return
        rounding = imgui.get_style().popup_rounding
        widgets.window_shadow("raised", radius=rounding)
        if frosted:
            widgets.window_backdrop(radius=rounding)
        if rise > 0.0:
            imgui.dummy((0, rise))
        widgets.section("Diagnostics")
        for check in checks:
            colour = theme.OK if check.ok else (theme.ERR if check.fatal else theme.WARN)
            # Lucide, as the status pills now are (UX.md Phase 2): "o" and "x"
            # were the last hand-spelled state glyphs in the app, and a lowercase
            # o at 11 px beside a red x is two letters rather than two shapes.
            widgets.text_colored(colour, icons.CHECK if check.ok else icons.CIRCLE_X)
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
        self._effective_config_section(ctx)
        self._toast_history_section(ctx)
        if ctx.state.dismissed_errors:
            # What Dismiss took off the banner (F59). Here rather than nowhere:
            # every writer of ``state.errors`` fires once, so clearing the list
            # used to destroy the only copy of the text -- and a dead worker is
            # reported through that list and through no doctor row at all, so it
            # was recoverable from nothing.
            widgets.section("Dismissed")
            for message in ctx.state.dismissed_errors:
                widgets.text_colored(theme.ERR, icons.TRIANGLE_ALERT)
                imgui.same_line()
                imgui.text_wrapped(message)
        imgui.separator()
        if imgui.button("Copy details"):
            imgui.set_clipboard_text(
                "\n".join(
                    f"{'ok' if c.ok else 'FAIL'} {c.name}: {c.detail}" for c in checks
                )
            )
        imgui.same_line()
        # Re-ask rather than wait out the poll (N111). The static half is only
        # recomputed on ``force``, which is what makes this button worth having
        # at all: having just installed the missing weights the popup names,
        # nothing short of a restart would otherwise change its mind.
        if imgui.button("Run checks again"):
            from ..service import system as svc_system

            ctx.submit("health", svc_system.current_checks, ctx.svc, force=True)
        imgui.same_line()
        if widgets.disabled_button("Open the log", _log_exists(ctx.runtime.config.data_dir)):
            ctx.open_log()
        imgui.same_line()
        # Chapter 12 (F57). The popup names the failing rows and their remedies;
        # what it cannot hold is what to do when a remedy does not take.
        from .manual import render as manual_render

        if manual_render.troubleshooting_button(ctx):
            imgui.close_current_popup()
        imgui.end_popup()
        imgui.pop_style_var()

    def _toast_history_section(self, ctx: Any) -> None:
        """Every notice this session raised, newest first (H67).

        In the diagnostics popup rather than behind a bell of its own: this is
        the same question as "what is wrong with my install" asked about the
        last ten seconds instead of about the machine, and a second icon in the
        top bar for it would be a second place to look. It sits *below* the
        checks for that reason -- the checks are the standing answer and this is
        the transient one.

        Collapsed by default, because on a healthy session it is a list of
        things that went right.
        """
        from imgui_bundle import imgui

        from . import theme, widgets
        from .tokens import sp

        log = ctx.state.toast_log
        if not log:
            return
        imgui.separator()
        if not imgui.collapsing_header(f"Notifications ({len(log)})##toast-history"):
            return
        now = time.monotonic()
        if imgui.begin_child("toast-history", (0, sp(160))):
            for entry in log:
                colour, glyph = widgets.toast_style(entry.level)
                widgets.text_colored(colour, glyph or "-")
                imgui.same_line()
                # Relative, not a clock time: what a reader wants from this
                # list is "was that the one from just now", and a wall clock
                # makes them do the subtraction.
                widgets.muted(_ago(now - entry.born))
                imgui.same_line()
                imgui.push_style_color(
                    imgui.Col_.text.value,
                    imgui.ImVec4(*theme.rgba(theme.TEXT if entry.level != "info" else theme.MUTED)),
                )
                imgui.text_wrapped(entry.text)
                imgui.pop_style_color()
        imgui.end_child()
        if imgui.small_button("Copy notifications"):
            imgui.set_clipboard_text(
                "\n".join(f"{e.level}: {e.text}" for e in reversed(log))
            )

    def _effective_config_section(self, ctx: Any) -> None:
        """What this process is running on, with the overridden rows marked.

        Collapsed by default and overridden rows first (S140). Thirty settings
        is a wall of text nobody reads; the two or three a host has actually
        changed are the whole diagnostic value, so they are what is visible when
        the section is opened, and the rest is there to confirm a suspicion
        rather than to be read through.

        Shares ``config.effective`` with ``warlock doctor``, which is the point
        of building the data source once: the copy a user pastes into an issue
        and the list they read on screen are the same answer.
        """
        from imgui_bundle import imgui

        from .panes import app_settings

        if not imgui.collapsing_header("Effective configuration"):
            return
        # The table itself lives in the app-Settings pane (K100), which is
        # where a user looking for configuration goes. It is drawn from here
        # too because this popup is where a user looking at a *failure* is,
        # and those are the same thirty rows -- so it is one function called
        # twice rather than two lists that would drift the first time a
        # variable was added.
        app_settings.config_table(ctx)

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
        from .panes import overlay

        texture = self.viewer.reference
        # UVs past 1.0 with the sampler set to repeat: one draw call, the
        # inker canvas's checkerboard idiom, rather than N**2 images that would
        # have to be positioned by hand and would show a seam of their own
        # wherever the arithmetic left a sub-pixel gap.
        repeat = 1
        if self.app_ctx.state.tile_preview and overlay.shows_tiled(
            self.app_ctx, self.app_ctx.job()
        ):
            repeat = overlay.TILE_REPEAT
        # Set on *both* branches, every frame. Turning the toggle on used to be
        # a one-way door: the sampler was switched to GL_REPEAT and never put
        # back, so the single-tile view that followed sampled a wrapped texture
        # at its own edges -- which is the one place a seamless tile is not
        # seamless, since bilinear filtering there blends the far edge in.
        # Idempotent and cheap: moderngl skips the GL call when the value is
        # already what it is being set to.
        texture.repeat_x = texture.repeat_y = repeat > 1
        scale = min(width / texture.size[0], height / texture.size[1])
        imgui.image(
            widgets.texture_ref(texture),
            (texture.size[0] * scale, texture.size[1] * scale),
            (0.0, 0.0),
            (float(repeat), float(repeat)),
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
            # Each mode's persist is its own step, so one raising cannot cost
            # the others -- but the *write* is one flush at the end, because
            # Settings holds the whole document and flushing per step wrote the
            # same file five times on the way out.
            _step("persist settings", lambda: self._persist(ctx))
            _step("persist inker", lambda: self._persist_inker(ctx))
            _step("persist clay", lambda: self._persist_clay(ctx))
            _step("persist plotter", lambda: self._persist_plotter(ctx))
            _step("persist packwright", lambda: self._persist_packwright(ctx))
            _step("write settings", ctx.settings.flush)
            if ctx.textures is not None:
                _step("release textures", ctx.textures.release)
            from .panes import sheet_panel

            _step("release sheet strip", lambda: sheet_panel.release_strip_texture(ctx))
            from . import packwright_mode, plotter_mode

            _step("release plotter textures", lambda: plotter_mode.release_all(ctx))
            _step("release atlas textures", lambda: packwright_mode.release_all(ctx))
        if self.viewer is not None:
            _step("release viewer", self.viewer.release)
        # ``getattr``, not an attribute access: teardown runs after a *failed*
        # setup too, and Clay's viewport is one of the last things constructed
        # -- an AttributeError here would skip runtime.shutdown, which is the
        # step that stops the worker loop and the trellis child.
        clay_view = getattr(self, "clay_view", None)
        if clay_view is not None:
            _step("release clay view", clay_view.release)
            # The ctx mirror dies with the view: a released view left on it
            # would hand the call sites dead GL objects, where None is the
            # answer every one of them already refuses.
            if ctx is not None:
                ctx.clay_view = None
        poser_viewer = getattr(self, "poser_viewer", None)
        if poser_viewer is not None:
            _step("release poser viewer", poser_viewer.release)
        if self.imgui_renderer is not None:
            _step("shutdown imgui", self.imgui_renderer.shutdown)
        _step("pygame.quit", pygame.quit)
        _step("runtime shutdown", self.runtime.shutdown)
        # The line whose *absence* is evidence: a session that ends without it
        # died somewhere no `except` could see.
        log.info("teardown complete")

    def _persist(self, ctx: Any) -> None:
        """The app's own settings. The write itself is teardown's last step.

        No mode: the app opens on Home every launch, so storing the one it
        happened to quit in would have no reader -- and quitting from the
        Manual or Settings would store a mode nothing would want restored.
        """
        from .settings import sanitise_form
        from .state import filters_to_store

        ctx.settings.set("show_fps", ctx.state.show_fps)
        ctx.settings.set("form_2d", sanitise_form(ctx.state.form_2d))
        ctx.settings.set("form_3d", sanitise_form(ctx.state.form_3d))
        ctx.settings.set("history", ctx.state.history)
        # Not ``vars``: the trash is a *view* rather than a filter, so quitting
        # from it must not reopen in it. See ``state.VOLATILE_FILTERS``.
        ctx.settings.set("filters", filters_to_store(ctx.state.filters))

    def _persist_inker(self, ctx: Any) -> None:
        from . import inker_mode

        inker_mode.persist(ctx)

    def _persist_clay(self, ctx: Any) -> None:
        from . import clay_mode

        clay_mode.persist(ctx)

    def _persist_plotter(self, ctx: Any) -> None:
        from . import plotter_mode

        plotter_mode.persist(ctx)

    def _persist_packwright(self, ctx: Any) -> None:
        from . import packwright_mode

        packwright_mode.persist(ctx)


# The diagnostics popup's log-file probe: (data dir, deadline, answer). One
# slot rather than a map -- a process has exactly one data directory, and a
# dict keyed by path would be a cache with no eviction for no benefit.
_LOG_PROBE: tuple[str, float, bool] = ("", 0.0, False)


def _log_exists(data_dir: Any) -> bool:
    """Whether there is a log file to open, re-stated at most once a second.

    The popup draws every frame while it is open, and this was a ``stat`` per
    frame for an answer that changes once per session -- when the first line is
    written. Cached rather than answered once, because on a clean data
    directory that first line lands *after* the window does, and a button
    permanently greyed because the popup happened to be open early is a worse
    failure than a second of staleness.

    A module function rather than a method: it needs no App, and the popup is
    drawn against stand-ins in the smoke suite.
    """
    global _LOG_PROBE
    key = str(data_dir)
    cached_key, deadline, exists = _LOG_PROBE
    now = time.monotonic()
    if key != cached_key or now >= deadline:
        exists = (Path(data_dir) / "warlock.log").exists()
        _LOG_PROBE = (key, now + LOG_STAT_SECONDS, exists)
    return exists


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
    from .. import migrate
    from ..config import get_config

    if migrate.MOVED:
        # Said twice on purpose. The move itself printed to stderr because it
        # happened before this handler existed; the log is where somebody looks
        # a week later to find out why their library is not where they left it.
        log.info("moved into %s: %s", get_config().home, ", ".join(migrate.MOVED))
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
