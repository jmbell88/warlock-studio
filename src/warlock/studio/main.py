"""The window, the frame loop, and everything wired together.

One pygame window with a GL 3.3 core context, one moderngl context over it, and
imgui drawing through that same context. The viewport is a texture the panels
show with ``imgui.image`` -- not a separate surface -- which is what makes
"panels over 3D" a layout question rather than a compositing one.

The frame is always the same six steps: collect finished tasks, refresh the job
cache, pump events, build the UI, render the viewport, present. Nothing in
those six may block.

**The function-local imports here are deliberate and they are not lazy.**
There are about 150 of them, and by frame 1 they buy nothing: ``journal``'s
``ensure_providers`` imports all six mode modules to register their document
kinds, so every module a ``from . import`` in this file names is already in
``sys.modules`` before the first frame draws. What they buy is *import order* --
this module is imported by the entry point before pygame has a display, and a
mode module pulled in at its top would drag imgui, moderngl and its own pane
tree into that moment. So they stay, and they are read as "imported at the
first call" rather than as a claim that the module might never load. The
alternative the review offered -- gating on ``state.inker is not None`` -- would
make the import conditional on state that says nothing about whether the module
is loaded, which is a worse lie than the one being replaced.
"""

from __future__ import annotations

import functools
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

from .. import memlog, winjob
from . import anchors, create_brief, filetypes, guard, probe, resources, tokens, viewer_embed
from . import fps as fps_mod

# The Ctrl+/ sheet's contents and its filter, in a file of their own since
# 2026-09-04 and named here where every caller already looks. See
# ``studio/shortcuts.py``.
from .clay_viewport import ClayViewport

# ``modal_open`` lives in ``dialogs`` now and is re-exported here, where every
# existing caller (and ``tests/test_mode_keys.py``) names it: the tour needs the
# same predicate, and a pane importing the frame loop for it is how a leaf comes
# to depend on the shell.
from .dialogs import modal_open
from .poser_viewport import PoserViewport
from .review_panes import ReviewPanes
from .shortcuts import filter_shortcuts, shortcut_sections

log = logging.getLogger(__name__)

WINDOW_TITLE = "Warlock Studio"
# How often the frame loop samples host memory. Long enough to be free, short
# enough that a 30-minute idle session yields 60 points to fit a slope through.
MEMORY_TICK_SECONDS = 30.0
# The task key the selection's GLB is parsed under. One key, so a selection
# moving faster than the disk cannot pile up loads: a refused submit is simply
# retried on the next tick, and a landed result is checked against
# ``viewer.pending`` before it is adopted.
#: ``viewer_embed.LOAD_KEY``, named here where the frame loop's three readers
#: already look. It moved so a mode module can ask for a picture without
#: importing the shell (``viewer_embed.request_reference``).
VIEWER_KEY = viewer_embed.LOAD_KEY
REVIEW_MESH_KEY = "viewer-review"
# The post-download re-probe. Its own key rather than "health"'s, so a slow
# forced verification cannot be mistaken for the periodic poll and dropped by
# key-dedupe while the user is watching for it (UX-09).
VERIFY_KEY = "verify-install"

#: Task keys whose **success** has nothing to do on the frame thread, listed so
#: that everything else arriving unclaimed can be reported. Each is silent for
#: its own reason, and the reasons are the point of the list:
#:
#: ``open-log``  ``os.startfile``. The outcome is a window the OS opened.
#: ``thumb:``    a card image written to disk. ``ThumbnailCache`` keys on
#:               mtime, so the library picks it up without being told.
#: ``derive:``   an artifact derived *inside* the job directory. Deliberately
#:               not ``save:`` -- ``app_ctx.derive_key`` says why -- because
#:               the user chose no destination and "Saved to <internal path>"
#:               is a sentence about a file they cannot find.
#: ``wrap:``     the wrap preview. The pane re-reads the file it asked for.
SILENT_TASK_KEYS = ("open-log", "thumb:", "derive:", "wrap:")


def _compare_key() -> str:
    """``library.COMPARE_KEY``, looked up lazily.

    A function rather than a module-level import: ``panes.library`` imports a
    great deal of the app and every other reference to it in this file is
    already deferred to its call site for that reason.
    """
    from .panes import library

    return library.COMPARE_KEY


def _import_mesh_key() -> str:
    """``library.IMPORT_MESH_KEY``, looked up lazily -- :func:`_compare_key`'s
    reason, and the same shape so the two read as one convention."""
    from .panes import library

    return library.IMPORT_MESH_KEY


DEFAULT_SIZE = (1600, 950)
MIN_SIZE = (1100, 700)
# Desired pane widths and their frame-local fit live in layout.py; named
# workspace layouts persist explicit horizontal and vertical splitter edits.
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
# categories partition ``modes.KEYS`` exactly, and the partition is the guard
# on ``_build_ui``'s dispatch.
#
# The Manual left this tuple when it stopped being a mode (the UI redesign,
# wave 3): it is drawn from ``_overlays`` now, so it has no dispatch branch
# to be reached by.
_SINGLE_PANE_MODES = ("home", "settings", "library")


# What a drop onto the window is allowed to be. The refusal message and every
# accept path have to agree about it (H71) -- and so do the file pickers, which
# is why the list itself lives in ``filetypes`` and this is a name for it
# rather than a copy of it.
DROPPABLE_IMAGES = filetypes.IMAGE_SUFFIXES


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


def _desktop_size(pygame: Any) -> tuple[int, int] | None:
    """The primary display's size in physical pixels, or None if SDL cannot say.

    ``get_desktop_sizes`` is the whole-display size rather than the work area,
    so this is a ceiling on what can be *asked for*, not a promise the window
    will not sit under the taskbar. That is the honest guarantee available: SDL
    exposes no work area, and a window one taskbar too tall is recoverable
    where one whose title bar is off the bottom of the screen is not.
    """
    try:
        sizes = pygame.display.get_desktop_sizes()
    except Exception:  # pragma: no cover - SDL without a display
        return None
    if not sizes:
        return None
    width, height = sizes[0]
    if width < 1 or height < 1:
        return None
    return (int(width), int(height))


def _window_size(
    stored: Any, *, override: Any, first_run_scale: float, desktop: tuple[int, int] | None
) -> tuple[int, int]:
    """The size to open at: validated, then clamped to the screen.

    Two separate bugs, both of which reached ``pygame.display.set_mode``.

    **The stored value was trusted.** ``Settings.load`` discards the whole file
    on a *version* mismatch, but a single malformed ``window_size`` in an
    otherwise-valid file sailed straight through to ``set_mode`` with no shape
    check and no floor -- and ``MIN_SIZE`` was enforced only on the live resize
    path, which is to say after the window already existed. A junk value there
    is not a cosmetic problem: if ``set_mode`` raises, ``run``'s handler reports
    the crash but never rewrites the key, so it recurs on every launch and a
    non-developer has no way back in. ``_ui_scale`` directly above already
    states the rule this setting was skipping -- *a junk value must not brick
    the window* -- so this is that rule, applied to the other stored geometry.

    **The default was never checked against the screen.** ``DEFAULT_SIZE``
    scaled by the monitor is 2000x1187 at the 125% Windows recommends for many
    1080p laptops, which does not fit a 1920x1080 panel; the unscaled 1600x950
    does not fit a 1366x768 one at all. Clamping is last so it applies to a
    stored size too -- the display a window was closed on may not be the
    display it reopens on.

    ``override`` wins outright and unclamped: it is the screenshot harness
    asking for an exact framebuffer, and a clamp there would silently produce
    shots of a size nothing asked for.
    """
    if override:
        return (int(override[0]), int(override[1]))
    default = (int(DEFAULT_SIZE[0] * first_run_scale), int(DEFAULT_SIZE[1] * first_run_scale))
    size = default
    try:
        width, height = (int(stored[0]), int(stored[1]))  # type: ignore[index]
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    else:
        floor = _min_window_size(first_run_scale)
        if width > 0 and height > 0:
            size = (max(width, floor[0]), max(height, floor[1]))
    if desktop is not None:
        size = (min(size[0], desktop[0]), min(size[1], desktop[1]))
    # Never zero, whatever the display claimed: ``set_mode((0, n))`` is a
    # fullscreen request to SDL, not a small window.
    return (max(size[0], 1), max(size[1], 1))


def _takes_pointer(target: Any, hovered: bool) -> bool:
    """The one hover/grab rule, for all three viewports.

    A viewport sees the mouse while the pointer is over it, and a gesture
    already in progress keeps it wherever the cursor goes -- so crossing onto
    a panel mid-orbit does not drop the drag. Written three times (the asset
    viewer, Clay's and Poser's) it drifted: only Clay's carried the
    ``tab.saving`` press gate, which is a *different* rule and stays where it
    is, beside the document it is about.
    """

    return bool(hovered or (target is not None and target.dragging))


def _ui_scale(settings: Any) -> float:
    """The stored multiplier, clamped. A junk value must not brick the window."""
    from . import tokens

    lo, hi = tokens.UI_SCALE_RANGE
    try:
        value = float(settings.get("ui_scale") or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return min(max(value, lo), hi)


def _split_column(
    ctx: Any,
    lay: Any,
    *,
    split_id: str,
    handle_length: float,
    width: float,
    edge: Any,
    top: tuple[str, Any, Callable[[Any], None]],
    bottom: tuple[str, Any, Callable[[Any], None]],
    before: tuple[str, Any, Callable[[Any], None], float] | None = None,
    middle: tuple[str, Any, Callable[[Any], None], float] | None = None,
    wanted: float | None = None,
    below_floor: float = 0.0,
) -> None:
    """One column of stacked panes with a drag handle between them.

    Every workspace builds the same shape twice -- a pane sized from a share,
    a pane taking what is left -- and each built it by hand. Two consequences
    the one function fixes at the source:

    * **A key per split.** ``split_id`` names *this* column, and the handle's
      id is derived from it (``f"{split_id}-share"``) rather than passed. So
      the two can no longer disagree, and a second column cannot be given the
      first one's key by copying the block.
    * **A handle at all.** Six of the workspaces drew a proportion the
      user could not change, because only the three columns that had a
      ``splitter`` call got one. It is drawn here, so having a split *is*
      having a handle.

    ``avail_y`` is captured before the first sized pane and after ``before``,
    which is the height the shares are really taken out of; measuring it after
    the top pane divides the drag delta by a height that pane already spent,
    and the handle then travels at the wrong rate.

    ``wanted``/``below_floor`` turn the plain proportion into Inker's give-way
    split, where a pane with a known minimum content height wins over the
    stored share and the pane beneath it keeps a floor of its own.
    """
    from imgui_bundle import imgui

    from . import layout as layout_mod
    from . import tokens

    imgui.begin_group()
    if before is not None:
        name, role, draw_fn, height = before
        with layout_mod.pane(name, (width, height), role, edge=edge) as visible:
            if visible:
                draw_fn(ctx)
    avail_y = imgui.get_content_region_avail().y
    if wanted is None:
        top_height = avail_y * lay.share(split_id)
    else:
        top_height = layout_mod.give_way(avail_y, lay.share(split_id), wanted, below_floor)
    with layout_mod.pane(top[0], (width, top_height), top[1], edge=edge) as visible:
        if visible:
            top[2](ctx)
    drag = layout_mod.splitter(f"{split_id}-share", vertical=False, length=handle_length)
    if drag and avail_y > 0:
        if wanted is None:
            share = lay.share(split_id) + drag * tokens.SCALE / avail_y
        else:
            # ``give_way_drag`` leaves the share alone whenever the pane under
            # the handle is pinned by its content and cannot follow the cursor.
            share = layout_mod.give_way_drag(
                avail_y, lay.share(split_id), wanted, below_floor, drag * tokens.SCALE
            )
        previous = lay.share(split_id)
        lay.set_share(split_id, share)
        if lay.share(split_id) != previous:
            lay.save()
    if middle is not None:
        name, role, draw_fn, height = middle
        with layout_mod.pane(name, (width, height), role, edge=edge) as visible:
            if visible:
                draw_fn(ctx)
    with layout_mod.pane(bottom[0], (width, 0), bottom[1], edge=edge) as visible:
        if visible:
            bottom[2](ctx)
    imgui.end_group()


def _right_column(
    ctx: Any,
    lay: Any,
    sidebar_w: float,
    *,
    inspector_draw: Callable[[Any], None],
    library_draw: Callable[[Any], None],
    share_key: str = "create-inspector",
) -> None:
    """The right sidebar: inspector on top, library on bottom.

    Kept as its own name because it is the one column a test drives directly
    -- it is :func:`_split_column` with Create's two panes filled in, and the
    geometry the frame draws is therefore the geometry the test measures.
    """
    from . import layout as layout_mod

    _split_column(
        ctx,
        lay,
        split_id=share_key,
        handle_length=sidebar_w,
        width=sidebar_w,
        edge=layout_mod.PaneEdge.LEFT,
        top=("inspector", layout_mod.PaneRole.INSPECTOR, inspector_draw),
        bottom=("library", layout_mod.PaneRole.SIDEBAR, library_draw),
    )


def _column_boundary(library: Any, workspace: str, side: str) -> None:
    """The draggable boundary between a side column and the centre anchor."""

    from imgui_bundle import imgui

    from . import layout as layout_mod

    imgui.same_line()
    layout_mod.column_splitter(library, workspace, side)
    imgui.same_line()


def _stage_pane(ctx: Any) -> None:
    """The settings column's body at the Create stage the user is standing on.

    A module-level function rather than a method for ``_right_column``'s
    reason: it needs no ``self``, and a test that wants to know every stage
    draws should call the dispatch the frame calls rather than a hand-copied
    reimplementation of it -- which is how the fifth stage comes to be missing
    from one of the two.

    The fall-through is Reference: the front of the pipeline, and the only
    stage that says something with nothing selected at all.
    """
    from . import icons, widgets
    from .panes import inspector, pose_panel, settings_2d, settings_3d, stage_rig

    stage = ctx.state.create_stage
    if stage == "mesh":
        settings_3d.draw(ctx)
    elif stage == "rig":
        stage_rig.draw(ctx)
    elif stage == "pose":
        job = ctx.job()
        if job is None:
            widgets.empty_state(
                icons.PERSON_STANDING, "No mesh selected.", "Pick a rigged mesh to pose it."
            )
        else:
            pose_panel.draw(ctx, job, hosted=True)
    elif stage == "export":
        job = ctx.job()
        if job is None:
            widgets.empty_state(
                icons.DOWNLOAD, "Nothing selected.", "Pick an asset to take files from it."
            )
        else:
            # The inspector's own grid, called rather than copied: it is the
            # one answer to "what can I take away from this", and a second
            # version of it is a second place for an artifact to be missed.
            inspector.downloads(ctx, job)
    else:
        settings_2d.draw(ctx)


class StartupRefused(Exception):
    """A named startup failure, with the sentence the user should read.

    ``_run_locked`` shows any exception out of ``App.__init__`` in a native
    dialog, which is already better than the log line it used to be -- but
    "AttributeError: 'NoneType' object has no attribute 'clear'" is a dialog
    the user still cannot act on. The two failures below are recoverable in
    principle and worth explaining, so they carry their own words.
    """

    def __init__(self, title: str, body: str) -> None:
        self.title = title
        self.body = body
        super().__init__(f"{title}: {body}")


class App(ClayViewport, PoserViewport, ReviewPanes):
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
        #: The caption's current state, so only a change is sent to the window
        #: manager. ``None`` until the first sync, which therefore always runs.
        self._title_marked: bool | None = None
        # The mode the last frame was built in, so a change into a viewport
        # mode can resync the viewer -- a mode change is not something the
        # job cache announces.
        self._last_mode: str | None = None
        # The Create stage the last frame was built in, for ``_last_mode``'s
        # reason and because the merge removed the mode change that used to
        # stand in for it: Reference -> Mesh is now one mode with a different
        # file under the viewport.
        self._last_stage: str | None = None
        # What the viewport was last asked to show. A selection change is not
        # announced by the cache, so without this the viewer only caught up on
        # the 3 s idle reread and the inspector described one asset while the
        # viewport drew another (UX-03).
        self._last_selected: str | None = None
        # How long the veil over the whole viewport takes to clear, or 0.0 for
        # "there is no transition running". A duration rather than a bool
        # because the two things that raise one want different lengths: a mode
        # switch is DUR_BASE, the splash dissolving into the app is DUR_SLOW.
        self._transition_duration = 0.0
        # Measured every frame, drawn only when state.show_fps is on (F10), and
        # logged once at teardown regardless -- the overlay answers "is it
        # smooth now", the log line is the evidence for "it ran at 60".
        self.fps = fps_mod.FpsMeter()
        # Task keys that have already been reported as arriving with nowhere to
        # go, so the report is once per key rather than once per arrival. See
        # the tail of ``_on_task_done``.
        self._unclaimed: set[str] = set()
        # One sampler for the app, because the CPU figure is a delta between
        # calls: two owners sharing a baseline would each eat the other's
        # interval. Ticked from ``_tick`` at one second, drawn by
        # ``status_bar.draw``.
        self.resources = resources.Sampler()
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

    def setup_window(self, *, size_override: tuple[int, int] | None = None) -> None:
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
        from . import layouts as layouts_mod
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
        size = _window_size(
            settings.get("window_size"),
            override=size_override,
            first_run_scale=first_run_scale,
            desktop=_desktop_size(pygame),
        )
        try:
            self.window = pygame.display.set_mode(
                size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
        except Exception as exc:
            # The GL attributes above ask for a 3.3 core context, and SDL
            # refuses here rather than degrading if the driver cannot give one.
            # It arrived as the generic "could not start" box, which says
            # nothing a user can act on -- and this is one of the few startup
            # failures that genuinely has a remedy.
            raise StartupRefused(
                "Warlock Studio needs OpenGL 3.3",
                f"{exc}\n\nWarlock draws its whole interface on the GPU and "
                "could not create an OpenGL 3.3 window.\n\nThis usually means "
                "the graphics driver is missing or out of date. It can also "
                "happen over Remote Desktop or in a virtual machine, where the "
                "session offers no hardware OpenGL.",
            ) from exc
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

        try:
            self.ctx = moderngl.create_context()
        except Exception as exc:
            # The window opened and the context still cannot be adopted, which
            # is a narrower fault than the one above -- a driver that advertises
            # 3.3 and does not deliver it. Same remedy, so the same sentence,
            # said separately because the two fail at different lines and a
            # log-reader should be able to tell them apart.
            raise StartupRefused(
                "Warlock Studio needs OpenGL 3.3",
                f"{exc}\n\nWarlock opened a window but could not use the "
                "OpenGL context behind it.\n\nThis usually means the graphics "
                "driver is missing or out of date. It can also happen over "
                "Remote Desktop or in a virtual machine, where the session "
                "offers no hardware OpenGL.",
            ) from exc
        imgui.create_context()
        imgui.get_io().set_ini_filename("")  # imgui's own layout file is not ours to keep
        # Before anything can raise inside a frame. imgui's error-recovery
        # assert defaults *on*, and under imgui-bundle an IM_ASSERT surfaces as
        # a RuntimeError -- so left alone, the unwind that saves a broken pane
        # is itself the exception that ends the session. See ``studio/guard.py``.
        guard.configure()
        # Keyboard navigation, app-wide (UX-02). It was off, and the shortcut
        # sheet made support look broader than it was: Settings, Profiles, the
        # library, the inspector and the mode switch had no focus traversal at
        # all, so a keyboard-only user could reach the two forms ``focus.py``
        # hand-rolls an order for and nothing else.
        #
        # Safe to switch on now only because ``imgui_backend`` arbitrates the
        # arrows and Space, which five surfaces already bind -- see
        # ``_NAV_KEYS`` there for the rule and why it lives at that door.
        imgui.get_io().config_flags |= imgui.ConfigFlags_.nav_enable_keyboard.value
        try:
            fonts.load(imgui)
        except fonts.FontsUnavailable as exc:
            # These ship in the wheel -- the offline invariant covers fonts as
            # much as weights -- so this is a partial install, an antivirus
            # quarantine or a half-copied directory, and every one of those is
            # fixed by reinstalling rather than by reading a stack trace.
            raise StartupRefused(
                "Warlock Studio is missing part of its installation",
                f"{exc}\n\nThese files ship with Warlock and are never "
                "downloaded, so one going missing means the installation is "
                "incomplete -- an interrupted install, or an antivirus tool "
                "that quarantined them.\n\nReinstalling Warlock restores them.",
            ) from exc
        theme.apply(imgui)
        self.layout = Layout(settings)
        # Saved arrangements within the fixed three-column skeleton (wave 5).
        # A separate object from ``Layout``, which owns the *proportions*: one
        # is a preference the user drags and the other is a named thing they
        # switch between, and the settings keys are top-level for the reason
        # ``layouts.py`` states.
        self.layouts = layouts_mod.Library(settings)
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
        from . import motion, textures
        from .app_ctx import Ctx
        from .jobs_cache import JobsCache
        from .settings import as_list, restore_form
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
        # Absent means on: this defaults to shown, so a settings file written
        # before it existed must not read as "the user turned it off".
        stored_resources = settings.get("show_resources")
        state.show_resources = True if stored_resources is None else bool(stored_resources)
        # Both halves, here rather than at the checkbox: the stored value has to
        # reach ``motion.REDUCED`` before the first frame is built, or the app
        # animates its own startup at somebody who asked it not to.
        state.reduce_motion = bool(settings.get("reduce_motion"))
        motion.set_reduced(state.reduce_motion)
        state.form_2d = restore_form(default_form_2d(), settings.get("form_2d"))
        state.form_3d = restore_form(DEFAULT_FORM_3D, settings.get("form_3d"))
        state.history = [str(entry) for entry in as_list(settings.get("history"))]
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
        from .panes import first_run

        # Sampled once. The marker is intentionally outside studio_settings:
        # resetting preferences must not turn setup into an annual popup.
        self.app_ctx.first_run = first_run.pending(self.svc.config)
        device = getattr(self.runtime, "device_memory", None)
        self.app_ctx.gpu_name = str(getattr(device, "name", "") or "")
        self.app_ctx.dpi_scale = monitor_scale
        # The status bar reads the sampler off the Ctx; the App owns it,
        # because the CPU figure is a delta and one owner has to tick it.
        self.app_ctx.resources = self.resources
        self.app_ctx.layout = self.layout
        self.app_ctx.layouts = self.layouts
        # Every ``widgets.field_error`` call site gets the Install offer at
        # once, without widgets importing a pane. Bound to this Ctx, so a
        # second App in one process replaces it rather than stacking.
        from . import widgets as widgets_mod
        from .panes import model_gate

        widgets_mod.set_install_offer(lambda field: model_gate.install_offer(self.app_ctx, field))
        # And the mode gate, at the one door every switch already goes through
        # (H14). Bound the same way and replaced the same way; ``state`` itself
        # must not learn what a Ctx is.
        from .state import set_mode_gate

        set_mode_gate(lambda key: not model_gate.mode_block(self.app_ctx, key))
        self.app_ctx.load_presets = self.load_presets
        self.app_ctx.refresh_rig_data = self._refresh_rig_side_data
        self.eta = Eta()
        self._load_static_answers()
        if self.app_ctx.first_run:
            self.app_ctx.first_run_info = first_run.snapshot(self.app_ctx)
        # Off the frame thread (C32): the walk stats every file under every
        # job directory, and nothing on the first frame needs the number --
        # the library's storage line simply appears when the task lands.
        self._request_storage()
        # Which tours have already been finished. Read here rather than
        # defaulted empty, so Home stops offering one the reader has done.
        from .panes import tour as tour_pane

        tour_pane.restore(self.app_ctx)
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
        ctx.ask_quit = self._ask_quit
        ctx.clear_viewport = self._clear_viewport
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
        self._report_failed_checks()

    def _report_failed_checks(self) -> None:
        """Every failing fatal row (and the trellis port) onto the banner.

        Also after each health poll: ``note_error`` deduplicates, so a row
        still failing costs nothing, and a row that turned fatal once torch
        imported -- no CUDA -- used to show only as a Home chip.
        """
        ctx = self.app_ctx
        # ``pending_install`` is excluded, and it is the reason this filter
        # exists in this shape: a fresh install has every model row failing,
        # and banner-ing them meant the first thing a new user saw was a red
        # wash listing downloads they had not made yet. Those rows are offered
        # by the first-run panel and by Home's status row instead.
        failed = [
            c
            for c in self.runtime.checks
            if not c.ok
            and not c.pending_install
            and (c.fatal or c.name == "trellis port")
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
        # Snapshot rather than iterate live: register_imported_loras/
        # remove_imported_lora can mutate this table from another thread
        # between frames (see models.STYLE_LORAS_LOCK).
        ctx.style_loras = [("", "no style LoRA")] + [
            (k, spec.label) for k, spec in models.style_loras_snapshot().items()
        ]
        # The Settings pane draws this and may not ask the service itself: it
        # is a pane, and ``recommended_base`` needs a resolved Plan. Empty when
        # there is no plan, which is the pane's "say nothing" value.
        ctx.recommended_base_label = (
            models.BASE_MODELS[vram.recommended_base(plan_)].label if plan_ is not None else ""
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
        crashed = False
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
            crashed = True
            if in_setup:
                log.exception("Warlock Studio could not start")
            else:
                log.exception(
                    "the frame loop crashed mid-session after %d frames (%.1f s up)",
                    self.fps.frames,
                    time.perf_counter() - self._started_at,
                )
        finally:
            self.teardown()
            if crashed:
                # *After* teardown, which is the whole reason the flag exists
                # rather than the report living in the except block: the
                # journal's last write goes through the task runner, and this
                # sentence counts what is on disk. Reporting first would tell
                # the user about a copy that had not been taken yet.
                self._report_crash(in_setup)
        return rc

    def _report_crash(self, in_setup: bool) -> None:
        """Say the app crashed, and offer the log (UX-06).

        The window has gone by now -- that is what a crash looks like from
        outside, and it is precisely the problem: the app vanished and the one
        artefact that could explain it was in a file the user had no reason to
        know about. A native box, because the GL context and imgui are gone.

        The journal sentence is computed rather than promised: "your work is
        safe" is only worth saying when it is true, and "nothing was waiting"
        is the honest answer the rest of the time.
        """
        from .. import instance
        from . import journal

        try:
            # ``self.runtime.config`` -- App has no ``config`` of its own, and
            # reading one here was an AttributeError this except swallowed, so
            # the dialog's "Open the log folder?" could never actually open it.
            data_dir = Path(self.runtime.config.data_dir)
        except Exception:  # noqa: BLE001 -- a crash report must not crash
            data_dir = None
        try:
            note = journal.status_line(self.app_ctx) if self.app_ctx else ""
        except Exception:  # noqa: BLE001
            note = ""
        when = "while starting" if in_setup else "and had to close"
        instance.alert_fatal(
            "Warlock Studio",
            f"Warlock Studio ran into a problem {when}.\n\n"
            + (note + "\n\n" if note else "")
            + "The details are in warlock.log. Open the log folder?",
            log_dir=data_dir,
        )

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
        # Troupe plays its sheet with no input at all, and ``advance`` only
        # runs inside the preview's draw -- so a skipped frame does not advance
        # playback, it *drops* it. Throttled to IDLE_FPS the preview becomes
        # coarse catch-up jumps that can step straight over the frame being
        # judged, which is the one thing the mode exists to make obvious.
        if state.mode == "troupe" and getattr(state.troupe, "playing", False):
            return True
        # Sirens for the same reason, and it was missing: the playhead is drawn
        # from the mixer's clock and nothing else moves, so at IDLE_FPS the row
        # cursor crawled down the pattern at 12 fps while the audio ran at full
        # speed -- the one readout that says *where in the song you are*,
        # visibly disagreeing with what you can hear.
        #
        # **Muse, word for word.** Its player draws a playhead from the same
        # clock across the same kind of picture, and nothing else on that
        # screen moves either. The two audio modes share one predicate because
        # they share one argument.
        if state.mode in ("sirens", "muse"):
            from . import sirens_audio

            if sirens_audio.playing():
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
        # Child commit is included: Warlock's subprocesses are not incidental
        # (the BiRefNet matting worker measured 6.5 GiB of private commit on
        # 2026-08-21), and a line reporting only ``private`` understated this
        # app's charge against the commit limit by 40% on the session that
        # prompted the reading.
        #
        # Over ``measured_pids()`` rather than ``tracked()``: the latter holds
        # the pids ``Popen`` returned, which under a uv venv are trampolines
        # rather than the interpreters holding the weights
        # (docs/measurements/2026-08-22-trampoline-child-pids.md).
        summary = memlog.summary(children=winjob.measured_pids())
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

        from . import imgui_backend, modes

        self.app_ctx.state.frame_index += 1
        # The caption tracks every unsaved document, not only a pose, so it is
        # sampled per frame rather than pushed from one callback. Cheap: it
        # returns immediately unless the answer changed.
        self._sync_title()
        self.app_ctx.textures.begin_frame()
        # Here rather than in ``_tick``: every path that draws a frame goes
        # through this method, and ``_tick`` belongs to the run loop alone --
        # the screenshot harness calls ``frame`` directly, and a meter that
        # was blank in every shipped picture is how that was found. Gated on
        # the setting, so the opt-out costs nothing at all; the sampler itself
        # is two ctypes calls and a driver ioctl, 0.047 ms measured, behind a
        # one-second cadence (see ``resources.Sampler.sample``).
        if self.app_ctx.state.show_resources:
            # The frame rate is handed over rather than measured there: the
            # meter is the frame loop's, and ``resources`` is pinned free of
            # pygame. ``None`` before the first recorded frame -- which is
            # every screenshot-harness frame, since the harness calls ``frame``
            # directly and never ``_tick`` -- so the segment is simply absent
            # rather than reading a confident 0.
            self.resources.tick(fps=self.fps.fps if self.fps.frames else None)
        self._collect_tasks()
        self._refresh()
        # Before ``_events``, which is where the keys are read: whether the
        # arrows reach imgui at all is a property of the surface they arrive
        # at, so it has to be settled for this frame before any of them is
        # dispatched (UX-02).
        imgui_backend.reserve_nav_keys(self.app_ctx.state.mode in modes.NAV_KEY_MODES)
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
            try:
                fonts.reload(imgui)
            except Exception:
                # Mid-session, from the UI-scale slider, and *not* fatal. The
                # atlas either kept the faces it had (the files-missing check
                # runs before ``clear_fonts``) or is empty and imgui falls back
                # to its own default font, which is the state every headless
                # test already runs in. Taking the session down over a type
                # ramp -- with an unsaved document open in every editor -- is
                # the wrong trade by a wide margin.
                log.exception("could not re-bake the font atlas")
                self.app_ctx.toast(
                    "The interface font could not be reloaded at this size.",
                    "error",
                    "log",
                )
        imgui.new_frame()
        self._build_ui()
        imgui.render()
        # After the frame, because ``want_text_input`` is only true once the
        # field that wants it has been drawn (UX-19). SDL emits no TEXTINPUT
        # while text input is stopped, so this is what makes typing work --
        # and stopping it again is what keeps an IME's candidate window off
        # the viewport while nobody is typing.
        imgui_backend.sync_text_input()

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
        # Not while a button is held: the debounced flush is a JSON encode of
        # the whole settings document plus an atomic file write, and a splitter
        # drag dirties the layout on every frame it moves -- so the write
        # landed once a second *inside* the drag, on the frame thread, as a
        # hitch under the pointer. Deferred to release, where the same flush
        # happens once. ``flush`` on exit covers a drag that ends the session.
        if not imgui.is_any_mouse_down():
            self.app_ctx.settings.tick()
        # One toast per problem, polled rather than pushed: ``Settings`` is a
        # plain file object with no way to reach the UI, and both of the things
        # it has to report -- a file that could not be read at startup, and one
        # that cannot be written now -- used to be log lines nobody saw while
        # every preference silently reverted or stopped persisting (UX-10).
        notice = self.app_ctx.settings.take_notice()
        if notice is not None:
            self.app_ctx.toast(notice, "error")

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
                    # ``rows`` is the refusal's other half: which registry rows
                    # would fix it. Written since the class was, and until now
                    # read by nothing outside the tests -- so "you haven't got
                    # these weights" arrived as a sentence with no action.
                    rows = tuple(getattr(done.error, "rows", ()) or ())
                    gib = 0.0
                    if rows:
                        try:
                            from ..service import downloads as svc_downloads

                            gib = svc_downloads.needed_gib(ctx.svc, list(rows))
                        except Exception:
                            # Path arithmetic over ~17 registry entries, but a
                            # figure is a courtesy: an unknown row must not
                            # cost the user the button as well as the number.
                            log.exception("could not size a refusal's install")
                    ctx.state.note_field_error(named, done.message or "", rows, gib)
                elif done.key == "submit":
                    # A refusal with no control to point at -- the VRAM door is
                    # the one of these, by its own recorded argument. It is
                    # kept so the plan block can say it, because a toast cannot
                    # hold ``vram.shortfall_message``'s list of remedies and
                    # the block was going on saying "Ready to generate."
                    ctx.state.submit_refusal = done.message or ""
                message = done.message or "That did not work."
                action = done.action
                if done.key.startswith("journal:"):
                    # The eighth prefix, and the one nobody had claimed. A
                    # journal write is the only task in the app the user did
                    # not start, and since the mark stopped advancing on
                    # *submit* (``journal.write``) its failure is also the only
                    # signal that the crash copy the app promised does not
                    # exist. "Something went wrong; see the log for details"
                    # names neither half of that, and it is the sentence that
                    # was being shown.
                    #
                    # No routing call beside it: the journal has no per-mode
                    # state to unlock, and the retry is already the debounce's
                    # -- ``_write_if_due`` comes back to this slot in
                    # JOURNAL_SECONDS whether or not anything is told here.
                    message = (
                        "Autosave could not write a recovery copy. Save your "
                        "work somewhere you choose."
                    )
                    action = "log"
                ctx.toast(message, "error", action)
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
                elif done.key.startswith("sirens-"):
                    from . import sirens_mode

                    # Same rule, plus one of its own: a failed *render* has to
                    # clear ``rendering`` and record why, or the transport
                    # shows a dead Play button with nothing beside it.
                    sirens_mode.on_task_failed(ctx, done)
                elif done.key.startswith("troupe-"):
                    from . import troupe_mode

                    # Both of Troupe's tasks are *doors*, so a failure here is
                    # always a refusal with a sentence in it -- and one the
                    # user is owed, since neither door's button can know in
                    # advance which of its options the service will object to.
                    troupe_mode.on_task_failed(ctx, done)
                elif done.key.startswith(("download:", "remove:")):
                    # A failed fetch has to be *routed* somewhere, not merely
                    # toasted: the rows carry a presence flag, and a fetch that
                    # got partway before failing has changed what is on disk.
                    # Re-probing costs a few stats and is the only thing that
                    # stops the pane staying optimistic about a download that
                    # did not happen. (Only the rows, not doctor's whole
                    # suite -- nothing succeeded, so the health state has
                    # nothing new to say.)
                    #
                    # A failed *removal* is the same fact from the other side,
                    # and more sharply so: ``uninstall`` renames a directory
                    # out of the way before it deletes it, so a failure part
                    # way through has already made the model absent.
                    self._refresh_model_answers()
                elif done.key == REVIEW_MESH_KEY:
                    self._adopt_review_model(done)
                elif done.key == VIEWER_KEY:
                    # ``pending`` still names the file, so nothing would retry
                    # it; that is the intent (the parse would fail again), but
                    # the flag has to come down or the next *different* asset
                    # is refused as a duplicate of this one.
                    if self.viewer.pending == done.tag:
                        self.viewer.pending = None
                        self.viewer.clear()
                        self.viewer.path = done.tag
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
                elif done.key.startswith("matte-"):
                    from . import matte_preview

                    # The seventh, and it was missing. ``matte_preview.pump``
                    # re-submits whenever there is no cached cutout for the
                    # current stamp, and ``settings_3d.matte_modal`` runs it
                    # every frame -- so a failure that left no note behind was
                    # re-submitted, re-failed and re-toasted at the frame rate.
                    matte_preview.on_task_failed(ctx, done)
                continue
            try:
                self._on_task_done(done)
            except Exception:
                # A handler that raises here used to end the session: this is
                # the frame loop's own thread, and the outer ``try`` in
                # ``App.run`` treats any escaped exception as the app dying,
                # not one task's landing going wrong (finding #1). Logged and
                # toasted the way ``guard`` announces a tripped pane -- the
                # task itself already finished, so there is nothing left to
                # retry, only the fact that its landing broke to report.
                log.exception("landing task %r raised", done.key)
                self.app_ctx.toast(f"That did not finish landing: {done.key}.", "error", "log")

    def _on_task_done(self, done: Any) -> None:
        ctx = self.app_ctx
        key = done.key
        if key == "preview" and isinstance(done.result, dict):
            ctx.state.preview.update(done.result)
            return
        if key == "health":
            # The status surfaces read runtime.checks each frame; replacing the
            # list wholesale is atomic enough for all of them.
            if isinstance(done.result, list):
                self.runtime.checks = done.result
                # The rows that were "still checking" at startup have their
                # answer now: a newly fatal one (no CUDA) joins the banner, and
                # the first-run panel's verdicts are retaken from the same list.
                self._report_failed_checks()
                if getattr(ctx, "first_run", False):
                    from .panes import first_run

                    ctx.first_run_info = first_run.snapshot(ctx)
                # The first poll is also what pays for the deferred bpy probe
                # (C30). If it says rigging works and the ctx does not yet,
                # re-ask for the templates -- the probe's answer is cached, so
                # the re-ask costs a directory read.
                blender_ok = any(c.name == "Blender (rigging)" and c.ok for c in done.result)
                if blender_ok and not ctx.rigging_available:
                    from ..service import rig as svc_rig

                    ctx.submit("rig-templates", svc_rig.rig_templates, self.svc)
            return
        if key == "model-storage":
            if isinstance(done.result, dict):
                ctx.model_storage = done.result
            return
        if key == "library-verify":
            self._report_library_check(done.result)
            return
        if key == "library-backup":
            out = done.result if isinstance(done.result, dict) else {}
            if out:
                from .state import format_bytes

                ctx.toast(
                    f"Library index backed up to {out['dir']} "
                    f"({format_bytes(int(out['store_bytes']))}).",
                    "success",
                )
            return
        if key == "sweep-staging":
            # Silent when there was nothing to reclaim, which is every launch
            # that did not follow a cancelled fetch. Said out loud when there
            # was: disk quietly reappearing is the kind of thing a user should
            # be told about rather than discover in a folder listing.
            removed = done.result if isinstance(done.result, list) else []
            if removed:
                noun = "tree" if len(removed) == 1 else "trees"
                ctx.toast(f"Reclaimed {len(removed)} staging {noun} left by a cancelled download.")
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

            # Off the frame thread. ``force=True`` re-runs *every* probe,
            # including the slow ones the startup path deliberately defers --
            # the torch import and the bpy subprocess, which is seconds of
            # frozen window on the frame that is supposed to say "Download
            # finished" (UX-09). The model answers below are refreshed when the
            # probe lands, so the pane catches up a moment later instead of the
            # whole app stopping for it.
            ctx.submit(VERIFY_KEY, svc_system.current_checks, self.svc, force=True)
            ctx.tasks.set_progress(VERIFY_KEY, 0.0, "Verifying installation...")
            # The untick happens when the probe lands (see ``VERIFY_KEY``
            # above), because the rows it reads are derived from the checks it
            # is still computing. Unticking against the *old* answers would
            # leave every row exactly as it was: the plan is deduped, so
            # fetching one SDXL 1.0 recipe satisfies the other three, and
            # leaving them ticked offers to download 7 GB that is already there.
            ctx.toast("Model removed." if key.startswith("remove:") else "Download finished.")
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
        if key.startswith("journal:"):
            # The mark, on the frame thread. A copy that landed changes nothing
            # on screen, which is why this key was in ``SILENT_TASK_KEYS`` until
            # the review's T3 moved the three slot attributes off the task
            # thread -- the write still says nothing, but the slot has to be
            # told here, beside ``drop``, rather than from the pool.
            from . import journal

            journal.on_task_done(ctx, done)
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
        if key.startswith("muse-"):
            from . import muse_mode

            muse_mode.on_task_done(ctx, done)
            return
        if key.startswith("sirens-"):
            from . import sirens_mode

            sirens_mode.on_task_done(ctx, done)
            return
        if key.startswith("troupe-"):
            from . import troupe_mode

            troupe_mode.on_task_done(ctx, done)
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
            # The press was taken, so whatever the last one was refused for is
            # no longer the state of things.
            ctx.state.submit_refusal = ""
            ctx.cache.invalidate()
            # Say where in line it landed: five rapid submits used to produce
            # five identical "Queued." toasts and no sense of depth.
            # ``+ 1`` for the job this submit just created. ``invalidate()``
            # above only marks the cache dirty -- it does not reread -- so
            # ``cache.jobs`` here is still the page from *before* this submit,
            # and the count was short by exactly one every time: the first
            # submit of an idle queue counted 0 and said "Queued." while two
            # jobs were in line (UX-25).
            waiting = sum(1 for j in ctx.cache.jobs if j.get("status") == "queued") + 1
            ctx.toast("Queued." if waiting <= 1 else f"Queued - {waiting} jobs in line.")
            return
        if key.startswith(("save:", "bake:", "sheet-save:")) or key == "screenshot":
            # ``screenshot`` was outside this and outside every other branch,
            # so a viewport capture the user had just chosen a destination for
            # finished in silence -- the one shape a save must never have,
            # since the only other thing that looks like it is a save that did
            # not happen. ``None`` is a cancelled picker and says nothing,
            # which is what the other three already rely on.
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
            # land here, each as a *reading* -- the sizes, or the one directory
            # to fold in, or why neither. The amendment happens here rather
            # than in the task because the library's size sort reads those
            # sizes and the memo key reads their generation (T3).
            ctx.cache.adopt_storage(done.result)
            return
        if (
            key.startswith(
                ("delete:", "prune", "rename:", "name:", "tags:", "fav:", "restore:", "purge:")
            )
            or key == "empty-trash"
        ):
            ctx.cache.invalidate()
            # ``restore:`` and ``purge:`` were missing from both lists, and each
            # absence showed differently. Restore (the toast's Undo, and the
            # trash's own button) left the row looking untouched for up to the
            # 3 s cache backstop, so the action read as inert and users clicked
            # it twice. Purge and Empty trash are worse: they are the actions
            # that actually *free disk*, and nothing re-measured -- so the
            # "N jobs - X GB" footer kept the pre-delete figure for the rest of
            # the session, which is the one number the whole affordance exists
            # to move (UX-11).
            if key.startswith(("delete:", "prune", "purge:")) or key == "empty-trash":
                self._request_storage()
            return
        if key == _import_mesh_key():
            # A new finished row, so the list has to refetch exactly as it does
            # after a delete or a rename. ``None`` is the picker being
            # cancelled and is silent: a toast saying nothing happened, after
            # the user chose for nothing to happen, is noise.
            if done.result is None:
                return
            ctx.cache.invalidate()
            self._request_storage()
            # Selected, because an import is a thing the user just *made* and
            # the next click is always on it -- the same reasoning ``create``
            # follows when a job it queued lands.
            ctx.state.select(done.result["id"])
            ctx.toast("Mesh imported. It is an ordinary asset now.", "success")
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
        if key == REVIEW_MESH_KEY:
            self._adopt_review_model(done)
            return
        if key == _compare_key():
            self._adopt_compare(done)
            return
        if key == VERIFY_KEY:
            # The slow half of the post-download refresh, landing off the frame
            # thread (UX-09). Everything that reads the checks happens here, so
            # the pane updates in one step rather than half now and half later.
            if done.result is not None:
                self.runtime.checks = done.result
            self._refresh_model_answers()
            present = {row["row_key"] for row in ctx.model_rows if row.get("present")}
            ctx.model_picks -= present
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
            return
        if key.startswith(SILENT_TASK_KEYS):
            return
        # Nothing claimed it. This module's own rule -- "the app claims results
        # by prefix, and a key without one is a result delivered nowhere" --
        # was enforced by nothing whatever, so a result arriving under an
        # unclaimed key was indistinguishable from one that had been handled,
        # in the one place where the difference is invisible from outside:
        # the *success* path. The two ways to get here are a mode closing
        # while its own task was in flight, and a new key whose author forgot
        # this file exists.
        #
        # Once per key rather than once per arrival: several of these are
        # resubmitted by a pane that runs every frame, and a line a frame is a
        # log nobody can read. ``log.info`` and not a warning, because a
        # deliberately silent key that nobody added to ``SILENT_TASK_KEYS``
        # would otherwise cry wolf for the life of the session.
        if key not in self._unclaimed:
            self._unclaimed.add(key)
            log.info("a %r task finished with nowhere to deliver its result", key)

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
        # flight, so a slow encode skips a beat rather than queuing.
        #
        # **This import is eager, and the comment that called it lazy was
        # wrong** (the review's theme T5, settled 2026-09-03 by saying so).
        # ``_refresh`` runs every frame, so every mode module is imported on
        # frame 1 whether or not its mode is ever opened -- and it would be
        # even without this line, because ``journal.snapshot`` below reaches
        # ``ensure_providers``, which imports all six to register their kinds
        # before the first recovery scan. Gating on ``state.inker is not None``
        # would therefore save nothing at all while adding a condition to read.
        # What is genuinely lazy is ``ensure``'s *state*, not this import.
        from . import inker_mode, journal

        # Every registered document kind, not only Inker (UX-05). Importing
        # ``inker_mode`` is what registers its provider; the other modes
        # register theirs the same way, lazily, so a session that never opens
        # Clay pays for nothing.
        journal.pump(self.app_ctx)
        # Beside it, and in every mode for the same reason: an export flattens
        # one frame per app frame rather than a whole clip on the frame the
        # button was clicked, and a user who started one and switched to the
        # library must still get their file.
        inker_mode.pump_export(self.app_ctx)
        # And beside it: the history drops its oldest steps when they get too
        # big to hold, and the press that did it is routinely the last thing
        # the user does in Inker before switching away.
        inker_mode.pump_undo_trim(self.app_ctx)
        # Scanned here, on the first frame that has a Ctx, and *offered* by the
        # home screen rather than by a modal in front of it. It has to be here
        # and not in the pane: the autosave directory is also where this
        # session's copies land, so a scan taken any later than the first frame
        # would hand the user their own open documents back. ``snapshot`` is a
        # no-op after the first call, which is what makes calling it per frame
        # correct rather than merely cheap.
        journal.snapshot(ctx)

    def _report_library_check(self, report: Any) -> None:
        """One line on screen, the whole report in the log.

        A findings *list* wants a modal with columns and a way to act on each
        row, and none of the five findings has an action this app should take on
        the user's behalf -- deleting an orphan directory or a stale verdict is
        exactly the bulk gesture that caused the 2026-08-09 loss the check
        exists to surface. So the pane says how many and where to read them, and
        ``warlock library verify --json`` is the surface that hands the detail
        to something that can act.
        """
        ctx = self.app_ctx
        if not isinstance(report, dict):
            return
        checked = report.get("checked", 0)
        noun = "asset" if checked == 1 else "assets"
        if report.get("ok"):
            ctx.toast(f"Library intact - {checked} {noun} checked.", "success")
            return
        log.warning("library verify: %s", json.dumps(report, default=str))
        findings = report.get("findings", 0)
        thing = "finding" if findings == 1 else "findings"
        ctx.toast(
            f"{findings} {thing} across {checked} {noun}. The detail is in warlock.log.",
            "warn",
            action="log",
        )

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

        **Every kind of unsaved, not only a pose.** The mark used to be
        ``pose_dirty`` alone, so a dirty drawing, sculpt, map, atlas or song
        left the caption clean -- the one place in the app that answers "have I
        saved this" at a glance, saying no to five of the six things it could
        be about. Derived from the quit-guard predicate
        (``docmodes.any_unsaved``), so the caption and the question asked on
        the way out cannot disagree.

        Swallows its own failure: a caption is not worth taking a frame down
        for, and this is reachable from a viewer callback that knows nothing
        about whether a display still exists (teardown releases the viewer
        after pygame has quit).
        """
        from . import docmodes

        ctx = self.app_ctx
        state = ctx.state if ctx is not None else None
        marked = bool(
            state is not None and (state.pose_dirty or docmodes.any_unsaved(ctx))
        )
        # Called once a frame now that it tracks five more things; setting the
        # caption is a window-manager round trip, so only a *change* is sent.
        if marked == self._title_marked:
            return
        self._title_marked = marked
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

    def _clear_viewport(self) -> None:
        """Empty the canvas of whichever Create stage is on screen.

        The hard part is not the clearing but making it *stay* clear.
        ``_sync_viewer`` runs from four places and again within ~3s off the
        cache tick, and it decides what to show from the selection alone -- so a
        plain ``clear()`` is undone by the next tick, which is exactly the
        "reference wants the picture, mesh wants the mesh" rule doing its job.
        What stops it is the short-circuit ``viewer.path == wanted``: leave the
        path naming the thing that *would* be loaded and the sync agrees there
        is nothing to do. This is ``_review_load``'s idiom -- clear, then pin --
        applied at the other end of the same mechanism.

        So the two stages clear differently and neither is arbitrary.
        ``clear_reference`` releases just the texture (forgetting the backend's
        registration before the release, which is the whole reason to call it
        rather than release the texture here), while ``clear()`` empties the
        mesh side and nulls path *and* pending. Nulling ``pending`` is not
        incidental: a parse may be in flight, and ``_adopt_model`` would
        otherwise land it on the emptied viewport a moment later. The pin is
        then written back explicitly in both branches, rather than relying on
        one of the two clears happening to leave ``path`` alone.

        The pin releases on its own the moment the selection or the stage
        changes, because ``wanted`` changes with it. Reselecting the same job
        re-shows it, which is what the tooltip says.
        """
        from . import create_stages, modes

        ctx = self.app_ctx
        viewer = self.viewer
        if viewer is None or ctx.state.mode not in modes.VIEWPORT_MODES:
            return
        # A strip renders off the mesh that is about to go; finishing it would
        # spend frames on an image of something no longer on screen.
        viewer.cancel_sheet_strip()
        if ctx.state.comparing:
            ctx.state.comparing = None
            viewer.exit_compare()
        job = ctx.job()
        job_dir = None if job is None else ctx.job_dir(job["id"])
        files = [] if job is None else (job.get("files") or [])
        # ``wanted``, computed exactly as ``_sync_viewer`` computes it -- the
        # two must agree or the pin names something the sync does not want and
        # the canvas refills on the next tick.
        if create_stages.at(ctx.state, "reference"):
            viewer.clear_reference()
            name = "input.png"
        else:
            viewer.clear()
            name = "model.glb"
        if job_dir is not None and name in files:
            viewer.path = job_dir / name

    def _sync_viewer(self) -> None:
        """Show whatever the selection implies, when it changes.

        Driven off the cache rather than off the click so a job that finishes
        while it is selected starts showing its mesh without another click.
        """
        from . import create_stages, modes

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
        # Reference wants the picture; Mesh, Rig and Pose all want the mesh --
        # the rig and its poses are fitted to *that* geometry, and a stage that
        # framed something else would be describing a different object. (The
        # pose editor swaps in ``rig.glb`` itself once it is entered, which is
        # why this returns early while ``viewer.pose_mode`` is on.)
        if create_stages.at(ctx.state, "reference"):
            if "input.png" in files:
                wanted = job_dir / "input.png"
        elif "model.glb" in files:
            wanted = job_dir / "model.glb"
        if wanted is None or self.viewer.path == wanted or self.viewer.pending == wanted:
            return
        # Both kinds are decoded off-thread and adopted when they land. This
        # runs on a *timer*, on the frame a job transitions to done -- which is
        # when the file is largest and coldest -- so doing the parse and the
        # texture decode here froze the frame that was meant to show the job
        # finishing. The GPU upload stays on the frame thread; see
        # ``_adopt_model``, which tells the two apart by the tag's suffix.
        parse = self.viewer.parse_reference if wanted.suffix == ".png" else self.viewer.parse_model
        self.viewer.pending = wanted
        if not ctx.submit(VIEWER_KEY, parse, wanted, tag=wanted):
            # Another load is already in flight. Its result is checked against
            # ``pending`` before it is adopted, so this one is simply retried
            # on the next tick rather than queued.
            self.viewer.pending = None

    def _adopt_compare(self, done: Any) -> None:
        """Take a parsed comparison mesh. Frame thread only, and contained.

        The mirror of ``_adopt_model`` and for the same three reasons: the GPU
        upload has to be on the frame thread, a result nobody is waiting for
        any more must be dropped rather than shown, and a failure must produce
        a toast rather than take the process out.

        That last one is what UX-04 is actually about. ``Viewer.compare`` used
        to parse *and* upload inline from the menu handler with no error
        boundary at all, so a corrupt GLB -- or an upload that failed on a
        driver hiccup -- propagated out of the frame loop and exited Studio. A
        comparison is a *look* at something; it must never be able to cost the
        session.
        """
        ctx = self.app_ctx
        wanted = done.tag
        if wanted is None or ctx.state.compare_pending != wanted:
            ctx.state.compare_pending = None
            return
        ctx.state.compare_pending = None
        try:
            self.viewer.adopt_compare(done.result)
        except Exception:
            log.exception("could not open %s for comparison", wanted)
            ctx.state.comparing = None
            ctx.toast("Could not open that asset to compare.", "error")

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
            if wanted.suffix == ".png":
                self.viewer.clear()
                self.viewer.adopt_reference(done.result)
                self.viewer.path = wanted
                self._refresh_rig_side_data()
                return
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
                # Through the *preflight* (the UI redesign, wave 3). This went
                # straight to ``_request_quit``, which walks the per-document
                # guards and asks nothing about a run in flight -- survivable
                # only while the header's power icon existed to carry
                # ``_ask_quit``'s generic summary. The header is gone, so the
                # window's X is the only interactive way out and it has to be
                # the one that asks.
                self._ask_quit()
                continue
            if event.type == pygame.VIDEORESIZE:
                # The *clamped* size is persisted, not the requested one: the
                # window that comes back is the clamped one, so storing the
                # raw event meant next launch opened below the resize floor
                # with no event to correct it.
                sized = (max(event.w, self._min_size[0]), max(event.h, self._min_size[1]))
                pygame.display.set_mode(sized, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
                ctx.settings.set("window_size", list(sized))
                continue
            if event.type in (pygame.WINDOWDISPLAYCHANGED, pygame.WINDOWMOVED):
                # UX-22. Both, because neither is sufficient: SDL2 reports the
                # display change when a window is dragged to another monitor,
                # but a display whose *own* scale is changed in Windows'
                # settings raises no such event and the window simply starts
                # being drawn at the wrong size. WINDOWMOVED catches the first
                # case again and costs one Win32 call, which is cheaper than
                # being wrong until the next restart.
                self._resample_display_scale()
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
            if _takes_pointer(self.viewer, self._viewport_hovered):
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
        if _takes_pointer(self.clay_view, hovered):
            self.clay_view.handle_event(tab.doc, event, hovered)

    def _poser_event(self, event: Any) -> None:
        """Route the mouse to Poser's viewer, on the same hover rule as Clay's.

        A drag already in progress ignores the hover, so crossing onto a panel
        mid-orbit does not drop it.
        """
        viewer = self.poser_viewer
        if viewer is None:
            return
        if _takes_pointer(viewer, self._poser_hovered):
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
        """Whether *any* modal is on screen and owns the keyboard."""
        return modal_open(self.app_ctx)

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

        from . import docmodes, modes

        ctx = self.app_ctx
        self._note_mode(ctx.state)
        # Ctrl+K, before everything, because it is the only binding that must
        # work in *every* mode and the workspace modes each consume whatever
        # reaches them. It is also, since the positional Alt+digit bindings went
        # away with the tenth-and-eleventh mode, the only keyboard route to a
        # mode at all. K is bound by neither Inker nor Clay, so no workspace
        # binding is displaced.
        # ``event.mod``, not ``pygame.key.get_mods()`` -- the rule
        # ``_passes_text_field`` states a few hundred lines down and
        # ``review_mode.handle_key`` already follows. ``mod`` is the modifier
        # state at the moment this key was *pressed*; ``get_mods()`` is the
        # state now, and events drain in a batch after the frame. A Ctrl
        # released between the press and this call made ``get_mods()`` lie, so
        # a fast Ctrl+K fell through to bare ``k`` -- which in Inker is the
        # **Rect tool**, so the palette failed to open *and* the active tool
        # changed under the user (UX-12).
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_k
            and event.mod & pygame.KMOD_CTRL
        ):
            from .panes import palette

            palette.toggle(ctx)
            return
        # **The palette owns the keyboard while it is up**, Esc included: its
        # query box holds the imgui focus and it reads its own Escape, Enter
        # and arrows there (``panes/palette.py``). Only Ctrl+K, above, is
        # exempt, because it is the way out. Without this the chords leaked
        # straight through -- ``palette_open`` was never one of
        # ``modal_open``'s answers, so Ctrl+Enter with the palette open in
        # Create queued a generation behind it.
        if event.type == pygame.KEYDOWN and ctx.state.palette_open:
            return
        # Beside Ctrl+K and for its reason: this is the second binding that
        # has to work in every mode, and the workspace modes below each consume
        # whatever reaches them. Slash rather than a letter because every
        # letter worth having is a tool in Inker or Clay, and Ctrl+/ is what
        # the rest of the world binds this to. It sets a flag; the header
        # consumes it (see ``_mode_switch``), because a key handler is not
        # inside the window the popup is registered in.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_SLASH
            and event.mod & pygame.KMOD_CTRL
        ):
            ctx.state.shortcuts_requested = True
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
            from .manual import render as manual_render

            manual_render.toggle(ctx)
            return
        # **And the Manual owns it too.** It covers the app, and the workspace
        # arms below consume whatever they are handed against a pane the reader
        # cannot see: Delete in Create trashed the selected asset unconfirmed,
        # and a bare tool letter switched Inker's tool under the overlay. Esc
        # passes because the branch immediately below is what answers it, and
        # Ctrl+K/Ctrl+//F1 are above for the reason they always are.
        if (
            event.type == pygame.KEYDOWN
            and ctx.state.manual.open
            and event.key != pygame.K_ESCAPE
        ):
            return
        # Esc closes the Manual before anything else looks at it, and that
        # ordering is the whole of why this sits here rather than in
        # ``_escape_mode``: the workspace modes below consume every key they
        # are handed, so an Esc dispatched to Inker with the overlay up would
        # drop a floating selection and leave the reference open on top of it.
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and ctx.state.manual.open:
            from .manual import render as manual_render

            manual_render.close(ctx)
            return
        # Then a running tour. Below the Manual because a step's "Read more"
        # raises the Manual over the tour, so the reference is the topmost
        # thing an Esc is about -- and above the workspaces for the same reason
        # the Manual is: Inker would consume the key and drop a floating
        # selection while the tour stayed up.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
            and getattr(ctx.state, "tour", None)
            and ctx.state.tour.running
        ):
            from .panes import tour as tour_pane

            tour_pane.stop(ctx)
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

                # A *grid* here, so Up and Down move by a row and Left and
                # Right by one -- the column count is whatever the grid drew
                # last frame. The sidebar keeps ``select_relative`` and its
                # one-card rows; which of the two a key means is a property of
                # the pane it was pressed in.
                if event.key in (pygame.K_UP, pygame.K_DOWN):
                    library.select_grid(ctx, 0, -1 if event.key == pygame.K_UP else 1)
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    library.select_grid(ctx, -1 if event.key == pygame.K_LEFT else 1, 0)
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    library.open_selected(ctx)
                elif event.key == pygame.K_DELETE and ctx.state.selected:
                    # The same binding the Create sidebar's library has, and
                    # the same reasoning: delete-to-trash is confirm-free here
                    # because the trash *is* the confirmation. The shortcuts
                    # sheet advertised it in both places and only one had it,
                    # which is a sheet that lies about the mode whose whole
                    # subject is the library.
                    library.delete_asset(ctx, ctx.state.selected)
            return
        if ctx.state.mode == "clay":
            from . import clay_mode

            # First refusal, and unconditional for the reason Inker's is:
            # handle_key returns False with no document open, and letting that
            # fall through meant F/W/S acted on a viewport Clay has replaced.
            # Every Clay binding is in ``clay_mode.handle_key`` now, F
            # included: it records ``state.frame_pending`` and the viewport
            # consumes it, because framing needs a viewport this module owns
            # and that one may not import (B6).
            clay_mode.handle_key(ctx, event)
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

            # Unconditional for the reason the four above are: handle_key
            # returns False for every key it does not bind, and letting that
            # fall through let the shared 2D/3D block below act on a library
            # and a viewport Packwright has replaced -- Delete trashed the
            # selected *library* asset (confirm-free, by that binding's own
            # design) and Ctrl+Enter queued a generation, from the atlas
            # packer. The return was lost when Troupe's branch was spliced in
            # ahead of it; a scan test now pins every workspace mode's arm.
            packwright_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "muse":
            from . import muse_mode

            # Unconditional and returning, for the reason every workspace arm
            # here is: ``handle_key`` answers False for every key it does not
            # bind, and letting that fall through would let the shared 2D/3D
            # block act on a library and a viewport Muse has replaced -- Delete
            # would trash the selected *library* asset, from a results tray.
            muse_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "sirens":
            from . import sirens_mode

            # Unconditional and returning, for the reason every workspace arm
            # above is: ``handle_key`` answers False for every key it does not
            # bind, and letting that fall through would let the shared 2D/3D
            # block act on a library and a viewport Sirens has replaced --
            # Delete would trash the selected *library* asset, from a tracker.
            # A scan test pins every workspace mode's arm.
            sirens_mode.handle_key(ctx, event)
            return
        if ctx.state.mode == "troupe":
            from . import troupe_mode

            # Unconditional and returning, for the reason the three above are:
            # ``handle_key`` answers False for every key it does not bind, and
            # letting that fall through would let F/W/S act on a viewport
            # Troupe has replaced with a sprite.
            troupe_mode.handle_key(ctx, event)
            return
        # Both edges reach this function, because Inker's space-to-pan is a
        # hold and needs the release. Nothing below is a hold: every one of
        # these is a toggle or an action, so acting on the release too undoes
        # the toggle the press just made and submits a second job for one
        # Ctrl+Enter.
        if event.type != pygame.KEYDOWN:
            return
        # ``event.mod``, for ``_passes_text_field``'s reason -- the state when
        # the key was pressed, not the state after the batch drained. A fast
        # Ctrl+Enter used to submit nothing at all, silently (UX-12).
        mods = event.mod
        # Before the 2D/3D bindings, because pose mode is drawn *over* them:
        # the inspector's pose editor is the same PoseEditor Poser authors
        # with, and it had no keyboard undo at all while Poser's got one. It
        # consumes only its own three chords and only while the editor is
        # bound, so nothing below moves when pose mode is off.
        if docmodes.pose_undo_key(self.viewer, event):
            return
        if event.key == pygame.K_RETURN and mods & pygame.KMOD_CTRL:
            from . import create_stages
            from .panes import settings_2d, settings_3d

            if create_stages.at(ctx.state, "reference"):
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
        elif event.key == pygame.K_DELETE and ctx.state.selected:
            # The library keyboard used to stop at navigation: Up/Down/Enter
            # moved and opened, and every action was mouse-only (UX-27).
            #
            # Delete specifically, and unguarded, because delete-to-trash is
            # deliberately confirm-free here -- "the trash *is* the
            # confirmation", which is the reasoning the menu item already
            # stands on. So this binding is exactly as safe as the menu item it
            # mirrors, and no safer or less safe.
            #
            # ``F`` is deliberately not bound to favourite despite the finding
            # offering it: F already frames the viewer a few lines below, and
            # taking a live 3D binding to add a library one would be a trade,
            # not a fix.
            from .panes import library

            library.delete_asset(ctx, ctx.state.selected)
        elif event.key == pygame.K_f:
            self.viewer.frame()
        elif event.key == pygame.K_w and event.mod & pygame.KMOD_SHIFT:
            # Shift+W: the layout editor (P5.4). Verified free -- plain W is
            # wireframe below, and every mode that takes W takes it before this
            # handler is reached.
            from . import layout_edit

            layout_edit.toggle(ctx.state)
        elif event.key == pygame.K_w:
            ctx.state.wireframe = not ctx.state.wireframe
            self.viewer.set_wireframe(ctx.state.wireframe)
        elif event.key == pygame.K_s:
            ctx.state.turntable = not ctx.state.turntable
            self.viewer.set_turntable(ctx.state.turntable)

    def _resample_display_scale(self) -> float:
        """Re-read the monitor's scale and rebuild what is baked at it (UX-22).

        DPI was sampled once, at startup, and the module comment said a
        mid-session change "would need a rebuild" as though that were
        unavailable -- but the UI-scale slider has done exactly this rebuild
        since K99, and every piece of it is reusable. Dragging the window from
        a 100% monitor to a 150% one left the whole UI drawn at the old scale
        until the next launch; on the pair of displays this is most likely to
        happen on, that is either a UI two-thirds the size it should be or one
        half again too big.

        -> the scale in force afterwards, so a caller can tell whether
        anything moved.

        The font atlas is *not* rebuilt here. It cannot be: rebuilding
        invalidates every ImFont handle, and those are pushed and popped all
        through ``_build_ui``, so the flag is raised and the frame loop
        consumes it between frames (K99) -- the same route the slider takes.
        """
        import pygame
        from imgui_bundle import imgui

        from . import dpi, theme, tokens

        monitor_scale = dpi.window_scale(pygame)
        if monitor_scale == self._monitor_scale:
            return tokens.SCALE
        # The user's zoom is re-read rather than divided back out of the old
        # product: ``set_scale`` clamps, so on a display scaled past the
        # ceiling the stored zoom and the zoom in force differ, and recovering
        # it by division would bake that clamp in permanently -- each move
        # between two such monitors shrinking the UI again.
        self._monitor_scale = monitor_scale
        self.app_ctx.dpi_scale = monitor_scale
        lo, hi = tokens.ui_scale_bounds(monitor_scale)
        tokens.set_scale(monitor_scale * min(max(_ui_scale(self.app_ctx.settings), lo), hi))
        theme.apply(imgui)
        self.app_ctx.state.fonts_dirty = True
        # The resize floor follows the monitor too, and a stale one is the
        # difference between a window that can be made small enough to fit and
        # one that cannot.
        self._min_size = _min_window_size(monitor_scale)
        log.info("display scale changed to %.2fx; rebuilding style and fonts", monitor_scale)
        return tokens.SCALE

    def _on_drop(self, path: Path) -> None:
        from . import create_stages
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
                ctx.toast("Packwright opens .wpack documents and packs image files.", "error")
            return
        if ctx.state.mode == "sirens":
            from . import sirens_mode, sirens_state

            suffix = path.suffix.lower()
            if suffix == sirens_state.WSNG_SUFFIX:
                sirens_mode.open_path(ctx, path)
            elif suffix == ".wav":
                # A WAV dropped here is a *sample*, not a song -- the Plotter
                # rule that a refusal and an accept both have to say what a
                # drop would do in this mode. It lands in the open song's sample
                # table; with no song open there is nowhere to put it, and
                # opening one silently to hold a drum hit would be a document
                # the user did not ask for.
                tab = sirens_mode.active(ctx)
                if tab is None:
                    ctx.toast(
                        "Open or start a song first: a sample belongs to one.", "error"
                    )
                else:
                    sirens_mode.import_sample(ctx, tab, path)
            else:
                ctx.toast("Sirens opens .wsng songs and .wav samples.", "error")
            return
        if ctx.state.mode in ("poser", "troupe"):
            # Neither mode opens a file, and neither said so: a drop here fell
            # through to Create's branches below, which would either refuse it
            # by describing a generation form that is not on screen or accept
            # it by switching modes out from under the user. The refusal names
            # what this mode works on instead, H71's rule.
            ctx.toast(
                "The Poser opens no files: it edits poses on a rig you already "
                "have, chosen from its own library."
                if ctx.state.mode == "poser"
                else "Troupe opens no files: it plays the character sheets a "
                "render has already produced.",
                "error",
            )
            return
        if ctx.state.mode in ("home", "library") and path.suffix.lower() == ".glb":
            # Home and Library take a mesh straight into the library, which is
            # the other half of the door ``library.pick_and_import_mesh``
            # opens: a user with a ``.glb`` reaches for a drop before a menu,
            # and until 2026-08-30 the only surface that accepted one was Clay
            # -- which converts it into an editable document and refuses a
            # rigged mesh outright. Create is deliberately *not* here: a mesh
            # dropped mid-generation is ambiguous between "start from this" and
            # "put this in my library", and the branch below already answers
            # that question for images.
            from .panes import library

            library.import_mesh_path(ctx, path)
            ctx.toast(f"Importing {path.name}...", "info")
            return
        if path.suffix.lower() not in DROPPABLE_IMAGES:
            # The refusal says what a drop would have *done here* (H71). One
            # sentence for both modes was wrong in 2D, where a dropped image is
            # a conditioning reference and never a mesh -- and the sentence is
            # the only thing that teaches the difference, since the two modes
            # accept the same file types.
            ctx.toast(
                "Drop an image to condition this generation on it."
                if create_stages.at(ctx.state, "reference")
                else "Drop an image to start a mesh from it.",
                "error",
            )
            return
        if create_stages.at(ctx.state, "reference"):
            # In the 2D pane a dropped image is a *conditioning reference*, not
            # a mesh to build -- forcing the mode switch here would throw away
            # the prompt the user is composing. One branch, and it is what
            # makes the feature discoverable at all.
            ctx.state.form_2d["ref_path"] = str(path)
            # H70's other half -- a ``widgets.request_open`` for the
            # References block -- is gone, not moved. It was written when that
            # block was a collapsible header that defaulted shut; it is always
            # open now, and no ``persist_key`` by that name is registered
            # anywhere, so the request matched nothing and merely accumulated
            # in ``widgets._OPEN_REQUESTS`` for the life of the process -- with
            # a comment beside it claiming it was why the drop was visible.
            # The flash below is what actually says the drop landed.
            self._flash_drop("2d-ref")
            ctx.toast(f"Using {path.name} as the reference.", "success")
            return
        # A drop is a start: it would otherwise land behind the chooser, with
        # nothing on screen saying anything had happened. ``follow=False``:
        # the file being dropped is the source, so walking the selection onto
        # a mesh the current one already has would describe the wrong asset.
        create_stages.go(ctx, "mesh", follow=False)
        self._flash_drop("3d-source")
        settings_3d.upload(ctx, path)

    def _flash_drop(self, slot: str) -> None:
        """Mark a slot as having just received a drop, for ``widgets.ring``."""
        state = self.app_ctx.state
        state.drop_flash_slot = slot
        state.drop_flash_at = time.monotonic()

    def _quit_summary(self) -> str:
        """What quitting would actually interrupt, in one sentence per thing.

        Empty when nothing is going on, which is the common case and the whole
        point: the confirm this feeds used to say "Anything still generating is
        cancelled" *unconditionally*, on an idle app with nothing unsaved --
        a warning about a thing that was not happening, which teaches people to
        click through warnings (UX-21).
        """
        ctx = self.app_ctx
        lines: list[str] = []
        if self.runtime.current_job_id is not None or ctx.cache.active is not None:
            lines.append("A job is still generating and will be cancelled.")
        # Named, not counted: "3 tasks" is not something a user can weigh, and
        # a 16 GB download is a very different thing to interrupt than a
        # thumbnail. Downloads and exports are the two worth calling out.
        busy = set(ctx.tasks.busy_keys)
        if any(k.startswith("download:") for k in busy):
            lines.append("A model download is in progress and will be stopped.")
        if any(k.startswith(("export", "save:", "bake:")) for k in busy):
            lines.append("An export is still being written.")
        return "\n".join(lines)

    def _ask_quit(self) -> None:
        """Ask once, about what is actually true, then run the guards.

        The chain below still asks per unsaved document, because each of those
        is a genuine question with a genuine answer ("discard *this* one?").
        What is gone is the unconditional preamble in front of it: on an idle
        app with nothing dirty -- the common state -- quitting used to raise a
        warning about generating that was not happening, and confirming it
        could then raise up to six more (UX-21). Now the generic question is
        asked only when it has something to say.
        """
        from . import dialogs

        summary = self._quit_summary()
        if not summary:
            self._request_quit()
            return
        self.app_ctx.confirms.ask(
            dialogs.Confirm(
                title="Quit Warlock Studio?",
                message=summary,
                confirm_label="Quit",
                cancel_label="Stay",
                on_confirm=self._request_quit,
            )
        )

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
        from . import clay_mode, inker_mode, packwright_mode, plotter_mode, poser_mode, sirens_mode
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
            sirens_mode.guard,
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
        from . import menus, modes, rail, status_bar, tokens
        from .panes import (
            app_settings,
            inspector,
            landing,
            library,
        )

        ctx = self.app_ctx
        # The rail first of all, because the sidebars are fitted against what
        # is left after it: a rail measured afterwards would leave the columns
        # disagreeing with the window by exactly its own width for one frame
        # every time it was toggled.
        rail.tick(self.layout)
        # Before any pane reads ``layout.SIDEBAR_W``: a width change eases, and
        # a half-eased width read by the left sidebar and the settled one read
        # by the right would be two columns disagreeing about the same frame.
        layout_mod.tick()
        # Straight after, and before any column is drawn: how wide a sidebar
        # can be is a fact about this frame's window, and the left column must
        # not settle it for itself and leave the right one to find out (UX-01).
        mode_for_layout = ctx.state.mode
        self.layout.bind_workspace(self.layouts, mode_for_layout)
        # No ``fixed_left`` any more: it pinned Inker's left column to the
        # toolbox rail's 90 px, and the rail is gone -- both of Inker's columns
        # are ordinary sidebars whose widths are the arrangement's to state.
        # (``fit_widths`` keeps the parameter: it is the general answer for a
        # column that is a fixed size rather than a preference, and deleting it
        # would have to be re-derived by the next workspace that wants one.)
        layout_mod.measure(self.layouts, mode_for_layout)
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
        #
        # A *stage* change is the same event since wave 5, and has to be
        # watched separately: Reference and Mesh are one mode now, so stepping
        # between them moves no mode at all -- and what the viewport should be
        # showing (``input.png`` against ``model.glb``) changed anyway.
        stage_moved = ctx.state.create_stage != self._last_stage
        self._last_stage = ctx.state.create_stage
        if (
            ctx.state.mode != self._last_mode or stage_moved
        ) and ctx.state.mode in modes.VIEWPORT_MODES:
            self._sync_viewer()
        # And on a *selection* change, which is the trigger UX-03 found missing.
        # ``_sync_viewer`` was driven off the cache reread (a 3 s idle timer)
        # and off mode transitions, so clicking a card updated ``state.selected``
        # -- and therefore the inspector -- immediately while the viewport went
        # on showing the previous asset for up to three seconds. The inspector
        # described B and the viewport drew A, with nothing on screen saying so,
        # which makes an export or a compare decision untrustworthy.
        if ctx.state.selected != self._last_selected:
            self._last_selected = ctx.state.selected
            if ctx.state.mode in modes.VIEWPORT_MODES:
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
            | imgui.WindowFlags_.menu_bar.value
        )
        imgui.begin("##host", None, flags)
        # One frame, one record of what stopped drawing. Above the three clears
        # below rather than beside them, which visibly breaks that block: the
        # menu bar draws first *and* is itself guarded, so its census has to be
        # empty before it runs.
        guard.begin_frame(ctx)
        # One stable command surface in every mode.  The menu rows are adapters
        # over the same command/operation registries used by Ctrl+K and keys.
        guard.run("shell/menus", menus.draw, ctx, self.layout, title="The menu bar")
        # One frame, one record of where every pane ended up -- and one answer
        # to "is the layout editor open", which every splitter reads (P5.4).
        from . import layout_edit

        layout_mod.begin_frame(layout_edit.ensure(ctx.state).open)
        # And one record of where every *control* that a tour can point at
        # ended up. Cleared here rather than in ``layout`` so the two clears
        # are visibly the same decision, made once, in one place.
        anchors.begin_frame()
        # And -- on a probe run only -- one record of every control the frame
        # submits, for the driver that clicks them. Same clear, same place, for
        # the same reason: a stale census points at whatever took the control's
        # place.
        probe.begin_frame()
        # And -- once every ten seconds rather than once a frame -- the
        # interpolator forgets the keys nothing is asking for any more. Here
        # rather than in ``motion`` itself because this is the one place that
        # knows a frame has started, which is the same reason the four clears
        # above it live here.
        from . import motion as motion_mod

        motion_mod.sweep()
        # The rail is drawn in every mode, Home included: it is how you leave
        # wherever you are, so a mode that hides it is a dead end.
        guard.run("shell/rail", rail.draw, self, ctx, title="The mode rail")
        # Shell utility popups are opened at host scope. Menu actions can
        # originate in child windows, while imgui resolves a popup in the
        # window that opens it, so they communicate through one-shot requests.
        if rail.take("layouts"):
            imgui.open_popup("layouts")
        # Under ``guard`` like every other surface: these three draw at host
        # scope, so a raise inside one took the whole frame down rather than
        # one pane's worth of it -- and the layouts popup is where finding 1 of
        # this review's section 2 shipped from.
        guard.run(
            "shell/layouts",
            self._layouts_popup,
            ctx,
            title="Workspace layout",
            draw_placeholder=False,
        )
        # Kept behind an explicit developer environment flag; normal installs
        # never gain a design-system destination in their navigation.
        from . import component_gallery

        guard.run(
            "shell/gallery",
            component_gallery.draw,
            title="Component gallery",
            draw_placeholder=False,
        )
        # Ctrl+/ and the palette's "Keyboard shortcuts" both set this flag,
        # because neither a key handler nor a palette command is inside the
        # window the popup is registered in. It was consumed by the header's
        # ``?`` button; the header is gone, so it is consumed here.
        if ctx.state.shortcuts_requested:
            ctx.state.shortcuts_requested = False
            # Cleared on the way in rather than on the way out: a popup can be
            # dismissed by clicking anywhere, which is not a moment this has a
            # hook in, and reopening onto last time's query would look like a
            # list that had lost most of its rows.
            self._shortcuts_query = ""
            imgui.open_popup("shortcuts")
        guard.run(
            "shell/shortcuts",
            self._shortcuts_popup,
            title="Keyboard shortcuts",
            draw_placeholder=False,
        )
        imgui.same_line()
        # Treat the workspace and its status as one vertical item beside the
        # full-height rail. Without this group imgui advances below the taller
        # rail before drawing the status, clipping it against the host edge.
        imgui.begin_group()
        # A negative child height leaves its magnitude below the child, but
        # the next item also consumes the parent's item spacing. Reserve both
        # so the shared status line is never clipped at the host's lower edge,
        # especially when that spacing is doubled by UI scale.
        status_reserve = tokens.sp(status_bar.STATUS_H) + imgui.get_style().item_spacing.y
        imgui.begin_child("##content", (0, -status_reserve))
        from .panes import overlay

        mode = ctx.state.mode
        # Tier two of the same net, and the reason the two tails below became
        # one: a single guarded region needs a single exit, so the duplicated
        # end_child/status/end_group/end/overlays that each branch used to
        # carry is now written once. The mark is taken inside ``##content``,
        # so a failure in the scaffolding *between* panes -- the groups and
        # columns no ``layout.pane`` covers -- costs the workspace and leaves
        # the rail, the menu bar and the status line live. That is the
        # difference between a broken workspace and a dead end.
        with guard.surface("shell/content", title="The workspace") as live:
            if live:
                overlay.doctor_banner(ctx)
                if mode in _SINGLE_PANE_MODES or mode in modes.WORKSPACE_MODES:
                    if mode == "home":
                        landing.draw(ctx)
                    elif mode == "settings":
                        app_settings.draw(ctx)
                    elif mode == "library":
                        # The full-window composition (the UI redesign, wave 4.4), not a
                        # second card list: ``library_full`` draws the *same* filters,
                        # the same cards' actions and the same inspector, arranged for
                        # a window rather than for a 300 px sidebar. The library itself
                        # is still one implementation -- this module composes it.
                        from .panes import library_full

                        library_full.draw(ctx)
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
                    elif mode == "muse":
                        self._muse_workspace()
                    elif mode == "sirens":
                        self._sirens_workspace()
                    elif mode == "troupe":
                        self._troupe_workspace()
                    else:
                        self._inker_workspace()
                else:

                    # The library used to share the left sidebar with settings, split by
                    # settings_share; it shares the right sidebar with the inspector now
                    # instead, so the left column is settings alone (nothing left to split
                    # against) and the right column is the two-scroller stack that used to
                    # live on the left.

                    lay = self.layout
                    left_w = layout_mod.sidebar_width("left")
                    right_w = layout_mod.sidebar_width("right")
                    # The rail first, above the columns it switches: it is a breadcrumb for
                    # what is under it, and one drawn at the bottom would be a tab strip
                    # that had lost its tabs.
                    #
                    # Spanning the whole content width rather than boxed inside the 300 dp
                    # settings column, which is where it used to live. Five labelled
                    # segments measure ~304 dp; a sidebar gives the widget ~276 dp, so the
                    # fitting ladder in ``widgets.stage_rail`` was pinned to its last rung
                    # -- five anonymous icons standing in for the app's central navigation
                    # metaphor, at every realistic window size rather than only at small
                    # ones. Full width, the same widget sits on rung 1 (labels and ticks)
                    # and the ladder goes back to being a response to a narrow window.
                    imgui.dummy((0, tokens.sp(tokens.SP_2)))
                    pad = tokens.sp(layout_mod.PANE_PADDING)
                    rail_w = imgui.get_content_region_avail().x - pad * 2
                    imgui.indent(pad)
                    self._stage_rail(ctx, max_width=rail_w)
                    imgui.unindent(pad)
                    # The brief, across the full content width under the rail
                    # and above the columns. Drawn only where it has something
                    # true to say -- the four other stages start their columns
                    # here instead, rather than reserving an inert strip.
                    if create_brief.shows(ctx):
                        with layout_mod.pane(
                            "brief",
                            (0, tokens.sp(create_brief.BAR_H)),
                            layout_mod.PaneRole.CONTENT,
                            edge=layout_mod.PaneEdge.BOTTOM,
                            title="The brief bar",
                        ) as visible:
                            if visible:
                                create_brief.draw(ctx)
                    with layout_mod.pane(
                        "settings",
                        (left_w, 0),
                        layout_mod.PaneRole.SIDEBAR,
                        edge=layout_mod.PaneEdge.RIGHT,
                    ) as visible:
                        if visible:
                            _stage_pane(ctx)

                    _column_boundary(self.layouts, "create", "left")
                    self._viewport_pane()
                    _column_boundary(self.layouts, "create", "right")

                    _right_column(
                        ctx,
                        lay,
                        right_w,
                        inspector_draw=inspector.draw,
                        library_draw=library.draw,
                    )

        imgui.end_child()
        guard.run("shell/status", status_bar.draw, ctx, title="The status bar")
        imgui.end_group()
        imgui.end()
        self._overlays(viewport)

    def _stage_rail(self, ctx: Any, *, max_width: float | None = None) -> None:
        """Create's breadcrumb, over the three columns.

        The pane dispatch that follows it reads ``state.create_stage``, and
        this is the only control that writes one -- through
        ``create_stages.go``, which is what makes "switching stage may move the
        selection" a rule rather than a thing this happens to remember.
        """
        from imgui_bundle import imgui

        from . import create_stages, tokens, widgets
        from .panes import inspector

        job = ctx.job()
        # The two pieces of evidence no job row carries. ``rig_meta`` is the
        # inspector's own mtime-cached read of ``rig.json`` -- the same call it
        # makes for the weighting line, so the rail costs no extra stat --
        # and ``poses`` is what ``_refresh_rig_side_data`` fetched off-thread.
        meta = inspector.rig_meta(ctx, job) if job is not None else None
        poses = ctx.state.preview.get("poses")
        items = [
            (
                stage,
                create_stages.LABELS[stage],
                create_stages.ICONS[stage],
                create_stages.available(stage, job, ctx),
            )
            for stage in create_stages.STAGES
        ]
        picked = widgets.stage_rail(
            "create-stages",
            items,
            ctx.state.create_stage,
            done=create_stages.reached(job, meta, poses),
            max_width=(imgui.get_content_region_avail().x if max_width is None else max_width),
        )
        anchors.mark("create/stages")
        if picked != ctx.state.create_stage:
            create_stages.go(ctx, picked)
        imgui.dummy((0, tokens.sp(tokens.SP_2)))

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
            # Level ``warn``, not ``error``: the export succeeded and only the
            # picture did not, so a red card over a build that worked would
            # overclaim -- but something *did* fail and the log has the
            # traceback, which is exactly the middle level H68 added. (It sat
            # at ``info`` only while info and error were the whole vocabulary.)
            ctx.toast("The asset was built, but its thumbnail could not be made.", "warn", "log")
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

    def _inker_workspace(self) -> None:
        """Colour / canvas / tools, with the timeline along the bottom.

        Aseprite's default arrangement, which is what ``skeletons.inker`` now
        declares -- the previous shape (a 90 px tool rail on the left, the
        palette on the right, no bottom region at all) was its *Mirrored
        Default* preset plus a bug.

        Deliberately not a takeover of the whole window: the progress card
        floats over every mode, so a trellis run started before switching here
        is still visible while painting.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import skeletons, tokens
        from .panes import inker_canvas, inker_timeline
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        state = ctx.state.inker
        tab = None if state is None else state.active
        # The tile panel appears with the tilesets, the way the preview appears
        # with the frames: a drawing that has never seen a tilemap layer is
        # byte-for-byte the workspace it always was, and a fixed-height palette
        # taken out of every Inker session for a feature most of them do not
        # use is a cost with no matching benefit. The *verbs* that make the
        # first tileset are not hidden with it -- they are menu rows, which are
        # always drawn.
        # (The tile panel's own ``when`` predicate answers that in the
        # skeleton table, which is where the shape of a workspace lives now.)
        #
        # Both sidebars go through ``layout.column`` over ``skeletons.inker``
        # (P5.1): one renderer, one height arithmetic, and a saved layout has
        # something to be a permutation of.
        columns = skeletons.for_mode(ctx, "inker")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "inker", columns["left"]),
            width=left_w,
            handle_length=left_w,
            on_hidden=lambda _slot: None,
        )

        _column_boundary(self.layouts, "inker", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        imgui.begin_group()
        # A *positive* height, never a bottom offset: with little room left a
        # negative height collapses the canvas child to nothing and the canvas
        # -- and its texture uploads -- silently stops being drawn. Same rule
        # the status bar inside ``inker_canvas`` already follows.
        #
        # **Unconditional, because the strip is unconditional.** It was gated
        # on ``doc.anim is not None`` from 2a56df6 until 2026-08-23, which is
        # what hid the layer list from every still document; it was then gated
        # on ``state.timeline_open``, which is what a ``Tab`` key could hide.
        # Both gates are gone: the strip holds the layers, so a hidden strip is
        # a document with no visible layer list, and the height drag is the
        # thing hiding it was really being used for.
        available_h = imgui.get_content_region_avail().y
        centre_h = 0.0
        strip_key = "inker-timeline"
        if tab is not None:
            # A high UI scale can make the timeline's preferred design-pixel
            # height consume the whole physical window. The canvas remains the
            # anchor in that case and the timeline becomes the compressed,
            # scrollable pane. The ratio is frame-local, just like horizontal
            # side-panel compression; it never rewrites a saved share.
            strip = max(
                sp(inker_timeline.STRIP_H),
                available_h * lay.share("inker-timeline"),
            )
            centre_h = max(available_h - strip, available_h * 0.62)
        with layout_mod.pane(
            "inker-centre",
            (width, centre_h),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                inker_canvas.draw(ctx)
            elif tab is not None:
                # A keyboard zoom rung banked by ``handle_key`` is consumed
                # inside the canvas child; on a frame the centre pane does not
                # draw at all it would survive and fire later, unprompted.
                # Dropped here for the same reason the canvas's own invisible
                # branch drops it.
                tab.view.pending_zoom_rung = 0
        if tab is not None:
            # The handle between the canvas and the strip. Dragging it *down*
            # is a smaller strip, which is why the delta is subtracted: the
            # share names the timeline's portion, not the canvas's.
            drag = layout_mod.splitter(f"{strip_key}-share", vertical=False, length=width)
            if drag and available_h > 0:
                before = lay.share(strip_key)
                lay.set_share(strip_key, before - drag * tokens.SCALE / available_h)
                if lay.share(strip_key) != before:
                    lay.save()
            with layout_mod.pane(
                "inker-timeline",
                (width, 0),
                layout_mod.PaneRole.SHEET,
                edge=layout_mod.PaneEdge.TOP,
            ) as visible:
                if visible:
                    inker_timeline.draw(ctx)
        imgui.end_group()

        _column_boundary(self.layouts, "inker", "right")
        # **Preview / Tools / Tiles / Generation**, top to bottom, with a
        # handle between each adjacent shareable pair. The toolbox moved here
        # from the left column in this wave: the note it used to carry said
        # that putting it on the right "would put the toolbox on the far side
        # of the canvas from the hand", and what answers that is Aseprite's own
        # default, which is the program these users already have open.
        #
        # Which panes are here, and in what order, is now the active saved
        # layout's answer (wave 5) -- reconciled against this table every read
        # and never written back.
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "inker", columns["right"]),
            width=right_w,
            handle_length=right_w,
            on_hidden=lambda _slot: None,
        )

    def _plotter_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses,
        arranged the way Tiled arranges its own:

            [ plotter-properties ]  the toolbar  [ plotter-layers  ]
            [ plotter-bridge     ]  the map      [ plotter-tileset ]

        Both sidebars are ``skeletons.plotter``, which is where the argument
        for that arrangement is written down. The toolbar is not a slot: it is
        a strip inside the centre pane, drawn by ``plotter_canvas`` between the
        tab bar and the map, exactly as Inker's context bar is.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import skeletons
        from .panes import plotter_canvas, plotter_tileset_editor

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        # Both sidebars through ``layout.column`` over ``skeletons.plotter``
        # (wave 5), so the arrangement is data a saved layout can permute.
        columns = skeletons.for_mode(ctx, "plotter")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "plotter", columns["left"]),
            width=left_w,
            handle_length=left_w,
        )

        _column_boundary(self.layouts, "plotter", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        # The tileset editor is a **sheet over the centre pane**, drawn instead
        # of the map: the branch ``_review_workspace`` already takes, with the
        # role that already exists for it. Not a mode (a 21-place checklist,
        # including prose asserting the mode count) and not a document kind
        # (which would teach nine places a second shape).
        sheet = plotter_tileset_editor.active(ctx)
        with layout_mod.pane(
            "plotter-centre",
            (width, 0),
            layout_mod.PaneRole.SHEET if sheet else layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                if sheet:
                    plotter_tileset_editor.draw(ctx)
                else:
                    plotter_canvas.draw(ctx)

        _column_boundary(self.layouts, "plotter", "right")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "plotter", columns["right"]),
            width=right_w,
            handle_length=right_w,
        )

    def _muse_workspace(self) -> None:
        """The brief across the top, the takes in the middle, the recipe beside:

            [ muse-brief                                        ]
            [ the takes                       ]  [ muse-recipe  ]

        Composed by hand rather than through ``skeletons``, and **one** sidebar
        rather than the pair that table is built for. Packwright is the
        precedent that hand composition here is current rather than legacy; the
        reason it applies is that Muse has nothing to put in a second column.
        Two columns of which one is empty is a worse answer than one column.

        The bar is a full-width pane above both, which is ``create_brief``'s
        arrangement -- except that it is unconditional, because Muse has no
        stages for it to be absent on.

        The player is a fourth pane along the bottom, full width:

            [ muse-brief                                        ]
            [ the takes                       ]  [ muse-recipe  ]
            [ muse-player                                       ]

        Full width for the reason its own docstring gives -- 240 seconds across
        a 260 dp sidebar is a second per pixel, and a loop marker dragged at
        that scale is a guess. Drawn only once a take has been auditioned, so
        the two columns get the whole height until there is something to put
        under them.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import muse_brief
        from .panes import muse_player, muse_recipe, muse_results

        ctx = self.app_ctx
        right_w = layout_mod.sidebar_width("right")

        with layout_mod.pane(
            "muse-brief",
            (0, tokens.sp(muse_brief.BAR_H)),
            layout_mod.PaneRole.CONTENT,
            edge=layout_mod.PaneEdge.BOTTOM,
            title="The brief bar",
        ) as visible:
            if visible:
                muse_brief.draw(ctx)

        # The two columns are bounded rather than filling, so the strip has a
        # row to be in. Measured here rather than passed as a negative height:
        # the strip is conditional, and "what is left" has to be the whole
        # remainder on the frames where there is no strip at all.
        strip = muse_player.should_draw(ctx)
        body = imgui.get_content_region_avail().y
        body_h = body - tokens.sp(muse_player.STRIP_H) if strip else 0.0

        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "muse-centre",
            (layout_mod.centre_width() + layout_mod.sidebar_width("left"), body_h),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                muse_results.draw(ctx)

        _column_boundary(self.layouts, "muse", "right")
        with layout_mod.pane(
            "muse-recipe",
            (right_w, body_h),
            layout_mod.PaneRole.SIDEBAR,
            edge=layout_mod.PaneEdge.LEFT,
        ) as visible:
            if visible:
                muse_recipe.draw(ctx)

        if strip:
            muse_player.draw(ctx)

    def _sirens_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses:

            [ sirens-transport ]                 [ sirens-instruments ]
            [ sirens-orders    ]  the grid       [ sirens-bridge      ]

        Both sidebars through ``layout.column`` over ``skeletons.sirens``,
        which is the direction of travel: Packwright still composes its columns
        by hand here, and a table is what a saved layout can permute.

        The centre column is one pane: the tab bar, the caret strip and the
        grid, in that order, all of which ``sirens_patterns.draw`` composes --
        the grid sizes its row count from what is left of the content region,
        so the strip has to be drawn before it rather than beside it here.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import skeletons
        from .panes import sirens_patterns

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        columns = skeletons.for_mode(ctx, "sirens")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "sirens", columns["left"]),
            width=left_w,
            handle_length=left_w,
        )

        _column_boundary(self.layouts, "sirens", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "sirens-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                sirens_patterns.draw(ctx)

        _column_boundary(self.layouts, "sirens", "right")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "sirens", columns["right"]),
            width=right_w,
            handle_length=right_w,
        )

    def _troupe_workspace(self) -> None:
        """The same skeleton the other five use:

            [ troupe-cast     ]                  [ troupe-sheets ]
            [ troupe-settings ]   the sprite     [ troupe-bridge ]

        The centre pane is also the mode's heartbeat -- there is no per-mode
        update hook, so the pane that draws is what pumps the preview clock.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import (
            troupe_bridge,
            troupe_characters,
            troupe_preview,
            troupe_settings,
            troupe_sheets,
        )

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")

        _split_column(
            ctx,
            lay,
            split_id="troupe-cast",
            handle_length=left_w,
            width=left_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=("troupe-cast", layout_mod.PaneRole.SIDEBAR, troupe_characters.draw),
            bottom=("troupe-settings", layout_mod.PaneRole.SIDEBAR, troupe_settings.draw),
        )

        _column_boundary(self.layouts, "troupe", "left")
        width = layout_mod.centre_width()
        # No scroll-with-mouse for the reason Plotter's centre has none: the
        # wheel belongs to the picture. It said so from the day the mode was
        # built and it was not true until W0.3 -- no Troupe pane read the
        # wheel, so the flag took it away from the pane's scrollbar and gave it
        # to nothing. ``troupe_preview`` zooms with it now, over the sprite.
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "troupe-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                troupe_preview.draw(ctx)

        _column_boundary(self.layouts, "troupe", "right")
        _split_column(
            ctx,
            lay,
            split_id="troupe-sheets",
            handle_length=right_w,
            width=right_w,
            edge=layout_mod.PaneEdge.LEFT,
            top=("troupe-sheets", layout_mod.PaneRole.INSPECTOR, troupe_sheets.draw),
            bottom=("troupe-bridge", layout_mod.PaneRole.INSPECTOR, troupe_bridge.draw),
        )

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

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")

        _split_column(
            ctx,
            lay,
            split_id="packwright-sources",
            handle_length=left_w,
            width=left_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=("packwright-sources", layout_mod.PaneRole.SIDEBAR, packwright_sources.draw),
            bottom=("packwright-settings", layout_mod.PaneRole.SIDEBAR, packwright_settings.draw),
        )

        _column_boundary(self.layouts, "packwright", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "packwright-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                packwright_preview.draw(ctx)

        _column_boundary(self.layouts, "packwright", "right")
        _split_column(
            ctx,
            lay,
            split_id="packwright-items",
            handle_length=right_w,
            width=right_w,
            edge=layout_mod.PaneEdge.LEFT,
            top=("packwright-items", layout_mod.PaneRole.INSPECTOR, packwright_items.draw),
            bottom=("packwright-bridge", layout_mod.PaneRole.INSPECTOR, packwright_bridge.draw),
        )

    def _overlays(self, viewport: Any) -> None:
        """Toasts and modals, drawn over whichever layout ran.

        Outside the host window and after it ends, because a modal is its own
        window: the landing screen needs them as much as the workspace does,
        which is why this is not inline in either.
        """
        from . import widgets
        from .panes import first_run, overlay, palette, settings_3d

        ctx = self.app_ctx
        # The layout editor, over the workspace that has just recorded its pane
        # rects and *outside* every pane -- see ``layout_edit``'s docstring for
        # why that is a construction rather than a habit.
        from . import layout_edit

        over = functools.partial(guard.run, draw_placeholder=False)
        over(
            "overlay/layout-editor",
            layout_edit.draw,
            self,
            ctx,
            viewport,
            title="The layout editor",
        )
        over("overlay/fps", overlay.fps_meter, ctx, self.fps, title="The frame meter")
        if ctx.state.mode != "home":
            over(
                "overlay/progress",
                overlay.progress_card,
                ctx,
                self.eta,
                title="The progress card",
            )
        over(
            "overlay/toasts",
            widgets.toasts,
            ctx.state,
            (viewport.work_size.x, viewport.work_size.y),
            on_action=self._toast_action,
            title="Notifications",
        )
        # The first-run question owns the screen before any workflow modal.
        # Its two exits close it permanently, then later questions can use the
        # one popup slot on the following frame.
        over("overlay/first-run", first_run.draw, ctx, title="The first-run panel")
        if first_run.is_open(ctx):
            self._transition_overlay(viewport)
            return
        # Before the confirms, because it is the same kind of thing and the
        # earlier one wins the single modal slot imgui gives a frame.
        over("overlay/matte", settings_3d.matte_modal, ctx, title="The cutout dialog")
        # The Manual, over whatever ran above (the UI redesign, wave 3). Before the
        # palette on purpose: Ctrl+K is how you leave anywhere, this included,
        # so it has to float above the reference rather than under it.
        from .manual import render as manual_render

        over("overlay/manual", manual_render.draw_overlay, ctx, title="The manual")
        # Above the confirms it can raise (Delete asks): the palette closes
        # itself in the same frame it runs a command, so the question it asks
        # takes the modal slot on the frame after, with nothing to contend
        # with.
        # The tour, over the Manual so a "Read more" does not bury the card
        # that offered it, and under the palette for the Manual's own reason:
        # Ctrl+K is how you leave anywhere, this included.
        from .panes import tour as tour_pane

        over(
            "overlay/tour",
            tour_pane.draw,
            ctx,
            title="The guided tour",
            on_failure=lambda: tour_pane.stop(ctx),
        )
        over("overlay/palette", palette.draw, ctx, title="The command palette")
        # A queue that stops drawing still reports ``modal_open``, so the
        # keyboard would be owned by a modal nobody can see. Dismissing is the
        # queue's own documented way out, and the reason ``pending`` is
        # read-only.
        over(
            "overlay/confirms",
            ctx.confirms.draw,
            title="A confirmation",
            on_failure=ctx.confirms.dismiss,
        )
        over(
            "overlay/prompts",
            ctx.prompts.draw,
            title="A prompt",
            on_failure=ctx.prompts.dismiss,
        )
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
            from . import asset_open

            # Through ``asset_open``, which knows that a follow-up row -- a
            # rig, a sheet, a sprite draft -- holds nothing of its own and
            # routes to the asset whose directory its artifacts landed in.
            # Routing by stage sent every one of them to the Mesh stage of a
            # row with no mesh, which is a toast saying "finished" followed by
            # a blank screen.
            asset_open.open_asset(ctx, arg)
            # The row that actually got selected, so the grid scrolls to what
            # is on screen rather than to an invisible follow-up.
            job = ctx.cache.get(arg)
            ctx.state.library_scroll_to = asset_open.route(job).job_id if job is not None else arg
        elif name == "undo" and arg:
            from .panes import library

            # Through the library's own restore, so the tick set and the
            # selection are handled exactly as they are when the trash view's
            # own Restore button is pressed.
            library.restore_asset(ctx, arg)
        elif name == "unlock" and arg:
            from . import plotter_mode

            plotter_mode.unlock_layer(ctx, arg)
        elif name == "review":
            # The sweep is not named here: Review rescans on arrival and its
            # run list is the thing that knows which directories exist. Landing
            # on the mode is the whole of what the button promises.
            self._set_mode("review")

    # Every binding the app answers to, in one place the user can find. The
    # tuples are (keys, what), grouped; Inker's letters come from TOOL_KEYS so
    # this list cannot drift from the handler.
    def _shortcuts_popup(self) -> None:
        from imgui_bundle import imgui

        from . import icons, widgets
        from .tokens import sp

        viewport = imgui.get_main_viewport()
        popup_width = min(sp(tokens.SURFACE_W_SHEET), viewport.work_size.x - sp(32))
        popup_height = min(sp(tokens.SURFACE_H_SHEET), viewport.work_size.y - sp(64))
        imgui.set_next_window_pos(
            (
                viewport.work_pos.x + viewport.work_size.x - sp(16),
                viewport.work_pos.y + sp(48),
            ),
            imgui.Cond_.appearing.value,
            (1.0, 0.0),
        )
        imgui.set_next_window_size((popup_width, popup_height))
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
        widgets.pane_header(
            "Keyboard shortcuts",
            actions=(("close", f"{icons.X} Close", imgui.close_current_popup),),
        )

        # Collected first, drawn after the box (UX.md Phase 4). The list is ~60
        # rows over eight groups, which is a scroll and a read rather than a
        # lookup -- and the subsequence matcher the command palette already
        # carries is the right instrument, so it is reused rather than
        # reimplemented. The rows themselves are ``shortcut_sections``, which
        # is module-level so the manual can be gated against it.
        sections = shortcut_sections()
        if imgui.begin_child("shortcuts/scroll", (0, 0)):
            self._draw_shortcut_rows(sections)
        imgui.end_child()
        imgui.end_popup()
        imgui.pop_style_var()

    def _draw_shortcut_rows(self, sections: list[tuple[str, list[tuple[str, str]]]]) -> None:
        """The filter box and whatever survives it."""
        from imgui_bundle import imgui

        from . import widgets

        imgui.set_next_item_width(-1)
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

    def _layouts_popup(self, ctx: Any) -> None:
        """The Window menu's layout switcher (P5.3).

        A switcher and nothing else: renaming, duplicating, deleting and
        resetting are Settings -> Advanced, which is **the canonical path**
        because Settings is reachable from the rail in every mode and no
        workspace layout can touch its single-column composition. This popup
        carries a Reset because that is the rung a user reaches for while
        looking at the layout that went wrong.
        """
        from imgui_bundle import imgui

        from . import controls, widgets

        if not imgui.begin_popup("layouts"):
            return
        widgets.popup_chrome(_imgui=imgui)
        widgets.secondary("Workspace layout")
        imgui.separator()
        for name, layout in sorted(self.layouts.layouts.items()):
            selected = name == self.layouts.active
            label = name if layout.readable else f"{name}  (a newer version)"
            if (
                controls.menu_item(
                    f"{label}##layout/{name}",
                    "",
                    selected,
                    layout.readable,
                    reason=(
                        "This layout was saved by a newer build. It is kept exactly "
                        "as it was found rather than reinterpreted."
                    ),
                )[0]
                and layout.readable
            ):
                self.layouts.set_active(name)
        imgui.separator()
        if controls.menu_item_simple("Reset this layout"):
            self.layouts.reset()
        if controls.menu_item_simple("Manage layouts..."):
            # Settings, rather than a second administration surface here: one
            # place that can rename and delete is one place to look for the
            # thing you deleted.
            from .state import set_mode

            set_mode(ctx.state, "settings")
        imgui.end_popup()

    def _viewport_pane(self) -> None:
        from imgui_bundle import imgui

        from . import create_stages, generation_workspace
        from . import layout as layout_mod
        from .panes import overlay
        from .tokens import sp

        ctx = self.app_ctx
        # Leave room for the inspector; the progress card floats over the image
        # now, so the full height is the image's.
        width = layout_mod.centre_width()
        # no_scroll_with_mouse: over the viewport the wheel can only mean dolly.
        with layout_mod.pane(
            "viewport",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=imgui.WindowFlags_.no_scroll_with_mouse.value,
        ) as visible:
            if visible:
                overlay.toolbar(ctx)
                avail = imgui.get_content_region_avail()
                height = max(avail.y, 64)
                reference_stage = create_stages.at(ctx.state, "reference")
                # Once a Create run exists, the canvas gains an in-context
                # results tray.  The viewer remains above it, so a reference
                # can still be judged at useful scale while progress and the
                # next variation stay in the same creative loop.
                tray = reference_stage and generation_workspace.should_draw(ctx)
                # The floor is **one whole card**, not a round number: heading,
                # caption, a 72 dp thumbnail and the two rows of actions under
                # it come to a little over 200 dp, and at the old 180 the
                # actions sat below the tray's fold on every card. A button
                # nobody can reach without scrolling a strip they cannot see
                # scrolls is a button that does nothing; ``/exercise-mode
                # create`` reported twelve of them, and no test can, because a
                # clipped button is still drawn.
                tray_height = min(sp(320), max(sp(232), height * 0.36)) if tray else 0.0
                gap = imgui.get_style().item_spacing.y if tray else 0
                canvas_height = max(height - tray_height - gap, sp(64))
                if tray:
                    # ``placeholder`` centres itself by consuming the available
                    # height, so it needs its own top child; otherwise it would
                    # consume the tray's room before the tray is drawn.
                    if imgui.begin_child(
                        "generation-canvas", (0, canvas_height), False,
                        imgui.WindowFlags_.no_scroll_with_mouse.value,
                    ):
                        if self.viewer.reference is not None:
                            self._draw_reference(width, canvas_height)
                        else:
                            overlay.placeholder(ctx)
                    imgui.end_child()
                elif not reference_stage and self.viewer.has_model:
                    self._draw_viewport_image(imgui.get_cursor_screen_pos(), width, height)
                elif reference_stage and self.viewer.reference is not None:
                    self._draw_reference(width, height)
                else:
                    overlay.placeholder(ctx)
                if tray:
                    imgui.separator()
                    generation_workspace.draw(ctx, tray_height)

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
                (cell, height),
                (0, 1),
                (1, 0),
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
            from . import troupe_mode
            from .panes import sheet_panel

            _step("release sheet strip", lambda: sheet_panel.release_strip_texture(ctx))
            # Troupe's atlas, for the same reason and by the same rule: it is
            # registered with the imgui backend by ``widgets.texture_ref``, so
            # it must be forgotten before it is released.
            _step("release troupe atlas", lambda: troupe_mode.release_texture(ctx))
            from . import inker_mode, packwright_mode, plotter_mode

            _step("release inker textures", lambda: inker_mode.release_all(ctx))
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
        ctx.settings.set("show_resources", ctx.state.show_resources)
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
        _crash_log.write(f"=== session {_utc_now()} pid={os.getpid()} warlock={_version()} ===\n")
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
    """The installed version, falling back to the packaged constant.

    A thin alias now: the implementation is ``warlock.installed_version``,
    because Home asked this module for it and a pane has no other business
    importing the frame loop.
    """
    from .. import installed_version

    return installed_version()


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
        # Unreachable in an ordinary launch since the single-instance lock went
        # in: a live second instance is refused before this runs. What can still
        # reach it is a *recycled* pid -- the marker names a process that died
        # and whose number the OS handed to something unrelated -- so the
        # sentence says what is actually known rather than asserting a second
        # Warlock the lock has already ruled out (RUN-01).
        log.warning(
            "the previous session's marker names pid %d (started %s), which is "
            "alive; the instance lock was free, so that is almost certainly a "
            "recycled pid rather than another Warlock",
            pid,
            data.get("started_at", "?"),
        )
        return
    log.warning(
        "the previous session (pid %d, started %s, warlock %s) did not shut down "
        "cleanly -- check crash.log and Windows event 2004",
        pid,
        data.get("started_at", "?"),
        data.get("version", "?"),
    )


def _write_session_marker() -> None:
    try:
        _marker_path().write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": _utc_now(),
                    "version": _version(),
                }
            ),
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
    _setup_logging()
    _install_excepthooks()
    log.info(
        "Warlock Studio %s starting: pid=%d python=%s argv=%s",
        _version(),
        os.getpid(),
        sys.version.split()[0],
        sys.argv[1:],
    )
    # Before ``migrate`` is even imported, because importing it *performs* the
    # one-time move: a second instance starting mid-copy is RUN-02's window, and
    # the whole point of this lock is to be taken before anything touches the
    # home directory. Also before the store is opened and before the runtime
    # claims the engine port.
    from .. import instance
    from ..config import get_config, source_checkout

    if not source_checkout():
        # Refused at the door with a sentence, rather than left to fail as a
        # missing ``trellis-server.exe`` at the first job (DST-01, D4). Every
        # native path resolves against the repository root, which inside a
        # wheel points under the environment -- and ``vendor/`` is not in the
        # wheel, because the binaries are manual downloads. Saying so is the
        # supported answer; pretending otherwise is not.
        log.error("not a source checkout; refusing to start (see DST-01)")
        instance.alert(
            "Warlock Studio must be run from its source checkout",
            "This copy of Warlock was installed as a package rather than run "
            "from a checkout, and it cannot find the native binaries it needs: "
            "they live in vendor/ beside the source and are downloaded by hand.\n\n"
            "Use the Warlock installer, or clone the repository and run:\n\n"
            "    uv sync --extra studio --extra text2image --extra rig\n"
            "    uv run warlock",
        )
        return 1
    try:
        config = get_config()
    except Exception as exc:
        # The first thing in this process that touches the disk, and until now
        # the only unguarded one. ``get_config`` runs ``migrate.run`` and then
        # ``mkdir``s four directories, so a home on a disconnected network
        # share, a read-only drive or a path the user has no rights to raises
        # ``OSError`` **here** -- before the window, before GL, before imgui,
        # and (under ``pythonw``) with stderr pointed at the null device. The
        # app simply did not appear, twice in a row, with nothing anywhere but
        # a log file in a directory that is itself the problem.
        #
        # ``instance.alert`` is the right tool and was already used twice in
        # the twenty lines above: it needs no window, no GL context and no
        # imgui, which is exactly the situation this is.
        log.exception("could not prepare the Warlock home directory")
        instance.alert(
            "Warlock Studio cannot use its home directory",
            f"{exc}\n\nWarlock keeps its library, job database and settings in "
            "a home directory it creates on first run, and it could not "
            "prepare that directory.\n\nCheck that the drive is connected and "
            "writable, or point WARLOCK_HOME at a directory you own.",
        )
        return 1
    lock = instance.InstanceLocks(instance.lock_paths(config))
    unsafe_lock = os.environ.get("WARLOCK_ALLOW_UNSAFE_LOCK") == "1"
    if not lock.acquire(allow_unsafe=unsafe_lock):
        # A dialog, not a log line. The behaviour this replaces wrote a warning
        # into warlock.log and carried on, so the second instance went on to
        # share the job database and the engine port with the first -- and
        # could terminate the first's trellis server -- with nothing on screen
        # to say why anything was going wrong (RUN-01).
        if lock.failure:
            log.error("instance locking failed for %s; refusing to start", lock.path)
            instance.alert(
                "Warlock Studio cannot protect its data",
                f"{lock.failure}\n\nWarlock stopped before opening the library because "
                "running without this protection can corrupt jobs or model files. "
                "Fix the directory permissions and try again. For emergency recovery "
                "only, set WARLOCK_ALLOW_UNSAFE_LOCK=1.",
            )
            return 1
        log.error("another Warlock instance holds %s; refusing to start", lock.path)
        instance.alert(
            "Warlock Studio is already running",
            "Another Warlock Studio is using this home, job database, or model "
            "directory.\n\nOnly one can use those resources at a time: sharing "
            "them can corrupt jobs or model installs.\n\nClose the other window "
            "and try again. A second copy needs a different WARLOCK_HOME, "
            "WARLOCK_DB, and WARLOCK_T2I_ROOT.",
        )
        return 1
    try:
        code = _run_locked()
    finally:
        lock.release()
    # The last thing this process does, and deliberately after the ``finally``
    # above: a worker parked on something that never returns -- the native file
    # dialogs block until dismissed, and by now the window they belong to is
    # gone -- is a non-daemon thread, so ``threading._shutdown`` would wait on
    # it forever with nothing on screen to say why. Everything that must happen
    # has happened by this line; what a hard exit skips is only the waiting.
    from .tasks import hard_exit_if_leaked

    return hard_exit_if_leaked(code)


def _offer_store_reset(exc: Any) -> bool:
    """Offer to set a broken job database aside. -> may we start over?

    The library index is a *record of jobs*, not the jobs themselves: the
    assets live in directories under the data dir and survive this untouched.
    What is lost is the history -- prompts, settings, verdicts, favourites --
    which is real and is why this is a question rather than a repair.

    Native, because it runs before the window exists; ``instance.ask`` answers
    No to anything that is not an explicit Yes, which is the right default for
    a button that moves somebody's library index.
    """
    from .. import db, instance

    log.error("the job database could not be opened", exc_info=exc.cause)
    agreed = instance.ask(
        "Warlock Studio cannot open its job database",
        f"{exc.path}\n\n{exc.cause}\n\nThis file is the library's index. Your "
        "generated assets are stored as ordinary folders beside it and are not "
        "affected, but the record of them -- prompts, settings, verdicts and "
        "favourites -- is in here.\n\nStart with an empty index? The damaged "
        "file is renamed and kept, not deleted, so it can be examined or "
        "recovered later.\n\nChoosing No leaves everything untouched and "
        "closes Warlock.",
    )
    if not agreed:
        return False
    moved = db.set_aside(exc.path)
    if moved is None:
        instance.alert(
            "Warlock Studio could not move the damaged database",
            f"{exc.path} could not be renamed, so a new index cannot be "
            "created beside it.\n\nThis usually means the file is open in "
            "another program or the directory is read-only.",
        )
        return False
    log.warning("job database set aside as %s; starting with an empty index", moved)
    return True


def _run_locked() -> int:
    """Everything after the single-instance lock is held."""
    from .. import migrate
    from ..config import get_config
    from ..db import StoreUnreadable
    from .runtime import Runtime

    if migrate.MOVED:
        # Said twice on purpose. The move itself printed to stderr because it
        # happened before this handler existed; the log is where somebody looks
        # a week later to find out why their library is not where they left it.
        log.info("moved into %s: %s", get_config().home, ", ".join(migrate.MOVED))
    _note_previous_session()
    _write_session_marker()
    try:
        try:
            return App(Runtime()).run()
        except StoreUnreadable as exc:
            # The one startup failure with an in-app way out, and it had none.
            # Offered rather than done, and offered exactly once: a second
            # ``StoreUnreadable`` after the rename is a fault in the *new*
            # file, which means the disk or the directory is the problem and
            # making another empty database would be a loop.
            if not _offer_store_reset(exc):
                return 1
            return App(Runtime()).run()
    except Exception as exc:
        # App.run reports and swallows its own failures, so anything arriving
        # here happened before the loop existed -- constructing the Runtime,
        # which opens the job store and claims the engine port.
        log.exception("Warlock Studio could not start")
        # And it is said out loud. This branch logged and returned 1, which
        # under ``pythonw`` is a process that starts, writes to a devnull
        # stderr and vanishes: the user double-clicks the icon and nothing at
        # all happens. ``instance.alert`` again -- there is no window to put a
        # dialog in, which is the whole reason that function exists.
        from .. import instance

        # A refusal with words of its own says them; anything else gets the
        # type and the message, which at least distinguishes "the port is in
        # use" from "the database is malformed" without opening the log.
        if isinstance(exc, StartupRefused):
            instance.alert(exc.title, exc.body)
        else:
            instance.alert(
                "Warlock Studio could not start",
                f"{type(exc).__name__}: {exc}\n\nWarlock stopped before its "
                "window opened. The full details are in warlock.log in your "
                "Warlock home directory.",
            )
        return 1
    finally:
        _clear_session_marker()


if __name__ == "__main__":
    sys.exit(run())
