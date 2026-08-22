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

from .. import memlog, winjob
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
# The post-download re-probe. Its own key rather than "health"'s, so a slow
# forced verification cannot be mistaken for the periodic poll and dropped by
# key-dedupe while the user is watching for it (UX-09).
VERIFY_KEY = "verify-install"


def _compare_key() -> str:
    """``library.COMPARE_KEY``, looked up lazily.

    A function rather than a module-level import: ``panes.library`` imports a
    great deal of the app and every other reference to it in this file is
    already deferred to its call site for that reason.
    """
    from .panes import library

    return library.COMPARE_KEY
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
# categories partition ``modes.KEYS`` exactly, and the partition is the guard
# on ``_build_ui``'s dispatch.
#
# The Manual and the profile manager both left this tuple when they stopped
# being modes (the UI redesign, wave 3): each is drawn from ``_overlays`` now, so
# neither has a dispatch branch to be reached by.
_SINGLE_PANE_MODES = ("home", "settings", "library")


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


def shortcut_sections() -> list[tuple[str, list[tuple[str, str]]]]:
    """Every binding the Ctrl+/ sheet lists, as data.

    Module-level and imgui-free so ``tests/manual/test_shortcuts.py`` can
    compare it against chapter 16 -- which is the gate that did not exist
    while the popup drifted into saying "F1 -- Switch to the Manual" and
    "thirteen modes" against a tree with ten and no such mode.
    Every group's rows are gathered before anything is drawn because the
    query decides which *groups* survive: a heading over nothing is a section
    that looks broken.
    """
    sections: list[tuple[str, list[tuple[str, str]]]] = []

    def table(title: str, rows: list[tuple[str, str]]) -> None:
        sections.append((title, rows))

    table(
        "Everywhere",
        [
            # No per-mode digit: ten modes against ten digits reads as a
            # promise of a stable mapping that the next mode breaks, and
            # the palette is the keyboard route to all of them.
            ("Ctrl+K", "Command palette -- switch mode, or open an asset"),
            ("Ctrl+/", "This list"),
            ("F1", "Open the manual over whatever is on screen"),
            ("Esc", "Close the topmost thing: the manual, then the profile "
                    "sheet, then a mode you passed through"),
            ("F10", "Toggle the frame-rate readout"),
        ],
    )
    # One heading, because there is one mode (the UI redesign, wave 5). The
    # rows that used to be split "2D" from "3D" are the same keys either
    # way -- what changed is which stage of Create you are standing on,
    # and the stage rail is a click rather than a shortcut, so there is
    # nothing here to key.
    table(
        "Create",
        [
            ("Ctrl+Enter", "Run the stage: Generate, or Make 3D"),
            ("Tab / Shift+Tab", "Move between the form's controls"),
            ("Enter", "Press the stage's button when it is the one focused"),
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
    table(
        "Review - a judging pass",
        [
            ("A", "Accept - files +3"),
            ("R", "Reject - files -3, rather than arming a negative"),
            ("S", "Skip, staying in the pass"),
            ("Esc", "End the pass and show its report"),
        ],
    )
    from . import inker_state
    from .clay_mode import TOOL_KEYS as CLAY_KEYS
    from .inker_mode import ALT_TOOL_CHORDS

    table(
        "Clay",
        [
            (
                " / ".join(k.upper() for k in CLAY_KEYS),
                # Capitalised here rather than in ``TOOL_KEYS``: those
                # values are the tool *ids* ``state.tool`` is compared
                # against and the saved documents carry, and this is the
                # one place they are read as English. Joined lowercase
                # they were the only row in the popup that did not start
                # with a capital, one line above "Vertex / edge / face".
                " / ".join(CLAY_KEYS.values()).capitalize(),
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
            # "Ctrl+N / O / W", matching Inker's row below rather than
            # stopping at O: Clay closes a document with Ctrl+W like every
            # other document mode, and the popup simply never said so
            # (UX-13). The axis views were missing from both this popup and
            # the manual's "full list", which made six of them
            # undiscoverable.
            ("Ctrl+N / O / W", "New / open / close"),
            ("Ctrl+E", "Export to the library"),
            ("Ctrl+Tab", "Next document"),
            ("Ctrl+1 / 3 / 7", "Look along front / right / top"),
            ("Ctrl+Shift+1 / 3 / 7", "The opposite view: back / left / bottom"),
            ("Ctrl+5", "Orthographic / perspective"),
        ],
    )
    # **The letters, named, six to a row.** This was one squashed row
    # reading "A, B, C, D, E, ..." with the note "hover a tool for its
    # letter" -- which is a shortcut sheet declining to be one, and the
    # only mode's table that did. Six per row keeps the two columns
    # readable, and the order is the toolbox's, so the pairs that sit
    # together there (brush/spray, line/curve, the two lassos) sit
    # together here.
    tool_rows = []
    band = list(inker_state.TOOLS)
    for start in range(0, len(band), 6):
        chunk = band[start : start + 6]
        tool_rows.append(
            (
                " / ".join(letter for _key, _label, letter in chunk),
                " / ".join(label for _key, label, _letter in chunk),
            )
        )
    # Aseprite files these two-to-a-slot and cycles with Shift; here they
    # are second bindings beside the plain letters, so they are listed
    # rather than left to the tooltips.
    alt = " / ".join(
        f"{chord} {inker_state.tool_label(tool)}"
        for tool, chord in ALT_TOOL_CHORDS.items()
    )
    table(
        "Inker",
        [
            *tool_rows,
            (alt, "The same tools on Aseprite's shifted letters"),
            ("X", "Swap colours"),
            ("1 - 0", "Brush opacity, 10% to 100%"),
            ("Alt+1 - 9", "Recall a numbered custom brush"),
            ("Alt+Shift+1 - 9", "Store the captured brush in that slot"),
            ("Tab", "Show or hide the timeline -- which is where the layers are"),
            ("Ctrl+Shift+N", "New layer"),
            ("Ctrl+Shift+Up / Down", "Move the layer up / down the stack"),
            ("[ / ]", "Brush size (Shift: hardness)"),
            ("Shift+click", "Paint a line from where the last stroke ended"),
            ("+ / -", "Zoom in / out, by whole scales"),
            ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
            ("Space / middle drag", "Pan (wheel zooms in 5% steps)"),
            ("Ctrl+4 / Ctrl+5", "Rotate the view a quarter turn / flip it"),
            ("Arrows", "Nudge a pixel (Shift: eight)"),
            ("Delete", "Delete what is selected"),
            ("Esc", "Cancel -- a move, playback, a float, then the selection"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            ("Ctrl+E", "Save as a reference in the library"),
            ("Ctrl+Shift+E", "Export PNG"),
            ("Ctrl+N / O / W", "New / open / close"),
            ("Ctrl+A / D", "Select all / deselect"),
            ("Ctrl+Shift+D", "Reselect what was last dismissed"),
            ("Ctrl+C / X / V", "Copy / cut / paste"),
            ("Ctrl+Shift+V", "Paste as a layer"),
            ("Ctrl+J / Ctrl+Shift+J", "Copy / move the selection to its own layer"),
            ("Ctrl+Shift+I", "Invert the selection"),
            ("Ctrl+T", "Free transform"),
            ("Ctrl+B", "Capture the selection as an image brush"),
            ("Ctrl+Tab", "Next tab"),
            (", / .", "Previous / next frame (animated)"),
            ("Enter", "Play or pause (animated)"),
        ],
    )
    from .plotter_state import TOOLS as PLOTTER_TOOLS

    table(
        "Plotter",
        [
            ("1 - 9", "Recall a numbered stamp"),
            ("Ctrl+Shift+1 - 9", "Store the stamp in hand in that slot"),
            ("Right-drag", "Capture a block off the map, keeping the tool"),
            ("Right-click an object", "Duplicate, raise, lower or delete it"),
            ("H", "Highlight the current layer"),
            ("+ / -", "Zoom in / out, by whole scales"),
            ("Ctrl+Shift+I", "Invert the selection"),
            (
                " / ".join(letter for _k, _l, letter in PLOTTER_TOOLS),
                " / ".join(label for _k, label, _letter in PLOTTER_TOOLS),
            ),
            ("X / Y / Z", "Flip the brush across, down; turn it (Shift turns back)"),
            ("Shift+click", "Stamp a line from the last cell painted"),
            ("Pick drag", "Capture a block off the map as the brush"),
            ("Wand Ctrl+click", "Select every cell of that tile, map-wide"),
            ("Shift / Alt", "Add to / subtract from the selection (Wand and marquee)"),
            ("Ctrl+A / Ctrl+D", "Select all / deselect (Ctrl+Shift+A also)"),
            ("Ctrl+C / Ctrl+X / Ctrl+V", "Copy / cut / paste as the brush"),
            ("Ctrl+J", "Duplicate the selected object"),
            ("Ctrl+click / Alt+click", "Insert / remove a polygon vertex"),
            ("Delete", "Clear the selection, or remove the object"),
            ("Ctrl+Z / Ctrl+Y", "Undo / redo"),
            ("Ctrl+S / Ctrl+Shift+S", "Save / save as"),
            ("Ctrl+E", "Export to the library"),
            ("Ctrl+Shift+E", "Export a Tiled .tmx"),
            ("Ctrl+N / O / W", "New / open / close"),
            ("Ctrl+G", "Toggle the grid"),
            ("Ctrl+Tab", "Next map"),
            ("Ctrl+0 / Ctrl+1", "Fit / 100%"),
            ("Space / middle drag", "Pan (wheel zooms)"),
            ("Esc", "Cancel a drag, then the object, then the selection"),
        ],
    )
    table(
        "Poser",
        [
            # The mode is otherwise mouse-shaped -- joints are clicked and
            # gizmos are dragged -- which is why two rows are the whole group
            # and not a sign that the rest were forgotten.
            ("Ctrl+Z / Ctrl+Y", "Undo / redo (Ctrl+Shift+Z also redoes)"),
            ("Esc", "Deselect the joint"),
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
    table(
        "Troupe",
        [
            ("Space", "Play / pause the preview"),
            # Stepping pauses, which is why the two rows are not "step" alone:
            # the binding does two things and a sheet that named one of them
            # would be describing a different control.
            ("Left / Right", "Step one frame, and pause"),
        ],
    )
    return sections


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
    * **A handle at all.** Six of the seven workspaces drew a proportion the
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
        width=0,
        edge=layout_mod.PaneEdge.LEFT,
        top=("inspector", layout_mod.PaneRole.INSPECTOR, inspector_draw),
        bottom=("library", layout_mod.PaneRole.SIDEBAR, library_draw),
    )


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
        fonts.load(imgui)
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
        from .panes import first_run

        # Sampled once. The marker is intentionally outside studio_settings:
        # resetting preferences must not turn setup into an annual popup.
        self.app_ctx.first_run = first_run.pending(self.svc.config)
        device = getattr(self.runtime, "device_memory", None)
        self.app_ctx.gpu_name = str(getattr(device, "name", "") or "")
        self.app_ctx.dpi_scale = monitor_scale
        self.app_ctx.layout = self.layout
        # Every ``widgets.field_error`` call site gets the Install offer at
        # once, without widgets importing a pane. Bound to this Ctx, so a
        # second App in one process replaces it rather than stacking.
        from . import widgets as widgets_mod
        from .panes import model_gate

        widgets_mod.set_install_offer(
            lambda field: model_gate.install_offer(self.app_ctx, field)
        )
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
                    self.fps.frames, time.perf_counter() - self._started_at,
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
        summary = memlog.summary(children=winjob.tracked())
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

        self.app_ctx.textures.begin_frame()
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
            fonts.reload(imgui)
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
                elif done.key.startswith("matte-"):
                    from . import matte_preview

                    # The seventh, and it was missing. ``matte_preview.pump``
                    # re-submits whenever there is no cached cutout for the
                    # current stamp, and ``settings_3d.matte_modal`` runs it
                    # every frame -- so a failure that left no note behind was
                    # re-submitted, re-failed and re-toasted at the frame rate.
                    matte_preview.on_task_failed(ctx, done)
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
        if key == "model-storage":
            if isinstance(done.result, dict):
                ctx.model_storage = done.result
            return
        if key == "sweep-staging":
            # Silent when there was nothing to reclaim, which is every launch
            # that did not follow a cancelled fetch. Said out loud when there
            # was: disk quietly reappearing is the kind of thing a user should
            # be told about rather than discover in a folder listing.
            removed = done.result if isinstance(done.result, list) else []
            if removed:
                noun = "tree" if len(removed) == 1 else "trees"
                ctx.toast(
                    f"Reclaimed {len(removed)} staging {noun} left by a "
                    f"cancelled download."
                )
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
        if key.startswith(
            ("delete:", "prune", "rename:", "name:", "tags:", "fav:", "restore:", "purge:")
        ) or key == "empty-trash":
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
        # Scanned here, on the first frame that has a Ctx, and *offered* by the
        # home screen rather than by a modal in front of it. It has to be here
        # and not in the pane: the autosave directory is also where this
        # session's copies land, so a scan taken any later than the first frame
        # would hand the user their own open documents back. ``snapshot`` is a
        # no-op after the first call, which is what makes calling it per frame
        # correct rather than merely cheap.
        journal.snapshot(ctx)

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
                pygame.display.set_mode(
                    sized, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
                )
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
        """Whether *any* modal is on screen and owns the keyboard.

        It used to know about exactly two: the confirm queue and the prompt
        queue. The matte preview is a third -- a real modal, drawn in front of
        the promotion, with its own Accept and Cancel -- and every global
        shortcut leaked straight through it (UX-08). Ctrl+K opened the palette
        behind it; Ctrl+Enter submitted the form the modal was a *question
        about*; a mode key left the app somewhere else with the modal still up.
        Ownership is a property of "a modal is up", not of which queue happens
        to hold it, so the predicate asks all three.
        """
        from . import matte_preview
        from .panes import first_run

        ctx = self.app_ctx
        return (
            ctx.confirms.pending is not None
            or ctx.prompts.pending is not None
            or matte_preview.is_open(ctx)
            or first_run.is_open(ctx)
        )

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
        # Esc closes the Manual before anything else looks at it, and that
        # ordering is the whole of why this sits here rather than in
        # ``_escape_mode``: the workspace modes below consume every key they
        # are handed, so an Esc dispatched to Inker with the overlay up would
        # drop a floating selection and leave the reference open on top of it.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
            and ctx.state.manual.open
        ):
            from .manual import render as manual_render

            manual_render.close(ctx)
            return
        # Then the profile sheet, for the same reason and in this order: the
        # Manual can be raised *over* the manager (its (?) does exactly that),
        # so the topmost surface is the one an Esc is about. This one goes
        # through the panel's own guard, so a half-typed profile still asks.
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
            and ctx.state.profiles_open
        ):
            from .panes import profiles_panel

            profiles_panel.close_sheet(ctx)
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
                ctx.toast(
                    "Packwright opens .wpack documents and packs image files.", "error"
                )
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
        from . import clay_mode, inker_mode, packwright_mode, plotter_mode, poser_mode
        from .panes import pose_panel, profiles_panel

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
            # A profile draft is a document too -- nine fields and an anchor
            # image -- and it was the only one this chain did not know about
            # (UX-17).
            profiles_panel.guard,
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
        from . import modes, rail, tokens
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
        layout_mod.measure()
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
        )
        imgui.begin("##host", None, flags)
        # One frame, one record of where every pane ended up -- and one answer
        # to "is the layout editor open", which every splitter reads (P5.4).
        from . import layout_edit

        layout_mod.begin_frame(layout_edit.ensure(ctx.state).open)
        # The rail is drawn in every mode, Home included: it is how you leave
        # wherever you are, so a mode that hides it is a dead end.
        rail.draw(self, ctx)
        # The two popups its footer can raise, opened at *host* scope. imgui
        # registers a popup in the window that opens it and the rail is a
        # child, so a footer that called ``open_popup`` itself would be naming
        # a popup nothing outside the child could find. The same one-shot the
        # shortcuts button has always used, generalised.
        if rail.take("diagnostics"):
            imgui.open_popup("diagnostics")
        self._diagnostics_popup(list(getattr(ctx.runtime, "checks", []) or []))
        if rail.take("layouts"):
            imgui.open_popup("layouts")
        self._layouts_popup(ctx)
        # Kept behind an explicit developer environment flag; normal installs
        # never gain a design-system destination in their navigation.
        from . import component_gallery

        component_gallery.draw()
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
        self._shortcuts_popup()
        imgui.same_line()
        imgui.begin_child("##content", (0, 0))
        from .panes import overlay

        overlay.doctor_banner(ctx)
        mode = ctx.state.mode
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
            elif mode == "troupe":
                self._troupe_workspace()
            else:
                self._inker_workspace()
            imgui.end_child()
            imgui.end()
            self._overlays(viewport)
            return

        # The library used to share the left sidebar with settings, split by
        # settings_share; it shares the right sidebar with the inspector now
        # instead, so the left column is settings alone (nothing left to split
        # against) and the right column is the two-scroller stack that used to
        # live on the left.

        lay = self.layout
        sidebar_w = layout_mod.sidebar_width()
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
        pad = tokens.sp(layout_mod.PANE_PADDING)
        rail_w = imgui.get_content_region_avail().x - pad * 2
        imgui.indent(pad)
        self._stage_rail(ctx, max_width=rail_w)
        imgui.unindent(pad)
        with layout_mod.pane(
            "settings",
            (sidebar_w, 0),
            layout_mod.PaneRole.SIDEBAR,
            edge=layout_mod.PaneEdge.RIGHT,
        ) as visible:
            if visible:
                _stage_pane(ctx)

        imgui.same_line()
        self._viewport_pane()
        imgui.same_line()

        _right_column(
            ctx, lay, sidebar_w, inspector_draw=inspector.draw, library_draw=library.draw
        )

        imgui.end_child()
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
            max_width=(
                imgui.get_content_region_avail().x if max_width is None else max_width
            ),
        )
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

    def _poser_workspace(self) -> None:
        """The sidebar / centre / sidebar skeleton, Poser's way:

            [ poser_library + poser_clips ]  viewport  [ poser_controls ]

        One pane per side rather than Clay's stacked pairs: the library and the
        controls are each one scroller, and an empty half-pane would be chrome.
        The clip editor is a *section* inside the left scroller for that same
        reason -- a skeleton with no clips shows one collapsed heading, where a
        split pane would show an empty half.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .panes import poser_clips, poser_controls, poser_library

        ctx = self.app_ctx
        sidebar_w = layout_mod.sidebar_width()
        with layout_mod.pane(
            "poser-library",
            (sidebar_w, 0),
            layout_mod.PaneRole.SIDEBAR,
            edge=layout_mod.PaneEdge.RIGHT,
        ) as visible:
            if visible:
                poser_library.draw(ctx)
                # Under the pose library rather than in a mode of its own: a
                # clip is a *library* of the same kind of thing, and the right
                # sidebar has to stay free for the joint controls, which are
                # the actual editing surface for a key.
                poser_clips.draw(ctx)

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "poser-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                self._poser_viewport(ctx)

        imgui.same_line()
        with layout_mod.pane(
            "poser-controls",
            (0, 0),
            layout_mod.PaneRole.INSPECTOR,
            edge=layout_mod.PaneEdge.LEFT,
        ) as visible:
            if visible:
                poser_controls.draw(ctx)

    def _poser_viewport(self, ctx: Any) -> None:
        from imgui_bundle import imgui

        from . import icons, poser_mode, widgets
        from .panes import overlay

        self._poser_hovered = False
        state = poser_mode.ensure(ctx)
        if not ctx.rigging_available:
            overlay.placeholder(ctx)
            return
        viewer = self._ensure_poser_viewer()
        showing = poser_mode.sync_preview(ctx, viewer)
        if not showing:
            # Both branches through ``centred_empty``, the shape every other
            # workspace's empty viewport takes. These two were the app's one
            # pair of top-left muted lines where nine centred cards go.
            if state.building:
                overlay.centred_empty(
                    icons.PERSON_STANDING,
                    "Building the skeleton preview",
                    "The armature is built by Blender once per skeleton and "
                    "cached; the first open of a template takes a moment.",
                )
            elif state.error:
                overlay.centred_empty(
                    icons.TRIANGLE_ALERT, "The skeleton did not build", state.error
                )
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
        self._poser_menu(ctx, viewer)

    def _poser_menu(self, ctx: Any, viewer: Any) -> None:
        """The joint's right-click menu (B7).

        Drawn here rather than in a pane because a popup belongs to the window
        that begins it and this is that window -- the same reason
        ``clay_menu`` is called from Clay's viewport. The viewer records
        ``menu_request`` and knows nothing about imgui.
        """
        from imgui_bundle import imgui

        from . import controls, widgets

        popup = "poser-joint-menu"
        if viewer.menu_request is not None:
            viewer.menu_request = None
            imgui.open_popup(popup)
        if not imgui.begin_popup(popup):
            return
        widgets.popup_chrome(_imgui=imgui)
        selected = viewer.editor.selected
        if selected is None:
            widgets.secondary("No joint selected")
        else:
            widgets.secondary(str(selected))
            imgui.separator()
            # Through the *viewer*, never the editor: every one of these has a
            # ``_after_pose_change`` behind it that re-skins the preview, which
            # is exactly the step a direct editor call would skip.
            if controls.menu_item_simple("Clear this joint's rotation"):
                viewer.reset_bone()
            if controls.menu_item_simple("Deselect"):
                viewer.editor.selected = None
        imgui.separator()
        if controls.menu_item_simple("Reset the whole pose"):
            viewer.reset_all()
        imgui.end_popup()

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

    def _clay_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses.

        Mirrors ``_inker_workspace`` line for line, including its share key
        for the vertical split, so the two editors do not drift into looking
        like different applications:

            [ clay_tools ]            [ clay_outliner ]
            [ clay_props ]  viewport  [ clay_bridge   ]
        """
        from imgui_bundle import imgui

        from . import clay_mode, widgets
        from . import layout as layout_mod
        from .panes import clay_bridge, clay_outliner, clay_props, clay_tools

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = layout_mod.sidebar_width()

        _split_column(
            ctx,
            lay,
            split_id="clay-tools",
            handle_length=sidebar_w,
            width=sidebar_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=("clay-tools", layout_mod.PaneRole.SIDEBAR, clay_tools.draw),
            bottom=("clay-props", layout_mod.PaneRole.SIDEBAR, clay_props.draw),
        )

        imgui.same_line()
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "clay-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                self._clay_viewport(ctx, clay_mode, widgets)

        imgui.same_line()
        _split_column(
            ctx,
            lay,
            split_id="clay-outliner",
            handle_length=sidebar_w,
            width=0,
            edge=layout_mod.PaneEdge.LEFT,
            top=("clay-outliner", layout_mod.PaneRole.INSPECTOR, clay_outliner.draw),
            bottom=("clay-bridge", layout_mod.PaneRole.INSPECTOR, clay_bridge.draw),
        )

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
        if state.frame_pending:
            # The other half of ``F``: the mode recorded the intent and this is
            # the only place that has the viewport to act on it (B6).
            state.frame_pending = False
            self._frame_clay_selection()
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

        from . import widgets

        # This was written as a copy of ``inker_canvas._empty`` and the copy
        # dropped the ``sp()`` scaling, so at 150 % the raster editor's empty
        # state grew with the text while Clay's kept 240-*physical*-pixel
        # buttons under 1.5x labels -- which is where a label stops fitting its
        # button. Both are one function now (the UI redesign, wave 2), which is the
        # only fix that also holds for the next copy.
        widgets.nothing_open(
            "Start a model, open a document, or drop a .wblk on the window.",
            [
                ("New model", lambda: clay_mode.new_document(ctx)),
                ("Open a file...", lambda: clay_mode.ask_open(ctx)),
            ],
            # No recent list here: it is the bridge panel's, on both of its
            # branches, which is where Plotter and Packwright keep theirs (B5).
        )

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
        from . import skeletons
        from .panes import inker_canvas, inker_timeline
        from .tokens import sp

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = layout_mod.sidebar_width()
        tab = None if ctx.state.inker is None else ctx.state.inker.active
        animated = tab is not None and tab.doc.anim is not None
        # The tile panel appears with the tilesets, the way the preview appears
        # with the frames: a drawing that has never seen a tilemap layer is
        # byte-for-byte the workspace it always was, and a fixed-height palette
        # taken out of every Inker session for a feature most of them do not
        # use is a cost with no matching benefit. The *verbs* that make the
        # first tileset are not hidden with it -- they are menu rows, which are
        # always drawn.
        # (The tile panel's own ``when`` predicate answers that in the
        # skeleton table, which is where the shape of a workspace lives now.)

        # **The toolbox is a fixed 90 px rail** -- no share, no handle, no
        # give-way. All three existed because the pane under the toolbox was
        # the tool options, whose height nobody could predict; the options are
        # a row above the canvas now (W2.4) and what is left is twelve buttons,
        # three toggles and two colour chips, which is a *known* width and no
        # height worth arguing about.
        #
        # Both sidebars go through ``layout.column`` over ``skeletons.inker``
        # (P5.1): one renderer, one height arithmetic, and a saved layout has
        # something to be a permutation of.
        columns = skeletons.for_mode(ctx, "inker")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "inker", columns["left"]),
            width=sp(columns["left"].width),
            handle_length=sp(columns["left"].width),
        )

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
        if animated:
            with layout_mod.pane(
                "inker-timeline",
                (width, 0),
                layout_mod.PaneRole.SHEET,
                edge=layout_mod.PaneEdge.TOP,
            ) as visible:
                if visible:
                    inker_timeline.draw(ctx)
        imgui.end_group()

        imgui.same_line()
        # **Preview / Colours / Tiles**, top to bottom, with a handle between
        # each adjacent shareable pair. The colour panel moved here from the
        # left column in W2.9 because the palette grid is a *panel* and always
        # wanted the width; the two colours themselves stayed at the foot of
        # the rail, which is Aseprite's "Mirrored Default" shape.
        #
        # Which panes are here, and in what order, is now the active saved
        # layout's answer (wave 5) -- reconciled against this table every read
        # and never written back.
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "inker", columns["right"]),
            handle_length=sidebar_w,
            on_hidden=lambda _slot: None,
        )

    def _plotter_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses:

            [ plotter-tools   ]           [ plotter-layers ]
            [ plotter-tileset ]  the map  [ plotter-bridge ]

        Mirrors ``_clay_workspace`` line for line, its share key included,
        so the editors do not drift into looking like different applications.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from . import skeletons
        from .panes import plotter_canvas, plotter_tileset_editor

        ctx = self.app_ctx
        lay = self.layout
        sidebar_w = layout_mod.sidebar_width()
        # Both sidebars through ``layout.column`` over ``skeletons.plotter``
        # (wave 5), so the arrangement is data a saved layout can permute.
        columns = skeletons.for_mode(ctx, "plotter")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "plotter", columns["left"]),
            width=sidebar_w,
            handle_length=sidebar_w,
        )

        imgui.same_line()
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

        imgui.same_line()
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "plotter", columns["right"]),
            handle_length=sidebar_w,
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
        sidebar_w = layout_mod.sidebar_width()

        _split_column(
            ctx,
            lay,
            split_id="troupe-cast",
            handle_length=sidebar_w,
            width=sidebar_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=("troupe-cast", layout_mod.PaneRole.SIDEBAR, troupe_characters.draw),
            bottom=("troupe-settings", layout_mod.PaneRole.SIDEBAR, troupe_settings.draw),
        )

        imgui.same_line()
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

        imgui.same_line()
        _split_column(
            ctx,
            lay,
            split_id="troupe-sheets",
            handle_length=sidebar_w,
            width=0,
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
        sidebar_w = layout_mod.sidebar_width()

        _split_column(
            ctx,
            lay,
            split_id="packwright-sources",
            handle_length=sidebar_w,
            width=sidebar_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=("packwright-sources", layout_mod.PaneRole.SIDEBAR, packwright_sources.draw),
            bottom=("packwright-settings", layout_mod.PaneRole.SIDEBAR, packwright_settings.draw),
        )

        imgui.same_line()
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

        imgui.same_line()
        _split_column(
            ctx,
            lay,
            split_id="packwright-items",
            handle_length=sidebar_w,
            width=0,
            edge=layout_mod.PaneEdge.LEFT,
            top=("packwright-items", layout_mod.PaneRole.INSPECTOR, packwright_items.draw),
            bottom=("packwright-bridge", layout_mod.PaneRole.INSPECTOR, packwright_bridge.draw),
        )

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

        ctx = self.app_ctx
        state = review_mode.ensure(ctx)
        lay = self.layout
        sidebar_w = layout_mod.sidebar_width()

        _split_column(
            ctx,
            lay,
            split_id="review-runs",
            handle_length=sidebar_w,
            width=sidebar_w,
            edge=layout_mod.PaneEdge.RIGHT,
            top=(
                "review-runs",
                layout_mod.PaneRole.SIDEBAR,
                lambda _ctx: self._review_runs(ctx, state, review_mode),
            ),
            bottom=(
                "review-units",
                layout_mod.PaneRole.SIDEBAR,
                lambda _ctx: self._review_units(state, review_mode),
            ),
        )

        imgui.same_line()
        width = layout_mod.centre_width()
        # The labelling grid replaces the viewport rather than sitting beside it:
        # a mesh on screen under a question about a *picture* is the mismatch that
        # files an accept about the wrong artifact. It also scrolls, so it must
        # not inherit the viewport's no-scroll flag.
        labelling = state.labels is not None
        flags = 0 if labelling else imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "review-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                if labelling:
                    self._review_labels(ctx, state, review_mode)
                else:
                    self._review_viewport(state, review_mode, width)

        imgui.same_line()
        with layout_mod.pane(
            "review-verdict",
            (0, 0),
            layout_mod.PaneRole.INSPECTOR,
            edge=layout_mod.PaneEdge.LEFT,
        ) as visible:
            if visible:
                if labelling:
                    self._review_label_panel(ctx, state, review_mode)
                else:
                    self._review_verdict(ctx, state, review_mode)

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

        from . import controls, widgets

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
        # One sentence for the three of them: they share a gate, and three
        # spellings of "there is nothing on screen to judge" would read as
        # three different problems. The ``_VIEWPORT_WHY`` pattern.
        no_row = "There is nothing left to label in this pass."
        if widgets.primary_button("Good (A)", enabled=row is not None):
            review_mode.record_label(ctx, "accept")
        imgui.same_line()
        if widgets.disabled_button("Bad (R)", row is not None, reason=no_row):
            review_mode.record_label(ctx, "reject")
        imgui.same_line()
        if widgets.disabled_button("Skip (S)", row is not None, reason=no_row):
            review_mode.advance_labels(labels)
        if controls.button("Done", role=controls.ButtonRole.GHOST):
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

        from . import controls, icons, widgets
        from .manual import render as manual_render

        self._review_judging_card(ctx, state, review_mode)
        widgets.section("Sweeps")
        manual_render.help_button(ctx, "review")
        if widgets.disabled_button(
            f"{icons.REFRESH} Rescan",
            not state.scanning,
            reason="A scan is already running.",
        ):
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
            if controls.selectable(f"{sweep['label']}##sweep-{sweep['id']}", selected)[0]:
                # Picking a sweep by hand leaves the pass: the pass is a walk
                # over every outstanding bucket in a stated order, and a user
                # who jumps out of that order is no longer on the walk its
                # header is counting. Cleared here rather than inside
                # ``open_sweep``, which the pass itself calls.
                state.judging = None
                review_mode.open_sweep(ctx, sweep["id"])
            widgets.muted(f"   {total - todo}/{total} reviewed")
            # What the run actually varied, under the name the user typed for
            # it at the time -- which is routinely "test2" by the time anyone
            # comes back to judge it.
            summary = review_mode.spec_summary(sweep.get("spec"))
            if summary:
                widgets.muted(f"   {summary}")
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
            if controls.selectable(f"{title}##label-{stage}", open_here)[0]:
                if open_here:
                    review_mode.close_labels(ctx)
                else:
                    review_mode.open_labels(ctx, stage)
        imgui.separator()
        self._review_form(ctx, state, review_mode)

    def _review_judging_card(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """The offer to start a guided pass, or the report from the last one.

        A card at the top of the column rather than a modal, deliberately: the
        pass is a *convenience* over the loop that is already on screen, and a
        dialog in front of Review would make judging feel like something you
        have to commit to before you can look at anything.
        """
        from imgui_bundle import imgui

        from . import widgets

        if state.judging_report is not None:
            self._review_judging_report(ctx, state, review_mode)
            return
        if state.judging is not None or state.scanning:
            # Nothing to offer: the pass is running (its controls are in the
            # verdict pane, beside the mesh they are about) or the list is still
            # being read and its counts are not yet true.
            return
        outstanding = review_mode.todo_total(state)
        if outstanding <= 0:
            return
        widgets.section("Judging")
        widgets.muted(f"{outstanding} unit(s) across every bucket have no verdict.")
        if widgets.primary_button("Start judging", (-1, 0)):
            review_mode.start_judging(ctx)
        # The up-front warning, and the only one there is. The user chose no
        # dialog, so this sentence is carrying the whole of the notice that a
        # judged sweep's files are about to go -- which is why it says what
        # survives as well as what does not.
        widgets.hint_text(
            "One at a time, Accept or Reject. Once every unit of a sweep has "
            "been judged its images and meshes are removed automatically; the "
            "verdicts and findings they produced are kept."
        )
        imgui.separator()

    def _review_judging_report(self, ctx: Any, state: Any, review_mode: Any) -> None:
        """What the pass that just ended did.

        Drawn from the stored dict and never recomputed: the numbers were
        tallied once, in memory, at the moment the pass ended -- recomputing
        per frame would be a table scan behind the one serialized connection,
        every frame, for a card that says the same thing each time.
        """
        from imgui_bundle import imgui

        from . import controls, widgets

        report = state.judging_report
        widgets.section("Judging pass")
        # Wrapped, not ``muted``: these are sentences with a sweep's own name in
        # them, in a 300 px sidebar, and the unwrapped form clipped both the
        # average grade off the end of every row and the word "do" off the
        # overall line -- so the two numbers the card exists to report were the
        # two the reader could not see.
        for row in report["sweeps"]:
            line = (
                f"{row['label']}: {row['accepted']} accepted / "
                f"{row['rejected']} rejected of {row['total']}"
            )
            if row["mean_grade"] is not None:
                line += f", avg {row['mean_grade']:+.1f}"
            widgets.muted_wrapped(line)
        widgets.muted_wrapped(
            f"{report['filed']} filed this pass - {report['accepted']} accepted, "
            f"{report['rejected']} rejected, {report['remaining']} still to do."
        )
        if controls.button("Dismiss", role=controls.ButtonRole.GHOST):
            review_mode.dismiss_report(ctx)
        imgui.separator()

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
        from . import controls, widgets

        # The label table for every guidance field, which is where a param's
        # human name already lives. Resolved here rather than in ``review_mode``
        # for that module's own rule: it may not import a pane.
        from .panes import settings_2d

        if not widgets.header("New sweep", default_open=False):
            return
        form = state.form
        widgets.field_label("prompt")
        form.prompt = widgets.multiline("##sweep-prompt", form.prompt, 60, 1000)
        widgets.field_label("name")
        form.label = widgets.input_text("##sweep-label", form.label, max_length=120)
        widgets.field_label("seeds")
        form.seeds = widgets.input_text("##sweep-seeds", form.seeds, max_length=120)

        if controls.button("Start from current 2D/3D settings"):
            form.base = review_mode.capture_base(ctx)
            form.base_note = f"{len(form.base)} setting(s) captured"
            ctx.toast("Captured the current settings as this sweep's baseline.")
        widgets.muted(
            form.base_note
            # Names the button, because "the defaults" is a fact about a sweep
            # that is unreproducible rather than merely unconfigured -- and the
            # remedy is one control away and was not being pointed at.
            or "No baseline captured; units use the defaults. Press "
            '"Start from current 2D/3D settings" above to use your own.'
        )

        # "what to vary", not "vary": the old label was a verb with no object,
        # over a combo of thirty raw param names.
        widgets.field_label("what to vary")
        rows = {row["param"]: row for row in review_mode.axis_options(ctx)}
        options = [("", "-")] + [
            (p, settings_2d.field_label(p)) for p in sweeps_mod.axis_params()
        ]
        for i, row in enumerate(form.axes):
            imgui.push_id(f"axis-{i}")
            row["param"] = widgets.combo("##param", row.get("param", ""), options, width=-1)
            self._review_axis_values(row, rows.get(row.get("param") or ""))
            imgui.pop_id()
        if controls.button("Add axis"):
            form.axes.append({"param": "", "values": ""})
        if len(form.axes) > 1:
            imgui.same_line()
            if controls.button("Remove axis", role=controls.ButtonRole.GHOST):
                form.axes.pop()

        planned = review_mode.preview_units(state)
        if planned < 0:
            widgets.muted("Fill in the prompt and one axis.")
        else:
            labels = {p: settings_2d.field_label(p) for p in rows}
            widgets.muted_wrapped(review_mode.preview_line(state, labels))
            widgets.muted(f"Roughly two minutes of GPU each - {planned * 2} minutes in all.")
        enabled = planned > 0 and not form.submitting and not state.scanning
        if widgets.primary_button("Launch sweep", (-1, 0), enabled=enabled):
            review_mode.launch(ctx)

    def _review_axis_values(self, row: dict[str, Any], spec: Any) -> None:
        """One axis row's values, drawn as whatever the param actually accepts.

        **Every kind writes back the same comma-separated string.** That is the
        whole design: ``build_plan`` and ``_coerce`` parse one representation
        and are untouched, so this is a better *control* over the existing
        field rather than a second way of storing an axis -- and a param the
        catalog cannot resolve falls back to the free-text field the row has
        always been, which is why an unknown param is less discoverable rather
        than broken.
        """
        from . import controls, widgets

        kind = (spec or {}).get("kind", "text")
        if spec and spec.get("help"):
            widgets.help_marker(spec["help"])
        if kind in ("options", "bool"):
            entries = (
                spec["options"]
                if kind == "options"
                else [{"key": "true", "label": "on"}, {"key": "false", "label": "off"}]
            )
            chosen = [v.strip() for v in (row.get("values") or "").split(",") if v.strip()]
            for entry in entries:
                key = entry["key"]
                changed, _ticked = controls.checkbox(f"{entry['label']}##{key}", key in chosen)
                if changed:
                    # Rebuilt from the *entry order* rather than by appending, so
                    # the string the user sees back is stable however they
                    # clicked -- and so a value typed by hand and then unticked
                    # cannot leave a duplicate behind.
                    picked = set(chosen) ^ {key}
                    chosen = [e["key"] for e in entries if e["key"] in picked]
                    row["values"] = ", ".join(chosen)
            return
        hint = "comma-separated"
        if kind == "number" and spec.get("range"):
            low, high = spec["range"]
            default = spec.get("default")
            hint = f"{low}-{high}" + (f", e.g. {default}" if default is not None else "")
        row["values"] = widgets.input_text(
            "##values", row.get("values", ""), max_length=200, hint=hint
        )

    def _review_units(self, state: Any, review_mode: Any) -> None:
        from . import controls, icons, widgets

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
            if controls.selectable(
                f"{mark} {review_mode.label(state, unit)}##unit-{unit['job_id']}",
                i == state.index,
            )[0]:
                review_mode.step(state, i - state.index)

    def _review_judging_controls(
        self, ctx: Any, state: Any, review_mode: Any, enabled: bool
    ) -> None:
        """Accept / Reject / Finish, for a pass that is running.

        The keys are named on the buttons, which is the whole of what licenses
        binding ``A`` again: the objection was never to the key, it was to a
        *silent* remap onto a grade the reviewer had not chosen. A button
        labelled "Accept (A)" in a mode entered on purpose says exactly what it
        files.
        """
        from imgui_bundle import imgui

        from ..vectors import BINARY_GRADES
        from . import controls, widgets

        reason = "A scan is running; the queue is being rebuilt."
        if widgets.disabled_button("Accept (A)", enabled, reason=reason):
            review_mode.record(ctx, BINARY_GRADES["accept"], state.pending_tags)
        imgui.same_line()
        if widgets.disabled_button("Reject (R)", enabled, reason=reason):
            review_mode.record(ctx, BINARY_GRADES["reject"], state.pending_tags)
        imgui.same_line()
        if controls.button("Finish", role=controls.ButtonRole.GHOST):
            review_mode.end_judging(ctx)
        widgets.hint_text(
            f"Files {BINARY_GRADES['accept']:+d} or {BINARY_GRADES['reject']:+d}. "
            "Use the grades below to say more; Esc ends the pass."
        )

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

        from . import forms, widgets

        unit = review_mode.current(state)
        if unit is None:
            widgets.muted("Pick a sweep on the left.")
            self._review_findings(ctx)
            return

        if state.judging is not None:
            # The pass's own position, above the unit's. It counts *filed*
            # against the total outstanding when the pass started, which is a
            # different question from "where in this sweep am I" -- and the one
            # a reviewer who has agreed to do twenty of these is asking.
            widgets.section(
                f"Judging {state.judging.filed + 1} of {state.judging.total}"
            )
            widgets.muted(review_mode.label(state, unit))
        else:
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

        if state.judging is not None:
            # Above the grade row, not instead of it. The binary pair is the
            # fast path; the eleven-point scale below is the power path and
            # keeps working, files a grade and advances the pass exactly as
            # these two do.
            self._review_judging_controls(ctx, state, review_mode, enabled)

        with forms.Form("review-verdict") as form_ui:
            with form_ui.field(
                "grade",
                "Grade",
                help_text="A digit grades; press R first for a negative grade.",
                helper="+5 ships as-is, +3 is usable, and -5 is unusable.",
            ):
                grade = widgets.grade_buttons("review", enabled)
            if grade is not None:
                review_mode.record(ctx, grade, state.pending_tags)

            with form_ui.field(
                "tags", "Tags", helper="Optional; S skips without filing a grade."
            ):
                tag = widgets.tag_toggles("review", state.pending_tags, enabled)
            if tag is not None:
                review_mode.toggle_tag(state, tag)

            if widgets.disabled_button(
                "Skip (S)",
                enabled,
                reason="A scan is running; the queue is being rebuilt.",
            ):
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
        from . import controls, review_mode, widgets

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
            summary = review_mode.describe_vector(entry["vector"])
            imgui.text_wrapped(f"{findings_lib.vector_line(entry)}  -  {summary}")
            measured = findings_lib.metrics_line(entry.get("metrics"))
            if measured:
                widgets.muted(measured)
            tagged = findings_lib.tag_line(entry)
            if tagged:
                widgets.muted(tagged)
            vector = entry["vector"]
            if controls.button(f"Apply to forms##apply-{entry['key']}"):
                review_mode.apply_vector(ctx.state, vector)
                ctx.toast("Applied those settings to the 2D and 3D forms.")
            imgui.separator()

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

        layout_edit.draw(self, ctx, viewport)
        overlay.fps_meter(ctx, self.fps)
        if ctx.state.mode != "home":
            overlay.progress_card(ctx, self.eta)
        widgets.toasts(
            ctx.state,
            (viewport.work_size.x, viewport.work_size.y),
            on_action=self._toast_action,
        )
        # The first-run question owns the screen before any workflow modal.
        # Its two exits close it permanently, then later questions can use the
        # one popup slot on the following frame.
        first_run.draw(ctx)
        if first_run.is_open(ctx):
            self._transition_overlay(viewport)
            return
        # Before the confirms, because it is the same kind of thing and the
        # earlier one wins the single modal slot imgui gives a frame.
        settings_3d.matte_modal(ctx)
        # The Manual, over whatever ran above (the UI redesign, wave 3). Before the
        # palette on purpose: Ctrl+K is how you leave anywhere, this included,
        # so it has to float above the reference rather than under it.
        from .manual import render as manual_render
        from .panes import profiles_panel

        # Under the Manual, because the (?) inside the manager opens the
        # manual *about* it: the reference has to land on top of the sheet it
        # was asked from, not behind it.
        profiles_panel.draw_sheet(ctx)
        manual_render.draw_overlay(ctx)
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
            ctx.state.library_scroll_to = (
                asset_open.route(job).job_id if job is not None else arg
            )
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

    # Every binding the app answers to, in one place the user can find. The
    # tuples are (keys, what), grouped; Inker's letters come from TOOL_KEYS so
    # this list cannot drift from the handler.
    def _shortcuts_popup(self) -> None:
        from imgui_bundle import imgui

        from . import icons, widgets
        from .tokens import sp

        viewport = imgui.get_main_viewport()
        popup_width = min(sp(520), viewport.work_size.x - sp(32))
        popup_height = min(sp(720), viewport.work_size.y - sp(64))
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
        """The rail footer's layout switcher (P5.3).

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
            if controls.menu_item(
                f"{label}##layout/{name}",
                "",
                selected,
                layout.readable,
                reason=(
                    "This layout was saved by a newer build. It is kept exactly "
                    "as it was found rather than reinterpreted."
                ),
            )[0] and layout.readable:
                self.layouts.set_active(name)
        imgui.separator()
        if controls.menu_item_simple("Reset this layout"):
            self.layouts.reset()
        if controls.menu_item_simple("Manage layouts..."):
            # Settings, rather than a second administration surface here: one
            # place that can rename and delete is one place to look for the
            # thing you deleted.
            from .state import set_mode

            set_mode(ctx, "settings")
        imgui.end_popup()

    def _diagnostics_popup(self, checks: list[Any]) -> None:
        from imgui_bundle import imgui

        from . import controls, icons, theme, widgets
        from .tokens import sp

        ctx = self.app_ctx
        viewport = imgui.get_main_viewport()
        popup_width = min(sp(480), viewport.work_size.x - sp(32))
        popup_height = min(sp(720), viewport.work_size.y - sp(64))
        imgui.set_next_window_pos(
            (
                viewport.work_pos.x + viewport.work_size.x - sp(16),
                viewport.work_pos.y + sp(48),
            ),
            imgui.Cond_.appearing.value,
            (1.0, 0.0),
        )
        imgui.set_next_window_size((popup_width, popup_height))
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
        widgets.pane_header(
            "Issues",
            actions=(("close", f"{icons.X} Close", imgui.close_current_popup),),
        )
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
        if controls.button("Copy details", role=controls.ButtonRole.GHOST):
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
        if controls.button("Run checks again", role=controls.ButtonRole.GHOST):
            from ..service import system as svc_system

            ctx.submit("health", svc_system.current_checks, ctx.svc, force=True)
        imgui.same_line()
        if widgets.disabled_button(
            "Open the log",
            _log_exists(ctx.runtime.config.data_dir),
            # The log is written on the first line logged, so its absence is
            # "nothing has gone wrong yet" rather than a fault to report.
            reason="There is no log file yet: nothing has been logged this install.",
        ):
            ctx.open_log()
        imgui.same_line()
        # Chapter 12 (F57). The popup names the failing rows and their remedies;
        # what it cannot hold is what to do when a remedy does not take.
        from .manual import render as manual_render

        if manual_render.troubleshooting_button(ctx):
            imgui.close_current_popup()
        from . import component_gallery

        if component_gallery.enabled():
            imgui.same_line()
            if controls.button(
                "Component gallery", role=controls.ButtonRole.GHOST
            ):
                component_gallery.request()
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

        from . import controls, theme, widgets
        from .tokens import sp

        log = ctx.state.toast_log
        if not log:
            return
        imgui.separator()
        if not controls.collapsing_header(
            f"Notifications ({len(log)})##toast-history"
        ):
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
        if controls.small_button(
            "Copy notifications", role=controls.ButtonRole.GHOST
        ):
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

        from . import controls
        from .panes import app_settings

        if not controls.collapsing_header("Effective configuration"):
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

        from . import create_stages
        from . import layout as layout_mod
        from .panes import overlay

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
                image_pos = imgui.get_cursor_screen_pos()
                avail = imgui.get_content_region_avail()
                height = max(avail.y, 64)
                reference_stage = create_stages.at(ctx.state, "reference")
                if not reference_stage and self.viewer.has_model:
                    self._draw_viewport_image(image_pos, width, height)
                elif reference_stage and self.viewer.reference is not None:
                    self._draw_reference(width, height)
                else:
                    overlay.placeholder(ctx)

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
    _setup_logging()
    _install_excepthooks()
    log.info(
        "Warlock Studio %s starting: pid=%d python=%s argv=%s",
        _version(), os.getpid(), sys.version.split()[0], sys.argv[1:],
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
    config = get_config()
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


def _run_locked() -> int:
    """Everything after the single-instance lock is held."""
    from .. import migrate
    from ..config import get_config
    from .runtime import Runtime

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
