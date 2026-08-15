"""Multi-document state for Paint mode, without imgui.

The same split ``state.py`` makes, for the same reason: which document is open,
which is dirty, where the view is and what the tool settings are would all still
make sense if the app were driven by a script, so none of it needs a window to
be tested.

Two conventions worth stating, both borrowed from Aseprite because they are
what a user coming from a paint program expects. **Tool settings belong to the
app, not to the document** -- switching tabs must not silently change your brush
size. And **the view belongs to the document** -- a tab remembers where it was
panned to, because scrolling back to where you were working is not something
the user should have to redo per tab switch.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import docmodes

MIN_ZOOM = 0.05
MAX_ZOOM = 32.0
ZOOM_STEP = 1.15

# The swatch row's own capacity. Not a palette editor -- the eyedropper is how
# a user gets the colours actually in their image; this only has to hold the
# handful they keep coming back to.
MAX_SWATCHES = 24

TOOLS = (
    ("brush", "Brush", "B"),
    ("eraser", "Eraser", "E"),
    ("fill", "Fill", "G"),
    ("gradient", "Gradient", "U"),
    ("blur", "Blur", "R"),
    ("smudge", "Smudge", "N"),
    ("line", "Line", "P"),
    ("rect", "Rect", "K"),
    ("ellipse", "Ellipse", "J"),
    ("select", "Marquee", "M"),
    ("select_ellipse", "Ellipse select", "S"),
    ("lasso", "Lasso", "Q"),
    ("wand", "Wand", "W"),
    ("move", "Move", "V"),
    ("eyedropper", "Pick", "I"),
    ("slice", "Slice", "C"),
)

# Tools whose drag paints into the layer.
PAINT_TOOLS = frozenset({"brush", "eraser", "blur", "smudge"})
SHAPE_TOOLS = frozenset({"line", "rect", "ellipse"})
SELECT_TOOLS = frozenset({"select", "select_ellipse", "lasso", "wand"})

# What each tool asks the brush engine for. Kept here rather than branched at
# the call site so a new mode is one row.
BRUSH_MODES = {"brush": "paint", "eraser": "erase", "blur": "blur", "smudge": "smudge"}

# The options a tool remembers *for itself*, and what a tool that has never
# been touched starts from.
#
# Every one of these used to be a single app-level value, which is wrong in the
# way that is hardest to notice: sizing the eraser to 60 to clean up a corner
# and going back to the brush found the brush at 60 too, so the user re-set it
# every time they switched -- and blamed themselves. Aseprite, Photoshop and
# Krita all key these on the tool, and a user arrives expecting that.
#
# What is *not* here is as deliberate. Symmetry, the grid, the foreground and
# background colours and the onion-skin settings stay app-level: they are
# properties of the canvas or of the session rather than of a tool, and a grid
# that switched off because you picked the eraser would be a bug.
TOOL_OPTION_DEFAULTS: dict[str, Any] = {
    "brush_size": 12,
    "hardness": 0.85,
    "opacity": 1.0,
    "spacing": 0.1,
    "strength": 0.5,
    "shape_filled": False,
    "wand_tolerance": 32,
    "wand_contiguous": True,
    "sample_layer": False,
    "stabilise": 0.0,
    "speed_taper": 0.0,
    # Per tool for the same reason every other brush setting is: a pixel nib is
    # a property of the tool in your hand, and an eraser that stayed soft after
    # the pencil was made hard is exactly the mismatch this table exists to
    # stop -- a one-pixel line rubbed out with a feathered eraser leaves a
    # fringe of half-alpha nobody asked for.
    "nib": "soft",
    "pixel_perfect": False,
}

DEFAULT_SWATCHES: tuple[tuple[int, int, int, int], ...] = (
    (0, 0, 0, 255),
    (255, 255, 255, 255),
    (128, 128, 128, 255),
    (200, 40, 40, 255),
    (230, 140, 40, 255),
    (230, 210, 60, 255),
    (60, 170, 80, 255),
    (60, 130, 220, 255),
    (140, 70, 200, 255),
    (110, 70, 40, 255),
)

_uids = itertools.count(1)


# --- the view ---------------------------------------------------------------


#: The quarter turns the view offers, in degrees. **Quarter turns only, and
#: that is a decision rather than a first instalment.** A free-angle canvas
#: rotation makes every overlay in the pane a rotated quantity: the grid stops
#: being two families of axis-aligned lines, the marquee preview stops being a
#: rect, the transform box's handles stop being squares, and each of those has
#: to be re-derived and re-tested. A quarter turn maps an axis-aligned image
#: rectangle onto an axis-aligned *screen* rectangle, so every one of those
#: stays exactly what it was -- and it delivers what canvas rotation is
#: actually reached for: turning the page to draw a curve, and checking a
#: drawing mirrored. The engine never sees any of it; pixels are untouched.
ROTATIONS = (0, 90, 180, 270)


@dataclass
class PaintView:
    """Where the canvas sits in its pane. Per document, so a tab switch does
    not lose your place."""

    zoom: float = 1.0
    pan: tuple[float, float] = (0.0, 0.0)
    # Whether the view has been framed yet. False asks the canvas to fit on the
    # next frame it draws, which is the only moment it knows how big the pane
    # is -- the state layer never does.
    fitted: bool = False
    # A zoom to snap to on the next frame, for the same reason: "100%, centred"
    # needs the pane's size, and a keypress does not have it.
    pending_zoom: float | None = None
    # Display only, both of them: see ROTATIONS. ``rotation`` is clockwise on
    # screen in degrees; ``flipped`` mirrors left-to-right *after* it, which is
    # the order a physical sheet of paper does the two in.
    rotation: int = 0
    flipped: bool = False


def clamp_zoom(zoom: float) -> float:
    return max(MIN_ZOOM, min(MAX_ZOOM, float(zoom)))


def _quarter(view: PaintView) -> int:
    """ROTATIONS' index for the view's rotation, in one spelling.

    A rotation somehow off the quarter lattice reads as 0 -- the answer
    ``basis`` has always given -- rather than raising out of ``index()``.
    Only code can produce one today, which is exactly why the guard lives
    here: ``rotate_view`` restated the lookup without it, so the two answered
    the same bad value differently, one silently and one with a ValueError.
    """
    rotation = int(view.rotation)
    return ROTATIONS.index(rotation % 360) if rotation % 90 == 0 else 0


def basis(view: PaintView) -> tuple[tuple[float, float], tuple[float, float]]:
    """The view's 2x2 orientation, as rows. Orthonormal, determinant +-1.

    Kept separate from the zoom because it is exactly the part that preserves
    *length*: the marching ants measure arc length in canvas space and dash
    along it, so a transform that scaled would have to be threaded through that
    arithmetic, and one that only turns does not.
    """
    quarter = _quarter(view)
    # (x, y) -> (-y, x) is one clockwise quarter turn on a screen whose y grows
    # downward, which is the direction the button's icon points.
    rows = (
        ((1.0, 0.0), (0.0, 1.0)),
        ((0.0, -1.0), (1.0, 0.0)),
        ((-1.0, 0.0), (0.0, -1.0)),
        ((0.0, 1.0), (-1.0, 0.0)),
    )[quarter]
    if view.flipped:
        # After the turn, and on screen x: mirroring in image space instead
        # would put the flip under the rotation and make "flip" mean two
        # different things depending on which way the page was turned.
        rows = ((-rows[0][0], -rows[0][1]), rows[1])
    return rows


def _oriented(view: PaintView, x: float, y: float) -> tuple[float, float]:
    (a, b), (c, d) = basis(view)
    return (a * x + b * y, c * x + d * y)


def view_extent(
    view: PaintView, size: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The canvas's oriented box at zoom 1, as ``(low, high)``.

    A quarter turn puts part of the canvas at negative coordinates, so the
    framing functions cannot assume the corner is at the origin any more --
    which is the whole of what rotation costs the layout, and it is contained
    here.
    """
    width, height = float(size[0]), float(size[1])
    corners = [
        _oriented(view, x, y) for x in (0.0, width) for y in (0.0, height)
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _place(
    view: PaintView, size: tuple[int, int], region: tuple[float, float], zoom: float
) -> None:
    """Set the zoom and centre the oriented canvas in the region."""
    (lo_x, lo_y), (hi_x, hi_y) = view_extent(view, size)
    view.zoom = clamp_zoom(zoom)
    view.pan = (
        (region[0] - (hi_x - lo_x) * view.zoom) * 0.5 - lo_x * view.zoom,
        (region[1] - (hi_y - lo_y) * view.zoom) * 0.5 - lo_y * view.zoom,
    )
    view.fitted = True


def fit(view: PaintView, size: tuple[int, int], region: tuple[float, float]) -> None:
    """Scale to show the whole document, centred."""
    (lo_x, lo_y), (hi_x, hi_y) = view_extent(view, size)
    zoom = min(region[0] / max(hi_x - lo_x, 1.0), region[1] / max(hi_y - lo_y, 1.0))
    _place(view, size, region, zoom)


def centre(
    view: PaintView, size: tuple[int, int], region: tuple[float, float], zoom: float
) -> None:
    """Set an explicit zoom and re-centre -- what Ctrl+1 (100%) does."""
    _place(view, size, region, zoom)


def rotate_view(view: PaintView, quarter_turns: int = 1) -> None:
    """Turn the page. The zoom is kept and the canvas re-centred next frame.

    Re-centred rather than left where it was, because a quarter turn about the
    view's origin sends the canvas off the pane -- and through ``pending_zoom``
    rather than by clearing ``fitted``, which would also re-scale and throw away
    a zoom the user chose.
    """
    view.rotation = ROTATIONS[(_quarter(view) + int(quarter_turns)) % 4]
    view.pending_zoom = view.zoom


def flip_view(view: PaintView) -> None:
    """Mirror the view left-to-right. The classic check on a drawing, and the
    reason this is a *view* flag rather than an edit: nothing about the document
    changes, so there is nothing to undo and nothing to save."""
    view.flipped = not view.flipped
    view.pending_zoom = view.zoom


def to_image(view: PaintView, origin: tuple[float, float], sx: float, sy: float):
    """Screen -> image coordinates, as floats.

    Floats, not ints: the brush walks sub-pixel positions, and rounding here
    would quantise every stroke to the zoom level it was drawn at.

    The orientation is inverted by **transposing** its matrix, which is exact
    rather than approximate: the basis is orthonormal, so its transpose is its
    inverse whichever of the eight it happens to be.
    """
    u = (sx - origin[0] - view.pan[0]) / view.zoom
    v = (sy - origin[1] - view.pan[1]) / view.zoom
    (a, b), (c, d) = basis(view)
    return (a * u + c * v, b * u + d * v)


def to_screen(view: PaintView, origin: tuple[float, float], x: float, y: float):
    u, v = _oriented(view, x, y)
    return (
        origin[0] + view.pan[0] + u * view.zoom,
        origin[1] + view.pan[1] + v * view.zoom,
    )


def zoom_about(
    view: PaintView, origin: tuple[float, float], mouse: tuple[float, float], steps: float
) -> None:
    """Zoom keeping whatever pixel is under the cursor under the cursor."""
    before = view.zoom
    after = clamp_zoom(before * (ZOOM_STEP**steps))
    if after == before:
        return
    local = (mouse[0] - origin[0], mouse[1] - origin[1])
    ratio = after / before
    view.zoom = after
    view.pan = (
        local[0] - (local[0] - view.pan[0]) * ratio,
        local[1] - (local[1] - view.pan[1]) * ratio,
    )


# --- one open document ------------------------------------------------------


@dataclass
class InkerDoc:
    """One tab.

    ``uid`` is stable and never reused, because imgui identifies a tab by its
    label: a title alone would make two files called "input.png" the same tab,
    and would move a tab's identity every time it was renamed by a Save As.
    """

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    file_format: str = "png"  # ora | png
    uid: str = field(default_factory=lambda: f"pd{next(_uids)}")
    view: PaintView = field(default_factory=PaintView)
    # The history position the file on disk was written from. Dirty is a
    # *comparison*, not a flag, so undoing back to the saved state correctly
    # stops being dirty -- which the document's revision cannot express,
    # because it counts changes and an undo is one.
    saved_head: int = 0
    saving: bool = False
    # The job this document writes back into, if any. ``link_kind`` says what
    # kind of write that is; empty means the document is a plain file.
    job_id: str = ""
    link_kind: str = ""  # "" | "reference-edit"
    has_original: bool = False

    # Playback, and it is deliberately transient per tab rather than on the
    # document: the playhead is view state and this is view state *about* the
    # playhead, so neither belongs in a file. ``play_accum_ms`` is the leftover
    # time carried between ticks, because a frame's duration is per-frame and a
    # clip is therefore not a rate.
    playing: bool = False
    play_index: int = 0
    play_accum_ms: float = 0.0
    # Which leg of a ping-pong the playhead is on. Only a ping-pong tag reads
    # it, and it lives here rather than in ``advance`` because that function is
    # pure and the leg has to survive between ticks.
    play_forward: bool = True

    # Crash-safety, owned by :mod:`studio.journal` (UX-05). ``journal_name`` is
    # the file this tab owns under the autosave directory and is minted once,
    # on the first copy: naming it eagerly would litter the directory with
    # entries for tabs nobody ever edited. ``journal_head`` is the history
    # position the last one captured, so an idle document is not rewritten
    # every two minutes -- the same comparison ``dirty`` is, against a
    # different mark. ``journal_at`` is the debounce.
    #
    # Named for the journal rather than for Inker, because they are the three
    # fields *every* document mode now carries: the mechanism Inker proved was
    # right and the whole of what was wrong was that it lived in one mode.
    journal_name: str = ""
    journal_head: int | None = None
    journal_at: float = 0.0

    @property
    def busy(self) -> bool:
        """Whether the document may be edited right now.

        One question with two answers behind it -- a save is encoding the layer
        stack off-thread, or playback is running and every control that would
        change the document is refused. Callers ask this rather than ``saving``
        so a third reason can never be added in one place and forgotten in nine.
        """
        return self.saving or self.playing

    @property
    def dirty(self) -> bool:
        return self.doc.history.head != self.saved_head

    @property
    def frame_uid(self) -> int | None:
        """The frame the playhead is on, or None on a still document.

        Here rather than in either pane because two of them ask -- the canvas
        resolves a slice's per-frame key to draw it and the tools panel resolves
        the same one to describe it -- and two spellings of "which frame is the
        user looking at" is one of them being wrong during playback.
        """
        anim = self.doc.anim
        return None if anim is None or not anim.frames else anim.frame.uid

    @property
    def linked(self) -> bool:
        return bool(self.job_id)

    @property
    def label(self) -> str:
        """What imgui draws on the tab. The id after ### is what it *matches*
        on, so the visible part is free to change without moving the tab."""
        return f"{self.title}###{self.uid}"

    def mark_saved(self, head: int | None = None) -> None:
        """Record which history position is now on disk.

        Captured when the *encode* starts, not when it finishes: an edit made
        while the file was being written is genuinely not in it, and clearing a
        flag here would call it saved.
        """
        self.saved_head = self.doc.history.head if head is None else head
        self.saving = False


# The same answer in three of the four modes; Clay's is on ``stem`` on purpose.
title_for = docmodes.title_for


def _tool_option(name: str) -> property:
    """``state.brush_size`` reading and writing the *active tool's* copy.

    A property rather than a rewrite of every call site, and that is the point:
    the panes, the canvas and the keyboard already say ``state.brush_size``, and
    each of them meant "the size of the tool in my hand" all along -- the state
    layer was the thing that disagreed. Nine call sites become per-tool with no
    edit and no chance of one being missed.

    An unannotated class attribute is not a dataclass field, so these coexist
    with ``@dataclass`` rather than fighting it; the defaults live in
    ``TOOL_OPTION_DEFAULTS`` because that is where a *tool's* defaults belong.
    """

    def get(self: InkerState) -> Any:
        return self.options_for(self.tool)[name]

    def put(self: InkerState, value: Any) -> None:
        self.options_for(self.tool)[name] = value

    return property(get, put, doc=f"{name}, remembered per tool.")


# --- everything Paint mode remembers ----------------------------------------


@dataclass
class InkerState:
    docs: list[InkerDoc] = field(default_factory=list)
    active_uid: str = ""
    # The in-flight sheet/GIF export's frame-by-frame read of a document, or
    # None. ``inker_mode._Export``, typed loosely here because this module is
    # the state and that one is the behaviour. One at a time by construction:
    # both exports share a task key and the tab is locked for the duration, so
    # a second click while one is stepping is refused rather than queued.
    export: Any = None

    # Tool settings: shared across documents on purpose.
    tool: str = "brush"
    # Per tool, keyed by tool name; see TOOL_OPTION_DEFAULTS. Populated lazily,
    # so a fresh session carries nothing and a tool that has never been adjusted
    # has no entry rather than a copy of the defaults.
    tool_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    feather_radius: float = 2.0
    # Grow / shrink / border, in whole pixels. App-level like every other
    # tool setting, and separate from ``feather_radius`` because the two have
    # different units and mean different things to an edge.
    select_steps: int = 2
    gradient_kind: str = "linear"
    gradient_to_transparent: bool = False
    # Extra colour stops, or empty for the foreground-to-background preset.
    # Empty rather than a materialised two-stop list on purpose: the preset has
    # to *follow* the two colours, so that swapping them with X changes the next
    # gradient the way it always has.
    gradient_stops: list[tuple[float, tuple[int, int, int, int]]] = field(
        default_factory=list
    )
    symmetry: str = "none"
    # Where the mirrors sit, in image coordinates, or None for the canvas
    # centre. None rather than a materialised centre: a document opened at a
    # different size must not inherit the last one's axis, and "the centre" is
    # the only answer that stays true across a resize.
    symmetry_axis: tuple[float, float] | None = None
    radial_count: int = 6
    # How a scale, and the free transform's own scale and rotate, decide what a
    # destination pixel holds; see ``transform.RESAMPLES``. App-level rather
    # than per tool or per document: it is a statement about the kind of art
    # being made, which does not change when the eraser is picked up and is the
    # same answer for every document open in a pixel-art session.
    resample: str = "smooth"
    grid: bool = False
    grid_size: int = 16
    # Whether a shape, a marquee or a line snaps to the grid. Deliberately not
    # applied to freehand strokes: quantising a brush to a 16-pixel lattice is
    # not a drawing aid, it is a different tool.
    grid_snap: bool = False

    # Onion skinning: app-level, like every other tool setting, because it is a
    # property of how the user works rather than of the drawing. Tinted red
    # behind and green ahead, which is the convention every 2D animation tool
    # has used for thirty years -- picking differently would be a novelty the
    # user has to learn for nothing.
    # Which tag the timeline is renaming, and the text being typed. Pure view
    # state -- not persisted, pushes no undo step -- for the same reason the
    # playhead is: a document must not ask to be saved because a name is being
    # typed. -1 is "nothing being renamed"; the buffer is only meaningful with
    # it set.
    tag_editing: int = -1
    tag_name: str = ""

    onion: bool = False
    onion_before: int = 1
    onion_after: int = 1
    onion_alpha: float = 0.35
    fg: tuple[int, int, int, int] = (0, 0, 0, 255)
    bg: tuple[int, int, int, int] = (255, 255, 255, 255)
    swatches: list[tuple[int, int, int, int]] = field(
        default_factory=lambda: list(DEFAULT_SWATCHES)
    )

    # -- slices, all of it view state --------------------------------------
    #
    # The slices themselves live on the *document* (``Document.slices``): they
    # are saved with the file and read by an export. What is here is which one
    # the user has selected and whether the overlay is on, neither of which is
    # picture data and neither of which may push an undo step.
    #
    # ``slice_uid`` is 0 for "none", and a stale one is tolerated rather than
    # policed: an undone add leaves the selection naming a slice that is not in
    # the document, every reader of it is a ``slice_by_uid`` that answers None,
    # and hunting the value down on every history move would be a second place
    # for the selection to be wrong.
    slice_uid: int = 0
    #: Whether the overlay draws while another tool is in hand. The slice tool
    #: forces it on -- see ``inker_canvas.slices_visible`` -- so this is only
    #: ever the answer to "keep showing them while I paint".
    show_slices: bool = False
    #: ``(properties at the press, which handle)`` for the drag in flight.
    #: The properties are what ``set_slice(was=...)`` records as the "before":
    #: the drag mutates the live slice every frame so the overlay follows the
    #: cursor, and reading the before at *release* would undo the gesture to
    #: itself.
    slice_drag: Any = None

    # -- indexed colour, all of it view state ------------------------------
    #
    # The palette itself lives on the *document* (``Document.palette``), which
    # is the only place it can live: it is saved with the file and it decides
    # what every write snaps to. What is here is which slot the user has
    # selected and the last usage count they asked for -- neither is picture
    # data, and neither may push an undo step.
    palette_slot: int = 0
    # ``(document rev the count was taken at, per-slot counts)``. Asked for
    # rather than recomputed: counting is a walk over every pixel of every cel,
    # so doing it per frame would cost a 40-frame clip's worth of scanning
    # sixty times a second to keep a number that changes on one dab.
    palette_usage: tuple[int, list[int]] | None = None

    # Drag state, decided on press because several tools start the same way.
    drag_kind: str = ""  # "" | paint | shape | marquee | lasso | move | gradient | pan
    drag_anchor: tuple[float, float] | None = None
    last_point: tuple[float, float] | None = None
    lasso: list[tuple[float, float]] = field(default_factory=list)
    combine: str = "replace"
    space_held: bool = False

    # The open filter session: which filter, the values every filter was last
    # run with, and whether the popup is up. Remembered per filter for the
    # reason Clay's op parameters are -- somebody applying the same levels to
    # six layers should not retype it six times -- and ``filter_open`` is what
    # notices imgui closing the popup on a click outside, which is a cancel.
    filter_name: str = ""
    # ``Any`` and not ``float``: the FX staples brought colours (an RGBA tuple)
    # and a choice (a string) into the same per-filter bag.
    filter_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    filter_open: bool = False

    # Free transform is a *state*, not a tool: it takes over the canvas until
    # it is committed or cancelled, and every other tool is unavailable while
    # it is on -- which is exactly what "modal" means and why it cannot live in
    # the tool list beside brush and fill.
    transforming: bool = False
    # What the handle was grabbed at, so a drag is measured against the press
    # rather than against the previous frame.
    transform_ref: tuple[float, float, float, float] | None = None

    # -- per-tool options ---------------------------------------------------
    #
    # Written out one per line rather than looped over TOOL_OPTION_DEFAULTS: a
    # generated attribute is invisible to a reader and to a type checker, and
    # the whole point of these is that they are the names the rest of the app
    # already says.

    brush_size = _tool_option("brush_size")
    hardness = _tool_option("hardness")
    opacity = _tool_option("opacity")
    strength = _tool_option("strength")
    spacing = _tool_option("spacing")
    shape_filled = _tool_option("shape_filled")
    wand_tolerance = _tool_option("wand_tolerance")
    wand_contiguous = _tool_option("wand_contiguous")
    sample_layer = _tool_option("sample_layer")
    stabilise = _tool_option("stabilise")
    speed_taper = _tool_option("speed_taper")
    nib = _tool_option("nib")
    pixel_perfect = _tool_option("pixel_perfect")

    def options_for(self, tool: str) -> dict[str, Any]:
        """One tool's option dictionary, created at the defaults on first ask.

        A tool that has never been adjusted has no entry at all, which is what
        makes "reset this tool" a ``pop`` and what keeps a saved session (if one
        is ever saved) to the settings the user actually changed.
        """
        got = self.tool_options.get(tool)
        if got is None:
            got = dict(TOOL_OPTION_DEFAULTS)
            self.tool_options[tool] = got
        return got

    def reset_tool_options(self, tool: str | None = None) -> None:
        self.tool_options.pop(tool or self.tool, None)

    # -- documents ---------------------------------------------------------

    @property
    def active(self) -> InkerDoc | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, doc: InkerDoc) -> InkerDoc:
        self.docs.append(doc)
        self.active_uid = doc.uid
        self.clear_drag()
        return doc

    def get(self, uid: str) -> InkerDoc | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None

    def close(self, uid: str) -> bool:
        doc = self.get(uid)
        if doc is None:
            return False
        index = self.docs.index(doc)
        self.docs.remove(doc)
        if self.active_uid == uid:
            # The neighbour, not the first: closing a tab should leave you next
            # to where you were rather than at the far end of the bar.
            self.active_uid = self.docs[min(index, len(self.docs) - 1)].uid if self.docs else ""
        self.clear_drag()
        return True

    def activate(self, uid: str) -> None:
        if uid != self.active_uid:
            self.active_uid = uid
            self.clear_drag()

    def cycle(self, step: int = 1) -> None:
        if len(self.docs) < 2:
            return
        current = self.active
        index = self.docs.index(current) if current in self.docs else 0
        self.activate(self.docs[(index + step) % len(self.docs)].uid)

    def find_path(self, path: Path) -> InkerDoc | None:
        """An already-open tab for this file, so opening twice focuses rather
        than forking -- two tabs over one path would race on save."""
        for doc in self.docs:
            if doc.path is not None and doc.path == path:
                return doc
        return None

    def find_job(self, job_id: str) -> InkerDoc | None:
        for doc in self.docs:
            if doc.job_id == job_id:
                return doc
        return None

    # -- drag ---------------------------------------------------------------

    def clear_drag(self) -> None:
        self.drag_kind = ""
        self.drag_anchor = None
        self.last_point = None
        self.lasso = []
        self.transform_ref = None
        self.slice_drag = None

    # -- colours ------------------------------------------------------------

    def swap_colours(self) -> None:
        self.fg, self.bg = self.bg, self.fg

    def add_swatch(self, colour: tuple[int, int, int, int]) -> None:
        colour = tuple(int(c) for c in colour)  # type: ignore[assignment]
        if colour in self.swatches:
            return
        self.swatches.append(colour)
        del self.swatches[:-MAX_SWATCHES]



# --- shape drag constraints -------------------------------------------------
#
# The two modifiers every drawing program binds, and they are deliberately
# scoped to the *shape* tools alone. Shift and Alt already mean add and subtract
# on the four selection tools, sampled at press into ``combine``; giving them a
# second meaning on the same drag would make one gesture ambiguous, and which
# reading won would depend on the order the branches happen to be written in.
# So the marquee keeps its combining modifiers and the line, rectangle and
# ellipse get these.


def constrain_line(
    anchor: tuple[float, float], point: tuple[float, float]
) -> tuple[float, float]:
    """The cursor snapped to the nearest eighth of a turn from the anchor.

    Length is preserved rather than projected, so the far end tracks the cursor
    at the same distance it is actually at -- a projection makes a line shrink
    to nothing as the cursor approaches the perpendicular, which reads as the
    constraint fighting the drag.
    """
    dx, dy = point[0] - anchor[0], point[1] - anchor[1]
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return point
    step = math.pi / 4.0
    angle = round(math.atan2(dy, dx) / step) * step
    return (anchor[0] + math.cos(angle) * length, anchor[1] + math.sin(angle) * length)


def constrain_square(
    anchor: tuple[float, float], point: tuple[float, float]
) -> tuple[float, float]:
    """The cursor with both sides made equal, keeping the quadrant it is in.

    The larger side wins, so the shape always covers what the cursor has reached
    on its dominant axis; taking the smaller one makes a drag feel like it is
    being pulled back.
    """
    dx, dy = point[0] - anchor[0], point[1] - anchor[1]
    size = max(abs(dx), abs(dy))
    return (
        anchor[0] + (size if dx >= 0 else -size),
        anchor[1] + (size if dy >= 0 else -size),
    )


def shape_endpoints(
    tool: str,
    anchor: tuple[float, float],
    point: tuple[float, float],
    *,
    constrain: bool = False,
    from_centre: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two points a shape drag describes, after the modifiers.

    One function for the preview and for the release, which is the whole reason
    it is here rather than inlined at either: the two used to be the same
    expression by coincidence, and a constraint applied to one of them would
    draw a square and commit a rectangle.

    Constraining happens *before* the centre expansion, or the two fight: an
    equal-sided box mirrored about the anchor is still equal-sided, while
    mirroring first and squaring afterwards moves the centre off the point the
    user pressed on -- which is the one thing "from centre" promises.
    """
    if constrain:
        point = constrain_line(anchor, point) if tool == "line" else constrain_square(
            anchor, point
        )
    if from_centre:
        # A line has a start and an end rather than a box, so "from centre"
        # means the anchor is the middle of it, which is the same reflection.
        anchor = (2.0 * anchor[0] - point[0], 2.0 * anchor[1] - point[1])
    return anchor, point


# --- brush size stepping ----------------------------------------------------


def step_size(size: int, delta: int) -> int:
    """``[`` and ``]``, accelerating.

    A flat +/-1 makes going from 8 to 200 a chore; scaling the step with the
    size keeps every press feel proportional, which is what every paint program
    does.
    """
    from .inker import clamp_brush

    step = max(1, int(abs(size) * 0.12))
    return clamp_brush(size + (step if delta > 0 else -step))
