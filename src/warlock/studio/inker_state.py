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

# Tool presets: how many are kept, and how long a name may be. Both are the
# swatch row's reasoning -- this is a shelf for the handful of setups a user
# keeps coming back to, not a library -- and the name cap is there because the
# name is drawn in a 104px sidebar and stored in a settings file.
MAX_PRESETS = 32
MAX_PRESET_NAME = 40

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
    ("spray", "Spray", "A"),
    ("lasso_poly", "Poly lasso", "D"),
    ("text", "Text", "T"),
    # ``H`` for shading, because Aseprite has no letter to borrow here -- its
    # shading is an *ink* on the ordinary brush rather than a tool -- and every
    # letter in the word that reads as a mnemonic was taken (S is the ellipse
    # marquee, A the spray, D and T are spoken for elsewhere).
    ("shade", "Shading", "H"),
    # The clicked shapes (Q-c). Their letters are what was left: of "polygon"
    # only ``O`` is free (P is the line, G the fill, N the smudge), so the
    # polygon takes it and the polyline takes ``L`` one word along. Not one
    # letter of "curve" is free -- C is the slice, U the gradient, R the blur, V
    # the move, E the eraser -- so the curve takes ``F``, for the *free-form*
    # curve it draws.
    ("polyline", "Polyline", "L"),
    ("polygon", "Polygon", "O"),
    ("curve", "Curve", "F"),
)

# Tools whose drag paints into the layer.
PAINT_TOOLS = frozenset({"brush", "eraser", "blur", "smudge", "spray", "shade"})
SHAPE_TOOLS = frozenset({"line", "rect", "ellipse", "polyline", "polygon", "curve"})

#: The shape tools whose points are **clicked** rather than dragged out (Q-c).
#: They are shape tools in every way the options panel and the lock check care
#: about -- a size, a fill, one write -- and not shape tools at all to the press
#: dispatcher, which has to route them into the multi-click gesture instead of
#: starting a ``drag_kind="shape"`` drag.
PATH_SHAPE_TOOLS = frozenset({"polyline", "polygon", "curve"})

#: The shape tools with no inside, and therefore no **Filled** checkbox. The
#: line was the only one before Q-c; a polyline and a curve are open paths for
#: the same reason it is, and a fill checkbox on a shape that encloses nothing
#: is a control that does nothing.
OPEN_SHAPE_TOOLS = frozenset({"line", "polyline", "curve"})
SELECT_TOOLS = frozenset({"select", "select_ellipse", "lasso", "wand", "lasso_poly"})

#: Tools a **right-click** drives, painting with the background colour instead
#: of the foreground one. Deliberately short: the selection tools stay inert on
#: the right button, which reserves it for whatever they may want it to mean,
#: and an inert button is a promise that can be kept later where a wrong one
#: cannot be taken back. Spray is not here for the same reason -- one gesture,
#: one meaning, and it can be added the day somebody asks.
BG_BUTTON_TOOLS = frozenset({"brush", "eraser", "fill"})

#: Tools that will stamp a captured image instead of a generated disc.
#:
#: The three paint tools whose ``BRUSH_MODES`` entry is in
#: ``brush.STAMP_MODES`` -- the ones that write a colour the caller supplies,
#: and an image tip is exactly "the colour, per pixel". Blur, smudge and shade
#: decide what they write from what is already on the layer, so a picture has
#: nothing to say to them; the engine drops a stamp handed to one rather than
#: half-applying it, and the panel does not offer the controls.
#:
#: **The tip is not a tool of its own**, which is Aseprite's arrangement and
#: the right one: a captured brush replaces the *tip* of the tool in your hand,
#: so everything the tool already does -- symmetry, the spray's emission,
#: tiling, the selection clip -- comes with it and no toolbox slot or shortcut
#: letter is spent.
STAMP_TOOLS = frozenset({"brush", "eraser", "spray"})

# What each tool asks the brush engine for. Kept here rather than branched at
# the call site so a new mode is one row. Spray asks for ``paint``: it is a
# different way to *emit* dabs, not a different kind of dab.
BRUSH_MODES = {
    "brush": "paint", "eraser": "erase", "blur": "blur", "smudge": "smudge",
    "spray": "paint", "shade": "shade",
}

#: Why the shading tool cannot be used on a document, keyed by what is wrong.
#: Written here rather than in the pane because the *canvas* says the same thing
#: in a toast when a press is refused, and two spellings of one refusal is how a
#: user comes to believe there are two different problems.
SHADE_REASONS = {
    "none": "Shading needs a palette. Make the document indexed in the Colour panel.",
    "one": "Shading needs at least two colours in the palette to step between.",
}


def tool_reason(tool: str, doc: Any = None) -> str:
    """Why *tool* cannot be used on *doc* right now, or ``""``.

    One function rather than a check per call site: the toolbox greys the
    button and shows this as its tooltip, the options panel prints it under the
    heading, and the canvas toasts it if a shortcut key got the tool selected
    anyway. A tool with nothing to say answers the empty string, which is every
    tool but one.

    ``doc`` is optional and ``None`` disables nothing: with no document open
    every tool is equally useless, and greying the whole toolbox to say so
    would be noise rather than information.
    """
    if tool != "shade" or doc is None:
        return ""
    palette = getattr(doc, "palette", None)
    if not palette:
        return SHADE_REASONS["none"]
    if len(palette) < 2:
        return SHADE_REASONS["one"]
    return ""


#: How wide a spray's dabs are, as a fraction of the tool's size. ``brush_size``
#: is the diameter of the *disc the dabs land in* for this one tool -- which is
#: what the brush cursor already draws and what a "spray width" control means
#: everywhere else -- so the dab itself has to be some part of it. A spray whose
#: dabs are as wide as its own disc is a blob; a quarter is a spray.
SPRAY_DAB_FRACTION = 0.25

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
    # Which ink the brush lays down: ``blend`` composites the colour over what
    # is there, ``replace`` writes it verbatim including alpha. Per tool like
    # everything else here, and offered on the brush alone -- this app has brush
    # modes and layer locks instead of Aseprite's per-tool ink selector.
    "paint_ink": "blend",
    # Dabs a second, for the spray tool. A rate rather than a per-frame count,
    # so a spray lays down the same cloud on a slow machine as on a fast one.
    "spray_rate": 90,
    # ``"none"`` or one of ``dither.ORDERED``. Per tool like everything else
    # here, and defaulted off because a dithered gradient is a pixel-art
    # decision rather than a better gradient -- on a photograph it is noise.
    "gradient_dither": "none",
    # The text tool's three, per tool like everything else here even though one
    # tool is the only one that reads them: a table with an exception in it is
    # a table somebody has to remember the exception to.
    #
    # ``font`` is a *path*, and empty means the vendored Inter face -- resolved
    # where the popup is drawn (``inker_canvas.font_choices``), because a
    # default that named a file would be a default that can stop existing. The
    # size is in pixels, which is what the rasteriser takes and what the canvas
    # shows; a point size would be a number about paper.
    "text_size": 24,
    "font": "",
    "aa": True,
    # Which way the shading ink walks its ramp: +1 toward the end of the
    # selected run, -1 toward its start. Per tool because it *is* the shading
    # tool's one setting, and +1 because a palette is conventionally arranged
    # dark to light, which makes the unmodified drag the one that lights.
    "shade_dir": 1,
    # The image brush's two, per tool like everything else here -- which is
    # what lets the eraser carry a stamp-shaped tip while the brush keeps its
    # round one, and what makes "use the captured tip" a thing a preset can
    # say. ``stamp_align`` is one of ``brush.STAMP_ALIGN``.
    "use_stamp": False,
    "stamp_align": "free",
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

    # Which axes this tab is drawn -- and painted -- wrapped on; one of
    # ``inker.tiling.TILED_AXES``. **Per tab and never in the file**: whether
    # you are looking at a texture as a tile is a property of how you are
    # working on it this afternoon, not of the picture, and a document that
    # opened tiled because somebody once toggled it would be a surprise with no
    # undo step to reverse. The engine is told explicitly at every call, so it
    # stays stateless about the UI.
    tiled: str = "off"

    # The timeline's range selection, as an inclusive cell rect
    # ``(t0, t1, f0, f1)`` of **indices**, or None for "nothing selected".
    #
    # View state on the tab, like the playhead and for the same reason: it
    # pushes no undo step and is in no file. Indices rather than uids is the
    # choice ``Tag`` already makes -- a range names a region of the timeline,
    # so a frame inserted inside it should widen it -- and it is **clamped at
    # use, never at store** (Plotter's selection rule): trimming it on every
    # delete would silently shrink it under the user, and the engine clamps
    # every op it is handed anyway.
    range_sel: tuple[int, int, int, int] | None = None

    # Preview-pane playback. A second, independent playhead: it never touches
    # ``playing``, ``play_index`` or the document's ``anim.current``, which is
    # the whole reason the preview can run while the canvas is being drawn on.
    preview_playing: bool = False
    preview_index: int = 0
    preview_accum_ms: float = 0.0
    preview_forward: bool = True
    preview_cycles: int = 0
    #: dt multiplier, 0.25 .. 4.0. A preview option rather than a document
    #: playback mode -- see the divergence list.
    preview_speed: float = 1.0
    #: "clip" plays the whole timeline; "tag" plays the tag under the preview's
    #: own index, honouring its direction, loop flag and repeat count.
    preview_scope: str = "clip"

    # How many times the *document* playhead has been round its span since play
    # started, for a tag with a finite repeat count. Reset on every play, so a
    # clip stopped and started again plays its full count rather than
    # remembering that it already finished once.
    play_cycles: int = 0

    # Which layer groups are folded shut in the panel, by group uid. **View
    # state, and deliberately on the tab rather than on the document**: whether
    # a folder is open says nothing about the picture, so it is neither
    # persisted into the ``.ora`` nor undoable -- for the reason the playhead is
    # neither. A document that asked to be saved because somebody collapsed a
    # folder would make ``dirty`` a lie.
    collapsed_groups: set[int] = field(default_factory=set)

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
    #: The captured image brush -- an ``inker.Stamp`` or None. App-level like
    #: every other tool setting, so one tip is held across tabs, and typed
    #: loosely so this module keeps importing nothing from the engine.
    #:
    #: **Not persisted, and not part of a preset.** It is pixels rather than a
    #: setting: the settings block is small JSON rewritten on every swatch
    #: click, and a quarter of a megabyte of base64 in it would be paid for on
    #: every one. A preset carries the options -- including which of them says
    #: "use the captured tip" -- and the tip is recaptured from the drawing,
    #: which is where it came from and where it still is.
    stamp: Any = None
    #: Named bundles of one tool's options; see :meth:`save_preset`. Persisted
    #: beside the swatches by ``inker_mode.persist``.
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    #: The slots the user has selected, in the order they picked them, with
    #: ``palette_slot`` as the anchor a Shift+click ranges from. A *list* rather
    #: than a set: "sort these five" and "ramp from this one to that one" are
    #: both about the order, and a set would have to invent one.
    palette_slots: list[int] = field(default_factory=list)
    #: The sort the palette panel would apply, and its direction, and how many
    #: colours "Insert ramp" would put between two slots. Remembered across
    #: documents like every other tool setting, and persisted like none of them:
    #: they describe a gesture, not a file.
    palette_sort: str = "luma"
    palette_sort_desc: bool = False
    palette_ramp: int = 3
    #: The open conversion session: **which tab owns it**, which method it is
    #: previewing, how many colours a built table may have, and the table
    #: itself. View state, none of it persisted -- the *document* keeps the
    #: table a conversion produces, and this is only what is being tried.
    #:
    #: ``convert_uid`` is a tab uid and not a bool, and that is the whole of the
    #: difference between this and the filter popup it was cloned from. A
    #: conversion session lives on one ``Document`` (``Document._convert``) while
    #: this state is one object shared by every tab, and the pane draws whichever
    #: tab is *active* -- so a plain "the popup is up" flag meant that switching
    #: tabs with it open cancelled the conversion on the wrong document and left
    #: the right one holding a preview nobody would ever answer. Holding the uid
    #: makes both halves address the same document by name; see
    #: ``inker_mode.end_convert_session``. Empty means no session.
    convert_uid: str = ""
    convert_method: str = "nearest"
    convert_max: int = 16
    #: ``convert_table`` is held rather than recomputed per frame because
    #: building one is a pass over every plane of the document, and because the
    #: preview moves ``doc.rev`` every frame -- so there is no cache key made of
    #: document state that would not thrash. It is rebuilt when the slider that
    #: decides it moves, and at no other time.
    convert_table: list[tuple[int, int, int, int]] = field(default_factory=list)

    # Drag state, decided on press because several tools start the same way.
    drag_kind: str = ""  # "" | paint | spray | shape | marquee | lasso | move |
    #                       layer_move | gradient | pan
    drag_anchor: tuple[float, float] | None = None
    last_point: tuple[float, float] | None = None
    lasso: list[tuple[float, float]] = field(default_factory=list)
    combine: str = "replace"
    space_held: bool = False
    # Which mouse button started the drag: 0 paints with the foreground colour,
    # 1 with the background one. Stored rather than re-read, because a gesture
    # belongs to the button that began it -- testing "is the left button down"
    # on the release of a right-drag ends the gesture on the wrong frame.
    drag_button: int = 0
    # The whole tile the press landed in, subtracted from every point of the
    # gesture. Computed once at press and *not* per point: folding each sample
    # independently makes the brush jump a full tile the moment the cursor
    # crosses a seam mid-stroke.
    tile_offset: tuple[float, float] = (0.0, 0.0)
    # The fractional part of ``spray_rate x delta_time`` carried between frames,
    # so a rate below one dab per frame still emits rather than rounding to
    # nothing sixty times a second.
    spray_carry: float = 0.0

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
    # rather than against the previous frame. Widened past four entries when
    # the scale became per-axis; ``inker_canvas._transform_input`` names them.
    transform_ref: tuple[float, ...] | None = None
    # Which grab point the drag started on -- ``inker_canvas.HANDLE_AXES`` maps
    # it to the axes it scales. A name rather than a second ``drag_kind``
    # spelling, because every one of them is the same *kind* of drag.
    transform_grab: str = ""
    # Whether the numeric Scale X and Scale Y fields move together. A view
    # preference, remembered across documents like every other tool setting.
    transform_link: bool = True

    # -- the timeline's range selection ------------------------------------
    #
    # The rect itself is per tab (``InkerDoc.range_sel``); what is app-level is
    # the *gesture*: where a drag or a Shift+click measures from, and the
    # clipboard a range copy fills. The clipboard is here rather than on the
    # document precisely so a range copied in one tab pastes into another --
    # which is the only reason to reach for it over Duplicate.
    timeline_anchor: tuple[int, int] | None = None
    cel_clip: Any = None
    # The duration the timeline's range menu applies, in ms. A setting rather
    # than a prompt: retiming a span is something a user does repeatedly with
    # the same number.
    range_ms: int = 100
    # Cel thumbnails in the timeline cells. App-level like every other view
    # preference, and off by default: the cells are 20px without it and the
    # off path draws exactly what it always did.
    timeline_thumbs: bool = False
    # Nearest-neighbour multiplier applied to sheet, GIF and PNG exports.
    export_scale: int = 1

    # -- importing a sprite sheet ------------------------------------------
    #
    # ``sheet_import`` is ``(atlas array, title)`` once a file has been chosen
    # and decoded off-thread, and None the rest of the time. The picture is
    # read *before* the grid is asked for, so the popup can say how many frames
    # the numbers actually produce -- which is the one thing that stops a
    # mistyped cell size becoming a document the user edits for ten minutes
    # before noticing. ``sheet_import_open`` is what notices imgui closing the
    # popup on a click outside, the ``filter_open`` idiom.
    sheet_import: Any = None
    sheet_import_open: bool = False
    sheet_cell: tuple[int, int] = (32, 32)
    sheet_offset: tuple[int, int] = (0, 0)
    sheet_padding: tuple[int, int] = (0, 0)
    #: 0 means "as many as the grid holds", so a user who never touches the
    #: field gets the whole sheet rather than one frame.
    sheet_count: int = 0

    # -- the multi-click gesture (C4) ---------------------------------------
    #
    # The second kind of gesture this pane has. A *drag* is a press, a run of
    # moves and a release, and ``drag_kind`` names it for as long as the button
    # is down; a multi-click gesture is a run of separate clicks with the button
    # up in between, so it cannot be held in ``drag_kind`` at all -- the press
    # dispatcher refuses a press while a gesture "owns the mouse" (see
    # ``inker_canvas._input``), and a poly-lasso click sequence must pass that
    # gate rather than be eaten by it. So each click is complete at the press:
    # ``drag_kind`` stays empty, no drag runs, no release fires, and what
    # survives between clicks is exactly these two fields.
    #
    # Shared infrastructure rather than the poly lasso's private state: the
    # curve and polygon shape tools are the same gesture with a different
    # landing, and the alternative is a second copy of the open/extend/close
    # arithmetic that can disagree with this one about what a click near the
    # first vertex means.
    #
    #: The vertices placed so far, in **image** space and already grid-snapped
    #: (``inker_canvas._snapped``), empty when no gesture is open.
    gesture_pts: list[tuple[float, float]] = field(default_factory=list)
    #: How the finished polygon combines with the live selection, captured at
    #: the **first** click and not re-read afterwards. A gesture is one act: the
    #: user decides "this one adds" when they start it, and reading Shift again
    #: at the closing click would let a modifier let go halfway through turn an
    #: add into a replace that throws the selection away.
    gesture_combine: str = "replace"

    # -- the text tool's open popup ----------------------------------------
    #
    # What is being typed, and where the click that opened the popup landed.
    # Pure view state: a stamp is a floating buffer once it is made, so nothing
    # here survives the OK button and nothing here may push an undo step.
    #
    # ``text_at`` is recorded at the *press* rather than read when OK is
    # clicked, because by then the mouse is over a button in a popup somewhere
    # else on the screen. The buffer is remembered across popups on purpose:
    # retyping is how this editor re-edits text (there are no text objects), so
    # a second stamp of the same word with a bigger size must not start empty.
    text_buffer: str = ""
    text_at: tuple[int, int] = (0, 0)
    #: Whether the user has set the Antialias box themselves. Until they have,
    #: the popup decides it from the document on every open -- off on an
    #: indexed one, on everywhere else -- which is what the manual promises,
    #: with no session in the sentence.
    #:
    #: A flag beside the option rather than a state *of* the option, and that
    #: is the whole of the bug it fixes: ``options_for`` materialises all of a
    #: tool's keys from the defaults the first time any one of them is read, so
    #: "has this ever been set" is a question the stored dictionary cannot
    #: answer. Asking it that way ("is there an entry for the text tool") made
    #: the promise last exactly one popup -- after a single stamp on an RGB
    #: document, every indexed document for the rest of the session opened with
    #: antialiasing on.
    text_aa_touched: bool = False
    #: What is typed in the preset panel's name box. Pure view state, like the
    #: tag rename buffer beside it: not persisted, and it pushes nothing.
    preset_name: str = ""

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
    paint_ink = _tool_option("paint_ink")
    spray_rate = _tool_option("spray_rate")
    gradient_dither = _tool_option("gradient_dither")
    text_size = _tool_option("text_size")
    font = _tool_option("font")
    aa = _tool_option("aa")
    shade_dir = _tool_option("shade_dir")
    use_stamp = _tool_option("use_stamp")
    stamp_align = _tool_option("stamp_align")

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
        tool = tool or self.tool
        self.tool_options.pop(tool, None)
        if tool == "text":
            # ``text_aa_touched`` is part of the text tool's stored settings --
            # it is only outside the dictionary because ``options_for`` cannot
            # represent "unset" -- so a reset that left it standing would put
            # the option back to its default and keep overriding it, which is a
            # Reset button that does not reset.
            self.text_aa_touched = False

    def tip_for(self, tool: str) -> Any:
        """The image tip *this tool* would stamp right now, or None.

        One answer with three readers -- the press that opens a stroke, the
        brush cursor that has to draw the right outline, and the options panel
        -- because the three disagreeing is exactly how a user comes to believe
        the feature is broken: a ring drawn round a captured tip that the press
        then does not use looks like the click missed.
        """
        if self.stamp is None or tool not in STAMP_TOOLS:
            return None
        return self.stamp if self.options_for(tool)["use_stamp"] else None

    # -- tool presets -------------------------------------------------------
    #
    # A preset is a named copy of one tool's options and nothing else. Not a
    # snapshot of the whole session: the colours, the symmetry, the grid and
    # the onion skin are app-level *because* they are properties of the canvas
    # or of the sitting rather than of a tool, and a preset that dragged them
    # along would turn "my inking pen" into "my inking pen, and also switch the
    # grid off". Not the captured tip either -- see ``stamp``.

    def save_preset(self, name: str) -> str:
        """Store the active tool's options under ``name``. -> the name used.

        The *whole* dictionary rather than what differs from the defaults, so a
        preset keeps meaning what it meant if a default is ever changed under
        it. Overwrites silently by name, which is what a dictionary of names
        does and what "save over it" means to a user typing the same one twice.

        Oldest first out at the cap: this is a session convenience persisted in
        a settings block, not a library, and a list that only ever grows is one
        the user cannot get out of once it fills their panel.
        """
        name = str(name).strip()[:MAX_PRESET_NAME]
        if not name:
            return ""
        self.presets[name] = {
            "tool": self.tool,
            "options": dict(self.options_for(self.tool)),
        }
        for stale in list(self.presets)[: max(0, len(self.presets) - MAX_PRESETS)]:
            del self.presets[stale]
        return name

    def apply_preset(self, name: str) -> bool:
        """Select the preset's tool and put its options on. -> whether it ran.

        Through :meth:`set_tool`, like every other way of picking one, so a
        half-drawn multi-click gesture goes with the switch.

        The tool's options are **replaced** rather than updated, over a base of
        the defaults: a preset saved before an option existed must leave that
        option at its default rather than at whatever the tool happened to be
        carrying, or applying the same preset twice in one session gives two
        different brushes. Keys the table no longer has are dropped, which is
        what makes a stored block from an older build safe to read.
        """
        saved = self.presets.get(str(name))
        if not isinstance(saved, dict):
            return False
        tool = saved.get("tool")
        if tool not in {key for key, _label, _short in TOOLS}:
            return False
        options = saved.get("options")
        self.set_tool(tool)
        self.tool_options[tool] = {
            **TOOL_OPTION_DEFAULTS,
            **{
                key: value
                for key, value in (options or {}).items()
                if key in TOOL_OPTION_DEFAULTS
            },
        }
        return True

    def delete_preset(self, name: str) -> bool:
        return self.presets.pop(str(name), None) is not None

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
        self.drag_button = 0
        self.tile_offset = (0.0, 0.0)
        self.spray_carry = 0.0
        self.transform_grab = ""
        # An open multi-click gesture goes with it, which is what makes a tab
        # switch, a tab close and Escape cancel a half-drawn polygon for free --
        # all three already come through here. Safe *because* a gesture holds no
        # ``drag_kind``: the click cycle that builds one never reaches
        # ``_release``, so this is only ever called from outside the gesture.
        self.clear_gesture()

    def clear_gesture(self) -> None:
        """Drop a half-finished multi-click gesture. Writes nothing.

        Separate from :meth:`clear_drag` as well as called by it, because the
        canvas cancels a gesture in places where a drag is not in flight at all
        (the tab going busy mid-polygon) and a tool switch must drop the
        polygon without also resetting the drag button under a live drag.
        """
        self.gesture_pts = []
        self.gesture_combine = "replace"

    def set_tool(self, tool: str) -> None:
        """Pick a tool, dropping whatever gesture was open.

        The one door, so a tool cannot be changed from the keyboard, the
        toolbox or a paste without the half-drawn polygon on screen going with
        it -- a gesture belongs to the tool that started it, and vertices left
        behind would be committed by the *next* tool's closing click.
        """
        if tool != self.tool:
            self.clear_gesture()
        self.tool = tool

    # -- colours ------------------------------------------------------------

    def swap_colours(self) -> None:
        self.fg, self.bg = self.bg, self.fg

    def add_swatch(self, colour: tuple[int, int, int, int]) -> None:
        colour = tuple(int(c) for c in colour)  # type: ignore[assignment]
        if colour in self.swatches:
            return
        self.swatches.append(colour)
        del self.swatches[:-MAX_SWATCHES]

    # -- palette slots ------------------------------------------------------

    def select_slot(self, index: int, *, ctrl: bool = False, shift: bool = False) -> None:
        """Click, Ctrl+click and Shift+click on a palette slot, in one place.

        The three-way gesture every list in every editor uses, and it lives on
        the state rather than in the pane so it can be asserted without a
        window. ``palette_slot`` stays the **anchor** throughout -- it is what a
        Shift+click ranges from, and what every single-slot control (the Slot
        picker, Remove, the reorder arrows) acts on -- and ``palette_slots``
        carries the selection.

        Ctrl+click toggling the anchor's own slot leaves the anchor where it is:
        the anchor is not a member of the selection, it is where the next range
        starts, and moving it on a *deselect* would make the next Shift+click
        range from a slot the user had just taken out.
        """
        index = max(0, int(index))
        if shift:
            # Anchored on ``palette_slot`` and *replacing* the selection, not
            # adding to it: a range is a statement about two endpoints, and the
            # editors that add instead make a stray Shift+click impossible to
            # take back without starting over.
            low, high = sorted((self.palette_slot, index))
            self.palette_slots = list(range(low, high + 1))
        elif ctrl:
            if index in self.palette_slots:
                self.palette_slots.remove(index)
            else:
                self.palette_slots.append(index)
                self.palette_slot = index
        else:
            self.palette_slots = [index]
            self.palette_slot = index

    def clamp_slots(self, count: int) -> None:
        """Drop selected slots the palette no longer has.

        Called wherever the table is drawn, because every op that shortens it --
        Remove, an undo, a whole reconversion -- can leave a selection pointing
        past the end, and every consumer of ``palette_slots`` indexes with it.
        """
        self.palette_slot = max(0, min(self.palette_slot, count - 1))
        self.palette_slots = [i for i in self.palette_slots if 0 <= i < count]

    @property
    def selected_slots(self) -> list[int]:
        """The selection, or the anchor alone when there is none.

        What every multi-slot op takes, so "sort" with nothing selected sorts
        the whole table rather than one slot -- see the call sites, which pass
        ``None`` for that case; this is for the ops that need at least one.
        """
        return list(self.palette_slots) if self.palette_slots else [self.palette_slot]



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
