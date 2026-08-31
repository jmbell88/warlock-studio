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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import docmodes
from .tilegrid import gid

MIN_ZOOM = 0.05
MAX_ZOOM = 32.0
ZOOM_STEP = 1.15

# Inker's own bounds. Separate constants rather than a change to the three
# above, because Plotter and Packwright import those and only the *ceiling* is
# Inker's own now: 32x is a tile map's magnifier and a drawing has a nib.
# Every view function below takes ``lo``/``hi`` defaulting to the globals, so
# those two callers are untouched and this pane passes its own pair.
#
# **The floor was 25% and is now the global 5%, and that is a reversal.** The
# argument for 25% was that a drawing at 5% is a postage stamp nobody can nib
# -- but the paragraph making it also recorded what it cost, three lines
# later: an image too large to fit at 25% centres and *overflows* the pane
# rather than shrinking to meet it, so Fit does not fit. A floor that breaks
# the one control whose entire meaning is "show me all of it" is buying the
# wrong thing: nobody nibs at 5% on purpose, and the way you leave 5% is the
# same wheel notch that got you there.
INKER_MIN_ZOOM = 0.05
INKER_MAX_ZOOM = 10.0

# The wheel's granularity, in percent. Additive and *snapped* rather than the
# multiplicative ``ZOOM_STEP``: a 15% ratio step from 100% lands on 115, 132.25,
# 152.09 -- the status bar reads a different arbitrary number every notch, and
# there is no way back to a round one. See :func:`zoom_step`.
ZOOM_PERCENT_STEP = 5

#: The keyboard's zoom ladder, as whole scales.
#:
#: **Integer above 1:1, halving below it.** This is the pixel-art rule and the
#: reason the keyboard does not simply reuse the wheel's 5% steps: at 135% a
#: source pixel is 1.35 screen pixels, so the renderer draws some of them one
#: pixel wide and some two, and a checkerboard dither comes out as bands. Every
#: rung here maps one source pixel onto a whole number of screen pixels (or a
#: whole number of source pixels onto one), which is the only family of zooms at
#: which pixel art is being shown rather than resampled.
#:
#: The wheel keeps its 5% notches deliberately -- it is the *fine* control, and
#: ``test_canvas_input`` pins the notch size. Aseprite splits the two the same
#: way round.
#: Three rungs below the old 25% floor arrived with it (1/20, 1/10 and 1/8 --
#: all whole numbers of source pixels onto one, so the rule above holds).
#:
#: **This is not :data:`ZOOM_PRESETS` and must not be "synced" with it.** The
#: two tables answer two questions. The combo is *"show me exactly this
#: number"*, and honouring 75% there is correct even though a source pixel is
#: then 0.75 screen pixels -- the user asked. The ladder is *"the next honest
#: scale"*, and putting 75% on it would walk +/- into banding unasked. A test
#: asserts the divergence so a future tidy-up of the two tables fails loudly.
ZOOM_LADDER = (0.05, 0.1, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)

#: What the zoom combo in the canvas footer offers, as whole scales.
#:
#: Aseprite's own list, and deliberately *wider* than the ladder: see the note
#: there for why 75% belongs on exactly one of the two.
ZOOM_PRESETS = (0.05, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0)


def zoom_key(zoom: float) -> str:
    """A preset's stable key: its percentage, trimmed. ``0.125`` -> ``"12.5"``.

    A string rather than Plotter's ``int(picked) / 100``, because this list
    holds a half-percent (12.5%) that an int round-trip silently turns into
    12%, which is not a rung and not what the label says.
    """
    return f"{round(zoom * 100, 2):g}"


def zoom_rung(
    zoom: float, direction: int, ladder: tuple[float, ...] = ZOOM_LADDER
) -> float:
    """The next rung of ``ladder`` in ``direction``. -> the new scale.

    Strictly past the current zoom rather than nearest-then-step, so a view
    sitting between two rungs at 135% zooms *out* to 100% and *in* to 200%
    instead of snapping sideways to 100% on a press labelled "in". At either end
    the ladder holds, which is the same answer the wheel gives at its bounds.

    ``ladder`` is a parameter because the Plotter tileset palette has its own
    (``plotter_state.PALETTE_ZOOM_LADDER``) and the *stepping rule* above is the
    part worth sharing -- particularly "strictly past", which is what makes a
    fit-derived zoom that sits between two rungs step sanely. Copying the scan
    over there would have been two implementations of one sentence.
    """
    if direction > 0:
        return next((rung for rung in ladder if rung > zoom + 1e-6), ladder[-1])
    return next((rung for rung in reversed(ladder) if rung < zoom - 1e-6), ladder[0])

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

# The toolbox, in the order the buttons are drawn. **Order is presentation and
# nothing reads it as data** -- every rule about a tool is a membership test
# against one of the frozensets below -- so this is arranged for the hand rather
# than for the parser, in Aseprite's four bands: what lays down colour, then the
# shapes, then the selections, then the utilities. Within a band the pairs that
# are reached for together sit together (brush/spray, line/curve, the two
# lassos), which is what makes a five-wide grid readable without hovering it.
TOOLS = (
    # Colour, and the three that decide what they write from what is already
    # there (blur, smudge, shading) at the end of the band.
    ("brush", "Brush", "B"),
    ("spray", "Spray", "A"),
    ("eraser", "Eraser", "E"),
    ("fill", "Fill", "G"),
    # ``K`` rather than the ``U`` this had, because ``U`` is Aseprite's
    # rectangle and a gradient is not what a hand trained there expects from
    # it. The gradient keeps a plain letter as well as ``Shift+G``, which is
    # where Aseprite actually files it -- sharing the paint bucket's slot.
    ("gradient", "Gradient", "K"),
    ("blur", "Blur", "R"),
    ("smudge", "Smudge", "N"),
    # ``H`` for shading, because Aseprite has no letter to borrow here -- its
    # shading is an *ink* on the ordinary brush rather than a tool -- and every
    # letter in the word that reads as a mnemonic was taken (S is the ellipse
    # marquee, A the spray, D and T are spoken for elsewhere).
    ("shade", "Shading", "H"),
    # The shapes, dragged ones and clicked ones interleaved by what they draw
    # rather than by how they are drawn: a user reaching for "a curved line"
    # wants it beside the straight one.
    #
    # **``L`` is the line and ``U`` is the rectangle, because that is where
    # Aseprite puts them.** They did not start here: the letters were handed out
    # in toolbox order, so ``L`` had gone to the polyline and ``U`` to the
    # gradient before the line and the rectangle were reached -- and a hand
    # trained on Aseprite pressing ``L`` for a line got a polyline, which is a
    # *different gesture* (click, click, Enter) rather than a near miss. The two
    # tools they displaced take the letters they vacated: the polyline gets
    # ``P``, still the first letter of its own name, and the gradient ``K``.
    #
    # The clicked shapes took what was left. Not one letter of "curve" is free
    # -- C is the slice, U the rectangle, R the blur, V the move, E the eraser --
    # so the curve takes ``F``, for the *free-form* curve it draws, and the
    # polygon ``O``, "polygon" having nothing else spare.
    ("line", "Line", "L"),
    ("curve", "Curve", "F"),
    ("rect", "Rect", "U"),
    ("ellipse", "Ellipse", "J"),
    ("polyline", "Polyline", "P"),
    ("polygon", "Polygon", "O"),
    # The selections, the freehand pair adjacent for the reason above.
    ("select", "Marquee", "M"),
    ("select_ellipse", "Ellipse select", "S"),
    ("lasso", "Lasso", "Q"),
    ("lasso_poly", "Poly lasso", "D"),
    ("wand", "Wand", "W"),
    # The utilities: everything whose gesture is about the drawing rather than
    # about paint.
    ("move", "Move", "V"),
    ("eyedropper", "Pick", "I"),
    ("text", "Text", "T"),
    ("slice", "Slice", "C"),
    # The tile stamp (Wave 3). **``Y`` because it is the only letter left that
    # is bound to nothing at all.** Every mnemonic in "tile", "stamp" and
    # "tilemap" was already spoken for -- T is the text tool, I the
    # eyedropper, L the line, E the eraser, S the elliptical marquee, M the
    # marquee, A the spray, P the polyline -- and of the three letters no tool
    # holds, ``X`` already swaps the colours and ``Z`` sits under Ctrl+Z, where
    # a slipped modifier would swap a tool for an undo. Aseprite has no letter
    # to borrow here: it has no tile *tool*, it puts the whole editor into a
    # tilemap mode instead.
    ("tile", "Tile stamp", "Y"),
)

#: The toolbox's twelve slots, each holding the tools that answer one question.
#:
#: **Separate from :data:`TOOLS`, not a fourth element of it.** ``TOOLS`` is
#: unpacked three-at-a-time in ``main.shortcut_sections``, the toolbox grid and
#: two tests; a fourth element breaks all four at once. It is also the wrong
#: shape: a group is a *set* of tools, and the row that owns a group is not a
#: property of any one tool in it.
#:
#: Aseprite's own arrangement, adapted to the tools this editor has: what the
#: hand reaches for is the group, and which member is on it is a detail the
#: user sets once by sliding onto it. Twelve slots is two columns of six, which
#: is what lets the toolbox be a 90 px rail instead of a 300 px column.
TOOL_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("brush", "Brush", ("brush", "spray")),
    ("eraser", "Eraser", ("eraser",)),
    ("fill", "Fill", ("fill", "gradient")),
    ("effects", "Effects", ("blur", "smudge", "shade")),
    ("line", "Line", ("line", "curve")),
    ("rect", "Shapes", ("rect", "ellipse")),
    ("path", "Paths", ("polyline", "polygon")),
    ("select", "Marquee", ("select", "select_ellipse")),
    ("lasso", "Lasso", ("lasso", "lasso_poly")),
    ("wand", "Wand", ("wand",)),
    ("move", "Move", ("move", "eyedropper")),
    ("util", "Utilities", ("text", "slice", "tile")),
)

#: Which group a tool sits in. Derived, so a tool can never be in two.
GROUP_OF: dict[str, str] = {
    tool: key for key, _label, tools in TOOL_GROUPS for tool in tools
}


def group_members(key: str) -> tuple[str, ...]:
    """The tools in one group, or empty if there is no such group."""

    return next((tools for name, _label, tools in TOOL_GROUPS if name == key), ())


# ``group_label`` was deleted on 2026-08-22 with zero callers. Its sibling
# ``group_members`` above is live; a label for a group is read off
# ``TOOL_GROUPS`` where it is needed.


def cycle_in_group(current: str, wanted: str) -> str:
    """Which tool a letter should select, given the tool in hand.

    Aseprite's rule, and the reason key cycling is additive here rather than a
    change to ``TOOLS``: **the first press always lands where it always did.**
    ``U`` picks the rectangle from anywhere else; pressing it again, with the
    rectangle already in hand, moves along its group to the ellipse. So every
    binding that existed still means what it meant, and the second press is a
    new fact rather than a moved one.
    """

    group = GROUP_OF.get(wanted)
    members = group_members(group) if group else ()
    if len(members) < 2 or current not in members:
        return wanted
    return members[(members.index(current) + 1) % len(members)]


def flyout_cells(
    rect: tuple[float, float, float, float], cell: tuple[float, float], count: int
) -> list[tuple[float, float, float, float]]:
    """Where a group's flyout strip draws, given the button it hangs off.

    Pure, in physical pixels, and **the same function places the cells and
    hit-tests them** (:func:`flyout_hit`) -- which is what makes it impossible
    for the picture and the hit box to disagree, the defect every hand-rolled
    flyout has.

    To the right of the button and vertically centred on it, which is where
    Aseprite's is: the toolbox is against the left edge of the window, so a
    strip anywhere else would open off screen.
    """

    x, y, w, h = rect
    cw, ch = cell
    top = y + (h - ch) * 0.5
    return [(x + w + index * cw, top, cw, ch) for index in range(count)]


def flyout_hit(
    rect: tuple[float, float, float, float],
    cell: tuple[float, float],
    count: int,
    mouse: tuple[float, float],
) -> int | None:
    """Which cell of the strip the mouse is over, or None."""

    for index, (x, y, w, h) in enumerate(flyout_cells(rect, cell, count)):
        if x <= mouse[0] < x + w and y <= mouse[1] < y + h:
            return index
    return None


#: The key contexts, in the order they are tested. **First match wins**, which
#: is what makes them mutually exclusive by construction rather than by six
#: predicates each having to know about the other five -- the way the modal
#: arms of ``handle_key`` used to, and the way they drifted.
#:
#: Each entry is ``(name, predicate)`` over ``(state, tab)``. ``"Normal"`` is
#: last and always true, so there is no such thing as no context.
KEY_CONTEXTS: tuple[tuple[str, Any], ...] = (
    ("Transformation", lambda state, tab: state.transforming),
    ("Float", lambda state, tab: tab is not None and tab.doc.floating is not None),
    ("Gesture", lambda state, tab: bool(state.gesture_pts)),
    (
        "FramesSelection",
        lambda state, tab: tab is not None and getattr(tab, "range_sel", None) is not None,
    ),
    ("Selection", lambda state, tab: tab is not None and tab.doc.mask is not None),
    ("MoveTool", lambda state, tab: state.tool == "move"),
    ("ShapeTool", lambda state, tab: state.tool in SHAPE_TOOLS),
    ("FreehandTool", lambda state, tab: state.tool in PAINT_TOOLS),
    ("Normal", lambda state, tab: True),
)


def key_context(state: Any, tab: Any) -> str:
    """Which context the keyboard is in. Pure, and the whole of the decision.

    Aseprite's contexts, and the reason to have them at all: Enter means
    "apply the transform", "close the polygon" and "play" in three different
    situations, and the only alternative to naming the situations is three
    branches at the top of the key handler that each have to remember the other
    two.
    """

    for name, applies in KEY_CONTEXTS:
        if applies(state, tab):
            return name
    return "Normal"  # pragma: no cover - the table's last row is always true


def tool_label(key: str) -> str:
    """The display name for a tool key, or the key if it is not one.

    One implementation, because there were four and two of them were wrong.
    The toolbox grid and the status bar each looked the name up out of
    ``TOOLS``; the Reset button and the preset list instead spelled the *key*
    with its underscores swapped for spaces, so the box that said "Ellipse
    select" was reset by a button that said "Reset select ellipse".
    """
    return next((label for tool, label, _ in TOOLS if tool == key), key)


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

#: Tools that use a captured tip as a **fill source** rather than as a dab.
#:
#: The bucket, and Aseprite's own arrangement: with an image brush loaded the
#: paint bucket pours the picture instead of the swatch. **A pattern is a stamp
#: used as a fill source**, which is why this reads ``state.stamp`` and
#: ``use_stamp`` rather than carrying a pattern of its own -- there is one
#: captured tip in the app, and a second image the bucket alone knew about
#: would be a second thing to capture, name, rotate and forget.
#:
#: Separate from :data:`STAMP_TOOLS` because that set is a *derivation* --
#: the tools whose ``BRUSH_MODES`` entry is in ``brush.STAMP_MODES`` -- and the
#: bucket has no brush mode at all: it never opens a stroke.
PATTERN_TOOLS = frozenset({"fill"})

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

#: Why the tile stamp is out. One sentence with the way forward in it, because
#: the answer is a verb the user has -- the Tiles panel's own buttons.
TILE_REASON = (
    "The tile stamp needs a tilemap layer. Make one in the Tiles panel, or "
    "convert this layer to a tilemap there."
)

#: The three ways a pixel edit on a tilemap layer may reach the tileset under
#: it -- Aseprite's own names, and its own meanings. View state: see
#: ``Document.tile_behavior``, which is never serialized.
TILE_BEHAVIORS = (
    (
        "manual",
        "Manual",
        "Paint is thrown away: the tileset is never changed and the cell is "
        "re-drawn from the tile it already names. Place tiles with the tile "
        "stamp instead.",
    ),
    (
        "auto",
        "Auto",
        "Paint edits the tile itself -- every placement of it, on every frame "
        "and every layer, changes with it.",
    ),
    (
        "stack",
        "Stack",
        "Paint appends a new tile and points this cell at it. No existing tile "
        "is ever modified.",
    ),
)

#: The revert sentence the canvas raises once per gesture, when a paint stroke
#: on a Manual-mode tilemap cel came back with nothing recorded. Derived from
#: view state at the pane -- the document has no channel for it, deliberately:
#: ``_commit_tilemap_patch`` reverts silently and the UI, which is the only
#: thing that knows a *gesture* happened, is where the sentence belongs.
TILE_MANUAL_REVERTED = "Manual mode: the tileset was not changed."


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
    if doc is None:
        return ""
    if tool == "tile":
        # Asked of the *active row* rather than of the document: a drawing may
        # well hold a tilemap layer and have an ordinary one selected, and the
        # stamp writes refs onto whichever row is active or onto nothing.
        probe = getattr(doc, "active_tilemap_uid", None)
        return "" if probe is not None and probe() is not None else TILE_REASON
    if tool != "shade":
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
#: The tool options that live on the context bar, in the order they are drawn.
#:
#: A **table rather than a chain of ifs**, so both directions are assertable:
#: which widgets a tool shows, and -- the one that earns its keep -- that every
#: key in :data:`TOOL_OPTION_DEFAULTS` is reachable from at least one tool's
#: bar. That assertion is what stops an option becoming unreachable the day it
#: leaves the sidebar, which is the whole risk of moving them.
#:
#: ``group`` is ``"dynamics"`` for the four that go behind one popup: spacing,
#: smoothing, taper and strength are the settings of a *stroke*, adjusted once
#: for a way of working and then left, and Aseprite files them the same way.
#: They are still options of the tool and still reachable from its bar, which
#: is why they are rows here rather than an exception to the table.
CONTEXT_WIDGETS: tuple[tuple[str, str, frozenset[str], str], ...] = ()


def _context_table() -> tuple[tuple[str, str, frozenset[str], str], ...]:
    """Built after the tool sets above exist, so the sets are named once."""

    sized = PAINT_TOOLS | SHAPE_TOOLS
    lined = PAINT_TOOLS - {"spray"}
    return (
        ("brush_size", "Size", frozenset(sized), ""),
        ("nib", "Nib", frozenset(PAINT_TOOLS), ""),
        ("brush_angle", "Angle", frozenset(PAINT_TOOLS), ""),
        ("pixel_perfect", "Pixel perfect", frozenset(PAINT_TOOLS - {"spray"}), ""),
        ("hardness", "Hardness", frozenset(PAINT_TOOLS), ""),
        ("opacity", "Opacity", frozenset(PAINT_TOOLS - {"shade"}), ""),
        ("paint_ink", "Ink", frozenset(PAINT_TOOLS), ""),
        ("shade_dir", "Direction", frozenset({"shade"}), ""),
        ("spray_rate", "Rate", frozenset({"spray"}), ""),
        ("spacing", "Spacing", frozenset(lined), "dynamics"),
        ("stabilise", "Smoothing", frozenset(lined), "dynamics"),
        ("speed_taper", "Taper", frozenset(lined), "dynamics"),
        ("strength", "Strength", frozenset({"blur", "smudge"}), "dynamics"),
        ("use_stamp", "Image brush", frozenset(STAMP_TOOLS), "dynamics"),
        ("stamp_align", "Pattern", frozenset(STAMP_TOOLS), "dynamics"),
        ("shape_filled", "Filled", frozenset(SHAPE_TOOLS - OPEN_SHAPE_TOOLS), ""),
        ("corner_radius", "Corners", frozenset({"rect"}), ""),
        ("wand_tolerance", "Tolerance", frozenset({"fill", "wand"}), ""),
        ("wand_contiguous", "Contiguous", frozenset({"fill", "wand"}), ""),
        ("wand_eight", "Diagonals", frozenset({"fill", "wand"}), ""),
        ("fill_refer", "Refer to", frozenset({"fill"}), ""),
        ("fill_stop_grid", "Stop at grid", frozenset({"fill"}), ""),
        ("sample_layer", "This layer only", frozenset({"eyedropper"}), ""),
        ("gradient_dither", "Dither", frozenset({"gradient"}), ""),
        ("text_size", "Size", frozenset({"text"}), ""),
        ("font", "Font", frozenset({"text"}), ""),
        ("aa", "Antialias", frozenset({"text"}), ""),
    )


def widgets_for(tool: str, doc: Any = None, state: Any = None) -> tuple[str, ...]:
    """Which option keys *tool*'s context bar shows, in drawn order.

    The three hiding rules the sidebar already applied, kept because each is
    the same argument: a control that cannot do what it says is worse than no
    control.

    * **A pixel nib has no hardness.** Coverage is 0 or 1 by definition, so
      there is no falloff to shape and a greyed slider would suggest there is
      one somewhere else.
    * **Pixel perfect is only offered on a pixel nib**, and never on the spray:
      the corner filter is about a *line*, and the canvas forces it off there.
    * **An indexed document has no soft nib**, so it has no hardness either.

    ``doc`` and ``state`` are optional; with neither, the answer is the tool's
    full set, which is what the coverage test wants.
    """

    from .inker.brush import ANGLED_NIBS, PIXEL_NIBS

    nib = "soft"
    if state is not None:
        nib = state.options_for(tool).get("nib", "soft")
    indexed = bool(getattr(doc, "is_indexed", False)) if doc is not None else False
    pixel = indexed or nib in PIXEL_NIBS
    keys = []
    for key, _label, tools, _group in CONTEXT_WIDGETS:
        if tool not in tools:
            continue
        if key == "hardness" and pixel:
            continue
        if key == "pixel_perfect" and not pixel:
            continue
        if key == "brush_angle" and nib not in ANGLED_NIBS:
            # Hidden rather than greyed, for hardness's reason: a turned disc
            # is a disc, so there is nothing here for the control to change and
            # a greyed slider would suggest there is one somewhere.
            continue
        keys.append(key)
    return tuple(keys)


def context_label(key: str) -> str:
    return next((label for name, label, _t, _g in CONTEXT_WIDGETS if name == key), key)


def context_group(key: str) -> str:
    return next((group for name, _l, _t, group in CONTEXT_WIDGETS if name == key), "")


#: Aseprite's five inks, and what each one *is* in this engine.
#:
#: ``mode`` is the ``brush.StrokeState`` mode it opens the stroke with and
#: ``lock`` forces the alpha lock on for the stroke. Written as a table rather
#: than a chain of ifs at the press for the reason every other table here is:
#: the labels, the engine modes and the tooltips cannot drift apart, and
#: "which inks are there" is a list rather than a function body.
#:
#: **Lock Alpha is an ink here and a layer flag as well**, and that is not a
#: duplicate: the flag is a property of the layer that outlives every tool, and
#: the ink is "for this stroke, paint inside what is already there". Aseprite
#: has both for the same reason.
INKS: tuple[tuple[str, str, str, bool, str], ...] = (
    (
        "simple",
        "Simple",
        "replace",
        False,
        "Writes the colour, alpha included, at the brush's own opacity -- so it "
        "can paint transparency down as well as up.",
    ),
    (
        "alpha",
        "Alpha",
        "paint",
        False,
        "Composites the colour over what is already there. The ordinary ink.",
    ),
    (
        "copy",
        "Copy",
        "copy",
        False,
        "The colour exactly, on every pixel the dab touches: no opacity, no "
        "antialiasing, no blend. What a pixel artist reaches for when the point "
        "is that only the chosen colours end up in the drawing.",
    ),
    (
        "lock_alpha",
        "Lock alpha",
        "paint",
        True,
        "Paints inside what this layer already has and never past its edge -- "
        "colours change, transparency does not.",
    ),
    (
        "shading",
        "Shading",
        "shade",
        False,
        "Moves each pixel one step along the selected ramp instead of painting "
        "a colour. The shading *tool* is where the ramp is chosen.",
    ),
)

INK_LABELS = [(key, label) for key, label, _mode, _lock, _hint in INKS]


def ink_mode(ink: str, fallback: str = "paint") -> str:
    """The engine mode an ink opens a stroke with."""

    return next((mode for key, _l, mode, _lock, _h in INKS if key == ink), fallback)


def ink_locks_alpha(ink: str) -> bool:
    return next((lock for key, _l, _m, lock, _h in INKS if key == ink), False)


def ink_hint(ink: str) -> str:
    return next((hint for key, _l, _m, _lock, hint in INKS if key == ink), "")


TOOL_OPTION_DEFAULTS: dict[str, Any] = {
    "brush_size": 12,
    "hardness": 0.85,
    "opacity": 1.0,
    "spacing": 0.1,
    "strength": 0.5,
    "shape_filled": False,
    #: How far a rectangle's corners are rounded, in pixels. Aseprite's
    #: hold-``C``-while-dragging control, and a per-tool option like every
    #: other shape setting so a rounded rectangle stays rounded next time.
    "corner_radius": 0,
    "wand_tolerance": 32,
    "wand_contiguous": True,
    #: Aseprite's pixel connectivity. Off is four-connected, which is what a
    #: fill has always been here; on, a region continues through a corner
    #: touch, which is what a diagonal pixel-art line asks for. Shared by the
    #: bucket and the wand because they share one predicate.
    "wand_eight": False,
    #: ``"canvas"`` (the composite, what you see) or ``"layer"`` (the active
    #: layer alone). Lineart over paint is the case the second exists for.
    "fill_refer": "canvas",
    #: Whether the bucket is confined to the grid cell it was clicked in. The
    #: *size* is the session's ``grid_size``; this is only the switch, because
    #: the grid is a property of the canvas and not of the tool.
    "fill_stop_grid": False,
    "sample_layer": False,
    "stabilise": 0.0,
    "speed_taper": 0.0,
    # Per tool for the same reason every other brush setting is: a pixel nib is
    # a property of the tool in your hand, and an eraser that stayed soft after
    # the pencil was made hard is exactly the mismatch this table exists to
    # stop -- a one-pixel line rubbed out with a feathered eraser leaves a
    # fringe of half-alpha nobody asked for.
    "nib": "soft",
    #: Which way an angled nib points, in degrees. Per tool like every other
    #: brush setting, and read only by the nibs an angle means anything to --
    #: a turned disc is a disc (``brush.ANGLED_NIBS``).
    "brush_angle": 0,
    "pixel_perfect": False,
    # Which ink this tool lays down; see :data:`INKS`. Per tool like everything
    # else here -- and offered on *every* painting tool now (6.1), which is
    # Aseprite's own arrangement: an ink is a property of the writing, not of
    # one tool.
    "paint_ink": "alpha",
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

# Built here rather than at its declaration: the sets it names are defined
# above ``TOOL_OPTION_DEFAULTS`` and the defaults are what it is checked
# against, so this is the first line at which both are true.
CONTEXT_WIDGETS = _context_table()


def _safe_int(value: Any, fallback: int, *, minimum: int) -> int:
    """``value`` as an int no smaller than ``minimum``, or ``fallback`` if it
    is not a number at all -- a hand-edited settings file's own doctrine,
    applied to a recorded export option the same way ``clamp_canvas`` applies
    it to a typed size."""
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return fallback


#: The nine export controls, and the value each starts at -- ``InkerState``'s
#: own field defaults, restated here as one dict because both a stored
#: settings block and a tab's own recorded set need to be applied *over* this
#: base rather than over whatever the controls currently hold, for
#: :meth:`InkerState.apply_preset`'s reason: a set saved before a key existed
#: must leave that key at its default, not at a stranger's leftover value.
EXPORT_OPTION_DEFAULTS: dict[str, Any] = {
    "arrange": None,
    # **What goes in the JSON sidecar** (6.9). Two switches rather than one,
    # because they answer two questions a pipeline asks separately: which
    # cell is which frame is always written (it is what a sheet *is*), and
    # these are the extras -- the tags an engine reads clips from, and the
    # slices a nine-patch or a hitbox lives in. Both default on, which is what
    # every export before this wrote. A third, "layer names for a compositor",
    # was declared here for a sidecar section nothing ever wrote and was
    # removed with it.
    "meta_tags": True,
    "meta_slices": True,
    "wrap": 4,
    "merge": False,
    "skip_empty": False,
    "trim": False,
    "padding": 0,
    "extrude": 0,
    "scale": 1,
    "template": "",
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
    # A zoom-ladder press waiting for a frame, +1 in or -1 out. Same reason
    # again, and a *direction* rather than a scale because the anchor decides
    # what the step means: over the canvas it holds the pixel under the cursor,
    # off it the middle of the pane, and neither is known to a key handler.
    pending_zoom_rung: int = 0
    # Display only, both of them: see ROTATIONS. ``rotation`` is clockwise on
    # screen in degrees; ``flipped`` mirrors left-to-right *after* it, which is
    # the order a physical sheet of paper does the two in.
    rotation: int = 0
    flipped: bool = False
    # Where the last paint stroke finished, for Shift-click's line. On the
    # *view* rather than on the session because it belongs to the drawing: a
    # tab switch and back should continue the line you were drawing, and a
    # session-wide field would carry one document's last point into another.
    last_paint: tuple[float, float] | None = None


def clamp_zoom(zoom: float, lo: float = MIN_ZOOM, hi: float = MAX_ZOOM) -> float:
    """Hold a zoom inside its bounds.

    The bounds are arguments rather than the module constants, because the
    three consumers of this view math want different ones -- see
    ``INKER_MIN_ZOOM``. Defaulting to the globals is what keeps Plotter's and
    Packwright's call sites unchanged.
    """

    return max(lo, min(hi, float(zoom)))


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
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    zoom: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Set the zoom and centre the oriented canvas in the region."""
    (lo_x, lo_y), (hi_x, hi_y) = view_extent(view, size)
    view.zoom = clamp_zoom(zoom, lo, hi)
    view.pan = (
        (region[0] - (hi_x - lo_x) * view.zoom) * 0.5 - lo_x * view.zoom,
        (region[1] - (hi_y - lo_y) * view.zoom) * 0.5 - lo_y * view.zoom,
    )
    view.fitted = True


def fit(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Scale to show the whole document, centred.

    Under a floor (``lo``) a document too large to fit is centred at the floor
    and overflows the pane. That is the stated cost of having a floor at all;
    the alternative is a "fit" that is not one, which is worse in the case the
    floor exists for.
    """
    (lo_x, lo_y), (hi_x, hi_y) = view_extent(view, size)
    zoom = min(region[0] / max(hi_x - lo_x, 1.0), region[1] / max(hi_y - lo_y, 1.0))
    _place(view, size, region, zoom, lo=lo, hi=hi)


def centre(
    view: PaintView,
    size: tuple[int, int],
    region: tuple[float, float],
    zoom: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Set an explicit zoom and re-centre -- what Ctrl+1 (100%) does."""
    _place(view, size, region, zoom, lo=lo, hi=hi)


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


def _anchor(
    view: PaintView, origin: tuple[float, float], mouse: tuple[float, float], after: float
) -> None:
    """Move to ``after``, keeping whatever pixel is under the cursor there.

    Factored out rather than written twice: :func:`zoom_about` and
    :func:`zoom_step` differ only in how they pick the new zoom, and a second
    copy of this pan correction is the classic way for one of the two routes
    into the same view to start drifting.
    """
    before = view.zoom
    if after == before:
        return
    local = (mouse[0] - origin[0], mouse[1] - origin[1])
    ratio = after / before
    view.zoom = after
    view.pan = (
        local[0] - (local[0] - view.pan[0]) * ratio,
        local[1] - (local[1] - view.pan[1]) * ratio,
    )


def zoom_about(
    view: PaintView,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    steps: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """Zoom keeping whatever pixel is under the cursor under the cursor.

    Multiplicative: a fixed *ratio* per step, which is the right shape for a
    keyboard zoom spanning three orders of magnitude. The wheel uses
    :func:`zoom_step` instead.
    """
    _anchor(view, origin, mouse, clamp_zoom(view.zoom * (ZOOM_STEP**steps), lo, hi))


def zoom_step(
    view: PaintView,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    notches: float,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """One wheel notch: +-``ZOOM_PERCENT_STEP`` percent, snapped to the grid.

    Snapped *first*, so a zoom arrived at by fitting (an arbitrary 83.4%) joins
    the lattice on the first notch instead of carrying its fraction forever --
    which is what makes the status bar read 85, 90, 95 rather than 88.4, 93.4.
    The rounding is what a user coming from any paint program expects and what
    the multiplicative ratio cannot give: 100% is reachable from either side.
    """
    percent = round(view.zoom * 100 / ZOOM_PERCENT_STEP) * ZOOM_PERCENT_STEP
    percent += ZOOM_PERCENT_STEP * notches
    _anchor(view, origin, mouse, clamp_zoom(percent / 100.0, lo, hi))


def zoom_ladder_step(
    view: PaintView,
    origin: tuple[float, float],
    mouse: tuple[float, float],
    direction: int,
    *,
    lo: float = MIN_ZOOM,
    hi: float = MAX_ZOOM,
) -> None:
    """One press of zoom in or out: the next whole scale, cursor held.

    Through the same ``_anchor`` the wheel and ``zoom_about`` use, for the
    reason that helper exists at all -- a third copy of the pan correction is a
    third route into the view that can start to drift from the other two.
    """
    _anchor(view, origin, mouse, clamp_zoom(zoom_rung(view.zoom, direction), lo, hi))


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
    file_format: str = "png"  # ora | png | aseprite
    uid: str = field(default_factory=lambda: f"pd{next(_uids)}")
    view: PaintView = field(default_factory=PaintView)
    # The history position the file on disk was written from. Dirty is a
    # *comparison*, not a flag, so undoing back to the saved state correctly
    # stops being dirty -- which the document's revision cannot express,
    # because it counts changes and an undo is one.
    saved_head: int = 0
    saving: bool = False
    # ``doc.history.trimmed`` as of the last time the user was told about it.
    # The history drops its oldest steps when they get too big to hold (see
    # ``studio.undo.UNDO_HARD_BYTES``), and a rotate on a large document can
    # take most of the stack with it in one press -- so the undo the user
    # reaches for a minute later is simply not there any more, with nothing
    # having said so. Compared with ``!=`` rather than ``>``: ``clear`` puts
    # the stack's counter back to zero along with the history.
    trim_seen: int = 0
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
    #: Play every frame at this many per second, or 0 for the durations the
    #: document stores. **A preview setting**: it changes nothing about the
    #: frames, which is what an animator asking "what does this look like at
    #: 12 fps" means -- the alternative is an undoable edit to every frame.
    #: Per tab, because it is a question about *this* clip.
    constant_rate: int = 0

    # Which layer groups are folded shut in the panel, by group uid. **View
    # state, and deliberately on the tab rather than on the document**: whether
    # a folder is open says nothing about the picture, so it is neither
    # persisted into the ``.ora`` nor undoable -- for the reason the playhead is
    # neither. A document that asked to be saved because somebody collapsed a
    # folder would make ``dirty`` a lie.
    collapsed_groups: set[int] = field(default_factory=set)

    # This tab's own last export: where it was written and the option set
    # (``InkerState.export_options_snapshot``) it was written with. **Session-
    # scoped and journal-exempt**, like ``playing``/``range_sel`` above --
    # neither is read by ``journal.head_of`` (``tab.doc.history.head`` alone)
    # or by ``dirty`` (``doc.history.head`` against ``saved_head``), so
    # recording an export mid-session neither dirties the document nor wakes
    # the crash-recovery copy. ``None``/empty until the first successful
    # export of this tab; :meth:`InkerState.apply_export_options` treats that
    # as "nothing to suggest" rather than "suggest nothing".
    export_dest: Path | None = None
    #: Which export wrote ``export_dest``; see ``inker_mode.REPEATABLE``.
    #: Session-scoped like the destination beside it, and set only when an
    #: export *lands* -- a cancelled file dialog is not an export to repeat.
    export_kind: str = ""
    export_options: dict[str, Any] = field(default_factory=dict)

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



#: How long a status-bar tip stays up, in seconds. Long enough to read a
#: sentence and reach for the remedy, short enough that a stale one is never
#: mistaken for a description of the state you are in now.
TIP_SECONDS = 6.0


@dataclass
class Tip:
    """One sentence under the canvas, answering a gesture that did not land.

    The distinction this exists to draw, and the one the toasts had lost:
    **a tip answers a gesture; a toast reports a job.** "This layer is locked"
    is about the click you just made, and belongs where your eye already is --
    under the cursor, in the status bar, gone in a moment. "Exported to
    sprites/hero.png" is about work that finished while you were looking
    somewhere else, and has to survive not being looked at.

    ``remedy`` is an **op name, never a callable**. A tip that offers a button
    the menu and the keyboard do not have is a second, invisible command
    surface; naming the op means the offer cannot outlive what it offers, and
    the button can show the op's own key.
    """

    text: str
    remedy: str = ""
    remedy_label: str = ""
    at: float = field(default_factory=time.monotonic)

    def alive(self, now: float | None = None) -> bool:
        """Whether it should still be drawn."""

        return (time.monotonic() if now is None else now) - self.at < TIP_SECONDS


@dataclass
class InkerState:
    docs: list[InkerDoc] = field(default_factory=list)
    active_uid: str = ""
    #: The most recent status-bar tip, or None; see :class:`Tip`. Not
    #: persisted and not per document -- it is about the gesture just made.
    tip: Tip | None = None
    #: The popup a menu row has asked a pane to open, by name, or "". See
    #: ``inker_ops``' module docstring: a popup belongs to the window that
    #: began it, so the registry can only ask.
    pending_dialog: str = ""
    #: **Same ink in all tools** -- Aseprite's own checkbox. Off by default,
    #: because the per-tool table is this app's rule and an app-level override
    #: of it should be something the user asked for. When on, every painting
    #: tool reads (and writes) the brush's ink.
    ink_shared: bool = False
    #: The group whose flyout strip is open, or "". Held only while the mouse
    #: is down on a group button -- it *is* the capture, in Aseprite's gesture
    #: where mouse-down selects, opens and captures in one event.
    flyout: str = ""
    #: Which member of each group the toolbox shows, keyed by group. Written by
    #: sliding onto a member, so the rail keeps showing the tool you use.
    group_tool: dict[str, str] = field(default_factory=dict)
    #: The parameterised op whose dialog is open, by name, or "". Held by
    #: name rather than as the ``Op`` for the same reason: the popup survives
    #: across frames and the registry owns the object.
    pending_op: str = ""
    #: The last values each parameterised op was run with, so the common case
    #: is two clicks rather than two clicks and a number.
    op_params: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Target-scoped many-to-many shortcut overrides.  Values stay as plain
    #: JSON-shaped records so this state module does not import ``inker_ops``
    #: (the command registry already imports the context table above).
    #: Presence with an empty list means deliberately unbound; an absent target
    #: inherits the Aseprite 1.3.15.5 Windows defaults.
    shortcut_overrides: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    shortcut_query: str = ""
    shortcut_target: str = ""
    shortcut_draft: str = ""
    shortcut_context: str = ""
    shortcut_trigger: str = "press"
    #: Previous tool and physical modifier while an Aseprite-style quick tool
    #: is held.  Key-up restores this even if the active context changed during
    #: the drag, which is why restoration cannot be another context lookup.
    quick_tool: str = ""
    quick_key: str = ""
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
    #: **The pixel grid** (6.8): one line per source pixel, drawn only past
    #: the zoom where the lines would not be most of what is on screen.
    #: Separate from ``grid``, which is the *tile* grid at ``grid_size`` -- two
    #: different questions, and Aseprite has both.
    pixel_grid: bool = False
    #: Outline the active layer's own bounds, so what is being drawn on is
    #: legible on a document where every layer is transparent at the edges.
    layer_edges: bool = False
    #: Draw each cell's local tile id on a tilemap layer. A reading aid for
    #: authoring a tileset, and off by default because it covers the art.
    tile_numbers: bool = False
    #: Which harmony the colour panel shows; see ``inker.indexed.HARMONIES``.
    #: App-level like every other tool setting, and not persisted: it is a way
    #: of *looking* at the colour in hand rather than a preference.
    harmony: str = "complement"
    #: Nine numbered custom brushes, by slot. Aseprite's ``Alt+1..9``.
    #:
    #: **Not persisted, for ``stamp``'s reason**: a tip is pixels rather than a
    #: setting, and a quarter of a megabyte of base64 in the settings block
    #: would be paid for on every swatch click. They live as long as the
    #: session, which is as long as the drawing they were cut out of is open.
    stamp_slots: dict[int, Any] = field(default_factory=dict)
    #: Named bundles of one tool's options; see :meth:`save_preset`. Persisted
    #: beside the swatches by ``inker_mode.persist``.
    presets: dict[str, dict[str, Any]] = field(default_factory=dict)
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
    # The Image size dialog's two modes, and the Canvas size dialog's one. All
    # three sit here beside ``resample`` and for its reason: they are statements
    # about the kind of work being done rather than facts about a document, and
    # Photoshop and GIMP both carry them from one document to the next.
    #
    # Deliberately **not persisted** -- ``_restore_canvas`` is untouched. They
    # are cheap to re-set and a remembered "percent" would make the first field
    # a returning user reads mean something other than what it says.
    #: "pixels" | "percent" -- what the Image size fields are measured in.
    scale_units: str = "pixels"
    #: Whether Image size holds the document's aspect ratio.
    scale_linked: bool = True
    #: Whether Canvas size's fields are a delta rather than an absolute size.
    canvas_relative: bool = False
    grid: bool = False
    # 32 by default: the most common tile and sprite cell in the corpus this
    # app feeds, and what the user asked the grid to assume. Changeable in the
    # tools pane's Canvas section, and persisted with the rest of the block.
    grid_size: int = 32
    # Whether a shape, a marquee or a line snaps to the grid. Deliberately not
    # applied to freehand strokes: quantising a brush to a 16-pixel lattice is
    # not a drawing aid, it is a different tool.
    grid_snap: bool = False
    # The pixel rulers along the canvas's top and left edges. On by default --
    # they exist to give a sense of size, which is exactly what a first-run
    # user lacks -- and metric in the decimal sense: tick steps come off the
    # 1/2/5 ladder, never inches or twelfths.
    rulers: bool = True

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
    #: Ghost only the *active* track's drawing rather than the whole frame's
    #: composite, which is what an animator inking one layer over a static
    #: background actually wants to see. App-level and unpersisted like the
    #: other four: onion is a property of the canvas, not of a tool.
    onion_current_layer: bool = False
    #: The two ghost tints, as 0xRRGGBB (6.7). State rather than the two
    #: constants they were, because which colours read as "before" and "after"
    #: depends on the drawing: red and green over a red-and-green sprite is two
    #: ghosts nobody can tell from the art.
    onion_tint_back: int = 0xE05050
    onion_tint_forward: int = 0x50C060
    #: How the ghosts fade with distance. ``1.0`` is the ``alpha / offset``
    #: falloff this always had; a higher power drops the far ones away faster,
    #: which is what a twelve-frame onion needs to stay readable, and ``0``
    #: makes every ghost the same strength for tracing a cycle.
    onion_falloff: float = 1.0
    #: Draw the ghosts **over** the live frame rather than under it. Aseprite's
    #: own switch: under is right for drawing the next pose and over is right
    #: for checking one you have just drawn against the last.
    onion_in_front: bool = False
    #: Stop at a tag's ends rather than at the document's (6.7). What an
    #: animator working inside a walk cycle means by "the frame before this
    #: one" is the tag's last frame, not the previous clip's.
    onion_wrap_tag: bool = False
    fg: tuple[int, int, int, int] = (0, 0, 0, 255)
    #: Which palette **slot** the foreground came from, or None when it came
    #: from the wheel, the eyedropper or the session swatch row.
    #:
    #: It only matters in a truly indexed document, where it becomes
    #: ``Document.paint_slot`` and decides which of two identical swatches a
    #: stroke lands in. A colour is not enough to answer that -- which is the
    #: whole reason an index plane exists -- so the answer has to be carried
    #: from the click that chose it, and it has to be **cleared** by every other
    #: way of choosing a colour or the brush would keep claiming a slot the user
    #: has moved away from. :meth:`set_fg` is the one door, for that reason.
    fg_slot: int | None = None
    bg: tuple[int, int, int, int] = (255, 255, 255, 255)
    #: Which of the two colours the picker pane's sliders edit -- ``"fg"`` or
    #: ``"bg"``. Session state rather than a setting: it is where you happen to
    #: be looking, not how you like the app, and a picker that reopened on the
    #: background after a restart would be a control that lies about which
    #: colour a drag is about to change.
    picker_target: str = "fg"
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

    # -- tiles, all of it view state ---------------------------------------
    #
    # The tilesets and the refs live on the *document*; what is here is the
    # brush -- which tileset, which tile out of it, and how it is turned. The
    # same split the palette block below makes, for the same reason: none of
    # this is picture data and none of it may push an undo step.
    #
    # ``tileset_uid`` is 0 for "whatever the active layer is bound to", which
    # is what a user reaching for the panel means every time but the one where
    # they are looking at another tileset on purpose. A stale uid is tolerated
    # rather than policed, ``slice_uid``'s rule: an undone add leaves it naming
    # a tileset the document does not have, the picker falls back to the bound
    # one, and hunting the value down on every history move would be a second
    # place for it to be wrong.
    tileset_uid: int = 0
    #: The local id the stamp puts down. 1 rather than 0 because local id 0 is
    #: the required-blank tile -- a real, useful choice (it is the tile eraser),
    #: but not the one a panel should open on.
    tile_local: int = 1
    #: The three flag bits, as three toggles. Held apart rather than as one
    #: encoded word so the panel's checkboxes are the state rather than a view
    #: of it; ``tile_gid`` is the one place they are folded together.
    tile_flip_h: bool = False
    tile_flip_v: bool = False
    tile_flip_d: bool = False
    #: The last cell a drag stamped, so a drag across a cell's interior costs
    #: one write and not one per mouse-move. Cleared with the drag.
    tile_cell: tuple[int, int] | None = None
    #: ``doc.history.head`` at the press of a paint gesture on a tilemap cel,
    #: or None. What the Manual-mode revert toast is derived from: a commit
    #: that reverted pushed no step, so the head is where it was.
    tile_head: int | None = None

    # -- indexed colour, all of it view state ------------------------------
    #
    # The palette itself lives on the *document* (``Document.palette``), which
    # is the only place it can live: it is saved with the file and it decides
    # what every write snaps to. What is here is which slot the user has
    # selected and the last usage count they asked for -- neither is picture
    # data, and neither may push an undo step.
    palette_slot: int = 0
    # ``(tab uid, document rev the count was taken at, per-slot counts)``. Asked
    # for rather than recomputed: counting is a walk over every pixel of every
    # cel, so doing it per frame would cost a 40-frame clip's worth of scanning
    # sixty times a second to keep a number that changes on one dab.
    #
    # The uid is in the key because this state is shared by every tab: two open
    # documents at the same ``rev`` with palettes of the same length used to
    # answer each other's counts, and "0 px, safe to delete" is the one thing
    # this number must never say wrongly.
    palette_usage: tuple[str, int, list[int]] | None = None
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
    #: Which palette the folder browser has selected, by stem. View state and
    #: nothing else: the folder is the truth, and a name remembered here that
    #: has since been deleted is answered by the browser drawing the first name
    #: it *does* have rather than by anything stored being wrong.
    palette_pick: str = ""
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
    #: Which question the open conversion session is asking. ``""`` is the snap
    #: this popup has always been -- adopt a table, rewrite the pixels, stay in
    #: RGB -- and ``"indexed"`` is the *mode* change the "Colour mode..." row
    #: that opens it has always been named after. One popup, because the
    #: controls, the preview and the refusals are the same either way; the
    #: difference is one call at Apply.
    convert_mode: str = ""
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
    # run with, and which tab owns the popup. Remembered per filter for the
    # reason Clay's op parameters are -- somebody applying the same levels to
    # six layers should not retype it six times -- and ``filter_uid`` is what
    # notices imgui closing the popup on a click outside, which is a cancel.
    filter_name: str = ""
    # ``Any`` and not ``float``: the FX staples brought colours (an RGBA tuple)
    # and a choice (a string) into the same per-filter bag.
    filter_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: **A tab uid and not a bool**, for the reason ``convert_uid`` above is one
    #: -- this is the popup that one was cloned from, and it kept the bug the
    #: clone was written to fix. A filter session lives on one ``Document``
    #: (``Document._filter``) while this state is shared by every tab and the
    #: pane draws whichever tab is *active*, so a plain "the popup is up" flag
    #: meant a tab switch previewed into the wrong document and left the right
    #: one holding pixels nobody would ever answer for -- pixels a save then
    #: wrote to disk and called clean. See ``inker_mode.end_filter_session``.
    #: Empty means no session.
    filter_uid: str = ""
    #: A regeneration in flight: the job id the bridge polls, the tab and layer
    #: it lands in, the box and the selection weight it was asked with. One at
    #: a time; the popup refuses a second while one is pending.
    inpaint_pending: dict[str, Any] | None = None
    #: The prompt typed into the last Regenerate popup, kept across uses.
    inpaint_prompt: str = ""
    inpaint_strength: float = 0.6

    # Free transform is a *state*, not a tool: it takes over the canvas until
    # it is committed or cancelled, and every other tool is unavailable while
    # it is on -- which is exactly what "modal" means and why it cannot live in
    # the tool list beside brush and fill.
    transforming: bool = False
    # **Which tab owns the transform** -- ``convert_uid``'s pattern, for its
    # exact bug shape: the modal lives on one ``Document``, this state object
    # is shared by every tab, and the panes draw whichever tab is in front. A
    # bare bool meant Enter after a mid-transform tab switch committed the
    # *new* tab's floating buffer and stranded the owner's lifted pixels.
    # Empty means no transform; see ``_settle_transform``.
    transform_uid: str = ""
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
    #: The state a drag down the eye column is painting, or None. See
    #: ``inker_timeline._drag_toggle``: the value is the one the *first* row
    #: took, so the drag paints rather than flipping each row it crosses.
    eye_drag: bool | None = None
    #: Which rows that drag has already written, and what each one held before
    #: it did. The gesture mutates live so the column follows the cursor, and
    #: asks for its undo step once, on release -- ``set_layers_props``' ``was``
    #: pre-image, for the reason ``header_controls`` needs one on the opacity
    #: drag: by release the rows already hold the new value, so reading
    #: "before" off them would compare a value against itself.
    eye_drag_was: dict[int, dict] = field(default_factory=dict)
    # Nearest-neighbour multiplier applied to sheet, GIF and PNG exports.
    export_scale: int = 1
    # How a sheet export packs its cells: None is the plain row-wrap this
    # always used; "horizontal"/"vertical" force one row/column; "rows"/
    # "columns" fix that side's count from ``export_wrap``. App-level like
    # ``export_scale`` -- remembered across documents, unused by GIF and PNG
    # exports, and ignored (the export refuses the combination by name)
    # whenever the document itself carries a directional layout.
    export_arrange: str | None = None
    #: The two JSON meta switches (6.9); see ``EXPORT_OPTION_DEFAULTS``.
    export_meta_tags: bool = True
    export_meta_slices: bool = True
    # The N for ``export_arrange in ("rows", "columns")``. Kept even while
    # ``export_arrange`` names neither, so switching back to Rows/Columns does
    # not forget what the user last typed.
    export_wrap: int = 4
    # Duplicate flattened frames (byte-identical, which a linked cel is for
    # free) share one cell instead of one each. Off by default -- the byte pin
    # over the whole default sheet export needs the untouched cell-per-frame
    # path, and a sheet an engine slices by fixed geometry would otherwise
    # break if a repeat silently vanished. Ignored, like ``export_arrange``,
    # whenever the document carries a directional layout: that grid's cells
    # are poses by yaws, not frames, so there is nothing here to merge.
    export_merge: bool = False
    # A fully-transparent flattened frame gets no cell at all. Off by default
    # for the same reason ``export_merge`` is, and refused together with a
    # directional layout for the same reason too.
    export_skip_empty: bool = False
    # Each cell shrinks to the largest trimmed frame's size instead of the
    # full canvas, every frame's own trimmed pixels placed flush at the
    # cell's corner. Off by default for the same reason ``export_merge`` is --
    # the byte pin over the whole default sheet export needs the untouched
    # full-cell path.
    export_trim: bool = False
    # A uniform border round the atlas and gutter between every cell, in
    # pixels; app-level like the switches above it. Zero is the sheet this
    # always packed.
    export_padding: int = 0
    # Replicates each placed rectangle's own border pixels outward into
    # whatever gutter ``export_padding`` left, so a filtered texture sampling
    # just past a sprite's edge finds that sprite's own colour. Refused at
    # export (by name, before the file dialog) whenever it exceeds half of
    # ``export_padding`` -- the same room guarantee ``packwright.PackSettings``
    # enforces at construction.
    export_extrude: int = 0
    # A user-chosen filename template, or "" for whichever default the export
    # in question falls back to -- ``sheetout.DEFAULT_FRAME_TEMPLATE`` for a
    # plain PNG sequence, ``DEFAULT_TAG_TEMPLATE``/``DEFAULT_LAYER_TEMPLATE``
    # for a split. App-level like every export control above it, and the
    # ninth key ``export_options_snapshot``/``apply_export_options`` carry.
    export_template: str = ""
    # Which tab's own recorded ``export_options`` the controls above were last
    # seeded from -- ``""`` seeds nothing. Compared by uid, not by identity,
    # against ``InkerDoc.uid``: a tab closed and never reopened cannot match
    # again, which is exactly "forget it" without a second field to clear.
    # Session-only bookkeeping, not a document fact and not persisted: it
    # exists so exporting the *same* tab twice in a row keeps whatever the
    # user just typed into the controls, and only a switch away and back
    # re-suggests that tab's own last settings over it.
    export_seed_uid: str = ""

    # -- importing a sprite sheet ------------------------------------------
    #
    # ``sheet_import`` is ``(atlas array, title)`` once a file has been chosen
    # and decoded off-thread, and None the rest of the time. The picture is
    # read *before* the grid is asked for, so the popup can say how many frames
    # the numbers actually produce -- which is the one thing that stops a
    # mistyped cell size becoming a document the user edits for ten minutes
    # before noticing. ``sheet_import_open`` is what notices imgui closing the
    # popup on a click outside, the ``filter_uid`` idiom -- as a bare bool,
    # because this popup holds nothing on any document to settle.
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
    #: Which tab's press opened the text popup, ``filter_uid``'s shape and for
    #: its reason. ``text_at`` is a point in *that* document's pixels and the
    #: popup is drawn for whichever tab is in front, so without this an OK after
    #: a tab switch stamped into the wrong document at the right document's
    #: coordinates -- with the antialias default decided from the other one's
    #: palette. Empty means no popup.
    text_uid: str = ""
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
    wand_eight = _tool_option("wand_eight")
    fill_refer = _tool_option("fill_refer")
    fill_stop_grid = _tool_option("fill_stop_grid")
    sample_layer = _tool_option("sample_layer")
    stabilise = _tool_option("stabilise")
    speed_taper = _tool_option("speed_taper")
    nib = _tool_option("nib")
    pixel_perfect = _tool_option("pixel_perfect")
    paint_ink = _tool_option("paint_ink")
    brush_angle = _tool_option("brush_angle")
    corner_radius = _tool_option("corner_radius")

    @property
    def ink(self) -> str:
        """The ink this stroke writes with, honouring "same in all tools"."""

        tool = "brush" if self.ink_shared else self.tool
        stored = self.options_for(tool).get("paint_ink", "alpha")
        # A settings file written before 6.1 carries the old two-value spelling.
        # Read forward rather than migrated: the value is per tool per profile,
        # and a migration would have to walk a dict of dicts to change a string
        # this function can answer for.
        return {"blend": "alpha", "replace": "simple"}.get(stored, stored)

    def set_ink(self, ink: str) -> None:
        tool = "brush" if self.ink_shared else self.tool
        self.options_for(tool)["paint_ink"] = ink
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

    def pattern_for(self, tool: str) -> Any:
        """The tip *this tool* would fill with right now, or None.

        :meth:`tip_for`'s sibling for :data:`PATTERN_TOOLS`, and separate from
        it on purpose: ``tip_for`` is also what the canvas draws the brush
        outline from, and a bucket that answered it would put a tip-shaped ring
        under a cursor that stamps nothing. Both read the same two things -- the
        one captured tip and this tool's own ``use_stamp`` -- so the panel's
        checkbox and the click cannot disagree.
        """
        if self.stamp is None or tool not in PATTERN_TOOLS:
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

    # -- export options ------------------------------------------------------
    #
    # The nine export controls (scale, arrange, wrap, merge, skip empty, trim,
    # padding, extrude, template) are app-level like the swatches -- one set,
    # shared by whichever tab is exporting -- but a *document* still has its
    # own preference: the walk cycle wants Merge on and the icon sheet does
    # not. ``InkerDoc.export_options`` is that per-tab memory (session-only,
    # journal-exempt); these two methods are its one door, matching and
    # inverting each other the way ``save_preset``/``apply_preset`` do.

    def export_options_snapshot(self) -> dict[str, Any]:
        """The nine export controls' current values, as one dict.

        One spelling for what a completed export records onto its tab
        (:func:`inker_mode.on_task_done`) and what ``inker_mode.persist``
        writes cross-session, so the two destinations cannot drift apart by a
        stray key.
        """
        return {
            "arrange": self.export_arrange,
            "meta_tags": bool(self.export_meta_tags),
            "meta_slices": bool(self.export_meta_slices),
            "wrap": int(self.export_wrap),
            "merge": bool(self.export_merge),
            "skip_empty": bool(self.export_skip_empty),
            "trim": bool(self.export_trim),
            "padding": int(self.export_padding),
            "extrude": int(self.export_extrude),
            "scale": int(self.export_scale),
            "template": str(self.export_template),
        }

    def apply_export_options(self, options: Any) -> None:
        """Replace the nine export controls from a recorded set.

        Over :data:`EXPORT_OPTION_DEFAULTS`, never over whatever the controls
        currently hold -- :meth:`apply_preset`'s reason, restated: a set saved
        before a key existed must leave that key at its default rather than at
        a stranger's value, or applying the same recorded set twice gives two
        different exports.

        A falsy ``options`` -- a tab that has never exported, ``InkerDoc``'s
        own default -- is a no-op: the shared controls keep whatever they
        already carry, which is either the cross-session default
        (``inker_mode._restore_export``) or the previous tab's leftover
        setting, and there is nothing here to correct that with.
        """
        if not options or not isinstance(options, dict):
            return
        merged = {
            **EXPORT_OPTION_DEFAULTS,
            **{k: v for k, v in options.items() if k in EXPORT_OPTION_DEFAULTS},
        }
        arrange = merged["arrange"]
        self.export_arrange = arrange if arrange is None or isinstance(arrange, str) else None
        # The meta switches were recorded by the snapshot and never read back
        # here, so a tab switch or a restart put them back to the defaults
        # while the rest of the recorded set was honoured.
        self.export_meta_tags = bool(merged["meta_tags"])
        self.export_meta_slices = bool(merged["meta_slices"])
        self.export_wrap = _safe_int(merged["wrap"], self.export_wrap, minimum=1)
        self.export_merge = bool(merged["merge"])
        self.export_skip_empty = bool(merged["skip_empty"])
        self.export_trim = bool(merged["trim"])
        self.export_padding = _safe_int(merged["padding"], self.export_padding, minimum=0)
        self.export_extrude = _safe_int(merged["extrude"], self.export_extrude, minimum=0)
        self.export_scale = _safe_int(merged["scale"], self.export_scale, minimum=1)
        self.export_template = str(merged.get("template") or "")

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
        self._settle_transform()
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
        self._settle_transform()
        self.clear_drag()
        return True

    def activate(self, uid: str) -> None:
        if uid != self.active_uid:
            self.active_uid = uid
            self._settle_transform()
            self.clear_drag()

    def _settle_transform(self) -> None:
        """Cancel an open free transform the moment its owner stops being the
        active tab.

        The transform is modal on one document, and Enter/Escape are answered
        against whichever tab is in front -- so a transform left open across a
        switch would commit or cancel the *wrong* tab's floating buffer and
        strand the owner's lifted pixels. Cancelled rather than committed, for
        ``end_convert_session``'s reason: an unanswered modal is not a yes,
        and the cancel only puts back what the lift itself cut. An owner that
        was just closed has nothing left to put back; the flags still clear.
        """
        if not self.transforming or self.transform_uid == self.active_uid:
            return
        owner = self.get(self.transform_uid)
        if owner is not None:
            owner.doc.cancel_floating()
        self.transforming = False
        self.transform_uid = ""
        self.transform_ref = None
        self.transform_grab = ""

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
        # Space-to-pan is a hold, and its release can be *dropped* rather than
        # merely late: ``main`` gates both key edges on ``_passes_text_field``,
        # which answers no for a plain Space, so a release that arrives while a
        # text field has focus never reaches ``handle_key`` and the flag stays
        # on -- every left-drag panning instead of painting. Cleared here
        # because every tab switch, add and close comes through this method, so
        # it is the one place a latched flag is certain to be let go of.
        self.space_held = False
        self.drag_anchor = None
        self.last_point = None
        self.lasso = []
        self.transform_ref = None
        self.slice_drag = None
        self.drag_button = 0
        self.tile_offset = (0.0, 0.0)
        # The stamp's per-drag memory, and the head a Manual-mode revert is
        # measured against. Both belong to one gesture and neither may survive
        # a tab switch, which is what brings every other field here.
        self.tile_cell = None
        self.tile_head = None
        self.spray_carry = 0.0
        self.transform_grab = ""
        # A half-typed tag rename goes with the gesture state: the index is
        # into the *active* document's tag list, so surviving a tab switch --
        # every switch comes through here -- would leave the rename box open
        # over some other document's tag of the same number.
        self.tag_editing = -1
        self.tag_name = ""
        # The three remaining pane-owned gestures, each of which outlived a tab
        # switch and then acted on the wrong document:
        #
        # ``timeline_anchor`` is an index pair into the *active* tab's grid and
        # ``range_sel`` lives on ``InkerDoc``, so clicking a cell in one tab and
        # Shift+clicking in another built the second tab's range from the
        # first's coordinates. It needs no keyboard trick at all, which makes it
        # the most reachable of the three.
        #
        # ``eye_drag``/``eye_drag_was`` is a visibility drag whose pre-image is
        # keyed by row index. Ctrl+Tab is a modifier chord and passes the
        # text-field gate, so a drag could carry on over another tab's rows and
        # then push one ``set_layers_props`` with a pre-image belonging partly
        # to each -- an undo that restores the wrong values.
        #
        # ``text_at`` is dropped with them; the popup that reads it is closed by
        # ``text_uid`` no longer naming the tab in front.
        self.timeline_anchor = None
        self.eye_drag = None
        self.eye_drag_was = {}
        self.text_at = (0, 0)
        self.text_uid = ""
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

    def store_stamp(self, slot: int) -> bool:
        """Put the captured tip into a numbered slot. -> whether there was one.

        The slots are the tips a drawing keeps coming back to -- a hatch, a
        leaf, a stipple -- and the reason they are numbered rather than named
        is the same reason Plotter's stamps are: recall happens hundreds of
        times a session and storing nine times.
        """

        if self.stamp is None:
            return False
        self.stamp_slots[int(slot)] = self.stamp
        return True

    def recall_stamp(self, slot: int) -> bool:
        """Take a numbered brush back into the hand, and switch the tip on.

        Switching ``use_stamp`` on with it is what makes the gesture one step:
        a tip loaded on a tool that is not set to use one is a recall that
        appears to have done nothing.
        """

        stamp = self.stamp_slots.get(int(slot))
        if stamp is None:
            return False
        self.stamp = stamp
        self.options_for(self.tool)["use_stamp"] = True
        return True

    def say(self, text: str, *, remedy: str = "", remedy_label: str = "") -> None:
        """Put a tip under the canvas. The refusal door for a gesture.

        ``remedy`` names an op in :mod:`~warlock.studio.inker_ops`; see
        :class:`Tip` for why it is a name and not a function.
        """

        self.tip = Tip(text, remedy=remedy, remedy_label=remedy_label)

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

    # -- tiles --------------------------------------------------------------

    def tile_gid(self) -> int:
        """The stamp's picked tile as one encoded ref word.

        The single place the three flip toggles are folded into the gid the
        engine takes, so the panel's checkboxes and what a press actually
        writes cannot come to disagree. Local id 0 keeps its flags stripped:
        every orientation of blank is blank, and a flagged zero would read back
        out of an ORA as a different word for the same picture.
        """
        local = max(0, int(self.tile_local)) & gid.GID_MASK
        if local == 0:
            return 0
        flags = 0
        if self.tile_flip_h:
            flags |= gid.FLIP_H
        if self.tile_flip_v:
            flags |= gid.FLIP_V
        if self.tile_flip_d:
            flags |= gid.FLIP_D
        return local | flags

    def clamp_tile_pick(self, uid: int, tile_count: int) -> None:
        """Keep the picked tile inside the tileset it is a local id into.

        Two ways the pick goes stale, and both land here because both have the
        same answer -- go back to the first real tile:

        * **The tileset changed.** A local id means nothing next to a different
          atlas, so carrying 12 across from a 40-tile tileset to a 3-tile one
          is not a preserved choice, it is a wrong one.
        * **The tileset shrank under it.** Undoing a Stack-mode append is
          exactly this: the tile the user is holding stops existing.

        Tile 1 rather than 0 wherever there is one, for the field's own reason:
        0 is the required blank and a real choice (it is the tile eraser), but
        not the one a panel should land on. The engine refuses an over-range id
        at the door regardless (``place_tiles``); this is what stops the user
        ever meeting that refusal.
        """
        tile_count = max(1, int(tile_count))
        if uid == self.tileset_uid and 0 <= self.tile_local < tile_count:
            return
        self.tileset_uid = int(uid)
        self.tile_local = min(1, tile_count - 1)

    def picked_tileset(self, doc: Any) -> int | None:
        """Which tileset the panel shows and the stamp writes into.

        **The active layer's binding wins whenever there is one.** A stamp
        writes a *local* id into the layer's own tileset, so a panel showing a
        different atlas while a tilemap layer is selected would hand the engine
        an id naming a completely different tile -- a mismatch with no honest
        recovery, and the reason this is derived rather than remembered. The
        remembered ``tileset_uid`` only decides when the active row is bound to
        nothing, which is the browsing case: looking at a tileset to export it
        or to send it to Plotter.
        """
        uids = [slot.uid for slot in getattr(doc, "tilesets", [])]
        if not uids:
            return None
        probe = getattr(doc, "active_tileset_uid", None)
        bound = probe() if probe is not None else None
        if bound in uids:
            return bound
        if self.tileset_uid in uids:
            return self.tileset_uid
        return uids[0]

    # -- colours ------------------------------------------------------------

    def swap_colours(self) -> None:
        # Beside ``set_fg``'s clearing rule rather than through it: the
        # foreground is now whatever the background held, and the background
        # never carries a slot -- so a claim left standing would land the next
        # stroke in the slot of a colour no longer in hand.
        self.fg, self.bg = self.bg, self.fg
        self.fg_slot = None

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

    def set_fg(self, colour: Any, slot: int | None = None) -> None:
        """Load the brush, recording which palette slot it came from.

        One door for all four ways a foreground is chosen -- the wheel, the
        eyedropper, the session swatch row, a palette slot -- because the thing
        that has to be right is the *clearing*, and a clear that lives at three
        of the four sites is a brush that goes on claiming a slot the user moved
        away from. Only a palette click passes ``slot``.
        """
        self.fg = tuple(int(c) for c in tuple(colour)[:4])  # type: ignore[assignment]
        self.fg_slot = None if slot is None else int(slot)

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
# scoped to the *shape* tools alone. The selection tools use Aseprite's own
# Shift / Alt+Shift / Ctrl+Shift combinations, sampled into ``combine``; giving them a
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
