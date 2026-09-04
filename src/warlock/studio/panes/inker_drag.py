"""The canvas drag: what a held pointer means, and what its release writes.

The pane's one hard rule lives here -- **a drag never commits anything**. A
press has already decided what the gesture is; ``_drag`` only updates a preview
or walks the brush, and ``_release`` is the single place that writes an
undoable step, which is what makes every gesture exactly one Ctrl+Z.
``_shape_drag`` is the endpoint arithmetic the two share, and it reads its
modifiers *live* rather than sampling them at press.

Lifted out of ``panes/inker_canvas`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed.

``inker_canvas`` is imported as a *module* and never ``from``-imported, and it
serves these names back through a PEP 562 ``__getattr__`` -- so the pair may be
imported in either order and every existing caller goes on naming
``inker_canvas.<name>``.
"""

from __future__ import annotations

import math
from typing import Any

from imgui_bundle import imgui

from .. import (
    inker_mode,
    inker_state,
)

#: The four pure helpers this module used to define. They live in
#: ``inker_state`` now (no imgui, no document, no side effects) and are named
#: here because every call site in this file and four test files already do.
from ..inker_state import is_click, marquee_rect
from . import inker_canvas, inker_slices


def _drag(state: Any, tab: Any, point) -> None:
    doc = tab.doc
    if state.drag_kind.startswith("slice-"):
        inker_slices._slice_drag(state, tab, point)
        state.last_point = point
        return
    if state.drag_kind == "tile":
        inker_canvas._tile_stamp(state, tab, point)
        state.last_point = point
        return
    if state.drag_kind == "paint":
        doc.stroke_to(point)
    elif state.drag_kind == "spray":
        # Every held frame, moving or not: an airbrush keeps emitting while the
        # button is down, which is the whole tool. A rate times a delta rather
        # than a count per frame, so the cloud is the same on a slow machine as
        # on a fast one -- with the fraction carried, or a rate under one dab a
        # frame would round to nothing sixty times a second.
        state.spray_carry += state.spray_rate * imgui.get_io().delta_time
        emit = int(state.spray_carry)
        if emit > 0:
            state.spray_carry -= emit
            doc.spray_at(point, emit)
    elif state.drag_kind == "layer_move":
        # From the anchor, not from the last point: the session re-renders from
        # its snapshot, so what it wants is the total offset.
        anchor = state.drag_anchor or point
        doc.preview_layer_move(round(point[0] - anchor[0]), round(point[1] - anchor[1]))
    elif state.drag_kind == "move" and doc.floating is not None:
        last = state.last_point or point
        doc.move_floating(round(point[0] - last[0]), round(point[1] - last[1]))
    elif state.drag_kind == "lasso" and (
        not state.lasso or math.dist(state.lasso[-1], point) >= 2.0
    ):
        # Sampled rather than every pixel: a lasso is a polygon, and a vertex
        # per mouse-move makes a thousand-point one out of a slow drag.
        state.lasso.append(point)
    state.last_point = point


def _shape_drag(state: Any, anchor, point):
    """A shape drag's two endpoints, with the modifiers read *now*.

    Live rather than sampled at press, unlike ``combine``: a user decides a
    rectangle should have been a square halfway through drawing it, and the
    preview has to be able to change its mind with them.
    """
    from .. import inker_ops

    held = inker_canvas.held_chord()
    return inker_state.shape_endpoints(
        state.tool,
        anchor,
        point,
        constrain=inker_ops.action_active(
            "shape_square", held, "ShapeTool", state.shortcut_overrides
        ),
        from_centre=inker_ops.action_active(
            "shape_center", held, "ShapeTool", state.shortcut_overrides
        ),
    )


def _release(ctx: Any, state: Any, tab: Any, point) -> None:
    from ..inker import SelectionMask

    doc = tab.doc
    anchor = state.drag_anchor or point
    kind = state.drag_kind

    if kind.startswith("slice-"):
        inker_slices._slice_release(ctx, state, tab, point)
        state.clear_drag()
        return
    if kind in ("paint", "spray"):
        doc.end_stroke()
        # Where the next Shift-click draws from. After ``end_stroke``, so a
        # refused or empty stroke still moves it -- the user's hand was here
        # either way, and a line back to some earlier point they have forgotten
        # about is more surprising than a line from where they just clicked.
        tab.view.last_paint = point
    elif kind == "layer_move":
        doc.commit_layer_move()
    elif kind == "shape":
        p0, p1 = _shape_drag(state, anchor, point)
        doc.shape(
            state.tool,
            (int(p0[0]), int(p0[1])),
            (int(p1[0]), int(p1[1])),
            # Never the background colour: a right-press on a shape tool is
            # inert and never reaches here (see ``BG_BUTTON_TOOLS``).
            state.fg,
            state.brush_size,
            filled=state.shape_filled,
            wrap=tab.tiled,
            # The rectangle's rounded corners (6.4). Read at the commit rather
            # than at the press, so ``C`` held mid-drag changes the shape under
            # the cursor -- which is what Aseprite's own gesture does.
            radius=int(state.corner_radius) if state.tool == "rect" else 0,
        )
    elif kind == "marquee":
        rect = marquee_rect(anchor, point)
        if not is_click(anchor, point) and rect[2] > rect[0] and rect[3] > rect[1]:
            build = (
                SelectionMask.from_ellipse
                if state.tool == "select_ellipse"
                else SelectionMask.from_rect
            )
            doc.select(build(doc.size, rect), state.combine)
        else:
            # A click with no drag inside a select tool means "deselect", which
            # is what every other editor does and what stops a stray click
            # leaving a one-pixel selection nothing can be painted outside.
            doc.deselect()
    elif kind == "mask-move":
        # One step for the whole drag: the live offset was drawn by shifting
        # the ants and touched nothing, so this is the first and only thing the
        # gesture pushes.
        dx, dy = inker_canvas._mask_shift(state, point)
        if (dx or dy) and doc.mask is not None:
            doc.select(doc.mask.translated(dx, dy))
    elif kind == "lasso":
        # The same landing the polygonal lasso commits through (C4), so the two
        # cannot come to disagree about what a run of vertices selects. A drag
        # too short to be a polygon is the marquee's stray click: deselect.
        if not inker_mode.polygon_select(doc, state.lasso, state.combine):
            doc.deselect()
    elif kind == "gradient":
        # To transparent means the *foreground* colour at zero alpha, not the
        # background one: fading to a transparent black leaves a dark fringe
        # wherever the two are blended.
        end = (*state.fg[:3], 0) if state.gradient_to_transparent else state.bg
        doc.gradient(
            anchor,
            point,
            state.fg,
            end,
            kind=state.gradient_kind,
            # Empty means the foreground-to-background preset, which is a live
            # reading of the two colours rather than a copy of them -- so
            # swapping with X changes the next gradient, as it always has.
            stops=state.gradient_stops or None,
            # "none" is the tool option's spelling and None is the engine's:
            # the engine's is a *path*, not a value, and keeping the two apart
            # is what makes the undithered arithmetic byte-identical.
            dither=None if state.gradient_dither == "none" else state.gradient_dither,
        )
    # Only the kinds that actually paint. A marquee drag pushes no history step
    # either, and reading a still head after one would announce a revert that
    # never happened; ``clear_drag`` below drops the banked head for every other
    # gesture. The tile stamp is excluded by construction -- it writes refs and
    # never reaches the tileset at all.
    if kind in ("paint", "spray", "shape", "gradient"):
        inker_canvas._manual_note(ctx, state, doc)
    state.clear_drag()
