"""Inker's slices: the nine-slice rectangles, their handles and their drag.

A slice is a *named rectangle on the canvas* rather than anything on a
layer, which is why it has its own hit test, its own eight handles and its
own drag session -- and why a still document's layers share one. What is
here is that gesture end to end: the grab, the nudge arithmetic, the drag,
the release and the outlines.

Lifted out of ``panes/inker_canvas`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed.

``inker_canvas`` is imported as a *module* and never ``from``-imported, and
it serves these names back through a PEP 562 ``__getattr__`` -- so the pair
may be imported in either order and every existing caller goes on naming
``inker_canvas.<name>``.
"""

from __future__ import annotations

import math
from typing import Any

from imgui_bundle import imgui

from .. import (
    inker_state,
    theme,
)
from ..inker.slices import SliceKey, slice_props

#: The four pure helpers this module used to define. They live in
#: ``inker_state`` now (no imgui, no document, no side effects) and are named
#: here because every call site in this file and four test files already do.
from ..inker_state import (
    marquee_rect,
)
from ..tokens import sp
from . import inker_canvas

# --- slices -------------------------------------------------------------------
#
# A tool rather than a pane, so there is no new help anchor and no fourth
# sidebar: the overlay is on the canvas where the rectangles are, and the list,
# the toggles and the delete button ride the tools panel like every other tool's
# options do.
#
# The whole surface obeys the pane's two existing rules. Every screen position
# goes through ``inker_canvas._corners``/``inker_canvas._box`` rather than
# ``origin + x * zoom``, which is
# what keeps it correct on a turned or mirrored page. And a drag commits
# nothing: the press records what the slice looked like, the drag mutates the
# live object so the overlay follows the cursor, and the release pushes exactly
# one ``set_slice`` -- one gesture, one Ctrl+Z.

#: A slice's corner grab squares and the pivot's ring, in **design** pixels --
#: everything below runs both through ``sp``, drawing and hit-testing alike, so
#: the grab radius cannot come out smaller than the handle a user can see on a
#: scaled display.
SLICE_HANDLE = 4.0
SLICE_PIVOT_RADIUS = 5.0
#: How much bigger the grab radius is than the thing drawn. Generous, because
#: the alternative to grabbing a corner is starting a new slice on top of the
#: one you were aiming at.
SLICE_GRAB = 2.5
#: The smallest rectangle a drag will make, in **image** pixels. Two rather than
#: one: a stationary cursor at a fractional position rounds outward to a 1x1
#: rectangle, so a one-pixel floor would put a slice down on every stray click.
#: A genuinely one-pixel slice is still reachable by dragging a corner in.
SLICE_MIN = 2
#: How long a dash of the nine-slice centre is, in screen pixels. Dashed rather
#: than solid because the centre sits inside the slice's own outline and two
#: solid rectangles a few pixels apart read as one thick border.
SLICE_DASH = 4.0

#: Which corner of the bounds a resize drag is holding, and the pair of indices
#: into ``(x0, y0, x1, y1)`` it writes. Named so the press can record one string
#: and the drag can stay arithmetic.
SLICE_CORNERS = {"nw": (0, 1), "ne": (2, 1), "sw": (0, 3), "se": (2, 3)}


def slices_visible(state: Any) -> bool:
    """Whether the overlay draws. Derived rather than a second flag to keep in
    step: the slice tool forces it on, and ``show_slices`` is only ever the
    answer to "keep showing them while I paint"."""
    return bool(state.show_slices) or state.tool == "slice"


def _near(a, b, radius: float) -> bool:
    return math.dist(a, b) <= radius


def _slice_grab(state: Any, tab: Any, origin, point) -> tuple[str, str]:
    """What a press at ``point`` is holding: ``(kind, corner)``.

    Handles are hit-tested in **screen** space so a grab square is the same size
    to the hand at every zoom -- an image-space radius is a hundredth of a pixel
    at 100x and half the canvas at 5%. The rest (is the cursor inside a slice)
    is image space, where the rectangle actually is.

    The order is outermost first: the bounds corners, then the pivot, then the
    centre's corners, then the body, then empty canvas. Two grabs can genuinely
    coincide -- a pivot parked on a corner -- and resizing is the gesture a user
    reaches for at a corner, so it wins there.
    """
    doc = tab.doc
    entry = doc.slice_by_uid(state.slice_uid)
    mouse = imgui.get_mouse_pos()
    at = (mouse.x, mouse.y)
    frame_uid = tab.frame_uid
    if entry is not None:
        key = entry.at(frame_uid)
        x0, y0, x1, y1 = key.bounds
        corners = {
            "nw": (x0, y0),
            "ne": (x1, y0),
            "sw": (x0, y1),
            "se": (x1, y1),
        }
        grab = sp(SLICE_HANDLE) * SLICE_GRAB
        for name, (cx, cy) in corners.items():
            if _near(inker_state.to_screen(tab.view, origin, cx, cy), at, grab):
                return "slice-resize", name
        if key.pivot is not None:
            pivot = inker_state.to_screen(tab.view, origin, x0 + key.pivot[0], y0 + key.pivot[1])
            if _near(pivot, at, sp(SLICE_PIVOT_RADIUS) * SLICE_GRAB):
                return "slice-pivot", ""
        if key.center is not None:
            cx0, cy0, cx1, cy1 = key.center
            inner = {
                "nw": (x0 + cx0, y0 + cy0),
                "ne": (x0 + cx1, y0 + cy0),
                "sw": (x0 + cx0, y0 + cy1),
                "se": (x0 + cx1, y0 + cy1),
            }
            for name, (cx, cy) in inner.items():
                if _near(inker_state.to_screen(tab.view, origin, cx, cy), at, grab):
                    return "slice-center", name
    # The body of *any* slice, last one first: the list is drawn in order, so
    # the last is the one on top and the one the click visibly landed on.
    for candidate in reversed(doc.slices):
        x0, y0, x1, y1 = candidate.at(frame_uid).bounds
        if x0 <= point[0] < x1 and y0 <= point[1] < y1:
            state.slice_uid = candidate.uid
            return "slice-move", ""
    return "slice-new", ""


def _slice_press(ctx: Any, state: Any, tab: Any, origin, point) -> None:
    """Decide what this gesture is, and record what it started from."""
    if state.transforming:
        # A free transform owns the canvas until it is applied or cancelled;
        # letting a slice drag start underneath it would commit a step against
        # a document that is mid-operation.
        state.drag_kind = ""
        return
    kind, corner = _slice_grab(state, tab, origin, point)
    entry = tab.doc.slice_by_uid(state.slice_uid)
    state.drag_kind = kind
    state.slice_drag = (
        None if entry is None or kind == "slice-new" else slice_props(entry),
        corner,
    )


def _nudged(rect, corner: str, dx: float, dy: float):
    """One corner of a rectangle moved, the other three left alone.

    The corner is allowed to cross the far one, which is what every editor's
    resize does; the ordering happens once, at release, inside
    ``clamp_rect``. Ordering *during* the drag would swap which pair of indices
    the corner's name points at, and the rectangle would stick at the crossing.
    """
    out = list(rect)
    ix, iy = SLICE_CORNERS[corner]
    out[ix] += dx
    out[iy] += dy
    return tuple(out)


def _slice_drag(state: Any, tab: Any, point) -> None:
    """Move the live slice so the overlay follows the cursor. Pushes nothing.

    Measured against what was true at the **press**, never against the previous
    frame -- the rule ``_transform_input`` states, and here it is the difference
    between a drag that works and one that does not: the geometry is integer, so
    a per-frame delta of a third of a pixel truncates to nothing every frame and
    a slow drag at a high zoom moves the rectangle not at all.

    **Whatever the overlay is drawing is what moves.** On a frame with a key of
    its own that is the key, not the slice's own rectangle -- they are the same
    thing only on an unkeyed frame, and dragging the base while the canvas draws
    the key is a gesture that visibly does nothing.
    """
    entry = tab.doc.slice_by_uid(state.slice_uid)
    if entry is None or state.slice_drag is None or state.drag_kind == "slice-new":
        return
    before, corner = state.slice_drag
    if before is None:
        return
    anchor = state.drag_anchor or point
    dx, dy = point[0] - anchor[0], point[1] - anchor[1]

    frame_uid = tab.frame_uid
    keyed = frame_uid is not None and frame_uid in before["keys"]
    start = (
        before["keys"][frame_uid]
        if keyed
        else SliceKey(before["bounds"], before["pivot"], before["center"])
    )
    bounds, pivot, center = start.bounds, start.pivot, start.center

    if state.drag_kind == "slice-move":
        x0, y0, x1, y1 = bounds
        bounds = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    elif state.drag_kind == "slice-resize":
        bounds = _nudged(bounds, corner, dx, dy)
    elif state.drag_kind == "slice-pivot" and pivot is not None:
        pivot = (pivot[0] + dx, pivot[1] + dy)
    elif state.drag_kind == "slice-center" and center is not None:
        center = _nudged(center, corner, dx, dy)

    if keyed:
        # A fresh frozen key in a fresh dictionary, never a write into the live
        # one: the step's "before" is holding that dictionary's contents and
        # must not move with the drag.
        entry.keys = {**entry.keys, frame_uid: SliceKey(bounds, pivot, center)}
        return
    entry.bounds, entry.pivot, entry.center = bounds, pivot, center
    entry.normalise()


def _slice_release(ctx: Any, state: Any, tab: Any, point) -> None:
    """One step for the whole gesture, or nothing at all."""
    doc = tab.doc
    if state.drag_kind == "slice-new":
        anchor = state.drag_anchor or point
        rect = marquee_rect(anchor, point)
        if rect[2] - rect[0] >= SLICE_MIN and rect[3] - rect[1] >= SLICE_MIN:
            state.slice_uid = doc.add_slice(rect).uid
            return
        # A click with no drag on empty canvas deselects, which is what the
        # marquee tool does with the same gesture -- and it is not "make a
        # one-pixel slice", which is what rounding a stationary cursor outward
        # would otherwise produce on every stray click.
        state.slice_uid = 0
        return
    if state.slice_drag is None:
        return
    before, _corner = state.slice_drag
    if before is not None:
        # ``was``, because the live object has already been dragged: reading the
        # before here would compare the slice against itself and push nothing.
        doc.set_slice(state.slice_uid, was=before)


def _dashed_rect(draw_list: Any, a, b, colour: int) -> None:
    """A rectangle in dashes, so it reads as *inside* the slice's own outline
    rather than as a second border a few pixels in from it."""
    corners = ((a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1]))
    for start, end in zip(corners, (*corners[1:], corners[0]), strict=True):
        length = math.dist(start, end)
        if length <= 0.0:
            continue
        steps = max(1, int(length // (SLICE_DASH * 2)))
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        for step in range(steps):
            head = step * length / steps
            tail = min(head + SLICE_DASH, length)
            draw_list.add_line(
                (start[0] + ux * head, start[1] + uy * head),
                (start[0] + ux * tail, start[1] + uy * tail),
                colour,
            )


def _slices(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """Every slice, and the handles of the selected one.

    Everything here is placed through ``inker_canvas._box``/``to_screen``, never through
    ``origin + x * zoom``: a quarter turn maps an axis-aligned image rectangle
    onto an axis-aligned *screen* rectangle, and that is only true of a position
    that has been through the view's basis.
    """
    frame_uid = tab.frame_uid
    outline = inker_canvas._u32(theme.ACCENT, 0.75)
    inner = inker_canvas._u32(theme.ACCENT, 0.45)
    hot = inker_canvas._u32(theme.ACCENT)
    for entry in tab.doc.slices:
        key = entry.at(frame_uid)
        x0, y0, x1, y1 = key.bounds
        selected = entry.uid == state.slice_uid
        a, b = inker_canvas._box(tab.view, origin, x0, y0, x1, y1)
        draw_list.add_rect(a, b, hot if selected else outline)
        if key.center is not None:
            cx0, cy0, cx1, cy1 = key.center
            ca, cb = inker_canvas._box(tab.view, origin, x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1)
            _dashed_rect(draw_list, ca, cb, inner)
        if key.pivot is not None:
            px, py = inker_state.to_screen(tab.view, origin, x0 + key.pivot[0], y0 + key.pivot[1])
            radius = sp(SLICE_PIVOT_RADIUS)
            draw_list.add_circle((px, py), radius, hot if selected else outline)
            draw_list.add_line((px - radius, py), (px + radius, py), hot)
            draw_list.add_line((px, py - radius), (px, py + radius), hot)
        if not selected:
            continue
        size = sp(SLICE_HANDLE)
        for cx, cy in inker_canvas._corners(tab.view, origin, x0, y0, x1, y1):
            draw_list.add_rect_filled((cx - size, cy - size), (cx + size, cy + size), hot)
