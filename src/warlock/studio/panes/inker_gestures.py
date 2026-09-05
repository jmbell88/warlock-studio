"""The multi-click gestures: the polygonal lasso, the curve and the text tool.

A gesture that takes several clicks is not a drag, and the difference is
the whole of this file: there is no button held down to end it, so it ends
on a click near its own first point (``closes_gesture``), on Enter, or on
Escape -- and until then the vertices live on the pane rather than in the
document. The text tool is here because its popup is the same shape: a
gesture with a modal middle.

Lifted out of ``panes/inker_canvas`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed.

``inker_canvas`` is imported as a *module* and never ``from``-imported, and
it serves these names back through a PEP 562 ``__getattr__`` -- so the pair
may be imported in either order and every existing caller goes on naming
``inker_canvas.<name>``.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import (
    controls,
    inker_mode,
    inker_state,
    theme,
    tokens,
    widgets,
)
from ..inker import textstamp
from ..inker.document import catmull_rom, curve_points, curve_spans

#: The four pure helpers this module used to define. They live in
#: ``inker_state`` now (no imgui, no document, no side effects) and are named
#: here because every call site in this file and four test files already do.
from ..inker_state import (
    closes_gesture,
)
from ..tokens import sp
from . import inker_canvas

# --- the polygonal lasso, and the multi-click gesture under it (C4) -----------
#
# The pane's second kind of gesture. A drag is press / move / release and
# ``drag_kind`` names it while the button is down; this one is a run of separate
# clicks with the button *up* in between, so it cannot be a ``drag_kind`` at all
# -- and must not be, because ``_input`` refuses a press while a gesture owns the
# mouse (the C12d guard) and a click sequence would be eaten by its own state.
# Each click is therefore complete at the press: nothing drags, nothing releases,
# and ``InkerState.gesture_pts`` is the whole of what survives between clicks.
#
# Built as the shared thing rather than the poly lasso's own, because the curve
# and polygon shape tools are this gesture with a different landing -- and Q-c
# is that landing: ``_gesture_press`` collects for all four tools, and
# ``inker_mode.commit_gesture`` decides whether the vertices become a selection
# or paint. The only thing they differ in on the way there is whether a click
# back on the first point closes them (``CLOSES_ON_FIRST``).

#: How near the first vertex a click has to land to close the polygon, in
#: **screen** pixels before ``sp`` -- the slice handles' rule, for the slice
#: handles' reason: an image-space radius is a hundredth of a pixel at 100x zoom
#: and half the canvas at 5%, so the target would be unhittable at one end of the
#: zoom range and unavoidable at the other.
POLY_CLOSE = 7.0

#: The gesture tools whose path is **closed**, and which therefore end on a
#: click back at the first vertex. The poly lasso (a polygon is what a selection
#: is) and the polygon shape tool; the polyline and the curve are open paths, so
#: a click near their first point is a click like any other and placing a vertex
#: there is the only thing it can honestly mean. They end on a double-click or
#: Enter instead, which every gesture tool answers.
CLOSES_ON_FIRST = frozenset({"lasso_poly", "polygon"})


def _gesture_press(ctx: Any, state: Any, tab: Any, point) -> None:
    """One click of a multi-click gesture: open, extend, or close the path.

    Shared by the polygonal lasso and by the three clicked shape tools (Q-c),
    which differ only in what ``commit_gesture`` does with the vertices and in
    whether a click back on the first one closes them. A second copy of this
    arithmetic is exactly where the two would come to disagree about what a
    double-click means.

    ``drag_kind`` is left empty on every arm, which is the load-bearing part:
    it keeps the next click out of the C12d guard, keeps ``_drag`` and
    ``_release`` out of the gesture entirely -- so this tool can never take the
    freehand ``drag_kind="lasso"`` path -- and keeps ``clear_drag`` free to mean
    "cancel the gesture" everywhere it is already called from.
    """
    doc = tab.doc
    if not state.gesture_pts:
        # The float lands on the first click, as it does for every other
        # selection tool's press and every paint tool's -- the user has moved on
        # from it.
        doc.commit_floating()
        # Captured once, here. ``state.combine`` is re-read at every press, and
        # letting go of Shift before the closing click would otherwise turn an
        # add into a replace that throws the selection away. Recorded for the
        # painting shapes too, which have no use for it: one gesture has one
        # opening arm, and a field left stale is a field the next tool reads.
        state.gesture_combine = state.combine
        state.gesture_pts = [point]
        state.drag_kind = ""
        return
    if imgui.is_mouse_double_clicked(state.drag_button) or (
        state.tool in CLOSES_ON_FIRST
        and closes_gesture(state.gesture_pts, point, tab.view.zoom, sp(POLY_CLOSE))
    ):
        # The closing click places no vertex of its own: a double-click's second
        # press lands on top of the first, and a click near vertex 0 *is*
        # vertex 0. Either would add a degenerate edge to the polygon.
        inker_mode.commit_gesture(state, tab)
    else:
        state.gesture_pts.append(point)
    state.drag_kind = ""


#: The settled part of the previewed curve: ``(vertices, samples)``, or None.
#:
#: The frame loop never blocks, and this overlay runs on it: a fifty-vertex
#: curve resampled from scratch every frame is milliseconds of pure arithmetic
#: per frame for a picture that changed in two segments. Module-level rather
#: than per-tab state because it is a *memo* of a pure function keyed on its own
#: argument -- a stale entry cannot be read, only missed, so nothing has to
#: invalidate it when the tab, the tool or the document changes.
_curve_settled: tuple[tuple[tuple[float, float], ...], list[tuple[float, float]]] | None
_curve_settled = None


def _curve_path(points, cursor) -> list[tuple[float, float]]:
    """The previewed curve through ``points`` and the cursor, resampling only
    what the cursor can move.

    A Catmull-Rom segment is a function of four consecutive control points, so
    the provisional cursor point reaches exactly the last two segments -- the
    one that ends at it and the one before, which has it as its far neighbour.
    Everything earlier is settled, and settled is what is memoised.

    The result is **identical** to ``catmull_rom([*points, cursor])``, sample
    for sample and bit for bit, which it has to be: the same four control points
    in the same order through the same arithmetic, and the tail is recomputed
    from a four-point slice whose middle two segments have the same neighbours
    they do in the whole path. That equality is the test.
    """
    global _curve_settled

    pts = curve_points(points)  # collapsed exactly as ``catmull_rom`` does
    if len(pts) < 3:
        # One or two control points make no settled segment to keep; the whole
        # path is at most the two segments this would recompute anyway.
        return catmull_rom([*pts, cursor])
    key = tuple(pts)
    if _curve_settled is None or _curve_settled[0] != key:
        _, spans = curve_spans(pts)
        settled = [pts[0]]
        # Segments 0 .. len(pts) - 3, which end at the second-to-last vertex.
        for span in spans[: len(pts) - 2]:
            settled.extend(span)
        _curve_settled = (key, settled)
    out = list(_curve_settled[1])
    # The last three vertices plus the cursor: its middle two segments have
    # exactly the neighbours they have in the whole path, so their samples are
    # the whole path's. The first one is the settled tail already in ``out``.
    for span in curve_spans([*pts[-3:], cursor])[1][1:]:
        out.extend(span)
    return out


def _gesture_preview(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """The open path, its rubber band, and where the closing click goes.

    Through ``to_screen`` like every other overlay (Ink9), so the polygon is
    drawn on the turned or mirrored page rather than a quarter turn away from
    it. The rubber band follows the *snapped* cursor rather than the raw one,
    which is the rule ``_preview``'s shape branch already states: a band drawn
    somewhere the next vertex will not go is worse than no band at all.

    The curve is previewed **segment by segment through the same arithmetic the
    commit rasterises** (``_curve_path``, which is ``catmull_rom`` with the
    settled segments memoised), rather than as the straight chords between its
    points: a spline preview that shows a polygon is a preview of a different
    tool, and one that samples the curve its own way is a promise the rasteriser
    has not made. The cursor rides along as a provisional last point, so what is
    on screen is the curve *if you click here* -- exactly what the polygon's
    rubber-band edge already means.
    """
    points = state.gesture_pts
    if not points:
        return
    view = tab.view
    colour = inker_canvas._u32(theme.ACCENT)
    mouse = imgui.get_mouse_pos()
    point = inker_state.to_image(view, origin, mouse.x, mouse.y)
    cursor = inker_canvas._snapped(state, inker_canvas._local(state, point))
    if state.tool == "curve":
        curve = [inker_state.to_screen(view, origin, x, y) for x, y in _curve_path(points, cursor)]
        for a, b in zip(curve, curve[1:], strict=False):
            draw_list.add_line(a, b, colour)
        return
    screen = [inker_state.to_screen(view, origin, x, y) for x, y in points]
    for a, b in zip(screen, screen[1:], strict=False):
        draw_list.add_line(a, b, colour)
    tip = inker_state.to_screen(view, origin, *cursor)
    draw_list.add_line(screen[-1], tip, colour)
    if state.tool in CLOSES_ON_FIRST and len(screen) >= 3:
        # The edge a commit would close with, and the target that closes it.
        # Fainter than the placed edges: it is what *would* happen, not what has.
        draw_list.add_line(tip, screen[0], inker_canvas._u32(theme.ACCENT, 0.4))
        draw_list.add_circle(screen[0], sp(POLY_CLOSE), colour)


# --- the text tool ------------------------------------------------------------
#
# A popup on the canvas rather than a section in the tools panel, because the
# gesture is "put a word *here*": the click chooses the spot and the popup is
# the only thing between it and a floating buffer. Both halves live in this
# module and neither may move to another pane -- an imgui popup is matched by an
# id computed off the id stack, so an ``open_popup`` in the canvas child and a
# ``begin_popup`` in a sidebar would never meet (the same trap
# ``inker_bridge.CONVERT_POPUP`` is written up under).
#
# There is no live preview. A stamp *is* a floating buffer, which the canvas
# already draws and the user can already drag, so previewing it before the OK
# button would be a second, worse copy of what the next click gives them.

TEXT_POPUP = "inker-text"

#: How tall the typing box is, in design pixels: four lines and a bit, which is
#: what a caption or a label takes and enough for a longer one to scroll in.
TEXT_BOX_HEIGHT = 88.0

#: The cap on what one stamp may hold. Generous rather than meaningful -- it is
#: there so a paste of a whole file into the box cannot ask FreeType to lay out
#: a megabyte on the frame thread.
TEXT_MAX = 4000


def _open_text(state: Any, tab: Any) -> None:
    """Start a text stamp at the press that just landed.

    The font scan happens here rather than at import: reading several hundred
    directory entries is a frame's worth of work that a session which never
    touches the tool must not pay, and ``font_choices`` caches it from the
    first open onwards.

    **Antialiasing follows the document until the user says otherwise**: off on
    an indexed one, on everywhere else, decided here on every open. A palette
    is a promise that the file holds exactly those colours, and an antialiased
    edge is a rim of blends that each snap to the nearest slot -- so the
    default that keeps the promise is the monochrome rasteriser.

    It is a *default* and not a rule: an indexed document with a soft-edged
    palette is a real thing, so ticking the box in the popup sets
    ``text_aa_touched`` and this stops deciding anything. That flag is the
    whole fix for what the first version got wrong -- it asked whether the text
    tool had a stored options *entry*, which ``options_for`` creates on the
    first read of any of the three, so one popup on an RGB document made every
    indexed document for the rest of the session open with AA on. The manual
    says "on an indexed document it starts off", with no session in the
    sentence, and a promise that holds until you have used the tool once is not
    the promise.
    """
    inker_mode.font_choices()
    if not state.text_aa_touched:
        state.aa = not tab.doc.palette
    imgui.open_popup(TEXT_POPUP)


def _text_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The typing box, the font, the size and the AA toggle.

    Nothing to clean up when imgui closes it on a click outside -- unlike the
    filter and conversion sessions this is cloned from, a text stamp previews
    nothing and holds nothing on the document, so an unanswered popup is
    simply a stamp that was never made. That is the whole of why there is no
    ``text_open`` flag beside ``filter_uid``.
    """
    if not imgui.begin_popup(TEXT_POPUP):
        state.text_uid = ""
        return
    if state.text_uid and state.text_uid != tab.uid:
        # The tab changed under the popup. ``text_at`` is a point in the *other*
        # document's pixels, so answering OK here would stamp into this one at
        # coordinates that mean nothing -- with the antialias default decided
        # from the other document's palette. Closed rather than redirected: the
        # user pointed at a place on a picture that is no longer on screen.
        state.text_uid = ""
        imgui.close_current_popup()
        imgui.end_popup()
        return
    widgets.popup_chrome(_imgui=imgui)
    state.text_buffer = widgets.multiline(
        "##inkertext", state.text_buffer, sp(TEXT_BOX_HEIGHT), TEXT_MAX
    )
    choices = inker_mode.font_choices()
    if choices:
        state.font = widgets.combo("##inkerfont", state.font, choices, sp(240))
    else:
        # Only reachable with the vendored face missing *and* no system font
        # directory, i.e. a broken install. Said out loud rather than drawn as
        # an empty combo the user would click at.
        widgets.muted("No fonts found.")
    changed, value = widgets.labeled_slider_int(
        "Size", int(state.text_size), textstamp.MIN_SIZE, textstamp.MAX_SIZE
    )
    if changed:
        state.text_size = int(value)
    changed, value = controls.checkbox("Antialias", bool(state.aa), _imgui=imgui)
    if changed:
        state.aa = bool(value)
        # The user has an opinion now, so ``_open_text`` stops forming one --
        # in *both* directions. Ticking it on an indexed document keeps it
        # ticked there, and unticking it on an RGB one keeps it unticked; a
        # default that reasserted itself on the next click would be a checkbox
        # the user cannot operate.
        state.text_aa_touched = True
    widgets.help_marker(
        "Off renders the glyphs as whole pixels -- no partial coverage "
        "anywhere -- which is what pixel art and an indexed palette want."
    )
    imgui.color_button(
        "##inkertextfg", imgui.ImVec4(*[c / 255.0 for c in state.fg]), 0, (sp(16), sp(16))
    )
    imgui.same_line()
    widgets.muted("the foreground colour")

    imgui.dummy((0, sp(tokens.SP_1)))
    if controls.button("OK##inkertext", (sp(90), 0), _imgui=imgui):
        # The tool becomes Move on success (``stamp_text``), so the popup must
        # close either way: a refusal has already toasted why.
        inker_mode.stamp_text(ctx, state, tab)
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel##inkertext", (sp(90), 0), _imgui=imgui):
        imgui.close_current_popup()
    imgui.end_popup()
