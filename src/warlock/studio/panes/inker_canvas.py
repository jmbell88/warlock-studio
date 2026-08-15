"""Paint mode's centre pane: the file row, the tab bar and the canvas itself.

Descended from the inline editor's canvas, which is why the view maths, the
press/drag/release split and the texture gating look familiar -- those were the
parts that worked. What is new is that the pane is now driving a document with
selections and layers, so the drawing order matters: checkerboard, composite,
floating pixels, grid, symmetry axes, marching ants, then whatever the current
drag is previewing, then the brush cursor. Everything after the composite is an
overlay and none of it touches pixels.

The one non-obvious rule is that a *drag* never commits anything. A press
decides what the drag means, the drag updates a preview or walks the brush, and
the release is the only thing that writes an undoable step -- which is what
makes every gesture exactly one Ctrl+Z.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np
from imgui_bundle import imgui

from .. import ants, icons, inker_mode, inker_state, theme, widgets
from ..inker import textstamp
from ..inker.indexed import shade_ramp
from ..inker.slices import SliceKey, slice_props
from ..inker.tiling import axes_of, canonical, tile_offset
from ..inker_state import BG_BUTTON_TOOLS, PAINT_TOOLS, SELECT_TOOLS, SHAPE_TOOLS
from ..tokens import sp
from . import inker_textures

#: The four positions of the Tiled control, one per :data:`TILED_AXES` entry.
#: Prefixed, because in a row of icon buttons a bare "X" says nothing about
#: what it is an axis of.
TILED_LABELS = (
    ("off", "Tiled: off"),
    ("x", "Tiled: X"),
    ("y", "Tiled: Y"),
    ("both", "Tiled: X+Y"),
)


def _u32(colour: int, alpha: float = 1.0) -> int:
    return imgui.get_color_u32(imgui.ImVec4(*theme.rgba(colour, alpha)))


# --- drawing through the view's orientation (Ink9) ---------------------------
#
# Every overlay below goes through one of these three, and that is the whole of
# what makes canvas rotation and the flipped view safe. The failure the feature
# invites is not a crash: it is one overlay left computing its own screen
# position from ``origin + x * zoom``, which is right at rotation 0 and silently
# a quarter turn out everywhere else -- a grid drawn across a canvas that is not
# there, ants beside the mask they describe. Quarter turns are what keeps these
# three enough (see ``inker_state.ROTATIONS``): an axis-aligned image rectangle
# comes out an axis-aligned *screen* rectangle, so a rect is still a rect and a
# grid is still two families of straight lines.


def _corners(view: Any, origin, x0: float, y0: float, x1: float, y1: float):
    """An image-space rectangle's four corners on screen, in image order:
    top-left, top-right, bottom-right, bottom-left."""
    to = inker_state.to_screen
    return (
        to(view, origin, x0, y0),
        to(view, origin, x1, y0),
        to(view, origin, x1, y1),
        to(view, origin, x0, y1),
    )


def _box(view: Any, origin, x0: float, y0: float, x1: float, y1: float):
    """An image-space rectangle as a screen AABB ``(top_left, bottom_right)``.

    Sound only because the orientation is a quarter turn: it maps the corner
    set onto itself, so the min and max of the four transformed corners *are*
    two opposite corners of the same rectangle rather than a box around a
    tilted one.
    """
    points = _corners(view, origin, x0, y0, x1, y1)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _blit(draw_list: Any, texture: Any, view: Any, origin, x0, y0, x1, y1, **kwargs) -> None:
    """One image drawn over an image-space rectangle, however the view is turned.

    ``add_image_quad`` rather than ``add_image``, because the two-corner form
    can only draw an upright rectangle: the quad's four *positions* carry the
    turn and the four uvs stay in the texture's own order, which is exactly a
    rotation of the picture and not a resampling of it.

    ``uv`` overrides the corner coordinates for the one caller that tiles
    (the checkerboard), in the same top-left/top-right/bottom-right/bottom-left
    order the positions are in. ``uv0``/``uv1`` is the two-corner spelling of
    the same thing, for the tiled composite: values outside 0..1 with the
    sampler set to repeat are what draw the 3x3 neighbourhood in one call
    rather than as nine images positioned by hand.
    """
    uv0, uv1 = kwargs.pop("uv0", None), kwargs.pop("uv1", None)
    if uv0 is not None or uv1 is not None:
        (u0, v0), (u1, v1) = uv0 or (0.0, 0.0), uv1 or (1.0, 1.0)
        kwargs.setdefault("uv", ((u0, v0), (u1, v0), (u1, v1), (u0, v1)))
    uv = kwargs.pop("uv", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    colour = kwargs.pop("colour", None)
    a, b, c, d = _corners(view, origin, x0, y0, x1, y1)
    ref = widgets.texture_ref(texture)
    if colour is None:
        draw_list.add_image_quad(ref, a, b, c, d, *uv)
    else:
        draw_list.add_image_quad(ref, a, b, c, d, *uv, colour)


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    _file_row(ctx, state)
    if not state.docs:
        _empty(ctx, state)
        return
    _tab_bar(ctx, state)
    tab = state.active
    if tab is not None:
        _canvas(ctx, state, tab)


# --- the file row -----------------------------------------------------------


def _file_row(ctx: Any, state: Any) -> None:
    tab = state.active
    # Undo and redo live beside the canvas they act on; the bridge panel's
    # pair was three panels away from the stroke it reversed.
    doc = tab.doc if tab is not None else None
    # Undo can rebind the stack mid-write; the saving gate here matches the
    # keyboard path (_MUTATING_CTRL) and the bridge panel's own pair.
    idle = tab is not None and not tab.busy
    if widgets.icon_button(
        icons.UNDO, "Undo (Ctrl+Z)", enabled=idle and doc.history.can_undo
    ):
        doc.undo()
    imgui.same_line()
    if widgets.icon_button(
        icons.REDO, "Redo (Ctrl+Y)", enabled=idle and doc.history.can_redo
    ):
        doc.redo()
    imgui.same_line()
    if imgui.button("New"):
        imgui.open_popup("new-canvas")
    _new_popup(ctx)
    imgui.same_line()
    if imgui.button("Open"):
        inker_mode.ask_open(ctx)
    imgui.same_line()
    if imgui.button("Recent"):
        imgui.open_popup("inker-recent")
    _recent_popup(ctx, state)
    imgui.same_line()
    # A save commits the floating buffer, so saving mid-transform would land
    # the transform with no confirm and leave the mode pointing at nothing.
    busy = tab is not None and (tab.busy or state.transforming)
    if widgets.disabled_button("Save", tab is not None and not busy):
        inker_mode.save(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button("Save as", tab is not None and not busy):
        inker_mode.save_as(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button("Export PNG", tab is not None and not busy):
        inker_mode.export_png(ctx, tab)
    if tab is not None:
        imgui.same_line()
        _view_row(tab)
        imgui.same_line()
        if tab.saving:
            widgets.muted("saving...")
        elif tab.playing:
            widgets.muted("playing...")
        elif tab.dirty:
            widgets.text_colored(theme.WARN, "unsaved")
    imgui.separator()
    if tab is not None and state.transforming:
        _transform_row(ctx, state, tab)


def _view_row(tab: Any) -> None:
    """Turn the page, and mirror it. Never disabled while a save is running.

    Everything else on this row is gated on ``busy`` because it changes the
    document; these two change nothing at all -- no pixels move, no step is
    pushed, nothing is written -- so gating them would be an editor that refuses
    to let you *look* at your drawing while it writes a file.
    """
    view = tab.view
    if widgets.icon_button(icons.ROTATE_CW, "Rotate the view (Ctrl+4)"):
        inker_state.rotate_view(view, 1)
    imgui.same_line()
    if widgets.icon_button(icons.FLIP_HORIZONTAL, "Flip the view (Ctrl+5)"):
        inker_state.flip_view(view)
    imgui.same_line()
    # One control driving the view *and* the writes, deliberately: a canvas
    # that showed its neighbours while the brush went on clamping at the edge
    # would be a picture of a seamless tile you cannot paint.
    tab.tiled = widgets.combo("##inkertiled", tab.tiled, list(TILED_LABELS), sp(104))
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Draws the eight neighbouring tiles around this one, and wraps "
            "every stroke, fill and shape across the seams it shows."
        )
    if view.rotation or view.flipped:
        imgui.same_line()
        # Said out loud, because the two are invisible once you have looked away
        # for a moment and a mirrored canvas silently teaches the wrong hand.
        parts = ([f"{view.rotation} deg"] if view.rotation else []) + (
            ["flipped"] if view.flipped else []
        )
        if imgui.small_button(f"{' + '.join(parts)}##inkerviewreset"):
            view.rotation, view.flipped = 0, False
            view.pending_zoom = view.zoom
        if imgui.is_item_hovered():
            imgui.set_tooltip("The view only -- click to set it upright")


def _transform_row(ctx: Any, state: Any, tab: Any) -> None:
    """Numeric handles for the same operations the drag handles do.

    A drag cannot express "exactly 90 degrees", and a rotation that is nearly
    square is worse than either -- so the buttons are not a convenience, they
    are the only way to get an exact one.
    """
    doc = tab.doc
    widgets.text_colored(theme.ACCENT, "Transform")
    imgui.same_line()
    if imgui.button("Flip H"):
        doc.flip_floating("horizontal")
    imgui.same_line()
    if imgui.button("Flip V"):
        doc.flip_floating("vertical")
    imgui.same_line()
    if imgui.button("-90"):
        doc.rotate_floating(-90.0)
    imgui.same_line()
    if imgui.button("+90"):
        doc.rotate_floating(90.0)
    imgui.same_line()
    if imgui.button("Apply"):
        inker_mode.end_transform(ctx, commit=True)
    imgui.same_line()
    if imgui.button("Cancel"):
        inker_mode.end_transform(ctx, commit=False)

    buf = doc.floating
    if buf is None:
        return
    imgui.set_next_item_width(sp(160))
    changed, angle = imgui.slider_float("Angle", buf.angle, -180.0, 180.0, "%.1f deg")
    if changed:
        doc.transform_floating(angle=angle, resample=state.resample)
    imgui.same_line()
    # Two sliders and a link, rather than the one that used to drive both axes
    # from ``scale[0]``: the engine has taken a per-axis scale all along and
    # the panel was the thing that could not express it.
    imgui.set_next_item_width(sp(110))
    changed_x, fx = imgui.slider_float("X##inkscalex", buf.scale[0], 0.05, 8.0)
    imgui.same_line()
    imgui.set_next_item_width(sp(110))
    changed_y, fy = imgui.slider_float("Y##inkscaley", buf.scale[1], 0.05, 8.0)
    imgui.same_line()
    linked, value = imgui.checkbox("Link##inkscalelink", state.transform_link)
    if linked:
        state.transform_link = value
    if imgui.is_item_hovered():
        imgui.set_tooltip("Scale both axes together. Shift does the same on a handle.")
    if changed_x or changed_y:
        if state.transform_link:
            fx = fy = fx if changed_x else fy
        doc.transform_floating(scale=(fx, fy), resample=state.resample)
    if state.resample == "rotsprite":
        from ..inker import transform

        if not transform.rotsprite_fits(buf.size):
            # The standing version of the toast Ctrl+T raised once: a drag
            # re-renders every frame and cannot say this every time, but the
            # user is looking at this row the whole while.
            widgets.muted("Too big for RotSprite -- turning with nearest neighbour.")
    imgui.separator()


def _new_popup(ctx: Any) -> None:
    """The presets, and the custom size the presets could not express.

    The other half of the 3x3 anchor grid's popup: a resize has taken typed
    width and height since it was written, and creating one could only ever
    offer three squares -- so a user who wanted 1920x1080 had to make a square
    and then resize it, which is two undo steps and a guess about the anchor.

    The fields are remembered per session in ``state.preview`` rather than per
    document, because there is no document yet -- and because reopening the
    dialog after a mistake should offer the number that was nearly right.
    """
    if not imgui.begin_popup("new-canvas"):
        return
    imgui.text("New canvas")
    for width, height in inker_mode.NEW_PRESETS:
        if imgui.button(f"{width} x {height}", (sp(160), 0)):
            inker_mode.new_document(ctx, width, height)
            imgui.close_current_popup()

    imgui.separator()
    key = "inker_new_size"
    width, height = ctx.state.preview.get(key) or inker_mode.NEW_PRESETS[1]
    imgui.set_next_item_width(sp(72))
    changed_w, width = imgui.input_int("W##newcanvas", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(72))
    changed_h, height = imgui.input_int("H##newcanvas", int(height), 0)
    if changed_w or changed_h:
        # Clamped on the way *into* the field as well as on the way out, or the
        # box goes on showing a number that is not what the button would make.
        ctx.state.preview[key] = inker_mode.clamp_canvas(width, height)
    width, height = inker_mode.clamp_canvas(width, height)
    if imgui.button(f"Create {width} x {height}", (sp(160), 0)):
        inker_mode.new_document(ctx, width, height)
        imgui.close_current_popup()
    widgets.muted(f"up to {inker_mode.NEW_MAX} px a side")
    imgui.end_popup()


def _recent_popup(ctx: Any, state: Any) -> None:
    from pathlib import Path

    if not imgui.begin_popup("inker-recent"):
        return
    found = inker_mode.recent_paths(ctx)
    if not found:
        widgets.muted("Nothing opened yet.")
    for path in found:
        # The full path in the id, not just the label: two files with the same
        # basename in different directories are an ordinary thing to have open,
        # and one imgui id between them is one row.
        if imgui.selectable(f"{Path(path).name}##{path}", False)[0]:
            inker_mode.open_path(ctx, Path(path))
            imgui.close_current_popup()
        if imgui.is_item_hovered():
            imgui.set_tooltip(path)
    imgui.end_popup()


def _empty(ctx: Any, state: Any) -> None:
    from pathlib import Path

    imgui.dummy((0, sp(40)))
    imgui.text("Nothing open")
    widgets.muted("Start a canvas, open an image, or send one here from the library.")
    imgui.dummy((0, sp(16)))
    for width, height in inker_mode.NEW_PRESETS:
        if imgui.button(f"New {width} x {height}", (sp(240), 0)):
            inker_mode.new_document(ctx, width, height)
    # The same popup the file row's New button opens, rather than a second set
    # of fields: it is registered earlier in this window, so opening it by name
    # from here is enough.
    if imgui.button("New custom size...", (sp(240), 0)):
        imgui.open_popup("new-canvas")
    imgui.dummy((0, sp(8)))
    if imgui.button("Open a file...", (sp(240), 0)):
        inker_mode.ask_open(ctx)
    found = inker_mode.recent_paths(ctx)
    if found:
        imgui.dummy((0, sp(16)))
        widgets.section("recent")
        for path in found[:6]:
            if imgui.selectable(f"{Path(path).name}##{path}", False)[0]:
                inker_mode.open_path(ctx, Path(path))


# --- tabs -------------------------------------------------------------------


def _tab_bar(ctx: Any, state: Any) -> None:
    flags = imgui.TabBarFlags_.reorderable.value | imgui.TabBarFlags_.auto_select_new_tabs.value
    if not imgui.begin_tab_bar("inker-tabs", flags):
        return
    for tab in list(state.docs):
        item_flags = 0
        if tab.dirty:
            # imgui's own dot, rather than a " *" in the title: the title is
            # also the tab's identity, and decorating it would move the tab.
            item_flags |= imgui.TabItemFlags_.unsaved_document.value
        opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
        if opened:
            state.activate(tab.uid)
            imgui.end_tab_item()
        if not keep:
            inker_mode.request_close(ctx, tab)
    imgui.end_tab_bar()


# --- the canvas -------------------------------------------------------------


def _canvas(ctx: Any, state: Any, tab: Any) -> None:
    flags = imgui.WindowFlags_.no_scroll_with_mouse.value | imgui.WindowFlags_.no_scrollbar.value
    hovered = False
    origin = None
    # A positive height, never a bottom offset: with little room left a "-26"
    # child collapses to nothing and the canvas (and its texture uploads)
    # silently stops being drawn.
    height = max(imgui.get_content_region_avail().y - sp(26), sp(16))
    if imgui.begin_child("inker-canvas", (0, height), imgui.ChildFlags_.borders.value, flags):
        origin = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        region = (max(avail.x, 16.0), max(avail.y, 16.0))
        view = tab.view
        if view.pending_zoom is not None:
            inker_state.centre(view, tab.doc.size, region, view.pending_zoom)
            view.pending_zoom = None
        elif not view.fitted:
            inker_state.fit(view, tab.doc.size, region)
        # The right button is taken as well as the left: it paints with the
        # background colour (C12d) and the canvas has no context menu to
        # conflict with. Without the flag imgui simply never reports the press.
        imgui.invisible_button(
            "##inker-surface",
            region,
            imgui.ButtonFlags_.mouse_button_left.value
            | imgui.ButtonFlags_.mouse_button_right.value,
        )
        active = imgui.is_item_active()
        hovered = imgui.is_item_hovered()
        _input(ctx, state, tab, (origin.x, origin.y), active=active, hovered=hovered)
        _paint(ctx, state, tab, (origin.x, origin.y), hovered=hovered)
        # Inside the child, because that is where ``_press`` opened it from and
        # an imgui popup's id is computed off the id stack it was opened on.
        _text_popup(ctx, state, tab)
    imgui.end_child()
    _status_bar(state, tab, origin, hovered)


def _status_bar(state: Any, tab: Any, origin: Any, hovered: bool) -> None:
    """Zoom, cursor position, active tool, document size -- the numbers a
    paint program keeps under the canvas, which used to live in the far-right
    bridge panel or nowhere."""
    view = tab.view
    parts = [f"{view.zoom * 100:.0f}%"]
    if hovered and origin is not None:
        mouse = imgui.get_mouse_pos()
        px, py = inker_state.to_image(view, (origin.x, origin.y), mouse.x, mouse.y)
        # Folded onto the canonical tile: over a neighbour in the 3x3 view the
        # raw number is off the canvas, and a readout saying "300, 40" on a
        # 256-wide document is a coordinate the user cannot use.
        px, py = canonical((px, py), tab.doc.size, axes_of(tab.tiled))
        parts.append(f"{int(px)}, {int(py)}")
    tool = next((label for key, label, _ in inker_state.TOOLS if key == state.tool), state.tool)
    parts.append(tool)
    parts.append(f"{tab.doc.size[0]} x {tab.doc.size[1]}")
    widgets.muted("   ".join(parts))


# --- input ------------------------------------------------------------------


def _input(ctx: Any, state: Any, tab: Any, origin, *, active: bool, hovered: bool) -> None:
    io = imgui.get_io()
    mouse = imgui.get_mouse_pos()
    point = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)

    if hovered and io.mouse_wheel:
        inker_state.zoom_about(tab.view, origin, (mouse.x, mouse.y), io.mouse_wheel)

    # Middle-drag always pans; space-drag pans with the left button, which is
    # what every paint program does and what makes a tablet usable.
    panning = imgui.is_mouse_dragging(2) or (state.space_held and imgui.is_mouse_dragging(0))
    if (hovered or state.drag_kind == "pan") and panning:
        button = 2 if imgui.is_mouse_dragging(2) else 0
        delta = imgui.get_mouse_drag_delta(button)
        imgui.reset_mouse_drag_delta(button)
        state.drag_kind = "pan"
        tab.view.pan = (tab.view.pan[0] + delta.x, tab.view.pan[1] + delta.y)
        return
    if state.drag_kind == "pan" and not (imgui.is_mouse_down(2) or imgui.is_mouse_down(0)):
        state.drag_kind = ""

    if tab.busy:
        # A save is encoding the live document; letting a stroke land in the
        # middle of it would put half a stroke in the file. Playback is the
        # other half of ``busy``: the canvas is showing a cached flatten of some
        # other frame, so a stroke would land invisibly on the frame underneath.
        #
        # A half-drawn multi-click gesture is dropped rather than merely
        # suspended: its next click cannot be delivered while this returns
        # early, so leaving the vertices up would draw a polygon over a document
        # that is being encoded or played and finish it whenever the tab came
        # back -- against whatever the document looked like by then.
        state.clear_gesture()
        return
    if state.transforming and tab.doc.floating is not None:
        _transform_input(state, tab, origin, point, active=active)
        return
    pressed = 0 if imgui.is_mouse_clicked(0) else 1 if imgui.is_mouse_clicked(1) else -1
    # **A press is refused while a gesture owns the mouse.** Two buttons is
    # exactly what C12d made possible and what nothing here could produce
    # before it: the right button coming down mid-left-drag used to reach
    # ``_press``, whose inert and Alt-pick arms clear ``drag_kind`` and return
    # -- abandoning the open gesture in place. A blur or spray stroke was left
    # with its pixels written and no step (pushed out of band by the *next*
    # stroke's ``end_stroke``), and a layer move was left previewed, undoable
    # by nothing and restorable from a snapshot the next ``begin_layer_move``
    # would have rolled the layer back to. One button owns a gesture from press
    # to release; the other one does nothing until it is over.
    holding = bool(state.drag_kind) and imgui.is_mouse_down(state.drag_button)
    if active and pressed >= 0 and not state.space_held and not holding:
        if state.drag_kind:
            # A gesture whose own button is already up but whose release has
            # not been dispatched yet -- press and release inside one frame.
            # Closed here, with the *previous* gesture's tile offset still in
            # ``state``, so it commits where it was drawn rather than being
            # orphaned by the press below.
            _release(ctx, state, tab, _snapped(state, _local(state, point)))
        state.drag_button = pressed
        # The tile the press landed in, fixed for the whole gesture. See
        # ``tiling.tile_offset``: folding per point would jump the brush a full
        # tile at the seam.
        state.tile_offset = tile_offset(point, tab.doc.size, axes_of(tab.tiled))
        # ``origin`` rides along for the slice tool alone: its handles are
        # hit-tested in screen space (``_slice_grab``), which the point on its
        # own cannot answer.
        _press(ctx, state, tab, _snapped(state, _local(state, point)), origin)
    elif state.drag_kind and imgui.is_mouse_down(state.drag_button):
        _drag(state, tab, _snapped(state, _local(state, point)))
    elif state.drag_kind and not imgui.is_mouse_down(state.drag_button):
        _release(ctx, state, tab, _snapped(state, _local(state, point)))


def _local(state: Any, point: tuple[float, float]) -> tuple[float, float]:
    """A cursor position in the *pressed* tile's coordinates.

    The whole of what the 3x3 view costs the input path, and it is a
    subtraction rather than a modulo for the reason above: within one gesture
    the offset never changes, so a stroke that crosses a seam carries on in a
    straight line and the engine wraps it.
    """
    return (point[0] - state.tile_offset[0], point[1] - state.tile_offset[1])


def _snapped(state: Any, point: tuple[float, float]) -> tuple[float, float]:
    """The cursor, on a grid intersection, for the tools that want one.

    Applied here rather than inside each tool so there is one answer to "where
    did the user click" -- a press that snapped and a release that did not
    would draw a rectangle whose far corner is off the grid.

    **Never for a freehand stroke.** Quantising a brush to a 16-pixel lattice
    is not a drawing aid, it is a different tool, and the paint tools are the
    ones a grid is most likely to be switched on around.
    """
    if not (state.grid and state.grid_snap):
        return point
    if state.tool in PAINT_TOOLS or state.tool in ("lasso", "wand", "eyedropper"):
        return point
    step = max(2, int(state.grid_size))
    return (round(point[0] / step) * step, round(point[1] / step) * step)


# Corner handles, in screen pixels.
HANDLE = 5.0
# How far above the box the rotate handle floats.
ROTATE_ARM = 28.0
# The radial symmetry pivot's crosshair, in screen pixels before ``sp``. A
# mirror shows as a line across the page; a rotation has no line to draw, only
# the point it turns about, so the guide is a small ring rather than nothing.
SYMMETRY_PIVOT_RADIUS = 7.0


#: Which axes each grab point scales. The corners take both, the four edge
#: handles take one -- which is the whole of what makes the scale non-uniform,
#: and it is a table rather than a chain of ``if``s so a handle drawn by
#: ``_transform_box`` cannot be one the drag code has no opinion about. The
#: names are image-space compass points, as ``_handles`` builds them.
HANDLE_AXES = {
    "nw": "xy", "ne": "xy", "sw": "xy", "se": "xy",
    "n": "y", "s": "y", "e": "x", "w": "x",
}


def _handles(tab: Any, origin) -> dict[str, tuple[float, float]]:
    """Where the transform box's grab points are, in screen space."""
    buf = tab.doc.floating
    x, y = buf.offset
    width, height = buf.size
    view = tab.view
    corners = {
        "nw": (x, y),
        "ne": (x + width, y),
        "sw": (x, y + height),
        "se": (x + width, y + height),
        # The edge midpoints, which is what a per-axis scale is grabbed by.
        "n": (x + width / 2.0, y),
        "s": (x + width / 2.0, y + height),
        "w": (x, y + height / 2.0),
        "e": (x + width, y + height / 2.0),
    }
    out = {k: inker_state.to_screen(view, origin, *p) for k, p in corners.items()}
    # The arm points away from the box's top edge *in the image*, carried on to
    # screen -- not straight up, which is only the same thing while the page is
    # upright and puts the handle inside the box at 180 degrees.
    top = inker_state.to_screen(view, origin, x + width / 2.0, y)
    above = inker_state.to_screen(view, origin, x + width / 2.0, y - 1.0)
    dx, dy = above[0] - top[0], above[1] - top[1]
    length = math.hypot(dx, dy) or 1.0
    out["rotate"] = (top[0] + dx / length * ROTATE_ARM, top[1] + dy / length * ROTATE_ARM)
    return out


def _transform_input(state: Any, tab: Any, origin, point, *, active: bool) -> None:
    """Drag a handle to scale, the arm to rotate, the middle to move.

    Every drag is measured against what was true at the *press*, not against
    the previous frame: accumulating per-frame deltas makes a slow drag and a
    fast one produce different results, and a scale that compounds frame by
    frame runs away.
    """
    doc = tab.doc
    buf = doc.floating
    mouse = imgui.get_mouse_pos()
    centre = inker_state.to_screen(tab.view, origin, *buf.centre)

    if active and imgui.is_mouse_clicked(0):
        handles = _handles(tab, origin)
        grab = min(handles, key=lambda k: math.dist(handles[k], (mouse.x, mouse.y)))
        near = math.dist(handles[grab], (mouse.x, mouse.y)) <= HANDLE * 2.5
        state.drag_anchor = point
        state.transform_grab = grab
        # Both axes' reference distances are in *image* space rather than
        # screen space, and that is what makes a per-axis scale correct under a
        # turned page: the view's rotation swaps which screen axis an image
        # axis lands on, so measuring "how far across" on screen would scale
        # the wrong one at 90 degrees. The screen distance stays beside them
        # for the two gestures that are genuinely about the screen -- the
        # uniform ratio and the rotate bearing.
        cx, cy = buf.centre
        state.transform_ref = (
            buf.scale[0],
            buf.scale[1],
            buf.angle,
            max(1.0, math.dist(centre, (mouse.x, mouse.y))),
            math.degrees(math.atan2(mouse.y - centre[1], mouse.x - centre[0])),
            max(1e-3, abs(point[0] - cx)),
            max(1e-3, abs(point[1] - cy)),
        )
        if near and grab == "rotate":
            state.drag_kind = "rotate"
        elif near:
            state.drag_kind = "scale"
        elif buf.contains((int(point[0]), int(point[1]))):
            state.drag_kind = "move"
        else:
            state.drag_kind = ""
        state.last_point = point
        return

    if not state.drag_kind or state.transform_ref is None:
        return
    if not imgui.is_mouse_down(0):
        state.drag_kind = ""
        state.transform_ref = None
        return

    scale0x, scale0y, angle0, dist0, bearing0, ref_x, ref_y = state.transform_ref
    if state.drag_kind == "scale":
        cx, cy = buf.centre
        axes = HANDLE_AXES.get(state.transform_grab, "xy")
        fx = abs(point[0] - cx) / ref_x if "x" in axes else 1.0
        fy = abs(point[1] - cy) / ref_y if "y" in axes else 1.0
        if imgui.get_io().key_shift:
            # Shift constrains to uniform. For a corner that is the screen
            # distance ratio (the same number the drag used to produce before
            # there were two axes); for an edge handle there is only one live
            # ratio, so it is simply applied to both.
            fx = fy = (
                math.dist(centre, (mouse.x, mouse.y)) / dist0 if axes == "xy" else fx * fy
            )
        doc.transform_floating(
            scale=(scale0x * fx, scale0y * fy), resample=state.resample
        )
    elif state.drag_kind == "rotate":
        bearing = math.degrees(math.atan2(mouse.y - centre[1], mouse.x - centre[0]))
        step = bearing0 - bearing  # screen y grows downward; the engine's does not
        if tab.view.flipped:
            # A rotation of the *view* cancels in a difference of bearings, but
            # a mirror does not: dragging the handle clockwise on a flipped page
            # is anticlockwise on the drawing, and without this the buffer turns
            # the opposite way to the cursor.
            step = -step
        if imgui.get_io().key_shift:
            step = round(step / 15.0) * 15.0
        doc.transform_floating(angle=angle0 + step, resample=state.resample)
    elif state.drag_kind == "move":
        last = state.last_point or point
        doc.move_floating(round(point[0] - last[0]), round(point[1] - last[1]))
        state.last_point = point


def _transform_box(state: Any, tab: Any, draw_list: Any, origin) -> None:
    buf = tab.doc.floating
    if buf is None:
        return
    handles = _handles(tab, origin)
    colour = _u32(theme.ACCENT)
    # The screen box of the four corners rather than the ``nw``/``se`` pair:
    # those names are image-space, and a turned page makes ``nw`` the bottom
    # right of what is on screen, which ``add_rect`` draws as nothing at all.
    a, b = _box(
        tab.view, origin,
        buf.offset[0], buf.offset[1],
        buf.offset[0] + buf.size[0], buf.offset[1] + buf.size[1],
    )
    draw_list.add_rect(a, b, colour)
    top = inker_state.to_screen(
        tab.view, origin, buf.offset[0] + buf.size[0] / 2.0, buf.offset[1]
    )
    draw_list.add_line(top, handles["rotate"], colour)
    for name, point in handles.items():
        if name == "rotate":
            draw_list.add_circle_filled(point, HANDLE, colour)
        else:
            draw_list.add_rect_filled(
                (point[0] - HANDLE, point[1] - HANDLE),
                (point[0] + HANDLE, point[1] + HANDLE),
                colour,
            )


def _combine_op() -> str:
    """Shift adds, Alt subtracts, both intersect -- the universal convention.

    The pair is checked *first*, or the Shift branch answers it and the fourth
    of ``selection.COMBINE_OPS`` stays unreachable, which is what it was.
    """
    io = imgui.get_io()
    if io.key_shift and io.key_alt:
        return "intersect"
    if io.key_shift:
        return "add"
    if io.key_alt:
        return "subtract"
    return "replace"


# The tools that read the document rather than write to it, so a content lock
# has nothing to say about them: the eyedropper samples, and the four selection
# tools build a mask that lives on the document rather than in a layer.
_READ_ONLY_TOOLS = frozenset({"eyedropper"}) | SELECT_TOOLS


def _locked_out(ctx: Any, state: Any, tab: Any) -> bool:
    """One toast per press when the active layer refuses tool-level writes.

    The toast lives here rather than at the engine's doors for the reason the
    engine refuses silently: ``write_colour`` is asked once per dab and once per
    preview frame, and this is the only place that knows a press is a press.
    """
    doc = tab.doc
    if state.tool in _READ_ONLY_TOOLS or not doc.write_locked():
        return False
    # Nudging a buffer that is already floating writes to no layer -- the hole
    # it came out of was cut before the lock went on, and ``commit_floating``
    # is deliberately not refused either.
    if state.tool == "move" and doc.floating is not None:
        return False
    ctx.toast("That layer is locked. Unlock it in the layers panel.", "warn")
    state.drag_kind = ""
    return True


def _press(ctx: Any, state: Any, tab: Any, point, origin=(0.0, 0.0)) -> None:
    doc = tab.doc
    tool = state.tool
    state.drag_anchor = point
    state.last_point = point
    state.combine = _combine_op()
    ipoint = (int(math.floor(point[0])), int(math.floor(point[1])))

    # Before every paint branch, and it returns rather than falling through:
    # the slice tool must never leave a dab behind, which is exactly what a
    # missing early-out looks like the first time somebody drags on the canvas.
    if tool == "slice":
        if state.drag_button == 0:
            _slice_press(ctx, state, tab, origin, point)
        else:
            # Inert on the right button, and said here rather than by the
            # ``BG_BUTTON_TOOLS`` check below because this branch returns above
            # it: the slice tool is one of the tools that reserve the button
            # instead of spending it, so a right-press must not start a slice.
            state.drag_kind = ""
        return

    # The text tool, on the slice arm's shape and for its reasons: it returns
    # before every paint branch so a click can never leave a dab behind, and it
    # is inert on the right button rather than spending it. The lock is asked
    # here rather than at the OK button -- offering a popup, a font list and a
    # size, and only then saying the layer is locked, is a form the app knew
    # the answer to before it drew it.
    if tool == "text":
        if state.drag_button == 0 and not _locked_out(ctx, state, tab):
            state.text_at = ipoint
            _open_text(state, tab)
        state.drag_kind = ""
        return

    # Alt over a paint tool picks the colour under the cursor, which is the one
    # convention a user coming from any other paint program reaches for without
    # thinking. Checked before the tool branches rather than inside the paint
    # one, so it reads as what it is: a modifier over the whole toolbox that the
    # paint tools happen to be the only ones with a free Alt for -- Alt already
    # subtracts on the selection tools and expands from centre on the shapes.
    if tool == "eyedropper" or (tool in PAINT_TOOLS and imgui.get_io().key_alt):
        picked = doc.eyedrop(ipoint, layer_only=state.sample_layer)
        if picked is not None:
            # Alt+right-click picks the *background* colour, which is the other
            # half of the right button painting with it: the two are one
            # gesture pair, and picking into fg from a right-click would make
            # the button mean two different things.
            if state.drag_button == 1:
                state.bg = picked
            else:
                state.fg = picked
        state.drag_kind = ""
        return
    if state.drag_button == 1 and tool not in BG_BUTTON_TOOLS:
        # Inert rather than a second meaning; see ``BG_BUTTON_TOOLS``.
        state.drag_kind = ""
        return
    # After the right-button check above, so a press that is inert anyway does
    # not toast about a lock it was never going to reach.
    if _locked_out(ctx, state, tab):
        return
    colour = state.bg if state.drag_button == 1 else state.fg
    if tool == "fill":
        doc.commit_floating()
        doc.fill(
            ipoint,
            colour,
            thresh=state.wand_tolerance,
            contiguous=state.wand_contiguous,
            wrap=tab.tiled,
        )
        state.drag_kind = ""
        return
    if tool == "wand":
        doc.commit_floating()
        doc.select_wand(
            ipoint,
            tolerance=state.wand_tolerance,
            op=state.combine,
            contiguous=state.wand_contiguous,
            wrap=tab.tiled,
        )
        state.drag_kind = ""
        return
    if tool == "move":
        if doc.floating is not None and doc.floating.contains(ipoint):
            state.drag_kind = "move"
        elif doc.mask is not None and doc.mask.contains(ipoint):
            doc.lift()
            state.drag_kind = "move"
        elif doc.begin_layer_move():
            # The third arm (C10): no buffer and no selection means "move what
            # is on this layer", which is what the move tool does everywhere
            # else and what this one used to answer with nothing at all.
            state.drag_kind = "layer_move"
        else:
            state.drag_kind = ""
        return
    if tool == "lasso_poly":
        # Before the shared selection branch and returning, because the whole
        # point of this tool is that it does *not* start a ``drag_kind="lasso"``
        # drag: its vertices are clicked, not dragged out.
        _poly_press(ctx, state, tab, point)
        return
    if tool in SELECT_TOOLS:
        doc.commit_floating()
        # An unmodified drag starting *inside* the selection moves its edges
        # rather than replacing them -- the marquee's own version of grabbing
        # what you can see. The modifier check comes first, so Shift and Alt
        # still start the combine drags they have always started: a user
        # Shift-dragging to add a second region routinely starts inside the
        # first one, and reading the modifiers second would have taken that
        # gesture away.
        if state.combine == "replace" and doc.mask is not None and doc.mask.contains(ipoint):
            state.drag_kind = "mask-move"
            return
        state.drag_kind = "lasso" if tool == "lasso" else "marquee"
        state.lasso = [point]
        return
    if tool in SHAPE_TOOLS:
        doc.commit_floating()
        state.drag_kind = "shape"
        return
    if tool == "gradient":
        doc.commit_floating()
        state.drag_kind = "gradient"
        return
    if tool in PAINT_TOOLS:
        # Said here rather than only in the tools panel: the panel greys the
        # button, but a shortcut key selects a tool without asking the panel
        # anything, so this is the door a shading press on a document with no
        # palette actually arrives at.
        refusal = inker_state.tool_reason(tool, doc)
        if refusal:
            ctx.toast(refusal, "warn")
            state.drag_kind = ""
            return
        doc.commit_floating()
        spraying = tool == "spray"
        # Asked once, and it decides two arguments rather than one; see
        # ``_press_mode`` for why the ink cannot be read independently of it.
        tip = state.tip_for(tool)
        doc.begin_stroke(
            point,
            colour,
            size=_dab_size(state, spraying),
            hardness=state.hardness,
            opacity=state.opacity,
            spacing=state.spacing,
            mode=_press_mode(state, tool, tip),
            strength=state.strength,
            nib=state.nib,
            # Both forced off for the spray: the corner filter is about a
            # *line* and there is no line here, and a lag on a stationary
            # airbrush would move the cloud away from the cursor.
            pixel_perfect=False if spraying else state.pixel_perfect,
            axis=state.symmetry_axis,
            radial=state.radial_count,
            stabilise=0.0 if spraying else state.stabilise,
            speed_taper=state.speed_taper,
            symmetry=state.symmetry,
            wrap=tab.tiled,
            scatter=state.brush_size / 2.0 if spraying else 0.0,
            # The determinism seam: the engine is a pure function of the seed
            # and the call sequence, and this is the one place entropy enters.
            seed=random.getrandbits(32) if spraying else 0,
            # Read at the press and carried for the whole stroke, so selecting
            # different slots mid-drag cannot change what the ramp is halfway
            # along -- and so nothing about the ramp reaches the document.
            ramp=shade_ramp(doc.palette, state.palette_slots) if tool == "shade" else (),
            shade_dir=state.shade_dir,
            # The captured tip, when this tool is set to use one. Asked through
            # ``tip_for`` rather than read off the two fields here, so the press
            # and the cursor outline drawn a frame earlier cannot disagree about
            # whether the picture is what is about to land.
            stamp=tip,
            stamp_align=state.stamp_align,
        )
        state.spray_carry = 0.0
        state.drag_kind = "spray" if spraying else "paint"


def _press_mode(state: Any, tool: str, tip: Any) -> str:
    """Which brush mode this press opens with. **The tip outranks the ink.**

    One line of policy, in a function of its own because it is the seam a real
    bug slipped through and because a test has to be able to pin it against
    ``tip_for`` directly.

    ``paint_ink`` is the brush's copy-colour toggle and it is the *only* thing
    that can send a mode other than ``BRUSH_MODES[tool]``. But ``replace`` is
    not a stamp mode -- a captured tip's alpha is both its shape and its
    transparency, so a copy ink has nothing left to say about one -- and
    ``StrokeState`` therefore drops a tip handed to it. Reading the ink
    independently of the tip made that a silent, unrecoverable failure: the
    panel hides the ink radio while a tip is loaded, the cursor draws the tip's
    box and the checkbox stays ticked, so the stroke stamped a round replace dab
    with nothing on screen to say why and no control left to turn the ink off.
    Two ordinary gestures reached it -- set Replace and then capture (Ctrl+B
    does not touch the ink), and apply a preset, since ``paint_ink`` is one of
    the options a preset carries.

    So ``tip_for`` is the single source of truth: whatever it advertises is what
    the press lays down, and a stale ink is ignored rather than obeyed. It is
    ignored rather than *cleared* because the tip is the transient thing --
    forget the tip and the ink the user chose is still theirs.
    """
    if tool == "brush" and tip is None and state.paint_ink == "replace":
        return "replace"
    return inker_state.BRUSH_MODES[tool]


def _dab_size(state: Any, spraying: bool) -> int:
    """The brush diameter one dab is stamped at.

    For every tool but the spray that is the size slider. For the spray the
    slider is the width of the *cloud* -- what the brush cursor draws and what
    "spray width" means in every other editor -- so the dab is a fraction of it
    (``inker_state.SPRAY_DAB_FRACTION``); a spray whose dabs are as wide as its
    own disc is a blob.
    """
    if not spraying:
        return state.brush_size
    return max(1, round(state.brush_size * inker_state.SPRAY_DAB_FRACTION))


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
# and polygon shape tools are this gesture with a different landing.

#: How near the first vertex a click has to land to close the polygon, in
#: **screen** pixels before ``sp`` -- the slice handles' rule, for the slice
#: handles' reason: an image-space radius is a hundredth of a pixel at 100x zoom
#: and half the canvas at 5%, so the target would be unhittable at one end of the
#: zoom range and unavoidable at the other.
POLY_CLOSE = 7.0


def closes_gesture(points, point, zoom: float, radius: float) -> bool:
    """Whether a click at ``point`` closes the polygon on its first vertex.

    Image-space distance times the zoom *is* the screen distance, because the
    view is a uniform scale after a quarter turn and a turn preserves length
    (``inker_state.basis`` is orthonormal). So this is quarter-turn and flip
    invariant without ever building a screen coordinate -- which is what lets it
    be a pure function of three numbers rather than of the view.

    Below three vertices there is nothing to close: two points are a line, and a
    click back on the first would otherwise end the gesture with a selection the
    rasteriser has to refuse anyway.
    """
    if len(points) < 3:
        return False
    return math.dist(points[0], point) * zoom <= radius


def _poly_press(ctx: Any, state: Any, tab: Any, point) -> None:
    """One click of the polygonal lasso: open, extend, or close the polygon.

    ``drag_kind`` is left empty on every arm, which is the load-bearing part:
    it keeps the next click out of the C12d guard, keeps ``_drag`` and
    ``_release`` out of the gesture entirely -- so this tool can never take the
    freehand ``drag_kind="lasso"`` path -- and keeps ``clear_drag`` free to mean
    "cancel the gesture" everywhere it is already called from.
    """
    doc = tab.doc
    if not state.gesture_pts:
        # The float lands on the first click, as it does for every other
        # selection tool's press: the user has moved on from it.
        doc.commit_floating()
        # Captured once, here. ``state.combine`` is re-read at every press, and
        # letting go of Shift before the closing click would otherwise turn an
        # add into a replace that throws the selection away.
        state.gesture_combine = state.combine
        state.gesture_pts = [point]
        state.drag_kind = ""
        return
    if imgui.is_mouse_double_clicked(state.drag_button) or closes_gesture(
        state.gesture_pts, point, tab.view.zoom, sp(POLY_CLOSE)
    ):
        # The closing click places no vertex of its own: a double-click's second
        # press lands on top of the first, and a click near vertex 0 *is*
        # vertex 0. Either would add a degenerate edge to the polygon.
        inker_mode.commit_gesture(state, tab)
    else:
        state.gesture_pts.append(point)
    state.drag_kind = ""


def _gesture_preview(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """The open polygon, its rubber band, and where the closing click goes.

    Through ``to_screen`` like every other overlay (Ink9), so the polygon is
    drawn on the turned or mirrored page rather than a quarter turn away from
    it. The rubber band follows the *snapped* cursor rather than the raw one,
    which is the rule ``_preview``'s shape branch already states: a band drawn
    somewhere the next vertex will not go is worse than no band at all.
    """
    points = state.gesture_pts
    if not points:
        return
    view = tab.view
    colour = _u32(theme.ACCENT)
    screen = [inker_state.to_screen(view, origin, x, y) for x, y in points]
    for a, b in zip(screen, screen[1:], strict=False):
        draw_list.add_line(a, b, colour)
    mouse = imgui.get_mouse_pos()
    tip = inker_state.to_screen(
        view,
        origin,
        *_snapped(state, _local(state, inker_state.to_image(view, origin, mouse.x, mouse.y))),
    )
    draw_list.add_line(screen[-1], tip, colour)
    if len(screen) >= 3:
        # The edge a commit would close with, and the target that closes it.
        # Fainter than the placed edges: it is what *would* happen, not what has.
        draw_list.add_line(tip, screen[0], _u32(theme.ACCENT, 0.4))
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
    ``text_open`` flag beside ``filter_open``.
    """
    if not imgui.begin_popup(TEXT_POPUP):
        return
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
    changed, value = imgui.checkbox("Antialias", bool(state.aa))
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

    imgui.dummy((0, 4))
    if imgui.button("OK##inkertext", (sp(90), 0)):
        # The tool becomes Move on success (``stamp_text``), so the popup must
        # close either way: a refusal has already toasted why.
        inker_mode.stamp_text(ctx, state, tab)
        imgui.close_current_popup()
    imgui.same_line()
    if imgui.button("Cancel##inkertext", (sp(90), 0)):
        imgui.close_current_popup()
    imgui.end_popup()


# --- slices -------------------------------------------------------------------
#
# A tool rather than a pane, so there is no new help anchor and no fourth
# sidebar: the overlay is on the canvas where the rectangles are, and the list,
# the toggles and the delete button ride the tools panel like every other tool's
# options do.
#
# The whole surface obeys the pane's two existing rules. Every screen position
# goes through ``_corners``/``_box`` rather than ``origin + x * zoom``, which is
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
            "nw": (x0, y0), "ne": (x1, y0), "sw": (x0, y1), "se": (x1, y1),
        }
        grab = sp(SLICE_HANDLE) * SLICE_GRAB
        for name, (cx, cy) in corners.items():
            if _near(inker_state.to_screen(tab.view, origin, cx, cy), at, grab):
                return "slice-resize", name
        if key.pivot is not None:
            pivot = inker_state.to_screen(
                tab.view, origin, x0 + key.pivot[0], y0 + key.pivot[1]
            )
            if _near(pivot, at, sp(SLICE_PIVOT_RADIUS) * SLICE_GRAB):
                return "slice-pivot", ""
        if key.center is not None:
            cx0, cy0, cx1, cy1 = key.center
            inner = {
                "nw": (x0 + cx0, y0 + cy0), "ne": (x0 + cx1, y0 + cy0),
                "sw": (x0 + cx0, y0 + cy1), "se": (x0 + cx1, y0 + cy1),
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

    Everything here is placed through ``_box``/``to_screen``, never through
    ``origin + x * zoom``: a quarter turn maps an axis-aligned image rectangle
    onto an axis-aligned *screen* rectangle, and that is only true of a position
    that has been through the view's basis.
    """
    frame_uid = tab.frame_uid
    outline = _u32(theme.ACCENT, 0.75)
    inner = _u32(theme.ACCENT, 0.45)
    hot = _u32(theme.ACCENT)
    for entry in tab.doc.slices:
        key = entry.at(frame_uid)
        x0, y0, x1, y1 = key.bounds
        selected = entry.uid == state.slice_uid
        a, b = _box(tab.view, origin, x0, y0, x1, y1)
        draw_list.add_rect(a, b, hot if selected else outline)
        if key.center is not None:
            cx0, cy0, cx1, cy1 = key.center
            ca, cb = _box(
                tab.view, origin, x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1
            )
            _dashed_rect(draw_list, ca, cb, inner)
        if key.pivot is not None:
            px, py = inker_state.to_screen(
                tab.view, origin, x0 + key.pivot[0], y0 + key.pivot[1]
            )
            radius = sp(SLICE_PIVOT_RADIUS)
            draw_list.add_circle((px, py), radius, hot if selected else outline)
            draw_list.add_line((px - radius, py), (px + radius, py), hot)
            draw_list.add_line((px, py - radius), (px, py + radius), hot)
        if not selected:
            continue
        size = sp(SLICE_HANDLE)
        for cx, cy in _corners(tab.view, origin, x0, y0, x1, y1):
            draw_list.add_rect_filled(
                (cx - size, cy - size), (cx + size, cy + size), hot
            )


def _drag(state: Any, tab: Any, point) -> None:
    doc = tab.doc
    if state.drag_kind.startswith("slice-"):
        _slice_drag(state, tab, point)
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
    io = imgui.get_io()
    return inker_state.shape_endpoints(
        state.tool, anchor, point, constrain=io.key_shift, from_centre=io.key_alt
    )


def marquee_rect(anchor, point) -> tuple[int, int, int, int]:
    """The pixel rectangle a marquee drag covers, whichever way it was drawn.

    Ordering the corners has to come *before* rounding them. Flooring the
    anchor and ceiling the release point rounds outward only while the drag
    runs down and to the right; reverse the drag and the same two rules round
    inward on both corners, so an identical gesture selects a smaller rectangle
    depending on the direction it was made in.
    """
    x0, x1 = sorted((float(anchor[0]), float(point[0])))
    y0, y1 = sorted((float(anchor[1]), float(point[1])))
    return (
        int(math.floor(x0)),
        int(math.floor(y0)),
        int(math.ceil(x1)),
        int(math.ceil(y1)),
    )


def _release(ctx: Any, state: Any, tab: Any, point) -> None:
    from ..inker import SelectionMask

    doc = tab.doc
    anchor = state.drag_anchor or point
    kind = state.drag_kind

    if kind.startswith("slice-"):
        _slice_release(ctx, state, tab, point)
        state.clear_drag()
        return
    if kind in ("paint", "spray"):
        doc.end_stroke()
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
        )
    elif kind == "marquee":
        rect = marquee_rect(anchor, point)
        if rect[2] > rect[0] and rect[3] > rect[1]:
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
        dx, dy = _mask_shift(state, point)
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
    state.clear_drag()


# --- drawing ----------------------------------------------------------------

#: Onion tints. Red behind, green ahead -- the convention every 2D animation
#: tool has used for decades, so picking differently would be a novelty the user
#: has to learn in exchange for nothing.
ONION_BACK = 0xE05050
ONION_FORWARD = 0x50C060


def _onion(ctx: Any, state: Any, tab: Any, draw_list, view, origin, size) -> None:
    """Neighbouring frames, tinted and faded, beneath the live one.

    Furthest first so the nearest neighbour ends up on top, and each step
    fainter than the last: the point is to see where a drawing *came from*, and
    three equally solid ghosts are just a mess.
    """
    anim = tab.doc.anim
    if anim is None:
        return
    current = anim.current
    for offset in range(max(state.onion_before, state.onion_after), 0, -1):
        for delta, colour in ((-offset, ONION_BACK), (offset, ONION_FORWARD)):
            limit = state.onion_before if delta < 0 else state.onion_after
            index = current + delta
            if offset > limit or not 0 <= index < len(anim.frames):
                continue
            texture = inker_textures.frame_texture(ctx, tab, anim.frames[index].uid)
            if texture is None:
                continue
            fade = state.onion_alpha / offset
            _blit(
                draw_list, texture, view, origin, 0, 0, size[0], size[1],
                colour=_u32(colour, fade),
            )


def _playback_frame(ctx: Any, tab: Any, draw_list, view, origin, size) -> None:
    anim = tab.doc.anim
    if anim is None or not anim.frames:
        return
    index = max(0, min(tab.play_index, len(anim.frames) - 1))
    texture = inker_textures.frame_texture(ctx, tab, anim.frames[index].uid)
    if texture is not None:
        _blit(draw_list, texture, view, origin, 0, 0, size[0], size[1])


def _paint(ctx: Any, state: Any, tab: Any, origin, *, hovered: bool) -> None:
    doc = tab.doc
    view = tab.view
    draw_list = imgui.get_window_draw_list()
    width, height = doc.size
    # The canvas's screen AABB, which under a quarter turn is still exactly the
    # canvas rather than a box around it -- see ``_box``.
    top_left, bottom_right = _box(view, origin, 0, 0, width, height)
    # And the neighbourhood the tiled view shows: one tile out along each
    # wrapped axis, the canvas itself along the others. Every overlay below
    # stays on the canonical box -- overlays are canonical-only in v1 -- but the
    # checkerboard is the surface the page sits on, so it extends with it.
    tiled = _tiled_extent(tab, doc.size)
    tiled_tl, tiled_br = _box(view, origin, *tiled)

    _checkerboard(ctx, draw_list, tiled_tl, tiled_br)
    # Before the composite and before its ``None`` early-out, so the strip is
    # genuinely beneath the live drawing rather than sometimes instead of it.
    if state.onion and not tab.playing:
        _onion(ctx, state, tab, draw_list, view, origin, doc.size)

    if tab.playing:
        # The cached flatten of the frame being played, not the document's
        # composite: the playhead on the document deliberately does not move
        # during playback, so the composite is still frame one.
        _playback_frame(ctx, tab, draw_list, view, origin, doc.size)
        draw_list.add_rect(top_left, bottom_right, _u32(theme.EDGE))
        return

    texture = inker_textures.composite(ctx, tab, nearest=view.zoom >= 1.0)
    if texture is None:
        return
    axes = axes_of(tab.tiled)
    # Set on **both** branches, every frame. Turning tiling on used to be a
    # one-way door in the reference viewer for exactly this reason: the sampler
    # was switched to GL_REPEAT and never put back, so the single-tile view that
    # followed sampled a wrapped texture at its own edges -- which is the one
    # place a seamless tile is not seamless, since LINEAR filtering there blends
    # the far edge in. moderngl skips the GL call when the value already
    # matches, so the idempotent write costs nothing.
    texture.repeat_x, texture.repeat_y = axes[0], axes[1]
    x0, y0, x1, y1 = tiled
    _blit(
        draw_list, texture, view, origin, x0, y0, x1, y1,
        uv0=(x0 / width, y0 / height), uv1=(x1 / width, y1 / height),
    )
    _floating(ctx, tab, draw_list, origin)
    draw_list.add_rect(top_left, bottom_right, _u32(theme.EDGE))

    if state.grid:
        _grid(state, draw_list, view, origin, doc.size, top_left, bottom_right)
    if state.symmetry != "none":
        _symmetry(state, draw_list, view, origin, doc.size)
    _ants(ctx, tab, draw_list, origin, state)
    if slices_visible(state):
        _slices(state, tab, draw_list, origin)
    if state.transforming:
        _transform_box(state, tab, draw_list, origin)
    _preview(state, tab, draw_list, origin)
    # Beside the drag preview rather than inside it: a multi-click gesture holds
    # no ``drag_kind``, which is the first thing ``_preview`` returns on.
    _gesture_preview(state, tab, draw_list, origin)
    if hovered and state.tool in PAINT_TOOLS:
        _cursor(state, draw_list, view)


def _tiled_extent(tab: Any, size) -> tuple[float, float, float, float]:
    """The image-space rectangle the canvas draws, in canvas coordinates.

    The canvas itself with tiling off, and one tile out along each *wrapped*
    axis with it on -- so X-only tiling shows a horizontal strip of three
    rather than a 3x3 block, which is the honest picture of what will actually
    wrap when you paint on it.
    """
    width, height = size
    wrap_x, wrap_y = axes_of(tab.tiled)
    x0, x1 = (-width, 2 * width) if wrap_x else (0, width)
    y0, y1 = (-height, 2 * height) if wrap_y else (0, height)
    return (x0, y0, x1, y1)


def _checkerboard(ctx: Any, draw_list: Any, top_left, bottom_right) -> None:
    """The transparency backdrop, drawn in **screen** space and deliberately not
    put through the view's turn.

    It is the surface the page sits on rather than part of the page: rotating it
    with the canvas would make the squares spin, which says nothing and looks
    like a bug. Under a quarter turn the canvas's screen AABB *is* the canvas,
    so an upright ``add_image`` over that box covers exactly what it should.
    """
    texture = inker_textures.checker(ctx)
    if texture is None:
        return
    # UVs in tile units, so one draw call covers the canvas at any zoom and the
    # squares stay a constant size on screen rather than in image space.
    span = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    tile = texture.size[0]
    draw_list.add_image(
        widgets.texture_ref(texture),
        top_left,
        bottom_right,
        (0, 0),
        (span[0] / tile, span[1] / tile),
    )


def _floating(ctx: Any, tab: Any, draw_list: Any, origin) -> None:
    buf = tab.doc.floating
    if buf is None:
        return
    texture = inker_textures.floating(ctx, tab, nearest=tab.view.zoom >= 1.0)
    if texture is None:
        return
    x, y = buf.offset
    fw, fh = buf.size
    _blit(draw_list, texture, tab.view, origin, x, y, x + fw, y + fh)
    a, b = _box(tab.view, origin, x, y, x + fw, y + fh)
    draw_list.add_rect(a, b, _u32(theme.ACCENT))


def _grid(state: Any, draw_list: Any, view: Any, origin, size, top_left, bottom_right) -> None:
    """Clipped to the visible rectangle: a 16-pixel grid over a 4096 canvas is
    a quarter of a million lines if it is drawn in full."""
    colour = _u32(theme.EDGE, 0.55)
    step = max(1, int(state.grid_size))
    # The line below the step is the real floor: six screen pixels between
    # grid lines. (There was a ``GRID_MIN_ZOOM`` branch here that computed
    # ``min(step, step)`` -- a no-op guarding a per-pixel grid that was never
    # implemented.)
    if step * view.zoom < 6.0:
        return
    width, height = size
    # Only the visible index range (B22): the canvas span intersected with
    # the window, turned back into grid indices. A 16 px grid over a zoomed
    # 4096 canvas used to walk all 257 columns to draw the dozen on screen.
    #
    # The range is derived by inverse-transforming the visible rectangle's
    # corners rather than by dividing a screen offset by the zoom, which is the
    # form that only worked while ``top_left`` was the image's origin: under a
    # quarter turn it is a different corner of the canvas, so the old
    # subtraction produced a range that was correct at rotation 0 and empty
    # (or reversed) at every other.
    win = imgui.get_window_pos()
    wsz = imgui.get_window_size()
    left = max(top_left[0], win.x)
    right = min(bottom_right[0], win.x + wsz.x)
    top = max(top_left[1], win.y)
    bottom = min(bottom_right[1], win.y + wsz.y)
    if left > right or top > bottom:
        return
    seen = [
        inker_state.to_image(view, origin, sx, sy)
        for sx in (left, right)
        for sy in (top, bottom)
    ]
    lo_x = max(0, int(min(p[0] for p in seen) / step) * step)
    hi_x = min(width, int(max(p[0] for p in seen)) + step)
    lo_y = max(0, int(min(p[1] for p in seen) / step) * step)
    hi_y = min(height, int(max(p[1] for p in seen)) + step)
    # An image-space line still lands as an axis-aligned screen line, because
    # the orientation is a quarter turn -- but which screen axis it lands on
    # swaps, so both endpoints are transformed rather than one coordinate being
    # borrowed from the canvas box.
    for x in range(lo_x, hi_x + 1, step):
        a = inker_state.to_screen(view, origin, x, 0)
        b = inker_state.to_screen(view, origin, x, height)
        draw_list.add_line(a, b, colour)
    for y in range(lo_y, hi_y + 1, step):
        a = inker_state.to_screen(view, origin, 0, y)
        b = inker_state.to_screen(view, origin, width, y)
        draw_list.add_line(a, b, colour)


def _symmetry(state: Any, draw_list: Any, view: Any, origin, size) -> None:
    """The guide, drawn where the engine actually reflects.

    It used to draw at ``width / 2`` and ``height / 2`` unconditionally, which
    was wrong twice over: ``brush._mirror`` reflects about ``(width - 1) / 2``
    by default, and it honours ``state.symmetry_axis`` when the user has moved
    it -- so a moved axis left the line pointing at the middle of the page while
    the strokes came out somewhere else. ``brush.axis_or_default`` is now the
    one answer both read. Radial had no guide at all and now gets the pivot,
    which is the only thing there is to show for it: its reflections are turns
    rather than lines.
    """
    from ..inker import brush

    width, height = size
    colour = _u32(theme.ACCENT, 0.6)
    ax, ay = brush.axis_or_default((int(width), int(height)), state.symmetry_axis)
    # Both endpoints through ``to_screen``, for ``_grid``'s reason: the axis a
    # mirror line lands on swaps with the page, so borrowing one coordinate
    # from a corner would draw it across the canvas the wrong way.
    if state.symmetry in ("x", "xy"):
        a = inker_state.to_screen(view, origin, ax, 0)
        b = inker_state.to_screen(view, origin, ax, height)
        draw_list.add_line(a, b, colour)
    if state.symmetry in ("y", "xy"):
        a = inker_state.to_screen(view, origin, 0, ay)
        b = inker_state.to_screen(view, origin, width, ay)
        draw_list.add_line(a, b, colour)
    if state.symmetry == "radial":
        centre = inker_state.to_screen(view, origin, ax, ay)
        radius = sp(SYMMETRY_PIVOT_RADIUS)
        draw_list.add_circle(centre, radius, colour)
        draw_list.add_line(
            (centre[0] - radius, centre[1]), (centre[0] + radius, centre[1]), colour
        )
        draw_list.add_line(
            (centre[0], centre[1] - radius), (centre[0], centre[1] + radius), colour
        )


def _mask_shift(state: Any, point: Any = None) -> tuple[int, int]:
    """Whole pixels a mask-move drag has travelled, or ``(0, 0)``.

    One function for the preview and for the release, so what the ants show
    while the drag runs and what ``select`` is handed when it ends cannot
    disagree by a pixel. Measured against the *press* rather than accumulated
    per frame, which is the rule the transform handles already follow.
    """
    if state.drag_kind != "mask-move" or state.drag_anchor is None:
        return (0, 0)
    tip = point if point is not None else state.last_point
    if tip is None:
        return (0, 0)
    return (
        int(round(tip[0] - state.drag_anchor[0])),
        int(round(tip[1] - state.drag_anchor[1])),
    )


def _contours(ctx: Any, tab: Any):
    """The selection outline, recomputed only when the mask object changes.

    ``select()`` always builds a new mask, so identity is a sound cache key --
    and it has to be one, because the document's revision ticks on every dab
    and tracing the boundary per stroke frame would be the most expensive thing
    on screen.

    What is cached is what :mod:`~warlock.studio.ants` prepared rather than the
    raw lattice points: the vertex arrays and the cumulative arc lengths are a
    pure function of the mask too, so measuring the perimeter belongs on the
    same side of this cache as tracing the boundary does.
    """
    key = f"paint_ants:{tab.uid}"
    cached = ctx.state.preview.get(key)
    mask = tab.doc.mask
    if cached is not None and cached[0] is mask:
        return cached[1]
    prepared = ants.prepare(mask.contours() if mask is not None else [])
    # The canvas-space AABB rides beside each loop (B23): a pure function of
    # the mask too, so it belongs on this side of the cache.
    loops = [(verts, cum, ants.loop_box(verts)) for verts, cum in prepared]
    ctx.state.preview[key] = (mask, loops)
    return loops


def _ants(ctx: Any, tab: Any, draw_list: Any, origin, state: Any = None) -> None:
    loops = _contours(ctx, tab)
    if not loops:
        return
    view = tab.view
    # A mask-move drag previews by sliding the *drawn* outline and nothing
    # else: the mask itself is untouched until the release, so the preview
    # costs no recompute of the contours and pushes no history. Expressed as an
    # image-space origin rather than a screen delta, so it goes through the
    # view's turn with everything else.
    shift = (0, 0) if state is None else _mask_shift(state)
    # Canvas (0, 0) on screen, from the same function every other overlay uses:
    # ``to_screen`` is a uniform scale plus this offset, and a second spelling
    # of it is how the ants end up one pixel off the mask they describe.
    offset = inker_state.to_screen(view, origin, float(shift[0]), float(shift[1]))
    phase = (time.monotonic() * ants.ANT_SPEED) % (ants.DASH * 2)
    light, dark = _u32(theme.TEXT), _u32(theme.BG)
    # The visible window, for the two culls below (B23): a whole loop whose
    # box misses it is skipped before any dash arithmetic, and the runs of a
    # loop that straddles it are masked before the per-run Python loop.
    win = imgui.get_window_pos()
    wsz = imgui.get_window_size()
    clip = (win.x, win.y, win.x + wsz.x, win.y + wsz.y)
    rows = inker_state.basis(view)
    matrix = np.asarray(rows, dtype=np.float64)
    for verts, cum, box in loops:
        # The loop's canvas box, put on screen through the *same* orientation
        # the dashes go through. Under a quarter turn the transformed corners
        # are still two opposite corners of an axis-aligned box, so a min/max
        # over the four is exact rather than conservative -- but the old form,
        # which scaled the box's own coordinates, silently culled every loop the
        # moment the page was turned.
        corners = [
            inker_state.to_screen(view, origin, x + shift[0], y + shift[1])
            for x in (box[0], box[2])
            for y in (box[1], box[3])
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        if min(xs) > clip[2] or max(xs) < clip[0] or min(ys) > clip[3] or max(ys) < clip[1]:
            continue
        starts, ends, on = ants.dash_segments(
            verts, cum, view.zoom, offset, phase, basis=matrix
        )
        starts, ends, on = ants.cull(starts, ends, on, clip)
        # One call per dash. imgui has no batched per-segment-colour API, so
        # this loop is irreducible -- but it is over dashes now rather than over
        # every unit lattice step, which at zoom 1 is six times fewer.
        for (ax, ay), (bx, by), lit in zip(
            starts.tolist(), ends.tolist(), on.tolist(), strict=True
        ):
            draw_list.add_line((ax, ay), (bx, by), light if lit else dark)


def _preview(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """What the current drag would produce, before it produces it."""
    if state.drag_anchor is None or not state.drag_kind:
        return
    view = tab.view
    mouse = imgui.get_mouse_pos()
    anchor = inker_state.to_screen(view, origin, *state.drag_anchor)
    tip = (mouse.x, mouse.y)
    colour = _u32(theme.ACCENT)
    kind, tool = state.drag_kind, state.tool

    if kind == "slice-new":
        # The rectangle a release would add. The other four slice drags need no
        # preview at all: they move the live slice, so ``_slices`` is already
        # drawing exactly what the release will keep.
        draw_list.add_rect(anchor, tip, colour)
        return
    if kind.startswith("slice-"):
        return

    if kind == "shape":
        # Through the same function the release goes through, and back to
        # screen: a preview drawn from the raw cursor while the commit applies a
        # constraint is a picture of a shape the user is not about to get. The
        # snap goes through here too, for the same reason -- the release reads a
        # snapped point and this used to read the cursor.
        p0, p1 = _shape_drag(
            state,
            state.drag_anchor,
            # Through ``_local`` as well, because the anchor already is: a
            # preview drawn a tile away from the shape the release will commit
            # is worse than no preview at all.
            _snapped(
                state,
                _local(state, inker_state.to_image(view, origin, mouse.x, mouse.y)),
            ),
        )
        anchor = inker_state.to_screen(view, origin, *p0)
        tip = inker_state.to_screen(view, origin, *p1)

    if kind == "lasso" and len(state.lasso) > 1:
        points = [inker_state.to_screen(view, origin, x, y) for x, y in state.lasso]
        for a, b in zip(points, points[1:], strict=False):
            draw_list.add_line(a, b, colour)
        draw_list.add_line(points[-1], tip, colour)
    elif kind == "gradient":
        draw_list.add_line(anchor, tip, colour, 2.0)
    elif kind == "marquee" and tool == "select_ellipse":
        _ellipse(draw_list, anchor, tip, colour)
    elif kind == "marquee" or (kind == "shape" and tool == "rect"):
        draw_list.add_rect(anchor, tip, colour)
    elif kind == "shape" and tool == "line":
        draw_list.add_line(anchor, tip, colour)
    elif kind == "shape" and tool == "ellipse":
        _ellipse(draw_list, anchor, tip, colour)


def _ellipse(draw_list: Any, a, b, colour: int) -> None:
    centre = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
    radii = (abs(b[0] - a[0]) * 0.5, abs(b[1] - a[1]) * 0.5)
    if radii[0] > 0.5 and radii[1] > 0.5:
        draw_list.add_ellipse(centre, radii, colour)


def _cursor(state: Any, draw_list: Any, view: Any) -> None:
    """A circle the size of the brush. The one piece of feedback that makes a
    variable-size brush usable at all."""
    mouse = imgui.get_mouse_pos()
    colour = _u32(theme.TEXT, 0.7)
    tip = state.tip_for(state.tool)
    if tip is not None:
        # The tip's own box, not the size slider's circle: an image brush is not
        # round and is not the slider's size, so the ring would be a picture of
        # a brush that is not the one in hand. Drawn at the *variant's* size, so
        # a quarter turn of a tall stamp shows a wide box -- which is what will
        # land.
        width, height = tip.size
        half_w, half_h = width * 0.5 * view.zoom, height * 0.5 * view.zoom
        draw_list.add_rect(
            (mouse.x - half_w, mouse.y - half_h),
            (mouse.x + half_w, mouse.y + half_h),
            colour,
        )
        return
    radius = max(2.0, state.brush_size * 0.5 * view.zoom)
    if state.nib == "square":
        draw_list.add_rect(
            (mouse.x - radius, mouse.y - radius), (mouse.x + radius, mouse.y + radius), colour
        )
        return
    draw_list.add_circle((mouse.x, mouse.y), radius, colour)
    # A pixel nib has no falloff, so it has no inner ring to draw -- and the
    # ring is read as "this is where it is solid", which for a hard-edged nib
    # would be a picture of a softness it does not have.
    if state.nib == "soft" and state.hardness < 0.99:
        inner = radius * max(state.hardness, 0.05)
        draw_list.add_circle((mouse.x, mouse.y), inner, _u32(theme.TEXT, 0.25))
