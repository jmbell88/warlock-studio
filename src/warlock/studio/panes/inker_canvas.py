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
import time
from typing import Any

import numpy as np
from imgui_bundle import imgui

from .. import ants, icons, inker_mode, inker_state, theme, widgets
from ..inker_state import PAINT_TOOLS, SELECT_TOOLS, SHAPE_TOOLS
from ..tokens import sp
from . import inker_textures


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
    order the positions are in.
    """
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
        doc.transform_floating(angle=angle)
    imgui.same_line()
    imgui.set_next_item_width(sp(160))
    changed, factor = imgui.slider_float("Scale", buf.scale[0], 0.05, 8.0)
    if changed:
        doc.transform_floating(scale=(factor, factor))
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
        imgui.invisible_button("##inker-surface", region)
        active = imgui.is_item_active()
        hovered = imgui.is_item_hovered()
        _input(ctx, state, tab, (origin.x, origin.y), active=active, hovered=hovered)
        _paint(ctx, state, tab, (origin.x, origin.y), hovered=hovered)
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
        return
    if state.transforming and tab.doc.floating is not None:
        _transform_input(state, tab, origin, point, active=active)
        return
    if active and imgui.is_mouse_clicked(0) and not state.space_held:
        _press(ctx, state, tab, _snapped(state, point))
    elif state.drag_kind and imgui.is_mouse_down(0):
        _drag(state, tab, _snapped(state, point))
    elif state.drag_kind and not imgui.is_mouse_down(0):
        _release(ctx, state, tab, _snapped(state, point))


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
        state.transform_ref = (
            buf.scale[0],
            buf.angle,
            max(1.0, math.dist(centre, (mouse.x, mouse.y))),
            math.degrees(math.atan2(mouse.y - centre[1], mouse.x - centre[0])),
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

    scale0, angle0, dist0, bearing0 = state.transform_ref
    if state.drag_kind == "scale":
        ratio = math.dist(centre, (mouse.x, mouse.y)) / dist0
        doc.transform_floating(scale=(scale0 * ratio, scale0 * ratio))
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
        doc.transform_floating(angle=angle0 + step)
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
    """Shift adds to the selection, Alt subtracts -- the universal convention."""
    io = imgui.get_io()
    if io.key_shift:
        return "add"
    if io.key_alt:
        return "subtract"
    return "replace"


def _press(ctx: Any, state: Any, tab: Any, point) -> None:
    doc = tab.doc
    tool = state.tool
    state.drag_anchor = point
    state.last_point = point
    state.combine = _combine_op()
    ipoint = (int(math.floor(point[0])), int(math.floor(point[1])))

    if tool == "eyedropper":
        picked = doc.eyedrop(ipoint, layer_only=state.sample_layer)
        if picked is not None:
            state.fg = picked
        state.drag_kind = ""
        return
    if tool == "fill":
        doc.commit_floating()
        doc.fill(ipoint, state.fg, thresh=state.wand_tolerance)
        state.drag_kind = ""
        return
    if tool == "wand":
        doc.commit_floating()
        doc.select_wand(
            ipoint,
            tolerance=state.wand_tolerance,
            op=state.combine,
            contiguous=state.wand_contiguous,
        )
        state.drag_kind = ""
        return
    if tool == "move":
        if doc.floating is not None and doc.floating.contains(ipoint):
            state.drag_kind = "move"
        elif doc.mask is not None and doc.mask.contains(ipoint):
            doc.lift()
            state.drag_kind = "move"
        else:
            state.drag_kind = ""
        return
    if tool in SELECT_TOOLS:
        doc.commit_floating()
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
        doc.commit_floating()
        doc.begin_stroke(
            point,
            state.fg,
            size=state.brush_size,
            hardness=state.hardness,
            opacity=state.opacity,
            spacing=state.spacing,
            mode=inker_state.BRUSH_MODES[tool],
            strength=state.strength,
            axis=state.symmetry_axis,
            radial=state.radial_count,
            stabilise=state.stabilise,
            speed_taper=state.speed_taper,
            symmetry=state.symmetry,
        )
        state.drag_kind = "paint"


def _drag(state: Any, tab: Any, point) -> None:
    doc = tab.doc
    if state.drag_kind == "paint":
        doc.stroke_to(point)
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

    if kind == "paint":
        doc.end_stroke()
    elif kind == "shape":
        doc.shape(
            state.tool,
            (int(anchor[0]), int(anchor[1])),
            (int(point[0]), int(point[1])),
            state.fg,
            state.brush_size,
            filled=state.shape_filled,
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
    elif kind == "lasso":
        if len(state.lasso) >= 3:
            doc.select(SelectionMask.from_polygon(doc.size, state.lasso), state.combine)
        else:
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

    _checkerboard(ctx, draw_list, top_left, bottom_right)
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
    _blit(draw_list, texture, view, origin, 0, 0, width, height)
    _floating(ctx, tab, draw_list, origin)
    draw_list.add_rect(top_left, bottom_right, _u32(theme.EDGE))

    if state.grid:
        _grid(state, draw_list, view, origin, doc.size, top_left, bottom_right)
    if state.symmetry != "none":
        _symmetry(state, draw_list, view, origin, doc.size)
    _ants(ctx, tab, draw_list, origin)
    if state.transforming:
        _transform_box(state, tab, draw_list, origin)
    _preview(state, tab, draw_list, origin)
    if hovered and state.tool in PAINT_TOOLS:
        _cursor(state, draw_list, view)


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
    width, height = size
    colour = _u32(theme.ACCENT, 0.6)
    # Both endpoints through ``to_screen``, for ``_grid``'s reason: the axis a
    # mirror line lands on swaps with the page, so borrowing one coordinate
    # from a corner would draw it across the canvas the wrong way.
    if state.symmetry in ("x", "xy"):
        a = inker_state.to_screen(view, origin, width / 2, 0)
        b = inker_state.to_screen(view, origin, width / 2, height)
        draw_list.add_line(a, b, colour)
    if state.symmetry in ("y", "xy"):
        a = inker_state.to_screen(view, origin, 0, height / 2)
        b = inker_state.to_screen(view, origin, width, height / 2)
        draw_list.add_line(a, b, colour)


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


def _ants(ctx: Any, tab: Any, draw_list: Any, origin) -> None:
    loops = _contours(ctx, tab)
    if not loops:
        return
    view = tab.view
    # Canvas (0, 0) on screen, from the same function every other overlay uses:
    # ``to_screen`` is a uniform scale plus this offset, and a second spelling
    # of it is how the ants end up one pixel off the mask they describe.
    offset = inker_state.to_screen(view, origin, 0.0, 0.0)
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
            inker_state.to_screen(view, origin, x, y)
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
    colour = _u32(theme.ACCENT)
    kind, tool = state.drag_kind, state.tool

    if kind == "lasso" and len(state.lasso) > 1:
        points = [inker_state.to_screen(view, origin, x, y) for x, y in state.lasso]
        for a, b in zip(points, points[1:], strict=False):
            draw_list.add_line(a, b, colour)
        draw_list.add_line(points[-1], (mouse.x, mouse.y), colour)
    elif kind == "gradient":
        draw_list.add_line(anchor, (mouse.x, mouse.y), colour, 2.0)
    elif kind == "marquee" and tool == "select_ellipse":
        _ellipse(draw_list, anchor, (mouse.x, mouse.y), colour)
    elif kind == "marquee" or (kind == "shape" and tool == "rect"):
        draw_list.add_rect(anchor, (mouse.x, mouse.y), colour)
    elif kind == "shape" and tool == "line":
        draw_list.add_line(anchor, (mouse.x, mouse.y), colour)
    elif kind == "shape" and tool == "ellipse":
        _ellipse(draw_list, anchor, (mouse.x, mouse.y), colour)


def _ellipse(draw_list: Any, a, b, colour: int) -> None:
    centre = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
    radii = (abs(b[0] - a[0]) * 0.5, abs(b[1] - a[1]) * 0.5)
    if radii[0] > 0.5 and radii[1] > 0.5:
        draw_list.add_ellipse(centre, radii, colour)


def _cursor(state: Any, draw_list: Any, view: Any) -> None:
    """A circle the size of the brush. The one piece of feedback that makes a
    variable-size brush usable at all."""
    mouse = imgui.get_mouse_pos()
    radius = max(2.0, state.brush_size * 0.5 * view.zoom)
    draw_list.add_circle((mouse.x, mouse.y), radius, _u32(theme.TEXT, 0.7))
    if state.hardness < 0.99:
        inner = radius * max(state.hardness, 0.05)
        draw_list.add_circle((mouse.x, mouse.y), inner, _u32(theme.TEXT, 0.25))
