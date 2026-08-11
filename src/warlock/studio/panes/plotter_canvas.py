"""Plotter's centre pane: the map, drawn cell by cell.

**One textured quad per *visible* cell, through imgui's draw list.** No
composited image exists anywhere in the mode -- a 200x200 map at 32px is a
40-megapixel RGBA buffer, and recompositing it per frame is not something that
happens between ``new_frame`` and ``render``. The flat renderer in
``plotter/render.py`` builds one, and only an export ever asks it to.

The quads are ``add_image_quad`` rather than ``add_image``, and that is what
makes the diagonal flip drawable at all: a transpose cannot be expressed by
swapping a UV pair, but it can by permuting four corners. The permutations here
are the same transpose-then-mirror order ``render.orient`` applies, which is why
what you paint and what you export agree.

Draw-only. Every decision -- which tool, which tile, which layer -- belongs to
one of the three panes around this, and this one turns a click into a call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import inker_state, plotter_mode, theme, widgets
from ..plotter import gid as gidlib
from ..plotter import tools as plotter_tools
from ..plotter.tilemap import ObjectLayer, TileLayer
from ..tokens import sp
from . import plotter_layers, plotter_textures

# Below this many pixels per tile the grid stops being drawn: at two pixels a
# cell it is a solid wash rather than a guide, and it costs one line per column.
MIN_GRID_PX = 6


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    _tabs(ctx, state)
    tab = state.active
    if tab is None:
        _empty(ctx)
        return

    doc = tab.doc
    view = tab.view
    avail = imgui.get_content_region_avail()
    region = (max(float(avail.x), 1.0), max(float(avail.y), 1.0))
    size_px = (doc.pixel_width, doc.pixel_height)

    if not view.fitted:
        inker_state.fit(view, size_px, region)
    if view.pending_zoom is not None:
        inker_state.centre(view, size_px, region, view.pending_zoom)
        view.pending_zoom = None

    origin = imgui.get_cursor_screen_pos()
    imgui.invisible_button("plotter-canvas", region)
    hovered = imgui.is_item_hovered()
    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect(
        (origin.x, origin.y), (origin.x + region[0], origin.y + region[1]), True
    )
    _backdrop(draw_list, view, (origin.x, origin.y), size_px)
    _layers(ctx, tab, draw_list, (origin.x, origin.y), region)
    if state.grid:
        _grid(draw_list, doc, view, (origin.x, origin.y), size_px)
    if state.show_objects:
        _objects(state, doc, draw_list, view, (origin.x, origin.y))
    _cursor(state, tab, draw_list, (origin.x, origin.y), hovered)
    draw_list.pop_clip_rect()

    _events(ctx, state, tab, (origin.x, origin.y), hovered)
    _status(ctx, state, tab)


# --- chrome -------------------------------------------------------------------


def _tabs(ctx: Any, state: Any) -> None:
    from imgui_bundle import imgui

    if not state.docs:
        return
    # ``auto_select_new_tabs`` for ``inker_canvas``'s reason: without it,
    # opening a second document adds its tab and leaves the *first* one
    # focused, so "Open" appears to do nothing until the user notices the new
    # tab and clicks it.
    flags = (
        imgui.TabBarFlags_.reorderable.value
        | imgui.TabBarFlags_.auto_select_new_tabs.value
    )
    if imgui.begin_tab_bar("plotter-tabs", flags):
        for tab in list(state.docs):
            # imgui's own dot, not a ``"* "`` prefix -- see ``inker_canvas``.
            item_flags = imgui.TabItemFlags_.unsaved_document.value if tab.dirty else 0
            opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
            if opened:
                state.activate(tab.uid)
                imgui.end_tab_item()
            if not keep:
                plotter_mode.close_tab(ctx, tab.uid)
        imgui.end_tab_bar()


def _empty(ctx: Any) -> None:
    from imgui_bundle import imgui

    imgui.dummy((0, 40))
    imgui.text("Nothing open")
    widgets.muted_wrapped("Start a map, open one, or drop a .wmap / .tmx / .tmj on the window.")
    imgui.dummy((0, 16))
    if imgui.button("New map", (240, 0)):
        plotter_mode.new_document(ctx)
    imgui.dummy((0, 8))
    if imgui.button("Open a file...", (240, 0)):
        plotter_mode.ask_open(ctx)


def _status(ctx: Any, state: Any, tab: Any) -> None:
    from imgui_bundle import imgui

    doc = tab.doc
    layer = doc.active()
    name = layer.name if layer is not None else "no layer"
    widgets.muted(
        f"{doc.width} x {doc.height}  --  {int(tab.view.zoom * 100)}%  --  {name}"
        + ("  --  saving" if tab.busy else "")
    )
    imgui.same_line()


# --- drawing ------------------------------------------------------------------


def _backdrop(draw_list: Any, view: Any, origin: tuple[float, float], size_px) -> None:
    from imgui_bundle import imgui

    lo = inker_state.to_screen(view, origin, 0, 0)
    hi = inker_state.to_screen(view, origin, size_px[0], size_px[1])
    draw_list.add_rect_filled(lo, hi, imgui.get_color_u32(theme.rgba(theme.ELEV_1)))
    draw_list.add_rect(lo, hi, imgui.get_color_u32(theme.rgba(theme.EDGE)))


def _visible_range(view: Any, doc: Any, origin, region) -> tuple[int, int, int, int]:
    """The inclusive tile rectangle the pane can actually see.

    Clamped to the map, so the loop below is bounded by the *window* rather
    than by the document: zooming out on a 512-square map must not become a
    quarter of a million draw calls.
    """
    x0, y0 = inker_state.to_image(view, origin, origin[0], origin[1])
    x1, y1 = inker_state.to_image(view, origin, origin[0] + region[0], origin[1] + region[1])
    c0 = max(0, int(x0 // doc.tile_w))
    r0 = max(0, int(y0 // doc.tile_h))
    c1 = min(doc.width - 1, int(x1 // doc.tile_w) + 1)
    r1 = min(doc.height - 1, int(y1 // doc.tile_h) + 1)
    return c0, r0, c1, r1


def _corner_uvs(uv, flip_h: bool, flip_v: bool, flip_d: bool):
    """The four corner UVs, in TL, TR, BR, BL order.

    Transpose first, then mirror -- the order ``render.orient`` applies, and the
    reason the canvas and an export never disagree about which way round a tile
    is. A transpose is a reflection across the main diagonal, so it swaps the
    two *off*-diagonal corners and leaves TL and BR alone.
    """
    u0, v0, u1, v1 = uv
    corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    if flip_d:
        corners = [corners[0], corners[3], corners[2], corners[1]]
    if flip_h:
        corners = [corners[1], corners[0], corners[3], corners[2]]
    if flip_v:
        corners = [corners[3], corners[2], corners[1], corners[0]]
    return corners


def _layers(ctx: Any, tab: Any, draw_list: Any, origin, region) -> None:
    from imgui_bundle import imgui

    doc = tab.doc
    view = tab.view
    c0, r0, c1, r1 = _visible_range(view, doc, origin, region)
    if c1 < c0 or r1 < r0:
        return
    zoom = view.zoom
    tile_w, tile_h = doc.tile_w, doc.tile_h
    # One texture per tileset per frame, resolved once: ``tileset_texture`` is a
    # dict lookup but ``resolve`` is a linear scan of the tileset list, and the
    # inner loop runs once per visible cell per layer.
    refs = {
        index: (ref, plotter_textures.tileset_texture(ctx, tab.uid, index, ref.tileset))
        for index, ref in enumerate(doc.tilesets)
    }
    if not refs:
        return

    for layer in doc.layers:
        if not isinstance(layer, TileLayer):
            continue
        if not layer.visible or layer.opacity <= 0.0:
            continue
        tint = imgui.get_color_u32((1.0, 1.0, 1.0, float(layer.opacity)))
        block = layer.data[r0 : r1 + 1, c0 : c1 + 1]
        ids = gidlib.tile_ids(block)
        flags = gidlib.flags(block)
        for row in range(ids.shape[0]):
            for column in range(ids.shape[1]):
                tile_id = int(ids[row, column])
                if not tile_id:
                    continue
                entry = None
                for index, (ref, texture) in refs.items():
                    if ref.holds(tile_id):
                        entry = (ref, texture, index)
                        break
                if entry is None or entry[1] is None:
                    continue
                ref, texture, _index = entry
                uv = ref.tileset.uv(ref.local(tile_id))
                mask = int(flags[row, column])
                corners = _corner_uvs(
                    uv,
                    bool(mask & gidlib.FLIP_H),
                    bool(mask & gidlib.FLIP_V),
                    bool(mask & gidlib.FLIP_D),
                )
                px = (c0 + column) * tile_w
                py = (r0 + row) * tile_h
                p0 = inker_state.to_screen(view, origin, px, py)
                p2 = (p0[0] + tile_w * zoom, p0[1] + tile_h * zoom)
                draw_list.add_image_quad(
                    widgets.texture_ref(texture),
                    p0,
                    (p2[0], p0[1]),
                    p2,
                    (p0[0], p2[1]),
                    corners[0],
                    corners[1],
                    corners[2],
                    corners[3],
                    tint,
                )


def _grid(draw_list: Any, doc: Any, view: Any, origin, size_px) -> None:
    from imgui_bundle import imgui

    step_x = doc.tile_w * view.zoom
    step_y = doc.tile_h * view.zoom
    if step_x < MIN_GRID_PX or step_y < MIN_GRID_PX:
        return
    colour = imgui.get_color_u32(theme.rgba(theme.EDGE, 0.6))
    top = inker_state.to_screen(view, origin, 0, 0)
    bottom = inker_state.to_screen(view, origin, size_px[0], size_px[1])
    for column in range(doc.width + 1):
        x = top[0] + column * step_x
        draw_list.add_line((x, top[1]), (x, bottom[1]), colour)
    for row in range(doc.height + 1):
        y = top[1] + row * step_y
        draw_list.add_line((top[0], y), (bottom[0], y), colour)


def _objects(state: Any, doc: Any, draw_list: Any, view: Any, origin) -> None:
    from imgui_bundle import imgui

    for layer in doc.layers:
        if not isinstance(layer, ObjectLayer) or not layer.visible:
            continue
        for obj in layer.objects:
            selected = state.selected_object == obj.uid
            alpha = 1.0 if obj.visible else 0.4
            colour = imgui.get_color_u32(
                theme.rgba(theme.ACCENT if selected else theme.OK, alpha)
            )
            p0 = inker_state.to_screen(view, origin, obj.x, obj.y)
            if obj.kind == "point":
                draw_list.add_circle_filled(p0, sp(4), colour, 12)
                draw_list.add_circle(p0, sp(7), colour, 12)
            else:
                p1 = inker_state.to_screen(view, origin, obj.x + obj.w, obj.y + obj.h)
                draw_list.add_rect(p0, p1, colour, 0.0, sp(2) if selected else sp(1))
            if obj.name:
                draw_list.add_text((p0[0] + sp(6), p0[1] - sp(14)), colour, obj.name)


def _cursor(state: Any, tab: Any, draw_list: Any, origin, hovered: bool) -> None:
    """The brush footprint under the pointer.

    Drawn from the *brush's* shape rather than one cell, so a 3x2 stamp shows
    what it is about to cover -- which is the only way to place one without
    guessing.
    """
    from imgui_bundle import imgui

    if not hovered or state.tool in ("object", "pick"):
        return
    cell = _cell_under(state, tab, origin)
    if cell is None:
        return
    doc, view = tab.doc, tab.view
    columns, rows = 1, 1
    if state.tool == "stamp" and state.brush is not None:
        rows, columns = state.brush.shape
    p0 = inker_state.to_screen(view, origin, cell[0] * doc.tile_w, cell[1] * doc.tile_h)
    p1 = inker_state.to_screen(
        view,
        origin,
        (cell[0] + columns) * doc.tile_w,
        (cell[1] + rows) * doc.tile_h,
    )
    draw_list.add_rect(p0, p1, imgui.get_color_u32((1.0, 1.0, 1.0, 0.75)), 0.0, sp(1.5))


# --- input --------------------------------------------------------------------


def _cell_under(state: Any, tab: Any, origin) -> tuple[int, int] | None:
    from imgui_bundle import imgui

    mouse = imgui.get_mouse_pos()
    x, y = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)
    return int(x // tab.doc.tile_w), int(y // tab.doc.tile_h)


def _events(ctx: Any, state: Any, tab: Any, origin, hovered: bool) -> None:
    from imgui_bundle import imgui

    if tab.busy:
        return
    io = imgui.get_io()
    view = tab.view

    if hovered and io.mouse_wheel:
        mouse = imgui.get_mouse_pos()
        inker_state.zoom_about(view, origin, (mouse.x, mouse.y), io.mouse_wheel)

    panning = state.space_held or imgui.is_mouse_down(2)
    if hovered and panning and imgui.is_mouse_dragging(2 if not state.space_held else 0):
        delta = imgui.get_mouse_drag_delta(2 if not state.space_held else 0)
        imgui.reset_mouse_drag_delta(2 if not state.space_held else 0)
        view.pan = (view.pan[0] + delta.x, view.pan[1] + delta.y)
        return

    if not hovered and state.drag_kind == "":
        return
    cell = _cell_under(state, tab, origin)
    if cell is None:
        return

    if state.tool == "object":
        _object_input(ctx, state, tab, origin, hovered)
        return

    if hovered and imgui.is_mouse_clicked(0):
        state.drag_kind = "rect" if state.tool == "rect" else "paint"
        state.drag_anchor = cell
        if state.tool != "rect":
            _apply(ctx, state, tab, cell)
    elif state.drag_kind == "paint" and imgui.is_mouse_down(0):
        _apply(ctx, state, tab, cell)
    elif state.drag_kind and imgui.is_mouse_released(0):
        if state.drag_kind == "rect" and state.drag_anchor is not None:
            _apply_rect(ctx, state, tab, state.drag_anchor, cell)
        state.clear_drag()


def _layer_for_paint(ctx: Any, tab: Any):
    layer = tab.doc.active()
    if isinstance(layer, TileLayer):
        return layer
    ctx.toast("Pick a tile layer to paint on.", "error")
    return None


def _apply(ctx: Any, state: Any, tab: Any, cell: tuple[int, int]) -> None:
    layer = _layer_for_paint(ctx, tab)
    if layer is None:
        return
    doc = tab.doc
    if state.tool == "pick":
        value = plotter_tools.pick(layer.data, *cell)
        if value:
            state.brush = np.array([[value]], gidlib.DTYPE)
            ref = doc.ref_for(value)
            if ref is not None:
                state.tileset_index = doc.tilesets.index(ref)
        return
    if state.tool == "stamp":
        if state.brush is None:
            ctx.toast("Pick a tile from the tileset first.", "error")
            return
        result = plotter_tools.stamp(layer.data, cell[0], cell[1], state.brush)
    elif state.tool == "erase":
        result = plotter_tools.erase(layer.data, cell[0], cell[1])
    elif state.tool == "fill":
        if state.brush is None:
            ctx.toast("Pick a tile from the tileset first.", "error")
            return
        result = plotter_tools.flood_fill(layer.data, cell[0], cell[1], int(state.brush[0, 0]))
    else:
        return
    if result is not None:
        doc.write_region(layer.uid, *result)


def _apply_rect(ctx: Any, state: Any, tab: Any, a: tuple[int, int], b: tuple[int, int]) -> None:
    layer = _layer_for_paint(ctx, tab)
    if layer is None:
        return
    if state.brush is None:
        ctx.toast("Pick a tile from the tileset first.", "error")
        return
    result = plotter_tools.fill_rect(layer.data, a[0], a[1], b[0], b[1], int(state.brush[0, 0]))
    if result is not None:
        tab.doc.write_region(layer.uid, *result)


def _object_input(ctx: Any, state: Any, tab: Any, origin, hovered: bool) -> None:
    """Click an object to select it, drag on empty space to draw a rectangle.

    A point is a rectangle with no drag, which is why the two are one gesture:
    releasing where you pressed means you wanted a marker, not a zero-sized box.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    layer = doc.active()
    if not isinstance(layer, ObjectLayer):
        if hovered and imgui.is_mouse_clicked(0):
            ctx.toast("Pick an object layer first.", "error")
        return
    mouse = imgui.get_mouse_pos()
    point = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)

    if hovered and imgui.is_mouse_clicked(0):
        hit = _object_at(layer, point)
        if hit is not None:
            state.selected_object = hit.uid
            state.drag_kind = ""
            return
        state.drag_kind = "object"
        state.drag_object = point
    elif state.drag_kind == "object" and imgui.is_mouse_released(0):
        start = state.drag_object or point
        x0, x1 = sorted((start[0], point[0]))
        y0, y1 = sorted((start[1], point[1]))
        w, h = x1 - x0, y1 - y0
        kind = "rect" if w >= tab.doc.tile_w * 0.25 and h >= tab.doc.tile_h * 0.25 else "point"
        obj = plotter_layers.add_object(
            doc, layer, kind, x0, y0, w if kind == "rect" else 0.0, h if kind == "rect" else 0.0
        )
        state.selected_object = obj.uid
        state.clear_drag()


def _object_at(layer: Any, point: tuple[float, float]):
    """Topmost first, so a small object drawn over a large one is reachable."""
    x, y = point
    for obj in reversed(layer.objects):
        if obj.kind == "point":
            if abs(obj.x - x) <= 8 and abs(obj.y - y) <= 8:
                return obj
        elif obj.x <= x <= obj.x + obj.w and obj.y <= y <= obj.y + obj.h:
            return obj
    return None
