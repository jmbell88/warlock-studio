"""Plotter's centre pane: the map, drawn cell by cell.

**One textured quad per *visible* cell, through imgui's draw list.** A
composited image is what this mode spends its effort not building -- a 200x200
map at 32px is a 40-megapixel RGBA buffer, and recompositing it per frame is
not something that happens between ``new_frame`` and ``render``. The flat
renderer in ``plotter/render.py`` builds one, and an export is its ordinary
caller.

**The one exception is a non-normal blend mode**, which the draw list cannot
express at all: it can tint a texture but has no per-quad blend switch, so
either the flat composite is drawn or the mode is not shown. :data:`BLEND_PREVIEW_PIXELS`
is the budget that keeps that exception from becoming the rule, and
:func:`_blended` is where it is applied -- above the budget, mid-stroke, or
with no GL context, the cell loop runs instead and the canvas says on its face
that the modes are not being shown.

The quads are ``add_image_quad`` rather than ``add_image``, and that is what
makes the diagonal flip drawable at all: a transpose cannot be expressed by
swapping a UV pair, but it can by permuting four corners. The permutations here
are the same transpose-then-mirror order ``render.orient`` applies, which is why
what you paint and what you export agree.

Draw-only. Every decision -- which tool, which tile, which layer -- belongs to
one of the three panes around this, and this one turns a click into a call.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from .. import (
    controls,
    icons,
    inker_state,
    plotter_mode,
    plotter_setup,
    plotter_state,
    theme,
    widgets,
)
from ..plotter import project
from ..plotter import render as plotter_render
from ..plotter import scene as plotter_scene
from ..plotter import terrain as plotter_terrain
from ..plotter import tools as plotter_tools
from ..plotter.tilemap import (
    ImageLayer,
    ObjectLayer,
    Polygon,
    Polyline,
    TileLayer,
)
from ..tilegrid import gid as gidlib
from ..tokens import sp
from . import plotter_layers, plotter_textures

# Below this many pixels per tile the grid stops being drawn: at two pixels a
# cell it is a solid wash rather than a guide, and it costs one line per column.
MIN_GRID_PX = 6

# Which tileset owns a tile id, memoised per document: ``{uid: (epoch, {id: index})}``.
#
# ``MapDoc.resolve`` is a linear scan of the tileset list, and the draw loop below
# asks once per *visible cell per tile layer* -- so a 60x40 viewport over four
# layers is ~9,600 scans a frame, every frame, for an answer that changes only
# when a tileset is attached, detached or swapped. That is what ``tileset_epoch``
# counts, and because the three hooks it counts are the tileset list's only
# mutators, an undo invalidates this cache exactly as the edit did.
#
# ``None`` is a real memoised answer -- "this id belongs to no tileset here" is
# the expensive question asked of every cell painted from a tileset that has
# since been detached -- so absence is spelled with a sentinel, not with ``None``.
_TILESET_MEMO: dict[str, tuple[int, dict[int, int | None]]] = {}
_UNMEMOISED = object()


def _index_memo(uid: str, epoch: int) -> dict[int, int | None]:
    """This document's id -> tileset-index map, emptied when the epoch moves."""
    entry = _TILESET_MEMO.get(uid)
    if entry is None or entry[0] != epoch:
        entry = (epoch, {})
        _TILESET_MEMO[uid] = entry
    return entry[1]


def forget_doc(uid: str) -> None:
    """Drop a closed document's memo, beside ``plotter_textures.release_doc``.

    A tab uid is never reissued, so a leak here is unbounded rather than merely
    stale -- the same reason the texture cache is released by uid at the same
    moment.
    """
    _TILESET_MEMO.pop(uid, None)


def forget_all() -> None:
    _TILESET_MEMO.clear()


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    _tabs(ctx, state)
    # Before the empty-state return, or "New map" would open a dialog that only
    # exists on the branch where a map is already open.
    setup_popup(ctx, state)
    tab = state.active
    if tab is None:
        _empty(ctx)
        return

    doc = tab.doc
    view = tab.view
    # A release can be missed -- focus lost, a tab switched, Esc, a save begun
    # mid-drag -- and a session left open is a document whose cells are ahead of
    # its history. ``end_stroke`` is idempotent, so this is a null check on
    # every frame the button is not down, over every tab rather than this one:
    # the tab you switched *away* from is not the tab drawing.
    if not imgui.is_mouse_down(0):
        for entry in state.docs:
            entry.doc.end_stroke()
            entry.doc.end_object_edit()
    avail = imgui.get_content_region_avail()
    # The status is drawn *after* the invisible canvas item, so its line has to
    # be reserved before that item claims the remaining region. Without this,
    # the cursor advances below the pane and the zoom/layer/cell readout is
    # clipped -- most painfully on isometric maps, where the cell coordinate is
    # the only unambiguous picking feedback.
    status_height = float(imgui.get_text_line_height_with_spacing())
    region = (
        max(float(avail.x), 1.0),
        max(float(avail.y) - status_height, 1.0),
    )
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
    _backdrop(draw_list, doc, view, (origin.x, origin.y))
    _layers(ctx, state, tab, draw_list, (origin.x, origin.y), region)
    if state.grid:
        _grid(draw_list, doc, view, (origin.x, origin.y))
    if state.show_objects:
        _objects(state, doc, draw_list, view, (origin.x, origin.y))
    # Above the grid so its edge reads against one, below the cursor so the
    # footprint stays the thing under the pointer.
    _marquee(state, tab, draw_list, (origin.x, origin.y))
    _cursor(state, tab, draw_list, (origin.x, origin.y), hovered)
    _minimap(ctx, state, tab, draw_list, (origin.x, origin.y), region)
    draw_list.pop_clip_rect()

    state.hover_cell = _cell_under(state, tab, (origin.x, origin.y)) if hovered else None
    _events(ctx, state, tab, (origin.x, origin.y), hovered, region)
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
    widgets.nothing_open(
        f"Start a map, open one, or drop a {plotter_state.MAP_SUFFIX_TEXT} on the window.",
        [
            ("New map...", lambda: plotter_mode.ask_new_document(ctx)),
            ("Open a file...", lambda: plotter_mode.ask_open(ctx)),
        ],
    )


#: The setup popup's name, which imgui also takes as its id -- so what is opened
#: and what is begun have to be this one string. One window owns it (this pane),
#: because a popup belongs to the window that begins it and this is the only
#: Plotter pane drawn on both the empty and the open branch.
SETUP_POPUP = "New map"


def setup_popup(ctx: Any, state: Any) -> None:
    """Ask what the new map is, then make it.

    Opened from ``state.setup_pending`` rather than from a button, because the
    five doors that ask for a map are spread across four windows and only this
    one can begin the popup. See :func:`.plotter_mode.ask_new_document`.
    """
    from imgui_bundle import imgui

    appearing = bool(state.setup_pending)
    if appearing:
        state.setup_pending = False
        imgui.open_popup(SETUP_POPUP)

    key = "plotter_new_map"
    form = ctx.state.preview.get(key)
    if not isinstance(form, dict):
        form = plotter_setup.blank_form()
        ctx.state.preview[key] = form

    centre = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
    alpha, rise = widgets.popover_enter("plotter-new-map", appearing)
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    opened, _ = imgui.begin_popup_modal(
        SETUP_POPUP, None, imgui.WindowFlags_.always_auto_resize.value
    )
    widgets.pop_surface_rounding()
    if not opened:
        imgui.pop_style_var()
        return
    widgets.window_shadow("overlay", radius=radius)
    if frosted:
        widgets.window_backdrop(radius=radius)
    if rise > 0.0:
        imgui.dummy((0, rise))
    _setup_body(ctx, form, key)
    imgui.end_popup()
    imgui.pop_style_var()


def _setup_body(ctx: Any, form: dict, key: str) -> None:
    from imgui_bundle import imgui

    # The three presets are the widest row in this modal.  Giving the form a
    # stable minimum measure keeps "Isometric, 64 x 32" intact instead of
    # allowing always-auto-resize to settle on the paragraph's narrower width.
    imgui.dummy((sp(480), 0))
    widgets.muted_wrapped(
        "The projection and the tile size are worth getting right now: a map's "
        "lattice is fixed once anything is painted on it, and a plain image is "
        "sliced at whatever tile size the map has when it is added."
    )
    imgui.dummy((0, sp(6)))

    widgets.field_label("Start from")
    width = widgets.grid_width(len(plotter_setup.PRESETS))
    for index, preset in enumerate(plotter_setup.PRESETS):
        if index:
            imgui.same_line()
        label = preset[0]
        if controls.button(f"{label}##preset-{index}", (width, 0)):
            plotter_setup.apply_preset(form, label)

    imgui.dummy((0, sp(6)))
    form["projection"] = widgets.labeled_combo(
        "Projection",
        form["projection"],
        [(name, name.capitalize()) for name in project.PROJECTIONS],
    )
    # ``W``/``H`` beside a field label rather than four spelled-out labels:
    # imgui draws an input's label to its right, so "Width (tiles)" on a pair of
    # fields on one line is two sentences competing for the same row. This is
    # ``inker_canvas``'s new-canvas layout.
    widgets.field_label("Map size, in tiles")
    imgui.set_next_item_width(sp(80))
    _, form["width"] = controls.input_int("W##map-w", int(form["width"]), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(80))
    _, form["height"] = controls.input_int("H##map-h", int(form["height"]), 0)

    widgets.field_label("Tile size, in pixels")
    imgui.set_next_item_width(sp(80))
    _, form["tile_w"] = controls.input_int("W##tile-w", int(form["tile_w"]), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(80))
    _, form["tile_h"] = controls.input_int("H##tile-h", int(form["tile_h"]), 0)
    # Clamped on the way in as well as on the way out, so the fields never go on
    # showing a number Create will not honour.
    plotter_setup.clamp(form)

    widgets.muted(plotter_setup.summary(form))
    warning = plotter_setup.isometric_warning(form)
    if warning:
        widgets.hint_text(warning)

    imgui.dummy((0, sp(6)))
    widgets.field_label("Then")
    for label, value in (
        (f"{icons.PLUS} Add a tileset from a file", plotter_setup.NEXT_FILE),
        ("Nothing yet", plotter_setup.NEXT_EMPTY),
    ):
        # Through ``controls`` rather than imgui: an AST guard rejects a pane
        # reaching for a presentational widget directly, and this is one of the
        # three states that has to look selected in the app's one language.
        if controls.radio_button(f"{label}##next-{value}", form["next"] == value):
            form["next"] = value

    imgui.dummy((0, sp(8)))
    if widgets.primary_button("Create", (sp(180), 0)):
        _create(ctx, form)
        ctx.state.preview.pop(key, None)
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel", (sp(120), 0)) or imgui.is_key_pressed(imgui.Key.escape):
        imgui.close_current_popup()


def _create(ctx: Any, form: dict) -> None:
    """Make the map, then open the tileset picker if that was chosen.

    The routing is here rather than in ``plotter_setup`` because the door is
    UI: it opens an OS picker on a task thread, which is not something a pure
    form module should know about.
    """
    plotter_mode.new_document(
        ctx, plotter_setup.size_of(form), projection=form["projection"]
    )
    choice = form.get("next")
    if choice == plotter_setup.NEXT_FILE:
        plotter_mode.ask_add_tileset(ctx)


def _status(ctx: Any, state: Any, tab: Any) -> None:
    doc = tab.doc
    layer = doc.active()
    name = layer.name if layer is not None else "no layer"
    bits = [
        f"{doc.width} x {doc.height}",
        doc.projection,
        f"{int(tab.view.zoom * 100)}%",
        name,
        f"tool {state.tool}",
    ]
    # The hovered cell is the affordance an isometric map actually needs:
    # nothing on screen says which diamond you are in, so "did I click the one
    # I meant" is otherwise unanswerable. Suppressed off-map so it cannot
    # flicker while the pointer is over the chrome.
    cell = state.hover_cell
    if cell is not None and 0 <= cell[0] < doc.width and 0 <= cell[1] < doc.height:
        bits.append(f"cell {cell[0]}, {cell[1]}")
        if state.tool == "terrain":
            ref = _terrain_ref(state, doc)
            layer = doc.active()
            if ref is not None and isinstance(layer, TileLayer):
                rank = plotter_terrain.terrain_at(layer.data, cell[0], cell[1], ref)
                if rank is not None:
                    bits.append(ref.tileset.terrains[rank].name)
    if tab.busy:
        bits.append("saving")
    widgets.muted("  --  ".join(bits))


# --- drawing ------------------------------------------------------------------


def _backdrop(draw_list: Any, doc: Any, view: Any, origin: tuple[float, float]) -> None:
    """The map's own extent, which is a diamond when the map is isometric.

    Drawn from the four lattice corners in both cases: for an orthogonal map
    those *are* the rectangle's corners, so there is one path and no branch that
    only one projection is ever tested through.
    """
    from imgui_bundle import imgui

    quad = [
        inker_state.to_screen(view, origin, *doc.cell_corner(column, row))
        for column, row in ((0, 0), (doc.width, 0), (doc.width, doc.height), (0, doc.height))
    ]
    colour = theme.rgba(theme.ELEV_1)
    if doc.backgroundcolor:
        text = str(doc.backgroundcolor).lstrip("#")
        if len(text) == 6:
            text = "ff" + text
        try:
            alpha, red, green, blue = (int(text[i : i + 2], 16) for i in range(0, 8, 2))
            colour = (red / 255.0, green / 255.0, blue / 255.0, alpha / 255.0)
        except (ValueError, TypeError):
            pass
    draw_list.add_convex_poly_filled(quad, imgui.get_color_u32(colour))
    # thickness *before* flags: the binding reverses imgui's own
    # ``AddPolyline(points, num, col, flags, thickness)``, and passing them the
    # C++ way is a TypeError rather than a wrong-looking line.
    draw_list.add_polyline(
        quad, imgui.get_color_u32(theme.rgba(theme.EDGE)), 1.0, imgui.ImDrawFlags_.closed.value
    )


def _visible_range(
    view: Any, doc: Any, origin, region, shift: tuple[float, float] = (0.0, 0.0)
) -> tuple[int, int, int, int]:
    """The inclusive tile rectangle the pane can actually see.

    Clamped to the map, so the loop below is bounded by the *window* rather
    than by the document: zooming out on a 512-square map must not become a
    quarter of a million draw calls.

    The arithmetic is ``MapDoc``'s, not this pane's, and it takes all four
    corners rather than two: under an isometric projection a screen rectangle
    maps to a *rhombus* in cell space, so the min and max of one diagonal pair
    culls cells that are on screen.

    ``shift`` is where a layer's own content sits relative to the grid --
    :func:`_layer_shift`'s answer -- and it is subtracted rather than ignored
    because the cull is what decides which cells are *asked for*: an offset
    layer's visible cells are a different rectangle, and using the grid's would
    clip a nudged layer along the pane edge it was nudged towards.
    """
    x0, y0 = inker_state.to_image(view, origin, origin[0], origin[1])
    x1, y1 = inker_state.to_image(view, origin, origin[0] + region[0], origin[1] + region[1])
    return doc.cell_bounds(x0 - shift[0], y0 - shift[1], x1 - shift[0], y1 - shift[1])


def _layer_shift(view: Any, origin, entry: Any) -> tuple[float, float]:
    """Where one resolved layer's content sits, in map pixels.

    Two things at once, because on screen they are one displacement: the
    layer's own summed pixel ``offset``, and what its ``parallax`` factor does
    against the view.

    The parallax half is ``(1 - p) * view_origin``. Read it off the two ends: at
    ``p == 1`` the term vanishes and the layer moves exactly with the map, which
    is why 1 is the identity; at ``p == 0`` the shift is the whole view origin,
    so the layer's own origin lands at the pane's top-left however far the map
    has been panned -- a background pinned to the screen. Everything between
    those is the parallax the name promises.

    ``render.py`` does none of this on purpose: an export is a still and a still
    has no camera, so there is no view origin for it to be a fraction of.
    """
    vx, vy = inker_state.to_image(view, origin, origin[0], origin[1])
    return (
        entry.offset[0] + (1.0 - entry.parallax[0]) * vx,
        entry.offset[1] + (1.0 - entry.parallax[1]) * vy,
    )


def _active_entry(tab: Any):
    """The active layer's resolved state -- its ancestry folded in -- or ``None``.

    Every input path that has to agree with what was drawn goes through here,
    because what was drawn came out of the same resolver. A walk from the root
    per event, which is what :func:`~..plotter.scene.resolved_for` is for and
    what the draw path already pays once per layer per frame.
    """
    return plotter_scene.resolved_for(tab.doc, tab.doc.active_layer)


def _active_locked(tab: Any) -> bool:
    """Whether the active layer is locked *or sits under something that is*.

    The layer's own flag is the wrong question: :mod:`~..plotter.scene` ORs the
    lock down the tree, the canvas hides an object's handles on the resolved
    answer, and an input path reading the leaf's flag would go on accepting
    exactly the gestures whose controls were taken away.
    """
    entry = _active_entry(tab)
    return entry is not None and entry.locked


def _active_shift(tab: Any, origin) -> tuple[float, float]:
    """The displacement the *active* layer is drawn at, or none.

    Hit-testing subtracts it before asking which cell the pointer is in: what
    the user is clicking is what they can see, and on an offset layer the cell
    under the pointer is not the cell the grid says it is.
    """
    entry = _active_entry(tab)
    return (0.0, 0.0) if entry is None else _layer_shift(tab.view, origin, entry)


def _layer_tint(imgui: Any, entry: Any) -> int:
    """One resolved layer's tint and opacity as imgui's single colour word.

    They multiply into one value because a draw-list quad takes one: the tint's
    own alpha is part of how transparent the layer is, exactly as the group
    opacity above it is.
    """
    r, g, b, a = entry.tint
    return imgui.get_color_u32(
        (r / 255.0, g / 255.0, b / 255.0, (a / 255.0) * float(entry.opacity))
    )


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


#: How many pixels this pane will composite between ``new_frame`` and
#: ``render``.
#:
#: Deliberately *not* ``render.MAX_RENDER_PIXELS``. That one is an arithmetic
#: refusal -- "past this the allocation is absurd" -- set at a gigabyte of RGBA
#: for a renderer that runs on a task thread and may take as long as it takes.
#: The question a frame asks is a different one: how much blending fits inside
#: a frame. 4 megapixels is a 2048-square image, which is a 64-square map at
#: 32px or a 128-square map at 16px -- tens of milliseconds of blend and
#: upload, so a hitch on the frame an edit lands rather than a stall. Above it
#: the pane declines and says why; the exporter is unaffected and still gets
#: the full ceiling.
BLEND_PREVIEW_PIXELS = 1 << 22


def _blended(ctx: Any, tab: Any, doc: Any, draw_list: Any, origin, view) -> bool:
    """Draw the map as one correctly-composited texture. -> whether it drew.

    Three reasons it declines, and the caller falls back to the ordinary cell
    loop -- which is what this mode did before blend modes existed at all, so
    the fallback is a documented rendering rather than a broken one.

    * **Past the frame budget.** ``render_map`` allocates and blends the whole
      map at full size; at 512 cells square and 32px tiles that is a
      268-megapixel composite plus a same-size GL upload, on the frame thread,
      on every edit. It also *raises* past its own ceiling, and a raise from a
      pane draw is not caught anywhere (``main.py``'s frame loop does not wrap
      pane draws), so on a large enough map the first frame that drew a
      multiply layer took the window down. Checked before the call rather than
      caught after it: the point is not to survive the allocation, it is not to
      make it.
    * **Mid-stroke.** The cache stamp is ``history.head`` and the tree shape,
      and a paint stroke writes the live array and pushes nothing until
      ``end_stroke``. Keyed on that alone the composite was pinned for the
      whole stroke, so painting on a map with any non-normal layer showed
      nothing at all until the mouse came up. Declining instead means the cells
      go down live and the modes come back on release.
    * **No GL context**, which is every headless test.

    ``ValueError`` is still caught around the call. The ceiling is not the only
    thing ``render_map`` raises for -- an unparseable ``backgroundcolor``
    reaches it as well -- and none of them is a reason to lose the window from
    inside a draw list.
    """
    if doc.stroking:
        return False
    if int(doc.pixel_width) * int(doc.pixel_height) > BLEND_PREVIEW_PIXELS:
        return False
    stamp = (doc.history.head, doc.tree_shape(), "blend")
    key = plotter_textures.blend_slot(tab.uid)
    cached = ctx.state.preview.get(key)
    if cached is None or cached[0] != stamp:
        try:
            cached = (stamp, plotter_render.render_map(doc))
        except ValueError:
            return False
        ctx.state.preview[key] = cached
    texture = plotter_textures.image_texture(
        ctx, tab.uid, "blended-map", cached[1], stamp
    )
    if texture is None:
        return False
    p0 = inker_state.to_screen(view, origin, 0.0, 0.0)
    draw_list.add_image(
        widgets.texture_ref(texture),
        p0,
        (
            p0[0] + doc.pixel_width * view.zoom,
            p0[1] + doc.pixel_height * view.zoom,
        ),
    )
    return True


def _layers(ctx: Any, state: Any, tab: Any, draw_list: Any, origin, region) -> None:
    """Every drawable layer, through the resolver both renderers share.

    ``scene.resolve`` is what decides visibility, opacity, tint, offset and
    parallax here exactly as it does in ``render.render_map`` -- the two
    renderers agree about a nested layer for the same reason they already agree
    about which way round a flipped tile is, which is that neither works it out
    for itself.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    view = tab.view
    zoom = view.zoom
    tile_w, tile_h = doc.tile_w, doc.tile_h
    resolved = plotter_scene.resolve(doc)
    # ImGui's draw list can tint a texture but has no per-quad blend-mode
    # switch, so the only way to show a non-normal mode is the flat renderer's
    # composited surface -- the one thing this module's header says never
    # happens here, for the reason it gives. :func:`_blended` is where that
    # exception is bounded; when it declines, the cell loop below draws every
    # layer as though its mode were normal and the note says so.
    if any(entry.blend_mode != "normal" for entry in resolved):
        if _blended(ctx, tab, doc, draw_list, origin, view):
            return
        # Said out loud rather than left looking like a rendering bug: a
        # multiply layer about to be drawn as an opaque one is a canvas that
        # disagrees with its own export, and silence gives the user nothing to
        # go on. Cheap enough to draw every frame -- it is one string.
        draw_list.add_text(
            (origin[0] + sp(8), origin[1] + sp(8)),
            imgui.get_color_u32(theme.rgba(theme.WARN)),
            "Blend modes are not shown while painting"
            if doc.stroking
            else "Blend modes are not shown on a map this large",
        )
    # One texture per tileset per frame. This hoist only ever covered the
    # *upload* -- ``tileset_texture`` is a dict lookup, but deciding which ref
    # holds a given id stayed a linear scan run once per visible cell per layer,
    # which is what ``_TILESET_MEMO`` now answers instead.
    refs = {
        index: (
            ref,
            plotter_textures.tileset_texture(
                ctx, tab.uid, index, ref.tileset, doc.tileset_epoch
            ),
        )
        for index, ref in enumerate(doc.tilesets)
    }
    memo = _index_memo(tab.uid, doc.tileset_epoch)

    # **In this pane only, never in ``scene.resolve``.** ``render.py`` composites
    # every export off the same resolver, so a dim living there would export
    # dimmed -- which is why the entry is replaced here, on the way to the draw
    # list, and the document is not touched at all.
    if state.highlight and doc.active_layer is not None:
        resolved = [
            entry
            if entry.layer.uid == doc.active_layer
            else dataclasses.replace(entry, opacity=entry.opacity * 0.3)
            for entry in resolved
        ]

    for entry in resolved:
        if entry.opacity <= 0.0:
            continue
        layer = entry.layer
        shift = _layer_shift(view, origin, entry)
        if isinstance(layer, ImageLayer):
            _image_layer(ctx, tab, draw_list, origin, layer, entry, shift)
            continue
        if not isinstance(layer, TileLayer) or not refs:
            continue
        c0, r0, c1, r1 = _visible_range(view, doc, origin, region, shift)
        if c1 < c0 or r1 < r0:
            continue
        tint = _layer_tint(imgui, entry)
        block = layer.data[r0 : r1 + 1, c0 : c1 + 1]
        ids = gidlib.tile_ids(block)
        flags = gidlib.flags(block)
        # Back to front. For an orthogonal map that is row-major and this is
        # the order the block already has; for an isometric one depth is
        # ``column + row``, and row-major is not monotone in it.
        rows = list(range(ids.shape[0]))
        columns = list(range(ids.shape[1]))
        if doc.renderorder.startswith("left"):
            columns.reverse()
        if doc.renderorder.endswith("up"):
            rows.reverse()
        cells = [(row, column) for row in rows for column in columns]
        if doc.isometric:
            cells.sort(key=lambda cell: cell[0] + cell[1])
        for row, column in cells:
            tile_id = int(ids[row, column])
            if not tile_id:
                continue
            index = memo.get(tile_id, _UNMEMOISED)
            if index is _UNMEMOISED:
                index = None
                for candidate, (ref, _texture) in refs.items():
                    if ref.holds(tile_id):
                        index = candidate
                        break
                memo[tile_id] = index
            if index is None:
                continue
            ref, texture = refs[index]
            if texture is None:
                continue
            uv = ref.tileset.uv(ref.local(tile_id))
            mask = int(flags[row, column])
            corners = _corner_uvs(
                uv,
                bool(mask & gidlib.FLIP_H),
                bool(mask & gidlib.FLIP_V),
                bool(mask & gidlib.FLIP_D),
            )
            px, py = doc.cell_origin(c0 + column, r0 + row)
            p0 = inker_state.to_screen(view, origin, px + shift[0], py + shift[1])
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


def _image_layer(
    ctx: Any, tab: Any, draw_list: Any, origin, layer: Any, entry: Any, shift
) -> None:
    """One image layer, as a quad -- or a grid of them when it repeats.

    Bounded by the *map's* extent rather than by the pane, which is what
    ``render.render_image`` does too: the two renderers have to put the same
    number of copies in the same places, and a viewport-bounded tiling would put
    copies on screen that an export does not carry.

    The texture is keyed on the pixel array's identity, the tileset rule --
    :class:`~..plotter.tilemap.ImageLayer` freezes its picture on construction,
    so a matching id is the same art and a replaced picture is a new array.
    """
    if layer.pixels.size == 0:
        return
    from imgui_bundle import imgui

    texture = plotter_textures.image_texture(
        ctx, tab.uid, f"img{layer.uid}", layer.pixels, id(layer.pixels)
    )
    if texture is None:
        return
    doc, view = tab.doc, tab.view
    zoom = view.zoom
    width, height = float(layer.width), float(layer.height)
    tint = _layer_tint(imgui, entry)
    for py in plotter_render.repeats(
        int(round(shift[1])), int(height), int(doc.pixel_height), layer.repeat_y
    ):
        for px in plotter_render.repeats(
            int(round(shift[0])), int(width), int(doc.pixel_width), layer.repeat_x
        ):
            p0 = inker_state.to_screen(view, origin, px, py)
            draw_list.add_image(
                widgets.texture_ref(texture),
                p0,
                (p0[0] + width * zoom, p0[1] + height * zoom),
                (0.0, 0.0),
                (1.0, 1.0),
                tint,
            )


def _grid(draw_list: Any, doc: Any, view: Any, origin) -> None:
    from imgui_bundle import imgui

    # The *projected* step, not the tile size: an isometric cell advances only
    # half a tile per lattice step, so the old test kept drawing a solid wash
    # at half the zoom it should have stopped at.
    scale = 0.5 if doc.isometric else 1.0
    step_x = doc.tile_w * view.zoom * scale
    step_y = doc.tile_h * view.zoom * scale
    if step_x < MIN_GRID_PX or step_y < MIN_GRID_PX:
        return
    colour = imgui.get_color_u32(theme.rgba(theme.EDGE, 0.6))

    def node(column: float, row: float):
        return inker_state.to_screen(view, origin, *doc.cell_corner(column, row))

    # Lines along the two *lattice* directions rather than the screen axes.
    # For an orthogonal map this reproduces the old lines exactly; for an
    # isometric one it draws the diamond mesh.
    for column in range(doc.width + 1):
        draw_list.add_line(node(column, 0), node(column, doc.height), colour)
    for row in range(doc.height + 1):
        draw_list.add_line(node(0, row), node(doc.width, row), colour)


def _objects(state: Any, doc: Any, draw_list: Any, view: Any, origin) -> None:
    """The object outlines, through the same resolver the tiles go through.

    An object layer draws no pixels, so it is absent from an export -- but it is
    still a layer in the tree, and a group that is hidden, offset or under a
    parallax factor has to move its objects with everything else or the handles
    stop sitting on what they resize.
    """
    from imgui_bundle import imgui

    for entry in plotter_scene.resolve(doc):
        layer = entry.layer
        if not isinstance(layer, ObjectLayer):
            continue
        dx, dy = _layer_shift(view, origin, entry)
        for obj in layer.objects:
            selected = state.selected_object == obj.uid
            alpha = float(obj.opacity) * float(entry.opacity) * (1.0 if obj.visible else 0.4)
            colour = imgui.get_color_u32(
                theme.rgba(theme.ACCENT if selected else theme.OK, alpha)
            )
            p0 = inker_state.to_screen(view, origin, obj.x + dx, obj.y + dy)
            if obj.kind == "point":
                draw_list.add_circle_filled(p0, sp(4), colour, 12)
                draw_list.add_circle(p0, sp(7), colour, 12)
            else:
                outline = [
                    inker_state.to_screen(view, origin, px + dx, py + dy)
                    for px, py in _object_outline(obj)
                ]
                closed = obj.kind != "polyline"
                draw_list.add_polyline(
                    outline,
                    colour,
                    sp(2) if selected else sp(1),
                    imgui.ImDrawFlags_.closed.value if closed else 0,
                )
                if selected and not entry.locked:
                    # Only on the selected one, and only when it can actually be
                    # dragged: a handle on a locked layer is a control that
                    # looks live and does nothing. ``entry.locked``, not the
                    # layer's own flag -- a lock on a group above it is just as
                    # real a reason the drag is refused, and ``_object_input``
                    # refuses it off the same resolved answer.
                    for corner in _HANDLES:
                        cx, cy = _handle_corners(obj)[corner]
                        sx, sy = inker_state.to_screen(view, origin, cx + dx, cy + dy)
                        draw_list.add_rect_filled(
                            (sx - sp(4), sy - sp(4)), (sx + sp(4), sy + sp(4)), colour
                        )
            if obj.name:
                draw_list.add_text((p0[0] + sp(6), p0[1] - sp(14)), colour, obj.name)


def _rotated(obj: Any, x: float, y: float) -> tuple[float, float]:
    """An object-local point in the map plane, using Tiled's clockwise angle."""
    # Lightweight integrations sometimes pass an object-shaped protocol value
    # rather than a MapObject. Absence means Tiled's default of zero rotation.
    angle = math.radians(float(getattr(obj, "rotation", 0.0)))
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        obj.x + x * cosine - y * sine,
        obj.y + x * sine + y * cosine,
    )


def _object_outline(obj: Any) -> list[tuple[float, float]]:
    """Polyline used for every Tiled object shape in the live canvas."""
    if obj.kind in ("polygon", "polyline"):
        local = list(obj.shape.points)
    elif obj.kind == "ellipse":
        local = [
            (
                obj.w * 0.5 + math.cos(index * math.tau / 32) * obj.w * 0.5,
                obj.h * 0.5 + math.sin(index * math.tau / 32) * obj.h * 0.5,
            )
            for index in range(32)
        ]
    elif obj.kind == "capsule":
        local = _capsule_outline(obj.w, obj.h)
    else:
        local = [(0.0, 0.0), (obj.w, 0.0), (obj.w, obj.h), (0.0, obj.h)]
    return [_rotated(obj, x, y) for x, y in local]


def _capsule_outline(width: float, height: float) -> list[tuple[float, float]]:
    steps = 16
    if width >= height:
        radius = height * 0.5
        centres = ((width - radius, radius, -math.pi / 2), (radius, radius, math.pi / 2))
    else:
        radius = width * 0.5
        centres = ((radius, height - radius, 0.0), (radius, radius, math.pi))
    points: list[tuple[float, float]] = []
    for cx, cy, start in centres:
        points.extend(
            (
                cx + math.cos(start + index * math.pi / steps) * radius,
                cy + math.sin(start + index * math.pi / steps) * radius,
            )
            for index in range(steps + 1)
        )
    return points


# The minimap's longest edge, in design pixels.
_MINIMAP_MAX = 160


def _minimap_rect(tab: Any, region) -> tuple[float, float, float, float]:
    """Where the minimap sits, bottom-right of the pane, in screen space."""
    doc = tab.doc
    longest = max(int(doc.width), int(doc.height)) or 1
    scale = sp(_MINIMAP_MAX) / longest
    w, h = doc.width * scale, doc.height * scale
    pad = sp(8)
    return region[0] - w - pad, region[1] - h - pad, w, h


def _minimap(ctx: Any, state: Any, tab: Any, draw_list: Any, origin, region) -> None:
    """The whole map in the corner, one pixel per cell.

    Cached on ``(history.head, tree_shape())`` rather than recomputed per frame:
    the head moves for every edit, which is exactly when the picture changes,
    and the tree's shape catches anything that moved a layer without moving the
    head. It was ``len(layers)``, which the tree quietly made dishonest --
    adding a layer inside a group, or dragging one out of it, leaves the root
    list exactly as long as it was, so the picture would change under a key that
    did not. Both halves are cheap, so the comparison costs nothing on the frame
    that does not need it.
    """
    from imgui_bundle import imgui

    if not state.minimap:
        return
    doc = tab.doc
    x, y, w, h = _minimap_rect(tab, region)
    x, y = origin[0] + x, origin[1] + y
    stamp = (doc.history.head, doc.tree_shape())
    key = f"plotter_minimap:{tab.uid}"
    entry = ctx.state.preview.get(key)
    if entry is None or entry[0] != stamp:
        entry = (stamp, plotter_render.minimap(doc))
        ctx.state.preview[key] = entry
    texture = plotter_textures.image_texture(ctx, tab.uid, "minimap", entry[1], stamp)

    draw_list.add_rect_filled((x, y), (x + w, y + h), imgui.get_color_u32(theme.rgba(theme.ELEV_1)))
    if texture is not None:
        draw_list.add_image(widgets.texture_ref(texture), (x, y), (x + w, y + h))
    # rounding is the 4th positional and thickness the *5th*; a flags value in
    # that slot silently draws a hairline and is how this was caught.
    draw_list.add_rect(
        (x, y), (x + w, y + h), imgui.get_color_u32(theme.rgba(theme.EDGE)), 0.0, sp(1)
    )
    # Where the pane is looking, so the minimap says *where you are* and not
    # only what the map holds.
    c0, r0, c1, r1 = _visible_range(tab.view, doc, origin, region)
    vx0 = x + (max(0, c0) / max(1, doc.width)) * w
    vy0 = y + (max(0, r0) / max(1, doc.height)) * h
    vx1 = x + (min(doc.width, c1 + 1) / max(1, doc.width)) * w
    vy1 = y + (min(doc.height, r1 + 1) / max(1, doc.height)) * h
    # When the whole map is visible this rectangle is exactly the minimap's
    # outside edge. Painting it purple made an empty map look selected and hid
    # the neutral container boundary underneath; in that state the location
    # marker carries no information, so leave the ordinary edge visible.
    whole_map = c0 <= 0 and r0 <= 0 and c1 >= doc.width - 1 and r1 >= doc.height - 1
    if not whole_map:
        draw_list.add_rect(
            (vx0, vy0),
            (vx1, vy1),
            imgui.get_color_u32(theme.rgba(theme.ACCENT)),
            0.0,
            sp(1.5),
        )


def _marquee(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """The selection, as a low-alpha fill with an outline on it.

    Drawn from the clamped rect rather than the stored one, so a marquee left
    hanging over a map that has since been resized shows what it actually
    constrains. A parallelogram through four lattice nodes, the footprint's
    idiom, so the isometric case is the same four calls.
    """
    from imgui_bundle import imgui

    rect = state.selection_in(tab.doc)
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    doc, view = tab.doc, tab.view
    quad = [
        inker_state.to_screen(view, origin, *doc.cell_corner(column, row))
        for column, row in ((x0, y0), (x1 + 1, y0), (x1 + 1, y1 + 1), (x0, y1 + 1))
    ]
    draw_list.add_convex_poly_filled(quad, imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.15)))
    draw_list.add_polyline(
        quad,
        imgui.get_color_u32(theme.rgba(theme.ACCENT)),
        sp(1.5),
        imgui.ImDrawFlags_.closed.value,
    )


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
    # Drawn where the *active layer* is drawn, because that is the layer the
    # cell under the pointer belongs to -- ``_cell_under`` already took this
    # off, and a footprint left on the bare grid would sit somewhere the stamp
    # is not about to land.
    shift = _active_shift(tab, origin)

    def outline(at: tuple[int, int], alpha: float) -> None:
        # The parallelogram through four lattice nodes, which degenerates to the
        # rectangle this used to draw when the map is orthogonal.
        quad = [
            inker_state.to_screen(view, origin, *_shifted(doc.cell_corner(column, row), shift))
            for column, row in (
                (at[0], at[1]),
                (at[0] + columns, at[1]),
                (at[0] + columns, at[1] + rows),
                (at[0], at[1] + rows),
            )
        ]
        draw_list.add_polyline(
            quad,
            imgui.get_color_u32((1.0, 1.0, 1.0, alpha)),
            sp(1.5),
            imgui.ImDrawFlags_.closed.value,
        )

    # A shape drag in progress: outline what would land at release. The rect
    # tool this replaced had no preview at all, so the only way to place one was
    # to draw it and undo.
    if state.drag_kind in ("rect", "erase-rect") and state.drag_anchor is not None:
        _shape_preview(
            state,
            tab,
            draw_list,
            origin,
            state.drag_anchor,
            cell,
            mode="rect" if state.drag_kind == "erase-rect" else None,
        )
        return

    # Shift with a "from" cell in hand: show the run that is about to land, not
    # just where the pointer is. Drawn from the same lattice corners as the
    # footprint, so an isometric map previews as a diagonal of diamonds for
    # free. Faint, because the cell under the pointer is still the cursor.
    if _line_pending(state, imgui.get_io()):
        for step in plotter_tools.line(*state.last_paint, cell[0], cell[1])[:-1]:
            outline(step, 0.3)
    outline(cell, 0.75)


# How many segments an ellipse preview is drawn with. Past about this the
# polyline costs more than the curve gains, and a shape drag lasts a handful of
# frames.
_ELLIPSE_SAMPLES = 32


def _shape_points(
    mode: str, a: tuple[int, int], b: tuple[int, int]
) -> list[tuple[float, float]]:
    """The outline of a shape drag, in *lattice* coordinates.

    Pure, and separate from the drawing for the reason ``_corner_uvs`` is: the
    arithmetic is the part worth pinning, and it is imgui-free.

    Lattice rather than screen coordinates because the projection is what turns
    a circle into the right ellipse on an isometric map -- sampling here and
    projecting afterwards means there is no second path for the second lattice.
    """
    lo_x, hi_x = sorted((a[0], b[0]))
    lo_y, hi_y = sorted((a[1], b[1]))
    # Node coordinates: the far edge of the last cell, not its near edge.
    x0, y0, x1, y1 = float(lo_x), float(lo_y), float(hi_x + 1), float(hi_y + 1)
    if mode != "ellipse":
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    centre_x, centre_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    semi_x, semi_y = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    return [
        (
            centre_x + semi_x * math.cos(math.tau * step / _ELLIPSE_SAMPLES),
            centre_y + semi_y * math.sin(math.tau * step / _ELLIPSE_SAMPLES),
        )
        for step in range(_ELLIPSE_SAMPLES)
    ]


def _shifted(point: tuple[float, float], shift: tuple[float, float]) -> tuple[float, float]:
    """One map-pixel point moved by a layer's displacement."""
    return (point[0] + shift[0], point[1] + shift[1])


def _shape_preview(
    state: Any,
    tab: Any,
    draw_list: Any,
    origin,
    a: tuple[int, int],
    b: tuple[int, int],
    *,
    mode: str | None = None,
) -> None:
    """The outline of what a shape drag would fill, rect or ellipse."""
    from imgui_bundle import imgui

    doc, view = tab.doc, tab.view
    shift = _active_shift(tab, origin)
    draw_list.add_polyline(
        [
            inker_state.to_screen(view, origin, *_shifted(doc.cell_corner(px, py), shift))
            for px, py in _shape_points(state.shape_mode if mode is None else mode, a, b)
        ],
        imgui.get_color_u32(theme.rgba(theme.ACCENT)),
        sp(1.5),
        imgui.ImDrawFlags_.closed.value,
    )


# --- input --------------------------------------------------------------------


def _cell_under(state: Any, tab: Any, origin) -> tuple[int, int] | None:
    from imgui_bundle import imgui

    mouse = imgui.get_mouse_pos()
    x, y = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)
    # The active layer's own displacement comes off first: what a click is about
    # is the cell the user can see under the pointer, and on an offset layer
    # that is not the cell the grid is drawn at.
    dx, dy = _active_shift(tab, origin)
    # ``MapDoc`` owns the inverse so this pane's hit test and the tools' cannot
    # drift; it is exact rather than a polygon test, and unclamped because every
    # tool clips its own placement.
    return tab.doc.cell_at(x - dx, y - dy)


def _events(ctx: Any, state: Any, tab: Any, origin, hovered: bool, region) -> None:
    from imgui_bundle import imgui

    if tab.busy:
        return
    io = imgui.get_io()
    view = tab.view

    if hovered and io.mouse_wheel:
        mouse = imgui.get_mouse_pos()
        inker_state.zoom_about(view, origin, (mouse.x, mouse.y), io.mouse_wheel)

    # Claimed before any tool sees the mouse, and before panning: the minimap
    # sits *over* the map, so a press inside it that fell through would paint a
    # cell on whatever is underneath.
    if _minimap_input(state, tab, origin, region, hovered):
        return

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

    if state.tool == "select":
        _select_input(state, tab, cell, hovered)
        return

    if state.tool == "wand":
        _wand_input(state, tab, cell, hovered)
        return

    if state.tool == "pick":
        # Pick's own gesture, because a *drag* means something different from a
        # click here: press-and-drag captures the marquee as a brush (Tiled's
        # capture), and a plain click stays today's single-cell pick byte for
        # byte -- ``_apply`` still runs it, from the release branch below.
        _pick_input(ctx, state, tab, cell, hovered)
        return

    if hovered and imgui.is_mouse_clicked(0):
        # The drag *kind* stays "rect" whatever the shape tool is filling: the
        # gesture is a corner-to-corner drag either way, and only what lands at
        # release differs. Renaming it would also collide with the object tool's
        # own "rect", which is a third, unrelated use of the word.
        if state.tool == "shape":
            state.drag_kind = "rect"
        elif state.tool == "erase" and io.key_shift:
            state.drag_kind = "erase-rect"
        else:
            state.drag_kind = "paint"
        state.drag_anchor = cell
        if state.drag_kind == "paint":
            # Stamp, erase and terrain are the three tools a *drag* repeats, and
            # so the three that would otherwise push one step per cell. Fill,
            # rect and pick fire once and need no session.
            layer = tab.doc.active()
            if state.tool in ("stamp", "erase", "terrain") and isinstance(layer, TileLayer):
                tab.doc.begin_stroke(layer.uid)
            if _line_pending(state, io):
                # Inside the session opened just above, so a forty-cell line is
                # one step exactly as a forty-cell drag is.
                _apply_line(ctx, state, tab, state.last_paint, cell)
            else:
                _apply(ctx, state, tab, cell)
            state.drag_last_cell = cell
            if state.tool == "stamp":
                state.last_paint = cell
    elif state.drag_kind == "paint" and imgui.is_mouse_down(0):
        _apply_drag(ctx, state, tab, cell)
    elif state.drag_kind and imgui.is_mouse_released(0):
        if state.drag_kind == "rect" and state.drag_anchor is not None:
            _apply_shape(ctx, state, tab, state.drag_anchor, cell)
        elif state.drag_kind == "erase-rect" and state.drag_anchor is not None:
            _apply_erase_rect(ctx, state, tab, state.drag_anchor, cell)
        tab.doc.end_stroke()
        state.clear_drag()


def _line_pending(state: Any, io: Any) -> bool:
    """Whether this click should draw a line rather than place one stamp.

    Tiled's rule, and Stamp only: the other tools that repeat on a drag have no
    "from" -- Erase and Terrain both self-select per cell, so a line of them is
    a stroke, which the drag already is.
    """
    return bool(io.key_shift) and state.tool == "stamp" and state.last_paint is not None


def _minimap_input(state: Any, tab: Any, origin, region, hovered: bool) -> bool:
    """Click or drag the minimap to recentre the view. ``True`` if it took the
    mouse, which is what keeps the tool underneath from also acting."""
    from imgui_bundle import imgui

    if not state.minimap:
        return False
    if state.drag_kind and state.drag_kind != "minimap":
        return False
    x, y, w, h = _minimap_rect(tab, region)
    x, y = origin[0] + x, origin[1] + y
    mouse = imgui.get_mouse_pos()
    inside = x <= mouse.x <= x + w and y <= mouse.y <= y + h

    if hovered and inside and imgui.is_mouse_clicked(0):
        state.drag_kind = "minimap"
    elif state.drag_kind == "minimap" and not imgui.is_mouse_down(0):
        state.clear_drag()
        return True
    if state.drag_kind != "minimap":
        # Still swallow a hover so the cursor footprint does not draw under it.
        return bool(inside and hovered)

    doc = tab.doc
    # Where in the map the pointer is, as a fraction, then centre the pane there.
    fx = min(max((mouse.x - x) / w, 0.0), 1.0)
    fy = min(max((mouse.y - y) / h, 0.0), 1.0)
    px, py = fx * doc.pixel_width, fy * doc.pixel_height
    zoom = tab.view.zoom
    tab.view.pan = (region[0] / 2.0 - px * zoom, region[1] / 2.0 - py * zoom)
    return True


def _normalized(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    """Two dragged corners as an inclusive ``(x0, y0, x1, y1)``, lowest first."""
    lo_x, hi_x = sorted((int(a[0]), int(b[0])))
    lo_y, hi_y = sorted((int(a[1]), int(b[1])))
    return lo_x, lo_y, hi_x, hi_y


def _pick_input(
    ctx: Any, state: Any, tab: Any, cell: tuple[int, int], hovered: bool
) -> None:
    """Pick: a click is one cell, a drag captures a block off the map.

    Tiled's capture. The marquee rectangle is reused for the feedback, so the
    gesture looks exactly like a selection while it is happening and the brush
    cursor takes over the moment it ends -- which is the paste precedent
    (``plotter_mode``'s paste loads the brush and switches to Stamp rather than
    dropping a block).

    A plain click -- press and release inside one cell -- is today's single-cell
    pick, byte for byte, and goes through :func:`_apply` exactly as it did.
    """
    from imgui_bundle import imgui

    if hovered and imgui.is_mouse_clicked(0):
        state.drag_kind = "capture"
        state.drag_anchor = cell
        return
    if state.drag_kind != "capture":
        return
    if imgui.is_mouse_down(0) and state.drag_anchor is not None:
        # Drawn with the marquee, so there is one rectangle-feedback path.
        state.set_selection(_normalized(state.drag_anchor, cell))
        return
    if not imgui.is_mouse_released(0):
        return
    anchor = state.drag_anchor
    state.set_selection(None)
    state.clear_drag()
    if anchor is None:
        return
    if anchor == cell:
        _apply(ctx, state, tab, cell)
        return
    layer = tab.doc.active()
    if not isinstance(layer, TileLayer):
        return
    block = plotter_tools.capture(layer.data, anchor[0], anchor[1], cell[0], cell[1])
    if block is None:
        # An all-empty capture would arm a brush that erases everything it
        # touches while claiming to be a stamp.
        ctx.toast("There is nothing to capture there.", "warn")
        return
    state.brush = block
    state.tool = "stamp"
    ctx.toast(f"Captured a {block.shape[1]} x {block.shape[0]} stamp.")


def _before_drag(state: Any):
    found = getattr(state, "select_before", None)
    return None if found is None else found[0]


def _mask_before_drag(state: Any):
    found = getattr(state, "select_before", None)
    return None if found is None else found[1]


def _selection_algebra(state: Any, doc: Any, wanted, add: bool, subtract: bool) -> None:
    """Store a new set of cells, combined with what is already selected.

    ``wanted`` is a full-map bool array. Shift adds, Alt subtracts, neither
    replaces -- the same three-way algebra on the wand and on the marquee, so
    "hold Shift to add" is one thing to learn rather than two.

    A result that is a solid rectangle is stored **as a rectangle** (mask
    ``None``): that is what keeps an ordinary marquee at an ordinary cost, and
    it means the mask only exists once the selection has actually stopped being
    one.
    """
    import numpy as np

    if add or subtract:
        rect = state.selection_in(doc)
        current = np.zeros((int(doc.height), int(doc.width)), dtype=bool)
        if rect is not None:
            existing = state.selection_mask_in(doc)
            if existing is not None:
                current |= existing
            else:
                x0, y0, x1, y1 = rect
                current[y0 : y1 + 1, x0 : x1 + 1] = True
        wanted = (current & ~wanted) if subtract else (current | wanted)
    if not wanted.any():
        state.set_selection(None)
        return
    ys, xs = np.nonzero(wanted)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    solid = bool(wanted[y0 : y1 + 1, x0 : x1 + 1].all())
    state.set_selection((x0, y0, x1, y1), None if solid else wanted)


def _wand_input(state: Any, tab: Any, cell: tuple[int, int], hovered: bool) -> None:
    """Click selects the contiguous run of one gid; Ctrl+click every cell of it.

    The contiguous half goes through the *same* ``flood_mask`` the fill does --
    one helper, three callers, and the C kernel behind it serves all of them. The
    Ctrl half is Tiled's Select Same Tile, folded in here because ``S`` is taken
    by Objects.

    Nothing is written to the document, so none of this is guarded by
    ``tab.busy``; the selection is view state exactly as the marquee is.
    """
    import numpy as np
    from imgui_bundle import imgui

    if not (hovered and imgui.is_mouse_clicked(0)):
        return
    layer = tab.doc.active()
    if not isinstance(layer, TileLayer):
        return
    data = np.asarray(layer.data)
    height, width = data.shape
    x, y = int(cell[0]), int(cell[1])
    if not (0 <= x < width and 0 <= y < height):
        return
    same = data == data[y, x]
    wanted = same if imgui.get_io().key_ctrl else plotter_tools.flood_mask(same, x, y)
    _selection_algebra(
        state,
        tab.doc,
        np.asarray(wanted, dtype=bool),
        imgui.get_io().key_shift,
        imgui.get_io().key_alt,
    )


def _select_input(state: Any, tab: Any, cell: tuple[int, int], hovered: bool) -> None:
    """The marquee gesture: press anchors, drag resizes, a plain click clears.

    Tiled's behaviour, including the last part -- a click with no drag is how
    you get rid of a selection without reaching for a menu. Stored unclamped
    (``selection_in`` clamps at use), and it writes nothing to the document, so
    none of this is guarded by ``tab.busy``: the guard above already returned.
    """
    from imgui_bundle import imgui

    if hovered and imgui.is_mouse_clicked(0):
        state.drag_kind = "select"
        state.drag_anchor = cell
        # Remembered so a Shift/Alt drag can combine with what was selected
        # before it started rather than with the rectangle it is drawing.
        state.select_before = (state.select, state.select_mask)
        state.set_selection(_normalized(cell, cell))
    elif state.drag_kind == "select" and imgui.is_mouse_down(0):
        if state.drag_anchor is not None:
            state.set_selection(_normalized(state.drag_anchor, cell))
    elif state.drag_kind == "select" and imgui.is_mouse_released(0):
        # A press and release inside one cell is a click, not a one-cell
        # selection: the marquee is already that rect, so compare rather than
        # tracking whether the pointer moved.
        if state.select == _normalized(cell, cell) and not (
            imgui.get_io().key_shift or imgui.get_io().key_alt
        ):
            state.set_selection(None)
        elif state.drag_anchor is not None and (
            imgui.get_io().key_shift or imgui.get_io().key_alt
        ):
            # The same algebra the wand uses, so "hold Shift to add" is one
            # thing to learn. Applied at release rather than during the drag,
            # because the rectangle is still being sized until then.
            import numpy as np

            wanted = np.zeros((int(tab.doc.height), int(tab.doc.width)), dtype=bool)
            x0, y0, x1, y1 = _normalized(state.drag_anchor, cell)
            x0, y0 = max(0, x0), max(0, y0)
            x1 = min(int(tab.doc.width) - 1, x1)
            y1 = min(int(tab.doc.height) - 1, y1)
            if x1 >= x0 and y1 >= y0:
                wanted[y0 : y1 + 1, x0 : x1 + 1] = True
            state.set_selection(_before_drag(state), _mask_before_drag(state))
            _selection_algebra(
                state, tab.doc, wanted, imgui.get_io().key_shift, imgui.get_io().key_alt
            )
        state.clear_drag()


def _apply_drag(ctx: Any, state: Any, tab: Any, cell: tuple[int, int]) -> None:
    """One frame of a paint drag, with the run since the last frame filled in.

    A drag is sampled once a frame, so a fast one arrives as a sequence of cells
    with gaps between them; painting only the cells the pointer was *seen* in
    leaves a dotted line. Every write goes into the session already open, so the
    interpolation costs no extra undo steps.
    """
    previous = state.drag_last_cell
    if previous is None:
        _apply(ctx, state, tab, cell)
    else:
        # ``[1:]`` drops the cell painted last frame. Two samples inside one cell
        # then leave nothing to do, which falls out rather than being tested for.
        for step in plotter_tools.line(previous[0], previous[1], cell[0], cell[1])[1:]:
            _apply(ctx, state, tab, step)
    state.drag_last_cell = cell
    if state.tool == "stamp":
        state.last_paint = cell


def _apply_line(ctx: Any, state: Any, tab: Any, a: tuple[int, int], b: tuple[int, int]) -> None:
    if state.brush is None:
        # Checked once here rather than per cell: ``_apply`` toasts about an
        # empty hand, and a forty-cell line would raise forty of them.
        ctx.toast("Pick a tile from the tileset first.", "error")
        return
    for cell in plotter_tools.line(a[0], a[1], b[0], b[1]):
        _apply(ctx, state, tab, cell)


def _terrain_ref(state: Any, doc: Any):
    """The tileset reference the chosen terrain belongs to, or ``None``.

    Re-resolved per call rather than cached on the state: tilesets are
    append-only but a *replace* swaps the reference, and a cached one would go
    on painting from the atlas the polish pass retired.
    """
    if state.terrain is None:
        return None
    index, rank = state.terrain
    if index >= len(doc.tilesets):
        return None
    ref = doc.tilesets[index]
    return ref if rank < len(ref.tileset.terrains) else None


def _terrain_cell_ref(doc: Any, data: Any, cell: tuple[int, int]):
    """The terrain set the cell under the cursor belongs to, or ``None``.

    Asked of the *cell* rather than of the palette, so Erase self-selects per
    cell: a plain tile erases as a plain tile even on a map that carries a
    terrain set, and a terrain cell re-fits its neighbours even when the tool in
    hand is not Terrain. That is the only rule that makes one eraser correct on
    a mixed map -- the alternative is a second eraser and a user deciding which.
    """
    for ref in doc.tilesets:
        if not ref.tileset.is_terrain_set:
            continue
        if plotter_terrain.terrain_at(data, cell[0], cell[1], ref) is not None:
            return ref
    return None


def _fill_rng(state: Any):
    """The generator a fill scatters with, or ``None`` for a pattern/plain fill.

    Random mode applied to Fill and Shape as it already was to Stamp, per landed
    cell, by ``random_stamp``'s own choice rule. ``None`` off, so the ordinary
    fill is byte-identical to what it was.
    """
    return plotter_tools.fill_rng() if state.random_mode else None


def _constrained(state: Any, doc: Any, result):
    """A tool's region cut down to the marquee, if there is one.

    One place rather than per tool: the tools go on computing what they would
    do, and this decides what is allowed to land. ``None`` in, ``None`` out --
    and ``None`` out is also the honest answer for a placement entirely outside
    the selection, which is the same thing every tool already says about a
    placement entirely off the map.
    """
    if result is None:
        return None
    rect = state.selection_in(doc)
    if rect is None:
        return result
    layer = doc.active()
    data = layer.data if isinstance(layer, TileLayer) else None
    mask = state.selection_mask_in(doc)
    if mask is None or data is None:
        return plotter_tools.clip_region(result, rect)
    # A non-rectangular selection puts the refused cells back to what the layer
    # already had, so the placement is still one rectangular write.
    return plotter_tools.clip_region_mask(result, data, rect, mask)


def _layer_for_paint(ctx: Any, tab: Any):
    layer = tab.doc.active()
    if not isinstance(layer, TileLayer):
        ctx.toast("Pick a tile layer to paint on.", "error")
        return None
    if _active_locked(tab):
        # Enforced here rather than in the engine, deliberately: ``write_region``
        # must go on working on a locked layer or an undo could not put back
        # what was there before the lock. The lock stops *the user* painting,
        # not the document from being written to.
        ctx.toast(f"{layer.name} is locked.", "error")
        return None
    return layer


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
                # By identity, not ``.index``: a ``TilesetRef`` holds ndarrays,
                # so ``==`` is an elementwise comparison whose truth value is
                # ambiguous, and ``list.index`` only survives it by
                # short-circuiting on the firstgid it happens to compare first.
                state.tileset_index = next(
                    i for i, entry in enumerate(doc.tilesets) if entry is ref
                )
        return
    # Whether what comes back is a blob *re-fit* rather than a placement. A
    # re-fit reaches the eight cells around what it touched by design, so
    # trimming it to a selection would leave that ring showing the edge art of a
    # neighbour that is no longer what it was -- a visibly broken field, which is
    # worse than a tool that reaches one cell past a marquee.
    refits = False
    if state.tool == "stamp":
        if state.brush is None:
            ctx.toast("Pick a tile from the tileset first.", "error")
            return
        result = (
            plotter_tools.random_stamp(layer.data, cell[0], cell[1], state.brush)
            if state.random_mode
            else plotter_tools.stamp(layer.data, cell[0], cell[1], state.brush)
        )
    elif state.tool == "erase":
        # Erasing a terrain cell is not erasing a tile: the hole has to grow an
        # outline on everything that now borders it, or the field keeps the edge
        # art of a neighbour that is no longer there. Self-selecting per cell, so
        # a plain tile on the same map still erases as a plain tile.
        ref = _terrain_cell_ref(doc, layer.data, cell)
        refits = ref is not None
        result = (
            plotter_tools.erase(layer.data, cell[0], cell[1])
            if ref is None
            else plotter_terrain.erase_terrain(layer.data, cell[0], cell[1], ref)
        )
    elif state.tool == "fill":
        ref = _terrain_ref(state, doc) if state.brush is None else None
        if ref is not None:
            # A terrain in hand and no tile picked is an unambiguous request:
            # flood the field. The flood is over the *rank* field, so it crosses
            # a terrain's own forty-seven cases rather than stopping at the
            # first edge tile -- see ``terrain.fill_terrain``.
            refits = True
            result = plotter_terrain.fill_terrain(
                layer.data, cell[0], cell[1], state.terrain[1], ref
            )
        elif state.brush is None:
            ctx.toast("Pick a tile from the tileset first.", "error")
            return
        else:
            # Bounded up front, not clipped afterwards: a fill trimmed at the
            # end would be free to leave the selection, run around the outside
            # and come back in, and the trim would hide the trip.
            result = plotter_tools.flood_fill(
                layer.data,
                cell[0],
                cell[1],
                int(state.brush[0, 0]),
                bounds=state.selection_in(doc),
                # The mask is applied to the *match* for the same reason the
                # bounds are, one step finer: a concave selection a fill could
                # leave and re-enter is exactly the escape the bounds prevent.
                mask=state.selection_mask_in(doc),
                brush=state.brush if state.brush.size > 1 else None,
                rng=_fill_rng(state),
            )
    elif state.tool == "terrain":
        ref = _terrain_ref(state, doc)
        if ref is None:
            ctx.toast("Pick a terrain first.", "error")
            return
        refits = True
        result = plotter_terrain.paint_terrain(
            layer.data, cell[0], cell[1], state.terrain[1], ref
        )
    else:
        return
    if not refits:
        result = _constrained(state, doc, result)
    if result is None:
        return
    # Inside a drag the write goes to the open session, which pushes nothing:
    # the whole stroke becomes one step at release. Outside one -- a single
    # click, a fill -- ``write_region`` takes the diff and pushes as it always
    # has, so the two paths differ only in when the step appears.
    if doc.stroking:
        doc.stroke_write(*result)
    else:
        doc.write_region(layer.uid, *result)


def _apply_shape(ctx: Any, state: Any, tab: Any, a: tuple[int, int], b: tuple[int, int]) -> None:
    """What a shape drag lands at release, rect or ellipse.

    One dispatch rather than two tools: the gesture, the preview and the undo
    step are the same either way, and only the set of cells differs.
    """
    layer = _layer_for_paint(ctx, tab)
    if layer is None:
        return
    if state.brush is None:
        ctx.toast("Pick a tile from the tileset first.", "error")
        return
    fill = plotter_tools.fill_ellipse if state.shape_mode == "ellipse" else plotter_tools.fill_rect
    result = _constrained(
        state,
        tab.doc,
        fill(
            layer.data,
            a[0],
            a[1],
            b[0],
            b[1],
            int(state.brush[0, 0]),
            # A multi-tile brush makes this a pattern fill, anchored to map
            # coordinates so two shapes continue one pattern; a 1x1 brush is
            # byte-identical to the plain fill it always was.
            brush=state.brush if state.brush.size > 1 else None,
            rng=_fill_rng(state),
        ),
    )
    if result is not None:
        tab.doc.write_region(layer.uid, *result)


def _apply_erase_rect(
    ctx: Any, state: Any, tab: Any, a: tuple[int, int], b: tuple[int, int]
) -> None:
    """Shift-drag Erase: clear one constrained rectangle in one undo step."""
    layer = _layer_for_paint(ctx, tab)
    if layer is None:
        return
    result = _constrained(
        state,
        tab.doc,
        plotter_tools.fill_rect(layer.data, a[0], a[1], b[0], b[1], gidlib.EMPTY),
    )
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
    # In the *layer's* space, not the map's: the handle tests below compare
    # against object coordinates, and an object on an offset layer is drawn
    # somewhere its own ``x``/``y`` does not say. One subtraction here rather
    # than an addition at every comparison.
    # One resolver walk for both answers the gesture needs -- where the layer's
    # content sits, and whether anything above it is locked -- rather than two.
    entry = _active_entry(tab)
    locked = entry is not None and entry.locked
    dx, dy = (0.0, 0.0) if entry is None else _layer_shift(tab.view, origin, entry)
    raw = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)
    point = (raw[0] - dx, raw[1] - dy)

    if hovered and imgui.is_mouse_clicked(0):
        # Priority: a handle on the selected rect, then any object's body, then
        # empty space. Handles win because they sit *on* the outline and so
        # overlap the body -- checking the body first would make them unreachable.
        selected = next((o for o in layer.objects if o.uid == state.selected_object), None)
        vertex = (
            _vertex_at(tab.view, origin, selected, (mouse.x, mouse.y), (dx, dy))
            if selected is not None and not locked
            else None
        )
        if vertex is not None:
            if imgui.get_io().key_alt:
                _remove_vertex(ctx, doc, layer, selected, vertex)
                return
            doc.begin_object_edit(layer.uid, selected.uid)
            state.drag_kind = "object-vertex"
            state.drag_vertex = vertex
            return
        if (
            selected is not None
            and not locked
            and imgui.get_io().key_ctrl
            and _vertex_points(selected) is not None
        ):
            found = _segment_at(tab.view, origin, selected, (mouse.x, mouse.y), (dx, dy))
            if found is not None:
                index, at = found
                points = list(_vertex_points(selected))
                points.insert(index + 1, at)
                _with_points(ctx, doc, layer, selected, tuple(points))
                return
        handle = (
            _handle_at(tab.view, origin, selected, (mouse.x, mouse.y), (dx, dy))
            if selected is not None and not locked
            else None
        )
        if handle is not None:
            doc.begin_object_edit(layer.uid, selected.uid)
            state.drag_kind = "object-resize"
            state.drag_handle = handle
            return
        hit = _object_at(layer, point)
        if hit is not None:
            # Selecting on a locked layer stays allowed -- it is how you read an
            # object's properties, and it changes nothing.
            state.selected_object = hit.uid
            state.drag_kind = ""
            if not locked:
                doc.begin_object_edit(layer.uid, hit.uid)
                state.drag_kind = "object-move"
                # The grab offset, so the object does not jump its own top-left
                # corner to the pointer on the first frame of the drag.
                state.drag_object = (point[0] - hit.x, point[1] - hit.y)
            return
        if locked:
            ctx.toast(f"{layer.name} is locked.", "error")
            return
        state.drag_kind = "object"
        state.drag_object = point
    elif state.drag_kind == "object-move" and imgui.is_mouse_down(0):
        offset = state.drag_object or (0.0, 0.0)
        x, y = point[0] - offset[0], point[1] - offset[1]
        if imgui.get_io().key_ctrl:
            x, y = doc.cell_corner(*doc.cell_at(x, y))
        doc.place_object(x=float(x), y=float(y))
    elif state.drag_kind == "object-resize" and imgui.is_mouse_down(0):
        target = next((o for o in layer.objects if o.uid == state.selected_object), None)
        if target is not None:
            fixed = _opposite(state.drag_handle, target)
            at = point
            if imgui.get_io().key_ctrl:
                at = doc.cell_corner(*doc.cell_at(*point))
            x, y, w, h = _resized(target, fixed, at)
            doc.place_object(x=float(x), y=float(y), w=float(w), h=float(h))
    elif state.drag_kind == "object-vertex" and imgui.is_mouse_down(0):
        target = next((o for o in layer.objects if o.uid == state.selected_object), None)
        points = None if target is None else _vertex_points(target)
        if points is not None and state.drag_vertex is not None:
            at = point
            if imgui.get_io().key_ctrl:
                at = doc.cell_corner(*doc.cell_at(*point))
            # **Into the object's own frame**, which is the ``_resized`` trap by
            # name: the outline is drawn by rotating each point about the
            # object's origin, so taking the map point straight would put the
            # vertex somewhere the outline does not go.
            local = _unrotated_about(
                (at[0] - target.x, at[1] - target.y), target.rotation
            )
            moved = list(points)
            moved[state.drag_vertex] = local
            kind = Polygon if isinstance(target.shape, Polygon) else Polyline
            doc.place_object(shape=kind(tuple(moved)))
    elif state.drag_kind == "object-vertex" and imgui.is_mouse_released(0):
        # One undo step for the whole drag, exactly as a move or a resize is.
        doc.end_object_edit()
        state.clear_drag()
    elif state.drag_kind in ("object-move", "object-resize") and imgui.is_mouse_released(0):
        # One step for the whole gesture, and none at all for a click that never
        # moved anything -- the session finds no diff and pushes nothing.
        doc.end_object_edit()
        state.clear_drag()
    elif state.drag_kind == "object" and imgui.is_mouse_released(0):
        start = state.drag_object or point
        x0, x1 = sorted((start[0], point[0]))
        y0, y1 = sorted((start[1], point[1]))
        w, h = x1 - x0, y1 - y0
        kind = state.object_shape
        if kind != "point":
            w = w if w >= tab.doc.tile_w * 0.25 else float(tab.doc.tile_w)
            h = h if h >= tab.doc.tile_h * 0.25 else float(tab.doc.tile_h)
        tile_gid = 0
        if kind == "tile":
            brush = np.asarray(state.brush).reshape(-1) if state.brush is not None else ()
            tile_gid = next((int(value) for value in brush if int(value)), 0)
            if not tile_gid and doc.tilesets:
                tile_gid = int(doc.tilesets[0].firstgid)
        obj = plotter_layers.add_object(
            doc,
            layer,
            kind,
            x0,
            y0,
            w if kind != "point" else 0.0,
            h if kind != "point" else 0.0,
            gid=tile_gid,
        )
        state.selected_object = obj.uid
        state.clear_drag()


# The four corners of a rect object, in the order the handles are drawn. Named
# by the corner that *moves*; a resize pins the opposite one.
_HANDLES = ("nw", "ne", "se", "sw")


def _handle_corners(obj: Any) -> dict[str, tuple[float, float]]:
    """Each handle's position in map-pixel space."""
    return {
        "nw": _rotated(obj, 0.0, 0.0),
        "ne": _rotated(obj, obj.w, 0.0),
        "se": _rotated(obj, obj.w, obj.h),
        "sw": _rotated(obj, 0.0, obj.h),
    }


def _opposite(handle: str, obj: Any) -> tuple[float, float]:
    """The corner a resize keeps still."""
    return _handle_corners(obj)[{"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}[handle]]


# --- polygon and polyline vertices -------------------------------------------
#
# Everything here works in the object's **own unrotated frame** and converts on
# the way in and out -- ``_resized``'s trap by name. The outline is drawn by
# rotating each point about the object's origin, so a vertex on a rotated
# polygon is nowhere near where its stored coordinates say; taking a map point
# straight as a vertex would move it somewhere the outline does not go, and the
# error grows with the rotation angle rather than announcing itself at zero.


def _vertex_points(obj: Any) -> tuple[tuple[float, float], ...] | None:
    """The stored points of a polygon or polyline, or ``None`` for anything else."""
    shape = getattr(obj, "shape", None)
    if isinstance(shape, (Polygon, Polyline)):
        return tuple(shape.points)
    return None


def _rotated_about(point: tuple[float, float], degrees: float) -> tuple[float, float]:
    angle = math.radians(float(degrees))
    cos, sin = math.cos(angle), math.sin(angle)
    return (
        point[0] * cos - point[1] * sin,
        point[0] * sin + point[1] * cos,
    )


def _unrotated_about(point: tuple[float, float], degrees: float) -> tuple[float, float]:
    return _rotated_about(point, -float(degrees))


def _vertex_at(
    view: Any,
    origin,
    obj: Any,
    mouse: tuple[float, float],
    shift: tuple[float, float] = (0.0, 0.0),
) -> int | None:
    """Which vertex the pointer is over, or ``None``.

    Screen space, ``_handle_at``'s rule and for its reason: the handles are
    drawn at a fixed ``sp()`` size, so a map-space radius would shrink to
    nothing at low zoom exactly where they are hardest to hit.
    """
    points = _vertex_points(obj)
    if points is None:
        return None
    for index, point in enumerate(points):
        turned = _rotated_about(point, obj.rotation)
        sx, sy = inker_state.to_screen(
            view, origin, obj.x + turned[0] + shift[0], obj.y + turned[1] + shift[1]
        )
        if abs(sx - mouse[0]) <= sp(6) and abs(sy - mouse[1]) <= sp(6):
            return index
    return None


def _segment_at(
    view: Any,
    origin,
    obj: Any,
    mouse: tuple[float, float],
    shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[int, tuple[float, float]] | None:
    """The segment the pointer is on and where on it, for an insert.

    Returns ``(index after which to insert, the point in the object's frame)``.
    A polygon's last segment closes back to its first point; a polyline's does
    not, which is the whole difference between the two shapes here.
    """
    points = _vertex_points(obj)
    if points is None or len(points) < 2:
        return None
    closed = isinstance(obj.shape, Polygon)
    count = len(points) if closed else len(points) - 1
    best: tuple[float, int, tuple[float, float]] | None = None
    for index in range(count):
        a = points[index]
        b = points[(index + 1) % len(points)]
        sa = inker_state.to_screen(
            view,
            origin,
            *(v + s for v, s in zip(_offset_point(obj, a), shift, strict=True)),
        )
        sb = inker_state.to_screen(
            view,
            origin,
            *(v + s for v, s in zip(_offset_point(obj, b), shift, strict=True)),
        )
        along = _projection(sa, sb, mouse)
        if along is None:
            continue
        distance, fraction = along
        if distance > sp(6):
            continue
        at = (a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction)
        if best is None or distance < best[0]:
            best = (distance, index, at)
    return None if best is None else (best[1], best[2])


def _offset_point(obj: Any, point: tuple[float, float]) -> tuple[float, float]:
    turned = _rotated_about(point, obj.rotation)
    return (obj.x + turned[0], obj.y + turned[1])


def _projection(
    a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]
) -> tuple[float, float] | None:
    """``(distance, fraction along)`` of ``p`` onto segment ``a``-``b``."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    if length <= 0.0:
        return None
    fraction = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length
    fraction = max(0.0, min(1.0, fraction))
    nx, ny = ax + dx * fraction, ay + dy * fraction
    return math.hypot(p[0] - nx, p[1] - ny), fraction


def _with_points(
    ctx: Any, doc: Any, layer: Any, obj: Any, points: tuple[tuple[float, float], ...]
) -> bool:
    """Replace an object's shape with the same kind holding new points.

    Shapes are frozen records, so a moved vertex is a *new* ``Polygon`` or
    ``Polyline`` through ``set_object`` -- ``merged_object_values`` is the one
    reconciliation door and there is no writing through a shape.
    """
    kind = Polygon if isinstance(obj.shape, Polygon) else Polyline
    try:
        shape = kind(tuple(points))
    except ValueError as exc:
        ctx.toast(f"{exc}.", "error")
        return False
    doc.set_object(layer.uid, obj.uid, shape=shape)
    return True


def _remove_vertex(ctx: Any, doc: Any, layer: Any, obj: Any, index: int) -> None:
    """Alt+click a vertex to delete it, floored at the shape's minimum.

    Refused by toast at the floor rather than silently: a polygon with two
    points and a polyline with one are not shapes, and the frozen record would
    raise out of the middle of the click.
    """
    points = _vertex_points(obj)
    if points is None:
        return
    floor = 3 if isinstance(obj.shape, Polygon) else 2
    if len(points) <= floor:
        what = "polygon" if floor == 3 else "polyline"
        ctx.toast(f"A {what} needs at least {floor} points.", "warn")
        return
    remaining = [p for i, p in enumerate(points) if i != index]
    _with_points(ctx, doc, layer, obj, tuple(remaining))


def _handle_at(
    view: Any,
    origin,
    obj: Any,
    mouse: tuple[float, float],
    shift: tuple[float, float] = (0.0, 0.0),
) -> str | None:
    """Which resize handle the pointer is over, or ``None``.

    Tested in *screen* space, because the handles are drawn at a fixed ``sp()``
    size: their grab radius has to be the same however far the map is zoomed
    out, and a map-space radius would shrink to nothing at low zoom exactly
    where the handles are hardest to hit. ``shift`` is the layer's displacement,
    the same one ``_objects`` drew the handles at.
    """
    shape = getattr(obj, "shape", None)
    if shape is None:
        # Backwards-compatible structural objects only carry a kind and
        # dimensions. Rectangles remain resizable; points do not gain handles.
        if getattr(obj, "kind", "") != "rect":
            return None
    elif not hasattr(shape, "w"):
        return None
    for name, (px, py) in _handle_corners(obj).items():
        sx, sy = inker_state.to_screen(view, origin, px + shift[0], py + shift[1])
        if abs(sx - mouse[0]) <= sp(6) and abs(sy - mouse[1]) <= sp(6):
            return name
    return None


def _resized(obj: Any, fixed: tuple[float, float], point: tuple[float, float]):
    """``(x, y, w, h)`` from the pinned corner and the pointer.

    **Worked in the object's own unrotated frame, then put back.** Everything
    around this already speaks rotated map space -- ``_handle_corners`` rotates
    the four corners, ``_opposite`` picks the rotated one to pin, ``_handle_at``
    hit-tests where they are actually drawn -- and this one function used to
    take the min and max of two *map* points and write the result straight onto
    ``x``/``y``/``w``/``h``. Those fields are the object's **unrotated** origin
    and extent (``_rotated`` turns them into what you see), so for any
    ``rotation != 0`` the answer described a different rectangle in a different
    place: grabbing a corner made the object jump. Rotation is newly authorable
    here and arrives on every import, so this stopped being unreachable.

    The arithmetic: with ``R`` the object's rotation and ``P`` the pinned corner
    (already in map space), the pointer's offset ``M - P`` rotated *back* by
    ``R`` is the new rectangle's diagonal in object space. Its absolute
    components are the new width and height, and the corner nearest the local
    origin -- ``min(0, ...)`` on each axis, which is what normalizes a drag past
    the opposite corner into a flip rather than a negative size -- rotated
    forward again and added to ``P`` is the new origin. At zero rotation every
    term collapses and this is exactly the sorted min/max it replaces, which is
    what ``test_dragging_a_corner_past_its_opposite_flips_rather_than_going_negative``
    still pins.

    ``handle`` is gone from the signature because ``fixed`` already says which
    corner is pinned and the normalization already says which way the drag
    went; taking both invited them to disagree.
    """
    angle = math.radians(float(getattr(obj, "rotation", 0.0)))
    cosine, sine = math.cos(angle), math.sin(angle)
    mx, my = point[0] - fixed[0], point[1] - fixed[1]
    # R is a rotation, so its inverse is its transpose -- no division, and no
    # special case at 90 degrees.
    local_x = mx * cosine + my * sine
    local_y = -mx * sine + my * cosine
    left, top = min(0.0, local_x), min(0.0, local_y)
    return (
        fixed[0] + left * cosine - top * sine,
        fixed[1] + left * sine + top * cosine,
        abs(local_x),
        abs(local_y),
    )


def _object_at(layer: Any, point: tuple[float, float]):
    """Topmost first, so a small object drawn over a large one is reachable."""
    x, y = point
    for obj in reversed(layer.objects):
        if obj.kind == "point":
            if abs(obj.x - x) <= 8 and abs(obj.y - y) <= 8:
                return obj
            continue
        local_x, local_y = _object_local(obj, x, y)
        if obj.kind == "ellipse":
            rx, ry = obj.w * 0.5, obj.h * 0.5
            if rx and ry and ((local_x - rx) / rx) ** 2 + ((local_y - ry) / ry) ** 2 <= 1:
                return obj
        # One arm per kind, and the *kind* alone decides the arm. These two
        # used to share an arm whose condition was "polygon and inside, or
        # polyline and near" -- so a polygon the pointer missed did not stop
        # there, it fell through the capsule arm into the rectangle test at the
        # bottom and was picked by its bounding box. Harmless only because a
        # polygon's derived ``w``/``h`` are currently zero, which makes that a
        # hit at the object's origin and nowhere else; it becomes a real
        # mis-selection the moment those shapes report an extent.
        elif obj.kind == "polygon":
            if _inside_polygon(local_x, local_y, obj.shape.points):
                return obj
        elif obj.kind == "polyline":
            if _near_polyline(local_x, local_y, obj.shape.points):
                return obj
        elif obj.kind == "capsule":
            if _inside_capsule(local_x, local_y, obj.w, obj.h):
                return obj
        elif 0 <= local_x <= obj.w and 0 <= local_y <= obj.h:
            return obj
    return None


def _object_local(obj: Any, x: float, y: float) -> tuple[float, float]:
    angle = math.radians(-float(obj.rotation))
    cosine, sine = math.cos(angle), math.sin(angle)
    dx, dy = x - obj.x, y - obj.y
    return (dx * cosine - dy * sine, dx * sine + dy * cosine)


def _inside_polygon(x: float, y: float, points: Any) -> bool:
    inside = False
    previous = points[-1]
    for current in points:
        x0, y0 = previous
        x1, y1 = current
        if (y0 > y) != (y1 > y):
            crossing = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _near_polyline(x: float, y: float, points: Any, tolerance: float = 8.0) -> bool:
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        dx, dy = x1 - x0, y1 - y0
        length_sq = dx * dx + dy * dy
        t = (
            0.0
            if not length_sq
            else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / length_sq))
        )
        if math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) <= tolerance:
            return True
    return False


def _inside_capsule(x: float, y: float, width: float, height: float) -> bool:
    if width >= height:
        radius = height * 0.5
        near_x = max(radius, min(width - radius, x))
        near_y = radius
    else:
        radius = width * 0.5
        near_x = radius
        near_y = max(radius, min(height - radius, y))
    return math.hypot(x - near_x, y - near_y) <= radius
