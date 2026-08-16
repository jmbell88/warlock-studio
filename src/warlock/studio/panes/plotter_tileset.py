"""Plotter's left-bottom pane: the map's tilesets and the tile picker.

**This pane is the sole owner of what a stamp puts down.** The tools pane says
what a click means and the layers pane says where it lands; the brush -- which
tile, or which rectangular block of tiles -- is decided here and nowhere else.

The picker draws the tileset's atlas as one image and the selection as a
draw-list rectangle over it, rather than a button per tile: a 16x16 tileset is
256 buttons, and imgui would spend a per-item id, a hover test and a draw call
on each of them every frame.

**The picker leads the pane, and the three ways of acquiring a tileset follow
it in order of how often they are reached for.** The pane used to open with
*Add from a file...* and the generator header, which put two once-per-map
controls above the one control that is used on every single click. Ordering by
frequency rather than by lifecycle is the rule: pick a tile, then add a file,
then repaint in Inker, then -- last -- generate a set from nothing. The empty
map is the one exception and says why inline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import icons, plotter_mode, widgets
from ..manual import render as manual_render
from ..plotter import gid as gidlib
from ..tokens import sp
from . import plotter_textures


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Tilesets")
    manual_render.help_button(ctx, "plotter-tileset")

    if tab is None:
        widgets.muted("Open or start a map first.")
        return

    doc = tab.doc
    disabled = tab.busy

    def add_button() -> None:
        if widgets.disabled_button(f"{icons.PLUS} Add from a file...", not disabled, (-1, 0)):
            plotter_mode.ask_add_tileset(ctx)

    if not doc.tilesets:
        # The one branch where *acquiring* a tileset is the whole pane. There is
        # nothing to pick and nothing to polish, so the two ways of getting one
        # are what the pane is for and they lead it -- burying the generator
        # under a picker that cannot draw would put a new map's most useful
        # control at the bottom of an otherwise empty panel.
        add_button()
        imgui.dummy((0, 4))
        widgets.muted_wrapped(
            "A map needs a tileset before anything can be painted. Add a PNG and it "
            f"is sliced at {doc.tile_w} x {doc.tile_h}, add a Tiled .tsx, or generate "
            "a ground set below."
        )
        imgui.dummy((0, 8))
        _generator(ctx, state, tab)
        return

    options = [
        (str(index), f"{ref.tileset.name} ({ref.tileset.tile_count})")
        for index, ref in enumerate(doc.tilesets)
    ]
    picked = widgets.labeled_combo("Tileset", str(state.tileset_index), options)
    if picked != str(state.tileset_index):
        state.tileset_index = int(picked)
        state.brush = None

    index = max(0, min(state.tileset_index, len(doc.tilesets) - 1))
    state.tileset_index = index
    ref = doc.tilesets[index]
    imgui.dummy((0, 4))
    _picker(ctx, state, ref, index, tab.uid)

    # Everything below here is about the pane's *contents* rather than about the
    # brush, in the order the questions get asked: which tile (above), then how
    # another tileset gets on, then how this one gets repainted, and last the
    # one that builds a set from nothing.
    imgui.dummy((0, 8))
    add_button()
    imgui.dummy((0, 4))
    _inker_row(ctx, state, tab, index)
    imgui.dummy((0, 8))
    _generator(ctx, state, tab)


def _picker(ctx: Any, state: Any, ref: Any, index: int, uid: str) -> None:
    from imgui_bundle import imgui

    tileset = ref.tileset
    texture = plotter_textures.tileset_texture(ctx, uid, index, tileset)
    avail = max(imgui.get_content_region_avail().x, sp(80))
    # Whole-pixel zoom, at least 1: a tileset drawn at 0.7 of a pixel per pixel
    # is unreadable, and the horizontal scrollbar that a >1 zoom needs is a
    # better trade than a blurred palette.
    zoom = max(1, int(avail // max(tileset.image_w, 1)))
    width = tileset.image_w * zoom
    height = tileset.image_h * zoom

    origin = imgui.get_cursor_screen_pos()
    if texture is None:
        # No GL context: the headless smoke suite and every state-only test.
        # A placeholder keeps the layout identical so a screenshot pass still
        # exercises the geometry around it.
        widgets.thumb_placeholder(min(width, avail), icons.GRID)
    else:
        imgui.image(widgets.texture_ref(texture), (width, height))

    draw = imgui.get_window_draw_list()
    step_w = (tileset.tile_w + tileset.spacing) * zoom
    step_h = (tileset.tile_h + tileset.spacing) * zoom
    margin = tileset.margin * zoom

    def cell_at(mx: float, my: float) -> tuple[int, int] | None:
        if step_w <= 0 or step_h <= 0:
            return None
        column = int((mx - origin.x - margin) // step_w)
        row = int((my - origin.y - margin) // step_h)
        if 0 <= column < tileset.columns and 0 <= row < tileset.rows:
            return column, row
        return None

    if imgui.is_item_hovered() and texture is not None:
        mouse = imgui.get_mouse_pos()
        hit = cell_at(mouse.x, mouse.y)
        if hit is not None:
            if imgui.is_mouse_clicked(0):
                state.palette_anchor = hit
            if imgui.is_mouse_down(0) and state.palette_anchor is not None:
                state.brush = _block(ref, state.palette_anchor, hit)
            elif imgui.is_mouse_released(0):
                state.palette_anchor = None

    # The current selection, drawn from the brush rather than from the drag, so
    # it survives the mouse leaving the pane and is still right after a tab
    # switch cleared the anchor.
    span = _brush_span(ref, state.brush)
    if span is not None:
        (c0, r0), (c1, r1) = span
        lo = (origin.x + margin + c0 * step_w, origin.y + margin + r0 * step_h)
        hi = (
            origin.x + margin + c1 * step_w + tileset.tile_w * zoom,
            origin.y + margin + r1 * step_h + tileset.tile_h * zoom,
        )
        draw.add_rect(lo, hi, imgui.get_color_u32((1.0, 1.0, 1.0, 0.9)), 0.0, sp(2))

    imgui.dummy((0, 2))
    if state.brush is None:
        widgets.muted_wrapped("Pick a tile -- drag for a multi-tile brush.")
    else:
        rows, columns = state.brush.shape
        widgets.muted(f"Brush: {columns} x {rows} tile(s)")


def _block(ref: Any, a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
    """The encoded gids for a rectangular palette drag.

    Global ids, not local ones: the brush is written straight into a layer, and
    a layer holds gids. Converting at paint time instead would mean the brush
    only made sense next to the tileset it came from, which is exactly the bug
    a tab switch would produce.
    """
    tileset = ref.tileset
    c0, c1 = sorted((a[0], b[0]))
    r0, r1 = sorted((a[1], b[1]))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), gidlib.DTYPE)
    for row in range(r0, r1 + 1):
        for column in range(c0, c1 + 1):
            out[row - r0, column - c0] = ref.firstgid + row * tileset.columns + column
    return out


def _brush_span(
    ref: Any, brush: np.ndarray | None
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Where the current brush sits in *this* tileset, or None if it does not.

    Derived from the brush rather than remembered, so a brush picked from
    another tileset simply draws no outline here instead of outlining the wrong
    tiles.
    """
    if brush is None or brush.size == 0:
        return None
    tileset = ref.tileset
    first = int(brush[0, 0]) & gidlib.GID_MASK
    last = int(brush[-1, -1]) & gidlib.GID_MASK
    if not ref.holds(first) or not ref.holds(last):
        return None
    a = ref.local(first)
    b = ref.local(last)
    return (
        (a % tileset.columns, a // tileset.columns),
        (b % tileset.columns, b // tileset.columns),
    )


# --- generating ---------------------------------------------------------------
#
# A collapsing section here rather than a fifth pane. This pane already answers
# "how does a tileset get into this map", and generating one is that question by
# a third route beside a file and a library asset. A new pane would cost an entry
# in the four-pane skeleton, a ``settings_share`` decision and a help-coverage
# entry; a header costs none of those and remembers whether it is open.
#
# Drawn last on a map that already has a tileset, because it is the last resort
# of the three routes: a generated ground set is what you reach for when there
# is no art to point at, and once there is, it is the least likely answer to
# "where does this tile come from". Collapsed by default, so its cost when it is
# not wanted is one row.


def _generator(ctx: Any, state: Any, tab: Any) -> None:
    from imgui_bundle import imgui

    from ..plotter import blob, project, tilegen

    if not widgets.header(
        f"{icons.WAND} Generate a ground set", default_open=False, persist_key="plotter/generate"
    ):
        return
    manual_render.help_button(ctx, "plotter-terrain")

    doc = tab.doc
    key = f"plotter_gen:{tab.uid}"
    stamp = (doc.tile_w, doc.tile_h, doc.projection)
    form = ctx.state.preview.get(key)
    if not isinstance(form, dict) or form.get("for") != stamp:
        # Rebuilt when the map's geometry moves under it, the idiom the resize
        # form uses: a half-typed set for a 16px map means nothing on a 32px one.
        form = {
            "for": stamp,
            "projection": doc.projection,
            "rows": [
                {"name": entry.name, "colour": [part / 255.0 for part in entry.fill[:3]]}
                for entry in tilegen.DEFAULT_TERRAINS
            ],
        }
        ctx.state.preview[key] = form

    widgets.muted_wrapped(
        f"Blob autotiling: {blob.TILE_COUNT} cells per terrain, so every edge and "
        "inner corner has its own tile. The whole set arrives as one tileset, and "
        "Ctrl+Z takes it back."
    )
    imgui.dummy((0, 4))

    # The projection is settable only while the map is empty, and this is the
    # *only* control that sets it. A set drawn for one lattice paints the wrong
    # shape into every cell already painted for the other, so the moment there
    # is anything to get wrong the answer stops being the user's to change.
    empty = not doc.tilesets and not any(
        bool(layer.data.any()) for layer in doc.tile_layers()
    )
    if empty:
        form["projection"] = widgets.labeled_combo(
            "Projection",
            form["projection"],
            [(name, name.capitalize()) for name in project.PROJECTIONS],
        )
    else:
        widgets.muted(f"Projection: {doc.projection}")
        widgets.help_marker(
            "A map's projection is fixed once anything is on it: a set drawn for the "
            "other lattice would paint the wrong shape into every cell already painted."
        )
    if form["projection"] == project.ISOMETRIC and doc.tile_w != doc.tile_h * 2:
        widgets.hint_text(
            f"An isometric cell is conventionally 2:1 -- {doc.tile_h * 2} x {doc.tile_h} "
            f"rather than {doc.tile_w} x {doc.tile_h}. It will still generate."
        )
    widgets.muted(f"Tiles are drawn at {doc.tile_w} x {doc.tile_h}")

    imgui.dummy((0, 4))
    widgets.muted("Terrains, in order. Later ones sit on top where they meet.")
    remove = None
    for index, row in enumerate(form["rows"]):
        imgui.push_id(index)
        imgui.set_next_item_width(sp(120))
        row["name"] = widgets.input_text("##name", row["name"], max_length=24)
        imgui.same_line()
        changed, colour = imgui.color_edit3(
            "##fill",
            row["colour"],
            imgui.ColorEditFlags_.no_inputs.value | imgui.ColorEditFlags_.no_label.value,
        )
        if changed:
            row["colour"] = list(colour)
        imgui.same_line()
        if widgets.disabled_button(icons.TRASH, len(form["rows"]) > 1, (0, 0)):
            remove = index
        imgui.pop_id()
    if remove is not None:
        form["rows"].pop(remove)

    if widgets.disabled_button(
        f"{icons.PLUS} Add a terrain", len(form["rows"]) < tilegen.MAX_TERRAINS, (-1, 0)
    ):
        form["rows"].append({"name": f"Terrain {len(form['rows']) + 1}", "colour": [0.5, 0.5, 0.5]})

    names = [row["name"].strip() for row in form["rows"]]
    problem = ""
    if "" in names:
        problem = "Every terrain needs a name."
    elif len({name.lower() for name in names}) != len(names):
        problem = "Two terrains cannot share a name."
    if problem:
        widgets.hint_text(problem)

    imgui.dummy((0, 4))
    if widgets.disabled_button("Generate", not tab.busy and not problem, (-1, 0)):
        plotter_mode.generate_terrain_set(
            ctx,
            tilegen.GenSpec(
                terrains=tuple(_spec_of(row) for row in form["rows"]),
                tile_w=doc.tile_w,
                tile_h=doc.tile_h,
                projection=form["projection"],
                name="Ground",
            ),
        )


def _spec_of(row: dict) -> Any:
    """One form row as a :class:`~..plotter.tileset.TerrainSpec`.

    The outline is derived from the fill rather than picked separately: the base
    style *is* a flat fill with a darker edge, so one colour is the whole of what
    a terrain is here, and a second control for it is how the two drift apart.
    """
    from ..plotter.tileset import TerrainSpec

    fill = tuple(int(round(part * 255)) for part in row["colour"]) + (255,)
    return TerrainSpec(
        name=row["name"].strip(),
        fill=fill,
        outline=(*(part * 3 // 5 for part in fill[:3]), 255),
    )


def _inker_row(ctx: Any, state: Any, tab: Any, index: int) -> None:
    """Out to Inker to be painted, and back onto the same tileset.

    The pull direction copies ``packwright_sources._from_inker``: Plotter reaches
    into Inker's open documents rather than Inker pushing into Plotter, so
    neither mode has to know when the other is finished.
    """
    from imgui_bundle import imgui

    if widgets.disabled_button(f"{icons.BRUSH} Polish in Inker", not tab.busy, (-1, 0)):
        plotter_mode.polish_in_inker(ctx, tab, index)

    inker = getattr(ctx.state, "inker", None)
    docs = list(getattr(inker, "docs", []) or [])
    if not docs:
        return
    imgui.dummy((0, 4))
    widgets.muted("from Inker")
    name = tab.doc.tilesets[index].tileset.name
    for entry in docs:
        label = f"{icons.IMAGE} Back onto {name}##ink-{entry.uid}"
        if widgets.disabled_button(label, not tab.busy, (-1, 0)):
            plotter_mode.tileset_from_inker(ctx, entry.doc, index=index)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Repaint {name} with {entry.title}. Every painted cell keeps its tile."
            )
