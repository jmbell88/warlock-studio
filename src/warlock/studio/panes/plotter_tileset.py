"""Plotter's left-bottom pane: the map's tilesets and the tile picker.

**This pane is the sole owner of what a stamp puts down.** The tools pane says
what a click means and the layers pane says where it lands; the brush -- which
tile, or which rectangular block of tiles -- is decided here and nowhere else.

The picker draws the tileset's atlas as one image and the selection as a
draw-list rectangle over it, rather than a button per tile: a 16x16 tileset is
256 buttons, and imgui would spend a per-item id, a hover test and a draw call
on each of them every frame.

**The picker leads the pane, and the ways of acquiring a tileset follow it in
order of how often they are reached for.** The pane used to open with *Add from
a file...*, which put a once-per-map control above the one control that is used
on every single click. Ordering by frequency rather than by lifecycle is the
rule: pick a tile, then add a file, then repaint in Inker. The empty map is the
one exception and says why inline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import controls, icons, plotter_mode, widgets
from ..manual import render as manual_render
from ..tilegrid import gid as gidlib
from ..tokens import sp
from . import plotter_textures

SHEET_POPUP = "plotter-sheet-import"


def _sheet_popup(ctx: Any, state: Any, tab: Any) -> None:
    """Confirm what to do with a sheet the detector found rules in.

    The whole reason this is a popup and not an automatic slice: a genuinely
    dark drawing -- a night scene, a silhouette set -- can rule itself off
    convincingly, and the user is the only one who can tell that from a real
    tilesheet. So the detector's answer is *shown*, and the second button does
    exactly what the editor did before the detector existed.
    """
    from imgui_bundle import imgui

    if not imgui.begin_popup(SHEET_POPUP):
        # imgui closes a popup on a click outside, and a parked 1024 sheet is
        # megabytes: dropping it here is what keeps a dismissed import from
        # pinning the pixels for the rest of the session.
        if state.sheet_import_open:
            plotter_mode.clear_sheet_import(ctx)
        return
    widgets.popup_chrome(_imgui=imgui)
    if state.sheet_import is None:
        imgui.end_popup()
        return
    _uid, name, _source, pixels, grid = state.sheet_import
    height, width = pixels.shape[:2]
    doc = tab.doc
    widgets.muted(f"{name} - {width} x {height}")
    if isinstance(grid, plotter_mode.SheetTerrain):
        # The third variant. "Import as plain tiles" is the false-positive
        # mitigation here, exactly as the blind slice is for a dark sheet: a
        # sprite set whose cells happen to cover all 47 silhouettes is a
        # coincidence only the user can rule out.
        imgui.text(
            f"These {grid.roles.tile_count} tiles look like a complete "
            f"47-case terrain set."
        )
        widgets.muted_wrapped(
            "Import reorders them into the canonical blob layout and turns on "
            "terrain painting for this tileset."
        )
        imgui.dummy((0, 4))
        if controls.button("Import as terrain set", (sp(170), 0)) and (
            plotter_mode.import_sheet_terrain(ctx)
        ):
            imgui.close_current_popup()
        imgui.same_line()
        if controls.button("Import as plain tiles##sheet-plain", (sp(170), 0)) and (
            plotter_mode.import_sheet_blind(ctx)
        ):
            imgui.close_current_popup()
        imgui.same_line()
        if controls.button("Cancel##sheet", (sp(90), 0)):
            plotter_mode.clear_sheet_import(ctx)
            imgui.close_current_popup()
        imgui.end_popup()
        return
    if isinstance(grid, plotter_mode.SheetMismatch):
        # The second variant: no rules were found, but the sheet's own sidecar
        # records a cell size the map does not share. A blind slice here cuts
        # every tile in half, which is exactly the failure this popup exists to
        # put in front of somebody.
        imgui.text(
            f"This sheet was generated at {grid.tile_w} x {grid.tile_h}; "
            f"this map's tiles are {doc.tile_w} x {doc.tile_h}."
        )
        widgets.muted_wrapped(
            f"Import cuts the sheet on its own grid and redraws each tile at "
            f"{doc.tile_w} x {doc.tile_h}."
        )
    else:
        rows, cols = grid.shape
        imgui.text(
            f"Detected a {cols} x {rows} tile grid with separator lines "
            f"(threshold {grid.threshold})."
        )
        widgets.muted_wrapped(
            f"Import strips the separator lines and redraws each cell at "
            f"{doc.tile_w} x {doc.tile_h}, so the tiles line up with the map."
        )
    imgui.dummy((0, 4))
    if controls.button("Import", (sp(90), 0)) and plotter_mode.import_detected_sheet(ctx):
        imgui.close_current_popup()
    imgui.same_line()
    label = f"Slice at {doc.tile_w} x {doc.tile_h} instead"
    if controls.button(f"{label}##sheet-blind", (sp(190), 0)) and (
        plotter_mode.import_sheet_blind(ctx)
    ):
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel##sheet", (sp(90), 0)):
        plotter_mode.clear_sheet_import(ctx)
        imgui.close_current_popup()
    imgui.end_popup()


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

    # **Above the empty-map early return, deliberately.** A ruled sheet arriving
    # onto a map with no tilesets yet is the single most common way this feature
    # is reached -- it is how a map gets its first tileset -- and a pump placed
    # after the return would leave that case parking pixels that no popup ever
    # asks about.
    parked = state.sheet_import
    if parked is not None and parked[0] == tab.uid and not state.sheet_import_open:
        state.sheet_import_open = True
        imgui.open_popup(SHEET_POPUP)
    _sheet_popup(ctx, state, tab)

    if not doc.tilesets:
        # The one branch where *acquiring* a tileset is the whole pane. There is
        # nothing to pick and nothing to polish, so the way of getting one is
        # what the pane is for and it leads.
        add_button()
        imgui.dummy((0, 4))
        widgets.muted_wrapped(
            "A map needs a tileset before anything can be painted. Add a PNG and it "
            f"is sliced at {doc.tile_w} x {doc.tile_h}, add a Tiled .tsx, or make a "
            "tile sheet in Create and bring it in from the library."
        )
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
    _picker(ctx, state, ref, index, tab.uid, doc.tileset_epoch)

    # Everything below here is about the pane's *contents* rather than about the
    # brush, in the order the questions get asked: which tile (above), then how
    # another tileset gets on, then how this one gets repainted.
    imgui.dummy((0, 8))
    add_button()
    imgui.dummy((0, 4))
    _inker_row(ctx, state, tab, index)


def _picker(ctx: Any, state: Any, ref: Any, index: int, uid: str, epoch: int) -> None:
    from imgui_bundle import imgui

    tileset = ref.tileset
    texture = plotter_textures.tileset_texture(ctx, uid, index, tileset, epoch)
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
