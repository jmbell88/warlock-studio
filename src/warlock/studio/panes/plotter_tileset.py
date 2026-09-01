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
from . import plotter_layers, plotter_textures

SHEET_POPUP = "plotter-sheet-import"

#: What this pane refuses to shrink past, in design pixels. It had no floor at
#: all while it sat under the tools pane and took whatever was left; it is the
#: fill slot of a column now, and a fill slot's floor is what
#: ``layout_skeleton.heights`` makes the share above it give way to. The number
#: is a heading, the tileset tabs, one row of the zoom bar and enough atlas to
#: recognise -- below that the picker is a scrollbar with a picture in it.
TILESET_FLOOR = 220.0


#: Every button in this pane that can grey out does so for this one cause, and
#: hoisting it is the ``_VIEWPORT_WHY`` pattern: four dead controls explaining
#: themselves in four different sentences read as four separate problems.
_BUSY_WHY = "This map is being written; the buttons come back when it lands."


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
    if isinstance(grid, plotter_mode.SheetLattice):
        # The fourth variant, and the only one about the *lattice* rather than
        # the cut. A map's projection is fixed once anything is painted on it --
        # a tileset drawn for one lattice paints the wrong shape into every cell
        # of the other -- so there is no "convert" to offer here and pretending
        # otherwise would be the offer-then-refuse shape. On an empty map this
        # popup never opens: the sheet simply brings its lattice with it.
        imgui.text(
            f"This sheet was drawn for an {grid.view} map; this one is "
            f"{grid.lattice}."
        )
        widgets.muted_wrapped(
            "This map has been painted, so its lattice is fixed. Adding the "
            "sheet anyway slices it on this map's grid, which will not line up "
            "with what the tiles were drawn as. Starting a new "
            f"{grid.view} map is the other way round it."
        )
        imgui.dummy((0, 4))
        if controls.button("Add anyway", (sp(120), 0)) and (
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
        # The heading and nothing else. One voice for one empty state:
        # the canvas's ``nothing_open`` is it, and four panels each
        # repeating it reads as four separate problems.
        return

    doc = tab.doc
    disabled = tab.busy

    def add_button() -> None:
        if widgets.disabled_button(
            f"{icons.PLUS} Add from a file...", not disabled, (-1, 0), reason=_BUSY_WHY
        ):
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

    _tileset_tabs(ctx, state, doc, tab.uid)

    index = max(0, min(state.tileset_index, len(doc.tilesets) - 1))
    state.tileset_index = index
    ref = doc.tilesets[index]
    imgui.dummy((0, 4))
    _tileset_bar(ctx, state, tab, ref, index)
    _picker(ctx, state, tab, ref, index, tab.uid, doc.tileset_epoch)
    _tile_form(ctx, state, tab, ref, index)


#: The tab strip's id, and the filter box's key on ``AppState.list_filters``.
TABS = "##plotter-tilesets"
FILTER_TAG = "plotter-tilesets"


def _tileset_tabs(ctx: Any, state: Any, doc: Any, uid: str) -> None:
    """Which tileset the picker shows, as a tab strip with a filter over it.

    **A strip, not a combo.** Tiled puts one tab per tileset here and the reason
    is the gesture rather than the look: swapping between two sets while
    painting is a thing you do dozens of times a minute, and a combo is a click
    to open, a read and a second click for every one of them. A tab is one
    click, and the names are on screen so *which sets this map has* is answered
    without opening anything.

    **The filter box is what makes the strip survive a map with a dozen sets**,
    which is where a strip beats a combo least. It appears at
    ``widgets.list_filter``'s own threshold, which is the count at which the
    tabs start scrolling -- so it arrives exactly when the strip needs it and
    clears itself when it does not.

    Selection is taken from whichever tab reports itself open and written back
    through ``state.tileset_index``, rather than left to imgui: the tab a filter
    hides must not silently become the tileset you are painting with. See
    :func:`.plotter_state.visible_tilesets`, which is why the set in hand is
    never one of the hidden ones.
    """
    from imgui_bundle import imgui

    from .. import plotter_state as ps

    names = [ref.tileset.name for ref in doc.tilesets]
    needle = widgets.list_filter(ctx, FILTER_TAG, len(names))
    current = max(0, min(state.tileset_index, len(names) - 1))
    shown = ps.visible_tilesets(names, needle, current)
    widgets.no_matches(needle, sum(1 for index in shown if index != current))

    # **``set_selected`` only when the index moved from outside the strip.**
    # Passing it every frame is the classic way to build a tab bar nobody can
    # click: imgui re-selects the flagged tab after the press, so every other
    # tab reverts on the same frame it is chosen. What is remembered here is
    # what the strip last reported, so a mismatch means somebody else -- a tab
    # switch, an undone add, the library door -- moved it, and only then is the
    # strip told where to be.
    memo = f"plotter_tabsel:{uid}"
    force = ctx.state.preview.get(memo) != current
    picked = current
    if imgui.begin_tab_bar(TABS, imgui.TabBarFlags_.fitting_policy_scroll.value):
        for index in shown:
            # ``###`` so the id carries the *index* and the visible part carries
            # the name: two tilesets with one name are two tabs rather than one,
            # and a rename cannot move the selection.
            opened, _ = imgui.begin_tab_item(
                f"{names[index]}###ts{index}",
                None,
                imgui.TabItemFlags_.set_selected.value
                if force and index == current
                else 0,
            )
            if opened:
                picked = index
                imgui.end_tab_item()
        imgui.end_tab_bar()
    ctx.state.preview[memo] = picked
    if picked != state.tileset_index:
        # The brush is numbered against the tileset it came from, so it goes
        # with the tab -- ``_forget_document_state``'s rule, one level down.
        state.tileset_index = picked
        state.brush = None
    ref = doc.tilesets[state.tileset_index]
    widgets.muted(f"{ref.tileset.tile_count} tile(s)")


def _picked_local(state: Any, ref: Any) -> int | None:
    """The one tile the palette selection names, or ``None`` for a block.

    A block selection has no single tile to attach metadata to, and offering the
    form for it would make "which tile did that class land on" a question the
    user has to guess the answer to.
    """
    brush = state.brush
    if brush is None or brush.size != 1:
        return None
    value = int(brush.reshape(-1)[0])
    tile = value & gidlib.GID_MASK
    if not ref.holds(tile):
        return None
    return tile - ref.firstgid


def _tile_form(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """The **Tile** header: what one tile carries beyond its picture.

    Under the palette rather than beside it, and shown only when the selection
    is a single tile -- see :func:`_picked_local`.
    """
    from imgui_bundle import imgui

    local = _picked_local(state, ref)
    if local is None:
        return
    imgui.dummy((0, 6))
    if not widgets.header("Tile", default_open=False, persist_key="plotter/tilemeta"):
        return
    meta = ref.tileset.meta_of(local)
    widgets.muted(f"local id {local}")

    class_name = widgets.input_text(
        "##tile-class", meta.class_name, max_length=64, hint="class"
    )
    changed, probability = controls.input_float("Probability", float(meta.probability))
    if class_name != meta.class_name or changed:
        tab.doc.set_tile_meta(
            index,
            local,
            _replaced(meta, class_name=class_name, probability=max(0.0, probability)),
        )
    if meta.probability == 0.0:
        widgets.muted_wrapped(
            "Never chosen by a random brush, and always placeable by hand."
        )

    imgui.dummy((0, 4))
    widgets.muted(
        f"{len(meta.animation)} animation frame(s), "
        f"{len(meta.collision)} collision shape(s)"
    )
    if controls.button("Add frame from selection", (-1, 0)):
        # The palette pick *is* the frame picker -- there is no second control
        # for choosing a tile, because the one above it already is one.
        tab.doc.set_tile_meta(
            index,
            local,
            _replaced(
                meta,
                animation=(*meta.animation, _frame(local)),
            ),
        )
    if meta.animation and controls.button("Remove last frame", (-1, 0)):
        tab.doc.set_tile_meta(
            index, local, _replaced(meta, animation=meta.animation[:-1])
        )
    if meta.collision and controls.button("Clear collision", (-1, 0)):
        tab.doc.set_tile_meta(index, local, _replaced(meta, collision=()))

    imgui.dummy((0, 4))
    widgets.muted("Properties")
    plotter_layers.property_editor(
        ctx,
        f"tilemeta:{tab.uid}:{index}:{local}",
        meta.properties,
        lambda values: tab.doc.set_tile_meta(
            index, local, _replaced(meta, properties=values)
        ),
    )


def _frame(local: int) -> Any:
    from ..tilegrid.tileset import TileFrame

    return TileFrame(local_id=int(local), duration_ms=100)


def _replaced(meta: Any, **values: Any) -> Any:
    """A ``TileMeta`` with some fields changed.

    ``dataclasses.replace`` rather than a constructor call listing every field:
    a seventh field added to the record must not need an edit here to survive
    a class being typed into the form.
    """
    import dataclasses

    return dataclasses.replace(meta, **values)


#: The picker's own scrolling well, in design pixels.
#:
#: A ceiling rather than a fixed size, which is ``widgets.card``'s no-scrollbar
#: argument read the other way round: an atlas is any height at all, and one
#: taller than the pane would push the Tile form, *Add from a file...* and the
#: Inker row off the bottom of a sidebar that has no scrollbar of its own to
#: get them back. A well that scrolls is the trade.
#:
#: The floor is deliberately small. It exists so a one-row tileset is still a
#: target you can drag across rather than a hairline, and for nothing else --
#: a larger one would pad every short atlas with dead space, which the first
#: screenshot of this pane showed it doing.
PICKER_MIN_H = 40.0
PICKER_MAX_H = 260.0


def _resolve_zoom(state: Any, tab: Any, tileset: Any, index: int, avail: float) -> float:
    """The palette's scale for this tileset, fitting it the first time.

    A missing key means "fit", which is what makes a wide atlas arrive whole
    rather than clipped -- the defect this pane had. The Fit button and the
    zoom nudges are folded in here too, so there is one place that decides the
    number and one place that stores it.
    """
    from .. import plotter_state as ps

    zoom = tab.palette_zoom.get(index)
    if zoom is None or state.palette_zoom_fit:
        # ``avail`` for both axes: the well is as wide as the pane and as tall
        # as it is allowed to be, and fitting to the smaller of the two is what
        # "the entire tile set" means.
        zoom = ps.palette_fit_zoom(
            tileset.image_w, tileset.image_h, avail, sp(PICKER_MAX_H)
        )
        state.palette_zoom_fit = False
    if state.palette_zoom_rung:
        zoom = ps.palette_zoom_rung(zoom, state.palette_zoom_rung)
        state.palette_zoom_rung = 0
    tab.palette_zoom[index] = zoom
    return zoom


SOURCES_POPUP = "plotter-tileset-sources"


def _tileset_bar(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """Edit, the zoom group, and every way a tileset gets on or off the map.

    One strip over the picker, where there used to be a zoom row above it and
    a hundred pixels of full-width buttons *below* it -- *Add from a file...*,
    *Polish in Inker* and one *Back onto...* per open Inker document. Those are
    reached once a session and the picker is reached on every click, so they
    were taking the bottom of the pane from the thing the pane is for. Behind
    the ``+`` they cost one glyph.

    **The zoom is Fit / out / in and no combo**, unlike the canvas footer's. A
    map's useful zoom is percentage-shaped and fixed percentages serve it; a
    palette's depends on the atlas *and* the pane width, both of which move, so
    a fixed list would spend most of its rows offering scales that show nothing.

    *Edit tileset* leads because it is the verb this pane could not reach at
    all: the sheet was a Tileset-menu row, three clicks from the picker whose
    tile you wanted to look at.
    """
    from imgui_bundle import imgui

    editable = not tab.busy
    if widgets.disabled_button(
        f"{icons.PENCIL} Edit",
        editable,
        reason=_BUSY_WHY,
        tooltip="Open this tileset's sheet over the map: per-tile class, "
        "properties, animation and terrain",
    ):
        plotter_mode.edit_tileset(ctx, index)
    imgui.same_line()
    if controls.button("Fit", tooltip="Zoom so the whole tileset is on screen."):
        state.palette_zoom_fit = True
    imgui.same_line()
    if controls.button(
        f"{icons.MINUS}##palette-zoom-out",
        tooltip="Zoom out (Ctrl+wheel over the palette)",
    ):
        state.palette_zoom_rung = -1
    imgui.same_line()
    if controls.button(
        f"{icons.PLUS}##palette-zoom-in",
        tooltip="Zoom in (Ctrl+wheel over the palette)",
    ):
        state.palette_zoom_rung = 1
    imgui.same_line()
    if controls.button(
        f"{icons.ELLIPSIS}##palette-sources",
        tooltip="Add a tileset, reload this one, or repaint it in Inker",
    ):
        imgui.open_popup(SOURCES_POPUP)
    _sources_popup(ctx, state, tab, ref, index)
    zoom = tab.palette_zoom.get(index)
    if zoom:
        imgui.same_line()
        # The canvas footer's own spelling, so one number does not read two
        # ways in one mode.
        widgets.muted(f"{zoom * 100:.0f}%")


def _sources_popup(ctx: Any, state: Any, tab: Any, ref: Any, index: int) -> None:
    """Where a tileset comes from, and where it goes to be repainted.

    The four rows the pane used to draw as full-width buttons, in the order the
    questions get asked: how another one gets on, how this one is re-read off
    disk, and how it is repainted -- out to Inker and back onto the same
    tileset, so every painted cell keeps its tile.

    The pull direction copies ``packwright_sources._from_inker``: Plotter reaches
    into Inker's open documents rather than Inker pushing into Plotter, so
    neither mode has to know when the other is finished.
    """

    with controls.menu_popup(SOURCES_POPUP) as opened:
        if not opened:
            return
        editable = not tab.busy
        name = ref.tileset.name
        if _menu_row("Add from a file...", editable):
            plotter_mode.ask_add_tileset(ctx)
        if _menu_row("Reload the image...", editable):
            # The other half of *Polish in Inker*, for a paint program that is
            # not Inker. Through ``MapDoc.replace_tileset``, so the ids, the
            # firstgid and the declared terrains are kept and the map redraws
            # rather than renumbering.
            plotter_mode.ask_replace_tileset(ctx, index)
        controls.menu_separator()
        if _menu_row(f"{icons.BRUSH} Polish in Inker", editable):
            plotter_mode.polish_in_inker(ctx, tab, index)
        inker = getattr(ctx.state, "inker", None)
        for entry in list(getattr(inker, "docs", []) or []):
            row = f"{icons.IMAGE} Back onto {name}##ink-{entry.uid}"
            if _menu_row(
                row,
                editable,
                tooltip=f"Repaint {name} with {entry.title}. Every painted cell "
                "keeps its tile.",
            ):
                plotter_mode.tileset_from_inker(ctx, entry.doc, index=index)


def _menu_row(label: str, enabled: bool, *, tooltip: str = "") -> bool:
    hit = controls.menu_item(
        f"{label}##plotter-tileset-src", "", False, enabled,
        reason=_BUSY_WHY, tooltip=tooltip,
    )
    return bool(hit[0] if isinstance(hit, tuple) else hit)


def _picker(
    ctx: Any, state: Any, tab: Any, ref: Any, index: int, uid: str, epoch: int
) -> None:
    from imgui_bundle import imgui

    tileset = ref.tileset
    texture = plotter_textures.tileset_texture(ctx, uid, index, tileset, epoch)
    avail = max(imgui.get_content_region_avail().x, sp(80))
    zoom = _resolve_zoom(state, tab, tileset, index, avail)
    # **One float decides both the picture and the hit grid.** Rounding the
    # image's size to whole pixels while leaving the step fractional lets the
    # drawn cell boundaries and the computed ones drift apart across a wide
    # atlas -- worst at the right-hand end, which is precisely the part that
    # used to be invisible and is the reason this pane was reported.
    width = tileset.image_w * zoom
    height = tileset.image_h * zoom

    # The well. Horizontal scrolling is what the old comment here already
    # assumed existed and never did: a pane-level flag was the alternative and
    # would have scrolled the Tile form and the buttons with it, as well as
    # letting the property editor's own width push a bar across the whole pane.
    # The scrollbar lives *inside* the child, so its height is reserved only
    # when there is going to be one -- otherwise every atlas that fits pays for
    # a bar it does not have, which is a strip of dead panel under the picture.
    bar = imgui.get_style().scrollbar_size if width > avail else 0.0
    inner_h = min(max(height + bar + sp(2), sp(PICKER_MIN_H)), sp(PICKER_MAX_H))
    imgui.begin_child(
        "##tileset-well",
        (-1, inner_h),
        imgui.ChildFlags_.borders.value,
        imgui.WindowFlags_.horizontal_scrollbar.value,
    )

    origin = imgui.get_cursor_screen_pos()
    if texture is None:
        # No GL context: the headless smoke suite and every state-only test.
        # The *same* rectangle the image would occupy, not a square of its
        # width -- a placeholder that is a different shape from the thing it
        # stands in for makes every headless assertion about this geometry a
        # claim about something the app never draws.
        widgets.thumb_placeholder(width, icons.GRID, height)
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
        if tileset.local_at(column, row) is not None:
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

    # Ctrl+wheel, the gesture a Tiled or Aseprite user already has. Guarded on
    # the modifier so a plain wheel still scrolls the well, and applied next
    # frame through the pending field rather than here -- the image for this
    # frame has already been submitted at the old scale, and re-deriving the
    # geometry underneath it would put the selection rectangle a rung out.
    if imgui.is_window_hovered():
        wheel = imgui.get_io().mouse_wheel
        if wheel and imgui.get_io().key_ctrl:
            state.palette_zoom_rung = 1 if wheel > 0 else -1

    imgui.end_child()

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
            # Through ``local_at`` rather than the row-major arithmetic: a
            # collection's ids are sparse and its last palette row is routinely
            # short, so the product would hand out gids that answer to nothing.
            # ``None`` leaves the cell empty, which a stamp already means.
            local = tileset.local_at(column, row)
            if local is not None:
                out[row - r0, column - c0] = ref.firstgid + local
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
    return (
        _palette_cell(tileset, ref.local(first)),
        _palette_cell(tileset, ref.local(last)),
    )


def _palette_cell(tileset: Any, local: int) -> tuple[int, int]:
    """Where one local id sits in the palette grid.

    The inverse of :meth:`Tileset.local_at`, and it has to go through the
    collection's own slot for the same reason that one does: sparse ids are not
    their own positions.
    """
    if tileset.collection is not None:
        slot = tileset.collection.slot_of(local)
        if slot is None:
            return (0, 0)
        return (slot % tileset.columns, slot // tileset.columns)
    return (local % tileset.columns, local // tileset.columns)


# --- generating ---------------------------------------------------------------
