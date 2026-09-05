"""Inker's tile panel: the document's tilesets, the tile the stamp puts down,
and the three verbs that make a tilemap layer.

**This pane is the sole owner of what the tile stamp writes.** The toolbox says
that a click means "put a tile down"; which tile, and how it is turned, is
decided here and nowhere else -- ``plotter_tileset``'s arrangement, one mode
over, and deliberately the same one: the two panels are the same idea in two
editors, so a user who has painted a map already knows this one.

Two things differ from Plotter's picker, and both come from the model rather
than from taste.

*The atlas is re-flowed rather than drawn whole.* An Inker tileset is an
Aseprite-style **vertical strip** -- one tile wide, however many tall -- so
Plotter's "draw the image and rectangle the selection" would give a column a
hundred tiles long in a 104 px sidebar. The tiles are therefore drawn one item
each, out of one texture with per-tile UVs, and wrapped to the pane's width. It
costs an item and a hover test per tile, which the strip layout buys back: a
tileset here is cut from one canvas, so a 256x256 drawing at 16 px is 256 tiles
at the very most.

*The tileset is not chosen freely.* A stamp writes a **local** id into whatever
tileset the active layer is bound to, so a picker showing another atlas beside
a selected tilemap layer would hand the engine an id naming a different tile.
``InkerState.picked_tileset`` derives the answer instead of remembering it, and
the combo is out (with the reason on it) exactly while a tilemap layer is
active.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, icons, inker_mode, plotter_tilesets, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_textures

#: How wide a tile is drawn in the picker, in design px, before the whole-pixel
#: zoom is worked out. A hair over two rows of text: big enough to tell two
#: 8 px tiles apart, small enough that a 64-tile atlas is a few rows rather
#: than a scroll.
TILE_PX = 28.0

#: The panel's own height in the sidebar, in design px. Fixed rather than a
#: share of the column, ``inker_preview.PREVIEW_H``'s reason: it is a palette,
#: and one that grew with the window would push the colour panel off the bottom
#: on a short screen.
PANEL_H = 240.0

#: The least of it that has to be on screen, in design px, once the panel is a
#: share of the right column rather than a fixed height. The picker's own first
#: row plus its heading: below this the pane is a title with a sliver of tiles
#: under it, which reads as broken rather than as small.
PANEL_FLOOR = 140.0

#: What a tileset may be before the picker stops drawing every tile of it. Far
#: past anything this editor authors (see the module docstring); it exists so a
#: hand-made ``.tsx`` with four thousand tiles in it cannot cost a frame.
MAX_DRAWN = 512


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    tab = state.active
    widgets.section("Tiles")
    manual_render.help_button(ctx, "inker-tiles")

    if tab is None:
        widgets.muted("Open a drawing first.")
        return

    doc = tab.doc
    _verbs(ctx, state, tab)
    imgui.dummy((0, sp(tokens.SP_2)))

    if not doc.tilesets:
        widgets.muted_wrapped(
            "A tilemap layer needs a tileset. Cut one out of the layer you have "
            "with Convert to tilemap, or import a Tiled .tsx."
        )
        imgui.dummy((0, sp(tokens.SP_1)))
        _files(ctx, state, tab, None)
        return

    uid = state.picked_tileset(doc)
    slot = doc.tileset_slot(uid)
    # Every frame, before anything reads the pick: the tileset under it can
    # change (another layer selected) or shrink (an undone Stack append) with
    # nothing to tell this pane it happened, and a pick past the end is a ref
    # that draws blank now and draws the wrong tile the moment the tileset
    # grows past it again. See ``InkerState.clamp_tile_pick``.
    state.clamp_tile_pick(uid, slot.tileset.tile_count)
    _chooser(ctx, state, tab, uid)
    _picker(ctx, state, tab, slot)
    imgui.dummy((0, sp(tokens.SP_1)))
    _flips(state, slot.tileset)
    imgui.dummy((0, sp(tokens.SP_2)))
    _files(ctx, state, tab, uid)


#: Why every button here is out while the document is being written. The
#: ``inker_bridge._busy_why`` sentence, said again rather than imported: this
#: pane draws with no knowledge of playback, which is that function's other
#: half.
BUSY_WHY = "This document is being written; the buttons come back when it lands."


def _verbs(ctx: Any, state: Any, tab: Any) -> None:
    """New tilemap layer, and the way back out of one.

    **Disabled, never hidden** -- the Wave 2 rule: a panel that quietly loses a
    button when a document is not shaped a certain way is one where the feature
    looks like it was imagined. Each reason is a sentence on the hover, which is
    where the way forward goes.

    The *third* verb, "Convert to tilemap", is not here: it is one of the two
    ways a document acquires its first tileset, and this whole pane only exists
    once it has one. It lives beside the other one -- the ``.tsx`` import -- in
    the bridge panel, which is drawn whether or not there is a tileset. See
    :func:`convert_row`.
    """
    doc = tab.doc
    width = widgets.grid_width(2)
    busy = tab.busy
    tilemap = doc.active_tilemap_uid() is not None
    if widgets.disabled_button(
        f"{icons.PLUS} New tilemap layer",
        not busy and bool(doc.tilesets),
        (width, 0),
        reason=(
            BUSY_WHY
            if busy
            else "This document has no tileset yet. Convert a layer, or import a .tsx."
        ),
        tooltip="A new layer whose cells name tiles instead of holding pixels.",
    ):
        inker_mode.new_tilemap_layer(ctx, tab, state.picked_tileset(doc))
    imgui.same_line()
    if widgets.disabled_button(
        "To a plain layer",
        not busy and tilemap,
        (width, 0),
        reason=BUSY_WHY if busy else "The active layer is not a tilemap layer.",
        tooltip=(
            "Keeps the picture exactly as it is and drops the cells. The "
            "tileset stays in the document."
        ),
    ):
        inker_mode.convert_to_raster(ctx, tab)


def can_convert(state: Any, tab: Any) -> bool:
    """Whether "Convert to tilemap" applies. The registry's predicate.

    Written here rather than in ``inker_ops`` because the question is about
    tiles and this module is where every other tile question is answered; the
    registry names it, which is the whole of the coupling.
    """
    if tab is None or tab.busy:
        return False
    return tab.doc.active_tilemap_uid() is None and bool(len(tab.doc.stack))


def convert_row(ctx: Any, state: Any, tab: Any) -> None:
    """The size popup "Convert to tilemap..." opens, registered by its window.

    The verb itself is a **menu row** now (Sprite -> Convert to tilemap...),
    because it is how a drawing gets its *first* tileset and the tile pane is
    not on screen until it has one -- so a button for it could only live
    somewhere that is always drawn. What has to stay a call from a pane is the
    popup: a popup belongs to the window that begins it, so whoever hosts the
    menu registers this too. ``inker_canvas.new_popup``'s arrangement, for
    ``inker_canvas.new_popup``'s reason.
    """
    _tile_size_popup(ctx, state, tab)


def _tile_size_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The tile size a conversion cuts at.

    A popup rather than two fields on the panel, for the reason the resize
    popup is one: it is asked once per conversion and never looked at again,
    and two permanent number boxes would be two controls that mean nothing for
    the rest of the session. The answer is remembered per tab in the preview
    dictionary, so a mistake is corrected rather than retyped.
    """
    if not imgui.begin_popup("inker-to-tilemap"):
        return
    widgets.popup_chrome(_imgui=imgui)
    key = f"inker_tile_size:{tab.uid}"
    tile_w, tile_h = ctx.state.preview.get(key) or (state.grid_size, state.grid_size)
    imgui.set_next_item_width(sp(90))
    changed_w, tile_w = controls.input_int("W", int(tile_w), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed_h, tile_h = controls.input_int("H", int(tile_h), 0)
    if changed_w or changed_h:
        ctx.state.preview[key] = (max(1, tile_w), max(1, tile_h))
    widgets.muted_wrapped(
        "Cells that hold the same picture share one tile, and an empty cell "
        "costs no tile at all."
    )
    imgui.dummy((0, sp(tokens.SP_1)))
    imgui.begin_disabled(tab.busy)
    if controls.button("Convert", (sp(180), 0)):
        inker_mode.convert_to_tilemap(ctx, tab, max(1, tile_w), max(1, tile_h))
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.end_popup()


def _chooser(ctx: Any, state: Any, tab: Any, uid: int) -> None:
    """Which tileset the picker shows.

    Out while a tilemap layer is active, and the reason is on the hover: the
    layer's own binding decides, because a local id only means anything next to
    the tileset it was picked from. One tileset in the document is the ordinary
    case and gets a line of text rather than a combo of one.
    """
    doc = tab.doc
    if len(doc.tilesets) < 2:
        widgets.muted(_summary(doc.tileset_slot(uid)))
        return
    bound = doc.active_tileset_uid()
    options = [(str(slot.uid), _summary(slot)) for slot in doc.tilesets]
    if bound is not None:
        imgui.begin_disabled()
    picked = widgets.labeled_combo(
        "Tileset",
        str(uid),
        options,
        help_text=(
            "A tile is a *local* id into one tileset, so the tileset is the "
            "layer's to choose while a tilemap layer is selected."
        ),
    )
    if bound is not None:
        imgui.end_disabled()
    elif picked != str(uid):
        state.tileset_uid = int(picked)


def _summary(slot: Any) -> str:
    tileset = slot.tileset
    return f"{tileset.name} - {tileset.tile_count} tiles at {tileset.tile_w}x{tileset.tile_h}"


def _picker(ctx: Any, state: Any, tab: Any, slot: Any) -> None:
    """The atlas, re-flowed into rows, with the picked tile outlined.

    Tile 0 is drawn like every other tile and *labelled*: it is the required
    blank, so it looks like an empty cell and would otherwise read as a gap in
    the atlas rather than as the tile eraser it is.

    **The UVs come from ``Tileset.uv``, which exists for exactly this.** They
    were sliced here as horizontal bands -- ``local / tile_count`` down a
    one-column strip -- which is right for a tileset this editor cut and wrong
    for every other shape the same type can hold: an imported grid ``.tsx`` is
    kept as it arrives (chunk 3.6), so it has columns, and it may have a margin
    and spacing too. Asking the class that owns the slicing is the same rule
    ``tile_pixels`` and the two renderers already follow.
    """
    tileset = slot.tileset
    texture = inker_textures.tileset_texture(ctx, tab, slot)
    zoom = max(1, int(sp(TILE_PX) // max(tileset.tile_w, 1)))
    cell_w = tileset.tile_w * zoom
    cell_h = tileset.tile_h * zoom
    spacing = imgui.get_style().item_spacing.x
    draw_list = imgui.get_window_draw_list()
    count = min(tileset.tile_count, MAX_DRAWN)

    for local in range(count):
        top_left = imgui.get_cursor_screen_pos()
        if texture is None:
            # No GL context: the headless smoke suite and every state-only
            # test. A placeholder keeps the geometry identical, so a screenshot
            # pass still exercises the layout around it.
            widgets.thumb_placeholder(cell_w, icons.GRID)
        else:
            u0, v0, u1, v1 = tileset.uv(local)
            imgui.image(
                widgets.texture_ref(texture), (cell_w, cell_h), (u0, v0), (u1, v1)
            )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Tile 0 -- the blank. Stamping it clears a cell."
                if local == 0
                else f"Tile {local}"
            )
            if imgui.is_mouse_clicked(0):
                state.tile_local = local
                state.tileset_uid = slot.uid
        if local == state.tile_local:
            draw_list.add_rect(
                (top_left.x, top_left.y),
                (top_left.x + cell_w, top_left.y + cell_h),
                imgui.get_color_u32(theme.rgba(theme.ACCENT)),
                0.0,
                sp(2),
            )
        if local != count - 1:
            widgets.same_line_or_wrap(cell_w + spacing)
    imgui.new_line()
    if tileset.tile_count > count:
        widgets.muted(f"showing the first {count} of {tileset.tile_count}")
    picked = int(state.tile_local)
    widgets.muted("Tile 0 (blank)" if picked == 0 else f"Tile {picked}")


def _flips(state: Any, tileset: Any) -> None:
    """The three flag bits, as three boxes.

    Named for the transform rather than for the bit -- Aseprite's X/Y/D, in the
    words this app already uses for a flip -- and folded into one gid by
    ``InkerState.tile_gid``, which is the only place the three are read
    together.

    D is disabled (and the held flag cleared) when the picked tileset's tiles
    are not square: a transpose swaps the footprint, and the document masks
    the flag off at every door for exactly that reason
    (``TileOps._strip_diagonal``) -- a live checkbox over a masked flag would
    just look broken.
    """
    square = int(tileset.tile_w) == int(tileset.tile_h)
    widgets.field_label("flip")
    changed, value = controls.checkbox("H##tileflip", state.tile_flip_h)
    if changed:
        state.tile_flip_h = value
    imgui.same_line()
    changed, value = controls.checkbox("V##tileflip", state.tile_flip_v)
    if changed:
        state.tile_flip_v = value
    imgui.same_line()
    if not square and state.tile_flip_d:
        state.tile_flip_d = False
    # Greyed with the sentence, not just greyed: the help marker below explains
    # the requirement, but a reader who has not opened it meets a dead tick box
    # with nothing on it to say why. ``checkbox`` has taken ``reason`` since it
    # was written.
    changed, value = controls.checkbox(
        "D##tileflip",
        state.tile_flip_d,
        enabled=square,
        reason="D is a transpose, which needs square tiles: it would not land "
        "on the grid otherwise.",
    )
    if changed:
        state.tile_flip_d = value
    widgets.help_marker(
        "How the tile is turned as it goes down. D is the diagonal -- a "
        "transpose -- and the three together give all eight orientations. The "
        "flags ride in the cell, so the tileset holds one copy of the tile. "
        "D needs square tiles: a transpose of a non-square tile would not "
        "fit its own cell."
    )


def _files(ctx: Any, state: Any, tab: Any, uid: int | None) -> None:
    """Out to a ``.tsx``, in from one, and across to Plotter.

    The export addresses **the picked tileset**, which is what this pane exists
    to know: it superseded a first cut that always reached index 0 because
    nothing yet had a selection to offer it.
    """
    doc = tab.doc
    width = widgets.grid_width(2)
    ready = not tab.busy
    why = BUSY_WHY
    if widgets.disabled_button(
        f"{icons.UPLOAD} Export tileset...",
        ready and uid is not None,
        (width, 0),
        reason=why if not ready else "This document has no tileset yet.",
    ):
        index = next(i for i, slot in enumerate(doc.tilesets) if slot.uid == uid)
        inker_mode.export_tileset(ctx, tab, index=index)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.FOLDER_OPEN} Import .tsx...", ready, (width, 0), reason=why
    ):
        inker_mode.import_tileset(ctx, tab)
    if widgets.disabled_button(
        f"{icons.GRID} Use in Plotter",
        ready and uid is not None,
        (-1, 0),
        reason=why if not ready else "This document has no tileset yet.",
        tooltip=(
            "Hands this tileset to the open map as it stands now. A snapshot, "
            "not a link: painting on it here afterwards leaves the map's copy "
            "alone."
        ),
    ):
        plotter_tilesets.use_inker_tileset(ctx, doc, uid)
