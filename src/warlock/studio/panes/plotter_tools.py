"""Plotter's left-top pane: the tools and the map's own properties.

The tools are what a click on the canvas *means*; the tileset pane owns which
tile it means, and the layers pane owns which layer it lands on. One control,
one owner -- the rule the two generate panes already follow for ``platform``.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, plotter_mode, plotter_setup, plotter_state, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import plotter_layers

# The tool letters, drawn on the buttons. From ``plotter_state.TOOLS`` rather
# than restated, so a tool added there cannot get a button with no key or a key
# with no button.
_ICONS = {
    "stamp": icons.BRUSH,
    "erase": icons.ERASER,
    "fill": icons.PAINT_BUCKET,
    "terrain": icons.WAND,
    "shape": icons.SQUARE,
    "select": icons.SQUARE_DASHED,
    "pick": icons.PIPETTE,
    "object": icons.FLAG,
}


def _tool_grid(state: Any) -> None:
    from imgui_bundle import imgui

    # Width from the style rather than a literal gap: ``theme.apply`` sets
    # item_spacing through ``sp()``, so a grid that subtracted a hard-coded 8
    # was right at UI scale 1.0 and short by five pixels per gap at 1.5 --
    # which is what dropped the raster editor's fifth toolbox column.
    width = widgets.grid_width(3)
    for index, (key, label, letter) in enumerate(plotter_state.TOOLS):
        if index % 3:
            imgui.same_line()
        active = state.tool == key
        if active:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        if controls.button(f"{_ICONS.get(key, icons.SQUARE)}##tool-{key}", (width, 0)):
            state.tool = key
        if active:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{label} ({letter})")


#: What the Shape tool can fill, in the order the buttons sit. A tuple rather
#: than two literals so the pane and any future keyboard route read one list.
SHAPES = (("rect", "Rectangle", icons.SQUARE), ("ellipse", "Ellipse", icons.CIRCLE))


def _shape_picker(state: Any) -> None:
    """Rectangle or ellipse -- Tiled's Shape Fill, as one tool with a mode.

    Drawn only while Shape is the tool in hand, the way the terrain swatches
    are: a control for a tool you are not holding is a control that has to
    explain itself.
    """
    from imgui_bundle import imgui

    width = widgets.grid_width(len(SHAPES))
    for index, (key, label, glyph) in enumerate(SHAPES):
        if index:
            imgui.same_line()
        active = state.shape_mode == key
        if active:
            imgui.push_style_color(
                imgui.Col_.button.value, imgui.get_style().color_(imgui.Col_.button_active.value)
            )
        if controls.button(f"{glyph}##shape-{key}", (width, 0)):
            state.shape_mode = key
        if active:
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip(label)


def terrains_of(doc: Any) -> list[tuple[int, int, Any]]:
    """``(tileset index, terrain index, spec)`` for every terrain on the map.

    Derived at draw time rather than cached, for the reason the terrain of a
    *cell* is derived: a cached list would be one more thing that can disagree
    with the tilesets, and the list is at most a few dozen entries long.
    """
    out = []
    for index, ref in enumerate(doc.tilesets):
        for rank, spec in enumerate(ref.tileset.terrains):
            out.append((index, rank, spec))
    return out


def _terrain_picker(ctx: Any, state: Any, tab: Any) -> None:
    """Which terrain the Terrain tool lays down.

    Here rather than in the tileset pane because this pane owns what a click
    *means* and the other owns which tile it means -- and a terrain is not a
    tile: it is a role that thirty-two cells of the atlas share, chosen by the
    neighbours rather than by the user.
    """
    from imgui_bundle import imgui

    entries = terrains_of(tab.doc)
    widgets.section("Terrain")
    if not entries:
        widgets.muted_wrapped(
            "This map has no terrain sets. Generate one under Tilesets, or add a "
            "tileset that carries one."
        )
        if controls.button("Open the generator", (-1, 0)):
            widgets.request_open("plotter/generate")
        return
    if state.terrain is None:
        state.terrain = (entries[0][0], entries[0][1])
    width = widgets.grid_width(2)
    for slot, (tileset_index, rank, spec) in enumerate(entries):
        if slot % 2:
            imgui.same_line()
        chosen = state.terrain == (tileset_index, rank)
        colour = tuple(part / 255.0 for part in spec.fill)
        imgui.push_style_color(imgui.Col_.button.value, imgui.get_color_u32(colour))
        imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.get_color_u32(colour))
        # The swatch *is* the terrain, so the selected one is marked by the text
        # colour rather than by a second colour fighting the fill.
        imgui.push_style_color(
            imgui.Col_.text.value,
            imgui.get_color_u32((1.0, 1.0, 1.0, 1.0) if chosen else (0.0, 0.0, 0.0, 0.55)),
        )
        if controls.button(f"{spec.name}##terrain-{tileset_index}-{rank}", (width, 0)):
            state.terrain = (tileset_index, rank)
        imgui.pop_style_color(3)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"{spec.name} - in {tab.doc.tilesets[tileset_index].tileset.name}"
            )
    widgets.muted_wrapped("Painting fits the eight cells around what you touch.")


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Tools")
    manual_render.help_button(ctx, "plotter-tools")
    _tool_grid(state)
    imgui.dummy((0, 6))

    if tab is None:
        widgets.muted("Open or start a map to draw on.")
        return

    doc = tab.doc
    if state.tool == "terrain":
        _terrain_picker(ctx, state, tab)
        imgui.dummy((0, 6))
    if state.tool == "shape":
        _shape_picker(state)
        imgui.dummy((0, 6))
    _, state.grid = widgets.toggle("Grid (Ctrl+G)", state.grid)
    _, state.show_objects = widgets.toggle("Show objects", state.show_objects)
    _, state.minimap = widgets.toggle("Minimap", state.minimap)

    imgui.dummy((0, 6))
    widgets.section("Map")
    widgets.muted(f"{doc.width} x {doc.height} tiles, {doc.tile_w} x {doc.tile_h} px")
    widgets.muted(f"{doc.pixel_width} x {doc.pixel_height} px overall")
    widgets.muted(f"{doc.projection} projection")

    if tab.busy:
        widgets.muted("Saving...")
        return

    imgui.dummy((0, 6))
    if widgets.header("Properties", default_open=False, persist_key="plotter/map-props"):
        # The map's own custom properties. They survive a Tiled round trip and
        # always have; this is the first way to set one without a text editor.
        plotter_layers.property_editor(
            ctx,
            f"plotter_map_prop:{tab.uid}",
            doc.properties,
            doc.set_map_properties,
        )

    imgui.dummy((0, 6))
    if widgets.header("Resize", default_open=False, persist_key="plotter/resize"):
        _resize_form(ctx, tab)


def _resize_form(ctx: Any, tab: Any) -> None:
    """Grow or crop the grid, anchoring the old content by an offset.

    Cached under ``state.preview`` per tab, so typing a width does not fight
    with the document and switching tabs does not carry a half-typed number
    onto a different map.
    """
    from imgui_bundle import imgui

    key = f"plotter_resize:{tab.uid}"
    form = ctx.state.preview.get(key)
    if form is None or form.get("for") != (tab.doc.width, tab.doc.height):
        form = {
            "for": (tab.doc.width, tab.doc.height),
            "w": tab.doc.width,
            "h": tab.doc.height,
            "dx": 0,
            "dy": 0,
        }
        ctx.state.preview[key] = form

    _, form["w"] = widgets.labeled_slider_int("Width", int(form["w"]), 1, 512)
    _, form["h"] = widgets.labeled_slider_int("Height", int(form["h"]), 1, 512)
    _, form["dx"] = widgets.labeled_slider_int("Offset X", int(form["dx"]), -64, 64)
    _, form["dy"] = widgets.labeled_slider_int("Offset Y", int(form["dy"]), -64, 64)
    imgui.dummy((0, 4))
    # ``##apply`` is load-bearing. The section around this is ``header("Resize")``
    # and imgui hashes an item's id from its label, so a button reading "Resize"
    # inside it claimed the *header's* id -- and imgui routes activation by id,
    # so the button never acted and the map could not be resized at all. The
    # suffix is invisible to the reader and makes the id its own.
    if widgets.primary_button("Resize##apply", (-1, 0)):
        try:
            tab.doc.resize(
                int(form["w"]), int(form["h"]),
                offset_x=int(form["dx"]), offset_y=int(form["dy"]),
            )
        except ValueError as exc:
            # Framed rather than forwarded: the engine's sentence says what was
            # wrong with the number and nothing about what was being attempted.
            ctx.toast(f"The map was not resized: {exc}.", "error")
            return
        ctx.state.preview.pop(key, None)
        tab.view.fitted = False

    imgui.dummy((0, 10))
    _tile_size_form(ctx, tab)


def _tile_size_form(ctx: Any, tab: Any) -> None:
    """How big a cell is, on a map that already exists.

    Beside the grid resize because they are the two halves of one question --
    how many cells, and how big is a cell -- and neither is a projection: the
    lattice is what cannot move once tiles are on the map, and a cell's *size*
    can, because a gid names the same tile whatever size the cell under it is.
    """
    from imgui_bundle import imgui

    key = f"plotter_tile_px:{tab.uid}"
    stamp = (tab.doc.tile_w, tab.doc.tile_h)
    form = ctx.state.preview.get(key)
    if form is None or form.get("for") != stamp:
        form = {"for": stamp, "w": tab.doc.tile_w, "h": tab.doc.tile_h}
        ctx.state.preview[key] = form

    widgets.field_label("Tile size, in pixels")
    imgui.set_next_item_width(sp(80))
    _, form["w"] = controls.input_int("W##tile-px-w", int(form["w"]), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(80))
    _, form["h"] = controls.input_int("H##tile-px-h", int(form["h"]), 0)
    width = max(1, min(int(form["w"] or 1), plotter_setup.MAX_TILE_PX))
    height = max(1, min(int(form["h"] or 1), plotter_setup.MAX_TILE_PX))
    form["w"], form["h"] = width, height

    widgets.muted_wrapped(
        "Cells are redrawn at the new size and every object scales with them. "
        "Tilesets already on the map keep the slicing they arrived with; only "
        "an image added *after* this is sliced at the new size."
    )
    if widgets.disabled_button(
        "Apply tile size", (width, height) != stamp, (-1, 0)
    ):
        try:
            tab.doc.set_tile_size(width, height)
        except ValueError as exc:
            # Framed rather than forwarded, ``_resize_form``'s rule.
            ctx.toast(f"The tile size was not changed: {exc}.", "error")
            return
        ctx.state.preview.pop(key, None)
        tab.view.fitted = False
