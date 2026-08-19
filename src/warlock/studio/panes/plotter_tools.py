"""Plotter's left-top pane: the tools and the map's own properties.

The tools are what a click on the canvas *means*; the tileset pane owns which
tile it means, and the layers pane owns which layer it lands on. One control,
one owner -- the rule the two generate panes already follow for ``platform``.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, plotter_mode, plotter_setup, plotter_state, widgets
from ..manual import render as manual_render
from ..plotter import project
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
        # A foreign Wang set contributes **one row per colour**, which is the
        # same thing a terrain row is from the user's side: a thing the Terrain
        # tool lays down. The two paint by different machinery -- the preset by
        # its positional collapse, a Wang colour by constraint matching -- and
        # the pane deliberately does not say which, because to the person
        # holding the tool there is no difference.
        for wangset in ref.tileset.wangsets:
            for colour_index, colour in enumerate(wangset.colours):
                out.append((index, -1 - colour_index, _wang_swatch(wangset, colour)))
    return out


def _wang_swatch(wangset: Any, colour: Any) -> Any:
    """A Wang colour dressed as a terrain spec, for the row that draws it.

    The outline is derived rather than carried, exactly as it is for a terrain
    read out of a ``.tsx``: the format does not store one and the swatch is the
    only thing that reads it.
    """
    from ..tilegrid.tileset import TerrainSpec

    fill = _hex_rgba(colour.colour)
    return TerrainSpec(
        name=colour.name or wangset.name,
        fill=fill,
        outline=(*(part * 3 // 5 for part in fill[:3]), fill[3]),
    )


def _hex_rgba(text: str) -> tuple[int, int, int, int]:
    """``#rrggbb`` or ``#aarrggbb`` as four channels; opaque white on nonsense.

    Tolerant rather than refusing, because this is a *swatch*: a colour nobody
    can parse is a row drawn in the wrong colour, and refusing the map over it
    would be the half-read trade run backwards.
    """
    raw = str(text or "").lstrip("#")
    try:
        if len(raw) == 8:
            a, r, g, b = (int(raw[i : i + 2], 16) for i in (0, 2, 4, 6))
            return (r, g, b, a)
        if len(raw) == 6:
            r, g, b = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
            return (r, g, b, 255)
    except ValueError:
        pass
    return (255, 255, 255, 255)


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
        # The old copy sent people to a generator that was deleted on
        # 2026-08-18, through a section key nothing has registered since -- a
        # route that was live and did nothing.
        widgets.muted_wrapped(
            "This map has no terrain sets. Add a Tiled .tsx that carries Wang "
            "sets, or a sheet holding all 47 blob cases, and Plotter will offer "
            "to turn it into one."
        )
        if controls.button("Add a tileset...", (-1, 0)):
            plotter_mode.ask_add_tileset(ctx)
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
    if state.tool == "stamp":
        _, state.random_mode = widgets.toggle("Random mode", state.random_mode)
        if state.random_mode:
            widgets.muted_wrapped("Each cell chooses from the non-empty tiles in the stamp.")
        imgui.dummy((0, 6))
    if state.tool == "erase":
        widgets.muted_wrapped("Drag to erase; Shift-drag clears a rectangle.")
        imgui.dummy((0, 6))
    # Headed, where they used to be three loose toggles under whatever the tool
    # in hand had drawn. They are about what the *canvas* shows rather than what
    # a click does, and with sections drawn as blocks the difference stopped
    # being invisible and became wrong: with Terrain in hand, "Grid",
    # "Show objects" and "Minimap" sat inside the Terrain block, which said they
    # were terrain settings.
    widgets.section("View")
    _, state.grid = widgets.toggle("Grid (Ctrl+G)", state.grid)
    _, state.show_objects = widgets.toggle("Show objects", state.show_objects)
    _, state.minimap = widgets.toggle("Minimap", state.minimap)
    _, state.highlight = widgets.toggle(
        "Highlight current layer", state.highlight
    )

    imgui.dummy((0, 6))
    widgets.section("Map")
    widgets.muted(f"{doc.width} x {doc.height} tiles, {doc.tile_w} x {doc.tile_h} px")
    widgets.muted(f"{doc.pixel_width} x {doc.pixel_height} px overall")
    widgets.muted(f"{doc.projection} projection")

    if tab.busy:
        widgets.muted("Saving...")
        return

    if widgets.header("Map metadata", default_open=False, persist_key="plotter/map-meta"):
        class_name = widgets.input_text(
            "##map-class", doc.class_name, max_length=64, hint="map class"
        )
        if class_name != doc.class_name:
            doc.set_map_settings(class_name=class_name)
        background = widgets.input_text(
            "##map-background",
            doc.backgroundcolor or "",
            max_length=9,
            hint="#RRGGBB or #AARRGGBB",
        )
        if background != (doc.backgroundcolor or ""):
            try:
                doc.set_map_settings(backgroundcolor=background or None)
            except ValueError:
                # Said in place rather than toasted. A text field is re-read on
                # every frame, and "#4a7" on the way to "#4a7c3f" is refused on
                # every one of them -- so the toast fired sixty times a second
                # for as long as the caret sat in the middle of a colour, which
                # is the exact failure ``ground_watch``'s docstring names. A
                # half-typed value is not an error, it is a value in progress:
                # the document keeps what it had until the text is a colour.
                widgets.muted("Not a colour yet - #RRGGBB or #AARRGGBB.")
        order = widgets.labeled_combo(
            "Render order",
            doc.renderorder,
            [(value, value) for value in project.RENDER_ORDERS],
        )
        if order != doc.renderorder:
            doc.set_map_settings(renderorder=order)
        changed, origin = controls.input_float2(
            "Parallax origin", [float(value) for value in doc.parallax_origin]
        )
        if changed:
            doc.set_map_settings(parallax_origin=(float(origin[0]), float(origin[1])))
        if doc.projection == project.OBLIQUE:
            changed, skew = controls.input_float2(
                "Skew", [float(doc.skew_x), float(doc.skew_y)]
            )
            if changed:
                doc.set_map_settings(skew_x=int(skew[0]), skew_y=int(skew[1]))

    imgui.dummy((0, 6))
    if widgets.header("Properties", default_open=False, persist_key="plotter/map-props"):
        # The map's own custom properties. They survive a Tiled round trip and
        # always have; this is the first way to set one without a text editor.
        plotter_layers.property_editor(
            ctx,
            f"plotter_map_prop:{tab.uid}",
            doc.properties,
            doc.set_map_properties,
            object_options=plotter_layers.object_options(doc),
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

    imgui.dummy((0, 8))
    _offset_form(ctx, tab)
    imgui.dummy((0, 8))
    # Tiled's Map -> Autocrop. Delegated to ``resize`` verbatim inside the
    # document, so objects travel by the rule that already exists.
    if controls.button("Autocrop to content", (-1, 0)):
        if tab.doc.autocrop():
            ctx.state.preview.pop(key, None)
            tab.view.fitted = False
        else:
            ctx.toast("There is nothing painted to crop to.", "warn")

    imgui.dummy((0, 10))
    _tile_size_form(ctx, tab)


#: Whether an offset moves every tile layer or only the active one, and how the
#: vacated cells are filled. Data rather than four booleans in the pane, for
#: ``TOOLS``' reason: the labels and the values cannot drift apart.
OFFSET_SCOPES = (("map", "Whole map"), ("layer", "This layer"))


def _offset_form(ctx: Any, tab: Any) -> None:
    """Tiled's Map -> Offset Map, inside the section that already owns geometry.

    Its own row rather than its own section: a user reaching for "move
    everything two cells left" is reaching for the same thing they reach for
    when they resize, and a second collapsible header for two sliders is a
    header nobody opens.
    """

    key = f"plotter_offset:{tab.uid}"
    form = ctx.state.preview.get(key)
    if form is None:
        form = {"dx": 0, "dy": 0, "wrap": True, "scope": "map"}
        ctx.state.preview[key] = form

    widgets.muted("Offset")
    _, form["dx"] = widgets.labeled_slider_int("Move X", int(form["dx"]), -64, 64)
    _, form["dy"] = widgets.labeled_slider_int("Move Y", int(form["dy"]), -64, 64)
    form["scope"] = widgets.labeled_combo(
        "Scope", str(form["scope"]), list(OFFSET_SCOPES)
    )
    _changed, form["wrap"] = controls.checkbox(
        "Wrap around the edges",
        bool(form["wrap"]),
        tooltip=(
            "Wrapping is an exact permutation -- nothing is lost, so offsetting "
            "back puts everything where it was. Without it the vacated cells "
            "are cleared."
        ),
    )
    if controls.button("Offset##apply", (-1, 0)):
        moved = tab.doc.offset(
            int(form["dx"]),
            int(form["dy"]),
            wrap=bool(form["wrap"]),
            scope=str(form["scope"]),
        )
        if not moved:
            ctx.toast("That offset moves nothing.", "warn")


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
