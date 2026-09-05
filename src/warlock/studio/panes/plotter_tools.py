"""Plotter's toolbar: the row across the top of the map, and the two dialogs.

Tiled puts the tools in a strip above the canvas and nothing else there. This
pane used to be a 300 px sidebar holding four different subjects -- the tool
grid, the tool's own options, the view aids and a read-only recital of the
map's size -- stacked down the left of the window with the map beside them.
Three of those four are settings you change *between* clicks on the canvas, and
a sidebar puts them the width of the window away from the click.

So the pane became the bar, and the layout it left behind is Tiled's:
Properties on the left, Layers over Tilesets on the right, this strip across
the top of the centre column. What was measured against the old arrangement
holds here too -- see :mod:`~warlock.studio.toolbar` for why a row degrades
rather than clips, and ``inker_context`` for the same move made a week earlier
against Aseprite's context bar.

**The pill is not part of the collapsing row.** Which tool is in your hand is
the one thing on this strip that is never optional, so it is drawn first, as
glyphs, and it never folds into an overflow menu. Everything after it -- the
brush transforms, Random, the tool's own field, the view aids and the snap
choice -- goes through :func:`toolbar.toolbar`, which gives up labels before it
gives up controls.

**Two surfaces read one table.** ``VIEW_TOGGLES`` is the grid, the rulers, the
objects, the minimap and the layer highlight; :func:`view_rows` draws them, and
both the ``View`` popover here and the ``View`` menu in ``plotter_menu`` call
it. They used to be five loose ``widgets.toggle`` calls in this pane and one
menu row in another, which is how "Highlight current layer" came to be filed
under *Layer* while the other four sat under a heading called *View*.

Everything below :data:`SETTINGS_POPUP` is unchanged: the two dialogs the Map
menu opens are hosted by whichever window drew the menu, and that is still the
canvas.
"""

from __future__ import annotations

from typing import Any

from .. import (
    controls,
    dialogs,
    icons,
    plotter_mode,
    plotter_setup,
    plotter_state,
    tokens,
    toolbar,
    widgets,
)
from ..manual import render as manual_render
from ..plotter import project
from ..tokens import sp
from . import plotter_layers

#: This bar's imgui id, and the prefix every one of its controls is keyed from.
BAR = "plotter-context"

#: The tool pill's id. Its segments are ``##plotter-tool/<key>``, which is what
#: the smoke suite presses; the old grid keyed them ``##tool-<key>``.
TOOL_PILL = "plotter-tool"

# One glyph per tool, and **every** tool in **both** palettes. Wand had no entry
# once, fell through to the ``icons.SQUARE`` default and drew the same picture
# as Shape two buttons away -- while ``icons.WAND`` was on Terrain, which is the
# one tool in the grid the word does not describe. A missing entry here is a
# *wrong* picture, not a blank one.
#
# The object palette had no entries at all until the bar landed, because the old
# grid drew it at 300 px wide where a wrong glyph beside a full label was
# survivable. On a 28 px pill the glyph is the only thing on screen, so all
# eight are named. ``icons.EGG`` for the capsule is the closest rounded oblong
# the vendored subset carries; ``icons.py`` forbids guessing a codepoint, and a
# stadium is not in lucide-static 0.525.0 under any name.
#
# ``test_every_plotter_tool_has_its_own_icon`` pins completeness against both
# palettes and distinctness *within* each -- across the two it cannot hold,
# because six letters already mean two things and ``object`` is in both lists.
_ICONS = {
    # The tile palette.
    "stamp": icons.BRUSH,
    "erase": icons.ERASER,
    "fill": icons.PAINT_BUCKET,
    "terrain": icons.BLEND,
    "shape": icons.SQUARE,
    "select": icons.SQUARE_DASHED,
    "wand": icons.WAND,
    "pick": icons.PIPETTE,
    "object": icons.FLAG,
    # The object palette. ``object`` above is its pointer and is shared.
    "object_rect": icons.RECTANGLE,
    "object_point": icons.CROSSHAIR,
    "object_ellipse": icons.CIRCLE,
    "object_polygon": icons.PENTAGON,
    "object_polyline": icons.SPLINE,
    "object_tile": icons.IMAGE,
    "object_text": icons.TYPE,
    "object_capsule": icons.EGG,
}

#: What the Shape tool can fill, in the order the buttons sit. A tuple rather
#: than two literals so the pane and any future keyboard route read one list.
SHAPES = (("rect", "Rectangle", icons.SQUARE), ("ellipse", "Ellipse", icons.CIRCLE))

#: Tiled's brush transforms as bar buttons: ``key``, label, glyph, the
#: ``plotter_mode`` transform name, and whether it is the reversed one.
#:
#: **Two of the four are words rather than glyphs**, and deliberately:
#: ``icons.py`` is a transcription of lucide-static 0.525.0's codepoint
#: assignments and its docstring forbids guessing one. The vendored subset
#: carries ``flip-horizontal-2`` and ``rotate-cw`` and neither of their mirrors,
#: so "Flip V" and "Rotate back" are the honest rendering. ``toolbar._measure``
#: hands an item with no glyph the same width at both tiers, which is what makes
#: "this group drops to icons" mean "this group gets no narrower" for those two
#: rather than "those two vanish".
BRUSH_TRANSFORMS: tuple[tuple[str, str, str, str, bool, str], ...] = (
    (
        "flip_h",
        "Flip H",
        icons.FLIP_HORIZONTAL,
        "x",
        False,
        "Mirror the brush left to right (X)",
    ),
    ("flip_v", "Flip V", "", "y", False, "Mirror the brush top to bottom (Y)"),
    (
        "rotate_cw",
        "Rotate",
        icons.ROTATE_CW,
        "z",
        False,
        "Turn the brush a quarter clockwise (Z)",
    ),
    (
        "rotate_ccw",
        "Rotate back",
        "",
        "z",
        True,
        "Turn the brush a quarter anticlockwise (Shift+Z)",
    ),
)

#: Said once, because four buttons and a test all want the same sentence.
NO_BRUSH = "Pick a tile in the tileset first -- there is no brush to transform."

RANDOM_TIP = "Each cell chooses from the non-empty tiles in the stamp."

#: The canvas aids, as one table. **Two surfaces read it** -- the ``View``
#: popover on this bar and the ``View`` menu in ``plotter_menu`` -- which is the
#: whole reason it exists: the five were four ``widgets.toggle`` calls in this
#: pane plus one menu row filed under *Layer*, and nothing made them one set.
#:
#: The attribute name on ``PlotterState`` is the key, so a toggle added there
#: and listed here needs no third place to be taught about it.
VIEW_TOGGLES: tuple[tuple[str, str, str], ...] = (
    ("grid", "Grid", "Ctrl+G"),
    ("rulers", "Rulers", "Ctrl+R"),
    ("show_objects", "Show objects", ""),
    ("minimap", "Minimap", ""),
    ("highlight", "Highlight current layer", "H"),
)

VIEW_POPUP = "plotter-view"


def view_rows(ctx: Any, state: Any) -> None:
    """:data:`VIEW_TOGGLES` as checked menu rows. Drawn in both View surfaces.

    Menu rows rather than ``widgets.toggle`` because the menu is the surface
    that cannot use a switch, and one drawing that works in both beats two that
    have to be kept saying the same thing. ``controls.menu_item`` renders
    identically inside :func:`controls.menu_popup`, which is what the popover
    on this bar is.
    """

    for key, label, chord in VIEW_TOGGLES:
        hit = controls.menu_item(
            f"{label}##plotter-view/{key}", chord, bool(getattr(state, key, False))
        )
        if bool(hit[0] if isinstance(hit, tuple) else hit):
            setattr(state, key, not bool(getattr(state, key, False)))


def _view_popup(ctx: Any, state: Any) -> None:
    with controls.menu_popup(VIEW_POPUP) as opened:
        if opened:
            view_rows(ctx, state)


def bar_items(state: Any, layer: Any) -> tuple[list[Any], list[str]]:
    """What the row holds for this layer: its buttons, and which fields it wants.

    **Pure** -- no imgui, no ctx, no document beyond the layer handed in -- so
    every combination the bar can be in is a plain assertion rather than a
    rendered frame. The fields come back as *names* rather than as
    :class:`toolbar.Field`s for the same reason: a Field carries a draw
    closure, and a closure is the one part of this that cannot be asserted
    about.

    An object layer gets neither the brush transforms nor Random: there is no
    brush on an object layer, and a row of five buttons that are permanently
    greyed is worse than a row that does not claim the width.
    """

    items: list[Any] = []
    fields: list[str] = []
    if plotter_state.layer_kind(layer) != "tile":
        return items, fields
    armed = getattr(state, "brush", None) is not None
    for key, label, glyph, _name, _back, tip in BRUSH_TRANSFORMS:
        items.append(
            toolbar.Item(
                key,
                label,
                glyph,
                tooltip=tip,
                enabled=armed,
                reason=NO_BRUSH,
                priority=1,
            )
        )
    items.append(
        toolbar.Item(
            "random",
            "Random",
            icons.SHUFFLE,
            tooltip=RANDOM_TIP,
            selected=bool(getattr(state, "random_mode", False)),
            priority=2,
        )
    )
    # The tool's own setting, at priority 0: it is what the row is *for* while
    # that tool is in hand, so it outlives the transforms in a narrow window.
    if state.tool == "shape":
        fields.append("shape")
    if state.tool == "terrain":
        fields.append("terrain")
    return items, fields


def _tool_pill(state: Any, layer: Any) -> None:
    """The palette **the layer in hand hosts**, as one pill group of glyphs.

    Tiled's split, and the reason for it: half the toolbox does nothing on the
    layer you are standing on, and a toolbox that offers a Fill on an object
    layer is one that has to refuse the click afterwards.

    A group or an image layer hosts no gesture at all, and the pill draws
    **disabled with a reason** rather than empty -- the house pattern. An empty
    toolbox reads as a broken pane; a greyed one with a sentence on the hover
    says what to select instead.

    Through ``controls.segmented_choice`` rather than ``widgets.icon_button``,
    which is load-bearing rather than tidy: ``icon_button`` draws through imgui
    directly and is invisible to ``probe``, so a tool button that stopped
    working could not be caught by a test that presses it. The old grid used
    ``controls.button`` for the same reason and this keeps it.
    """

    palette = plotter_state.tools_for(layer)
    reason = ""
    if not palette:
        palette = plotter_state.TILE_TOOLS
        reason = (
            "Nothing is drawn on this layer directly. Select a tile layer or "
            "an object layer -- a group holds layers and an image layer holds "
            "a picture."
            if layer is not None
            else "Nothing is open."
        )
    options = [(key, _ICONS.get(key, icons.SQUARE)) for key, _label, _letter in palette]
    tips = {
        key: (f"{label} ({letter})\n{reason}" if reason else f"{label} ({letter})")
        for key, label, letter in palette
    }
    changed, picked = controls.segmented_choice(
        TOOL_PILL,
        options,
        "" if reason else state.tool,
        enabled=not reason,
        reason=reason,
        tooltips=tips,
        compact=True,
    )
    if changed and not reason:
        state.tool = picked
        shape = plotter_state.OBJECT_SHAPES.get(picked)
        if shape:
            state.object_shape = shape


def _shape_field(state: Any) -> Any:
    """Rectangle or ellipse -- Tiled's Shape Fill, as one tool with a mode."""

    def draw_it(_compact: bool) -> None:
        from imgui_bundle import imgui

        for index, (key, label, glyph) in enumerate(SHAPES):
            if index:
                imgui.same_line(0.0, 0.0)
            if controls.button(
                f"{glyph}##{BAR}/shape/{key}",
                role=controls.ButtonRole.GHOST,
                control_size=controls.ControlSize.COMPACT,
                selected=state.shape_mode == key,
                tooltip=label,
            ):
                state.shape_mode = key

    # Glyphs at both tiers: two pills are already the smallest this can be, and
    # a field whose compact size equals its full one is what ``Field`` means by
    # a control that gets no narrower.
    return toolbar.Field("shape", "Shape", draw_it, width=64.0, compact=64.0)


NO_TERRAIN = (
    "This map has no terrain sets. Add a Tiled .tsx that carries Wang sets, or "
    "a sheet holding all 47 blob cases, and Plotter will offer to turn it into "
    "one."
)


def _terrain_field(state: Any, tab: Any) -> Any:
    """Which terrain the Terrain tool lays down.

    A combo rather than the grid of coloured swatches this was in the sidebar,
    and that is the one thing the move to a bar genuinely costs: a 38 px row has
    no space for a colour per row. What it buys is that the choice sits beside
    the tool it belongs to instead of a window's width away, and the *name* is
    how a terrain is chosen -- the fill colour was only ever how it was
    recognised once chosen, and the swatch is still drawn on the tileset
    editor's Terrain tab off the same spec.

    The empty branch is a *disabled* combo carrying the sentence rather than a
    button that opens the importer, because a field is not a place to put a
    verb: Tileset > Import a tileset is where a tileset is added, and the
    reason says so.
    """

    def draw_it(_compact: bool) -> None:
        if tab is None:
            controls.combo(
                f"##{BAR}/terrain", "", [], enabled=False, reason="Nothing is open."
            )
            return
        entries = terrains_of(tab.doc)
        if not entries:
            controls.combo(f"##{BAR}/terrain", "", [], enabled=False, reason=NO_TERRAIN)
            return
        if state.terrain is None:
            state.terrain = first_terrain(entries)
        options = [(f"{index}:{rank}", spec.name) for index, rank, spec in entries]
        keys = {key for key, _label in options}
        current = (
            "" if state.terrain is None else f"{state.terrain[0]}:{state.terrain[1]}"
        )
        if current not in keys:
            current = options[0][0]
        changed, picked = controls.combo(
            f"##{BAR}/terrain",
            current,
            options,
            tooltip="Which terrain the Terrain tool lays down. Painting fits "
            "the eight cells around what you touch.",
        )
        if changed:
            index, rank = picked.split(":")
            state.terrain = (int(index), int(rank))

    return toolbar.Field("terrain", "Terrain", draw_it, width=140.0, compact=96.0)


def _field(state: Any, tab: Any, key: str) -> Any:
    return _shape_field(state) if key == "shape" else _terrain_field(state, tab)


def _trailing(ctx: Any, state: Any) -> Any:
    """``View``, the snap choice and the (?), at the **end** of the row.

    A trailing block rather than three more items, for ``inker_context``'s
    reason: ``toolbar`` draws its items before its fields, so as items these
    would land between Random and the tool's own setting -- in the middle of the
    tool's settings, which is the one place a session-wide setting must not be.
    Trailing puts them past the fields, at the right-hand end.

    The block collapses to ``View`` plus the (?), and the snap pills are what it
    gives up. That is the ordering ``toolbar.Trailing`` states -- a label is
    cheaper to lose than a control -- with one addition this bar can make and
    Inker's could not: a control that exists in *two* places is cheaper still,
    and every snap mode is also a row in the Map menu's Snap group.
    """

    from imgui_bundle import imgui

    style = imgui.get_style()
    gap = style.item_spacing.x
    pad = style.frame_padding.x * 2.0
    square = imgui.get_frame_height()
    view_w = imgui.calc_text_size("View").x + pad
    # ``ControlSize.COMPACT`` changes only the vertical padding, so a pill is
    # exactly its text plus the frame padding and this measurement is not an
    # estimate.
    pills = sum(imgui.calc_text_size(label).x + pad for _key, label in SNAP_LABELS)
    full = view_w + gap + pills + gap + square
    compact = view_w + gap + square

    def draw_it(tight: bool) -> None:
        if controls.button(
            f"View##{BAR}/view",
            role=controls.ButtonRole.GHOST,
            control_size=controls.ControlSize.COMPACT,
            tooltip="What the canvas shows: the grid, the rulers, the objects, "
            "the minimap and the layer highlight",
        ):
            imgui.open_popup(VIEW_POPUP)
        _view_popup(ctx, state)
        if not tight:
            imgui.same_line()
            _snap_choice(state)
        imgui.same_line()
        manual_render.help_button_inline(ctx, "plotter-tools")

    return toolbar.Trailing(full, compact, draw_it)


def _hit(ctx: Any, state: Any, key: str) -> None:
    """One click on the row. The transforms go through ``plotter_mode`` so the
    button and the keystroke are one line rather than two that agree."""

    if key == "random":
        state.random_mode = not bool(getattr(state, "random_mode", False))
        return
    for item_key, _label, _glyph, name, back, _tip in BRUSH_TRANSFORMS:
        if item_key == key:
            plotter_mode.transform_brush(state, name, back=back)
            return


def draw(ctx: Any) -> None:
    """The bar. Called by the canvas, between the tab bar and the map.

    No ``section_blocks`` and no ``widgets.section``: this is not one of the
    four narrow sidebars the tinted grouping was written for, it is a strip over
    the canvas, and a heading on a toolbar is a heading on a toolbar.
    ``tests/test_section_blocks.py`` lists the panes that must ask; this one
    left that list when it left the sidebar.
    """

    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    layer = None if tab is None else tab.doc.active()
    # Before the pill draws, so the frame that follows a layer switch already
    # has a legal tool in hand: ``sync_tool`` is idempotent and is called from
    # here, from the canvas and from the key handler, which are the places a
    # gesture can start.
    plotter_state.sync_tool(state, layer)
    _tool_pill(state, layer)
    items, fields = bar_items(state, layer)
    imgui.same_line()
    hit = toolbar.toolbar(
        BAR,
        items,
        fields=[_field(state, tab, key) for key in fields],
        trailing=_trailing(ctx, state),
    )
    if hit:
        _hit(ctx, state, hit)
    widgets.divider()


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
        # the bar deliberately does not say which, because to the person holding
        # the tool there is no difference.
        for wangset in ref.tileset.wangsets:
            for colour_index, colour in enumerate(wangset.colours):
                out.append((index, -1 - colour_index, _wang_swatch(wangset, colour)))
    return out


def first_terrain(entries: list[tuple[int, int, Any]]) -> tuple[int, int] | None:
    """What the Terrain field arms when nothing is chosen yet.

    A function rather than two lines inside the field because **no click is
    needed to reach a terrain**: merely picking the tool arms one, so what it
    arms is a rule the canvas is tested against rather than a detail of a draw
    call nothing headless can run.
    """
    if not entries:
        return None
    index, rank, _spec = entries[0]
    return (index, rank)


def _wang_swatch(wangset: Any, colour: Any) -> Any:
    """A Wang colour dressed as a terrain spec, for the row that names it.

    The outline is derived rather than carried, exactly as it is for a terrain
    read out of a ``.tsx``: the format does not store one and the swatch is the
    only thing that reads it.
    """
    from ..tilegrid.tileset import TerrainSpec

    fill = hex_rgba(colour.colour)
    return TerrainSpec(
        name=colour.name or wangset.name,
        fill=fill,
        outline=(*(part * 3 // 5 for part in fill[:3]), fill[3]),
    )


def hex_rgba(text: str) -> tuple[int, int, int, int]:
    """``#rrggbb`` or ``#aarrggbb`` as four channels; opaque white on nonsense.

    Public since 2026-08-30, for one caller and one reason: the tileset editor's
    Terrain tab draws a swatch off the same spec, and it has to be the same
    colour by construction rather than by a second parser beside it agreeing for
    a while.

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


#: The snap setting's three pills, and the sentence each one is worth.
SNAP_LABELS = (("off", "Off"), ("grid", "Grid"), ("pixel", "Pixel"))

SNAP_TIPS = {
    "off": "Object drags land wherever the pointer is. Hold Ctrl to snap to the grid.",
    "grid": "Object drags land on cell corners, and rotation on 15 degrees. "
    "Hold Ctrl for one unsnapped drag.",
    "pixel": "Object drags land on whole map pixels, and rotation on 15 degrees. "
    "Hold Ctrl for one unsnapped drag.",
}


def _snap_choice(state: Any) -> None:
    """What object gestures snap to, as one row of three.

    A choice rather than a switch because there are three answers and Tiled
    offers all three; the tooltips are where "and Ctrl inverts it" is said,
    because a modifier nobody documents is a modifier nobody finds.
    """
    changed, picked = controls.segmented_choice(
        "plotter-snap",
        list(SNAP_LABELS),
        state.snap if state.snap in plotter_state.SNAP_MODES else "off",
        tooltips=SNAP_TIPS,
        compact=True,
    )
    if changed:
        state.snap = picked


SETTINGS_POPUP = "plotter-map-settings"


def map_settings_popup(ctx: Any, state: Any, tab: Any) -> None:
    """Map -> Map properties: the metadata and the custom-property table.

    Two accordions in the tools pane, opened when a map is set up and closed
    for the rest of the session -- which is a dialog wearing a sidebar's
    clothes. Same door as the resize form, same flag pattern, same reason.
    """
    from imgui_bundle import imgui

    if state.map_settings_pending:
        state.map_settings_pending = False
        if tab is not None:
            imgui.open_popup(SETTINGS_POPUP)
    if tab is None or not imgui.begin_popup(SETTINGS_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    doc = tab.doc
    widgets.popup_title("Map properties")
    if tab.busy:
        widgets.busy("Saving")
        imgui.end_popup()
        return
    widgets.section("Metadata")
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
    if doc.projection in (project.STAGGERED, project.HEXAGONAL):
        # Shown only for the two projections that read them, which is the
        # same rule Skew follows above: a field that means nothing on an
        # orthogonal map is a question with no answer.
        axis = widgets.labeled_combo(
            "Stagger axis",
            doc.stagger_axis,
            [("y", "Rows (y)"), ("x", "Columns (x)")],
            help_text="Which way the offset runs. Tiled's staggeraxis.",
        )
        if axis != doc.stagger_axis:
            doc.set_map_settings(stagger_axis=axis)
        index = widgets.labeled_combo(
            "Stagger index",
            doc.stagger_index,
            [("odd", "Odd"), ("even", "Even")],
            help_text="Which rows (or columns) are the shifted ones.",
        )
        if index != doc.stagger_index:
            doc.set_map_settings(stagger_index=index)
    if doc.projection == project.HEXAGONAL:
        changed, side = controls.input_int("Hex side", int(doc.hex_side))
        if changed:
            doc.set_map_settings(hex_side=max(0, int(side)))
            # It changes ``pixel_width``, so the fit is stale: a map that
            # kept its old frame after the lattice changed shape is one
            # whose edge is off screen with no way to tell why.
            tab.view.fitted = False
        widgets.help_marker(
            "How long the hexagon's flat run is, in pixels. Zero is a "
            "staggered diamond rather than a hexagon -- which is what a "
            "hexagonal map made here wrote before this field existed."
        )

    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.section("Custom properties")
    # The map's own custom properties. They survive a Tiled round trip and
    # always have; this is the first way to set one without a text editor.
    plotter_layers.property_editor(
        ctx,
        f"plotter_map_prop:{tab.uid}",
        doc.properties,
        doc.set_map_properties,
        object_options=plotter_layers.object_options(doc),
    )
    imgui.end_popup()


RESIZE_POPUP = "plotter-resize"


def resize_popup(ctx: Any, state: Any, tab: Any) -> None:
    """Map -> Resize, as a dialog rather than as an accordion in the sidebar.

    Two hundred lines of column -- width, height, the offset pair, autocrop,
    the infinite conversion and the tile size -- for a form opened once when a
    map is set up and then never again. It is a *dialog* by nature: it is
    answered and dismissed, and everything in it is about the document rather
    than about the click you are about to make.

    Opened by name off ``state.resize_pending``, which is how a menu row asks
    for a popup: a popup belongs to the window that begins it, and a menu is
    not that window.
    """
    from imgui_bundle import imgui

    if state.resize_pending:
        state.resize_pending = False
        if tab is not None:
            imgui.open_popup(RESIZE_POPUP)
    if tab is None or not imgui.begin_popup(RESIZE_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    # The title says which map this is, because the form under it is a
    # different form on each: an infinite map has no width and height to set.
    widgets.popup_title("Map size" if tab.doc.infinite else "Resize the map")
    _resize_form(ctx, tab)
    imgui.end_popup()


def _resize_form(ctx: Any, tab: Any) -> None:
    """Grow or crop the grid, anchoring the old content by an offset.

    Cached under ``state.preview`` per tab, so typing a width does not fight
    with the document and switching tabs does not carry a half-typed number
    onto a different map.
    """
    from imgui_bundle import imgui

    key = f"plotter_resize:{tab.uid}"
    if tab.doc.infinite:
        # **No width/height fields.** An infinite map's rectangle is not
        # something the user sets: it is the window over what has been painted,
        # and typing a number into it would either clip content or grow a
        # rectangle the next stroke re-derives anyway. What is left is the two
        # things that still mean something -- moving the content, and giving the
        # map an edge again.
        widgets.muted(
            f"Infinite. {tab.doc.width} x {tab.doc.height} tiles painted so far, "
            f"growing as you paint past the edge."
        )
        imgui.dummy((0, sp(tokens.SP_2)))
        _offset_form(ctx, tab)
        imgui.dummy((0, sp(tokens.SP_2)))
        if controls.button("Shrink to content", (-1, 0)):
            if tab.doc.autocrop():
                ctx.state.preview.pop(key, None)
                tab.view.fitted = False
            else:
                ctx.toast("There is nothing painted to shrink to.", "warn")
        imgui.dummy((0, sp(tokens.SP_2)))
        _infinite_form(ctx, tab)
        imgui.dummy((0, 10))
        _tile_size_form(ctx, tab)
        return

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
    imgui.dummy((0, sp(tokens.SP_1)))
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

    imgui.dummy((0, sp(tokens.SP_2)))
    _offset_form(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    # Tiled's Map -> Autocrop. Delegated to ``resize`` verbatim inside the
    # document, so objects travel by the rule that already exists.
    if controls.button("Autocrop to content", (-1, 0)):
        if tab.doc.autocrop():
            ctx.state.preview.pop(key, None)
            tab.view.fitted = False
        else:
            ctx.toast("There is nothing painted to crop to.", "warn")

    imgui.dummy((0, sp(tokens.SP_2)))
    _infinite_form(ctx, tab)

    imgui.dummy((0, 10))
    _tile_size_form(ctx, tab)


def _infinite_form(ctx: Any, tab: Any) -> None:
    """The conversion, both ways, behind the confirm the destructive way needs.

    Only one of the two directions loses anything, and only that one asks: going
    infinite keeps every cell where it is, while coming back has to pick a fixed
    rectangle and the cells outside it are gone. The button says which it is.
    """
    doc = tab.doc
    if not doc.infinite:
        if controls.button("Make infinite", (-1, 0)):
            doc.set_infinite(True)
            tab.view.fitted = False
        widgets.muted("The map grows as you paint past its edge. Nothing is lost.")
        return
    if controls.button("Give the map a fixed size", (-1, 0)):
        bounds = doc.content_bounds()
        if bounds is None:
            # Nothing painted, so nothing to lose and nothing to say: the
            # rectangle it keeps is the one already on screen.
            doc.set_infinite(False)
            tab.view.fitted = False
            return
        x0, y0, x1, y1 = bounds
        dialogs.ask_delete(
            ctx,
            title="Give the map a fixed size",
            message=(
                f"The map will be cropped to the {x1 - x0 + 1} x {y1 - y0 + 1} "
                "tiles that hold something, and will stop growing when you "
                "paint past the edge."
                "\n\nThis is one undo step."
            ),
            on_confirm=lambda: _make_finite(tab),
        )


def _make_finite(tab: Any) -> None:
    """The confirm's other half. Named rather than inlined as a lambda so the
    view reset travels with the conversion into every future caller."""
    tab.doc.set_infinite(False)
    tab.view.fitted = False


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
    # Not offered on an infinite map, and not merely disabled: there is no
    # edge for a roll to wrap around, so the box would be asking a question
    # with no answer. The engine refuses it by name for the same reason; this
    # is that refusal made unreachable rather than survivable.
    infinite = bool(tab.doc.infinite)
    if infinite:
        widgets.muted_wrapped(
            "An infinite map has no edge, so an offset slides its window "
            "rather than wrapping or clearing -- nothing is lost either way."
        )
    else:
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
        try:
            moved = tab.doc.offset(
                int(form["dx"]),
                int(form["dy"]),
                wrap=False if infinite else bool(form["wrap"]),
                scope=str(form["scope"]),
            )
        except ValueError as exc:
            # ``_resize``'s framing: the engine's sentence says what was wrong
            # and nothing about what was being attempted.
            ctx.toast(f"The map was not offset: {exc}.", "error")
            return
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
        "Apply tile size",
        (width, height) != stamp,
        (-1, 0),
        reason="That is the size the map already uses.",
    ):
        try:
            tab.doc.set_tile_size(width, height)
        except ValueError as exc:
            # Framed rather than forwarded, ``_resize_form``'s rule.
            ctx.toast(f"The tile size was not changed: {exc}.", "error")
            return
        ctx.state.preview.pop(key, None)
        tab.view.fitted = False
