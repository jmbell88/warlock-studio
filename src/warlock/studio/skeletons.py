"""Each workspace's columns, as :class:`~.layout_skeleton.Slot` tables.

One place that says what a workspace is made of, so a saved layout has
something to be a permutation *of* -- and so the answer to "which panes does
Inker have" is a list rather than a function body.

**Built lazily, per call.** A slot holds its pane's ``draw``, and importing
every pane at module scope would drag imgui into anything that imports this;
the tables are cheap to build and are built once a frame, beside the drawing
they describe.

The centre column is deliberately *not* here yet: it is an anchor with strips
under it rather than an ordered stack, its flags are load-bearing (a canvas's
height reservation reads the content region), and nothing in wave 5 asks to
reorder it. The two sidebars are what a saved layout moves.
"""

from __future__ import annotations

from typing import Any

from .layout_skeleton import FILL, FIXED, SHARE, Column, Slot

#: A sidebar column's design width. The same number ``layout.SIDEBAR_WIDTHS``
#: calls "default" -- said here so a table can state a width without importing
#: the imgui-bearing layout module.
COLUMN_W = 300.0


def _role(name: str) -> Any:
    from .layout import PaneRole

    return PaneRole(name)


def _edge(name: str) -> Any:
    from .layout import PaneEdge

    return PaneEdge(name)


def inker(ctx: Any) -> dict[str, Column]:
    """Inker's two sidebars, in Aseprite's *default* arrangement.

    Colour on the left with the picker under it, the toolbox on the right over
    the preview, the tiles and the generation verbs -- and the timeline along
    the bottom of the centre column, which the workspace composes by hand.

    This reverses W2.9's "Mirrored Default" (tools left, colour right). The
    argument that put the toolbox on the left was that moving it to the right
    "would put the toolbox on the far side of the canvas from the hand"; what
    answers it is that Aseprite ships the other way round and this app is
    trying to be the program its users already have in their hands. The rail
    also cost the toolbox its heading, its help button and two clipped rows --
    at a full sidebar width all three come back.
    """

    from .panes import (
        inker_colors,
        inker_generate,
        inker_picker,
        inker_preview,
        inker_tiles,
        inker_tools,
    )

    def animated(context: Any) -> bool:
        state = getattr(context.state, "inker", None)
        tab = None if state is None else state.active
        return tab is not None and tab.doc.anim is not None

    def tiled(context: Any) -> bool:
        state = getattr(context.state, "inker", None)
        tab = None if state is None else state.active
        return tab is not None and bool(tab.doc.tilesets)

    left = Column(
        "left",
        (
            Slot(
                "inker-colors",
                "Colour",
                inker_colors.draw,
                role=_role("inspector"),
                edge=_edge("right"),
                sizing=SHARE,
                share_key="inker-colors",
                floor=inker_colors.PANEL_FLOOR,
            ),
            Slot(
                "inker-picker",
                "Picker",
                inker_picker.draw,
                role=_role("inspector"),
                edge=_edge("right"),
                sizing=FILL,
                floor=inker_picker.PICKER_FLOOR,
            ),
        ),
        width=COLUMN_W,
    )
    right = Column(
        "right",
        (
            Slot(
                "inker-preview",
                "Preview",
                inker_preview.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=FIXED,
                height=inker_preview.PREVIEW_H,
                when=animated,
            ),
            Slot(
                "inker-tools",
                "Tools",
                inker_tools.draw,
                role=_role("sidebar"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="inker-tools",
                floor=inker_tools.TOOLS_FLOOR,
            ),
            Slot(
                "inker-tiles",
                "Tiles",
                inker_tiles.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="inker-tiles",
                floor=inker_tiles.PANEL_FLOOR,
                when=tiled,
            ),
            Slot(
                "inker-generate",
                "Generation",
                inker_generate.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=FILL,
                floor=inker_generate.GENERATE_FLOOR,
            ),
        ),
        width=COLUMN_W,
    )
    return {"left": left, "right": right}


def plotter(ctx: Any) -> dict[str, Column]:
    """Plotter's two sidebars, in **Tiled's** default arrangement.

    Properties on the left with the map file under it; the layer stack on the
    right over the tileset palette; the tools in a strip across the top of the
    centre column, which ``plotter_canvas`` composes by hand.

    This reverses the shape the workspace shipped with -- tools over tilesets
    on the left, layers over properties over the file on the right -- and it is
    the same argument that turned Inker round on 2026-08-31. What put Plotter's
    panes where they were was that they mirrored ``_clay_workspace`` "so the
    editors do not drift into looking like different applications". That is a
    real cost and it is the smaller one: a user arriving at Plotter has Tiled
    in their hands, not Clay, and every reach they have learnt -- the layer list
    on the right, the properties of the thing they just clicked on the left --
    was answered here by the opposite side of the window. Internal consistency
    is worth having between two panels nobody has seen before; it is not worth
    having against the muscle memory of the program this one is trying to be.

    The tileset palette takes the **fill** slot rather than sharing, because it
    is the pane whose useful size has no ceiling: an atlas is scrolled through,
    where a layer stack is read down. Its floor is declared for the first time
    here, which is what stops a short window squeezing the picker to a heading.

    No ``layouts.VERSION`` bump and no migration. ``layout_skeleton.reconcile``
    is per column against ``set(builtin)``, so a saved v2 arrangement --
    ``left=[tools, tileset]``, ``right=[layers, properties, bridge]`` -- lands
    on this one on its own: the two slots that changed column are unknown to
    their old side and are dropped there, and unlisted on their new side and
    are appended there. An orphaned ``shares["plotter-tools"]`` and any hidden
    entry naming a moved slot are inert rather than wrong.
    """

    from .panes import (
        plotter_bridge,
        plotter_layers,
        plotter_objects,
        plotter_stamps,
        plotter_tileset,
    )

    left = Column(
        "left",
        (
            # **The selected thing, then the file.** These fields used to be
            # drawn *inside* the layer list, between two sibling rows, so
            # choosing a layer pushed its neighbours a hundred and fifty lines
            # apart and there was no column of names left to read down. They
            # have been their own pane since 2026-08-29; this moves that pane
            # to the side Tiled puts it on.
            Slot(
                "plotter-properties",
                "Properties",
                plotter_layers.draw_properties,
                role=_role("inspector"),
                edge=_edge("right"),
                sizing=SHARE,
                share_key="plotter-properties",
                floor=plotter_layers.PROPERTIES_FLOOR,
            ),
            # Under the selected thing and over the file, and only on a tile
            # layer: a stamp is a block of tiles, so the pane belongs where the
            # tileset palette belongs. On an object layer it would be nine
            # controls that cannot act.
            Slot(
                "plotter-stamps",
                "Tile stamps",
                plotter_stamps.draw,
                role=_role("inspector"),
                edge=_edge("right"),
                sizing=SHARE,
                share_key="plotter-stamps",
                floor=plotter_stamps.STAMPS_FLOOR,
                when=plotter_stamps.on_tile_layer,
            ),
            Slot(
                "plotter-bridge",
                "Map file",
                plotter_bridge.draw,
                role=_role("inspector"),
                edge=_edge("right"),
                sizing=FILL,
            ),
        ),
    )
    right = Column(
        "right",
        (
            Slot(
                "plotter-layers",
                "Layers",
                plotter_layers.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="plotter-layers",
                floor=plotter_layers.LAYERS_FLOOR,
            ),
            # **Between the stack and the palette, and only when there is
            # something to list.** Objects were rows inside the layer list,
            # which put the stack sixty rows down the pane on a map with sixty
            # triggers -- and left "where is the door I named" answerable only
            # by knowing which layer it was on.
            Slot(
                "plotter-objects",
                "Objects",
                plotter_objects.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="plotter-objects",
                floor=plotter_objects.OBJECTS_FLOOR,
                when=plotter_objects.has_object_layer,
            ),
            Slot(
                "plotter-tileset",
                "Tilesets",
                plotter_tileset.draw,
                role=_role("sidebar"),
                edge=_edge("left"),
                sizing=FILL,
                floor=plotter_tileset.TILESET_FLOOR,
            ),
        ),
    )
    return {"left": left, "right": right}


def clay(ctx: Any) -> dict[str, Column]:
    """Clay's two sidebars: the verbs on the left, the document on the right.

    The last workspace composed by hand in ``main``, and the change is not only
    tidiness: a hand-composed workspace is one a saved layout cannot permute, so
    Clay was the one editor whose panes a user could not rearrange.

    **The left column is one pane now.** It was Tools over Properties, split by
    a handle, and half of what the Tools pane held has gone to the viewport
    header -- the mode row, the tool grid, snapping, proportional editing and
    the view aids. What is left is what a *sidebar* is for: the primitives you
    add and the operations you invoke, which is a list that wants the height.

    Properties moves to the right, under the outliner, which is where the
    thing-you-have-selected belongs and where Plotter and Tiled both put it:
    what the scene *contains*, then what the selected part of it *is*, then the
    file. ``clay-props`` is a new share key, and ``clay-tools`` stops being one
    -- a column of one FILL slot has nothing to share against. An orphaned
    ``shares["clay-tools"]`` in a user's settings is inert, exactly as
    Plotter's is; see :func:`plotter` for why no migration is owed.
    """

    from .panes import clay_bridge, clay_outliner, clay_props, clay_tools

    left = Column(
        "left",
        (
            Slot(
                "clay-tools",
                "Tools",
                clay_tools.draw,
                role=_role("sidebar"),
                edge=_edge("right"),
                sizing=FILL,
            ),
        ),
    )
    right = Column(
        "right",
        (
            Slot(
                "clay-outliner",
                "Outliner",
                clay_outliner.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="clay-outliner",
            ),
            Slot(
                "clay-props",
                "Properties",
                clay_props.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="clay-props",
            ),
            Slot(
                "clay-bridge",
                "Document",
                clay_bridge.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=FILL,
            ),
        ),
    )
    return {"left": left, "right": right}


def sirens(ctx: Any) -> dict[str, Column]:
    """Sirens' two sidebars: the transport over the order list, the instrument
    list and its envelopes over the song file.

    Plotter's shape, deliberately, so the editors do not drift into looking
    like different applications -- what you *do* on the left, what the document
    *is* on the right.

    **The share keys are declared, not derived.** A key derived from the slot
    id would tie a saved layout to a pane's name, so renaming a pane would
    silently reset every user's column heights; these four are written out for
    the reason ``layout_skeleton`` states once.
    """

    from .panes import (
        sirens_bridge,
        sirens_effects,
        sirens_envelopes,
        sirens_instruments,
        sirens_orders,
        sirens_transport,
    )

    left = Column(
        "left",
        (
            Slot(
                "sirens-transport",
                "Transport",
                sirens_transport.draw,
                role=_role("sidebar"),
                edge=_edge("right"),
                sizing=SHARE,
                share_key="sirens-transport",
            ),
            Slot(
                "sirens-orders",
                "Order",
                sirens_orders.draw,
                role=_role("sidebar"),
                edge=_edge("right"),
                sizing=FILL,
            ),
        ),
    )
    right = Column(
        "right",
        (
            Slot(
                "sirens-instruments",
                "Instruments",
                sirens_instruments.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="sirens-instruments",
            ),
            Slot(
                "sirens-envelopes",
                "Envelopes",
                sirens_envelopes.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="sirens-envelopes",
                floor=sirens_envelopes.ENVELOPES_FLOOR,
            ),
            Slot(
                # Under the instrument that plays it and over the file it is
                # written into: a sound effect is a little song, so it belongs
                # with the things a song is made of rather than beside Save.
                "sirens-effects",
                "Sound effects",
                sirens_effects.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="sirens-effects",
            ),
            Slot(
                "sirens-bridge",
                "Song file",
                sirens_bridge.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=FILL,
            ),
        ),
    )
    return {"left": left, "right": right}


#: Which builder serves which workspace. A workspace with no entry keeps its
#: hand-written composition, which is what the centre-heavy ones (Create,
#: Review, Troupe, Poser) still have -- and Packwright, which is the last of
#: the sidebar-shaped ones left composing by hand.
BUILDERS = {
    "clay": clay,
    "inker": inker,
    "plotter": plotter,
    "sirens": sirens,
}


def for_mode(ctx: Any, mode: str) -> dict[str, Column]:
    builder = BUILDERS.get(mode)
    return {} if builder is None else builder(ctx)


def ordered(ctx: Any, library: Any, mode: str, column: Column) -> list[Slot]:
    """One column's slots, in the order the active saved layout wants them.

    Reconciled every read and never written back (``layouts.Library.order``),
    and **hidden slots are dropped here rather than in the renderer**: a
    hideable slot the layout hides is not part of the frame at all, which is
    what keeps a hidden pane from paying for a child window nobody sees.
    """

    live = {slot.id: slot for slot in column.live(ctx)}
    if library is None:
        return list(live.values())
    hidden = library.hidden(mode)
    order = library.order(mode, column.id, list(live))
    out = []
    for slot_id in order:
        slot = live.get(slot_id)
        if slot is None:
            continue
        if slot.hideable and slot_id in hidden:
            continue
        out.append(slot)
    return out
