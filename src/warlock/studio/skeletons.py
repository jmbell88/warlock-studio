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
    """Plotter's two sidebars: tools over the tileset palette, layers over the
    bridge -- Clay's shape, deliberately, so the editors do not drift into
    looking like different applications."""

    from .panes import plotter_bridge, plotter_layers, plotter_tileset, plotter_tools

    left = Column(
        "left",
        (
            Slot(
                "plotter-tools",
                "Tools",
                plotter_tools.draw,
                role=_role("sidebar"),
                edge=_edge("right"),
                sizing=SHARE,
                share_key="plotter-tools",
            ),
            Slot(
                "plotter-tileset",
                "Tileset",
                plotter_tileset.draw,
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
                "plotter-layers",
                "Layers",
                plotter_layers.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="plotter-layers",
                floor=plotter_layers.LAYERS_FLOOR,
            ),
            # **The stack, then the selected thing, then the file** -- Tiled's
            # own arrangement, and the reason this slot exists at all: these
            # fields used to be drawn *inside* the list, between two sibling
            # rows, so choosing a layer pushed its neighbours a hundred and
            # fifty lines apart and there was no column of names left to read
            # down.
            Slot(
                "plotter-properties",
                "Properties",
                plotter_layers.draw_properties,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="plotter-properties",
                floor=plotter_layers.PROPERTIES_FLOOR,
            ),
            Slot(
                "plotter-bridge",
                "Map file",
                plotter_bridge.draw,
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
#: Review, Troupe, Poser) still have.
BUILDERS = {
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
