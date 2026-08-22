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


def _role(name: str) -> Any:
    from .layout import PaneRole

    return PaneRole(name)


def _edge(name: str) -> Any:
    from .layout import PaneEdge

    return PaneEdge(name)


def inker(ctx: Any) -> dict[str, Column]:
    """Inker's two sidebars after wave 2: a rail, and a stack of three."""

    from .panes import inker_colors, inker_preview, inker_tiles, inker_tools

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
                "inker-tools",
                "Tools",
                inker_tools.draw,
                role=_role("sidebar"),
                edge=_edge("right"),
                sizing=FILL,
                # The rail is the workspace: hiding it would leave a column of
                # nothing, and moving it into the right column would put the
                # toolbox on the far side of the canvas from the hand.
                movable=False,
                hideable=False,
            ),
        ),
        width=inker_tools.RAIL_W,
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
                "inker-colors",
                "Colour",
                inker_colors.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=SHARE,
                share_key="inker-colors",
                floor=inker_colors.PANEL_FLOOR,
            ),
            Slot(
                "inker-tiles",
                "Tiles",
                inker_tiles.draw,
                role=_role("inspector"),
                edge=_edge("left"),
                sizing=FILL,
                floor=inker_tiles.PANEL_FLOOR,
                when=tiled,
            ),
        ),
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


#: Which builder serves which workspace. A workspace with no entry keeps its
#: hand-written composition, which is what the centre-heavy ones (Create,
#: Review, Troupe, Poser) still have.
BUILDERS = {
    "inker": inker,
    "plotter": plotter,
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
