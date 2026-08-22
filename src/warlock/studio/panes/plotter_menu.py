"""Plotter's menu strip: Map, Layer, Tileset.

``inker_menu``'s shape, and Tiled's three menus. Drawn from
``plotter_canvas.draw`` before the tab bar, inside the centre window, because
an imgui popup only renders in the id stack of the window that opened it -- and
through :mod:`~warlock.studio.toolbar`, so a strip too wide for the pane
collapses into an overflow with the names back rather than clipping.

**A row here is a verb the panels no longer have to carry.** The bridge panel
sheds nine of its eleven controls and keeps the *facts* -- the path, the
unsaved marker, the undo pair, the step count -- which is what a panel is for;
the tools pane sheds the whole Resize/Offset/Autocrop/Infinite/Tile-size
accordion, about two hundred lines, to Map ▸ Resize. That dialog is begun by a
pane through ``state.resize_pending``, the pattern ``state.setup_pending``
already establishes with its reasoning written out.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, plotter_mode, plotter_tilesets, toolbar

BAR = "plotter-menu"

#: The strip, in Tiled's order.
MENUS: tuple[str, ...] = ("Map", "Layer", "Tileset")

BUSY = "This map is being written; the rows come back when it lands."
NO_TILESET = "This map has no tileset yet, so there is nothing to write."


def draw(ctx: Any) -> None:
    state = plotter_mode.ensure(ctx)
    tab = state.active
    items = [toolbar.Item(name, name, role=controls.ButtonRole.GHOST) for name in MENUS]
    clicked = toolbar.toolbar(BAR, items)
    if clicked:
        imgui.open_popup(controls.menu_bar_id(BAR, clicked))
    for name in MENUS:
        with controls.menu_popup(controls.menu_bar_id(BAR, name)) as opened:
            if opened:
                _rows(ctx, state, tab, name)


def _row(label: str, key: str = "", *, enabled: bool = True, reason: str = "") -> bool:
    hit = controls.menu_item(f"{label}##{BAR}/{label}", key, False, enabled, reason=reason)
    return bool(hit[0] if isinstance(hit, tuple) else hit)


def _rows(ctx: Any, state: Any, tab: Any, name: str) -> None:
    if name == "Map":
        _map_rows(ctx, state, tab)
    elif name == "Layer":
        _layer_rows(ctx, state, tab)
    else:
        _tileset_rows(ctx, state, tab)


def _map_rows(ctx: Any, state: Any, tab: Any) -> None:
    ready = tab is not None and not tab.busy
    tilesets = tab is not None and bool(tab.doc.tilesets)
    if _row("New...", "Ctrl+N"):
        plotter_mode.ask_new_document(ctx)
    if _row("Open...", "Ctrl+O"):
        plotter_mode.ask_open(ctx)
    controls.menu_separator()
    if _row("Save", "Ctrl+S", enabled=ready, reason=BUSY):
        plotter_mode.save(ctx, tab)
    if _row("Save As...", "Ctrl+Shift+S", enabled=ready, reason=BUSY):
        plotter_mode.save_as(ctx, tab)
    controls.menu_separator()
    if _row("Resize...", enabled=ready, reason=BUSY):
        # A flag, not a call: the dialog is a popup and a popup belongs to the
        # window that begins it. ``plotter_state.setup_pending`` is the same
        # pattern with the same reason written out.
        state.resize_pending = True
    if _row("Map properties...", enabled=ready, reason=BUSY):
        state.map_settings_pending = True
    controls.menu_separator()
    if _row(
        "Export .tmx",
        "Ctrl+Shift+E",
        enabled=ready and tilesets,
        reason=BUSY if not ready else NO_TILESET,
    ):
        plotter_mode.export_map(ctx, "tmx", tab)
    if _row(
        "Export .tmj", enabled=ready and tilesets, reason=BUSY if not ready else NO_TILESET
    ):
        plotter_mode.export_map(ctx, "tmj", tab)
    if _row(
        "Export to the library",
        "Ctrl+E",
        enabled=ready and tilesets,
        reason=BUSY if not ready else NO_TILESET,
    ):
        plotter_mode.export_library(ctx, tab)
    controls.menu_separator()
    if _row("Close", "Ctrl+W", enabled=tab is not None, reason="Nothing is open."):
        plotter_mode.close_tab(ctx, tab.uid)


def _layer_rows(ctx: Any, state: Any, tab: Any) -> None:
    doc = None if tab is None else tab.doc
    ready = tab is not None and not tab.busy
    active = None if doc is None else doc.active_layer
    many = doc is not None and len(doc.layers) > 1
    if _row("New tile layer", enabled=ready, reason=BUSY):
        doc.add_tile_layer()
    if _row("New object layer", enabled=ready, reason=BUSY):
        doc.add_object_layer()
    if _row("New group", enabled=ready, reason=BUSY):
        doc.add_group_layer()
    if _row("Duplicate layer", enabled=ready and active is not None, reason=BUSY):
        doc.duplicate_layer(active)
    if _row(
        "Delete layer",
        enabled=ready and many and active is not None,
        reason="A map keeps at least one layer." if ready else BUSY,
    ):
        doc.remove_layer(active)
    controls.menu_separator()
    if _row(
        "Raise layer",
        "Ctrl+Shift+Up",
        enabled=ready and many and active is not None,
        reason=BUSY,
    ):
        _shift(doc, active, 1)
    if _row(
        "Lower layer",
        "Ctrl+Shift+Down",
        enabled=ready and many and active is not None,
        reason=BUSY,
    ):
        _shift(doc, active, -1)
    controls.menu_separator()
    if _row("Highlight current layer", "H", enabled=tab is not None, reason="Nothing is open."):
        # Tiled's own View toggle, filed under Layer because that is what it is
        # about. A canvas setting and nothing more -- see ``PlotterState``.
        state.highlight = not state.highlight


def _shift(doc: Any, uid: int, delta: int) -> None:
    """Move a layer one place within its own parent, clamped at the ends."""
    found = doc._locate(uid)
    if found is None:
        return
    _layer, parent_uid, _index = found
    siblings = [layer.uid for layer in doc.children_of(parent_uid)]
    if uid not in siblings:
        return
    at = siblings.index(uid)
    doc.move_layer(uid, max(0, min(len(siblings) - 1, at + delta)))


def _tileset_rows(ctx: Any, state: Any, tab: Any) -> None:
    ready = tab is not None and not tab.busy
    tilesets = tab is not None and bool(tab.doc.tilesets)
    if _row("Import a tileset...", enabled=ready, reason=BUSY):
        plotter_tilesets.ask_add_tileset(ctx)
    if _row(
        "Edit tileset...",
        enabled=ready and tilesets,
        reason=BUSY if not ready else NO_TILESET,
    ):
        # The sheet over the centre pane; see ``plotter_tileset_editor``.
        # By *index*, which is what the sheet addresses: order is firstgid
        # order, so an index is stable for as long as the list is.
        state.editing_tileset = max(0, state.tileset_index)
