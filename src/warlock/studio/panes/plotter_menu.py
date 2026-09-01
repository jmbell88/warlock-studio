"""Plotter's menu strip: Map, View, Layer, Tileset.

``inker_menu``'s shape, and Tiled's four menus. Drawn from
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

from .. import controls, layout_edit, plotter_mode, plotter_tilesets, toolbar
from . import plotter_tools

BAR = "plotter-menu"

#: The strip, in Tiled's order.
#:
#: **View is second**, which is where Tiled puts it, and it is the menu that
#: closes the split this app had: the canvas aids were four toggles in the
#: tools sidebar and a fifth row filed under *Layer*, so there was no one
#: place to ask "what is the canvas showing". Both surfaces read
#: ``plotter_tools.VIEW_TOGGLES`` now -- the popover on the bar and these rows.
MENUS: tuple[str, ...] = ("Map", "View", "Layer", "Tileset")

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
    elif name == "View":
        _view_rows(ctx, state, tab)
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
    if _row("Go to coordinate...", enabled=tab is not None, reason="Nothing is open."):
        # Outside the busy gate, unlike the two above it: a jump moves the view
        # and writes nothing, so there is no reason a map being saved cannot be
        # scrolled. ``resize_pending``'s flag pattern all the same -- the popup
        # belongs to the canvas.
        state.goto_pending = True
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


def _view_rows(ctx: Any, state: Any, tab: Any) -> None:
    """The canvas aids, then the pane arrangement.

    The five toggles are drawn by ``plotter_tools.view_rows`` rather than
    restated here: they are also the toolbar's View popover, and a menu that
    kept its own copy is a menu that would come to disagree with the button
    three inches above it. "Highlight current layer" arrives with them, having
    been the one row of the five that lived under *Layer*.

    *Rearrange panes* is here because View is where a user looks for it and
    because Shift+W is otherwise a chord with no on-screen door at all.
    """

    plotter_tools.view_rows(ctx, state)
    controls.menu_separator()
    if _row("Rearrange panes...", "Shift+W"):
        layout_edit.toggle(ctx.state)


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
        plotter_mode.shift_layer(doc, active, 1)
    if _row(
        "Lower layer",
        "Ctrl+Shift+Down",
        enabled=ready and many and active is not None,
        reason=BUSY,
    ):
        plotter_mode.shift_layer(doc, active, -1)


def _tileset_rows(ctx: Any, state: Any, tab: Any) -> None:
    ready = tab is not None and not tab.busy
    tilesets = tab is not None and bool(tab.doc.tilesets)
    if _row("Import a tileset...", enabled=ready, reason=BUSY):
        plotter_tilesets.ask_add_tileset(ctx)
    if _row(
        "Reload the image...",
        enabled=ready and tilesets,
        reason=BUSY if not ready else NO_TILESET,
    ):
        # The other half of *Polish in Inker*, for a paint program that is not
        # Inker: an atlas exported, edited in Aseprite or Photoshop and saved
        # back had no way in, and re-importing it made a *second* tileset that
        # every gid on the map still ignored. Through the same
        # ``MapDoc.replace_tileset`` door the Inker round trip uses, so the ids,
        # the firstgid and the declared terrains are kept and the map redraws
        # rather than renumbering.
        plotter_tilesets.ask_replace_tileset(ctx, max(0, state.tileset_index))
    if _row(
        "Edit tileset...",
        enabled=ready and tilesets,
        reason=BUSY if not ready else NO_TILESET,
    ):
        # The sheet over the centre pane; see ``plotter_tileset_editor``. The
        # palette's own footer opens the same one through the same door.
        plotter_mode.edit_tileset(ctx)
