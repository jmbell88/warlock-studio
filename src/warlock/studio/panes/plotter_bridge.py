"""Plotter's right-bottom pane: files, Tiled interop and the library.

The two ways out, and they are genuinely different. **The library** mints an
ordinary asset -- a flat render as a ``done`` reference row, with ``map.wmap``
beside it -- so the map joins the same library, inspector and 2D pipeline every
other asset is in. **Tiled** writes a ``.tmx`` (or ``.tmj``) plus a ``.tsx`` and
a ``.png`` per tileset, which is what an engine's importer expects.

Exporting for Tiled deliberately does *not* retarget Ctrl+S: the ``.wmap``
holds things the ``.tmx`` cannot -- an embedded tileset image above all -- so
silently making the export the document's home would lose them on the next save.
"""

from __future__ import annotations

from typing import Any

from .. import icons, plotter_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Map file")
    manual_render.help_button(ctx, "plotter-bridge")

    if tab is None:
        # The recent list and nothing else. New/Open and the drop hint are on
        # the empty canvas one column to the left, word for word -- one pair of
        # buttons in two places, and the same sentence twice on one screen.
        _recent(ctx)
        return

    # **The facts, and the two verbs a hand reaches for without a menu.** The
    # nine buttons that were here -- New, Open, Save, Save As, the two exports,
    # the library export -- are the Map menu's rows now (W3.1), which is where
    # a user coming from Tiled looks for them. What a panel is for is what is
    # left: where this map lives, whether it is written, and how deep the undo
    # stack is. Clay's bridge is the same size for the same reason.
    if tab.path is not None:
        imgui.text_wrapped(str(tab.path))
    else:
        widgets.muted("Not saved to a file yet.")
    if tab.dirty:
        widgets.muted("Unsaved changes.")
    else:
        widgets.muted("Saved.")
    if not tab.doc.tilesets:
        widgets.muted_wrapped(
            "No tileset yet -- Tileset > Import a tileset. A Tiled map needs "
            "at least one before it can be exported."
        )

    imgui.dummy((0, 8))
    _history(ctx, tab)
    _recent(ctx)


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen.

    This mode had a full undo stack and no visible control for it, so the
    feature existed only for a user who already knew Ctrl+Z -- while Inker drew
    the same pair twice. ``plotter_mode.undo``/``redo`` rather than
    ``tab.doc.undo()`` here, so the button and the chord carry the same side
    effects (see the history block in that module).
    """
    from imgui_bundle import imgui

    doc = tab.doc
    width = widgets.grid_width(2)
    if widgets.disabled_button(
        f"{icons.UNDO} Undo", doc.history.can_undo, (width, 0), reason="Nothing to undo yet."
    ):
        plotter_mode.undo(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.REDO} Redo",
        doc.history.can_redo,
        (width, 0),
        reason="Nothing to redo: this is the newest step.",
    ):
        plotter_mode.redo(ctx, tab)
    widgets.muted(f"{len(doc.history)} step(s)")


def _recent(ctx: Any) -> None:
    from pathlib import Path

    widgets.recent_files(
        plotter_mode.recent_paths(ctx),
        lambda path: plotter_mode.open_path(ctx, Path(path)),
    )
