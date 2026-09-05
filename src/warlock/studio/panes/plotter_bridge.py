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

from .. import icons, plotter_mode, tokens, verbs, widgets
from ..manual import render as manual_render
from ..tokens import sp

#: The undo-history popover's name, which imgui also takes as its id. Opened and
#: begun in this pane, which is what an imgui popup requires.
HISTORY_POPUP = "plotter-undo-history"

#: What this pane refuses to shrink past, in design pixels: the path line, the
#: unsaved marker, the undo pair and the step count.
#:
#: It had none until 2026-09-01 and did not need one, because it was the *only*
#: fill slot under a single share. Adding the Tile stamps pane put two shares
#: above it, and ``layout_skeleton.heights`` gives each share its proportion of
#: the room before the fill sees any -- so two at the default 0.5 left this pane
#: exactly zero pixels and the Map file panel simply was not on screen. The
#: floor is what ``heights`` reserves out of the shares' headroom, and the rule
#: it enforces ("a share gives way to what the fill under it needs") only has
#: something to enforce once the fill says what it needs.
#:
#: Found by a screenshot, which is the only thing that could have found it: the
#: pane still drew, still passed every test that calls its ``draw``, and was
#: simply allocated no height by the column.
BRIDGE_FLOOR = 150.0


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

    # The shared document header (2026-09-05). This pane had deliberately
    # given its nine buttons to the Map menu (W3.1); the four file verbs came
    # back the day every workspace got the same header, because a user who
    # found Save in Packwright's pane and not in this one read that as a gap.
    # The status line is the same one-sentence ladder the first draft of this
    # pane was fixed to -- it lives in ``widgets.document_status_text`` now.
    widgets.document_header(
        tab,
        new=lambda: plotter_mode.ask_new_document(ctx),
        open_=lambda: plotter_mode.ask_open(ctx),
        save=lambda: plotter_mode.save(ctx, tab),
        save_as=lambda: plotter_mode.save_as(ctx, tab),
    )
    if not tab.doc.tilesets:
        widgets.muted_wrapped(
            "No tileset yet -- Tileset > Import a tileset. A Tiled map needs "
            "at least one before it can be exported."
        )

    imgui.dummy((0, sp(tokens.SP_2)))
    _history(ctx, tab)
    _exits(ctx, tab)
    _recent(ctx)


def _exits(ctx: Any, tab: Any) -> None:
    """The library export, under the heading every workspace's exits share.
    It was Ctrl+E and a Map-menu row only: the one document mode whose bridge
    offered no way out of the mode (2026-09-05)."""
    from imgui_bundle import imgui

    imgui.dummy((0, sp(tokens.SP_2)))
    widgets.exits()
    ready = bool(tab.doc.tilesets) and not tab.busy
    if widgets.disabled_button(
        f"{icons.UPLOAD} {verbs.EXPORT_TO_LIBRARY}",
        ready,
        (-1, 0),
        reason="This map is being written." if tab.doc.tilesets else "Add a tileset first.",
        tooltip="Ctrl+E",
    ):
        plotter_mode.export_library(ctx, tab)


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen.

    This mode had a full undo stack and no visible control for it, so the
    feature existed only for a user who already knew Ctrl+Z -- while Inker drew
    the same pair twice. ``plotter_mode.undo``/``redo`` rather than
    ``tab.doc.undo()`` here, so the button and the chord carry the same side
    effects (see the history block in that module).
    """
    widgets.history_block(
        ctx,
        tab,
        key="plotter",
        undo=lambda: plotter_mode.undo(ctx, tab),
        redo=lambda: plotter_mode.redo(ctx, tab),
        step=lambda index: plotter_mode.step_history(ctx, tab, index),
        opened="(the map as opened)",
    )


def _history_popup(ctx: Any, tab: Any) -> None:
    """The Undo History popover, by the name the smoke test opens it under.
    The drawing is ``widgets.history_popup``, shared with every bridge."""
    widgets.history_popup(
        HISTORY_POPUP,
        tab,
        lambda index: plotter_mode.step_history(ctx, tab, index),
        key="plotter",
        opened="(the map as opened)",
    )


def _recent(ctx: Any) -> None:
    from pathlib import Path

    widgets.recent_files(
        plotter_mode.recent_paths(ctx),
        lambda path: plotter_mode.open_path(ctx, Path(path)),
    )
