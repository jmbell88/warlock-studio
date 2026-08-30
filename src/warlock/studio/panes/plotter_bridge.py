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

from .. import controls, icons, plotter_mode, widgets
from ..manual import render as manual_render

#: The undo-history popover's name, which imgui also takes as its id. Opened and
#: begun in this pane, which is what an imgui popup requires.
HISTORY_POPUP = "plotter-undo-history"


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
    # **One sentence about where this map stands**, not two. Written as a
    # ladder because the states are exclusive and the first draft printed
    # "Not saved to a file yet." *and* "Saved." on the same screen -- which the
    # screenshot pass caught, and which is the kind of thing only a picture
    # does catch.
    if tab.path is None:
        widgets.muted(
            "Not saved to a file yet." if tab.dirty else "Nothing to save yet."
        )
    else:
        imgui.text_wrapped(str(tab.path))
        widgets.muted("Unsaved changes." if tab.dirty else "Saved.")
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
    if controls.button(
        f"{len(doc.history)} step(s)##plotter-history",
        (-1, 0),
        tooltip="Every step, with the head marked. Click one to go there.",
    ):
        imgui.open_popup(HISTORY_POPUP)
    _history_popup(ctx, tab)


def _history_popup(ctx: Any, tab: Any) -> None:
    """The Undo History panel, as a popover rather than a tenth pane.

    ``inker_menu._history_popup``'s shape and its whole argument: a pane would
    be a column of one list, on screen all session, for a thing reached when
    something has gone wrong -- and it would want a share, a floor, a help
    target and a place in every saved layout.

    **It holds no list of its own.** The rows are ``UndoStack.history()`` and a
    click is ``step_to``, so what is drawn is what the stack holds -- which is
    how an undo panel goes wrong, by keeping a copy that drifts once the byte
    budget evicts a step.

    The step count was already on screen and was already the right label for
    this; making it the button is one control rather than two, and it is the
    thing a reader is looking at when they want the list.
    """
    from imgui_bundle import imgui

    if not imgui.begin_popup(HISTORY_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    history = tab.doc.history
    steps = history.history()
    done = sum(1 for _label, is_done in steps if is_done)
    widgets.secondary(f"{len(steps)} step(s)")
    imgui.separator()
    if controls.selectable("(the map as opened)##plotter-undo-0", done == 0)[0]:
        plotter_mode.step_history(ctx, tab, 0)
    for index, (label, is_done) in enumerate(steps):
        # The *count of done steps* this row stands for, which is the number
        # ``step_to`` takes: row 0 is "one step done".
        wanted = index + 1
        at_head = is_done and wanted == done
        row = f"{label}  <" if at_head else (label if is_done else f"{label}  (undone)")
        if controls.selectable(f"{row}##plotter-undo{index}", at_head)[0]:
            plotter_mode.step_history(ctx, tab, wanted)
    imgui.end_popup()


def _recent(ctx: Any) -> None:
    from pathlib import Path

    widgets.recent_files(
        plotter_mode.recent_paths(ctx),
        lambda path: plotter_mode.open_path(ctx, Path(path)),
    )
