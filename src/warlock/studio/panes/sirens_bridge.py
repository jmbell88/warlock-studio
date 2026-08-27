"""Sirens' right-bottom pane: where this song lives, and the undo stack.

``plotter_bridge``'s shape and its reasoning: the facts about the file, the two
history verbs, and the recent list. New/Open/Save are the Song menu's rows,
which is where a user looks for them.

**Export is here rather than in the transport**, next to Save and the file's
own path, because that is what it is: the second thing this document can be
written out as. The transport is about what you are hearing now.

**The button names the folder rather than a file**, because that is what the
picker asks for -- ``song.wav`` lands beside a ``stems/`` and an ``sfx/``
directory, all of them derived from the ``.wsng`` and none of them named by the
user. The panel says so before the click rather than after it.
"""

from __future__ import annotations

from typing import Any

from .. import icons, sirens_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = sirens_mode.ensure(ctx)
    tab = state.active
    widgets.section("Song file")
    manual_render.help_button(ctx, "sirens-bridge")

    if tab is None:
        # The recent list and nothing else -- the grid's empty state already
        # carries the New/Open pair and the same sentence twice on one screen
        # reads as two problems.
        _recent(ctx)
        return

    # **One sentence about where this song stands**, not two: the states are
    # exclusive and a panel that printed "Not saved to a file yet." *and*
    # "Saved." on the same screen is what ``plotter_bridge`` was fixed for.
    if tab.path is None:
        widgets.muted("Not saved to a file yet." if tab.dirty else "Nothing to save yet.")
    else:
        imgui.text_wrapped(str(tab.path))
        widgets.muted("Unsaved changes." if tab.dirty else "Saved.")

    imgui.dummy((0, 8))
    _history(ctx, tab)
    imgui.dummy((0, 8))
    _export(ctx, tab)
    _recent(ctx)


def _export(ctx: Any, tab: Any) -> None:
    """The audio, out. One button and one sentence about what it writes.

    Enabled on a document with *something* to render -- an order list or a
    sound effect -- rather than always, because a brand-new song would otherwise
    open a folder picker and then write a folder of empty WAVs.
    ``sirens_io.export_files`` refuses the same case; this is the same refusal
    said before the click instead of after it.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    ready = bool(doc.order or doc.oneshots)
    if widgets.disabled_button(
        f"{icons.DOWNLOAD} Export audio...",
        ready and not tab.busy,
        (-1, 0),
        reason=(
            "This song is being written; the button comes back when it lands."
            if ready
            else "There is nothing in the order list to export yet."
        ),
    ):
        sirens_mode.export_files(ctx, tab)
    counts = [f"{len(doc.channels)} stem(s)"]
    if doc.oneshots:
        counts.append(f"{len(doc.oneshots)} effect(s)")
    imgui.dummy((0, 4))
    widgets.muted_wrapped(
        f"Into a folder you pick: {sirens_mode.SONG_NAME}, then"
        f" {sirens_mode.STEM_DIR}/ and {sirens_mode.SFX_DIR}/ -- "
        + " and ".join(counts)
        + ". The .wsng is the composition; every WAV is derived from it."
    )


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen.

    ``sirens_mode.undo``/``redo`` rather than ``tab.doc.undo()``, so the button
    and the chord carry the same side effects -- the caret clamp and the
    re-render, both of which belong to *undoing* rather than to the keyboard.
    """
    from imgui_bundle import imgui

    doc = tab.doc
    width = widgets.grid_width(2)
    if widgets.disabled_button(
        f"{icons.UNDO} Undo", doc.history.can_undo, (width, 0), reason="Nothing to undo yet."
    ):
        sirens_mode.undo(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.REDO} Redo",
        doc.history.can_redo,
        (width, 0),
        reason="Nothing to redo: this is the newest step.",
    ):
        sirens_mode.redo(ctx, tab)
    widgets.muted(f"{len(doc.history)} step(s)")


def _recent(ctx: Any) -> None:
    from pathlib import Path

    widgets.recent_files(
        sirens_mode.recent_paths(ctx),
        lambda path: sirens_mode.open_path(ctx, Path(path)),
    )
