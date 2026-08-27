"""Sirens' right-bottom pane: where this song lives, and the undo stack.

``plotter_bridge``'s shape and its reasoning: the facts about the file, the two
history verbs, and the recent list. New/Open/Save are the Song menu's rows,
which is where a user looks for them.

**There is no Export button and this pane says so.** WAV, stems and one-shot
export are Phase 4. A panel that offered the control and toasted "not yet"
would be worse than one that does not offer it, and one that offered nothing at
all would send the user hunting -- so the absence is written down instead.
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
    widgets.muted_wrapped(
        "Exporting a WAV is not built yet -- the .wsng is the composition, and"
        " every audio file will be derived from it."
    )
    _recent(ctx)


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
