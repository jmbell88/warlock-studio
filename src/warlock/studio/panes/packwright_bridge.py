"""Packwright's right-bottom pane: the document's file, and the two exports.

**Files** writes the atlas PNG plus the TexturePacker JSON sidecar beside it,
and a ``.tsx`` as well when the pack is a grid -- which is the whole payoff of
grid mode being a real mode: what comes out is a *tileset*, usable in Plotter or
in Tiled with no conversion.

**Library** mints an ordinary reference asset from the atlas with ``pack.wpack``
beside it, so the result joins the same library every other asset is in and can
be reopened here later.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, packwright_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("Atlas file")
    manual_render.help_button(ctx, "packwright-bridge")

    if tab is None:
        # The recent list and nothing else -- see ``plotter_bridge``. The hint
        # this used to draw was the empty canvas's sentence verbatim.
        _recent(ctx)
        return

    width = widgets.grid_width(2)
    if controls.button(f"{icons.PLUS} New", (width, 0)):
        packwright_mode.new_document(ctx)
    imgui.same_line()
    if controls.button(f"{icons.FOLDER_OPEN} Open...", (width, 0)):
        packwright_mode.ask_open(ctx)

    ready = not tab.busy
    packed = tab.layout is not None and tab.atlas is not None
    # Two gates, and every button below is behind one or both of them. Hoisted
    # so the four of them explain the same state in the same words -- the
    # ``_VIEWPORT_WHY`` pattern.
    busy_why = "This atlas is being written; the buttons come back when it lands."
    # There is no Pack button and there never was: packing is automatic, run
    # from the centre pane's pump whenever ``pack_dirty`` is set. This used to
    # send the user hunting for a control that does not exist, which is the
    # worst kind of empty state -- one that reads as a working instruction.
    packed_why = "Nothing is packed yet. Add images -- packing runs by itself."
    imgui.dummy((0, 4))
    if widgets.disabled_button(
        f"{icons.SAVE} Save (Ctrl+S)", ready, (width, 0), reason=busy_why
    ):
        packwright_mode.save(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button("Save As...", ready, (width, 0), reason=busy_why):
        packwright_mode.save_as(ctx, tab)
    if tab.busy:
        # What the two greyed buttons above mean. ``clay_bridge._facts`` is the
        # model: a disabled control with nothing saying why reads as broken.
        widgets.muted("Saving...")
    if tab.path is not None:
        widgets.muted(str(tab.path))
    if tab.dirty:
        widgets.muted("Unsaved changes.")

    imgui.dummy((0, 8))
    _history(ctx, tab)

    imgui.dummy((0, 8))
    widgets.section("Export")
    if widgets.disabled_button(
        f"{icons.DOWNLOAD} Atlas + JSON (Ctrl+Shift+E)",
        ready and packed,
        (-1, 0),
        reason=busy_why if not ready else packed_why,
    ):
        packwright_mode.export_files(ctx, tab)
    if packed and tab.layout.is_grid:
        widgets.muted_wrapped("A .tsx goes beside them: this grid is a tileset.")
    else:
        widgets.muted_wrapped("TexturePacker's JSON (Array) schema, which most engines read.")

    imgui.dummy((0, 8))
    # The one heading every mode's exits are under -- see ``inker_bridge``'s
    # ``_pipeline``. The Export block above stays named for what it writes: it
    # produces files for another application, not a move inside the app.
    widgets.section("Take it somewhere")
    if widgets.disabled_button(
        f"{icons.UPLOAD} Export to the library (Ctrl+E)",
        ready and packed,
        (-1, 0),
        reason=busy_why if not ready else packed_why,
    ):
        packwright_mode.export_library(ctx, tab)
    widgets.muted_wrapped(
        "The atlas becomes an ordinary reference asset, with this document kept "
        "beside it so it can be reopened from the library."
    )

    _recent(ctx)


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen.

    This mode had a full undo stack and no visible control for it, so the
    feature existed only for a user who already knew Ctrl+Z -- while Inker drew
    the same pair twice. ``packwright_mode.undo``/``redo`` rather than
    ``tab.doc.undo()`` here, so the button and the chord carry the same side
    effects (see the history block in that module).
    """
    from imgui_bundle import imgui

    doc = tab.doc
    width = widgets.grid_width(2)
    if widgets.disabled_button(
        f"{icons.UNDO} Undo", doc.history.can_undo, (width, 0), reason="Nothing to undo yet."
    ):
        packwright_mode.undo(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button(
        f"{icons.REDO} Redo",
        doc.history.can_redo,
        (width, 0),
        reason="Nothing to redo: this is the newest step.",
    ):
        packwright_mode.redo(ctx, tab)
    widgets.muted(f"{len(doc.history)} step(s)")


def _recent(ctx: Any) -> None:
    from pathlib import Path

    widgets.recent_files(
        packwright_mode.recent_paths(ctx),
        lambda path: packwright_mode.open_path(ctx, Path(path)),
    )
