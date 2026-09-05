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

from .. import icons, packwright_mode, tokens, verbs, widgets
from ..manual import render as manual_render
from ..tokens import sp


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

    # The shared header -- see ``widgets.document_header``. The busy gate is
    # ``tab.busy`` rather than ``tab.saving`` because an export in flight also
    # has to hold the four buttons.
    widgets.document_header(
        tab,
        new=lambda: packwright_mode.new_document(ctx),
        open_=lambda: packwright_mode.ask_open(ctx),
        save=lambda: packwright_mode.save(ctx, tab),
        save_as=lambda: packwright_mode.save_as(ctx, tab),
        saving=tab.busy,
    )

    ready = not tab.busy
    # ``pack_stale_why`` and not merely "is there an atlas": a failed repack
    # leaves the previous one in place, and exporting it writes a file about
    # sprites this document no longer holds.
    packed = tab.layout is not None and tab.atlas is not None and not tab.pack_stale_why
    # Two gates, and every button below is behind one or both of them. Hoisted
    # so the four of them explain the same state in the same words -- the
    # ``_VIEWPORT_WHY`` pattern.
    busy_why = widgets.DOCUMENT_SAVING_WHY
    # There is no Pack button and there never was: packing is automatic, run
    # from the centre pane's pump whenever ``pack_dirty`` is set. This used to
    # send the user hunting for a control that does not exist, which is the
    # worst kind of empty state -- one that reads as a working instruction.
    packed_why = tab.pack_stale_why or (
        "Nothing is packed yet. Add images -- packing runs by itself."
    )

    imgui.dummy((0, sp(tokens.SP_2)))
    _history(ctx, tab)

    imgui.dummy((0, sp(tokens.SP_2)))
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
        # The schema the document is *set* to, not the default: the settings
        # pane offers Array and Hash and this line said Array either way, so an
        # atlas exported as Hash was described as something else in the one
        # place a reader looks to check.
        schema = str(tab.doc.settings.json_schema or "array")
        label = "Hash" if schema == "hash" else "Array"
        widgets.muted_wrapped(
            f"TexturePacker's JSON ({label}) schema, which most engines read."
        )

    imgui.dummy((0, sp(tokens.SP_2)))
    # The one heading every mode's exits are under -- see ``inker_bridge``'s
    # ``_pipeline``. The Export block above stays named for what it writes: it
    # produces files for another application, not a move inside the app.
    widgets.section("Take it somewhere")
    if widgets.disabled_button(
        f"{icons.UPLOAD} {verbs.EXPORT_TO_LIBRARY} (Ctrl+E)",
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
