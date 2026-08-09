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

from .. import icons, packwright_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("atlas file")
    manual_render.help_button(ctx, "packwright-bridge")

    width = widgets.grid_width(2)
    if imgui.button(f"{icons.PLUS} New", (width, 0)):
        packwright_mode.new_document(ctx)
    imgui.same_line()
    if imgui.button(f"{icons.FOLDER_OPEN} Open...", (width, 0)):
        packwright_mode.ask_open(ctx)

    if tab is None:
        imgui.dummy((0, 8))
        widgets.muted_wrapped("Start an atlas, open one, or drop images on the window.")
        _recent(ctx, state)
        return

    ready = not tab.busy
    packed = tab.layout is not None and tab.atlas is not None
    imgui.dummy((0, 4))
    if widgets.disabled_button(f"{icons.SAVE} Save (Ctrl+S)", ready, (width, 0)):
        packwright_mode.save(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button("Save as...", ready, (width, 0)):
        packwright_mode.save_as(ctx, tab)
    if tab.path is not None:
        widgets.muted(str(tab.path))
    if tab.dirty:
        widgets.muted("Unsaved changes.")

    imgui.dummy((0, 8))
    widgets.section("export")
    if widgets.disabled_button(
        f"{icons.DOWNLOAD} Atlas + JSON (Ctrl+Shift+E)", ready and packed, (-1, 0)
    ):
        packwright_mode.export_files(ctx, tab)
    if packed and tab.layout.is_grid:
        widgets.muted_wrapped("A .tsx goes beside them: this grid is a tileset.")
    else:
        widgets.muted_wrapped("TexturePacker's JSON (Array) schema, which most engines read.")

    imgui.dummy((0, 8))
    widgets.section("library")
    if widgets.disabled_button(
        f"{icons.UPLOAD} Export to the library (Ctrl+E)", ready and packed, (-1, 0)
    ):
        packwright_mode.export_library(ctx, tab)
    widgets.muted_wrapped(
        "The atlas becomes an ordinary reference asset, with this document kept "
        "beside it so it can be reopened from the library."
    )

    _recent(ctx, state)


def _recent(ctx: Any, state: Any) -> None:
    from pathlib import Path

    from imgui_bundle import imgui

    if not state.recent:
        return
    imgui.dummy((0, 8))
    widgets.section("recent")
    for path in list(state.recent)[:6]:
        if imgui.selectable(f"{Path(path).name}##{path}", False)[0]:
            packwright_mode.open_path(ctx, Path(path))
        if imgui.is_item_hovered():
            imgui.set_tooltip(path)
