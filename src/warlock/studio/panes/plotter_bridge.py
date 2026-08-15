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

from .. import icons, plotter_mode, plotter_state, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = plotter_mode.ensure(ctx)
    tab = state.active
    widgets.section("Map file")
    manual_render.help_button(ctx, "plotter-bridge")

    width = widgets.grid_width(2)
    if imgui.button(f"{icons.PLUS} New", (width, 0)):
        plotter_mode.new_document(ctx)
    imgui.same_line()
    if imgui.button(f"{icons.FOLDER_OPEN} Open...", (width, 0)):
        plotter_mode.ask_open(ctx)

    if tab is None:
        imgui.dummy((0, 8))
        widgets.muted_wrapped(
            f"Start a map, open one, or drop a {plotter_state.MAP_SUFFIX_TEXT} on the window."
        )
        _recent(ctx)
        return

    ready = not tab.busy
    # ``packwright_bridge``'s two hoisted sentences, for the same reason: the
    # five buttons below share two gates, and five spellings of one state is
    # five problems for the user to look for.
    busy_why = "This map is being written; the buttons come back when it lands."
    tileset_why = "This map has no tileset yet, so there is nothing to write."
    imgui.dummy((0, 4))
    if widgets.disabled_button(
        f"{icons.SAVE} Save (Ctrl+S)", ready, (width, 0), reason=busy_why
    ):
        plotter_mode.save(ctx, tab)
    imgui.same_line()
    if widgets.disabled_button("Save as...", ready, (width, 0), reason=busy_why):
        plotter_mode.save_as(ctx, tab)
    if tab.path is not None:
        widgets.muted(str(tab.path))
    if tab.dirty:
        widgets.muted("Unsaved changes.")

    imgui.dummy((0, 8))
    widgets.section("Tiled")
    has_tilesets = bool(tab.doc.tilesets)
    if widgets.disabled_button(
        "Export .tmx (Ctrl+Shift+E)",
        ready and has_tilesets,
        (-1, 0),
        reason=busy_why if not ready else tileset_why,
    ):
        plotter_mode.export_map(ctx, "tmx", tab)
    if widgets.disabled_button(
        "Export .tmj",
        ready and has_tilesets,
        (-1, 0),
        reason=busy_why if not ready else tileset_why,
    ):
        plotter_mode.export_map(ctx, "tmj", tab)
    if not has_tilesets:
        widgets.muted_wrapped("A Tiled map needs at least one tileset.")
    else:
        widgets.muted_wrapped(
            "Writes the map beside one .tsx and one .png per tileset, which is "
            "the layout Tiled and every engine importer expects."
        )

    imgui.dummy((0, 8))
    widgets.section("Library")
    if widgets.disabled_button(
        f"{icons.UPLOAD} Export to the library (Ctrl+E)",
        ready and has_tilesets,
        (-1, 0),
        reason=busy_why if not ready else tileset_why,
    ):
        plotter_mode.export_library(ctx, tab)
    widgets.muted_wrapped(
        "A flat render becomes an ordinary reference asset, with the map itself "
        "kept beside it so this document can be reopened from the library."
    )

    _recent(ctx)


def _recent(ctx: Any) -> None:
    from pathlib import Path

    widgets.recent_files(
        plotter_mode.recent_paths(ctx),
        lambda path: plotter_mode.open_path(ctx, Path(path)),
    )
