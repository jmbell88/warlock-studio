"""Packwright's left-top pane: what is going into the atlas.

Three ways in, and the third is the reason the mode exists beside Inker: an
**open Inker document** contributes one sprite per frame of an animated clip, or
one per layer of a still one. That enumeration runs on the frame thread on
purpose -- see ``packwright_mode.add_inker_document`` for why -- and the two
loose-file paths run on a task thread because they decode PNGs.

A duplicate is skipped rather than refused when a *batch* is added, which is the
caller's decision and not the document's: dropping twenty files of which one is
already present should add nineteen.
"""

from __future__ import annotations

from typing import Any

from .. import controls, icons, packwright_mode, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    from imgui_bundle import imgui

    state = packwright_mode.ensure(ctx)
    tab = state.active
    widgets.section("Sources")
    manual_render.help_button(ctx, "packwright-sources")

    if tab is None:
        # The heading and nothing else. One voice for one empty state:
        # the canvas's ``nothing_open`` is it, and four panels each
        # repeating it reads as four separate problems.
        return

    editable = not tab.busy
    if widgets.disabled_button(f"{icons.PLUS} Add an image...", editable, (-1, 0)):
        packwright_mode.ask_add_sources(ctx)

    if widgets.disabled_button(
        f"{icons.GRID} Add a tile set...",
        editable,
        (-1, 0),
        tooltip="Loads an already-made tile sheet, slices it on a grid you "
        "set, drops the empty cells, and packs what is left -- so a sparse "
        "sheet comes back as a smaller one.",
    ):
        packwright_mode.ask_add_tileset(ctx)
    if state.tileset_import is not None and not state.tileset_import_open:
        state.tileset_import_open = True
        imgui.open_popup(TILESET_POPUP)
    _tileset_popup(ctx, state)

    _from_inker(ctx, tab, editable)

    imgui.dummy((0, 6))
    sources = tab.doc.sources
    if not sources:
        widgets.muted_wrapped(
            "Drop images on the window, add one above, or pull the frames out of "
            "a document open in Inker."
        )
        return

    widgets.muted(f"{len(sources)} sprite(s)")
    imgui.dummy((0, 2))
    for source in sources:
        _row(ctx, state, tab, source, editable)


TILESET_POPUP = "packwright-tileset-import"


def _cell_pair(value: tuple[int, int]) -> tuple[int, int]:
    """Two small integer fields on one row -- ``inker_bridge._pair``'s shape."""
    from imgui_bundle import imgui

    from ..tokens import sp

    imgui.set_next_item_width(sp(70))
    _changed_x, x = controls.input_int("##tilew", int(value[0]), 1, 8)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    _changed_y, y = controls.input_int("##tileh", int(value[1]), 1, 8)
    imgui.same_line()
    widgets.muted("tile size")
    return (max(1, int(x)), max(1, int(y)))


def _tileset_popup(ctx: Any, state: Any) -> None:
    from imgui_bundle import imgui

    from .. import theme
    from ..packwright.layout import MAX_SPRITES
    from ..packwright.sources import dedup_tiles, sprites_from_tileset, tileset_occupancy
    from ..tokens import sp

    if not imgui.begin_popup(TILESET_POPUP):
        # imgui closes a popup on a click outside, and the sheet is a megabyte
        # or two: dropping it here is what keeps a cancelled import from
        # pinning the pixels for the rest of the session.
        if state.tileset_import_open:
            state.tileset_import_open = False
            state.tileset_import = None
        return
    widgets.popup_chrome(_imgui=imgui)
    if state.tileset_import is None:
        imgui.end_popup()
        return
    path, stem, pixels = state.tileset_import
    height, width = pixels.shape[:2]
    widgets.muted(f"{stem} - {width} x {height}")

    state.tileset_cell = _cell_pair(state.tileset_cell)
    state.tileset_dedup = controls.checkbox(
        "Drop duplicate tiles", state.tileset_dedup
    )[1]
    if state.tileset_dedup:
        imgui.indent()
        state.tileset_dedup_flips = controls.checkbox(
            "match flipped / rotated", state.tileset_dedup_flips
        )[1]
        imgui.unindent()

    # The counts the numbers above actually produce, computed from the same
    # occupancy grid the import slices by -- so what the popup promises and
    # what the import does cannot disagree.
    occupied = tileset_occupancy(pixels, tile=state.tileset_cell)
    rows, columns = occupied.shape
    kept = int(occupied.sum())
    dropped = rows * columns - kept
    duplicates = 0
    if state.tileset_dedup and kept and kept <= MAX_SPRITES:
        # **The same call the import runs**, not a second count of its own: the
        # popup-promise contract ``tileset_occupancy`` already enforces for
        # emptiness, applied to the dedup it now also promises.
        tile = state.tileset_cell
        _kept, duplicates = dedup_tiles(
            sprites_from_tileset(
                pixels, tile=tile, prefix=f"{path}@{tile[0]}x{tile[1]}", name=stem
            ),
            orientations=state.tileset_dedup_flips,
        )
        kept -= duplicates
    problem = ""
    if kept == 0:
        problem = "No occupied cells at that tile size."
    elif kept > MAX_SPRITES:
        problem = f"{kept} tiles; the packer's ceiling is {MAX_SPRITES}."
    if problem:
        widgets.text_colored(theme.WARN, problem)
    elif duplicates:
        widgets.muted(
            f"{kept} unique of {kept + duplicates} tiles "
            f"({duplicates} duplicate(s) dropped), {dropped} empty dropped"
        )
    else:
        widgets.muted(f"{columns} x {rows} cells - {kept} tile(s), {dropped} empty dropped")

    imgui.dummy((0, 4))
    imgui.begin_disabled(bool(problem))
    if controls.button("Import", (sp(90), 0)) and packwright_mode.import_tileset(ctx):
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    if controls.button("Cancel##tileset", (sp(90), 0)):
        state.tileset_import = None
        state.tileset_import_open = False
        imgui.close_current_popup()
    imgui.end_popup()


def _from_inker(ctx: Any, tab: Any, editable: bool) -> None:
    """One button per open Inker document, or nothing at all.

    Nothing rather than a disabled button: with no document open the control
    would be explaining a mode the user may not have visited, and the sources
    list already says what can be added.
    """
    from imgui_bundle import imgui

    inker = ctx.state.inker
    if inker is None or not inker.docs:
        return
    imgui.dummy((0, 4))
    widgets.muted("From Inker")
    for doc in inker.docs:
        frames = len(doc.doc.anim.frames) if doc.doc.anim is not None else len(doc.doc.stack)
        what = "frame" if doc.doc.anim is not None else "layer"
        label = f"{doc.title} ({frames} {what}{'s' if frames != 1 else ''})"
        if widgets.disabled_button(f"{icons.FILM} {label}##ink-{doc.uid}", editable, (-1, 0)):
            packwright_mode.add_inker_document(ctx, doc)


def _row(ctx: Any, state: Any, tab: Any, source: Any, editable: bool) -> None:
    from imgui_bundle import imgui

    imgui.push_id(str(source.uid))
    selected = state.selected == source.uid
    if controls.selectable(f"{source.name}##src", selected)[0]:
        state.selected = None if selected else source.uid
    if imgui.is_item_hovered():
        sprite = source.sprite
        imgui.set_tooltip(f"{sprite.width} x {sprite.height}\n{sprite.key}")
    if imgui.begin_popup_context_item("src-menu"):
        widgets.popup_chrome(_imgui=imgui)
        if controls.menu_item_simple("Remove") and editable:
            packwright_mode.remove_source(ctx, source.uid, tab)
        imgui.end_popup()
    if selected and editable:
        name = widgets.input_text("##rename", source.name, max_length=64)
        if name != source.name:
            # Through the mode, not onto the document: the mode is what re-arms
            # the pack, and a name that never reaches the layout is a name the
            # exported sidecar does not carry.
            packwright_mode.rename_source(ctx, tab, source.uid, name)
        if widgets.destructive_button(f"{icons.TRASH} Remove", (-1, 0)):
            packwright_mode.remove_source(ctx, source.uid, tab)
    imgui.pop_id()
