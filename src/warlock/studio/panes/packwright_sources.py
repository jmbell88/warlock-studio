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
        widgets.muted("Start or open an atlas first.")
        return

    editable = not tab.busy
    if widgets.disabled_button(f"{icons.PLUS} Add an image...", editable, (-1, 0)):
        packwright_mode.ask_add_sources(ctx)

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
