"""Foreground, background and the swatch row.

Two colours rather than one because the gradient tool needs both ends and X
swapping them is universal muscle memory. The swatches are persisted (unlike
the old editor's fixed palette) because a project has a palette and retyping it
every session is the kind of small friction that makes a tool feel unfinished.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import inker_mode, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_bridge

# One swatch, in *design* pixels -- see ``_swatches``.
SWATCH = 20.0
# imgui's own picker rather than hand-rolled hex and HSV fields. It already has
# both, plus a wheel and an eyedropper, and every one of those is a widget that
# would otherwise have to be written and then kept agreeing with the others
# about rounding. ``display_hex`` puts the hex box on the inline row too, so the
# common case -- reading or typing a colour somebody sent you -- needs no popup
# at all.
FLAGS = (
    imgui.ColorEditFlags_.no_inputs.value
    | imgui.ColorEditFlags_.alpha_bar.value
    | imgui.ColorEditFlags_.display_hex.value
    | imgui.ColorEditFlags_.picker_hue_bar.value
)


def _to_rgba(value: Any) -> tuple[int, int, int, int]:
    """imgui's float colour -> the 8-bit tuple the engine writes with."""
    return (
        int(round(value.x * 255)),
        int(round(value.y * 255)),
        int(round(value.z * 255)),
        int(round(value.w * 255)),
    )


def _vec(colour: tuple[int, int, int, int]) -> Any:
    return imgui.ImVec4(*[c / 255.0 for c in colour])


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    widgets.section("colour")
    # After the heading, never before it: help_button is a same_line, and
    # same_line returns to the previous row unconditionally -- called first it
    # lands on whatever the pane above drew.
    manual_render.help_button(ctx, "inker-colors")

    changed, value = imgui.color_edit4("Foreground", _vec(state.fg), FLAGS)
    if changed:
        state.fg = _to_rgba(value)
    changed, value = imgui.color_edit4("Background", _vec(state.bg), FLAGS)
    if changed:
        state.bg = _to_rgba(value)

    if imgui.button("Swap (X)"):
        state.swap_colours()
    imgui.same_line()
    if imgui.button("+ swatch"):
        state.add_swatch(state.fg)
        inker_mode.persist(ctx)

    imgui.dummy((0, 4))
    _swatches(ctx, state)
    imgui.dummy((0, 4))
    _palette_files(ctx, state)
    imgui.dummy((0, 6))
    _indexed(ctx, state)


# --- indexed colour ---------------------------------------------------------


def _indexed(ctx: Any, state: Any) -> None:
    """The document's own colour table, and what can be done to it.

    Below the swatch row and deliberately separate from it: a swatch is a
    colour you keep reaching for this session, and a palette *slot* is a colour
    this file is made of. The two look alike and behave nothing alike -- adding
    a swatch changes no pixel, and editing a slot repaints every frame.
    """
    tab = inker_mode.active(ctx)
    if tab is None:
        return
    doc = tab.doc
    widgets.section("palette")
    widgets.help_marker(
        "Indexed colour constrains every write to this table: a stroke, a fill "
        "or a filter lands on the nearest colour in it. Alpha is untouched, so "
        "a soft brush still fades -- it just bands. Editing a slot repaints "
        "every pixel painted in it, across every layer and frame, as one undo."
    )
    # ``busy`` for ``inker_bridge._canvas_ops``'s reason: everything below
    # rebinds whole layer planes, and one landing mid-save writes an archive
    # whose parts disagree about the document.
    imgui.begin_disabled(tab.busy)
    if not doc.palette:
        _not_indexed(ctx, state, tab)
    else:
        _slots(ctx, state, tab)
    imgui.end_disabled()
    # Outside the disabled scope, for the reason ``inker_bridge._canvas_ops``
    # states: a popup is its own window, and imgui's disabled state is not meant
    # to span a Begin/End pair. It carries its own ``busy`` gate on Apply.
    inker_bridge.convert_popup(ctx, tab)


def _not_indexed(ctx: Any, state: Any, tab: Any) -> None:
    widgets.muted("Not indexed - any colour is allowed.")
    if widgets.disabled_button(
        "Index to the swatches",
        bool(state.swatches),
        reason="The swatch row above is empty.",
    ):
        inker_mode.index_to(ctx, tab, list(state.swatches))
    if imgui.button("Index to a palette file..."):
        inker_mode.import_document_palette(ctx)
    if imgui.button("Palette from an image..."):
        inker_mode.palette_from_image(ctx)
    if imgui.button("Convert..."):
        inker_bridge.open_convert(ctx, tab)
    widgets.help_marker(
        "Convert builds a palette out of this drawing's own colours and shows "
        "the result before you commit to it -- including the dither, which is "
        "the setting nobody can predict on their own picture."
    )


def _slots(ctx: Any, state: Any, tab: Any) -> None:
    doc = tab.doc
    palette = list(doc.palette)
    state.clamp_slots(len(palette))
    counts = _usage(state, doc, len(palette))

    avail = imgui.get_content_region_avail().x
    gap = imgui.get_style().item_spacing.x
    side = sp(SWATCH)
    per_row = max(1, int((avail + gap) // (side + gap)))
    for index, colour in enumerate(palette):
        imgui.push_id(f"slot{index}")
        # The anchor is outlined in the accent and the rest of a multi-slot
        # selection in the text colour: two of these are routinely the same
        # colour at a glance, and the single-slot controls below act on the
        # anchor alone while Sort and Ramp act on the whole selection.
        chosen = index == state.palette_slot
        member = index in state.palette_slots and not chosen
        if chosen or member:
            tint = theme.ACCENT if chosen else theme.TEXT
            imgui.push_style_color(imgui.Col_.border.value, imgui.ImVec4(*theme.rgba(tint)))
            imgui.push_style_var(imgui.StyleVar_.frame_border_size.value, sp(2.0))
        if imgui.color_button("##slot", _vec(colour), 0, (side, side)):
            io = imgui.get_io()
            state.select_slot(index, ctrl=io.key_ctrl, shift=io.key_shift)
            if not (io.key_ctrl or io.key_shift):
                # Painting with it as well as selecting it: the reason to click
                # a palette slot is almost always to use it. Only on a plain
                # click -- a Ctrl+click that *deselects* a slot must not leave
                # the brush loaded with it.
                state.fg = colour
        if chosen or member:
            imgui.pop_style_var()
            imgui.pop_style_color()
        if imgui.is_item_hovered():
            used = "" if counts is None else f"  -  {counts[index]} px"
            imgui.set_tooltip(f"{colour}{used}\nCtrl-click to add, Shift-click for a range")
        imgui.pop_id()
        if index % per_row != per_row - 1:
            imgui.same_line()
    imgui.new_line()

    slot = state.palette_slot
    changed, value = imgui.color_edit4("Slot", _vec(palette[slot]), FLAGS)
    if changed and doc.recolour_slot(slot, _to_rgba(value)):
        state.palette_usage = None
    if imgui.button("+ from colour") and doc.add_slot(state.fg):
        state.palette_slot = len(doc.palette) - 1
        state.palette_usage = None
    imgui.same_line()
    if widgets.disabled_button(
        "Remove",
        len(palette) > 1,
        reason="An indexed document keeps at least one colour.",
        tooltip="Pixels painted in it merge into the nearest remaining colour.",
    ) and doc.remove_slot(slot):
        state.palette_usage = None
    if widgets.disabled_button("<", slot > 0, reason="Already first."):
        doc.move_slot(slot, slot - 1)
        state.palette_slot = slot - 1
    imgui.same_line()
    if widgets.disabled_button(">", slot < len(palette) - 1, reason="Already last."):
        doc.move_slot(slot, slot + 1)
        state.palette_slot = slot + 1
    imgui.same_line()
    widgets.muted(f"{slot + 1} of {len(palette)}")

    _sort_and_ramp(ctx, state, doc, counts)

    if imgui.small_button("Count usage"):
        state.palette_usage = (doc.rev, doc.palette_usage())
    imgui.same_line()
    if imgui.small_button("Export palette"):
        inker_mode.export_document_palette(ctx)
    imgui.same_line()
    if imgui.small_button("Not indexed"):
        inker_mode.index_to(ctx, tab, None)
    if imgui.small_button("Re-convert..."):
        inker_bridge.open_convert(ctx, tab)
    imgui.same_line()
    widgets.muted("try a dither")


#: The sort keys, labelled. Two-way against ``inker.PALETTE_SORT_KEYS``: a key
#: the pane offers and the engine does not fails on the first click, and one the
#: engine has and the pane does not is a feature nobody can reach.
SORT_LABELS: tuple[tuple[str, str], ...] = (
    ("hue", "Hue"),
    ("saturation", "Saturation"),
    ("luma", "Brightness"),
    ("red", "Red"),
    ("green", "Green"),
    ("blue", "Blue"),
    ("alpha", "Alpha"),
    ("usage", "Usage"),
)


def _sort_and_ramp(ctx: Any, state: Any, doc: Any, counts: list[int] | None) -> None:
    """Reorder the table, and fill the gap between two slots.

    Neither pushes an undo step and neither moves a pixel -- order is
    presentation in an indexed document, and a new swatch is a colour you *may*
    paint with. The engine states the rule; this is only where it is reached.
    """
    selection = list(state.palette_slots)
    widgets.field_label("sort")
    state.palette_sort = widgets.combo(
        "##palsort", state.palette_sort, list(SORT_LABELS), sp(110)
    )
    imgui.same_line()
    if imgui.small_button("Sort"):
        if state.palette_sort == "usage" and counts is None:
            # Counting is a walk over every pixel of every cel, so it is asked
            # for rather than kept live -- and a sort by a figure nobody has
            # taken yet has to take it, once, rather than sort by zeros.
            counts = doc.palette_usage()
            state.palette_usage = (doc.rev, counts)
        if doc.sort_palette(
            state.palette_sort,
            indices=selection or None,
            counts=counts,
            descending=state.palette_sort_desc,
        ):
            state.palette_usage = None
    imgui.same_line()
    changed, value = imgui.checkbox("Down##palsortdir", state.palette_sort_desc)
    if changed:
        state.palette_sort_desc = value
    if selection:
        widgets.muted(f"{len(selection)} slot(s) selected; sorting them in place")

    widgets.field_label("ramp")
    imgui.set_next_item_width(sp(70))
    changed, value = imgui.slider_int("##palramp", int(state.palette_ramp), 1, 16)
    if changed:
        state.palette_ramp = int(value)
    imgui.same_line()
    if widgets.disabled_button(
        "Insert",
        len(selection) >= 2,
        reason="Ctrl-click or Shift-click two slots to ramp between them.",
        tooltip="Interpolated colours between the two, inserted between them.",
    ) and not doc.insert_ramp(min(selection), max(selection), state.palette_ramp):
        ctx.toast("That ramp is already in the palette.")


def _usage(state: Any, doc: Any, slots: int) -> list[int] | None:
    """The last usage count the user asked for, or None.

    Asked for rather than recomputed, and dropped the moment the document
    moves: counting walks every pixel of every cel, so a live figure would cost
    a whole clip's worth of scanning per frame to keep a number that changes on
    one dab. A stale count is worse than no count -- "0 px, safe to delete" is
    the one thing it must never say wrongly -- so it is discarded rather than
    shown greyed.
    """
    cached = state.palette_usage
    if cached is None or cached[0] != doc.rev or len(cached[1]) != slots:
        state.palette_usage = None
        return None
    return cached[1]


def _palette_files(ctx: Any, state: Any) -> None:
    """Import and export the swatch row as a ``.gpl``.

    Both go through ``ctx.submit``: a native picker is modal to the OS and
    blocks until it is dismissed, so neither may touch the frame thread. The
    export's *bytes* are built here, before the submit, for the reason every
    save in this app is -- serialising after an unbounded modal would write
    whatever the user changed while it was open.
    """
    if imgui.small_button("Import palette"):
        inker_mode.import_palette(ctx)
    imgui.same_line()
    if widgets.disabled_button("Export palette", bool(state.swatches)):
        inker_mode.export_palette(ctx)
    widgets.help_marker(
        "GIMP .gpl, which Krita, Aseprite and Inkscape all read, or JASC .pal. "
        "The format follows the suffix you save under; neither has an alpha "
        "channel, so exported swatches are written opaque. An import adds to "
        "the row rather than replacing it."
    )


def _swatches(ctx: Any, state: Any) -> None:
    avail = imgui.get_content_region_avail().x
    # The gap between two swatches is the style's, not 6: a row of n costs
    # n * SWATCH + (n - 1) * spacing, and pricing the gap two pixels under the
    # real one bought a swatch that did not fit and was clipped at the edge.
    gap = imgui.get_style().item_spacing.x
    # Through sp() (K97): a swatch that stayed 20 physical pixels while the
    # panel around it grew with the monitor was a grid of dots on a 4K display.
    side = sp(SWATCH)
    per_row = max(1, int((avail + gap) // (side + gap)))
    for index, colour in enumerate(list(state.swatches)):
        imgui.push_id(f"swatch{index}")
        if imgui.color_button("##swatch", _vec(colour), 0, (side, side)):
            state.fg = colour
        # Right-click removes: a swatch row with no way to prune it fills up
        # with mistakes and stops being useful within a session.
        if imgui.is_item_clicked(1):
            state.swatches.remove(colour)
            inker_mode.persist(ctx)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{colour}  -  right-click to remove")
        imgui.pop_id()
        if index % per_row != per_row - 1:
            imgui.same_line()
    imgui.new_line()
