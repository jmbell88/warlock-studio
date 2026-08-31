"""Foreground, background and the swatch row.

Two colours rather than one because the gradient tool needs both ends and X
swapping them is universal muscle memory. The swatches are persisted (unlike
the old editor's fixed palette) because a project has a palette and retyping it
every session is the kind of small friction that makes a tool feel unfinished.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import anchors, controls, inker_mode, theme, widgets
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


#: The least this panel may be squeezed to, in design px, before the pane
#: above it stops taking room: the heading, the two colour chips, the Swap row
#: and one row of swatches. The tools pane can grow to fit its toolbox, and
#: this is the number that stops it doing so by eating the colours entirely.
PANEL_FLOOR = 210.0


def draw(ctx: Any) -> None:
    anchors.mark_window("inker/colors")
    state = inker_mode.ensure(ctx)
    widgets.section("Colour")
    # After the heading, never before it: help_button is a same_line, and
    # same_line returns to the previous row unconditionally -- called first it
    # lands on whatever the pane above drew.
    manual_render.help_button(ctx, "inker-colors")

    changed, value = controls.color_edit4("Foreground", _vec(state.fg), FLAGS)
    if changed:
        state.set_fg(_to_rgba(value))
    changed, value = controls.color_edit4("Background", _vec(state.bg), FLAGS)
    if changed:
        state.bg = _to_rgba(value)

    if controls.button("Swap (X)"):
        state.swap_colours()
    imgui.same_line()
    if controls.button("+ swatch"):
        state.add_swatch(state.fg)
        inker_mode.persist(ctx)

    imgui.dummy((0, 4))
    _swatches(ctx, state)
    _harmonies(ctx, state)
    imgui.dummy((0, 4))
    _palette_files(ctx, state)
    imgui.dummy((0, 6))
    _indexed(ctx, state)
    # Unconditionally, and at *this* level rather than inside ``_indexed``: a
    # popup is matched by an id computed off the id stack that opened it, which
    # is this one, and a conversion session's only per-frame hook is this call.
    # ``_indexed`` returns early when there is no active tab -- and "there is no
    # active tab any more" is one of the ways a session gets stranded, so it has
    # to be a frame this still runs on. It takes the tab or None and settles the
    # session against whichever document actually owns it.
    inker_bridge.convert_popup(ctx, inker_mode.active(ctx))


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
    # **A header, and closed until this document has a table.** As a plain
    # section this block ran off the bottom of the pane: measured at the app's
    # own default 1600x950, the Colour pane is 470 px over 516 px of content,
    # so the palette-source row sat below the fold and imgui clipped it away
    # entirely -- and the Picker under it is 37 px over its own allotment, so
    # there is no height in the column to move across. Collapsed it costs one
    # row. Open once the document *is* indexed, because the slots are then the
    # document's own storage rather than four ways to make one, and
    # ``persist_key`` keeps whichever the user settles on.
    if not widgets.header(
        "Palette", default_open=bool(doc.palette), persist_key="inker/palette"
    ):
        return
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
    _mode_row(ctx, tab)
    if not doc.palette:
        _not_indexed(ctx, state, tab)
    else:
        _slots(ctx, state, tab)
    imgui.end_disabled()


def _mode_row(ctx: Any, tab: Any) -> None:
    """The document's colour mode, as three buttons with the current one out.

    Buttons rather than a combo, and the current mode *disabled* rather than
    highlighted: each of these is a whole-document conversion and one undo step,
    so the control that says "you are already here" has to be the one that
    cannot be clicked. The disabled reason is what a combo could not say.
    """
    doc = tab.doc
    widgets.muted("Mode")
    imgui.same_line()
    for mode in inker_mode.COLOR_MODES:
        label = inker_mode.COLOR_MODE_LABELS[mode]
        if widgets.disabled_button(
            f"{label}##mode{mode}",
            doc.color_mode != mode,
            reason="Already this mode.",
            tooltip=_MODE_HELP[mode],
        ):
            if mode == "indexed":
                # Through the Convert popup rather than straight into the
                # conversion: entering indexed mode is a dither like every
                # other conversion in the app, and this was the only one that
                # took ``"nearest"`` with nothing on screen to say so.
                from . import inker_bridge

                ctx.state.inker.pending_dialog = inker_bridge.CONVERT_MODE_POPUP
            else:
                inker_mode.set_color_mode(ctx, tab, mode)
        if mode != inker_mode.COLOR_MODES[-1]:
            imgui.same_line()


#: What each mode does, said on the button that enters it. Long enough to carry
#: the one consequence a user cannot guess: that indexed mode is destructive to
#: colours off the table, and that leaving it is not.
_MODE_HELP = {
    "rgb": (
        "Any colour, no constraint. Leaving indexed or grayscale mode repaints "
        "nothing -- the pixels you have are the pixels you keep."
    ),
    "indexed": (
        "Every pixel becomes a numbered slot in the palette. Editing a slot "
        "repaints its pixels across every layer and frame instantly, and two "
        "slots holding the same colour stay separate. Colours off the table are "
        "snapped onto it, once, as one undo step."
    ),
    "grayscale": (
        "Every write lands on a grey. The colour that is there now is flattened "
        "to its brightness -- one undo step -- and there is nothing to restore "
        "it from afterwards."
    ),
}


def _not_indexed(ctx: Any, state: Any, tab: Any) -> None:
    widgets.muted_wrapped("Not indexed - any colour is allowed.")
    if widgets.disabled_button(
        "Index to the swatches",
        bool(state.swatches),
        reason="The swatch row above is empty.",
    ):
        inker_mode.index_to(ctx, tab, list(state.swatches))
    # **One row, three words.** As four stacked sentences this block ran past
    # the foot of the pane -- measured at 1600x950, the content region ended
    # part-way through the button above, and imgui clipped these three away
    # entirely, so the only way to reach them was to scroll a pane most people
    # would not guess scrolls. Widening the pane is not the fix either: the
    # Picker below it is over its own allotment already. The sentence each one
    # used to be is its tooltip now; the label only has to name the source.
    if controls.button("File...", tooltip="Index to a palette file on disk"):
        inker_mode.import_document_palette(ctx)
    imgui.same_line()
    if controls.button("Image...", tooltip="Build a palette out of an image"):
        inker_mode.palette_from_image(ctx)
    imgui.same_line()
    if controls.button(
        "Convert...", tooltip="Build a palette out of this drawing's own colours"
    ):
        inker_bridge.open_convert(ctx, tab)
    widgets.help_marker(
        "Convert builds a palette out of this drawing's own colours and shows "
        "the result before you commit to it -- including the dither, which is "
        "the setting nobody can predict on their own picture."
    )


def _hole_marker(at: Any, side: float) -> None:
    """A small notch on the transparent slot's top-left corner."""
    draw = imgui.get_window_draw_list()
    size = max(sp(4.0), side * 0.3)
    draw.add_triangle_filled(
        imgui.ImVec2(at.x, at.y),
        imgui.ImVec2(at.x + size, at.y),
        imgui.ImVec2(at.x, at.y + size),
        imgui.get_color_u32(imgui.ImVec4(*theme.rgba(theme.ACCENT))),
    )


def _slots(ctx: Any, state: Any, tab: Any) -> None:
    doc = tab.doc
    palette = list(doc.palette)
    state.clamp_slots(len(palette))
    counts = _usage(state, tab, len(palette))
    # ``is_indexed`` and not ``bool(palette)``: a palette-constrained RGB
    # document has a table and no transparent index, and marking a slot on one
    # would be marking a meaning it does not have.
    hole = doc.transparent_index if doc.is_indexed else -1

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
                #
                # The *slot* rides along, which is what makes painting with the
                # second of two identical swatches land in the second one.
                state.set_fg(colour, index)
        if chosen or member:
            imgui.pop_style_var()
            imgui.pop_style_color()
        if hole == index:
            # A checker corner rather than a border: the two borders above are
            # already spent on the selection, and the thing this marks is not a
            # selection state -- it is what the slot *means*. Drawn after the
            # button so it sits on top of the swatch's own fill.
            _hole_marker(imgui.get_item_rect_min(), side)
        if imgui.is_item_hovered():
            used = "" if counts is None else f"  -  {counts[index]} px"
            note = "\nThis slot is transparent." if hole == index else ""
            imgui.set_tooltip(
                f"{colour}{used}{note}\nCtrl-click to add, Shift-click for a range"
            )
        imgui.pop_id()
        if index % per_row != per_row - 1:
            imgui.same_line()
    imgui.new_line()

    slot = state.palette_slot
    changed, value = controls.color_edit4("Slot", _vec(palette[slot]), FLAGS)
    if changed and doc.recolour_slot(slot, _to_rgba(value)):
        state.palette_usage = None
    if controls.button("+ from colour") and doc.add_slot(state.fg):
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

    if doc.is_indexed and widgets.disabled_button(
        "Make transparent",
        slot != hole,
        reason="This slot is already the transparent one.",
        tooltip=(
            "Pixels in this slot become holes and the old transparent slot goes "
            "solid. Nothing is repainted -- only what the numbers mean changes."
        ),
    ):
        inker_mode.set_transparent_slot(ctx, tab, slot)

    _sort_and_ramp(ctx, state, tab, counts)

    if controls.small_button("Count usage"):
        state.palette_usage = (tab.uid, doc.rev, doc.palette_usage())
    imgui.same_line()
    if controls.small_button("Export palette"):
        inker_mode.export_document_palette(ctx)
    imgui.same_line()
    if controls.small_button("Not indexed"):
        inker_mode.index_to(ctx, tab, None)
    # A row of its own rather than a fourth on the one above: the Colour pane
    # is 300 design px wide and ``same_line`` past the content region draws a
    # control nobody can reach, which is exactly how the palette-source row
    # below was lost once already.
    if controls.small_button(
        "Export image",
        tooltip=(
            "The palette as a PNG swatch strip -- the other half of "
            "Image..., and how it reaches a tool that reads no palette file."
        ),
    ):
        inker_mode.export_palette_image(ctx)
    if controls.small_button("Re-convert..."):
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


def _sort_and_ramp(ctx: Any, state: Any, tab: Any, counts: list[int] | None) -> None:
    """Reorder the table, and fill the gap between two slots.

    Neither pushes an undo step and neither moves a pixel -- order is
    presentation in an indexed document, and a new swatch is a colour you *may*
    paint with. The engine states the rule; this is only where it is reached.
    """
    doc = tab.doc
    selection = list(state.palette_slots)
    widgets.field_label("sort")
    state.palette_sort = widgets.combo(
        "##palsort", state.palette_sort, list(SORT_LABELS), sp(110)
    )
    imgui.same_line()
    if controls.small_button("Sort"):
        if state.palette_sort == "usage" and counts is None:
            # Counting is a walk over every pixel of every cel, so it is asked
            # for rather than kept live -- and a sort by a figure nobody has
            # taken yet has to take it, once, rather than sort by zeros.
            counts = doc.palette_usage()
            state.palette_usage = (tab.uid, doc.rev, counts)
        if doc.sort_palette(
            state.palette_sort,
            indices=selection or None,
            counts=counts,
            descending=state.palette_sort_desc,
        ):
            state.palette_usage = None
    imgui.same_line()
    changed, value = controls.checkbox("Down##palsortdir", state.palette_sort_desc)
    if changed:
        state.palette_sort_desc = value
    if selection:
        widgets.muted_wrapped(f"{len(selection)} slot(s) selected; sorting them in place")

    widgets.field_label("ramp")
    imgui.set_next_item_width(sp(70))
    changed, value = controls.slider_int("##palramp", int(state.palette_ramp), 1, 16)
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


def _usage(state: Any, tab: Any, slots: int) -> list[int] | None:
    """The last usage count the user asked for, or None.

    Asked for rather than recomputed, and dropped the moment the document
    moves: counting walks every pixel of every cel, so a live figure would cost
    a whole clip's worth of scanning per frame to keep a number that changes on
    one dab. A stale count is worse than no count -- "0 px, safe to delete" is
    the one thing it must never say wrongly -- so it is discarded rather than
    shown greyed.
    """
    cached = state.palette_usage
    # The *tab* is part of the key, not only the revision. Two open documents
    # sitting at the same ``rev`` with palettes of the same length answered each
    # other's counts -- and this is the one number whose docstring says it must
    # never be wrong, because "0 px, safe to delete" is a thing a user acts on.
    if (
        cached is None
        or cached[0] != tab.uid
        or cached[1] != tab.doc.rev
        or len(cached[2]) != slots
    ):
        state.palette_usage = None
        return None
    return cached[2]


def _palette_files(ctx: Any, state: Any) -> None:
    """Import and export the swatch row as a ``.gpl``.

    Both go through ``ctx.submit``: a native picker is modal to the OS and
    blocks until it is dismissed, so neither may touch the frame thread. The
    export's *bytes* are built here, before the submit, for the reason every
    save in this app is -- serialising after an unbounded modal would write
    whatever the user changed while it was open.
    """
    if controls.small_button("Import swatches"):
        inker_mode.import_palette(ctx)
    imgui.same_line()
    if widgets.disabled_button(
        "Export swatches",
        bool(state.swatches),
        reason="The swatch row is empty.",
    ):
        inker_mode.export_palette(ctx)
    widgets.help_marker(
        "GIMP .gpl, which Krita, Aseprite and Inkscape all read, JASC .pal, "
        "Lospec .hex or Paint.NET .txt. The format follows the suffix you save "
        "under; only .txt has an alpha channel, so swatches exported to the "
        "other three are written opaque. An import adds to the row rather than "
        "replacing it."
    )
    _palette_folder(ctx, state)


def _palette_folder(ctx: Any, state: Any) -> None:
    """The palette *folder*: every palette file the user has dropped in one place.

    The one palette door in this workspace that is not an OS file dialog, and
    the reason to have it: a picker starts wherever it was last, so the palettes
    a user keeps are three navigations away every single time. This is the same
    zero-registry convention the rest of the app already uses -- a palette is a
    file in ``config.palette_dir``, there is nothing to install and nothing to
    register -- drawn where the Inker actually needs it.

    ``service.palettes`` is called from **here** and not from
    ``studio/inker/``: the headless package may not import the service layer.
    The listing goes through ``inspector.palette_names`` rather than calling
    ``available`` directly, because that function owns the once-per-directory-
    version remembering and a second copy of it would be a second thing to
    invalidate when a file lands.
    """
    from ...service import palettes as svc_palettes
    from . import inspector

    names = inspector.palette_names(ctx)
    imgui.dummy((0, 2))
    if not names:
        # A muted line and not a warning: the folder is opt-in, and having no
        # palettes in it is the ordinary state of a fresh install.
        widgets.muted_wrapped(f"No palette files yet. {svc_palettes.SUFFIX_HELP}")
        return
    if state.palette_pick not in names:
        state.palette_pick = names[0]
    widgets.field_label("folder")
    state.palette_pick = widgets.combo(
        "##palfolder",
        state.palette_pick,
        [(name, name) for name in names],
        sp(150),
        tooltip=svc_palettes.SUFFIX_HELP,
    )
    imgui.same_line()
    if controls.button(
        "Load",
        tooltip="Add this palette's colours to the swatch row above.",
    ):
        _load_named_palette(ctx, state.palette_pick)


def _load_named_palette(ctx: Any, name: str) -> None:
    """Read one palette out of the folder and add it to the swatch row.

    Off the frame thread under ``import_palette``'s own key, which is what
    makes the landing free: this ends up in exactly the same place a file
    dialog's import does -- appended to the row, with the count said out loud
    -- and a second landing branch that did the same thing differently is how
    two doors to one act drift apart. Sharing the key also means the two cannot
    both be in flight, which is correct: they are the same act.

    A file that is not a palette, or one deleted between the listing and the
    click, comes back as ``service.errors.Invalid`` -- a ``ServiceError``, so
    the task runner surfaces its message rather than "something went wrong".
    """
    from ...service import palettes as svc_palettes

    config = ctx.svc.config

    def run() -> list[tuple[int, int, int, int]]:
        _name, colours, _digest = svc_palettes.load(config, name)
        # Opaque: three of the four formats have no alpha at all, and the
        # service's own reader answers in RGB triples whichever it read.
        return [(r, g, b, 255) for r, g, b in colours]

    ctx.submit("inker-palette", run)


def _harmonies(ctx: Any, state: Any) -> None:
    """The Shades strip, and the harmonies of the colour in hand (6.6).

    Both are **derived from the foreground colour**, which is what keeps them
    out of the document: there is nothing to store, nothing to undo, and
    nothing to get out of step with the swatch they came from. A click takes
    one as the foreground; a right-click keeps it as a swatch, which is the
    gesture the swatch row above already uses.
    """
    from ..inker import indexed as ix

    imgui.dummy((0, 6))
    if not widgets.header("Shades", default_open=False, persist_key="inker/shades"):
        return
    side = sp(SWATCH)
    for index, colour in enumerate(ix.shades(state.fg, 7)):
        imgui.push_id(f"shade{index}")
        if imgui.color_button("##shade", _vec(colour), 0, (side, side)):
            state.set_fg(colour)
        if imgui.is_item_clicked(1):
            state.add_swatch(colour)
            inker_mode.persist(ctx)
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"{colour}  -  right-click to keep it as a swatch")
        imgui.pop_id()
        imgui.same_line()
    imgui.new_line()

    state.harmony = widgets.labeled_combo(
        "Harmony",
        state.harmony if state.harmony in ix.HARMONIES else "complement",
        [(key, key.title()) for key in ix.HARMONIES],
        help_text=(
            "Hue rotation only: the saturation and lightness are the ones you "
            "chose, because a harmony that changed them would be a palette "
            "generator rather than an answer to what goes with this colour."
        ),
    )
    for index, colour in enumerate(ix.harmony(state.fg, state.harmony)):
        imgui.push_id(f"harm{index}")
        if imgui.color_button("##harm", _vec(colour), 0, (side, side)):
            state.set_fg(colour)
        if imgui.is_item_clicked(1):
            state.add_swatch(colour)
            inker_mode.persist(ctx)
        imgui.pop_id()
        imgui.same_line()
    imgui.new_line()


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
            state.set_fg(colour)
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
