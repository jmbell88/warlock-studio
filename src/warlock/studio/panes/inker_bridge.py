"""The Inker's dialogs, and no panel at all any more.

This *was* the bridge panel: five blocks of buttons in the left column saying
what a painting could become. Those verbs are now rows in ``inker_menu``,
which is where a user coming from Aseprite looks for them and which costs the
canvas nothing -- the panel was 300 px of column for eleven buttons pressed
once a session.

What could not move is what is left: four popups -- resize, filter, sheet
import, colour-mode convert -- and the several hundred lines of machinery
behind them. A popup belongs to the window that began it, so they are drawn by
:func:`popups` from inside the centre pane rather than from a pane of their
own, and **there is no ``draw``**: this module is not in the workspace.

The distinction the *linked* verbs turn on is still worth stating, because the
ops registry encodes it: a linked document writes back into a job's input.png
(with the layered source kept beside it); an unlinked one is a plain file that
has never been part of a job.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import controls, inker_mode, theme, widgets
from ..inker import transform
from ..tokens import sp
from . import inker_colors


def _busy_why(tab: Any) -> str:
    """Why every button on this panel is out, when one of them is.

    ``tab.busy`` is deliberately one question with two answers behind it -- a
    save is encoding off-thread, or playback is running -- and a user reading
    "Saving..." while the clip is looping would go and look for a save. So the
    sentence separates them here, once, and the panel's six buttons share it:
    the ``_VIEWPORT_WHY`` pattern.
    """
    if getattr(tab, "playing", False):
        return "Playback is running. Stop it to edit the document."
    return "This document is being written; the buttons come back when it lands."


def popups(ctx: Any) -> None:
    """Every dialog this module owns, drawn in the caller's window.

    The whole of what is left of a pane. ``inker_menu`` and the context bar
    replaced the five blocks of buttons; the four popups behind them -- resize,
    filter, sheet import, colour-mode convert -- stayed, because an imgui popup
    belongs to the window that began it and the machinery behind these is
    several hundred lines that has nothing to do with where a button sits.

    Called from ``inker_canvas.draw``, which is the window the menu strip is
    drawn in. ``state.pending_dialog`` is how a menu row asks for one: the
    registry names a popup, this opens it, and nothing in ``inker_ops`` has to
    know a window exists.
    """
    state = inker_mode.ensure(ctx)
    tab = state.active
    wanted, state.pending_dialog = state.pending_dialog, ""
    if wanted and tab is not None:
        if wanted == "inker-resize":
            _measure_pixel_grid(ctx, tab)
            imgui.open_popup("inker-resize")
        elif wanted == FILTER_POPUP:
            _open_filter(ctx, tab)
        elif wanted == CONVERT_POPUP:
            open_convert(ctx, tab)
        elif wanted:
            # Not this module's popup -- hand it back for whoever owns it
            # (the canvas's New, the menu strip's layer properties). Rewritten
            # rather than swallowed: a dialog request that silently evaporates
            # is a menu row that does nothing when clicked.
            state.pending_dialog = wanted
    if state.sheet_import is not None and not state.sheet_import_open:
        state.sheet_import_open = True
        imgui.open_popup(SHEET_IMPORT_POPUP)
    _sheet_import_popup(ctx, state)
    if tab is None:
        return
    _resize_popup(ctx, tab)
    _filter_popup(ctx, tab)



def _measure_pixel_grid(ctx: Any, tab: Any) -> None:
    """Measure the document's pixel lattice, once, as the Resize popup opens.

    **Once, and not per frame.** The measurement is a gradient sweep over the
    whole flattened document at every candidate scale; on a 2048-square drawing
    that is a frame-thread stall, and the answer cannot change while a modal
    popup owns the frame. Parked in ``ctx.state.preview`` beside the popup's own
    ``inker_resize:`` entry, keyed the same way.
    """
    key = f"inker_grid:{tab.uid}"
    try:
        found = transform.detect_pixel_grid(tab.doc.flatten(matte=False))
    except ValueError:
        found = {"scale": None}
    ctx.state.preview[key] = found


def _descale_row(ctx: Any, tab: Any) -> bool:
    """The detected-lattice line and its button, or nothing at all.

    Nothing at all is the common case and the important one: an ordinary
    drawing has no lattice, and the popup must then be exactly what it was
    before this existed. Never applied silently -- the same rule the tilesheet
    detector follows at the import doors.
    """
    found = ctx.state.preview.get(f"inker_grid:{tab.uid}") or {}
    scale = found.get("scale")
    if not scale:
        return False
    width, height = transform.descale_size(tab.doc.size, scale, found["phase"])
    if not width or not height:
        return False
    widgets.muted(f"Detected a {scale} px pixel grid - true size {width} x {height}")
    if controls.button("Descale", (sp(180), 0)):
        tab.doc.descale_to_grid(scale, found["phase"])
        tab.view.fitted = False
        return True
    return False


def _resize_popup(ctx: Any, tab: Any) -> None:
    if not imgui.begin_popup("inker-resize"):
        return
    widgets.popup_chrome(_imgui=imgui)
    key = f"inker_resize:{tab.uid}"
    width, height = ctx.state.preview.get(key) or tab.doc.size
    imgui.set_next_item_width(sp(90))
    changed_w, width = controls.input_int("W", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed_h, height = controls.input_int("H", int(height), 0)
    if changed_w or changed_h:
        ctx.state.preview[key] = (max(1, width), max(1, height))
    state = inker_mode.ensure(ctx)
    state.resample = widgets.labeled_combo(
        "Resample",
        state.resample,
        [(k, k) for k in transform.RESAMPLES],
        help_text=(
            "Nearest copies each source pixel whole, which is what pixel art needs "
            "-- a filtered scale of a 32x32 sprite comes back blurred and with "
            "thousands of colours in it. Smooth is right for everything else."
        ),
    )
    anchor = _anchor_grid(ctx, tab)
    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if controls.button("Scale image", (sp(180), 0)):
        tab.doc.scale((max(1, width), max(1, height)), resample=state.resample)
        tab.view.fitted = False
        imgui.close_current_popup()
    # Two different operations that a single "resize" would conflate: one
    # resamples the picture, the other changes how much room it has. The anchor
    # belongs to the second: scaling has nowhere to put slack.
    if controls.button("Resize canvas", (sp(180), 0)):
        tab.doc.resize_canvas((max(1, width), max(1, height)), anchor=anchor)
        tab.view.fitted = False
        imgui.close_current_popup()
    if _descale_row(ctx, tab):
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.end_popup()


# The grid, drawn in reading order. Taken from ``transform.ANCHORS`` rather
# than written again here: the pane decides the layout and the engine owns what
# each name means, which is why that table is written out rather than derived
# from a 3x3 index.
ANCHOR_ROWS = (
    ("top-left", "top", "top-right"),
    ("left", "centre", "right"),
    ("bottom-left", "bottom", "bottom-right"),
)


def _anchor_grid(ctx: Any, tab: Any) -> str:
    """Nine buttons saying where the old image sits in the new canvas.

    Remembered per tab in the preview dictionary beside the width and height,
    so reopening the popup after a mistake offers the same answer rather than
    silently going back to the corner.
    """
    from ..inker import transform as tf

    key = f"inker_anchor:{tab.uid}"
    current = ctx.state.preview.get(key) or "top-left"
    widgets.field_label("anchor")
    for row in ANCHOR_ROWS:
        for name in row:
            selected = name == current
            if selected:
                imgui.push_style_color(
                    imgui.Col_.button.value,
                    imgui.get_style().color_(imgui.Col_.button_active.value),
                )
            if controls.button(f" ##anchor{name}", (sp(28), sp(24))):
                ctx.state.preview[key] = name
                current = name
            if selected:
                imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip(name)
            if name != row[-1]:
                imgui.same_line()
        imgui.new_line()
    return current if current in tf.ANCHORS else "top-left"


# --- filters ----------------------------------------------------------------
#
# A live preview rather than an apply-and-look: every one of these is a value
# nobody can predict, and a filter you have to undo to judge is a filter you
# stop using. The document owns the session (``begin_filter`` takes the copy
# every preview recomputes from) so nothing here holds pixels, and the whole
# thing is one undo step however many times a slider moved.


FILTER_POPUP = "inker-filter"


def _open_filter(ctx: Any, tab: Any) -> None:
    from ..inker import filters

    state = inker_mode.ensure(ctx)
    if tab.busy:
        return
    if not state.filter_name:
        state.filter_name = next(iter(filters.FILTERS))
    if tab.doc.begin_filter() is None:
        ctx.toast("There is nothing to filter.", "warn")
        return
    state.filter_open = True
    imgui.open_popup(FILTER_POPUP)


def _filter_values(state: Any, name: str) -> dict[str, Any]:
    from ..inker import filters

    got = state.filter_params.get(name)
    if got is None:
        got = filters.popup_values(name)
        state.filter_params[name] = got
    return got


# A parameter whose *name* is not what to call it on screen. ``replace colour``
# takes ``old`` and ``new`` because ``from`` is a keyword and cannot be a keyword
# argument -- which is a fact about Python and not something a user should have
# to read the source to translate. The manual says From and To, so the popup
# does too.
_PARAM_LABELS = {"old": "From", "new": "To"}


def _param_label(key: str) -> str:
    return _PARAM_LABELS.get(key, key)


def _filter_control(
    state: Any, values: dict[str, Any], key: str, filter_name: str = ""
) -> None:
    """One parameter row, drawn by the kind the registry says it is.

    Four kinds rather than a slider and a special case, because the FX staples
    brought parameters a slider cannot hold: a colour, an on/off, and a choice
    between two numbers that has nothing in between. Which kind a name is lives
    in ``filters`` beside the filter that declares it -- see ``COLOUR_PARAMS``.

    Every id carries the *parameter* name rather than the label, so what a
    control is called and what it is are free to differ. (The choice combos go
    through ``labeled_combo``, whose id is its label -- which is the same string
    for both of them, neither being relabelled.)
    """
    from ..inker import filters

    label = _param_label(key)
    if key in filters.COLOUR_PARAMS:
        # ``inker_colors``' own conversions rather than a second pair here: the
        # rounding between imgui's floats and the 8-bit tuple the engine writes
        # with is a rule, and two copies of a rule are one disagreement waiting.
        changed, value = controls.color_edit4(
            f"{label}##{key}", inker_colors._vec(tuple(values[key])), inker_colors.FLAGS
        )
        if changed:
            values[key] = inker_colors._to_rgba(value)
        imgui.same_line()
        # The colour a user wants is nearly always the one they are painting
        # with, and picking it twice in two widgets is the friction this button
        # exists to remove.
        if controls.button(f"use FG##fg{key}"):
            values[key] = tuple(state.fg)
        return
    if key in filters.toggles_for(filter_name or state.filter_name):
        # Stored as 0.0/1.0, not as a bool: the registry holds one kind of value
        # and ``apply_named`` passes it straight through.
        changed, on = controls.checkbox(f"{label}##{key}", bool(values[key]))
        if changed:
            values[key] = 1.0 if on else 0.0
        return
    choices = filters.CHOICE_PARAMS.get(key)
    if choices is not None:
        # ``labeled_combo`` and not ``combo``: imgui draws a combo's label to its
        # *right* and the default width is -1, so a named combo puts its name
        # past the content region, where same_line clips rather than wraps and
        # the name is simply not drawn. ``widgets.combo``'s docstring is where
        # that rule is written down.
        picked = widgets.labeled_combo(
            label, str(values[key]), [(str(choice), str(choice)) for choice in choices]
        )
        if picked != str(values[key]):
            values[key] = next(c for c in choices if str(c) == picked)
        return
    low, high = filters.RANGES.get(key, (0.0, 1.0))
    imgui.set_next_item_width(sp(160))
    changed, value = controls.slider_float(f"{label}##{key}", float(values[key]), low, high)
    if changed:
        values[key] = float(value)


def _filter_popup(ctx: Any, tab: Any) -> None:
    from ..inker import filters

    state = inker_mode.ensure(ctx)
    if not imgui.begin_popup(FILTER_POPUP):
        # imgui closes a popup on a click outside, and the user did not answer
        # the question -- so the pixels on screen are a preview nobody
        # approved. Cancel, never commit.
        if state.filter_open:
            state.filter_open = False
            tab.doc.cancel_filter()
        return
    widgets.popup_chrome(_imgui=imgui)

    name = widgets.labeled_combo(
        "Filter", state.filter_name, [(key, key) for key in filters.FILTERS]
    )
    if name != state.filter_name:
        state.filter_name = name
    values = _filter_values(state, state.filter_name)
    for key in filters.FILTERS[state.filter_name][0]:
        _filter_control(state, values, key, state.filter_name)
    if controls.button("Reset##filterreset"):
        # Back to what opening the popup gave, not to the identity defaults --
        # Reset on Invert that unticked all three channels would be a button
        # that turns the filter off.
        state.filter_params[state.filter_name] = filters.popup_values(state.filter_name)

    # Every frame, not only on a change: the combo above can switch filters,
    # and a preview that only ran on a slider move would leave the last
    # filter's pixels under the new filter's controls.
    tab.doc.preview_filter(state.filter_name, **_filter_values(state, state.filter_name))

    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if controls.button("Apply", (sp(90), 0)):
        tab.doc.commit_filter()
        state.filter_open = False
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    _apply_to_range(ctx, tab)
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss -- the trap the params popup in Clay documents.
    if controls.button("Cancel", (sp(90), 0)):
        tab.doc.cancel_filter()
        state.filter_open = False
        imgui.close_current_popup()
    imgui.end_popup()


def _apply_to_range(ctx: Any, tab: Any) -> None:
    """Run the filter over every cel of the timeline's range, in one step.

    **Cancels the preview session first**, which is the whole of what makes
    this safe beside Apply: the session has already written its preview into
    the cel on screen, and running the range filter over that cel would filter
    an already-filtered plane -- the compounding ``preview_filter`` exists to
    avoid, arriving by a different door. Cancelling puts the pixels back, and
    ``filter_range`` then reads every cel including this one exactly once.

    Disabled with no range rather than hidden, the rule the timeline's own menu
    follows: a button that appears and disappears is one the user has to
    rediscover.
    """
    state = inker_mode.ensure(ctx)
    rect = tab.range_sel
    imgui.begin_disabled(tab.busy or rect is None or tab.doc.anim is None)
    if controls.button("Apply to range", (sp(120), 0)):
        values = dict(_filter_values(state, state.filter_name))
        tab.doc.cancel_filter()
        state.filter_open = False
        tab.doc.filter_range(state.filter_name, values, *rect)
        imgui.close_current_popup()
    imgui.end_disabled()
    if rect is None:
        widgets.help_marker(
            "Drag across the timeline to select a range of cels first. Every"
            " distinct cel in it is filtered once, so a linked cel is filtered"
            " once however many frames it appears on."
        )


# --- importing a sprite sheet -----------------------------------------------

SHEET_IMPORT_POPUP = "inker-sheet-import"




def _pair(label: str, value: tuple[int, int], low: int = 0) -> tuple[int, int]:
    """Two small integer fields on one row. -> the pair, floored at ``low``."""
    imgui.set_next_item_width(sp(70))
    _changed_x, x = controls.input_int(f"##{label}x", int(value[0]), 1, 8)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    _changed_y, y = controls.input_int(f"##{label}y", int(value[1]), 1, 8)
    imgui.same_line()
    widgets.muted(label)
    return (max(low, int(x)), max(low, int(y)))


def _sheet_import_popup(ctx: Any, state: Any) -> None:
    from ..inker import sheetin

    if not imgui.begin_popup(SHEET_IMPORT_POPUP):
        # imgui closes a popup on a click outside, and the picture is a
        # megabyte or two: dropping it here is what keeps a cancelled import
        # from pinning the atlas for the rest of the session.
        if state.sheet_import_open:
            state.sheet_import_open = False
            state.sheet_import = None
        return
    widgets.popup_chrome(_imgui=imgui)
    if state.sheet_import is None:
        imgui.end_popup()
        return
    atlas, title = state.sheet_import
    height, width = atlas.shape[:2]
    widgets.muted(f"{title} - {width} x {height}")

    state.sheet_cell = _pair("cell", state.sheet_cell, low=1)
    state.sheet_offset = _pair("offset", state.sheet_offset)
    state.sheet_padding = _pair("padding", state.sheet_padding)
    imgui.set_next_item_width(sp(70))
    _changed, count = controls.input_int("frames (0 = all)", int(state.sheet_count), 1, 8)
    state.sheet_count = max(0, int(count))

    # The count the numbers above actually produce, computed every frame from
    # the same function the import runs -- so what the popup promises and what
    # the import does cannot disagree, and a mistyped cell size says so here
    # rather than in a document twenty frames long.
    try:
        rects = sheetin.grid_rects(
            (int(width), int(height)),
            state.sheet_cell,
            state.sheet_offset,
            state.sheet_padding,
            state.sheet_count or None,
        )
        problem = ""
    except ValueError as exc:
        rects, problem = [], str(exc)
    if problem:
        widgets.text_colored(theme.WARN, problem)
    else:
        widgets.muted(f"{len(rects)} frames")

    imgui.dummy((0, 4))
    imgui.begin_disabled(not rects)
    if controls.button("Import", (sp(90), 0)) and inker_mode.import_sheet(ctx):
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    if controls.button("Cancel##sheetin", (sp(90), 0)):
        state.sheet_import = None
        state.sheet_import_open = False
        imgui.close_current_popup()
    imgui.end_popup()


# --- palette conversion -----------------------------------------------------
#
# The filter popup's mechanism against a different session, and for the same
# reason: nobody can predict what Floyd-Steinberg does to *their* drawing on
# *their* palette, and a conversion you have to undo to judge is one you stop
# trying. The document owns the session, so nothing here holds pixels, and
# committing is the ordinary one-undo ``convert_to_palette``.
#
# Opened and drawn from ``panes/inker_colors`` rather than from ``_canvas_ops``
# below, even though it is written here beside its twin: an imgui popup is
# matched by an id computed off the current id stack, and the colours pane and
# this one are different child windows -- ``open_popup`` here and
# ``begin_popup`` there would never meet. The palette section is where the
# controls belong anyway.

CONVERT_POPUP = "inker-convert"


def _convert_table(state: Any, doc: Any) -> list[tuple[int, int, int, int]]:
    """The table the conversion would use: the document's own, or a built one.

    No third choice and no source selector. A document that *has* a palette is
    being re-dithered onto the palette it has -- offering to replace it here
    would put "change my colours" and "change how my pixels reach them" behind
    one button. A document that has none has nothing else to convert to but its
    own pixels, and the swatch row is a session's favourites rather than a
    statement about this file (see ``inker_colors._indexed``).
    """
    if doc.palette:
        return [tuple(c) for c in doc.palette]
    return doc.built_palette(state.convert_max)


def open_convert(ctx: Any, tab: Any) -> None:
    from ..inker import dither

    state = inker_mode.ensure(ctx)
    if tab.busy:
        return
    if state.convert_method not in dither.METHODS:
        state.convert_method = dither.METHODS[0]
    if not tab.doc.begin_convert():
        ctx.toast("There is nothing to convert.", "warn")
        return
    # After ``begin_convert``, which is what makes the built table read the
    # session's snapshot rather than a preview of itself.
    state.convert_table = _convert_table(state, tab.doc)
    state.convert_uid = tab.uid
    imgui.open_popup(CONVERT_POPUP)


def convert_popup(ctx: Any, tab: Any) -> None:
    """Draw the open session's popup, or settle a session nothing will answer.

    Called unconditionally from ``inker_colors.draw`` -- including with no tab
    at all -- because this is the only per-frame hook the session has, and every
    way it can be stranded is a frame where the popup does not get drawn.

    Two of those ways are handled below and neither may act on ``tab``:

    * ``begin_popup`` says no. The user clicked outside, or the pane stopped
      being submitted (leaving Inker mode), and imgui closed the popup. An
      unanswered question is not a yes, so the session is cancelled.
    * the popup is up but this pane is drawing a *different* document. The user
      switched tabs. The session belongs to the tab it was opened on and nothing
      about the new one has anything to do with it.

    In both, ``end_convert_session`` resolves the owner by uid. Reaching for
    ``tab`` here -- which the filter popup this was cloned from does -- is
    exactly how a tab switch came to restore planes that were never previewed
    while the previewed document kept a dither nobody approved, with no hook
    left to take it back.
    """
    from ..inker import dither

    state = inker_mode.ensure(ctx)
    owner = state.get(state.convert_uid) if state.convert_uid else None
    if not imgui.begin_popup(CONVERT_POPUP):
        inker_mode.end_convert_session(ctx)
        return
    widgets.popup_chrome(_imgui=imgui)
    if owner is None or tab is None or owner.uid != tab.uid:
        inker_mode.end_convert_session(ctx)
        imgui.close_current_popup()
        imgui.end_popup()
        return

    state.convert_method = widgets.labeled_combo(
        "Dither", state.convert_method, [(key, key) for key in dither.METHODS]
    )
    if not tab.doc.palette:
        imgui.set_next_item_width(sp(160))
        changed, value = controls.slider_int("Colours", int(state.convert_max), 2, 64)
        if changed:
            state.convert_max = int(value)
        if imgui.is_item_deactivated_after_edit():
            # On release, not on every frame of the drag: building a table is a
            # pass over every plane in the document followed by a median cut,
            # and a slider dragged across its range would ask for sixty of them.
            # This is the only control that changes what the table would be.
            state.convert_table = _convert_table(state, tab.doc)
    widgets.muted(f"{len(state.convert_table)} colour(s); this frame is previewed")
    widgets.help_marker(
        "The preview shows the current frame. Applying converts the whole "
        "document -- every layer and every frame -- as one undo step, because "
        "the palette it installs constrains every write afterwards."
    )

    # Every frame, not only on a change: the combo and the slider can both move
    # the answer, and a preview that only ran on a change would leave the last
    # method's pixels under the new method's controls.
    tab.doc.preview_convert(state.convert_table, state.convert_method)

    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if controls.button("Apply##convert", (sp(90), 0)):
        table = list(state.convert_table)
        if tab.doc.commit_convert(table, state.convert_method):
            state.palette_slot = 0
            state.palette_slots = []
            state.palette_usage = None
            ctx.toast(f"Converted to {len(table)} colour(s).", "success")
        state.convert_uid = ""
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss.
    if controls.button("Cancel##convert", (sp(90), 0)):
        tab.doc.cancel_convert()
        state.convert_uid = ""
        imgui.close_current_popup()
    imgui.end_popup()
