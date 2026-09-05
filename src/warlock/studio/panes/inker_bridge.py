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

from .. import controls, icons, inker_mode, theme, tokens, widgets
from ..inker import transform
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_colors
from . import inker_flourish as inker_flourish_pane


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
    if wanted and tab is None:
        # **Handed back, not dropped.** Taken-and-cleared above, so a request
        # arriving on a frame with no active tab used to evaporate here -- which
        # is the failure the comment below says this branch exists to prevent,
        # by the one route it did not cover. A tab is a frame away (an open, a
        # recovery offer), and the request is answered then.
        state.pending_dialog = wanted
    elif wanted and tab is not None:
        if wanted == "inker-scale":
            # Measured once, as the dialog opens: it is a whole-document scan
            # and the frame loop must never carry one per frame.
            _measure_pixel_grid(ctx, tab)
            imgui.open_popup(SCALE_DIALOG)
        elif wanted == "inker-resize":
            imgui.open_popup(CANVAS_DIALOG)
        elif wanted == FILTER_POPUP:
            _open_filter(ctx, tab)
        elif wanted == INPAINT_POPUP:
            _open_inpaint(ctx, tab)
        elif wanted == CONVERT_POPUP:
            open_convert(ctx, tab)
        elif wanted == CONVERT_MODE_POPUP:
            open_convert(ctx, tab, to_mode="indexed")
        elif wanted == inker_flourish_pane.FLOURISH_POPUP:
            inker_flourish_pane.open_popup(ctx, tab)
        elif wanted == inker_flourish_pane.SNIPPET_POPUP:
            inker_flourish_pane.open_snippet_popup(ctx, tab)
        elif wanted == inker_flourish_pane.TEXTURE_POPUP:
            inker_flourish_pane.open_texture_popup(ctx, tab)
        elif wanted == inker_flourish_pane.RESTYLE_POPUP:
            inker_flourish_pane.open_restyle_popup(ctx, tab)
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
    # ``opening`` is threaded from the dispatcher rather than sniffed with
    # ``is_popup_open``: ``popover_enter`` needs *the frame it appeared on*,
    # and this is the only place that knows which one that was.
    _scale_dialog(ctx, tab, opening=(wanted == "inker-scale"))
    _canvas_dialog(ctx, tab, opening=(wanted == "inker-resize"))
    _filter_popup(ctx, tab)
    _inpaint_popup(ctx, tab)
    inker_flourish_pane.popup(ctx, tab)
    inker_flourish_pane.snippet_popup(ctx, tab)
    inker_flourish_pane.texture_popup(ctx, tab)
    inker_flourish_pane.restyle_popup(ctx, tab)
    poll_inpaint(ctx)


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


def _descale_row(ctx: Any, tab: Any, *, refused: bool = False) -> bool:
    """The detected-lattice line and its button, or nothing at all.

    Nothing at all is the common case and the important one: an ordinary
    drawing has no lattice, and the dialog must then be exactly what it was
    before this existed. Never applied silently -- the same rule the tilesheet
    detector follows at the import doors.

    Lives on **Image size** rather than Canvas size: undoing an upscale is a
    resampling question, and an anchor has nothing to say about it.

    ``refused`` is the tilemap case. ``descale_to_grid`` refuses one by name
    exactly as ``scale`` does, so the button says why instead of raising out of
    the frame loop.
    """
    found = ctx.state.preview.get(f"inker_grid:{tab.uid}") or {}
    scale = found.get("scale")
    if not scale:
        return False
    width, height = transform.descale_size(tab.doc.size, scale, found["phase"])
    if not width or not height:
        return False
    widgets.divider()
    widgets.muted(f"Detected a {scale} px pixel grid - true size {width} x {height}")
    if widgets.disabled_button(
        "Descale",
        not refused,
        (sp(180), 0),
        reason=_NO_TILEMAP_SCALE,
        tooltip=(
            "Undo an upscale: take one pixel per detected cell, rather than "
            "resampling to that size."
        ),
    ):
        tab.doc.descale_to_grid(scale, found["phase"])
        tab.view.fitted = False
        return True
    return False


#: The two dialogs' imgui ids, which are also their **titles**:
#: ``begin_popup_modal`` draws a title bar and "inker-resize" is not a title.
#: The request keys the menu writes into ``pending_dialog`` stay what they were
#: -- ``CONVERT_POPUP``'s precedent, and ``tests/inker/test_inker_ops.py`` pins
#: ``inker-resize`` as the id one menu row asks for.
SCALE_DIALOG = "Image size"
CANVAS_DIALOG = "Canvas size"

#: The narrowest either dialog may be. ``always_auto_resize`` otherwise settles
#: on whichever line happens to be widest, which makes the modal's width a
#: function of the document's current pixel count.
DIALOG_W = 360.0

#: The canvas preview's side, in design pixels.
PREVIEW_BOX = 96.0

#: Scaling a tilemap would have to rescale the tileset with it, which is not
#: modelled -- ``Document.scale`` refuses by name. Said on the button rather
#: than raised out of the frame loop, which is what used to happen.
_NO_TILEMAP_SCALE = (
    "A document with a tilemap layer cannot be scaled: the tileset would have "
    "to be re-cut, which is a different operation. Canvas size still works."
)

#: Where each anchor cell's arrow points, by the unit vector ``anchor_cell``
#: answers with. ``(0, 0)`` is the anchor's own cell and holds the picture.
_ANCHOR_ARROWS = {
    (0, 0): icons.IMAGE,
    (1, 0): icons.ARROW_RIGHT,
    (-1, 0): icons.ARROW_LEFT,
    (0, 1): icons.ARROW_DOWN,
    (0, -1): icons.ARROW_UP,
    (1, 1): icons.ARROW_DOWN_RIGHT,
    (-1, 1): icons.ARROW_DOWN_LEFT,
    (1, -1): icons.ARROW_UP_RIGHT,
    (-1, -1): icons.ARROW_UP_LEFT,
}


def _begin_dialog(name: str, appearing: bool) -> tuple[bool, float]:
    """Open one of this module's modals with the house's surface treatment.

    ``plotter_canvas.setup_popup``'s twelve lines, factored so the two dialogs
    cannot drift apart -- ``popup_chrome``'s own argument ("stops a newly added
    popup choosing its own depth") one level up.

    Depth is ``overlay``, the ``ConfirmQueue``'s, not ``popup_chrome``'s raised:
    a modal is the surface that stops the app, and it should sit at the height
    of the thing that stops the app.
    """
    centre = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
    alpha, rise = widgets.popover_enter(f"inker/{name}", appearing)
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    opened, _ = imgui.begin_popup_modal(
        name, None, imgui.WindowFlags_.always_auto_resize.value
    )
    widgets.pop_surface_rounding()
    if not opened:
        imgui.pop_style_var()
        return False, 0.0
    widgets.window_shadow("overlay", radius=radius)
    if frosted:
        widgets.window_backdrop(radius=radius)
    if rise > 0.0:
        imgui.dummy((0, rise))
    imgui.dummy((sp(DIALOG_W), 0))
    return True, rise


def _end_dialog() -> None:
    imgui.end_popup()
    imgui.pop_style_var()


def _wh_row(
    prefix: str, value: tuple[float, float], *, integer: bool, step: float
) -> tuple[str, tuple[float, float]]:
    """Width and Height side by side. -> ``(which axis moved, the new pair)``.

    **Which axis, not a bare "changed" flag.** The proportion chain has to know
    what the user typed, or whichever field is read second wins and typing a
    width silently rewrites it from the height that has not moved.
    """
    axis = ""
    out = [float(value[0]), float(value[1])]
    for index, (label, tag) in enumerate((("W", "w"), ("H", "h"))):
        if index:
            imgui.same_line()
        imgui.set_next_item_width(sp(90))
        # **Step zero, which is what hides imgui's own -/+ buttons.** They are
        # drawn *inside* the item's width, so at ``sp(90)`` a four-digit size
        # came out as "16" with the rest clipped -- the field stopped showing
        # the number it holds, which on a dialog whose entire subject is that
        # number is worse than having no stepper.
        if integer:
            changed, typed = controls.input_int(
                f"{label}##{prefix}{tag}", int(round(out[index])), int(step)
            )
        else:
            changed, typed = controls.input_float(
                f"{label}##{prefix}{tag}", out[index], step, format="%.1f"
            )
        if changed:
            out[index] = float(typed)
            axis = tag
    return axis, (out[0], out[1])


def _scale_dialog(ctx: Any, tab: Any, *, opening: bool = False) -> None:
    """Photoshop's Image Size / GIMP's Scale Image: resample the picture.

    **Pixels are the one stored truth** and percent is derived every frame.
    That keeps ``inker_mode.clamp_resize`` the single ceiling and makes it
    impossible for the field to promise a size the document will not get. The
    cost is a snap -- typing 50 on a three-pixel axis gives two pixels and
    redisplays 66.7 -- which is honest, and is what Photoshop does; a stored
    percentage would be a second source of truth ``clamp_resize`` cannot
    govern.
    """
    opened, _rise = _begin_dialog(SCALE_DIALOG, opening)
    if not opened:
        return
    state = inker_mode.ensure(ctx)
    old = tab.doc.size
    key = f"inker_scale:{tab.uid}"
    # Clamped on the way *in* as well as out: the stored pair outlives one
    # opening, and a document since cropped must not carry a size typed against
    # the one it used to be.
    width, height = inker_mode.clamp_resize(
        old, *(ctx.state.preview.get(key) or old)
    )

    widgets.muted(f"Current: {old[0]} x {old[1]} px")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-image-size")
    widgets.divider()

    units = widgets.segmented_control(
        "##scale-units",
        [("pixels", "Pixels"), ("percent", "Percent")],
        state.scale_units,
    )
    if units != state.scale_units:
        state.scale_units = units

    if state.scale_units == "percent":
        axis, typed = _wh_row(
            "scale", transform.size_percent(old, (width, height)),
            integer=False, step=0.0,
        )
        if axis:
            width, height = transform.percent_size(old, typed)
    else:
        axis, typed = _wh_row(
            "scale", (float(width), float(height)), integer=True, step=0
        )
        if axis:
            width, height = int(typed[0]), int(typed[1])

    imgui.same_line()
    chain = icons.LINK if state.scale_linked else icons.UNLINK
    if widgets.icon_button(
        f"{chain}##scale-chain",
        "Keep the width and height in proportion",
        selected=state.scale_linked,
    ):
        state.scale_linked = not state.scale_linked
    if axis and state.scale_linked:
        width, height = transform.linked_size(old, (int(width), int(height)), axis)
    if axis:
        ctx.state.preview[key] = inker_mode.clamp_resize(old, width, height)
    width, height = inker_mode.clamp_resize(old, width, height)

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

    # **Both readings, always.** A units toggle that hid the other number would
    # make the reader flip back and forth to answer "is that the size I meant".
    pct = transform.size_percent(old, (width, height))
    widgets.muted(
        f"New size: {width} x {height} px  ({pct[0]:.0f}% x {pct[1]:.0f}%)"
    )

    tilemap = bool(tab.doc._holds_tilemap())
    unchanged = (width, height) == tuple(old)

    imgui.begin_disabled(tab.busy)
    # **Above the action row, not beside it.** Descale is a separate offer with
    # its own explanatory line, and drawn after the buttons it put Cancel on
    # the same line as itself -- which reads as though Cancel belonged to the
    # descale rather than to the dialog.
    if _descale_row(ctx, tab, refused=tilemap):
        imgui.close_current_popup()
    imgui.end_disabled()

    widgets.divider()
    imgui.begin_disabled(tab.busy)
    if widgets.disabled_button(
        "Scale",
        not tilemap and not unchanged,
        (sp(120), 0),
        reason=_NO_TILEMAP_SCALE if tilemap else "That is already the size it is.",
        tooltip="Resample the picture to this size.",
    ):
        try:
            tab.doc.scale((width, height), resample=state.resample)
        except ValueError as exc:
            ctx.toast(f"Not scaled: {exc}.", "error")
        else:
            tab.view.fitted = False
            imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    # **Never disabled.** A save starting while this is up must not leave a
    # modal the user cannot dismiss.
    if controls.button(
        "Cancel##scale", (sp(120), 0), role=controls.ButtonRole.GHOST
    ):
        imgui.close_current_popup()
    _end_dialog()


def _canvas_dialog(ctx: Any, tab: Any, *, opening: bool = False) -> None:
    """Photoshop's / GIMP's Canvas Size: change the room around the picture.

    **Relative is a display mode over an absolute stored pair.** Storing the
    delta instead would make the stored value mean different things depending
    on a flag, and ``clamp_resize`` would have nothing to clamp.
    """
    opened, _rise = _begin_dialog(CANVAS_DIALOG, opening)
    if not opened:
        return
    state = inker_mode.ensure(ctx)
    old = tab.doc.size
    key = f"inker_resize:{tab.uid}"
    width, height = inker_mode.clamp_resize(
        old, *(ctx.state.preview.get(key) or old)
    )

    widgets.muted(f"Current: {old[0]} x {old[1]} px")
    imgui.same_line()
    manual_render.help_button_inline(ctx, "inker-canvas-size")
    widgets.divider()

    changed, relative = controls.checkbox(
        "Relative",
        state.canvas_relative,
        tooltip="Type how much to add or take away, rather than the new size.",
    )
    if changed:
        state.canvas_relative = bool(relative)

    if state.canvas_relative:
        axis, typed = _wh_row(
            "canvas", (float(width - old[0]), float(height - old[1])),
            integer=True, step=0,
        )
        if axis:
            width, height = old[0] + int(typed[0]), old[1] + int(typed[1])
    else:
        axis, typed = _wh_row(
            "canvas", (float(width), float(height)), integer=True, step=0
        )
        if axis:
            width, height = int(typed[0]), int(typed[1])
    if axis:
        ctx.state.preview[key] = inker_mode.clamp_resize(old, width, height)
    width, height = inker_mode.clamp_resize(old, width, height)

    anchor = _anchor_grid(ctx, tab)
    imgui.same_line()
    _canvas_preview(old, (width, height), anchor)

    offset = transform.anchor_offset(old, (width, height), anchor)
    delta = (width - old[0], height - old[1])
    widgets.muted(
        f"New size: {width} x {height} px  ({delta[0]:+d}, {delta[1]:+d})"
    )
    # The one thing the numbers cannot say is which side the room lands on, and
    # this is the sentence that says it.
    widgets.muted(f"The old image lands at {offset[0]}, {offset[1]}.")

    widgets.divider()
    imgui.begin_disabled(tab.busy)
    if widgets.disabled_button(
        "Resize",
        (width, height) != tuple(old),
        (sp(120), 0),
        reason="That is already the size it is.",
        tooltip="Grow or crop the canvas, leaving the picture unresampled.",
    ):
        try:
            tab.doc.resize_canvas((width, height), anchor=anchor)
        except ValueError as exc:
            # ``resize_canvas`` refuses a non-tile-aligned offset **by name and
            # before ``commit_floating``**, so nothing has changed when it does
            # -- the toast is the whole remedy and the dialog stays open on the
            # numbers that caused it.
            ctx.toast(f"Not resized: {exc}.", "error")
        else:
            tab.view.fitted = False
            imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    if controls.button(
        "Cancel##canvas", (sp(120), 0), role=controls.ButtonRole.GHOST
    ):
        imgui.close_current_popup()
    _end_dialog()


def _canvas_preview(
    old: tuple[int, int], new: tuple[int, int], anchor: str
) -> None:
    """The new canvas outlined, with the old image where the anchor puts it.

    Two rectangles on the draw list, no texture. GIMP draws a full drag
    preview and Photoshop draws none; this is the half of GIMP's that answers
    what the numbers cannot -- which side the new room opens on.
    """
    box = sp(PREVIEW_BOX)
    origin = imgui.get_cursor_screen_pos()
    imgui.dummy((box, box))
    draw = imgui.get_window_draw_list()
    new_rect, old_rect = transform.preview_boxes(old, new, anchor, box)
    draw.add_rect(
        (origin.x + new_rect[0], origin.y + new_rect[1]),
        (origin.x + new_rect[2], origin.y + new_rect[3]),
        imgui.get_color_u32(theme.rgba(theme.MUTED, 0.55)),
        sp(2),
    )
    draw.add_rect_filled(
        (origin.x + old_rect[0], origin.y + old_rect[1]),
        (origin.x + old_rect[2], origin.y + old_rect[3]),
        imgui.get_color_u32(theme.rgba(theme.ACCENT, 0.30)),
        sp(2),
    )


def _anchor_grid(ctx: Any, tab: Any) -> str:
    """Nine cells saying where the old image sits in the new canvas.

    **Photoshop's grid.** The chosen cell holds the picture; every cell
    *adjacent* to it holds an arrow pointing away -- the direction the new room
    opens in -- and a cell that is not adjacent is blank, because an arrow
    there would promise room that anchor never makes. All nine stay clickable.

    Which cell shows what is ``transform.anchor_cell``'s answer, not this
    function's: two spellings of "which cell is where" is one edit away from a
    grid that highlights one cell and resizes towards another.

    Remembered per tab beside the width and height, so reopening after a
    mistake offers the same answer rather than silently going back to the
    corner.
    """
    key = f"inker_anchor:{tab.uid}"
    current = ctx.state.preview.get(key) or "top-left"
    if current not in transform.ANCHORS:
        current = "top-left"
    widgets.field_label("Anchor")
    imgui.begin_group()
    for row in transform.ANCHOR_GRID:
        for name in row:
            if name != row[0]:
                imgui.same_line()
            # ``icon_button`` rather than a bare ``controls.button`` with a
            # padded label: it centres the glyph in a square and carries the
            # selection ring, neither of which nine blank buttons did.
            glyph = _ANCHOR_ARROWS.get(transform.anchor_cell(current, name), " ")
            if widgets.icon_button(
                f"{glyph}##anchor-{name}", name, selected=name == current
            ):
                ctx.state.preview[key] = name
                current = name
    imgui.end_group()
    return current


# --- filters ----------------------------------------------------------------
#
# A live preview rather than an apply-and-look: every one of these is a value
# nobody can predict, and a filter you have to undo to judge is a filter you
# stop using. The document owns the session (``begin_filter`` takes the copy
# every preview recomputes from) so nothing here holds pixels, and the whole
# thing is one undo step however many times a slider moved.


FILTER_POPUP = "inker-filter"
INPAINT_POPUP = "inker-inpaint"


# --- regenerate a selection --------------------------------------------------------


def _open_inpaint(ctx: Any, tab: Any) -> None:
    state = inker_mode.ensure(ctx)
    if tab.busy:
        return
    if tab.doc.mask is None or tab.doc.mask.bounds is None:
        ctx.toast("Select the area to regenerate first.", "warn")
        return
    if state.inpaint_pending is not None:
        ctx.toast("A regeneration is already on its way.", "warn")
        return
    imgui.open_popup(INPAINT_POPUP)


def _inpaint_popup(ctx: Any, tab: Any) -> None:
    state = inker_mode.ensure(ctx)
    if not imgui.begin_popup(INPAINT_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    widgets.muted("Redraw what is selected, from a prompt. Everything outside stays.")
    _changed, state.inpaint_prompt = controls.input_text(
        "##inpaint_prompt", state.inpaint_prompt
    )
    changed, value = controls.slider_float(
        "Strength##inpaint", float(state.inpaint_strength), 0.3, 0.65
    )
    if changed:
        state.inpaint_strength = value
    widgets.help_marker(
        "How far from the current pixels the model may go inside the selection."
    )
    imgui.dummy((0, sp(tokens.SP_1)))
    problems = []
    if not state.inpaint_prompt.strip():
        problems.append("Describe what should be there.")
    if tab.doc.mask is None or tab.doc.mask.bounds is None:
        problems.append("The selection is gone.")
    for problem in problems:
        widgets.muted(problem)
    imgui.begin_disabled(bool(problems) or tab.busy)
    if controls.button("Generate", (sp(90), 0)):
        submit_inpaint(ctx, tab, state.inpaint_prompt, float(state.inpaint_strength))
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    if controls.button("Cancel", (sp(90), 0)):
        imgui.close_current_popup()
    imgui.end_popup()


def submit_inpaint(ctx: Any, tab: Any, prompt: str, strength: float) -> bool:
    """Send the selection to the image model as a masked img2img reference job.

    What is remembered for the landing -- tab, layer uid, box, the selection's
    weight -- is taken *now*: the user carries on editing while the queue
    works, and the result must go back to the layer it was asked about, not
    to whatever is active when it arrives.
    """
    from ..inker import inpaint

    state = inker_mode.ensure(ctx)
    doc = tab.doc
    if doc.mask is None or doc.mask.bounds is None:
        return False
    if doc.write_locked():
        ctx.toast("The active layer is locked.", "warn")
        return False
    crop_png, mask_png, box = inpaint.prepare(
        doc.flatten(matte=False), doc.mask.mask, doc.mask.bounds
    )
    x0, y0, x1, y1 = box
    pending = {
        "tab_uid": tab.uid,
        "layer_uid": doc.stack.active.uid,
        "box": box,
        "weight": doc.mask.mask[y0:y1, x0:x1].copy(),
        "job_id": "",
        "next_poll": 0.0,
    }
    key = f"inker-inpaint:{tab.uid}"

    def run() -> Any:
        from ...service import jobs as svc_jobs

        return svc_jobs.create_job(
            ctx.svc,
            kind="text",
            prompt=prompt.strip(),
            reference=crop_png,
            mask=mask_png,
            init_image=True,
            init_strength=strength,
            output="reference",
            count=1,
            # The crop is a window on a drawing, not a subject to frame:
            # normalising it would move the pixels the mask is aligned to.
            reference_prep=False,
        )

    if not ctx.submit(key, run):
        return False
    state.inpaint_pending = pending
    ctx.toast("Regenerating the selection...")
    return True


def on_inpaint_queued(ctx: Any, result: Any) -> None:
    """``inker_mode.on_task_done``'s branch: the door answered with a job id."""
    state = inker_mode.ensure(ctx)
    pending = state.inpaint_pending
    if pending is None:
        return
    job_id = ""
    if isinstance(result, dict):
        job_id = str(result.get("id") or "")
        if not job_id:
            ids = result.get("ids") or result.get("jobs") or []
            job_id = str(ids[0]) if ids else ""
    if not job_id:
        state.inpaint_pending = None
        ctx.toast("The regeneration was not queued.", "warn")
        return
    pending["job_id"] = job_id


#: How often the bridge asks the store about a pending regeneration.
INPAINT_POLL_S = 0.5


def poll_inpaint(ctx: Any) -> None:
    """Once every ``INPAINT_POLL_S``: is the regeneration done, and land it."""
    import time

    state = inker_mode.ensure(ctx)
    pending = state.inpaint_pending
    if pending is None or not pending.get("job_id"):
        return
    now = time.monotonic()
    if now < float(pending.get("next_poll") or 0.0):
        return
    pending["next_poll"] = now + INPAINT_POLL_S
    try:
        job = ctx.svc.store.get(pending["job_id"])
    except Exception:  # noqa: BLE001 - the store answers next tick
        return
    if job is None:
        state.inpaint_pending = None
        return
    status = job.get("status")
    if status in ("queued", "running"):
        return
    state.inpaint_pending = None
    if status != "done":
        ctx.toast(f"The regeneration {status}: {job.get('error') or 'no result'}.", "warn")
        return
    # **Off the frame thread.** This is called from ``popups``, which
    # ``inker_canvas.draw`` runs every frame, and the landing is a disk read, a
    # PNG decode and a LANCZOS resize up to the size of the user's selection --
    # which can be the whole canvas. The hitch also arrives on whichever frame
    # the job happens to finish, so it is both avoidable and unpredictable.
    # ``inker_mode.on_task_done`` applies what this returns.
    image_path = ctx.svc.job_dir(pending["job_id"]) / "input.png"
    ctx.submit(
        f"inker-inpaint-land:{pending['tab_uid']}",
        _decode_inpaint,
        pending,
        image_path,
    )


def _decode_inpaint(pending: dict[str, Any], image_path: Any) -> dict[str, Any] | None:
    """Blocking; task thread only. The picture, fitted to the box it fills."""
    from PIL import Image

    from ..inker import inpaint

    try:
        with Image.open(image_path) as im:
            pixels = inpaint.fit_back(im, tuple(pending["box"]))
    except OSError:
        return None
    return {"pending": pending, "pixels": pixels}


def land_inpaint(ctx: Any, pending: dict[str, Any], pixels: Any) -> bool:
    """Blend the finished picture into the layer it was asked about.

    The frame-thread half: the document is frame-thread state and so is the
    toast. ``pixels`` comes from :func:`_decode_inpaint`, which did the reading
    and the resizing on a task thread.
    """
    state = inker_mode.ensure(ctx)
    tab = state.get(pending["tab_uid"])
    if tab is None:
        ctx.toast("The document the regeneration was for is closed.", "warn")
        return False
    if pixels is None:
        ctx.toast("The regeneration produced no picture.", "warn")
        return False
    ok = tab.doc.apply_pixels(
        int(pending["layer_uid"]), tuple(pending["box"]), pixels, pending.get("weight")
    )
    if ok:
        ctx.toast("Regeneration landed.")
    elif _still_in_the_stack(tab.doc, int(pending["layer_uid"])):
        ctx.toast("The layer it was for is gone.", "warn")
    else:
        # The uid names a *cel*, and stepping the playhead takes that cel out
        # of the stack. "The layer is gone" reads as data loss for what is
        # really "you are looking at a different frame".
        ctx.toast("It was asked for on a frame that is no longer showing.", "warn")
    return ok


def _still_in_the_stack(doc: Any, layer_uid: int) -> bool:
    try:
        doc.layer_by_uid(layer_uid)
    except KeyError:
        return False
    return True


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
    # The *owner*, not a bare flag: everything below addresses the document
    # that opened the session by name. See ``InkerState.filter_uid``.
    state.filter_uid = tab.uid
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
        if state.filter_uid:
            # Through the session-ender rather than ``tab.doc``: the tab in
            # front may not be the one that opened this.
            inker_mode.end_filter_session(ctx)
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

    imgui.dummy((0, sp(tokens.SP_1)))
    imgui.begin_disabled(tab.busy)
    if controls.button("Apply", (sp(90), 0)):
        tab.doc.commit_filter()
        state.filter_uid = ""
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    _apply_to_range(ctx, tab)
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss -- the trap the params popup in Clay documents.
    if controls.button("Cancel", (sp(90), 0)):
        tab.doc.cancel_filter()
        state.filter_uid = ""
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
        state.filter_uid = ""
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

    imgui.dummy((0, sp(tokens.SP_1)))
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
#: A second *request* key for the same popup, asking it in mode-change flavour.
#: Two keys rather than a parameter on ``pending_dialog`` because that field is
#: one string and the flavour has to survive the frame between the click and the
#: dispatch -- and a flavour left on the state by a request that never opened
#: (the tab was busy) would silently change what the next plain request did.
CONVERT_MODE_POPUP = "inker-convert-mode"


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


def open_convert(ctx: Any, tab: Any, *, to_mode: str = "") -> None:
    from ..inker import dither

    state = inker_mode.ensure(ctx)
    # Before the busy return: whoever opens the popup decides the flavour, and
    # a request that could not open must not leave the last one behind.
    state.convert_mode = to_mode
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


def apply_convert(ctx: Any, tab: Any) -> bool:
    """Answer the open session: snap onto a table, or enter indexed mode.

    Free of imgui so the branch can be asserted without a window -- which is
    the point of it being a function at all, since "did Apply change the mode"
    is the whole of what distinguishes the two sessions.

    A **mode** session cancels the preview before converting rather than
    committing it, for ``commit_convert``'s own reason one level down: the
    preview has already written converted pixels onto the current frame, and
    ``convert_to_indexed``'s snapshot would otherwise record *those* as the
    state to undo to -- one Ctrl+Z landing on a document that never existed.
    """
    state = inker_mode.ensure(ctx)
    table = list(state.convert_table)
    mode, state.convert_mode = state.convert_mode, ""
    state.convert_uid = ""
    if mode:
        tab.doc.cancel_convert()
        return inker_mode.set_color_mode(
            ctx, tab, mode, method=state.convert_method, max_colours=state.convert_max
        )
    if not tab.doc.commit_convert(table, state.convert_method):
        return False
    state.palette_slot = 0
    state.palette_slots = []
    state.palette_usage = None
    ctx.toast(f"Converted to {len(table)} colour(s).", "success")
    return True


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

    imgui.dummy((0, sp(tokens.SP_1)))
    imgui.begin_disabled(tab.busy)
    if controls.button("Apply##convert", (sp(90), 0)):
        apply_convert(ctx, tab)
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss.
    if controls.button("Cancel##convert", (sp(90), 0)):
        tab.doc.cancel_convert()
        state.convert_uid = ""
        state.convert_mode = ""
        imgui.close_current_popup()
    imgui.end_popup()
