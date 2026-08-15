"""What connects a painting to the rest of the app.

Paint is a third mode rather than a separate program, and this is the panel
that makes that true: a document can become a reference the pipeline accepts,
or go straight to the mesh stage, and a document that came *from* a job can be
saved back into it.

The distinction the whole panel turns on is whether the document is *linked*.
A linked one writes back into a job's input.png (with the layered source kept
beside it); an unlinked one is a plain file that has never been part of a job.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import inker_mode, theme, widgets
from ..inker import transform
from ..manual import render as manual_render
from ..tokens import sp


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


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    tab = state.active
    widgets.section("document")
    manual_render.help_button(ctx, "inker-bridge")
    if tab is None:
        widgets.muted("Nothing open.")
        return

    doc = tab.doc
    width, height = doc.size
    widgets.muted(f"{width} x {height} - {len(doc.stack)} layer(s)")
    widgets.muted(f"{tab.view.zoom * 100:.0f}%  -  {tab.file_format.upper()}")
    if tab.path is not None:
        imgui.text_wrapped(str(tab.path))
    if tab.linked:
        widgets.text_colored(theme.OK, f"linked to job {tab.job_id[:8]}")
    else:
        widgets.muted("not part of a job")

    imgui.dummy((0, 8))
    _animation(ctx, tab)
    imgui.dummy((0, 8))
    _pipeline(ctx, tab)
    imgui.dummy((0, 8))
    _canvas_ops(ctx, tab)


def _animation(ctx: Any, tab: Any) -> None:
    """The one door into animating a document.

    Deliberately a button rather than a mode or a checkbox: animating is an
    *edit* -- one undo step that turns the layers into the first frame's cels --
    so it belongs on the same footing as adding a layer, and Ctrl+Z takes it
    back. Once the document is animated this row goes quiet and the timeline
    strip owns everything else.
    """
    widgets.section("animation")
    if tab.doc.anim is None:
        if widgets.disabled_button(
            "Animate", not tab.busy, (-1, 0), reason=_busy_why(tab)
        ):
            inker_mode.animate(ctx, tab)
        widgets.help_marker(
            "Turns this drawing into frame one of an animation and adds a second"
            " frame. The layers become tracks; Ctrl+Z undoes the whole thing."
        )
        return
    anim = tab.doc.anim
    widgets.muted(f"{len(anim.frames)} frames, {len(anim.tracks)} tracks")
    widgets.muted(f"{anim.duration_ms()} ms total")


def _pipeline(ctx: Any, tab: Any) -> None:
    widgets.section("pipeline")
    busy = tab.busy
    why = _busy_why(tab)
    if not tab.linked:
        if widgets.disabled_button("Save as reference", not busy, (-1, 0), reason=why):
            inker_mode.save_as_reference(ctx, tab)
        widgets.help_marker(
            "Adds this image to the library as a finished reference, so it can be"
            " meshed, promoted and rerun like a generated one."
        )
    if widgets.disabled_button("Send to 3D", not busy, (-1, 0), reason=why):
        inker_mode.send_to_3d(ctx, tab)
    widgets.help_marker("Queues the mesh stage from the flattened image.")
    if tab.linked and widgets.disabled_button(
        "Revert to original",
        tab.has_original and not busy,
        (-1, 0),
        reason=why
        if busy
        else "There is no original kept for this reference: it has never been edited.",
    ):
        inker_mode.revert(ctx, tab)


def _canvas_ops(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    widgets.section("canvas")
    # Every control below either rebinds a layer's pixels (the geometry ops,
    # via _map_planes) or rebinds the stack wholesale (undo, via
    # restore_snapshot). ``write_ora`` flattens, writes stack.xml and then
    # walks the stack writing one PNG per layer, so either one landing
    # mid-save produces an archive whose parts disagree about the canvas size.
    # The canvas, the layers panel and the keyboard path all gate on this
    # flag; this panel was the hole.
    imgui.begin_disabled(tab.busy)
    if imgui.button("Flip H"):
        doc.flip("horizontal")
    imgui.same_line()
    if imgui.button("Flip V"):
        doc.flip("vertical")
    imgui.same_line()
    if imgui.button("Rotate"):
        doc.rotate90()
    if imgui.button("Resize..."):
        imgui.open_popup("inker-resize")
    imgui.same_line()
    if imgui.button("Fit view"):
        tab.view.fitted = False
    if imgui.button("Filter..."):
        _open_filter(ctx, tab)

    imgui.dummy((0, 6))
    if widgets.disabled_button(
        "Undo", doc.history.can_undo, reason="Nothing to undo yet."
    ):
        doc.undo()
    imgui.same_line()
    if widgets.disabled_button(
        "Redo", doc.history.can_redo, reason="Nothing to redo: this is the newest step."
    ):
        doc.redo()
    imgui.same_line()
    widgets.muted(f"{len(doc.history)} step(s)")
    imgui.end_disabled()

    # Outside the disabled scope: a popup is its own window, and imgui's
    # disabled state is not meant to span a Begin/End pair. It carries the
    # same gate on the two buttons that actually resample the document -- a
    # popup already open when a save starts would otherwise still fire them.
    _resize_popup(ctx, tab)
    _filter_popup(ctx, tab)


def _resize_popup(ctx: Any, tab: Any) -> None:
    if not imgui.begin_popup("inker-resize"):
        return
    key = f"inker_resize:{tab.uid}"
    width, height = ctx.state.preview.get(key) or tab.doc.size
    imgui.set_next_item_width(sp(90))
    changed_w, width = imgui.input_int("W", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(90))
    changed_h, height = imgui.input_int("H", int(height), 0)
    if changed_w or changed_h:
        ctx.state.preview[key] = (max(1, width), max(1, height))
    state = inker_mode.ensure(ctx)
    state.resample = widgets.labeled_combo(
        "Resample", state.resample, [(k, k) for k in transform.RESAMPLES]
    )
    widgets.help_marker(
        "Nearest copies each source pixel whole, which is what pixel art needs "
        "-- a filtered scale of a 32x32 sprite comes back blurred and with "
        "thousands of colours in it. Smooth is right for everything else."
    )
    anchor = _anchor_grid(ctx, tab)
    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if imgui.button("Scale image", (sp(180), 0)):
        tab.doc.scale((max(1, width), max(1, height)), resample=state.resample)
        tab.view.fitted = False
        imgui.close_current_popup()
    # Two different operations that a single "resize" would conflate: one
    # resamples the picture, the other changes how much room it has. The anchor
    # belongs to the second: scaling has nowhere to put slack.
    if imgui.button("Resize canvas", (sp(180), 0)):
        tab.doc.resize_canvas((max(1, width), max(1, height)), anchor=anchor)
        tab.view.fitted = False
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
            if imgui.button(f" ##anchor{name}", (sp(28), sp(24))):
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


def _filter_values(state: Any, name: str) -> dict[str, float]:
    from ..inker import filters

    got = state.filter_params.get(name)
    if got is None:
        got = dict(filters.FILTERS[name][0])
        state.filter_params[name] = got
    return got


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

    name = widgets.combo(
        "Filter", state.filter_name, [(key, key) for key in filters.FILTERS]
    )
    if name != state.filter_name:
        state.filter_name = name
    values = _filter_values(state, state.filter_name)
    for key in filters.FILTERS[state.filter_name][0]:
        low, high = filters.RANGES.get(key, (0.0, 1.0))
        imgui.set_next_item_width(sp(160))
        changed, value = imgui.slider_float(key, float(values[key]), low, high)
        if changed:
            values[key] = float(value)
    if imgui.button("Reset##filterreset"):
        state.filter_params[state.filter_name] = dict(
            filters.FILTERS[state.filter_name][0]
        )

    # Every frame, not only on a change: the combo above can switch filters,
    # and a preview that only ran on a slider move would leave the last
    # filter's pixels under the new filter's controls.
    tab.doc.preview_filter(state.filter_name, **_filter_values(state, state.filter_name))

    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if imgui.button("Apply", (sp(90), 0)):
        tab.doc.commit_filter()
        state.filter_open = False
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss -- the trap the params popup in Clay documents.
    if imgui.button("Cancel", (sp(90), 0)):
        tab.doc.cancel_filter()
        state.filter_open = False
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
    if owner is None or tab is None or owner.uid != tab.uid:
        inker_mode.end_convert_session(ctx)
        imgui.close_current_popup()
        imgui.end_popup()
        return

    state.convert_method = widgets.combo(
        "Dither", state.convert_method, [(key, key) for key in dither.METHODS]
    )
    if not tab.doc.palette:
        imgui.set_next_item_width(sp(160))
        changed, value = imgui.slider_int("Colours", int(state.convert_max), 2, 64)
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
    if imgui.button("Apply##convert", (sp(90), 0)):
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
    if imgui.button("Cancel##convert", (sp(90), 0)):
        tab.doc.cancel_convert()
        state.convert_uid = ""
        imgui.close_current_popup()
    imgui.end_popup()
