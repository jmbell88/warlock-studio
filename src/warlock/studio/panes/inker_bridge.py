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
    # Above the tab check: importing a sheet *makes* a document, so it has to
    # be reachable when there is none open, which is exactly the moment a user
    # is most likely to want it.
    _sheet_import(ctx)
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
    _apply_to_range(ctx, tab)
    imgui.same_line()
    # Never disabled: a save starting while this is open must not leave a modal
    # the user cannot dismiss -- the trap the params popup in Clay documents.
    if imgui.button("Cancel", (sp(90), 0)):
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
    if imgui.button("Apply to range", (sp(120), 0)):
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


def _sheet_import(ctx: Any) -> None:
    """The Import sheet button, and the grid popup once a file is chosen."""
    state = inker_mode.ensure(ctx)
    if imgui.button("Import sprite sheet...", (-1, 0)):
        inker_mode.ask_import_sheet(ctx)
    widgets.help_marker(
        "Slices any image into one frame per cell, row by row. For a sheet this"
        " app generated, opening the draft from the library carries its"
        " directions and tags as well; this is for a sheet from anywhere else."
    )
    if state.sheet_import is not None and not state.sheet_import_open:
        state.sheet_import_open = True
        imgui.open_popup(SHEET_IMPORT_POPUP)
    _sheet_import_popup(ctx, state)


def _pair(label: str, value: tuple[int, int], low: int = 0) -> tuple[int, int]:
    """Two small integer fields on one row. -> the pair, floored at ``low``."""
    imgui.set_next_item_width(sp(70))
    _changed_x, x = imgui.input_int(f"##{label}x", int(value[0]), 1, 8)
    imgui.same_line()
    imgui.set_next_item_width(sp(70))
    _changed_y, y = imgui.input_int(f"##{label}y", int(value[1]), 1, 8)
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
    _changed, count = imgui.input_int("frames (0 = all)", int(state.sheet_count), 1, 8)
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
    if imgui.button("Import", (sp(90), 0)) and inker_mode.import_sheet(ctx):
        imgui.close_current_popup()
    imgui.end_disabled()
    imgui.same_line()
    if imgui.button("Cancel##sheetin", (sp(90), 0)):
        state.sheet_import = None
        state.sheet_import_open = False
        imgui.close_current_popup()
    imgui.end_popup()
