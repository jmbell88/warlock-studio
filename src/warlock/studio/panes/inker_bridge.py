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
from ..manual import render as manual_render
from ..tokens import sp


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
        if widgets.disabled_button("Animate", not tab.busy, (-1, 0)):
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
    if not tab.linked:
        if widgets.disabled_button("Save as reference", not busy, (-1, 0)):
            inker_mode.save_as_reference(ctx, tab)
        widgets.help_marker(
            "Adds this image to the library as a finished reference, so it can be"
            " meshed, promoted and rerun like a generated one."
        )
    if widgets.disabled_button("Send to 3D", not busy, (-1, 0)):
        inker_mode.send_to_3d(ctx, tab)
    widgets.help_marker("Queues the mesh stage from the flattened image.")
    if tab.linked and widgets.disabled_button(
        "Revert to original", tab.has_original and not busy, (-1, 0)
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
    if widgets.disabled_button("Undo", doc.history.can_undo):
        doc.undo()
    imgui.same_line()
    if widgets.disabled_button("Redo", doc.history.can_redo):
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
    anchor = _anchor_grid(ctx, tab)
    imgui.dummy((0, 4))
    imgui.begin_disabled(tab.busy)
    if imgui.button("Scale image", (sp(180), 0)):
        tab.doc.scale((max(1, width), max(1, height)))
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
        ctx.toast("There is nothing to filter.", "warning")
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
