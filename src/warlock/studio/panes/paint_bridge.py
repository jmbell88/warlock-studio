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

from .. import paint_mode, theme, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    state = paint_mode.ensure(ctx)
    tab = state.active
    widgets.section("document")
    manual_render.help_button(ctx, "paint-bridge")
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
    _pipeline(ctx, tab)
    imgui.dummy((0, 8))
    _canvas_ops(ctx, tab)


def _pipeline(ctx: Any, tab: Any) -> None:
    widgets.section("pipeline")
    busy = tab.saving
    if not tab.linked:
        if widgets.disabled_button("Save as reference", not busy, (-1, 0)):
            paint_mode.save_as_reference(ctx, tab)
        widgets.help_marker(
            "Adds this image to the library as a finished reference, so it can be"
            " meshed, promoted and rerun like a generated one."
        )
    if widgets.disabled_button("Send to 3D", not busy, (-1, 0)):
        paint_mode.send_to_3d(ctx, tab)
    widgets.help_marker("Queues the mesh stage from the flattened image.")
    if tab.linked and widgets.disabled_button(
        "Revert to original", tab.has_original and not busy, (-1, 0)
    ):
        paint_mode.revert(ctx, tab)


def _canvas_ops(ctx: Any, tab: Any) -> None:
    doc = tab.doc
    widgets.section("canvas")
    if imgui.button("Flip H"):
        doc.flip("horizontal")
    imgui.same_line()
    if imgui.button("Flip V"):
        doc.flip("vertical")
    imgui.same_line()
    if imgui.button("Rotate"):
        doc.rotate90()
    if imgui.button("Resize..."):
        imgui.open_popup("paint-resize")
    imgui.same_line()
    if imgui.button("Fit view"):
        tab.view.fitted = False
    _resize_popup(ctx, tab)

    imgui.dummy((0, 6))
    if widgets.disabled_button("Undo", doc.history.can_undo):
        doc.undo()
    imgui.same_line()
    if widgets.disabled_button("Redo", doc.history.can_redo):
        doc.redo()
    imgui.same_line()
    widgets.muted(f"{len(doc.history)} step(s)")


def _resize_popup(ctx: Any, tab: Any) -> None:
    if not imgui.begin_popup("paint-resize"):
        return
    key = f"paint_resize:{tab.uid}"
    width, height = ctx.state.preview.get(key) or tab.doc.size
    imgui.set_next_item_width(90)
    changed_w, width = imgui.input_int("W", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(90)
    changed_h, height = imgui.input_int("H", int(height), 0)
    if changed_w or changed_h:
        ctx.state.preview[key] = (max(1, width), max(1, height))
    imgui.dummy((0, 4))
    if imgui.button("Scale image", (180, 0)):
        tab.doc.scale((max(1, width), max(1, height)))
        tab.view.fitted = False
        imgui.close_current_popup()
    # Two different operations that a single "resize" would conflate: one
    # resamples the picture, the other changes how much room it has.
    if imgui.button("Resize canvas", (180, 0)):
        tab.doc.resize_canvas((max(1, width), max(1, height)))
        tab.view.fitted = False
        imgui.close_current_popup()
    imgui.end_popup()
