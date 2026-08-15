"""What the document is, and the two ways out of it.

The counts and the save state on top, the pipeline buttons underneath -- the
same shape the raster editor's bridge takes, and for the same reason: a panel
that offers to send something somewhere should first say what it is going to
send.

**The two output paths are genuinely different things, not two encodings of
one.** Export puts the *exact* geometry in the library as an ordinary asset,
which is what a user wants when the shape they modelled is the shape they
meant. Make 3D renders the document flat and hands the picture to trellis,
which reinterprets it -- the blockout becomes a suggestion rather than a
specification, and what comes back is a reconstruction with surface detail
nobody modelled. Choosing between them is the whole point of having both, so
the panel says which is which rather than labelling them "Export" and "Export".

Both buttons are disabled while a save is in flight, for the reason the tool
panel states.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, controls, icons, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    widgets.section("Document")
    manual_render.help_button(ctx, "clay-bridge")
    _files(ctx, tab)
    if tab is None:
        widgets.muted("Nothing open.")
        return

    _facts(tab)
    imgui.dummy((0, 8))
    _outputs(ctx, tab)


def _facts(tab: Any) -> None:
    doc = tab.doc
    visible = [obj for obj in doc.objects if obj.visible]
    triangles = sum(_triangles(obj.mesh) for obj in visible)
    widgets.muted(
        f"{len(visible)} of {len(doc.objects)} objects visible  -  "
        f"{triangles:,} triangles  -  {len(doc.materials)} materials"
    )
    if tab.saving:
        widgets.spinner()
        imgui.same_line()
        widgets.muted("saving...")
    elif tab.dirty:
        widgets.muted("unsaved changes")
    else:
        widgets.muted("saved")


def _triangles(mesh: Any) -> int:
    """Counted the way the renderer fans them, so the number matches the
    exported file rather than the face count the outliner would give."""
    import numpy as np

    counts = np.diff(mesh.starts)
    return int(np.maximum(counts - 2, 0).sum()) if len(counts) else 0


def _files(ctx: Any, tab: Any | None) -> None:
    width = widgets.grid_width(2)
    if controls.button(f"{icons.PLUS} New", (width, 0)):
        clay_mode.new_document(ctx)
    imgui.same_line()
    if controls.button(f"{icons.FOLDER_OPEN} Open...", (width, 0)):
        clay_mode.ask_open(ctx)
    if tab is None:
        return
    imgui.begin_disabled(tab.saving)
    if controls.button(f"{icons.SAVE} Save (Ctrl+S)", (width, 0)):
        clay_mode.save(ctx, tab)
    imgui.same_line()
    if controls.button("Save As...", (width, 0)):
        clay_mode.save_as(ctx, tab)
    imgui.end_disabled()
    if tab.path is not None:
        widgets.muted(str(tab.path))
    imgui.dummy((0, 8))


def _outputs(ctx: Any, tab: Any) -> None:
    widgets.field_label("send")
    doc = tab.doc
    ready = any(obj.visible for obj in doc.objects) and not tab.saving
    # One sentence for both buttons below, because they are refused for the
    # same two reasons and a user reading two different explanations of one
    # state would look for two different problems. The ``_VIEWPORT_WHY``
    # pattern: a shared gate gets a shared sentence, hoisted to a local.
    why = (
        "Saving..."
        if tab.saving
        else "Nothing visible to send -- every object is hidden."
    )

    if widgets.primary_button(f"{icons.DOWNLOAD} Export to library", enabled=ready):
        clay_mode.export_asset(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "The exact geometry, as an ordinary asset. It picks up rigging, posing, "
            "sprite sheets, the triangle retarget and every mesh export, because all "
            "of those are functions of model.glb."
        )

    # "Make 3D", matching the Mesh stage's own button and Inker's -- wave 5
    # left no "3D" to send anything to.
    if widgets.disabled_button(f"{icons.SEND} Make 3D", ready, reason=why):
        # The App owns the offscreen render: the picture has to be drawn on
        # the frame thread because it needs the GL context, and the bridge is
        # not where that belongs.
        send_to_3d(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Renders the document flat and hands the picture to trellis, which "
            "reinterprets it: the blockout becomes a suggestion, and what comes back "
            "has surface detail nobody modelled."
        )

    if tab.job_id:
        widgets.muted(f"Last exported as {tab.job_id}")


def send_to_3d(ctx: Any, tab: Any) -> None:
    """Hand the document to the App's offscreen render (``_clay_send_to_3d``).

    The indirection through ``ctx`` is the point: the render needs the GL
    context and therefore the frame thread, which is the App's business. A
    headless ctx that never attached the handler gets a clear refusal rather
    than a half-drawn frame.

    The refusal stays -- ``Ctx.clay_send_to_3d`` defaults to None and only the
    App assigns it, so a ctx built without one is a real construction and not a
    hypothetical -- but its wording did not. It said the feature was "not wired
    up yet", which was true of the branch's own first draft and has not been
    true of the app since: a user who saw it went looking for a setting to turn
    on. What the branch actually knows is that *this* window has nothing to
    render from, so that is what it now says.
    """
    handler = getattr(ctx, "clay_send_to_3d", None)
    if handler is None:
        ctx.toast("Could not make a mesh: this window has no viewport to render from.", "error")
        return
    handler(tab)
