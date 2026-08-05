"""What the document is, and the two ways out of it.

The counts and the save state on top, the pipeline buttons underneath -- the
same shape the raster editor's bridge takes, and for the same reason: a panel
that offers to send something somewhere should first say what it is going to
send.

**The two output paths are genuinely different things, not two encodings of
one.** Export puts the *exact* geometry in the library as an ordinary asset,
which is what a user wants when the shape they modelled is the shape they
meant. Send to 3D renders the document flat and hands the picture to trellis,
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

from .. import build_mode, icons, widgets


def draw(ctx: Any) -> None:
    state = build_mode.ensure(ctx)
    tab = state.active
    widgets.section("document")
    if tab is None:
        widgets.muted("Nothing open.")
        return

    _facts(tab)
    imgui.dummy((0, 8))
    _files(ctx, tab)
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


def _files(ctx: Any, tab: Any) -> None:
    widgets.field_label("document")
    imgui.begin_disabled(tab.saving)
    if imgui.button(f"{icons.SAVE} Save"):
        build_mode.save(ctx, tab)
    imgui.same_line()
    if imgui.button("Save as..."):
        build_mode.save_as(ctx, tab)
    imgui.same_line()
    if imgui.button(f"{icons.FOLDER_OPEN} Open"):
        build_mode.ask_open(ctx)
    imgui.end_disabled()
    if tab.path is not None:
        widgets.muted(str(tab.path))


def _outputs(ctx: Any, tab: Any) -> None:
    widgets.field_label("send")
    doc = tab.doc
    ready = any(obj.visible for obj in doc.objects) and not tab.saving

    if widgets.primary_button(f"{icons.DOWNLOAD} Export to library", enabled=ready):
        build_mode.export_asset(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "The exact geometry, as an ordinary asset. It picks up rigging, posing, "
            "sprite sheets, the triangle retarget and every mesh export, because all "
            "of those are functions of model.glb."
        )

    if widgets.disabled_button(f"{icons.SEND} Send to 3D", ready):
        # Wired in Task 13, which owns the offscreen render: the picture has to
        # be drawn on the frame thread because it needs the GL context, and the
        # bridge is not where that belongs.
        send_to_3d(ctx, tab)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Renders the document flat and hands the picture to trellis, which "
            "reinterprets it: the blockout becomes a suggestion, and what comes back "
            "has surface detail nobody modelled."
        )

    if tab.job_id:
        widgets.muted(f"last exported as {tab.job_id}")


def send_to_3d(ctx: Any, tab: Any) -> None:
    """Placeholder until Task 13 supplies the offscreen render.

    Named and called from here rather than left as a dead button, so the wiring
    that Task 13 completes has exactly one place to land and the pane needs no
    edit when it does.
    """
    handler = getattr(ctx, "build_send_to_3d", None)
    if handler is None:
        ctx.toast("Sending a build to 3D is not wired up yet.", "error")
        return
    handler(tab)
