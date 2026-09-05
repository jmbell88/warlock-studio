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

from .. import clay_mode, icons, tokens, verbs, widgets
from ..manual import render as manual_render
from ..tokens import sp

#: What this pane refuses to shrink past, in design pixels: the path line, the
#: undo pair, the step count and the two ways out.
#:
#: It had none while the right column was a hand-composed pair. The declarative
#: layout stacks the outliner and the properties above it, and
#: ``layout_skeleton.heights`` gives each share its proportion of the room
#: before the fill sees any -- so two at the default 0.5 leave this pane exactly
#: zero pixels and the Document panel is not on screen at all. Plotter's map
#: file panel went the same way on the same day and for the same reason.
BRIDGE_FLOOR = 170.0


def draw(ctx: Any) -> None:
    state = clay_mode.ensure(ctx)
    tab = state.active
    # "Model file", the shape of every other bridge's heading ("Drawing file",
    # "Map file", "Song file", "Atlas file"); this one said "Document".
    widgets.section("Model file")
    manual_render.help_button(ctx, "clay-bridge")
    if tab is None:
        # The recent list and nothing else -- Plotter's bridge exactly (B5).
        # New/Open are on the empty canvas two columns to the left, and drawing
        # them here as well was one pair of buttons in two places; the *list*
        # is the opposite case, because the moment it matters most is the
        # moment there is nothing open.
        _recent(ctx)
        return
    _files(ctx, tab)

    _facts(tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _history(ctx, tab)
    imgui.dummy((0, sp(tokens.SP_2)))
    _outputs(ctx, tab)
    _recent(ctx)


def _history(ctx: Any, tab: Any) -> None:
    """Undo and Redo, on screen.

    This mode had a full undo stack and no visible control for it, so the
    feature existed only for a user who already knew Ctrl+Z -- while Inker drew
    the same pair twice. ``clay_mode.undo``/``redo`` rather than
    ``tab.doc.undo()`` here, so the button and the chord carry the same side
    effects (see the history block in that module).
    """
    widgets.history_block(
        ctx,
        tab,
        key="clay",
        undo=lambda: clay_mode.undo(ctx, tab),
        redo=lambda: clay_mode.redo(ctx, tab),
        step=lambda index: clay_mode.step_history(ctx, tab, index),
    )


def _facts(tab: Any) -> None:
    doc = tab.doc
    visible = [obj for obj in doc.objects if obj.visible]
    triangles = sum(_triangles(obj.mesh) for obj in visible)
    widgets.muted(
        f"{len(visible)} of {len(doc.objects)} objects visible  -  "
        f"{triangles:,} triangles  -  {len(doc.materials)} materials"
    )


def _triangles(mesh: Any) -> int:
    """Counted the way the renderer fans them, so the number matches the
    exported file rather than the face count the outliner would give."""
    import numpy as np

    counts = np.diff(mesh.starts)
    return int(np.maximum(counts - 2, 0).sum()) if len(counts) else 0


def _files(ctx: Any, tab: Any) -> None:
    """The file row for an *open* document -- the shared header, so it is the
    same four buttons and the same status ladder every other workspace has.
    ``draw`` returns before this when there is none, because New and Open also
    belong to the empty canvas."""
    widgets.document_header(
        tab,
        new=lambda: clay_mode.new_document(ctx),
        open_=lambda: clay_mode.ask_open(ctx),
        save=lambda: clay_mode.save(ctx, tab),
        save_as=lambda: clay_mode.save_as(ctx, tab),
    )
    imgui.dummy((0, sp(tokens.SP_2)))


def _outputs(ctx: Any, tab: Any) -> None:
    # The one heading every mode's exits are under. See ``inker_bridge``'s
    # ``_pipeline`` for why the five of them agree on a name.
    widgets.section("Take it somewhere")
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

    if widgets.primary_button(f"{icons.DOWNLOAD} {verbs.EXPORT_TO_LIBRARY}", enabled=ready):
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


def _recent(ctx: Any) -> None:
    """The recent list, on the bridge, on **both** branches.

    Plotter's bridge already draws its list whether or not a document is open,
    and that is the answer: a recent list is how you get *back* to work, so the
    one moment it matters most is the moment there is nothing open. Clay's was
    on the empty canvas instead, which is the one screen it disappears from as
    soon as it becomes useful again.
    """
    from pathlib import Path

    widgets.recent_files(
        clay_mode.recent_paths(ctx),
        lambda path: clay_mode.open_path(ctx, Path(path)),
    )
