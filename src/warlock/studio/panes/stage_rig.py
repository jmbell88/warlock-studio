"""The Rig stage's settings column: pick a skeleton, fit it to the mesh.

A *form*, like the two stages before it -- which is the whole reason this is a
pane rather than a section of the inspector. Rigging was reachable from three
places (a library card's **Rig**, the 3D form's "rig when the mesh lands", and a
button buried in the inspector's Pose panel), all three doing the same thing to
the same mesh, and none of them was where a user looking at a finished mesh
would look. The stage is that place; the other three still work, and the
skeleton they all use is chosen here (:func:`skeleton`).

What the rig *produced* -- the weighting method, the bone count, the
deformation sheet -- stays in the inspector beside it. This column is the
decision and the button; the column on the right is the evidence.
"""

from __future__ import annotations

from typing import Any

from ...service import rig as svc_rig
from .. import icons, widgets
from ..manual import render as manual_render


def skeleton(ctx: Any) -> str | None:
    """The skeleton every rig submission uses. None means "no explicit choice".

    Lifted from ``library._skeleton`` unchanged, and kept as one function for
    its reason: rigging an existing mesh used to pass nothing at all, so picking
    "quadruped" and then rigging a finished mesh silently used the config
    default. It reads ``form_3d`` because that is where the combo has always
    written -- the control moved to this pane in wave 5, the field did not, so a
    session that set the skeleton before generating still has it set here.
    """
    return (ctx.state.form_3d or {}).get("rig_template") or None


def rig_key(job: Any) -> str:
    """The submit key, one per job -- so a second press while the first is in
    flight is refused rather than queueing a second rig of the same mesh."""
    return f"rig:{job['id']}"


def draw(ctx: Any) -> None:
    job = ctx.job()
    widgets.section("Skeleton")
    # On the first *real* row rather than under the heading: ``section`` ends
    # in a spacer, so a right-aligned (?) called straight after a title floats
    # alone on an empty line (the UI redesign, wave 4).
    manual_render.help_button(ctx, "settings-rig")
    if job is None:
        widgets.empty_state(
            icons.BONE, "No mesh selected.", "Pick a finished mesh; the skeleton fits to it."
        )
        return
    if not ctx.rigging_available:
        # The stage is still *enterable* without Blender -- the rail says why
        # rather than hiding the segment -- so this pane answers the same
        # question again for anyone who arrived from a library card.
        widgets.muted("Rigging needs Blender, which is not installed.")
        return

    _skeleton_picker(ctx)

    rigged = "rig.glb" in (job.get("files") or [])
    if rigged:
        widgets.muted("This mesh is rigged. Rigging it again replaces the rig.")
    # S138, and it stays said here: rigging is a real queued job -- Blender out
    # of process, behind whatever the serial worker is already doing -- and a
    # bare button looked like something that happens now.
    widgets.cost_note(
        "Rigging is queued like a generation: it runs Blender in a separate "
        "process and waits behind anything already running."
    )
    key = rig_key(job)
    label = "Rig this mesh again" if rigged else "Rig this mesh"
    if widgets.disabled_button(
        label, not ctx.busy(key), reason="This mesh is already being rigged."
    ):
        ctx.submit(key, svc_rig.create_rig, ctx.svc, job["id"], template=skeleton(ctx))


def _skeleton_picker(ctx: Any) -> None:
    """The template combo. Silent with no templates rather than drawing an
    empty select: a list with nothing in it is a configuration failure, and
    ``ctx.rig_templates`` being empty is already reported by the doctor."""
    options = [(t["key"], t["label"]) for t in ctx.rig_templates]
    if not options:
        return
    form = ctx.state.form_3d
    form["rig_template"] = widgets.labeled_combo(
        "Skeleton",
        form.get("rig_template") or ctx.rig_default,
        options,
        help_text=(
            "Which template's joints are fitted onto the mesh. The fit is by "
            "proportion unless the mesh's own landmarks are found."
        ),
    )
    # ``settings_3d``'s reason: ``service/rig.py`` and ``service/poses.py``
    # both refuse on ``rig_template``, and a generic toast that leaves the
    # dropdown at fault unmarked is the one failure the form cannot point at.
    widgets.field_error(ctx.state, "rig_template")
