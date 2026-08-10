"""Posing a rigged mesh, and correcting where its joints sit.

Two modes over the same markers, and the unsaved-edit guard is the thing that
matters most: a pose lives only in the viewer until it is saved, so every exit
route -- switching jobs, starting a comparison, leaving edit mode, closing the
window -- has to ask first.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from imgui_bundle import imgui

from ...service import rig as svc_rig
from .. import dialogs, icons, theme, widgets
from ..manual import render as manual_render

log = logging.getLogger(__name__)


def draw(ctx: Any, job: Any) -> None:
    files = job.get("files") or []
    if "model.glb" not in files:
        return
    rigged = "rig.glb" in files
    if not widgets.header("Pose"):
        return
    manual_render.help_button(ctx, "pose")
    if not rigged:
        # Every other rig control hides itself when Blender is missing, so
        # "rig this mesh first" was instructing the user to press a button that
        # is not on screen and cannot be.
        if not ctx.rigging_available:
            widgets.muted("Posing needs Blender, which is not installed.")
            return
        widgets.muted("Posing needs a rig.")
        # S138. Rigging is a real queued job -- Blender out of process, behind
        # whatever the serial worker is already doing -- and looked from here
        # like a button that does something immediately.
        widgets.cost_note(
            "Rigging is queued like a generation: it runs Blender in a separate "
            "process and waits behind anything already running."
        )
        rig_key = f"rig:{job['id']}"
        if widgets.disabled_button("Rig this mesh", not ctx.busy(rig_key)):
            ctx.submit(
                rig_key, svc_rig.create_rig, ctx.svc, job["id"], template=ctx.rig_default or None
            )
        return

    viewer = ctx.viewer
    editing = viewer is not None and viewer.pose_mode
    if not editing:
        if imgui.button("Edit pose", (-1, 0)):
            _enter(ctx, job)
        _saved_list(ctx, job)
        return
    if not _is_bound_job(viewer, job):
        _elsewhere(ctx, viewer)
        return

    _banner(ctx, viewer)
    if viewer.editor.mode == "joints":
        _joints(ctx, job, viewer)
    else:
        _pose(ctx, job, viewer)
    _saved_list(ctx, job)


# --- entering and leaving ---------------------------------------------------


def _is_bound_job(viewer: Any, job: Any) -> bool:
    """Whether this panel is looking at the asset the editor is editing.

    A bound id of ``None`` means an editor opened by something that did not say
    whose rig it was; treated as "this one", which is the pre-binding behaviour
    and the only answer that keeps a test double working.
    """
    bound = getattr(viewer, "pose_job_id", None)
    return bound is None or bound == job.get("id")


def _elsewhere(ctx: Any, viewer: Any) -> None:
    """The editor is open on a different asset than the one selected.

    Reachable because ``_sync_viewer`` deliberately does not reload the viewer
    while posing -- so clicking another asset in the library leaves this rig on
    screen under that asset's inspector. Every control is withheld rather than
    retargeted: Save wrote the rotations on screen into the *selected* job's
    pose directory, and joints mode re-skinned it against a skeleton fitted to
    something else, neither of which rig validation could catch (it checks bone
    names, and two jobs on one template have the same ones).
    """
    widgets.text_colored(theme.ACCENT, "Pose editing - another asset")
    widgets.muted(
        "The pose editor is open on the asset in the viewport, not this one. "
        "Select it again to carry on, or leave the editor."
    )
    if imgui.button("Leave pose mode"):
        guard(ctx, "leave edit mode", lambda: leave(ctx))
    imgui.same_line()
    if imgui.button("Show the asset being posed"):
        # No guard: nothing is discarded by looking at the rig again.
        ctx.state.select(viewer.pose_job_id)


def _enter(ctx: Any, job: Any) -> None:
    """Load the rig into the viewer and bind the editor to it."""
    job_id = job["id"]
    rig_path = ctx.job_dir(job_id) / "rig.glb"
    if not rig_path.exists():
        ctx.toast("This mesh has no rig yet.", "error")
        return
    try:
        ctx.viewer.load_model(rig_path)
    except Exception:
        # Logged as well as toasted: a rig GLB that will not open is a real
        # failure with a real traceback, and every comparable site in the app
        # writes one. Without it the only trace was a four-word toast.
        log.exception("could not open the rig for job %s", job_id)
        ctx.toast("Could not open the rig.", "error")
        return
    rig = None
    # Posing still works by hand without rig.json; only the mirror button and
    # the joint editor need what it carries, and both hide themselves without it.
    with contextlib.suppress(Exception):
        rig = svc_rig.get_rig(ctx.svc, job_id)
    if not ctx.viewer.enter_pose_mode(rig, job_id):
        ctx.toast("That GLB carries no skeleton.", "error")
        return
    # The preset library follows the rig's skeleton, so it is fetched here
    # rather than at startup -- there is no single answer before a rig is open.
    ctx.load_presets((rig or {}).get("template"))
    # And the saved poses, because opening the editor is the one moment the
    # viewer changes what it shows without the selection changing -- which is
    # what otherwise triggers a refresh.
    ctx.refresh_rig_data()


def guard(ctx: Any, action: str, proceed: Any) -> bool:
    """Ask before discarding unsaved edits. -> whether it went ahead now."""
    viewer = ctx.viewer
    if viewer is None or not viewer.pose_mode or not viewer.editor.has_unsaved_edits():
        proceed()
        return True
    what = "joint" if viewer.editor.mode == "joints" else "pose"
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved changes?",
            message=f"Unsaved {what} changes will be lost if you {action}.",
            on_confirm=proceed,
        )
    )
    return False


def leave(ctx: Any) -> None:
    if ctx.viewer is not None:
        ctx.viewer.exit_pose_mode()


# --- pose mode --------------------------------------------------------------


def _banner(ctx: Any, viewer: Any) -> None:
    joints = viewer.editor.mode == "joints"
    label = "Joint placement" if joints else "Pose editing"
    if viewer.editor.has_unsaved_edits():
        label += " - unsaved changes"
    widgets.text_colored(theme.ACCENT, label)
    if imgui.button("Done"):
        guard(ctx, "leave edit mode", lambda: leave(ctx))


def _pose(ctx: Any, job: Any, viewer: Any) -> None:
    selected = viewer.selected_bone
    widgets.muted(selected or "Click a joint to rotate it.")
    if widgets.disabled_button("Reset joint", selected is not None):
        viewer.reset_bone()
    imgui.same_line()
    if imgui.button("Reset all"):
        viewer.reset_all()
    if viewer.editor.mirror_pairs:
        # Hidden for a serpent or a fish: a skeleton with no mirror pairs has
        # nothing to mirror, and a button that can only no-op is worse than none.
        imgui.same_line()
        if imgui.button("Mirror"):
            viewer.mirror()

    presets = ctx.state.preview.get("presets") if ctx.state.preview else None
    if presets:
        chosen = widgets.combo(
            "##preset", "", [("", "preset...")] + [(p["name"], p["name"]) for p in presets]
        )
        if chosen:
            preset = next((p for p in presets if p["name"] == chosen), None)
            if preset:
                # apply_preset resets every joint first, so hand-made
                # rotations need the same confirm as any other discard.
                guard(ctx, "apply a preset", lambda p=preset: viewer.apply_preset(p))

    if viewer.editor.current:
        # Before the button, the retarget panel's rule (S139): saving over an
        # existing pose deletes its cached bake, because the GLB on disk depicts
        # the rotations that are about to be replaced. The service is right to
        # delete it and wrong to be the only thing that mentions it.
        widgets.text_colored(
            theme.WARN, "Saving under the same name replaces that pose and its saved GLB."
        )
    if imgui.button("Save pose...", (-1, 0)):
        _save(ctx, job, viewer)
    if viewer.editor.fitted and imgui.button("Adjust joints", (-1, 0)):
        # A rotation left over from posing would put the markers where the
        # posed bones are, and a correction is against the rest skeleton.
        guard(ctx, "move joints", viewer.enter_joints_mode)


def _save(ctx: Any, job: Any, viewer: Any) -> None:
    job_id = job["id"]
    existing = viewer.editor.current

    def accept(name: str) -> None:
        payload: dict[str, Any] = {"name": name, "bones": viewer.get_pose()}
        if existing:
            # Saving under the same pose replaces it, rather than leaving two
            # called "idle" that differ by one shoulder.
            payload["id"] = existing
        # Dirty is cleared when the save *lands*, not here: clearing it at
        # submit time means a failed write leaves the editor claiming the pose
        # is safe, and the next Escape or selection change discards it silently.
        ctx.submit(f"pose-save:{job_id}", svc_rig.save_pose, ctx.svc, job_id, payload)

    ctx.prompts.ask(dialogs.Prompt(title="Name this pose", label="Name", on_accept=accept))


# --- joints mode ------------------------------------------------------------


def _joints(ctx: Any, job: Any, viewer: Any) -> None:
    moved = viewer.joint_moves()
    widgets.muted(viewer.selected_bone or "Click a joint to move it.")
    widgets.muted(f"{len(moved)} joint(s) moved")
    job_id = job["id"]
    busy = ctx.busy(f"joints:{job_id}")
    if widgets.disabled_button("Apply joint positions", bool(moved) and not busy, (-1, 0)):
        # A re-rig, not an edit: skinning is minutes of CPU and must never
        # overlap a trellis run, so it goes through the queue like the first
        # rig did.
        ctx.submit(
            f"joints:{job_id}",
            svc_rig.adjust_joints,
            ctx.svc,
            job_id,
            {"bones": viewer.corrected_bones()},
        )
        viewer.exit_joints_mode()
    if widgets.disabled_button("Revert", bool(moved)):
        viewer.editor.revert_joints()
    if imgui.button("Back to posing", (-1, 0)):
        guard(ctx, "return to pose editing", viewer.exit_joints_mode)


# --- the saved list ---------------------------------------------------------


def _library_list(ctx: Any, job: Any) -> None:
    """Poses authored in the Poser, offered to any asset on the same skeleton.

    Applying one *snapshots* it into this job's own poses/ directory (see
    ``service.poses.apply_library_pose``): the copy is the provenance record,
    so editing or deleting the library pose afterwards changes nothing here.
    The rows come off the ``pose-library:{job_id}`` task the side-data refresh
    already submits; an empty answer draws nothing at all, because an asset
    with no applicable poses does not need a section saying so -- the Poser is
    where the library is explained.
    """
    from ...service import poses as svc_poses

    poses = (ctx.state.preview or {}).get("library_poses") or []
    if not poses:
        return
    widgets.section("Library poses")
    widgets.help_marker(
        "Poses from the Poser workspace that fit this skeleton. Applying one "
        "copies it onto this asset, so later library edits never change it."
    )
    job_id = job["id"]
    needle = widgets.list_filter(ctx, "library-poses", len(poses))
    for pose in poses:
        pose_id = str(pose.get("id") or "")
        name = str(pose.get("name") or pose_id)
        if needle and needle not in name.lower():
            continue
        imgui.push_id(f"lib-{pose_id}")
        imgui.text(name)
        imgui.same_line()
        key = f"pose-lib-apply:{job_id}:{pose_id}"
        if widgets.disabled_button("Apply", not ctx.busy(key)):
            ctx.submit(key, svc_poses.apply_library_pose, ctx.svc, job_id, pose_id)
        imgui.pop_id()


def _saved_list(ctx: Any, job: Any) -> None:
    _library_list(ctx, job)
    poses = (ctx.state.preview or {}).get("poses") or []
    widgets.section("Saved poses")
    if not poses:
        # H73, and the same reason the sheet list has one: an absent section
        # is indistinguishable from a broken one.
        widgets.empty_state(
            icons.PERSON_STANDING,
            "No saved poses",
            "Enter edit mode, rotate a joint, then Save.",
        )
        return
    # S138, and the interesting half of it: a pose bake is deliberately *not*
    # queued -- it is a one-second Blender subprocess derived on request, for
    # the same reason the STL and OBJ exports are -- so saying so is what stops
    # "Save GLB..." reading like another two-minute wait.
    widgets.cost_note("Saving a posed GLB runs on the spot; it does not join the queue.")
    widgets.help_marker(
        "A pose is the joint rotations, saved beside the mesh. Applying one "
        "replaces whatever the editor is holding."
    )
    job_id = job["id"]
    needle = widgets.list_filter(ctx, "poses", len(poses))
    for pose in poses:
        pose_id = pose["id"]
        if needle and needle not in str(pose.get("name") or pose_id).lower():
            continue
        imgui.push_id(pose_id)
        imgui.text(pose.get("name") or pose_id)
        if pose.get("source_pose"):
            # The provenance marker: this row is a snapshot applied from the
            # global library, not a hand-made pose.
            imgui.same_line()
            widgets.muted("(library)")
        imgui.same_line()
        if imgui.small_button("Apply") and ctx.viewer is not None:
            # Overwrites the editor's rotations and clears dirty, so it is an
            # exit route like Done/Escape and takes the same confirm.
            def _apply(pose=pose, pose_id=pose_id):
                if not ctx.viewer.pose_mode:
                    _enter(ctx, job)
                ctx.viewer.set_pose(pose.get("bones") or {}, pose_id=pose_id, dirty=False)

            guard(ctx, "apply a saved pose", _apply)
        imgui.same_line()
        key = f"bake:{job_id}:{pose_id}"
        if widgets.disabled_button("Save GLB...", not ctx.busy(key)):
            _bake(ctx, job_id, pose_id)
        imgui.same_line()
        if imgui.small_button("Delete"):
            ctx.submit(
                f"pose-del:{job_id}:{pose_id}", svc_rig.delete_pose, ctx.svc, job_id, pose_id
            )
        imgui.pop_id()


def _bake(ctx: Any, job_id: str, pose_id: str) -> None:
    """Bake the pose into a GLB and save it where the user says.

    On a task thread rather than the job queue: it is a one-second Blender
    subprocess, and putting it behind the serial GPU queue would make it wait
    on a trellis run.
    """

    def run():
        source = svc_rig.posed_model(ctx.svc, job_id, pose_id)
        dest = dialogs.save_file("Save posed GLB", f"{job_id}_{pose_id}.glb", dialogs.GLB_FILTER)
        if dest is None:
            return None
        dest.write_bytes(source.read_bytes())
        return dest

    ctx.submit(f"bake:{job_id}:{pose_id}", run)
