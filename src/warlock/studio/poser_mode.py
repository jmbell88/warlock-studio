"""Poser mode's controller: authoring reusable poses against a skeleton template.

The ``clay_mode.py`` pattern -- state and logic here, drawing in ``main.py``
and the two panes, no imgui anywhere under this import -- so everything about
what a session holds is assertable without a GL context.

What it is for. A pose authored here is a *complete* bone map against one of
the shipped skeleton templates, stored globally under ``data_dir/poser/`` and
applied to any rigged asset of the same template from the asset's Pose panel.
The preview it is authored on is an armature-only GLB built by the same
Blender code path as a real rig (``op_armature``), over the canonical unit box
-- so the bone frames the editor rotates are the frames every bake will see,
and model units are character heights literally.

**Poser owns its own Viewer instance.** ``adopt_model`` on the shared viewer
calls ``exit_pose_mode`` unconditionally, so loading the preview into it would
silently discard unsaved inspector pose edits, bypassing ``pose_panel.guard``
-- the exact hazard class the shared viewer's own docs describe. A second
Viewer on the one GL context is already the app's shape (the compare viewport,
ClayView), and with it every cross-mode conflict vanishes structurally: the
inspector session survives on the shared viewer, the Poser session survives
mode trips like an open Inker document, and no guard is needed on *leaving*
the mode -- only on quit and on destructive in-mode actions. The instance
lives on the App/Ctx (``ctx.poser_viewer``), constructed lazily on the frame
thread at first entry and released in teardown; this module only ever reads
it through ``viewer_of``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import poselib, rigging
from . import dialogs, journal

log = logging.getLogger(__name__)

# Task keys. Prefixed "poser-" because the app claims results by prefix.
LIST_KEY = "poser-list"
SAVE_KEY = "poser-save"
DELETE_KEY = "poser-delete"
DUPLICATE_KEY = "poser-duplicate"
RENAME_KEY = "poser-rename"
PREVIEW_KEY_PREFIX = "poser-preview:"
CLIPS_KEY = "poser-clips"
CLIPS_SAVE_KEY = "poser-clips-save"

# What pose_job_id carries in an authoring session. Can never equal a 12-hex
# job id (a colon fails is_valid_id), belt-and-braces under the separate
# viewer instance: a save can never be addressed to a job.
TOKEN_PREFIX = "poser:"


@dataclass
class PoserState:
    """One authoring session: which template, what the library holds.

    Nothing here is persisted -- a stored template key is the only candidate
    and re-deriving the default is cheaper than migrating a setting.
    """

    template: str = ""
    # The library records for ``template``, and the shipped presets beside
    # them. Both read by one task, adopted wholesale.
    poses: list[dict[str, Any]] = field(default_factory=list)
    presets: list[dict[str, Any]] = field(default_factory=list)
    loading: bool = False
    # Whether the library on screen is behind the store -- the findings_dirty
    # idiom: ``TaskRunner.submit`` refuses a key already in flight and nothing
    # else re-arms it, so a refresh wanted *while the list task runs* must be
    # a flag something pumps, cleared only when a submit is accepted.
    refresh_dirty: bool = False
    # The preview build in flight, and its answer once landed. The path is
    # bound to the viewer by ``sync_preview`` on the frame thread -- the
    # ``viewer.path`` comparison idiom Review uses, so an answer landing after
    # the user switched templates simply never binds.
    building: bool = False
    preview_path: Any = None
    preview_template: str = ""
    error: str = ""

    # -- the clip editor ------------------------------------------------------
    #
    # ``LPC_ALT.md`` Phase 2's open half: a clip was editable only as raw JSON
    # in the package tree. The library here is the *whole* file -- see
    # ``service.clips``, which will not save less than one -- held as the
    # editor's working copy and written back on Save.
    clips: dict[str, Any] = field(default_factory=dict)
    clips_loading: bool = False
    clips_dirty_flag: bool = False
    #: Which clip is open, by name. Names are a clip's identity here for the
    #: same reason they are in the file: that is what a clip's keys reference.
    clip: str = ""
    #: Which key of the open clip the editor is on, by *position*, because a
    #: clip may legitimately use one key twice (a contact pose either side of a
    #: passing one) and a name would not say which.
    key_index: int = 0
    #: Where the scrubber is, in expanded frames of the open clip. -1 is "off
    #: the scrubber, on a key" -- the state the pose gizmos are meaningful in,
    #: since a scrubbed frame is interpolated and editing it would have nowhere
    #: to be stored.
    frame: int = -1
    #: The expanded frames of the open clip, rebuilt whenever the working copy
    #: changes. Held rather than recomputed per draw: the scrubber reads it at
    #: frame rate.
    frames: list[dict[str, Any]] = field(default_factory=list)
    #: Whether the working copy differs from what is on disk. Set by every
    #: mutation, cleared when a save lands -- ``dirty``'s rule everywhere else
    #: in this app: never at submit, because a failed write has to leave the
    #: guard standing.
    clips_unsaved: bool = False
    clips_error: str = ""
    #: Whether the neighbouring keys are ghosted in the viewport. Session
    #: state, not a document fact: it is how the user is *looking* at the
    #: clip, the same argument Inker's tiled view makes about itself.
    onion: bool = False

    def open_clip(self) -> dict[str, Any] | None:
        """The clip record being edited, or None."""
        for record in self.clips.get("clips") or ():
            if record.get("name") == self.clip:
                return record
        return None

    def key_names(self) -> list[str]:
        """Every key pose name in the library, for the add/replace pickers."""
        return [str(p.get("name") or "") for p in self.clips.get("poses") or ()]

    def key_pose(self, name: str) -> dict[str, Any] | None:
        for pose in self.clips.get("poses") or ():
            if pose.get("name") == name:
                return pose
        return None

    def find(self, pose_id: Any) -> dict[str, Any] | None:
        """The library record with this id, or None.

        One method rather than five copies of the same generator expression:
        every caller here is answering "which record is the editor on", and a
        list is the right shape for a library the user reads top to bottom.
        """
        return next((p for p in self.poses if p.get("id") == pose_id), None)


def ensure(ctx: Any) -> PoserState:
    """The mode's state, built on first use -- lazy for the reason Clay's is."""
    state = ctx.state.poser
    if state is None:
        state = PoserState()
        ctx.state.poser = state
    if not state.template:
        entries = rigging.catalog()
        default = str(getattr(ctx, "rig_default", "") or "")
        keys = {e["key"] for e in entries}
        state.template = default if default in keys else (entries[0]["key"] if entries else "")
    return state


def viewer_of(ctx: Any) -> Any:
    return getattr(ctx, "poser_viewer", None)


def token(template: str) -> str:
    return f"{TOKEN_PREFIX}{template}"


# --- entering and refreshing -------------------------------------------------


def enter(ctx: Any) -> None:
    """Arriving in the mode: refresh the library and ask for the preview.

    Driven off the mode change (the Review-arrival rule), not off "the list is
    empty" -- which would submit a directory walk every frame on a library
    that genuinely holds nothing.
    """
    state = ensure(ctx)
    if not state.template:
        return
    refresh(ctx)
    clips_refresh(ctx)
    request_preview(ctx)


def _collect(svc: Any, template: str) -> dict[str, Any]:
    """The library rows and the shipped presets, in one task -- both are disk
    reads and the panes need them together."""
    from ..service import poses as svc_poses
    from ..service import rig as svc_rig

    return {
        "template": template,
        "poses": svc_poses.list_library(svc, template)["poses"],
        # Through the service, like the library rows beside it: the service
        # layer is the only business logic, and its refusal wording is the one
        # the generic failure toast shows.
        "presets": svc_rig.template_presets(template)["poses"],
    }


def refresh(ctx: Any) -> None:
    """Ask for the library to be re-read. The read itself is ``pump``'s."""
    ensure(ctx).refresh_dirty = True
    pump(ctx)


def pump(ctx: Any) -> None:
    """Submit the wanted refresh if nothing stands in the way.

    Called every frame from the library pane's draw, and again when the list
    task lands -- the frame pump is what covers a submit the runner refused,
    and the landing pump is what lets a save that arrived mid-list re-read
    without waiting for a frame. The flag clears *only* when the submit is
    accepted, the findings_dirty rule; a refusal leaves it set for the next
    pump rather than dropping the refresh for good.
    """
    state = ensure(ctx)
    if not state.refresh_dirty or state.loading or not state.template:
        return
    state.loading = True
    if ctx.submit(LIST_KEY, _collect, ctx.svc, state.template):
        state.refresh_dirty = False
    else:
        # The runner refuses a key already in flight; leaving the flag set
        # would make the mode permanently inert after a double press.
        state.loading = False


def request_preview(ctx: Any) -> None:
    """Ask for the current template's armature preview to exist.

    A cache hit inside ``template_preview`` is a couple of stats, so this is
    safe to ask on every arrival; a cold build is a Blender subprocess, which
    is exactly why it is a task and the viewport draws a progress row.
    """
    from ..service import poses as svc_poses

    state = ensure(ctx)
    if not state.template or not getattr(ctx, "rigging_available", False):
        return
    key = f"{PREVIEW_KEY_PREFIX}{state.template}"
    if ctx.busy(key):
        return
    state.building = True
    state.error = ""
    if not ctx.submit(key, svc_poses.template_preview, ctx.svc, state.template):
        state.building = False


def set_template(ctx: Any, template: str) -> None:
    """Switch skeletons, behind the guard: the editor holds one template's
    pose, and a switch discards it."""
    state = ensure(ctx)
    if template == state.template:
        return

    def proceed() -> None:
        state.template = template
        state.poses, state.presets = [], []
        state.preview_path, state.preview_template = None, ""
        viewer = viewer_of(ctx)
        if viewer is not None:
            # The old template's armature must not stay poseable under the new
            # template's library; sync_preview binds the new one when it lands.
            viewer.clear()
        refresh(ctx)
        request_preview(ctx)

    guard(ctx, "switch skeletons", proceed)


# --- the preview -------------------------------------------------------------


def preview_bounds(template_key: str) -> tuple[list[float], list[float]]:
    """The glTF-space box to frame the preview with, computed host-side.

    Over every fitted bone's head *and tail* -- leaf tails have no nodes, so a
    joint-node box would clip the skull -- from the same unit-box fit
    ``op_armature`` builds, converted Blender -> glTF per component
    ((x, z, -y), the ``m3.blender_delta_to_gltf`` mapping) and padded ~10%.
    Pure, so the framing is assertable with no GL and no file.
    """
    # Function-level, the module's own rule: nothing under this import may pull
    # in a GL context, and math3d is only wanted by this one function.
    from .viewer import math3d as m3

    template = rigging.get_template(template_key)
    fitted = rigging.fit_template(template, poselib.UNIT_LO, poselib.UNIT_HI)
    # Converted first, then boxed. The hand-coded corner swap this replaces was
    # mathematically the same thing -- min/max commute with a signed axis
    # permutation -- but it restated the mapping, which is exactly what
    # ``blender_delta_to_gltf`` exists to be the only copy of. Its
    # delta-not-absolute caveat does not apply: these are freshly computed fits
    # in the unit box, not offsets against some other frame.
    points = [
        m3.blender_delta_to_gltf(p) for bone in fitted for p in (bone["head"], bone["tail"])
    ]
    glo = [min(float(p[i]) for p in points) for i in range(3)]
    ghi = [max(float(p[i]) for p in points) for i in range(3)]
    pad = 0.10 * max(b - a for a, b in zip(glo, ghi, strict=True))
    return [v - pad for v in glo], [v + pad for v in ghi]


def sync_preview(ctx: Any, viewer: Any) -> bool:
    """Bind the built preview to the Poser viewer if it is not already shown.

    Frame thread only (it loads a model and frames a camera). What decides is
    ``viewer.path`` against the landed answer -- never a remembered flag, the
    Review lesson -- and a preview built for a template the user has switched
    away from never binds. -> whether the viewer is showing the preview.
    """
    state = ensure(ctx)
    path = state.preview_path
    if path is None or state.preview_template != state.template:
        return viewer.path is not None
    if viewer.path == Path(path):
        return True
    try:
        viewer.load_model(Path(path))
    except Exception:
        log.exception("could not open the %s pose preview", state.template)
        state.preview_path = None
        state.error = "Could not open the skeleton preview."
        return False
    bind_preview(ctx, viewer, state.template)
    return True


def bind_preview(ctx: Any, viewer: Any, template_key: str) -> None:
    """Enter the authoring session over whatever the viewer just loaded."""
    template = rigging.get_template(template_key)
    bones = [b["name"] for b in template.bones]
    viewer.enter_pose_authoring(
        bones, [list(p) for p in template.mirror_pairs], token(template.key)
    )
    viewer.editor.root = template.root
    lo, hi = preview_bounds(template_key)
    viewer.frame_bounds(lo, hi)


# --- applying ----------------------------------------------------------------


def apply_pose(ctx: Any, pose_id: str) -> None:
    """Load a library pose into the editor, behind the guard -- it overwrites
    whatever is being authored."""
    state = ensure(ctx)
    record = state.find(pose_id)
    viewer = viewer_of(ctx)
    if record is None or viewer is None or not viewer.pose_mode:
        return

    def proceed() -> None:
        # Reset first, the apply_preset order: set_pose writes only the bones
        # the record lists, so a partial record (pre-completeness saves exist
        # on disk) applied over a posed editor would keep stale rotations --
        # and get_pose() would then save them as authored.
        viewer.reset_all(dirty=False)
        viewer.set_pose(record.get("bones") or {}, pose_id=record["id"], dirty=False)
        viewer.set_root_translation(
            record.get("root_translation") or [0.0, 0.0, 0.0], dirty=False
        )

    guard(ctx, "apply a saved pose", proceed)


def apply_preset(ctx: Any, preset: dict[str, Any]) -> None:
    """Load a shipped preset, behind the guard. Presets are read-only;
    apply-then-Save-as is the promotion path into the library."""
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    guard(ctx, "apply a preset", lambda: viewer.apply_preset(preset))


def new_pose(ctx: Any) -> None:
    """Back to rest with nothing being edited, behind the guard."""
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    guard(ctx, "start a new pose", lambda: viewer.reset_all(dirty=False))


# --- saving ------------------------------------------------------------------


def _payload(state: PoserState, viewer: Any, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "template": state.template,
        "bones": viewer.get_pose(),
        "root_translation": viewer.editor.root_translation(),
    }


def _mutate(ctx: Any, key: str, fn: Any, *args: Any) -> bool:
    """Submit a library write, saying so when the runner refuses it.

    A refusal means the previous write on this key is still in flight.
    Dropping the click strands nothing -- dirty clears only on landing -- but
    it also answers nothing, and a press that does nothing reads as a broken
    button, so the refusal becomes a sentence rather than silence.
    """
    if ctx.submit(key, fn, *args):
        return True
    ctx.toast("Still working on the previous pose-library change.", "info")
    return False


def save(ctx: Any) -> None:
    """Save over the pose being edited, or fall through to Save-as."""
    from ..service import poses as svc_poses

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return
    existing = viewer.editor.current
    record = state.find(existing)
    if record is None:
        save_as(ctx)
        return
    # Dirty is cleared when the save *lands* (on_task_done), never at submit:
    # a failed write must leave the guard standing in front of the exits.
    _mutate(
        ctx,
        SAVE_KEY,
        svc_poses.update_library_pose,
        ctx.svc,
        existing,
        _payload(state, viewer, str(record.get("name") or "")),
    )


def save_as(ctx: Any) -> None:
    from ..service import poses as svc_poses

    state = ensure(ctx)
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return

    def accept(name: str) -> None:
        _mutate(
            ctx, SAVE_KEY, svc_poses.create_library_pose, ctx.svc, _payload(state, viewer, name)
        )

    ctx.prompts.ask(dialogs.Prompt(title="Name this pose", label="Name", on_accept=accept))


def rename(ctx: Any, pose_id: str) -> None:
    from ..service import poses as svc_poses

    state = ensure(ctx)
    record = state.find(pose_id)
    if record is None:
        return

    def accept(name: str) -> None:
        _mutate(ctx, RENAME_KEY, svc_poses.rename_library_pose, ctx.svc, pose_id, name)

    ctx.prompts.ask(
        dialogs.Prompt(
            title="Rename this pose",
            label="Name",
            value=str(record.get("name") or ""),
            on_accept=accept,
        )
    )


def duplicate(ctx: Any, pose_id: str) -> None:
    from ..service import poses as svc_poses

    _mutate(ctx, DUPLICATE_KEY, svc_poses.duplicate_library_pose, ctx.svc, pose_id)


def delete(ctx: Any, pose_id: str) -> None:
    """Behind a confirm: the library has no trash, so this one is genuinely
    irreversible -- the paths that are keep their question."""
    from ..service import poses as svc_poses

    state = ensure(ctx)
    record = state.find(pose_id)
    name = str((record or {}).get("name") or "this pose")

    def proceed() -> None:
        # The id rides in the key so the completion knows which pose died --
        # the result of a delete is only {"ok": True}.
        _mutate(ctx, f"{DELETE_KEY}:{pose_id}", svc_poses.delete_library_pose, ctx.svc, pose_id)

    ctx.confirms.ask(
        dialogs.Confirm(
            title="Delete this pose?",
            message=f'"{name}" will be removed from the library. '
            "Assets it was applied to keep their snapshots.",
            on_confirm=proceed,
        )
    )


# --- the guard ---------------------------------------------------------------


def guard(ctx: Any, verb: str, proceed: Any) -> bool:
    """Ask before discarding unsaved Poser edits. -> whether it went ahead now.

    Asks only about the *Poser* viewer's editor, which is what makes it
    mutually exclusive with ``pose_panel.guard`` by construction: that one
    reads the shared viewer, this one reads ``ctx.poser_viewer``, and no edit
    can live in both. Leaving the mode needs no guard at all -- the session
    survives on its own viewer, like an open Inker document -- so this runs
    only on quit and on destructive in-mode actions.
    """
    from . import docmodes

    return docmodes.viewer_guard(ctx, viewer_of(ctx), "pose", verb, proceed)


# --- keys and task results ---------------------------------------------------


def handle_key(ctx: Any, event: Any) -> bool:
    """Poser's shortcuts. -> whether the key was consumed.

    Esc deselects the joint and Ctrl+Z/Ctrl+Shift+Z/Ctrl+Y walk the pose
    history; nothing else is bound -- the mode is otherwise mouse-shaped. The
    caller returns unconditionally either way, the workspace-mode rule.

    The undo binding is shared with the inspector's asset pose mode through
    ``docmodes.pose_undo_key``, because it is one editor with two doors.
    """
    import pygame

    from . import docmodes

    if event.type != pygame.KEYDOWN:
        return False
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return False
    if docmodes.pose_undo_key(viewer, event):
        return True
    if event.key == pygame.K_ESCAPE and viewer.editor.selected is not None:
        viewer.editor.selected = None
        return True
    return False


def on_task_done(ctx: Any, done: Any) -> None:
    """Called from the app for every ``poser-`` key."""
    state = ensure(ctx)
    key = done.key
    if key == LIST_KEY:
        state.loading = False
        if isinstance(done.result, dict) and done.result.get("template") == state.template:
            state.poses = list(done.result.get("poses") or ())
            state.presets = list(done.result.get("presets") or ())
        # A save landing while this list was in flight set refresh_dirty and
        # could submit nothing; the landing is the moment the key is free.
        pump(ctx)
        return
    if key.startswith(PREVIEW_KEY_PREFIX):
        state.building = False
        template = key[len(PREVIEW_KEY_PREFIX):]
        if done.result is not None:
            state.preview_path = Path(done.result)
            state.preview_template = template
        return
    if key == SAVE_KEY:
        viewer = viewer_of(ctx)
        if viewer is not None and viewer.pose_mode and isinstance(done.result, dict):
            # Only now is the pose on disk -- the pose_panel _save rule: a
            # failed write leaves dirty set and the guard standing.
            viewer.editor.dirty = False
            viewer.editor.current = done.result.get("id")
        refresh(ctx)
        return
    if key.startswith(DELETE_KEY):
        deleted = key.partition(":")[2]
        viewer = viewer_of(ctx)
        if viewer is not None and viewer.editor.current == deleted:
            # The record Save would have written to is gone; the edits stay in
            # the editor, and Save now falls through to Save-as.
            viewer.editor.current = None
        refresh(ctx)
        return
    if key in (RENAME_KEY, DUPLICATE_KEY):
        refresh(ctx)
        return
    if key in (CLIPS_KEY, CLIPS_SAVE_KEY):
        # Both land the same way: the door hands back the library as it now is
        # on disk, and that becomes the working copy. Which is also what clears
        # ``clips_unsaved`` -- on the *landing*, never at submit, so a refused
        # write leaves the editor holding the edits that were refused.
        state.clips_loading = False
        if isinstance(done.result, dict) and done.result.get("template") == state.template:
            adopt_clips(ctx, done.result)
        return


def on_task_failed(ctx: Any, done: Any) -> None:
    """Flags only: the generic failure path has already toasted the service's
    own message, which for a save names the duplicate or the bad field."""
    state = ensure(ctx)
    if done.key == LIST_KEY:
        # ``loading`` gates the refresh; leaving it set makes the mode inert.
        state.loading = False
        # A refresh wanted while the failed list was in flight is still wanted.
        pump(ctx)
        return
    if done.key.startswith(PREVIEW_KEY_PREFIX):
        state.building = False
        # What the viewport's empty state shows under the placeholder, so a
        # broken Blender is a sentence on screen rather than a toast that
        # scrolled away.
        state.error = str(getattr(done, "message", "") or "Could not build the pose preview.")
        return
    if done.key in (CLIPS_KEY, CLIPS_SAVE_KEY):
        # ``clips_loading`` gates the read, exactly as ``loading`` does above;
        # ``clips_unsaved`` is deliberately *not* cleared, which is the whole
        # point of clearing it on the landing instead: a refused save leaves the
        # editor holding what it refused, with the reason already toasted by the
        # generic failure path.
        state.clips_loading = False
        clips_pump(ctx)


# --- crash recovery (UX-05) ---------------------------------------------------
#
# A pose is the smallest thing in the app and the easiest to lose: it lives
# entirely in a ``PoseEditor`` until somebody presses Save, so a crash during an
# authoring session took the whole of it. Both entry points are covered by one
# provider because both are one editor -- Poser's own viewer and the
# inspector's shared one -- which is the same argument
# ``docmodes.pose_undo_key`` makes about the keyboard.
#
# **Payload equality is the head.** A pose has no undo serial to compare
# against... except that since A-2 it does, and it is deliberately not used: the
# editor's history is dropped on every rebind, so its head restarts and an
# unchanged pose would be re-encoded after every model adoption. The payload is
# a few dozen floats and comparing it is cheaper than the encode it prevents.



# --- the clip editor ---------------------------------------------------------
#
# ``LPC_ALT.md`` Phase 2's open half. The clip *format* and its expansion
# shipped with Troupe; what did not was any way to change a clip that is not
# editing JSON in the package tree, which the plan calls "the most important art
# task in the program".
#
# **The editor holds a whole library as its working copy**, because
# ``service.clips`` will not save less than one -- a clip library is internally
# consistent by construction, and saving a clip alone would let a key rename
# land while another clip still pointed at the old name. Every mutation below
# therefore edits ``state.clips`` in place and sets ``clips_unsaved``; nothing
# touches disk until :func:`save_clips`.
#
# **A key is edited by posing the preview armature.** That is the whole design:
# Poser already has a skeleton, gizmos and a pose editor, so "author a
# keyframe" is "load the key onto the armature, drag joints, put it back". No
# second posing surface, and the poses a clip is made of are the same kind of
# thing the pose library already holds.


def clips_refresh(ctx: Any) -> None:
    """Ask for the clip library to be re-read."""
    state = ensure(ctx)
    state.clips_dirty_flag = True
    clips_pump(ctx)


def clips_pump(ctx: Any) -> None:
    """Submit the wanted clip read. ``pump``'s rule, on its own key.

    Guarded on ``clips_unsaved`` as well as on the flag: a re-read while the
    user has unsaved keys would silently discard them, and a background refresh
    is not a thing anybody asked for. The editor is re-read on arrival and after
    a save, which is when it is safe.
    """
    state = ensure(ctx)
    if not state.clips_dirty_flag or state.clips_loading or not state.template:
        return
    if state.clips_unsaved:
        return
    state.clips_loading = True
    if ctx.submit(CLIPS_KEY, _collect_clips, ctx.svc, state.template):
        state.clips_dirty_flag = False
    else:
        state.clips_loading = False


def _collect_clips(svc: Any, template: str) -> dict[str, Any]:
    from ..service import clips as svc_clips

    return svc_clips.library(svc, template)


def adopt_clips(ctx: Any, library: dict[str, Any]) -> None:
    """Install a freshly read library as the working copy.

    Keeps the open clip *by name* when the new library still carries one of
    that name -- a save-then-reload must not send the user back to the top of
    the list -- and falls back to the first clip otherwise.
    """
    state = ensure(ctx)
    state.clips = dict(library)
    state.clips_loading = False
    state.clips_unsaved = False
    state.clips_error = ""
    names = [str(c.get("name") or "") for c in state.clips.get("clips") or ()]
    if state.clip not in names:
        state.clip = names[0] if names else ""
        state.key_index = 0
    state.frame = -1
    rebuild_frames(ctx)


def rebuild_frames(ctx: Any) -> None:
    """Re-expand the open clip, for the scrubber.

    Through ``sheet.interpolate_clip`` -- the renderer's own interpolator --
    rather than a preview-shaped reimplementation, so what the scrubber shows is
    what the sheet will draw. A clip that cannot be expanded (segments that do
    not match its keys, mid-edit) leaves the frames empty and says why, rather
    than raising into a draw.
    """
    from ..pipelines import sheet as sheetlib

    state = ensure(ctx)
    record = state.open_clip()
    state.frames = []
    if record is None:
        return
    try:
        keys = [state.key_pose(name) for name in record.get("keys") or ()]
        if any(k is None for k in keys):
            raise ValueError("a key pose is missing from the library")
        state.frames = sheetlib.interpolate_clip(
            [dict(k) for k in keys],
            [int(n) for n in record.get("segments") or ()],
            closed=bool(record.get("closed")),
            easing=str(record.get("easing") or "linear"),
            space=str(state.clips.get("space") or "node"),
            clip_id=str(record.get("name") or ""),
        )
        state.clips_error = ""
    except Exception as exc:
        # Not a refusal -- the user is mid-edit and the clip is momentarily
        # inconsistent, which is ordinary. The scrubber empties and the reason
        # is shown; Save is what actually refuses.
        state.clips_error = str(exc)


def select_clip(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    if state.clip == name:
        return
    state.clip = str(name)
    state.key_index = 0
    state.frame = -1
    rebuild_frames(ctx)
    apply_key(ctx)


def _viewer_editor(ctx: Any) -> Any:
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return None
    return viewer.editor



# --- rotation space ---------------------------------------------------------
#
# Two frames, and they are not interchangeable (``blender_worker.POSE_SPACES``).
# ``PoseEditor`` is unconditionally *node-local*: parent-relative absolute
# rotations, which is what ``Model.set_rotation`` writes. A clip library may
# declare ``space="delta"`` instead -- each value a rotation from the bone's own
# rest, composed as ``node = rest * delta``. The shipped humanoid library is
# delta and its arms carry a constant -58 degrees off rest, so pushing its
# values through the editor raw contorts the skeleton, and capturing back out
# raw writes node-local values into a delta-declared file. Everything crossing
# the boundary goes through these two.


def _clip_space(state: Any) -> str:
    return "delta" if str(state.clips.get("space") or "node") == "delta" else "node"


def _to_node(editor: Any, bones: dict[str, Any], space: str) -> dict[str, list[float]]:
    """Library rotations -> the editor's node-local frame."""
    from .viewer import math3d as m3

    out: dict[str, list[float]] = {}
    for name, quat in bones.items():
        value = [float(v) for v in quat]
        if space == "delta":
            rest = editor.rest.get(name) if editor is not None else None
            if rest is not None:
                value = [float(v) for v in m3.quat_mul(rest, value)]
        out[name] = value
    return out


def _from_node(editor: Any, bones: dict[str, Any], space: str) -> dict[str, list[float]]:
    """The editor's node-local frame -> library rotations."""
    from .viewer import math3d as m3

    out: dict[str, list[float]] = {}
    for name, quat in bones.items():
        value = [float(v) for v in quat]
        if space == "delta":
            rest = editor.rest.get(name) if editor is not None else None
            if rest is not None:
                value = [float(v) for v in m3.quat_mul(m3.quat_conjugate(rest), value)]
        out[name] = value
    return out


def apply_key(ctx: Any) -> None:
    """Put the selected key's rotations onto the preview armature.

    Straight onto the editor, not through the pose *library*: a clip's keys are
    not library poses and giving them ids there would put twenty-two rows a
    user never asked for into the list every asset poses from.
    """
    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    record = state.open_clip()
    if editor is None or record is None:
        return
    keys = list(record.get("keys") or ())
    if not 0 <= state.key_index < len(keys):
        return
    pose = state.key_pose(keys[state.key_index])
    if pose is None:
        return
    state.frame = -1
    # ``apply_preset`` and not ``apply``: a key lists only the bones it moves,
    # so the rest have to go back to rest first, or loading "walk contact"
    # after "attack strike" leaves the sword arm up. It is one undo step, which
    # is right for a discrete "go to this key".
    editor.apply_preset(
        {"bones": _to_node(editor, dict(pose.get("bones") or {}), _clip_space(state))}
    )
    root = pose.get("root_translation")
    if root:
        editor.set_root_translation([float(v) for v in root])
    sync_onion(ctx)


def select_key(ctx: Any, index: int) -> None:
    state = ensure(ctx)
    state.key_index = max(0, int(index))
    apply_key(ctx)


def scrub(ctx: Any, frame: int) -> None:
    """Put an *interpolated* frame onto the armature.

    ``state.frame`` going non-negative is what tells the pane the gizmos are no
    longer meaningful: a scrubbed frame is between two keys and has nowhere to
    store an edit. Judged as motion here and as pixels in the sprite preview --
    the plan's "judge clips as pixels, not as viewport playback" is about the
    *verdict*, and this is the fast loop that gets you close enough to render.
    """
    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    if editor is None or not state.frames:
        return
    index = max(0, min(int(frame), len(state.frames) - 1))
    state.frame = index
    # Scrubbing puts the live skeleton on an in-between pose, so the key ghosts
    # stop meaning anything -- see ``sync_onion``.
    sync_onion(ctx)
    pose = state.frames[index]
    # Rest overlaid with the frame's own bones, through the **undecorated**
    # ``apply``. ``apply_preset`` would be the natural call and is exactly
    # wrong here: it is ``@_undoable``, and this runs once per frame of a
    # slider drag -- a step per frame is a stack full of one gesture, which is
    # the rule ``rotate_selected`` and ``move_root`` already follow. Building
    # the full map first is what makes one non-undoable call equivalent to
    # reset-then-apply.
    full = {name: [float(v) for v in quat] for name, quat in editor.rest.items()}
    full.update(_to_node(editor, dict(pose.get("bones") or {}), _clip_space(state)))
    editor.apply(full, pose_id=None, dirty=False)
    editor.set_root_translation(
        [float(v) for v in (pose.get("root_translation") or (0.0, 0.0, 0.0))],
        dirty=False,
    )


def capture_key(ctx: Any) -> None:
    """Write the armature's current pose back into the selected key.

    Refused while scrubbing, by name: the armature is showing an interpolated
    frame then, and storing it into a key would silently replace an authored
    pose with a computed one. Refused rather than "capture into the nearest
    key", which is the same act with the mistake hidden.
    """
    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    record = state.open_clip()
    if editor is None or record is None:
        return
    if state.frame >= 0:
        ctx.toast(
            "That is an in-between frame, not a key. Pick a key first.", "warn"
        )
        return
    keys = list(record.get("keys") or ())
    if not 0 <= state.key_index < len(keys):
        return
    pose = state.key_pose(keys[state.key_index])
    if pose is None:
        return
    pose["bones"] = _from_node(editor, dict(editor.pose()), _clip_space(state))
    root = editor.root_translation() if editor.root is not None else None
    if root and any(root):
        pose["root_translation"] = [float(v) for v in root]
    else:
        pose.pop("root_translation", None)
    _touch(ctx)


def set_onion(ctx: Any, on: bool) -> None:
    state = ensure(ctx)
    state.onion = bool(on)
    sync_onion(ctx)


def sync_onion(ctx: Any) -> None:
    """Put the neighbouring keys' poses on the viewer, or clear them.

    The **keys** either side of the selected one, not the frames either side of
    the playhead: a keyframe is judged against the keys it steps between, and
    the in-between frames are what the scrubber is for. A closed clip wraps, so
    the first key's "before" is the last one -- which is exactly the comparison
    a looping walk needs and the one that is easiest to get wrong by hand.

    Cleared while scrubbing: the live skeleton is then an in-between pose, and
    ghosts of the neighbouring *keys* around it would be three poses on screen
    with no stated relationship between them.
    """
    viewer = viewer_of(ctx)
    if viewer is None:
        return
    state = ensure(ctx)
    record = state.open_clip()
    if not state.onion or record is None or state.frame >= 0:
        viewer.onion = []
        return
    keys = list(record.get("keys") or ())
    if not keys:
        viewer.onion = []
        return
    closed = bool(record.get("closed"))
    out: list[dict[str, Any]] = []
    for step in (-1, 1):
        index = state.key_index + step
        if closed:
            index %= len(keys)
        elif not 0 <= index < len(keys):
            out.append({})
            continue
        pose = state.key_pose(keys[index])
        out.append(
            _to_node(_viewer_editor(ctx), dict(pose.get("bones") or {}), _clip_space(state))
            if pose
            else {}
        )
    viewer.onion = out


def _touch(ctx: Any) -> None:
    """Mark the working copy changed and re-expand it. Every mutation ends
    here, so neither half can be forgotten at one call site."""
    state = ensure(ctx)
    state.clips_unsaved = True
    rebuild_frames(ctx)


def set_segment(ctx: Any, index: int, frames: int) -> None:
    from ..service import clips as svc_clips

    state = ensure(ctx)
    record = state.open_clip()
    if record is None:
        return
    segments = list(record.get("segments") or ())
    if not 0 <= index < len(segments):
        return
    value = max(svc_clips.MIN_SEGMENT, min(int(frames), svc_clips.MAX_SEGMENT))
    if segments[index] == value:
        return
    segments[index] = value
    record["segments"] = segments
    _touch(ctx)


def set_easing(ctx: Any, easing: str) -> None:
    state = ensure(ctx)
    record = state.open_clip()
    if record is None or record.get("easing") == easing:
        return
    record["easing"] = str(easing)
    _touch(ctx)


def set_closed(ctx: Any, closed: bool) -> None:
    """Open or close the loop -- which changes how many segments the clip needs.

    An open clip of N keys has N-1 steps and a closed one has N, so the segment
    list is resized here rather than left for the save to refuse. Growing takes
    the last segment's length (a new step most like its neighbour); shrinking
    drops the one that no longer exists. Doing it silently is right *because*
    the count is derived: there is no other value it could take.
    """
    state = ensure(ctx)
    record = state.open_clip()
    if record is None or bool(record.get("closed")) == bool(closed):
        return
    record["closed"] = bool(closed)
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    wanted = len(keys) if closed else max(0, len(keys) - 1)
    while len(segments) < wanted:
        segments.append(segments[-1] if segments else 1)
    del segments[wanted:]
    record["segments"] = segments
    _touch(ctx)


def move_key(ctx: Any, index: int, delta: int) -> None:
    """Reorder one key, carrying the segment that *follows* it.

    A segment is the step out of a key, so moving a key without its segment
    would reorder the poses and leave the timing where it was -- which reads as
    the reorder having corrupted the clip.
    """
    state = ensure(ctx)
    record = state.open_clip()
    if record is None:
        return
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    target = index + int(delta)
    if not (0 <= index < len(keys) and 0 <= target < len(keys)):
        return
    keys[index], keys[target] = keys[target], keys[index]
    if index < len(segments) and target < len(segments):
        segments[index], segments[target] = segments[target], segments[index]
    record["keys"] = keys
    record["segments"] = segments
    state.key_index = target
    _touch(ctx)
    apply_key(ctx)


def insert_key(ctx: Any, name: str) -> None:
    """Add an existing key pose to the clip, after the selection.

    From the library's own poses rather than from nothing: a clip references
    keys by name, so "add a key" is either "use one of these" or "author a new
    pose first", and :func:`new_key` is the second.
    """
    state = ensure(ctx)
    record = state.open_clip()
    if record is None or state.key_pose(name) is None:
        return
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    at = min(state.key_index + 1, len(keys))
    keys.insert(at, str(name))
    # One more step exists now, wherever the loop stands: a closed clip gains a
    # segment at the same index, an open one gains the step out of the new key.
    segments.insert(min(at, len(segments)), segments[min(at, len(segments)) - 1] if segments else 1)
    record["keys"] = keys
    record["segments"] = segments
    state.key_index = at
    _touch(ctx)
    apply_key(ctx)


def new_key(ctx: Any, name: str) -> None:
    """Author a brand-new key pose from the armature and add it to the clip."""
    from ..service import clips as svc_clips

    state = ensure(ctx)
    editor = _viewer_editor(ctx)
    label = str(name or "").strip()
    if editor is None or not label:
        return
    if state.key_pose(label) is not None:
        ctx.toast(f'a key pose named "{label}" already exists', "warn")
        return
    poses = list(state.clips.get("poses") or ())
    if len(poses) >= svc_clips.MAX_LIBRARY_KEYS:
        ctx.toast(
            f"a clip library holds at most {svc_clips.MAX_LIBRARY_KEYS} key poses",
            "warn",
        )
        return
    record = {
        "name": label,
        "bones": _from_node(editor, dict(editor.pose()), _clip_space(state)),
    }
    root = editor.root_translation() if editor.root is not None else None
    if root and any(root):
        record["root_translation"] = [float(v) for v in root]
    poses.append(record)
    state.clips["poses"] = poses
    insert_key(ctx, label)


def remove_key(ctx: Any, index: int) -> None:
    """Drop one key from the clip. The *pose* stays in the library.

    Two different things, deliberately: another clip may use it, and a delete
    that silently reached into the shared pose list would be a delete with a
    blast radius the button does not describe.
    """
    from ..service import clips as svc_clips

    state = ensure(ctx)
    record = state.open_clip()
    if record is None:
        return
    keys = list(record.get("keys") or ())
    segments = list(record.get("segments") or ())
    if not 0 <= index < len(keys):
        return
    if len(keys) <= svc_clips.MIN_KEYS:
        ctx.toast(
            f"a clip needs at least {svc_clips.MIN_KEYS} keys; one key is a pose",
            "warn",
        )
        return
    del keys[index]
    if index < len(segments):
        del segments[index]
    elif segments:
        del segments[-1]
    record["keys"] = keys
    record["segments"] = segments
    state.key_index = max(0, min(state.key_index, len(keys) - 1))
    _touch(ctx)
    apply_key(ctx)


def save_clips(ctx: Any) -> None:
    """Write the working copy. One task, on its own key.

    ``clips_unsaved`` clears when the save *lands*, never at submit: a refused
    write has to leave the editor holding the edits that were refused.
    """
    from ..service import clips as svc_clips

    state = ensure(ctx)
    if not state.template or not state.clips:
        return
    payload = {
        "space": state.clips.get("space") or "node",
        "poses": state.clips.get("poses") or [],
        "clips": state.clips.get("clips") or [],
    }
    if not ctx.submit(CLIPS_SAVE_KEY, svc_clips.save, ctx.svc, state.template, payload):
        ctx.toast("Still saving the previous clip change.", "info")


def revert_clips(ctx: Any) -> None:
    """Throw the working copy away and go back to the shipped clips.

    Behind the guard for ``reset_all``'s reason: it discards authored work and
    this mode has no undo.
    """
    from ..service import clips as svc_clips

    state = ensure(ctx)
    if not state.template:
        return

    def proceed() -> None:
        if not ctx.submit(CLIPS_SAVE_KEY, svc_clips.revert, ctx.svc, state.template):
            ctx.toast("Still saving the previous clip change.", "info")

    if state.clips_unsaved or state.clips.get("edited"):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Revert clips",
                message=(
                    "This throws away every change to this skeleton's clips and "
                    "goes back to the ones the build ships. There is no undo."
                ),
                confirm_label="Revert",
                on_confirm=proceed,
            )
        )
    else:
        proceed()

def _journal_slot_for(ctx: Any, viewer: Any, key: str) -> Any:
    """One journallable pose session, or None.

    A ``SimpleNamespace``-shaped slot rather than a real object: the editor is
    not a document and has no place to keep three bookkeeping fields, so the
    journal's marks live on the *viewer*, which has exactly the lifetime the
    session does. ``key`` distinguishes the two viewers so the inspector's
    session and Poser's cannot share a filename.
    """
    if viewer is None or not getattr(viewer, "pose_mode", False):
        return None
    editor = getattr(viewer, "editor", None)
    if editor is None or not editor.bound or not editor.has_unsaved_edits():
        return None
    return _PoseSlot(viewer=viewer, editor=editor, key=key)


class _PoseSlot:
    """A pose session as the journal sees it. Marks proxy onto the viewer."""

    def __init__(self, viewer: Any, editor: Any, key: str) -> None:
        self.viewer = viewer
        self.editor = editor
        self.key = key

    @property
    def journal_name(self) -> str:
        return getattr(self.viewer, "journal_name", "") or ""

    @journal_name.setter
    def journal_name(self, value: str) -> None:
        self.viewer.journal_name = value

    @property
    def journal_head(self) -> Any:
        return getattr(self.viewer, "journal_head", None)

    @journal_head.setter
    def journal_head(self, value: Any) -> None:
        self.viewer.journal_head = value

    @property
    def journal_at(self) -> float:
        return float(getattr(self.viewer, "journal_at", 0.0) or 0.0)

    @journal_at.setter
    def journal_at(self, value: float) -> None:
        self.viewer.journal_at = value


def _pose_payload(slot: Any) -> bytes:
    import json as _json

    editor = slot.editor
    return _json.dumps(
        {
            "bones": editor.pose(),
            "root_translation": editor.root_translation(),
            "moved": {name: list(delta) for name, delta in editor.moved.items()},
            "mode": editor.mode,
            # Which job's rig this was bound to, so the inspector's copy can
            # refuse to land on a different asset.
            "job_id": getattr(slot.viewer, "pose_job_id", None) or "",
            "where": slot.key,
        },
        sort_keys=True,
    ).encode("utf-8")


def _journal_slots(ctx: Any) -> list[Any]:
    """Both pose sessions -- Poser's own viewer and the inspector's."""
    out = []
    for viewer, key in ((viewer_of(ctx), "poser"), (getattr(ctx, "viewer", None), "asset")):
        slot = _journal_slot_for(ctx, viewer, key)
        if slot is not None:
            out.append(slot)
    return out


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Put a recovered pose back onto a bound editor, or say why not.

    **A pose is not a document and cannot be opened on its own**: it is a set
    of rotations for a named skeleton, and applying it needs that skeleton
    loaded. So this is the one adopter that can decline for a reason the user
    has to hear -- the job it was authored against may have been deleted, or
    the session may simply not be open -- and it declines by *keeping* the file
    and saying so, because the alternative is applying a stranger's rotations
    to whatever rig happens to be in front of them.
    """
    import json as _json

    try:
        data = _json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.exception("could not read the recovered pose at %s", path)
        return False
    where = str(data.get("where") or "poser")
    viewer = viewer_of(ctx) if where == "poser" else getattr(ctx, "viewer", None)
    editor = getattr(viewer, "editor", None)
    if viewer is None or editor is None or not editor.bound:
        ctx.toast(
            "An unsaved pose was recovered. Open the rig it belongs to and it "
            "will be offered again.",
            "warn",
        )
        return False
    job_id = str(data.get("job_id") or "")
    if job_id and str(getattr(viewer, "pose_job_id", "") or "") != job_id:
        # The asset it was authored against is not the one in front of us. Kept
        # rather than applied: rotations by bone name land on whatever shares a
        # name, silently and wrongly.
        ctx.toast(
            "An unsaved pose was recovered, but it belongs to a different "
            "asset. Open that one and it will be offered again.",
            "warn",
        )
        return False
    editor.apply(data.get("bones") or {}, pose_id=None, dirty=True)
    if data.get("root_translation"):
        editor.set_root_translation(data["root_translation"])
    if str(data.get("mode") or "pose") == "joints":
        # The payload records everything ``_pose_payload`` writes, and joint
        # corrections are half of it: a crash mid-placement recovered as a rest
        # pose under a success toast, the corrections silently gone. Entering
        # joints mode takes fresh homes (the rotations in a joints session are
        # rest anyway -- ``enter_joints_mode`` resets them on the way in, as it
        # did in the crashed session), and each recorded displacement is then
        # replayed through ``move_handle``, the same door a drag uses, so the
        # markers land where the user put them and ``moved`` is re-derived
        # rather than trusted.
        import numpy as np

        from .viewer import math3d as m3

        editor.enter_joints_mode()
        for name, delta in (data.get("moved") or {}).items():
            home = editor.home.get(name)
            if home is None:
                continue
            editor.move_handle(
                name, home + m3.blender_delta_to_gltf(np.asarray(delta, dtype="f8"))
            )
    after = getattr(viewer, "_after_pose_change", None)
    if after is not None:
        after()
    ctx.toast("An unsaved pose was recovered.", "success")
    return True


JOURNAL = journal.register(
    journal.Provider(
        kind="pose",
        ext=".pose.json",
        label="pose",
        slots=_journal_slots,
        uid_of=lambda slot: slot.key,
        title_of=lambda slot: "Pose" if slot.key == "poser" else "Asset pose",
        # Payload equality: see the section note above.
        head_of=_pose_payload,
        encode=_pose_payload,
        adopt=_journal_adopt,
    )
)
