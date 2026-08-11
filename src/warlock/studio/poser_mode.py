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
from . import dialogs

log = logging.getLogger(__name__)

# Task keys. Prefixed "poser-" because the app claims results by prefix.
LIST_KEY = "poser-list"
SAVE_KEY = "poser-save"
DELETE_KEY = "poser-delete"
DUPLICATE_KEY = "poser-duplicate"
RENAME_KEY = "poser-rename"
PREVIEW_KEY_PREFIX = "poser-preview:"

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

    Esc deselects the joint, nothing else is bound -- the mode is
    mouse-shaped, and the shortcuts popup deliberately has no Poser section.
    The caller returns unconditionally either way, the workspace-mode rule.
    """
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    viewer = viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        return False
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
