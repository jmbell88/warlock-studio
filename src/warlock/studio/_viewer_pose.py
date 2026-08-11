"""The Viewer's pose half: binding an editor to a rig, and drawing its controls.

Split out of :mod:`~warlock.studio.viewer_embed` as pure code motion, the
``clay_view`` mixin pattern: methods only, no state and no ``__init__`` -- every
attribute these touch is still built and owned by ``Viewer.__init__``, so the
class is a namespace rather than a second object.

Why this half and not another. Posing is the one section of the Viewer with a
contract of its own: an editor bound to *whose* rig, a dirty listener, and two
gizmos whose choice is a function of the editor's mode. The two render helpers
come with it rather than staying behind, because they are pose-only -- they
return nothing at all outside pose mode -- and reading ``_active_gizmo`` next to
``root_translate`` is what makes the move-root rule legible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .viewer import bonelines as bonelineslib
from .viewer import math3d as m3
from .viewer import picking

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .viewer_embed import Viewer


class PoseOps:
    """``Viewer``'s pose mode. See the module docstring."""

    # -- pose --------------------------------------------------------------

    def enter_pose_mode(
        self: Viewer, rig: dict[str, Any] | None = None, job_id: str | None = None
    ) -> bool:
        """Bind the editor to the loaded rig. ``job_id`` is *whose* rig it is.

        Recorded because the editor outlives the selection: ``_sync_viewer``
        deliberately does not reload while posing, so clicking another asset
        leaves this rig on screen with the inspector pointing somewhere else.
        Without a bound id the panel used the selected job for both, and Save
        wrote one asset's rotations into another's pose directory while joints
        mode re-skinned it against the wrong skeleton -- silently, because rig
        validation checks bone *names* and two jobs on one template share them.
        """
        if self.model is None or not self.model.skins:
            return False
        self.editor.bind(self.model)
        self.editor.mirror_pairs = list((rig or {}).get("mirror_pairs") or [])
        self.editor.fitted = list((rig or {}).get("bones") or [])
        self.pose_mode = True
        self.pose_job_id = job_id
        self._bone_pairs = bonelineslib.bone_segments(self.model, self.editor.bones)
        self._render_dirty = True
        self._notify_pose_dirty()
        return True

    def enter_pose_authoring(
        self: Viewer, bones: list[str], mirror_pairs: list[Any], token: str
    ) -> bool:
        """Bind the editor to a meshless armature with an explicit bone list.

        A separate entry point rather than a relaxed ``enter_pose_mode``: that
        one's skins check is its job-scoped contract (a mesh with no skin has
        nothing to pose) and stays intact. ``token`` fills ``pose_job_id`` --
        it can never equal a 12-hex job id, belt-and-braces under the separate
        Poser Viewer instance, so a save can never be addressed to a job.
        """
        if self.model is None:
            return False
        self.editor.bind(self.model, bones)
        self.editor.mirror_pairs = [list(p) for p in mirror_pairs]
        self.pose_mode = True
        self.pose_job_id = token
        self._bone_pairs = bonelineslib.bone_segments(self.model, self.editor.bones)
        self._render_dirty = True
        self._notify_pose_dirty()
        return True

    def exit_pose_mode(self: Viewer) -> None:
        self._render_dirty = True
        self.pose_mode = False
        self.pose_job_id = None
        self.rotate_gizmo.end_drag()
        self.translate_gizmo.end_drag()
        self.editor.clear()
        self._bone_pairs = []
        # Both ends of the editor's life report, or an indicator raised on the
        # way in survives every mesh loaded after it -- ``adopt_model`` and
        # ``clear`` both come through here.
        self._notify_pose_dirty()

    def get_pose(self: Viewer) -> dict[str, list[float]]:
        return self.editor.pose()

    def set_pose(
        self: Viewer, bones: dict[str, Any], *, pose_id: str | None = None, dirty=True
    ) -> None:
        self.editor.apply(bones, pose_id=pose_id, dirty=dirty)
        self._after_pose_change()

    def apply_preset(self: Viewer, preset: dict[str, Any]) -> None:
        self.editor.apply_preset(preset)
        self._after_pose_change()

    def reset_bone(self: Viewer) -> None:
        self.editor.reset_bone()
        self._after_pose_change()

    def reset_all(self: Viewer, *, dirty: bool = True) -> None:
        self.editor.reset_all(dirty=dirty)
        self._after_pose_change()

    def mirror(self: Viewer) -> None:
        self.editor.mirror()
        self._after_pose_change()

    def set_root_translation(self: Viewer, v: Any, *, dirty: bool = True) -> None:
        self.editor.set_root_translation(v, dirty=dirty)
        self._after_pose_change()

    def _after_pose_change(self: Viewer) -> None:
        self._render_dirty = True
        if self.gpu is not None:
            self.gpu.refresh_palettes()
        self._notify_pose_dirty()

    def _notify_pose_dirty(self: Viewer) -> None:
        """Tell the listener whether unsaved pose edits exist *right now*.

        Answered as ``pose_mode and ...`` rather than by asking the editor
        alone, so leaving the editor -- or loading another mesh, which exits it
        -- reports False. A listener told "dirty" on the way out and never told
        otherwise would keep an indicator up for edits nothing can reach any
        more, which is worse than no indicator at all.

        The listener is a mirror, never the authority: ``pose_panel.guard``
        goes on asking ``editor.has_unsaved_edits()`` itself, so a missed
        notification costs a stale marker rather than discarded work.
        """
        if self.on_pose_dirty is not None:
            self.on_pose_dirty(self.pose_mode and self.editor.has_unsaved_edits())

    @property
    def selected_bone(self: Viewer) -> str | None:
        return self.editor.selected

    def enter_joints_mode(self: Viewer) -> None:
        self.editor.enter_joints_mode()
        self._after_pose_change()

    def exit_joints_mode(self: Viewer) -> None:
        self.editor.exit_joints_mode()
        self._after_pose_change()

    def joint_moves(self: Viewer) -> dict[str, list[float]]:
        return dict(self.editor.moved)

    def corrected_bones(self: Viewer) -> list[dict[str, Any]]:
        return self.editor.corrected_bones()

    # -- the pose-only half of the render list -----------------------------

    def _overlays(self: Viewer, height: int) -> list[Any]:
        if not self.pose_mode or not self.editor.bound:
            return []
        radius = picking.marker_radius(self.radius)
        # Lines first, markers over them: the skeleton is context, the joints
        # are the controls.
        items = self.bonelines.draws(
            self.editor.handles, self._bone_pairs, self.editor.selected, self.placement
        )
        items += self.markers.draws(
            self.editor.handles, radius, self.editor.selected, self.placement
        )
        gizmo = self._active_gizmo()
        if gizmo is not None and self.editor.selected is not None:
            # Looked up rather than indexed: a selection naming a bone the
            # current handles or model no longer carry costs the gizmo for a
            # frame, which is the right degrade on the render path. Indexing
            # raised KeyError inside the frame loop instead.
            handle = self.editor.handles.get(self.editor.selected)
            index = self.model.by_name.get(self.editor.selected)
            if handle is None or index is None:
                return items
            origin = picking.to_world(self.placement, handle)
            # A translate gizmo always works in world axes -- the joints-mode
            # convention, kept for the root translate; the rotate gizmo's
            # rings are drawn in the joint's own frame.
            node = self.model.nodes[index]
            basis = (
                m3.identity()
                if gizmo is self.translate_gizmo
                else self.placement @ node.world
            )
            gizmo.place(origin, basis, self.camera, height)
            items += gizmo.draws()
        return items

    def _active_gizmo(self: Viewer):
        if not self.pose_mode:
            return None
        if self.editor.mode == "joints":
            return self.translate_gizmo
        if (
            self.editor.root_translate
            and self.editor.root is not None
            and self.editor.selected == self.editor.root
        ):
            # Move-root in the authoring session: the root joint translates,
            # every other joint keeps its rotate rings.
            return self.translate_gizmo
        return self.rotate_gizmo
