"""The pose editor's state: rotations, joint moves, mirroring.

Two modes over one set of markers. In **pose** mode a marker shows where its
bone ended up and the gizmo rotates the bone; in **joints** mode the marker
*is* the thing being dragged and the gizmo translates it, because a joint's
position is a property of the armature's rest pose that the viewer's copy of
the rig cannot express -- so the marker is the handle and the server re-skins.

The mirror comes from :func:`warlock.rigging.mirror_pose`, imported rather than
reimplemented. The browser had its own copy with a comment insisting the two
stay identical, which is exactly the kind of sign convention that is wrong in a
way you cannot see: a mirrored arm rotating the wrong way about one axis still
looks plausible in a still.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ... import poselib, rigging
from . import math3d as m3
from .gltf import Model

# Re-exported so nothing downstream is tempted to write the sign flip out again.
mirror_quaternion = rigging.mirror_quaternion


class PoseEditor:
    """Rotations and joint moves for one rigged model."""

    def __init__(self) -> None:
        self.model: Model | None = None
        self.bones: list[str] = []
        self.rest: dict[str, np.ndarray] = {}
        self.selected: str | None = None
        self.mode = "pose"  # or "joints"
        self.current: str | None = None  # the saved pose being edited
        self.dirty = False
        self.mirror_pairs: list[list[str]] = []
        self.fitted: list[dict[str, Any]] = []
        # bone -> [dx, dy, dz] in *Blender* space, which is what the API wants.
        self.moved: dict[str, list[float]] = {}
        # Where each marker started this joints-mode session, in model space.
        self.home: dict[str, np.ndarray] = {}
        self.handles: dict[str, np.ndarray] = {}
        # Root-translation authoring (Poser only; pose_panel's entry path never
        # sets these, which is what keeps asset pose mode behaviourally
        # unchanged). ``root`` names the bone whose node translation may move;
        # ``root_translate`` is whether the gizmo currently translates it.
        self.root: str | None = None
        self.root_translate = False

    # -- binding -----------------------------------------------------------

    def bind(self, model: Model, bones: list[str] | None = None) -> None:
        """Attach to a loaded rig. Joint order follows the skin's palette."""
        self.model = model
        if bones is None:
            bones = _skin_bones(model)
        self.bones = [b for b in bones if b in model.by_name]
        self.rest = {b: model.get_rotation(b) for b in self.bones}
        self.selected = None
        self.current = None
        self.dirty = False
        self.moved.clear()
        self.home.clear()
        self.root = None
        self.root_translate = False
        self.handles = {b: model.nodes[model.by_name[b]].world[:3, 3].copy() for b in self.bones}

    def clear(self) -> None:
        self.model = None
        self.bones = []
        self.rest.clear()
        self.handles.clear()
        self.home.clear()
        self.moved.clear()
        self.selected = None
        self.current = None
        self.dirty = False
        self.mode = "pose"
        self.root = None
        self.root_translate = False

    @property
    def bound(self) -> bool:
        return self.model is not None and bool(self.bones)

    def has_unsaved_edits(self) -> bool:
        return bool(self.dirty or self.moved)

    # -- rotations ---------------------------------------------------------

    def pose(self) -> dict[str, list[float]]:
        """Every bound bone's current local rotation, XYZW."""
        if self.model is None:
            return {}
        out: dict[str, list[float]] = {}
        for bone in self.bones:
            quat = self.model.get_rotation(bone)
            out[bone] = [float(v) for v in (m3.quat_identity() if quat is None else quat)]
        return out

    def apply(self, bones: dict[str, Any], *, pose_id: str | None = None, dirty: bool = True):
        if self.model is None:
            return
        for name, quat in (bones or {}).items():
            self.model.set_rotation(name, quat)
        self.model.update_world()
        self._resync_handles()
        self.current = pose_id
        self.dirty = dirty

    def apply_preset(self, preset: dict[str, Any]) -> None:
        """A preset lists only the bones it moves, so the rest is reset first
        -- otherwise applying "idle" after "wave" leaves the arm up."""
        self.reset_all(dirty=True)
        self.apply(preset.get("bones", {}), pose_id=None, dirty=True)

    def reset_bone(self, bone: str | None = None) -> None:
        bone = bone or self.selected
        if self.model is None or bone is None:
            return
        self.model.set_rotation(bone, self.rest.get(bone, m3.quat_identity()))
        if bone == self.root:
            self._restore_root_rest()
        self.model.update_world()
        self._resync_handles()
        self.dirty = True

    def reset_all(self, *, dirty: bool = True) -> None:
        if self.model is None:
            return
        for bone, quat in self.rest.items():
            self.model.set_rotation(bone, quat)
        # The root's rest translation comes back with the rotations, which is
        # also what makes a preset zero-translation: apply_preset resets first.
        self._restore_root_rest()
        self.model.update_world()
        self._resync_handles()
        self.current = None
        self.dirty = dirty

    def rotate_selected(self, delta: np.ndarray) -> None:
        """Post-multiply a local-space delta onto the selected bone.

        Post- rather than pre-multiply: the gizmo's rings are drawn in the
        joint's own frame, so a drag round the visible red ring must turn the
        bone about *its* X, not the world's.
        """
        if self.model is None or self.selected is None:
            return
        current = self.model.get_rotation(self.selected)
        if current is None:
            return
        self.model.set_rotation(self.selected, m3.quat_normalize(m3.quat_mul(current, delta)))
        self.model.update_world()
        self._resync_handles()
        self.dirty = True

    def mirror(self) -> None:
        """Copy every posed bone onto its mirror partner, reflected."""
        if not self.mirror_pairs:
            return
        # ``pose_id=self.current``: apply's clearing of ``current`` is for
        # *loading* a different pose, and a mirror is still the same one --
        # dirty yes, identity no. Without it, Mirror silently turned the next
        # Save into Save-as.
        self.apply(rigging.mirror_pose(self.pose(), self.mirror_pairs), pose_id=self.current)
        if self.root is not None:
            # The positional half of the same reflection: a pose that steps
            # left must step right when mirrored.
            self.set_root_translation(
                poselib.mirror_root_translation(self.root_translation())
            )

    def _resync_handles(self) -> None:
        """Markers follow their bones -- except in joints mode, where a marker
        *is* the drag and snapping it back would undo it as fast as it is made."""
        if self.model is None or self.mode == "joints":
            return
        self.handles = {
            b: self.model.nodes[self.model.by_name[b]].world[:3, 3].copy() for b in self.bones
        }

    # -- root translation --------------------------------------------------
    #
    # Poser-only: ``root`` is set by the authoring session after bind, never
    # by pose_panel's entry path, so asset pose mode cannot reach any of this.
    # The offset previews live as ``node.translation = rest + delta`` -- the
    # exact arithmetic the bake performs -- with the rest translation
    # remembered on the Model the way rest_rotations already are.

    def _root_index(self) -> int | None:
        if self.model is None or self.root is None:
            return None
        return self.model.by_name.get(self.root)

    def _restore_root_rest(self) -> None:
        index = self._root_index()
        if index is None:
            return
        self.model.nodes[index].translation = self.model.rest_translations[index].copy()

    def move_root(self, point: Any) -> None:
        """Place the root joint during a translate drag.

        ``point`` is in model space -- the caller has already divided the
        placement out, the ``move_handle`` convention. The displacement is
        carried into the root's parent frame (where a glTF node translation
        lives) and soft-clamped to ±MAX_ROOT_TRANSLATION per component: on the
        canonical Poser armature model units are character heights literally,
        and an offset past two of them is a slipped drag, not a pose.
        """
        index = self._root_index()
        if index is None:
            return
        node = self.model.nodes[index]
        # parent_world = world @ local^-1: no parent map needed.
        parent_world = node.world @ np.linalg.inv(node.local())
        target = np.linalg.inv(parent_world) @ np.array(
            [float(point[0]), float(point[1]), float(point[2]), 1.0], dtype="f8"
        )
        rest = self.model.rest_translations[index]
        delta = np.clip(
            target[:3] - rest, -poselib.MAX_ROOT_TRANSLATION, poselib.MAX_ROOT_TRANSLATION
        )
        node.translation = rest + delta
        self.model.update_world()
        self._resync_handles()
        self.dirty = True

    def set_root_translation(self, v: Any, *, dirty: bool = True) -> None:
        """Set the root offset from a stored record: Blender axes,
        character-height units. Used when a saved pose is loaded back."""
        index = self._root_index()
        if index is None:
            return
        delta = np.clip(
            m3.blender_delta_to_gltf(np.asarray(v, dtype="f8")),
            -poselib.MAX_ROOT_TRANSLATION,
            poselib.MAX_ROOT_TRANSLATION,
        )
        node = self.model.nodes[index]
        node.translation = self.model.rest_translations[index] + delta
        self.model.update_world()
        self._resync_handles()
        self.dirty = dirty

    def root_translation(self) -> list[float]:
        """The current root offset -- Blender axes, character-height units.
        Zeros when nothing is bound or the root never moved, so a caller can
        always store what this returns."""
        index = self._root_index()
        if index is None:
            return [0.0, 0.0, 0.0]
        delta = self.model.nodes[index].translation - self.model.rest_translations[index]
        return [float(x) for x in m3.gltf_delta_to_blender(delta)]

    # -- joint placement ---------------------------------------------------

    def enter_joints_mode(self) -> None:
        # A rotation left over from posing would put the markers where the
        # *posed* bones are, and a joint correction is against the rest skeleton.
        self.reset_all(dirty=False)
        self.mode = "joints"
        self.moved.clear()
        self.selected = None
        self.home = {name: point.copy() for name, point in self.handles.items()}

    def exit_joints_mode(self) -> None:
        self.mode = "pose"
        self.moved.clear()
        self.selected = None
        self.revert_joints()

    def revert_joints(self) -> None:
        self.moved.clear()
        for name, point in self.home.items():
            self.handles[name] = point.copy()

    def move_handle(self, bone: str, position: np.ndarray) -> None:
        """Place a marker during a joints-mode drag and record its displacement."""
        if bone not in self.handles:
            return
        self.handles[bone] = np.asarray(position, dtype="f8").copy()
        home = self.home.get(bone)
        if home is None:
            return
        delta = self.handles[bone] - home
        if not np.any(delta):
            self.moved.pop(bone, None)
        else:
            # Recorded in Blender space, because that is the space rig.json's
            # joint positions are already in -- converting the *delta* rather
            # than the absolutes is what keeps this lossless.
            self.moved[bone] = [float(v) for v in m3.gltf_delta_to_blender(delta)]

    def corrected_bones(self) -> list[dict[str, Any]]:
        """The whole skeleton, fitted positions with the dragged joints substituted.

        A bone's tail follows its first child's head where one exists, and
        otherwise moves rigidly with its own head. That is the rule that keeps
        a dragged chain connected -- moving a knee has to shorten the thigh and
        lengthen the shin, not leave a gap where the thigh used to end.
        """

        def shift(point, name):
            d = self.moved.get(name)
            return [point[0] + d[0], point[1] + d[1], point[2] + d[2]] if d else list(point)

        heads = {b["name"]: shift(b["head"], b["name"]) for b in self.fitted}
        first_child: dict[str, str] = {}
        for bone in self.fitted:
            parent = bone.get("parent")
            if parent and parent not in first_child:
                first_child[parent] = bone["name"]
        return [
            {
                "name": bone["name"],
                "head": heads[bone["name"]],
                "tail": heads[first_child[bone["name"]]]
                if bone["name"] in first_child
                else shift(bone["tail"], bone["name"]),
            }
            for bone in self.fitted
        ]


def _skin_bones(model: Model) -> list[str]:
    """Every joint node of every skin, in palette order.

    Palette order rather than name order: it is the order the rig was built in,
    so a marker list reads root-outwards the way a skeleton does.
    """
    names: list[str] = []
    for skin in model.skins:
        for index in skin.joints:
            name = model.nodes[index].name
            if name and name not in names:
                names.append(name)
    return names
