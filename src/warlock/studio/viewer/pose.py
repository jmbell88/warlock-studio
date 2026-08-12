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

import contextlib
import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ... import poselib, rigging
from ..undo import Edit, UndoStack
from . import math3d as m3
from .gltf import Model

# Re-exported so nothing downstream is tempted to write the sign flip out again.
mirror_quaternion = rigging.mirror_quaternion


@dataclass
class PoseSnapshotEdit(Edit):
    """One reversible pose step, as a pair of whole-editor snapshots.

    Whole snapshots rather than a diff, which is the opposite of what the
    raster editor does and is right for the same reason: a ``PatchEdit`` stores
    a rectangle because a canvas is megabytes, and a pose is a few dozen
    quaternions -- under a kilobyte for the canonical armature. A diff would buy
    nothing and would have to describe six different kinds of change (a
    rotation, the root translation, a joint drag, the mode, the identity of the
    pose being edited, the dirty flag) in six ways, any one of which could be
    forgotten when a seventh is added.

    ``cost`` is measured rather than assumed, so the stack's byte budget is
    honest about a two-hundred-bone rig.
    """

    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cost = _snapshot_cost(self.before) + _snapshot_cost(self.after)

    def undo(self, doc: Any) -> None:
        doc.restore(self.before)

    def redo(self, doc: Any) -> None:
        doc.restore(self.after)


def _undoable(method):
    """Mark a discrete operation as one undo step.

    The decorator rather than a ``with`` inside each body, because the thing
    being asserted is uniform -- "this whole call is one step" -- and writing
    it out seven times is seven chances to indent one of them wrong. The
    gesture-level mutators (``rotate_selected``, ``move_root``,
    ``move_handle``) are deliberately *not* decorated: each is called once per
    frame of a drag, and a step per frame is a stack full of one gesture. The
    pane brackets those with ``record()`` across the whole grab instead.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.record():
            return method(self, *args, **kwargs)

    return wrapper


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two snapshots describe the same editor state.

    Written out rather than ``a == b`` because the handle maps hold numpy
    arrays, whose ``==`` is an array and whose truth value raises. Selection is
    deliberately *not* compared: clicking a different marker changes what the
    gizmo points at and nothing about the pose, and a stack that recorded it
    would spend a user's undo presses putting the highlight back.
    """
    for key in ("rotations", "moved", "root_translation", "mode", "current", "dirty"):
        if a.get(key) != b.get(key):
            return False
    for key in ("handles", "home"):
        left, right = a.get(key) or {}, b.get(key) or {}
        if left.keys() != right.keys():
            return False
        if any(not np.array_equal(left[name], right[name]) for name in left):
            return False
    return True


def _snapshot_cost(snap: dict[str, Any]) -> int:
    """Bytes, roughly: eight per float, which is what the arrays hold."""
    rotations = len(snap.get("rotations") or ()) * 4 * 8
    handles = len(snap.get("handles") or ()) * 3 * 8
    home = len(snap.get("home") or ()) * 3 * 8
    moved = len(snap.get("moved") or ()) * 3 * 8
    return rotations + handles + home + moved + 128


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
        # Undo, per pose *session*. Dropped by ``bind`` and ``clear`` rather
        # than carried, because adopting a different model rebuilds ``bones``
        # and ``rest`` -- a surviving step would restore rotations onto a
        # skeleton that never had them, by name, silently.
        self.history = UndoStack()
        # Re-entrancy depth for ``record``; nonzero means a step is open.
        self._depth = 0
        # Bumped by every ``_reset_history``. An open ``record`` that spans a
        # rebind -- a drag live when the model is adopted -- compares this and
        # declines to push, because its 'before' describes a skeleton that is
        # no longer here.
        self._generation = 0

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
        self._reset_history()

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
        self._reset_history()

    def _reset_history(self) -> None:
        self.history.clear()
        self._depth = 0
        self._generation += 1

    @property
    def bound(self) -> bool:
        return self.model is not None and bool(self.bones)

    def has_unsaved_edits(self) -> bool:
        return bool(self.dirty or self.moved)

    # -- undo ---------------------------------------------------------------
    #
    # The stack lives here rather than in the pane because both entry points
    # into pose editing -- Poser's authoring session and the inspector's asset
    # pose mode -- want the same history over the same object, and a pane that
    # owned it would own it twice. ``studio/undo`` is the shared engine Clay
    # already borrows; nothing in it is about pixels.
    #
    # A *snapshot* is the unit, not a diff. See ``PoseSnapshotEdit``.

    def snapshot(self) -> dict[str, Any]:
        """Everything an edit can change, as plain data owned by the caller.

        Copies throughout, for ``undo``'s rule that an edit owns its data: the
        handle arrays are live numpy state, and storing views of them would
        make an undo restore whatever the value happened to become.
        """
        model = self.model
        return {
            "rotations": {} if model is None else {
                bone: [float(v) for v in (m3.quat_identity() if q is None else q)]
                for bone, q in ((b, model.get_rotation(b)) for b in self.bones)
            },
            "root_translation": self.root_translation(),
            "moved": {name: list(delta) for name, delta in self.moved.items()},
            "handles": {name: point.copy() for name, point in self.handles.items()},
            "home": {name: point.copy() for name, point in self.home.items()},
            "mode": self.mode,
            "selected": self.selected,
            "current": self.current,
            "dirty": self.dirty,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Put the editor back to a snapshot. The other half of an edit.

        ``dirty`` is restored rather than set, which is the whole reason it is
        in the snapshot: undoing back to the state a save was taken at has to
        leave the session clean, or the quit guard asks about work the user
        already stored.
        """
        if self.model is None:
            return
        for bone, quat in (snap.get("rotations") or {}).items():
            if bone in self.model.by_name:
                self.model.set_rotation(bone, quat)
        # After the rotations and before ``update_world``: the root's offset is
        # a node translation, and both feed the same matrix.
        if self.root is not None:
            self.set_root_translation(snap.get("root_translation") or [0.0, 0.0, 0.0])
        self.model.update_world()
        self.mode = snap.get("mode", "pose")
        self.moved = {name: list(delta) for name, delta in (snap.get("moved") or {}).items()}
        self.home = {name: point.copy() for name, point in (snap.get("home") or {}).items()}
        # Handles last and verbatim, because ``_resync_handles`` deliberately
        # refuses to touch them in joints mode -- where the marker *is* the edit.
        self._resync_handles()
        self.handles = {
            name: point.copy() for name, point in (snap.get("handles") or {}).items()
        }
        self.selected = snap.get("selected")
        self.current = snap.get("current")
        self.dirty = bool(snap.get("dirty"))

    @contextlib.contextmanager
    def record(self):
        """Bracket one undoable step. Re-entrant; only the outermost pushes.

        Re-entrant because the discrete operations compose -- ``apply_preset``
        is a ``reset_all`` and an ``apply``, and a user pressing a preset made
        one edit, not two. A gizmo drag brackets differently: the pane opens
        this on grab and closes it on release, so the hundreds of intermediate
        ``rotate_selected`` calls collapse into the single step the gesture was.

        Nothing is pushed when nothing changed, so a click that selects a bone
        and a drag that ends where it started both leave the stack alone.
        """
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        before = self.snapshot()
        generation = self._generation
        self._depth = 1
        try:
            yield
        finally:
            self._depth = 0
        if generation != self._generation:
            # The session was rebound or torn down inside the step. Pushing now
            # would file a rotation map keyed on the *previous* skeleton's bone
            # names, which restores silently and wrongly onto whatever shares a
            # name with it.
            return
        after = self.snapshot()
        if not _same(before, after):
            self.history.push(PoseSnapshotEdit(before, after))

    def undo(self) -> bool:
        return self.history.undo(self)

    def redo(self) -> bool:
        return self.history.redo(self)

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

    @_undoable
    def apply_preset(self, preset: dict[str, Any]) -> None:
        """A preset lists only the bones it moves, so the rest is reset first
        -- otherwise applying "idle" after "wave" leaves the arm up."""
        self.reset_all(dirty=True)
        self.apply(preset.get("bones", {}), pose_id=None, dirty=True)

    @_undoable
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

    @_undoable
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

    @_undoable
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

    @_undoable
    def enter_joints_mode(self) -> None:
        # A rotation left over from posing would put the markers where the
        # *posed* bones are, and a joint correction is against the rest skeleton.
        self.reset_all(dirty=False)
        self.mode = "joints"
        self.moved.clear()
        self.selected = None
        self.home = {name: point.copy() for name, point in self.handles.items()}

    @_undoable
    def exit_joints_mode(self) -> None:
        self.mode = "pose"
        self.moved.clear()
        self.selected = None
        self.revert_joints()

    @_undoable
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
