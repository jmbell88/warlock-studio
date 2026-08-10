"""The viewer half of Poser: explicit-bone binding, bone lines, root translate.

Everything here runs over a hand-built skinless ``gltf.Model`` -- the exact
shape the meshless armature preview loads as -- with no GL context anywhere:
``bone_segments`` is pure, the editor is pure, and the two ``Viewer`` methods
under test are exercised unbound over a stub, the pattern that keeps a GL
object out of a logic test.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import poselib
from warlock.studio.viewer import bonelines
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.camera import Camera
from warlock.studio.viewer.gltf import Model, Node
from warlock.studio.viewer.pose import PoseEditor
from warlock.studio.viewer_embed import Viewer

BONES = ["hips", "spine", "arm.L", "arm.R"]


def _armature_model() -> Model:
    """A meshless armature: a 'rig' object node over a small joint tree --
    what op_armature's export loads as, minus the file."""
    nodes = [
        Node(name="rig", children=[1]),
        Node(name="hips", translation=m3.vec3(0.0, 0.53, 0.0), children=[2, 3, 4]),
        Node(name="spine", translation=m3.vec3(0.0, 0.07, 0.0)),
        Node(name="arm.L", translation=m3.vec3(-0.2, 0.1, 0.0)),
        Node(name="arm.R", translation=m3.vec3(0.2, 0.1, 0.0)),
    ]
    return Model(nodes, roots=[0], meshes=[], skins=[])


def _editor() -> PoseEditor:
    editor = PoseEditor()
    editor.bind(_armature_model(), BONES)
    editor.mirror_pairs = [["arm.L", "arm.R"]]
    editor.root = "hips"
    return editor


# --- bone segments ----------------------------------------------------------


def test_bone_segments_connect_a_bone_to_its_bound_parent():
    model = _armature_model()
    pairs = bonelines.bone_segments(model, BONES)
    assert pairs == [("hips", "spine"), ("hips", "arm.L"), ("hips", "arm.R")]


def test_the_root_gets_no_segment_because_its_parent_is_not_a_bone():
    """The armature object node above the root is a helper, not a joint; a
    line up to it would draw a bone that does not exist."""
    model = _armature_model()
    pairs = bonelines.bone_segments(model, BONES)
    assert all(bone != "hips" for _, bone in pairs)


def test_bone_segments_ignore_names_the_model_lacks():
    model = _armature_model()
    assert bonelines.bone_segments(model, ["hips", "tail_99"]) == []


# --- explicit-bone binding --------------------------------------------------


class _ViewerStub:
    """Just the attributes enter_pose_authoring touches."""

    def __init__(self, model) -> None:
        self.model = model
        self.editor = PoseEditor()
        self.pose_mode = False
        self.pose_job_id = None
        self._bone_pairs: list = []
        self._render_dirty = False
        self.dirty_reports: list = []

    def _notify_pose_dirty(self) -> None:
        self.dirty_reports.append(True)


def test_enter_pose_authoring_binds_explicit_bones_over_a_skinless_model():
    """enter_pose_mode's skins check is its contract and stays intact; this is
    the separate door the Poser walks through, with the bone list supplied."""
    stub = _ViewerStub(_armature_model())
    ok = Viewer.enter_pose_authoring(
        stub, BONES, [["arm.L", "arm.R"]], "poser:humanoid"
    )
    assert ok is True
    assert stub.pose_mode is True
    assert stub.pose_job_id == "poser:humanoid"
    assert stub.editor.bones == BONES
    assert stub.editor.mirror_pairs == [["arm.L", "arm.R"]]
    assert stub._bone_pairs == bonelines.bone_segments(stub.model, BONES)
    assert stub.dirty_reports, "binding reports the (clean) editor state"


def test_the_token_can_never_be_a_job_id():
    """pose_job_id carries 'poser:<template>' in the authoring session --
    belt-and-braces under the separate Viewer instance, because a 12-hex job
    id can never contain a colon."""
    from warlock import rigging

    assert not rigging.is_valid_id("poser:humanoid")


def test_enter_pose_authoring_refuses_with_no_model():
    stub = _ViewerStub(None)
    assert Viewer.enter_pose_authoring(stub, BONES, [], "poser:humanoid") is False
    assert stub.pose_mode is False


# --- root translation -------------------------------------------------------


def test_move_root_offsets_the_node_and_reports_blender_units():
    editor = _editor()
    assert editor.root_translation() == [0.0, 0.0, 0.0]
    assert editor.dirty is False

    # hips sits at glTF (0, 0.53, 0) under an identity parent, so a target of
    # (0.1, 0.63, -0.2) is a delta of (0.1, 0.1, -0.2) -- which in Blender
    # axes reads (0.1, 0.2, 0.1).
    editor.move_root(np.array([0.1, 0.63, -0.2]))
    assert editor.dirty is True
    assert editor.root_translation() == pytest.approx([0.1, 0.2, 0.1])
    # The handles follow: the marker is the drag handle.
    assert editor.handles["hips"] == pytest.approx([0.1, 0.63, -0.2])
    # And the children ride along with their parent.
    assert editor.handles["spine"] == pytest.approx([0.1, 0.70, -0.2])


def test_move_root_soft_clamps_each_component():
    editor = _editor()
    editor.move_root(np.array([50.0, 0.53, 0.0]))
    assert editor.root_translation() == pytest.approx(
        [poselib.MAX_ROOT_TRANSLATION, 0.0, 0.0]
    )


def test_set_root_translation_round_trips():
    editor = _editor()
    editor.set_root_translation([0.1, -0.2, 0.3])
    assert editor.root_translation() == pytest.approx([0.1, -0.2, 0.3])


def test_reset_all_restores_the_rest_translation():
    editor = _editor()
    editor.move_root(np.array([0.3, 0.53, 0.0]))
    editor.reset_all(dirty=False)
    assert editor.root_translation() == [0.0, 0.0, 0.0]
    assert editor.handles["hips"] == pytest.approx([0.0, 0.53, 0.0])
    assert editor.dirty is False


def test_reset_bone_on_the_root_restores_its_translation_too():
    editor = _editor()
    editor.move_root(np.array([0.3, 0.53, 0.0]))
    editor.reset_bone("hips")
    assert editor.root_translation() == [0.0, 0.0, 0.0]


def test_apply_preset_is_zero_translation():
    """A preset lists rotations only; applying one after a root move must not
    keep the character floating where the last pose left it."""
    editor = _editor()
    editor.move_root(np.array([0.0, 1.0, 0.0]))
    editor.apply_preset({"name": "idle", "bones": {"spine": [0, 0, 0, 1]}})
    assert editor.root_translation() == [0.0, 0.0, 0.0]


def test_mirror_reflects_the_root_translation():
    editor = _editor()
    editor.set_root_translation([0.3, 0.1, 0.2])
    editor.mirror()
    assert editor.root_translation() == pytest.approx([-0.3, 0.1, 0.2])


def test_mirror_keeps_the_pose_being_edited():
    """Mirror goes through apply, whose clearing of ``current`` is for
    *loading* -- a mirror is still the same pose, so it stays addressed and
    the next Save saves rather than silently becoming Save-as."""
    editor = _editor()
    editor.apply({"arm.L": [0.0, 0.0, 0.7071068, 0.7071068]}, pose_id="0123456789ab")
    editor.mirror()
    assert editor.current == "0123456789ab"
    assert editor.dirty is True


def test_reset_all_clears_the_pose_being_edited():
    """new_pose's half of the pair: back to rest is back to nothing-open."""
    editor = _editor()
    editor.apply({"arm.L": [0.0, 0.0, 0.7071068, 0.7071068]}, pose_id="0123456789ab")
    editor.reset_all(dirty=False)
    assert editor.current is None


def test_without_a_root_nothing_moves():
    """pose_panel's entry path never sets editor.root, which is what keeps
    asset pose mode behaviourally unchanged."""
    editor = _editor()
    editor.root = None
    editor.move_root(np.array([5.0, 5.0, 5.0]))
    editor.set_root_translation([1.0, 1.0, 1.0])
    assert editor.root_translation() == [0.0, 0.0, 0.0]
    assert editor.dirty is False


def test_a_root_move_is_an_unsaved_edit():
    editor = _editor()
    editor.move_root(np.array([0.1, 0.53, 0.0]))
    assert editor.has_unsaved_edits()


# --- framing ----------------------------------------------------------------


class _RendererStub:
    def __init__(self) -> None:
        self.fitted = None

    def fit_grid(self, lo, hi) -> None:
        self.fitted = (np.asarray(lo), np.asarray(hi))


def test_frame_bounds_frames_the_supplied_box_not_the_models():
    """A meshless armature's Model.bounds() is zeros, which frame() clamps to
    a 1e-4 radius -- markers invisible, picking dead. frame_bounds takes the
    host-computed box over the fitted bones instead."""

    class Stub:
        pass

    stub = Stub()
    stub.camera = Camera()
    stub.renderer = _RendererStub()
    stub.radius = 0.0
    stub._render_dirty = False

    radius = Viewer.frame_bounds(stub, [-0.55, 0.0, -0.55], [0.55, 1.1, 0.55])
    assert radius == stub.radius
    assert stub.radius > 0.5, "a unit-character box frames at a usable radius"
    assert stub.renderer.fitted is not None
    assert stub._render_dirty is True
