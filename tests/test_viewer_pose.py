"""The pose editor and its handles, without a GL context.

The interesting contracts here are all conventions -- which way a mirror flips,
which space a joint delta is recorded in, which side a local rotation
multiplies on -- and every one of them is wrong in a way that still looks
plausible in a still image. So they are pinned as numbers next to the Python
the Blender worker actually uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from warlock import rigging
from warlock.studio.viewer import gizmo as gizmolib
from warlock.studio.viewer import markers, picking
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer.camera import Camera, screen_ray
from warlock.studio.viewer.pose import PoseEditor, mirror_quaternion


class FakeNode:
    def __init__(self, world):
        self.world = world


class FakeModel:
    """Just enough of gltf.Model for the editor: named nodes with rotations."""

    def __init__(self, names):
        self.by_name = {n: i for i, n in enumerate(names)}
        self.rotations = {n: m3.quat_identity() for n in names}
        self.nodes = [FakeNode(m3.translation(m3.vec3(i, 0, 0))) for i, _ in enumerate(names)]
        self.skins = []

    def get_rotation(self, bone):
        return self.rotations.get(bone)

    def set_rotation(self, bone, quat):
        if bone not in self.rotations:
            return False
        self.rotations[bone] = m3.quat_normalize(np.asarray(quat, dtype="f8"))
        return True

    def update_world(self):
        pass


@pytest.fixture
def editor():
    e = PoseEditor()
    e.bind(FakeModel(["hip", "upper_arm.L", "upper_arm.R"]), ["hip", "upper_arm.L", "upper_arm.R"])
    e.mirror_pairs = [["upper_arm.L", "upper_arm.R"]]
    return e


# --- the mirror -------------------------------------------------------------


def test_the_mirror_is_the_one_in_rigging_not_a_copy_of_it():
    """The browser kept its own and a comment insisting they stay identical.
    This is that comment made unnecessary."""
    assert mirror_quaternion is rigging.mirror_quaternion


def test_mirroring_flips_the_components_perpendicular_to_the_mirror_normal():
    assert mirror_quaternion([0.1, 0.2, 0.3, 0.9]) == [0.1, -0.2, -0.3, 0.9]


def test_mirroring_a_pose_swaps_both_sides_rather_than_overwriting_one(editor):
    left = m3.quat_from_axis_angle(m3.vec3(0, 0, 1), 0.5)
    right = m3.quat_from_axis_angle(m3.vec3(0, 0, 1), -0.2)
    editor.apply({"upper_arm.L": list(left), "upper_arm.R": list(right)})
    editor.mirror()
    posed = editor.pose()
    assert posed["upper_arm.R"] == pytest.approx(mirror_quaternion(left))
    assert posed["upper_arm.L"] == pytest.approx(mirror_quaternion(right))


def test_a_bone_with_no_partner_is_left_alone(editor):
    q = m3.quat_from_axis_angle(m3.vec3(1, 0, 0), 0.3)
    editor.apply({"hip": list(q)})
    editor.mirror()
    # A spine sits on the mirror plane; reflecting it would rotate a centred
    # limb off centre.
    assert editor.pose()["hip"] == pytest.approx(list(q))


# --- rotations --------------------------------------------------------------


def test_a_gizmo_delta_post_multiplies_onto_the_bones_local_rotation(editor):
    """Post-multiply is what makes the visible red ring turn the bone about
    *its* X rather than the world's."""
    start = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.4)
    editor.apply({"hip": list(start)})
    editor.selected = "hip"
    delta = m3.quat_from_axis_angle(m3.vec3(1, 0, 0), 0.25)
    editor.rotate_selected(delta)
    assert editor.pose()["hip"] == pytest.approx(list(m3.quat_mul(start, delta)), abs=1e-9)


def test_resetting_one_bone_leaves_the_others_posed(editor):
    editor.apply({"hip": [0, 0, 0.3, 0.954], "upper_arm.L": [0, 0.3, 0, 0.954]})
    editor.selected = "hip"
    editor.reset_bone()
    assert editor.pose()["hip"] == pytest.approx([0, 0, 0, 1])
    assert editor.pose()["upper_arm.L"] != pytest.approx([0, 0, 0, 1])


def test_applying_a_preset_clears_whatever_was_posed_before(editor):
    """A preset lists only the bones it moves, so applying 'idle' after 'wave'
    must not leave the arm up."""
    editor.apply({"upper_arm.L": [0, 0.3, 0, 0.954]})
    editor.apply_preset({"name": "idle", "bones": {"hip": [0, 0, 0.1, 0.995]}})
    assert editor.pose()["upper_arm.L"] == pytest.approx([0, 0, 0, 1])
    assert editor.pose()["hip"] != pytest.approx([0, 0, 0, 1])


def test_editing_marks_the_pose_dirty_and_a_reset_can_clear_it(editor):
    assert editor.has_unsaved_edits() is False
    editor.apply({"hip": [0, 0, 0.3, 0.954]})
    assert editor.has_unsaved_edits() is True
    editor.reset_all(dirty=False)
    assert editor.has_unsaved_edits() is False


# --- joint placement --------------------------------------------------------


@pytest.fixture
def joints_editor(editor):
    editor.fitted = [
        {"name": "hip", "head": [0, 0, 0.5], "tail": [0, 0, 0.7], "parent": None},
        {"name": "upper_arm.L", "head": [0.1, 0, 0.7], "tail": [0.3, 0, 0.7], "parent": "hip"},
    ]
    editor.enter_joints_mode()
    return editor


def test_a_joint_drag_is_recorded_as_a_blender_space_delta(joints_editor):
    """[x, y, z] -> [x, -z, y]. Deltas only: rig.json's positions are already
    Blender's, and round-tripping them would only lose precision."""
    home = joints_editor.home["hip"].copy()
    joints_editor.move_handle("hip", home + m3.vec3(0.5, 1.0, -2.0))
    assert joints_editor.moved["hip"] == pytest.approx([0.5, 2.0, 1.0])


def test_a_drag_back_to_where_it_started_is_not_a_move(joints_editor):
    home = joints_editor.home["hip"].copy()
    joints_editor.move_handle("hip", home + m3.vec3(0.1, 0, 0))
    assert "hip" in joints_editor.moved
    joints_editor.move_handle("hip", home)
    assert "hip" not in joints_editor.moved


def test_a_bones_tail_follows_its_first_childs_head(joints_editor):
    """The rule that keeps a dragged chain connected: moving a knee shortens
    the thigh and lengthens the shin rather than leaving a gap."""
    joints_editor.moved["upper_arm.L"] = [0.0, 0.0, 0.2]
    corrected = {b["name"]: b for b in joints_editor.corrected_bones()}
    moved_head = corrected["upper_arm.L"]["head"]
    assert moved_head == pytest.approx([0.1, 0, 0.9])
    # The parent's tail is its child's head, not its own recorded tail.
    assert corrected["hip"]["tail"] == pytest.approx(moved_head)


def test_a_leaf_bones_tail_moves_rigidly_with_its_head(joints_editor):
    joints_editor.moved["upper_arm.L"] = [0.0, 0.0, 0.2]
    corrected = {b["name"]: b for b in joints_editor.corrected_bones()}
    assert corrected["upper_arm.L"]["tail"] == pytest.approx([0.3, 0, 0.9])


def test_the_correction_is_the_whole_skeleton_not_only_what_moved(joints_editor):
    joints_editor.moved["hip"] = [0.0, 0.0, 0.1]
    assert len(joints_editor.corrected_bones()) == len(joints_editor.fitted)


def test_entering_joints_mode_resets_the_pose_first(editor):
    """The markers must show the *rest* skeleton: a correction is against it."""
    editor.apply({"hip": [0, 0, 0.3, 0.954]})
    editor.fitted = [{"name": "hip", "head": [0, 0, 0], "tail": [0, 0, 1], "parent": None}]
    editor.enter_joints_mode()
    assert editor.pose()["hip"] == pytest.approx([0, 0, 0, 1])
    assert editor.has_unsaved_edits() is False


def test_reverting_puts_every_handle_back(joints_editor):
    home = joints_editor.home["hip"].copy()
    joints_editor.move_handle("hip", home + m3.vec3(1, 1, 1))
    joints_editor.revert_joints()
    assert joints_editor.moved == {}
    assert joints_editor.handles["hip"] == pytest.approx(home)


# --- picking ----------------------------------------------------------------


def test_a_ray_hits_a_sphere_in_front_and_misses_one_behind():
    origin, direction = m3.vec3(0, 0, 5), m3.vec3(0, 0, -1)
    assert picking.ray_sphere(origin, direction, m3.vec3(0, 0, 0), 1.0) == pytest.approx(4.0)
    assert picking.ray_sphere(origin, direction, m3.vec3(0, 0, 10), 1.0) is None
    assert picking.ray_sphere(origin, direction, m3.vec3(5, 0, 0), 1.0) is None


def test_the_nearest_of_several_markers_wins():
    origin, direction = m3.vec3(0, 0, 10), m3.vec3(0, 0, -1)
    centres = {"far": m3.vec3(0, 0, -5), "near": m3.vec3(0, 0, 5)}
    assert picking.nearest_hit(origin, direction, centres, 1.0) == "near"


def test_a_screen_ray_through_the_centre_points_at_the_target():
    camera = Camera()
    camera.set_target(m3.vec3(0, 0, 0))
    camera.aspect = 1.0
    origin, direction = screen_ray(camera, 64, 64, 128, 128)
    to_target = -origin / np.linalg.norm(origin)
    assert direction == pytest.approx(to_target, abs=1e-9)


def test_a_screen_ray_at_the_edge_leans_the_right_way():
    camera = Camera()
    camera.theta, camera.phi, camera.distance = 0.0, math.pi / 2, 5.0
    camera.set_target(m3.vec3(0, 0, 0))
    camera.aspect = 1.0
    _origin, right = screen_ray(camera, 127, 64, 128, 128)
    # Camera at +Z looking at the origin: screen-right is world +X.
    assert right[0] > 0


def test_an_orthographic_screen_ray_is_parallel_and_offset_not_fanned():
    """Clay exposes the ortho toggle and then picks through this function.

    Perspective-only, every ray was cast from a single apex the orthographic
    render does not have, so the further a click landed from screen centre the
    further what got picked was from what was drawn under the cursor.
    """
    camera = Camera()
    camera.theta, camera.phi, camera.distance = 0.0, math.pi / 2, 5.0
    camera.set_target(m3.vec3(0, 0, 0))
    camera.aspect = 1.0
    camera.orthographic = True

    centre_o, centre_d = screen_ray(camera, 64, 64, 128, 128)
    edge_o, edge_d = screen_ray(camera, 127, 64, 128, 128)

    # Parallel: the direction is the view direction wherever the click landed.
    assert edge_d == pytest.approx(centre_d, abs=1e-9)
    assert centre_d == pytest.approx(m3.vec3(0, 0, -1), abs=1e-9)
    # It is the origin that moves, and screen-right is world +X from here.
    assert edge_o[0] > centre_o[0]
    # Sized like projection(): the half-width at the target plane is
    # distance * tan(fov/2), and the click one pixel short of the right edge is
    # just inside it.
    half = 5.0 * math.tan(math.radians(camera.fov * 0.5))
    assert 0 < edge_o[0] < half


def test_an_orthographic_ray_starts_behind_everything_the_render_draws():
    """projection() deliberately puts the near plane behind the eye, because an
    orthographic frustum has no apex to clip against. A ray starting at the eye
    would miss exactly the geometry that choice exists to keep visible."""
    camera = Camera()
    camera.theta, camera.phi, camera.distance = 0.0, math.pi / 2, 5.0
    camera.set_target(m3.vec3(0, 0, 0))
    camera.aspect = 1.0
    camera.orthographic = True

    origin, direction = screen_ray(camera, 64, 64, 128, 128)
    # A point between the target and the eye is still in front of the ray.
    between = m3.vec3(0, 0, 2.5)
    assert float(np.dot(between - origin, direction)) > 0


def test_a_perspective_screen_ray_is_unchanged_by_the_orthographic_branch():
    camera = Camera()
    camera.theta, camera.phi, camera.distance = 0.0, math.pi / 2, 5.0
    camera.set_target(m3.vec3(0, 0, 0))
    camera.aspect = 1.0
    assert camera.orthographic is False

    origin, direction = screen_ray(camera, 127, 64, 128, 128)
    assert origin == pytest.approx(camera.position)
    assert direction[0] > 0


def test_a_mirrored_camera_copies_the_projection_too():
    """"Mirror another camera exactly" -- the compare view's premise is that
    both halves are the same picture of two meshes."""
    left, right = Camera(), Camera()
    left.orthographic = True
    right.copy_from(left)
    assert right.orthographic is True


def test_the_signed_angle_between_two_spokes_has_a_sign():
    axis = m3.vec3(0, 0, 1)
    a, b = m3.vec3(1, 0, 0), m3.vec3(0, 1, 0)
    assert picking.signed_angle(a, b, axis) == pytest.approx(math.pi / 2)
    assert picking.signed_angle(b, a, axis) == pytest.approx(-math.pi / 2)


def test_closest_on_axis_finds_the_crossing_point():
    # A ray down -Y crossing the X axis at x = 3.
    s, distance = picking.closest_on_axis(
        m3.vec3(3, 5, 0), m3.vec3(0, -1, 0), m3.vec3(0, 0, 0), m3.vec3(1, 0, 0)
    )
    assert s == pytest.approx(3.0)
    assert distance == pytest.approx(0.0, abs=1e-9)


def test_a_marker_is_a_fixed_fraction_of_the_model():
    assert picking.marker_radius(2.0) == pytest.approx(0.044)


def test_a_gizmo_keeps_its_screen_size_as_the_camera_pulls_back():
    camera = Camera()
    camera.set_target(m3.vec3(0, 0, 0))
    camera.distance = 5.0
    near = picking.screen_scale(camera, m3.vec3(0, 0, 0), 90.0, 600)
    camera.distance = 20.0
    far = picking.screen_scale(camera, m3.vec3(0, 0, 0), 90.0, 600)
    assert far == pytest.approx(near * 4.0, rel=1e-6)


# --- gizmo geometry and drags -----------------------------------------------


class FakeGizmo(gizmolib.RotateGizmo):
    """The drag maths without a GL context: geometry is never asked for."""

    def __init__(self):
        self.hover = None
        self.drag = None
        self.origin = m3.vec3()
        self.basis = m3.identity()
        self.scale = 1.0


def test_a_rotate_ring_is_hit_on_its_line_not_inside_it():
    g = FakeGizmo()
    # The Z ring lies in the XY plane; look down -Z from +Z.
    origin, direction = m3.vec3(1.0, 0, 5), m3.vec3(0, 0, -1)
    assert g.hit(origin, direction) == "z"
    # Through the middle: inside every ring, on none of them.
    assert g.hit(m3.vec3(0, 0, 5), m3.vec3(0, 0, -1)) is None


def test_a_rotate_drag_produces_an_incremental_local_quaternion():
    g = FakeGizmo()
    assert g.begin("z", m3.vec3(1.0, 0, 5), m3.vec3(0, 0, -1))
    # A quarter turn round the ring, in two steps.
    first = g.update(m3.vec3(0.7071, 0.7071, 5), m3.vec3(0, 0, -1))
    second = g.update(m3.vec3(0.0, 1.0, 5), m3.vec3(0, 0, -1))
    assert first is not None and second is not None
    total = m3.quat_mul(first, second)
    expected = m3.quat_from_axis_angle(m3.vec3(0, 0, 1), math.pi / 2)
    assert m3.quat_to_mat4(total) == pytest.approx(m3.quat_to_mat4(expected), abs=1e-4)


def test_a_rotate_drag_that_does_not_move_reports_nothing():
    g = FakeGizmo()
    g.begin("z", m3.vec3(1.0, 0, 5), m3.vec3(0, 0, -1))
    assert g.update(m3.vec3(1.0, 0, 5), m3.vec3(0, 0, -1)) is None


def test_the_ring_axes_follow_the_targets_own_frame():
    """Local space is what makes 'bend the elbow' a single-ring drag."""
    g = FakeGizmo()
    g.basis = m3.quat_to_mat4(m3.quat_from_axis_angle(m3.vec3(1, 0, 0), math.pi / 2))
    # The joint's Z now points along world -Y, so the Z ring lies in the XZ plane.
    assert g._axis_world("z") == pytest.approx([0, -1, 0], abs=1e-12)


def test_a_scaled_armature_does_not_scale_the_gizmo():
    frame = gizmolib._frame(m3.vec3(1, 2, 3), m3.scaling(5.0), 0.25)
    assert np.linalg.norm(frame[:3, 0]) == pytest.approx(0.25)
    assert frame[:3, 3] == pytest.approx([1, 2, 3])


class FakeTranslate(gizmolib.TranslateGizmo):
    def __init__(self):
        self.hover = None
        self.drag = None
        self.origin = m3.vec3()
        self.basis = m3.identity()
        self.scale = 1.0


def test_a_translate_arrow_is_hit_along_its_own_half_only():
    g = FakeTranslate()
    # Just above the +X arrow's midpoint, looking down.
    assert g.hit(m3.vec3(0.5, 5, 0), m3.vec3(0, -1, 0)) == "x"
    # Past its tip: the drawn arrow is the grabbable part.
    assert g.hit(m3.vec3(3.0, 5, 0), m3.vec3(0, -1, 0)) is None
    # Behind the origin, where nothing is drawn.
    assert g.hit(m3.vec3(-0.5, 5, 0), m3.vec3(0, -1, 0)) is None


def test_a_translate_drag_is_constrained_to_its_axis():
    g = FakeTranslate()
    g.begin("x", m3.vec3(0.5, 5, 0), m3.vec3(0, -1, 0))
    moved = g.update(m3.vec3(2.5, 5, 3), m3.vec3(0, -1, 0))
    assert moved == pytest.approx([2.0, 0.0, 0.0])


def test_a_ring_closes_on_itself():
    verts = gizmolib.ring("x")
    assert verts[0] == pytest.approx(verts[-1])
    assert (np.abs(verts[:, 0]) < 1e-6).all()


def test_a_marker_sphere_is_made_of_whole_triangles():
    verts = markers.sphere(1.0)
    assert len(verts) % 3 == 0
    radii = np.linalg.norm(verts, axis=1)
    assert radii.max() == pytest.approx(1.0, abs=1e-6)
