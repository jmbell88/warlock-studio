"""The two conventions the whole viewer rests on, pinned.

No GL here on purpose: if a limb bends the wrong way, it should be provable
from these numbers before anything is rendered.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from warlock.studio.viewer import math3d as m3


def test_a_mat4_uniform_is_transposed_on_its_way_to_glsl():
    """GLSL reads a mat4's memory column-major; we store row-major. That
    transpose is the only place the two orders are allowed to meet."""
    mat = m3.translation(m3.vec3(1, 2, 3))
    floats = np.frombuffer(m3.gl_bytes(mat), dtype="f4")
    # Column-major: the translation lands in elements 12..14, which is exactly
    # where a GLSL `mat4[3].xyz` reads it from.
    assert list(floats[12:15]) == [1.0, 2.0, 3.0]


def test_translation_lives_in_the_last_column():
    mat = m3.translation(m3.vec3(1, 2, 3))
    assert list(mat[:3, 3]) == [1.0, 2.0, 3.0]
    point = mat @ np.array([0.0, 0.0, 0.0, 1.0])
    assert list(point[:3]) == [1.0, 2.0, 3.0]


def test_a_quaternion_is_xyzw():
    """Same order as THREE.Quaternion.toArray() and as every pose on disk."""
    q = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), math.pi / 2)
    assert q[3] == pytest.approx(math.cos(math.pi / 4))
    assert q[1] == pytest.approx(math.sin(math.pi / 4))
    assert q[0] == q[2] == 0


def test_a_quarter_turn_about_y_sends_x_to_minus_z():
    q = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), math.pi / 2)
    out = m3.quat_rotate(q, m3.vec3(1, 0, 0))
    assert out == pytest.approx([0, 0, -1], abs=1e-12)


def test_quat_multiplication_applies_the_right_hand_operand_first():
    """Same order as a matrix product, which is what makes
    quat_to_mat4(a*b) == quat_to_mat4(a) @ quat_to_mat4(b) hold."""
    a = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.7)
    b = m3.quat_from_axis_angle(m3.vec3(1, 0, 0), -0.4)
    assert m3.quat_to_mat4(m3.quat_mul(a, b)) == pytest.approx(
        m3.quat_to_mat4(a) @ m3.quat_to_mat4(b), abs=1e-12
    )


def test_a_quaternion_times_its_conjugate_is_identity():
    q = m3.quat_normalize(np.array([0.3, -0.5, 0.2, 0.8]))
    assert m3.quat_mul(q, m3.quat_conjugate(q)) == pytest.approx([0, 0, 0, 1], abs=1e-12)


def test_compose_and_decompose_round_trip():
    t = m3.vec3(1.5, -2.0, 0.25)
    q = m3.quat_normalize(np.array([0.2, 0.3, -0.1, 0.9]))
    s = m3.vec3(2.0, 0.5, 3.0)
    t2, q2, s2 = m3.decompose(m3.compose(t, q, s))
    assert t2 == pytest.approx(t)
    assert s2 == pytest.approx(s)
    # A quaternion and its negation are the same rotation.
    assert m3.quat_to_mat4(q2) == pytest.approx(m3.quat_to_mat4(q), abs=1e-12)


def test_compose_orders_the_scale_innermost():
    """T * R * S: a glTF node scales in its own frame, then rotates, then
    moves. The other order would move the translation by the scale."""
    mat = m3.compose(m3.vec3(10, 0, 0), m3.quat_identity(), m3.vec3(2, 2, 2))
    out = mat @ np.array([1.0, 0.0, 0.0, 1.0])
    assert list(out[:3]) == [12.0, 0.0, 0.0]


def test_mat3_to_quat_survives_every_diagonal_branch():
    """Shepperd's method picks a different branch per axis; each has to work."""
    for axis in (m3.vec3(1, 0, 0), m3.vec3(0, 1, 0), m3.vec3(0, 0, 1)):
        for angle in (0.1, math.pi - 0.05, math.pi * 0.75):
            q = m3.quat_from_axis_angle(axis, angle)
            back = m3.mat3_to_quat(m3.quat_to_mat4(q)[:3, :3])
            assert m3.quat_to_mat4(back) == pytest.approx(m3.quat_to_mat4(q), abs=1e-9)


def test_slerp_takes_the_short_way_round():
    a = m3.quat_identity()
    b = -m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.4)  # same rotation, negated
    mid = m3.quat_slerp(a, b, 0.5)
    expected = m3.quat_from_axis_angle(m3.vec3(0, 1, 0), 0.2)
    assert m3.quat_to_mat4(mid) == pytest.approx(m3.quat_to_mat4(expected), abs=1e-9)


def test_a_perspective_matrix_puts_near_and_far_on_the_clip_bounds():
    proj = m3.perspective(45.0, 16 / 9, 0.1, 100.0)
    for depth, expected in ((-0.1, -1.0), (-100.0, 1.0)):
        clip = proj @ np.array([0.0, 0.0, depth, 1.0])
        assert clip[2] / clip[3] == pytest.approx(expected, abs=1e-6)


def test_look_at_puts_the_eye_at_the_view_origin():
    view = m3.look_at(m3.vec3(3, 4, 5), m3.vec3(0, 1, 0), m3.vec3(0, 1, 0))
    eye_in_view = view @ np.array([3.0, 4.0, 5.0, 1.0])
    assert eye_in_view[:3] == pytest.approx([0, 0, 0], abs=1e-12)


def test_look_at_survives_a_target_directly_overhead():
    """Straight up is where the naive cross product collapses, and a NaN there
    blanks the whole frame rather than merely looking wrong."""
    view = m3.look_at(m3.vec3(0, 0, 0), m3.vec3(0, 5, 0), m3.vec3(0, 1, 0))
    assert np.isfinite(view).all()


def test_look_at_survives_a_zero_length_forward():
    view = m3.look_at(m3.vec3(2, 2, 2), m3.vec3(2, 2, 2), m3.vec3(0, 1, 0))
    assert np.isfinite(view).all()


def test_a_gltf_delta_becomes_a_blender_delta_and_back():
    """[x, y, z] -> [x, -z, y]. Deltas only: an absolute position in rig.json
    is already Blender's, and round-tripping it would only lose precision."""
    d = m3.vec3(0.5, 1.0, -2.0)
    assert list(m3.gltf_delta_to_blender(d)) == [0.5, 2.0, 1.0]
    assert m3.blender_delta_to_gltf(m3.gltf_delta_to_blender(d)) == pytest.approx(d)


def test_an_ortho_matrix_maps_its_box_onto_the_clip_cube():
    proj = m3.orthographic(-2, 2, -1, 1, 0.5, 10.0)
    corner = proj @ np.array([2.0, 1.0, -0.5, 1.0])
    assert corner[:3] == pytest.approx([1, 1, -1], abs=1e-12)


def test_the_joint_palette_is_packed_as_a_flat_column_major_array():
    mats = [m3.translation(m3.vec3(i, 0, 0)) for i in range(3)]
    floats = np.frombuffer(m3.gl_bytes_many(mats), dtype="f4")
    assert len(floats) == 48
    for i in range(3):
        assert floats[i * 16 + 12] == float(i)
