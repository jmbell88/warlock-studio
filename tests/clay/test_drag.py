"""Axis constraints and typed values: the keyboard's half of a gizmo drag.

Every rule here is ``(quantity, lock) -> quantity``, which is exactly why the
module exists as a filter rather than as a second kind of drag -- the whole of
"what does X then 2 do" is three numbers, with no viewport anywhere near it.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import drag as bd
from warlock.studio.viewer import math3d as m3


def _typed(axis: str = "", typed: str = "") -> bd.DragInput:
    return bd.DragInput(axis=axis, typed=typed)


# --- the input itself ---------------------------------------------------------


def test_an_axis_key_locks_and_the_same_key_again_clears_it():
    """The only convention that leaves a mistyped X recoverable without
    abandoning the drag."""
    entry = bd.DragInput()
    assert entry.key("x") and entry.axis == "x"
    assert entry.key("x") and entry.axis == ""


def test_a_second_axis_key_replaces_the_first():
    entry = bd.DragInput()
    entry.key("x")
    entry.key("z")
    assert entry.axis == "z"


def test_digits_a_dot_and_a_minus_are_taken_and_letters_are_not():
    entry = bd.DragInput()
    for ch in "-1.5":
        assert entry.key(ch), ch
    assert entry.typed == "-1.5"
    assert not entry.key("q")


def test_backspace_takes_the_last_character_back():
    entry = bd.DragInput(typed="12")
    assert entry.key("backspace")
    assert entry.typed == "1"


def test_a_half_typed_number_is_kept_rather_than_discarded():
    """A bare minus and a trailing dot both parse to nothing and are both states
    a real number passes through; dropping them makes the minus unreachable."""
    entry = bd.DragInput()
    entry.key("-")
    assert entry.typed == "-"
    assert entry.value() is None
    entry.key("2")
    assert entry.value() == -2.0


def test_an_empty_input_is_not_active_and_a_lock_alone_is():
    assert not bd.DragInput().active
    assert bd.DragInput(axis="y").active
    assert bd.DragInput(typed="-").active


# --- translation --------------------------------------------------------------


def test_an_unconstrained_displacement_passes_straight_through():
    d = np.array([1.0, 2.0, 3.0])
    assert np.allclose(bd.constrain_translation(d, _typed()), d)


def test_a_lock_keeps_only_that_axis_of_the_displacement():
    out = bd.constrain_translation(np.array([1.0, 2.0, 3.0]), _typed(axis="y"))
    assert np.allclose(out, [0.0, 2.0, 0.0])


def test_a_typed_value_with_a_lock_is_the_whole_answer():
    out = bd.constrain_translation(np.array([1.0, 2.0, 3.0]), _typed("x", "2.5"))
    assert np.allclose(out, [2.5, 0.0, 0.0])


def test_a_typed_value_with_no_lock_keeps_the_direction_the_mouse_chose():
    """A lock is a statement about direction and a number is the amount, which
    is why the number is meaningful on its own."""
    out = bd.constrain_translation(np.array([3.0, 4.0, 0.0]), _typed(typed="10"))
    assert np.allclose(out, [6.0, 8.0, 0.0])


def test_a_typed_value_against_a_zero_displacement_stays_zero():
    """There is no direction to keep, and inventing one would put the object
    somewhere the user never pointed."""
    out = bd.constrain_translation(np.zeros(3), _typed(typed="5"))
    assert np.allclose(out, 0.0)


# --- scale --------------------------------------------------------------------


def test_a_locked_scale_returns_the_other_axes_to_one():
    """Not to whatever the gizmo said: the point of locking is that nothing else
    moves, and leaving the other two dragged would be a lock that only renamed
    the readout."""
    out = bd.constrain_scale(np.array([2.0, 3.0, 4.0]), _typed(axis="z"))
    assert np.allclose(out, [1.0, 1.0, 4.0])


def test_a_typed_scale_with_no_lock_is_uniform():
    assert np.allclose(bd.constrain_scale(np.array([2.0, 3.0, 4.0]), _typed(typed="0.5")), 0.5)


def test_a_typed_scale_with_a_lock_touches_one_axis():
    out = bd.constrain_scale(np.ones(3), _typed("x", "3"))
    assert np.allclose(out, [3.0, 1.0, 1.0])


# --- rotation -----------------------------------------------------------------


def _angle_about(quat: np.ndarray, axis: np.ndarray) -> float:
    length = float(np.linalg.norm(quat[:3]))
    angle = 2.0 * float(np.arctan2(length, float(quat[3])))
    if length < 1e-12:
        return 0.0
    return float(np.degrees(angle * (quat[:3] / length @ axis)))


def test_an_unconstrained_rotation_passes_straight_through():
    q = m3.quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5)
    assert np.allclose(bd.constrain_rotation(q, _typed()), q)


def test_a_typed_angle_with_a_lock_is_exactly_that_many_degrees():
    out = bd.constrain_rotation(m3.quat_identity(), _typed("y", "90"))
    assert _angle_about(out, np.array([0.0, 1.0, 0.0])) == pytest.approx(90.0)


def test_a_lock_projects_the_swept_angle_rather_than_keeping_it_whole():
    """Which is what keeps the result continuous as the lock goes on: a sweep
    mostly about Y contributes mostly to a Y lock."""
    axis = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    q = m3.quat_from_axis_angle(axis, np.radians(90.0))
    out = bd.constrain_rotation(q, _typed(axis="y"))
    assert _angle_about(out, np.array([0.0, 1.0, 0.0])) == pytest.approx(90.0 / np.sqrt(2.0))


def test_a_typed_angle_with_no_lock_keeps_the_drags_own_axis():
    q = m3.quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.2)
    out = bd.constrain_rotation(q, _typed(typed="45"))
    assert _angle_about(out, np.array([1.0, 0.0, 0.0])) == pytest.approx(45.0)


def test_a_typed_angle_against_the_identity_stays_the_identity():
    """It has no axis, and inventing one would turn the object about something
    nobody chose."""
    out = bd.constrain_rotation(m3.quat_identity(), _typed(typed="45"))
    assert np.allclose(out, m3.quat_identity())


# --- proportional editing -----------------------------------------------------


def test_the_falloff_is_one_at_the_selection_and_zero_at_the_radius():
    out = bd.falloff(np.array([0.0, 0.5, 1.0]), 1.0)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(0.0)


def test_the_falloff_has_no_crease_at_either_end():
    """A linear ramp creases at the selection boundary and again at the radius,
    both of which show as a visible ridge -- the artefact this exists to avoid."""
    t = np.linspace(0.0, 1.0, 201)
    w = bd.falloff(t, 1.0)
    slope = np.diff(w)
    assert abs(float(slope[0])) < abs(float(slope[len(slope) // 2])) / 10.0
    assert abs(float(slope[-1])) < abs(float(slope[len(slope) // 2])) / 10.0


def test_a_zero_radius_is_a_hard_selection_rather_than_a_division_by_zero():
    """The ``snap_value`` convention: the off switch is the same control."""
    assert bd.falloff(np.array([0.0, 0.1]), 0.0).tolist() == [1.0, 0.0]


def test_the_neighbourhood_measures_from_the_nearest_selected_vertex():
    """Not from the centroid, which is a point the geometry may not pass
    through: dragging one end of a long strip would fade out along the strip."""
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype="f8")
    verts, weights = bd.proportional_set(positions, np.array([0, 1]), 1.5)
    assert verts.tolist() == [0, 1, 2]
    assert weights[:2].tolist() == [1.0, 1.0]
    assert 0.0 < float(weights[2]) < 1.0


def test_a_vertex_exactly_at_the_radius_is_dropped_rather_than_carried():
    """Carrying it means a drag that reports moving geometry it does not move."""
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype="f8")
    verts, _ = bd.proportional_set(positions, np.array([0]), 1.0)
    assert verts.tolist() == [0]


def test_a_mesh_too_big_for_the_broadcast_falls_back_to_a_hard_selection():
    """Declining beats both a stall at the press and a spatial index nothing
    else in this package needs."""
    positions = np.zeros((10, 3))
    verts, weights = bd.proportional_set(positions, np.arange(10), 1.0)
    assert len(verts) == 10
    saved, bd.MAX_FALLOFF_PAIRS = bd.MAX_FALLOFF_PAIRS, 1
    try:
        verts, weights = bd.proportional_set(positions, np.array([0, 1]), 1.0)
    finally:
        bd.MAX_FALLOFF_PAIRS = saved
    assert verts.tolist() == [0, 1]
    assert weights.tolist() == [1.0, 1.0]


# --- the readout --------------------------------------------------------------


def test_the_readout_shows_the_result_rather_than_what_was_typed():
    text = bd.readout("move", np.array([0.0, 2.0, 0.0]), _typed(axis="y"))
    assert "[Y]" in text and "2.000" in text


def test_the_readout_shows_a_half_typed_number_as_well_as_the_value():
    """A field you cannot see is not a field."""
    text = bd.readout("move", np.zeros(3), _typed("x", "-"))
    assert "= -" in text


def test_a_rotation_reads_in_degrees_and_a_scale_carries_no_unit():
    assert "deg" in bd.readout("rotate", np.array([45.0]), _typed())
    assert "m" not in bd.readout("scale", np.ones(3), _typed())


# --- the signed screen angle (Clay W3) ---------------------------------------


def test_the_screen_angle_is_signed_rather_than_absolute():
    """``arccos`` of the dot product gives the angle and loses the direction,
    so a rotate drag turned the same way whichever way the mouse went round."""
    import math

    from warlock.studio.clay import drag as bdrag

    pivot = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    assert bdrag.screen_angle(pivot, axis, right, up) == pytest.approx(math.pi / 2)
    assert bdrag.screen_angle(pivot, axis, up, right) == pytest.approx(-math.pi / 2)


def test_the_screen_angle_ignores_the_component_along_the_axis():
    """Every ray hit is a hair off the plane at float precision, and a tilt
    that leaked in would make the same drag report a different angle."""
    from warlock.studio.clay import drag as bdrag

    pivot = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    flat = bdrag.screen_angle(
        pivot, axis, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    off = bdrag.screen_angle(
        pivot, axis, np.array([1.0, 0.0, 7.0]), np.array([0.0, 1.0, -3.0])
    )
    assert off == pytest.approx(flat)


def test_a_degenerate_pair_reports_no_angle_rather_than_refusing():
    """The caller is a live drag that must go on drawing."""
    from warlock.studio.clay import drag as bdrag

    pivot = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    assert bdrag.screen_angle(pivot, axis, pivot, np.array([1.0, 0.0, 0.0])) == 0.0


def test_a_full_turn_wraps_rather_than_accumulating():
    """``atan2`` is what the caller's own accumulator expects: the increment is
    per frame, and a drag that passes the half turn must not jump."""
    import math

    from warlock.studio.clay import drag as bdrag

    pivot = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])
    angle = bdrag.screen_angle(
        pivot, axis, np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
    )
    assert abs(angle) == pytest.approx(math.pi)
