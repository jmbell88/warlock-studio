"""The motion, pinned. These are the claims the prototype's correctness rests on.

None of them says the walk looks good -- that verdict is a person's, and
``TODO.md`` owes it. What they say is that a limb never grows, a stance foot
never skates or floats, and the cycle closes: three things a viewer notices
immediately and cannot un-notice, and three things arithmetic can settle.
"""

from __future__ import annotations

import math

import pytest
from _figure import figure

from warlock.studio.inker.walk import gait
from warlock.studio.inker.walk import rig as R

SIDES = ("near", "far")


def _settings(rig: R.Rig, **changes: float) -> gait.WalkSettings:
    return gait.defaults_for(rig).replaced(**changes)


def _distance(a: R.Point, b: R.Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def test_limb_lengths_are_the_same_in_every_frame():
    """The invariant the whole module is built to protect: a pose is reached by
    turning bones, never by stretching them."""
    rig = figure()
    lengths = R.segment_lengths(rig)
    for pose in gait.cycle(rig, _settings(rig)):
        for side in SIDES:
            hip = pose.joints[f"{side}_hip"]
            knee = pose.joints[f"{side}_knee"]
            ankle = pose.joints[f"{side}_ankle"]
            assert _distance(hip, knee) == pytest.approx(lengths[f"{side}_thigh"], abs=1e-6)
            assert _distance(knee, ankle) == pytest.approx(lengths[f"{side}_shin"], abs=1e-6)


def test_limb_lengths_hold_even_at_a_stride_the_leg_cannot_reach():
    """The case the clamp exists for. Asked for a step twice the leg's span, the
    step comes out shorter -- the leg does not come out longer."""
    rig = figure()
    lengths = R.segment_lengths(rig)
    huge = _settings(rig, stride=R.leg_length(rig) * 4.0)
    for pose in gait.cycle(rig, huge):
        for side in SIDES:
            hip, knee, ankle = (pose.joints[f"{side}_{n}"] for n in ("hip", "knee", "ankle"))
            assert _distance(hip, knee) == pytest.approx(lengths[f"{side}_thigh"], abs=1e-6)
            assert _distance(knee, ankle) == pytest.approx(lengths[f"{side}_shin"], abs=1e-6)


def test_the_arms_never_stretch_either():
    """The arms are forward kinematics rather than a solve, so nothing clamps
    them -- which means nothing can shorten them either. Pinned separately
    because it is a different mechanism reaching the same guarantee."""
    rig = figure()
    lengths = R.segment_lengths(rig)
    for pose in gait.cycle(rig, _settings(rig, arm_swing=80.0)):
        for side in SIDES:
            shoulder, elbow, wrist = (
                pose.joints[f"{side}_{n}"] for n in ("shoulder", "elbow", "wrist")
            )
            assert _distance(shoulder, elbow) == pytest.approx(
                lengths[f"{side}_upper_arm"], abs=1e-6
            )
            assert _distance(elbow, wrist) == pytest.approx(
                lengths[f"{side}_lower_arm"], abs=1e-6
            )


def test_every_frame_has_exactly_one_foot_on_the_ground():
    """Stance is half the cycle and the legs are opposites, so a frame with two
    stance feet or none is a phase error."""
    rig = figure()
    for pose in gait.cycle(rig, _settings(rig)):
        assert sum(pose.grounded.values()) == 1


def test_the_stance_foot_sits_exactly_on_the_ground_line():
    """Not "close to": the body's height is derived from where the stance foot
    has to be, so any drift here means the derivation stopped being authoritative
    and the leg is reaching."""
    rig = figure()
    for pose in gait.cycle(rig, _settings(rig)):
        for side in SIDES:
            if pose.grounded[side]:
                assert pose.joints[f"{side}_ankle"][1] == pytest.approx(rig.ground_y, abs=1e-6)


def test_the_stance_foot_stays_on_the_ground_at_a_stride_it_cannot_reach():
    """The clamp shortens the step; it must not lift the foot off the floor."""
    rig = figure()
    huge = _settings(rig, stride=R.leg_length(rig) * 4.0)
    for pose in gait.cycle(rig, huge):
        for side in SIDES:
            if pose.grounded[side]:
                assert pose.joints[f"{side}_ankle"][1] == pytest.approx(rig.ground_y, abs=1e-6)


def test_the_stance_foot_moves_backward_every_frame():
    """This is an in-place walk, so the ground moves under the figure. A stance
    foot that stalled or crept forward is the skate a viewer reads as sliding."""
    rig = figure()
    facing = rig.facing
    poses = gait.cycle(rig, _settings(rig))
    for side in SIDES:
        travel = [p.joints[f"{side}_ankle"][0] for p in poses if p.grounded[side]]
        # The stance frames of one leg are contiguous within the cycle, so
        # consecutive samples are consecutive steps of the same stance.
        for before, after in zip(travel, travel[1:], strict=False):
            assert (after - before) * facing < 0.0


def test_the_swing_foot_leaves_the_ground_and_comes_back():
    rig = figure()
    poses = gait.cycle(rig, _settings(rig))
    heights = [
        rig.ground_y - p.joints["near_ankle"][1] for p in poses if not p.grounded["near"]
    ]
    assert max(heights) > 0.0
    assert min(heights) == pytest.approx(0.0, abs=1e-6)


def test_the_cycle_hands_its_last_frame_back_to_its_first():
    """Loop continuity, which is true by construction here: the poses are a
    periodic function of phase and the frames are samples of it, so there is no
    seam rule to get wrong. Pinned anyway, because "by construction" is a claim
    about the code that a refactor can quietly stop honouring."""
    rig = figure()
    settings = _settings(rig)
    start = gait.pose(rig, settings, 0.0)
    wrapped = gait.pose(rig, settings, 1.0)
    assert start.joints == wrapped.joints
    assert start.angles == wrapped.angles
    assert start.grounded == wrapped.grounded


def test_the_sample_points_do_not_repeat_the_first_frame_at_the_end():
    """Eight frames means eight distinct phases; a ninth equal to the first would
    show as a held frame at the loop point."""
    assert gait.phases() == tuple(i / 8.0 for i in range(8))
    assert len(set(gait.phases())) == gait.WALK_FRAMES


def test_the_two_legs_are_half_a_cycle_apart():
    rig = figure()
    settings = _settings(rig)
    for t in gait.phases():
        here = gait.pose(rig, settings, t)
        there = gait.pose(rig, settings, (t + 0.5) % 1.0)
        assert here.grounded["near"] == there.grounded["far"]
        assert here.joints["near_ankle"][0] == pytest.approx(there.joints["far_ankle"][0])


def test_the_knee_bends_the_way_it_was_told_to():
    """Forward for a figure facing right, which is what a knee does: flex it and
    the knee goes in front of the hip-to-ankle line while the heel goes back."""
    hip, ankle = (0.0, 0.0), (0.0, 10.0)
    forward = gait.two_bone(hip, ankle, 8.0, 8.0, 1)
    backward = gait.two_bone(hip, ankle, 8.0, 8.0, -1)
    assert forward[0] > 0.0
    assert backward[0] < 0.0
    assert forward[1] == pytest.approx(backward[1])


def test_a_facing_left_figure_bends_its_knees_the_other_way():
    """One sign, no second code path. Mirroring the rig must mirror the bend."""
    rig = figure()
    assert rig.facing == 1
    mirrored = R.set_joint(rig, "near_toe", (25.0, 58.0))
    assert mirrored.facing == -1
    settings = _settings(rig)
    right = gait.pose(rig, settings, 0.15)
    left = gait.pose(mirrored, _settings(mirrored), 0.15)
    hip = rig.joints["near_hip"][0]
    assert (right.joints["near_knee"][0] - hip) * (left.joints["near_knee"][0] - hip) < 0.0


def test_a_target_beyond_the_chain_is_pulled_in_rather_than_the_bones_pushed_out():
    root, target = (0.0, 0.0), (100.0, 0.0)
    reached = gait.clamp_target(root, target, 5.0, 5.0)
    assert reached[0] < 10.0
    assert reached[1] == pytest.approx(0.0)
    knee = gait.two_bone(root, target, 5.0, 5.0, 1)
    assert math.hypot(*knee) == pytest.approx(5.0, abs=1e-6)


def test_a_reachable_target_is_left_exactly_alone():
    """The clamp is a last resort, not a rounding pass -- a target the chain can
    hold must come back bit-for-bit, or every pose acquires a small drift."""
    root, target = (3.0, 4.0), (6.0, 8.0)
    assert gait.clamp_target(root, target, 4.0, 4.0) == target


def test_the_stride_bound_moves_when_the_hip_does():
    """The bound is geometry, not a constant, so dragging the hip up -- which
    gives the leg more slack over the ground -- must widen it."""
    rig = figure()
    lower = gait.reachable_stride(rig)
    raised = R.set_joint(rig, "near_hip", (32.0, 34.0))
    assert gait.reachable_stride(raised) > lower


def test_a_stride_is_never_longer_than_the_leg_allows():
    rig = figure()
    assert 0.0 < gait.reachable_stride(rig) <= R.leg_length(rig) * gait.MAX_STRIDE


def test_the_body_sinks_at_the_contacts_and_rises_through_the_passing_poses():
    """The bob is derived from the reach rather than decorated on, so it exists
    even with the slider at zero -- and a body that did not sink would be a body
    whose stance foot could not touch the floor."""
    rig = figure()
    settings = _settings(rig, bob=0.0)
    contact = gait.root_offset(rig, settings, 0.0)[1]
    passing = gait.root_offset(rig, settings, 0.25)[1]
    assert contact > passing


def test_the_bob_slider_deepens_the_sink_and_never_lifts_the_body_out_of_reach():
    rig = figure()
    quiet = _settings(rig, bob=0.0)
    loud = _settings(rig, bob=R.leg_length(rig) * 0.25)
    for t in gait.phases():
        assert gait.root_offset(rig, loud, t)[1] >= gait.root_offset(rig, quiet, t)[1] - 1e-9
    for pose in gait.cycle(rig, loud):
        for side in SIDES:
            if pose.grounded[side]:
                assert pose.joints[f"{side}_ankle"][1] == pytest.approx(rig.ground_y, abs=1e-6)


def test_the_arms_swing_against_the_legs_on_the_same_side():
    """"Opposing arm swing" is the near arm forward while the near leg is back."""
    rig = figure()
    settings = _settings(rig)
    facing = rig.facing
    for pose in gait.cycle(rig, settings):
        hip = pose.joints["near_hip"][0]
        shoulder = pose.joints["near_shoulder"][0]
        leg = (pose.joints["near_ankle"][0] - hip) * facing
        arm = (pose.joints["near_wrist"][0] - shoulder) * facing
        if abs(leg) > 1.0:
            assert leg * arm <= 0.0


def test_the_foot_is_flat_through_stance():
    """It is standing on the ground line, and the rest drawing already had it
    flat -- which is the whole reason a foot is a part of its own."""
    rig = figure()
    for pose in gait.cycle(rig, _settings(rig)):
        for side in SIDES:
            if pose.grounded[side]:
                assert pose.angles[f"{side}_foot"] == pytest.approx(0.0)


def test_a_turn_the_long_way_round_is_never_taken():
    """An angle difference either side of straight down is a couple of degrees,
    not three hundred and fifty-eight; without the wrap a part spins for one
    frame."""
    rig = figure()
    for pose in gait.cycle(rig, _settings(rig)):
        for name, degrees in pose.angles.items():
            assert -180.0 < degrees <= 180.0, name


def test_defaults_are_a_fraction_of_the_leg_rather_than_a_pixel_count():
    """A sprite may be twelve pixels tall or two hundred, so a default in pixels
    is right for one of them."""
    small = figure()
    big = small.copy()
    big.joints = {name: (x * 3.0, y * 3.0) for name, (x, y) in small.joints.items()}
    big.ground_y = small.ground_y * 3.0
    ratio = gait.defaults_for(big).stride / gait.defaults_for(small).stride
    assert ratio == pytest.approx(3.0, rel=1e-6)
