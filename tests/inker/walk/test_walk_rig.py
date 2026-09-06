"""Assembling a rig, and what it says when it is not finished yet.

The refusals matter more than they look. A greyed Bake button with no reason is
the defect this project has a written rule about, and the string these functions
return is what that button's tooltip says -- so it is asserted here, in the
engine, where it is a plain unit test rather than something only a screenshot
could catch.
"""

from __future__ import annotations

import numpy as np
import pytest
from _figure import BLOCKS, JOINTS, SIZE, figure, plane

from warlock.studio.inker.walk import rig as R


def test_a_blank_rig_has_a_part_slot_for_every_spec():
    blank = R.blank(SIZE)
    assert set(blank.parts) == set(R.PART_NAMES)
    assert not any(part.assigned for part in blank.parts.values())


def test_a_finished_rig_refuses_nothing():
    assert R.refusal(figure()) == ""


def test_an_unassigned_body_part_is_named_in_the_refusal():
    rig = R.set_part(figure(), "head", R.Part())
    assert R.refusal(rig) == "No art assigned to the head."


def test_two_missing_parts_are_listed_with_an_and():
    rig = R.set_part(R.set_part(figure(), "head", R.Part()), "torso", R.Part())
    assert R.refusal(rig) == "No art assigned to the torso and head."


def test_a_half_assigned_limb_names_the_pieces_that_are_missing():
    """A figure drawn with one arm hidden is legitimate, so a limb is refused
    only when it is *part* assigned -- which is always a mistake."""
    rig = R.set_part(figure(), "far_hand", R.Part())
    assert R.refusal(rig) == "No art assigned to the far hand."


def test_a_limb_left_out_entirely_is_not_refused():
    rig = figure(far=False)
    for name in ("far_upper_arm", "far_lower_arm", "far_hand"):
        assert not rig.parts[name].assigned
    assert "arm" not in R.missing_parts(rig)


def test_a_rig_with_no_legs_at_all_is_refused():
    """A walk with nothing to stand on is not a walk."""
    rig = figure()
    for spec in R.PARTS:
        if spec.limb.endswith("leg"):
            rig = R.set_part(rig, spec.name, R.Part())
    assert "thigh" in R.refusal(rig)


def test_an_unplaced_joint_is_named_once_the_art_is_there():
    rig = R.blank(SIZE)
    for name, (x0, y0, x1, y1, colour) in BLOCKS.items():
        rig = R.set_part(rig, name, R.part_from_plane(plane((x0, y0, x1, y1), colour)))
    for name in ("far_upper_arm", "far_lower_arm", "far_hand"):
        rig = R.set_part(rig, name, rig.parts["near" + name.removeprefix("far")])
    assert "Place the" in R.refusal(rig)
    assert "shoulder" in R.refusal(rig)


def test_both_legs_joints_are_required_even_with_no_leg_art():
    """The gait is driven off the hip, knee and ankle whether or not a thigh was
    ever cut out -- an arms-only rig still has to know where the body is."""
    rig = R.blank(SIZE)
    assert set(R.required_joints(rig)) >= {
        f"{side}_{joint}"
        for side in ("near", "far")
        for joint in ("hip", "knee", "ankle", "toe")
    }


def test_a_segment_whose_joints_sit_on_top_of_each_other_is_refused():
    """Two joints a pixel apart give an angle that swings wildly for a one-pixel
    drag, so the rig is refused rather than rendered into a part that spins."""
    rig = R.set_joint(figure(), "near_knee", JOINTS["near_hip"])
    assert "too short to turn" in R.refusal(rig)


def test_segment_lengths_are_measured_from_the_rest_pose():
    rig = figure()
    lengths = R.segment_lengths(rig)
    assert lengths["near_thigh"] == pytest.approx(10.0)
    assert lengths["near_shin"] == pytest.approx(9.0)
    assert R.leg_length(rig) == pytest.approx(19.0)


def test_copying_a_limb_brings_its_joints_across_too():
    """Pixels and joints together, because a far thigh with the near thigh's art
    and no hip placed is not a head start, it is half a job."""
    rig = R.blank(SIZE)
    for name, (x0, y0, x1, y1, colour) in BLOCKS.items():
        rig = R.set_part(rig, name, R.part_from_plane(plane((x0, y0, x1, y1), colour)))
    for name, point in JOINTS.items():
        rig = R.set_joint(rig, name, point)
    copied = R.copy_near_to_far(rig, "leg")
    for joint in ("hip", "knee", "ankle", "toe"):
        assert copied.joints[f"far_{joint}"] == rig.joints[f"near_{joint}"]
    assert "far_shoulder" not in copied.joints


def test_copying_a_limb_leaves_the_other_limbs_alone():
    rig = figure(far=False)
    copied = R.copy_near_to_far(rig, "leg")
    assert copied.parts["far_thigh"].assigned
    assert not copied.parts["far_upper_arm"].assigned


def test_a_limb_is_an_arm_or_a_leg():
    with pytest.raises(ValueError, match="arm or a leg"):
        R.copy_near_to_far(figure(), "tail")


def test_every_mutation_bumps_the_revision():
    """The preview caches on this number rather than hashing fourteen planes, so
    a mutator that forgot to bump it would show a stale walk."""
    rig = figure()
    assert R.set_joint(rig, "neck", (1.0, 2.0)).rev == rig.rev + 1
    assert R.set_ground(rig, 3.0).rev == rig.rev + 1
    assert R.set_part(rig, "head", R.Part()).rev == rig.rev + 1
    assert R.set_order(rig, reversed(R.PART_NAMES)).rev == rig.rev + 1
    assert R.copy_near_to_far(rig, "arm").rev == rig.rev + 1


def test_a_mutation_leaves_the_rig_it_was_given_alone():
    """Every mutator returns a new rig, which is what lets the pane hold one and
    the preview hold another without a defensive copy at every call site."""
    rig = figure()
    before = dict(rig.joints)
    moved = R.set_joint(rig, "neck", (99.0, 99.0))
    assert rig.joints == before
    assert moved.joints["neck"] == (99.0, 99.0)


def test_a_draw_order_names_every_part_exactly_once():
    with pytest.raises(ValueError, match="every part exactly once"):
        R.set_order(figure(), R.PART_NAMES[:-1])


def test_trimming_finds_the_art_and_says_where_it_was():
    art = plane((10, 20, 14, 26), (1, 2, 3, 255))
    crop, origin = R.trim(art)
    assert origin == (10, 20)
    assert crop.shape == (6, 4, 4)


def test_trimming_an_empty_plane_gives_something_rather_than_nothing():
    """A part assigned from a blank layer is a rig problem, reported by
    ``missing_parts`` -- not something every caller downstream must branch on."""
    crop, origin = R.trim(np.zeros((8, 8, 4), dtype=np.uint8))
    assert crop.shape == (1, 1, 4)
    assert origin == (0, 0)


def test_a_part_taken_from_a_cutout_remembers_where_the_cutout_was():
    """A selection cutout arrives already cropped, so its own offset has to
    compose with the trim or every joint lands in the wrong place."""
    cutout = np.zeros((6, 4, 4), dtype=np.uint8)
    cutout[2:5, 1:3] = (9, 9, 9, 255)
    part = R.part_from_plane(cutout, origin=(30, 40))
    assert part.origin == (31, 42)


def test_facing_is_read_off_the_toe_rather_than_asked_for():
    rig = figure()
    assert rig.facing == 1
    assert R.set_joint(rig, "near_toe", (20.0, 58.0)).facing == -1


def test_facing_falls_back_to_right_when_there_is_nothing_to_read():
    assert R.blank(SIZE).facing == 1


def test_the_ground_line_starts_under_the_lower_foot():
    """A better guess than the canvas floor: a sprite is normally drawn with air
    beneath it, and a ground line at the last row puts the walk below the feet."""
    rig = figure()
    assert R.default_ground(rig, SIZE[1]) == JOINTS["near_ankle"][1]
    assert R.default_ground(R.blank(SIZE), SIZE[1]) == SIZE[1] - 1
