"""Measuring joints off a mesh, against synthetic bodies whose answers are known.

Synthetic rather than the supplied base meshes for the reason every fixture in
this suite is: the test has to state what it expects, and a real mesh's true
elbow is whatever somebody modelled. What these bodies *do* carry is the one
property the module exists for -- the same skeleton held in a T-pose and in an
A-pose -- because the shipped humanoid template is an A-pose and fitting it to
a T-pose mesh is what puts the arm chain through the ribcage.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.pipelines import jointfit

HEIGHT = 1.8


def _capsule(a, b, radius, n=300, seed=0):
    """A cloud of points around the segment ``a``-``b``."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = rng.random((n, 1))
    spine = a + (b - a) * t
    return spine + rng.normal(0.0, radius, (n, 3))


def _body(arm_drop: float, seed=0) -> np.ndarray:
    """A stick figure standing on Z=0, arms out to x=+-0.7 and dropping by
    ``arm_drop`` over that reach. 0.0 is a T-pose."""
    shoulder_z = HEIGHT * 0.80
    parts = [
        _capsule((0, 0, 0.50), (0, 0, shoulder_z), 0.09, 700, seed),         # torso
        _capsule((0, 0, shoulder_z), (0, 0, HEIGHT), 0.07, 300, seed + 1),   # head
        _capsule((0.10, 0, 0.94), (0.10, 0, 0.04), 0.055, 500, seed + 2),    # leg L
        _capsule((-0.10, 0, 0.94), (-0.10, 0, 0.04), 0.055, 500, seed + 3),  # leg R
        _capsule((0.10, 0, shoulder_z), (0.70, 0, shoulder_z - arm_drop),
                 0.045, 500, seed + 4),                                      # arm L
        _capsule((-0.10, 0, shoulder_z), (-0.70, 0, shoulder_z - arm_drop),
                 0.045, 500, seed + 5),                                      # arm R
    ]
    pts = np.vstack(parts)
    pts[:, 2] = np.clip(pts[:, 2], 0.0, HEIGHT)
    return pts


@pytest.fixture(scope="module")
def tpose():
    return jointfit.measure(_body(0.0))


@pytest.fixture(scope="module")
def apose():
    return jointfit.measure(_body(0.45))


def test_every_template_bone_is_measured(tpose):
    assert set(tpose) == set(jointfit.BONES)


def test_the_payload_is_in_template_order_and_the_shape_the_validator_takes():
    from warlock.rigging import get_template, validate_joints

    payload = jointfit.payload(_body(0.2))
    assert [b["name"] for b in payload["bones"]] == list(jointfit.BONES)
    # The real refusal path, not a restatement of it.
    assert validate_joints(payload, get_template("humanoid"))


def test_the_two_halves_are_exact_mirrors(tpose):
    for name in jointfit.BONES:
        if not name.endswith(".L"):
            continue
        left, right = tpose[name], tpose[name[:-2] + ".R"]
        for end in (0, 1):
            assert right[end][0] == -left[end][0]
            assert right[end][1:] == left[end][1:]


def test_the_chain_is_connected(tpose):
    """A gap between a bone's tail and its child's head is a rig with a joint
    floating in space, which Blender will happily build and nobody will like."""
    for parent, child in (
        ("hips", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
        ("shoulder.L", "upper_arm.L"), ("upper_arm.L", "forearm.L"),
        ("forearm.L", "hand.L"), ("thigh.L", "shin.L"), ("shin.L", "foot.L"),
    ):
        assert tpose[parent][1] == tpose[child][0], (parent, child)


def test_no_bone_is_zero_length(tpose):
    """Blender silently deletes one on leaving edit mode **and takes its
    children with it**, so a zero-length bone costs a limb, quietly."""
    for name, (head, tail) in tpose.items():
        assert np.linalg.norm(np.array(tail) - np.array(head)) > 1e-3, name


def test_the_arm_follows_the_pose_it_was_measured_in(tpose, apose):
    """The property the whole module exists for. In a T-pose the arm chain is
    level; in an A-pose it descends -- and a bbox fit would return the same
    skeleton for both."""
    def slope(bones):
        head, tail = bones["upper_arm.L"][0], bones["hand.L"][1]
        return (tail[2] - head[2]) / (tail[0] - head[0])

    assert abs(slope(tpose)) < 0.05
    assert slope(apose) < -0.5


def test_the_arm_reaches_the_fingertips(tpose):
    """The failure this replaces: a template scaled to the bounding box ended
    its arm chain well short of the hand, so everything past it skinned to the
    chest."""
    reach = 0.70
    assert tpose["hand.L"][1][0] > reach * 0.9


def test_the_arm_stays_in_the_vertical_plane(tpose, apose):
    """Y is pinned to zero along the arm on purpose: a shoulder measured a few
    centimetres forward tilts the bone, and a swing about its own local X then
    carries the arm forward as well as down -- a character walking with its
    arms folded across its chest."""
    for bones in (tpose, apose):
        for name in ("shoulder.L", "upper_arm.L", "forearm.L", "hand.L"):
            assert bones[name][0][1] == 0.0 and bones[name][1][1] == 0.0


def test_the_legs_stand_on_the_floor_and_the_head_tops_the_body(tpose):
    assert tpose["foot.L"][1][2] == 0.0
    assert tpose["head"][1][2] == pytest.approx(HEIGHT, abs=0.02)


def test_the_spine_climbs(tpose):
    heights = [tpose[n][0][2] for n in ("hips", "spine", "chest", "neck", "head")]
    assert heights == sorted(heights)


@pytest.mark.parametrize(
    "cloud, message",
    [
        (np.zeros((10, 3)), "vertex cloud"),
        (np.zeros((100, 2)), "vertex cloud"),
        (np.zeros((100, 3)), "no height"),
    ],
)
def test_a_cloud_that_cannot_be_measured_is_refused_by_name(cloud, message):
    with pytest.raises(ValueError, match=message):
        jointfit.measure(cloud)


def test_a_body_with_no_arms_is_refused_rather_than_guessed():
    """Legs only. Inventing an arm chain from nothing would produce a rig that
    looks fine in rig.json and binds the shoulders to the hips."""
    legs = np.vstack([
        _capsule((0.10, 0, 0.94), (0.10, 0, 0.04), 0.055, 500),
        _capsule((-0.10, 0, 0.94), (-0.10, 0, 0.04), 0.055, 500, 1),
    ])
    with pytest.raises(ValueError, match="not standing in a T-pose"):
        jointfit.measure(legs)
