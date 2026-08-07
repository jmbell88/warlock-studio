"""The pure half of pose2d: COCO-17 landmarks -> template-space bones.

Headless and torch-free by construction. ``refit`` takes keypoints and a
subject bbox and returns normalized bones, so every rule about where a joint
lands is assertable from synthetic numbers -- which is the whole reason the
module is split the way ``pipelines/reference.py`` is.

The figures below face the camera. A subject's *left* is on the viewer's
*right*, i.e. at the larger pixel x, and the template's +X is the subject's
left; ``test_a_subjects_left_arm_lands_on_the_templates_positive_x_side`` is
the one that pins that, because a sign error there mirrors every rig in the
app and a symmetric pose cannot show it.
"""

from __future__ import annotations

import dataclasses

import pytest

from warlock import rigging
from warlock.pipelines import pose2d

# x0, y0, x1, y1 -- 200 wide, 400 tall, centred on x=200. Chosen so every
# expected normalized coordinate below is exact in binary floating point.
BBOX = (100, 50, 300, 450)


def _norm(px: float, py: float) -> tuple[float, float]:
    """The conversion the tests assert against, written out independently of
    the implementation: x about the bbox centre, z up from its floor."""
    return ((px - 200.0) / 200.0, 1.0 - (py - 50.0) / 400.0)


# A standing figure with its arms down at its sides -- deliberately not the
# T-pose the template describes, because "the reference was not in a T-pose"
# is the case this feature exists for.
STANDING = {
    "nose": (200, 90),
    "left_eye": (210, 85),
    "right_eye": (190, 85),
    "left_ear": (220, 90),
    "right_ear": (180, 90),
    "left_shoulder": (240, 150),
    "right_shoulder": (160, 150),
    "left_elbow": (255, 220),
    "right_elbow": (145, 220),
    "left_wrist": (250, 290),
    "right_wrist": (150, 290),
    "left_hip": (225, 260),
    "right_hip": (175, 260),
    "left_knee": (225, 350),
    "right_knee": (175, 350),
    # Above the bbox floor, because an ankle is: the sole is what makes the
    # bottom edge. Keeping them off it is what lets the foot-tail test below
    # see the template's drop rather than the clamp.
    "left_ankle": (225, 420),
    "right_ankle": (175, 420),
}


def _keypoints(pixels: dict[str, tuple[float, float]], **scores: float):
    """COCO-17 keypoints from name -> (px, py), confident unless told otherwise.

    A name absent from ``pixels`` is absent from the returned list: that is a
    landmark the detector never produced, which is a different code path from
    one it produced with no confidence.
    """
    return [
        pose2d.Keypoint(name, px, py, scores.get(name, 0.9))
        for name, (px, py) in pixels.items()
    ]


def _fit(pixels=None, **scores):
    return pose2d.refit(
        rigging.get_template("humanoid"),
        _keypoints(pixels or STANDING, **scores),
        BBOX,
    )


def _by_name(bones):
    return {b["name"]: b for b in bones}


def _template_bone(name: str) -> dict:
    return _by_name(rigging.get_template("humanoid").bones)[name]


# --- the direct anchors -----------------------------------------------------


def test_the_arm_chain_lands_on_the_detected_shoulder_elbow_and_wrist():
    bones = _by_name(_fit())
    shoulder_x, shoulder_z = _norm(*STANDING["left_shoulder"])
    elbow_x, elbow_z = _norm(*STANDING["left_elbow"])
    wrist_x, wrist_z = _norm(*STANDING["left_wrist"])

    assert bones["upper_arm.L"]["head"][0] == pytest.approx(shoulder_x)
    assert bones["upper_arm.L"]["head"][2] == pytest.approx(shoulder_z)
    assert bones["forearm.L"]["head"][0] == pytest.approx(elbow_x)
    assert bones["forearm.L"]["head"][2] == pytest.approx(elbow_z)
    assert bones["hand.L"]["head"][0] == pytest.approx(wrist_x)
    assert bones["hand.L"]["head"][2] == pytest.approx(wrist_z)
    # And the chain is continuous: a tail is its child's head.
    assert bones["upper_arm.L"]["tail"] == bones["forearm.L"]["head"]
    assert bones["forearm.L"]["tail"] == bones["hand.L"]["head"]


def test_the_leg_chain_lands_on_the_detected_hip_knee_and_ankle():
    bones = _by_name(_fit())
    for bone, landmark in (
        ("thigh.R", "right_hip"),
        ("shin.R", "right_knee"),
        ("foot.R", "right_ankle"),
    ):
        x, z = _norm(*STANDING[landmark])
        assert bones[bone]["head"][0] == pytest.approx(x), bone
        assert bones[bone]["head"][2] == pytest.approx(z), bone


def test_the_root_sits_at_the_midpoint_of_the_two_hip_landmarks():
    bones = _by_name(_fit())
    left_x, left_z = _norm(*STANDING["left_hip"])
    right_x, right_z = _norm(*STANDING["right_hip"])
    assert bones["hips"]["head"][0] == pytest.approx((left_x + right_x) / 2)
    assert bones["hips"]["head"][2] == pytest.approx((left_z + right_z) / 2)


def test_a_subjects_left_arm_lands_on_the_templates_positive_x_side():
    """The sign convention, stated once and pinned here.

    COCO's ``left_*`` is the subject's own left. The template's +X is the
    subject's left too. A subject facing the camera has that side at the
    *larger* pixel x, so image x maps straight through with no flip -- and the
    only way to see a mistake is an asymmetric figure, because a mirrored rig
    on a symmetric pose looks perfect.
    """
    raised = {**STANDING, "left_wrist": (250, 100), "left_elbow": (255, 160)}
    bones = _by_name(_fit(raised))
    assert bones["hand.L"]["head"][0] > 0.0
    assert bones["hand.R"]["head"][0] < 0.0
    # The raised arm is the subject's left one.
    assert bones["hand.L"]["head"][2] > bones["hand.R"]["head"][2]


# --- the derived joints -----------------------------------------------------


def test_the_crown_is_pinned_to_the_top_of_the_frame():
    """The bbox top *is* the crown -- it is the topmost subject pixel -- while
    the nose keypoint is a face landmark well below it."""
    assert _by_name(_fit())["head"]["tail"][2] == pytest.approx(1.0)


def test_the_head_leans_towards_the_face_landmarks():
    tilted = {
        **STANDING,
        "nose": (240, 90),
        "left_eye": (250, 85),
        "right_eye": (230, 85),
        "left_ear": (260, 90),
        "right_ear": (220, 90),
    }
    upright = _by_name(_fit())["head"]["tail"][0]
    leaning = _by_name(_fit(tilted))["head"]["tail"][0]
    assert leaning > upright


def test_the_spine_climbs_from_the_hips_to_the_shoulders():
    bones = _by_name(_fit())
    chain = [
        bones["hips"]["head"][2],
        bones["spine"]["head"][2],
        bones["chest"]["head"][2],
        bones["neck"]["head"][2],
    ]
    assert chain == sorted(chain), chain
    hips_z = _norm(*STANDING["left_hip"])[1]
    shoulder_z = _norm(*STANDING["left_shoulder"])[1]
    assert chain[0] == pytest.approx(hips_z)
    assert hips_z < chain[1] < chain[2] <= shoulder_z < chain[3]


def test_a_leaning_torso_leans_the_spine():
    """The chain is interpolated along the measured hips->shoulders axis, not
    stood up vertically from the hips: a subject leaning to one side has a
    spine that leans with them."""
    leaning = {**STANDING, "left_shoulder": (280, 150), "right_shoulder": (200, 150)}
    bones = _by_name(_fit(leaning))
    hips_x = bones["hips"]["head"][0]
    shoulders_x = (_norm(280, 150)[0] + _norm(200, 150)[0]) / 2
    assert hips_x < bones["spine"]["head"][0] < bones["chest"]["head"][0]
    assert bones["chest"]["head"][0] < shoulders_x


def test_the_hand_extends_along_the_forearm_past_the_wrist():
    bones = _by_name(_fit())
    elbow = bones["forearm.L"]["head"]
    wrist = bones["hand.L"]["head"]
    tail = bones["hand.L"]["tail"]
    # Colinear with the forearm, and beyond the wrist rather than back up it.
    forearm = (wrist[0] - elbow[0], wrist[2] - elbow[2])
    hand = (tail[0] - wrist[0], tail[2] - wrist[2])
    assert hand[0] * forearm[1] == pytest.approx(hand[1] * forearm[0], abs=1e-9)
    assert hand[0] * forearm[0] + hand[1] * forearm[1] > 0.0
    # Template-proportional: shorter than the forearm it extends.
    assert 0.0 < (hand[0] ** 2 + hand[1] ** 2) < (forearm[0] ** 2 + forearm[1] ** 2)


def test_the_feet_keep_the_templates_forward_offset():
    bones = _by_name(_fit())
    template_foot = _template_bone("foot.L")
    offset = template_foot["tail"][2] - template_foot["head"][2]
    ankle_z = _norm(*STANDING["left_ankle"])[1]
    assert bones["foot.L"]["tail"][1] == pytest.approx(template_foot["tail"][1])
    assert bones["foot.L"]["tail"][1] < 0.0, "-Y is forward; a foot points forward"
    assert bones["foot.L"]["tail"][2] == pytest.approx(max(0.0, ankle_z + offset))


def test_a_foot_never_lands_below_the_floor():
    """z=0 is the bbox floor. The template's foot tail drops below its head,
    so a detected ankle already at the floor would otherwise put the toe under
    the mesh, where automatic weights have nothing to bind to."""
    planted = {**STANDING, "left_ankle": (225, 450), "right_ankle": (175, 450)}
    bones = _by_name(_fit(planted))
    assert bones["foot.L"]["tail"][2] == pytest.approx(0.0)


# --- what the mapping does not claim to know --------------------------------


def test_depth_comes_from_the_template_for_every_joint():
    """One image gives x and z. Nothing in it gives y, so every joint keeps the
    depth the template chose -- which is what makes this an *informed* fit
    rather than a flattened one."""
    fitted = _by_name(_fit())
    for name, bone in fitted.items():
        template_bone = _template_bone(name)
        assert bone["head"][1] == pytest.approx(template_bone["head"][1]), name
        assert bone["tail"][1] == pytest.approx(template_bone["tail"][1]), name


def test_every_template_bone_comes_back_with_its_parentage():
    template = rigging.get_template("humanoid")
    fitted = _fit()
    assert [b["name"] for b in fitted] == [b["name"] for b in template.bones]
    assert [b["parent"] for b in fitted] == [b["parent"] for b in template.bones]


def test_the_result_survives_the_bbox_scaling_the_worker_applies():
    """The whole seam: refit produces a Template the worker fits exactly as it
    fits the shipped one, so the output has to satisfy the same validation a
    hand-adjusted skeleton does."""
    template = rigging.get_template("humanoid")
    informed = dataclasses.replace(template, bones=tuple(_fit()))
    fitted = rigging.fit_template(informed, [-1.0, -0.5, 0.0], [1.0, 0.5, 2.0])
    assert rigging.validate_joints({"bones": fitted}, template)


# --- the sanity gates -------------------------------------------------------


def test_a_landmark_below_the_confidence_floor_refuses_the_whole_fit():
    assert _fit(left_knee=pose2d.MIN_KEYPOINT_CONFIDENCE - 0.01) is None


def test_a_landmark_the_detector_never_produced_refuses_the_whole_fit():
    without_ankle = {k: v for k, v in STANDING.items() if k != "right_ankle"}
    assert _fit(without_ankle) is None


def test_an_implausible_vertical_ordering_refuses_the_fit():
    """Shoulders above hips above knees above ankles. A detection that says
    otherwise has found something that is not a standing figure, and a
    skeleton built from it would be worse than the bbox fit it replaces."""
    upside_down = {name: (px, 500 - py) for name, (px, py) in STANDING.items()}
    assert _fit(upside_down) is None


def test_a_landmark_far_outside_the_subject_bbox_refuses_the_fit():
    """The bbox is the subject. A landmark well outside it is a detection of
    something else in the frame, and the normalized coordinates it produces
    would be meaningless."""
    stray = {**STANDING, "right_wrist": (-400, 290)}
    assert _fit(stray) is None


def test_a_landmark_just_outside_the_bbox_is_tolerated():
    """Detection is approximate and the bbox is a threshold of the same image;
    refusing on a pixel of overshoot would refuse most real figures."""
    edge = {**STANDING, "left_wrist": (302, 290)}
    assert _fit(edge) is not None


def test_a_template_the_mapping_does_not_know_is_refused():
    """The COCO-17 skeleton says nothing about a quadruped, and there is no
    honest partial answer -- the caller falls back to the bbox fit."""
    assert (
        pose2d.refit(rigging.get_template("quadruped"), _keypoints(STANDING), BBOX)
        is None
    )
    assert "quadruped" not in pose2d.POSE_FIT_TEMPLATES
    assert "humanoid" in pose2d.POSE_FIT_TEMPLATES


def test_a_degenerate_bbox_is_refused_rather_than_dividing_by_zero():
    template = rigging.get_template("humanoid")
    assert pose2d.refit(template, _keypoints(STANDING), (5, 5, 5, 5)) is None


# --- the model half ---------------------------------------------------------
#
# Only its edges: whether it is asked at all, what it does when it cannot
# answer, and the two conversions between the model's vocabulary and this
# module's. The forward pass itself needs weights and is the manual check.


@pytest.fixture(autouse=True)
def _no_model_carried_between_tests():
    # The cache, and the failure sentinel beside it, is module state that
    # outlives a test by design. Nothing here should inherit either.
    pose2d.unload()
    yield
    pose2d.unload()


def _config(tmp_path):
    from types import SimpleNamespace

    return SimpleNamespace(t2i_model_root=tmp_path)


def _weights(tmp_path):
    from warlock import models

    root = tmp_path / models.POSE_MODELS[models.DEFAULT_POSE_MODEL].dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    return root


def test_the_registry_entry_carries_a_download_command():
    from warlock import models

    spec = models.POSE_MODELS[models.DEFAULT_POSE_MODEL]
    assert spec.dir_name and "hf download" in spec.download


def test_no_weights_means_not_available(tmp_path):
    assert pose2d.available(_config(tmp_path)) is False


def test_weights_on_disk_mean_available(tmp_path):
    _weights(tmp_path)
    assert pose2d.available(_config(tmp_path)) is True


def test_detection_without_weights_never_reaches_the_model(tmp_path, monkeypatch):
    """The existence check precedes the torch import, which is the ordering
    every model in this codebase keeps: a checkout with no image extra has to
    get the bbox fit, not an ImportError about a dependency that was never the
    problem."""

    def explode(*_a, **_k):
        raise AssertionError("the model must not be asked for without weights")

    monkeypatch.setattr(pose2d, "_detect", explode)
    assert pose2d.detect(tmp_path / "input.png", BBOX, _config(tmp_path)) is None


def test_a_detection_failure_is_a_none_not_a_raise(tmp_path, monkeypatch):
    """Everything downstream of this is a fallback, so a broken checkpoint, an
    unreadable PNG or an out-of-memory costs the informed fit and never the
    rig job."""
    _weights(tmp_path)
    monkeypatch.setattr(
        pose2d, "_detect", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert pose2d.detect(tmp_path / "input.png", BBOX, _config(tmp_path)) is None


def test_a_checkpoint_that_failed_once_is_not_loaded_again(tmp_path, monkeypatch):
    """from_pretrained is seconds of work per attempt and a queue draining a
    sweep rigs job after job; one broken install must not be paid for by every
    one of them."""
    import sys
    import types

    attempts: list[str] = []

    def refuse(*_a, **_k):
        attempts.append("tried")
        raise OSError("no such checkpoint")

    fake = types.ModuleType("transformers")
    fake.AutoImageProcessor = types.SimpleNamespace(from_pretrained=refuse)
    fake.VitPoseForPoseEstimation = types.SimpleNamespace(from_pretrained=refuse)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    with pytest.raises(OSError):
        pose2d._load(tmp_path)
    with pytest.raises(RuntimeError, match="earlier this session"):
        pose2d._load(tmp_path)
    assert attempts == ["tried"]

    # ...and repairing the install gets another go, without restarting the app.
    pose2d.unload()
    with pytest.raises(OSError):
        pose2d._load(tmp_path)
    assert attempts == ["tried", "tried"]


def test_the_person_box_is_handed_over_in_coco_format():
    """The processor's post-processing maps heatmap coordinates back through
    this box, and it documents (x, y, width, height). Handing it the corners
    instead puts every landmark somewhere plausible and wrong, with nothing
    raised to say so."""
    assert pose2d._coco_box((100, 50, 300, 450)) == [100.0, 50.0, 200.0, 400.0]


def test_landmarks_are_named_by_the_models_label_not_by_position():
    """``post_process_pose_estimation`` can drop low-scoring keypoints, and
    when it does the remaining rows shift up. Reading names off the row index
    would then silently relabel an ankle as an eye."""
    named = pose2d._named(
        [[10.0, 20.0], [30.0, 40.0]], [0.9, 0.8], [16, 0]
    )
    assert [(k.name, k.x, k.score) for k in named] == [
        ("right_ankle", 10.0, 0.9),
        ("nose", 30.0, 0.8),
    ]


def test_a_label_outside_the_coco_set_is_dropped_rather_than_guessed():
    named = pose2d._named([[1.0, 2.0], [3.0, 4.0]], [0.9, 0.9], [99, 5])
    assert [k.name for k in named] == ["left_shoulder"]
