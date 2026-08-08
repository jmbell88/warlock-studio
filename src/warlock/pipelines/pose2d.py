"""Where the subject's joints actually are, read off the reference image.

Every generated mesh is a reconstruction of one image, and until now the rig
fitter never looked at it: ``rigging.fit_template`` scales a template's
normalized landmarks onto the mesh bounding box and nothing else, so a
reference that is not standing in a T-pose gets a T-pose skeleton laid over it.
A 2D pose estimator run on that same image gives the shoulders, elbows, hips,
knees and ankles in the subject's own frame -- and because the mesh *is* that
image reconstructed, those positions map straight onto the mesh's X (side) and
Z (height) axes. Only depth is unrecoverable from one view, and depth keeps the
template's values.

The module is split the way ``pipelines/reference.py`` is, and for the same
reason:

* the *pure* half (:func:`refit`) is arithmetic on landmarks and needs nothing
  but the standard library, so every rule about where a joint lands is
  assertable headlessly from synthetic numbers;
* the *impure* half (:func:`available`, :func:`detect`) is the model, and
  imports torch and transformers inside the functions that use them -- never at
  module scope -- so ``available()``, the question every caller asks first, is
  answerable on a checkout with no image extra installed at all.

Nothing here ever fails a rig. Every refusal is a ``None`` and the caller falls
back to the bbox-proportional fit, which is what the app did before this
existed and is still exactly right when the reference is a plain T-pose.

**Wholesale or not at all.** A detection that clears the gates places every
joint; one that does not places none. Filling some joints from the image and
the rest from the template would put two coordinate systems in one skeleton --
an arm measured against the subject and a shoulder assumed from a T-pose -- and
the result is not "partly better", it is a skeleton that is internally
inconsistent in a way nothing downstream can detect.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import models

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ..rigging import Template

log = logging.getLogger(__name__)

# The templates this mapping knows how to speak about. COCO-17 is a *human*
# skeleton: it says nothing about a quadruped's shoulder or a bird's wing, and
# there is no honest partial answer for one. Widening this set means finding a
# model trained on that animal (AP-10K for quadrupeds) and writing its own
# mapping -- not reusing this one.
POSE_FIT_TEMPLATES = frozenset({"humanoid"})

# The 17 landmarks COCO defines, in the order every model trained on it emits
# them. Written out rather than derived from the checkpoint's config so the
# pure half can be tested without one.
COCO_KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# How sure the detector has to be about a landmark before a joint is placed on
# it. 0.3 is low as thresholds go, and deliberately: this is not a decision
# about whether a person is present -- the subject bbox already came from the
# composition gate, which refuses a reference with no subject in it -- but
# about whether *this* landmark was found or guessed. Below it the whole fit is
# refused, so the cost of being generous is a bad skeleton and the cost of
# being strict is the bbox fit, which is only the status quo.
MIN_KEYPOINT_CONFIDENCE = 0.3

# The twelve limb landmarks the mapping actually consumes, each of which has to
# clear the confidence floor.
#
# Wider than "both shoulders, hips, knees and ankles", which is the set that
# decides whether the detection found a *standing figure*. The elbows and
# wrists are here because the arm chain is built from them and there is no
# per-joint fallback by design: a missing elbow cannot be filled from the
# template without mixing an assumed T-pose into a measured skeleton, so it
# refuses the fit instead. Refusing costs the bbox fit; mixing costs an arm
# bent to a joint that is not in the mesh.
REQUIRED_KEYPOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# The landmarks that fix the head's lean. Optional, all of them: a subject
# facing away has no nose and no eyes and is still perfectly riggable, so the
# crown simply falls back to sitting above the shoulders.
FACE_KEYPOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")

# How far outside the subject bbox a required landmark may sit before the fit
# is refused, as a fraction of the bbox's own extent.
#
# Not zero. The bbox is a threshold of the same image the detector read, and
# the two disagree at the edges by a few pixels on almost every figure -- an
# outstretched hand is exactly where both are least certain. What this is
# actually catching is a landmark placed on something *else* in the frame,
# which misses by a whole subject rather than by a rim.
BBOX_TOLERANCE = 0.15

# Every bone the mapping places, which is every bone the humanoid template has.
# The match has to be exact, not a subset: a template carrying these plus a
# tail (``biped_tail`` does) would come back missing the tail bones, and their
# absence is not a rig with no tail -- it is a bone list whose parents do not
# resolve.
_HUMANOID_BONES = frozenset(
    {
        "hips", "spine", "chest", "neck", "head",
        "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
        "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
        "thigh.L", "shin.L", "foot.L",
        "thigh.R", "shin.R", "foot.R",
    }
)


@dataclass(frozen=True, slots=True)
class Keypoint:
    """One detected landmark: a COCO name, a pixel position, a confidence."""

    name: str
    x: float
    y: float
    score: float


# --- the pure half ----------------------------------------------------------


def refit(
    template: Template,
    keypoints: Sequence[Keypoint],
    subject_bbox: Sequence[int],
) -> list[dict[str, Any]] | None:
    """Landmarks -> the template's own normalized bones, or ``None``.

    The return value is a drop-in replacement for ``template.bones``: the same
    names, the same parentage, the same normalized space (Blender axes, x/y
    spanning -0.5..0.5 about the bbox centre, z spanning 0 at the floor to 1 at
    the crown). So the worker keeps running ``rigging.fit_template`` on it
    exactly as it runs the shipped one, and the bbox scaling stays owned by the
    worker rather than being reimplemented here.

    ``subject_bbox`` is ``reference.Report.bbox``: ``(x0, y0, x1, y1)`` in
    pixels with the right and bottom edges exclusive and y running *down*,
    which is why the z conversion below is a subtraction from 1.

    ``None`` on any doubt at all -- a landmark the detector was unsure of, a
    figure whose knees are above its hips, a template this mapping does not
    speak for. Every one of those is the caller's cue to fall back to the
    bbox-proportional fit, which is a worse skeleton and never a wrong one.
    """
    by_name = {bone["name"]: bone for bone in template.bones}
    if by_name.keys() != _HUMANOID_BONES:
        log.info("pose fit: no landmark mapping for template %r", template.key)
        return None

    found = {kp.name: kp for kp in keypoints}
    missing = [n for n in REQUIRED_KEYPOINTS if n not in found]
    if missing:
        log.info("pose fit: the detector produced no %s", ", ".join(missing))
        return None
    weak = [n for n in REQUIRED_KEYPOINTS if found[n].score < MIN_KEYPOINT_CONFIDENCE]
    if weak:
        log.info("pose fit: not confident enough about %s", ", ".join(weak))
        return None

    to_frame = _framer(subject_bbox)
    if to_frame is None:
        return None
    p = {n: to_frame(found[n].x, found[n].y) for n in REQUIRED_KEYPOINTS}

    outside = [n for n, point in p.items() if not _inside(point)]
    if outside:
        log.info("pose fit: %s landed outside the subject", ", ".join(outside))
        return None
    if not _stands_up(p):
        log.info("pose fit: the landmarks are not a standing figure")
        return None

    face = [
        to_frame(found[n].x, found[n].y)
        for n in FACE_KEYPOINTS
        if n in found and found[n].score >= MIN_KEYPOINT_CONFIDENCE
    ]
    placed = _place(by_name, p, face)
    if placed is None:
        return None
    return [
        {
            "name": bone["name"],
            "parent": bone["parent"],
            # Depth is the template's, always. One image fixes x and z and says
            # nothing whatever about y, and inventing a depth from a silhouette
            # is how a skeleton ends up in front of the chest it belongs in.
            "head": [placed[bone["name"]][0][0], bone["head"][1], placed[bone["name"]][0][1]],
            "tail": [placed[bone["name"]][1][0], bone["tail"][1], placed[bone["name"]][1][1]],
        }
        for bone in template.bones
    ]


Point = tuple[float, float]


def _framer(bbox: Sequence[int]):
    """The pixel -> normalized conversion for one subject bbox, or ``None``.

    A degenerate bbox is refused rather than divided by: ``measure`` can report
    a one-pixel subject, and an infinity here would sail through every gate
    below and only fail inside Blender.
    """
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    width, height = x1 - x0, y1 - y0
    if width <= 0.0 or height <= 0.0:
        log.info("pose fit: degenerate subject bbox %r", tuple(bbox))
        return None
    centre_x = (x0 + x1) / 2.0

    def convert(px: float, py: float) -> Point:
        return ((px - centre_x) / width, 1.0 - (py - y0) / height)

    return convert


def _inside(point: Point) -> bool:
    x, z = point
    return abs(x) <= 0.5 + BBOX_TOLERANCE and -BBOX_TOLERANCE <= z <= 1.0 + BBOX_TOLERANCE


def _stands_up(p: dict[str, Point]) -> bool:
    """Shoulders above hips above knees above ankles.

    The cheapest possible check that the detection is of a figure standing the
    way the template assumes, and it is worth having because the failure it
    catches is the expensive one: landmarks scattered over a character sheet or
    split between two subjects still clear every confidence floor individually.
    """
    levels = [
        _mid(p["left_shoulder"], p["right_shoulder"])[1],
        _mid(p["left_hip"], p["right_hip"])[1],
        _mid(p["left_knee"], p["right_knee"])[1],
        _mid(p["left_ankle"], p["right_ankle"])[1],
    ]
    return all(a > b for a, b in itertools.pairwise(levels))


def _place(
    template: dict[str, dict[str, Any]], p: dict[str, Point], face: list[Point]
) -> dict[str, tuple[Point, Point]] | None:
    """Every bone's (head, tail) in the frame's x/z, or ``None``.

    Proportions are read out of the template at call time rather than written
    down here as constants: where the shoulder bone starts along the collar,
    how long a hand is relative to its forearm, how far up the spine the chest
    begins. Editing ``humanoid.json`` then moves this fit with it, which is the
    only way the informed fit and the bbox fit can stay two views of one
    skeleton.
    """
    pelvis = _mid(p["left_hip"], p["right_hip"])
    chest_top = _mid(p["left_shoulder"], p["right_shoulder"])

    # The template's own pelvis and shoulder line, measured the same way the
    # detected ones are, so the fractions below are comparable.
    t_pelvis = _mid(_xz(template["thigh.L"]["head"]), _xz(template["thigh.R"]["head"]))
    t_chest_top = _mid(
        _xz(template["upper_arm.L"]["head"]), _xz(template["upper_arm.R"]["head"])
    )
    torso = t_chest_top[1] - t_pelvis[1]
    if torso <= 0.0:
        log.info("pose fit: the template's torso has no height")
        return None

    def up(name: str, end: str) -> Point:
        """A template joint's height as a fraction of its torso, applied to the
        measured one -- so a leaning subject gets a leaning spine rather than a
        vertical one stood up from the hips."""
        return _lerp(pelvis, chest_top, (_xz(template[name][end])[1] - t_pelvis[1]) / torso)

    spine_head = up("spine", "head")
    chest_head = up("chest", "head")
    neck_head = up("neck", "head")

    # The crown is pinned to the top of the frame because that is what the top
    # of the frame *is*: the highest subject pixel, which on anything humanoid
    # is the top of the head. The face landmarks only decide which way it
    # leans; with none of them confident it sits straight above the shoulders.
    crown = (_mean(x for x, _ in face) if face else chest_top[0], 1.0)
    head_span = _xz(template["head"]["tail"])[1] - _xz(template["neck"]["head"])[1]
    head_head = _lerp(
        neck_head,
        crown,
        (_xz(template["head"]["head"])[1] - _xz(template["neck"]["head"])[1]) / head_span
        if head_span > 0.0
        else 0.5,
    )

    placed: dict[str, tuple[Point, Point]] = {
        "hips": (pelvis, spine_head),
        "spine": (spine_head, chest_head),
        "chest": (chest_head, neck_head),
        "neck": (neck_head, head_head),
        "head": (head_head, crown),
    }
    for side, tag in (("left", "L"), ("right", "R")):
        shoulder, elbow = p[f"{side}_shoulder"], p[f"{side}_elbow"]
        wrist = p[f"{side}_wrist"]
        # The collar bone starts near the spine and ends at the shoulder joint;
        # the template says how near, as a fraction of that same span.
        inset = _ratio(
            _xz(template[f"shoulder.{tag}"]["head"]),
            t_chest_top,
            _xz(template[f"shoulder.{tag}"]["tail"]),
        )
        # A hand has no landmark of its own -- COCO stops at the wrist -- so it
        # continues the forearm, at the length the template gives it relative
        # to that forearm. A pose estimator will not tell us where the fingers
        # point, and continuing the arm is the one answer that is never
        # gratuitously wrong.
        reach = _ratio(
            _xz(template[f"hand.{tag}"]["tail"]),
            _xz(template[f"hand.{tag}"]["head"]),
            _xz(template[f"forearm.{tag}"]["head"]),
        )
        hip, knee, ankle = p[f"{side}_hip"], p[f"{side}_knee"], p[f"{side}_ankle"]
        toe = _xz(template[f"foot.{tag}"]["tail"])
        heel = _xz(template[f"foot.{tag}"]["head"])
        placed.update(
            {
                f"shoulder.{tag}": (_lerp(chest_top, shoulder, inset), shoulder),
                f"upper_arm.{tag}": (shoulder, elbow),
                f"forearm.{tag}": (elbow, wrist),
                f"hand.{tag}": (wrist, _extend(elbow, wrist, reach)),
                f"thigh.{tag}": (hip, knee),
                f"shin.{tag}": (knee, ankle),
                f"foot.{tag}": (
                    ankle,
                    # Clamped at the floor: the template's toe sits below its
                    # ankle, and a subject already standing on the bottom edge
                    # would otherwise get a toe *under* the mesh, where
                    # automatic weights have nothing to bind to.
                    (ankle[0] + toe[0] - heel[0], max(0.0, ankle[1] + toe[1] - heel[1])),
                ),
            }
        )
    return placed


def _xz(point: Sequence[float]) -> Point:
    """A template's 3-vector as the two axes an image can speak about."""
    return (float(point[0]), float(point[2]))


def _mid(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _extend(start: Point, end: Point, scale: float) -> Point:
    """``end``, carried on past itself by ``scale`` of the ``start``->``end``
    step. A zero-length step gives a zero-length bone, which ``fit_template``
    already nudges apart -- it is the one guard that has to live there, because
    only the world-space scale says how short is too short."""
    return (end[0] + (end[0] - start[0]) * scale, end[1] + (end[1] - start[1]) * scale)


def _ratio(a: Point, b: Point, reference: Point) -> float:
    """``|a - b|`` as a fraction of ``|reference - b|``. Zero when the
    reference span is degenerate, which places the joint on ``b`` rather than
    raising -- a template edit should cost proportion, never the fit."""
    span = math.dist(reference, b)
    return math.dist(a, b) / span if span > 0.0 else 0.0


def _mean(values) -> float:
    collected = list(values)
    return sum(collected) / len(collected)


# --- the impure half --------------------------------------------------------


def model_dir(config: Any = None) -> Path:
    from ..config import get_config

    spec = models.POSE_MODELS[models.DEFAULT_POSE_MODEL]
    return Path((config or get_config()).t2i_model_root) / spec.dir_name


def available(config: Any = None) -> bool:
    """Whether the weights are on disk. Checked before any torch import, the
    same load-bearing ordering ``matting.available`` and ``bench.metrics`` use:
    a checkout with no image extra must get "download it once with ..." rather
    than an ImportError about a dependency that was never the problem."""
    return (model_dir(config) / "config.json").exists()


def unload() -> None:
    """Drop the loaded model, and the memory of one that would not load.

    The counterpart to ``_load``, and it forgets the failure sentinel too, so a
    user who repairs a half-downloaded checkpoint gets another attempt without
    restarting the app.
    """
    _cache.clear()


# The sentinel a failed load leaves behind, in the same dict as the model so
# clearing one clears the other. A checkpoint that cannot load cannot load
# again, and from_pretrained is seconds of work per attempt -- on a queue
# draining a sweep, one broken install would otherwise cost every rig in it.
_FAILED = object()

_cache: dict[str, Any] = {}


def _load(path: Path):
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    key = str(path)
    hit = _cache.get(key)
    if hit is _FAILED:
        raise RuntimeError(f"the pose model at {path} failed to load earlier this session")
    if hit is not None:
        return hit
    try:
        processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
        model = VitPoseForPoseEstimation.from_pretrained(str(path), local_files_only=True)
        model.eval()
        # CPU, deliberately and not as a fallback. This runs on the job queue
        # beside a resident trellis-server and a resident SDXL pipe, and a
        # 90 MB pose model taking VRAM from the two models that are actually
        # producing the asset would be a bad trade for a second of host compute
        # -- the same one matting.py makes, and the reason vram.py's cost table
        # needs no entry for this.
        model = model.to("cpu")
    except Exception:
        _cache[key] = _FAILED
        raise
    _cache[key] = (processor, model)
    return _cache[key]


def detect(
    image_path: Path, subject_bbox: Sequence[int], config: Any = None
) -> list[Keypoint] | None:
    """The COCO-17 landmarks of the subject in ``image_path``, or ``None``.

    ``subject_bbox`` is the person box handed to the model, and it comes from
    ``reference.measure`` rather than from a detector of its own: the reference
    has exactly one subject -- the composition gate refuses it otherwise -- and
    its silhouette is already measured, so a second network to find a box that
    is already known would be pure cost.

    Blocking (a torch forward on the CPU), so every caller dispatches it
    through ``asyncio.to_thread``. Returns ``None`` rather than raising on any
    failure, because the only thing downstream of it is a fallback.
    """
    if not available(config):
        return None
    try:
        return _detect(image_path, subject_bbox, model_dir(config))
    except Exception:
        # Never fatal, and never a traceback per job either: a rig whose joints
        # come from the bbox is the behaviour this whole module is optional on
        # top of.
        log.warning("pose detection failed for %s; falling back to the bbox fit", image_path)
        log.debug("pose detection traceback", exc_info=True)
        return None


def _detect(image_path: Path, subject_bbox: Sequence[int], path: Path) -> list[Keypoint] | None:
    """The model half, isolated so the gating above can be tested without
    weights by patching exactly this."""
    import torch
    from PIL import Image

    processor, model = _load(path)
    boxes = [[_coco_box(subject_bbox)]]
    with Image.open(image_path) as image:
        image.load()
        inputs = processor(image.convert("RGB"), boxes=boxes, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # No score threshold, deliberately: filtering here drops rows and the
    # gating in refit is what decides confidence, on a floor this module owns
    # rather than one buried in a call site.
    people = processor.post_process_pose_estimation(outputs, boxes=boxes)[0]
    if not people:
        return None
    person = people[0]
    return _named(
        person["keypoints"].tolist(),
        person["scores"].tolist(),
        person["labels"].tolist(),
    )


def _coco_box(bbox: Sequence[int]) -> list[float]:
    """``(x0, y0, x1, y1)`` -> the ``(x, y, width, height)`` the processor
    documents.

    Its own function because it is the one silent failure in the whole model
    half: ``post_process_pose_estimation`` maps heatmap coordinates back onto
    the image *through* this box, so handing it corners produces seventeen
    landmarks in plausible-looking wrong places and raises nothing.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return [x0, y0, x1 - x0, y1 - y0]


def _named(
    points: list[list[float]], scores: list[float], labels: list[int]
) -> list[Keypoint]:
    """The model's rows -> named landmarks, addressed by its own label index.

    By label rather than by row, because the row order is not a contract: the
    processor's optional score threshold drops rows and the survivors shift up,
    which by position relabels an ankle as an eye. A label outside COCO's 17 is
    dropped rather than guessed at -- the gate in ``refit`` then sees a missing
    landmark, which is exactly what it is.
    """
    found = []
    for point, score, label in zip(points, scores, labels, strict=False):
        if 0 <= int(label) < len(COCO_KEYPOINTS):
            found.append(
                Keypoint(
                    COCO_KEYPOINTS[int(label)], float(point[0]), float(point[1]), float(score)
                )
            )
    return found
