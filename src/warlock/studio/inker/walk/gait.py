"""The walk itself: a rest rig plus four numbers, evaluated at a phase.

**Everything here is a periodic function of ``t``, and the frames are samples of
it.** That is the design decision the rest follows from. A keyframe table would
need a seam rule -- Poser's clips have one (``closed``, and no segment landing on
its far key) because a table has no choice but to say what happens between the
last key and the first. A function of phase has no seam to get wrong: ``pose(0)``
and ``pose(1)`` are the same call with the same answer, and the test that says so
is checking arithmetic rather than a convention.

**The target is clamped, never the bones.** ``two_bone`` shortens what it is
reaching for until the leg can reach it, so a stride past the leg's span comes
out as a shorter step. The alternative -- clamping the stride slider -- puts the
same rule somewhere a moved joint can invalidate it, and a rig whose hip has just
been dragged upward would then pose with a stretched thigh until somebody
re-clamped. Limb lengths are measured once by ``rig.segment_lengths`` and are
never an output of anything in this module.

**Legs are solved, arms are not**, and the asymmetry is deliberate. A leg has a
constraint worth solving for: the stance foot is on the ground line, and where
the knee goes follows from that. An arm has none, so a solver would need a wrist
target invented for it to reach -- which is FK with extra steps and one more
thing to get wrong. The arms swing on a sine and the forearm leads slightly,
which is what an arm does.

Angles are **screen degrees**: measured with ``atan2`` in image space, where y
runs down, so a positive angle turns clockwise on screen. ``render`` is the one
place that converts to Pillow's counter-clockwise convention, and it does it
once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from . import rig as R

#: The frame count is fixed for this prototype. It is the classic eight-frame
#: walk -- contact, down, passing, up, and the same four with the legs swapped --
#: and exposing it would mean deciding what a five-frame walk's passing pose is,
#: which is a question about animation rather than about a control.
WALK_FRAMES = 8

#: Fraction of the cycle each leg spends on the ground. A half, so the two legs
#: are exact opposites and every frame has exactly one stance foot. A real walk
#: is nearer 0.6 with both feet down at the contacts, and that difference is a
#: thing to judge on the ogre rather than to assume.
STANCE = 0.5

#: How much further the forearm swings than the upper arm. Slightly more than
#: one because the elbow trails the shoulder through a swing; below one the arm
#: reads as a plank.
FOREARM_LEAD = 1.35

#: How much of the shin's turn the foot takes through the swing. The foot is
#: flat through stance -- it is on the ground, which is the whole reason it is a
#: separate part -- and picks the shin up gradually rather than snapping to it.
FOOT_FOLLOW = 0.45

#: Knees bend forward for a figure facing right: flex a knee and the knee goes
#: forward while the heel goes back. Multiplied by ``rig.facing``, so a drawing
#: facing left needs no second code path.
KNEE_BEND = 1

#: Defaults, as fractions of the leg's own length. A sprite may be twelve pixels
#: tall or two hundred, so a pixel default would be wrong for all but one of
#: them.
DEFAULT_STRIDE = 0.45
DEFAULT_LIFT = 0.12
DEFAULT_BOB = 0.045
DEFAULT_ARM_SWING = 24.0

#: How far the hip may sink below its rest height at a contact, as a fraction of
#: the leg. The sink is not decoration -- see :func:`contact_drop` -- so this is
#: what really bounds the stride, and 0.3 is already a deep, deliberate-looking
#: step for a sprite.
MAX_DROP = 0.30

#: The longest step, as a fraction of the leg, regardless of what the geometry
#: would allow. Past this it stops reading as a walk and starts reading as a
#: lunge, and no amount of reach makes that the thing the user asked for.
MAX_STRIDE = 0.90

#: How close to full extension a solved leg may come. A leg at exactly its span
#: has a knee angle whose second derivative is unbounded, so a contact frame
#: snaps straight and the frame either side does not; a hair short of it reads as
#: a straight leg and moves continuously.
REACH_MARGIN = 0.02


@dataclass(frozen=True)
class WalkSettings:
    """The four numbers the panel exposes, plus the frame duration.

    Pixels, not fractions, because that is what the sliders say and what a user
    reasons about when a foot clips the canvas. :func:`defaults_for` mints them
    from the rig once; after that they are the user's.
    """

    stride: float = 0.0
    lift: float = 0.0
    bob: float = 0.0
    arm_swing: float = DEFAULT_ARM_SWING
    duration_ms: int = 100

    def replaced(self, **changes: float) -> WalkSettings:
        return replace(self, **changes)


def defaults_for(rig: R.Rig) -> WalkSettings:
    """Sensible numbers for this rig, in pixels, minted from its leg length."""
    leg = R.leg_length(rig)
    return WalkSettings(
        stride=min(leg * DEFAULT_STRIDE, reachable_stride(rig)),
        lift=leg * DEFAULT_LIFT,
        bob=leg * DEFAULT_BOB,
        arm_swing=DEFAULT_ARM_SWING,
        duration_ms=100,
    )


def _span(rig: R.Rig, side: str) -> float:
    return R.leg_length(rig, side) * (1.0 - REACH_MARGIN)


def rest_drop(rig: R.Rig, side: str = "near") -> float:
    """Hip to ground line, straight down, in the rest pose."""
    hip = rig.joints.get(f"{side}_hip")
    return 0.0 if hip is None else rig.ground_y - hip[1]


def contact_drop(rig: R.Rig, stride: float, side: str = "near") -> float:
    """How far the hip must sink for the leg to reach a contact of this length.

    **This is why the body bobs at all.** A figure is normally drawn standing
    with its legs straight, so the hip is already exactly a leg's length above
    the ground and a step of *any* size is out of reach -- the leg would have to
    grow. Every animator's answer is the same: the body drops on the contacts and
    rises through the passing pose. So the sink is derived from the stride rather
    than asked for, and the ``bob`` slider only ever adds to it.

    Solving ``hypot(stride/2, rest_drop - sink) == span`` for the sink.
    """
    span = _span(rig, side)
    half = max(0.0, stride) / 2.0
    if span <= 0.0:
        return 0.0
    if half >= span:
        return max(0.0, rest_drop(rig, side))
    return max(0.0, rest_drop(rig, side) - math.sqrt(span * span - half * half))


def reachable_stride(rig: R.Rig, side: str = "near") -> float:
    """The longest step this rig can hold, and the stride slider's upper bound.

    Bounded twice, and by the tighter of the two: by how deep a sink
    :data:`MAX_DROP` permits, and by :data:`MAX_STRIDE`. Exact rather than a
    guess, because the bound has to move the moment the user drags the hip or the
    ground line.
    """
    leg = R.leg_length(rig, side)
    span = _span(rig, side)
    if span <= 0.0:
        return 0.0
    reach_at_lowest = max(0.0, rest_drop(rig, side) - leg * MAX_DROP)
    if reach_at_lowest >= span:
        return 0.0
    geometric = 2.0 * math.sqrt(span * span - reach_at_lowest * reach_at_lowest)
    return min(geometric, leg * MAX_STRIDE)


def clamp_target(
    root: R.Point, target: R.Point, upper: float, lower: float
) -> R.Point:
    """``target`` pulled along the root-to-target line until the chain reaches it.

    **The one place a pose is compromised, and it compromises the target.** The
    bones are measured once from the rest drawing and are never an output of
    anything here; a target outside ``[|upper - lower|, upper + lower]`` has no
    solution at all, and the honest answer is the nearest pose the limb can
    actually hold. Clamping the stride slider instead would put the same rule
    somewhere a dragged hip could invalidate.
    """
    rx, ry = float(root[0]), float(root[1])
    dx, dy = float(target[0]) - rx, float(target[1]) - ry
    distance = math.hypot(dx, dy)
    if distance < 1e-9:
        # Degenerate: the target is the root. Point the chain along +x, which is
        # at least continuous with the poses either side.
        dx, dy, distance = 1.0, 0.0, 1.0
    span = (upper + lower) * (1.0 - REACH_MARGIN)
    gap = abs(upper - lower) + 1e-6
    clamped = min(max(distance, gap), max(gap, span))
    if abs(clamped - distance) < 1e-12:
        return (float(target[0]), float(target[1]))
    return (rx + dx / distance * clamped, ry + dy / distance * clamped)


def two_bone(root: R.Point, target: R.Point, upper: float, lower: float, bend: int) -> R.Point:
    """Where the middle joint goes for a two-bone chain reaching ``target``.

    The standard circle intersection: the middle joint is ``upper`` from the root
    and ``lower`` from the target, so it sits where two circles meet, and
    ``bend`` picks which of the two meeting points.

    The target is put through :func:`clamp_target` first, so this always returns
    a joint for which both bones are exactly their measured length. A caller that
    needs to know where the chain actually ended up asks ``clamp_target`` for the
    same answer rather than re-deriving it.
    """
    reached = clamp_target(root, target, upper, lower)
    rx, ry = float(root[0]), float(root[1])
    dx, dy = reached[0] - rx, reached[1] - ry
    distance = math.hypot(dx, dy) or 1.0
    ux, uy = dx / distance, dy / distance
    # Distance from the root to the foot of the middle joint's perpendicular.
    along = (distance * distance + upper * upper - lower * lower) / (2.0 * distance)
    across = math.sqrt(max(0.0, upper * upper - along * along))
    # The perpendicular, a quarter turn in screen space (y down), so a positive
    # ``bend`` puts the middle joint on the +x side of a chain pointing straight
    # down -- a knee in front of a standing leg, for a figure facing right.
    px, py = -uy, ux
    return (rx + ux * along - px * across * bend, ry + uy * along - py * across * bend)


def reach(rig: R.Rig, side: str = "near") -> float:
    """How far this leg can actually reach, margin included."""
    return _span(rig, side)


def screen_angle(a: R.Point, b: R.Point) -> float:
    """The direction from ``a`` to ``b``, in degrees, clockwise-positive.

    Image space has y running down, so ``atan2(dy, dx)`` grows clockwise as it
    is drawn. Everything in this package works in that convention and ``render``
    converts once, at the one place a Pillow rotation is asked for.
    """
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


@dataclass(frozen=True)
class Pose:
    """One frame: where every joint is, and how far every part has turned.

    Both, and not one derived from the other on demand, because the two have
    different consumers -- the overlay draws joints, the renderer turns parts --
    and deriving either from the other at draw time is how a preview comes to
    disagree with a bake.
    """

    joints: dict[str, R.Point]
    angles: dict[str, float]
    #: Whether each leg is on the ground this frame, keyed by side.
    grounded: dict[str, bool]


def _phase(side: str) -> float:
    """Where in the cycle this leg's stance begins. The two legs are opposites."""
    return 0.0 if side == "near" else 0.5


def _ankle(rig: R.Rig, settings: WalkSettings, side: str, t: float) -> tuple[R.Point, bool]:
    """Where this leg's ankle is at phase ``t``, and whether it is on the ground.

    Stance is the first half of the leg's own cycle and the ankle travels
    *backwards* along the ground line, from ``+stride/2`` to ``-stride/2``
    relative to the hip. That is what makes this an in-place walk: the body does
    not move, so the ground does.

    Swing is the second half and brings it forward again on an arc that peaks at
    ``lift``. A sine arc rather than a parabola so that the vertical velocity is
    zero at both ends -- the foot lands and leaves flat, instead of stabbing at
    the ground line and bouncing off it.
    """
    # The *rest* hip, not the posed one: the body bobs vertically and the foot
    # travel is horizontal, so anchoring the step on a hip that is moving would
    # make the bob shorten and lengthen the stride.
    hip = rig.joints[f"{side}_hip"]
    facing = rig.facing
    local = (t - _phase(side)) % 1.0
    half = settings.stride / 2.0
    if local < STANCE:
        u = local / STANCE if STANCE > 0.0 else 0.0
        offset = half - settings.stride * u
        return ((hip[0] + offset * facing, rig.ground_y), True)
    u = (local - STANCE) / (1.0 - STANCE) if STANCE < 1.0 else 0.0
    offset = -half + settings.stride * u
    lift = settings.lift * math.sin(math.pi * u)
    return ((hip[0] + offset * facing, rig.ground_y - lift), False)


def required_drop(rig: R.Rig, settings: WalkSettings, t: float) -> float:
    """How far the hip **must** sink at phase ``t`` for the stance foot to reach.

    The stance foot's position is the authority and the body's height follows
    from it, rather than the other way round. That inversion is what makes "the
    stance foot is on the ground line" true by construction instead of true to
    within whatever the clamp happened to allow -- and the clamp firing is
    exactly what a walk looks like when the feet skate.

    It also explains why a body bobs at all. A figure is drawn standing with its
    legs straight, so the hip is already a full leg above the ground; a step of
    any length puts the foot further away than that, and the only way to reach it
    without growing the leg is to come down.
    """
    worst = 0.0
    for side in ("near", "far"):
        # The hip first: a rig is asked to pose while it is still being
        # assembled, so a leg whose joints are not placed yet has no demand to
        # make and ``_ankle`` has nothing to answer with.
        hip = rig.joints.get(f"{side}_hip")
        if hip is None:
            continue
        target, down = _ankle(rig, settings, side, t)
        if not down:
            continue
        span = _span(rig, side)
        dx = target[0] - hip[0]
        reachable_drop = math.sqrt(max(0.0, span * span - dx * dx))
        worst = max(worst, rest_drop(rig, side) - reachable_drop)
    return max(0.0, worst)


def root_offset(rig: R.Rig, settings: WalkSettings, t: float) -> R.Point:
    """How far the body has moved from its rest position at phase ``t``.

    Horizontally nothing -- this is an in-place walk, so the ground moves under
    the figure and the figure does not move.

    Vertically, the deeper of two curves: what the stance leg demands
    (:func:`required_drop`) and the bob the user asked for, the classic
    two-per-cycle shape that is lowest at the contacts and highest through the
    passing poses. Taking the maximum rather than the sum means the slider
    *deepens* the natural sink instead of stacking on top of it, and can never
    take the body above where the leg can still reach the ground.
    """
    bob = max(0.0, settings.bob) * (1.0 + math.cos(4.0 * math.pi * t)) / 2.0
    return (0.0, max(required_drop(rig, settings, t), bob))


def _arm_angles(rig: R.Rig, settings: WalkSettings, side: str, t: float) -> dict[str, float]:
    """Upper and lower arm turns, in screen degrees, relative to the rest pose.

    The arm opposes the leg on the *same* side -- a near arm back when the near
    foot is forward -- which is what "opposing arm swing" means.

    **A cosine on the leg's own phase, and deliberately not a sine half a cycle
    later**, which is the shape this looked like it wanted and is wrong. The foot
    is driven by a *trajectory* that peaks forward at the contact, so its
    fundamental is a cosine; the arm is driven by a *rotation*, and a positive
    screen angle swings a downward-hanging limb backwards. Those two facts cancel
    one sign between them, so opposition is a cosine on the same phase, and half
    a cycle of offset would put the arm back in step with the leg it is meant to
    fight.
    """
    facing = rig.facing
    local = (t - _phase(side)) % 1.0
    upper = settings.arm_swing * math.cos(2.0 * math.pi * local) * facing
    return {
        f"{side}_upper_arm": upper,
        f"{side}_lower_arm": upper * FOREARM_LEAD,
        f"{side}_hand": upper * FOREARM_LEAD,
    }


def pose(rig: R.Rig, settings: WalkSettings, t: float) -> Pose:
    """The whole figure at phase ``t``. Periodic: ``pose(0)`` equals ``pose(1)``."""
    lengths = R.segment_lengths(rig)
    offset = root_offset(rig, settings, t)
    joints: dict[str, R.Point] = {}
    angles: dict[str, float] = {name: 0.0 for name in R.PART_NAMES}
    grounded: dict[str, bool] = {}

    # The body translates and does not turn: rigid segments, by decision. Every
    # joint that hangs off the torso rather than off a limb moves with it.
    for name in ("neck", "near_shoulder", "far_shoulder", "near_hip", "far_hip"):
        if name in rig.joints:
            rest = rig.joints[name]
            joints[name] = (rest[0] + offset[0], rest[1] + offset[1])

    for side in ("near", "far"):
        _pose_leg(rig, settings, side, t, lengths, joints, angles, grounded)
        _pose_arm(rig, settings, side, t, joints, angles)

    return Pose(joints=joints, angles=angles, grounded=grounded)


def _pose_leg(
    rig: R.Rig,
    settings: WalkSettings,
    side: str,
    t: float,
    lengths: dict[str, float],
    joints: dict[str, R.Point],
    angles: dict[str, float],
    grounded: dict[str, bool],
) -> None:
    hip_name, knee_name = f"{side}_hip", f"{side}_knee"
    ankle_name, toe_name = f"{side}_ankle", f"{side}_toe"
    if hip_name not in joints or knee_name not in rig.joints or ankle_name not in rig.joints:
        return
    hip = joints[hip_name]
    wanted, down = _ankle(rig, settings, side, t)
    grounded[side] = down
    thigh = lengths.get(f"{side}_thigh", 0.0)
    shin = lengths.get(f"{side}_shin", 0.0)
    # The reachable ankle, not the requested one. Recording the request would
    # make the shin measure longer than it is on any frame the clamp fired --
    # which is the bug this whole rule exists to prevent, one level up.
    ankle = clamp_target(hip, wanted, thigh, shin)
    knee = two_bone(hip, ankle, thigh, shin, KNEE_BEND * rig.facing)
    joints[knee_name] = knee
    joints[ankle_name] = ankle

    rest_hip = rig.joints[hip_name]
    rest_knee = rig.joints[knee_name]
    rest_ankle = rig.joints[ankle_name]
    thigh_turn = screen_angle(hip, knee) - screen_angle(rest_hip, rest_knee)
    shin_turn = screen_angle(knee, ankle) - screen_angle(rest_knee, rest_ankle)
    angles[f"{side}_thigh"] = _wrapped(thigh_turn)
    angles[f"{side}_shin"] = _wrapped(shin_turn)
    # Flat through stance -- it is standing on the ground line, and the rest
    # drawing already had it flat -- and picking the shin up through the swing.
    foot_turn = 0.0 if down else _wrapped(shin_turn) * FOOT_FOLLOW
    angles[f"{side}_foot"] = foot_turn
    if toe_name in rig.joints:
        joints[toe_name] = _turned(rig.joints[toe_name], rest_ankle, ankle, foot_turn)


def _pose_arm(
    rig: R.Rig,
    settings: WalkSettings,
    side: str,
    t: float,
    joints: dict[str, R.Point],
    angles: dict[str, float],
) -> None:
    """Forward kinematics: turn the rest arm about the shoulder, then the elbow.

    No lengths are read, and that is the point of doing the arms this way -- the
    rest geometry *is* the length, so a turn about a joint cannot change one.
    """
    shoulder_name = f"{side}_shoulder"
    if shoulder_name not in joints:
        return
    turns = _arm_angles(rig, settings, side, t)
    angles.update(turns)
    shoulder = joints[shoulder_name]
    rest_shoulder = rig.joints[shoulder_name]
    elbow_name, wrist_name = f"{side}_elbow", f"{side}_wrist"
    if elbow_name in rig.joints:
        joints[elbow_name] = _turned(
            rig.joints[elbow_name], rest_shoulder, shoulder, turns[f"{side}_upper_arm"]
        )
    if wrist_name in rig.joints and elbow_name in joints:
        joints[wrist_name] = _turned(
            rig.joints[wrist_name],
            rig.joints[elbow_name],
            joints[elbow_name],
            turns[f"{side}_lower_arm"],
        )


def _turned(point: R.Point, rest_pivot: R.Point, pivot: R.Point, degrees: float) -> R.Point:
    """``point`` carried from ``rest_pivot`` to ``pivot`` and turned about it.

    The same arithmetic the renderer applies to pixels, applied to a point --
    which is what keeps the overlay's joints on top of the art the bake produces.
    """
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    dx, dy = point[0] - rest_pivot[0], point[1] - rest_pivot[1]
    return (pivot[0] + dx * cos - dy * sin, pivot[1] + dx * sin + dy * cos)


def _wrapped(degrees: float) -> float:
    """An angle difference brought into (-180, 180].

    A thigh whose rest direction is just short of straight down and whose posed
    direction is just past it differs by 359 degrees without this, and the part
    spins the long way round for one frame.
    """
    return (degrees + 180.0) % 360.0 - 180.0


def phases(count: int = WALK_FRAMES) -> tuple[float, ...]:
    """The sample points. ``i / count``, so the last frame is one step short of
    the first and the cycle closes with no duplicated frame."""
    return tuple(i / float(count) for i in range(count))


def cycle(rig: R.Rig, settings: WalkSettings, count: int = WALK_FRAMES) -> list[Pose]:
    return [pose(rig, settings, t) for t in phases(count)]
