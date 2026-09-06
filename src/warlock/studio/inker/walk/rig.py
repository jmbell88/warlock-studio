"""What the user assembles: cut-out body parts, and the joints between them.

A rig is a **rest pose**. The parts are planes lifted off the drawing as it was
drawn, the joints are points the user dragged onto that same drawing, and the
segment lengths are measured from those points **once**. Nothing downstream may
change a length -- ``gait`` clamps the target it is reaching for instead, which
is the one rule this whole feature rests on: a limb that stretches to hit a pose
stops being the limb the user drew.

Parts carry their own crop and origin rather than a canvas-sized plane. Fourteen
canvas-sized planes is fourteen copies of a drawing that is mostly transparent,
and the crop is what ``render`` wants anyway -- RotSprite's cost is quadratic in
the plane it is handed.

There is no persistence here, deliberately: a rig lives in the session that
built it and dies with it. Saved rigs are deferred, and the absence of a schema
is what keeps that decision cheap to revisit: there is no format to migrate and
nothing on disk that a later shape would have to stay compatible with.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]

#: Every joint the user places, in the order the panel lists them. ``toe`` is
#: here to give the foot a direction; without it a foot has a pivot and no
#: orientation, and there is no second point on a drawing to infer one from.
JOINTS: tuple[str, ...] = (
    "neck",
    "near_shoulder",
    "near_elbow",
    "near_wrist",
    "near_hip",
    "near_knee",
    "near_ankle",
    "near_toe",
    "far_shoulder",
    "far_elbow",
    "far_wrist",
    "far_hip",
    "far_knee",
    "far_ankle",
    "far_toe",
)

#: The near-limb joint each far-limb joint copies from, for "far starts as a
#: copy of near". Body joints (``neck``) have no pair and are not in the table.
FAR_OF_NEAR: dict[str, str] = {
    f"far_{name}": f"near_{name}"
    for name in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "toe")
}


@dataclass(frozen=True)
class PartSpec:
    """One cut-out and how it hangs off the skeleton.

    ``pivot`` is the joint the part turns about. ``direction`` is the joint pair
    whose angle *is* the part's orientation, so a part's rotation is measured
    from the drawing rather than accumulated down a chain of stored numbers --
    two spellings of an angle is one place for them to disagree. A part with no
    direction of its own (a head, a hand) names the part it ``follows``.
    """

    name: str
    pivot: str
    direction: tuple[str, str] | None = None
    follows: str = ""
    limb: str = ""  # "near_arm", "far_leg", ... ; "" for the body


def _arm(side: str) -> tuple[PartSpec, ...]:
    limb = f"{side}_arm"
    return (
        PartSpec(
            f"{side}_upper_arm",
            f"{side}_shoulder",
            (f"{side}_shoulder", f"{side}_elbow"),
            limb=limb,
        ),
        PartSpec(
            f"{side}_lower_arm",
            f"{side}_elbow",
            (f"{side}_elbow", f"{side}_wrist"),
            limb=limb,
        ),
        PartSpec(f"{side}_hand", f"{side}_wrist", follows=f"{side}_lower_arm", limb=limb),
    )


def _leg(side: str) -> tuple[PartSpec, ...]:
    limb = f"{side}_leg"
    return (
        PartSpec(f"{side}_thigh", f"{side}_hip", (f"{side}_hip", f"{side}_knee"), limb=limb),
        PartSpec(f"{side}_shin", f"{side}_knee", (f"{side}_knee", f"{side}_ankle"), limb=limb),
        PartSpec(f"{side}_foot", f"{side}_ankle", (f"{side}_ankle", f"{side}_toe"), limb=limb),
    )


#: Bottom-first, which is both the default draw order and the order the tracks
#: are laid down in. The far limbs are behind the body and the near ones in
#: front of it; for a side view that is the whole of the depth problem, which is
#: why nothing here reaches for ``cel_z``.
PARTS: tuple[PartSpec, ...] = (
    *_arm("far"),
    *_leg("far"),
    PartSpec("torso", "near_hip"),
    PartSpec("head", "neck"),
    *_leg("near"),
    *_arm("near"),
)

PART_NAMES: tuple[str, ...] = tuple(spec.name for spec in PARTS)
BY_NAME: dict[str, PartSpec] = {spec.name: spec for spec in PARTS}

#: The four limbs, for the copy-near-to-far button and the parts list's headings.
LIMBS: tuple[str, ...] = ("near_arm", "near_leg", "far_arm", "far_leg")

_KINDS = (
    "upper_arm",
    "lower_arm",
    "hand",
    "thigh",
    "shin",
    "foot",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
    "toe",
)

#: Human-readable, for refusals and for the panel. Kept here rather than in the
#: pane so a refusal reads the same in a test as it does on screen.
LABELS: dict[str, str] = {
    "torso": "torso",
    "head": "head",
    "neck": "neck",
    **{
        f"{side}_{kind}": f"{side} {kind.replace('_', ' ')}"
        for side in ("near", "far")
        for kind in _KINDS
    },
}


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


@dataclass
class Part:
    """One cut-out: a trimmed RGBA crop and where its top-left sits on the canvas.

    ``source`` is only ever shown to the user -- which layer or selection this
    came from. The engine never reads it, which is what keeps a document out of
    this package.
    """

    pixels: np.ndarray | None = None
    origin: tuple[int, int] = (0, 0)
    source: str = ""

    @property
    def assigned(self) -> bool:
        return self.pixels is not None

    def copy(self) -> Part:
        return Part(
            pixels=None if self.pixels is None else self.pixels.copy(),
            origin=self.origin,
            source=self.source,
        )


def trim(plane: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """A plane cropped to what it actually covers, and where the crop sits.

    Returns a 1x1 empty crop at the origin for a plane with no alpha at all,
    rather than ``None``: a part assigned from a blank layer is a rig problem,
    reported by :func:`missing_parts`, and not something every caller downstream
    should have to branch on.
    """
    alpha = plane[:, :, 3]
    rows = np.flatnonzero(alpha.any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return np.zeros((1, 1, 4), dtype=plane.dtype), (0, 0)
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return np.ascontiguousarray(plane[y0:y1, x0:x1]), (x0, y0)


def part_from_plane(
    plane: np.ndarray, *, origin: tuple[int, int] = (0, 0), source: str = ""
) -> Part:
    """A part from a canvas-sized layer or from a selection cutout.

    ``origin`` is where ``plane`` itself sits, so a cropped cutout and a
    full-canvas layer both arrive the same way and the trim composes with it.
    """
    crop, (x0, y0) = trim(plane)
    return Part(pixels=crop, origin=(origin[0] + x0, origin[1] + y0), source=source)


@dataclass
class Rig:
    """The rest pose: parts, joints, the ground line, and the draw order."""

    parts: dict[str, Part] = field(default_factory=lambda: {n: Part() for n in PART_NAMES})
    joints: dict[str, Point] = field(default_factory=dict)
    ground_y: float = 0.0
    order: tuple[str, ...] = PART_NAMES
    #: Bumped by every mutation, so a preview cache can be keyed on it without
    #: hashing fourteen planes. Every mutator below honours it, which is the
    #: only reason the pane can render on a revision comparison alone.
    rev: int = 0

    def copy(self) -> Rig:
        return Rig(
            parts={name: part.copy() for name, part in self.parts.items()},
            joints=dict(self.joints),
            ground_y=self.ground_y,
            order=self.order,
            rev=self.rev,
        )

    @property
    def facing(self) -> int:
        """+1 for a figure facing right, -1 for one facing left.

        Read off the near foot, which is the one part of a side view whose
        direction is unambiguous: a toe is in front of an ankle. Derived rather
        than asked for, because a control for it would be a second answer to a
        question the drawing has already answered.
        """
        ankle = self.joints.get("near_ankle")
        toe = self.joints.get("near_toe")
        if ankle is None or toe is None or abs(toe[0] - ankle[0]) < 1e-6:
            return 1
        return 1 if toe[0] > ankle[0] else -1


def blank(size: tuple[int, int]) -> Rig:
    """A rig with nothing assigned and the ground on the bottom row."""
    return Rig(ground_y=float(max(0, size[1] - 1)))


def default_ground(rig: Rig, height: int) -> float:
    """Where the ground line starts: under the lower of the two feet.

    A guess the user then drags, and a better one than the canvas floor -- a
    sprite is normally drawn with air beneath it, and a ground line at the last
    row would put the whole walk below the feet.
    """
    feet = [rig.joints[name] for name in ("near_ankle", "far_ankle") if name in rig.joints]
    if not feet:
        return float(max(0, height - 1))
    return float(max(point[1] for point in feet))


def set_joint(rig: Rig, name: str, point: Point) -> Rig:
    """One joint moved. Returns a new rig; the old one is left alone."""
    out = rig.copy()
    out.joints[name] = (float(point[0]), float(point[1]))
    out.rev = rig.rev + 1
    return out


def set_part(rig: Rig, name: str, part: Part) -> Rig:
    out = rig.copy()
    out.parts[name] = part
    out.rev = rig.rev + 1
    return out


def set_ground(rig: Rig, y: float) -> Rig:
    out = rig.copy()
    out.ground_y = float(y)
    out.rev = rig.rev + 1
    return out


def set_order(rig: Rig, order: Iterable[str]) -> Rig:
    order = tuple(order)
    if sorted(order) != sorted(PART_NAMES):
        raise ValueError("a draw order names every part exactly once")
    out = rig.copy()
    out.order = order
    out.rev = rig.rev + 1
    return out


def copy_near_to_far(rig: Rig, limb: str, *, brightness: float = 1.0) -> Rig:
    """Seed a far limb from the near one: **pixels and joints together**.

    The brief's starting point, and the reason it is one press rather than six.
    Copying the joints as well is what makes the copy usable immediately -- a far
    thigh with the near thigh's art and no hip placed is not a head start, it is
    half a job.

    ``brightness`` is the explicit adjustment the copy needs to stop reading as
    the near limb: a far arm that is pixel-identical to the near one in front of
    it reads as one arm. It multiplies colour only, never alpha, so a silhouette
    is untouched.
    """
    if limb not in ("arm", "leg"):
        raise ValueError(f"a limb is an arm or a leg, not {limb!r}")
    out = rig.copy()
    for spec in PARTS:
        if spec.limb != f"far_{limb}":
            continue
        near_name = "near_" + spec.name.removeprefix("far_")
        out.parts[spec.name] = _shaded(rig.parts[near_name], brightness)
    for far, near in FAR_OF_NEAR.items():
        if joint_limb(far) == f"far_{limb}" and near in rig.joints:
            out.joints[far] = rig.joints[near]
    out.rev = rig.rev + 1
    return out


def joint_limb(joint: str) -> str:
    """Which limb a joint belongs to, or ``""`` for a body joint."""
    side, _, kind = joint.partition("_")
    if kind in ("shoulder", "elbow", "wrist"):
        return f"{side}_arm"
    if kind in ("hip", "knee", "ankle", "toe"):
        return f"{side}_leg"
    return ""


def _shaded(part: Part, brightness: float) -> Part:
    out = part.copy()
    if out.pixels is None or abs(brightness - 1.0) < 1e-6:
        return out
    rgb = out.pixels[:, :, :3].astype(np.float32) * float(brightness)
    out.pixels[:, :, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return out


def required_joints(rig: Rig) -> tuple[str, ...]:
    """Every joint some assigned part depends on, plus both legs' own.

    The legs are always required even when unassigned, because the gait is
    driven off the hip, knee and ankle whether or not a thigh was ever cut out
    -- an arms-only rig still has to know where the body is.
    """
    needed: set[str] = set()
    for side in ("near", "far"):
        needed |= {f"{side}_hip", f"{side}_knee", f"{side}_ankle", f"{side}_toe"}
    for spec in PARTS:
        if not rig.parts[spec.name].assigned:
            continue
        needed.add(spec.pivot)
        if spec.direction is not None:
            needed.update(spec.direction)
    return tuple(name for name in JOINTS if name in needed)


def missing_joints(rig: Rig) -> tuple[str, ...]:
    return tuple(name for name in required_joints(rig) if name not in rig.joints)


def missing_parts(rig: Rig) -> tuple[str, ...]:
    """Parts with no art. The body is required; a limb is required as a set.

    A rig with no far arm at all is a legitimate drawing -- a figure in profile
    with one arm hidden behind it may simply not have one -- so a limb is
    refused only when it is *half* assigned, which is always a mistake rather
    than a choice. A rig with no leg at all is refused, because a walk with
    nothing to stand on is not a walk.
    """
    out: list[str] = []
    for name in ("torso", "head"):
        if not rig.parts[name].assigned:
            out.append(name)
    for limb in LIMBS:
        members = [spec.name for spec in PARTS if spec.limb == limb]
        assigned = [name for name in members if rig.parts[name].assigned]
        if assigned and len(assigned) != len(members):
            out += [name for name in members if name not in assigned]
    legs = [spec.name for spec in PARTS if spec.limb.endswith("leg")]
    if not any(rig.parts[name].assigned for name in legs):
        out.append("near_thigh")
    return tuple(dict.fromkeys(out))


#: Pixels below which a segment has no usable direction. Two joints a pixel
#: apart give an angle that swings wildly for a one-pixel drag, so the rig is
#: refused rather than rendered into a part that spins.
MIN_SEGMENT = 2.0


def segment_lengths(rig: Rig) -> dict[str, float]:
    """Every directed segment's length, measured **once**, from the rest pose.

    Keyed by the part that spans it. This is the table ``gait`` treats as
    immutable: a pose that cannot be reached shortens the step, it does not
    lengthen a thigh.
    """
    out: dict[str, float] = {}
    for spec in PARTS:
        if spec.direction is None:
            continue
        a = rig.joints.get(spec.direction[0])
        b = rig.joints.get(spec.direction[1])
        if a is None or b is None:
            continue
        out[spec.name] = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    return out


def leg_length(rig: Rig, side: str = "near") -> float:
    """Hip to ankle, fully extended. The scale every default is a fraction of."""
    lengths = segment_lengths(rig)
    return lengths.get(f"{side}_thigh", 0.0) + lengths.get(f"{side}_shin", 0.0)


def refusal(rig: Rig) -> str:
    """Why this rig cannot be baked yet, in one sentence, or ``""``.

    One string and not a list, because it is what a disabled button's tooltip
    says and what a test asserts, and two spellings of a refusal drift.
    """
    parts = missing_parts(rig)
    if parts:
        return "No art assigned to the " + _list(label(name) for name in parts) + "."
    joints = missing_joints(rig)
    if joints:
        return "Place the " + _list(label(name) for name in joints) + "."
    for name, length in sorted(segment_lengths(rig).items()):
        if length < MIN_SEGMENT:
            return f"The {label(name)} is too short to turn -- move its joints apart."
    return ""


def _list(names: Iterable[str]) -> str:
    items = list(names)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"
