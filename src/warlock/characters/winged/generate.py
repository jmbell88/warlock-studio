"""One parameterised winged generator, baked per silhouette group.

The humanoid generator's shape again -- primitives to a boolean union, one weld,
one Catmull-Clark level, one weld; regions from geometry after the union;
channels as per-vertex displacement fields evaluated by the same function over
the vertices and over the joints. Read ``characters/humanoid/generate.py``
first, and ``characters/quadruped/generate.py`` for why every extreme of the
bounding box is an absolute constant rather than a fraction of the box being
fitted.

**A dragon is a species here, not a substitution.** The brief's rule is that
anything nameable is makeable, and the winged archetype is where the nameable
flying things live, dragons included. What the shipped ``bird`` template settles
is not whether a dragon exists but *what shape* it is: the skeleton has one pair
of legs and one wing chain per side, so a dragon here is a winged biped -- which
is exactly what a wyvern already is, and what a griffin is approximated to.
A four-legged dragon is a winged-quadruped template plus a clip library plus a
generator, which is a body plan and not a parameter. That is recorded here
rather than hidden, because the difference is visible in the first render.

**A membrane and a feather are the same three bones.** ``wing_base``,
``wing_mid`` and ``wing_tip`` carry a leading-edge chain of capsules and one
flattened vane, whatever the group; what a bat has that a raven does not is
*finger spars* radiating through the membrane, which is extra geometry and
therefore a group rather than a channel. The wing chain itself never changes,
which is why one clip library flies all six species.

**Where the box comes from.** ``bird``'s landmarks are all strictly inside their
box -- the wingtip at 0.96 of the half width, the beak at 0.92 of the depth, the
tail tip at 0.96 -- with the single exception of the feet, which sit on the
floor. So unlike the quadruped there is no landmark on a corner and nothing has
to end in a point: the wing ends in a capsule cap on the box's side, the beak
and the tail in caps on its front and back, the skull's pole is its ceiling, and
the soles are its floor. Every one of those is written in absolute units so the
fixed-point loop in :func:`build` converges onto the smoothing's shrinkage
instead of chasing it down.

Clay is imported **inside the functions that use it**: this package is reachable
from ``warlock.characters``, which is imported by the door, and the boolean
kernel has no business loading behind a registry lookup.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import family as familylib
from ..errors import CharacterError

__all__ = [
    "GROUPS",
    "Baked",
    "Group",
    "bake",
    "build",
    "write_assets",
]

# --- the neutral flyer -------------------------------------------------------
#
# Fractions of total height, glTF axes (Y up, +X the subject's left, +Z
# forward), soles on y = 0 and the crown at y = 1.

R_BODY = 0.150
R_NECK = 0.058
R_SKULL = 0.105
R_BEAK = 0.038
R_TAIL = 0.052
R_THIGH = 0.042
R_SHIN = 0.032
R_WING = 0.028
R_SPAR = 0.013

#: Half-thickness of a wing vane. Thin, and the leading-edge chain is thicker
#: than it on purpose: the capsules stand proud of the plate on both faces, so
#: the two surfaces cross transversally and the union has no tangency to leave a
#: sliver at.
WING_THICK = 0.017

SEG = 10
RINGS = 3
HEAD_SEG = 12
HEAD_RINGS = 6

MAX_FACES = 12000

#: Weld tolerances, before and after the smoothing. The second is a hundredth of
#: the first for the quadruped module's measured reason: at 1e-5 it stops
#: tidying what Catmull-Clark recreated and starts collapsing real geometry
#: where three round surfaces meet at a shallow angle.
SEAM_EPS = 1e-5
SMOOTH_EPS = 1e-7

FIT_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class Group:
    """One silhouette group: the box it fills and what it grows."""

    #: Half the wingspan, as a fraction of height. The wingtip caps reach it.
    half_span: float
    #: Beak-to-tail depth, as a fraction of height.
    depth: float
    wing: str  # "feather" | "membrane"
    face: str  # "beak" | "snout"
    tail: str  # "fan" | "taper"
    horns: bool = False
    ears: bool = False
    #: Half-chord of the wing vane, as a fraction of height.
    chord: float = 0.17


#: ``silhouette -> Group``. The keys are exactly ``family.silhouettes`` for the
#: winged archetype, which ``test_every_silhouette_group_has_an_asset`` pins.
GROUPS: dict[str, Group] = {
    "feathered": Group(
        half_span=0.62, depth=1.02, wing="feather", face="beak", tail="fan", chord=0.17
    ),
    "membrane": Group(
        half_span=0.78, depth=0.80, wing="membrane", face="snout", tail="taper",
        ears=True, chord=0.21,
    ),
    "drake": Group(
        half_span=0.72, depth=1.34, wing="membrane", face="snout", tail="taper",
        horns=True, chord=0.20,
    ),
}


@dataclass
class Baked:
    """One silhouette group's mesh, its rig, its regions and its channels."""

    silhouette: str
    positions: np.ndarray
    loops: np.ndarray
    starts: np.ndarray
    regions: np.ndarray
    joints: list[dict[str, Any]]
    displacements: dict[str, np.ndarray]
    joint_displacements: dict[str, np.ndarray]
    bounds: tuple[np.ndarray, np.ndarray] = field(default=None)  # type: ignore[assignment]

    @property
    def face_count(self) -> int:
        return len(self.starts) - 1


# --- axes --------------------------------------------------------------------


def _to_gltf(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _to_blender(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


# --- the skeleton the mesh is grown around -----------------------------------


def _fit_joints(half_span: float, depth: float) -> list[dict[str, Any]]:
    from ... import rigging

    template = rigging.get_template("bird")
    lo = [-half_span, -depth / 2.0, 0.0]
    hi = [half_span, depth / 2.0, 1.0]
    return rigging.fit_template(template, lo, hi)


def _joint_points(joints: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {b["name"]: (_to_gltf(b["head"]), _to_gltf(b["tail"])) for b in joints}


# --- primitives placed in the world ------------------------------------------


def _basis_from_y(direction: np.ndarray) -> np.ndarray:
    d = np.asarray(direction, dtype="f8")
    d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = float(np.dot(y, d))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _placed(mesh: Any, matrix3: np.ndarray, centre: np.ndarray) -> Any:
    from ...studio.clay import mesh as bm

    matrix = np.eye(4)
    matrix[:3, :3] = matrix3
    matrix[:3, 3] = centre
    return bm.transformed(mesh, matrix)


def _capsule_between(a: Any, b: Any, radius: float, *, segments: int = SEG) -> Any:
    from ...studio.clay import primitives

    a = np.asarray(a, dtype="f8")
    b = np.asarray(b, dtype="f8")
    span = b - a
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return _sphere_at(a, radius)
    body = primitives.capsule(radius=radius, height=length, segments=segments, rings=RINGS)
    return _placed(body, _basis_from_y(span), (a + b) / 2.0)


def _sphere_at(centre: Any, radius: float, *, segments: int = SEG, rings: int = RINGS * 2) -> Any:
    from ...studio.clay import primitives

    body = primitives.uv_sphere(radius=radius, segments=segments, rings=rings)
    return _placed(body, np.eye(3), np.asarray(centre, dtype="f8"))


def _ellipsoid_at(
    centre: Any, radii: Any, *, segments: int = SEG, rings: int = RINGS * 2
) -> Any:
    """A sphere through a diagonal scale. The vane, and nothing else, needs it.

    A flattened *ellipsoid* rather than a flattened box because the plate has to
    cross the body and the leading-edge capsules, and every one of those
    intersections is between two curved surfaces -- a box's flat face lying
    almost in a capsule's tangent plane is exactly the shallow crossing that
    leaves a sliver the weld then pinches.
    """
    from ...studio.clay import primitives

    body = primitives.uv_sphere(radius=1.0, segments=segments, rings=rings)
    return _placed(body, np.diag(np.asarray(radii, dtype="f8")), np.asarray(centre, dtype="f8"))


def _cone_between(a: Any, b: Any, radius: float, *, segments: int = 8) -> Any:
    from ...studio.clay import primitives

    a = np.asarray(a, dtype="f8")
    b = np.asarray(b, dtype="f8")
    span = b - a
    length = float(np.linalg.norm(span))
    body = primitives.cone(radius=radius, height=length, segments=segments)
    return _placed(body, _basis_from_y(span), (a + b) / 2.0)


def _box_at(centre: Any, size: Any) -> Any:
    from ...studio.clay import primitives

    return _placed(primitives.box(size=size), np.eye(3), np.asarray(centre, dtype="f8"))


def _extended(a: np.ndarray, b: np.ndarray, axis: int, target: float) -> np.ndarray:
    """The point on the ray a->b whose *axis* coordinate is *target*.

    How a limb reaches the side of the box without the box being read off the
    estimate: the direction comes from the landmarks, the stopping point is an
    absolute number.
    """
    span = np.asarray(b, dtype="f8") - np.asarray(a, dtype="f8")
    if abs(float(span[axis])) < 1e-12:  # pragma: no cover - a template change
        raise CharacterError("a limb that does not run toward the box cannot reach it")
    t = (target - float(b[axis])) / float(span[axis])
    return np.asarray(b, dtype="f8") + t * span


# --- named landmarks the mesh and the regions both read ----------------------


def _skull_centre(group: Group, at) -> np.ndarray:
    """The head sphere's centre: its pole *is* the top of the box."""
    head_a, head_b = at["head"]
    return np.array([0.0, 1.0 - R_SKULL, float((head_a[2] + head_b[2]) / 2.0) + 0.02])


def _beak_tip(group: Group, at) -> np.ndarray:
    a, b = at["beak"]
    return _extended(a, b, 2, group.depth / 2.0 - R_BEAK * 0.55)


def _wing_chain(group: Group, at, side: str) -> list[np.ndarray]:
    """The four points of one wing's leading edge, tip last and on the box."""
    base_a = at[f"wing_base.{side}"][0]
    base_b = at[f"wing_base.{side}"][1]
    mid_b = at[f"wing_mid.{side}"][1]
    tip_b = at[f"wing_tip.{side}"][1]
    sign = 1.0 if side == "L" else -1.0
    cap = _extended(mid_b, tip_b, 0, sign * (group.half_span - R_WING))
    return [base_a, base_b, mid_b, tip_b, cap]


def _horn_segments(group: Group, at) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """``(base, tip, radius)`` per horn. Swept back, and under the crown."""
    if not group.horns:
        return []
    skull = _skull_centre(group, at)
    out = []
    for side in (1.0, -1.0):
        base = skull + [side * 0.048, 0.036, -0.030]
        tip = base + [side * 0.030, 0.022, -0.130]
        out.append((base, tip, 0.019))
        out.append((base + [side * 0.010, -0.020, 0.010],
                    base + [side * 0.048, -0.006, -0.070], 0.013))
    return out


def _spar_segments(group: Group, at, side: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """The finger spars of a membrane wing: leading edge to trailing edge."""
    if group.wing != "membrane":
        return []
    chain = _wing_chain(group, at, side)
    root = chain[1]
    out = []
    for point in chain[2:]:
        out.append((point, point + [0.0, -0.010, -group.chord * 1.55], R_SPAR))
    out.append((root, root + [0.0, -0.008, -group.chord * 1.05], R_SPAR))
    return out


def _eye_centres(group: Group, at) -> list[np.ndarray]:
    skull = _skull_centre(group, at)
    return [skull + [side * 0.062, 0.024, 0.058] for side in (1.0, -1.0)]


# --- the body ----------------------------------------------------------------

#: ``face -> (x, y, z)`` of the talon box. Centred on the ankle landmark, for
#: the quadruped module's measured reason: one Catmull-Clark level puts a quad's
#: new vertex at its own centroid, so the sole's lowest point is the bottom
#: face's centre and nowhere else.
FOOT_SIZE = (0.058, 0.042, 0.100)


def _parts(joints: list[dict[str, Any]], group: Group) -> list[tuple[str, Any]]:
    """Every closed solid the union is built from, named for its refusal text.

    The five parts that set the box, and which are therefore placed in absolute
    units rather than off the estimate: the two wingtip caps at ``+-half_span``,
    the beak cap at the front, the tail cap at the back, the skull's pole at
    ``y = 1`` and the two soles on ``y = 0``.
    """
    at = _joint_points(joints)
    half = group.depth / 2.0
    parts: list[tuple[str, Any]] = []

    hips = at["hips"][0]
    chest_tail = at["chest"][1]
    parts.append(("body", _capsule_between(hips, chest_tail, R_BODY)))
    parts.append(("breast", _sphere_at(at["chest"][0] + [0.0, -0.020, 0.045], R_BODY * 0.86)))

    neck_a, neck_b = at["neck"]
    parts.append(("neck", _capsule_between(neck_a, at["head"][0], R_NECK)))
    skull = _skull_centre(group, at)
    parts.append(("skull", _sphere_at(skull, R_SKULL, segments=HEAD_SEG, rings=HEAD_RINGS)))
    del neck_b

    tip = _beak_tip(group, at)
    if group.face == "beak":
        parts.append(("beak", _cone_between(skull + [0.0, -0.010, 0.030], tip, R_BEAK,
                                            segments=8)))
    else:
        parts.append((
            "snout",
            _capsule_between(skull + [0.0, -0.014, 0.020], tip, R_BEAK * 1.20, segments=8),
        ))
        parts.append(("jaw", _sphere_at(skull + [0.0, -0.052, 0.058], R_SKULL * 0.58)))

    for i, (a, b, r) in enumerate(_horn_segments(group, at)):
        parts.append((f"horn{i}", _cone_between(a, b, r, segments=6)))

    if group.ears:
        for side in (1.0, -1.0):
            base = skull + [side * 0.058, 0.030, -0.020]
            apex = base + [side * 0.040, 0.048, -0.070]
            parts.append((f"ear{side:+.0f}", _cone_between(base, apex, 0.042, segments=6)))

    # -- the tail: a chain out to a cap on the back of the box, plus a fan for
    # the feathered group. The fan stops short of the cap so the chain keeps the
    # extreme; a fan that reached past it would move the box every round.
    tail_a, tail_b = at["tail"]
    tip_b = at["tail_tip"][1]
    cap = _extended(tail_b, tip_b, 2, -half + R_TAIL * 0.55)
    parts.append(("tail", _capsule_between(tail_a, tail_b, R_TAIL, segments=8)))
    parts.append(("tail_tip", _capsule_between(tail_b, cap, R_TAIL * 0.62, segments=8)))
    if group.tail == "fan":
        mid = (tail_b + cap) / 2.0
        parts.append((
            "fan",
            _ellipsoid_at(mid, (group.chord * 0.95, WING_THICK, float(cap[2] - tail_b[2]) / 2.2),
                          segments=12, rings=6),
        ))

    # -- the wings.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        chain = _wing_chain(group, at, side)
        for i in range(len(chain) - 1):
            radius = R_WING * (1.0 - 0.10 * i)
            parts.append((f"wing{side}{i}", _capsule_between(chain[i], chain[i + 1], radius,
                                                             segments=8)))
        inner, outer = chain[1], chain[4]
        centre = (inner + outer) / 2.0 + [0.0, 0.0, -group.chord * 0.72]
        parts.append((
            f"vane.{side}",
            _ellipsoid_at(
                centre,
                (float(abs(outer[0] - inner[0])) / 2.0 + 0.02, WING_THICK, group.chord),
                segments=14,
                rings=6,
            ),
        ))
        for i, (a, b, r) in enumerate(_spar_segments(group, at, side)):
            parts.append((f"spar{side}{i}", _capsule_between(a, b, r, segments=6)))
        del sign

    # -- the legs.
    for side in ("L", "R"):
        thigh_a, thigh_b = at[f"thigh.{side}"]
        shin_a, shin_b = at[f"shin.{side}"]
        foot_b = at[f"foot.{side}"][1]
        parts.append((f"thigh.{side}", _capsule_between(thigh_a, thigh_b, R_THIGH, segments=8)))
        parts.append((f"shin.{side}", _capsule_between(shin_a, shin_b, R_SHIN, segments=8)))
        parts.append((
            f"foot.{side}",
            _box_at([float(foot_b[0]), FOOT_SIZE[1] / 2.0, float(foot_b[2])], FOOT_SIZE),
        ))

    for i, eye in enumerate(_eye_centres(group, at)):
        parts.append((f"eye{i}", _sphere_at(eye, 0.019, segments=8, rings=6)))

    return parts


def _solid(parts: list[tuple[str, Any]]) -> Any:
    """Union every part into one closed solid, smooth it once, weld it."""
    from ...studio.clay import elements, ops_boolean, ops_subdiv, ops_topo
    from ...studio.clay import mesh as bm
    from ...studio.clay.document import Obj

    objs = [Obj(uid=i + 1, name=name, mesh=m) for i, (name, m) in enumerate(parts)]
    merged = ops_boolean.union(objs)
    merged, _ = ops_topo.weld(merged, elements.empty(), eps=SEAM_EPS)
    merged, _ = ops_subdiv.catmull_clark(merged, elements.empty(), levels=1)
    merged, _ = ops_topo.weld(merged, elements.empty(), eps=SMOOTH_EPS)
    bm.validate(merged)
    return merged


# --- regions -----------------------------------------------------------------


def _centroids(positions: np.ndarray, loops: np.ndarray, starts: np.ndarray) -> np.ndarray:
    counts = np.diff(starts.astype("i8"))
    face_of = np.repeat(np.arange(len(counts), dtype="i8"), counts)
    total = np.zeros((len(counts), 3), dtype="f8")
    np.add.at(total, face_of, positions[loops.astype("i8")])
    return total / counts[:, None]


def _hash_noise(points: np.ndarray) -> np.ndarray:
    """A deterministic per-point value in ``[0, 1)``. Not ``numpy.random``."""
    v = np.sin(points @ np.array([127.1, 311.7, 74.7])) * 43758.5453
    w = np.sin(points @ np.array([269.5, 183.3, 246.1])) * 21943.1719
    return np.modf(np.abs(v) + np.abs(w))[0]


def _near_segment(points: np.ndarray, a: Any, b: Any) -> np.ndarray:
    a = np.asarray(a, dtype="f8")
    b = np.asarray(b, dtype="f8")
    span = b - a
    denom = float(np.dot(span, span)) or 1.0
    t = np.clip(((points - a) @ span) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * span), axis=1)


def _regions(
    positions: np.ndarray,
    loops: np.ndarray,
    starts: np.ndarray,
    joints: list[dict[str, Any]],
    group: Group,
    region_names: tuple[str, ...],
) -> np.ndarray:
    """One region id per face, from geometry."""
    index = {name: i for i, name in enumerate(region_names)}
    at = _joint_points(joints)
    c = _centroids(positions, loops, starts)
    ids = np.full(len(c), index["hide"], dtype="i4")

    # Wing: everything outboard of the shoulder, and the tail's fan with it --
    # a fan is flight feathers and is painted as one.
    ids[np.abs(c[:, 0]) > at["wing_base.L"][1][0] * 0.92] = index["wing"]
    if group.tail == "fan":
        tail_b = at["tail"][1]
        tip_b = at["tail_tip"][1]
        ids[
            (_near_segment(c, tail_b, tip_b) > R_TAIL * 0.9)
            & (c[:, 2] < float(tail_b[2]))
        ] = index["wing"]

    # Underbelly: the front and underside of the body only. The radial test is
    # what keeps it off the legs.
    body = _near_segment(c, at["hips"][0], at["chest"][1])
    ids[(body < R_BODY * 1.12) & (c[:, 2] > 0.02) & (c[:, 1] < 0.78)] = index["underbelly"]

    # Keratin: the beak or snout tip, the talons, the horns.
    skull = _skull_centre(group, at)
    ids[_near_segment(c, skull + [0.0, -0.010, 0.030], _beak_tip(group, at))
        < R_BEAK * 1.5] = index["beak"]
    ids[c[:, 1] < FOOT_SIZE[1] * 0.95] = index["horn"]
    for a, b, r in _horn_segments(group, at):
        ids[_near_segment(c, a, b) < r * 2.4] = index["horn"]

    hide = ids == index["hide"]
    noise = _hash_noise(c * 37.0)
    ids[hide & (c[:, 1] > 0.30) & (noise > 0.86)] = index["accent"]

    for eye in _eye_centres(group, at):
        ids[np.linalg.norm(c - eye, axis=1) < 0.024] = index["eye"]
    return ids


# --- the appearance channels -------------------------------------------------


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _along(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    span = b - a
    denom = float(np.dot(span, span)) or 1.0
    return _smoothstep(((points - a) @ span) / denom)


def _within(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """A smooth 1 near the segment a->b, falling to 0 by *radius*.

    The companion :func:`_along` needs: a projection alone is blind to how far
    off the line a point is, and without this the tail's channel reaches down
    the legs -- measured on the quadruped, where it pulled two ankle joints out
    through the surface at ``tail_length = -1``.
    """
    return _smoothstep((radius - _near_segment(points, a, b)) / (radius * 0.6))


def _nearest_bone(points: np.ndarray, joints: list[dict[str, Any]]):
    heads = _to_gltf(np.array([b["head"] for b in joints], dtype="f8"))
    tails = _to_gltf(np.array([b["tail"] for b in joints], dtype="f8"))
    span = tails - heads
    denom = np.einsum("ij,ij->i", span, span)
    denom[denom == 0] = 1.0
    rel = points[:, None, :] - heads[None, :, :]
    t = np.clip(np.einsum("pbj,bj->pb", rel, span) / denom[None, :], 0.0, 1.0)
    closest = heads[None, :, :] + t[:, :, None] * span[None, :, :]
    offset = points[:, None, :] - closest
    dist = np.linalg.norm(offset, axis=2)
    which = np.argmin(dist, axis=1)
    rows = np.arange(len(points))
    return which, offset[rows, which]


def _fields(
    points: np.ndarray, joints: list[dict[str, Any]], group: Group
) -> dict[str, np.ndarray]:
    """Every channel's displacement, per unit of channel, at *points*."""
    at = _joint_points(joints)
    names = [b["name"] for b in joints]
    which, perp = _nearest_bone(points, joints)
    torso = {i for i, n in enumerate(names) if n in ("hips", "spine", "chest")}
    on_torso = np.isin(which, list(torso)).astype("f8")
    y = points[:, 1]
    x = points[:, 0]

    out: dict[str, np.ndarray] = {}

    # Bulk: away from the bone, and mostly the body. Weighted rather than global
    # because a wing vane is seventeen thousandths of a body height thick and a
    # bulk that inflated it would make a slider that turns a raven into a manta.
    out["bulk"] = (0.10 + 0.26 * on_torso)[:, None] * perp

    # Wing span: a scale in x that starts outboard of the shoulder, so the body
    # never moves and the two wings can never cross the midline. Proportional to
    # x for the humanoid's ``shoulder_width`` reason -- a constant push would
    # move a wing joint further than its own distance from the centre.
    shoulder = float(at["wing_base.L"][1][0])
    reach = _smoothstep((np.abs(x) - shoulder * 0.35) / (shoulder * 0.9))
    out["wingspan"] = np.stack(
        [0.42 * reach * x, np.zeros_like(y), np.zeros_like(y)], axis=1
    )

    neck_a = at["neck"][0]
    head_b = at["head"][1]
    axis = head_b - neck_a
    near_head = _within(points, neck_a, head_b, float(np.linalg.norm(axis)) * 2.2)
    t = _along(points, neck_a, head_b) * near_head
    out["neck_length"] = 0.42 * t[:, None] * axis[None, :]

    skull = _skull_centre(group, at)
    head_mask = _along(points, neck_a + 0.55 * axis, head_b) * near_head
    out["head_size"] = 0.30 * head_mask[:, None] * (points - skull[None, :])

    tail_a, tail_b = at["tail"][0], at["tail_tip"][1]
    tail_mask = _along(points, tail_a, tail_b) * _within(
        points, tail_a, tail_b, float(np.linalg.norm(tail_b - tail_a)) * 0.75
    )
    out["tail_length"] = 0.50 * tail_mask[:, None] * (points - tail_a[None, :])

    hip_y = float(at["hips"][0][1])
    lift = np.clip(y / max(hip_y, 1e-6), 0.0, 1.0)
    out["leg_length"] = np.stack([np.zeros_like(y), 0.22 * lift, np.zeros_like(y)], axis=1)
    return out


# --- the bake ----------------------------------------------------------------


def build(silhouette: str) -> Baked:
    """Generate one silhouette group from scratch. Deterministic, no I/O."""
    from ...studio.clay import adjacency
    from ...studio.clay import mesh as bm

    try:
        group = GROUPS[silhouette]
    except KeyError:
        raise CharacterError(
            f"{silhouette!r} is not a winged silhouette; try " + ", ".join(sorted(GROUPS)),
            field="family",
        ) from None

    half_span, depth = group.half_span, group.depth
    solid: Any = None
    for _ in range(8):
        joints = _fit_joints(half_span, depth)
        solid = _solid(_parts(joints, group))
        lo, hi = bm.bounds(solid)
        measured_span = float(max(-lo[0], hi[0]))
        measured_depth = float(hi[2] - lo[2])
        if (
            abs(measured_span - half_span) < FIT_TOLERANCE
            and abs(measured_depth - depth) < FIT_TOLERANCE
        ):
            break
        half_span, depth = measured_span, measured_depth
    else:  # pragma: no cover - a contraction that stops contracting is a bug
        raise CharacterError(f"{silhouette}: the landmark fit did not settle")

    assert solid is not None
    report = adjacency.check_manifold(solid)
    if not report.clean:
        raise CharacterError(
            f"{silhouette}: the union is not a closed solid "
            f"({len(report.boundary_edges)} boundary, "
            f"{len(report.nonmanifold_edges)} non-manifold edges)"
        )
    if bm.face_count(solid) > MAX_FACES:
        raise CharacterError(
            f"{silhouette}: {bm.face_count(solid):,} faces, past the {MAX_FACES:,} budget"
        )

    lo, hi = bm.bounds(solid)
    scale = 1.0 / float(hi[1] - lo[1])
    centre = np.array([(lo[0] + hi[0]) / 2.0, float(lo[1]), (lo[2] + hi[2]) / 2.0])
    positions = (np.asarray(solid.positions, dtype="f8") - centre) * scale

    joints = _fit_joints(
        float(np.abs(positions[:, 0]).max()),
        float(positions[:, 2].max() - positions[:, 2].min()),
    )

    region_names = familylib.get_archetype("winged").regions
    regions = _regions(
        positions, np.asarray(solid.loops), np.asarray(solid.starts),
        joints, group, region_names,
    )

    fields = _fields(positions, joints, group)
    joint_points = np.stack(
        [_to_gltf(np.array([b["head"] for b in joints])),
         _to_gltf(np.array([b["tail"] for b in joints]))],
        axis=1,
    )
    flat = joint_points.reshape(-1, 3)
    joint_fields = {
        k: v.reshape(len(joints), 2, 3) for k, v in _fields(flat, joints, group).items()
    }

    return Baked(
        silhouette=silhouette,
        positions=positions,
        loops=np.asarray(solid.loops, dtype="i4"),
        starts=np.asarray(solid.starts, dtype="i4"),
        regions=regions,
        joints=joints,
        displacements={k: v.astype("f4") for k, v in fields.items()},
        joint_displacements={k: v.astype("f4") for k, v in joint_fields.items()},
        bounds=(positions.min(axis=0), positions.max(axis=0)),
    )


def primitives_of(baked: Baked) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """``(region id, positions, indices)`` per region, in region-id order."""
    from ...studio.clay import mesh as bm
    from ...studio.clay import topo

    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    source = bm.Mesh(
        positions=baked.positions.astype("f4"),
        loops=baked.loops,
        starts=baked.starts,
        material=baked.regions.astype("i4"),
        smooth=np.ones(baked.face_count, dtype=bool),
        uv=None,
    )
    for region in range(int(baked.regions.max()) + 1):
        faces = np.flatnonzero(baked.regions == region).astype("i8")
        if not len(faces):
            continue
        part = topo.take_faces(source, faces)
        part, _ = topo.compact_vertices(part)
        tris, _ = bm.triangulate(part)
        out.append((region, np.asarray(part.positions, dtype="f4"), tris.astype("u4").reshape(-1)))
    return out


def _smooth_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    tri = indices.reshape(-1, 3).astype("i8")
    a, b, c = positions[tri[:, 0]], positions[tri[:, 1]], positions[tri[:, 2]]
    face = np.cross(b - a, c - a)
    out = np.zeros_like(positions, dtype="f8")
    for k in range(3):
        np.add.at(out, tri[:, k], face)
    norm = np.linalg.norm(out, axis=1)
    norm[norm == 0] = 1.0
    return (out / norm[:, None]).astype("f4")


def bake(baked: Baked) -> tuple[bytes, dict[str, np.ndarray]]:
    """``(glb bytes, npz arrays)`` for one silhouette group. Pure."""
    from ...studio.viewer import glbwrite, gltf

    parts = primitives_of(baked)
    prims: list[gltf.Primitive] = []
    offsets = [0]
    region_ids: list[int] = []
    for region, positions, indices in parts:
        prims.append(
            gltf.Primitive(
                positions=positions,
                indices=indices,
                normals=_smooth_normals(positions.astype("f8"), indices),
                material=gltf.Material(name=f"region:{region}", metallic_factor=0.0),
            )
        )
        offsets.append(offsets[-1] + len(positions))
        region_ids.append(region)

    node = gltf.Node(name="character", mesh=0)
    model = gltf.Model(nodes=[node], roots=[0], meshes=[prims], skins=[])
    data = glbwrite.write_glb(model)

    stacked = np.concatenate([p for _r, p, _i in parts]).astype("f4")
    remap = _remap_to_primitives(baked, parts)
    arrays: dict[str, np.ndarray] = {
        "positions_digest": np.frombuffer(
            hashlib.blake2b(stacked.tobytes(), digest_size=16).digest(), dtype="u1"
        ),
        "prim_offsets": np.array(offsets, dtype="i4"),
        "prim_regions": np.array(region_ids, dtype="i4"),
        "joint_names": np.array([b["name"] for b in baked.joints]),
        "joint_parents": np.array([b["parent"] or "" for b in baked.joints]),
        "joints": np.array([[b["head"], b["tail"]] for b in baked.joints], dtype="f4"),
    }
    for key, value in baked.displacements.items():
        arrays[f"disp/{key}"] = value[remap].astype("f4")
    for key, value in baked.joint_displacements.items():
        arrays[f"jdisp/{key}"] = value.astype("f4")
    return data, arrays


def _remap_to_primitives(
    baked: Baked, parts: list[tuple[int, np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Source vertex index for every vertex of the concatenated primitives."""
    source = baked.positions.astype("f4")
    lookup: dict[bytes, int] = {}
    for i, row in enumerate(source):
        lookup.setdefault(row.tobytes(), i)
    out: list[int] = []
    for _region, positions, _indices in parts:
        for row in positions:
            out.append(lookup[row.tobytes()])
    return np.array(out, dtype="i8")


def write_assets(silhouette: str, directory: Any) -> dict[str, int]:
    """Bake one silhouette and stage both files onto their served names."""
    import os
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    baked = build(silhouette)
    data, arrays = bake(baked)

    glb = directory / f"{silhouette}.glb"
    npz = directory / f"{silhouette}.masks.npz"
    tmp_glb = glb.with_suffix(".glb.tmp")
    tmp_npz = npz.with_suffix(".npz.tmp")
    tmp_glb.write_bytes(data)
    with tmp_npz.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp_glb, glb)
    os.replace(tmp_npz, npz)
    return {
        "faces": baked.face_count,
        "vertices": len(baked.positions),
        "glb_bytes": glb.stat().st_size,
        "npz_bytes": npz.stat().st_size,
    }
