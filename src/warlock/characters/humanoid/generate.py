"""One parameterised humanoid generator, baked per silhouette group.

**Why a generator at all, and why its output is checked in.** The shape has to
exist before a prompt does: the reference-image path can take a minute and a
card, and "make me an ogre" has to be able to answer in the time it takes to
load a file. So the body is built from Clay primitives, unioned into one closed
solid, smoothed once and written out -- and the *result* is what ships, because
``manifold3d`` does not promise byte-stable output across versions and Blender
has to import a file either way. ``scripts/author_humanoid.py --write``
regenerates them; ``tests/characters/test_humanoid.py`` re-runs this module and
compares against what is checked in, so a kernel that changes its mind is a red
test rather than a silently different ogre.

**One mesh per silhouette group, not per species.** Twelve species share four
baked meshes. What separates two species inside a group is a set of smooth
displacement fields -- bulk, stature, head size, limb length, shoulder width,
hunch -- evaluated once here and stored per vertex, so instantiating a dwarf is
an add and a scale rather than a rebuild. The groups exist only where a species
differs *topologically*: ``plain`` is round-eared and tuskless, ``pointed`` grows
long ears, ``tusked`` grows tusks and a brow, ``saurian`` grows a snout and drops
the ears. No displacement field can grow a tusk, which is the whole test for
whether something is a channel or a group.

**The skeleton is not negotiable.** The mesh is built *around* the shipped
``humanoid`` template's landmarks, fitted to the mesh's own bounding box -- which
is circular, so :func:`_fit_joints` iterates it to a fixed point (the bbox width
is set by the hands, the hands are placed by the bbox width, and the map is a
contraction). The pay-off is that every fitted joint lands inside solid geometry
by construction, which is what ``test_every_species_fits_its_template`` measures
and what keeps Blender's bone-heat solve out of the envelope fallback.

**Materials, not textures.** Faces carry a region id -- an index into
``family.Archetype.regions`` -- assigned *after* the boolean and the subdivision,
from geometry, because a union returns a triangle soup with the target's first
material on every face and UVs gone. A theme then paints the regions at
instantiation. No UV, no PNG, no image decode in the Blender worker.

Clay is imported **inside the functions that use it**. This module is reachable
from ``warlock.characters`` and that package is imported by the door; dragging
``trimesh`` and ``manifold3d`` in behind a registry lookup would cost every
cold start the boolean kernel's import time for nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import family as familylib
from ..errors import CharacterError

__all__ = [
    "FEATURES",
    "Baked",
    "Features",
    "bake",
    "build",
    "write_assets",
]

# --- the neutral figure ------------------------------------------------------
#
# Every number is a fraction of total height, glTF axes (Y up, +X the subject's
# left, +Z forward), feet on y = 0 and the crown on y = 1. Uniform scaling to a
# species' ``height_m`` is the last thing instantiation does, so the figure is
# authored once and a troll is a troll-sized copy of it rather than its own bake.

#: Radii, as height fractions.
R_HAND = 0.048
R_FOREARM = 0.040
R_UPPER_ARM = 0.050
R_SHOULDER = 0.062
R_CHEST = 0.100
R_WAIST = 0.088
R_PELVIS = 0.086
R_BELLY = 0.093
R_NECK = 0.042
R_HEAD = 0.077
R_THIGH = 0.062
R_SHIN = 0.045

#: Torso segments, as (y_low, y_high) height fractions.
CHEST_SPAN = (0.640, 0.770)
WAIST_SPAN = (0.550, 0.660)

#: Offsets along +Z (forward) from the mesh's own depth centre.
Z_BELLY = 0.028
Z_CHEST = -0.006
Z_HEAD = 0.004
Z_FOOT = 0.030

FOOT_SIZE = (0.090, 0.058, 0.155)  # x, y, z as height fractions
SNOUT_LENGTH = 0.075

#: Segment counts. Low on purpose: one Catmull-Clark level turns every triangle
#: of the boolean's output into three quads, so the face budget is spent before
#: the smoothing rather than after it. See the authoring script's report.
SEG = 12
RINGS = 3
HEAD_SEG = 14
HEAD_RINGS = 7

#: How many faces a baked silhouette may have. Not a hard engine limit -- it is
#: the budget that keeps four assets under a megabyte and keeps Blender's
#: bone-heat solve fast enough to run inside a job.
MAX_FACES = 12000

#: How close the landmark fit has to close, as a fraction of total height.
#: 1e-5 of a three-metre troll is thirty microns; the loop is chasing a fixed
#: point, not a measurement, and a tighter bar just buys more unions.
FIT_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class Features:
    """The topological differences between silhouette groups."""

    ears: str = "round"  # "round" | "pointed" | "none"
    tusks: bool = False
    brow: bool = False
    snout: bool = False


#: ``silhouette -> Features``. The keys are exactly ``family.silhouettes`` for
#: the humanoid archetype, which ``test_every_silhouette_has_features`` pins.
FEATURES: dict[str, Features] = {
    "plain": Features(ears="round"),
    "pointed": Features(ears="pointed"),
    "tusked": Features(ears="round", tusks=True, brow=True),
    "saurian": Features(ears="none", snout=True),
}


@dataclass
class Baked:
    """One silhouette group's mesh, its rig, its regions and its channels."""

    silhouette: str
    positions: np.ndarray  # (V, 3) f8, glTF axes, unit height, grounded
    loops: np.ndarray
    starts: np.ndarray
    regions: np.ndarray  # (F,) i4 -- an index into the archetype's regions
    joints: list[dict[str, Any]]  # rigging-shaped, Blender axes, unit height
    displacements: dict[str, np.ndarray]  # channel -> (V, 3) f4
    joint_displacements: dict[str, np.ndarray]  # channel -> (J, 2, 3) f4
    bounds: tuple[np.ndarray, np.ndarray] = field(default=None)  # type: ignore[assignment]

    @property
    def face_count(self) -> int:
        return len(self.starts) - 1


# --- axes --------------------------------------------------------------------
#
# Two conventions meet here and the conversion is one line, so it is written
# once and never inlined: the rig, the clips and Blender speak (+X left, -Y
# forward, +Z up); Clay, glTF and everything downstream speak (+X left, +Y up,
# +Z forward).


def _to_gltf(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _to_blender(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


# --- the skeleton the mesh is grown around -----------------------------------


def _fit_joints(half_width: float, z_lo: float, z_hi: float) -> list[dict[str, Any]]:
    """The shipped humanoid landmarks, fitted to a candidate bounding box."""
    from ... import rigging

    template = rigging.get_template("humanoid")
    # Blender's y is glTF's -z, so the box's y range is the negated z range.
    lo = [-half_width, -z_hi, 0.0]
    hi = [half_width, -z_lo, 1.0]
    return rigging.fit_template(template, lo, hi)


def _joint_points(joints: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``bone -> (head, tail)`` in glTF axes."""
    return {b["name"]: (_to_gltf(b["head"]), _to_gltf(b["tail"])) for b in joints}


# --- primitives placed in the world ------------------------------------------


def _basis_from_y(direction: np.ndarray) -> np.ndarray:
    """A rotation taking +Y onto *direction*, by the shortest arc.

    The same construction Blender's roll-0 bone matrix uses, for the same
    reason: it is the one rotation with no arbitrary twist in it, so a capsule
    laid along a bone has no seam that moves when the bone does.
    """
    d = np.asarray(direction, dtype="f8")
    d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = float(np.dot(y, d))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _placed(mesh: Any, rotation: np.ndarray, centre: np.ndarray) -> Any:
    from ...studio.clay import mesh as bm

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = centre
    return bm.transformed(mesh, matrix)


def _capsule_between(a: np.ndarray, b: np.ndarray, radius: float, *, segments: int = SEG) -> Any:
    """A capsule whose cylindrical section runs exactly from *a* to *b*."""
    from ...studio.clay import primitives

    span = np.asarray(b, dtype="f8") - np.asarray(a, dtype="f8")
    length = float(np.linalg.norm(span))
    if length < 1e-9:
        return _sphere_at(a, radius)
    body = primitives.capsule(radius=radius, height=length, segments=segments, rings=RINGS)
    return _placed(body, _basis_from_y(span), (np.asarray(a, dtype="f8") + b) / 2.0)


def _sphere_at(centre: Any, radius: float, *, segments: int = SEG, rings: int = RINGS * 2) -> Any:
    from ...studio.clay import primitives

    body = primitives.uv_sphere(radius=radius, segments=segments, rings=rings)
    return _placed(body, np.eye(3), np.asarray(centre, dtype="f8"))


def _cone_between(a: np.ndarray, b: np.ndarray, radius: float, *, segments: int = 8) -> Any:
    """A cone with its base at *a* and its apex at *b*."""
    from ...studio.clay import primitives

    span = np.asarray(b, dtype="f8") - np.asarray(a, dtype="f8")
    length = float(np.linalg.norm(span))
    body = primitives.cone(radius=radius, height=length, segments=segments)
    return _placed(body, _basis_from_y(span), (np.asarray(a, dtype="f8") + b) / 2.0)


def _box_at(centre: Any, size: Any) -> Any:
    from ...studio.clay import primitives

    return _placed(primitives.box(size=size), np.eye(3), np.asarray(centre, dtype="f8"))


# --- the body ----------------------------------------------------------------


def _parts(joints: list[dict[str, Any]], features: Features) -> list[tuple[str, Any]]:
    """Every closed solid the union is built from, named for its refusal text.

    Depth offsets are **absolute**, never measured off the bounding box's own
    centre. Placing a part at ``centre_z + offset`` was the first attempt and it
    diverged: the box's centre is computed *from* the parts, so every round
    walked the whole figure another few millimetres forward and the landmark fit
    never closed. Only the fitted joints carry the box's depth, and they carry it
    into limbs whose radius is an order of magnitude larger, which is what makes
    the remaining coupling a contraction.
    """
    at = _joint_points(joints)
    parts: list[tuple[str, Any]] = []

    hip_l, _ = at["thigh.L"]
    hip_r, _ = at["thigh.R"]
    parts.append(("pelvis", _capsule_between(hip_l, hip_r, R_PELVIS)))

    axis = np.zeros(3)
    parts.append(
        ("waist", _capsule_between(
            axis + [0.0, WAIST_SPAN[0], 0.0], axis + [0.0, WAIST_SPAN[1], 0.0], R_WAIST))
    )
    parts.append(
        ("chest", _capsule_between(
            axis + [0.0, CHEST_SPAN[0], Z_CHEST], axis + [0.0, CHEST_SPAN[1], Z_CHEST], R_CHEST))
    )
    parts.append(("belly", _sphere_at([0.0, 0.585, Z_BELLY], R_BELLY)))

    neck_head, neck_tail = at["neck"]
    parts.append(("neck", _capsule_between(neck_head, neck_tail, R_NECK)))

    # The crown *is* the top of the bounding box: the template's head tail sits
    # at z = 1.0 normalized, so a head whose pole were anywhere else would put
    # that landmark outside the mesh no matter how the box was measured.
    head_centre = np.array([0.0, 1.0 - R_HEAD, Z_HEAD])
    parts.append(("head", _sphere_at(head_centre, R_HEAD, segments=HEAD_SEG, rings=HEAD_RINGS)))
    parts.append(("jaw", _sphere_at(head_centre + [0.0, -0.030, 0.026], R_HEAD * 0.72)))

    if features.brow:
        parts.append(("brow", _sphere_at(head_centre + [0.0, 0.028, 0.030], R_HEAD * 0.62)))
    if features.snout:
        snout_base = head_centre + [0.0, -0.020, R_HEAD * 0.4]
        parts.append((
            "snout",
            _capsule_between(snout_base, snout_base + [0.0, -0.012, SNOUT_LENGTH], R_HEAD * 0.42),
        ))
    if features.ears == "round":
        for side in (1.0, -1.0):
            parts.append((
                f"ear{side:+.0f}",
                _sphere_at(head_centre + [side * R_HEAD * 0.92, 0.006, -0.004], R_HEAD * 0.30),
            ))
    elif features.ears == "pointed":
        for side in (1.0, -1.0):
            base = head_centre + [side * R_HEAD * 0.78, 0.004, -0.006]
            tip = base + [side * R_HEAD * 0.85, R_HEAD * 0.95, -R_HEAD * 0.30]
            parts.append((f"ear{side:+.0f}", _cone_between(base, tip, R_HEAD * 0.30)))
    if features.tusks:
        for side in (1.0, -1.0):
            base = head_centre + [side * 0.026, -0.052, 0.040]
            tip = base + [side * 0.008, 0.062, 0.016]
            parts.append((f"tusk{side:+.0f}", _cone_between(base, tip, 0.013, segments=6)))

    for side in ("L", "R"):
        shoulder_head, shoulder_tail = at[f"shoulder.{side}"]
        upper_head, upper_tail = at[f"upper_arm.{side}"]
        fore_head, fore_tail = at[f"forearm.{side}"]
        hand_head, hand_tail = at[f"hand.{side}"]
        parts.append((f"shoulder.{side}", _sphere_at(shoulder_tail, R_SHOULDER)))
        parts.append((f"upper_arm.{side}", _capsule_between(upper_head, upper_tail, R_UPPER_ARM)))
        parts.append((f"forearm.{side}", _capsule_between(fore_head, fore_tail, R_FOREARM)))
        # Centred on the *tail* -- the fingertip landmark -- because that point
        # is what sets the bounding box's width and therefore where every other
        # landmark lands. Centring on the wrist would leave the fingertip on the
        # hull and the fit would never close.
        parts.append((f"hand.{side}", _sphere_at(hand_tail, R_HAND)))
        del shoulder_head, hand_head

        thigh_head, thigh_tail = at[f"thigh.{side}"]
        shin_head, shin_tail = at[f"shin.{side}"]
        parts.append((f"thigh.{side}", _capsule_between(thigh_head, thigh_tail, R_THIGH)))
        parts.append((f"shin.{side}", _capsule_between(shin_head, shin_tail, R_SHIN)))
        # A box, not a capsule: the sole has to be flat because grounding puts
        # y = 0 on it and a sphere touching the floor at one point makes a
        # character that visibly rocks as the sheet's yaw turns.
        foot_x = float(shin_tail[0])
        length = FOOT_SIZE[2] * (1.25 if features.snout else 1.0)
        parts.append((
            f"foot.{side}",
            _box_at([foot_x, FOOT_SIZE[1] / 2.0, Z_FOOT], (FOOT_SIZE[0], FOOT_SIZE[1], length)),
        ))

    for side in (1.0, -1.0):
        parts.append((
            f"eye{side:+.0f}",
            _sphere_at(
                head_centre + [side * 0.028, 0.012, R_HEAD * 0.86],
                0.014, segments=8, rings=6,
            ),
        ))

    return parts


def _solid(parts: list[tuple[str, Any]]) -> Any:
    """Union every part into one closed solid, smooth it once, weld it.

    The order is deliberate. The union first, because a boolean over already
    subdivided inputs is the same shape for four times the kernel time. The weld
    before the smoothing, because ``manifold3d`` returns coincident vertices at
    the seams it cut and Catmull-Clark's crease rule reads a doubled vertex as a
    boundary -- which is how a smoothed union grows a visible crack along every
    join. Then one weld after, for the handful the smoothing recreates.
    """
    from ...studio.clay import elements, ops_boolean, ops_subdiv, ops_topo
    from ...studio.clay import mesh as bm
    from ...studio.clay.document import Obj

    objs = [Obj(uid=i + 1, name=name, mesh=m) for i, (name, m) in enumerate(parts)]
    merged = ops_boolean.union(objs)
    merged, _ = ops_topo.weld(merged, elements.empty(), eps=1e-5)
    merged, _ = ops_subdiv.catmull_clark(merged, elements.empty(), levels=1)
    merged, _ = ops_topo.weld(merged, elements.empty(), eps=1e-5)
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
    """A deterministic per-point value in ``[0, 1)``.

    Not ``numpy.random``: the corpus this feeds is compared against a checked-in
    asset to 1e-5, so the noise has to be a pure function of the position and
    survive a different numpy, a different seed policy and a different machine.
    """
    v = np.sin(points @ np.array([127.1, 311.7, 74.7])) * 43758.5453
    w = np.sin(points @ np.array([269.5, 183.3, 246.1])) * 21943.1719
    return np.modf(np.abs(v) + np.abs(w))[0]


def _regions(
    positions: np.ndarray,
    loops: np.ndarray,
    starts: np.ndarray,
    joints: list[dict[str, Any]],
    features: Features,
    cz: float,
    region_names: tuple[str, ...],
) -> np.ndarray:
    """One region id per face, from geometry.

    Assigned *after* the boolean and the subdivision because a union keeps only
    the target's first face material and drops UVs entirely -- so there is
    nothing to carry through, and a rule stated over the finished surface is the
    only kind that stays true when a segment count changes.
    """
    index = {name: i for i, name in enumerate(region_names)}
    c = _centroids(positions, loops, starts)
    ids = np.full(len(c), index["skin"], dtype="i4")

    head_centre = np.array([0.0, 1.0 - R_HEAD, Z_HEAD])

    # Garment: a band about the hips, everywhere except the head and hands.
    band = (c[:, 1] > 0.455) & (c[:, 1] < 0.585)
    ids[band] = index["garment"]

    # Belly: the front of the paunch, and only the front -- the back of that
    # sphere is inside the torso and would paint the spine.
    belly_centre = np.array([0.0, 0.585, Z_BELLY])
    to_belly = np.linalg.norm(c - belly_centre, axis=1)
    ids[(to_belly < R_BELLY * 1.06) & (c[:, 2] > 0.02)] = index["belly"]

    # The accent region -- the lava crack a fire theme lights, the rune a cursed
    # one does. A noise threshold over the skin only, so it never eats an eye.
    skin = ids == index["skin"]
    noise = _hash_noise(c * 37.0)
    torso = (c[:, 1] > 0.30) & (c[:, 1] < 0.86)
    ids[skin & torso & (noise > 0.86)] = index["accent"]

    if features.tusks:
        for side in (1.0, -1.0):
            base = head_centre + [side * 0.026, -0.052, 0.040]
            tip = base + [side * 0.008, 0.062, 0.016]
            ids[_near_segment(c, base, tip) < 0.020] = index["tooth"]

    # Eyes last: the smallest region wins every overlap, because a face painted
    # skin that should be an eye is invisible and the reverse is a stare.
    for side in (1.0, -1.0):
        eye = head_centre + [side * 0.028, 0.012, R_HEAD * 0.86]
        ids[np.linalg.norm(c - eye, axis=1) < 0.017] = index["eye"]

    del joints
    return ids


def _near_segment(points: np.ndarray, a: Any, b: Any) -> np.ndarray:
    a = np.asarray(a, dtype="f8")
    b = np.asarray(b, dtype="f8")
    span = b - a
    denom = float(np.dot(span, span)) or 1.0
    t = np.clip(((points - a) @ span) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * span), axis=1)


# --- the appearance channels -------------------------------------------------
#
# Every channel is a pure function of a point cloud, so the *same* function runs
# over the vertices and over the joints. That is not tidiness: a channel that
# moved the mesh and the skeleton by different rules would push a joint out
# through the surface at some value of the slider, and the only way to be sure
# it never does is for there to be one rule.


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bone_segments(joints: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    heads = _to_gltf(np.array([b["head"] for b in joints], dtype="f8"))
    tails = _to_gltf(np.array([b["tail"] for b in joints], dtype="f8"))
    return heads, tails, [b["name"] for b in joints]


def _nearest_bone(points: np.ndarray, joints: list[dict[str, Any]]):
    """``(index, perpendicular offset)`` of the closest bone to every point."""
    heads, tails, _names = _bone_segments(joints)
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
    points: np.ndarray, joints: list[dict[str, Any]], cz: float
) -> dict[str, np.ndarray]:
    """Every channel's displacement, per unit of channel, at *points*."""
    at = _joint_points(joints)
    hip_y = float(at["hips"][0][1])
    which, perp = _nearest_bone(points, joints)
    names = [b["name"] for b in joints]
    arm_bones = {
        i for i, n in enumerate(names)
        if n.split(".")[0] in ("shoulder", "upper_arm", "forearm", "hand")
    }
    is_arm = np.isin(which, list(arm_bones))
    y = points[:, 1]

    out: dict[str, np.ndarray] = {}

    # Bulk: push the surface away from the bone it belongs to. A joint sits on
    # its own bone, so its perpendicular offset is zero and bulk moves it not at
    # all -- which is the correct answer and falls out rather than being cased.
    out["bulk"] = 0.32 * perp

    # Stature: the legs stretch and everything above the hips rides up.
    lift = np.clip(y / max(hip_y, 1e-6), 0.0, 1.0)
    out["stature"] = np.stack([np.zeros_like(y), 0.22 * lift, np.zeros_like(y)], axis=1)

    # Head size: a scale about the base of the neck.
    neck_base = at["neck"][0]
    mask = _smoothstep((y - float(neck_base[1])) / max(1.0 - float(neck_base[1]), 1e-6))
    out["head_size"] = 0.34 * mask[:, None] * (points - neck_base[None, :])

    # Limb length: the arms grow away from the shoulder they hang off.
    grow = np.zeros_like(points)
    for side in ("L", "R"):
        shoulder = at[f"shoulder.{side}"][1]
        on_side = is_arm & (np.sign(points[:, 0]) == (1.0 if side == "L" else -1.0))
        grow[on_side] = 0.26 * (points[on_side] - shoulder[None, :])
    out["limb_length"] = grow

    # Shoulder width: the yoke widens and takes the arms with it. **Proportional
    # to x, not a constant push outward** -- a constant moved the shoulder joints
    # by more than their own distance from the midline at the low end of the
    # slider, so a narrow-shouldered species had its shoulder bones crossed over
    # each other and sitting outside the mesh. Scaling keeps the sign of every
    # point's x, which is the property that makes the field safe at both bounds.
    yoke = np.exp(-(((y - 0.78) / 0.13) ** 2))
    weight = np.maximum(yoke, is_arm.astype("f8"))
    out["shoulder_width"] = np.stack(
        [0.55 * weight * points[:, 0], np.zeros_like(y), np.zeros_like(y)], axis=1
    )

    # Hunch: everything above the hips rotates forward about the hip line.
    pivot = np.array([0.0, hip_y, cz])
    w = _smoothstep((y - hip_y) / max(1.0 - hip_y, 1e-6))
    theta = 0.36 * w
    rel = points - pivot
    turned = np.stack(
        [
            rel[:, 0],
            rel[:, 1] * np.cos(theta) - rel[:, 2] * np.sin(theta),
            rel[:, 1] * np.sin(theta) + rel[:, 2] * np.cos(theta),
        ],
        axis=1,
    )
    out["hunch"] = turned - rel
    return out


# --- the bake ----------------------------------------------------------------


def build(silhouette: str) -> Baked:
    """Generate one silhouette group from scratch. Deterministic, no I/O."""
    from ...studio.clay import adjacency
    from ...studio.clay import mesh as bm

    try:
        features = FEATURES[silhouette]
    except KeyError:
        raise CharacterError(
            f"{silhouette!r} is not a humanoid silhouette; try "
            + ", ".join(sorted(FEATURES)),
            field="family",
        ) from None

    # The bounding box sets the landmarks and the landmarks set the bounding
    # box: the hands are placed at 0.36 of the width and *are* the width, plus a
    # hand's radius. The map is affine and contracts by ~0.72 a round, so plain
    # iteration converges -- but at 0.72 a round it needs thirty-odd unions to
    # reach a tolerance worth having, and a union is half a second. A secant step
    # on ``f(w) - w`` lands on the fixed point of an affine map in one, so the
    # loop takes four rounds instead of thirty-five and the assertion below is
    # what makes that an optimisation rather than a guess.
    half_width, z_lo, z_hi = 0.18, -0.11, 0.12
    solid: Any = None
    history: list[tuple[float, float]] = []
    for _ in range(8):
        joints = _fit_joints(half_width, z_lo, z_hi)
        solid = _solid(_parts(joints, features))
        lo, hi = bm.bounds(solid)
        z_lo, z_hi = float(lo[2]), float(hi[2])
        measured = float(max(-lo[0], hi[0]))
        if abs(measured - half_width) < FIT_TOLERANCE:
            break
        history.append((half_width, measured))
        if len(history) >= 2:
            (x0, y0), (x1, y1) = history[-2], history[-1]
            denom = (y1 - x1) - (y0 - x0)
            if abs(denom) > 1e-12:
                half_width = x1 - (y1 - x1) * (x1 - x0) / denom
                continue
        half_width = measured
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

    # Grounded and unit-tall, in that order, so the recorded landmarks and the
    # positions are in one frame -- the artifact rule for source.glb, applied
    # before anything derived from it exists.
    lo, hi = bm.bounds(solid)
    scale = 1.0 / float(hi[1] - lo[1])
    centre = np.array([(lo[0] + hi[0]) / 2.0, float(lo[1]), (lo[2] + hi[2]) / 2.0])
    positions = (np.asarray(solid.positions, dtype="f8") - centre) * scale

    cz = 0.0
    half_width = float(np.abs(positions[:, 0]).max())
    z_lo, z_hi = float(positions[:, 2].min()), float(positions[:, 2].max())
    joints = _fit_joints(half_width, z_lo, z_hi)

    region_names = familylib.get_archetype("humanoid").regions
    regions = _regions(
        positions, np.asarray(solid.loops), np.asarray(solid.starts),
        joints, features, cz, region_names,
    )

    fields = _fields(positions, joints, cz)
    joint_points = np.stack(
        [_to_gltf(np.array([b["head"] for b in joints])),
         _to_gltf(np.array([b["tail"] for b in joints]))],
        axis=1,
    )
    flat = joint_points.reshape(-1, 3)
    joint_fields = {k: v.reshape(len(joints), 2, 3) for k, v in _fields(flat, joints, cz).items()}

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
    """``(region id, positions, indices)`` per region, in region-id order.

    One primitive per region rather than one mesh with a per-face attribute,
    because glTF has no per-face anything: a material is a primitive. The order
    is the region order, so the concatenated positions are addressable from the
    mask file by a single offsets array.
    """
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
    """``(glb bytes, npz arrays)`` for one silhouette group.

    Pure: the caller decides where the bytes go, which is what lets the test
    re-bake and compare without writing anything into the source tree.
    """
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
        "joints": np.array(
            [[b["head"], b["tail"]] for b in baked.joints], dtype="f4"
        ),
    }
    for key, value in baked.displacements.items():
        arrays[f"disp/{key}"] = value[remap].astype("f4")
    for key, value in baked.joint_displacements.items():
        arrays[f"jdisp/{key}"] = value.astype("f4")
    return data, arrays


def _remap_to_primitives(
    baked: Baked, parts: list[tuple[int, np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Source vertex index for every vertex of the concatenated primitives.

    ``topo.take_faces`` + ``compact_vertices`` renumber, and the displacement
    fields are indexed on the *source* mesh -- so the mapping has to be
    recovered before a channel can be stored against the file that ships. Done
    by position, which is exact here: the primitives are cut out of one mesh, so
    every primitive vertex is bit-identical to the source vertex it came from.
    """
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
