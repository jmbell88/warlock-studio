"""One parameterised quadruped generator, baked per silhouette group.

The humanoid generator's shape, applied to a body plan that runs along its own
depth rather than standing up in it. Everything structural is the same and is
the same for the same reasons: primitives to a boolean union, one weld, one
Catmull-Clark level, one weld; regions assigned from geometry *after* the union
because a union keeps one material and drops UVs; channels stored as per-vertex
displacement fields evaluated by the same function over the vertices and over
the joints, so a slider can never move the two apart. Read
``characters/humanoid/generate.py`` first -- this module is deliberately its
sibling and not its subclass, because the two share no line worth the import.

**Where the ``quadruped`` template forces the silhouette.** The shipped skeleton
puts ``head``'s tail at the normalized ``(0, -0.5, 1.0)`` -- the *front-top
corner* of the bounding box. That is not a stylistic choice this module could
make differently: ``rigging.fit_template`` is bbox-proportional, so whatever the
mesh is, that landmark lands on that corner, and the containment bar says the
landmark has to be in solid geometry. A sphere tangent to two perpendicular
planes misses their shared corner by ``r*(sqrt(2)-1)`` however it is placed, so
the only shape that reaches it is a *point*. Hence: every quadruped here carries
its muzzle raised, the nose tip is both the highest and the frontmost point of
the body, and the muzzle is a cone whose apex is exactly that landmark. The
visible consequence is a head carriage -- right for a horse or a deer, alert for
a wolf -- and the invisible one is that **antlers and ears must stay below the
nose**, because anything that outgrew it would become the top of the box and
leave the head landmark hanging in the air above the skull.

**Every extreme is an absolute constant, and the box is measured.** The
landmarks are fitted to the bounding box and the parts are placed on the
landmarks, so the fit is circular here as it is in the humanoid. It is broken
the same way -- iterate to a fixed point -- but with one rule the humanoid did
not need: *nothing that sets an extreme of the box may be placed off the box*.
The nose apex, the tail apex, the soles and the barrel's radius are all written
down in absolute units, never as a fraction of the current estimate. The reason
is measured. Catmull-Clark approximates rather than interpolates, so a ten-sided
capsule of radius r comes out about 0.977 r; a barrel whose radius was *set to*
the estimated half-width would therefore lose 2.3% every round and the loop
would walk the animal to nothing. With the radius absolute, the same shrinkage
is a constant the loop converges *onto* in a single step.

**One mesh per silhouette group, not per species.** Eight species share five
baked meshes. A group exists only where a species differs *topologically*:
``paw`` has soft feet and no horns, ``hoof`` adds hooves and a mane, ``antlered``
grows antlers, ``tusked`` grows tusks off the jaw, ``scaled`` drops the ears and
grows a dorsal ridge. Everything else -- and a wolf, a bear and a big cat are
all of it -- is channel defaults, height and a palette.

Clay is imported **inside the functions that use it**, for the humanoid
module's reason: this package is reachable from ``warlock.characters``, which is
imported by the door, and the boolean kernel has no business loading behind a
registry lookup.
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

# --- the neutral animal ------------------------------------------------------
#
# Every number is a fraction of total *height*, in glTF axes (Y up, +X the
# subject's left, +Z forward), soles on y = 0 and the nose tip at y = 1. The
# body's length is a group parameter and is always greater than 1: a quadruped
# that was taller than it is long would be a quadruped standing up.

R_NECK = 0.072
R_SKULL = 0.088
R_JAW = 0.056
R_MUZZLE = 0.058
R_RUMP = 0.128
R_UPPER_LEG = 0.046
R_LOWER_LEG = 0.038
R_TAIL = 0.046

#: How far a leg is pushed outboard of the bone it is grown around.
#:
#: The template places the legs at 0.12 of the box width, which on a body whose
#: width *is* the barrel puts the left and right columns 0.07 of a body height
#: apart -- close enough that two capsules of a plausible radius merge into one
#: pillar through the midline. The offset separates them while leaving every
#: leg joint comfortably inside its own capsule, which is the property the
#: containment test measures and the reason the offset is bounded by the radius
#: rather than chosen by eye.
LEG_OUTBOARD = 0.016

#: Segment counts. Low on purpose: one Catmull-Clark level turns every triangle
#: of the boolean's output into three quads, so the budget is spent before the
#: smoothing rather than after it.
SEG = 10
RINGS = 3
HEAD_SEG = 12
HEAD_RINGS = 6

#: The face budget, the humanoid's for the humanoid's reason: it is what keeps a
#: group's pair of files inside a megabyte and Blender's bone-heat solve fast
#: enough to run inside a job.
MAX_FACES = 12000

#: Weld tolerances, before and after the smoothing. See :func:`_solid`.
SEAM_EPS = 1e-5
SMOOTH_EPS = 1e-7

#: How close the landmark fit has to close, as a fraction of total height.
#: 1e-5 of a 1.7 m horse is seventeen microns; the loop is chasing a fixed
#: point, not a measurement, and a tighter bar just buys more unions.
FIT_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class Group:
    """One silhouette group: the box it fills and what it grows."""

    #: Nose-to-tail depth, as a multiple of height. Always > 1: an animal that
    #: was taller than it is long would be an animal standing up.
    length: float
    #: The barrel's radius. Nothing on the body is wider, so this is what the
    #: bounding box's half-width settles on -- *near* it rather than at it,
    #: because Catmull-Clark approximates and a ten-sided capsule loses a couple
    #: of per cent of its radius to the smoothing. Which is exactly why the
    #: radius is written down here and the half-width is measured: a radius set
    #: *to* the box's half-width would lose that couple of per cent again every
    #: round and the fit would walk to zero.
    barrel: float
    feet: str  # "paw" | "hoof" | "claw"
    ears: str  # "pointed" | "round" | "none"
    horns: str = "none"  # "none" | "antler" | "tusk"
    mane: bool = False
    dorsal: bool = False
    #: Height of the withers as a fraction of total height. Sets where the
    #: barrel hangs; a cat's back is lower than a horse's relative to its head.
    back: float = 0.72


#: ``silhouette -> Group``. The keys are exactly ``family.silhouettes`` for the
#: quadruped archetype, which ``test_every_silhouette_group_has_an_asset`` pins.
GROUPS: dict[str, Group] = {
    "paw": Group(length=1.36, barrel=0.146, feet="paw", ears="pointed"),
    "hoof": Group(length=1.30, barrel=0.140, feet="hoof", ears="pointed", mane=True),
    "antlered": Group(length=1.28, barrel=0.138, feet="hoof", ears="pointed", horns="antler"),
    "tusker": Group(
        length=1.34, barrel=0.166, feet="hoof", ears="round", horns="tusk", mane=True
    ),
    "scaled": Group(
        length=1.72, barrel=0.132, feet="claw", ears="none", dorsal=True, back=0.66
    ),
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


def _to_gltf(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def _to_blender(p: Any) -> np.ndarray:
    p = np.asarray(p, dtype="f8")
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


# --- the skeleton the mesh is grown around -----------------------------------


def _fit_joints(half_width: float, length: float) -> list[dict[str, Any]]:
    """The shipped quadruped landmarks, fitted to the group's box."""
    from ... import rigging

    template = rigging.get_template("quadruped")
    # Blender's y is glTF's -z, so a box centred on z = 0 has y from -L/2 to L/2.
    lo = [-half_width, -length / 2.0, 0.0]
    hi = [half_width, length / 2.0, 1.0]
    return rigging.fit_template(template, lo, hi)


def _joint_points(joints: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``bone -> (head, tail)`` in glTF axes."""
    return {b["name"]: (_to_gltf(b["head"]), _to_gltf(b["tail"])) for b in joints}


# --- primitives placed in the world ------------------------------------------


def _basis_from_y(direction: np.ndarray) -> np.ndarray:
    """A rotation taking +Y onto *direction*, by the shortest arc."""
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


def _capsule_between(a: Any, b: Any, radius: float, *, segments: int = SEG) -> Any:
    """A capsule whose cylindrical section runs exactly from *a* to *b*."""
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


def _cone_between(a: Any, b: Any, radius: float, *, segments: int = 8) -> Any:
    """A cone with its base at *a* and its apex at *b*."""
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


# --- named landmarks the mesh and the regions both read ----------------------
#
# Written once and read twice. A region rule that re-derived a tusk's line from
# its own copy of the numbers would be one edit away from painting the air
# beside it, which is the failure the humanoid module's ``_regions`` avoids the
# same way.


def _muzzle_apex(group: Group) -> np.ndarray:
    """The nose tip: the front-top corner of the box, and the head's tail."""
    return np.array([0.0, 1.0, group.length / 2.0])


def _skull_centre(at: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return at["head"][0]


def _antler_segments(group: Group, at) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """``(base, tip, radius)`` per antler beam and tine, both sides.

    Swept back and out rather than up, and the reason is the module docstring's:
    the nose is the top of the box by the template's own arithmetic, so an
    antler that rose over the skull would take the head landmark with it.
    """
    if group.horns != "antler":
        return []
    skull = _skull_centre(at)
    span = group.barrel
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for side in (1.0, -1.0):
        base = skull + [side * 0.034, 0.030, -0.014]
        beam = base + [side * 0.040, 0.012, -0.140]
        tip = beam + [side * (span * 0.22), 0.006, -0.086]
        out.append((base, beam, 0.016))
        out.append((beam, tip, 0.011))
        out.append((beam, beam + [side * 0.014, 0.018, 0.048], 0.010))
        out.append((base + [side * 0.004, 0.008, -0.026],
                    base + [side * 0.022, 0.026, 0.030], 0.009))
    return out


def _tusk_segments(group: Group, at) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """``(base, tip, radius)`` per tusk. Off the jaw, curving up and forward."""
    if group.horns != "tusk":
        return []
    jaw = _skull_centre(at) + [0.0, -0.052, 0.060]
    out = []
    for side in (1.0, -1.0):
        base = jaw + [side * 0.030, -0.006, 0.010]
        out.append((base, base + [side * 0.014, 0.075, 0.030], 0.012))
    return out


def _eye_centres(at) -> list[np.ndarray]:
    skull = _skull_centre(at)
    return [skull + [side * 0.052, 0.026, 0.048] for side in (1.0, -1.0)]


def _dorsal_segments(group: Group, at) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """A ridge of spines along the spine, for the scaled group."""
    if not group.dorsal:
        return []
    a = at["chest"][0]
    b = at["tail_01"][0]
    out = []
    for i in range(7):
        t = (i + 0.5) / 7.0
        root = a + t * (b - a)
        root = np.array([0.0, group.back + 0.10, float(root[2])])
        out.append((root, root + [0.0, 0.048, 0.0], 0.016))
    return out


# --- the body ----------------------------------------------------------------


def _parts(joints: list[dict[str, Any]], group: Group) -> list[tuple[str, Any]]:
    """Every closed solid the union is built from, named for its refusal text.

    The four parts that *define* the box are the ones to read first, because
    :func:`build` refuses the group if they stop defining it: the barrel's
    radius is the width, the muzzle cone's apex is the front and the top, the
    tail cone's apex is the back, and the feet's soles are the floor.
    """
    at = _joint_points(joints)
    half = group.length / 2.0
    parts: list[tuple[str, Any]] = []

    # -- the barrel. Its radius *is* half_width, which is what puts the widest
    # part of the animal on the box and keeps the fit a fixed point.
    hips = at["hips"][0]
    chest_tail = at["chest"][1]
    parts.append(("barrel", _capsule_between(hips, chest_tail, group.barrel)))
    parts.append(("rump", _sphere_at(hips + [0.0, -0.010, -0.030], R_RUMP)))
    parts.append(("chest", _sphere_at(at["chest"][0] + [0.0, -0.012, 0.020],
                                      group.barrel * 0.97)))

    # -- neck and head.
    neck_head, neck_tail = at["neck"]
    parts.append(("neck", _capsule_between(neck_head, neck_tail, R_NECK)))
    skull = _skull_centre(at)
    parts.append(("skull", _sphere_at(skull, R_SKULL, segments=HEAD_SEG, rings=HEAD_RINGS)))
    # Behind the nose, always. A jaw far enough forward becomes the frontmost
    # point of the box and the head landmark lands on *it* instead of the nose,
    # which is what happened to the short-bodied groups first time round: the
    # apex has to win both extremes or it wins neither.
    parts.append(("jaw", _sphere_at(skull + [0.0, -0.048, 0.030], R_JAW)))
    # The muzzle is a *cone*, and its apex is the head bone's tail. See the
    # module docstring: nothing rounded reaches the corner of its own box.
    parts.append(("muzzle", _cone_between(skull + [0.0, -0.006, 0.010],
                                          _muzzle_apex(group), R_MUZZLE, segments=10)))

    if group.ears == "pointed":
        for side in (1.0, -1.0):
            base = skull + [side * 0.058, 0.038, -0.028]
            tip = base + [side * 0.026, 0.052, -0.048]
            parts.append((f"ear{side:+.0f}", _cone_between(base, tip, 0.030, segments=6)))
    elif group.ears == "round":
        for side in (1.0, -1.0):
            parts.append((
                f"ear{side:+.0f}",
                _sphere_at(skull + [side * 0.070, 0.040, -0.030], 0.034, segments=8, rings=6),
            ))

    for i, (a, b, r) in enumerate(_antler_segments(group, at)):
        parts.append((f"antler{i}", _cone_between(a, b, r, segments=6)))
    for i, (a, b, r) in enumerate(_tusk_segments(group, at)):
        parts.append((f"tusk{i}", _cone_between(a, b, r, segments=6)))
    for i, (a, b, r) in enumerate(_dorsal_segments(group, at)):
        parts.append((f"dorsal{i}", _cone_between(a, b, r, segments=6)))

    if group.mane:
        crest_a = at["neck"][0] + [0.0, 0.036, 0.0]
        crest_b = at["neck"][1] + [0.0, 0.030, -0.020]
        parts.append(("mane", _capsule_between(crest_a, crest_b, 0.048, segments=8)))

    # -- the tail. Two segments because the template's tail bends downward, and
    # a cone at the end because the apex has to *be* the back of the box.
    # The tail's last segment is a *capsule*, and where its cap centre sits is
    # arithmetic rather than taste. The landmark ``tail_02``'s tail is the
    # rearmost point of the box, and the rearmost point of a capsule is its end
    # sphere's -z pole -- so the cap centre goes exactly one radius in front of
    # the box's back face, at the landmark's own height. A cone apex was the
    # first attempt and it failed the containment bar at every group: one
    # Catmull-Clark level rounds a cone's tip to almost nothing, and the nudge a
    # boundary landmark gets is bigger than what is left.
    t1_head, t1_tail = at["tail_01"]
    _t2_head, t2_tail = at["tail_02"]
    r_tip = R_TAIL * 0.72
    cap = np.array([0.0, float(t2_tail[1]), -half + r_tip])
    parts.append(("tail_01", _capsule_between(t1_head, t1_tail, R_TAIL, segments=8)))
    parts.append(("tail_02", _capsule_between(t1_tail, cap, r_tip, segments=8)))

    # -- the legs. Pushed outboard so the two columns do not merge through the
    # midline; the joints stay inside because the offset is bounded by the
    # radius, not chosen by eye.
    sole = _FOOT_SIZE[group.feet]
    for pair in ("front", "rear"):
        for side, sign in (("L", 1.0), ("R", -1.0)):
            out = np.array([sign * LEG_OUTBOARD, 0.0, 0.0])
            upper_head, upper_tail = at[f"{pair}_upper.{side}"]
            lower_head, lower_tail = at[f"{pair}_lower.{side}"]
            foot_head, foot_tail = at[f"{pair}_foot.{side}"]
            parts.append((
                f"{pair}_upper.{side}",
                _capsule_between(upper_head + out * 0.6, upper_tail + out, R_UPPER_LEG),
            ))
            parts.append((
                f"{pair}_lower.{side}",
                _capsule_between(lower_head + out, lower_tail + out * 0.9, R_LOWER_LEG,
                                 segments=8),
            ))
            # A box, and centred on the ankle landmark in x and z rather than
            # outboard with the leg. One Catmull-Clark level replaces a quad
            # face with its own centroid, so the sole's lowest point is exactly
            # the bottom face's centre and nowhere else -- put that centre
            # anywhere but on the landmark and the landmark is left hanging
            # under a rounded corner, which is how all four feet of three groups
            # failed containment before this line said what it says.
            centre = np.array([float(foot_tail[0]), sole[1] / 2.0, float(foot_tail[2])])
            parts.append((f"{pair}_foot.{side}", _box_at(centre, sole)))
            del foot_head

    for i, eye in enumerate(_eye_centres(at)):
        parts.append((f"eye{i}", _sphere_at(eye, 0.016, segments=8, rings=6)))

    return parts


#: ``feet -> (x, y, z)`` of the sole box.
_FOOT_SIZE: dict[str, tuple[float, float, float]] = {
    "paw": (0.062, 0.052, 0.110),
    "hoof": (0.050, 0.058, 0.076),
    "claw": (0.070, 0.044, 0.118),
}


def _solid(parts: list[tuple[str, Any]]) -> Any:
    """Union every part into one closed solid, smooth it once, weld it.

    The order is the humanoid module's and is load-bearing for its reason: the
    union first because a boolean over subdivided inputs is the same shape for
    four times the kernel time, a weld before the smoothing because
    ``manifold3d`` leaves coincident vertices at every seam it cuts and
    Catmull-Clark reads a doubled vertex as a boundary, and a weld after for
    the handful the smoothing recreates.

    **The second weld's tolerance is a hundredth of the first's, and the
    difference is measured.** At the humanoid's 1e-5 the paw group came out with
    ten non-manifold edges, all of them on the muzzle, where the skull sphere,
    the jaw sphere and the muzzle cone meet at a shallow angle and the union
    leaves distinct vertices a thousandth of a body height apart. Catmull-Clark
    recreated *no* duplicates there -- the second weld at 1e-5 was not tidying
    the smoothing's output, it was collapsing real geometry and pinching the
    surface. 1e-7 keeps the safety net and stops it catching the animal.
    """
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
    """A deterministic per-point value in ``[0, 1)``.

    Not ``numpy.random``: the checked-in asset is compared against a rebake to
    1e-5, so the noise has to be a pure function of the position and survive a
    different numpy, a different seed policy and a different machine.
    """
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
    """One region id per face, from geometry.

    Assigned *after* the boolean and the subdivision because a union keeps only
    the target's first face material and drops UVs entirely, so there is nothing
    to carry through and a rule stated over the finished surface is the only
    kind that stays true when a segment count changes.
    """
    index = {name: i for i, name in enumerate(region_names)}
    at = _joint_points(joints)
    c = _centroids(positions, loops, starts)
    ids = np.full(len(c), index["hide"], dtype="i4")

    # Underbelly: below the barrel's axis and inside its radius. The radial test
    # is what keeps it off the legs, which are below the same line and would
    # otherwise come out pale to the ankle.
    axis_a, axis_b = at["hips"][0], at["chest"][1]
    radial = _near_segment(c, axis_a, axis_b)
    ids[(c[:, 1] < group.back - 0.012) & (radial < group.barrel * 1.10)] = index["underbelly"]
    throat = _near_segment(c, at["neck"][0], at["neck"][1])
    ids[(throat < R_NECK * 1.15) & (c[:, 1] < _skull_centre(at)[1] - 0.02)] = index["underbelly"]

    # Keratin: hooves, paw pads, claws, antlers, tusks and the dorsal ridge --
    # one region because one material is what they all are.
    ids[c[:, 1] < _FOOT_SIZE[group.feet][1] * 0.92] = index["horn"]
    for a, b, r in (
        _antler_segments(group, at) + _tusk_segments(group, at) + _dorsal_segments(group, at)
    ):
        ids[_near_segment(c, a, b) < r * 2.4] = index["horn"]

    if group.mane:
        crest_a = at["neck"][0] + [0.0, 0.036, 0.0]
        crest_b = at["neck"][1] + [0.0, 0.030, -0.020]
        ids[_near_segment(c, crest_a, crest_b) < 0.062] = index["mane"]

    # The accent -- the stripe a tiger theme paints, the ember a hellhound one
    # lights. A noise threshold over the hide only, so it never eats an eye.
    hide = ids == index["hide"]
    noise = _hash_noise(c * 37.0)
    flank = (c[:, 1] > group.back - 0.14) & (c[:, 1] < group.back + group.barrel)
    ids[hide & flank & (noise > 0.86)] = index["accent"]

    # Eyes last: the smallest region wins every overlap, because a face painted
    # hide that should be an eye is invisible and the reverse is a stare.
    for eye in _eye_centres(at):
        ids[np.linalg.norm(c - eye, axis=1) < 0.020] = index["eye"]
    return ids


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


def _along(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smoothstep of the clamped projection of *points* onto the segment a->b.

    A smooth mask with no nearest-bone lookup in it. The humanoid module masks
    its limb channels by which bone is closest, which is a step function and
    tears the surface where two bones meet; here the head and the tail both hang
    off the end of a line, so the projection parameter *is* the mask and the
    field is continuous everywhere by construction.
    """
    span = b - a
    denom = float(np.dot(span, span)) or 1.0
    return _smoothstep(((points - a) @ span) / denom)


def _within(points: np.ndarray, a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """A smooth 1 near the segment a->b, falling to 0 by *radius*.

    The companion :func:`_along` needs, because a projection alone is blind to
    how far off the line a point is. The tail's mask is the measured case: a
    rear foot sits *forward* of the tail's root but still projects positively
    onto the tail's axis, so at ``tail_length = -1`` the shortening field
    reached down the hind legs and pulled two ankle joints out through the
    surface. Multiplying by distance-to-the-line is what confines a channel to
    the limb it names.
    """
    d = _near_segment(points, a, b)
    return _smoothstep((radius - d) / (radius * 0.6))


def _nearest_bone(points: np.ndarray, joints: list[dict[str, Any]]):
    """``(index, perpendicular offset)`` of the closest bone to every point."""
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

    out: dict[str, np.ndarray] = {}

    # Bulk: push the surface away from the bone it belongs to, and mostly the
    # barrel. Weighted rather than global because the legs are grown outboard of
    # their bones -- a bulk that shrank a leg by a third of its radius would
    # eventually pull the capsule inside the joint it was built around, which is
    # exactly what ``test_a_channel_at_either_bound_keeps_the_skeleton_in_the_body``
    # is there to catch and what this weighting prevents by construction.
    out["bulk"] = (0.16 + 0.24 * on_torso)[:, None] * perp

    # Leg length: the legs stretch and everything above the withers rides up. A
    # point on the sole has y = 0 and does not move, which is what keeps the
    # animal on the floor at both bounds.
    lift = np.clip(y / max(group.back, 1e-6), 0.0, 1.0)
    out["leg_length"] = np.stack([np.zeros_like(y), 0.20 * lift, np.zeros_like(y)], axis=1)

    # Neck length and head size: one mask, two uses. ``t`` is how far along the
    # neck a point is, extended past the head, so the barrel is 0 and the muzzle
    # is 1 with everything between smooth.
    neck_a, neck_b = at["neck"]
    axis = neck_b - neck_a
    reach = float(np.linalg.norm(axis)) * 1.8
    near_neck = _within(points, neck_a, _muzzle_apex(group), reach)
    t = _along(points, neck_a, neck_b) * near_neck
    out["neck_length"] = 0.55 * t[:, None] * axis[None, :]

    skull = _skull_centre(at)
    head_mask = _along(points, neck_a + 0.5 * axis, neck_b) * near_neck
    out["head_size"] = 0.30 * head_mask[:, None] * (points - skull[None, :])

    # Body length: a scale in depth about the middle of the animal. Global,
    # because "longer" is a claim about the whole silhouette and a masked
    # version would leave the head hanging off a stretched barrel.
    out["body_length"] = np.stack(
        [np.zeros_like(y), np.zeros_like(y), 0.18 * points[:, 2]], axis=1
    )

    # Tail length: a scale about the tail's root, along the same mask shape as
    # the neck's, so a bare tail and a plumed one are one slider apart.
    tail_a, tail_b = at["tail_01"][0], at["tail_02"][1]
    tail_mask = _along(points, tail_a, tail_b) * _within(
        points, tail_a, tail_b, float(np.linalg.norm(tail_b - tail_a)) * 0.55
    )
    out["tail_length"] = 0.45 * tail_mask[:, None] * (points - tail_a[None, :])
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
            f"{silhouette!r} is not a quadruped silhouette; try " + ", ".join(sorted(GROUPS)),
            field="family",
        ) from None

    # The landmarks come from the box and the parts come from the landmarks, so
    # the two are chased to a fixed point. Every extreme is absolute (see the
    # module docstring), which makes the map a contraction with a very small
    # constant: it closes in two or three rounds, and the ``else`` below is a
    # bug rather than a tolerance to loosen.
    half_width, length = group.barrel, group.length
    solid: Any = None
    for _ in range(8):
        joints = _fit_joints(half_width, length)
        solid = _solid(_parts(joints, group))
        lo, hi = bm.bounds(solid)
        measured_w = float(max(-lo[0], hi[0]))
        measured_l = float(hi[2] - lo[2])
        if (
            abs(measured_w - half_width) < FIT_TOLERANCE
            and abs(measured_l - length) < FIT_TOLERANCE
        ):
            break
        half_width, length = measured_w, measured_l
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

    # Grounded, unit-tall and centred, in that order, so the recorded landmarks
    # and the positions are in one frame -- the artifact rule for source.glb,
    # applied before anything derived from it exists.
    lo, hi = bm.bounds(solid)
    scale = 1.0 / float(hi[1] - lo[1])
    centre = np.array([(lo[0] + hi[0]) / 2.0, float(lo[1]), (lo[2] + hi[2]) / 2.0])
    positions = (np.asarray(solid.positions, dtype="f8") - centre) * scale

    joints = _fit_joints(
        float(np.abs(positions[:, 0]).max()),
        float(positions[:, 2].max() - positions[:, 2].min()),
    )

    region_names = familylib.get_archetype("quadruped").regions
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
    """``(region id, positions, indices)`` per region, in region-id order.

    One primitive per region rather than one mesh with a per-face attribute,
    because glTF has no per-face anything: a material is a primitive.
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
    """Source vertex index for every vertex of the concatenated primitives.

    Done by position, which is exact here: the primitives are cut out of one
    mesh, so every primitive vertex is bit-identical to the source vertex it
    came from.
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
