"""One parameterised amorphous generator, baked per silhouette group.

The humanoid generator's shape a third time -- primitives to a boolean union,
one weld, one Catmull-Clark level, one weld; regions from geometry after the
union; channels as per-vertex displacement fields evaluated by the same
function over the vertices and over the joints. Read
``characters/humanoid/generate.py`` first, and ``characters/quadruped/`` for why
every extreme of the box is an absolute constant.

**Translucency is a palette here, and that is a finding rather than a choice.**
A slime wants to be glass with something suspended in it, and glTF can say so:
``gltf.Material`` carries ``alpha_mode`` and an alpha in ``base_color_factor``,
and ``glbwrite`` writes both. What cannot say so is the thing that renders it.
The sprite pipeline's ``_make_flat`` rewires every material to drive an Emission
node from whatever fed Base Color and links that straight to the surface output
-- alpha is not read, and there is no transparent BSDF anywhere in the path. So
a body marked ``BLEND`` at 0.5 renders exactly as the same body at 1.0, and the
setting would have been a number in a file that changed nothing on screen.
Rather than ship that, the nucleus **breaks the surface**: shards on an
elemental, a bright cap everywhere else, painted from the ``core`` region. What
you see through a slime is a thing actually sticking out of it, which is honest
at every camera angle a sheet renders from. If the worker ever grows a
transparent path, the change is a material flag in ``instantiate`` and this
paragraph, and no re-bake.

**Where the ``blob`` template puts its landmarks.** ``base`` is at the floor
centre and ``top`` at the ceiling centre -- both on the vertical axis, so a
flat-bottomed cylinder and the body ellipsoid's own pole answer them without
anything having to end in a point (the quadruped's problem, and it is not this
archetype's). The four lobes sit at 0.88 of the half width and 0.84 of the half
depth, four tenths of the way up, and the bulges grown on them are sized to
reach the sides of the box exactly -- so the body ellipsoid stops seven per cent
short on all four sides and the *lobes* are what the fit measures. Both numbers
are absolute, so the loop still closes in one round, and every lobe joint is at
the centre of its own bulge rather than near the edge of something else's.

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

# --- the neutral body --------------------------------------------------------
#
# Fractions of total height, glTF axes (Y up, +X the subject's left, +Z
# forward), the sole on y = 0 and the crown at y = 1.

#: The body ellipsoid's vertical centre and radius. It reaches y = 1 exactly and
#: stops short of the floor, which the base cylinder covers -- two surfaces that
#: cross transversally rather than one that is tangent to the ground plane.
BODY_Y = 0.520
BODY_RY = 0.480

#: The base: a cylinder, and a cylinder because one Catmull-Clark level replaces
#: a face with its own centroid, so an n-gon bottom keeps a vertex exactly on
#: y = 0 at x = z = 0 -- which is where the ``base`` landmark is.
BASE_HEIGHT = 0.300
BASE_FRACTION = 0.700  # of the half width

#: The body ellipsoid's radii, as a fraction of the box's. It stops short on
#: all four sides so the *lobes* reach them: a lobe that could not reach the box
#: would be a bulge you cannot see, and the template's own lobe landmarks sit at
#: 0.88 and 0.84 of the half width and half depth, which is where the bulges go
#: and therefore what the two radii below have to add up to.
BODY_FRACTION = 0.930
LOBE_X = 0.880
LOBE_Z = 0.840
LOBE_R_X = 0.120  # of the half width;  0.880 + 0.120 == 1
LOBE_R_Z = 0.160  # of the half depth;  0.840 + 0.160 == 1

SEG = 16
RINGS = 10
MAX_FACES = 12000

#: Weld tolerances, before and after the smoothing. The second is a hundredth of
#: the first for the quadruped module's measured reason.
SEAM_EPS = 1e-5
SMOOTH_EPS = 1e-7

FIT_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class Group:
    """One silhouette group: the box it fills and what it grows."""

    #: Half the width of the box. The L and R lobe bulges reach it exactly; the
    #: body ellipsoid stops at :data:`BODY_FRACTION` of it.
    half_width: float
    #: Half the depth of the box, reached the same way by the front and back
    #: lobes. A shard may stand outside both -- see :func:`_shard_segments`.
    half_depth: float
    #: Crystal shards breaking the surface, painted from the ``core`` region.
    shards: bool = False
    #: Extra lumps around the upper body, for a silhouette with no one outline.
    lumps: bool = False


#: ``silhouette -> Group``. The keys are exactly ``family.silhouettes`` for the
#: amorphous archetype, which ``test_every_silhouette_group_has_an_asset`` pins.
GROUPS: dict[str, Group] = {
    "smooth": Group(half_width=0.550, half_depth=0.500),
    "crystal": Group(half_width=0.440, half_depth=0.420, shards=True),
    "puff": Group(half_width=0.600, half_depth=0.560, lumps=True),
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


def _fit_joints(half_width: float, half_depth: float) -> list[dict[str, Any]]:
    from ... import rigging

    template = rigging.get_template("blob")
    lo = [-half_width, -half_depth, 0.0]
    hi = [half_width, half_depth, 1.0]
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


def _sphere_at(centre: Any, radius: float, *, segments: int = 12, rings: int = 8) -> Any:
    from ...studio.clay import primitives

    body = primitives.uv_sphere(radius=radius, segments=segments, rings=rings)
    return _placed(body, np.eye(3), np.asarray(centre, dtype="f8"))


def _ellipsoid_at(
    centre: Any, radii: Any, *, segments: int = SEG, rings: int = RINGS
) -> Any:
    from ...studio.clay import primitives

    body = primitives.uv_sphere(radius=1.0, segments=segments, rings=rings)
    return _placed(body, np.diag(np.asarray(radii, dtype="f8")), np.asarray(centre, dtype="f8"))


def _cylinder_at(centre: Any, radius: float, height: float, *, segments: int = SEG) -> Any:
    from ...studio.clay import primitives

    body = primitives.cylinder(radius=radius, height=height, segments=segments)
    return _placed(body, np.eye(3), np.asarray(centre, dtype="f8"))


def _cone_between(a: Any, b: Any, radius: float, *, segments: int = 6) -> Any:
    from ...studio.clay import primitives

    a = np.asarray(a, dtype="f8")
    b = np.asarray(b, dtype="f8")
    span = b - a
    length = float(np.linalg.norm(span))
    body = primitives.cone(radius=radius, height=length, segments=segments)
    return _placed(body, _basis_from_y(span), (a + b) / 2.0)


# --- named landmarks the mesh and the regions both read ----------------------

#: ``(azimuth, height, radius, length)`` per shard, in body-relative units. Six
#: of them, at angles that are not a multiple of each other so the silhouette
#: never reads as a ring.
_SHARDS: tuple[tuple[float, float, float, float], ...] = (
    (0.35, 0.78, 0.055, 0.230),
    (1.90, 0.62, 0.070, 0.280),
    (3.05, 0.84, 0.048, 0.200),
    (4.10, 0.48, 0.075, 0.260),
    (5.35, 0.70, 0.060, 0.240),
    (2.55, 0.34, 0.065, 0.210),
)


def _shard_segments(group: Group) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """``(base, tip, radius)`` per shard, all of them inside the box.

    The tip's reach is scaled by how much room the body ellipsoid leaves in that
    direction, so a shard is always proud of the surface and never past the
    side: it is the ``core`` region a theme lights, and a shard that set the
    bounding box would put the fit's width in the hands of a decoration.
    """
    if not group.shards:
        return []
    out = []
    for angle, height, radius, length in _SHARDS:
        direction = np.array([np.cos(angle), 0.30, np.sin(angle)])
        direction = direction / np.linalg.norm(direction)
        surface = np.array([
            group.half_width * np.cos(angle) * 0.62,
            height,
            group.half_depth * np.sin(angle) * 0.62,
        ])
        base = surface - direction * 0.10
        tip = surface + direction * length
        out.append((base, tip, radius))
    return out


def _lump_centres(group: Group) -> list[tuple[np.ndarray, float]]:
    """``(centre, radius)`` per lump. A cloud has no one outline."""
    if not group.lumps:
        return []
    out = []
    for i, (angle, height, radius) in enumerate((
        # Every one of these clears the crown: a lump whose top passed y = 1
        # would become the ceiling of the box off the axis, and the ``top``
        # landmark -- which is on the axis by the template's arithmetic -- would
        # be left hanging in the air beside it.
        (0.60, 0.74, 0.190),
        (2.20, 0.79, 0.150),
        (3.70, 0.68, 0.205),
        (5.10, 0.77, 0.165),
        (1.35, 0.52, 0.175),
        (4.55, 0.46, 0.185),
    )):
        centre = np.array([
            group.half_width * np.cos(angle) * 0.52,
            height,
            group.half_depth * np.sin(angle) * 0.52,
        ])
        out.append((centre, radius))
        del i
    return out


def _eye_centres(group: Group) -> list[np.ndarray]:
    """Two dark spots on the front of the upper body. Every group has them: a
    slime with eyes is a character and a slime without them is a prop."""
    return [
        np.array([side * group.half_width * 0.26, 0.660, group.half_depth * 0.82])
        for side in (1.0, -1.0)
    ]


# --- the body ----------------------------------------------------------------


def _parts(
    joints: list[dict[str, Any]], group: Group, shift: np.ndarray
) -> list[tuple[str, Any]]:
    """Every closed solid the union is built from, named for its refusal text.

    The parts that set the box: the lobe bulges on all four sides, the body
    ellipsoid's pole at ``y = 1``, and the base cylinder's flat bottom on
    ``y = 0``.

    *shift* moves the body, its base, its lobes and its eyes in x and z, and
    moves **nothing else**. It is what answers a decoration that is not
    symmetric: the ``top`` landmark sits on the *bounding box's* axis, so a
    shard cluster leaning one way pulls the box's centre away from the body's
    own pole and leaves that landmark in the air beside the crown -- measured,
    on the crystal group, where it was the only joint of any archetype outside
    its mesh. Shifting the body and not the shards makes the map from shift to
    measured centre a contraction, so :func:`build` closes it in a few rounds.
    """
    del joints  # every part is placed in absolute units; see the docstring
    parts: list[tuple[str, Any]] = []
    shift = np.asarray(shift, dtype="f8")

    parts.append((
        "body",
        _ellipsoid_at(
            shift + [0.0, BODY_Y, 0.0],
            (group.half_width * BODY_FRACTION, BODY_RY, group.half_depth * BODY_FRACTION),
        ),
    ))
    parts.append((
        "base",
        _cylinder_at(
            shift + [0.0, BASE_HEIGHT / 2.0, 0.0],
            group.half_width * BASE_FRACTION,
            BASE_HEIGHT,
        ),
    ))

    # Lobes: four bulges on the equator, on the template's own lobe landmarks
    # and reaching exactly the side of the box.
    for name, dx, dz, radius in (
        ("L", 1.0, 0.0, group.half_width * LOBE_R_X),
        ("R", -1.0, 0.0, group.half_width * LOBE_R_X),
        ("front", 0.0, 1.0, group.half_depth * LOBE_R_Z),
        ("back", 0.0, -1.0, group.half_depth * LOBE_R_Z),
    ):
        centre = shift + [
            dx * group.half_width * LOBE_X, 0.400, dz * group.half_depth * LOBE_Z,
        ]
        parts.append((f"lobe.{name}", _sphere_at(centre, radius, segments=12, rings=8)))

    for i, (centre, r) in enumerate(_lump_centres(group)):
        parts.append((f"lump{i}", _sphere_at(centre, r, segments=12, rings=8)))
    for i, (a, b, r) in enumerate(_shard_segments(group)):
        parts.append((f"shard{i}", _cone_between(a, b, r)))

    for i, eye in enumerate(_eye_centres(group)):
        parts.append((f"eye{i}", _sphere_at(shift + eye, 0.062, segments=10, rings=6)))

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
    """One region id per face, from geometry.

    ``core`` is the nucleus and it is on the *outside*, because nothing in the
    render path reads alpha -- see the module docstring. On the crystal group it
    is the shards; everywhere else it is the cap the body thins to, which is
    where a nucleus would show through if anything could see through.
    """
    del joints
    index = {name: i for i, name in enumerate(region_names)}
    c = _centroids(positions, loops, starts)
    ids = np.full(len(c), index["body"], dtype="i4")

    if group.shards:
        for a, b, r in _shard_segments(group):
            ids[_near_segment(c, a, b) < r * 2.6] = index["core"]
    else:
        ids[c[:, 1] > 0.845] = index["core"]

    noise = _hash_noise(c * 37.0)
    ids[(ids == index["body"]) & (noise > 0.84) & (c[:, 1] > 0.12)] = index["accent"]

    for eye in _eye_centres(group):
        ids[np.linalg.norm(c - eye, axis=1) < 0.070] = index["eye"]
    return ids


# --- the appearance channels -------------------------------------------------


def _smoothstep(x: np.ndarray) -> np.ndarray:
    t = np.clip(x, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wave(points: np.ndarray) -> np.ndarray:
    """A smooth, band-limited, deterministic scalar in ``[-1, 1]``.

    Not :func:`_hash_noise`, and the difference matters here rather than in the
    region rules: a region id is per face and may be as discontinuous as it
    likes, but a *displacement* field is read at every vertex and a hash would
    move each of them independently -- which is not a rippled surface, it is a
    torn one. Three sines are continuous everywhere, are the same on every
    machine, and repeat at a scale you can see.
    """
    return (
        np.sin(6.1 * points[:, 0] + 1.3)
        * np.sin(5.7 * points[:, 1] + 2.1)
        * np.sin(6.9 * points[:, 2] + 0.7)
    )


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
    """Every channel's displacement, per unit of channel, at *points*.

    Every one of them is a function of the *radius* from the vertical axis or of
    the height, and never of which bone is nearest -- a blob's bones are one
    column and four spokes, so a nearest-bone mask would cut the equator into
    four visible quadrants and a slider would show the seams.
    """
    del joints
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radial = np.stack([x, np.zeros_like(y), z], axis=1)
    zero = np.zeros_like(y)

    out: dict[str, np.ndarray] = {}

    # Bulk: a scale about the vertical axis. Radial rather than away-from-bone,
    # so the surface grows evenly and the four lobe joints ride out with it.
    out["bulk"] = 0.22 * radial

    # Viscosity: a thick body stands up, a thin one spreads. One volume-ish
    # trade -- taller *and* narrower -- because a slider that only made things
    # taller would be a second bulk with the sign flipped.
    out["viscosity"] = np.stack([-0.16 * x, 0.26 * (y - 0.0), -0.16 * z], axis=1)

    # Lobes: push the equator out and pull the poles in, so the outline goes
    # from an egg to a clover without the height changing.
    band = np.exp(-(((y - 0.40) / 0.26) ** 2))
    out["lobe"] = 0.30 * band[:, None] * radial

    # Surface: a smooth standing wave, radial. See :func:`_wave` for why it is
    # not the hash the region rules use.
    out["ripple"] = 0.10 * _wave(points)[:, None] * radial

    # Crown: raise or flatten the top half about the equator.
    lift = _smoothstep((y - 0.42) / 0.50)
    out["crown"] = np.stack([zero, 0.26 * lift, zero], axis=1)
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
            f"{silhouette!r} is not an amorphous silhouette; try " + ", ".join(sorted(GROUPS)),
            field="family",
        ) from None

    half_width, half_depth = group.half_width, group.half_depth
    shift = np.zeros(3)
    solid: Any = None
    for _ in range(12):
        joints = _fit_joints(half_width, half_depth)
        solid = _solid(_parts(joints, group, shift))
        lo, hi = bm.bounds(solid)
        centre = np.array([(lo[0] + hi[0]) / 2.0, 0.0, (lo[2] + hi[2]) / 2.0])
        measured_w = float(hi[0] - lo[0]) / 2.0
        measured_d = float(hi[2] - lo[2]) / 2.0
        if (
            abs(measured_w - half_width) < FIT_TOLERANCE
            and abs(measured_d - half_depth) < FIT_TOLERANCE
            and float(np.abs(centre - shift).max()) < FIT_TOLERANCE
        ):
            break
        half_width, half_depth, shift = measured_w, measured_d, centre
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
        float(np.abs(positions[:, 0]).max()), float(np.abs(positions[:, 2]).max())
    )

    region_names = familylib.get_archetype("amorphous").regions
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
