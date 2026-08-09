"""The nine shapes a user can place, and the registry the panel is built from.

Each generator is a plain function of its parameters returning a :class:`Mesh`,
and :data:`GENERATORS` maps a name to ``(defaults, builder)``. The registry is
the point of the module rather than an index over it: the properties panel is
generated from those default dictionaries, so adding a seventh primitive is
adding a function and one registry line, in the same spirit as "add a skeleton
by adding a JSON file, never by hardcoding bones in ``blender_worker``". A
panel that switched on a hardcoded list of shape names would be a second place
that has to know what a cylinder's parameters are, and the two would drift the
first time a parameter was renamed.

Three rules hold across all six, and each of them is pinned by a test:

**Every primitive is built centred on the origin.** ``Obj`` carries the
translation, so geometry that baked its placement in would make the numeric TRS
panel lie -- a box "at the origin" would sit somewhere else, and moving it back
would leave the panel reading a position the object is not at. ``plane`` is
centred too: it lies *in* the XZ plane at y = 0, not on top of it.

**Caps are n-gons, not fans.** The CSR storage exists precisely so a cylinder's
lid can be one face with sixteen corners, and it matters twice over: a fan cap
is sixteen faces where one will do, which the triangle budget notices, and a
user who clicks a cap in face mode expects to select *the cap*, not one wedge
of it. It is also why a cylinder at sixteen segments has thirty-two
vertices and not thirty-four -- there is no cap-centre vertex, because there is
no fan to radiate from one.

**Every face is wound counter-clockwise seen from outside, and convex.** The
winding is what makes the Newell normal point outward, and a flipped cap is the
single nastiest defect this module can ship: the viewport draws with back-face
culling off, so it looks perfect, and the exported GLB is inside out in every
engine with nothing in the file to explain why. Convexity is what makes
``mesh.triangulate``'s fan correct -- it fans from each face's first corner and
a fan across a reflex corner puts a triangle outside the polygon.

That first claim takes **two** assertions, not one, and the obvious one is the
weaker of the pair. Summing ``(centroid - centre) . normal`` over the faces is
six times the enclosed volume, so it says the shell is oriented *outward* --
but reversing a single face changes it by only twice that face's own
contribution, which is small beside the whole volume. Measured on these
generators, flipping one face leaves the sum positive every time, and flipping
the cone's base leaves it at ``7e-18``: positive by float noise. So the tests
also assert that **no ordered corner pair ``(a, b)`` occurs twice across all
faces**, which is what "consistently oriented" means -- neighbouring faces
traverse their shared edge in opposite directions, so a directed edge is used
once and a flipped face immediately collides with its neighbour. The undirected
edge-use count cannot stand in for it: it keys on the sorted pair and is
orientation-blind by construction. Consistency plus a positive volume is the
whole claim; either alone is not.

**Sizes are taken as magnitudes.** Every extent -- a box's ``size``, a radius, a
height, the tube -- is passed through ``abs``, because a negative one does not
mean a mirrored primitive, it means an inside-out one: a negative height puts a
cylinder's bottom ring above its top and every side quad and both caps then
face inward, and ``validate`` would accept it happily because it checks CSR
structure and index ranges and nothing geometric. A numeric property field is
one keystroke away from a minus sign, and mirroring is ``mesh.transformed``'s
job -- it reverses the loops to keep the winding honest, which a generator
handed a negative number cannot do on the caller's behalf.

Every face comes back **flat-shaded on material zero**. Flat rather than smooth
on the curved shapes deliberately: ``smooth`` is per-face and a shading tool
sets it later, and until there is one, faceted geometry that tells the truth
about the mesh it is about to export beats a rounded lie about a sixteen-sided
cylinder. Axes are glTF's -- Y up, right-handed -- because that is the space
the viewer, the exporter and everything downstream of Clay already speak.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .mesh import Mesh

# The low-end clamps. A slider dragged to zero must not be able to produce a
# mesh ``validate`` rejects, and clamping is the right shape for a control that
# is being *dragged*: raising mid-drag would have to be caught by the panel,
# which has nothing sensible to show for it. Three is the smallest ring that is
# a polygon at all; two is the smallest number of latitude bands that leaves a
# sphere with any surface between its poles.
MIN_SEGMENTS = 3
MIN_RINGS = 2

# A grid's own floor, deliberately *not* ``MIN_SEGMENTS``. One division is a
# legitimate grid -- it is ``plane`` -- so clamping it to three would refuse a
# shape the registry already ships, which is why the parameter is called
# ``divisions`` rather than ``segments``: the shared name would drag the shared
# clamp along with it.
MIN_DIVISIONS = 1

# How far an icosphere may be subdivided from the properties panel. Each step
# quadruples the face count (20 -> 80 -> 320 -> 1280 -> 5120 -> 20480), and the
# undo stack holds two meshes per step, so an unbounded integer field is one
# keystroke away from a multi-second stall on a control that is being *typed*.
MAX_SUBDIVISIONS = 5


def _mesh(
    positions: np.ndarray,
    faces: Sequence[Sequence[int]],
    uv: Sequence[Sequence[Sequence[float]]] | None = None,
) -> Mesh:
    """Assemble the CSR arrays from a list of corner loops.

    Every generator funnels through here so that the offsets are computed once,
    in one place, rather than six times with six chances to leave ``starts``
    one short.

    ``uv`` is one ``(u, v)`` per corner of each face, in the same nesting as
    ``faces`` -- per *corner* rather than per vertex, because that is what the
    field is and because it is the only shape that can put the two sides of a
    wrap seam at u = 0 and u = 1 while sharing one position.
    """
    counts = [len(f) for f in faces]
    flat = None
    if uv is not None:
        flat = np.array([c for face in uv for c in face], dtype="f4").reshape(-1, 2)
    return Mesh(
        positions=np.asarray(positions, dtype="f4"),
        loops=np.array([i for f in faces for i in f], dtype="i4"),
        starts=np.concatenate([[0], np.cumsum(counts)]).astype("i4"),
        material=np.zeros(len(faces), dtype="i4"),
        smooth=np.zeros(len(faces), dtype=bool),
        uv=flat,
    )


def _disc_uv(segments: int, centre: tuple[float, float], radius: float, reverse: bool = False):
    """A cap's corners on a circle in UV space, matching ``_ring``'s order.

    Caps go in their own corner of the square rather than sharing it with the
    sides: a canonical unwrap whose islands overlap is one that cannot be baked
    to, and baking is what these coordinates exist for.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    if reverse:
        theta = theta[::-1]
    return [
        (centre[0] + radius * float(np.cos(t)), centre[1] + radius * float(np.sin(t)))
        for t in theta
    ]


def _ring(radius: float, y: float, segments: int) -> np.ndarray:
    """``segments`` points on a circle in the XZ plane, at height *y*.

    Angle zero sits on +X and the sequence runs towards +Z. Seen from above --
    that is, looking down -Y -- that is *clockwise*, because Y-up
    right-handedness puts +Z towards the viewer's chin. So a ring in this order
    is the correct winding for a **bottom** cap, and a top cap is the same ring
    reversed. Getting that backwards is the flipped-cap bug, which is why it is
    written down here rather than rediscovered at each call site.

    A ring is a polygon, so a primitive's measured bounds reach its full radius
    only when a vertex happens to land on the axis -- true for any ``segments``
    divisible by four, which every default here is, and slightly short of it
    otherwise. That is honest behaviour for a polygonal approximation rather
    than something to correct for, but a properties panel showing a measured
    "size" beside a requested radius should expect the two to differ.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    return np.stack(
        [radius * np.cos(theta), np.full(segments, float(y)), radius * np.sin(theta)],
        axis=1,
    )


def _side_quads(lower: int, upper: int, segments: int) -> list[list[int]]:
    """The band of quads between two rings of ``segments`` vertices each.

    ``lower`` and ``upper`` are the first vertex index of each ring. The corner
    order -- lower *i*, upper *i*, upper *i+1*, lower *i+1* -- is the one whose
    Newell normal points away from the axis; the other traversal of the same
    four corners points into it.
    """
    return [
        [
            lower + i,
            upper + i,
            upper + (i + 1) % segments,
            lower + (i + 1) % segments,
        ]
        for i in range(segments)
    ]


# --- the generators ----------------------------------------------------------


def box(size: Sequence[float] = (1.0, 1.0, 1.0)) -> Mesh:
    """An axis-aligned box of the given full extents, centred on the origin.

    A negative extent is taken as its magnitude, not as a mirror -- see the
    module docstring.
    """
    hx, hy, hz = (abs(float(s)) * 0.5 for s in size)
    positions = np.array(
        [
            [-hx, -hy, -hz],
            [+hx, -hy, -hz],
            [+hx, -hy, +hz],
            [-hx, -hy, +hz],
            [-hx, +hy, -hz],
            [+hx, +hy, -hz],
            [+hx, +hy, +hz],
            [-hx, +hy, +hz],
        ],
        dtype="f4",
    )
    faces = [
        [0, 1, 2, 3],  # -Y
        [7, 6, 5, 4],  # +Y
        [3, 2, 6, 7],  # +Z
        [1, 0, 4, 5],  # -Z
        [2, 1, 5, 6],  # +X
        [0, 3, 7, 4],  # -X
    ]
    # A box's canonical unwrap *is* the cube projection, so it is that call
    # rather than a second table of the same six squares written out here.
    from .uv import box_unwrap

    return box_unwrap(_mesh(positions, faces))


def plane(size: Sequence[float] = (1.0, 1.0)) -> Mesh:
    """A single quad in the XZ plane at y = 0, facing +Y.

    The one open primitive: it has a boundary, no volume and no outward
    direction to derive, so its facing is a decision rather than a consequence.
    +Y, because a sheet a user drops into the scene is a floor.
    """
    hx, hz = (abs(float(s)) * 0.5 for s in size)
    positions = np.array(
        [[-hx, 0.0, +hz], [+hx, 0.0, +hz], [+hx, 0.0, -hz], [-hx, 0.0, -hz]],
        dtype="f4",
    )
    return _mesh(positions, [[0, 1, 2, 3]], [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]])


def cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 16) -> Mesh:
    """A tube with an n-gon cap at each end, axis along Y."""
    n = max(int(segments), MIN_SEGMENTS)
    r, h = abs(float(radius)), abs(float(height)) * 0.5
    positions = np.concatenate([_ring(r, -h, n), _ring(r, +h, n)])
    faces = _side_quads(0, n, n)
    faces.append(list(range(n)))  # bottom cap, ring order, normal -Y
    faces.append(list(range(2 * n - 1, n - 1, -1)))  # top cap, reversed, normal +Y

    # The band across the top half of the square, the two caps in the bottom
    # corners. ``u`` runs to exactly 1 on the last quad rather than wrapping to
    # 0, which is the whole reason UVs are per corner.
    uv = [
        [(i / n, 0.5), (i / n, 1.0), ((i + 1) / n, 1.0), ((i + 1) / n, 0.5)]
        for i in range(n)
    ]
    uv.append(_disc_uv(n, (0.25, 0.25), 0.24))
    uv.append(_disc_uv(n, (0.75, 0.25), 0.24, reverse=True))
    return _mesh(positions, faces, uv)


def cone(radius: float = 0.5, height: float = 1.0, segments: int = 16) -> Mesh:
    """A base ring, an n-gon bottom and a fan of triangles up to a single apex.

    The apex is one vertex, not one per side: the sides meet there, and
    splitting them would produce a seam an exporter has no reason to keep. The
    faceting a shared apex would otherwise cause is not a concern here because
    every face is flat-shaded anyway -- ``render_arrays`` splits a flat face's
    corners on the way to the GPU regardless.
    """
    n = max(int(segments), MIN_SEGMENTS)
    r, h = abs(float(radius)), abs(float(height)) * 0.5
    positions = np.concatenate([_ring(r, -h, n), [[0.0, +h, 0.0]]])
    apex = n
    faces: list[list[int]] = [[i, apex, (i + 1) % n] for i in range(n)]
    faces.append(list(range(n)))  # base cap, normal -Y

    # The apex gets a *different* u per side triangle -- it is one vertex and
    # many corners, which is exactly the case a per-corner field exists for. A
    # shared apex uv would smear the whole top of the texture into one point.
    uv = [
        [(i / n, 0.5), ((i + 0.5) / n, 1.0), ((i + 1) / n, 0.5)] for i in range(n)
    ]
    uv.append(_disc_uv(n, (0.25, 0.25), 0.24))
    return _mesh(positions, faces, uv)


def uv_sphere(radius: float = 0.5, segments: int = 16, rings: int = 8) -> Mesh:
    """A latitude/longitude sphere: quads between rings, triangles at the poles.

    ``rings`` counts the *bands* from pole to pole, so there are ``rings - 1``
    intermediate latitude rings and the vertex count is
    ``(rings - 1) * segments + 2``. The poles are single vertices and the bands
    that touch them are therefore triangles, not degenerate quads with two
    coincident corners -- a degenerate quad survives ``validate`` and then
    produces a zero-area triangle in every consumer downstream of it.
    """
    n = max(int(segments), MIN_SEGMENTS)
    m = max(int(rings), MIN_RINGS)
    r = abs(float(radius))

    top, bottom = 0, 1 + (m - 1) * n
    rows = []
    for j in range(1, m):
        phi = np.pi * j / m
        rows.append(_ring(r * np.sin(phi), r * np.cos(phi), n))
    positions = np.concatenate([[[0.0, +r, 0.0]], *rows, [[0.0, -r, 0.0]]])

    def row(j: int) -> int:
        """First vertex index of intermediate ring *j*, counting from the top."""
        return 1 + (j - 1) * n

    faces: list[list[int]] = [[row(1) + i, top, row(1) + (i + 1) % n] for i in range(n)]
    for j in range(1, m - 1):
        faces.extend(_side_quads(row(j + 1), row(j), n))
    faces.extend([bottom, row(m - 1) + i, row(m - 1) + (i + 1) % n] for i in range(n))

    # Equirectangular: u around, v from the south pole up. The whole square,
    # with no caps to pack, because a sphere has no faces that are not part of
    # the same surface.
    def band(j: int) -> float:
        return 1.0 - j / m

    uv: list[list[tuple[float, float]]] = [
        [(i / n, band(1)), ((i + 0.5) / n, 1.0), ((i + 1) / n, band(1))] for i in range(n)
    ]
    for j in range(1, m - 1):
        uv.extend(
            [
                (i / n, band(j + 1)),
                (i / n, band(j)),
                ((i + 1) / n, band(j)),
                ((i + 1) / n, band(j + 1)),
            ]
            for i in range(n)
        )
    uv.extend(
        [((i + 0.5) / n, 0.0), (i / n, band(m - 1)), ((i + 1) / n, band(m - 1))]
        for i in range(n)
    )
    return _mesh(positions, faces, uv)


def torus(
    radius: float = 0.35,
    tube: float = 0.15,
    segments: int = 24,
    sides: int = 12,
) -> Mesh:
    """A ring of ``segments`` tube cross-sections, each of ``sides`` corners.

    ``radius`` is to the centre of the tube and ``tube`` is the tube's own
    radius, so the outer extent is their sum -- the same pair of numbers
    Blender's torus takes, and the reason the default pair adds to 0.5: every
    primitive here defaults to fitting a one-metre box.

    **A ``tube`` larger than ``radius`` self-intersects and is not refused
    here.** The mesh stays valid, closed and outward-wound -- the tube simply
    passes through its own axis -- so there is no geometric rule to enforce, and
    the two are legitimately independent right up to the point where they are
    not. Keeping the pair sane is the properties panel's business (a soft clamp
    on the tube slider), which is the same division of labour as ``segments``:
    the generator refuses only what it cannot represent.
    """
    n = max(int(segments), MIN_SEGMENTS)
    k = max(int(sides), MIN_SEGMENTS)
    R, r = abs(float(radius)), abs(float(tube))

    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)[:, None]
    phi = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)[None, :]
    ring_radius = R + r * np.cos(phi)
    positions = np.stack(
        [
            ring_radius * np.cos(theta),
            np.broadcast_to(r * np.sin(phi), (n, k)),
            ring_radius * np.sin(theta),
        ],
        axis=-1,
    ).reshape(-1, 3)

    # Around the tube first, then around the ring: the reverse traversal of the
    # same four corners points into the tube rather than out of it.
    faces = [
        [
            i * k + j,
            i * k + (j + 1) % k,
            ((i + 1) % n) * k + (j + 1) % k,
            ((i + 1) % n) * k + j,
        ]
        for i in range(n)
        for j in range(k)
    ]
    # Both seams run to exactly 1 rather than wrapping to 0, for the reason the
    # cylinder's does: the corners are per face, so the last ring and the last
    # tube section can each close the square instead of folding it.
    uv = [
        [
            (i / n, j / k),
            (i / n, (j + 1) / k),
            ((i + 1) / n, (j + 1) / k),
            ((i + 1) / n, j / k),
        ]
        for i in range(n)
        for j in range(k)
    ]
    return _mesh(positions, faces, uv)


def grid(size: Sequence[float] = (1.0, 1.0), divisions: int = 4) -> Mesh:
    """A subdivided sheet in the XZ plane at y = 0, facing +Y.

    ``plane`` is this at one division and stays a separate entry rather than
    being folded in: it is a single quad, which is what a decal, a billboard or
    a backdrop wants, and a user who asks for "plane" and gets sixteen faces has
    to go and find the number that undoes that. What a grid is *for* is the
    thing a single quad cannot do -- carry a displacement, a proportional edit
    or a terrain sculpt, all of which need interior vertices to move.

    Open, like ``plane``: it has a boundary and no volume, so there is no
    outward direction to derive and +Y is the same decision for the same reason.

    UV runs 0..1 across the whole sheet, so a texture fits it once however many
    divisions it is cut into -- the coordinates describe the surface, not the
    faces it happens to be made of.
    """
    hx, hz = (abs(float(s)) * 0.5 for s in size)
    n = max(int(divisions), MIN_DIVISIONS)
    xs = np.linspace(-hx, +hx, n + 1)
    zs = np.linspace(+hz, -hz, n + 1)
    positions = np.array(
        [[x, 0.0, z] for z in zs for x in xs],
        dtype="f4",
    )

    def at(i: int, j: int) -> int:
        """Vertex index of column *i*, row *j*, rows running +Z to -Z."""
        return j * (n + 1) + i

    # The same corner order ``plane`` uses -- (-x,+z), (+x,+z), (+x,-z), (-x,-z)
    # -- which is the traversal whose Newell normal points at +Y.
    faces = [
        [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]
        for j in range(n)
        for i in range(n)
    ]
    uv = [
        [
            (i / n, j / n),
            ((i + 1) / n, j / n),
            ((i + 1) / n, (j + 1) / n),
            (i / n, (j + 1) / n),
        ]
        for j in range(n)
        for i in range(n)
    ]
    return _mesh(positions, faces, uv)


def capsule(
    radius: float = 0.25,
    height: float = 0.5,
    segments: int = 16,
    rings: int = 4,
) -> Mesh:
    """A cylinder with a hemisphere on each end, axis along Y.

    ``height`` is the **cylindrical** section alone, so the whole shape spans
    ``height + 2 * radius``. That is Blender's reading of the same pair and it is
    the one that keeps the two numbers independent: a capsule whose ``height``
    meant the total would silently stop being a capsule -- and start being a
    sphere, or nothing -- as the radius passed half of it, which is a shape
    change with no control the user can see it in. The defaults still add up to
    a one-metre box, as every other primitive's do.

    ``rings`` counts the latitude bands in **one** hemisphere, so the profile is
    ``rings`` bands up, one cylinder band across the middle, and ``rings`` bands
    down. There is no cap n-gon and no pole fan sharing a ring with the tube:
    the two hemisphere rings that meet the cylinder are separate rows of
    vertices at the same radius, which is what lets the cylinder band be a plain
    quad strip and keeps every band's winding the same expression.

    UV is a cylindrical unwrap whose ``v`` follows **arc length along the
    profile** rather than height, so the texel density does not collapse at the
    ends: a naive v = (y - min) / span squashes each hemisphere into a band as
    tall as its own bulge, which on a stubby capsule is most of the texture in
    a tenth of the square. ``u`` runs to exactly 1 on the last column, per
    corner, for the reason the cylinder's does.
    """
    n = max(int(segments), MIN_SEGMENTS)
    m = max(int(rings), MIN_RINGS)
    r, h = abs(float(radius)), abs(float(height)) * 0.5

    # The profile, north to south, as (ring radius, y). The two poles are the
    # ends and are single vertices; everything between them is a full ring.
    profile: list[tuple[float, float]] = [(0.0, +h + r)]
    for k in range(1, m + 1):
        phi = 0.5 * np.pi * k / m
        profile.append((r * float(np.sin(phi)), +h + r * float(np.cos(phi))))
    for k in range(m):
        psi = 0.5 * np.pi * k / m
        profile.append((r * float(np.cos(psi)), -h - r * float(np.sin(psi))))
    profile.append((0.0, -h - r))

    rows = profile[1:-1]
    positions = np.concatenate(
        [[[0.0, profile[0][1], 0.0]]]
        + [_ring(rr, y, n) for rr, y in rows]
        + [[[0.0, profile[-1][1], 0.0]]]
    )
    top, bottom = 0, 1 + len(rows) * n

    def row(j: int) -> int:
        """First vertex index of intermediate ring *j*, counting from the top."""
        return 1 + j * n

    # The pole fans are wound the way ``uv_sphere``'s are, and the bands between
    # rings the way ``_side_quads`` orders them -- upper ring first, because the
    # rings run downward and "lower" there means the ring further from +Y.
    faces: list[list[int]] = [[row(0) + i, top, row(0) + (i + 1) % n] for i in range(n)]
    for j in range(len(rows) - 1):
        faces.extend(_side_quads(row(j + 1), row(j), n))
    faces.extend(
        [bottom, row(len(rows) - 1) + i, row(len(rows) - 1) + (i + 1) % n] for i in range(n)
    )

    # v by arc length along the profile, measured south-to-north so v = 0 is the
    # bottom pole -- the same direction ``uv_sphere``'s bands run.
    points = np.array(profile, dtype="f8")
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(along[-1])
    v = 1.0 - along / total if total > 0.0 else np.zeros(len(along))

    uv: list[list[tuple[float, float]]] = [
        [(i / n, float(v[1])), ((i + 0.5) / n, float(v[0])), ((i + 1) / n, float(v[1]))]
        for i in range(n)
    ]
    for j in range(len(rows) - 1):
        uv.extend(
            [
                (i / n, float(v[j + 2])),
                (i / n, float(v[j + 1])),
                ((i + 1) / n, float(v[j + 1])),
                ((i + 1) / n, float(v[j + 2])),
            ]
            for i in range(n)
        )
    uv.extend(
        [
            ((i + 0.5) / n, float(v[-1])),
            (i / n, float(v[-2])),
            ((i + 1) / n, float(v[-2])),
        ]
        for i in range(n)
    )
    return _mesh(positions, faces, uv)


# The regular icosahedron, before normalisation: three mutually perpendicular
# golden rectangles. Written out rather than derived because the *face* list
# below is written out and the two have to agree about which vertex is which --
# deriving one and not the other is how a shell comes out with five faces
# reversed and passes every test that only counts things.
_ICO_T = (1.0 + 5.0**0.5) * 0.5

_ICO_VERTS = (
    (-1.0, _ICO_T, 0.0), (1.0, _ICO_T, 0.0), (-1.0, -_ICO_T, 0.0), (1.0, -_ICO_T, 0.0),
    (0.0, -1.0, _ICO_T), (0.0, 1.0, _ICO_T), (0.0, -1.0, -_ICO_T), (0.0, 1.0, -_ICO_T),
    (_ICO_T, 0.0, -1.0), (_ICO_T, 0.0, 1.0), (-_ICO_T, 0.0, -1.0), (-_ICO_T, 0.0, 1.0),
)

# Every triangle counter-clockwise seen from outside, which is what makes the
# Newell normal point away from the centre -- the module's third rule, and the
# one whose failure is invisible in the viewport.
_ICO_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


def _subdivide(
    verts: list[np.ndarray], faces: list[tuple[int, int, int]]
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """One Loop-style split: each triangle into four, midpoints on the sphere.

    The midpoint cache is keyed on the *sorted* vertex pair, which is what makes
    two triangles sharing an edge share the vertex on it. Without it the shell
    cracks: each face would mint its own copy, the positions would coincide, and
    the mesh would be two-manifold in appearance and open in every measurement.

    The four children are wound in the parent's own direction, so consistency is
    a property of the base list and is never re-derived.
    """
    cache: dict[tuple[int, int], int] = {}

    def middle(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        found = cache.get(key)
        if found is not None:
            return found
        point = verts[a] + verts[b]
        verts.append(point / np.linalg.norm(point))
        cache[key] = len(verts) - 1
        return cache[key]

    out: list[tuple[int, int, int]] = []
    for a, b, c in faces:
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        out.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
    return verts, out


def _sphere_uv(
    positions: np.ndarray, faces: Sequence[Sequence[int]]
) -> list[list[tuple[float, float]]]:
    """A latitude/longitude unwrap of a sphere whose faces do not share rings.

    ``uv_sphere`` and ``capsule`` can put ``u`` on a corner directly, because
    their faces are laid out in columns and the last column simply closes at 1.
    An icosphere has no columns: its triangles straddle the seam meridian, and a
    per-*vertex* longitude therefore hands one of them the corner pair
    ``0.98, 0.02`` and wraps the whole texture backwards across it.

    So longitude is unwrapped **per face**: each corner is taken to the branch
    nearest the face's first corner, which makes the island contiguous. That can
    put it outside the square by up to its own width, and the island is then
    *translated* back in -- never scaled, so the mapping stays exact and the only
    thing paid is continuity, at the seam, where a sphere has none to keep.

    A corner on the axis has no longitude at all; it takes the mean of its
    face's other two, which is the same answer ``cone`` gives its apex and for
    the same reason.
    """
    positions = np.asarray(positions, dtype="f8")
    radius = float(np.max(np.linalg.norm(positions, axis=1))) or 1.0
    lon = (np.arctan2(positions[:, 2], positions[:, 0]) / (2.0 * np.pi)) % 1.0
    axial = np.hypot(positions[:, 0], positions[:, 2]) < 1e-9
    lat = 0.5 + np.arcsin(np.clip(positions[:, 1] / radius, -1.0, 1.0)) / np.pi

    out: list[list[tuple[float, float]]] = []
    for face in faces:
        known = [lon[i] for i in face if not axial[i]]
        anchor = known[0] if known else 0.0
        us = []
        for i in face:
            if axial[i]:
                us.append(None)
                continue
            # The branch nearest the anchor: ``round`` of the difference is the
            # number of whole turns to take off.
            us.append(float(lon[i] - round(float(lon[i]) - anchor)))
        fallback = float(np.mean([u for u in us if u is not None])) if known else 0.0
        row = [fallback if u is None else u for u in us]
        low, high = min(row), max(row)
        shift = -low if low < 0.0 else (1.0 - high if high > 1.0 else 0.0)
        out.append([(u + shift, float(lat[i])) for u, i in zip(row, face, strict=True)])
    return out


def icosphere(radius: float = 0.5, subdivisions: int = 2) -> Mesh:
    """A geodesic sphere: an icosahedron, split and pushed back out.

    The one sphere with **even triangles**. ``uv_sphere``'s poles are fans of
    slivers and its equator quads are many times their area, which is what makes
    it the wrong ball to sculpt, to bevel, or to bake to -- and the right one to
    wrap an equirectangular texture round, which is why both ship rather than
    one replacing the other.

    ``subdivisions`` is capped at :data:`MAX_SUBDIVISIONS`; see the comment
    there. Zero is the bare icosahedron, which is a legitimate shape to place.
    """
    r = abs(float(radius))
    depth = min(max(int(subdivisions), 0), MAX_SUBDIVISIONS)
    verts = [np.asarray(v, dtype="f8") / np.linalg.norm(v) for v in _ICO_VERTS]
    faces = [tuple(f) for f in _ICO_FACES]
    for _ in range(depth):
        verts, faces = _subdivide(verts, faces)
    unit = np.asarray(verts, dtype="f8")
    return _mesh(unit * r, faces, _sphere_uv(unit, faces))


# --- the registry ------------------------------------------------------------

GENERATORS: dict[str, tuple[dict[str, Any], Callable[..., Mesh]]] = {
    "box": ({"size": (1.0, 1.0, 1.0)}, box),
    "plane": ({"size": (1.0, 1.0)}, plane),
    "cylinder": ({"radius": 0.5, "height": 1.0, "segments": 16}, cylinder),
    "cone": ({"radius": 0.5, "height": 1.0, "segments": 16}, cone),
    "uv_sphere": ({"radius": 0.5, "segments": 16, "rings": 8}, uv_sphere),
    "torus": ({"radius": 0.35, "tube": 0.15, "segments": 24, "sides": 12}, torus),
    "grid": ({"size": (1.0, 1.0), "divisions": 4}, grid),
    "capsule": ({"radius": 0.25, "height": 0.5, "segments": 16, "rings": 4}, capsule),
    "icosphere": ({"radius": 0.5, "subdivisions": 2}, icosphere),
}
"""Name -> ``(defaults, builder)``. Every default dictionary is a complete call.

Complete because that is exactly what the properties panel does with it: splat
it into the builder to make the object the user just asked for, then bind each
key to a widget. A missing key would be a ``TypeError`` raised the first time
somebody picked that shape, which is why a test calls every entry this way.
"""
