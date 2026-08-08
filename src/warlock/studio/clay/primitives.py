"""The six shapes a user can place, and the registry the panel is built from.

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


# --- the registry ------------------------------------------------------------

GENERATORS: dict[str, tuple[dict[str, Any], Callable[..., Mesh]]] = {
    "box": ({"size": (1.0, 1.0, 1.0)}, box),
    "plane": ({"size": (1.0, 1.0)}, plane),
    "cylinder": ({"radius": 0.5, "height": 1.0, "segments": 16}, cylinder),
    "cone": ({"radius": 0.5, "height": 1.0, "segments": 16}, cone),
    "uv_sphere": ({"radius": 0.5, "segments": 16, "rings": 8}, uv_sphere),
    "torus": ({"radius": 0.35, "tube": 0.15, "segments": 24, "sides": 12}, torus),
}
"""Name -> ``(defaults, builder)``. Every default dictionary is a complete call.

Complete because that is exactly what the properties panel does with it: splat
it into the builder to make the object the user just asked for, then bind each
key to a widget. A missing key would be a ``TypeError`` raised the first time
somebody picked that shape, which is why a test calls every entry this way.
"""
