"""Ray tests, analytically.

No GPU id buffer anywhere. Everything the viewer lets you click is a handle at
a known place with a known radius -- a joint marker, a gizmo ring, a gizmo
arrow -- so a closed-form test against a sphere, a torus band or a segment is
both cheaper than a readback and, unlike one, testable without a context.

Clay adds the one case the analytic handles never covered: clicking the
*geometry*. :func:`ray_triangles` is Moller-Trumbore, vectorised over the whole
triangle array at once because a blockout mesh is hundreds of triangles and a
Python loop per mouse move would be felt; :func:`ray_object` is the object-space
wrapper that puts the ray through the inverse transform once instead of putting
every triangle through the forward one. It is still not an id buffer, and for
the same reason as everything else here -- a readback cannot be asserted without
a window, and every quiet way this arithmetic goes wrong (a hit behind the eye,
a NaN out of a degenerate triangle, the further of two triangles) is a thing a
test should be able to state.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np


def ray_sphere(
    origin: np.ndarray, direction: np.ndarray, centre: np.ndarray, radius: float
) -> float | None:
    """-> distance along the ray to the near intersection, or None.

    A hit behind the eye is not a hit: the markers are drawn with depth testing
    off, so a joint on the far side of the model is legitimately clickable, but
    one behind the camera is not.
    """
    offset = np.asarray(origin, dtype="f8") - np.asarray(centre, dtype="f8")
    b = float(np.dot(offset, direction))
    c = float(np.dot(offset, offset)) - radius * radius
    disc = b * b - c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    for t in (-b - root, -b + root):
        if t >= 0:
            return t
    return None


def nearest_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    centres: dict[str, np.ndarray],
    radius: float,
) -> str | None:
    """The closest of several equal-radius spheres, by name."""
    best: tuple[float, str] | None = None
    for name, centre in centres.items():
        t = ray_sphere(origin, direction, centre, radius)
        if t is not None and (best is None or t < best[0]):
            best = (t, name)
    return None if best is None else best[1]


def ray_plane(
    origin: np.ndarray, direction: np.ndarray, point: np.ndarray, normal: np.ndarray
) -> np.ndarray | None:
    """Where a ray meets a plane, or None if it runs parallel to it."""
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(np.asarray(point, dtype="f8") - origin, normal)) / denom
    if t < 0:
        return None
    return np.asarray(origin, dtype="f8") + direction * t


def ray_ring(
    origin: np.ndarray,
    direction: np.ndarray,
    centre: np.ndarray,
    normal: np.ndarray,
    radius: float,
    tolerance: float,
) -> np.ndarray | None:
    """A rotate-gizmo ring: the plane hit, if it landed in the ring's band.

    Returns the hit point rather than a distance, because the caller
    immediately needs it to work out the drag's starting angle.
    """
    hit = ray_plane(origin, direction, centre, normal)
    if hit is None:
        return None
    if abs(float(np.linalg.norm(hit - centre)) - radius) > tolerance:
        return None
    return hit


def closest_on_axis(
    origin: np.ndarray,
    direction: np.ndarray,
    point: np.ndarray,
    axis: np.ndarray,
) -> tuple[float, float]:
    """-> (parameter along the axis, distance between the two lines).

    The standard closest-approach-of-two-lines solve. The distance is what a
    translate arrow's hit test thresholds on; the parameter is what the drag
    then tracks.
    """
    axis = np.asarray(axis, dtype="f8")
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    w0 = np.asarray(point, dtype="f8") - np.asarray(origin, dtype="f8")
    b = float(np.dot(axis, direction))
    denom = 1.0 - b * b
    if abs(denom) < 1e-9:
        # Looking straight down the axis: every point on it is equidistant, so
        # report the projection and let the caller's threshold decide.
        return 0.0, float(np.linalg.norm(np.cross(w0, direction)))
    d = float(np.dot(axis, w0))
    e = float(np.dot(direction, w0))
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    on_axis = np.asarray(point, dtype="f8") + axis * s
    on_ray = np.asarray(origin, dtype="f8") + direction * t
    return s, float(np.linalg.norm(on_axis - on_ray))


# Below this, a triangle is treated as edge-on or degenerate and skipped. It is
# a determinant of edge vectors, so it scales with the square of the mesh's
# units -- generous rather than tight, because the cost of skipping a triangle
# the ray grazes is nothing (its neighbours catch the click) and the cost of
# not skipping one is a division by almost zero and a NaN in the comparison
# that picks the nearest hit.
TRIANGLE_EPS = 1e-12


# --- the bounding-volume hierarchy ------------------------------------------


@dataclass(frozen=True)
class BVH:
    """A median-split AABB tree over a triangle list, flattened into arrays.

    Built once per *mesh*, which is what makes it affordable: a ``Mesh`` is
    immutable, so the tree is a pure function of one frozen object and the cache
    that holds it is keyed on that object -- **never on an array's identity**,
    which is a stamp that means nothing the moment numpy recycles an allocation.

    ``left``/``right`` are child node indices, ``-1`` at a leaf; ``first``/
    ``count`` index into ``order``, which is the triangle indices sorted so each
    leaf's are contiguous. Flat arrays rather than Python objects because the
    traversal is a hot loop on the frame thread and an object graph would put a
    dereference on every step of it.
    """

    lo: np.ndarray
    hi: np.ndarray
    left: np.ndarray
    right: np.ndarray
    first: np.ndarray
    count: np.ndarray
    order: np.ndarray


#: Leaves below this many triangles are not split further. Small enough that a
#: leaf test is a handful of triangles, large enough that the tree does not
#: become one node per triangle -- at which point the traversal overhead is more
#: than the arithmetic it saves.
BVH_LEAF = 8

#: Below this many triangles the linear sweep wins outright: the vectorised
#: Moller-Trumbore over a few hundred triangles is a handful of numpy calls,
#: where a tree walk is a Python loop per node.
BVH_MIN_TRIS = 512


def build_bvh(positions: np.ndarray, tris: np.ndarray) -> BVH | None:
    """A median-split BVH over ``tris``, or ``None`` when it is not worth one.

    Median split on the widest axis of the centroid bounds: cheap to build (one
    ``argpartition`` per node) and good enough that the traversal is the win.
    A surface-area heuristic would build a better tree and cost more to build
    than the picks it saves, which is the wrong trade for something rebuilt
    whenever the user edits a mesh.
    """
    tris = np.asarray(tris)
    positions = np.asarray(positions, dtype="f8")
    if len(tris) < BVH_MIN_TRIS:
        return None
    corners = positions[tris]  # (T, 3, 3)
    tri_lo = corners.min(axis=1)
    tri_hi = corners.max(axis=1)
    centroid = (tri_lo + tri_hi) * 0.5

    order = np.arange(len(tris), dtype="i8")
    lo: list[np.ndarray] = []
    hi: list[np.ndarray] = []
    left: list[int] = []
    right: list[int] = []
    first: list[int] = []
    count: list[int] = []

    # An explicit stack rather than recursion: a degenerate mesh could otherwise
    # reach Python's recursion limit from inside a mouse move.
    stack = [(0, len(tris))]
    nodes: list[tuple[int, int]] = []
    while stack:
        begin, end = stack.pop()
        node = len(lo)
        span = order[begin:end]
        lo.append(tri_lo[span].min(axis=0))
        hi.append(tri_hi[span].max(axis=0))
        left.append(-1)
        right.append(-1)
        first.append(begin)
        count.append(end - begin)
        nodes.append((begin, end))
        if end - begin <= BVH_LEAF:
            continue
        extent = centroid[span].max(axis=0) - centroid[span].min(axis=0)
        axis = int(np.argmax(extent))
        if extent[axis] <= 0.0:
            continue  # every centroid coincident: splitting would not separate them
        middle = (end - begin) // 2
        keys = centroid[span, axis]
        order[begin:end] = span[np.argpartition(keys, middle)]
        left[node] = -2  # a placeholder; the children's indices are filled below
        stack.append((begin + middle, end))
        stack.append((begin, begin + middle))

    # The stack above pushes children immediately after their parent is
    # appended, and pops depth-first, so the two children of a node marked -2
    # are the next two subtrees in append order. Resolving that here rather than
    # threading indices through the loop keeps the loop readable.
    return _link(
        np.asarray(lo, dtype="f8"),
        np.asarray(hi, dtype="f8"),
        np.asarray(left, dtype="i8"),
        np.asarray(right, dtype="i8"),
        np.asarray(first, dtype="i8"),
        np.asarray(count, dtype="i8"),
        order,
        nodes,
    )


def _link(lo, hi, left, right, first, count, order, nodes) -> BVH:
    """Fill in child indices from the append order the build produced."""
    # A node's children are the two subtrees that follow it and together cover
    # its own [first, first + count) span. Walking backwards, the most recent
    # unclaimed nodes covering the two halves are exactly them.
    pending: dict[tuple[int, int], int] = {}
    for index in range(len(lo) - 1, -1, -1):
        begin, end = nodes[index]
        if left[index] == -2:
            middle = begin + (end - begin) // 2
            left[index] = pending.pop((begin, middle))
            right[index] = pending.pop((middle, end))
            count[index] = 0  # an interior node holds no triangles of its own
        pending[(begin, end)] = index
    return BVH(lo=lo, hi=hi, left=left, right=right, first=first, count=count, order=order)


_BVH_CACHE: weakref.WeakKeyDictionary[Any, tuple[BVH | None]] = weakref.WeakKeyDictionary()


def cached_bvh(mesh: Any, positions: np.ndarray, tris: np.ndarray) -> BVH | None:
    """The BVH for *mesh*, memoised against it. ``None`` for a small mesh.

    **Keyed on the mesh object, which is the revision stamp**, and the key is
    the *only* thing this needs from it -- so the cache lives here rather than
    beside the triangulation cache in ``clay/adjacency``: that package is pinned
    against importing anything outward, and a ``clay -> viewer`` edge to reach a
    tree builder would be the wrong direction entirely.

    A ``Mesh`` is immutable, so the same object is the same geometry and a new
    object is a new edit -- where keying on ``positions``' ``id`` would key on an
    allocation numpy is free to recycle, a bug class that was live in three
    caches in this tree before 2026-08-17.

    That immutability is also what makes this safe under a drag: the drag
    previews into a separate positions array and leaves the object's mesh alone
    until it commits, so picking mid-drag reads this entry rather than rebuilding
    a tree sixty times a second. The commit mints a new mesh and the next pick
    builds once.
    """
    got = _BVH_CACHE.get(mesh)
    if got is None:
        got = (build_bvh(positions, tris),)
        _BVH_CACHE[mesh] = got
    return got[0]


def _bvh_candidates(bvh: BVH, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Triangle indices in every leaf the ray's slab test admits.

    The slab test errs towards yes exactly as :func:`ray_aabb` does, and for the
    same reason: a rejected box the triangle test would have hit is a click that
    does nothing.
    """
    found: list[np.ndarray] = []
    stack = [0]
    # The whole traversal runs under one ``errstate``: an origin exactly on a
    # slab gives ``0 * inf`` and a ray parallel to one gives an infinity, both
    # of which :func:`ray_aabb` already resolves in favour of a hit rather than
    # treating as errors.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / direction
        while stack:
            node = stack.pop()
            t1 = (bvh.lo[node] - origin) * inv
            t2 = (bvh.hi[node] - origin) * inv
            near = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf)
            far = np.nan_to_num(np.maximum(t1, t2), nan=np.inf)
            if not (near.max() <= far.min() and far.min() >= 0.0):
                continue
            if bvh.left[node] < 0:
                begin = int(bvh.first[node])
                found.append(bvh.order[begin : begin + int(bvh.count[node])])
                continue
            stack.append(int(bvh.left[node]))
            stack.append(int(bvh.right[node]))
    if not found:
        return np.zeros(0, dtype="i8")
    return np.concatenate(found)


def ray_triangles(
    origin: np.ndarray,
    direction: np.ndarray,
    positions: np.ndarray,
    tris: np.ndarray,
    bvh: BVH | None = None,
) -> tuple[float, int] | None:
    """-> ``(distance, triangle index)`` of the nearest hit, or None.

    Moller-Trumbore, computed for every triangle at once. ``tris`` is an
    ``(T, 3)`` index array into ``positions``, which is what
    :func:`~warlock.studio.clay.mesh.triangulate` hands back -- so the index
    returned is a *triangle* index, and a caller who wants the face the user
    selected maps it through that function's ``tri_face``.

    **Two-sided.** No back-face rejection: clicking a box you are inside must
    select it, and a blockout mesh is routinely inspected from within. The sign
    of the determinant is therefore only ever used through ``abs``.

    **A hit behind the ray's origin is not a hit**, matching :func:`ray_sphere`.
    A click selects what is in front of the camera.

    **A degenerate or edge-on triangle is skipped rather than divided by.** Both
    give a determinant at zero, and the naive form divides by it: the result is
    an infinity or a NaN, and a NaN in ``t`` silently *wins* an unguarded
    ``argmin`` on some paths and silently loses on others. Here the mask decides
    before any division happens.

    ``bvh`` narrows the sweep to the triangles in the leaves a ray actually
    meets, which is a complexity change rather than a constant factor: over
    *every* triangle the cost is the mesh, and a 200k-triangle import is a
    quarter of a million Moller-Trumbore evaluations per mouse move. **It must
    have been built over these exact positions** -- the tree is geometry, not
    topology -- which is why it is passed in rather than looked up: the caller
    holds the mesh both came from. The result is identical either way; the
    subset is a superset of every hit.
    """
    tris = np.asarray(tris)
    if len(tris) == 0:
        return None
    origin = np.asarray(origin, dtype="f8")
    direction = np.asarray(direction, dtype="f8")
    positions = np.asarray(positions, dtype="f8")

    picked: np.ndarray | None = None
    if bvh is not None:
        picked = _bvh_candidates(bvh, origin, direction)
        if not len(picked):
            return None
        tris = tris[picked]

    v0 = positions[tris[:, 0]]
    edge1 = positions[tris[:, 1]] - v0
    edge2 = positions[tris[:, 2]] - v0

    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    live = np.abs(det) > TRIANGLE_EPS
    if not live.any():
        return None

    # Every division is masked, so nothing that was skipped ever produces a
    # value -- the zeros left behind are then excluded by ``live`` again below.
    inv = np.zeros_like(det)
    np.divide(1.0, det, out=inv, where=live)

    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inv
    t = np.einsum("ij,ij->i", edge2, qvec) * inv

    ok = live & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t >= 0.0)
    if not ok.any():
        return None
    hits = np.flatnonzero(ok)
    nearest = t[hits].min()
    # **The tie-break is the lowest index in the caller's list, stated rather
    # than inherited from the sweep order.** Several triangles meeting at a
    # point -- the pole of a UV sphere is the standard case -- are all hit at
    # exactly the same ``t``, and ``argmin`` then returns whichever came first
    # in whatever order the candidates arrived in. That is the natural triangle
    # order for the full sweep and leaf order for a narrowed one, so without
    # this the tree would quietly select a different *face* at a pole. For the
    # full sweep the rule is what ``argmin`` already did.
    tied = hits[t[hits] == nearest]
    index = int(tied[0]) if picked is None else int(tied[np.argmin(picked[tied])])
    # The returned index names a triangle in the *caller's* list, so a narrowed
    # sweep maps back through the candidate set -- ``tri_face`` is indexed by it.
    return float(t[index]), index if picked is None else int(picked[index])


def ray_aabb(
    origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> bool:
    """Does the ray meet the box at or in front of its origin? The slab test.

    A prefilter, so it errs towards *yes*: rejecting something the triangle test
    would have hit is a click that does nothing, which is far worse than a few
    wasted triangle tests. A ray parallel to a slab divides by zero here quite
    deliberately -- the infinities that come out are the right answer, and the
    ``nan_to_num`` handles the one case that is not (an origin exactly on the
    slab, which gives ``0/0``) by resolving it in favour of a hit.
    """
    origin = np.asarray(origin, dtype="f8")
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.asarray(direction, dtype="f8")
        t1 = (np.asarray(lo, dtype="f8") - origin) * inv
        t2 = (np.asarray(hi, dtype="f8") - origin) * inv
    near = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf)
    far = np.nan_to_num(np.maximum(t1, t2), nan=np.inf)
    return bool(near.max() <= far.min() and far.min() >= 0.0)


def ray_object(
    origin: np.ndarray,
    direction: np.ndarray,
    matrix: np.ndarray,
    positions: np.ndarray,
    tris: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    bvh: BVH | None = None,
) -> tuple[float, int] | None:
    """:func:`ray_triangles` against a placed object, in the object's own space.

    The ray goes through the inverse transform once; the alternative is putting
    every vertex through the forward one on every mouse move, which is the same
    arithmetic multiplied by the vertex count.

    **The object-space direction is deliberately left un-normalised.** Both the
    origin and the direction map linearly, so the parameter ``t`` that comes
    back is the *world* ray's own -- which is what makes two objects comparable.
    Normalising would measure each object in its own scaled units, and the
    nearest-object test would pick whichever happened to be scaled up.

    ``bounds`` is the object-space AABB if the caller already has it (the
    document does, per mesh); it is measured here when not supplied. A
    singular transform -- a zero scale, which the properties panel will accept
    -- returns None rather than raising a ``LinAlgError`` in the middle of a
    mouse move.
    """
    try:
        inverse = np.linalg.inv(np.asarray(matrix, dtype="f8"))
    except np.linalg.LinAlgError:
        return None

    local_origin = (inverse @ np.append(np.asarray(origin, dtype="f8"), 1.0))[:3]
    local_dir = inverse[:3, :3] @ np.asarray(direction, dtype="f8")
    if not np.isfinite(local_origin).all() or not np.isfinite(local_dir).all():
        return None

    positions = np.asarray(positions, dtype="f8")
    if len(positions) == 0:
        return None
    lo, hi = bounds if bounds is not None else (positions.min(axis=0), positions.max(axis=0))
    if not ray_aabb(local_origin, local_dir, lo, hi):
        return None
    return ray_triangles(local_origin, local_dir, positions, tris, bvh)


def screen_scale(camera, point: np.ndarray, pixels: float, viewport_height: int) -> float:
    """The world size that covers ``pixels`` on screen at ``point``.

    What keeps a gizmo the same size however far away the joint is -- the
    alternative is a handle that is either invisible on a building or larger
    than a gem.
    """
    to_point = np.asarray(point, dtype="f8") - camera.position
    depth = abs(float(np.dot(to_point, -camera.view()[2, :3])))
    height = 2.0 * depth * math.tan(math.radians(camera.fov * 0.5))
    return height * pixels / max(viewport_height, 1)


def marker_radius(bounding_radius: float) -> float:
    """Joint markers are a fixed fraction of the model, as in the frontend:
    2.2% of the bounding radius. Constant in *world* space, not on screen, so
    zooming in on a hand actually helps you hit the right knuckle."""
    return bounding_radius * 0.022


def to_world(placement: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Model space -> world, for a point. The placement transform is the
    centre-and-ground the viewer applies on top of the GLB's own."""
    return (placement @ np.append(np.asarray(point, dtype="f8"), 1.0))[:3]


def from_world(placement: np.ndarray, point: np.ndarray) -> np.ndarray:
    return (np.linalg.inv(placement) @ np.append(np.asarray(point, dtype="f8"), 1.0))[:3]


def signed_angle(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    """The rotation about ``axis`` that takes a to b, in radians.

    atan2 rather than acos: the sign is the whole point (a drag has to be able
    to go both ways), and acos loses it.
    """
    axis = np.asarray(axis, dtype="f8")
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    a = a - axis * np.dot(a, axis)
    b = b - axis * np.dot(b, axis)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    a, b = a / na, b / nb
    return math.atan2(float(np.dot(np.cross(a, b), axis)), float(np.dot(a, b)))

