"""Ray tests, analytically.

No GPU id buffer anywhere. Everything the viewer lets you click is a handle at
a known place with a known radius -- a joint marker, a gizmo ring, a gizmo
arrow -- so a closed-form test against a sphere, a torus band or a segment is
both cheaper than a readback and, unlike one, testable without a context.

Build mode adds the one case the analytic handles never covered: clicking the
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

import numpy as np

from . import math3d as m3


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


def ray_triangles(
    origin: np.ndarray,
    direction: np.ndarray,
    positions: np.ndarray,
    tris: np.ndarray,
) -> tuple[float, int] | None:
    """-> ``(distance, triangle index)`` of the nearest hit, or None.

    Moller-Trumbore, computed for every triangle at once. ``tris`` is an
    ``(T, 3)`` index array into ``positions``, which is what
    :func:`~warlock.studio.build.mesh.triangulate` hands back -- so the index
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
    """
    tris = np.asarray(tris)
    if len(tris) == 0:
        return None
    origin = np.asarray(origin, dtype="f8")
    direction = np.asarray(direction, dtype="f8")
    positions = np.asarray(positions, dtype="f8")

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
    index = int(np.flatnonzero(ok)[np.argmin(t[ok])])
    return float(t[index]), index


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
    return ray_triangles(local_origin, local_dir, positions, tris)


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


def world_positions(model, bones: list[str]) -> dict[str, np.ndarray]:
    """Where each named joint currently is, in the model's own space."""
    out: dict[str, np.ndarray] = {}
    for name in bones:
        index = model.by_name.get(name)
        if index is not None:
            out[name] = model.nodes[index].world[:3, 3].copy()
    return out


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


def rotation_between(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """The quaternion for :func:`signed_angle`."""
    return m3.quat_from_axis_angle(axis, signed_angle(a, b, axis))
