"""Ray tests, analytically.

No GPU id buffer anywhere. Everything the viewer lets you click is a handle at
a known place with a known radius -- a joint marker, a gizmo ring, a gizmo
arrow -- so a closed-form test against a sphere, a torus band or a segment is
both cheaper than a readback and, unlike one, testable without a context.
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
