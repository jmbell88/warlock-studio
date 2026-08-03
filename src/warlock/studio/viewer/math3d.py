"""Vectors, matrices and quaternions, on numpy.

Deliberately hand-rolled rather than pulled from a library. The whole viewer
turns on two conventions being right, and a dependency that quietly disagrees
about either is a class of bug that only shows up as a limb bending the wrong
way:

* **Matrices are ``M @ v`` with column vectors.** ``M[row][col]``, the
  translation lives in the last *column*. GLSL reads a mat4's memory in
  column-major order, so anything on its way to a uniform goes through
  :func:`gl_bytes`, which transposes -- that transpose is the only place the
  two orders meet.
* **Quaternions are XYZW.** Same order as ``THREE.Quaternion.toArray()`` and
  as every pose file on disk. The Blender worker converts to WXYZ at its own
  door; nothing between here and there reorders.
"""

from __future__ import annotations

import math

import numpy as np

Vec3 = np.ndarray
Mat4 = np.ndarray
Quat = np.ndarray


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return np.array([x, y, z], dtype="f8")


def identity() -> Mat4:
    return np.eye(4, dtype="f8")


def gl_bytes(m: Mat4) -> bytes:
    """A mat4 uniform's payload. The transpose *is* the convention change."""
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


def gl_bytes_many(mats: list[Mat4]) -> bytes:
    """The same, for a mat4 array uniform (the joint palette)."""
    if not mats:
        return b""
    return np.ascontiguousarray(
        np.stack([m.T for m in mats]).astype("f4")
    ).tobytes()


# --- construction -----------------------------------------------------------


def translation(t: Vec3) -> Mat4:
    m = identity()
    m[:3, 3] = t
    return m


def scaling(s: Vec3 | float) -> Mat4:
    m = identity()
    m[0, 0], m[1, 1], m[2, 2] = (s, s, s) if np.isscalar(s) else tuple(s)
    return m


def quat_to_mat4(q: Quat) -> Mat4:
    x, y, z, w = q
    m = identity()
    m[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return m


def compose(t: Vec3, q: Quat, s: Vec3) -> Mat4:
    """A glTF node's local transform: T * R * S, in that order."""
    m = quat_to_mat4(q)
    m[:3, :3] *= np.asarray(s, dtype="f8")  # scales each column
    m[:3, 3] = t
    return m


def decompose(m: Mat4) -> tuple[Vec3, Quat, Vec3]:
    """The inverse of :func:`compose`, for a matrix with no shear.

    Only ever applied to transforms we or a glTF exporter built, so the
    no-shear assumption holds; a negative determinant flips X, matching what
    three.js does with a mirrored node.
    """
    t = m[:3, 3].copy()
    basis = m[:3, :3].astype("f8")
    s = np.linalg.norm(basis, axis=0)
    if np.linalg.det(basis) < 0:
        s[0] = -s[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rot = basis / np.where(s == 0, 1.0, s)
    return t, mat3_to_quat(rot), s


def mat3_to_quat(r: np.ndarray) -> Quat:
    """Shepperd's method: pick the largest diagonal term to divide by, so the
    square root is never taken of something near zero."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype="f8")


# --- quaternions ------------------------------------------------------------


def quat_identity() -> Quat:
    return np.array([0.0, 0.0, 0.0, 1.0])


def quat_mul(a: Quat, b: Quat) -> Quat:
    """``a * b``: apply b first, then a -- the same order as a matrix product."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conjugate(q: Quat) -> Quat:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_normalize(q: Quat) -> Quat:
    n = float(np.linalg.norm(q))
    return quat_identity() if n == 0 else np.asarray(q, dtype="f8") / n


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    axis = np.asarray(axis, dtype="f8")
    n = float(np.linalg.norm(axis))
    if n == 0:
        return quat_identity()
    axis = axis / n
    half = angle * 0.5
    s = math.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)])


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    return (quat_to_mat4(q)[:3, :3] @ np.asarray(v, dtype="f8")).astype("f8")


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0:  # take the short way round
        b, dot = -b, -dot
    if dot > 0.9995:
        return quat_normalize(a + t * (b - a))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(theta)
    return (math.sin((1 - t) * theta) / s) * a + (math.sin(t * theta) / s) * b


# --- camera matrices --------------------------------------------------------


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> Mat4:
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4), dtype="f8")
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> Mat4:
    m = identity()
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    """A *view* matrix: world -> camera."""
    eye = np.asarray(eye, dtype="f8")
    f = np.asarray(target, dtype="f8") - eye
    n = float(np.linalg.norm(f))
    # A degenerate eye==target would otherwise produce NaNs that quietly
    # blank the whole frame; an arbitrary forward is at least a picture.
    f = np.array([0.0, 0.0, -1.0]) if n == 0 else f / n
    s = np.cross(f, np.asarray(up, dtype="f8"))
    ns = float(np.linalg.norm(s))
    if ns == 0:  # looking straight along `up`
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
        ns = float(np.linalg.norm(s))
        if ns == 0:
            s = np.array([1.0, 0.0, 0.0])
            ns = 1.0
    s = s / ns
    u = np.cross(s, f)
    m = identity()
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return m


# --- axis conventions -------------------------------------------------------
#
# glTF/three are Y-up with -Z forward; Blender is Z-up with -Y forward. The
# viewer only ever converts *deltas* -- a joint the user dragged -- never
# absolute positions, because a rig.json's joint positions are already in
# Blender space and re-round-tripping them would accumulate error for nothing.


def gltf_delta_to_blender(d: Vec3) -> Vec3:
    """[x, y, z] -> [x, -z, y]."""
    return np.array([d[0], -d[2], d[1]], dtype="f8")


def blender_delta_to_gltf(d: Vec3) -> Vec3:
    """The inverse: [x, y, z] -> [x, z, -y]."""
    return np.array([d[0], d[2], -d[1]], dtype="f8")
