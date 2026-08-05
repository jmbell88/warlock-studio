"""Rotate and translate handles: the subset of TransformControls this app used.

Both are constant screen size (three's ``size = 0.6`` against its own sizing),
both hit-test analytically, and neither knows anything about bones. The rotate
gizmo works in the joint's *local* frame -- a ring aligned to the joint's own
axes is what makes "bend the elbow" one drag rather than three -- and the
translate gizmo works in world axes, because a joint correction is a
displacement in the skeleton's space.

Coordinate conversion lives in :mod:`.pose`; these two only ever produce a
quaternion delta or a world-space point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import math3d as m3
from . import picking
from .render import DrawItem

# three's TransformControls colours, and its 0.6 size scalar.
AXIS_COLORS = {"x": 0xFF3653, "y": 0x8ADB00, "z": 0x2C8FFF}
HOVER_COLOR = 0xFFFF00
GIZMO_PIXELS = 90.0
SIZE = 0.6
RING_SEGMENTS = 48
# How close to the ring's line a click must land, as a fraction of its radius.
RING_TOLERANCE = 0.12
ARROW_TOLERANCE = 0.10
# The uniform-scale handle at the gizmo's centre, as a fraction of its size.
CENTRE_RADIUS = 0.14
# A scale factor is clamped here rather than allowed to reach zero or go
# negative. Zero is a transform that cannot be inverted -- picking and the
# gizmo's own placement both stop working -- and negative is a mirror, which
# has to be baked into the mesh (``build.ops.mirror``) because glTF readers
# disagree about whether a negative scale flips the winding.
MIN_SCALE_FACTOR = 1e-4

AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _rgb(value: int) -> tuple[float, float, float]:
    return tuple(((value >> shift) & 0xFF) / 255.0 for shift in (16, 8, 0))


def ring(axis: str, segments: int = RING_SEGMENTS) -> np.ndarray:
    """-> (n, 3) line-strip vertices for a unit circle in the plane of ``axis``."""
    angles = np.linspace(0.0, 2 * math.pi, segments + 1)
    cos, sin = np.cos(angles), np.sin(angles)
    zero = np.zeros_like(cos)
    return np.array(
        {"x": (zero, cos, sin), "y": (cos, zero, sin), "z": (cos, sin, zero)}[axis],
        dtype="f4",
    ).T.copy()


def arrow(axis: str) -> np.ndarray:
    """-> (n, 3) line vertices: a shaft along the axis with a small head.

    Lines rather than a cone: the whole overlay pass is unlit and depth-less,
    and a cone with no shading reads as a flat blob.
    """
    direction = AXES[axis]
    perp = AXES["y"] if axis != "y" else AXES["x"]
    other = np.cross(direction, perp)
    tip = direction
    verts = [np.zeros(3), tip]
    for side in (perp, -perp, other, -other):
        verts += [tip, tip - direction * 0.25 + side * 0.1]
    return np.array(verts, dtype="f4")


def scale_handle(axis: str) -> np.ndarray:
    """-> (n, 3) line vertices: a shaft along ``axis`` ending in a small cube.

    A cube rather than an arrowhead, which is the convention every modelling
    package uses to tell a scale handle from a translate one at a glance; drawn
    as its twelve edges rather than as a solid, because the overlay pass is
    unlit and depth-less and a shaded box reads as a flat blob.
    """
    direction = AXES[axis]
    perp = AXES["y"] if axis != "y" else AXES["x"]
    other = np.cross(direction, perp)
    half = 0.06
    # The box is centred so that its far face lands exactly at unit length, as
    # ``arrow()``'s tip does: the two handles are the same length, and the hit
    # test's reach is one number for both.
    centre = direction * (1.0 - half)
    corners = [
        centre + perp * (a * half) + other * (b * half) + direction * (c * half)
        for a in (-1, 1)
        for b in (-1, 1)
        for c in (-1, 1)
    ]
    verts = [np.zeros(3), centre]
    # The twelve edges of the box: every pair of corners differing in one axis.
    for i, a in enumerate(corners):
        for j, b in enumerate(corners):
            if j > i and np.count_nonzero(~np.isclose(a, b)) == 1:
                verts += [a, b]
    return np.array(verts, dtype="f4")


@dataclass
class Drag:
    """What a in-progress drag needs to remember."""

    axis: str
    start: np.ndarray
    origin: np.ndarray
    normal: np.ndarray
    accumulated: float = 0.0


class Gizmo:
    """Shared plumbing: the buffers, the screen-size scale, the hover state."""

    def __init__(self, ctx: Any, programs: Any) -> None:
        self.ctx = ctx
        self.program = programs.get("solid")
        self._buffers: dict[str, Any] = {}
        self._vaos: dict[str, Any] = {}
        self.hover: str | None = None
        self.drag: Drag | None = None
        self.origin = m3.vec3()
        self.basis = m3.identity()
        self.scale = 1.0

    def _geometry(self, key: str, data: np.ndarray) -> Any:
        vao = self._vaos.get(key)
        if vao is None:
            buffer = self.ctx.buffer(np.ascontiguousarray(data, dtype="f4").tobytes())
            vao = self.ctx.vertex_array(self.program, [(buffer, "3f", "a_position")])
            self._buffers[key] = buffer
            self._vaos[key] = vao
        return vao

    def place(self, origin: np.ndarray, basis: np.ndarray, camera: Any, height: int) -> None:
        """Put the gizmo at a point, in a frame, at a constant screen size."""
        self.origin = np.asarray(origin, dtype="f8").copy()
        self.basis = basis
        self.scale = picking.screen_scale(camera, self.origin, GIZMO_PIXELS, height) * SIZE

    @property
    def dragging(self) -> bool:
        return self.drag is not None

    def end_drag(self) -> None:
        self.drag = None

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        for buffer in self._buffers.values():
            buffer.release()
        self._vaos.clear()
        self._buffers.clear()


class RotateGizmo(Gizmo):
    """Three rings in the target's local frame; a drag is an angle about one."""

    def _axis_world(self, axis: str) -> np.ndarray:
        direction = self.basis[:3, :3] @ AXES[axis]
        return direction / max(float(np.linalg.norm(direction)), 1e-12)

    def hit(self, origin: np.ndarray, direction: np.ndarray) -> str | None:
        """Which ring a ray crosses, nearest first."""
        best: tuple[float, str] | None = None
        for axis in AXES:
            hit = picking.ray_ring(
                origin,
                direction,
                self.origin,
                self._axis_world(axis),
                self.scale,
                self.scale * RING_TOLERANCE,
            )
            if hit is None:
                continue
            distance = float(np.linalg.norm(hit - origin))
            if best is None or distance < best[0]:
                best = (distance, axis)
        return None if best is None else best[1]

    def begin(self, axis: str, origin: np.ndarray, direction: np.ndarray) -> bool:
        normal = self._axis_world(axis)
        hit = picking.ray_plane(origin, direction, self.origin, normal)
        if hit is None:
            return False
        self.drag = Drag(axis=axis, start=hit - self.origin, origin=self.origin, normal=normal)
        return True

    def update(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        """-> the *incremental* local-space quaternion since the last update.

        Incremental rather than absolute so the caller can post-multiply it
        straight onto the bone: an absolute would have to know the rotation the
        drag started from, which is exactly the state a gizmo should not hold.
        """
        if self.drag is None:
            return None
        hit = picking.ray_plane(origin, direction, self.drag.origin, self.drag.normal)
        if hit is None:
            return None
        offset = hit - self.drag.origin
        angle = picking.signed_angle(self.drag.start, offset, self.drag.normal)
        delta = angle - self.drag.accumulated
        self.drag.accumulated = angle
        if abs(delta) < 1e-9:
            return None
        # About the *local* axis: the rings are drawn in the joint's own frame,
        # so a delta in world axes would turn the wrong way for any bone whose
        # rest rotation is not the identity -- i.e. all of them.
        return m3.quat_from_axis_angle(AXES[self.drag.axis], delta)

    def draws(self) -> list[DrawItem]:
        import moderngl

        active = self.drag.axis if self.drag else self.hover
        model = _frame(self.origin, self.basis, self.scale)
        items = []
        for axis in AXES:
            colour = HOVER_COLOR if axis == active else AXIS_COLORS[axis]
            items.append(
                DrawItem(
                    vao=self._geometry(f"ring:{axis}", ring(axis)),
                    color=(*_rgb(colour), 1.0),
                    model=model,
                    mode=moderngl.LINE_STRIP,
                )
            )
        return items


class TranslateGizmo(Gizmo):
    """Three arrows on the world axes; a drag slides along one."""

    def hit(self, origin: np.ndarray, direction: np.ndarray) -> str | None:
        best: tuple[float, str] | None = None
        for axis, unit in AXES.items():
            s, distance = picking.closest_on_axis(origin, direction, self.origin, unit)
            # Only the drawn half of the arrow is grabbable, so an axis
            # pointing away from the camera does not steal clicks from the
            # marker behind it.
            if not (0.0 <= s <= self.scale * 1.15):
                continue
            if distance <= self.scale * ARROW_TOLERANCE and (best is None or distance < best[0]):
                best = (distance, axis)
        return None if best is None else best[1]

    def begin(self, axis: str, origin: np.ndarray, direction: np.ndarray) -> bool:
        s, _distance = picking.closest_on_axis(origin, direction, self.origin, AXES[axis])
        self.drag = Drag(
            axis=axis,
            start=np.array([s, 0.0, 0.0]),
            origin=self.origin.copy(),
            normal=AXES[axis],
        )
        return True

    def update(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        """-> the gizmo's new world position, constrained to the drag's axis."""
        if self.drag is None:
            return None
        s, _distance = picking.closest_on_axis(
            origin, direction, self.drag.origin, self.drag.normal
        )
        moved = self.drag.origin + self.drag.normal * (s - self.drag.start[0])
        self.origin = moved
        return moved

    def draws(self) -> list[DrawItem]:
        import moderngl

        active = self.drag.axis if self.drag else self.hover
        # World axes, not the target's: a joint correction is a displacement in
        # the skeleton's space, and a per-bone frame would make "up" mean
        # something different for every joint.
        model = _frame(self.origin, m3.identity(), self.scale)
        items = []
        for axis in AXES:
            colour = HOVER_COLOR if axis == active else AXIS_COLORS[axis]
            items.append(
                DrawItem(
                    vao=self._geometry(f"arrow:{axis}", arrow(axis)),
                    color=(*_rgb(colour), 1.0),
                    model=model,
                    mode=moderngl.LINES,
                )
            )
        return items


class ScaleGizmo(Gizmo):
    """Three axis handles and a uniform one at the centre.

    Built the same way :class:`TranslateGizmo` is and on the same
    ``closest_on_axis`` solve, because a scale drag along an axis *is* a
    translate drag whose result is read as a ratio rather than as a
    displacement. What it hands back is a per-axis factor **relative to where
    the drag started**, never an absolute scale: the gizmo then needs to
    remember nothing about the object, and the caller multiplies the factor
    onto the scale it recorded when the drag began -- the same shape
    ``ClayDoc.set_transform``'s ``was`` argument already takes.

    Existing gizmos are reused as-is. ``RotateGizmo`` and ``TranslateGizmo``
    are already documented as knowing nothing about bones, and nothing about a
    scale handle changes that.
    """

    def _centre_radius(self) -> float:
        return self.scale * CENTRE_RADIUS

    def hit(self, origin: np.ndarray, direction: np.ndarray) -> str | None:
        # The uniform handle first, and not by distance: all three axis handles
        # *start* at the origin, so a click on the centre is within tolerance
        # of every one of them and would go to whichever the loop reached
        # first. The centre owns its own radius.
        if picking.ray_sphere(origin, direction, self.origin, self._centre_radius()) is not None:
            return "xyz"
        best: tuple[float, str] | None = None
        for axis, unit in AXES.items():
            s, distance = picking.closest_on_axis(origin, direction, self.origin, unit)
            if not (0.0 <= s <= self.scale * 1.15):
                continue
            if distance <= self.scale * ARROW_TOLERANCE and (best is None or distance < best[0]):
                best = (distance, axis)
        return None if best is None else best[1]

    def _reach(self, axis: str, origin: np.ndarray, direction: np.ndarray) -> float:
        """How far along the drag's measuring line the ray currently points.

        For an axis handle that is the parameter along the axis. For the
        uniform handle there is no axis, so it is the distance from the gizmo's
        centre to the ray's closest approach -- camera-free, and it behaves the
        way a user expects: drag away from the centre and the object grows.
        """
        if axis in AXES:
            s, _distance = picking.closest_on_axis(origin, direction, self.origin, AXES[axis])
            return s
        offset = np.asarray(origin, dtype="f8") - self.origin
        along = float(np.dot(offset, direction))
        closest = offset - np.asarray(direction, dtype="f8") * along
        return float(np.linalg.norm(closest))

    def begin(self, axis: str, origin: np.ndarray, direction: np.ndarray) -> bool:
        self.drag = Drag(
            axis=axis,
            start=np.array([self._reach(axis, origin, direction), 0.0, 0.0]),
            origin=self.origin.copy(),
            normal=AXES.get(axis, np.ones(3) / math.sqrt(3.0)),
        )
        return True

    def update(self, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        """-> a ``(3,)`` factor relative to the drag's start, or None.

        None when there is no drag, and also when the drag began *at* the
        gizmo's origin: there is no ratio to take from a start of zero, and the
        infinity a naive divide produces would reach the document as an
        object's scale.
        """
        if self.drag is None:
            return None
        start = float(self.drag.start[0])
        if abs(start) < 1e-9:
            return None
        ratio = max(self._reach(self.drag.axis, origin, direction) / start, MIN_SCALE_FACTOR)
        factors = np.ones(3)
        if self.drag.axis in AXES:
            factors["xyz".index(self.drag.axis)] = ratio
        else:
            factors[:] = ratio
        return factors

    def draws(self) -> list[DrawItem]:
        import moderngl

        active = self.drag.axis if self.drag else self.hover
        # World axes, as the translate gizmo uses: Clay's objects carry a
        # rotation, and a scale typed into the properties panel is applied
        # before it, so a handle drawn in the object's rotated frame would drag
        # along an axis the number it edits does not lie on.
        model = _frame(self.origin, m3.identity(), self.scale)
        items = []
        for axis in AXES:
            colour = HOVER_COLOR if axis == active else AXIS_COLORS[axis]
            items.append(
                DrawItem(
                    vao=self._geometry(f"scale:{axis}", scale_handle(axis)),
                    color=(*_rgb(colour), 1.0),
                    model=model,
                    mode=moderngl.LINES,
                )
            )
        return items


def _frame(origin: np.ndarray, basis: np.ndarray, scale: float) -> np.ndarray:
    """Place-and-size, with any scale already in ``basis`` divided out.

    A bone under a scaled armature would otherwise hand the gizmo its scale as
    well as its rotation, and a handle that is 1.8x on one joint and 0.4x on
    the next is unusable.
    """
    rotation = basis[:3, :3].astype("f8")
    lengths = np.linalg.norm(rotation, axis=0)
    rotation = rotation / np.where(lengths == 0, 1.0, lengths)
    out = m3.identity()
    out[:3, :3] = rotation * scale
    out[:3, 3] = origin
    return out
