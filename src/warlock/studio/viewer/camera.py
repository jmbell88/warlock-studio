"""The orbit camera: framing, damping, pan, dolly, turntable.

A port of what the frontend got from OrbitControls, with the numbers kept
literal rather than tidied -- ``0.62/0.47/0.62`` is not a nicer ratio in
disguise, it is the angle the app has always opened a model at, and a
"cleaner" 3/4 view would make every existing thumbnail look wrong.

State is spherical about a target: the pole is +Y, ``theta`` is the azimuth and
``phi`` the polar angle, clamped just short of the poles because the up vector
degenerates exactly at them.
"""

from __future__ import annotations

import math

import numpy as np

from . import math3d as m3

FOV_Y = 45.0
# OrbitControls' defaults, which the frontend never overrode.
DAMPING = 0.05
ROTATE_SPEED = 1.0
ZOOM_SPEED = 1.0
# OrbitControls.autoRotateSpeed is 2.0, defined as "30 seconds per orbit at
# 60fps" -- i.e. 2*pi/30 radians per second.
AUTO_ROTATE_SPEED = 2.0
_POLE_EPSILON = 1e-6


class Camera:
    def __init__(self, aspect: float = 1.0) -> None:
        self.aspect = aspect
        self.fov = FOV_Y
        self.near = 0.01
        self.far = 100.0
        self.target = m3.vec3(0.0, 0.0, 0.0)
        self.distance = 3.0
        self.theta = math.pi / 4
        self.phi = math.pi / 3
        self.min_distance = 0.01
        self.max_distance = 1000.0
        self.auto_rotate = False
        # Damping works on a *goal* the input moves and the camera chases.
        self._goal_theta = self.theta
        self._goal_phi = self.phi
        self._goal_distance = self.distance
        self._goal_target = self.target.copy()

    # -- derived state -----------------------------------------------------

    @property
    def position(self) -> np.ndarray:
        sin_phi = math.sin(self.phi)
        return self.target + self.distance * m3.vec3(
            sin_phi * math.sin(self.theta),
            math.cos(self.phi),
            sin_phi * math.cos(self.theta),
        )

    def view(self) -> np.ndarray:
        return m3.look_at(self.position, self.target, m3.vec3(0, 1, 0))

    def projection(self) -> np.ndarray:
        return m3.perspective(self.fov, max(self.aspect, 1e-6), self.near, self.far)

    # -- framing -----------------------------------------------------------

    def frame(self, lo: np.ndarray, hi: np.ndarray) -> float:
        """Put a bounding box on screen. -> the bounding radius.

        Everything is derived from the box rather than fixed because guidance
        sizes a model anywhere from 1 cm to 100 m: the near/far plane and the
        orbit limits scale with it, or a 100 m building clips through the far
        plane and a 1 cm gem sits inside the near one.
        """
        size = np.asarray(hi, dtype="f8") - np.asarray(lo, dtype="f8")
        radius = max(float(np.linalg.norm(size)) * 0.5, 1e-4)
        self.near = radius / 1000.0
        self.far = radius * 100.0
        distance = radius / math.sin(math.radians(self.fov * 0.5)) * 1.25
        self.set_position(m3.vec3(distance * 0.62, distance * 0.47, distance * 0.62))
        self.set_target(m3.vec3(0.0, size[1] * 0.5, 0.0))
        self.min_distance = radius * 0.5
        self.max_distance = radius * 20.0
        return radius

    def set_target(self, target: np.ndarray) -> None:
        self.target = np.asarray(target, dtype="f8").copy()
        self._goal_target = self.target.copy()

    def set_position(self, position: np.ndarray) -> None:
        """Place the eye, converting to the spherical state that drives it."""
        offset = np.asarray(position, dtype="f8") - self.target
        self.distance = max(float(np.linalg.norm(offset)), 1e-6)
        self.phi = math.acos(max(-1.0, min(1.0, offset[1] / self.distance)))
        self.theta = math.atan2(offset[0], offset[2])
        self._goal_theta, self._goal_phi = self.theta, self.phi
        self._goal_distance = self.distance

    # -- input -------------------------------------------------------------

    def orbit(self, dx: float, dy: float, viewport_height: int) -> None:
        """A drag, in pixels. OrbitControls maps a full height to 2*pi/…"""
        height = max(viewport_height, 1)
        self._goal_theta -= 2 * math.pi * dx / height * ROTATE_SPEED
        self._goal_phi -= 2 * math.pi * dy / height * ROTATE_SPEED
        self._goal_phi = min(math.pi - _POLE_EPSILON, max(_POLE_EPSILON, self._goal_phi))

    def pan(self, dx: float, dy: float, viewport_height: int) -> None:
        """Move the target in the camera's own plane.

        Scaled by distance and the field of view so a drag moves the same
        number of *world* units as it does screen pixels -- panning that
        accelerates as you zoom out is the thing that makes a viewer feel
        broken.
        """
        height = max(viewport_height, 1)
        scale = 2.0 * self.distance * math.tan(math.radians(self.fov * 0.5)) / height
        view = self.view()
        right = view[0, :3]
        up = view[1, :3]
        self._goal_target = self._goal_target - right * (dx * scale) + up * (dy * scale)

    def dolly(self, steps: float) -> None:
        factor = 0.95**ZOOM_SPEED
        self._goal_distance *= factor ** (-steps)

    def update(self, dt: float) -> None:
        """Advance one frame of damping.

        Frame-rate independent: OrbitControls applies a fixed 0.05 lerp per
        *frame*, which at 144 Hz converges nearly three times faster than at
        60. Raising the retention to the frame's share of 1/60 s keeps the feel
        the same on any monitor.
        """
        if self.auto_rotate:
            self._goal_theta -= (2 * math.pi / 60.0) * AUTO_ROTATE_SPEED * dt
        retain = (1.0 - DAMPING) ** max(dt * 60.0, 0.0)
        alpha = 1.0 - retain
        self.theta += (self._goal_theta - self.theta) * alpha
        self.phi += (self._goal_phi - self.phi) * alpha
        self._goal_distance = min(
            self.max_distance, max(self.min_distance, self._goal_distance)
        )
        self.distance += (self._goal_distance - self.distance) * alpha
        self.target = self.target + (self._goal_target - self.target) * alpha

    def settled(self) -> bool:
        """Whether damping has effectively converged.

        The lerp is asymptotic and never *exactly* reaches its goals, so
        "still moving" needs an epsilon (B11/B12). The angular tolerance is
        ~0.06 deg -- a fraction of a pixel at any sane viewport size -- and
        the distance/target tolerances scale with the distance, because a
        millimetre matters on a gem and not on a building. Auto-rotate is
        never settled: its goal moves every frame by definition.
        """
        if self.auto_rotate:
            return False
        scale = max(self.distance, 1e-6) * 1e-3
        return (
            abs(self._goal_theta - self.theta) < 1e-3
            and abs(self._goal_phi - self.phi) < 1e-3
            and abs(self._goal_distance - self.distance) < scale
            and float(np.max(np.abs(self._goal_target - self.target))) < scale
        )

    def copy_from(self, other: Camera) -> None:
        """Mirror another camera exactly. The compare view's whole point is
        that both halves are seen from the identical angle, so the second
        camera copies rather than having controls of its own."""
        self.fov, self.near, self.far = other.fov, other.near, other.far
        self.theta, self.phi, self.distance = other.theta, other.phi, other.distance
        self.target = other.target.copy()


def screen_ray(
    camera: Camera, x: float, y: float, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """-> (origin, unit direction) for a pixel, in world space.

    Analytic rather than an unproject: the projection is a plain perspective
    matrix, so building the ray from the field of view directly is both shorter
    and free of the inverse's conditioning problems at a grazing far plane.
    """
    ndc_x = (2.0 * x / max(width, 1)) - 1.0
    ndc_y = 1.0 - (2.0 * y / max(height, 1))
    tan_half = math.tan(math.radians(camera.fov * 0.5))
    view = camera.view()
    right, up, back = view[0, :3], view[1, :3], view[2, :3]
    forward = -back
    direction = forward + right * (ndc_x * tan_half * camera.aspect) + up * (ndc_y * tan_half)
    direction = direction / np.linalg.norm(direction)
    return camera.position, direction
