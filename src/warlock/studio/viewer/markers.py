"""Joint markers: the clickable spheres a pose is edited through.

One sphere mesh, drawn once per joint with a different model matrix -- which is
also why they are a fixed *world* size rather than children of the bones: an
armature is free to scale, and a handle that shrinks with its bone is a handle
you cannot hit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import math3d as m3
from .render import DrawItem

IDLE = 0x7C6CF0
ACTIVE = 0x4CC38A
SEGMENTS = (12, 8)


def sphere(radius: float = 1.0, segments: tuple[int, int] = SEGMENTS) -> np.ndarray:
    """-> (n, 3) triangle vertices for a UV sphere. Small enough not to index."""
    cols, rows = segments
    grid = np.zeros((rows + 1, cols + 1, 3))
    for r in range(rows + 1):
        phi = math.pi * r / rows
        for c in range(cols + 1):
            theta = 2 * math.pi * c / cols
            grid[r, c] = (
                radius * math.sin(phi) * math.cos(theta),
                radius * math.cos(phi),
                radius * math.sin(phi) * math.sin(theta),
            )
    tris = []
    for r in range(rows):
        for c in range(cols):
            a, b = grid[r, c], grid[r, c + 1]
            d, e = grid[r + 1, c], grid[r + 1, c + 1]
            tris += [a, d, b, b, d, e]
    return np.array(tris, dtype="f4")


class JointMarkers:
    """The GPU buffer for the marker sphere, plus the draw list builder."""

    def __init__(self, ctx: Any, programs: Any) -> None:
        self.ctx = ctx
        self.program = programs.get("solid")
        data = sphere(1.0)
        self._vbo = ctx.buffer(data.tobytes())
        self._vao = ctx.vertex_array(self.program, [(self._vbo, "3f", "a_position")])

    def draws(
        self,
        positions: dict[str, np.ndarray],
        radius: float,
        selected: str | None,
        placement: np.ndarray | None = None,
    ) -> list[DrawItem]:
        """One DrawItem per joint, the selected one in the accent-green."""
        placement = m3.identity() if placement is None else placement
        items: list[DrawItem] = []
        for name, point in positions.items():
            colour = ACTIVE if name == selected else IDLE
            model = placement @ m3.translation(point) @ m3.scaling(radius)
            items.append(
                DrawItem(vao=self._vao, color=(*_rgb(colour), 1.0), model=model)
            )
        return items

    def release(self) -> None:
        self._vao.release()
        self._vbo.release()


def _rgb(value: int) -> tuple[float, float, float]:
    return tuple(((value >> shift) & 0xFF) / 255.0 for shift in (16, 8, 0))
