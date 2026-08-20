"""The ground grid, as a line list.

three's GridHelper in ten lines: a square of ``divisions`` cells, the two
centre lines in the brighter colour. The span follows the model -- a
power-of-ten that comfortably contains its footprint -- because a fixed 4 m
grid frames a 1 m prop and nothing else.
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

from . import math3d as m3

DIVISIONS = 16
CENTRE_COLOR = 0x2C2F3A
GRID_COLOR = 0x232530


def span_for(width: float, depth: float) -> float:
    """A power-of-ten span that comfortably contains the footprint."""
    extent = max(width, depth) * 2.5
    if extent <= 0 or not math.isfinite(extent):
        return 1.0
    return 10.0 ** math.ceil(math.log10(extent))


def _rgb(value: int) -> tuple[float, float, float]:
    return tuple(((value >> shift) & 0xFF) / 255.0 for shift in (16, 8, 0))


def build(span: float, divisions: int = DIVISIONS) -> tuple[np.ndarray, np.ndarray]:
    """-> (positions (n, 3), colors (n, 3)) for a GL_LINES draw."""
    half = span * 0.5
    step = span / divisions
    centre = divisions // 2
    positions: list[list[float]] = []
    colors: list[tuple[float, float, float]] = []
    for i in range(divisions + 1):
        offset = -half + i * step
        color = _rgb(CENTRE_COLOR if i == centre else GRID_COLOR)
        positions += [[-half, 0.0, offset], [half, 0.0, offset]]
        positions += [[offset, 0.0, -half], [offset, 0.0, half]]
        colors += [color] * 4
    return np.array(positions, dtype="f4"), np.array(colors, dtype="f4")


class Grid:
    """The grid's GPU buffers, rebuilt only when the span changes."""

    def __init__(self, ctx, programs) -> None:
        self.ctx = ctx
        self.program = programs.get("lines")
        self.span = 0.0
        self._vbo = None
        self._vao = None
        self.set_span(4.0)

    def set_span(self, span: float) -> None:
        if span == self.span:
            return
        self.release()
        self.span = span
        positions, colors = build(span)
        data = np.concatenate([positions, colors], axis=1)
        self._vbo = self.ctx.buffer(np.ascontiguousarray(data).tobytes())
        self._vao = self.ctx.vertex_array(
            self.program, [(self._vbo, "3f 3f", "a_position", "a_color")]
        )

    def render(self, view, proj) -> None:
        self.program["u_view"].write(m3.gl_bytes(view))
        self.program["u_proj"].write(m3.gl_bytes(proj))
        self.program["u_exposure"].value = 1.0
        self.program["u_alpha"].value = 1.0
        self._vao.render(mode=moderngl.LINES)

    def release(self) -> None:
        for obj in (self._vao, self._vbo):
            if obj is not None:
                obj.release()
        self._vao = self._vbo = None
        self.span = 0.0
