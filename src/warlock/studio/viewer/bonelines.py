"""Bone connection lines: the skeleton drawn between the joint markers.

On a meshless armature -- the Poser preview -- the markers alone are a cloud of
dots with nothing to say which joint hangs off which; these are the lines that
make it read as a skeleton. Same mechanism as the gizmo rings and the markers:
a coloured vertex array through the one "solid" program, described as
``DrawItem``s so the renderer needs no change at all.

``bone_segments`` is the pure half: which pairs to connect, derived from the
node graph and nothing else, so it answers identically for a meshless armature
and a fully rigged GLB -- both are the same node-graph shape.
"""

from __future__ import annotations

from typing import Any

import moderngl
import numpy as np

from . import math3d as m3
from .markers import ACTIVE, _rgb
from .render import DrawItem

# Quieter than a marker on purpose: the skeleton is context, the joints are
# the controls.
IDLE = 0x55517E


def bone_segments(model: Any, bones: list[str]) -> list[tuple[str, str]]:
    """``(parent bone, bone)`` name pairs to draw lines between.

    A bone whose parent *node* is also a bound bone gets one segment between
    their world positions; the root -- and any bone whose glTF parent is a
    helper node the skin does not own -- gets none.
    """
    bound = set(bones)
    parent_of: dict[int, int] = {}
    for i, node in enumerate(model.nodes):
        for child in node.children:
            parent_of[child] = i
    pairs: list[tuple[str, str]] = []
    for bone in bones:
        index = model.by_name.get(bone)
        if index is None:
            continue
        parent = parent_of.get(index)
        if parent is None:
            continue
        parent_name = model.nodes[parent].name
        if parent_name in bound:
            pairs.append((parent_name, bone))
    return pairs


class BoneLines:
    """The GPU buffer for the connection lines, plus the draw list builder."""

    def __init__(self, ctx: Any, programs: Any) -> None:
        self.ctx = ctx
        self.program = programs.get("solid")
        self._vbo: Any = None
        self._vao: Any = None
        self._capacity = 0

    def draws(
        self,
        positions: dict[str, np.ndarray],
        pairs: list[tuple[str, str]],
        selected: str | None,
        placement: np.ndarray | None = None,
    ) -> list[DrawItem]:
        """At most two DrawItems: every segment muted, then the selected
        bone's *incoming* segment redrawn in the accent over the top.

        Rebuilt from the live handle positions on every call rather than
        cached on a pose-change hook: a gizmo drag moves the handles every
        frame without one, and forty vertices are cheaper than a stale
        skeleton. The active segments sit first in the buffer because a
        ``DrawItem`` counts vertices from zero -- the second, shorter draw
        lands exactly on them.
        """
        placement = m3.identity() if placement is None else placement
        idle: list[tuple[Any, Any]] = []
        active: list[tuple[Any, Any]] = []
        for parent, bone in pairs:
            a, b = positions.get(parent), positions.get(bone)
            if a is None or b is None:
                continue
            (active if bone == selected else idle).append((a, b))
        segments = active + idle
        if not segments:
            return []
        data = np.asarray(
            [point for segment in segments for point in segment], dtype="f4"
        )
        self._upload(data.tobytes())
        items = [
            DrawItem(
                vao=self._vao,
                color=(*_rgb(IDLE), 1.0),
                model=placement,
                mode=moderngl.LINES,
                vertices=len(segments) * 2,
            )
        ]
        if active:
            items.append(
                DrawItem(
                    vao=self._vao,
                    color=(*_rgb(ACTIVE), 1.0),
                    model=placement,
                    mode=moderngl.LINES,
                    vertices=len(active) * 2,
                )
            )
        return items

    def _upload(self, blob: bytes) -> None:
        if self._vbo is None or len(blob) > self._capacity:
            self.release()
            self._capacity = max(len(blob), 24)
            self._vbo = self.ctx.buffer(reserve=self._capacity, dynamic=True)
            self._vao = self.ctx.vertex_array(
                self.program, [(self._vbo, "3f", "a_position")]
            )
        self._vbo.write(blob)

    def release(self) -> None:
        if self._vao is not None:
            self._vao.release()
            self._vao = None
        if self._vbo is not None:
            self._vbo.release()
            self._vbo = None
        self._capacity = 0
