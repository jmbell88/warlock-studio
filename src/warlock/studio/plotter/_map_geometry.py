"""Changing the grid itself: the one operation no rectangle describes.

A resize is a **snapshot** step rather than a patch, and that is why it is its
own module rather than a third method in ``_map_paint``: every layer's *shape*
changes at once, so there is no dirty rect to record and nothing the patch
machinery can carry. Objects move with the content, because their coordinates
are absolute pixels and leaving them put would silently detach every trigger
volume from the geometry it was drawn around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import gid as gidlib
from ._map_model import ObjectLayer, TileLayer, _dimension
from .edits import ResizeEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


class GeometryOps:
    """Grid resizing, mixed into :class:`~.tilemap.MapDoc`."""

    def resize(
        self: MapDoc, width: int, height: int, *, offset_x: int = 0, offset_y: int = 0
    ) -> bool:
        """Change the grid, anchoring the old content at ``(offset_x, offset_y)``."""
        width, height = _dimension(width, "width"), _dimension(height, "height")
        dx, dy = int(offset_x), int(offset_y)
        if (width, height) == (self.width, self.height) and (dx, dy) == (0, 0):
            return False

        before = {layer.uid: layer.data for layer in self.tile_layers()}
        after: dict[int, np.ndarray] = {}
        for uid, data in before.items():
            grown = gidlib.empty_layer(width, height)
            # The overlap of the old rectangle, shifted, with the new one.
            sx0, sy0 = max(0, -dx), max(0, -dy)
            tx0, ty0 = max(0, dx), max(0, dy)
            span_w = min(data.shape[1] - sx0, width - tx0)
            span_h = min(data.shape[0] - sy0, height - ty0)
            if span_w > 0 and span_h > 0:
                grown[ty0 : ty0 + span_h, tx0 : tx0 + span_w] = data[
                    sy0 : sy0 + span_h, sx0 : sx0 + span_w
                ]
            after[uid] = grown

        shift_x, shift_y = dx * self.tile_w, dy * self.tile_h
        before_objects: dict[int, list[tuple[float, float]]] = {}
        after_objects: dict[int, list[tuple[float, float]]] = {}
        # ``all_layers`` rather than ``self.layers``: an object layer inside a
        # group is still on the grid being resized, and leaving its objects put
        # is exactly the detachment this whole shift exists to prevent.
        for layer in self.all_layers():
            if not isinstance(layer, ObjectLayer):
                continue
            before_objects[layer.uid] = [(o.x, o.y) for o in layer.objects]
            after_objects[layer.uid] = [(o.x + shift_x, o.y + shift_y) for o in layer.objects]

        self.history.push(
            ResizeEdit(
                before_size=(self.width, self.height),
                after_size=(width, height),
                before=before,
                after=after,
                before_objects=before_objects,
                after_objects=after_objects,
            )
        )
        self._apply_resize((width, height), after, after_objects)
        return True

    def _apply_resize(
        self: MapDoc,
        size: tuple[int, int],
        data: dict[int, np.ndarray],
        objects: dict[int, list[tuple[float, float]]],
    ) -> None:
        self.width, self.height = int(size[0]), int(size[1])
        for uid, array in data.items():
            layer = self.layer(uid)
            if isinstance(layer, TileLayer):
                # A copy, or undo and the document would share one array and
                # the next stamp would write into the history.
                layer.data = array.copy()
        for uid, positions in objects.items():
            layer = self.layer(uid)
            if isinstance(layer, ObjectLayer):
                # ``strict``, like every other zip in this package. The two
                # lists are recorded from one walk of the same layer and
                # replayed in LIFO order, so a length mismatch means an object
                # arrived or left between the record and the replay -- at which
                # point the positions no longer describe these objects and
                # truncating quietly leaves the tail sitting on the old grid,
                # detached from the geometry it was drawn around. That is the
                # thing the resize moves objects to prevent.
                for obj, (x, y) in zip(layer.objects, positions, strict=True):
                    obj.x, obj.y = float(x), float(y)
