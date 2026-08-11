"""Whole-canvas geometry: flip, rotate, scale, crop, canvas resize.

Every one of these is a canvas-level operation recorded by snapshot rather than
by patch, so they all take the same two-line shape -- commit whatever is
floating, then hand ``_replay`` a closure over ``_map_planes``. The machinery
they lean on (``_replay``, ``_grid_snapshot``, ``_map_planes``) stays in
``document.py``: it is shared with the indexed-colour ops and with anything else
that rewrites every plane at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import transform as tf

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document


class GeometryOps:
    """Canvas-level geometry, mixed into :class:`~.document.Document`."""

    def flip(self: Document, axis: str) -> None:
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.flip(plane, axis)))

    def rotate90(self: Document, quarters: int = 1) -> None:
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.rotate90(plane, quarters)))

    def scale(self: Document, size: tuple[int, int], *, resample: str = "smooth") -> None:
        self.commit_floating()
        self._replay(
            lambda: self._map_planes(lambda plane: tf.scale(plane, size, resample=resample))
        )

    def crop(self: Document, rect: tuple[int, int, int, int]) -> bool:
        box = self.clip(rect)
        if box is None:
            return False
        self.commit_floating()
        self._replay(lambda: self._map_planes(lambda plane: tf.crop(plane, box)))
        return True

    def crop_to_selection(self: Document) -> bool:
        bounds = self.mask.bounds if self.mask is not None else None
        return self.crop(bounds) if bounds else False

    def resize_canvas(
        self: Document,
        size: tuple[int, int],
        offset: tuple[int, int] | None = None,
        *,
        anchor: str = "top-left",
    ) -> None:
        """Grow or crop the canvas, placing the old image by *anchor*.

        An explicit ``offset`` still wins, because it is the general form and
        the anchor is a name for nine of its values -- and because every caller
        that already computed one should keep working unchanged.
        """
        self.commit_floating()
        if offset is None:
            offset = tf.anchor_offset(self.size, size, anchor)
        self._replay(lambda: self._map_planes(lambda plane: tf.resize_canvas(plane, size, offset)))
