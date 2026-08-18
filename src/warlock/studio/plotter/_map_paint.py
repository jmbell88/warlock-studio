"""The two write doors into a tile layer, and the stroke session over them.

**A drag is one gesture and must be one undo step.** Before the session existed,
the canvas called :meth:`PaintOps.write_region` on every frame the button was
down, so a stamp dragged across forty cells pushed forty steps -- forty things
to undo for one movement, and forty entries competing for the history's byte
budget. ``compound`` cannot repair that after the fact: the steps are already
pushed and the stack has no squash.

So this is ``inker.Document``'s stroke session over gids, deliberately the same
three calls in the same order: take a copy at press, write the live array with
no history at all while the button is down, and push one patch over the union of
what moved at release. It being a *second* door is why :meth:`PaintOps.end_stroke`
is idempotent, why the canvas closes any open session on the first frame the
mouse is not down, and why ``undo``/``redo`` commit one first -- a session left
open is a document whose cells are ahead of its head.

The diff is taken here rather than in :mod:`.tools`, so that every path into a
layer -- a stamp, a fill, a rectangle, a paste -- gets the no-op rule for free
rather than each remembering to apply it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..tilegrid import gid as gidlib
from ._map_model import TileLayer
from .edits import TilePatchEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


class PaintOps:
    """Writing gids into a tile layer, mixed into :class:`~.tilemap.MapDoc`."""

    def write_region(self: MapDoc, uid: int, x0: int, y0: int, after: np.ndarray) -> bool:
        """Put a rectangle of gids into a tile layer, undoably.

        Returns whether anything changed.
        """
        layer = self.layer(uid)
        if not isinstance(layer, TileLayer):
            raise KeyError(f"no tile layer {uid}")
        block = np.ascontiguousarray(after, dtype=gidlib.DTYPE)
        if block.ndim != 2:
            raise ValueError("a tile region is two-dimensional")
        h, w = block.shape
        x0, y0 = int(x0), int(y0)
        if x0 < 0 or y0 < 0 or x0 + w > layer.width or y0 + h > layer.height:
            raise ValueError("that region falls outside the layer")
        before = layer.data[y0 : y0 + h, x0 : x0 + w]
        if np.array_equal(before, block):
            return False
        self.history.push(
            TilePatchEdit(layer_uid=int(uid), x0=x0, y0=y0, before=before, after=block)
        )
        self._blit(uid, x0, y0, block)
        return True

    # -- the stroke session ----------------------------------------------------

    @property
    def stroking(self: MapDoc) -> bool:
        return self._stroke is not None

    def begin_stroke(self: MapDoc, uid: int) -> None:
        """Open a session on one tile layer. Re-opening is harmless."""
        layer = self.layer(uid)
        if not isinstance(layer, TileLayer):
            raise KeyError(f"no tile layer {uid}")
        if self._stroke is not None:
            self.end_stroke()
        self._stroke = {
            "uid": int(uid),
            "before": np.array(layer.data, dtype=gidlib.DTYPE),
            "box": None,
        }

    def stroke_write(self: MapDoc, x0: int, y0: int, after: np.ndarray) -> bool:
        """Write into the open session's layer, pushing nothing."""
        if self._stroke is None:
            raise RuntimeError("no stroke is open")
        block = np.ascontiguousarray(after, dtype=gidlib.DTYPE)
        uid = int(self._stroke["uid"])
        layer = self.layer(uid)
        if not isinstance(layer, TileLayer):
            raise KeyError(f"no tile layer {uid}")
        h, w = block.shape
        x0, y0 = int(x0), int(y0)
        if x0 < 0 or y0 < 0 or x0 + w > layer.width or y0 + h > layer.height:
            raise ValueError("that region falls outside the layer")
        if np.array_equal(layer.data[y0 : y0 + h, x0 : x0 + w], block):
            return False
        self._blit(uid, x0, y0, block)
        box = self._stroke["box"]
        self._stroke["box"] = (
            (x0, y0, x0 + w, y0 + h)
            if box is None
            else (min(box[0], x0), min(box[1], y0), max(box[2], x0 + w), max(box[3], y0 + h))
        )
        return True

    def end_stroke(self: MapDoc) -> bool:
        """Close the session and push one step. ``False`` if nothing moved.

        Idempotent on purpose: a release can be missed -- focus loss, a tab
        switch, Esc, a save beginning mid-drag, an undo issued mid-drag -- and
        every one of those recovery paths would otherwise need to know whether a
        stroke was open.
        """
        stroke, self._stroke = self._stroke, None
        if stroke is None or stroke["box"] is None:
            return False
        x0, y0, x1, y1 = stroke["box"]
        uid = int(stroke["uid"])
        layer = self.layer(uid)
        if not isinstance(layer, TileLayer):
            return False
        before = stroke["before"][y0:y1, x0:x1]
        after = layer.data[y0:y1, x0:x1]
        if np.array_equal(before, after):
            return False
        self.history.push(
            TilePatchEdit(
                layer_uid=uid,
                x0=x0,
                y0=y0,
                before=before,
                after=np.ascontiguousarray(after, dtype=gidlib.DTYPE),
            )
        )
        return True

    # -- the hook the edits call back into -------------------------------------

    def _blit(self: MapDoc, uid: int, x0: int, y0: int, block: np.ndarray) -> None:
        layer = self.layer(uid)
        if not isinstance(layer, TileLayer):
            raise KeyError(f"no tile layer {uid}")
        h, w = block.shape
        layer.data[y0 : y0 + h, x0 : x0 + w] = block
