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

    def _stroke_baseline(self: MapDoc, uid: int) -> np.ndarray | None:
        """The pre-stroke contents of ``uid``, if a session is open on it.

        What a mid-drag ``resize`` must record as its *before*: the live array
        is already carrying half a stroke, and a ``ResizeEdit`` that snapshot
        that would put those cells back on undo -- after the stroke's own patch
        had just taken them away.
        """
        if self._stroke is None or int(self._stroke["uid"]) != int(uid):
            return None
        return self._stroke["before"]

    def _reframe_stroke(self: MapDoc, dx: int, dy: int, width: int, height: int) -> None:
        """Move an open session into a resized grid.

        An infinite map grows *while the drag is happening* -- that is what
        makes it infinite -- and :meth:`~._map_geometry.GeometryOps.resize`
        reallocates every tile layer under a session that has already snapshot
        the old array at the old coordinates. Left alone, ``end_stroke`` slices
        ``before`` out of the stale array at post-resize coordinates: the two
        halves of the patch then describe *different* true cells, and Ctrl+Z
        plants stale tiles at unrelated coordinates while leaving some of the
        painted ones unreverted.

        The remap is the same overlap arithmetic ``resize`` applies to the
        layers themselves, applied to the snapshot and to the dirty box, so the
        session comes out the far side describing exactly what it did before.
        ``undo``/``redo`` close a session instead; that is not an option here,
        because the pane only writes ``if doc.stroking`` and a closed session
        would silently drop the rest of the drag.
        """
        if self._stroke is None:
            return
        data = self._stroke["before"]
        grown = gidlib.empty_layer(width, height)
        sx0, sy0 = max(0, -dx), max(0, -dy)
        tx0, ty0 = max(0, dx), max(0, dy)
        span_w = min(data.shape[1] - sx0, width - tx0)
        span_h = min(data.shape[0] - sy0, height - ty0)
        if span_w > 0 and span_h > 0:
            grown[ty0 : ty0 + span_h, tx0 : tx0 + span_w] = data[
                sy0 : sy0 + span_h, sx0 : sx0 + span_w
            ]
        self._stroke["before"] = grown
        box = self._stroke["box"]
        if box is not None:
            x0, y0, x1, y1 = box
            x0, x1 = max(0, min(x0 + dx, width)), max(0, min(x1 + dx, width))
            y0, y1 = max(0, min(y0 + dy, height)), max(0, min(y1 + dy, height))
            self._stroke["box"] = (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

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
