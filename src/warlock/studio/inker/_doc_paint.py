"""Everything that puts colour onto a layer: strokes, fills, shapes, gradients,
filters -- and the eyedropper that reads it back.

``SHAPES`` and ``normalise_rect`` live here rather than in ``document.py``
because ``shape()`` is their only runtime user, and leaving them behind would
make this module import ``document`` at runtime -- the one edge the split is
built to avoid. ``document.py`` re-exports both, so ``document.SHAPES`` and
``inker.normalise_rect`` are unchanged.

Every method below ends at ``_commit_patch``, which stays in ``document.py``:
it is the single place an indexed document is snapped and the single place an
undo step is pushed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from . import brush as brush_mod
from . import composite as cp
from . import dither, filters, tiling
from . import gradient as grad
from ._doc_ranges import masked_apply
from .brush import DEFAULT_SPACING, StrokeState, clamp_brush
from .layers import Layer
from .selection import magic_wand

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import RGBA, Document

#: The shape kinds whose geometry is a **path** -- a run of points, however
#: many -- rather than the two corners a drag produces (Q-c). Named as a group
#: because that is exactly the set :meth:`PaintOps.shape_path` accepts and the
#: set the pane collects by clicking instead of dragging; ``shape()`` takes them
#: too, with the two points it was handed, so the two entry points cannot come
#: to disagree about what a polyline is.
PATH_SHAPES = ("polyline", "polygon", "curve")

SHAPES = ("line", "rect", "ellipse", *PATH_SHAPES)

#: How far apart, in image pixels, the samples of a rasterised curve are. A
#: length rather than a per-segment count: a count that looks smooth on a 32px
#: sprite is a visible chain of chords across a 2000px canvas, and one that is
#: right at 2000px is hundreds of wasted points on the sprite.
CURVE_STEP = 2.0
#: The ceiling on one segment's samples, so a curve drawn corner to corner on a
#: huge canvas cannot cost thousands of points for a smoothness no pixel can
#: show. At 64 the worst chord on a 4096px span is under 32 pixels of a curve
#: that is a *stroke* several pixels wide.
CURVE_MAX_SAMPLES = 64


def normalise_rect(p0: tuple[int, int], p1: tuple[int, int]) -> tuple[int, int, int, int]:
    """Two corners in any order -> (x0, y0, x1, y1) with x0 <= x1."""
    x0, y0 = p0
    x1, y1 = p1
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _knot(a: tuple[float, float], b: tuple[float, float]) -> float:
    """The centripetal knot interval between two control points.

    ``dist ** 0.5`` is the alpha=0.5 parameterisation, and it is the whole
    reason this is centripetal Catmull-Rom rather than the uniform kind. A
    clicked path is unevenly spaced by nature -- three quick clicks in a corner
    and one across the canvas -- and uniform Catmull-Rom answers that with
    overshoots and self-intersecting loops between the close pairs. Centripetal
    is the standard cure and provably makes neither.

    Floored away from zero because the spline divides by knot differences;
    coincident points are collapsed before this is ever reached, so the floor is
    a guard rather than a path anything takes.
    """
    return max(math.dist(a, b) ** 0.5, 1e-6)


def _lerp(
    a: tuple[float, float], b: tuple[float, float], ta: float, tb: float, t: float
) -> tuple[float, float]:
    """The point at ``t`` on the line through ``a`` at ``ta`` and ``b`` at ``tb``."""
    span = tb - ta
    u, v = (tb - t) / span, (t - ta) / span
    return (a[0] * u + b[0] * v, a[1] * u + b[1] * v)


def _curve_point(p0, p1, p2, p3, knots, u: float) -> tuple[float, float]:
    """One point of the Catmull-Rom segment from ``p1`` to ``p2``.

    Barry-Goldman's pyramid rather than the matrix form, because it takes the
    knot values directly and so is the same three lines whatever the
    parameterisation is. At ``u == 0`` every lerp collapses onto ``p1`` and at
    ``u == 1`` onto ``p2``, which is the property the tool is sold on: the curve
    passes through **every point the user clicked**, not near them.

    ``knots`` arrives from :func:`_curve_span` rather than being derived here,
    which is the whole reason it is a parameter: the four knot values are a
    property of the *segment*, so computing them per sample was three square
    roots per point -- 192 of them across a 64-sample segment where three do.
    """
    t0, t1, t2, t3 = knots
    t = t1 + (t2 - t1) * u
    a1 = _lerp(p0, p1, t0, t1, t)
    a2 = _lerp(p1, p2, t1, t2, t)
    a3 = _lerp(p2, p3, t2, t3, t)
    b1 = _lerp(a1, a2, t0, t2, t)
    b2 = _lerp(a2, a3, t1, t3, t)
    return _lerp(b1, b2, t1, t2, t)


def _curve_span(p0, p1, p2, p3, step: float, samples: int) -> list[tuple[float, float]]:
    """The samples of one segment, from just after ``p1`` through ``p2``.

    The knots are computed once here and handed down. The sample count is the
    chord's length in ``step``-sized pieces, floored at two so even a tiny
    segment bends, and capped so one segment cannot cost thousands of points.
    """
    t0 = 0.0
    t1 = t0 + _knot(p0, p1)
    t2 = t1 + _knot(p1, p2)
    t3 = t2 + _knot(p2, p3)
    knots = (t0, t1, t2, t3)
    count = max(2, min(samples, math.ceil(math.dist(p1, p2) / max(step, 1e-6))))
    return [_curve_point(p0, p1, p2, p3, knots, s / count) for s in range(1, count + 1)]


def _mirrored(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """``a`` reflected through ``b`` -- the phantom control point past an end.

    A Catmull-Rom segment needs a neighbour on each side, and the first and last
    points have only one. Reflecting the inner neighbour makes the end segment
    leave along its own chord, which is what a user drawing a curve expects: the
    stroke starts off towards the second point rather than curling away from it.
    """
    return (2.0 * b[0] - a[0], 2.0 * b[1] - a[1])


def curve_points(points: Any) -> list[tuple[float, float]]:
    """The control points a curve actually has: floats, duplicates collapsed.

    Its own function because it decides *how many segments there are*, and a
    caller splitting a curve by segment index has to be counting the same
    points :func:`curve_spans` counted. Consecutive duplicates go because a
    double-click's second press lands on the first and a zero-length knot
    interval is a division by zero.
    """
    pts: list[tuple[float, float]] = []
    for x, y in points:
        p = (float(x), float(y))
        if not pts or p != pts[-1]:
            pts.append(p)
    return pts


def curve_spans(
    points: Any, *, step: float = CURVE_STEP, samples: int = CURVE_MAX_SAMPLES
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    """``(control points, one sample list per segment between them)``.

    :func:`catmull_rom` is this concatenated, and it is split out because a
    *segment* is the unit the preview can cache by: segment ``j`` is a function
    of the four control points ``j-1 .. j+2`` and of nothing else, so appending
    a point to the end of a path leaves every earlier segment's samples
    untouched -- which is what lets ``inker_canvas`` redraw a fifty-vertex curve
    without resampling it (see ``_curve_path``).

    The control points come back **collapsed** by :func:`curve_points` rather
    than as they went in, because that collapse decides how many segments there
    are. Fewer than three of them make no segments at all: two points are a
    straight line and one is a dot.
    """
    pts = curve_points(points)
    if len(pts) < 3:
        return pts, []
    ext = [_mirrored(pts[1], pts[0]), *pts, _mirrored(pts[-2], pts[-1])]
    spans = [
        _curve_span(*ext[index : index + 4], step, samples)
        for index in range(len(pts) - 1)
    ]
    return pts, spans


def catmull_rom(
    points: Any, *, step: float = CURVE_STEP, samples: int = CURVE_MAX_SAMPLES
) -> list[tuple[float, float]]:
    """A smooth path **through** every one of ``points``, as a polyline.

    A curve tool that interpolates is the only honest one for a click sequence:
    a Bezier's handles are points the drawing never passes through, and there is
    nowhere in a click-per-vertex gesture to put them. So the clicked points are
    the curve's, and what is returned is the same chain of straight segments the
    rasteriser and the on-canvas preview both draw -- one function, so the
    preview cannot promise a curve the commit does not lay down.

    Fewer than three points have no curvature to find and come back as they went
    in: two points are a straight line and one is a dot. Consecutive duplicates
    are collapsed first, because a double-click's second press lands on the
    first and a zero-length knot interval is a division by zero.
    """
    pts, spans = curve_spans(points, step=step, samples=samples)
    if not spans:
        return pts
    out = [pts[0]]
    for span in spans:
        out.extend(span)
    return out


def _translated(pixels: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """A canvas-sized plane shifted by (dx, dy), cropped, zero-filled behind.

    Not ``np.roll``: a layer is canvas-sized and its edge *is* the canvas edge,
    so what leaves one side is gone rather than arriving on the other. Rolling
    would put a sprite's head on the far side of the frame the moment somebody
    nudged it past the border, which reads as corruption.
    """
    out = np.zeros_like(pixels)
    height, width = pixels.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = pixels[sy0:sy1, sx0:sx1]
    return out


def _content_box(pixels: np.ndarray) -> tuple[int, int, int, int] | None:
    """The half-open bbox of every pixel that is not all zeros, or None.

    Any channel, not alpha alone -- and that is the difference between an undo
    that restores the layer and one that nearly does. The eraser cuts alpha and
    leaves RGB where it was, so an erased region is ``(r, g, b, 0)``: invisible,
    outside any alpha-only box, and *zeroed* by the translate below. A patch
    measured on alpha would not record that, and the colours would be gone with
    no step to bring them back -- silently, because nothing on screen changed.
    """
    rows = np.flatnonzero(pixels.any(axis=(1, 2)))
    cols = np.flatnonzero(pixels.any(axis=(0, 2)))
    if rows.size == 0:
        return None
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def _union(
    a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    if a is None or b is None:
        return a or b
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class PaintOps:
    """Writes into a layer's pixels, mixed into :class:`~.document.Document`."""

    def eyedrop(
        self: Document, xy: tuple[int, int], *, layer_only: bool = False
    ) -> RGBA | None:
        """The colour at a point: the composite by default, one layer on ask.

        The composite is the default because the colour a user is pointing at
        is the one they can see, and that is what they mean nine times in ten.
        ``layer_only`` is the tenth: picking up a line colour off a lineart
        layer that has a colour layer under it reads the *blend* otherwise,
        which is a colour that exists nowhere in the document.

        It reads the layer's own stored pixels, so it is unaffected by the
        layer's opacity, its blend mode and whether it is even visible -- all
        three are properties of how the layer is shown rather than of what was
        painted into it.
        """
        if not self.in_bounds(xy):
            return None
        source = self.stack.active.pixels if layer_only else self._composite
        r, g, b, a = source[int(xy[1]), int(xy[0])]
        return (int(r), int(g), int(b), int(a))

    # -- writing to a layer -------------------------------------------------

    def _weights(
        self: Document, rect: tuple[int, int, int, int], weight: np.ndarray
    ) -> np.ndarray:
        """Clip a write to the selection. The same multiply as a brush stamp --
        one rule, so a feathered selection softens every tool identically."""
        if self.mask is None:
            return weight
        x0, y0, x1, y1 = rect
        return weight * (self.mask.mask[y0:y1, x0:x1].astype(np.float32) / 255.0)

    def write_colour(
        self: Document, rect: tuple[int, int, int, int], colour: RGBA, weight: np.ndarray
    ) -> bool:
        """Composite a flat colour into a region of the active layer.

        The door the fill and the three shapes come through as well, which is
        why the content lock is checked here rather than three times over.
        """
        box = self.clip(rect)
        if box is None:
            return False
        if self.write_locked():
            return False
        self._ensure_active_cel()
        layer = self.stack.active
        x0, y0, x1, y1 = box
        before = layer.pixels[y0:y1, x0:x1].copy()
        out = cp.paint_colour(
            before.astype(np.float32), colour, self._weights(box, weight)
        )
        if layer.alpha_lock:
            # The lock, in one line and after the formula rather than inside
            # it: "preserve transparency" is exactly *the alpha does not
            # change*, so restoring the channel is the definition rather than
            # an approximation of it. Colour written where alpha is zero is
            # invisible, which is what makes this enough on its own.
            out[..., 3] = before[..., 3]
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(out)
        self._commit_patch(layer, box, before)
        return True

    # -- filters ------------------------------------------------------------

    def begin_filter(self: Document) -> tuple[int, int, int, int] | None:
        """Open a filter session over the selection, or the whole layer.

        Returns the rect it will write, or None when there is nothing to
        filter -- which includes a content-locked layer, refused here at the
        top so no preview is ever shown for a filter that cannot be applied.
        Nothing is pushed and nothing is changed: this only takes the
        copy that :meth:`preview_filter` recomputes from and
        :meth:`cancel_filter` restores.
        """
        self.end_filter()
        if self.write_locked():
            return None
        # A tilemap cel's pixels are re-derived from its tile references, so
        # this write would be silently discarded in manual mode and diced
        # into the shared tileset in auto/stack mode -- corrupting every
        # other placement of those tiles. Refused by name, like lift and cut.
        self._refuse_tilemap_layer(self.stack.active.uid, "filtering")
        width, height = self.size
        bounds = self.mask.bounds if self.mask is not None else None
        box = self.clip(bounds or (0, 0, width, height))
        if box is None:
            return None
        self._ensure_active_cel()
        x0, y0, x1, y1 = box
        layer = self.stack.active
        self._filter = (box, layer.pixels[y0:y1, x0:x1].copy(), layer.uid)
        self._filter_memo = None
        return box

    def _filter_layer(self: Document) -> Layer | None:
        """The layer an open filter session belongs to, or None if it is gone.

        ``layer_by_uid`` and not ``stack.by_uid``, for exactly the reason
        ``PatchEdit._put`` gives: on an animated document the cel being filtered
        may be on a frame the playhead has since moved off, and it is still the
        cel these pixels came from and the one they must go back to. The frame
        therefore needs no recording of its own -- a cel's uid names it on every
        frame at once, and that is the whole point of addressing by uid.

        Two answers are None rather than a layer, and both mean "cancel this
        cleanly instead of writing somewhere". A uid nothing carries any more is
        a layer (or a whole track) deleted while the popup was up. A uid that
        now resolves to a *placeholder* is an autovivified cel undone while the
        popup was up: placeholder uids are stable per slot, so the empty slot
        takes the same number back -- and its plane is the shared read-only one,
        which raises on the first write.
        """
        if self._filter is None:
            return None
        try:
            layer = self.layer_by_uid(self._filter[2])
        except KeyError:
            return None
        if self.anim is not None and self.anim.is_placeholder(layer):
            return None
        return layer

    def _abandon_filter(self: Document) -> bool:
        """Drop a session whose layer went, restoring nothing and pushing
        nothing. There is no target left to put the pixels back into, and the
        cel this session brought into existence has to go with it."""
        self._filter = None
        self._filter_memo = None
        self._discard_pending_cel()
        return False

    def preview_filter(self: Document, name: str, **params: Any) -> bool:
        """Show the filter without recording it. Cheap to call every frame.

        The selection is honoured as a *weight*, not as a rectangle: a
        feathered edge fades the filter in, which is the same rule every other
        write in this class follows and the reason feathering means one thing
        across the whole app.

        **The filter itself is memoised for the life of the session; the write
        below it is not.** ``inker_bridge`` calls this on every frame the popup
        is up, deliberately, because the combo can switch filters -- but
        ``before`` is the snapshot :meth:`begin_filter` took and never changes,
        so ``apply_named`` is a pure function of ``(name, params)`` within one
        session and recomputing it sixty times a second is the whole of what
        made a 2048 square blur unusable (measured at 1.1 s per call, i.e. per
        frame). Only the expensive half is cached: the blend, the alpha lock
        and the invalidate below still run every frame, so switching filters
        still repaints exactly as before.
        """
        if self._filter is None:
            return False
        layer = self._filter_layer()
        if layer is None:
            return self._abandon_filter()
        box, before, _uid = self._filter
        x0, y0, x1, y1 = box
        key = (name, tuple(sorted(params.items())))
        if self._filter_memo is not None and self._filter_memo[0] == key:
            filtered = self._filter_memo[1]
        else:
            filtered = filters.apply_named(name, before, **params)
            self._filter_memo = (key, filtered)
        # ``masked_apply`` rather than the blend spelled out here: the range
        # filter (``_doc_ranges.filter_range``) has to fade a filter into a
        # selection by exactly the same rule, and two copies of it is how
        # "feathering" comes to mean two things in one editor.
        layer.pixels[y0:y1, x0:x1] = masked_apply(
            before,
            filtered,
            None if self.mask is None else self.mask.mask[y0:y1, x0:x1],
            alpha_lock=layer.alpha_lock,
        )
        self.invalidate(box, layer_uid=layer.uid)
        return True

    def commit_filter(self: Document) -> bool:
        """Turn whatever is previewed into exactly one undo step."""
        if self._filter is None:
            return False
        layer = self._filter_layer()
        if layer is None:
            return self._abandon_filter()
        box, before, _uid = self._filter
        self._filter = None
        self._filter_memo = None
        # ``_commit_patch`` compares before against after and pushes nothing
        # when they match, which is what makes opening a filter, moving nothing
        # and pressing Apply a no-op rather than a step that dirties the file.
        self._commit_patch(layer, box, before)
        return True

    def cancel_filter(self: Document) -> bool:
        """Put the pixels back. Nothing was ever pushed, so nothing is undone."""
        if self._filter is None:
            return False
        layer = self._filter_layer()
        if layer is None:
            return self._abandon_filter()
        box, before, _uid = self._filter
        self._filter = None
        self._filter_memo = None
        x0, y0, x1, y1 = box
        layer.pixels[y0:y1, x0:x1] = before
        self._discard_pending_cel()
        self.invalidate(box, layer_uid=layer.uid)
        return True

    def end_filter(self: Document) -> None:
        """Abandon a session left open by a pane that went away.

        Cancels rather than commits: an unanswered question is not a yes, and
        the pixels on screen are a preview the user never approved.
        """
        self.cancel_filter()

    # -- moving a layer's pixels --------------------------------------------
    #
    # The move tool's third arm: with no floating buffer and no selection, a
    # drag translates what is *on* the layer. Translate-with-crop rather than
    # ``np.roll``: a layer is canvas-sized by invariant and its edges are the
    # canvas's edges, so pixels pushed off one side are gone rather than
    # arriving on the other. (A tiled-wrap variant of exactly this would be one
    # ``tiling.pieces`` call, and it is deliberately unwired -- moving a layer
    # is not a paint stroke, and "the drawing wrapped round while I nudged it"
    # is not something a user asks for by pressing an arrow key.) A cel-offset
    # model, where the layer keeps its pixels and grows an origin, was
    # considered and rejected: it is a second coordinate space through every op
    # in the package, which is the trade ``layers.py`` already refused.
    #
    # The session mirrors the filter session exactly -- begin snapshots, preview
    # re-renders *from the snapshot* so a drag cannot compound, commit pushes
    # one patch, cancel puts the pixels back and pushes nothing -- with one
    # deliberate difference at the abandon door: see :meth:`end_layer_move`.

    def begin_layer_move(self: Document) -> bool:
        """Open a move session on the active layer. -> whether it opened.

        Refused on a content-locked layer (or one inside a locked group),
        through ``write_locked`` -- the same door every other write goes
        through.
        """
        self.end_layer_move()
        # The lock is read *before* the float is committed: a refusal must
        # leave the document exactly as it found it, and committing first would
        # land a floating buffer as the side effect of a move that never
        # happened.
        if self.write_locked():
            return False
        # A tilemap cel's pixels are re-derived from its tile references, so
        # this write would be silently discarded in manual mode and diced
        # into the shared tileset in auto/stack mode -- corrupting every
        # other placement of those tiles. Refused by name, like lift and cut.
        self._refuse_tilemap_layer(self.stack.active.uid, "moving")
        self.commit_floating()
        self._ensure_active_cel()
        layer = self.stack.active
        self._move = (layer.pixels.copy(), layer.uid)
        return True

    def _move_layer(self: Document) -> Layer | None:
        """The layer an open move session belongs to, or None if it is gone.

        ``_filter_layer``'s rule and for its reasons: a session lives across
        frames, a cel is addressed by uid on every frame at once, and a uid that
        now resolves to a placeholder is an autovivified cel undone underneath
        us -- whose plane is the shared read-only one.
        """
        if self._move is None:
            return None
        try:
            layer = self.layer_by_uid(self._move[1])
        except KeyError:
            return None
        if self.anim is not None and self.anim.is_placeholder(layer):
            return None
        return layer

    def _abandon_move(self: Document) -> bool:
        self._move = None
        self._discard_pending_cel()
        return False

    def preview_layer_move(self: Document, dx: int, dy: int) -> bool:
        """Show the layer at a **total** offset from where the session opened.

        Total, not incremental: re-rendering from the snapshot every frame is
        what stops a slow drag and a fast one producing different results, and
        it is the same anti-compounding rule ``preview_filter`` follows.
        """
        if self._move is None:
            return False
        layer = self._move_layer()
        if layer is None:
            return self._abandon_move()
        source, _uid = self._move
        # In place, so a cel linked across several frames stays *one object*
        # and the move shows on every frame it appears in. Rebinding
        # ``layer.pixels`` would silently break every link in the row.
        layer.pixels[:] = _translated(source, int(dx), int(dy))
        width, height = self.size
        self.invalidate((0, 0, width, height), layer_uid=layer.uid)
        return True

    def commit_layer_move(self: Document) -> bool:
        """Turn whatever is previewed into exactly one undo step.

        The patch covers the union of where the content *was* and where it
        *is*, rather than the whole canvas: a 16-pixel nudge of a sprite on a
        2048 square would otherwise cost 32 MiB of history for a change that
        touches a few thousand pixels. See :func:`_content_box` for why "the
        content" is not "the non-transparent pixels".
        """
        if self._move is None:
            return False
        layer = self._move_layer()
        if layer is None:
            return self._abandon_move()
        source, _uid = self._move
        self._move = None
        box = _union(_content_box(source), _content_box(layer.pixels))
        if box is None:
            # Both empty: an empty layer was moved, which changed nothing.
            self._discard_pending_cel()
            return False
        x0, y0, x1, y1 = box
        self._commit_patch(layer, box, source[y0:y1, x0:x1])
        return True

    def cancel_layer_move(self: Document) -> bool:
        """Put the pixels back. Nothing was pushed, so nothing is undone.

        The user-initiated half of the pair, and the only one that restores:
        reachable from Escape and from the canvas, both of which act on a
        session that is still live. Every *automatic* path -- anything that
        finds a session somebody else left open -- goes through
        :meth:`end_layer_move` instead, because restoring from a snapshot of
        unknown age is how an unrelated edit gets wiped.
        """
        if self._move is None:
            return False
        layer = self._move_layer()
        if layer is None:
            return self._abandon_move()
        source, _uid = self._move
        self._move = None
        layer.pixels[:] = source
        self._discard_pending_cel()
        width, height = self.size
        self.invalidate((0, 0, width, height), layer_uid=layer.uid)
        return True

    def end_layer_move(self: Document) -> bool:
        """Close a session left open by a gesture that went away.

        ``end_filter``'s role, with the opposite answer, and the difference is
        the point. A filter session is a *question* -- a popup with an Apply
        button the user never pressed -- so abandoning it cancels: an unanswered
        question is not a yes. A move session is a *drag*: the pixels on screen
        are where the user put them, and there is no unpressed button.

        So this commits, and committing is also the only safe answer. Cancelling
        restores ``layer.pixels`` wholesale from the snapshot taken at
        ``begin_layer_move``, which is correct while the session is live and
        catastrophic once it is stale -- anything painted in between is
        overwritten by a snapshot older than it, while those strokes' own
        ``PatchEdit``s stay in history describing pixels that are no longer
        there. Committing can never lose a pixel: it records what actually
        happened as one step and leaves the layer exactly as it looks.
        """
        return self.commit_layer_move()

    # -- palette conversion -------------------------------------------------
    #
    # The filter session's mechanism, generalised one dimension: a filter writes
    # one rectangle of one layer, and a conversion rewrites every plane of the
    # document. Everything else is deliberately the same -- the snapshot is
    # taken once and every preview recomputes from it (never from the last
    # preview, which would compound), each plane is addressed by uid, and
    # committing is the ordinary one-undo op rather than anything the session
    # knows how to push.
    #
    # **The preview covers the current frame only, and the commit covers the
    # document.** Previewing forty frames would mean converting forty planes per
    # parameter change to show one, and the frames the user cannot see are the
    # ones they are not judging. The popup says so.

    def begin_convert(self: Document) -> bool:
        """Open a conversion preview over the current frame's real layers.

        Nothing is pushed and nothing is changed: this only takes the copies
        that :meth:`preview_convert` recomputes from and :meth:`cancel_convert`
        restores.

        **Empty cels are skipped rather than autovivified**, which is the one
        place this deliberately parts company with ``begin_filter``. A filter is
        a write to the layer you have selected, so bringing that cel into
        existence is what the user asked for. A conversion is a document-wide
        mode change that nobody would expect to *populate* the timeline -- and
        the cels it would create are exactly the empty ones, where a conversion
        has nothing to do anyway.
        """
        self.end_convert()
        anim = self.anim
        planes = [
            layer
            for layer in self.stack
            if anim is None or not anim.is_placeholder(layer)
        ]
        if not planes:
            return False
        self._convert = [(layer.uid, layer.pixels.copy()) for layer in planes]
        self._convert_memo = None
        return True

    def _convert_target(self: Document, uid: int, before: np.ndarray) -> Layer | None:
        """The layer one recorded plane belongs to, or None to skip it.

        Three answers are None, and all three mean "this plane is not ours to
        write any more" rather than "cancel the session": the uid names nothing
        (a layer or a whole track deleted under the popup), it names a
        *placeholder* (a cel undone under the popup -- placeholder uids are
        stable per slot, and the plane behind one is the shared read-only
        blank), or the plane has been resized under the popup and the pixels no
        longer describe it.

        Skipping rather than abandoning is the difference from ``_filter``,
        which has exactly one layer and nothing left to do when it goes. Here
        the other planes are still previewing and still have to be put back.
        """
        try:
            layer = self.layer_by_uid(uid)
        except KeyError:
            return None
        if self.anim is not None and self.anim.is_placeholder(layer):
            return None
        if layer.pixels.shape != before.shape:
            return None
        return layer

    def preview_convert(
        self: Document, colours: Any, method: str = "nearest"
    ) -> bool:
        """Show the conversion without recording it. Safe to call every frame.

        ``inker_bridge`` calls this on every frame the popup is up because the
        controls above it can change either argument, and both halves are
        memoised on ``(table, method)``: an unchanged key does nothing at all.

        **That is one step further than ``preview_filter``, which recomputes its
        write every frame, and the difference is measured rather than stylistic.**
        A filter's write is one rectangle of one layer plus a matching
        ``invalidate``; this one is every plane of the document plus
        ``invalidate_all``, whose own docstring puts it at a fraction of a
        second on a 2048-square ten-layer document and says in as many words
        that it is affordable *because* it happens on a click and never on a
        frame. Re-laying the same bytes sixty times a second to defend against a
        write nothing can make -- the popup owns the frame, and anything that
        does reach the planes cancels the session through ``_convert_target`` --
        would spend exactly that budget for nothing.

        **Grouped's table is document-wide even though only the frame is shown.**
        That is not an approximation to be tidied up later: during a preview
        ``_palette_planes`` yields the current frame's *snapshots* plus the other
        frames' true planes, which is byte-for-byte what ``commit_convert`` reads
        after its restore -- so the preview on the visible frame is exactly the
        commit on the visible frame. A table built from the visible frame alone
        would look right and then move when committed.
        """
        if self._convert is None:
            return False
        wanted = [tuple(c) for c in colours]
        if not wanted or method not in dither.METHODS:
            return False
        key = (tuple(wanted), method)
        if self._convert_memo is not None and self._convert_memo[0] == key:
            return True  # already on screen, and nothing about it has moved
        table = (
            dither.grouped_table(self._palette_planes(), wanted)
            if method == "grouped"
            else None
        )
        self._convert_memo = (
            key,
            [
                dither.convert(before, wanted, method, table=table)
                for _uid, before in self._convert
            ],
        )
        for (uid, before), converted in zip(
            self._convert, self._convert_memo[1], strict=True
        ):
            layer = self._convert_target(uid, before)
            if layer is not None:
                layer.pixels[:, :] = converted
        self.invalidate_all()
        return True

    def commit_convert(self: Document, colours: Any, method: str = "nearest") -> bool:
        """Put the preview back, then convert for real as one undo step.

        The restore is not wasted work and not belt-and-braces: the preview has
        already written the converted pixels onto the current frame, so without
        it ``convert_to_palette``'s snapshot would record *those* as the state
        to undo to -- and one Ctrl+Z would land on a document that was never
        the user's.
        """
        if self._convert is None:
            return False
        self._restore_convert()
        return self.convert_to_palette(colours, method)

    def cancel_convert(self: Document) -> bool:
        """Put the pixels back. Nothing was ever pushed, so nothing is undone."""
        if self._convert is None:
            return False
        self._restore_convert()
        return True

    def _restore_convert(self: Document) -> None:
        """Undo the preview and close the session, writing no history."""
        recorded, self._convert = self._convert, None
        self._convert_memo = None
        for uid, before in recorded or ():
            layer = self._convert_target(uid, before)
            if layer is not None:
                layer.pixels[:, :] = before
        self.invalidate_all()

    def end_convert(self: Document) -> None:
        """Abandon a session left open by a pane that went away. Cancels rather
        than commits, for :meth:`end_filter`'s reason."""
        self.cancel_convert()

    # -- strokes ------------------------------------------------------------

    def begin_stroke(
        self: Document,
        point: tuple[float, float],
        colour: RGBA,
        *,
        size: int = 8,
        hardness: float = 0.8,
        opacity: float = 1.0,
        spacing: float | None = None,
        mode: str = "paint",
        strength: float = 0.5,
        nib: str = "soft",
        angle: float = 0.0,
        pixel_perfect: bool = False,
        symmetry: str = "none",
        axis: tuple[float, float] | None = None,
        radial: int = brush_mod.DEFAULT_RADIAL,
        stabilise: float = 0.0,
        speed_taper: float = 0.0,
        wrap: str | tuple[bool, bool] = "off",
        scatter: float = 0.0,
        seed: int = 0,
        ramp: Any = (),
        shade_dir: int = 1,
        stamp: Any = None,
        stamp_align: str = "free",
        lock_alpha: bool = False,
    ) -> bool:
        """Open a stroke on the active layer. False when the layer is locked.

        ``wrap`` is one of :data:`.tiling.TILED_AXES` and is passed in by the
        canvas from the tab's own view state -- tiling is never read off the
        document, because it is a property of how a tab is being looked at and
        must not travel in a file. ``scatter``/``seed`` are the spray tool's
        emission; both default to the values that make this the stroke the
        editor has always opened.

        ``stamp`` is a :class:`.brush.Stamp` -- an image used as the brush tip
        instead of a generated disc -- and ``stamp_align`` is one of
        :data:`.brush.STAMP_ALIGN`. Both arrive per stroke for ``wrap``'s
        reason: a captured brush is a thing the *app* is holding, and nothing
        about it belongs in a document or in a file. A stamp handed to a mode
        that decides its own colours (blur, smudge, shade) is dropped by
        ``StrokeState`` rather than refused here -- it is a tip that stays
        loaded while the user tries the smudge tool.

        ``ramp``/``shade_dir`` are the shading ink's, and they arrive the same
        way and for the same reason: the ramp is a *selection in the palette
        panel*, which is UI state the engine must not read. Passing it per
        stroke also means no ramp is persisted anywhere -- the document keeps
        the palette, and which run of it was being shaded along is a thing about
        the gesture. See :meth:`.brush.StrokeState._shade`.

        The refusal comes *before* ``_ensure_active_cel``, so a brush-down on a
        locked track of an animated document does not autovivify a cel that
        nothing may then write to.
        """
        self.end_stroke()
        if self.write_locked():
            return False
        self._ensure_active_cel()
        layer = self.stack.active
        self._stroke = StrokeState(
            layer_uid=layer.uid,
            size=self.size,
            before=layer.pixels.copy(),
            colour=tuple(colour),
            diameter=clamp_brush(size),
            hardness=hardness,
            opacity=opacity,
            spacing=DEFAULT_SPACING if spacing is None else spacing,
            mode=mode,
            strength=strength,
            nib=nib,
            angle=angle,
            pixel_perfect=pixel_perfect,
            symmetry=symmetry,
            axis=axis,
            radial=radial,
            stabilise=stabilise,
            speed_taper=speed_taper,
            clip=self.mask,
            # The layer's flag **or** the ink's (6.1). Or-ed rather than
            # overridden either way round: the layer lock is a property of the
            # layer that outlives every tool, and the Lock Alpha ink is "for
            # this stroke" -- neither may quietly switch the other off.
            alpha_lock=layer.alpha_lock or bool(lock_alpha),
            wrap_axes=tiling.axes_of(wrap),
            scatter=scatter,
            seed=seed,
            ramp=tuple(tuple(int(c) for c in colour) for colour in ramp),
            shade_dir=shade_dir,
            stamp=stamp,
            stamp_align=stamp_align,
        )
        self._stroke.begin(point, layer.pixels)
        self._touch_stroke()
        return True

    def _stroke_layer(self: Document) -> Layer | None:
        """The layer an open stroke is painting into, or None if it is gone.

        The stroke session's answer to ``_filter_layer``, and it exists for the
        same reason: a session spans many calls, and between two of them the
        layer it named can be deleted -- by a shortcut, a menu op, or an undo
        arriving while the button is still down. ``stack.by_uid`` raises
        ``KeyError`` on a uid nothing carries any more, and this used to be
        called unguarded from all three of ``stroke_to``, ``spray_at`` and
        ``end_stroke``. The last one is what made it worth fixing rather than
        noting: with ``end_stroke`` raising, the stroke could never be closed,
        so every later call raised too and the document was left permanently
        mid-stroke.
        """
        stroke = self._stroke
        if stroke is None:
            return None
        try:
            return self.stack.by_uid(stroke.layer_uid)
        except KeyError:
            return None

    def _abandon_stroke(self: Document) -> bool:
        """Drop a stroke whose layer went. Nothing to put back, nothing to push."""
        self._stroke = None
        self._discard_pending_cel()
        return False

    def stroke_to(self: Document, point: tuple[float, float]) -> None:
        if self._stroke is None:
            return
        layer = self._stroke_layer()
        if layer is None:
            self._abandon_stroke()
            return
        self._stroke.to(point, layer.pixels)
        self._touch_stroke()

    def spray_at(self: Document, point: tuple[float, float], count: int) -> None:
        """Scatter ``count`` dabs about a point, inside the open stroke.

        The spray tool's ``stroke_to``: it does *not* walk the spacing, because
        an airbrush is not a line -- the canvas calls it every frame the button
        is held, so a stationary cursor keeps emitting, which is the whole
        behaviour. One press-to-release is still one ``PatchEdit``, since the
        dabs go through the same stroke buffer everything else does.
        """
        if self._stroke is None:
            return
        layer = self._stroke_layer()
        if layer is None:
            self._abandon_stroke()
            return
        self._stroke.spray(point, count, layer.pixels)
        self._touch_stroke()

    def _touch_stroke(self: Document) -> None:
        """Recomposite what the last dab touched, without pushing history: the
        undo step is the whole stroke, and it is pushed once at release.

        ``take_touched`` and **not** ``dirty``: the latter is the stroke's
        accumulated union, which is the right answer for the undo patch and the
        wrong one here twice over -- it grows as the stroke moves, and a mirror
        puts one dab in two or four places far apart. Both made this call
        recomposite most of the canvas for a dab the size of the nib; its
        docstring carries the measurements.
        """
        if self._stroke is None:
            raise ValueError("no stroke is open")
        for rect in self._stroke.take_touched():
            self.invalidate(rect)

    def end_stroke(self: Document) -> bool:
        """Close the stroke and make it exactly one undo step."""
        if self._stroke is not None and self._stroke_layer() is None:
            return self._abandon_stroke()
        stroke, self._stroke = self._stroke, None
        if stroke is not None and stroke.pending:
            # The pixel-perfect filter holds one pixel back to see whether the
            # next makes it an elbow, and at release there is no next: without
            # this flush a click marks nothing and every stroke is one pixel
            # short. Before the ``dirty is None`` test, since for a click the
            # flush is the only thing that makes it non-None.
            stroke.finish(self.stack.by_uid(stroke.layer_uid).pixels)
        if stroke is None or stroke.dirty is None:
            # A brush-down with no dab. ``begin_stroke`` may have autovivified a
            # cel for it, and nothing below will reach ``_commit_patch`` to
            # decide the cel's fate, so it is decided here.
            self._discard_pending_cel()
            return False
        box = self.clip(stroke.dirty)
        if box is None:
            self._discard_pending_cel()
            return False
        layer = self.stack.by_uid(stroke.layer_uid)
        x0, y0, x1, y1 = box
        self._commit_patch(layer, box, stroke.before[y0:y1, x0:x1])
        return True

    # -- fill, shapes, gradients -------------------------------------------

    def fill(
        self: Document,
        xy: tuple[int, int],
        colour: RGBA,
        *,
        thresh: int = 32,
        contiguous: bool = True,
        wrap: str | tuple[bool, bool] = "off",
        refer: str = "canvas",
        eight_connected: bool = False,
        stop_at_grid: int | tuple[int, int] | None = None,
    ) -> bool:
        """Flood fill from a point, on the composite's colours.

        The threshold is what makes this usable on generated art: an SDXL image
        has no flat regions, only nearly-flat ones behind an antialiased edge,
        and an exact-match fill stops after a few hundred pixels every time.
        The region comes from the same predicate the magic wand uses, so the
        two can never disagree about what "similar" means -- ``wrap`` included,
        which is why it is threaded straight through rather than reimplemented.

        Three Aseprite options ride on top of that one predicate, and each is
        about *which pixels the flood may read or reach* rather than about what
        similar means:

        * ``refer`` is ``"canvas"`` (the composite, the default and what you
          see) or ``"layer"`` (the active layer alone). Lineart on its own
          layer over a painted background is the case: referring to the canvas,
          every fill stops at the paint underneath it.
        * ``eight_connected`` continues a region through a corner touch, which
          is what a diagonal pixel-art line asks for.
        * ``stop_at_grid`` confines the region to the grid cell the seed is in,
          for filling one tile of a tileset without masking it first. It is the
          grid's size, so the caller decides whether the grid is on; a *crop*
          rather than a post-hoc clip, because a region that left the cell and
          came back would otherwise count.

        ``stop_at_grid`` and ``wrap`` do not combine: a cell is not a torus, so
        the crop wins and the fold has nothing left to fold.
        """
        if refer not in ("canvas", "layer"):
            raise ValueError(f"unknown fill refer {refer!r}")
        if not self.in_bounds(xy):
            return False
        source = self.stack.active.pixels if refer == "layer" else self._composite
        seed = (int(xy[0]), int(xy[1]))
        cell = self._grid_cell(seed, stop_at_grid)
        if cell is not None:
            cx0, cy0, cx1, cy1 = cell
            source = source[cy0:cy1, cx0:cx1]
            seed = (seed[0] - cx0, seed[1] - cy0)
            wrap = "off"
        region = magic_wand(
            source,
            seed,
            tolerance=thresh,
            contiguous=contiguous,
            wrap=wrap,
            eight_connected=eight_connected,
        )
        bounds = region.bounds
        if bounds is None:
            return False
        x0, y0, x1, y1 = bounds
        weight = region.mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        if cell is not None:
            x0, y0, x1, y1 = x0 + cell[0], y0 + cell[1], x1 + cell[0], y1 + cell[1]
        return self.write_colour((x0, y0, x1, y1), colour, weight)

    def _grid_cell(
        self: Document,
        seed: tuple[int, int],
        grid: int | tuple[int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        """The grid cell *seed* falls in, clipped to the canvas, or ``None``."""

        if grid is None:
            return None
        cell_w, cell_h = (grid, grid) if isinstance(grid, int) else grid
        cell_w, cell_h = int(cell_w), int(cell_h)
        if cell_w <= 0 or cell_h <= 0:
            return None
        width, height = self.size
        x0 = (seed[0] // cell_w) * cell_w
        y0 = (seed[1] // cell_h) * cell_h
        return (x0, y0, min(width, x0 + cell_w), min(height, y0 + cell_h))

    def shape(
        self: Document,
        kind: str,
        p0: tuple[int, int],
        p1: tuple[int, int],
        colour: RGBA,
        size: int,
        *,
        filled: bool = False,
        wrap: str | tuple[bool, bool] = "off",
        radius: int = 0,
    ) -> bool:
        """Line, rectangle or ellipse, antialiased through a coverage mask.

        ``radius`` rounds a rectangle's corners, Aseprite's own hold-``C``-
        while-dragging control. It is a *parameter of the shape* rather than a
        fourth kind, because everything else about it -- the coverage plane,
        the tiled fold, the brush width, the fill -- is the rectangle's, and a
        "roundrect" kind would have to repeat all of it to change one call.

        The two-point door, which is what a *drag* produces. A path shape
        (:data:`PATH_SHAPES`) named here is routed to :meth:`shape_path` with
        the two points rather than refused: they are all in ``SHAPES``, and one
        list whose entries mean different things depending on who asks is worse
        than either list on its own.

        With ``wrap`` on, the coverage is rasterised over a plane big enough to
        hold the whole gesture *and* the brush width that spills past the canvas
        edge, and then folded home by :func:`.tiling.fold_coverage`. Rasterising
        into the canvas and wrapping afterwards is the only order that works:
        Pillow clips to its image, so anything drawn past the edge would be gone
        before there was anything to fold.

        With it off the plane *is* the canvas at the origin, so the fold is the
        identity and the bounding box is the one this method always computed.
        """
        if kind not in SHAPES:
            raise ValueError(f"unknown shape {kind!r}")
        if kind in PATH_SHAPES:
            # A path shape handed the two points a drag produces: a polyline of
            # two is the line it draws, and routing rather than reimplementing
            # is what keeps ``SHAPES`` one list with one meaning per entry.
            return self.shape_path(kind, (p0, p1), colour, size, filled=filled, wrap=wrap)
        from PIL import Image, ImageDraw

        size = clamp_brush(size)
        axes = tiling.axes_of(wrap)
        (ox, oy), (pw, ph) = self._shape_plane(p0, p1, size, axes)
        canvas = Image.new("L", (pw, ph), 0)
        draw = ImageDraw.Draw(canvas)
        q0 = (p0[0] - ox, p0[1] - oy)
        q1 = (p1[0] - ox, p1[1] - oy)
        if kind == "line":
            draw.line([q0, q1], fill=255, width=size, joint="curve")
        else:
            x0, y0, x1, y1 = normalise_rect(q0, q1)
            corner = max(0, int(radius))
            if kind == "rect" and corner:
                # Clamped to half the shorter side: a radius past that is a
                # stadium, and Pillow draws it as one rather than refusing --
                # so the clamp is what keeps the number on screen and the
                # shape drawn in agreement.
                corner = min(corner, (x1 - x0) // 2, (y1 - y0) // 2)
            if kind == "rect" and corner > 0:
                draw.rounded_rectangle(
                    (x0, y0, x1, y1),
                    radius=corner,
                    fill=255 if filled else None,
                    outline=255,
                    width=size,
                )
            else:
                method = draw.rectangle if kind == "rect" else draw.ellipse
                if filled:
                    method((x0, y0, x1, y1), fill=255, outline=255, width=size)
                else:
                    method((x0, y0, x1, y1), fill=None, outline=255, width=size)

        folded = tiling.fold_coverage(
            np.asarray(canvas, dtype=np.uint8), (ox, oy), self.size, axes
        )
        if folded is None:
            return False
        rect, coverage = folded
        return self.write_colour(rect, colour, coverage.astype(np.float32) / 255.0)

    def shape_path(
        self: Document,
        kind: str,
        points: Any,
        colour: RGBA,
        size: int,
        *,
        filled: bool = False,
        wrap: str | tuple[bool, bool] = "off",
    ) -> bool:
        """A polyline, a polygon or a curve through ``points`` (Q-c).

        :meth:`shape`'s sibling for the shapes a *path* defines, and everything
        below the rasterising is deliberately the same code: the same plane, the
        same fold home for a wrapped axis, and the same single ``write_colour``
        -- so a polygon honours the selection, the alpha lock and an indexed
        document's palette exactly as a rectangle does, and however many clicks
        it took it is **one undo step**.

        ``curve`` is sampled to a polyline by :func:`catmull_rom` first, which
        is the whole of the difference between it and ``polyline``. ``polygon``
        is the closed one and the only one ``filled`` means anything to: an open
        path has no inside, and rather than invent one the flag is ignored.

        Two distinct points is the floor -- one point is a dot the brush would
        draw better, and the antialiased coverage of a zero-length line is
        nothing at all. It is deliberately *not* the poly-lasso's three: the
        engine draws what it is given, and "a polygon of two clicks commits
        nothing" is a rule about the gesture, kept where the gesture is.
        """
        if kind not in PATH_SHAPES:
            raise ValueError(f"unknown path shape {kind!r}")
        from PIL import Image, ImageDraw

        size = clamp_brush(size)
        path = catmull_rom(points) if kind == "curve" else list(points)
        path = [(float(x), float(y)) for x, y in path]
        if len(set(path)) < 2:
            return False
        axes = tiling.axes_of(wrap)
        (ox, oy), (pw, ph) = self._path_plane(path, size, axes)
        canvas = Image.new("L", (pw, ph), 0)
        draw = ImageDraw.Draw(canvas)
        local = [(x - ox, y - oy) for x, y in path]
        closed = kind == "polygon"
        if closed and filled:
            # Fill first and stroke over it, which is the order ``shape`` draws
            # a filled rectangle in: the outline is what carries the brush size,
            # so a filled polygon is exactly its outlined self plus its inside.
            draw.polygon(local, fill=255)
        draw.line(
            [*local, local[0]] if closed else local, fill=255, width=size, joint="curve"
        )

        folded = tiling.fold_coverage(
            np.asarray(canvas, dtype=np.uint8), (ox, oy), self.size, axes
        )
        if folded is None:
            return False
        rect, coverage = folded
        return self.write_colour(rect, colour, coverage.astype(np.float32) / 255.0)

    def _path_plane(
        self: Document,
        points: Any,
        size: int,
        axes: tuple[bool, bool],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """:meth:`_shape_plane` for a path: its bounding box, then that.

        A whole plane per shape rather than per segment, so a polyline that
        crosses a wrapped seam twice is folded home once and lands as one
        connected stroke -- folding segment by segment would rasterise each into
        its own plane and lose the joins between them.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        lo = (math.floor(min(xs)), math.floor(min(ys)))
        hi = (math.ceil(max(xs)), math.ceil(max(ys)))
        return self._shape_plane(lo, hi, size, axes)

    def _shape_plane(
        self: Document,
        p0: tuple[int, int],
        p1: tuple[int, int],
        size: int,
        axes: tuple[bool, bool],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """The rasterisation plane a shape needs: ``(origin, (w, h))``.

        The canvas exactly, on an unwrapped axis -- so nothing about the
        untiled path changes. On a wrapped one it grows to hold the gesture and
        the brush width around it, clamped to the 3x3 neighbourhood the tiled
        view actually shows: a drag cannot land further out than one tile away,
        and a clamp is cheaper than trusting a number that arrives from a mouse.
        """
        width, height = self.size
        pad = size + 1
        origin, extent = [], []
        for wrapped, lo_p, hi_p, span in (
            (axes[0], min(p0[0], p1[0]), max(p0[0], p1[0]), width),
            (axes[1], min(p0[1], p1[1]), max(p0[1], p1[1]), height),
        ):
            if not wrapped:
                origin.append(0)
                extent.append(span)
                continue
            lo = max(-span, min(0, int(lo_p) - pad))
            hi = min(2 * span, max(span, int(hi_p) + pad + 1))
            origin.append(lo)
            extent.append(hi - lo)
        return (origin[0], origin[1]), (extent[0], extent[1])

    def gradient(
        self: Document,
        p0: tuple[float, float],
        p1: tuple[float, float],
        start: RGBA | None = None,
        end: RGBA | None = None,
        *,
        kind: str = "linear",
        stops: Any = None,
        dither: str | None = None,
    ) -> bool:
        """Fill the selection (or the whole layer) with a ramp.

        ``stops`` is the general form; ``start``/``end`` is the two-stop
        shorthand every existing caller uses. Both go through one interpolator.

        **A gradient deliberately does not wrap**, and takes no ``wrap``
        argument to say so. A ramp has two endpoints: wrapping one puts the last
        stop against the first, which is a hard edge at the seam -- the one
        thing tiled mode exists to remove. A seamless ramp is a different
        object (a periodic one), not this one with a flag.

        ``dither`` thresholds the ramp onto its own stops instead of blending
        between them -- see :func:`gradient.dithered`. It is applied inside the
        render, never afterwards: dithering the *result* would be quantising a
        blend that has already been made, which is a different picture and one
        that no longer lands on the stop colours exactly.

        The **selection weight is never dithered**, which is why it comes back
        from the render separately and is multiplied in below. A feathered
        selection means one thing across every tool in this class, and a soft
        edge chopped into a chequer is not that thing.
        """
        if self.write_locked():
            return False
        width, height = self.size
        rect = self.mask.bounds if self.mask is not None else None
        box = self.clip(rect or (0, 0, width, height))
        if box is None:
            return False
        rgba, weight = grad.render(
            (width, height), p0, p1, start, end, kind, stops=stops, dither=dither
        )
        x0, y0, x1, y1 = box
        self._ensure_active_cel()
        layer = self.stack.active
        before = layer.pixels[y0:y1, x0:x1].copy()
        crop = rgba[y0:y1, x0:x1]
        clipped = self._weights(box, weight[y0:y1, x0:x1])

        # Per-pixel colour, so ``paint_colour``'s flat-colour form does not fit;
        # this is the same arithmetic with the colour varying.
        src_a = clipped
        dst_a = before[..., 3].astype(np.float32) / 255.0
        out_a = src_a + dst_a * (1.0 - src_a)
        share = np.divide(src_a, out_a, out=np.zeros_like(src_a), where=out_a > 0.0)
        out = np.empty(before.shape, dtype=np.float32)
        rgb = before[..., :3].astype(np.float32)
        out[..., :3] = rgb + (crop[..., :3] * 255.0 - rgb) * share[..., None]
        out[..., 3] = out_a * 255.0
        if layer.alpha_lock:
            # The same one line ``write_colour`` and ``preview_filter`` already
            # had, and the one write path that was missing it: a ramp run over
            # a layer with "preserve transparency" on painted straight through
            # the edge of the shape it was meant to stay inside, because the
            # composite above sets ``out_a`` from the ramp's own coverage.
            # Restoring the channel *is* the definition of the lock rather than
            # an approximation of it, which is why it goes after the formula.
            out[..., 3] = before[..., 3]
        layer.pixels[y0:y1, x0:x1] = cp.to_uint8_255(out)
        self._commit_patch(layer, box, before)
        return True
