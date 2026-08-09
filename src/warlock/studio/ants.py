"""Where the marching-ants dashes fall along a selection outline.

The pane that draws them imports imgui at module scope, so the arithmetic lives
here instead: this module is numpy and ``math`` and nothing else, which is what
makes the one part of the ants that can be wrong -- where a dash starts, where
it ends, and whether it is the light half or the dark half -- assertable
headlessly.

The rewrite it exists for: the old walk transformed every vertex in a Python
list comprehension and then stepped every unit segment with an inner
``while travelled < length`` loop, issuing an ``add_line`` per dash *per unit
segment*. A 2048-square lasso is tens of thousands of vertices, so that was
tens of thousands of Python iterations every frame, forever. Here the geometry
is prepared once per selection (:func:`prepare`) and the per-frame work is a
handful of whole-array passes and one ``add_line`` per straight *run*.

Three properties are load-bearing and all three are pinned by tests:

**The dash pattern is continuous across vertices.** It is a function of arc
length from the start of the loop, not of position within a segment -- which
falls out of measuring against the cumulative length, and is what keeps a
one-pixel lattice step from restarting the pattern sixty times a corner.

**A dash is measured in screen pixels.** ``DASH`` is a constant on screen, so
the loop's own arc length is scaled by the zoom rather than the dash being
scaled by its inverse; the ants keep their spacing at every zoom, as before.

**A dash follows the outline; it never cuts across it.** The tempting version
of this module interpolates each dash's two endpoints and draws the chord
between them -- one line per dash rather than per lattice step, which is the
whole speed argument. It is also wrong: a contour is a staircase, and a chord
across six steps of one straightens the staircase out and skips a one-pixel
spike entirely. So a dash is split at every vertex it crosses, and the *runs*
that come back out are then merged wherever consecutive pieces are collinear
and equally lit. That merge is exact -- collinear pieces of one straight run
are one line -- and it is where the speed actually comes from: a marquee's
edge is one call rather than two thousand, and a staircase, which genuinely
has a direction change per pixel, costs what it always did.
"""

from __future__ import annotations

import math

import numpy as np

# Dash length in screen pixels, and how fast the pattern crawls along the
# outline. Both were the pane's; they are the ants' rather than the canvas's.
DASH = 6.0
ANT_SPEED = 12.0

# Two directions count as one straight run below this cross product of their
# unit vectors -- a sine, so it is an angle rather than a distance. Lattice
# steps are axis-aligned and give exactly zero; the tolerance is for the two
# interpolated endpoints a dash boundary introduces mid-segment.
_STRAIGHT = 1e-6


def prepare(loops: list[list[tuple[int, int]]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(vertices, cumulative arc length)`` per loop, in *canvas* units.

    The vertex array is closed -- the first point is repeated at the end -- so
    that a dash spanning the seam interpolates like every other one instead of
    needing a wrap-around case. Lengths are float64 and computed rather than
    assumed: today's tracer emits unit steps, but that is a property of the
    tracer and not of ``contours()``'s contract.
    """
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for loop in loops:
        verts = np.asarray(loop, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[0] < 2:
            continue
        verts = np.concatenate((verts, verts[:1]), axis=0)
        deltas = verts[1:] - verts[:-1]
        cum = np.concatenate(((0.0,), np.cumsum(np.hypot(deltas[:, 0], deltas[:, 1]))))
        if cum[-1] <= 0.0:
            continue
        prepared.append((verts, cum))
    return prepared


def dash_segments(
    verts: np.ndarray,
    cum: np.ndarray,
    zoom: float,
    offset: tuple[float, float],
    phase: float,
    dash: float = DASH,
    basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lines to draw for one loop: ``(starts, ends, lit)`` in screen space.

    Each row is a straight piece of the outline lying inside a single dash, so
    drawing them all in ``lit``'s two colours reproduces the outline exactly and
    the dash pattern along it.

    ``offset`` is the screen position of canvas ``(0, 0)`` -- the caller gets it
    from ``inker_state.to_screen`` rather than restating the formula, because
    the ants sitting one pixel off the mask they describe is exactly what a
    duplicated affine looks like.

    ``basis`` is the view's 2x2 orientation (``inker_state.basis``), or ``None``
    for the upright view. It is deliberately **separate from the zoom** and
    deliberately orthonormal: every distance in this function is an arc length
    measured in canvas space, so a transform that turns leaves the whole dash
    calculation exactly as it was, and one that scaled would have to be threaded
    through all of it. The whole of the turn is one matmul in ``_points_at``,
    and the straight-run merge survives it because a rotation maps a straight
    run onto a straight run.

    Boundaries land at ``phase + k * dash`` and the even ``k`` are lit, which is
    the old walk's ``int((walked + travelled) // DASH) % 2 == 0`` with ``walked``
    starting at ``-phase``. The dash straddling the loop's start is clipped at
    both ends rather than wrapped: it is drawn from 0, exactly as before.
    """
    empty = np.zeros((0, 2), dtype=np.float64)
    total = float(cum[-1]) * float(zoom)
    if not (total > 0.0) or not math.isfinite(total) or dash <= 0.0:
        return empty, empty.copy(), np.zeros(0, dtype=bool)

    # Break the loop at every dash boundary *and* at every vertex. The vertices
    # are already the sorted array `cum`, so the boundaries are spliced into it
    # rather than concatenated and re-sorted: one pass instead of a sort of
    # tens of thousands of values, every frame.
    first = math.floor(-phase / dash)
    last = math.floor((total - phase) / dash)
    edges = np.clip(phase + np.arange(first + 1, last + 1) * dash, 0.0, total)
    edges /= float(zoom)
    bounds = np.insert(cum, np.searchsorted(cum, edges), edges)

    starts, ends = bounds[:-1], bounds[1:]
    # A boundary landing exactly on a vertex is now in there twice; so is the
    # clipped tail of the dash that began before the loop did.
    keep = ends > starts
    if not keep.any():
        return empty, empty.copy(), np.zeros(0, dtype=bool)

    points = _points_at(verts, cum, bounds, zoom, offset, basis)
    starts, ends = starts[keep], ends[keep]
    head, tail = points[:-1][keep], points[1:][keep]
    # Which dash a piece belongs to, asked at its midpoint: at its start the
    # answer is a boundary case, and every piece lies wholly within one dash.
    index = np.floor(((starts + ends) * 0.5 * float(zoom) - phase) / dash)
    lit = (index % 2.0) == 0.0

    return _merge_runs(head, tail, lit)


def cull(
    starts: np.ndarray,
    ends: np.ndarray,
    lit: np.ndarray,
    rect: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the runs whose AABB touches ``rect`` (screen space).

    ``rect`` is ``(left, top, right, bottom)``. The pane calls this before its
    per-run ``add_line`` loop -- the one irreducible Python loop the ants have
    -- so a selection mostly off screen costs only its visible edge (B23).
    Pure and conservative: a run is dropped only when its box provably misses
    the rectangle, so drawing the survivors reproduces the visible outline
    exactly.
    """
    if len(starts) == 0:
        return starts, ends, lit
    left, top, right, bottom = rect
    xs = np.minimum(starts[:, 0], ends[:, 0])
    xe = np.maximum(starts[:, 0], ends[:, 0])
    ys = np.minimum(starts[:, 1], ends[:, 1])
    ye = np.maximum(starts[:, 1], ends[:, 1])
    keep = (xe >= left) & (xs <= right) & (ye >= top) & (ys <= bottom)
    if keep.all():
        return starts, ends, lit
    return starts[keep], ends[keep], lit[keep]


def loop_box(verts: np.ndarray) -> tuple[float, float, float, float]:
    """A loop's canvas-space AABB ``(x0, y0, x1, y1)``, computed once beside
    ``prepare``'s output so the per-frame pre-filter (B23) is four floats."""
    return (
        float(verts[:, 0].min()),
        float(verts[:, 1].min()),
        float(verts[:, 0].max()),
        float(verts[:, 1].max()),
    )


def _merge_runs(
    head: np.ndarray, tail: np.ndarray, lit: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse consecutive collinear pieces of the same colour into one line.

    Exact rather than approximate: the pieces are contiguous (each ends where
    the next begins) and collinear, so their union *is* the line this emits.
    What it buys is the draw call count -- a rectangle's edge arrives here as
    one piece per pixel and leaves as one per dash.
    """
    step = tail - head
    length = np.hypot(step[:, 0], step[:, 1])
    safe = np.where(length > 0.0, length, 1.0)
    unit = step / safe[:, None]
    cross = np.abs(unit[:-1, 0] * unit[1:, 1] - unit[:-1, 1] * unit[1:, 0])

    fresh = np.empty(len(head), dtype=bool)
    fresh[0] = True
    fresh[1:] = (cross > _STRAIGHT) | (lit[1:] != lit[:-1])

    begin = np.flatnonzero(fresh)
    finish = np.append(begin[1:] - 1, len(head) - 1)
    return head[begin], tail[finish], lit[begin]


def _points_at(
    verts: np.ndarray,
    cum: np.ndarray,
    distance: np.ndarray,
    zoom: float,
    offset: tuple[float, float],
    basis: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate canvas-space arc positions and put them on screen.

    ``side="right"`` minus one puts a distance landing exactly on a vertex *at*
    that vertex with ``t == 0``, which is where the old per-segment walk put it:
    it reset ``travelled`` at every vertex, so a boundary there was the start of
    the next segment rather than the end of the previous one.
    """
    index = np.searchsorted(cum, distance, side="right") - 1
    index = np.clip(index, 0, cum.size - 2)
    span = cum[index + 1] - cum[index]
    safe = np.where(span > 0.0, span, 1.0)
    ratio = np.where(span > 0.0, (distance - cum[index]) / safe, 0.0)[:, None]
    here = verts[index]
    points = here + (verts[index + 1] - here) * ratio
    if basis is not None:
        points = points @ np.asarray(basis, dtype=np.float64).T
    return points * float(zoom) + np.asarray(offset, dtype=np.float64)
