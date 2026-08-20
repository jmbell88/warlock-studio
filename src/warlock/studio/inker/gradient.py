"""Linear and radial gradient rasterisers.

Pure functions from two points and a list of colour stops to an RGBA plane plus
a weight plane. The weight is returned separately rather than baked in so the
caller can clip it to a selection with the same multiply a brush stamp uses --
one clipping rule for everything that writes pixels.

**Two colours is the two-stop case, not a separate path.** ``render`` still
takes ``start`` and ``end`` and builds ``[(0, start), (1, end)]`` from them, so
the common call is unchanged and there is exactly one interpolator to be right
about -- a shortcut with its own arithmetic is how the two come to disagree in
the last ulp at the ends.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Aliased, because ``render`` takes a keyword called ``dither`` and a parameter
# that shadows the module it needs is the sort of thing that works until
# somebody adds one line inside the function.
from . import dither as dith

KINDS = ("linear", "radial")


def ramp(
    size: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    kind: str = "linear",
) -> np.ndarray:
    """0..1 float plane: 0 at ``p0``, 1 at ``p1``, clamped past both ends."""
    width, height = size
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    dx, dy = float(p1[0] - p0[0]), float(p1[1] - p0[1])

    if kind == "linear":
        length2 = dx * dx + dy * dy
        if length2 <= 0.0:
            return np.ones((height, width), dtype=np.float32)
        t = ((xs - p0[0]) * dx + (ys - p0[1]) * dy) / length2
    elif kind == "radial":
        radius = float(np.hypot(dx, dy))
        if radius <= 0.0:
            return np.ones((height, width), dtype=np.float32)
        t = np.hypot(xs - p0[0], ys - p0[1]) / radius
    else:
        raise ValueError(f"unknown gradient kind {kind!r}")
    return np.clip(t, 0.0, 1.0).astype(np.float32)


Stop = tuple[float, tuple[int, int, int, int]]


def sample(stops: Sequence[Stop], t: np.ndarray) -> np.ndarray:
    """Piecewise-linear colour along *t*, straight alpha, 0..1 float32.

    Interpolation is per channel through ``np.interp``, which clamps outside
    the stop range -- the same "clamped past both ends" rule :func:`ramp`
    already applies to the position. Stops are sorted here rather than
    required sorted, because the UI's list is in the order they were added and
    dragging one past another must not invert the ramp.

    Two stops at the same position are a *hard edge*, which is the useful
    reading and the one every gradient editor gives it.
    """
    if not stops:
        raise ValueError("a gradient needs at least one stop")
    ordered = sorted(stops, key=lambda stop: float(stop[0]))
    positions = np.array([float(p) for p, _ in ordered], dtype=np.float32)
    colours = np.array([c for _, c in ordered], dtype=np.float32) / 255.0
    if len(ordered) == 1:
        # A flat colour rather than an error: dragging a two-stop gradient down
        # to one stop is a thing a user does on the way to building a
        # three-stop one, and refusing mid-edit would be refusing a keystroke.
        return np.broadcast_to(colours[0], (*t.shape, 4)).astype(np.float32)
    return np.stack(
        [np.interp(t, positions, colours[:, c]).astype(np.float32) for c in range(4)],
        axis=-1,
    )


def dithered(stops: Sequence[Stop], t: np.ndarray, method: str) -> np.ndarray:
    """:func:`sample` with the blend replaced by an ordered threshold.

    Every pixel comes out as one of the two stops that bracket it, chosen by
    where it falls between them against ``dither.bayer_matrix``. So a two-stop
    black-to-white ramp lands entirely on black and white, which is what
    dithering a gradient *is* -- and on an indexed document the snap in
    ``_commit_patch`` is then a no-op, because both candidates are already
    palette members whenever the stops are.

    Kind-agnostic by construction: it takes the ramp parameter and knows nothing
    about how it was computed, so linear and radial dither identically.

    **Alpha dithers with its stop**, rather than being interpolated under a
    thresholded colour. The alternative reads as a soft gradient with hard
    colour banding on top -- two different gradients in one drag.

    The matrix is anchored at the canvas origin, so a ramp drawn in two halves
    interlocks with itself instead of showing a seam.
    """
    if method not in dith.BAYER_SIZES:
        raise ValueError(f"unknown gradient dither {method!r}")
    if not stops:
        raise ValueError("a gradient needs at least one stop")
    ordered = sorted(stops, key=lambda stop: float(stop[0]))
    colours = np.array([c for _, c in ordered], dtype=np.float32) / 255.0
    if len(ordered) == 1:
        return np.broadcast_to(colours[0], (*t.shape, 4)).astype(np.float32)
    positions = np.array([float(p) for p, _ in ordered], dtype=np.float32)

    # ``side="right"`` then clamped into the segment range: a parameter sitting
    # exactly on a stop belongs to the segment *after* it, which is what makes
    # two stops at one position the hard edge ``sample`` already gives them.
    low = np.clip(np.searchsorted(positions, t, side="right") - 1, 0, len(ordered) - 2)
    span = positions[low + 1] - positions[low]
    # ``composite.over``'s masked-lane fix: ``where=`` does not promise the
    # masked lanes go unevaluated, so a SIMD lane with a zero span (two stops
    # at one position) still ran x/0 and raised under
    # ``np.errstate(all="raise")``. Divide by one there and select -- the
    # masked lanes keep the one a zero-width segment always answered.
    lit = span > 0.0
    where = np.empty_like(t)
    np.divide(t - positions[low], np.where(lit, span, 1.0), out=where)
    where = np.where(lit, where, 1.0)
    where = np.clip(where, 0.0, 1.0)
    threshold = dith.tile_matrix(dith.bayer_matrix(dith.BAYER_SIZES[method]), t.shape)
    return colours[np.where(where > threshold, low + 1, low)]


def render(
    size: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    start: tuple[int, int, int, int] | None = None,
    end: tuple[int, int, int, int] | None = None,
    kind: str = "linear",
    *,
    stops: Sequence[Stop] | None = None,
    dither: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (rgba float32 straight 0..1, weight float32 0..1).

    The alpha ramp comes back as the *weight* rather than as the colour's own
    alpha, which is what makes "foreground to transparent" fade out instead of
    fading to black: the colour stays the foreground everywhere and only the
    amount written changes. With three stops that is still true per stop, so a
    gradient can fade out in the middle and back in.

    ``dither`` names one of the ordered matrices (``bayer2``/``4``/``8``) and
    replaces the blend between adjacent stops with a threshold against it; see
    :func:`dithered`. ``None`` is not merely the default but a separate path:
    the undithered arithmetic below is untouched, byte for byte, so turning the
    option on and off returns the exact pixels it always produced.
    """
    if stops is None:
        if start is None or end is None:
            raise ValueError("render needs either two colours or a stop list")
        stops = [(0.0, start), (1.0, end)]
    t = ramp(size, p0, p1, kind)
    mixed = sample(stops, t) if dither is None else dithered(stops, t, dither)
    rgba = np.empty(mixed.shape, dtype=np.float32)
    rgba[..., :3] = mixed[..., :3]
    rgba[..., 3] = 1.0
    return rgba, mixed[..., 3]
