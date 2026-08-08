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


def render(
    size: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    start: tuple[int, int, int, int] | None = None,
    end: tuple[int, int, int, int] | None = None,
    kind: str = "linear",
    *,
    stops: Sequence[Stop] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (rgba float32 straight 0..1, weight float32 0..1).

    The alpha ramp comes back as the *weight* rather than as the colour's own
    alpha, which is what makes "foreground to transparent" fade out instead of
    fading to black: the colour stays the foreground everywhere and only the
    amount written changes. With three stops that is still true per stop, so a
    gradient can fade out in the middle and back in.
    """
    if stops is None:
        if start is None or end is None:
            raise ValueError("render needs either two colours or a stop list")
        stops = [(0.0, start), (1.0, end)]
    mixed = sample(stops, ramp(size, p0, p1, kind))
    rgba = np.empty(mixed.shape, dtype=np.float32)
    rgba[..., :3] = mixed[..., :3]
    rgba[..., 3] = 1.0
    return rgba, mixed[..., 3]
