"""Linear and radial gradient rasterisers.

Pure functions from two points and two colours to an RGBA plane plus a weight
plane. The weight is returned separately rather than baked in so the caller can
clip it to a selection with the same multiply a brush stamp uses -- one clipping
rule for everything that writes pixels.
"""

from __future__ import annotations

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


def render(
    size: tuple[int, int],
    p0: tuple[float, float],
    p1: tuple[float, float],
    start: tuple[int, int, int, int],
    end: tuple[int, int, int, int],
    kind: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """-> (rgba float32 straight 0..1, weight float32 0..1).

    The alpha ramp comes back as the *weight* rather than as the colour's own
    alpha, which is what makes "foreground to transparent" fade out instead of
    fading to black: the colour stays the foreground everywhere and only the
    amount written changes.
    """
    t = ramp(size, p0, p1, kind)[..., None]
    a = np.array(start, dtype=np.float32) / 255.0
    b = np.array(end, dtype=np.float32) / 255.0
    mixed = a + (b - a) * t
    rgba = np.empty(mixed.shape, dtype=np.float32)
    rgba[..., :3] = mixed[..., :3]
    rgba[..., 3] = 1.0
    return rgba, mixed[..., 3]
