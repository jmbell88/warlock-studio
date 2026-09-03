"""Heat shimmer: displaces everything beneath by a scrolling noise field.

The one primitive that *replaces* the composite below rather than painting
over it, which ``REPLACES_BELOW`` tells the compositor. Displacement is by
nearest-sample gather -- no interpolation -- so the result stays exact
arithmetic on the input and pixel-art output does not gain intermediate
colours.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import POSITION, Param, fbm_plane, val

REPLACES_BELOW = True

PARAMS = {
    **POSITION,
    "radius": Param("curve", 40.0, 0.0, 1024.0, label="px, 0 = whole frame"),
    "strength": Param("curve", 3.0, 0.0, 64.0, label="px"),
    "scale": Param("float", 10.0, 1.0, 128.0, label="px per cell"),
    "speed": Param("float", 3.0, 0.0, 20.0),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    if below is None:
        return None
    strength = val(layer, "strength", ctx)
    if strength <= 0.0:
        return below
    scale = val(layer, "scale", ctx)
    drift = val(layer, "speed", ctx) * ctx.time
    seed = ctx.lseed()
    dx = (fbm_plane(ctx, seed + 3, scale=scale, dy=-drift, octaves=2, smooth=False) - 0.5) * 2.0
    dy = (
        fbm_plane(ctx, seed + 5, scale=scale, dx=51.0, dy=-drift, octaves=2, smooth=False) - 0.5
    ) * 2.0
    radius = val(layer, "radius", ctx)
    if radius > 0.0:
        cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))
        x, y = ctx.coords()
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / np.float32(radius)
        fall = np.clip(1.0 - d, 0.0, 1.0)
        dx = dx * fall
        dy = dy * fall
    px = np.float32(strength * ctx.scale)
    ys, xs = ctx.index()
    h, w = below.shape[:2]
    sx = np.clip(np.rint(xs + dx * px), 0, w - 1).astype(np.intp)
    sy = np.clip(np.rint(ys + dy * px), 0, h - 1).astype(np.intp)
    return below[sy, sx]
