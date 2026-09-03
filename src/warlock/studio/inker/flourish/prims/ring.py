"""A shockwave: an expanding ring whose thickness and opacity are curves."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import POSITION, Param, _falloff, color, fbm_plane, premultiply, val, window

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "radius": Param(
        "curve", {"keys": [[0.0, 4.0], [1.0, 64.0]], "easing": "ease_out"}, 0.0, 1024.0
    ),
    "thickness": Param("curve", {"keys": [[0.0, 8.0], [1.0, 1.0]]}, 0.1, 256.0),
    "alpha": Param("curve", {"keys": [[0.0, 1.0], [1.0, 0.0]]}, 0.0, 1.0),
    "color": Param("color", "#FFE9C0"),
    "softness": Param("float", 0.5, 0.0, 1.0),
    "unevenness": Param("float", 0.0, 0.0, 1.0, label="noise on the radius"),
    "squash": Param("float", 1.0, 0.1, 1.0, label="y/x, for a ground ring"),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    radius = val(layer, "radius", ctx)
    thick = val(layer, "thickness", ctx)
    alpha = val(layer, "alpha", ctx)
    if radius <= 0.0 or thick <= 0.0 or alpha <= 0.0:
        return None
    cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))
    uneven = val(layer, "unevenness", ctx)
    reach = radius * (1.0 + 0.3 * uneven) + thick
    win = window(ctx, cx, cy, reach)
    if win is None:
        return None
    squash = val(layer, "squash", ctx)
    d = np.sqrt((win.x - cx) ** 2 + ((win.y - cy) / np.float32(squash)) ** 2)
    if uneven > 0.0:
        n = fbm_plane(
            ctx, ctx.lseed(), scale=max(radius / 3.0, 2.0), dy=ctx.time, octaves=2, win=win
        )
        d = d * (1.0 + (n - 0.5) * 0.5 * uneven)
    edge = np.abs(d - radius) / np.float32(thick * 0.5)
    cov = _falloff(edge, val(layer, "softness", ctx), thick * 0.5 * ctx.scale)
    col = color(layer, "color")
    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    out[win.rows, win.cols] = premultiply(
        np.broadcast_to(col[:3], cov.shape + (3,)), cov * np.float32(alpha * col[3])
    )
    return out
