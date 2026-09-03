"""An impact flash: a bright soft disc, gone in a few frames."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import POSITION, Param, color, disc, premultiply, val

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "radius": Param(
        "curve",
        {"keys": [[0.0, 10.0], [0.3, 40.0], [1.0, 30.0]], "easing": "ease_out"},
        0.0,
        1024.0,
    ),
    "alpha": Param("curve", {"keys": [[0.0, 1.0], [0.4, 0.8], [1.0, 0.0]]}, 0.0, 1.0),
    "color": Param("color", "#FFFFFF"),
    "softness": Param("float", 0.8, 0.0, 1.0),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    alpha = val(layer, "alpha", ctx)
    radius = val(layer, "radius", ctx)
    if alpha <= 0.0 or radius <= 0.0:
        return None
    cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))
    cov = disc(ctx, cx, cy, radius, val(layer, "softness", ctx))
    col = color(layer, "color")
    return premultiply(np.broadcast_to(col[:3], cov.shape + (3,)), cov * np.float32(alpha * col[3]))
