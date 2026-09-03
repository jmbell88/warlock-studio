"""A sprite: an ingredient texture stamped, scaled, turned and faded.

The door for pictures that are not arithmetic -- a rune, a skull-shaped ember,
a flame somebody painted or a diffusion model produced. The texture is an
*asset* the recipe names by id and the document holds beside the recipe
(``_doc_flourish``'s ``assets``); the engine only ever sees a straight-alpha
RGBA array through ``ctx.asset``. No asset, or an id nothing holds, renders
nothing rather than a placeholder: an effect must not grow a grey square
because a file went missing.

Stamped by nearest sampling of the texture at the supersampled raster, so a
pixel-art texture stays pixel art and a painterly one is smoothed by the
bake's own reduction. Rotation and scale are curves over the phase; ``spin``
adds a constant turn per second on top, and ``flicker`` modulates the alpha
with the frame noise so a static picture reads as alive.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import POSITION, Param, color, fbm_plane, stamp, val

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "texture": Param("asset", "", label="asset id"),
    "size": Param("curve", 32.0, 1.0, 1024.0, label="px wide"),
    "rotation": Param("curve", 0.0, -720.0, 720.0, label="degrees"),
    "spin": Param("float", 0.0, -1440.0, 1440.0, label="degrees per second"),
    "alpha": Param("curve", 1.0, 0.0, 1.0),
    "tint": Param("color", "#FFFFFF"),
    "flicker": Param("float", 0.0, 0.0, 1.0, label="noise on the alpha"),
    "flicker_hz": Param("float", 8.0, 0.0, 60.0),
    "face_direction": Param("bool", True, label="turn with the facing"),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    texture = ctx.asset(str(layer.params.get("texture", "")))
    if texture is None:
        return None
    alpha = val(layer, "alpha", ctx)
    size = val(layer, "size", ctx)
    if alpha <= 0.0 or size <= 0.0:
        return None
    flicker = val(layer, "flicker", ctx)
    if flicker > 0.0:
        # One noise sample per frame: the whole stamp breathes together.
        t = ctx.time * val(layer, "flicker_hz", ctx)
        n = fbm_plane(ctx, ctx.lseed(), scale=8.0, dx=t, octaves=2)[0, 0]
        alpha *= 1.0 - flicker * float(n)
    cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))
    angle = val(layer, "rotation", ctx) + val(layer, "spin", ctx) * ctx.time
    if bool(layer.params.get("face_direction", True)):
        angle += ctx.direction
    tint = color(layer, "tint")
    return stamp(ctx, texture, cx, cy, size, angle, tint, alpha)
