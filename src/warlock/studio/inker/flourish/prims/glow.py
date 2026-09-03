"""Glow: a blurred, tinted copy of everything beneath, added on top.

Reads the composite below rather than any one layer, which is what makes it
one control for the whole effect: brighten the core and the glow brightens
with it. Blend ``add`` is what a glow means; the layer's blend is ignored.

Blurred at logical resolution: a glow is by definition low-frequency, so the
supersampled raster is reduced, blurred, and brought back up -- sixteen times
cheaper at 4x than blurring the raster, and indistinguishable after the bake's
own reduction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import Param, blur, color, downsample, upsample, val

REPLACES_BELOW = False
FORCE_BLEND = "add"

PARAMS = {
    "radius": Param("curve", 8.0, 0.0, 128.0, label="px"),
    "strength": Param("curve", 0.8, 0.0, 4.0),
    "tint": Param("color", "#FFFFFF"),
    "threshold": Param("float", 0.0, 0.0, 1.0, label="only above this brightness"),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    if below is None:
        return None
    strength = val(layer, "strength", ctx)
    radius = val(layer, "radius", ctx)
    if strength <= 0.0 or radius <= 0.0:
        return None
    s = int(ctx.scale)
    src = downsample(below, s)
    threshold = val(layer, "threshold", ctx)
    if threshold > 0.0:
        lum = src[..., :3].max(axis=-1)
        keep = np.clip((lum - threshold) / max(1.0 - threshold, 1e-3), 0.0, 1.0)
        src = src * keep[..., None]
    blurred = upsample(blur(src, radius), s, smooth=False)
    tint = color(layer, "tint")
    out = blurred * np.float32(strength)
    out[..., :3] *= tint[:3]
    out[..., 3] *= tint[3]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
