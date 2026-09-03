"""A trail: faded copies of a head along where it has been.

Stateless like everything else here -- the head's ``x``/``y`` curves are
re-evaluated at earlier times, so the trail follows exactly the path the
core took, and the recipe's direction turns the whole path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..curves import Curve
from . import POSITION, Param, color, over_into, premultiply, raw, val, window

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "length": Param("float", 0.25, 0.0, 3.0, label="seconds of history"),
    "samples": Param("int", 12, 1, 64),
    "radius": Param("curve", 8.0, 0.0, 256.0, label="px at the head"),
    "taper": Param("float", 0.8, 0.0, 1.0, label="thinner toward the tail"),
    "color_head": Param("color", "#FFD27A"),
    "color_tail": Param("color", "#B0301480"),
    "softness": Param("float", 0.6, 0.0, 1.0),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    length = val(layer, "length", ctx)
    samples = int(raw(layer, "samples"))
    radius = val(layer, "radius", ctx)
    if length <= 0.0 or samples <= 0 or radius <= 0.0:
        return None
    span = max(ctx.phase_seconds, 1e-3)
    xcurve = Curve.from_json(raw(layer, "x"))
    ycurve = Curve.from_json(raw(layer, "y"))
    ts = ctx.t - (np.arange(samples, dtype=np.float32) / max(samples - 1, 1)) * (length / span)
    ts = np.clip(ts, 0.0, 1.0)
    hx = xcurve.sample(ts)
    hy = ycurve.sample(ts)
    head = color(layer, "color_head")
    tail = color(layer, "color_tail")
    soft = max(val(layer, "softness", ctx), 0.02)
    taper = val(layer, "taper", ctx)
    out_a = np.zeros((ctx.height, ctx.width), dtype=np.float32)
    out_rgb = np.zeros((ctx.height, ctx.width, 3), dtype=np.float32)
    # Tail first, so the head paints over it.
    for i in range(samples - 1, -1, -1):
        u = i / max(samples - 1, 1)  # 0 at the head, 1 at the tail
        cx, cy = ctx.turn(float(hx[i]), float(hy[i]))
        r = radius * (1.0 - taper * u)
        if r <= 0.0:
            continue
        win = window(ctx, cx, cy, r)
        if win is None:
            continue
        d = np.sqrt((win.x - cx) ** 2 + (win.y - cy) ** 2) / np.float32(r)
        cov = np.clip((1.0 - d) / soft, 0.0, 1.0).astype(np.float32)
        rgb = head[:3] + (tail[:3] - head[:3]) * np.float32(u)
        cov *= np.float32(head[3] + (tail[3] - head[3]) * u)
        over_into(out_rgb[win.rows, win.cols], out_a[win.rows, win.cols], rgb, cov)
    return premultiply(out_rgb, out_a)
