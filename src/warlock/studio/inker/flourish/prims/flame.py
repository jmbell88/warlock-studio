"""A flame: a tapered silhouette eroded by scrolling turbulence.

In flame-local coordinates ``u`` runs across the base and ``v`` runs from the
base (0) to the tip (1) along ``rise``. The silhouette is a half-width profile
that narrows to the tip; fbm scrolled along ``v`` eats into it, which is what
makes the tongues. Colour runs base to tip, and the same noise brightens the
hotter folds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import POSITION, Param, color, fbm_plane, premultiply, ramp, rotate_arrays, val, window

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "width": Param("curve", 24.0, 0.0, 512.0, label="px at the base"),
    "height": Param("curve", 48.0, 0.0, 1024.0, label="px to the tip"),
    "rise": Param("float", -90.0, -360.0, 360.0, label="degrees"),
    "color_base": Param("color", "#FFE08A"),
    "color_tip": Param("color", "#E0341C"),
    "turbulence": Param("float", 0.6, 0.0, 1.0),
    "speed": Param("float", 1.6, 0.0, 10.0, label="scroll"),
    "scale": Param("float", 12.0, 2.0, 128.0, label="px per cell"),
    "spikiness": Param("float", 0.5, 0.0, 1.0),
    "opacity": Param("curve", 1.0, 0.0, 1.0),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    width = val(layer, "width", ctx)
    height = val(layer, "height", ctx)
    if width <= 0.0 or height <= 0.0:
        return None
    cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))
    # The flame fits in a disc of its height about its base, whatever it faces.
    reach = max(width, height) * 1.1
    win = window(ctx, cx, cy, reach)
    if win is None:
        return None
    rise = val(layer, "rise", ctx) + ctx.direction
    lx, ly = rotate_arrays(win.x - cx, win.y - cy, -(rise + 90.0))
    u = lx / np.float32(width * 0.5)  # -1..1 across the base
    v = -ly / np.float32(height)  # 0 at the base, 1 at the tip

    scale = val(layer, "scale", ctx)
    drift = val(layer, "speed", ctx) * ctx.time
    # Noise scrolls toward the tip: sample the frame's fbm shifted along
    # the rise direction by ``drift`` cells.
    ang = np.deg2rad(rise)
    n = fbm_plane(
        ctx,
        ctx.lseed(),
        scale=scale,
        dx=-np.cos(ang) * drift * 2.0,
        dy=-np.sin(ang) * drift * 2.0,
        octaves=4,
        win=win,
    )
    turb = val(layer, "turbulence", ctx)
    spike = val(layer, "spikiness", ctx)
    profile = np.clip(1.0 - v, 0.0, 1.0) ** (0.6 + spike * 1.2)
    eroded = profile - (n - 0.35) * turb * (0.4 + v * 1.2)
    inside = (eroded - np.abs(u)) * np.float32(width * 0.5) * ctx.scale  # px to the edge
    alpha = np.clip(inside / 1.5, 0.0, 1.0)
    alpha = np.where((v < -0.05) | (v > 1.05), 0.0, alpha).astype(np.float32)
    alpha *= np.clip(v * 8.0 + 0.4, 0.0, 1.0)

    base = color(layer, "color_base")
    tip = color(layer, "color_tip")
    heat = np.clip(v + (n - 0.5) * 0.6, 0.0, 1.0)
    rgb = ramp(base, tip, heat)
    centre = np.clip(1.0 - np.abs(u), 0.0, 1.0)[..., None]
    hot = np.clip((0.5 - heat) * 2.0, 0.0, 1.0)[..., None] * centre
    rgb = rgb + (1.0 - rgb) * hot * 0.6
    opacity = val(layer, "opacity", ctx) * min(base[3], tip[3])
    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    out[win.rows, win.cols] = premultiply(np.clip(rgb, 0.0, 1.0), alpha * np.float32(opacity))
    return out
