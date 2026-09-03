"""Smoke: drifting noise blobs that expand, rise, darken and fade.

Each blob is a soft disc whose coverage is modulated by fbm sampled in the
blob's own moving frame, so the raggedness travels with it. Like the sparks
it is closed-form in age: no stepping. Each blob paints only inside its own
window, and its noise is one small coarse plane.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import (
    POSITION,
    Param,
    color,
    curve,
    fbm_plane,
    hashed,
    over_into,
    particle_times,
    premultiply,
    raw,
    rotate,
    val,
    window,
)

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "count": Param("int", 10, 0, 80),
    "emission": Param("choice", "continuous", choices=("burst", "continuous")),
    "lifetime": Param("float", 1.2, 0.05, 6.0, label="seconds"),
    "size": Param("float", 10.0, 0.5, 256.0, label="px at birth"),
    "expand": Param("float", 2.2, 0.0, 8.0, label="x size over life"),
    "rise": Param("float", 40.0, -400.0, 400.0, label="px/s"),
    "rise_direction": Param("float", -90.0, -360.0, 360.0, label="degrees"),
    "drift": Param("float", 12.0, 0.0, 200.0, label="px/s sideways"),
    "spawn_radius": Param("float", 6.0, 0.0, 256.0, label="px"),
    "color": Param("color", "#3A3A3A"),
    "darken": Param("float", 0.5, 0.0, 1.0, label="over life"),
    "alpha_over_life": Param(
        "life", {"keys": [[0.0, 0.0], [0.15, 0.7], [1.0, 0.0]]}, 0.0, 1.0
    ),
    "raggedness": Param("float", 0.6, 0.0, 1.0),
    "noise_scale": Param("float", 10.0, 1.0, 128.0, label="px per cell"),
}


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    count = int(raw(layer, "count"))
    if count <= 0:
        return None
    seed = ctx.lseed()
    span = max(ctx.phase_seconds, 1e-3)
    born = particle_times(count, span, str(raw(layer, "emission")), seed)
    life = val(layer, "lifetime", ctx)
    age = ctx.phase_time - born
    u_all = age / life
    alive = np.nonzero((u_all >= 0.0) & (u_all < 1.0))[0]
    if len(alive) == 0:
        return None

    rise_dir = val(layer, "rise_direction", ctx) + ctx.direction
    rx, ry = rotate(1.0, 0.0, rise_dir)
    rise = val(layer, "rise", ctx)
    drift = val(layer, "drift", ctx)
    spawn = val(layer, "spawn_radius", ctx)
    size0 = val(layer, "size", ctx)
    expand = val(layer, "expand", ctx)
    rag = val(layer, "raggedness", ctx)
    nscale = val(layer, "noise_scale", ctx)
    darken = val(layer, "darken", ctx)
    alpha_curve = curve(layer, "alpha_over_life")
    col = color(layer, "color")
    cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))

    out_a = np.zeros((ctx.height, ctx.width), dtype=np.float32)
    out_rgb = np.zeros((ctx.height, ctx.width, 3), dtype=np.float32)
    ja = hashed(seed + 1, count)
    jr = hashed(seed + 2, count)
    jd = hashed(seed + 3, count)
    for i in alive:
        a_i = float(age[i])
        u = float(u_all[i])
        sa = float(ja[i]) * 2.0 * np.pi
        sr = float(np.sqrt(jr[i])) * spawn
        side = (float(jd[i]) - 0.5) * 2.0 * drift
        px = cx + np.cos(sa) * sr + rx * rise * a_i + (-ry) * side * a_i
        py = cy + np.sin(sa) * sr + ry * rise * a_i + rx * side * a_i
        radius = size0 * (1.0 + expand * u)
        win = window(ctx, px, py, radius * (1.0 + 0.8 * rag))
        if win is None:
            continue
        d = np.sqrt((win.x - px) ** 2 + (win.y - py) ** 2) / np.float32(radius)
        if rag > 0.0:
            n = fbm_plane(
                ctx,
                seed + int(i),
                scale=nscale,
                dx=-px / nscale + float(ja[i]) * 37.0,
                dy=-py / nscale + a_i * 0.4,
                octaves=3,
                win=win,
            )
            d = d * (1.0 + (n - 0.5) * 1.6 * rag)
        cov = np.clip(1.0 - d, 0.0, 1.0)
        cov = cov * cov * np.float32(float(alpha_curve.at(u)) * col[3])
        rgb = col[:3] * np.float32(1.0 - darken * u)
        over_into(out_rgb[win.rows, win.cols], out_a[win.rows, win.cols], rgb, cov)
    return premultiply(out_rgb, out_a)
