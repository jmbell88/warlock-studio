"""Sparks, embers, debris: a stateless particle emitter.

Each particle is born at a deterministic time with a deterministic direction,
speed and size, and its position at any later time is closed-form --
``p = p0 + v0 * age + g * age^2 / 2``, with drag as an exponential on the
velocity term. No stepping, no state: the frame after can be rendered before
the frame before, and a slider change re-renders only this layer.

The two curves over *life* -- ``size_over_life`` and ``alpha_over_life`` --
take a particle's age as 0..1, so one set of keys serves a burst of sparks and
a stream of embers alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..curves import Curve
from . import POSITION, Param, color, curve, hashed, particle_times, ramp, raw, rotate_arrays, val

REPLACES_BELOW = False

PARAMS = {
    **POSITION,
    "count": Param("int", 24, 0, 400),
    "emission": Param("choice", "burst", choices=("burst", "continuous")),
    "lifetime": Param("float", 0.5, 0.02, 5.0, label="seconds"),
    "lifetime_jitter": Param("float", 0.4, 0.0, 1.0),
    "speed": Param("float", 90.0, 0.0, 2000.0, label="px/s"),
    "speed_jitter": Param("float", 0.5, 0.0, 1.0),
    "direction": Param("float", 0.0, -360.0, 360.0, label="degrees"),
    "spread": Param("float", 360.0, 0.0, 360.0, label="degrees"),
    "gravity": Param("float", 120.0, -2000.0, 2000.0, label="px/s^2"),
    "drag": Param("float", 1.5, 0.0, 20.0),
    "spawn_radius": Param("float", 2.0, 0.0, 256.0, label="px"),
    "size": Param("float", 2.0, 0.1, 64.0, label="px"),
    "size_jitter": Param("float", 0.5, 0.0, 1.0),
    "size_over_life": Param("life", {"keys": [[0.0, 1.0], [1.0, 0.2]]}, 0.0, 4.0),
    "alpha_over_life": Param(
        "life", {"keys": [[0.0, 1.0], [0.7, 1.0], [1.0, 0.0]]}, 0.0, 1.0
    ),
    "color_start": Param("color", "#FFF4C0"),
    "color_end": Param("color", "#FF5A1A"),
    "softness": Param("float", 0.4, 0.0, 1.0),
    "streak": Param("float", 0.0, 0.0, 1.0, label="stretch along motion"),
    "texture": Param("asset", "", label="asset id, instead of a disc"),
    "spin": Param("float", 0.0, -1440.0, 1440.0, label="degrees per second, textured"),
}


def _state(layer: Any, ctx: Any) -> dict[str, np.ndarray] | None:
    count = int(raw(layer, "count"))
    if count <= 0:
        return None
    seed = ctx.lseed()
    span = max(ctx.phase_seconds, 1e-3)
    born = particle_times(count, span, str(raw(layer, "emission")), seed)
    age = ctx.phase_time - born
    life = val(layer, "lifetime", ctx) * (
        1.0 + (hashed(seed + 1, count) - 0.5) * 2.0 * val(layer, "lifetime_jitter", ctx)
    )
    life = np.maximum(life, 0.02).astype(np.float32)
    alive = (age >= 0.0) & (age < life)
    if not np.any(alive):
        return None
    idx = np.nonzero(alive)[0]
    age = age[idx]
    life = life[idx]
    u = age / life

    spread = val(layer, "spread", ctx)
    direction = val(layer, "direction", ctx) + ctx.direction
    ang = direction + (hashed(seed + 2, count)[idx] - 0.5) * spread
    speed = val(layer, "speed", ctx) * (
        1.0 + (hashed(seed + 3, count)[idx] - 0.5) * 2.0 * val(layer, "speed_jitter", ctx)
    )
    rad = np.deg2rad(ang).astype(np.float32)
    vx = np.cos(rad) * speed
    vy = np.sin(rad) * speed

    drag = val(layer, "drag", ctx)
    if drag > 0.0:
        # Integrated velocity under exponential drag.
        k = np.float32(drag)
        travel = (1.0 - np.exp(-k * age)) / k
    else:
        travel = age
    g = val(layer, "gravity", ctx)
    gx, gy = rotate_arrays(np.zeros_like(age), np.full_like(age, g), ctx.direction)

    spawn = val(layer, "spawn_radius", ctx)
    sa = hashed(seed + 4, count)[idx] * 2.0 * np.pi
    sr = np.sqrt(hashed(seed + 5, count)[idx]) * spawn
    cx, cy = ctx.turn(val(layer, "x", ctx), val(layer, "y", ctx))
    px = cx + np.cos(sa) * sr + vx * travel + 0.5 * gx * age * age
    py = cy + np.sin(sa) * sr + vy * travel + 0.5 * gy * age * age

    size = val(layer, "size", ctx) * (
        1.0 + (hashed(seed + 6, count)[idx] - 0.5) * 2.0 * val(layer, "size_jitter", ctx)
    )
    size = size * curve(layer, "size_over_life").sample(u)
    alpha = curve(layer, "alpha_over_life").sample(u)
    # Instantaneous velocity, for streaks.
    damp = np.exp(-np.float32(drag) * age) if drag > 0.0 else np.ones_like(age)
    ivx = vx * damp + gx * age
    ivy = vy * damp + gy * age
    return {
        "x": px.astype(np.float32),
        "y": py.astype(np.float32),
        "size": np.maximum(size, 0.05).astype(np.float32),
        "alpha": alpha.astype(np.float32),
        "u": u.astype(np.float32),
        "vx": ivx.astype(np.float32),
        "vy": ivy.astype(np.float32),
    }


def render(layer: Any, ctx: Any, below: np.ndarray | None) -> np.ndarray | None:
    st = _state(layer, ctx)
    if st is None:
        return None
    texture = ctx.asset(str(layer.params.get("texture", "")))
    if texture is not None:
        return _render_textured(layer, ctx, st, texture)
    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    c0 = color(layer, "color_start")
    c1 = color(layer, "color_end")
    softness = max(val(layer, "softness", ctx), 0.0)
    streak = val(layer, "streak", ctx)
    s = ctx.scale
    half_w = ctx.width / 2.0
    half_h = ctx.height / 2.0
    for i in range(len(st["x"])):
        r = st["size"][i]
        a = st["alpha"][i] * min(c0[3], c1[3])
        if a <= 0.0 or r <= 0.0:
            continue
        rgb = ramp(c0, c1, np.asarray(st["u"][i]))
        px = st["x"][i] * s + half_w
        py = st["y"][i] * s + half_h
        rp = r * s
        # Streak: stretch along the velocity by up to 4x.
        vx, vy = float(st["vx"][i]), float(st["vy"][i])
        vlen = (vx * vx + vy * vy) ** 0.5
        stretch = 1.0 + streak * min(vlen / 60.0, 3.0)
        ex = rp * stretch + 1.0
        x0 = int(max(0, np.floor(px - ex)))
        x1 = int(min(ctx.width, np.ceil(px + ex) + 1))
        y0 = int(max(0, np.floor(py - ex)))
        y1 = int(min(ctx.height, np.ceil(py + ex) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs.astype(np.float32) - px
        dy = ys.astype(np.float32) - py
        if stretch > 1.0 and vlen > 0.0:
            ux, uy = vx / vlen, vy / vlen
            along = dx * ux + dy * uy
            across = -dx * uy + dy * ux
            d = np.sqrt((along / stretch) ** 2 + across**2) / rp
        else:
            d = np.sqrt(dx * dx + dy * dy) / rp
        inner = 1.0 - softness
        aa = 1.0 / max(rp, 1e-3)
        cov = np.clip((max(1.0, inner + aa) - d) / max(1.0 - inner, aa), 0.0, 1.0)
        cov = cov.astype(np.float32) * np.float32(a)
        tile = out[y0:y1, x0:x1]
        # Additive within the layer: overlapping sparks brighten.
        tile[..., :3] += rgb * cov[..., None]
        tile[..., 3] = tile[..., 3] + cov - tile[..., 3] * cov
    np.clip(out, 0.0, 1.0, out=out)
    return out


def default_life_curve(name: str) -> Curve:
    return Curve.from_json(PARAMS[name].default)


def _render_textured(
    layer: Any, ctx: Any, st: dict[str, np.ndarray], texture: np.ndarray
) -> np.ndarray:
    """Each particle is the texture, ``size`` px wide, tinted along the colour
    ramp and turned by ``spin`` over its age; additive within the layer, like
    the discs."""
    from . import stamp

    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    c0 = color(layer, "color_start")
    c1 = color(layer, "color_end")
    spin = val(layer, "spin", ctx)
    seed = ctx.lseed(17)
    phases = hashed(seed, len(st["x"])) * 360.0
    for i in range(len(st["x"])):
        a = float(st["alpha"][i]) * min(c0[3], c1[3])
        width = float(st["size"][i]) * 2.0
        if a <= 0.0 or width <= 0.0:
            continue
        tint = np.append(ramp(c0, c1, np.asarray(st["u"][i])), 1.0).astype(np.float32)
        angle = float(phases[i]) + spin * float(st["u"][i]) * float(ctx.phase_seconds)
        plane = stamp(ctx, texture, float(st["x"][i]), float(st["y"][i]), width, angle, tint, a)
        if plane is None:
            continue
        out[..., :3] += plane[..., :3]
        out[..., 3] = out[..., 3] + plane[..., 3] - out[..., 3] * plane[..., 3]
    np.clip(out, 0.0, 1.0, out=out)
    return out
