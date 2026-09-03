"""The primitives: one module per kind, one ``render`` each.

A primitive is a module with three names. ``PARAMS`` declares every parameter
it reads -- default, range, and whether it is a plain number, a curve over the
phase, a curve over a particle's life, a colour or a choice -- so the recipe
codec can clamp what it loads and the inspector can draw a control without a
table of its own. ``REPLACES_BELOW`` says whether the layer's output *is* the
composite beneath it, transformed (a distortion), rather than a plane to blend
over it. ``render(layer, ctx, below)`` returns one premultiplied float32
``(H, W, 4)`` plane at the frame's supersampled size, or ``None`` for nothing.

Every render is a pure function of ``(layer, ctx)``: a primitive has no state
between frames. A particle system does not step -- it computes where a particle
born at ``t0`` is at ``t`` -- and a trail does not remember, it re-evaluates
the head's curve at earlier times. That is what makes frame 40 renderable
before frame 39, a single slider change re-render only the layer it touched,
and the whole thing deterministic.

Coordinates are *logical* pixels with the origin at the canvas centre and +y
down, scaled by ``ctx.scale`` when rasterised. Directions are degrees with 0
pointing right and 90 pointing down (screen convention); ``ctx.direction``
rotates every vector a primitive emits, which is how eight facings come from
one recipe.

Two cost rules, both measured (a 128px frame at 4x is 262,144 pixels, and the
first draft spent a second per frame on nine layers). **Noise is sampled at
logical resolution and upsampled** -- it is smooth by construction at cells of
several pixels, so the supersampled raster adds nothing but sixteen times the
arithmetic -- and **a disc paints only inside its window**: a smoke blob or a
trail sample touches a few thousand pixels, not the whole plane.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import noise
from ..curves import Curve

#: ``kind`` -> the values a parameter may take.
PARAM_KINDS = ("float", "int", "curve", "life", "color", "choice", "bool", "asset")


@dataclass(frozen=True)
class Param:
    kind: str
    default: Any
    lo: float = 0.0
    hi: float = 1.0
    choices: tuple[str, ...] = ()
    #: For the inspector: what the number means.
    label: str = ""

    def clamp(self, raw: Any) -> Any:
        """The stored form of ``raw``, inside this parameter's range."""
        if self.kind in ("curve", "life"):
            try:
                curve = Curve.from_json(raw)
            except (TypeError, ValueError):
                curve = Curve.from_json(self.default)
            return curve.clamped(self.lo, self.hi).to_json()
        if self.kind == "float":
            try:
                return min(self.hi, max(self.lo, float(raw)))
            except (TypeError, ValueError):
                return float(self.default)
        if self.kind == "int":
            try:
                return int(min(self.hi, max(self.lo, int(raw))))
            except (TypeError, ValueError):
                return int(self.default)
        if self.kind == "bool":
            return bool(raw)
        if self.kind == "color":
            try:
                parse_color(str(raw))
            except ValueError:
                return str(self.default)
            return str(raw)
        if self.kind == "choice":
            return str(raw) if str(raw) in self.choices else str(self.default)
        if self.kind == "asset":
            # An id the document resolves; anything else renders nothing.
            text = str(raw or "")
            return text if len(text) <= 64 and text.replace("_", "").isalnum() else ""
        raise ValueError(f"unknown parameter kind {self.kind!r}")


#: Shared by every primitive that has a position.
POSITION = {
    "x": Param("curve", 0.0, -512.0, 512.0, label="px from centre"),
    "y": Param("curve", 0.0, -512.0, 512.0, label="px from centre"),
}

KINDS = (
    "core",
    "flame",
    "particles",
    "smoke",
    "ring",
    "flash",
    "glow",
    "trail",
    "distortion",
    "sprite",
)

_MODULES: dict[str, Any] = {}


def module(kind: str) -> Any:
    """The primitive module for ``kind``; imported on first use."""
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not a primitive this build knows")
    mod = _MODULES.get(kind)
    if mod is None:
        mod = importlib.import_module(f"{__name__}.{kind}")
        _MODULES[kind] = mod
    return mod


def params_of(kind: str) -> dict[str, Param]:
    return dict(module(kind).PARAMS)


# -- reading a layer's parameters --------------------------------------------


def raw(layer: Any, name: str) -> Any:
    spec = params_of(layer.kind).get(name)
    if spec is None:
        raise KeyError(f"{layer.kind} has no parameter {name!r}")
    return layer.params.get(name, spec.default)


def val(layer: Any, name: str, ctx: Any) -> float:
    """A number now: a plain parameter, or a curve sampled at the phase's ``t``."""
    spec = params_of(layer.kind)[name]
    value = layer.params.get(name, spec.default)
    if spec.kind == "curve":
        return Curve.from_json(value).at(ctx.t)
    return float(value)


def curve(layer: Any, name: str) -> Curve:
    return Curve.from_json(raw(layer, name))


def color(layer: Any, name: str) -> np.ndarray:
    return parse_color(str(raw(layer, name)))


# -- colour ---------------------------------------------------------------------


def parse_color(text: str) -> np.ndarray:
    """``#RRGGBB`` or ``#RRGGBBAA`` -> float32 ``[r, g, b, a]`` in 0..1."""
    s = text.strip().lstrip("#")
    if len(s) == 6:
        s += "ff"
    if len(s) != 8:
        raise ValueError(f"not a colour: {text!r}")
    try:
        vals = [int(s[i : i + 2], 16) / 255.0 for i in range(0, 8, 2)]
    except ValueError as exc:
        raise ValueError(f"not a colour: {text!r}") from exc
    return np.asarray(vals, dtype=np.float32)


def ramp(colour_a: np.ndarray, colour_b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``(..., 3)`` colours blended from ``a`` at ``u=0`` to ``b`` at ``u=1``."""
    u = np.clip(u, 0.0, 1.0).astype(np.float32)[..., None]
    return colour_a[:3] + (colour_b[:3] - colour_a[:3]) * u


# -- geometry -------------------------------------------------------------------


def rotate(x: float, y: float, degrees: float) -> tuple[float, float]:
    """A vector turned by ``degrees`` in screen space (+y down)."""
    if degrees == 0.0:
        return x, y
    a = np.deg2rad(degrees)
    c, s = float(np.cos(a)), float(np.sin(a))
    return x * c - y * s, x * s + y * c


def rotate_arrays(x: np.ndarray, y: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    if degrees == 0.0:
        return x, y
    a = np.deg2rad(degrees)
    c, s = np.float32(np.cos(a)), np.float32(np.sin(a))
    return x * c - y * s, x * s + y * c


@dataclass(frozen=True)
class Window:
    """A rectangle of the raster, with the logical coordinates inside it."""

    y0: int
    y1: int
    x0: int
    x1: int
    x: np.ndarray  # logical coords, shape (y1-y0, x1-x0)
    y: np.ndarray

    @property
    def rows(self) -> slice:
        return slice(self.y0, self.y1)

    @property
    def cols(self) -> slice:
        return slice(self.x0, self.x1)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)


def window(
    ctx: Any, cx: float, cy: float, half_w: float, half_h: float | None = None
) -> Window | None:
    """The raster rectangle covering ``(cx, cy) +- (half_w, half_h)`` logical px,
    clipped to the frame; ``None`` when nothing of it is on the raster."""
    if half_h is None:
        half_h = half_w
    s = ctx.scale
    x0 = int(np.floor((cx - half_w) * s + ctx.width / 2.0)) - 1
    x1 = int(np.ceil((cx + half_w) * s + ctx.width / 2.0)) + 2
    y0 = int(np.floor((cy - half_h) * s + ctx.height / 2.0)) - 1
    y1 = int(np.ceil((cy + half_h) * s + ctx.height / 2.0)) + 2
    x0, x1 = max(0, x0), min(ctx.width, x1)
    y0, y1 = max(0, y0), min(ctx.height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    x, y = ctx.coords()
    return Window(y0, y1, x0, x1, x[y0:y1, x0:x1], y[y0:y1, x0:x1])


def disc(ctx: Any, cx: float, cy: float, radius: float, softness: float = 0.5) -> np.ndarray:
    """Coverage in 0..1 for a disc of ``radius`` logical px at ``(cx, cy)``,
    as a full-frame plane (painted only inside its window).

    ``softness`` is the fraction of the radius over which the edge fades: 0 is a
    hard (still anti-aliased) edge, 1 fades from the centre.
    """
    out = np.zeros((ctx.height, ctx.width), dtype=np.float32)
    if radius <= 0.0:
        return out
    win = window(ctx, cx, cy, radius)
    if win is None:
        return out
    out[win.rows, win.cols] = disc_in(win, cx, cy, radius, softness, ctx.scale)
    return out


def disc_in(
    win: Window, cx: float, cy: float, radius: float, softness: float, scale: float
) -> np.ndarray:
    d = np.sqrt((win.x - cx) ** 2 + (win.y - cy) ** 2) / np.float32(radius)
    return _falloff(d, softness, radius * scale)


def _falloff(d: np.ndarray, softness: float, radius_px: float) -> np.ndarray:
    inner = 1.0 - min(max(softness, 0.0), 1.0)
    aa = 1.0 / max(radius_px, 1e-3)  # one raster pixel of anti-aliasing on a hard edge
    lo = max(inner, 0.0)
    hi = max(1.0, lo + aa)
    return np.clip((hi - d) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


# -- noise at logical resolution -------------------------------------------------


def fbm_plane(
    ctx: Any,
    seed: int,
    *,
    scale: float,
    dx: float = 0.0,
    dy: float = 0.0,
    octaves: int = 3,
    win: Window | None = None,
    smooth: bool = True,
) -> np.ndarray:
    """fbm over the frame in 0..1, ``scale`` logical px per cell, offset by
    ``(dx, dy)`` cells, sampled at logical resolution and upsampled to the
    raster. With ``win`` only that rectangle is returned. ``smooth=False``
    repeats samples instead of interpolating -- right for anything the bake's
    own reduction will average back to one logical pixel anyway."""
    s = int(ctx.scale)
    scale = max(float(scale), 1e-3)
    if win is None:
        cy0, cy1, cx0, cx1 = 0, ctx.height // s, 0, ctx.width // s
    else:
        cy0, cy1 = win.y0 // s, -(-win.y1 // s)
        cx0, cx1 = win.x0 // s, -(-win.x1 // s)
    lx, ly = ctx.coarse()
    x = lx[cy0:cy1, cx0:cx1] / np.float32(scale) + np.float32(dx)
    y = ly[cy0:cy1, cx0:cx1] / np.float32(scale) + np.float32(dy)
    coarse = noise.fbm(x, y, seed, octaves=octaves)
    full = upsample(coarse, s, smooth=smooth)
    if win is None:
        return full
    return full[win.y0 - cy0 * s : win.y1 - cy0 * s, win.x0 - cx0 * s : win.x1 - cx0 * s]


def upsample(plane: np.ndarray, s: int, *, smooth: bool = True) -> np.ndarray:
    """Nearest repeat by ``s``, then -- when ``smooth`` -- one box pass of
    radius ``s // 2``: close to bilinear, and every step exact arithmetic on
    the input."""
    if s <= 1:
        return plane
    big = np.repeat(np.repeat(plane, s, axis=0), s, axis=1)
    return blur(big, s // 2, passes=1) if smooth else big


def downsample(plane: np.ndarray, s: int) -> np.ndarray:
    """Box mean by ``s`` on the leading two axes."""
    if s <= 1:
        return plane
    h, w = plane.shape[0] // s, plane.shape[1] // s
    trimmed = plane[: h * s, : w * s]
    if plane.ndim == 3:
        return trimmed.reshape(h, s, w, s, plane.shape[2]).mean(axis=(1, 3), dtype=np.float32)
    return trimmed.reshape(h, s, w, s).mean(axis=(1, 3), dtype=np.float32)


# -- blur -----------------------------------------------------------------------


def blur(plane: np.ndarray, radius_px: float, passes: int = 3) -> np.ndarray:
    """A box blur repeated ``passes`` times: close to Gaussian, fully deterministic.

    ``radius_px`` is in raster pixels. Works on ``(H, W)`` and ``(H, W, C)`` alike.
    """
    r = int(round(radius_px))
    if r <= 0:
        return plane
    out = plane.astype(np.float32, copy=True)
    for _ in range(max(1, passes)):
        out = _box1d(out, r, axis=1)
        out = _box1d(out, r, axis=0)
    return out


def _box1d(arr: np.ndarray, r: int, axis: int) -> np.ndarray:
    n = arr.shape[axis]
    if n == 0:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    padded = np.pad(arr, pad, mode="constant")
    csum = np.cumsum(padded, axis=axis, dtype=np.float32)
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    csum = np.concatenate([np.zeros(zero_shape, dtype=np.float32), csum], axis=axis)
    hi = np.take(csum, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(csum, np.arange(0, n), axis=axis)
    return (hi - lo) / np.float32(2 * r + 1)


# -- planes ---------------------------------------------------------------------


def premultiply(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """``rgb`` ``(..., 3)`` and ``alpha`` ``(...)`` -> premultiplied ``(..., 4)``."""
    out = np.empty(alpha.shape + (4,), dtype=np.float32)
    out[..., :3] = rgb * alpha[..., None]
    out[..., 3] = alpha
    return out


def over_into(out_rgb: np.ndarray, out_a: np.ndarray, rgb: np.ndarray, cov: np.ndarray) -> None:
    """Paint a straight-colour patch under coverage ``cov`` onto a straight
    ``(rgb, a)`` pair in place -- the accumulator the blob primitives share."""
    out_rgb *= (1.0 - cov)[..., None]
    out_rgb += rgb * cov[..., None]
    out_a += cov - out_a * cov


# -- particles -------------------------------------------------------------------


def particle_times(count: int, span: float, emission: str, seed: int) -> np.ndarray:
    """Birth time of each of ``count`` particles within a phase of ``span`` seconds.

    ``burst`` births everything at 0; ``continuous`` spreads births evenly over
    the phase with a deterministic jitter so the stream does not pulse.
    """
    if count <= 0:
        return np.zeros((0,), dtype=np.float32)
    if emission == "burst":
        return np.zeros((count,), dtype=np.float32)
    idx = np.arange(count, dtype=np.float32)
    jitter = hashed(seed + 977, count) - 0.5
    return ((idx + 0.5 + jitter * 0.8) / count * span).astype(np.float32)


def hashed(seed: int, count: int) -> np.ndarray:
    """``count`` deterministic uniforms in [0, 1) for ``seed``."""
    ix = np.arange(count, dtype=np.int64)
    return noise.hash_lattice(ix, np.zeros_like(ix), seed)


# -- stamping a texture ----------------------------------------------------------------


def stamp(
    ctx: Any,
    texture: np.ndarray,
    cx: float,
    cy: float,
    width: float,
    degrees: float,
    tint: np.ndarray,
    alpha: float,
) -> np.ndarray | None:
    """``texture`` (straight uint8 RGBA) centred at logical ``(cx, cy)``,
    ``width`` logical px wide, turned by ``degrees``, as a premultiplied
    full-frame plane. Nearest sampling, so a pixel-art texture stays one.
    ``None`` when nothing of it lands on the raster."""
    th, tw = texture.shape[:2]
    if tw == 0 or th == 0 or width <= 0.0 or alpha <= 0.0:
        return None
    scale = width / tw  # logical px per texel
    height = th * scale
    reach = 0.5 * (width**2 + height**2) ** 0.5
    win = window(ctx, cx, cy, reach)
    if win is None:
        return None
    # Raster -> texture space: undo the rotation about the centre, then scale.
    dx, dy = rotate_arrays(win.x - cx, win.y - cy, -degrees)
    u = np.floor(dx / scale + tw / 2.0).astype(np.int64)
    v = np.floor(dy / scale + th / 2.0).astype(np.int64)
    inside = (u >= 0) & (u < tw) & (v >= 0) & (v < th)
    if not inside.any():
        return None
    uu = np.clip(u, 0, tw - 1)
    vv = np.clip(v, 0, th - 1)
    texel = texture[vv, uu].astype(np.float32) / 255.0
    cov = texel[..., 3] * inside.astype(np.float32) * np.float32(alpha * tint[3])
    rgb = texel[..., :3] * tint[:3]
    out = np.zeros((ctx.height, ctx.width, 4), dtype=np.float32)
    out[win.rows, win.cols] = premultiply(rgb, cov)
    return out
