"""Seeded value noise, fractal sums and domain warping. Pure numpy.

Every organic edge in this package -- a flame's silhouette, a smoke blob's
raggedness, a shockwave's unevenness -- comes from here, and the one property
that matters more than the look is that the same ``(x, y, seed)`` gives the
same number on every machine. So the lattice values are an integer hash rather
than a random-number generator (whose stream order is an implementation detail)
and the interpolation is smoothstep in float32 with no fused-multiply-add
opportunities worth worrying about at this precision.

Value noise rather than simplex or Perlin: the gradient variants look better
at one octave, but under three or four octaves of fbm the difference is gone,
and value noise is a tenth of the code. The whole file is what a test can
read in one sitting.
"""

from __future__ import annotations

import numpy as np

_M1 = np.uint32(0x85EBCA6B)
_M2 = np.uint32(0xC2B2AE35)
_M3 = np.uint32(0x9E3779B1)


def hash_lattice(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Lattice value in [0, 1) for integer coordinates. Murmur-style mix."""
    with np.errstate(over="ignore"):
        h = ix.astype(np.uint32) * _M1
        h ^= iy.astype(np.uint32) * _M2
        h ^= np.uint32(seed & 0xFFFFFFFF) * _M3
        h ^= h >> np.uint32(15)
        h *= _M1
        h ^= h >> np.uint32(13)
        h *= _M2
        h ^= h >> np.uint32(16)
    return (h.astype(np.float64) / 4294967296.0).astype(np.float32)


def value2d(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth value noise in [0, 1] at float coordinates ``x``, ``y``."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0
    ix = x0.astype(np.int64)
    iy = y0.astype(np.int64)
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    a = hash_lattice(ix, iy, seed)
    b = hash_lattice(ix + 1, iy, seed)
    c = hash_lattice(ix, iy + 1, seed)
    d = hash_lattice(ix + 1, iy + 1, seed)
    top = a + (b - a) * sx
    bottom = c + (d - c) * sx
    return (top + (bottom - top) * sy).astype(np.float32)


def fbm(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    *,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
) -> np.ndarray:
    """Fractal sum of ``octaves`` value-noise layers, normalised to [0, 1]."""
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float32)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for i in range(max(1, int(octaves))):
        total += amp * value2d(x * freq, y * freq, seed + 131 * i)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / np.float32(norm)


def warp(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    *,
    amount: float,
    octaves: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Domain warp: offset each coordinate by noise, so what is sampled at the
    warped coordinates folds and curls instead of merely rippling."""
    if amount <= 0.0:
        return x, y
    dx = fbm(x, y, seed + 7, octaves=octaves) - 0.5
    dy = fbm(x, y, seed + 19, octaves=octaves) - 0.5
    return x + dx * (2.0 * amount), y + dy * (2.0 * amount)


def grid(width: int, height: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate planes for a ``height`` x ``width`` raster, ``scale`` pixels per
    noise cell, centred on the raster so a rotation happens about the middle."""
    scale = max(1e-3, float(scale))
    xs = (np.arange(width, dtype=np.float32) - width / 2.0) / scale
    ys = (np.arange(height, dtype=np.float32) - height / 2.0) / scale
    return np.meshgrid(xs, ys)
