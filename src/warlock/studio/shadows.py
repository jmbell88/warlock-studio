"""Blurred shadow sprites, nine-sliced (UX.md Phase 5, item 1).

Phase 2 approximated a blur with four concentric *stroked* bands, because
imgui's draw list has no blur and the frame loop may not grow one. This is the
real thing, and it is the cheapest of the four GPU-tier items for exactly the
reason the phase gives: the blur happens **once**, off the frame loop entirely,
into a texture -- so the per-frame cost is eight textured quads where it was
four stroked rounded rects, which is fewer vertices than it replaces.

Three decisions are load-bearing.

**The sprite is generated, not vendored.** It is a function of two numbers (the
surface's corner radius and how far the shadow reaches) and both are in
*physical* pixels, so a UI scale change and an elevation change are the same
kind of change: a different sprite, built on first use and cached under those
two numbers. A vendored PNG would be one scale's answer, and this app is
routinely run at 1.5.

**The centre is not drawn.** A nine-slice has nine pieces and this draws eight,
which is what lets one recipe sit under a card (drawn *before* the surface) and
under an imgui window (drawn *after* it, because ``begin`` has already painted
the background). It is the same property the stroked bands had, arrived at the
same way: leave the interior alone and the surface above cannot be darkened by
its own shadow. It also means the sprite's middle pixel is never sampled.

**Failure is permanent and silent.** No renderer, no GL context, a texture that
would not build: :func:`sprite` returns ``None`` for good and every call site
falls back to the Phase 2 bands. A shadow is not worth a traceback out of a
frame, and a fallback that is retried every frame on a broken host is a stall
rather than a degradation.

Pure in the way ``vram.py`` and ``memlog.py`` are, up to the one GL call:
:func:`pixels` is numpy and stdlib, imports nothing from ``service``, ``queue``
or the rest of ``studio`` bar ``tokens``, and is what the tests measure.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from . import ninepatch, tokens

log = logging.getLogger(__name__)

# The shadow's darkness at its darkest, before ``strength`` and before the
# elevation's own reach is applied. A blurred step edge reads about half its
# peak *at* the edge, so this is roughly twice the innermost stroked band's
# alpha -- which is what makes the two recipes read as the same depth rather
# than as a settings change.
PEAK = 0.5

# How far the shadow reaches past the surface, in design pixels, at elevation
# 1.0. The outermost stroked band's outer bound, so the two recipes occupy the
# same space and a call site cannot tell them apart by layout.
SPREAD = tokens.SHADOW_STEPS[-1][1]


def pixels(radius: float, spread: float) -> np.ndarray:
    """A blurred rounded rectangle, as ``uint8`` alpha. Pure.

    The rectangle is inset by ``spread`` on every side of a square sprite, so
    the blur has exactly the room it needs and the middle row and column are
    constant along their own axis -- which is what makes the nine-slice
    *correct* rather than merely convenient: the strip that gets stretched is
    the one the true shape is uniform along.
    """
    corner = _corner(radius, spread)
    size = 2 * corner + 1
    coords = np.arange(size, dtype=np.float64)
    low, high = spread, size - 1.0 - spread
    # Distance outside the rounded rect, per axis, so the corner test is one
    # hypot rather than four quadrant branches.
    dx = np.maximum(np.maximum(low + radius - coords, coords - (high - radius)), 0.0)
    inside_x = (coords >= low) & (coords <= high)
    mask_x = np.where(inside_x, 1.0, 0.0)
    grid_x = np.minimum(dx[None, :], radius)
    grid_y = np.minimum(dx[:, None], radius)
    corner_ok = np.hypot(grid_x, grid_y) <= radius
    mask = mask_x[None, :] * mask_x[:, None] * corner_ok
    return (_blur(mask, spread) * 255.0 + 0.5).astype(np.uint8)


def _corner(radius: float, spread: float) -> int:
    """The side of the sprite's corner block: the arc plus the whole blur."""
    return max(int(math.ceil(radius + spread)), 2)


def _blur(mask: np.ndarray, spread: float) -> np.ndarray:
    """Separable Gaussian, with the kernel sized off the reach asked for.

    ``spread`` is where the shadow is meant to be *gone*, and a Gaussian is
    ~0.006 of its peak at 2.5 sigma, so that is the sigma this reads back out.
    """
    sigma = max(spread / 2.5, 0.5)
    reach = max(int(math.ceil(sigma * 3.0)), 1)
    x = np.arange(-reach, reach + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, mask)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, out)


# Built sprites, keyed by the two numbers that decide one. Small and bounded:
# three elevations times the handful of radii the app draws, times a UI scale
# that changes rarely.
_SPRITES: dict[tuple[int, int], ninepatch.Sprite] = {}

# Set once, for good, when anything about building a sprite fails.
_BROKEN = False


def available() -> bool:
    """Whether the textured path is switched on and has not failed."""
    from . import effects

    return bool(effects.SOFT_SHADOWS) and not _BROKEN


def sprite(radius: float, spread: float) -> ninepatch.Sprite | None:
    """The sprite for a surface of this radius and reach, building it if new."""
    if not available():
        return None
    radius = max(float(radius), 0.0)
    spread = max(float(spread), 1.0)
    key = (int(round(radius)), int(round(spread)))
    found = _SPRITES.get(key)
    if found is not None:
        return found
    built = _build(float(key[0]), float(key[1]))
    if built is not None:
        _SPRITES[key] = built
    return built


def _build(radius: float, spread: float) -> ninepatch.Sprite | None:
    global _BROKEN

    if not ninepatch.have_renderer():
        # Not a failure: a headless test and the smoke suite both reach here
        # with no renderer, and both want the fallback rather than a broken
        # flag that outlives them.
        return None
    try:
        return ninepatch.build(pixels(radius, spread), _corner(radius, spread))
    except Exception:
        log.warning("soft shadows are unavailable; falling back to the drawn bands", exc_info=True)
        _BROKEN = True
        return None


def release_all() -> None:
    """Drop every built sprite. Frame thread only, at teardown or a rebuild."""
    ninepatch.release(_SPRITES.values())
    _SPRITES.clear()
