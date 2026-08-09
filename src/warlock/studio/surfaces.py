"""Continuous-curvature corners for the surfaces that are big enough to show
them (UX.md Phase 5, item 4).

A circular corner joins a straight edge with a step change in curvature -- zero
along the edge, one over the radius the instant the arc starts -- and at a 10 px
radius on a card that reads as a slightly pinched corner. A superellipse
(``|x|^n + |y|^n = 1``) ramps the curvature instead, which is the "squircle"
every platform toolkit with a design team has converged on.

**It is a nine-patch, not a shader.** The phase's own sketch was an SDF fragment
shader through a custom draw callback, and the mask is the cheaper half of that
idea with the same output: the shape is a function of one number (the corner
radius, in physical pixels), so it is one small texture per radius, built once,
stretched. A shader would mean a second program, a second vertex format and a
draw callback that suspends imgui's own batching -- for a shape that does not
change.

**Only above a radius the difference is visible at.** The phase's own rule:
controls at 4 to 6 px stay circular, because below about eight the superellipse
and the arc differ by less than a pixel and the whole pipeline would be paid for
nothing. :func:`enabled_for` is that threshold, in one place, so a call site
cannot disagree with it.

Pure up to the upload, like :mod:`.shadows`: :func:`pixels` is numpy and
stdlib, and it is what the tests measure.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from . import ninepatch

log = logging.getLogger(__name__)

# The exponent. 5 is the usual reading of Apple's continuous corner: 2 is a
# circle exactly, 4 is visibly boxy at a large radius, and past about 6 the
# corner stops looking round at all.
EXPONENT = 5.0

# Below this radius, in physical pixels, the shape is not drawn from a sprite
# at all. See the module docstring: the difference is under a pixel and the
# sprite would be four texels a side.
MIN_RADIUS = 8.0

# Samples per pixel per axis when the mask is rasterised. Four is enough that
# the edge reads as anti-aliased at every radius the app draws; the cost is a
# 4x4 grid over a sprite that is at most ninety pixels a side, once.
SUPERSAMPLE = 4


def pixels(radius: float, exponent: float = EXPONENT) -> np.ndarray:
    """A superellipse-cornered square, as ``uint8`` coverage. Pure.

    The sprite is exactly two corners plus the one middle pixel that gets
    stretched, so the shape it describes is "a rectangle whose corners are
    superelliptical arcs of this radius" at *every* size it is drawn at.
    """
    corner = max(int(math.ceil(radius)), 2)
    size = 2 * corner + 1
    # Sample centres, supersampled: the edge of the shape falls between pixels
    # at most radii, and a single sample per pixel makes that edge crawl by a
    # whole pixel as the radius changes.
    step = 1.0 / SUPERSAMPLE
    fine = np.arange(size * SUPERSAMPLE, dtype=np.float64) * step + step * 0.5
    # Distance into the corner block, per axis, measured from the arc's own
    # origin -- so a point on a straight edge has zero on one axis and the test
    # below reduces to "inside", which is what makes this one expression rather
    # than four quadrant branches.
    span = float(size)
    dx = np.maximum(np.maximum(radius - fine, fine - (span - radius)), 0.0) / max(radius, 1e-6)
    implicit = np.power(dx[None, :], exponent) + np.power(dx[:, None], exponent)
    inside = (implicit <= 1.0).astype(np.float64)
    # Average each pixel's samples back down.
    coverage = inside.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE).mean(axis=(1, 3))
    return (coverage * 255.0 + 0.5).astype(np.uint8)


_SPRITES: dict[int, ninepatch.Sprite] = {}
_BROKEN = False


def available() -> bool:
    """Whether the effect is switched on and has not failed."""
    from . import effects

    return bool(effects.SQUIRCLES) and not _BROKEN


def enabled_for(radius: float) -> bool:
    """Whether a surface of this radius gets the squircle at all."""
    return available() and radius >= MIN_RADIUS


def sprite(radius: float) -> ninepatch.Sprite | None:
    """The sprite for this corner radius, building it if it is new."""
    if not enabled_for(radius):
        return None
    key = int(round(radius))
    found = _SPRITES.get(key)
    if found is not None:
        return found
    built = _build(float(key))
    if built is not None:
        _SPRITES[key] = built
    return built


def _build(radius: float) -> ninepatch.Sprite | None:
    global _BROKEN

    if not ninepatch.have_renderer():
        return None
    try:
        return ninepatch.build(pixels(radius), max(int(math.ceil(radius)), 2))
    except Exception:
        log.warning("continuous corners are unavailable; using circular ones", exc_info=True)
        _BROKEN = True
        return None


def release_all() -> None:
    """Drop every built sprite. Frame thread only, at teardown."""
    ninepatch.release(_SPRITES.values())
    _SPRITES.clear()
