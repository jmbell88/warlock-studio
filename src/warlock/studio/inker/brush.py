"""The brush: a cached coverage stamp, a spacing walk, and one stroke buffer.

Three things here are the difference between a paint program and a line-drawing
program.

*Coverage accumulates with maximum, not with addition.* A stroke keeps its own
float coverage buffer for its whole life and the layer is recomputed from the
pixels as they were when the stroke began. Without that, a half-opacity brush
darkens wherever the stroke crosses itself or wherever the spacing put two
stamps close together -- which is everywhere, since spacing is a fraction of
the radius.

*Spacing is walked with a carry.* The distance left over from one segment is
carried into the next, so density does not depend on how fast the mouse moved
or how often the frame loop sampled it.

*The stamp is a cached float disc.* Rebuilding a Gaussian-ish falloff per dab
is what makes a naive painter stutter; the falloff only depends on radius and
hardness, and there are a few dozen of those in a session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import composite
from .selection import SelectionMask

MIN_BRUSH = 1
MAX_BRUSH = 256

# A fraction of the diameter. 0.1 is dense enough that a hard round brush has
# no scallops at any speed a mouse can produce.
DEFAULT_SPACING = 0.1

MODES = ("paint", "erase", "blur", "smudge")

SYMMETRY = ("none", "x", "y", "xy")


def clamp_brush(size: int) -> int:
    return max(MIN_BRUSH, min(MAX_BRUSH, int(size)))


@lru_cache(maxsize=256)
def make_stamp(diameter: int, hardness: float) -> np.ndarray:
    """A float32 coverage disc, ``diameter`` square, 0..1.

    Even at hardness 1 the rim is antialiased over the last half pixel: a hard
    brush should have a crisp edge, not a jagged one, and a stamp that is
    exactly 0/1 is how you get a staircase on every diagonal.
    """
    diameter = max(1, int(diameter))
    hardness = min(1.0, max(0.0, float(hardness)))
    radius = diameter / 2.0
    axis = np.arange(diameter, dtype=np.float32) + 0.5 - radius
    distance = np.hypot(axis[None, :], axis[:, None])

    # Where the falloff starts. At hardness 1 that is half a pixel in from the
    # rim, which is exactly the AA band.
    inner = max(0.0, radius * hardness - 0.5) if hardness < 1.0 else max(0.0, radius - 0.5)
    if radius <= inner:
        return (distance <= radius).astype(np.float32)
    t = np.clip((radius - distance) / (radius - inner), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep


def _mirror(
    point: tuple[float, float], size: tuple[int, int], symmetry: str
) -> list[tuple[float, float]]:
    """A point and its reflections. Applied at the *position* level, so every
    mode -- erase, blur, smudge -- inherits symmetry without knowing about it."""
    x, y = point
    width, height = size
    points = [(x, y)]
    if symmetry in ("x", "xy"):
        points.append((width - 1 - x, y))
    if symmetry in ("y", "xy"):
        points.append((x, height - 1 - y))
    if symmetry == "xy":
        points.append((width - 1 - x, height - 1 - y))
    return points


@dataclass
class StrokeState:
    """One stroke, from press to release, on one layer.

    ``before`` is the layer as it was at press time and doubles as the undo
    patch's source -- there is no second copy, and no moment where the two
    could disagree.
    """

    layer_uid: int
    size: tuple[int, int]
    before: np.ndarray
    colour: tuple[int, int, int, int]
    diameter: int = 8
    hardness: float = 0.8
    opacity: float = 1.0
    spacing: float = DEFAULT_SPACING
    mode: str = "paint"
    strength: float = 0.5
    symmetry: str = "none"
    clip: SelectionMask | None = None

    coverage: np.ndarray = field(init=False)
    dirty: tuple[int, int, int, int] | None = field(init=False, default=None)
    _carry: float = field(init=False, default=0.0)
    _last: tuple[float, float] | None = field(init=False, default=None)
    _pickup: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        width, height = self.size
        self.coverage = np.zeros((height, width), dtype=np.float32)
        self.diameter = clamp_brush(self.diameter)

    # -- the walk ----------------------------------------------------------

    @property
    def step(self) -> float:
        return max(0.5, self.diameter * self.spacing)

    def begin(self, point: tuple[float, float], target: np.ndarray) -> None:
        """A click is one dab -- press must mark, not wait for a drag."""
        self._last = point
        self._carry = 0.0
        self._dab(point, target)

    def to(self, point: tuple[float, float], target: np.ndarray) -> None:
        if self._last is None:
            self.begin(point, target)
            return
        x0, y0 = self._last
        x1, y1 = point
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0.0:
            return
        step = self.step
        travelled = self._carry
        while travelled + step <= length:
            travelled += step
            t = travelled / length
            self._dab((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), target)
        self._carry = travelled - length
        self._last = point

    # -- one dab -----------------------------------------------------------

    def _dab(self, point: tuple[float, float], target: np.ndarray) -> None:
        for mirrored in _mirror(point, self.size, self.symmetry):
            self._stamp(mirrored, target)

    def _stamp(self, point: tuple[float, float], target: np.ndarray) -> None:
        width, height = self.size
        stamp = make_stamp(self.diameter, self.hardness)
        radius = self.diameter / 2.0
        left = int(math.floor(point[0] - radius + 0.5))
        top = int(math.floor(point[1] - radius + 0.5))

        x0, y0 = max(0, left), max(0, top)
        x1 = min(width, left + self.diameter)
        y1 = min(height, top + self.diameter)
        if x1 <= x0 or y1 <= y0:
            return
        piece = stamp[y0 - top : y1 - top, x0 - left : x1 - left]

        if self.mode in ("blur", "smudge"):
            self._filter(piece, (x0, y0, x1, y1), target)
        else:
            region = self.coverage[y0:y1, x0:x1]
            np.maximum(region, piece, out=region)
            self._resolve((x0, y0, x1, y1), target)
        self._mark((x0, y0, x1, y1))

    def _weights(self, rect: tuple[int, int, int, int], piece: np.ndarray) -> np.ndarray:
        """Stamp coverage after the selection clip. One multiply -- which is
        the reason feathering and brush softness use the same representation."""
        if self.clip is None:
            return piece
        x0, y0, x1, y1 = rect
        return piece * (self.clip.mask[y0:y1, x0:x1].astype(np.float32) / 255.0)

    def _resolve(self, rect: tuple[int, int, int, int], target: np.ndarray) -> None:
        """Recompute the region from the pre-stroke pixels and the coverage.

        Not "blend onto what is there": that is what makes overlapping dabs
        pile up. Coverage is the whole record of the stroke, applied once.
        """
        x0, y0, x1, y1 = rect
        cov = self._weights(rect, self.coverage[y0:y1, x0:x1])
        if self.clip is not None:
            cov = np.minimum(cov, 1.0)
        alpha = (cov * self.opacity)[..., None]
        before = self.before[y0:y1, x0:x1].astype(np.float32)

        if self.mode == "erase":
            out = before.copy()
            out[..., 3] = before[..., 3] * (1.0 - alpha[..., 0])
        else:
            out = composite.paint_colour(before, self.colour, alpha[..., 0])
        target[y0:y1, x0:x1] = composite.to_uint8_255(out)

    def _filter(
        self, piece: np.ndarray, rect: tuple[int, int, int, int], target: np.ndarray
    ) -> None:
        """Blur and smudge read the *live* layer, not the pre-stroke copy.

        They are accumulation tools -- the second pass over the same place is
        supposed to blur more -- which is exactly why they cannot use the
        coverage-recompute path the paint modes use.
        """
        x0, y0, x1, y1 = rect
        crop = target[y0:y1, x0:x1].astype(np.float32)
        weight = (self._weights(rect, piece) * self.strength)[..., None]

        if self.mode == "blur":
            from PIL import Image, ImageFilter

            blurred = Image.fromarray(target[y0:y1, x0:x1], "RGBA").filter(
                ImageFilter.GaussianBlur(max(1.0, self.diameter / 8.0))
            )
            source = np.asarray(blurred, dtype=np.float32)
        else:
            if self._pickup is None or self._pickup.shape != crop.shape:
                self._pickup = crop.copy()
            source = self._pickup
            # The pickup buffer trails the brush: it takes on what it passes
            # over, which is what makes a smudge fade rather than smear one
            # colour across the canvas forever.
            self._pickup = source + (crop - source) * float(self.strength)

        out = crop + (source - crop) * weight
        target[y0:y1, x0:x1] = composite.to_uint8_255(out)

    def _mark(self, rect: tuple[int, int, int, int]) -> None:
        if self.dirty is None:
            self.dirty = rect
            return
        a, b, c, d = self.dirty
        x0, y0, x1, y1 = rect
        self.dirty = (min(a, x0), min(b, y0), max(c, x1), max(d, y1))
