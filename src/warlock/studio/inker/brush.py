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

SYMMETRY = ("none", "x", "y", "xy", "radial")

#: The three nibs, and the split is between *antialiased* and *not*.
#:
#: ``soft`` is the disc this brush has always stamped: a smoothstep falloff with
#: an antialiased rim even at hardness 1, which is right for everything the 3D
#: pipeline wants a reference painted for and wrong for every pixel-art use of
#: the same tool. The other two produce coverage that is exactly 0 or 1 -- no
#: fractional pixel anywhere, so a stroke has hard edges and a colour count that
#: does not grow -- ``pixel`` as a disc and ``square`` as the flat nib a
#: one-pixel pencil actually wants. Hardness means nothing to either: coverage
#: with no intermediate values has no falloff to shape, which is why
#: ``_stamp`` passes 1.0 for them rather than letting the slider quietly widen
#: the cache with values that change nothing.
NIBS = ("soft", "pixel", "square")

#: The nibs whose dabs land on whole pixels and whose coverage is binary.
PIXEL_NIBS = frozenset(NIBS[1:])

# How many ways a radial symmetry divides the circle by default. Six is the
# snowflake/mandala number every tool that has this control opens on.
DEFAULT_RADIAL = 6
MIN_RADIAL = 2
MAX_RADIAL = 32

# The distance between two input samples, in pixels, at which speed taper is
# fully applied. Samples arrive one per frame, so distance *is* speed -- there
# is no clock in this class and adding one would make a stroke depend on the
# frame rate it was drawn at, which is exactly what the spacing carry exists to
# avoid. 40 px/frame is a brisk drag at 60 fps.
TAPER_SPEED = 40.0

# How much of the previous dab's size carries into the next. Without it a
# single fast frame puts one thin dab in the middle of a thick stroke, which
# reads as a gap rather than as a taper.
TAPER_SMOOTHING = 0.6

# The most a stabiliser may lag. At 1.0 the brush never reaches the cursor at
# all, so the stroke stops where it started and the control looks broken.
MAX_STABILISE = 0.95


def clamp_brush(size: int) -> int:
    return max(MIN_BRUSH, min(MAX_BRUSH, int(size)))


@lru_cache(maxsize=256)
def make_stamp(diameter: int, hardness: float, nib: str = "soft") -> np.ndarray:
    """A float32 coverage stamp, ``diameter`` square, 0..1.

    For the ``soft`` nib the rim is antialiased over the last half pixel even at
    hardness 1: a hard brush should have a crisp edge, not a jagged one, and a
    stamp that is exactly 0/1 is how you get a staircase on every diagonal.

    For the two pixel nibs that staircase *is* the drawing, so their coverage is
    exactly 0 or 1 and hardness is not read at all. The assertion those two owe
    is one a test can make directly: no value strictly between the two, ever, at
    any diameter -- which is what stops a "hard" stroke laying down a fringe of
    near-colours that a palette, an outline pass or a colour-key export then has
    to deal with.
    """
    diameter = max(1, int(diameter))
    hardness = min(1.0, max(0.0, float(hardness)))
    radius = diameter / 2.0
    axis = np.arange(diameter, dtype=np.float32) + 0.5 - radius
    if nib == "square":
        return np.ones((diameter, diameter), dtype=np.float32)
    distance = np.hypot(axis[None, :], axis[:, None])
    if nib == "pixel":
        # ``<=`` rather than ``<``: at diameter 1 the single sample sits exactly
        # on the radius, and a strict test would make the one-pixel pencil --
        # the whole reason this nib exists -- stamp nothing at all.
        return (distance <= radius).astype(np.float32)

    # Where the falloff starts. At hardness 1 that is half a pixel in from the
    # rim, which is exactly the AA band.
    inner = max(0.0, radius * hardness - 0.5) if hardness < 1.0 else max(0.0, radius - 0.5)
    if radius <= inner:
        return (distance <= radius).astype(np.float32)
    t = np.clip((radius - distance) / (radius - inner), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep


def default_axis(size: tuple[int, int]) -> tuple[float, float]:
    """The centre of the canvas, in the coordinates ``_mirror`` reflects about.

    ``(width - 1) / 2`` rather than ``width / 2``: a reflection about it sends
    column 0 to column ``width - 1``, which is what the fixed formula this
    replaced did and is the only placement where a symmetric drawing is
    symmetric to the pixel.
    """
    return ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0)


def _mirror(
    point: tuple[float, float],
    size: tuple[int, int],
    symmetry: str,
    axis: tuple[float, float] | None = None,
    radial: int = DEFAULT_RADIAL,
) -> list[tuple[float, float]]:
    """A point and its reflections. Applied at the *position* level, so every
    mode -- erase, blur, smudge -- inherits symmetry without knowing about it.

    ``axis`` is the point the mirrors reflect about and the point a radial
    symmetry turns around; it defaults to the canvas centre, which is where it
    always was. Reflections are ``2a - x`` rather than ``width - 1 - x``: the
    two agree exactly at the centre, and only the first generalises.
    """
    x, y = point
    ax, ay = default_axis(size) if axis is None else axis
    points = [(x, y)]
    if symmetry == "radial":
        # Whole turns of 2pi/n about the axis. The point itself is the n = 0
        # case and is already in the list, so the loop starts at one.
        count = max(MIN_RADIAL, min(MAX_RADIAL, int(radial)))
        dx, dy = x - ax, y - ay
        for step in range(1, count):
            angle = 2.0 * math.pi * step / count
            cos, sin = math.cos(angle), math.sin(angle)
            points.append((ax + dx * cos - dy * sin, ay + dx * sin + dy * cos))
        return points
    if symmetry in ("x", "xy"):
        points.append((2.0 * ax - x, y))
    if symmetry in ("y", "xy"):
        points.append((x, 2.0 * ay - y))
    if symmetry == "xy":
        points.append((2.0 * ax - x, 2.0 * ay - y))
    return points


def _whole(point: tuple[float, float]) -> tuple[int, int]:
    """The pixel a float position is inside. Floor, not round: a position is
    inside the pixel whose index is its floor, and rounding would put the left
    half of pixel 3 into pixel 2."""
    return (int(math.floor(point[0])), int(math.floor(point[1])))


def _centre(pixel: tuple[int, int]) -> tuple[float, float]:
    """A whole pixel back as the float position ``_stamp`` and ``_mirror`` take.

    The centre rather than the corner, so a mirror about the canvas -- which
    reflects positions, not indices -- sends a pixel to a pixel rather than to a
    boundary between two.
    """
    return (pixel[0] + 0.5, pixel[1] + 0.5)


def line_pixels(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Every whole pixel between two, inclusive of both. Bresenham.

    A pixel nib cannot use the spacing walk: that places dabs a *fraction of a
    diameter* apart along a float segment, which for a one-pixel pencil means
    either a gap wherever the mouse moved faster than one pixel per frame or a
    second dab on a pixel already drawn. Neither is visible with a soft brush --
    coverage accumulates with maximum, so a repeat is free and a sub-pixel gap
    is filled by the rim -- and both are the whole failure mode at one pixel.
    """
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    out = [(x0, y0)]
    while (x0, y0) != (x1, y1):
        double = error * 2
        if double >= dy:
            error += dy
            x0 += sx
        if double <= dx:
            error += dx
            y0 += sy
        out.append((x0, y0))
    return out


def is_corner(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
    """Whether ``b`` is the elbow of an L that ``a`` and ``c`` already imply.

    Pixel-perfect drawing is exactly this predicate and nothing else: a diagonal
    line drawn freehand comes out as a staircase with a doubled pixel at every
    step, and each doubled pixel is an ``a``/``c`` pair one diagonal apart with
    ``b`` orthogonally adjacent to both. Dropping ``b`` leaves the two touching
    at their corner, which is what a clean pixel diagonal is.

    It is asked *before* ``b`` is stamped rather than by erasing it afterwards,
    because coverage accumulates with maximum and has no subtraction -- undoing
    a dab would mean recomputing the whole stroke's coverage from its history.
    """
    if abs(c[0] - a[0]) != 1 or abs(c[1] - a[1]) != 1:
        return False
    return b in ((a[0], c[1]), (c[0], a[1]))


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
    #: Which stamp; see NIBS. ``soft`` is every stroke this class drew before
    #: the pixel nibs existed, and the default keeps it that way.
    nib: str = "soft"
    #: Drop the elbow of every staircase step; see :func:`is_corner`. Only
    #: meaningful for a pixel nib -- a soft dab has a rim wider than the elbow
    #: it would remove -- and ignored for the others rather than refused, since
    #: it is a checkbox that stays ticked while the user tries a soft brush.
    pixel_perfect: bool = False
    symmetry: str = "none"
    #: Where the mirrors reflect and a radial symmetry turns. None is the
    #: canvas centre, which is where it was fixed before it was a field.
    axis: tuple[float, float] | None = None
    radial: int = DEFAULT_RADIAL
    #: 0..1. How far the drawn point lags the cursor -- a "lazy mouse", which
    #: is what turns a shaky hand's line into a smooth one.
    stabilise: float = 0.0
    #: 0..1. How much speed thins the stroke, for a pen-like taper on a fast
    #: flick. Speed is measured in pixels between input samples; see
    #: TAPER_SPEED.
    speed_taper: float = 0.0
    clip: SelectionMask | None = None
    # The layer's "preserve transparency". Applied per dab rather than once at
    # release, because the canvas draws every dab: enforcing it only at the end
    # would show the stroke spilling past the shape for the whole drag and then
    # snap it back, which reads as a bug in the lock rather than as the lock.
    alpha_lock: bool = False

    coverage: np.ndarray = field(init=False)
    dirty: tuple[int, int, int, int] | None = field(init=False, default=None)
    _carry: float = field(init=False, default=0.0)
    _last: tuple[float, float] | None = field(init=False, default=None)
    _pickup: np.ndarray | None = field(init=False, default=None)
    #: The stabilised cursor: what the brush is chasing. Distinct from
    #: ``_last``, which is where the brush *is* -- the two are the same point
    #: only when stabilisation is off.
    _target: tuple[float, float] | None = field(init=False, default=None)
    #: The tapered diameter carried between segments; see TAPER_SMOOTHING.
    _width: float = field(init=False, default=0.0)
    #: The pixel-perfect filter's two-pixel window: the last dab stamped, and
    #: the candidate held back to see whether the next one makes it an elbow.
    _prev_pixel: tuple[int, int] | None = field(init=False, default=None)
    _pending_pixel: tuple[int, int] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        width, height = self.size
        self.coverage = np.zeros((height, width), dtype=np.float32)
        self.diameter = clamp_brush(self.diameter)
        self.stabilise = min(MAX_STABILISE, max(0.0, float(self.stabilise)))
        self.speed_taper = min(1.0, max(0.0, float(self.speed_taper)))
        self._width = float(self.diameter)

    # -- the walk ----------------------------------------------------------

    @property
    def step(self) -> float:
        return max(0.5, self.diameter * self.spacing)

    @property
    def pixel(self) -> bool:
        """Whether this stroke walks whole pixels rather than the spacing."""
        return self.nib in PIXEL_NIBS

    def begin(self, point: tuple[float, float], target: np.ndarray) -> None:
        """A click is one dab -- press must mark, not wait for a drag.

        The stabiliser starts *at* the press rather than lagging into it: a
        brush that crept toward the first click would put the dab somewhere the
        user did not press, which is the one place a lag is not forgivable.

        Under the pixel-perfect filter the press is *held* rather than stamped,
        because whether it is an elbow is not known until two more pixels have
        arrived. ``finish`` is what makes a click still mark: it flushes the
        held pixel, so a press and release with no movement draws exactly one.
        """
        self._last = point
        self._target = point
        self._carry = 0.0
        self._width = float(self.diameter)
        if self.pixel:
            self._feed(_whole(point), target)
        else:
            self._dab(point, target)

    def to(self, point: tuple[float, float], target: np.ndarray) -> None:
        if self._last is None:
            self.begin(point, target)
            return
        previous = self._target or point
        speed = math.hypot(point[0] - previous[0], point[1] - previous[1])
        self._target = point
        # The point the brush chases. An exponential lag rather than a rolling
        # average of the last n samples: it needs no history, it arrives at the
        # cursor when the cursor stops, and one number is the whole control.
        keep = self.stabilise
        goal = (
            point
            if keep <= 0.0
            else (
                self._last[0] + (point[0] - self._last[0]) * (1.0 - keep),
                self._last[1] + (point[1] - self._last[1]) * (1.0 - keep),
            )
        )
        self._advance(goal, speed, target)

    def _advance(
        self, goal: tuple[float, float], speed: float, target: np.ndarray
    ) -> None:
        assert self._last is not None
        if self.pixel:
            # Every whole pixel between here and there, and no carry: the walk
            # is the line rather than a spacing along it. The first is the one
            # already fed on the previous segment, so it is dropped.
            for pixel in line_pixels(_whole(self._last), _whole(goal))[1:]:
                self._feed(pixel, target)
            self._last = goal
            return
        x0, y0 = self._last
        x1, y1 = goal
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0.0:
            return
        width = self._taper(speed)
        step = self.step
        travelled = self._carry
        while travelled + step <= length:
            travelled += step
            t = travelled / length
            self._dab((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), target, width)
        self._carry = travelled - length
        self._last = goal

    def _taper(self, speed: float) -> int:
        """The diameter for this segment's dabs, thinned by how fast it moved.

        Smoothed against the previous segment, because a single fast frame
        otherwise puts one thin dab in the middle of a thick stroke and reads
        as a gap rather than as a taper. Never below one pixel: a stamp with no
        diameter is a stroke that stops.
        """
        if self.speed_taper <= 0.0:
            return self.diameter
        fraction = min(1.0, speed / TAPER_SPEED)
        wanted = self.diameter * (1.0 - self.speed_taper * fraction)
        self._width += (wanted - self._width) * (1.0 - TAPER_SMOOTHING)
        return max(MIN_BRUSH, int(round(self._width)))

    # -- the pixel walk ----------------------------------------------------

    def _feed(self, pixel: tuple[int, int], target: np.ndarray) -> None:
        """Offer one whole pixel to the stroke, through the corner filter."""
        if not self.pixel_perfect:
            self._dab(_centre(pixel), target)
            return
        held = self._pending_pixel
        if held is None:
            self._pending_pixel = pixel
            return
        if pixel == held:
            return
        if self._prev_pixel is not None and is_corner(self._prev_pixel, held, pixel):
            # The elbow goes, and ``_prev_pixel`` stays where it was: the pixel
            # before the elbow is still the one the *next* triple is measured
            # from, and advancing it to the dropped pixel would let a long
            # diagonal shed a second pixel per step.
            self._pending_pixel = pixel
            return
        self._dab(_centre(held), target)
        self._prev_pixel = held
        self._pending_pixel = pixel

    @property
    def pending(self) -> bool:
        """Whether the corner filter is holding a pixel that is not drawn yet.

        Asked by ``Document.end_stroke`` so the flush -- and the layer lookup it
        needs -- is reached only by a stroke that has something to flush, which
        is none of them unless the pixel-perfect filter is on.
        """
        return self._pending_pixel is not None

    def finish(self, target: np.ndarray) -> None:
        """Stamp whatever the corner filter is still holding back.

        Called once, at release. Idempotent, because ``end_stroke`` is reached
        from several places and a second flush would otherwise re-stamp the last
        pixel -- harmless under maximum-coverage, but only by accident.
        """
        held, self._pending_pixel = self._pending_pixel, None
        if held is not None:
            self._dab(_centre(held), target)
            self._prev_pixel = held

    # -- one dab -----------------------------------------------------------

    def _dab(
        self, point: tuple[float, float], target: np.ndarray, diameter: int | None = None
    ) -> None:
        for mirrored in _mirror(point, self.size, self.symmetry, self.axis, self.radial):
            self._stamp(mirrored, target, self.diameter if diameter is None else diameter)

    def _stamp(self, point: tuple[float, float], target: np.ndarray, diameter: int) -> None:
        width, height = self.size
        # 1.0 rather than ``self.hardness`` for a pixel nib: neither reads it,
        # and passing it through would key the shared stamp cache on a value
        # that cannot change the answer.
        stamp = make_stamp(diameter, 1.0 if self.pixel else self.hardness, self.nib)
        radius = diameter / 2.0
        if self.pixel:
            # Anchored on the pixel the dab is *on*, so an odd nib is centred on
            # it and an even one grows down and right. The rounding form below
            # is a half-pixel out for an even diameter, which a soft rim hides
            # and a hard one does not.
            left = int(math.floor(point[0])) - (diameter - 1) // 2
            top = int(math.floor(point[1])) - (diameter - 1) // 2
        else:
            left = int(math.floor(point[0] - radius + 0.5))
            top = int(math.floor(point[1] - radius + 0.5))

        x0, y0 = max(0, left), max(0, top)
        x1 = min(width, left + diameter)
        y1 = min(height, top + diameter)
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
        if self.alpha_lock:
            # Which makes the eraser a no-op on a locked layer, and that is the
            # correct reading rather than a gap: erasing *is* changing alpha,
            # and every other editor with this lock behaves the same way.
            out[..., 3] = before[..., 3]
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
        if self.alpha_lock:
            out[..., 3] = crop[..., 3]
        target[y0:y1, x0:x1] = composite.to_uint8_255(out)

    def _mark(self, rect: tuple[int, int, int, int]) -> None:
        if self.dirty is None:
            self.dirty = rect
            return
        a, b, c, d = self.dirty
        x0, y0, x1, y1 = rect
        self.dirty = (min(a, x0), min(b, y0), max(c, x1), max(d, y1))
