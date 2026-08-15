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
from typing import Any

import numpy as np

from . import composite, tiling
from .selection import SelectionMask

MIN_BRUSH = 1
MAX_BRUSH = 256

# A fraction of the diameter. 0.1 is dense enough that a hard round brush has
# no scallops at any speed a mouse can produce.
DEFAULT_SPACING = 0.1

#: ``replace`` is Aseprite's copy-colour ink and it is the fifth rather than a
#: flag on ``paint`` because it is genuinely a different arithmetic: paint
#: composites the colour *over* what is there, replace writes it -- alpha
#: included, so it can paint transparency down as well as up.
#:
#: ``shade`` is Aseprite's shading ink and it is the odd one out for a reason
#: worth stating: it is the only mode that does not paint the stroke's colour at
#: all. It moves each pixel it covers **one step along a ramp**, so what it
#: writes is decided by what is already there -- see
#: :meth:`StrokeState._shade`.
MODES = ("paint", "erase", "blur", "smudge", "replace", "shade")

#: How much of a pixel a dab must cover for the shading ink to shift it.
#:
#: A threshold rather than a blend, because there is nothing to blend: a shift
#: lands on the next swatch of the ramp exactly or it does not happen, and half
#: a step is a colour that is on no palette. Half a pixel is the same rule the
#: rest of this package uses for "is this pixel inside" -- and for a pixel nib,
#: whose coverage is only ever 0 or 1, it is not a rule at all.
SHADE_COVERAGE = 0.5

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
    """A float32 coverage stamp, ``diameter`` square, 0..1. **Read-only.**

    Cached, and the cache hands the *same array* to every caller that asks for
    the same brush -- so it is returned write-locked. This is public API and it
    is a plain ndarray, which is exactly the combination where one caller
    scaling a stamp in place would silently change every stroke drawn with that
    brush for the rest of the session, and only for that brush size. A caller
    that needs to modify one copies it; the sibling caches in ``clay/mesh.py``
    and ``plotter/tileset.py`` freeze theirs for the same reason.
    """
    stamp = _stamp(diameter, hardness, nib)
    stamp.setflags(write=False)
    return stamp


def _stamp(diameter: int, hardness: float, nib: str) -> np.ndarray:
    """``make_stamp``'s body, uncached and writable.

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


def axis_or_default(
    size: tuple[int, int], axis: tuple[float, float] | None
) -> tuple[float, float]:
    """Where the mirrors reflect and a radial symmetry turns, resolved.

    One spelling of ``axis if axis is not None else default_axis(size)``,
    exported because the *guide* has to agree with the engine and there was a
    stretch where it did not: :func:`_mirror` honoured a moved axis and used
    ``(width - 1) / 2`` for the default, while the canvas drew its line at
    ``width / 2`` unconditionally. Half a pixel of that is invisible; ignoring a
    moved axis is a guide pointing at the wrong place, and radial had no guide
    at all. Two readers of one answer, in the imgui-free half of the tree so the
    pane can import it.
    """
    return default_axis(size) if axis is None else axis


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
    ax, ay = axis_or_default(size, axis)
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
    #: Which axes this stroke wraps on; see :mod:`.tiling`. A pair rather than
    #: a bool because tiled mode is per axis, and it is passed in by the canvas
    #: rather than read off the document: tiling is a property of how the tab
    #: is being *viewed*, and the engine stays stateless about the UI.
    wrap_axes: tuple[bool, bool] = (False, False)
    #: The radius dabs scatter within, for the spray tool. Zero is every stroke
    #: this class drew before spraying existed -- ``spray`` is a second way to
    #: *emit* dabs and changes nothing about what a dab does.
    scatter: float = 0.0
    #: The scatter's seed. The engine is deterministic given a seed and a call
    #: sequence; the canvas draws a fresh one at every press, and a test injects
    #: one instead. There is no other source of randomness in a stroke.
    seed: int = 0
    #: The shading ink's ramp, in the order it is walked; see :meth:`_shade`.
    #: Passed in rather than read off a document, for ``wrap_axes``' reason: the
    #: ramp is a *selection* in the palette panel, which is UI state, and the
    #: engine stays stateless about the UI. Empty is every stroke this class
    #: drew before shading existed, and makes a ``shade`` stroke a no-op.
    ramp: tuple[tuple[int, int, int, int], ...] = ()
    #: Which way along the ramp a shade dab moves: +1 toward its end, -1 toward
    #: its start. Clamped at both ends rather than wrapping -- see :meth:`_shade`.
    shade_dir: int = 1

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
    #: The scatter's generator, built from ``seed`` and never reseeded. Its own
    #: stream rather than ``np.random``'s global one, so two documents sprayed
    #: in one session cannot draw from each other's sequence.
    _rng: Any = field(init=False, default=None, repr=False)
    #: Which pixels this stroke has already shaded, ``(H, W)`` bool -- **one
    #: step per stroke**, which is the whole of what makes the shading ink
    #: usable. See :meth:`_shade`. Allocated only for a ``shade`` stroke, so
    #: every other mode pays nothing for it.
    _shifted: np.ndarray | None = field(init=False, default=None, repr=False)
    #: The ramp as packed RGB keys, and as an (N, 3) uint8 table. Both are
    #: derived from ``ramp`` once here rather than per dab: a stroke is hundreds
    #: of dabs and the ramp cannot change inside one.
    _ramp_keys: np.ndarray | None = field(init=False, default=None, repr=False)
    _ramp_rgb: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        width, height = self.size
        self.coverage = np.zeros((height, width), dtype=np.float32)
        self.diameter = clamp_brush(self.diameter)
        self.stabilise = min(MAX_STABILISE, max(0.0, float(self.stabilise)))
        self.speed_taper = min(1.0, max(0.0, float(self.speed_taper)))
        self._width = float(self.diameter)
        self._rng = np.random.default_rng(int(self.seed))
        if self.mode == "shade":
            self.ramp = tuple(tuple(int(c) for c in colour) for colour in self.ramp)  # type: ignore[misc]
            self._shifted = np.zeros((height, width), dtype=bool)
            rgb = np.asarray([c[:3] for c in self.ramp], dtype=np.uint8).reshape(-1, 3)
            self._ramp_rgb = rgb
            self._ramp_keys = (
                rgb[:, 0].astype(np.uint32) << 16
                | rgb[:, 1].astype(np.uint32) << 8
                | rgb[:, 2].astype(np.uint32)
            )

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

    def spray(self, point: tuple[float, float], count: int, target: np.ndarray) -> None:
        """Emit ``count`` dabs scattered uniformly in a disc of ``scatter``.

        A second way to *emit* dabs, not a second kind of dab: every one goes
        through :meth:`_dab`, so symmetry, the selection clip, the alpha lock,
        tiled wrapping and the single-patch undo all apply with no code of their
        own. That is the whole design -- an airbrush is a distribution over
        positions and nothing else.

        ``sqrt`` of a uniform sample for the radius, because a uniform *radius*
        piles the density up at the centre: the area of an annulus grows with
        r, so the number of dabs landing in it has to as well or the spray reads
        as a dot with a halo.
        """
        whole = int(count)
        if whole <= 0:
            return
        radius = max(0.0, float(self.scatter))
        # One draw for the whole batch rather than two per dab: the stream is
        # the same either way only if the shape is fixed, so this is also what
        # pins determinism to (seed, call sequence) rather than to how the
        # caller happened to chunk its frames -- see the class docstring.
        sample = self._rng.random((whole, 2))
        distance = radius * np.sqrt(sample[:, 0])
        angle = sample[:, 1] * (2.0 * math.pi)
        xs = (point[0] + distance * np.cos(angle)).tolist()
        ys = (point[1] + distance * np.sin(angle)).tolist()
        for x, y in zip(xs, ys, strict=True):
            self._dab((x, y), target)

    @property
    def _axes(self) -> tuple[bool, bool]:
        """Which axes *this dab* wraps on.

        **Smudge is excluded and that is a decision, not an omission.** Its
        pickup buffer trails the brush and is shaped by the region it last
        touched; carrying it across a seam would mean deciding what "the pixels
        the brush just passed over" are when the brush is in two places at once,
        and every answer is arbitrary. It falls back to the clamped behaviour,
        so a smudge near the edge stops at the edge. Blur wraps, but each piece
        blurs its own destination rectangle -- so the blur kernel does not reach
        across the seam either, which is a stated limitation rather than a
        promise this version makes.
        """
        return (False, False) if self.mode == "smudge" else self.wrap_axes

    def _stamp(self, point: tuple[float, float], target: np.ndarray, diameter: int) -> None:
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

        # The single choke point for tiled painting: every mode, every nib and
        # every emission reaches the layer through here, so wrapping is one loop
        # rather than a rule each of them has to remember. With no wrap the
        # helper returns the one clipped rectangle the body below used to
        # compute inline, which is what makes tiled-off byte-identical.
        for rect, (sx, sy) in tiling.pieces(
            (left, top, left + diameter, top + diameter), self.size, self._axes
        ):
            x0, y0, x1, y1 = rect
            piece = stamp[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
            if self.mode in ("blur", "smudge"):
                self._filter(piece, rect, target)
            elif self.mode == "shade":
                self._shade(piece, rect, target)
            else:
                region = self.coverage[y0:y1, x0:x1]
                np.maximum(region, piece, out=region)
                self._resolve(rect, target)
            # Per piece, so the union is the dirty box the undo patch covers.
            # One patch over the union rather than a rect per piece: a tile is a
            # small canvas, and a multi-rect edit type would need eviction
            # accounting of its own for a saving measured in kilobytes.
            self._mark(rect)

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

        ``replace`` is Aseprite's copy-colour ink and it lives here rather than
        beside the blur/smudge branch on purpose: it is a *coverage* mode, so it
        gets recompute-from-``before`` -- and therefore no compounding where a
        stroke crosses itself -- for free, along with the alpha lock, the
        selection clip and the indexed snap that runs at commit.

        At full coverage it writes the foreground RGBA verbatim, alpha
        included, so it can paint transparency *down* as well as up -- which is
        the whole reason a copy ink exists. Under partial coverage it lerps
        rather than snapping to a hard edge: this repository's doctrine is that
        feathering means one thing everywhere, so a soft nib, a feathered
        selection and a low opacity all soften a replace stroke exactly as they
        soften a paint one. That is a visible divergence from Aseprite, and only
        on soft nibs -- with a pixel nib the coverage is 0 or 1 and the two
        agree exactly.
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
        elif self.mode == "replace":
            colour = np.asarray(self.colour, dtype=np.float32)
            out = before + (colour - before) * alpha
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

    def _shade(
        self, piece: np.ndarray, rect: tuple[int, int, int, int], target: np.ndarray
    ) -> None:
        """Move every covered ramp pixel one step along the ramp.

        Aseprite's shading ink, and the one mode that does not paint the
        stroke's colour: what it writes is decided by what is already there.
        Four rules, and each of them is what stops the tool being useless.

        *It reads the live layer, not ``before``.* ``_filter``'s rule, for a
        stronger version of its reason -- the pixel it has to look at is the one
        it may have written itself a dab ago, and reading the pre-stroke copy
        would make every dab decide from a picture that no longer exists.

        *The match is exact packed RGB.* A pixel is on the ramp or it is not:
        nearest would drag every colour in the drawing onto the nearest swatch,
        which is a conversion rather than a shade, and it would make the
        selected slots mean nothing. So a colour that is not on the ramp -- and
        a fully transparent pixel, which has no colour at all -- is left exactly
        as it is.

        *One step per stroke.* ``_shifted`` records the pixels this stroke has
        already moved, so painting back and forth over a shoulder walks it one
        swatch and stops, rather than running it off the end of the ramp in the
        time it takes to notice. That is what makes the ink controllable, and it
        is per *stroke* -- lift the button and drag again to take another step.

        *Clamped, not wrapped.* The ends of a ramp are the darkest and lightest
        the user chose; wrapping would send the deepest shadow to the brightest
        highlight in one dab, which reads as corruption.

        Alpha is never touched, which is why there is no alpha-lock branch here
        -- "preserve transparency" is exactly *the alpha does not change*, and
        this mode cannot change it. Symmetry, tiled wrapping, the selection clip
        and the single-patch undo all come free, because this is called under
        ``_stamp`` like every other kind of dab.
        """
        keys, table = self._ramp_keys, self._ramp_rgb
        if keys is None or table is None or keys.size < 2 or self._shifted is None:
            # No ramp, or one swatch: there is nowhere to step. A no-op rather
            # than a refusal, because the tool is reached with an empty ramp
            # only on a document with no palette, which the UI already gates.
            return
        x0, y0, x1, y1 = rect
        covered = self._weights(rect, piece) >= SHADE_COVERAGE
        covered &= ~self._shifted[y0:y1, x0:x1]
        crop = target[y0:y1, x0:x1]
        covered &= crop[..., 3] > 0
        if not covered.any():
            return

        rgb = crop[..., :3]
        packed = (
            rgb[..., 0].astype(np.uint32) << 16
            | rgb[..., 1].astype(np.uint32) << 8
            | rgb[..., 2].astype(np.uint32)
        )
        # Which ramp entry each pixel sits on, -1 for none. Walked backwards so
        # the *first* entry wins a ramp that names one colour twice, which is
        # the same "earlier slot owns it" rule the palette panel shows.
        where = np.full(packed.shape, -1, dtype=np.int32)
        for index in range(keys.size - 1, -1, -1):
            where[packed == keys[index]] = index
        hit = covered & (where >= 0)
        if not hit.any():
            return

        step = 1 if int(self.shade_dir) >= 0 else -1
        moved = np.clip(where + step, 0, keys.size - 1)
        crop[..., :3][hit] = table[moved[hit]]
        # Marked even where the clamp made the step a no-op: a pixel already at
        # the end of the ramp has *had* this stroke's step, and leaving it
        # unmarked would only matter if a later dab could move it, which the
        # clamp already forbids.
        self._shifted[y0:y1, x0:x1][hit] = True

    def _mark(self, rect: tuple[int, int, int, int]) -> None:
        if self.dirty is None:
            self.dirty = rect
            return
        a, b, c, d = self.dirty
        x0, y0, x1, y1 = rect
        self.dirty = (min(a, x0), min(b, y0), max(c, x1), max(d, y1))
