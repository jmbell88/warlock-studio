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
#: ``copy`` is Aseprite's *Copy Color+Alpha*, and it is a sixth mode rather
#: than a flag on ``replace`` because it deliberately ignores **both** the
#: stroke opacity and the dab's antialiasing: what it writes is the colour, at
#: full strength, on every pixel the dab covers at all. That is the ink a
#: pixel artist reaches for when the point is that the result has exactly the
#: colours they chose in it -- and a version of it that honoured opacity would
#: be ``replace`` again under a second name.
MODES = ("paint", "erase", "blur", "smudge", "replace", "copy", "shade")

#: How much of a pixel a dab must cover for the shading ink to shift it.
#:
#: A threshold rather than a blend, because there is nothing to blend: a shift
#: lands on the next swatch of the ramp exactly or it does not happen, and half
#: a step is a colour that is on no palette. Half a pixel is the same rule the
#: rest of this package uses for "is this pixel inside" -- and for a pixel nib,
#: whose coverage is only ever 0 or 1, it is not a rule at all.
SHADE_COVERAGE = 0.5

#: ``diag`` and ``anti`` are the 45-degree mirrors (6.8): a reflection about
#: the line through the axis at +45 and at -45. Aseprite offers both, and they
#: are the two an isometric tile is drawn with -- neither is expressible as a
#: combination of the axis-aligned pair.
SYMMETRY = ("none", "x", "y", "xy", "radial", "diag", "anti")

#: The modes an image stamp is honoured for. A stamp handed to any other is
#: dropped at ``__post_init__`` rather than half-applied -- it is a tip that
#: stays loaded while the user tries another tool.
#:
#: Blur, smudge and shade are out because they decide what they write from what
#: is already on the layer, so there is nothing for a picture to say to them.
#:
#: **``replace`` is out for a subtler reason worth stating.** For a generated
#: disc, coverage and colour are two independent quantities -- the disc says how
#: much, the swatch says what -- and a copy ink is what lets the swatch's alpha
#: be written *down* rather than composited up. A captured picture has only one:
#: its alpha is both its shape and its transparency (that is the invariant
#: ``_masked_alpha`` gives a capture, and it is what makes a lasso-captured tip
#: stamp its shape rather than its bounding box). So "write this alpha verbatim"
#: and "cover this much" are the same number, and there is nothing left for a
#: copy ink to express. The brush's ink radio is hidden while a tip is loaded,
#: so the two settings never contradict each other on screen.
STAMP_MODES = ("paint", "erase")

#: Where an image stamp's dabs land; see :meth:`StrokeState._image_dab`.
#:
#: ``free`` puts the picture under the cursor, which is what a brush does.
#: ``aligned`` snaps every dab to a lattice of the stamp's own size anchored at
#: the canvas origin, which is what a *pattern* does -- and which is what makes
#: dragging back and forth over one cell idempotent to the byte.
STAMP_ALIGN = ("free", "aligned")

#: The largest image that may become a brush tip, per side.
#:
#: Not an engine limit -- numpy is happy with anything -- but a limit on what a
#: *dab* may cost. A dab is stamped per spacing step and the spacing is a
#: fraction of the tip, so a 2048-square tip is a 16 MiB read-modify-write
#: several times a frame for the whole drag. 512 is larger than any brush tip a
#: user draws and small enough that the worst case is a few milliseconds.
MAX_STAMP = 512

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
#: ``line`` is Aseprite's line nib: a one-pixel-thick run at the brush angle,
#: which is what a calligraphic stroke is made of. It is a *pixel* nib -- its
#: coverage is 0 or 1 -- because a feathered one-pixel line is a grey smear
#: rather than a line.
NIBS = ("soft", "pixel", "square", "line")

#: The nibs whose dabs land on whole pixels and whose coverage is binary.
PIXEL_NIBS = frozenset(NIBS[1:])

#: The nibs a brush *angle* means anything to. A disc turned is a disc.
ANGLED_NIBS = frozenset({"square", "line"})

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
def make_stamp(
    diameter: int, hardness: float, nib: str = "soft", angle: float = 0.0
) -> np.ndarray:
    """A float32 coverage stamp, ``diameter`` square, 0..1. **Read-only.**

    Cached, and the cache hands the *same array* to every caller that asks for
    the same brush -- so it is returned write-locked. This is public API and it
    is a plain ndarray, which is exactly the combination where one caller
    scaling a stamp in place would silently change every stroke drawn with that
    brush for the rest of the session, and only for that brush size. A caller
    that needs to modify one copies it; the sibling caches in ``clay/mesh.py``
    and ``plotter/tileset.py`` freeze theirs for the same reason.
    """
    stamp = _stamp(diameter, hardness, nib, angle)
    stamp.setflags(write=False)
    return stamp


def _stamp(diameter: int, hardness: float, nib: str, angle: float = 0.0) -> np.ndarray:
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
    if nib in ANGLED_NIBS:
        # **Rotated in the stamp's own space**, not by turning the drawing:
        # every sample asks where it lands in the nib's frame and answers 0 or
        # 1, so an angled square is still a square with hard edges at every
        # angle rather than a resampled one with a grey rim.
        radians = math.radians(float(angle))
        cos, sin = math.cos(radians), math.sin(radians)
        ys, xs = np.meshgrid(axis, axis, indexing="ij")
        local_x = xs * cos + ys * sin
        local_y = -xs * sin + ys * cos
        if nib == "square":
            inside = (np.abs(local_x) <= radius) & (np.abs(local_y) <= radius)
        else:
            # A run of ``diameter`` pixels, one pixel thick. ``<= 0.5`` is the
            # same half-pixel rule the rest of this module uses for "inside".
            inside = (np.abs(local_y) <= 0.5) & (np.abs(local_x) <= radius)
        return inside.astype(np.float32)
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
    if symmetry in ("diag", "anti"):
        # Reflection about the 45-degree line through the axis: swap the two
        # offsets (and negate both for the anti-diagonal). Written as offsets
        # from the axis for ``2a - x``'s reason -- the two agree at the centre
        # and only this form generalises to a moved axis.
        dx, dy = x - ax, y - ay
        if symmetry == "diag":
            points.append((ax + dy, ay + dx))
        else:
            points.append((ax - dy, ay - dx))
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


def _variant(source: np.ndarray, rotation: int, flip_x: bool, flip_y: bool) -> np.ndarray:
    """One transform of a stamp's pixels. **Flips first, then the rotation.**

    An order has to be picked because the two do not commute -- a flip then a
    quarter turn is not a quarter turn then a flip -- and this one is picked so
    that the two flip toggles keep meaning "mirror the picture I drew" no matter
    which turn is showing. The alternative (rotate first) makes the horizontal
    flip button mirror vertically at 90 degrees, which reads as a bug in the
    button rather than as an order of operations.

    Every step is an exact reindexing of the source bytes -- no resampling, no
    interpolation, nothing that could tint a pixel -- so a variant of a captured
    brush is byte-identical to the same variant taken by hand.
    """
    out = source
    if flip_x:
        out = out[:, ::-1]
    if flip_y:
        out = out[::-1, :]
    turns = (int(rotation) // 90) % 4
    if turns:
        # Negative k, because ``rot90`` turns anticlockwise and ``rotation`` is
        # clockwise -- the direction the canvas's own view rotation uses, and
        # the direction a "rotate" button is read as everywhere.
        out = np.rot90(out, -turns)
    return np.ascontiguousarray(out)


@dataclass(frozen=True, eq=False)
class Stamp:
    """An image used as a brush tip: the pixels, plus which variant is showing.

    The other half of :func:`make_stamp`, and deliberately the same word: both
    are "what one dab lays down", one computed from a radius and one captured
    from the drawing. What makes this a *brush* rather than a paste is that it
    goes through :meth:`StrokeState._dab` like every other dab, so symmetry, the
    selection clip, the alpha lock, tiled wrapping, the spray's emission and the
    single-patch undo all apply to it with no code of its own.

    Immutable, and the pixels are write-locked, for :func:`make_stamp`'s reason:
    one capture is held by the app for the rest of the session and handed to
    every stroke drawn with it, so a caller scaling or drawing into one in place
    would silently change every stroke still to come. The variants are the same
    object with a different ``rotation``/``flip``, computed once at construction
    -- a variant is made by a button click and then used for thousands of dabs.

    Equality is identity (``eq=False``): a dataclass ``__eq__`` over an ndarray
    field returns an array, and the first ``stamp == other`` anywhere would
    raise about an ambiguous truth value.
    """

    #: The captured pixels, ``(H, W, 4)`` uint8, in their untransformed
    #: orientation. Read-only.
    pixels: np.ndarray
    #: Clockwise, one of 0/90/180/270. Anything else is snapped to a quarter.
    rotation: int = 0
    flip_x: bool = False
    flip_y: bool = False
    #: ``pixels`` with the flips and the rotation applied; what a dab reads.
    image: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        source = np.ascontiguousarray(np.asarray(self.pixels, dtype=np.uint8))
        if source.ndim != 3 or source.shape[2] != 4 or min(source.shape[:2]) < 1:
            raise ValueError("a stamp is a (H, W, 4) uint8 image")
        if max(source.shape[:2]) > MAX_STAMP:
            raise ValueError(f"a stamp may not exceed {MAX_STAMP} pixels a side")
        source.setflags(write=False)
        image = _variant(source, self.rotation, bool(self.flip_x), bool(self.flip_y))
        image.setflags(write=False)
        # ``object.__setattr__`` because the dataclass is frozen; this is the
        # documented way to fill a derived field on one.
        object.__setattr__(self, "pixels", source)
        object.__setattr__(self, "rotation", (int(self.rotation) // 90 % 4) * 90)
        object.__setattr__(self, "flip_x", bool(self.flip_x))
        object.__setattr__(self, "flip_y", bool(self.flip_y))
        object.__setattr__(self, "image", image)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the variant that is showing -- so a quarter
        turn of a tall stamp reports a wide one, which is what places it."""
        return (int(self.image.shape[1]), int(self.image.shape[0]))

    @property
    def step(self) -> int:
        """What ``spacing`` is a fraction of, standing in for a diameter.

        The longer side rather than the shorter or the mean: spacing is "how far
        apart the dabs are as a share of the dab", and taking the short side of
        a long thin tip would put a hundred stamps where the user asked for ten.
        """
        width, height = self.size
        return max(width, height)

    def rotated(self, quarters: int = 1) -> Stamp:
        """The next quarter turn clockwise. The variants *cycle* -- four turns
        is where you started -- because the control is one button pressed
        repeatedly rather than a number to be typed."""
        return Stamp(
            self.pixels,
            rotation=self.rotation + 90 * int(quarters),
            flip_x=self.flip_x,
            flip_y=self.flip_y,
        )

    def flipped(self, axis: str) -> Stamp:
        """Toggle one mirror. ``"x"`` mirrors left-to-right, ``"y"`` top-to-
        bottom -- the axis the flip is *about*, which is how the transform menu
        beside it already names the pair."""
        return Stamp(
            self.pixels,
            rotation=self.rotation,
            flip_x=self.flip_x if axis != "x" else not self.flip_x,
            flip_y=self.flip_y if axis != "y" else not self.flip_y,
        )


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
    #: Which way an angled nib points, in degrees. Meaningless to a disc (a
    #: turned circle is a circle) and read only by :data:`ANGLED_NIBS`.
    angle: float = 0.0
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
    #: The image this stroke stamps instead of a coverage disc, or None for
    #: every stroke this class drew before image brushes existed. Dropped in
    #: ``__post_init__`` for a mode that has no colour of its own to place; see
    #: :data:`STAMP_MODES`.
    stamp: Stamp | None = None
    #: One of :data:`STAMP_ALIGN`. Meaningless without a stamp.
    stamp_align: str = "free"

    coverage: np.ndarray = field(init=False)
    #: The whole stroke's union, for the single undo patch pushed at release.
    dirty: tuple[int, int, int, int] | None = field(init=False, default=None)
    #: What has been marked *since the last recomposite*, as separate
    #: rectangles. See :meth:`take_touched` for why this is a list where
    #: ``dirty`` is one box.
    touched: list[tuple[int, int, int, int]] = field(init=False, default_factory=list)
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
    #: Which lattice cells an *aligned* image stroke has already stamped. A
    #: performance guard rather than a semantic one: re-stamping a cell writes
    #: exactly the same bytes (that is what alignment buys), so skipping it
    #: cannot change the picture -- it only stops a stationary cursor doing a
    #: full-tip composite every frame for as long as the button is held.
    _cells: set[tuple[int, int]] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self) -> None:
        width, height = self.size
        self.coverage = np.zeros((height, width), dtype=np.float32)
        if self.stamp is not None and self.mode not in STAMP_MODES:
            self.stamp = None
        if self.stamp_align not in STAMP_ALIGN:
            self.stamp_align = STAMP_ALIGN[0]
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
        if self.stamp is not None:
            return max(0.5, self.stamp.step * self.spacing)
        return max(0.5, self.diameter * self.spacing)

    @property
    def pixel(self) -> bool:
        """Whether this stroke walks whole pixels rather than the spacing.

        Never, with an image stamp, whatever the nib says. The pixel walk emits
        a dab per whole pixel along the line -- which is what a one-pixel pencil
        wants and is a full-tip composite per pixel of travel for anything else.
        The nib is not refused, because it is a checkbox that stays ticked while
        the user tries a captured brush; it is simply not what places a picture.
        """
        return self.stamp is None and self.nib in PIXEL_NIBS

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
        if self.stamp is not None:
            # Before the disc is even asked for. Hardness, the nib and the speed
            # taper all describe a *generated* falloff, and a captured picture
            # has none of the three: scaling it per dab would mean resampling
            # the user's own pixels several times a frame, which is a different
            # tool (a scaled brush) rather than this one honouring a slider.
            self._image_dab(point, target)
            return
        # 1.0 rather than ``self.hardness`` for a pixel nib: neither reads it,
        # and passing it through would key the shared stamp cache on a value
        # that cannot change the answer.
        stamp = make_stamp(
            diameter,
            1.0 if self.pixel else self.hardness,
            self.nib,
            # Quantised into the cache key on purpose: a stamp per tenth of a
            # degree is a cache miss per dab of a rotating stroke, and half a
            # degree is below what a hard-edged nib can even express.
            round(float(self.angle) * 2.0) / 2.0 if self.nib in ANGLED_NIBS else 0.0,
        )
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

    def _image_dab(self, point: tuple[float, float], target: np.ndarray) -> None:
        """One dab of an image stamp: where it lands, and its wrapped pieces.

        **Two placements, and the difference between them is the whole of the
        overlap question.**

        *Free* centres the picture on the cursor, which is what a brush does.
        Consecutive dabs then overlap by whatever the spacing left over, and the
        accumulation rule in :meth:`_place` is what stops that compounding.

        *Aligned* snaps the dab to a lattice of the stamp's own size, phased on
        the canvas origin -- so every position inside one cell produces the
        **same rectangle of the same pixels**. Dragging back and forth over a
        cell therefore restamps it identically, and identical is idempotent
        under the coverage-max rule below: the second visit adds no coverage and
        writes nothing. That is what makes this mode a *pattern* -- two strokes
        that cover the same area produce the same picture, and two neighbouring
        cells line up rather than meeting at a seam wherever the mouse happened
        to be. The lattice is anchored on the canvas rather than on the press,
        so it is the same lattice in every stroke, on every layer and on every
        frame, which is the only anchoring under which a pattern painted in two
        sittings still tiles.
        """
        stamp = self.stamp
        assert stamp is not None
        width, height = stamp.size
        if self.stamp_align == "aligned":
            left = math.floor(math.floor(point[0]) / width) * width
            top = math.floor(math.floor(point[1]) / height) * height
            if (left, top) in self._cells:
                return
            self._cells.add((left, top))
        else:
            # The same anchoring the pixel nib uses: floor to the pixel the dab
            # is on, then grow left and up by half the tip. An even tip has no
            # centre pixel, so the extra pixel lands on the left/up side --
            # ``p - width // 2`` puts more of an even tip before the anchor
            # than after it, which is what ``test_image_brush`` pins -- and it
            # is a half-pixel choice that a picture with a hard edge would
            # otherwise make visible as a one-pixel jitter along a stroke.
            left = int(math.floor(point[0])) - width // 2
            top = int(math.floor(point[1])) - height // 2
        for rect, (sx, sy) in tiling.pieces(
            (left, top, left + width, top + height), self.size, self._axes
        ):
            x0, y0, x1, y1 = rect
            self._place(stamp.image[sy : sy + (y1 - y0), sx : sx + (x1 - x0)], rect, target)
            self._mark(rect)

    def _place(
        self, piece: np.ndarray, rect: tuple[int, int, int, int], target: np.ndarray
    ) -> None:
        """Composite one image dab onto the **live** layer, coverage-max.

        The ``_filter`` path rather than the ``_resolve`` one, and for a reason
        of arithmetic rather than of taste: ``_resolve`` recomputes a region
        from ``before`` and one coverage number, which works because every dab
        of a paint stroke lays down the *same colour*. An image stamp lays down
        a different colour per pixel per dab, so "recompute from the pre-stroke
        pixels" would need the whole stroke's colours kept as well -- a
        canvas-sized RGBA plane, four times what the coverage buffer costs, to
        defend against a compounding this does not have.

        **The overlap rule: coverage still accumulates with maximum**, exactly
        as the class docstring says it does for every other dab, and this is how
        that survives a live composite. ``self.coverage`` keeps the greatest
        coverage this stroke has laid at each pixel. A dab writes not its own
        alpha but the *increment* that takes the pixel from that recorded
        coverage to its own::

            share = (a - covered) * opacity / (1 - covered * opacity)

        Composite that over what is already there and the stroke's total
        contribution comes out at exactly ``max(a, covered) * opacity`` -- the
        identity ``1 - (1 - share)(1 - covered * opacity) = a * opacity``. So a
        half-transparent tip dragged slowly over itself is half-transparent,
        not opaque; a stroke drawn at two frames a second and the same stroke
        drawn at sixty are the same picture; and a spray of image dabs does not
        darken where the cloud happens to be dense.

        Where a dab covers a pixel *less* than the stroke already has, its share
        is zero and it writes nothing -- so within one stroke the first dab to
        reach a pixel at full alpha owns its colour. That is a real divergence
        from Aseprite, where a later stamp paints over an earlier one, and it is
        the price of a path that never rewrites a pixel it has already shown.
        Lift the button and stamp again to put a picture *over* one already
        placed: coverage is per stroke, so a second stroke starts clear.

        Erasing rides the same increment from the other side: it multiplies
        alpha by ``1 - share`` per dab, and that product telescopes to
        ``1 - max(a)`` by the same identity -- so an image-shaped eraser cuts
        exactly the tip's alpha however many times it passes over.
        """
        x0, y0, x1, y1 = rect
        src = piece.astype(np.float32)
        alpha = self._weights(rect, src[..., 3] / 255.0)
        if self.clip is not None:
            alpha = np.minimum(alpha, 1.0)

        covered = self.coverage[y0:y1, x0:x1]
        gain = np.clip(alpha - covered, 0.0, None) * self.opacity
        denominator = 1.0 - covered * self.opacity
        # ``composite.over``'s masked-lane fix: ``where=`` does not promise the
        # masked lanes go unevaluated, so a SIMD lane with a zero denominator
        # (fully covered at full opacity -- its gain is zero too) still ran
        # 0/0 and raised under ``np.errstate(all="raise")``. Divide by one
        # there and select; bit-identical everywhere the old form defined a
        # value.
        lit = denominator > 0.0
        share = np.empty_like(gain)
        np.divide(gain, np.where(lit, denominator, 1.0), out=share)
        share = np.where(lit, share, 0.0)
        # After the share is computed from it, never before.
        np.maximum(covered, alpha, out=covered)
        if not share.any():
            return

        crop = target[y0:y1, x0:x1].astype(np.float32)
        if self.mode == "erase":
            out = crop.copy()
            out[..., 3] = crop[..., 3] * (1.0 - share)
        else:
            # ``composite.paint_colour``'s formula with a per-pixel colour --
            # which is why it is spelled out here rather than called: that one
            # takes a single RGBA. Straight alpha throughout, so stamping onto
            # emptiness gives the tip's colours rather than the tip faded
            # toward black.
            dst_a = crop[..., 3] / 255.0
            out_a = share + dst_a * (1.0 - share)
            # The same masked-lane fix as ``share`` above: ``out_a == 0``
            # implies ``share == 0``, so the masked lanes were 0/0.
            lit = out_a > 0.0
            frac = np.empty_like(share)
            np.divide(share, np.where(lit, out_a, 1.0), out=frac)
            frac = np.where(lit, frac, 0.0)
            out = np.empty_like(crop)
            out[..., :3] = crop[..., :3] + (src[..., :3] - crop[..., :3]) * frac[..., None]
            out[..., 3] = out_a * 255.0
        if self.alpha_lock:
            # Restored from the live crop rather than from ``before``, which is
            # the same value: the lock is applied on every dab, so this stroke
            # has never moved the channel.
            out[..., 3] = crop[..., 3]
        target[y0:y1, x0:x1] = composite.to_uint8_255(out)

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
        elif self.mode == "copy":
            # Neither opacity nor coverage: a threshold, at the same half-pixel
            # this package uses everywhere else for "is this pixel inside".
            colour = np.asarray(self.colour, dtype=np.float32)
            hit = (cov >= SHADE_COVERAGE)[..., None]
            out = np.where(hit, colour, before)
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

    #: When the union of the touched rectangles is no more than this multiple of
    #: their combined area, :meth:`take_touched` hands back the union instead of
    #: the parts. That is exactly the case where one recomposite is the better
    #: call: the pieces overlap or sit next to each other, so the union costs
    #: barely more area and saves every per-call cost. Consecutive dabs along a
    #: stroke are that case; mirrored dabs at opposite corners are emphatically
    #: not (their union is hundreds of times their area), which is what makes
    #: this one rule cover both.
    TOUCH_UNION_RATIO = 2.0

    #: A hard ceiling on the number of separate recomposites one flush can ask
    #: for, whatever the ratio says. A ``spray`` burst emits its whole count
    #: before the flush, times the mirrors, so this is the case it exists for.
    TOUCH_RECTS = 256

    def take_touched(self) -> list[tuple[int, int, int, int]]:
        """The rectangles marked since the last call, and clear them.

        **Separate from ``dirty``, and the split is the whole point.** The two
        answer different questions: ``dirty`` is "what does the undo patch have
        to cover", asked once at release, where one box is right (a multi-rect
        edit type would need eviction accounting of its own for a saving
        measured in kilobytes). This is "what has to be recomposited", asked
        after **every dab**, and there one box is badly wrong twice over:

        * **A stroke's union grows as it moves**, so dab N recomposited the
          bounding box of dabs 1..N. Measured on a 512x512 canvas with an 8px
          nib: the last quarter of a 200-move stroke recomposited **33x** the
          area of the first quarter, for a stroke that never covers more of the
          canvas per dab than the nib does.
        * **A mirror puts one dab in two or four places far apart**, so the
          union of a ``symmetry="xy"`` dab is a box spanning both mirrored
          positions on both axes -- **95%** of the canvas per dab against 33%
          with symmetry off, and 611ms against 213ms of wall clock over the
          same stroke. That is the cliff the Aseprite parity programme's P1
          appendix carried as "the
          measured symmetry=xy 16x per-dab invalidation cliff (union-rect
          defect)".

        Both are the same defect -- a union standing in for a set -- and both go
        away by keeping the pieces. See
        ``docs/measurements/2026-08-20-stroke-invalidation.md``.
        """
        rects, self.touched = self.touched, []
        if len(rects) < 2:
            return rects
        x0 = min(r[0] for r in rects)
        y0 = min(r[1] for r in rects)
        x1 = max(r[2] for r in rects)
        y1 = max(r[3] for r in rects)
        union = (x0, y0, x1, y1)
        if len(rects) > self.TOUCH_RECTS:
            return [union]
        area = max(1, (x1 - x0) * (y1 - y0))
        parts = sum(max(0, r[2] - r[0]) * max(0, r[3] - r[1]) for r in rects)
        # Consecutive dabs along a stroke overlap heavily, so their union is
        # barely bigger than their sum and one call is the better answer.
        # Mirrored dabs at opposite corners have a union hundreds of times
        # their sum, and there the parts win by the same arithmetic.
        return [union] if area <= parts * self.TOUCH_UNION_RATIO else rects

    def _mark(self, rect: tuple[int, int, int, int]) -> None:
        self.touched.append(rect)
        if self.dirty is None:
            self.dirty = rect
            return
        a, b, c, d = self.dirty
        x0, y0, x1, y1 = rect
        self.dirty = (min(a, x0), min(b, y0), max(c, x1), max(d, y1))
