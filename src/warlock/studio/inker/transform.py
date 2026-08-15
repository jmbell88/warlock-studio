"""Whole-plane geometry: flip, rotate, scale, crop, canvas resize.

Pure array functions, one plane at a time. The document applies each of them to
every layer and to the selection mask, which is the only way the three can
never disagree about what size the canvas is.

Resampling is Pillow's, on straight alpha, and that is deliberate: a bilinear
filter on straight alpha bleeds the colour of fully transparent pixels into the
edge. Every function here that resamples premultiplies first and unpremultiplies
after -- the one place in the codebase that touches premultiplied alpha, and it
is confined to these few lines.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from . import composite

#: The axes :func:`flip` accepts, and the list a menu is built from. One owner:
#: the function validates against this tuple rather than against a chain of its
#: own ``if``s, so a third axis cannot be offered by a pane and refused by the
#: function, or accepted by the function and missing from every menu.
FLIPS = ("horizontal", "vertical")

#: Which numpy axis each of them reverses. Keyed off ``FLIPS`` above rather
#: than restating the names, so the two cannot drift apart either.
_FLIP_AXIS = dict(zip(FLIPS, (1, 0), strict=True))


def flip(pixels: np.ndarray, axis: str) -> np.ndarray:
    which = _FLIP_AXIS.get(axis)
    if which is None:
        raise ValueError(f"unknown flip axis {axis!r}")
    return np.ascontiguousarray(np.flip(pixels, axis=which))


def rotate90(pixels: np.ndarray, quarters: int = 1) -> np.ndarray:
    """Counter-clockwise quarter turns. Lossless -- no resampling at all."""
    return np.ascontiguousarray(np.rot90(pixels, int(quarters) % 4))


def _premultiplied(pixels: np.ndarray) -> np.ndarray:
    out = pixels.astype(np.float32)
    out[..., :3] *= out[..., 3:4] / 255.0
    return out


def _unpremultiplied(pixels: np.ndarray) -> np.ndarray:
    alpha = pixels[..., 3:4] / 255.0
    rgb = np.divide(pixels[..., :3], alpha, out=np.zeros_like(pixels[..., :3]), where=alpha > 0.0)
    out = np.empty_like(pixels)
    out[..., :3] = rgb
    out[..., 3:4] = pixels[..., 3:4]
    # ``to_uint8_255``: the same expression, in the one place that owns it.
    return composite.to_uint8_255(out)


#: How a resize or a rotate decides what a destination pixel holds. ``smooth``
#: is the filtered path this module has always taken -- Lanczos for a scale,
#: bicubic for a turn -- and ``nearest`` is the one that copies a source pixel
#: whole.
#:
#: The second is not a lower-quality option, it is the only correct one for a
#: drawing whose pixels are the artwork: a filter over a 32x32 sprite scaled to
#: 128 produces a blurred sprite with a few thousand new colours in it, and a
#: filter over a pixel-art *rotation* is worse again. Offered as a choice rather
#: than inferred from the canvas size, because "small" is not what makes a
#: document pixel art.
RESAMPLES = ("smooth", "nearest")


def _filter(resample: str, smooth: int) -> int:
    from PIL import Image

    return Image.NEAREST if resample == "nearest" else smooth


def _resample(pixels: np.ndarray, run, *, straight: bool = False) -> np.ndarray:
    """Run a Pillow operation over a plane, premultiplied unless ``straight``.

    ``straight`` is for nearest neighbour and for nothing else. Every filtered
    path has to premultiply, because it *mixes* pixels and a mix with a fully
    transparent one drags that pixel's colour into the edge -- but nearest
    mixes nothing, so the round trip through premultiplied alpha would be pure
    loss: dividing a rounded product back out moves the colour of every
    partially transparent pixel, which is the one thing a copy must not do.
    """
    from PIL import Image

    if pixels.ndim == 2:  # a mask: no alpha to bleed, so no premultiply
        return np.asarray(run(Image.fromarray(pixels, "L")), dtype=np.uint8).copy()
    if straight:
        return np.asarray(
            run(Image.fromarray(pixels, "RGBA")), dtype=np.uint8
        ).copy()
    plane = _premultiplied(pixels)
    channels = [
        np.asarray(run(Image.fromarray(plane[..., i].astype(np.uint8), "L")), dtype=np.uint8)
        for i in range(4)
    ]
    return _unpremultiplied(np.stack(channels, axis=2).astype(np.float32))


def scale(
    pixels: np.ndarray, size: tuple[int, int], *, resample: str = "smooth"
) -> np.ndarray:
    from PIL import Image

    width, height = max(1, int(size[0])), max(1, int(size[1]))
    how = _filter(resample, Image.LANCZOS)
    return _resample(
        pixels,
        lambda im: im.resize((width, height), how),
        straight=resample == "nearest",
    )


def rotate(
    pixels: np.ndarray, degrees: float, *, expand: bool = False, resample: str = "smooth"
) -> np.ndarray:
    from PIL import Image

    how = _filter(resample, Image.BICUBIC)
    return _resample(
        pixels,
        # A single 0 is the fill for the one-band mask; an RGBA plane is only
        # reached on the straight path and needs the four-tuple spelling of the
        # same transparent black.
        lambda im: im.rotate(
            float(degrees),
            how,
            expand=expand,
            fillcolor=0 if im.mode == "L" else (0, 0, 0, 0),
        ),
        straight=resample == "nearest",
    )


def crop(pixels: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in rect)
    return np.ascontiguousarray(pixels[y0:y1, x0:x1])


#: The nine anchor positions, as the fraction of the *slack* that goes before
#: the image on each axis. Written out as a table rather than derived from a
#: 3x3 index so the names are the contract: a pane draws the grid in whatever
#: order it likes and cannot get the mapping subtly wrong.
ANCHORS: dict[str, tuple[float, float]] = {
    "top-left": (0.0, 0.0),
    "top": (0.5, 0.0),
    "top-right": (1.0, 0.0),
    "left": (0.0, 0.5),
    "centre": (0.5, 0.5),
    "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0),
    "bottom": (0.5, 1.0),
    "bottom-right": (1.0, 1.0),
}


def anchor_offset(
    old: tuple[int, int], new: tuple[int, int], anchor: str = "top-left"
) -> tuple[int, int]:
    """Where the old image lands inside the new canvas, for a named anchor.

    Negative when the canvas shrinks, which is exactly what ``resize_canvas``
    already means by a negative offset -- growing and cropping are one
    operation and an anchor is one number either way. Rounded rather than
    floored, so a two-pixel growth centred puts one pixel on each side rather
    than both on the right.
    """
    fx, fy = ANCHORS.get(anchor, ANCHORS["top-left"])
    return (
        int(round((new[0] - old[0]) * fx)),
        int(round((new[1] - old[1]) * fy)),
    )


def resize_canvas(
    pixels: np.ndarray, size: tuple[int, int], offset: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """A bigger or smaller canvas with the pixels placed, never rescaled."""
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    shape = (height, width) if pixels.ndim == 2 else (height, width, pixels.shape[2])
    out = np.zeros(shape, dtype=np.uint8)
    ox, oy = int(offset[0]), int(offset[1])

    sx0, sy0 = max(0, -ox), max(0, -oy)
    dx0, dy0 = max(0, ox), max(0, oy)
    copy_w = min(pixels.shape[1] - sx0, width - dx0)
    copy_h = min(pixels.shape[0] - sy0, height - dy0)
    if copy_w > 0 and copy_h > 0:
        out[dy0 : dy0 + copy_h, dx0 : dx0 + copy_w] = pixels[
            sy0 : sy0 + copy_h, sx0 : sx0 + copy_w
        ]
    return out


# --- carrying a rectangle through the same geometry --------------------------
#
# A slice is metadata *about* the canvas rather than a plane on it, so every
# whole-plane op above needs a partner that puts a rectangle through the
# identical transform. They are separate functions rather than a mode on the
# ones above because there is nothing here to resample: a flip and a quarter
# turn are exact, and the one thing that can go wrong -- a rectangle left
# describing pixels that have moved -- is invisible in the image and only shows
# up in an export somebody else reads.
#
# What is here is a *point* mapper per operation plus one :func:`map_rect` that
# carries a rectangle through any of them, which is the same argument
# ``panes/inker_canvas._corners`` makes about drawing: mapping the two corners
# and re-ordering them is right for all eight orientations, where mapping x and
# y independently is right only at rotation 0.
#
# Bounds are ``x0 y0 x1 y1`` with the far edge **exclusive**, as everywhere else
# in this package. That is what makes mirroring exact rather than off by one:
# ``[x0, x1)`` reflects to ``[w - x1, w - x0)`` with no rounding anywhere.


def flip_point(
    point: tuple[float, float], size: tuple[int, int], axis: str
) -> tuple[float, float]:
    """One canvas point through :func:`flip`. Validated against ``FLIPS``, so a
    third axis cannot be accepted here and refused there."""
    if axis not in _FLIP_AXIS:
        raise ValueError(f"unknown flip axis {axis!r}")
    x, y = float(point[0]), float(point[1])
    width, height = float(size[0]), float(size[1])
    return (width - x, y) if axis == "horizontal" else (x, height - y)


def rotate90_point(
    point: tuple[float, float], size: tuple[int, int], quarters: int = 1
) -> tuple[float, float]:
    """One canvas point through :func:`rotate90`, i.e. ``np.rot90``.

    ``np.rot90`` sends the element at ``(row y, column x)`` of a ``w`` wide
    plane to ``(row w - 1 - x, column y)``. In the continuous coordinates a
    rectangle's exclusive edge lives in, that is ``(x, y) -> (y, w - x)``, and
    applying it ``quarters`` times -- swapping the size each turn -- is the
    whole of the mapping.
    """
    x, y = float(point[0]), float(point[1])
    width, height = float(size[0]), float(size[1])
    for _ in range(int(quarters) % 4):
        x, y, width, height = y, width - x, height, width
    return (x, y)


def rotate90_size(size: tuple[int, int], quarters: int = 1) -> tuple[int, int]:
    """The canvas size after ``quarters`` turns. An odd number transposes it."""
    width, height = int(size[0]), int(size[1])
    return (height, width) if int(quarters) % 2 else (width, height)


def scale_point(
    point: tuple[float, float], old: tuple[int, int], new: tuple[int, int]
) -> tuple[float, float]:
    """One canvas point through :func:`scale`. A ratio, not a resample."""
    ox, oy = max(1, int(old[0])), max(1, int(old[1]))
    return (
        float(point[0]) * float(max(1, int(new[0]))) / float(ox),
        float(point[1]) * float(max(1, int(new[1]))) / float(oy),
    )


def offset_point(
    point: tuple[float, float], offset: tuple[int, int]
) -> tuple[float, float]:
    """One canvas point through a crop or a canvas resize -- both are a
    translation, and a crop's is the negated box origin."""
    return (float(point[0]) + float(offset[0]), float(point[1]) + float(offset[1]))


def rect_from_points(
    a: tuple[float, float], b: tuple[float, float]
) -> tuple[int, int, int, int]:
    """Two mapped corners back into an ordered integer rectangle.

    The origin floors and the far edge ceils, which is ``pixelsheet``'s trim
    rule and for its reason: a rectangle that rounded inward would clip a pixel
    off the thing it exists to describe. Ordering happens *before* rounding, or
    an identical mapping rounds outward one way round and inward the other.
    """
    x0, x1 = sorted((float(a[0]), float(b[0])))
    y0, y1 = sorted((float(a[1]), float(b[1])))
    return (
        int(math.floor(x0)),
        int(math.floor(y0)),
        int(math.ceil(x1)),
        int(math.ceil(y1)),
    )


def clamp_rect(
    rect: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """A rectangle brought inside the canvas, never emptied.

    The floor is 1x1 rather than nothing, deliberately. A slice is a named
    thing with an exported identity, and a crop that happened to miss it must
    cost it its rectangle rather than its existence -- deleting it would take
    the name, the pivot and the nine-slice centre with it, none of which can be
    recovered by undoing the crop's *pixels*.
    """
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    x0, y0, x1, y1 = (int(v) for v in rect)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return (x0, y0, x1, y1)


def map_rect(
    rect: tuple[int, int, int, int],
    point: Callable[[float, float], tuple[float, float]],
    box: tuple[int, int],
) -> tuple[int, int, int, int]:
    """A rectangle through a point mapper: map both corners, order, round, clamp.

    **One function rather than one per operation.** There were five -- a
    ``flip_rect`` beside ``scale_rect`` beside ``offset_rect`` -- and each was
    the same three lines with a different point mapper substituted, which is a
    composition and not five behaviours. Worse, the caller that actually maps
    slices takes the point mapper as an argument (it has to: the pivot and the
    nine-slice centre go through the *same* mapper as the bounds, so a per-op
    rect function could not have served it) and so spelled the composition
    inline. That left the five as untested duplicates of the real path, which is
    exactly the shape two spellings drift from.

    ``box`` is what the result is clamped into, and it is a parameter rather
    than derived because it is not always the canvas: a nine-slice centre is
    clamped into *its own slice*, which is the same composition against a
    different box.
    """
    a = point(float(rect[0]), float(rect[1]))
    b = point(float(rect[2]), float(rect[3]))
    return clamp_rect(rect_from_points(a, b), box)
