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


def upscale(pixels: np.ndarray, factor: int) -> np.ndarray:
    """Magnify by a whole number, exactly: one source pixel, ``n`` x ``n`` out.

    Not ``scale`` with ``resample="nearest"``, and the difference is that this
    one cannot be approximate. Pillow's nearest resize picks a source pixel per
    destination pixel by rounding a ratio, which is right for an arbitrary size
    and puts a one-pixel jitter into the block edges at an integer one --
    ``np.repeat`` places every block exactly, so an 8x magnification of pixel
    art is the artwork with each pixel drawn eight times and nothing else.

    A factor of 1 hands the array straight back rather than copying it: this is
    an *export* path and every caller reads the result, so the off path is
    byte-identical and costs nothing at all.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return pixels
    return np.ascontiguousarray(
        np.repeat(np.repeat(pixels, factor, axis=0), factor, axis=1)
    )
