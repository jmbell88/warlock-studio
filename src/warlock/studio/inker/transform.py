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
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _resample(pixels: np.ndarray, run) -> np.ndarray:
    from PIL import Image

    if pixels.ndim == 2:  # a mask: no alpha to bleed, so no premultiply
        return np.asarray(run(Image.fromarray(pixels, "L")), dtype=np.uint8).copy()
    plane = _premultiplied(pixels)
    channels = [
        np.asarray(run(Image.fromarray(plane[..., i].astype(np.uint8), "L")), dtype=np.uint8)
        for i in range(4)
    ]
    return _unpremultiplied(np.stack(channels, axis=2).astype(np.float32))


def scale(pixels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    width, height = max(1, int(size[0])), max(1, int(size[1]))
    return _resample(pixels, lambda im: im.resize((width, height), Image.LANCZOS))


def rotate(pixels: np.ndarray, degrees: float, *, expand: bool = False) -> np.ndarray:
    from PIL import Image

    return _resample(
        pixels,
        lambda im: im.rotate(float(degrees), Image.BICUBIC, expand=expand, fillcolor=0),
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
