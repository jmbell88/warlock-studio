"""Whole-layer image filters: pure functions from pixels to pixels.

Every one takes ``(H, W, 4)`` uint8 straight-alpha pixels and returns a new
array of the same shape. No layer, no document, no selection and no history --
the document owns all four, and keeping them out is what makes "does levels
clip at the right end" a plain assertion on a five-pixel array.

Three rules run through the whole file.

**Colour filters do not touch alpha.** Brightness, contrast, levels and
hue/saturation are about the colour of what is there, and changing coverage as
a side effect of a tone adjustment is a bug in every editor that has ever done
it. Blur is the one exception, and it is an exception on purpose: blurring a
layer's edge is most of why anybody blurs a layer.

**Blur and sharpen premultiply first.** A straight-alpha buffer stores an
arbitrary colour under a transparent pixel -- ``paint_colour`` leaves the paint
there and the alpha at zero, which is invisible and correct -- so a blur that
averaged the raw channels would drag that invisible colour into the visible
edge as a dark or bright halo. Premultiplying, blurring and dividing back out
is the standard fix and the only one that is right rather than close.

**The parameters are declared as data.** :data:`FILTERS` maps a name onto its
defaults and its function, and the panel enumerates it -- the same reason
``primitives.GENERATORS`` is data in Clay. A sixth filter is an entry here and
no edit in any pane.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from . import composite as cp

__all__ = ["FILTERS", "apply_named", "blur", "brightness_contrast", "hue_saturation",
           "levels", "sharpen"]


def _rgb(pixels: np.ndarray) -> np.ndarray:
    return pixels[..., :3].astype(np.float32)


def _rejoin(pixels: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """New RGB, the original alpha, back to uint8 with the house rounding."""
    out = pixels.copy()
    # ``to_uint8_255`` rather than the expression: identical semantics, and it
    # carries the native kernel and the float32 gate that decides when the
    # kernel is safe. The enumeration in ``native/warlockc.h`` names the call
    # sites, and a hand-rolled copy is one this file would not appear in.
    out[..., :3] = cp.to_uint8_255(rgb)
    return out


# --- tone -------------------------------------------------------------------


def brightness_contrast(
    pixels: np.ndarray, *, brightness: float = 0.0, contrast: float = 0.0
) -> np.ndarray:
    """Both in -1..1, both zero being the identity.

    Contrast pivots on mid-grey rather than on the image's own mean: a pivot
    that moved with the content would make the same slider do different things
    to two halves of one drawing, and the point of a contrast control is that
    it is predictable.
    """
    rgb = _rgb(pixels)
    if brightness:
        rgb = rgb + float(brightness) * 255.0
    if contrast:
        # tan-shaped rather than linear, so the top of the slider is a real
        # increase rather than a factor of two. The +1 / -1 ends are clamped
        # short of infinity for the obvious reason.
        factor = (1.0 + float(np.clip(contrast, -0.999, 0.999))) / (
            1.0 - float(np.clip(contrast, -0.999, 0.999))
        )
        rgb = (rgb - 127.5) * factor + 127.5
    return _rejoin(pixels, rgb)


def levels(
    pixels: np.ndarray, *, black: float = 0.0, white: float = 255.0, gamma: float = 1.0
) -> np.ndarray:
    """Remap ``black``..``white`` onto 0..255 with a gamma in between.

    A white point at or below the black point would divide by zero; it is
    treated as the degenerate ramp it is (everything above black goes to
    white), rather than refused, because a slider can be dragged there and a
    filter that raised mid-drag would take the popup down with it.
    """
    rgb = _rgb(pixels)
    lo, hi = float(black), float(white)
    span = hi - lo
    if span <= 0.0:
        normalised = (rgb > lo).astype(np.float32)
    else:
        normalised = np.clip((rgb - lo) / span, 0.0, 1.0)
    if gamma > 0.0 and gamma != 1.0:
        normalised = normalised ** (1.0 / float(gamma))
    return _rejoin(pixels, normalised * 255.0)


# --- colour -----------------------------------------------------------------


def hue_saturation(
    pixels: np.ndarray, *, hue: float = 0.0, saturation: float = 0.0, lightness: float = 0.0
) -> np.ndarray:
    """Rotate hue (-1..1 is a full turn), scale saturation, lift lightness.

    HSL rather than HSV, because the lightness control has to be able to reach
    white as well as black and HSV's "value" cannot. Written out rather than
    taken from a library so it stays numpy-only and vectorised: Pillow's
    conversions are per-image mode changes that would drop the alpha channel
    this has to preserve.
    """
    rgb = _rgb(pixels) / 255.0
    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    light = (high + low) * 0.5
    span = high - low

    # Saturation is undefined for a grey pixel; zero is the value that makes
    # every formula below leave it grey, which is what it should stay.
    denominator = 1.0 - np.abs(2.0 * light - 1.0)
    sat = np.divide(span, denominator, out=np.zeros_like(span), where=denominator > 1e-6)

    hue_deg = _hue_of(rgb, high, span)
    hue_deg = np.mod(hue_deg + float(hue) * 360.0, 360.0)
    sat = np.clip(sat * (1.0 + float(saturation)), 0.0, 1.0)
    light = np.clip(light + float(lightness), 0.0, 1.0)
    return _rejoin(pixels, _from_hsl(hue_deg, sat, light) * 255.0)


def _hue_of(rgb: np.ndarray, high: np.ndarray, span: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    safe = np.where(span > 1e-6, span, 1.0)
    out = np.where(
        high == r,
        np.mod((g - b) / safe, 6.0),
        np.where(high == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    return np.where(span > 1e-6, out * 60.0, 0.0)


def _from_hsl(hue_deg: np.ndarray, sat: np.ndarray, light: np.ndarray) -> np.ndarray:
    chroma = (1.0 - np.abs(2.0 * light - 1.0)) * sat
    sixth = hue_deg / 60.0
    second = chroma * (1.0 - np.abs(np.mod(sixth, 2.0) - 1.0))
    zero = np.zeros_like(chroma)

    sector = np.floor(sixth).astype(np.int32) % 6
    table = np.stack(
        [
            np.stack([chroma, second, zero], axis=-1),
            np.stack([second, chroma, zero], axis=-1),
            np.stack([zero, chroma, second], axis=-1),
            np.stack([zero, second, chroma], axis=-1),
            np.stack([second, zero, chroma], axis=-1),
            np.stack([chroma, zero, second], axis=-1),
        ],
        axis=0,
    )
    picked = np.take_along_axis(table, sector[None, ..., None], axis=0)[0]
    return np.clip(picked + (light - chroma * 0.5)[..., None], 0.0, 1.0)


# --- spatial ----------------------------------------------------------------


def _premultiplied(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = pixels[..., 3].astype(np.float32) / 255.0
    return pixels[..., :3].astype(np.float32) * alpha[..., None], alpha


def _straight(rgb: np.ndarray, alpha: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Divide the premultiplication back out and narrow to uint8.

    The narrowing goes through :func:`composite.to_uint8_255` rather than
    spelling ``clip(x + 0.5)`` again: that expression is what the shipped
    ``warlockc_to_uint8_255_f32`` kernel computes, and ``out`` is C-contiguous
    float32 by construction, so it satisfies that function's own gate. Worth
    about 2% of ``blur`` on a 2048 square layer -- kept for the deduplicated
    expression rather than for the speed.

    The division writes into ``out`` directly. ``np.divide(..., out=zeros_like,
    where=...)`` allocated a second full plane per call purely to supply the
    zero an unlit pixel keeps, which ``out[..., :3] = 0.0`` states in place.
    """
    out = np.empty(pixels.shape, dtype=np.float32)
    lit = alpha[..., None]
    out[..., :3] = 0.0
    np.divide(rgb, lit, out=out[..., :3], where=lit > 1e-6)
    out[..., 3] = alpha * 255.0
    return cp.to_uint8_255(out)


def blur(pixels: np.ndarray, *, radius: float = 2.0) -> np.ndarray:
    """Gaussian blur, alpha included, premultiplied so edges do not halo."""
    if radius <= 0.0:
        return pixels.copy()
    rgb, alpha = _premultiplied(pixels)
    return _straight(
        np.stack([_gaussian(rgb[..., c], radius) for c in range(3)], axis=-1),
        _gaussian(alpha, radius),
        pixels,
    )


def sharpen(pixels: np.ndarray, *, amount: float = 0.5, radius: float = 1.5) -> np.ndarray:
    """Unsharp mask: the image plus its difference from a blurred copy.

    Premultiplied like the blur, and for the same reason -- an unsharp mask is
    a blur with a subtraction on top, so it inherits the halo exactly.
    """
    if amount <= 0.0:
        return pixels.copy()
    rgb, alpha = _premultiplied(pixels)
    soft = np.stack([_gaussian(rgb[..., c], radius) for c in range(3)], axis=-1)
    sharp = np.clip(rgb + (rgb - soft) * float(amount), 0.0, 255.0)
    # Alpha is *not* sharpened: the coverage a user drew is the coverage they
    # meant, and ringing on it shows up as a bitten-out edge rather than as
    # crispness.
    return _straight(sharp, alpha, pixels)


def _gaussian(plane: np.ndarray, radius: float) -> np.ndarray:
    """One separable pass per axis, in float32.

    Pillow's ``GaussianBlur`` is the obvious alternative and is not usable
    here: it works on 8-bit channels, so premultiplying before it and dividing
    after would quantise twice and leave banding exactly where the halo used
    to be.

    **This stays on ``np.convolve``, and a native kernel for it was built,
    measured and rejected** -- see docs/measurements/2026-08-09-native-batch-2.md.
    The short version is that the two halves of the usual argument pull against
    each other here. ``np.convolve`` reaches ``cblas_sdot`` for contiguous
    float32, so its summation order is OpenBLAS's and varies with CPU dispatch;
    a kernel's bar in this repository is bit-identity, and there is no fixed
    order here to be identical *to*. Writing a deterministic reference to
    restore that bar costs 1.7x on its own (273 ms against 164 ms at 2048
    square, radius 8), because a shifted-slice accumulation is 33 whole-array
    passes where BLAS keeps a row in cache -- and a bit-identical kernel may not
    vectorise its reduction, so it wins that back and little more. Net: 24%
    faster with the DLL, 34% slower without one, and vendor/ is gitignored, so
    the second is what a checkout gets.

    What actually made the live preview affordable is upstream of here:
    ``Document.preview_filter`` memoises the filtered array for the life of a
    session, so this runs once per parameter change instead of once per frame.
    """
    kernel = _kernel(radius)
    if len(kernel) < 2:
        return plane.astype(np.float32, copy=True)
    padded = np.pad(plane.astype(np.float32), len(kernel) // 2, mode="edge")
    rows = np.apply_along_axis(lambda row: np.convolve(row, kernel, "valid"), 1, padded)
    return np.apply_along_axis(lambda col: np.convolve(col, kernel, "valid"), 0, rows)


def _kernel(radius: float) -> np.ndarray:
    sigma = max(float(radius), 1e-3) / 2.0
    half = max(int(round(float(radius) * 2.0)), 1)
    x = np.arange(-half, half + 1, dtype=np.float32)
    weights = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (weights / weights.sum()).astype(np.float32)


# --- the registry -----------------------------------------------------------

# name -> (defaults, function). The defaults are a complete call and every value
# is the identity, so opening a filter's popup changes nothing until a slider is
# moved -- which is what makes the live preview safe to run on every frame.
FILTERS: dict[str, tuple[dict[str, float], Callable[..., np.ndarray]]] = {
    "brightness / contrast": ({"brightness": 0.0, "contrast": 0.0}, brightness_contrast),
    "hue / saturation": (
        {"hue": 0.0, "saturation": 0.0, "lightness": 0.0},
        hue_saturation,
    ),
    "levels": ({"black": 0.0, "white": 255.0, "gamma": 1.0}, levels),
    "blur": ({"radius": 0.0}, blur),
    "sharpen": ({"amount": 0.0, "radius": 1.5}, sharpen),
}

# What each parameter's slider spans. Held beside the registry rather than in
# the pane so a filter is one entry in one file; a parameter with no range here
# is drawn as a plain number rather than dropped.
RANGES: dict[str, tuple[float, float]] = {
    "brightness": (-1.0, 1.0),
    "contrast": (-1.0, 1.0),
    "hue": (-1.0, 1.0),
    "saturation": (-1.0, 1.0),
    "lightness": (-1.0, 1.0),
    "black": (0.0, 255.0),
    "white": (0.0, 255.0),
    "gamma": (0.1, 4.0),
    "radius": (0.0, 32.0),
    "amount": (0.0, 3.0),
}


def apply_named(name: str, pixels: np.ndarray, **params: Any) -> np.ndarray:
    """Run a registered filter, filling in any parameter not supplied.

    Unknown *keys* are dropped rather than raising: the panel remembers a
    filter's last values per filter, and a renamed parameter would otherwise
    turn a stale settings entry into a TypeError out of a slider drag.
    """
    defaults, func = FILTERS[name]
    return func(pixels, **{key: params.get(key, value) for key, value in defaults.items()})
