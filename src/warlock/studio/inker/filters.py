"""Whole-layer image filters: pure functions from pixels to pixels.

Every one takes ``(H, W, 4)`` uint8 straight-alpha pixels and returns a new
array of the same shape. No layer, no document, no selection and no history --
the document owns all four, and keeping them out is what makes "does levels
clip at the right end" a plain assertion on a five-pixel array.

Three rules run through the whole file.

**Colour filters do not touch alpha.** Brightness, contrast, levels and
hue/saturation are about the colour of what is there, and changing coverage as
a side effect of a tone adjustment is a bug in every editor that has ever done
it. The exceptions are the *spatial* filters, and each is an exception on
purpose. Blur and :func:`despeckle` both filter alpha along with colour --
blurring a layer's edge is most of why anybody blurs a layer, and a despeckle
that left a stray pixel's coverage behind would delete its colour and keep its
hole. :func:`outline` *placed outside* adds coverage that was not there, which
is the strongest form of the exception and the only one worth restating: the
ring is by construction drawn where the shape is not, so a version that could
not add alpha would draw nothing. Placed inside it touches no alpha at all, and
:func:`invert` and :func:`replace_colour` never do.

Two of the matte-cleanup four are alpha exceptions as well, and both are the
point rather than a side effect. :func:`alpha_threshold` exists *to* make
coverage binary, and :func:`matte_grow` exists *to* move the silhouette in or
out. :func:`defringe` deliberately is **not** one -- it recolours the
semi-transparent rim and leaves every alpha exactly where it was, so it is an
ordinary colour filter -- and :func:`remove_orphans` leaves transparent pixels
alone in both roles.

**The matte pack copies colours; it never averages them.** Averaging is
precisely the halo being removed, and on a palette-locked document a copied
colour is already a palette member where a blended one would fight the
write-path snap on the very next stroke.

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

__all__ = ["FILTERS", "alpha_threshold", "apply_named", "blur",
           "brightness_contrast", "defringe", "despeckle", "hue_saturation",
           "invert", "levels", "matte_grow", "outline", "popup_values",
           "remove_orphans", "replace_colour", "sharpen"]

RGBA = tuple[int, int, int, int]


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


def invert(
    pixels: np.ndarray, *, red: float = 0.0, green: float = 0.0, blue: float = 0.0
) -> np.ndarray:
    """Invert the channels that are switched on. 255 - c, per channel.

    Three toggles rather than one, because inverting a single channel is how
    you get a complementary palette out of a drawing and inverting all three is
    only the most common case of it. They are declared as 0/1 numbers rather
    than as bools so the registry stays one shape -- a name mapped to a value a
    slider or a checkbox can hold -- and the panel draws these three as
    checkboxes because :data:`TOGGLE_PARAMS` says so.

    **The declared defaults are all off**, i.e. the identity, which is the rule
    every filter in this file obeys and the reason a popup can preview on the
    frame it opens. What a user means by "invert" is of course all three, so the
    *popup* seeds them on: see :func:`popup_values`, which is the one place the
    two differ and is deliberately not the defaults table.

    Alpha is untouched: an inverted transparent pixel is still transparent.
    """
    out = pixels.copy()
    for index, on in enumerate((red, green, blue)):
        if on:
            # 255 - c on uint8, computed in uint8: exact, and the operand is the
            # storage format rather than a rounded trip through float.
            out[..., index] = 255 - out[..., index]
    return out


def replace_colour(
    pixels: np.ndarray,
    *,
    old: RGBA = (0, 0, 0, 255),
    new: RGBA = (0, 0, 0, 255),
    tolerance: float = 0.0,
) -> np.ndarray:
    """Rewrite every pixel within *tolerance* of *old* to *new*'s colour.

    The parameters are ``old``/``new`` and not ``from``/``to`` for the dull
    reason that ``from`` is a keyword, so ``func(**params)`` could never pass
    one -- and they are the names :func:`indexed.remap` already uses for the
    same two things.

    Two rules make this the tool people expect rather than a near miss.

    **Alpha is kept, on both sides.** ``new``'s own alpha is ignored: this is a
    recolour, and a user replacing a shade of green in a drawing has not asked
    for the antialiased edge of that shade to become opaque. Fully transparent
    pixels are skipped for the reason ``indexed.snap`` skips them -- a straight
    buffer keeps invisible colour under a zero alpha, and matching on it would
    "replace" pixels nobody can see.

    **At tolerance 0 this is exactly** :func:`indexed.remap`, which is what
    makes the two safe to have in one program: recolouring a palette slot and
    replacing a colour by hand cannot disagree about which pixels are that
    colour. Pinned by a parity test rather than by a shared implementation,
    because remap is on the write path of an indexed document and must stay a
    plain equality.

    Distance is Euclidean in RGB, so the tolerance is a radius in colour space
    -- a 0..255 slider reaches most of a cube whose long diagonal is 441.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("replace_colour takes (H, W, 4) uint8")
    out = pixels.copy()
    if out.size == 0:
        return out
    want = np.asarray(tuple(old)[:3], dtype=np.float32)
    delta = out[..., :3].astype(np.float32) - want
    distance = np.sqrt((delta * delta).sum(axis=-1))
    hit = (distance <= float(tolerance)) & (out[..., 3] > 0)
    if hit.any():
        out[..., :3][hit] = np.asarray(tuple(new)[:3], dtype=np.uint8)
    return out


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


def _shift(mask: np.ndarray, dy: int, dx: int, *, wrap: bool) -> np.ndarray:
    """``mask`` moved by one step, off-canvas filled with False -- or rolled.

    ``wrap`` is the tiling case: a document drawn as a repeating tile has no
    edge, so an outline that stopped at the border would leave a seam exactly
    where the tile joins itself. It is ``np.roll`` and nothing else, deliberately
    -- no import of ``tiling.py``, which is about *previewing* a tiled canvas and
    owns no wrapping arithmetic this could share.
    """
    if wrap:
        return np.roll(mask, (dy, dx), axis=(0, 1))
    out = np.zeros_like(mask)
    height, width = mask.shape
    out[max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = mask[
        max(-dy, 0):height + min(-dy, 0), max(-dx, 0):width + min(-dx, 0)
    ]
    return out


_CORNERS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CORNERS_8 = _CORNERS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grow(mask: np.ndarray, steps: int, corners: int, *, wrap: bool) -> np.ndarray:
    """Dilate by ``steps`` single-pixel steps.

    Iterated one step at a time rather than done in one pass with a radius: at
    the sizes an outline is drawn at (1 to 32 px) the loop is cheap, and a
    single-step dilation repeated is what makes the 4-connected case a diamond
    and the 8-connected case a square -- which is the whole difference between
    a rounded outline and a boxy one.
    """
    neighbours = _CORNERS_8 if int(corners) >= 8 else _CORNERS_4
    out = mask
    for _ in range(steps):
        grown = out
        for dy, dx in neighbours:
            grown = grown | _shift(out, dy, dx, wrap=wrap)
        out = grown
    return out


def outline(
    pixels: np.ndarray,
    *,
    colour: RGBA = (0, 0, 0, 255),
    size: float = 0.0,
    place: str = "outside",
    corners: int = 8,
    wrap: float = 0.0,
) -> np.ndarray:
    """Draw a ring of *colour* along the edge of whatever is painted.

    The shape is "alpha above zero", dilated (placed outside) or eroded (placed
    inside) and differenced against itself, which is the definition an outline
    of an *antialiased* drawing needs: a threshold on coverage would make the
    outline follow the hard edge of a soft stroke and leave the stroke's own
    fringe outside it.

    **Placed outside, this adds alpha.** It is the one stated exception to this
    file's alpha rule and it is not negotiable: the ring is by construction
    where the drawing is not. Placed inside it recolours pixels that were
    already there and leaves their alpha exactly as it found it, including a
    half-covered edge pixel, so an inside outline cannot sharpen an edge.

    Two things it does not do, both because a filter sees only the pixels it is
    handed. It is **clipped to the session rect** -- the crop
    ``Document.begin_filter`` took, which is the selection's bounds when there is
    a selection -- so an outline drawn outside a selected region stops at the
    selection, exactly as a brush would. And off-crop counts as *inside* the
    shape for the inside case, so a selection cutting through a filled area does
    not draw an outline along the cut: what is past the edge is unknown, not
    empty.
    """
    steps = int(round(float(size)))
    if steps <= 0:
        return pixels.copy()
    tiled = bool(wrap)
    shape = pixels[..., 3] > 0
    if place == "inside":
        # Erode by growing the *complement*. Off-canvas is False in the
        # complement, i.e. shape, which is the "unknown, not empty" rule above.
        ring = shape & _grow(~shape, steps, corners, wrap=tiled)
        out = pixels.copy()
        out[..., :3][ring] = np.asarray(tuple(colour)[:3], dtype=np.uint8)
        return out
    ring = _grow(shape, steps, corners, wrap=tiled) & ~shape
    out = pixels.copy()
    out[ring] = np.asarray(tuple(colour), dtype=np.uint8)
    return out


#: The widest window :func:`despeckle` will build. See it for why it is small.
DESPECKLE_MAX = 9


def despeckle(pixels: np.ndarray, *, speck: float = 0.0) -> np.ndarray:
    """Median filter: takes stray pixels out without softening an edge.

    ``speck`` is a radius in pixels -- how big a stray thing this deletes -- and
    the window it sorts is ``2·speck + 1`` across.

    A median is the right tool and a blur is not: a blur spreads the speck over
    its neighbourhood, where a median deletes it outright and leaves a hard line
    hard, which is the whole point on pixel art.

    **Its own parameter name with a small span, rather than the shared
    ``radius``, and a hard ceiling underneath that.** A median is a rank filter
    -- it sorts every window -- so it costs O(k^2 log k) per pixel per band where
    a Gaussian is O(k) and separable. On the 0..32 span ``radius`` carries for
    blur and sharpen, the top of the slider is a 65x65 window: four thousand
    samples sorted per pixel, four times over, which is seconds per call on a
    2048 square. That call is on the **frame thread** -- ``preview_filter``
    recomputes for every new parameter value and a dragged slider mints one per
    frame -- so it would read as the app hanging, against the one invariant the
    frame loop has. :data:`DESPECKLE_MAX` clamps the window as well, so a stale
    settings entry or a caller passing 32 cannot reach the unaffordable case by
    going round the slider.

    The ceiling costs nothing real: past a 9x9 window a median stops deleting
    specks and starts deleting *detail*, which is what blur is for.

    Premultiplied like the blur and for the same reason: the median of a set of
    straight-alpha colours can pick a colour that is invisible in the source and
    paint it into a visible pixel. Pillow rather than a hand-rolled sort because
    ``ImageFilter.MedianFilter`` is a C loop over a window and numpy's
    equivalent is one full-image copy per window position; imported lazily, as
    everything Pillow in this package is, so the engine still imports on a
    machine without it.
    """
    if speck <= 0.0:
        return pixels.copy()
    from PIL import Image, ImageFilter

    # Pillow takes an odd *window size*, not a radius: 1 is the identity, 3 is
    # the 3x3 median every despeckle means by "1".
    size = min(int(round(float(speck))) * 2 + 1, DESPECKLE_MAX)
    rgb, alpha = _premultiplied(pixels)
    flat = np.empty(pixels.shape, dtype=np.uint8)
    flat[..., :3] = cp.to_uint8_255(rgb)
    flat[..., 3] = cp.to_uint8_255(alpha * 255.0)
    with Image.fromarray(flat, "RGBA") as im:
        got = np.asarray(im.filter(ImageFilter.MedianFilter(size)), dtype=np.uint8)
    return _straight(
        got[..., :3].astype(np.float32), got[..., 3].astype(np.float32) / 255.0, pixels
    )


# --- the matte-cleanup pack -------------------------------------------------
#
# A BiRefNet cutout lands with a halo and a semi-transparent fringe
# (``service/matte.py`` hands the raw mask to ``apply_matte``), and the remedy
# until now was the eraser. These four are the ones that actually fix it, and
# they are entries in the registry below rather than a new pane -- which is
# exactly what the registry is for: live preview, feathered selection weighting,
# alpha-lock respect, one-undo commit and Apply-to-range all arrive for free.


def alpha_threshold(pixels: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    """Alpha becomes 255 at or above *threshold* and 0 below it.

    ``pipelines/pixel.snap_alpha``'s rule, at the editor. A partial-alpha pixel
    in a 32px sprite reads as a smudge in every engine that does not blend the
    way the preview did, and a matte's rim is nothing but partial alpha.

    0.0 is the identity, so the live preview is safe on the frame the popup
    opens; the popup itself seeds 128, because nobody opens this to snap nothing
    (``invert``'s precedent, and the second entry in :data:`POPUP_VALUES`).

    An alpha exception by design -- see the module head.
    """
    out = pixels.copy()
    level = float(threshold)
    if level <= 0.0:
        return out
    out[..., 3] = np.where(out[..., 3] >= level, 255, 0).astype(np.uint8)
    return out


def _opaque_source(pixels: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel, the RGB of the nearest fully-opaque pixel within
    *steps* 8-neighbour hops, and a mask of which pixels found one.

    Iterative propagation rather than a distance transform: *steps* is at most
    eight (both callers give it a small span for despeckle's stated reason), so
    this is at most eight full-image maximum-filter passes, and a distance
    transform would be a dependency and a second definition of "nearest" for no
    measurable gain at that size.

    Colours are **copied**, never blended -- the module head says why.
    """
    height, width = pixels.shape[:2]
    filled = pixels[..., 3] == 255
    colour = pixels[..., :3].copy()
    for _ in range(max(0, int(steps))):
        if filled.all():
            break
        # One 8-neighbour dilation: each unfilled pixel takes the first filled
        # neighbour found in a fixed order, so the result is deterministic.
        grew = filled.copy()
        picked = colour.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                src_y = slice(max(0, -dy), height - max(0, dy))
                src_x = slice(max(0, -dx), width - max(0, dx))
                dst_y = slice(max(0, dy), height - max(0, -dy))
                dst_x = slice(max(0, dx), width - max(0, -dx))
                donor = filled[src_y, src_x]
                target = ~grew[dst_y, dst_x] & donor
                if not target.any():
                    continue
                window = picked[dst_y, dst_x]
                window[target] = colour[src_y, src_x][target]
                picked[dst_y, dst_x] = window
                grown = grew[dst_y, dst_x]
                grown[target] = True
                grew[dst_y, dst_x] = grown
        if not grew.any() or np.array_equal(grew, filled):
            break
        filled, colour = grew, picked
    return colour, filled


def defringe(pixels: np.ndarray, *, fringe: float = 0.0) -> np.ndarray:
    """Semi-transparent pixels take the colour of the nearest opaque one.

    The halo a matte leaves is a rim of pixels whose *colour* is a blend of the
    subject and whatever was behind it, at partial coverage. Replacing that
    colour with the subject's own -- copied whole, from the nearest fully-opaque
    pixel within ``fringe`` steps -- removes the halo and leaves the coverage
    exactly as the matte decided it. So this is a colour filter, not an alpha
    exception: every alpha comes through untouched.

    **Its own parameter with a small span** rather than the shared ``radius``,
    for :func:`despeckle`'s stated reason: each step is a full-image pass and
    this runs on the frame thread under a live preview, where ``radius`` tops
    out at 32.
    """
    out = pixels.copy()
    steps = int(round(float(fringe)))
    if steps <= 0:
        return out
    rim = (out[..., 3] > 0) & (out[..., 3] < 255)
    if not rim.any():
        return out
    colour, filled = _opaque_source(out, steps)
    take = rim & filled
    out[..., :3][take] = colour[take]
    return out


def matte_grow(pixels: np.ndarray, *, grow: float = 0.0) -> np.ndarray:
    """Move the silhouette out (positive) or in (negative), in whole pixels.

    Positive dilates: a new rim pixel takes the nearest opaque neighbour's
    colour -- :func:`defringe`'s propagation, with the coverage following the
    colour -- and full alpha. Negative erodes: any opaque pixel with a
    non-opaque 8-neighbour loses its coverage, once per step.

    An alpha exception on purpose; coverage is the whole point.

    Grow-then-shrink is deliberately **not** an identity and is not asserted as
    one: a dilation rounds a corner off and no erosion puts it back. What holds
    is monotonicity per sign, which is what the tests pin.
    """
    out = pixels.copy()
    steps = int(round(float(grow)))
    if steps == 0:
        return out
    height, width = out.shape[:2]
    if steps > 0:
        colour, filled = _opaque_source(out, steps)
        added = filled & (out[..., 3] != 255)
        out[..., :3][added] = colour[added]
        out[..., 3][added] = 255
        return out
    for _ in range(-steps):
        opaque = out[..., 3] == 255
        if not opaque.any():
            break
        # An opaque pixel on the rim: some 8-neighbour is not opaque. Outside
        # the array counts as not opaque, so the border erodes like an edge.
        exposed = np.zeros((height, width), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.zeros((height, width), dtype=bool)
                src_y = slice(max(0, -dy), height - max(0, dy))
                src_x = slice(max(0, -dx), width - max(0, dx))
                dst_y = slice(max(0, dy), height - max(0, -dy))
                dst_x = slice(max(0, dx), width - max(0, -dx))
                shifted[dst_y, dst_x] = opaque[src_y, src_x]
                exposed |= opaque & ~shifted
        out[..., 3][exposed] = 0
    return out


def remove_orphans(pixels: np.ndarray, *, orphans: float = 0.0) -> np.ndarray:
    """An opaque pixel with no same-coloured 8-neighbour takes their commonest.

    ``pipelines/pixel.clean_orphans``'s rule at the editor. A pixel none of
    whose neighbours shares its colour is, at sprite sizes, an artefact of a
    reduction rather than a decision.

    The contrast with :func:`despeckle` is the reason both exist: a median
    deletes one-pixel *detail* wholesale, wherever it is; this deletes only
    friendless pixels, so a deliberate two-pixel highlight survives -- each of
    the pair has the other.

    Transparent pixels are left alone in **both** roles: an isolated hole is a
    silhouette decision, not an artefact.

    A toggle rather than a slider: there is no size to choose.

    The per-lonely-pixel loop at the end is fine and the vectorised test above
    it is load-bearing -- lonely pixels are few by definition, but the *test*
    runs on every pixel and a Python loop over that under a live preview is not
    acceptable.
    """
    out = pixels.copy()
    if float(orphans) <= 0.0:
        return out
    height, width = out.shape[:2]
    if height < 3 or width < 3:
        return out
    opaque = out[..., 3] > 0
    # One integer per distinct colour, so "same colour" is one comparison.
    rgb = out[..., :3].astype(np.int64)
    code = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    code = np.where(opaque, code, -1)
    padded = np.pad(code, 1, constant_values=-1)
    stack = np.stack(
        [
            padded[1 + dy: 1 + dy + height, 1 + dx: 1 + dx + width]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
        ],
        axis=-1,
    )
    lonely = opaque & ~(stack == code[..., None]).any(axis=-1)
    if not lonely.any():
        return out
    for y, x in zip(*(a.tolist() for a in np.nonzero(lonely)), strict=True):
        values = [v for v in stack[y, x].tolist() if v >= 0]
        if not values:
            continue
        winner = max(set(values), key=values.count)
        out[y, x, :3] = ((winner >> 16) & 0xFF, (winner >> 8) & 0xFF, winner & 0xFF)
    return out


# --- the registry -----------------------------------------------------------

# name -> (defaults, function). The defaults are a complete call and every value
# is the identity, so a filter run at them changes nothing -- which is what makes
# the live preview safe to run on every frame, including the first. What the
# *popup* opens on is one step away from that (:func:`popup_values`) and Invert
# is the only filter where the two differ.
FILTERS: dict[str, tuple[dict[str, Any], Callable[..., np.ndarray]]] = {
    "brightness / contrast": ({"brightness": 0.0, "contrast": 0.0}, brightness_contrast),
    "hue / saturation": (
        {"hue": 0.0, "saturation": 0.0, "lightness": 0.0},
        hue_saturation,
    ),
    "levels": ({"black": 0.0, "white": 255.0, "gamma": 1.0}, levels),
    "blur": ({"radius": 0.0}, blur),
    "sharpen": ({"amount": 0.0, "radius": 1.5}, sharpen),
    "invert": ({"red": 0.0, "green": 0.0, "blue": 0.0}, invert),
    "replace colour": (
        # Identity by being a colour replaced with itself, which is a real
        # no-op rather than a sentinel the function has to test for.
        {"old": (0, 0, 0, 255), "new": (0, 0, 0, 255), "tolerance": 0.0},
        replace_colour,
    ),
    "outline": (
        {
            "colour": (0, 0, 0, 255),
            "size": 0.0,
            "place": "outside",
            "corners": 8,
            "wrap": 0.0,
        },
        outline,
    ),
    "despeckle": ({"speck": 0.0}, despeckle),
    "alpha threshold": ({"threshold": 0.0}, alpha_threshold),
    "defringe": ({"fringe": 0.0}, defringe),
    "grow / shrink matte": ({"grow": 0.0}, matte_grow),
    "remove orphans": ({"orphans": 0.0}, remove_orphans),
}

# The three parameter kinds the panel draws with something other than a slider.
# Held here beside the registry for :data:`RANGES`' reason -- a filter is one
# entry in one file, and a pane that had to know which of "colour" and "size"
# was a colour would be a second place to keep in step.
COLOUR_PARAMS: frozenset[str] = frozenset({"old", "new", "colour"})
TOGGLE_PARAMS: frozenset[str] = frozenset(
    {"red", "green", "blue", "wrap", "orphans"}
)
CHOICE_PARAMS: dict[str, tuple[Any, ...]] = {
    "place": ("outside", "inside"),
    # 4-connected is a diamond step and 8-connected is a square one; there is
    # nothing in between, so this is a pair of buttons and not a slider that
    # would let a user ask for 6.
    "corners": (4, 8),
}

# What the *popup* seeds a filter with the first time it is opened, where that
# differs from the defaults. Exactly one entry, and the reason it exists is that
# the two answer different questions: the defaults table answers "what does this
# filter do when nothing has been chosen", which must be nothing at all so a
# live preview is safe to run on the frame the popup opens; this answers "what
# did the user mean by choosing this filter", and nobody has ever opened Invert
# to invert no channels.
POPUP_VALUES: dict[str, dict[str, Any]] = {
    "invert": {"red": 1.0, "green": 1.0, "blue": 1.0},
    # Same argument as invert's, three more times: nobody opens "alpha
    # threshold" to snap nothing, "defringe" to clean a nought-pixel rim, or
    # "remove orphans" to remove none.
    "alpha threshold": {"threshold": 128.0},
    "defringe": {"fringe": 2.0},
    "remove orphans": {"orphans": 1.0},
}


def popup_values(name: str) -> dict[str, Any]:
    """The values a freshly opened popup starts a filter on."""
    return {**FILTERS[name][0], **POPUP_VALUES.get(name, {})}

# What each parameter's slider spans. Held beside the registry rather than in
# the pane so a filter is one entry in one file; a parameter with no range here
# is drawn as a plain number rather than dropped. A parameter in one of the
# three tables above is drawn by that table instead and wants no entry here.
RANGES: dict[str, tuple[float, float]] = {
    "brightness": (-1.0, 1.0),
    "contrast": (-1.0, 1.0),
    "hue": (-1.0, 1.0),
    "saturation": (-1.0, 1.0),
    "lightness": (-1.0, 1.0),
    "black": (0.0, 255.0),
    "white": (0.0, 255.0),
    "gamma": (0.1, 4.0),
    # Shared by blur and sharpen: one name, one span, because a
    # radius that meant 0..32 in one popup and 0..8 in the next is a slider a
    # user has to relearn per filter.
    "radius": (0.0, 32.0),
    "amount": (0.0, 3.0),
    # A radius in RGB space, whose long diagonal is 441 -- so 255 is not the
    # maximum possible distance, it is the largest one that is still a colour
    # match rather than a select-all.
    "tolerance": (0.0, 255.0),
    "size": (0.0, 32.0),
    # Deliberately *not* the shared ``radius`` span: the top of that one is a
    # 65x65 sort per pixel on the frame thread. See :func:`despeckle`.
    "speck": (0.0, (DESPECKLE_MAX - 1) / 2),
    "threshold": (0.0, 255.0),
    # Deliberately *not* the shared ``radius`` span, for ``speck``'s reason:
    # each step is a full-image pass on the frame thread under a live preview.
    "fringe": (0.0, 8.0),
    # Signed: negative erodes the silhouette, positive dilates it.
    "grow": (-8.0, 8.0),
}


def apply_named(name: str, pixels: np.ndarray, **params: Any) -> np.ndarray:
    """Run a registered filter, filling in any parameter not supplied.

    Unknown *keys* are dropped rather than raising: the panel remembers a
    filter's last values per filter, and a renamed parameter would otherwise
    turn a stale settings entry into a TypeError out of a slider drag.
    """
    defaults, func = FILTERS[name]
    return func(pixels, **{key: params.get(key, value) for key, value in defaults.items()})
