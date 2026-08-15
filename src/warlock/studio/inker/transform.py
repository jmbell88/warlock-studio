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
#:
#: ``rotsprite`` is the third and it is about *rotation* alone -- see
#: :func:`rotsprite`. A scale or a shear asked for it gets nearest, which is
#: the honest answer: the algorithm has nothing to say about either, and
#: silently smoothing a pixel-art scale because the user picked the pixel-art
#: rotation would be exactly backwards.
RESAMPLES = ("smooth", "nearest", "rotsprite")


def _filter(resample: str, smooth: int) -> int:
    from PIL import Image

    # Anything that is not ``smooth`` copies a source pixel whole. Written this
    # way round rather than as ``== "nearest"``, because a third pixel-art mode
    # arriving and quietly getting Lanczos is the failure this spelling
    # prevents.
    return smooth if resample == "smooth" else Image.NEAREST


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
        straight=resample != "smooth",
    )


def rotate(
    pixels: np.ndarray, degrees: float, *, expand: bool = False, resample: str = "smooth"
) -> np.ndarray:
    from PIL import Image

    if resample == "rotsprite":
        if rotsprite_fits((pixels.shape[1], pixels.shape[0])):
            return rotsprite(pixels, degrees, expand=expand)
        # Silently, here; the pane says so out loud. See ``ROTSPRITE_MAX_PIXELS``.
        resample = "nearest"
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
        straight=resample != "smooth",
    )


# --- RotSprite --------------------------------------------------------------
#
# Nearest-neighbour is the right answer for scaling pixel art and the wrong one
# for *turning* it: a hard-edged diagonal rotated by copying whole pixels comes
# out as a staircase with a different tread on every step, and every antialiased
# alternative invents colours the palette does not have. RotSprite is the
# standard way out -- upscale with an edge-preserving filter, turn the big
# image with nearest neighbour, and sample the middle of each block back down.
# Nothing is interpolated at any stage, so the result is drawn entirely from
# colours the source already had, and the staircase is decided by an 8x lattice
# rather than a 1x one.

#: Doublings before the turn. Three is the published figure and the one every
#: implementation uses: two leaves visible chunking on shallow angles and four
#: quadruples the memory for a difference that does not show.
ROTSPRITE_ROUNDS = 3
ROTSPRITE_SCALE = 2**ROTSPRITE_ROUNDS

#: Source pixels above which the turn falls back to plain nearest neighbour.
#: The upscale is 64x in *area*, so this ceiling is already 16.7 M pixels and
#: 67 MB in the intermediate RGBA plane alone, with the rotation making more --
#: and it is a free-transform drag, so the whole thing runs again on every
#: mouse-move. Above it the honest answer is the cheaper filter and a word to
#: the user rather than a frozen editor. Pixel art is small by definition, so
#: this is not a limit anybody drawing a sprite will meet.
ROTSPRITE_MAX_PIXELS = 512 * 512


def rotsprite_fits(size: tuple[int, int]) -> bool:
    """Whether a plane this size may be turned with RotSprite."""
    return int(size[0]) * int(size[1]) <= ROTSPRITE_MAX_PIXELS


def _packed(plane: np.ndarray) -> np.ndarray:
    """One integer per pixel, for **exact** equality.

    EPX's whole decision is "are these two pixels the same colour", and a
    per-channel comparison folded down with ``all`` costs four passes and a
    reduction to answer what one integer comparison answers. Packing RGBA into
    a uint32 is exact -- no arithmetic, only shifts -- so two pixels compare
    equal here if and only if all four of their bytes match.
    """
    if plane.ndim == 2:
        return plane.astype(np.uint32)
    p = plane.astype(np.uint32)
    return (p[..., 0] << 24) | (p[..., 1] << 16) | (p[..., 2] << 8) | p[..., 3]


def _neighbours(plane: np.ndarray) -> tuple[np.ndarray, ...]:
    """``(up, right, left, down)``, with the border pixels repeating themselves.

    Edge replication rather than a transparent pad: EPX only ever *interpolates
    a corner between two equal neighbours*, and a pad would make the border's
    neighbour a colour that is not in the drawing, rounding off every shape
    that touches the edge of its own cel.
    """
    return (
        np.concatenate((plane[:1], plane[:-1]), axis=0),
        np.concatenate((plane[:, 1:], plane[:, -1:]), axis=1),
        np.concatenate((plane[:, :1], plane[:, :-1]), axis=1),
        np.concatenate((plane[1:], plane[-1:]), axis=0),
    )


def epx(plane: np.ndarray) -> np.ndarray:
    """One EPX / Scale2x round: every pixel becomes four, vectorised.

    Each output quadrant takes the neighbour it points at *only* when that
    neighbour agrees with the one beside it and disagrees with the two across
    from it -- which is the whole of the algorithm and the reason it rounds a
    staircase without touching a flat area::

        1 2      1 = A if C == A and C != D and A != B
        3 4      2 = B if A == B and A != C and B != D
                 3 = C if D == C and D != B and C != A
                 4 = D if B == D and B != A and D != C

    with ``A`` above, ``B`` right, ``C`` left and ``D`` below, and every
    quadrant otherwise the pixel itself. Works on an ``(H, W, 4)`` plane and on
    an ``(H, W)`` mask alike, because the comparison goes through
    :func:`_packed` and the copy goes through fancy indexing.
    """
    up, right, left, down = _neighbours(plane)
    a, b, c, d = (_packed(x) for x in (up, right, left, down))
    out = np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)
    # ``out[0::2, 0::2]`` is a strided *view*, so the masked assignment writes
    # straight through into ``out`` -- no scatter and no second buffer.
    m = (c == a) & (c != d) & (a != b)
    out[0::2, 0::2][m] = up[m]
    m = (a == b) & (a != c) & (b != d)
    out[0::2, 1::2][m] = right[m]
    m = (d == c) & (d != b) & (c != a)
    out[1::2, 0::2][m] = left[m]
    m = (b == d) & (b != a) & (d != c)
    out[1::2, 1::2][m] = down[m]
    return out


def rotsprite(pixels: np.ndarray, degrees: float, *, expand: bool = False) -> np.ndarray:
    """Turn a plane the pixel-art way: upscale, turn, sample back down.

    The downsample takes ``[4::8, 4::8]`` -- the *middle* of each 8x8 block
    rather than its corner. A corner sample sits on the boundary between two
    source pixels and rounds one way at one angle and the other way at the
    next, so a slow rotate drag shimmers; the centre is the only offset with no
    tie to break.

    Deterministic by construction: every step is an integer copy, so the same
    plane and the same angle give the same bytes every time. That matters more
    than usual here, because a free transform re-renders from the lifted pixels
    on every mouse-move and a wobbling result would look like a bug in the
    drag rather than in the filter.
    """
    big = pixels
    for _ in range(ROTSPRITE_ROUNDS):
        big = epx(big)
    turned = rotate(big, degrees, expand=True, resample="nearest")
    half = ROTSPRITE_SCALE // 2
    small = np.ascontiguousarray(turned[half::ROTSPRITE_SCALE, half::ROTSPRITE_SCALE])
    if expand:
        return small
    # ``expand=False`` means "the same frame as it went in", which for a turn
    # is the centre of the grown one -- the same answer Pillow's own
    # ``expand=False`` gives, reached with the module's own two helpers.
    size = (pixels.shape[1], pixels.shape[0])
    return resize_canvas(
        small,
        size,
        anchor_offset((small.shape[1], small.shape[0]), size, "centre"),
    )


#: How far a shear may be pushed, per axis, in degrees. The limit is not
#: cosmetic: a shear matrix is ``[[1, tan sx], [tan sy, 1]]`` and its
#: determinant is ``1 - tan(sx)tan(sy)``, so the pair (45, 45) is singular --
#: the plane collapses onto a line and there is no inverse to sample through.
#: 60 degrees is well clear of that on both axes (``tan 60`` squared is 3, so
#: the determinant is -2) and is already further than anybody italicises a
#: sprite.
SHEAR_MAX = 60.0


def shear(
    pixels: np.ndarray, degrees: tuple[float, float], *, resample: str = "smooth"
) -> np.ndarray:
    """Slant a plane: ``x' = x + tan(sx) y``, ``y' = tan(sy) x + y``.

    Degrees rather than the tangents themselves, because that is what a user
    means by "italic 15 degrees" and what the numeric field shows. The output
    grows to hold the slanted rectangle -- a shear of a canvas-sized plane
    would otherwise lose exactly the corners the shear created.

    Pillow's ``AFFINE`` takes the *inverse* map (it walks the destination and
    asks where each pixel came from), so the matrix below is inverted here
    rather than at the call site: writing the forward matrix and handing it
    over unchanged shears the picture the other way, which is a bug that looks
    like a sign error in the UI.

    It goes through ``_resample`` like every other filtered path here, so it
    premultiplies and unpremultiplies around the interpolation -- a bilinear
    mix with a fully transparent pixel drags that pixel's colour into the edge.
    """
    from PIL import Image

    sx = max(-SHEAR_MAX, min(float(degrees[0]), SHEAR_MAX))
    sy = max(-SHEAR_MAX, min(float(degrees[1]), SHEAR_MAX))
    kx, ky = math.tan(math.radians(sx)), math.tan(math.radians(sy))
    det = 1.0 - kx * ky
    height, width = pixels.shape[:2]
    if abs(det) < 1e-6 or (abs(kx) < 1e-9 and abs(ky) < 1e-9):
        return pixels.copy()

    xs = [x + kx * y for x, y in ((0, 0), (width, 0), (0, height), (width, height))]
    ys = [ky * x + y for x, y in ((0, 0), (width, 0), (0, height), (width, height))]
    min_x, min_y = min(xs), min(ys)
    new_w = max(1, int(round(max(xs) - min_x)))
    new_h = max(1, int(round(max(ys) - min_y)))

    a, b = 1.0 / det, -kx / det
    d, e = -ky / det, 1.0 / det
    coeffs = (a, b, a * min_x + b * min_y, d, e, d * min_x + e * min_y)
    how = _filter(resample, Image.BICUBIC)
    return _resample(
        pixels,
        lambda im: im.transform(
            (new_w, new_h),
            Image.AFFINE,
            coeffs,
            how,
            fillcolor=0 if im.mode == "L" else (0, 0, 0, 0),
        ),
        straight=resample != "smooth",
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
