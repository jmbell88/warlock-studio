"""Whole-plane geometry: flip, rotate, scale, crop, canvas resize, descale.

Pure array functions, one plane at a time. The document applies each of them to
every layer and to the selection mask, which is the only way the three can
never disagree about what size the canvas is.

Resampling is Pillow's, on straight alpha, and that is deliberate: a bilinear
filter on straight alpha bleeds the colour of fully transparent pixels into the
edge. Every function here that resamples premultiplies first and unpremultiplies
after -- the one place in the codebase that touches premultiplied alpha, and it
is confined to these few lines.

**The descale trio at the foot of this file is a fourth grid measurement in the
tree, and the four must not be merged.** They ask different questions at
different doors:

* :mod:`warlock.pipelines.pixel` -- the luma *lattice* period and phase of an
  upscaled render, PIL, on the export path. This is a numpy port of exactly that
  measure, sharing its constants and its provisional threshold.
* :func:`detect_pixel_grid` here -- the same lattice measure, inside the editor,
  on the open document, offered as a button rather than applied on export.
* :mod:`warlock.studio.tilegrid.slicing` -- dark separator *bands* between the
  cells of a ruled tilesheet, at the import doors.
* :mod:`warlock.studio.tilegrid.roles` -- which blob role each cell of an
  already-gridded sheet plays.

A future reader who sees "grid detection" in two of them should read this list
before unifying anything.
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


def translate(
    pixels: np.ndarray, dx: int, dy: int, *, wrap: bool = False, fill: int = 0
) -> np.ndarray:
    """Shift a plane by whole pixels. Nothing is resampled.

    ``wrap`` carries content round the far edge, which makes the result an
    exact *permutation* -- no pixel is invented and none is lost, so shifting
    back is the identity. That is what the timeline's range shift needs: an
    index plane shifted this way keeps two slots holding one colour as two
    slots, because the slots are only being moved.

    Without it the vacated cells take ``fill``, which is zero -- transparent
    black -- for every RGBA caller and the *transparent index* for an index
    plane. Those are only the same value by coincidence, and a document whose
    transparent index is 7 would otherwise vacate into a band of solid slot-0
    colour. See :func:`resize_canvas`, whose ``fill`` exists for this reason
    and whose one caller passes exactly the same thing.
    """
    dx, dy = int(dx), int(dy)
    if wrap:
        return np.ascontiguousarray(np.roll(pixels, (dy, dx), axis=(0, 1)))
    out = np.full_like(pixels, int(fill))
    height, width = pixels.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx0 < sx1 and sy0 < sy1:
        out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = pixels[sy0:sy1, sx0:sx1]
    return out


def _premultiplied(pixels: np.ndarray) -> np.ndarray:
    out = pixels.astype(np.float32)
    out[..., :3] *= out[..., 3:4] / 255.0
    return out


def _unpremultiplied(pixels: np.ndarray) -> np.ndarray:
    alpha = pixels[..., 3:4] / 255.0
    # ``composite.over``'s masked-lane fix: ``where=`` does not promise the
    # masked lanes go unevaluated, so a SIMD lane with zero alpha under a
    # filtered colour still ran x/0 and raised under
    # ``np.errstate(all="raise")``. Divide by one there and select --
    # bit-identical everywhere the old form defined a value.
    shown = alpha > 0.0
    rgb = np.empty_like(pixels[..., :3])
    np.divide(pixels[..., :3], np.where(shown, alpha, 1.0), out=rgb)
    rgb = np.where(shown, rgb, 0.0)
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


#: How far a shear may be pushed **per axis**, in degrees. A bound on each
#: number the panel can send, and nothing more -- 60 degrees is already further
#: than anybody italicises a sprite, and past it the tangent grows fast enough
#: that the output plane is mostly empty.
#:
#: It is deliberately *not* what keeps the transform invertible. See
#: :data:`SHEAR_MIN_DET`, which is the guard that actually does, because the
#: degenerate case is a property of the **pair** and sits well inside this
#: bound.
SHEAR_MAX = 60.0

#: The smallest area factor a shear may have. A shear matrix is
#: ``[[1, tan sx], [tan sy, 1]]``, so its determinant -- which is exactly the
#: factor the plane's area is multiplied by -- is ``1 - tan(sx)tan(sy)``. Two
#: *equal-signed* slants therefore fight each other, and the pair (45, 45) is
#: singular: the plane collapses onto a line and there is no inverse to sample
#: through. That pair is nowhere near :data:`SHEAR_MAX`, so the per-axis clamp
#: never came close to preventing it.
#:
#: A tenth rather than an epsilon, because "not quite singular" is not a state
#: worth rendering: at (44, 44) the determinant is 0.067, so a sprite comes back
#: as a sliver of a fifteenth its area, and every pixel of it is a nearest
#: sample of a plane that has been squeezed flat. The pair is refused outright
#: instead -- the plane comes back unslanted -- which is the same answer this
#: function already gave for a pair of zeros, and it is reached only from the
#: numeric Slant fields, deliberately, and never from a handle.
SHEAR_MIN_DET = 0.1


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

    **A degenerate pair comes back unslanted**, which is a refusal and not an
    approximation: see :data:`SHEAR_MIN_DET`. Each axis is clamped to
    :data:`SHEAR_MAX` first, but that clamp is a bound on the numbers and does
    not reach the pair -- (45, 45) is singular and sits well inside it.
    """
    from PIL import Image

    sx = max(-SHEAR_MAX, min(float(degrees[0]), SHEAR_MAX))
    sy = max(-SHEAR_MAX, min(float(degrees[1]), SHEAR_MAX))
    kx, ky = math.tan(math.radians(sx)), math.tan(math.radians(sy))
    det = 1.0 - kx * ky
    height, width = pixels.shape[:2]
    if abs(det) < SHEAR_MIN_DET or (abs(kx) < 1e-9 and abs(ky) < 1e-9):
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


#: The anchor grid as it is drawn: three rows of three, top row first.
#:
#: Here rather than in the pane because the *arrangement* is a fact about the
#: anchors, and :func:`anchor_cell` below has to agree with it -- two spellings
#: of "which cell is where" is one edit away from a grid that highlights one
#: cell and resizes towards another.
ANCHOR_GRID: tuple[tuple[str, str, str], ...] = (
    ("top-left", "top", "top-right"),
    ("left", "centre", "right"),
    ("bottom-left", "bottom", "bottom-right"),
)


def anchor_cell(anchor: str, cell: str) -> tuple[int, int] | None:
    """Which way ``cell`` points when the image is anchored at ``anchor``.

    ``(0, 0)`` is the anchor's own cell -- the one that holds the picture. A
    unit ``(dx, dy)`` is an arrow pointing that way: the direction the new room
    opens in. ``None`` is a cell with nothing in it, which is what a corner
    anchor leaves five of, because an arrow there would promise room that
    anchor never makes.

    Photoshop's grid, and the reason it is worth copying: the numbers say *how
    much* room is being added and nothing at all about *which side* it lands
    on, which is the only thing a person actually gets wrong here.

    An unknown name behaves as ``top-left``, which is :func:`anchor_offset`'s
    own rule -- two different answers to a typo would be exactly the
    highlight-one-cell-resize-towards-another bug.
    """
    known = {name for row in ANCHOR_GRID for name in row}
    if anchor not in known:
        anchor = "top-left"
    if cell not in known:
        return None
    here = next(
        (r, c) for r, row in enumerate(ANCHOR_GRID) for c, name in enumerate(row)
        if name == anchor
    )
    there = next(
        (r, c) for r, row in enumerate(ANCHOR_GRID) for c, name in enumerate(row)
        if name == cell
    )
    dy, dx = there[0] - here[0], there[1] - here[1]
    if (dx, dy) == (0, 0):
        return (0, 0)
    # Adjacent only. A cell two steps away would be an arrow pointing at room
    # this anchor does not make on that side at all.
    if abs(dx) <= 1 and abs(dy) <= 1:
        return (dx, dy)
    return None


def percent_size(
    old: tuple[int, int], percent: tuple[float, float]
) -> tuple[int, int]:
    """``old`` scaled by a percentage per axis. -> whole pixels, floored at 1.

    **Deliberately does not clamp the upper end.** The growth ceiling is a
    policy about what a user may type into a form and it lives in
    ``inker_mode.clamp_resize``, one layer up, which is the single place that
    knows it. Duplicating it here would make two ceilings to keep in step, and
    the one that drifted would be the one nothing routed through.
    """
    return (
        max(1, int(round(old[0] * float(percent[0]) / 100.0))),
        max(1, int(round(old[1] * float(percent[1]) / 100.0))),
    )


def size_percent(
    old: tuple[int, int], size: tuple[int, int]
) -> tuple[float, float]:
    """``size`` as a percentage of ``old``, per axis. The inverse of above."""
    return (
        100.0 * float(size[0]) / max(1, int(old[0])),
        100.0 * float(size[1]) / max(1, int(old[1])),
    )


def linked_size(
    old: tuple[int, int], size: tuple[int, int], axis: str
) -> tuple[int, int]:
    """``size`` with the *untyped* axis pulled back onto ``old``'s ratio.

    ``axis`` is ``"w"`` or ``"h"`` -- **which field the user just typed in**,
    and it is not optional. A single "something changed" flag makes whichever
    field is read second win, so typing a width would silently rewrite it from
    the height that has not moved.

    The ratio is taken from ``old`` and never from the previous pending pair.
    Deriving it from the last value lets a chain of roundings walk a 3:2
    document off its own ratio after a few keystrokes, which is a proportion
    lock that does not lock.
    """
    ow, oh = max(1, int(old[0])), max(1, int(old[1]))
    if axis == "w":
        return (max(1, int(size[0])), max(1, int(round(int(size[0]) * oh / ow))))
    return (max(1, int(round(int(size[1]) * ow / oh))), max(1, int(size[1])))


def preview_boxes(
    old: tuple[int, int], new: tuple[int, int], anchor: str, box: float
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """``(new_rect, old_rect)`` for the canvas-size preview, in a ``box`` square.

    Both at one scale, fitted to whichever is larger per axis and centred, so a
    growth shows the picture inside the frame and a crop shows the frame inside
    the picture. Rects are ``(x0, y0, x1, y1)`` relative to the box's corner.

    This is the half of GIMP's drag-preview worth having: the numbers already
    say how much room is being added, and this is the only thing on the dialog
    that says which side it lands on.
    """
    ow, oh = max(1, int(old[0])), max(1, int(old[1]))
    nw, nh = max(1, int(new[0])), max(1, int(new[1]))
    span_w, span_h = max(ow, nw), max(oh, nh)
    scale = min(box / span_w, box / span_h)
    off_x, off_y = anchor_offset((ow, oh), (nw, nh), anchor)
    # The union's own origin: with a crop the old image starts left of and
    # above the new canvas, and both rects have to move by the same amount or
    # the picture and the frame stop being drawn in one space.
    base_x = min(0, off_x)
    base_y = min(0, off_y)
    pad_x = (box - span_w * scale) * 0.5
    pad_y = (box - span_h * scale) * 0.5
    new_rect = (
        pad_x + (0 - base_x) * scale,
        pad_y + (0 - base_y) * scale,
        pad_x + (0 - base_x + nw) * scale,
        pad_y + (0 - base_y + nh) * scale,
    )
    old_rect = (
        pad_x + (off_x - base_x) * scale,
        pad_y + (off_y - base_y) * scale,
        pad_x + (off_x - base_x + ow) * scale,
        pad_y + (off_y - base_y + oh) * scale,
    )
    return new_rect, old_rect


def resize_canvas(
    pixels: np.ndarray, size: tuple[int, int], offset: tuple[int, int] = (0, 0), fill: int = 0
) -> np.ndarray:
    """A bigger or smaller canvas with the pixels placed, never rescaled.

    ``fill`` is what the new room is made of. Zero -- transparent black -- for
    every RGBA caller there has ever been, and therefore the default, so this
    stays byte-identical for them. An **index plane** is the one caller that
    needs another value: its empty room has to be the document's transparent
    index, which is only slot 0 by coincidence, and filling it with zero on a
    document whose transparent index is 7 would grow the canvas by a rectangle
    of solid slot-0 colour.
    """
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    shape = (height, width) if pixels.ndim == 2 else (height, width, pixels.shape[2])
    out = np.full(shape, int(fill), dtype=np.uint8)
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


# --------------------------------------------------------------------------
# The pixel lattice: measuring it, and reducing onto it
# --------------------------------------------------------------------------
#
# Copied verbatim from ``pipelines/pixel.py``, which owns the same three
# constants and the measurement they parametrize. A copy rather than an import
# because this package is headless and pinned against reaching into
# ``pipelines`` (which imports PIL at module scope); the numbers are identical
# so the two detectors give one answer on one image.

#: Cell sizes a generator plausibly draws on a 1024 canvas. Integers only: a
#: fractional period would need sub-pixel resampling to reduce, which is exactly
#: the blending a descale exists to avoid.
GRID_SCALES: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 16)

#: Normalized within-cell gradient ratio below which the lattice is real.
#: **Provisional** in the sibling too, and governed by the same document:
#: ``docs/measurements/2026-08-06-pixel-art-xl.md`` (procedure pre-registered).
#: Copying it to a second surface moves nothing -- one document governs both.
GRID_RESIDUAL_MAX = 0.05

#: Below this a "grid" is a handful of cells and the statistic is noise.
_MIN_CELLS = 4

#: Rec. 601 luma coefficients, deliberately **not** :func:`dither.luma`, which
#: is Rec. 709 and a per-colour helper. The sibling detector uses these, and two
#: lattice detectors that disagreed about brightness would disagree about the
#: phase of the same image.
_GRID_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _grid_luma(pixels: np.ndarray) -> np.ndarray:
    return np.asarray(pixels, dtype=np.float64)[:, :, :3] @ _GRID_LUMA


def _axis_phase(luma: np.ndarray, scale: int, axis: int) -> int:
    """Where the cell boundaries sit along one axis.

    A block lattice puts all of its change *at* the boundaries, so the mean
    absolute gradient summed by position-modulo-period peaks at the boundary
    offset. One argmax rather than a search over reconstructions.
    """
    grad = np.abs(np.diff(luma, axis=axis)).mean(axis=1 - axis)
    if not grad.size:
        return 0
    # ``grad[i]`` is the change between index i and i+1, i.e. a boundary
    # *before* index i+1; the phase is where a cell starts.
    positions = np.arange(grad.size) % scale
    totals = np.bincount(positions, weights=grad, minlength=scale)
    return int((int(np.argmax(totals)) + 1) % scale)


def grid_residual(pixels: np.ndarray, scale: int, phase: tuple[int, int]) -> float:
    """How much of this image's change happens *between* cells, not inside them.

    Mean absolute gradient at non-boundary positions over the mean absolute
    gradient everywhere -- the same ratio shape ``seam.py`` uses, and for the
    same reason: an absolute number of levels means nothing without the
    picture's own contrast to divide by. A block lattice scores 0.0; a smooth
    gradient scores about 1.0, because a gradient changes as much mid-cell as it
    does at a boundary. Comparing within-cell variance to total variance instead
    would call every smooth image a grid, since a small patch of a gradient is
    always flatter than the whole frame.

    **The worse of the two axes decides.** A lattice holds in both directions;
    an image with horizontal banding and nothing vertical is not pixel art.
    """
    luma = _grid_luma(pixels)
    py, px = phase
    if min(luma.shape) < scale * _MIN_CELLS:
        return 1.0

    def axis_ratio(plane: np.ndarray, offset: int) -> float:
        grad = np.abs(np.diff(plane, axis=1))
        if not grad.size:
            return 1.0
        positions = np.arange(grad.shape[1])
        boundary = ((positions + 1 - offset) % scale) == 0
        total = float(grad.mean())
        if total <= 0.0:
            # Nothing changes anywhere: a flat image is not evidence of a
            # lattice, however trivially constant each of its cells is.
            return 1.0
        interior = grad[:, ~boundary]
        return float(interior.mean()) / total if interior.size else 1.0

    return max(axis_ratio(luma, px), axis_ratio(luma.T, py))


def detect_pixel_grid(pixels: np.ndarray) -> dict:
    """The cell size and phase this drawing was made on, if any.

    Measured on the whole plane and on luminance: the lattice is a property of
    the generation, not of the subject, and a crop would move the phase before
    it was ever measured. ``scale`` is ``None`` when nothing beats
    :data:`GRID_RESIDUAL_MAX` -- an ordinary drawing, which is the common case
    and must see no change at all.
    """
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("detect_pixel_grid takes (H, W, 3|4)")
    luma = _grid_luma(array)
    measured: list[tuple[int, tuple[int, int], float]] = []
    for scale in GRID_SCALES:
        if min(luma.shape) < scale * _MIN_CELLS:
            continue
        phase = (_axis_phase(luma, scale, 0), _axis_phase(luma, scale, 1))
        measured.append((scale, phase, grid_residual(array, scale, phase)))
    if not measured:
        return {"scale": None, "phase": (0, 0), "residual": 1.0, "candidate": None}
    passing = [m for m in measured if m[2] < GRID_RESIDUAL_MAX]
    # The *largest* passing scale, not the best-scoring one. Every divisor of a
    # true period passes just as cleanly -- an image of 8px blocks is trivially
    # also an image of 4px blocks -- and taking the smallest would halve the
    # size of every authored pixel.
    scale, phase, residual = (
        max(passing, key=lambda m: m[0]) if passing else min(measured, key=lambda m: m[2])
    )
    return {
        "scale": scale if passing else None,
        "phase": phase,
        "residual": float(residual),
        "candidate": scale,
    }


def descale_size(
    size: tuple[int, int], scale: int, phase: tuple[int, int]
) -> tuple[int, int]:
    """The ``(width, height)`` :func:`descale` will produce, without doing it.

    The document needs it to move its slices before the planes are touched, and
    deriving it from the same ``arange`` the sampler uses is what keeps the two
    from disagreeing by one cell at the edge.
    """
    width, height = int(size[0]), int(size[1])
    scale = int(scale)
    if scale < 2:
        raise ValueError("a descale reduces by at least two")
    ys = np.arange(int(phase[0]) + scale // 2, height, scale)
    xs = np.arange(int(phase[1]) + scale // 2, width, scale)
    return int(xs.size), int(ys.size)


def descale(pixels: np.ndarray, scale: int, phase: tuple[int, int]) -> np.ndarray:
    """One output pixel per lattice cell, sampled at the cell's centre.

    The centre and not a corner: a corner sample sits on the boundary the
    generator drew, where a half-pixel of the neighbouring cell routinely
    bleeds. Whole-plane, because the phase is only meaningful against the plane
    the lattice was drawn on.

    Pure element *selection* -- ``arr[np.ix_(ys, xs)]``, no arithmetic anywhere
    -- so it can never mint a colour that was not already there, and it is
    exact on an index plane by construction rather than by a special case.
    """
    array = np.asarray(pixels)
    scale = int(scale)
    if scale < 2:
        raise ValueError("a descale reduces by at least two")
    py, px = int(phase[0]), int(phase[1])
    height, width = array.shape[:2]
    ys = np.arange(py + scale // 2, height, scale)
    xs = np.arange(px + scale // 2, width, scale)
    if not ys.size or not xs.size:
        raise ValueError("that grid leaves no cells in this image")
    return np.ascontiguousarray(array[np.ix_(ys, xs)])
