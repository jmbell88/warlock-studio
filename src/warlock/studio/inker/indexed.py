"""Indexed colour: a document palette, and every write snapped onto it.

**This is constrain-on-write, not index storage, and that is a decision rather
than a shortcut.** ``Layer.__post_init__`` hard-enforces ``(H, W, 4) uint8``,
and every blend mode, brush coverage accumulation, filter, selection mask and
native kernel in this package is written against RGBA. A document that stored
an index plane would be a rewrite of the whole package, and it would buy
nothing a user can see: what "indexed" is *for* is no stray near-colours, a
palette-wide recolour that actually reaches every pixel, and an export whose
colour table is exactly the one you authored. All three fall out of carrying a
palette and snapping writes onto it. So a later reader does not "fix" this:
the pixels stay RGBA on purpose.

Nearest is measured in **straight** (non-premultiplied) RGB, for the reason
``transform._resample`` has a ``straight=`` path at all -- premultiplying moves
the colour of every partially transparent pixel towards black, so a soft edge
would snap to a darker swatch than the one it was painted with.

Alpha is never snapped and never quantised. A palette is a set of colours, not
a set of opacities: a soft nib is still legal in indexed mode and simply bands,
which is what the mode is for. A **fully transparent** pixel is left alone
outright -- it has no colour to snap, and rewriting its dead RGB would make a
no-op write look like an edit to ``Document._commit_patch`` and push an undo
step for a gesture that changed nothing.

Pure numpy, no imgui and no service layer, like the rest of this package.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import index_plane as ixp

__all__ = [
    "SORT_KEYS",
    "grayscale",
    "histogram",
    "nearest",
    "ramp_between",
    "remap",
    "shade_ramp",
    "snap",
    "sort_order",
]

RGBA = tuple[int, int, int, int]

#: What a palette can be sorted by. Aseprite's full set, and the whole set is
#: here rather than the two obvious ones because the *point* of sorting a
#: palette is to find the ramp hiding in it -- hue groups the families,
#: saturation separates the greys from the colours, luma orders a ramp, the
#: three channels answer "which of these is the reddest", alpha finds the
#: swatches carrying transparency, and usage puts the slots nothing is painted
#: in at one end where they can be deleted.
SORT_KEYS = ("hue", "saturation", "luma", "red", "green", "blue", "alpha", "usage")

# Rec. 709. The same coefficients ``dither.luma`` uses, and deliberately: "sort
# by brightness" and "the built palette's order" must mean one thing.
_LUMA = (0.2126, 0.7152, 0.0722)

# The same three numbers as an array, for :func:`grayscale`'s one matrix
# multiply. Derived from the tuple above rather than written out again, so
# "sorted by brightness" and "what grey this colour becomes" cannot drift.
_LUMA_F32 = np.asarray(_LUMA, dtype=np.float32)


def _table(palette: Sequence[RGBA]) -> np.ndarray:
    """The palette as an (P, 3) int16 array of straight RGB."""
    if not palette:
        raise ValueError("an indexed document needs at least one colour")
    return np.asarray([tuple(c)[:3] for c in palette], dtype=np.int16)


def nearest(colour: RGBA, palette: Sequence[RGBA]) -> int:
    """The index of the palette entry closest to *colour*.

    Squared Euclidean distance in RGB. Deliberately not a perceptual metric:
    the palettes this serves are hand-authored pixel-art ramps of a dozen or so
    colours, where the nearest entry is unambiguous under any metric, and a
    weighted one would make "which swatch did my brush land on" depend on a
    formula the user cannot see.
    """
    table = _table(palette)
    delta = table.astype(np.int32) - np.asarray(tuple(colour)[:3], dtype=np.int32)
    return int(np.argmin((delta * delta).sum(axis=1)))


def snap(pixels: np.ndarray, palette: Sequence[RGBA]) -> np.ndarray:
    """*pixels* with every visible colour replaced by its nearest swatch.

    Returns a new array; the input is not touched. Alpha rides through
    unchanged and fully transparent pixels are returned verbatim -- see the
    module docstring for both.

    The distances are computed over the region's *distinct* colours rather than
    over its pixels. A palette edit remaps a whole 40-frame document, which at
    2048 square is 160 million pixels against maybe 32 swatches; done per pixel
    that is five billion subtractions, and done per distinct colour it is a few
    thousand. ``np.unique`` on a packed uint32 view is what makes the
    difference, and pixel art is the case where it wins by the most.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("snap takes (H, W, 4) uint8")
    table = _table(palette)
    out = pixels.copy()
    if out.size == 0:
        return out
    visible = out[..., 3] > 0
    if not visible.any():
        return out

    rgb = out[..., :3][visible]
    # Packed into one integer per pixel so ``unique`` sorts on a scalar: a
    # structured or axis-wise unique over three columns is an order of
    # magnitude slower and answers the same question.
    packed = (
        rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
    )
    keys, inverse = np.unique(packed, return_inverse=True)
    colours = np.stack(
        [(keys >> 16) & 0xFF, (keys >> 8) & 0xFF, keys & 0xFF], axis=1
    ).astype(np.int16)
    picks = ixp.nearest_entries(colours, table)
    out[..., :3][visible] = table[picks][inverse].astype(np.uint8)
    return out


def grayscale(pixels: np.ndarray) -> np.ndarray:
    """*pixels* with every visible pixel's ``r``, ``g`` and ``b`` made equal.

    **Grayscale is a constraint over RGBA storage, not a storage change**, and
    that is a decision rather than a shortcut. Aseprite stores ``(value,
    alpha)``; we keep ``(v, v, v, a)``. Three arguments, in order of weight:

    *There is no identity problem.* The whole reason an indexed document needs
    an index plane is that two palette slots can be the same colour and an RGBA
    plane cannot tell them apart. ``(v, a)`` and ``(v, v, v, a)`` are
    informationally equivalent and ``v`` is exactly recoverable from either, so
    a second representation buys nothing.

    *A two-channel plane forks every consumer.* The compositor, the caches, the
    texture uploader, the native kernels, five export formats -- the same list
    the indexed design was shaped to avoid forking, and here with nothing on the
    other side of the trade.

    *All nineteen blend modes preserve grayness.* The channelwise ones trivially
    (they compute each channel from equal inputs by one formula); the HSL family
    because a grey has zero saturation, so hue and saturation transfers from a
    grey source leave a grey and luminosity transfer is grey by definition. So
    even the **composite** of a grayscale document is grey, which is what makes
    the constraint honest rather than merely enforced at the door.

    Alpha rides through untouched and fully transparent pixels are returned
    verbatim, for ``indexed.snap``'s reasons: a transparent pixel has no colour
    to convert, and rewriting its dead RGB would make a no-op write look like an
    edit to the funnel and push an undo step for a gesture that did nothing.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("grayscale takes (H, W, 4) uint8")
    out = pixels.copy()
    if out.size == 0:
        return out
    visible = out[..., 3] > 0
    if not visible.any():
        return out
    rgb = out[..., :3][visible].astype(np.float32)
    # ``+ 0.5`` then floor, rather than numpy's round-half-to-even: the values
    # are 0..255 and the tie-break has to be the one every other integer
    # conversion in this package uses, or a flat 50% grey lands on a different
    # byte here than in ``composite.to_uint8``.
    value = np.floor(rgb @ _LUMA_F32 + 0.5).clip(0, 255).astype(np.uint8)
    out[..., :3][visible] = value[:, None]
    return out


def remap(pixels: np.ndarray, old: RGBA, new: RGBA) -> np.ndarray:
    """*pixels* with every occurrence of *old* rewritten to *new*.

    Exact match on RGB, and alpha is again untouched: this is what editing a
    palette slot does to the pixels already painted in it, and it has to be
    exact or it would drag neighbouring swatches along with the one being
    edited. Transparent pixels are skipped for ``snap``'s reason.
    """
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("remap takes (H, W, 4) uint8")
    out = pixels.copy()
    if out.size == 0:
        return out
    want = np.asarray(tuple(old)[:3], dtype=np.uint8)
    hit = (out[..., :3] == want).all(axis=2) & (out[..., 3] > 0)
    if hit.any():
        out[..., :3][hit] = np.asarray(tuple(new)[:3], dtype=np.uint8)
    return out


def _hue_saturation(colour: Sequence[int]) -> tuple[float, float]:
    """HSV hue and saturation of one colour, both 0..1.

    Written out rather than taken from ``colorsys`` because that module works in
    floats the caller has to scale on both sides, and because a grey has to
    answer hue **0** rather than whatever falls out of a division by zero -- the
    greys then group together at one end of a hue sort instead of scattering
    through the colours, which is the only useful answer.
    """
    r, g, b = (int(c) / 255.0 for c in tuple(colour)[:3])
    high, low = max(r, g, b), min(r, g, b)
    spread = high - low
    if spread <= 0.0:
        return 0.0, 0.0
    if high == r:
        hue = ((g - b) / spread) % 6.0
    elif high == g:
        hue = (b - r) / spread + 2.0
    else:
        hue = (r - g) / spread + 4.0
    return hue / 6.0, spread / high


def _metric(colour: Sequence[int], key: str, count: int) -> float:
    if key == "luma":
        return sum(w * int(c) for w, c in zip(_LUMA, tuple(colour)[:3], strict=True))
    if key == "red":
        return float(colour[0])
    if key == "green":
        return float(colour[1])
    if key == "blue":
        return float(colour[2])
    if key == "alpha":
        return float(tuple(colour)[3]) if len(tuple(colour)) > 3 else 255.0
    if key == "usage":
        return float(count)
    hue, saturation = _hue_saturation(colour)
    return hue if key == "hue" else saturation


def sort_order(
    palette: Sequence[RGBA],
    key: str,
    *,
    counts: Sequence[int] | None = None,
    descending: bool = False,
) -> list[int]:
    """The permutation that puts *palette* in *key* order.

    A permutation rather than a sorted table, because the caller may be sorting
    a *subset* of the slots in place and needs to know where each one went.

    Stable, and the tie-break is written down rather than left to the sort: two
    identical colours keep their relative order ascending **and** descending,
    which is why this negates the metric instead of passing ``reverse=True``.
    A reversed sort flips ties too, so sorting descending and then ascending
    would not give the original table back -- and the user would watch two
    swatches swap places for no reason they can see.
    """
    if key not in SORT_KEYS:
        raise ValueError(f"unknown palette sort key {key!r}")
    if counts is None:
        counts = [0] * len(palette)
    if len(counts) != len(palette):
        raise ValueError("a usage count per slot, or none at all")
    metrics = [_metric(c, key, counts[i]) for i, c in enumerate(palette)]
    sign = -1.0 if descending else 1.0
    return sorted(range(len(palette)), key=lambda i: (sign * metrics[i], i))


def ramp_between(start: RGBA, end: RGBA, steps: int) -> list[RGBA]:
    """*steps* colours strictly between two swatches, straight RGB (and alpha).

    Straight rather than premultiplied, and linear rather than perceptual, for
    the reason this whole module works in straight RGB: the ramp has to land
    where the user expects the midpoint of two swatches to be, and a perceptual
    interpolation puts it somewhere they cannot predict from the two ends.

    The endpoints are **not** included -- they are already in the table, and
    returning them would have the caller inserting duplicates of the two slots
    the ramp was drawn between.
    """
    if steps < 1:
        return []
    a = tuple(int(c) for c in tuple(start)[:4])
    b = tuple(int(c) for c in tuple(end)[:4])
    out: list[RGBA] = []
    for step in range(1, steps + 1):
        t = step / (steps + 1)
        out.append(
            tuple(int(np.floor(x + (y - x) * t + 0.5)) for x, y in zip(a, b, strict=True))  # type: ignore[arg-type]
        )
    return out


def shade_ramp(
    palette: Sequence[RGBA] | None, slots: Sequence[int] | None = None
) -> list[RGBA]:
    """The ramp a slot selection describes, in **palette order**.

    What the shading ink shifts along; see :meth:`.brush.StrokeState._shade`.
    Two rules, and both are about the order rather than about the colours.

    *Palette order, not click order.* The slots come back sorted by position in
    the table, so a ramp is the run of swatches the user can see laid out left
    to right -- picking the dark end first and the light end second describes
    the same ramp as the other way round, and the direction toggle is what
    reverses it. A click-ordered ramp would make the same five swatches mean
    five different things depending on how they were picked.

    *Adjacency is the selection's, not the table's.* Slots 2, 5 and 9 are three
    **adjacent** steps of a three-colour ramp; the swatches between them are not
    on it and a pixel painted in one is left alone. That is what makes a ramp
    pickable out of a table holding several.

    A selection of fewer than two slots falls back to the whole palette, which
    is the useful answer for a table that *is* one ramp -- and the only answer
    that makes the tool work before the user has selected anything.
    """
    table = [tuple(c) for c in (palette or ())]
    if not table:
        return []
    wanted = sorted({int(i) for i in (slots or ()) if 0 <= int(i) < len(table)})
    if len(wanted) < 2:
        return list(table)
    return [table[i] for i in wanted]


def histogram(pixels: np.ndarray, palette: Sequence[RGBA]) -> list[int]:
    """How many visible pixels sit exactly on each palette entry.

    Exact rather than nearest, because the question it answers is "is this slot
    used" -- and in a snapped document every visible pixel is exactly on some
    entry, so a nearest-based count would report a slot as used by pixels that
    belong to its neighbour. A slot with a zero here is one the user can delete
    without losing a pixel.
    """
    counts = [0] * len(palette)
    if pixels.size == 0:
        return counts
    visible = pixels[..., 3] > 0
    if not visible.any():
        return counts
    rgb = pixels[..., :3][visible]
    # One pass over the canvas, not one per entry. The loop this replaces ran a
    # full-canvas ``(rgb == want).all(axis=1).sum()`` per palette slot, so a
    # 2048-square document with a full 256-entry palette made 256 passes over
    # four million pixels: 11.4 s, against 117 ms here. The packed key is the
    # same idiom ``snap`` above uses, and for the same reason -- and because
    # the count is per *distinct colour*, the cost stops depending on how many
    # entries the palette has at all.
    packed = (
        rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
    )
    keys, hits = np.unique(packed, return_counts=True)
    want = np.asarray(
        [(c[0] << 16) | (c[1] << 8) | c[2] for c in (tuple(e)[:3] for e in palette)],
        dtype=np.uint32,
    )
    # ``searchsorted`` finds where each entry *would* sit; the equality test is
    # what turns "would sit here" into "is here", and an unused slot still
    # reports zero -- which is the answer this function exists to give.
    at = np.clip(np.searchsorted(keys, want), 0, len(keys) - 1)
    return np.where(keys[at] == want, hits[at], 0).astype("i8").tolist()


#: The colour harmonies the wheel offers, as the hue offsets each one is.
#:
#: Degrees rather than named recipes, because that is all a harmony *is* -- and
#: a table of numbers is something a test can check against the wheel it is
#: drawn on, where a function per harmony would be five places for the same
#: arithmetic to be slightly wrong in.
HARMONIES: dict[str, tuple[float, ...]] = {
    "complement": (0.0, 180.0),
    "triad": (0.0, 120.0, 240.0),
    "tetrad": (0.0, 90.0, 180.0, 270.0),
    "analogous": (0.0, 30.0, -30.0),
    "split": (0.0, 150.0, 210.0),
    "square": (0.0, 90.0, 180.0, 270.0),
}


def harmony(colour: tuple[int, int, int, int], kind: str) -> list[tuple[int, int, int, int]]:
    """The colours *kind* makes of this one, the first being it.

    Hue rotation only: saturation and lightness are what the user chose, and a
    harmony that changed them would be a palette generator rather than the
    answer to "what goes with this".
    """
    import colorsys

    offsets = HARMONIES.get(kind)
    if not offsets:
        return [colour]
    r, g, b, a = (int(channel) for channel in colour)
    hue0, light, sat = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    out: list[tuple[int, int, int, int]] = []
    for offset in offsets:
        hue = (hue0 + offset / 360.0) % 1.0
        rr, gg, bb = colorsys.hls_to_rgb(hue, light, sat)
        out.append(
            (
                max(0, min(255, round(rr * 255))),
                max(0, min(255, round(gg * 255))),
                max(0, min(255, round(bb * 255))),
                a,
            )
        )
    return out


def shades(colour: tuple[int, int, int, int], steps: int = 5) -> list[tuple[int, int, int, int]]:
    """A ramp from black through *colour* to white, ``steps`` entries.

    Aseprite's Shades widget, and the reason it is one function rather than a
    stored object: it is derived from the colour in hand, so there is nothing
    to persist, nothing to undo and nothing to get out of step with the swatch
    it came from.
    """
    steps = max(2, int(steps))
    r, g, b, a = (int(channel) for channel in colour)
    out = []
    for index in range(steps):
        t = index / (steps - 1)
        if t < 0.5:
            factor = t * 2.0
            mixed = (round(r * factor), round(g * factor), round(b * factor))
        else:
            factor = (t - 0.5) * 2.0
            mixed = (
                round(r + (255 - r) * factor),
                round(g + (255 - g) * factor),
                round(b + (255 - b) * factor),
            )
        out.append((*mixed, a))
    return out
