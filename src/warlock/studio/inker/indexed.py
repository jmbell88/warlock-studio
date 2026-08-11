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

__all__ = ["snap", "remap", "histogram", "nearest"]

RGBA = tuple[int, int, int, int]


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
    delta = colours[:, None, :].astype(np.int32) - table[None, :, :].astype(np.int32)
    picks = np.argmin((delta * delta).sum(axis=2), axis=1)
    out[..., :3][visible] = table[picks][inverse].astype(np.uint8)
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
    for index, colour in enumerate(palette):
        want = np.asarray(tuple(colour)[:3], dtype=np.uint8)
        counts[index] = int((rgb == want).all(axis=1).sum())
    return counts
