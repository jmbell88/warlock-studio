"""The arithmetic of mirror-assisted cleanup and of the propagation mark.

Pure numpy over ``(H, W, 4)`` uint8 planes; nothing here touches a document.
``_doc_sheet.SheetOps`` is what applies these, and the canvas overlay is what
draws :func:`diff_report`'s map before anything is applied.

**Why a face box at all.** Measured on the reference sheets (the Troupe
programme's phase 6): mirroring west onto east leaves 36-37 differing pixels, and every
one of them is in the face -- while any non-zero centring shift is far worse
(443 pixels at +-1). The asymmetry is real (a face is not symmetric) and it is
*local*, so the useful offer is "the mirror of your fix, everywhere but the
face". The box is the top ``fraction`` of the sprite's own alpha bbox at the
bbox's full width, which is a head on every humanoid the sheet renderer
draws; the fraction is the user's to move because a helmet or a mane is not.
"""

from __future__ import annotations

import numpy as np

from . import transform

__all__ = [
    "FACE_FRACTION",
    "changed_weight",
    "diff",
    "diff_report",
    "face_box",
    "face_weight",
    "mirrored",
    "translate_within",
]

#: Default share of the alpha bbox's height, from the top, excluded from a
#: mirror. Three tenths is a head-and-neck on the shipped humanoid at 32 px
#: (roughly 9 rows of 30); the strip's slider is the door for anything else.
FACE_FRACTION = 0.30


def _check(pixels: np.ndarray) -> None:
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("mirror takes (H, W, 4) uint8")


def face_box(
    pixels: np.ndarray, fraction: float = FACE_FRACTION
) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` of the top ``fraction`` of the alpha bbox, or None
    on an empty plane or a zero fraction. Half-open on the far edges."""
    _check(pixels)
    fraction = max(0.0, min(float(fraction), 1.0))
    if fraction <= 0.0:
        return None
    rows = np.flatnonzero(pixels[..., 3].any(axis=1))
    cols = np.flatnonzero(pixels[..., 3].any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    depth = max(1, int(round((y1 - y0) * fraction)))
    return (x0, y0, x1, min(y1, y0 + depth))


def face_weight(
    shape_hw: tuple[int, int], box: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    """A ``masked_apply`` weight: 255 everywhere but inside ``box``, where it
    is 0. None for no box, which ``masked_apply`` reads as the whole plane."""
    if box is None:
        return None
    height, width = int(shape_hw[0]), int(shape_hw[1])
    weight = np.full((height, width), 255, dtype=np.uint8)
    x0, y0, x1, y1 = box
    weight[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)] = 0
    return weight


def mirrored(pixels: np.ndarray) -> np.ndarray:
    """The plane flipped left-to-right. Exact; nothing is resampled."""
    _check(pixels)
    return transform.flip(pixels, "horizontal")


def diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Where two planes differ, as a bool ``(H, W)`` map.

    Two fully transparent pixels are equal whatever colour hides under their
    zero alpha: a straight buffer keeps invisible colour there and reporting it
    would count pixels nobody can see.
    """
    _check(a)
    _check(b)
    if a.shape != b.shape:
        raise ValueError("diff takes two planes of one shape")
    differs = (a != b).any(axis=-1)
    both_clear = (a[..., 3] == 0) & (b[..., 3] == 0)
    return differs & ~both_clear


def diff_report(
    src: np.ndarray, dst: np.ndarray, weight: np.ndarray | None
) -> tuple[int, int, np.ndarray]:
    """What applying ``mirrored(src)`` onto ``dst`` would change.

    Returns ``(outside, inside, map)``: the count of differing pixels the
    weight lets through, the count it holds back (the face), and the full
    difference map so the overlay can draw both in their own colours.
    """
    changed = diff(mirrored(src), dst)
    if weight is None:
        return int(changed.sum()), 0, changed
    if weight.shape != changed.shape:
        raise ValueError("the weight must match the plane")
    open_ = weight > 0
    return int((changed & open_).sum()), int((changed & ~open_).sum()), changed


def changed_weight(before: np.ndarray, now: np.ndarray) -> np.ndarray | None:
    """The propagation mark as a weight: 255 where the cel changed since
    ``before`` was taken, 0 elsewhere. None when nothing changed, so the
    caller can say so instead of writing a mask of nothing."""
    changed = diff(before, now)
    if not changed.any():
        return None
    return np.where(changed, 255, 0).astype(np.uint8)


def translate_within(
    pixels: np.ndarray, weight: np.ndarray, dx: int, dy: int
) -> np.ndarray:
    """Move the weighted pixels by ``(dx, dy)`` inside their own plane.

    The selection is lifted, its footprint cleared to transparent, and the
    lift dropped at the offset -- ``shift_selected``'s lift-and-commit as one
    pure function, which is what lets it run over frames that are not on
    screen. Content pushed past the edge is clipped, never wrapped: a wrapped
    arm reappearing on the far side of a sprite is not a translation anybody
    asked for. A partial weight fades the lifted pixels in over what was
    already at the destination, the same way every other weighted write here
    does.
    """
    _check(pixels)
    if weight.shape != pixels.shape[:2]:
        raise ValueError("the weight must match the plane")
    dx, dy = int(dx), int(dy)
    fade = weight.astype(np.float32)[..., None] / 255.0
    lifted = pixels.astype(np.float32) * fade
    remaining = pixels.astype(np.float32) * (1.0 - fade)
    moved = transform.translate(lifted, dx, dy)
    moved_fade = transform.translate(fade, dx, dy)
    out = remaining * (1.0 - moved_fade) + moved
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
