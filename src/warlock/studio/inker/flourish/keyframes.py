"""Keyframes: a few restyled anchor frames become a whole phase, procedurally.

The third and least reliable of the AI doors, and the one the plan said to
measure before offering by default. The idea: render a phase, hand a handful
of its frames -- the *anchors* -- to the image model as img2img, get back the
same frames in a look the primitives cannot make (a painted flame, a woodcut
burst), and fill the frames between by **crossfading under the recipe's own
motion**. A plain crossfade between two pictures produces a ghost; sliding
each anchor along the displacement field the effect already computes for its
heat shimmer gives the blend a direction, and at 128px that is the difference
between a fade and a morph. No optical flow, no learned interpolation: the
field is the recipe's, so the result stays deterministic and offline once the
anchors exist.

Pure numpy. ``interpolate`` takes straight uint8 RGBA anchors keyed by frame
index and returns one plane per frame of the span, with the anchors returned
verbatim on their own frames -- what the model gave is never resampled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

#: How far, in pixels, the field may carry a pixel across one whole gap.
MAX_SHIFT_PX = 6.0


def anchor_frames(first: int, last: int, count: int) -> list[int]:
    """``count`` frame indices spread evenly over ``first..last``, ends included."""
    count = max(2, min(int(count), last - first + 1)) if last > first else 1
    if count == 1:
        return [first]
    return sorted({first + round(i * (last - first) / (count - 1)) for i in range(count)})


def _straight_to_premul(plane: np.ndarray) -> np.ndarray:
    f = plane.astype(np.float32) / 255.0
    out = f.copy()
    out[..., :3] *= f[..., 3:4]
    return out


def _premul_to_straight(plane: np.ndarray) -> np.ndarray:
    alpha = plane[..., 3]
    safe = np.maximum(alpha, 1e-6)[..., None]
    rgb = np.where(alpha[..., None] > 1e-6, plane[..., :3] / safe, 0.0)
    out = np.empty(plane.shape[:2] + (4,), dtype=np.uint8)
    out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    out[..., :3] = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    out[..., :3] *= (out[..., 3:4] > 0).astype(np.uint8)
    return out


def _shift(plane: np.ndarray, dx: np.ndarray, dy: np.ndarray, amount: float) -> np.ndarray:
    """Gather ``plane`` along ``(dx, dy) * amount`` pixels, nearest sampling."""
    if amount == 0.0:
        return plane
    h, w = plane.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    sx = np.clip(np.rint(xs - dx * amount), 0, w - 1).astype(np.intp)
    sy = np.clip(np.rint(ys - dy * amount), 0, h - 1).astype(np.intp)
    return plane[sy, sx]


def interpolate(
    anchors: dict[int, np.ndarray],
    first: int,
    last: int,
    field: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[np.ndarray]:
    """One straight-alpha uint8 plane per frame in ``first..last``.

    ``anchors`` maps frame index -> plane (all the same shape). Frames before
    the first anchor hold it; frames after the last hold that; between two
    anchors the blend is ``smoothstep(t)`` of the two, each carried along
    ``field(frame)`` -- unit displacement planes ``(dx, dy)`` in -1..1 -- by up
    to ``MAX_SHIFT_PX``: the earlier anchor drifts forward with ``t`` and the
    later one back with ``1 - t``, so the two meet in the middle.
    """
    if not anchors:
        raise ValueError("interpolation needs at least one anchor")
    keys = sorted(anchors)
    shapes = {anchors[k].shape for k in keys}
    if len(shapes) != 1:
        raise ValueError("every anchor is the same size")
    out: list[np.ndarray] = []
    for frame in range(first, last + 1):
        if frame in anchors:
            out.append(anchors[frame].copy())
            continue
        before = max((k for k in keys if k < frame), default=None)
        after = min((k for k in keys if k > frame), default=None)
        if before is None:
            out.append(anchors[after].copy())
            continue
        if after is None:
            out.append(anchors[before].copy())
            continue
        t = (frame - before) / (after - before)
        t = t * t * (3.0 - 2.0 * t)
        a = _straight_to_premul(anchors[before])
        b = _straight_to_premul(anchors[after])
        if field is not None:
            dx, dy = field(frame)
            a = _shift(a, dx, dy, MAX_SHIFT_PX * t)
            b = _shift(b, dx, dy, -MAX_SHIFT_PX * (1.0 - t))
        blend = a * np.float32(1.0 - t) + b * np.float32(t)
        out.append(_premul_to_straight(blend))
    return out


def field_from_recipe(
    recipe, *, direction: float = 0.0, scale: float = 10.0, speed: float = 3.0, seed_salt: int = 0
):
    """A displacement field per frame from the recipe's own noise -- the heat
    shimmer's field, whether or not the recipe has a distortion layer -- so
    the morph moves the way the effect moves. ``(dx, dy)`` at logical size."""
    from . import noise
    from . import render as R

    def field(frame: int) -> tuple[np.ndarray, np.ndarray]:
        ctx = R.frame_ctx(recipe, min(max(frame, 0), recipe.frame_count - 1), direction)
        x, y = ctx.coarse()
        drift = speed * ctx.time
        seed = ctx.seed + 7919 * 61 + seed_salt
        dx = (noise.fbm(x / scale, y / scale - drift, seed + 3, octaves=2) - 0.5) * 2.0
        dy = (noise.fbm(x / scale + 51.0, y / scale - drift, seed + 5, octaves=2) - 0.5) * 2.0
        return dx.astype(np.float32), dy.astype(np.float32)

    return field


def which_anchors(anchors: Sequence[int], frame: int) -> tuple[int | None, int | None]:
    """The anchor at or before ``frame`` and the one after, for a caller
    that wants to say which two a frame came from."""
    before = max((k for k in anchors if k <= frame), default=None)
    after = min((k for k in anchors if k > frame), default=None)
    return before, after
