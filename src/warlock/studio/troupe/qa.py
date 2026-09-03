"""Animation-quality scores for a rendered character sheet. Pure.

The frame table says which cells are one movement in one direction; this
module reads the atlas along those runs and says, per cell, how far it is from
its neighbours -- in silhouette, in position, in colour -- and across
directions, how far a direction has drifted from the rest of the sheet. The
scores **rank and never gate**: nothing in ``service/``, the queue or a worker
reads them, and the panel that draws them offers a place to look, not a
verdict. They are computed in the app when a sheet is selected, off the frame
thread, and never written anywhere.

Inputs are the atlas as ``(H, W, 4)`` uint8 and the layout dict
``troupe_mode.preview_layout`` already returns -- ``movements`` with ``key``,
``frames``, ``loop`` and ``directions``, and ``runs`` with ``movement``,
``direction``, ``start`` and ``end`` -- so a pre-v2 sheet scores through the
same door as a configured one.

**Limb continuity is not measured**, and the omission is deliberate: telling a
swapped limb from a swinging one needs segmentation or the rig, and the atlas
carries neither. A limb that vanishes is caught by the silhouette delta and the
palette flicker, which is the gross case that matters at 32 px.

Thresholds are in :data:`THRESHOLDS` with the reasoning in
``docs/measurements/2026-09-02-troupe-qa-thresholds.md``; numpy is imported
inside the functions, as every module of this package does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FLAGS",
    "LEVEL_BAD",
    "LEVEL_OK",
    "LEVEL_WARN",
    "METRICS",
    "THRESHOLDS",
    "CellScore",
    "SheetScore",
    "cell_box",
    "level",
    "score_sheet",
]

METRICS: tuple[str, ...] = (
    "shape_delta",
    "seam_delta",
    "centroid_jitter",
    "foot_jitter",
    "palette_flicker",
    "drift_w",
    "drift_h",
    "drift_occupancy",
)

FLAGS: tuple[str, ...] = ("blank", "shape", "seam", "jitter", "foot", "flicker", "drift")

LEVEL_OK, LEVEL_WARN, LEVEL_BAD = 0, 1, 2

#: ``metric -> (warn, bad)``. A metric at or above ``warn`` flags the cell; at
#: or above ``bad`` it is the cell to look at first. Silhouette and seam are a
#: ``1 - IoU`` so 0.35 is "a third of the shape moved"; the jitters are
#: fractions of the cell; flicker is a share of the sprite's area that changed
#: colour; drift is relative to the median over directions and deliberately
#: lenient, because a side view is legitimately narrower than a front one.
THRESHOLDS: dict[str, tuple[float, float]] = {
    "shape_delta": (0.35, 0.60),
    "seam_delta": (0.35, 0.60),
    "centroid_jitter": (0.06, 0.15),
    "foot_jitter": (0.04, 0.10),
    "palette_flicker": (0.35, 0.60),
    "drift_w": (0.35, 0.70),
    "drift_h": (0.35, 0.70),
    "drift_occupancy": (0.35, 0.70),
}

#: Which flag a metric raises. The three drift metrics share one.
_FLAG_OF: dict[str, str] = {
    "shape_delta": "shape",
    "seam_delta": "seam",
    "centroid_jitter": "jitter",
    "foot_jitter": "foot",
    "palette_flicker": "flicker",
    "drift_w": "drift",
    "drift_h": "drift",
    "drift_occupancy": "drift",
}


@dataclass(frozen=True)
class CellScore:
    cell: int
    animation: str
    direction: str
    frame: int
    metrics: dict[str, float]
    flags: tuple[str, ...]


@dataclass(frozen=True)
class SheetScore:
    cells: tuple[CellScore, ...]
    #: ``(metric, value, cell)`` of the cell furthest past its bad threshold,
    #: or None when no cell is flagged.
    worst: tuple[str, float, int] | None
    flagged: int

    def lookup(self) -> dict[tuple[str, str, int], CellScore]:
        return {(c.animation, c.direction, c.frame): c for c in self.cells}


def cell_box(
    index: int, columns: int, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` of cell ``index`` on a ``columns``-wide atlas."""
    column, row = int(index) % int(columns), int(index) // int(columns)
    return (
        column * int(frame_w),
        row * int(frame_h),
        (column + 1) * int(frame_w),
        (row + 1) * int(frame_h),
    )


def level(score: CellScore) -> int:
    """``LEVEL_BAD`` at or past any bad threshold or blank, ``LEVEL_WARN`` at
    or past any warn threshold, else ``LEVEL_OK``."""
    if "blank" in score.flags:
        return LEVEL_BAD
    worst = LEVEL_OK
    for name, value in score.metrics.items():
        bounds = THRESHOLDS.get(name)
        if bounds is None:
            continue
        if value >= bounds[1]:
            return LEVEL_BAD
        if value >= bounds[0]:
            worst = LEVEL_WARN
    return worst


# --- measurement --------------------------------------------------------------


@dataclass(frozen=True)
class _Shape:
    """What one cell contributes to every metric, measured once."""

    area: int
    mask: Any
    centroid: tuple[float, float]
    foot: float
    width: int
    height: int
    colours: dict[int, int]


def _measure(crop: Any) -> _Shape:
    import numpy as np

    alpha = crop[..., 3] > 0
    area = int(alpha.sum())
    if area == 0:
        return _Shape(0, alpha, (0.0, 0.0), 0.0, 0, 0, {})
    rows = np.flatnonzero(alpha.any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0))
    ys, xs = np.nonzero(alpha)
    height, width = crop.shape[:2]
    centroid = (float(xs.mean()) / width, float(ys.mean()) / height)
    foot = float(rows[-1] + 1) / height
    rgb = crop[..., :3][alpha].astype(np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    keys, counts = np.unique(packed, return_counts=True)
    colours = {int(k): int(c) for k, c in zip(keys, counts, strict=True)}
    return _Shape(
        area,
        alpha,
        centroid,
        foot,
        int(cols[-1] - cols[0] + 1),
        int(rows[-1] - rows[0] + 1),
        colours,
    )


def _shape_delta(a: _Shape, b: _Shape) -> float:
    if a.area == 0 and b.area == 0:
        return 0.0
    if a.area == 0 or b.area == 0:
        return 1.0
    inter = int((a.mask & b.mask).sum())
    union = int((a.mask | b.mask).sum())
    return 1.0 - (inter / union if union else 1.0)


def _flicker(a: _Shape, b: _Shape) -> float:
    if a.area == 0 or b.area == 0:
        return 0.0 if a.area == b.area else 1.0
    keys = set(a.colours) | set(b.colours)
    total = sum(abs(a.colours.get(k, 0) - b.colours.get(k, 0)) for k in keys)
    return 0.5 * total / max(a.area, b.area, 1)


def _relative(value: float, median: float) -> float:
    if median <= 0:
        return 0.0 if value <= 0 else 1.0
    return abs(value - median) / median


def score_sheet(
    atlas: Any,
    layout: Mapping[str, Any],
    *,
    columns: int,
    frame_w: int,
    frame_h: int,
) -> SheetScore:
    """Every cell the runs name, scored. Runs off the atlas are skipped."""
    import numpy as np

    atlas = np.asarray(atlas)
    if atlas.ndim != 3 or atlas.shape[2] != 4:
        raise ValueError("score_sheet takes an (H, W, 4) atlas")
    height, width = atlas.shape[:2]
    loops = {
        str(m.get("key") or ""): bool(m.get("loop", True))
        for m in layout.get("movements") or ()
    }

    # Measure once per cell, keyed by cell index; a run is a span of them.
    measured: dict[int, _Shape] = {}
    runs: list[tuple[str, str, int, int]] = []
    for run in layout.get("runs") or ():
        try:
            animation = str(run.get("movement") or "")
            direction = str(run.get("direction") or "")
            start, end = int(run.get("start")), int(run.get("end"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not animation or not direction or end < start or start < 0:
            continue
        x0, y0, x1, y1 = cell_box(end, columns, frame_w, frame_h)
        if x1 > width or y1 > height:
            continue
        runs.append((animation, direction, start, end))
        for index in range(start, end + 1):
            if index not in measured:
                bx0, by0, bx1, by1 = cell_box(index, columns, frame_w, frame_h)
                measured[index] = _measure(atlas[by0:by1, bx0:bx1])

    scores: dict[int, tuple[str, str, int, dict[str, float], set[str]]] = {}
    for animation, direction, start, end in runs:
        cells = list(range(start, end + 1))
        for offset, index in enumerate(cells):
            metrics: dict[str, float] = {}
            flags: set[str] = set()
            here = measured[index]
            if here.area == 0:
                flags.add("blank")
            if offset > 0:
                prev = measured[cells[offset - 1]]
                metrics["shape_delta"] = _shape_delta(prev, here)
                metrics["centroid_jitter"] = float(
                    np.hypot(
                        here.centroid[0] - prev.centroid[0],
                        here.centroid[1] - prev.centroid[1],
                    )
                )
                metrics["foot_jitter"] = abs(here.foot - prev.foot)
                metrics["palette_flicker"] = _flicker(prev, here)
            elif len(cells) > 1 and loops.get(animation, True):
                last = measured[cells[-1]]
                metrics["seam_delta"] = _shape_delta(last, here)
            scores[index] = (animation, direction, offset, metrics, flags)

    # Drift: per (animation, offset), each direction against the median.
    by_frame: dict[tuple[str, int], list[int]] = {}
    for index, (animation, _direction, offset, _m, _f) in scores.items():
        by_frame.setdefault((animation, offset), []).append(index)
    for indices in by_frame.values():
        if len(indices) < 2:
            for index in indices:
                scores[index][3].update(drift_w=0.0, drift_h=0.0, drift_occupancy=0.0)
            continue
        widths = [measured[i].width for i in indices]
        heights = [measured[i].height for i in indices]
        areas = [measured[i].area for i in indices]
        mw, mh, ma = (float(np.median(v)) for v in (widths, heights, areas))
        for index, w, h, a in zip(indices, widths, heights, areas, strict=True):
            scores[index][3].update(
                drift_w=_relative(w, mw),
                drift_h=_relative(h, mh),
                drift_occupancy=_relative(a, ma),
            )

    cells_out: list[CellScore] = []
    worst: tuple[str, float, int] | None = None
    worst_ratio = 0.0
    flagged = 0
    for index in sorted(scores):
        animation, direction, offset, metrics, flags = scores[index]
        for name, value in metrics.items():
            warn, bad = THRESHOLDS[name]
            if value >= warn:
                flags.add(_FLAG_OF[name])
            ratio = value / bad if bad > 0 else 0.0
            if value >= warn and ratio > worst_ratio:
                worst_ratio = ratio
                worst = (name, float(value), index)
        if "blank" in flags and worst is None:
            worst = ("blank", 1.0, index)
        ordered = tuple(flag for flag in FLAGS if flag in flags)
        if ordered:
            flagged += 1
        cells_out.append(
            CellScore(index, animation, direction, offset, dict(metrics), ordered)
        )
    return SheetScore(tuple(cells_out), worst, flagged)
