"""Does a camera-matched view exist at all?

This is the gate under everything in B5/B7. Every automatic metric assumes
that TRELLIS aligns its reconstruction to the input camera and that
``normalize_glb`` preserves that alignment -- so that one of the eight
rendered views can be compared against the reference image directly. Nothing
in this repo has established either. PROMPT_TEMPLATE asks for a "3/4
perspective view" rather than an axis-aligned front, and trellis-server.exe is
a vendored binary whose behaviour is confirmed here by measurement, not by
upstream docs.

So: sweep yaw and elevation over a finished mesh, score every cell against
that job's own ``input.png``, and look at where the maximum lands. Run it over
several jobs spanning categories. If the argmax is stable, set
``views.REFERENCE_YAW`` / ``REFERENCE_ELEVATION`` to the mode and paste the
table into that module -- the pattern ``Config.trellis_band`` follows. If it
scatters, there is no fixed match, and the metric has to become
max-over-8-views: weaker, but honest.

One job is an anecdote, not a mode. The summary says so rather than implying
a result the sample size cannot support.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import metrics as metrics_mod
from . import views as views_mod

DEFAULT_YAWS = 36
DEFAULT_ELEVATIONS = (0.0, 10.0, 20.0, 30.0)
# Smaller than a benchmark view: 144 cells per job, and the argmax is a
# coarse question about orientation, not about detail.
SWEEP_SIZE = 256

# How far the per-job argmax may spread before "there is a fixed matched view"
# stops being a defensible reading of the data.
STABLE_YAW_SPREAD = 30.0


@dataclass(frozen=True, slots=True)
class Cell:
    yaw: float
    elevation: float
    scores: dict[str, float | None]

    def score(self, metric: str) -> float:
        value = self.scores.get(metric)
        return -1.0 if value is None else value


def sweep_cells(yaws: int = DEFAULT_YAWS) -> list[dict[str, Any]]:
    """The worker cells for one elevation of the sweep.

    Built directly rather than through ``sheet.plan``, which would refuse this:
    its MAX_ATLAS_PX guard is about a *packed atlas*, and 36 views at 256px is
    9216px wide. Nothing here packs an atlas -- the cells are rendered as
    individual PNGs and scored one at a time -- so the guard is measuring a
    constraint this sweep does not have.
    """
    from ..pipelines import sheet

    return [
        {"index": i, "row": 0, "column": i, "yaw": yaw, "frame": 0, "pose": None, "bones": {}}
        for i, yaw in enumerate(sheet.yaw_angles(yaws))
    ]


def sweep_job(
    job_dir: Path,
    out_dir: Path,
    *,
    yaws: int = DEFAULT_YAWS,
    elevations: tuple[float, ...] = DEFAULT_ELEVATIONS,
    frame_size: int = SWEEP_SIZE,
    config: Any = None,
    on_event: Any = None,
) -> dict[str, Any]:
    """Render and score the whole grid for one finished job."""
    from .. import rigging

    say = on_event or (lambda msg: None)
    model = job_dir / "model.glb"
    reference = job_dir / "input.png"
    if not model.exists():
        raise ValueError(f"{job_dir.name} has no model.glb")
    if not reference.exists():
        raise ValueError(f"{job_dir.name} has no input.png")

    out_dir.mkdir(parents=True, exist_ok=True)
    cells: list[Cell] = []
    # One worker call per elevation: the spec carries a single elevation for
    # the whole render, which is also what keeps the camera framed once.
    for elevation in elevations:
        say(f"  elevation {elevation:.1f}: rendering {yaws} views")
        # Named to the hundredth of a degree: truncating to whole degrees
        # folded 22.0 and 22.5 into one directory, so the second elevation
        # rendered over the first and any frame it failed to write was then
        # scored from the stale render. Underscore for the point so the name
        # never reads as a file extension.
        frames = out_dir / f"e{elevation:06.2f}".replace(".", "_")
        frames.mkdir(parents=True, exist_ok=True)
        angles = sweep_cells(yaws)
        spec = rigging.sheet_spec(
            model,
            frames,
            angles,
            frame_size=frame_size,
            elevation=elevation,
            lighting=views_mod.VIEW_LIGHTING,
        )
        rigging.run_worker(spec, timeout=1800.0)
        for entry in angles:
            render = frames / f"{entry['index']:04d}.png"
            if not render.exists():
                continue
            cells.append(
                Cell(
                    yaw=entry["yaw"],
                    elevation=elevation,
                    scores=metrics_mod.score_view(reference, render, config),
                )
            )
    return summarise(job_dir.name, cells, config)


def summarise(job_id: str, cells: list[Cell], config: Any = None) -> dict[str, Any]:
    out: dict[str, Any] = {"job": job_id, "cells": [
        {"yaw": c.yaw, "elevation": c.elevation, **c.scores} for c in cells
    ]}
    for metric in metrics_mod.available(config):
        scored = [c for c in cells if c.scores.get(metric) is not None]
        if not scored:
            continue
        best = max(scored, key=lambda c: c.score(metric))
        out[metric] = {
            "best_yaw": best.yaw,
            "best_elevation": best.elevation,
            "best_score": best.scores[metric],
            "worst_score": min(c.score(metric) for c in scored),
            "mean_score": statistics.fmean(c.score(metric) for c in scored),
        }
    return out


def _circular_spread(angles: list[float]) -> float:
    """The tightest arc containing every angle, in degrees.

    Circular on purpose: 350 and 10 are 20 degrees apart, and a plain stdev
    would call them a 240-degree disagreement.
    """
    if len(angles) < 2:
        return 0.0
    ordered = sorted(a % 360.0 for a in angles)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    gaps.append(360.0 - ordered[-1] + ordered[0])
    return 360.0 - max(gaps)


def verdict(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Is there a fixed matched view, on this evidence?

    Deliberately conservative about the sample: the plan calls for 5-10 jobs
    spanning categories, and fewer than three cannot show a mode at all -- one
    argmax is an anecdote. Saying so is the point of this function.
    """
    picks = [r[metric] for r in results if metric in r]
    if not picks:
        return {"metric": metric, "verdict": "no data", "n": 0}
    yaws = [p["best_yaw"] for p in picks]
    spread = _circular_spread(yaws)
    elevations = [p["best_elevation"] for p in picks]

    if len(picks) < 3:
        answer = "inconclusive: too few jobs to show a mode"
    elif spread <= STABLE_YAW_SPREAD:
        answer = "stable: set REFERENCE_YAW/REFERENCE_ELEVATION to the mode"
    else:
        answer = "scattered: there is no fixed matched view -- use max-over-8-views"
    return {
        "metric": metric,
        "verdict": answer,
        "n": len(picks),
        "yaws": yaws,
        "yaw_spread": round(spread, 1),
        "modal_yaw": statistics.mode(yaws),
        "modal_elevation": statistics.mode(elevations),
        "scores": [round(p["best_score"], 4) for p in picks],
    }


def write_report(out_dir: Path, results: list[dict[str, Any]], config: Any = None) -> Path:
    payload = {
        "results": results,
        "verdicts": [verdict(results, m) for m in metrics_mod.available(config)],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibrate.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def verdict_lines(results: list[dict[str, Any]], config: Any = None) -> list[str]:
    """The verdicts as sentences, for the CLI."""
    out = []
    for metric in metrics_mod.available(config):
        v = verdict(results, metric)
        if v.get("n"):
            out.append(
                f"{metric}: {v['verdict']} "
                f"(n={v['n']}, argmax yaws {v['yaws']}, spread {v['yaw_spread']} deg)"
            )
        else:
            out.append(f"{metric}: no data")
    return out


def format_table(result: dict[str, Any], metric: str, *, top: int = 8) -> str:
    """The best cells for one job, as a table to paste into views.py."""
    rows = [c for c in result["cells"] if c.get(metric) is not None]
    rows.sort(key=lambda c: c[metric], reverse=True)
    lines = [f"{result['job']}  best {metric} cells:", "    yaw   elev    score"]
    for row in rows[:top]:
        lines.append(f"  {row['yaw']:6.1f} {row['elevation']:6.1f}   {row[metric]:.4f}")
    if rows:
        worst = rows[-1]
        lines.append(f"  (worst: yaw {worst['yaw']:.1f} elev {worst['elevation']:.1f} "
                     f"= {worst[metric]:.4f})")
    return "\n".join(lines)
