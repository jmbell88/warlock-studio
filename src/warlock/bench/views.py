"""Eight views of a finished mesh, rendered for measurement.

Reuses ``pipelines/sheet.py`` directly and deliberately does *not* go through
``service/sheets.py``. That route queues a job on the serial GPU queue, writes
into the source job's directory (160 rows of pollution in the user's
history), enforces a per-job sheet cap that has nothing to do with a
benchmark, and packs to an atlas the metrics would have to crop back out of.
``blender_worker.op_sheet`` already writes one square transparent PNG per cell
into a scratch directory, which is exactly the per-view output wanted here,
and it already handles an unrigged mesh -- every pose call in it is guarded on
``armature is not None``.

The camera constants below are the whole reason ``bench calibrate`` exists.
Nothing in this repo has established that TRELLIS aligns a reconstruction to
its input camera, and PROMPT_TEMPLATE asks for a 3/4 perspective view rather
than an axis-aligned front. The sweep has now been run and it says there is no
fixed matched view at all, so these remain a guess permanently -- see the note
on the constants below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..pipelines import sheet

# Which of the eight yaws is meant to be the camera-matched view, as an offset
# added before Blender sees a cell. Column 0 is by construction the matched
# view and the other seven are its turntable.
#
# UNCALIBRATED, and now measurably uncalibratable -- do not repeat the sweep
# expecting a number. `python -m warlock.bench calibrate --all` was run on
# 2026-08-05 over 37 finished jobs spanning every category in core-v1, and the
# per-job argmax scattered by 330 degrees (silhouette_iou) and 300 degrees
# (dino_cosine) against a STABLE_YAW_SPREAD of 30. The two metrics do not agree
# with each other either, and it scatters within every category as well as
# across them: there is no fixed matched view, so these stay a guess and column
# 0 is not the matched view of anything. Any fidelity metric built on top of
# this has to be max-over-8-views. See docs/measurements/
# 2026-08-04-view-calibration.md for the tables and the reasoning.
REFERENCE_YAW = 0.0
REFERENCE_ELEVATION = 20.0

VIEW_COUNT = 8
VIEW_SIZE = 512

# Pinned, never a recipe field. DINOv2 measures style as much as identity, so
# the absolute numbers are only meaningful A-against-B -- and only while the
# lighting is identical across the runs being compared. Making this
# configurable would let two runs differ in a way the score cannot see.
VIEW_LIGHTING = "lit"

SIDECAR = "views.json"


def view_plan(*, frame_size: int = VIEW_SIZE, elevation: float | None = None) -> sheet.Plan:
    """One row of ``VIEW_COUNT`` yaws around the unposed mesh."""
    return sheet.plan(
        [],
        frame_size=frame_size,
        elevation=REFERENCE_ELEVATION if elevation is None else elevation,
        lighting=VIEW_LIGHTING,
        yaws=VIEW_COUNT,
    )


def worker_cells(plan: sheet.Plan, yaw_offset: float = REFERENCE_YAW) -> list[dict[str, Any]]:
    """The plan's cells with the calibration offset already applied.

    Added here rather than in the renderer so column 0 *is* the matched view
    by construction, and the sidecar records the offset separately -- a
    pre-calibration run stays interpretable once the real number is known.
    """
    cells = []
    for cell in plan.cells:
        entry = cell.as_dict(plan.frame_size)
        entry["yaw"] = (cell.yaw + yaw_offset) % 360.0
        entry["bones"] = {}
        cells.append(entry)
    return cells


def render_views(
    source_glb: Path,
    out_dir: Path,
    *,
    yaw_offset: float = REFERENCE_YAW,
    elevation: float | None = None,
    frame_size: int = VIEW_SIZE,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Render the views and write ``views.json`` beside them.

    Calls ``rigging.run_worker`` synchronously on the calling thread. That is
    correct here and only here: the CLI thread is a may-block thread with no
    event loop and no frame deadline, and the run loop guarantees no trellis
    job is in flight -- one job at a time is the whole submission model.
    """
    from .. import rigging

    out_dir.mkdir(parents=True, exist_ok=True)
    # Cleared, not merged into. A resumed unit renders into a directory that
    # already holds the previous mesh's frames, and a render that fails part
    # way through would otherwise leave a set that is half one reconstruction
    # and half another -- scored, silently, as if it were one.
    for stale in out_dir.glob("*.png"):
        stale.unlink(missing_ok=True)
    plan = view_plan(frame_size=frame_size, elevation=elevation)
    cells = worker_cells(plan, yaw_offset)
    spec = rigging.sheet_spec(
        source_glb,
        out_dir,
        cells,
        frame_size=plan.frame_size,
        elevation=plan.elevation,
        lighting=plan.lighting,
    )
    result = rigging.run_worker(spec, timeout=timeout)

    payload = sheet.sidecar(
        plan,
        sheet_id="0" * 12,
        source_job=source_glb.parent.name,
        image="",
        created=0.0,
        name="bench-views",
    )
    payload["bench"] = {
        "yaw_offset": yaw_offset,
        "elevation": plan.elevation,
        "lighting": plan.lighting,
        "reference_column": 0,
        "calibrated": False,
        "frames": [f"{i:04d}.png" for i in range(len(plan.cells))],
    }
    payload["worker"] = {"frames": result.get("frames"), "rendered": result.get("rendered")}
    (out_dir / SIDECAR).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def view_paths(out_dir: Path) -> list[Path]:
    """The rendered frames in cell order, whichever names the worker used."""
    return sorted(p for p in out_dir.glob("*.png"))


def reference_view(out_dir: Path) -> Path | None:
    """The camera-matched view -- column 0 by construction."""
    paths = view_paths(out_dir)
    return paths[0] if paths else None
