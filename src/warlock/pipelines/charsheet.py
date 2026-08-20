"""Troupe's sheet planning: clip x direction -> frame table -> ``sheet.Plan``.

Pure arithmetic. No torch, no Blender, no ``service``, no filesystem -- the
same bargain ``sheet.py`` makes, and for the same reason: the browser preview,
the Blender renderer and the sidecar must never disagree about what cell 137
depicts, and the way to guarantee that is for one testable function to decide.

**The frame table is held twice.** ``studio.troupe.spec`` holds it as the
studio's answer and this module holds it as the pipeline's, because a
``pipelines`` module runs inside worker and Blender processes where ``studio``
is not importable at all, and ``studio/troupe`` imports nothing outward. That is
the ``spritesynth`` / ``inker.animation`` ``DIRECTION_ORDER`` arrangement at its
second instance, and it takes the same safeguard:
``tests/troupe/test_troupe_geometry_agreement.py`` is the **sole owner** of the
agreement between the two copies. A change to one is a change to both plus that
test, or a Troupe sheet and the editor that opens it come to mean different
things by ``walk_left``.

Cell order is grouped by ``(animation, direction)`` and dense -- the argument is
written out in ``studio.troupe.spec``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import sheet

__all__ = [
    "ANIMATIONS",
    "COLUMNS",
    "DIRECTIONS",
    "SIZES",
    "TroupeCell",
    "animation_block",
    "check_frame_counts",
    "frame_table",
    "plan",
    "spans",
]

#: ``(name, frames, loop, duration_ms)``, as a literal table rather than a
#: loop, so the frame table is readable in one glance.
ANIMATIONS: tuple[tuple[str, int, bool, int], ...] = (
    ("idle", 4, True, 150),
    ("walk", 8, True, 100),
    ("run", 8, True, 60),
    ("attack", 6, False, 80),
    ("jump", 6, False, 100),
)

#: ``(name, yaw)``, degrees clockwise from the front view. The four the rest of
#: the repo already names keep the yaws they already have.
DIRECTIONS: tuple[tuple[str, float], ...] = (
    ("front", 0.0),
    ("front_left", 45.0),
    ("left", 90.0),
    ("back_left", 135.0),
    ("back", 180.0),
    ("back_right", 225.0),
    ("right", 270.0),
    ("front_right", 315.0),
)

COLUMNS = 8
SIZES = (16, 24, 32, 48, 64, 96, 128)
RENDER_SIZE = 512


@dataclass(frozen=True, slots=True)
class TroupeCell:
    index: int
    animation: str
    direction: str
    yaw: float
    frame: int


def frames_per_direction() -> int:
    return sum(a[1] for a in ANIMATIONS)


def frame_table() -> tuple[TroupeCell, ...]:
    """Every cell of a character sheet, in pack-and-play order."""
    out: list[TroupeCell] = []
    for name, frames, _loop, _ms in ANIMATIONS:
        for direction, yaw in DIRECTIONS:
            for frame in range(frames):
                out.append(
                    TroupeCell(
                        index=len(out),
                        animation=name,
                        direction=direction,
                        yaw=yaw,
                        frame=frame,
                    )
                )
    return tuple(out)


def spans() -> tuple[tuple[str, str, int, int, bool], ...]:
    """``(animation, direction, start, end, loop)`` per contiguous run."""
    out = []
    index = 0
    for name, frames, loop, _ms in ANIMATIONS:
        for direction, _yaw in DIRECTIONS:
            out.append((name, direction, index, index + frames - 1, loop))
            index += frames
    return tuple(out)


def check_frame_counts(records: Mapping[str, Sequence[Any]]) -> None:
    """Refuse a set of expanded clips that does not fill the frame table.

    By name and with both numbers, rather than by padding or truncating: a
    seven-frame walk laid into an eight-frame table renders one cell of some
    other animation, and the user would go looking at the rig.
    """
    for name, frames, _loop, _ms in ANIMATIONS:
        got = len(records.get(name) or ())
        if got != frames:
            raise ValueError(
                f"the {name} clip expands to {got} frames and the table wants {frames}"
            )
    extra = sorted(set(records) - {a[0] for a in ANIMATIONS})
    if extra:
        raise ValueError(f"not Troupe animations: {extra}")


def plan(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    frame_size: int = 128,
    elevation: float = sheet.DEFAULT_ELEVATION,
    lighting: str = "flat",
) -> sheet.Plan:
    """The whole character sheet as one ``sheet.Plan``.

    ``records`` is ``animation name -> the expanded pose records`` that
    ``sheet.interpolate_clip`` produced for it; this module never expands a
    clip itself, because that needs the pose library and this file is meant to
    stay loadable anywhere.

    Built directly rather than through ``sheet.plan``: that one lays out poses
    down and yaws across, one row per pose, which for 256 cells would be a
    32-row-by-8-column grid whose rows are *poses* -- and Troupe's rows are a
    dense run of ``(animation, direction)`` groups instead. Both go through
    ``check_atlas_size``, which is the part that must not be approximated
    twice.
    """
    if lighting not in sheet.LIGHTING:
        raise ValueError(f"lighting must be one of {list(sheet.LIGHTING)}")
    if not -89.0 <= elevation <= 89.0:
        raise ValueError("elevation must be between -89 and 89 degrees")
    if frame_size not in SIZES and frame_size not in sheet.FRAME_SIZES:
        raise ValueError(f"frame_size must be one of {list(SIZES)}")
    check_frame_counts(records)

    table = frame_table()
    rows = (len(table) + COLUMNS - 1) // COLUMNS
    sheet.check_atlas_size(COLUMNS * frame_size, rows * frame_size)

    cells: list[sheet.Cell] = []
    poses: list[dict[str, Any]] = []
    for cell in table:
        record = records[cell.animation][cell.frame]
        column, row = cell.index % COLUMNS, cell.index // COLUMNS
        cells.append(
            sheet.Cell(
                index=cell.index,
                row=row,
                column=column,
                x=column * frame_size,
                y=row * frame_size,
                pose=record.get("id"),
                pose_name=cell.animation,
                yaw=cell.yaw,
                frame=cell.frame,
            )
        )
        poses.append(dict(record))
    return sheet.Plan(
        frame_size=frame_size,
        columns=COLUMNS,
        rows=rows,
        # Every direction the sheet contains, in table order -- ``Plan.yaws``
        # is the sheet's set of view directions, and here it is not the same
        # thing as "one per column" the way it is for a pose-per-row sheet.
        yaws=tuple(y for _n, y in DIRECTIONS),
        elevation=float(elevation),
        lighting=lighting,
        poses=tuple(poses),
        cells=tuple(cells),
    )


def animation_block() -> dict[str, Any]:
    """The sidecar's ``animation`` block: durations and per-direction tags.

    Gap 4 of the plan, closed. The block has been defined in ``sheet.sidecar``
    since the format was written and populated only by the Inker exporter, so a
    *rendered* sheet reached an engine as frame indices with no fps and no loop
    tags -- and the fps was the one thing the renderer knew and the importer
    could not guess.

    ``repeat: 1`` on the one-shots is the same spelling Inker's exporter uses
    for a play-once tag, so the two writers produce one format rather than two
    dialects of it.
    """
    table = frame_table()
    durations = {a[0]: a[3] for a in ANIMATIONS}
    return {
        "frames": [
            {"cell_index": c.index, "duration_ms": durations[c.animation]}
            for c in table
        ],
        "tags": [
            {
                "name": f"{animation}_{direction}",
                "start": start,
                "end": end,
                "loop": loop,
                "direction": "forward",
                **({} if loop else {"repeat": 1}),
            }
            for animation, direction, start, end, loop in spans()
        ],
    }


def pivot_in_cell(
    pivot: tuple[float, float] | None, frame_size: int
) -> tuple[float, float] | None:
    """A pivot the worker projected at ``RENDER_SIZE``, in *cell* pixels.

    The sidecar documents its pivot as pixels within a cell, and every reader
    of it assumes exactly that -- ``sheet.sidecar`` defaults the field to
    ``(cell_w / 2, cell_h)`` for the same reason. ``blender_worker.op_sheet``
    projects the ground origin in the pixels it rendered at, which on every
    *other* sheet path is the cell size and on this one deliberately is not:
    Troupe renders at :data:`RENDER_SIZE` and packs at the logical size,
    because a 256-cell atlas at 512 would be refused at the 8192 ceiling.

    So the number needs converting, and the conversion had been missing: a
    32px cell recorded a pivot near ``(256, 470)``, sixteen times outside
    itself, and an engine placing sprites from the sidecar put the character's
    feet far below the sprite. Placing without drift is the one property the
    field exists for.

    Here rather than inline in ``_q_troupe`` because this module is the
    filesystem-free half of the character sheet -- it decides what cell 137
    depicts and never reads a file -- which is what makes the arithmetic
    testable at all.
    """
    if pivot is None:
        return None
    scale = float(frame_size) / float(RENDER_SIZE)
    return (float(pivot[0]) * scale, float(pivot[1]) * scale)
