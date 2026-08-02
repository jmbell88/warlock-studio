"""Sprite-sheet layout, atlas packing and the engine-neutral sidecar.

Deliberately pure: no Blender, no three.js, no filesystem beyond reading the
rendered frames and writing the two outputs. Everything about *where a cell
lands and what it depicts* is decided here, so the browser preview, the Blender
renderer and the sidecar can never disagree about the grid -- and so the grid
is testable without a GPU.

The layout is poses down, view directions across: one row per pose, one column
per yaw. The sidecar records the grid twice over -- as ``columns``/``rows`` for
an importer that just wants to slice a regular atlas, and as a flat ``cells``
list that names what each cell actually contains. The flat list is the
extension point: an animated clip becomes more cells with a ``frame`` above
zero, which costs importers nothing and needs no new format version.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHEET_VERSION = 1

# Eight compass directions, the near-universal choice for 2D-from-3D: enough to
# read a turn, cheap enough to render, and it divides cleanly into the 4- and
# 8-direction conventions every 2D engine already has.
DEFAULT_YAWS = 8

DEFAULT_FRAME_SIZE = 128
FRAME_SIZES = (64, 128, 256, 512)
DEFAULT_ELEVATION = 30.0
LIGHTING = ("flat", "lit")

# Above this, engines start refusing the texture outright rather than scaling
# it. Checked at plan time so the failure is a rejected request, not a job that
# renders for two minutes and then cannot be loaded.
MAX_ATLAS_PX = 8192

REST_POSE_NAME = "rest"


@dataclass(frozen=True, slots=True)
class Cell:
    """One sprite in the atlas, and what it depicts."""

    index: int
    row: int
    column: int
    x: int
    y: int
    pose: str | None      # pose id, or None for the unposed rest row
    pose_name: str
    yaw: float            # degrees clockwise from front (-Y in Blender axes)
    frame: int            # always 0 today; the seam animated clips arrive on

    def as_dict(self, size: int) -> dict[str, Any]:
        return {
            "index": self.index,
            "row": self.row,
            "column": self.column,
            "x": self.x,
            "y": self.y,
            "w": size,
            "h": size,
            "pose": self.pose,
            "pose_name": self.pose_name,
            "yaw": self.yaw,
            "frame": self.frame,
        }


@dataclass(frozen=True, slots=True)
class Plan:
    frame_size: int
    columns: int
    rows: int
    yaws: tuple[float, ...]
    elevation: float
    lighting: str
    poses: tuple[dict[str, Any], ...]
    cells: tuple[Cell, ...]

    @property
    def width(self) -> int:
        return self.columns * self.frame_size

    @property
    def height(self) -> int:
        return self.rows * self.frame_size


def yaw_angles(count: int = DEFAULT_YAWS) -> tuple[float, ...]:
    """Evenly spaced yaws starting at the front view."""
    return tuple(round(i * 360.0 / count, 4) for i in range(count))


def plan(
    poses: Sequence[Mapping[str, Any]],
    *,
    frame_size: int = DEFAULT_FRAME_SIZE,
    elevation: float = DEFAULT_ELEVATION,
    lighting: str = "flat",
    yaws: int = DEFAULT_YAWS,
) -> Plan:
    """Work out the grid, or raise ValueError.

    ``poses`` is the saved-pose records to render, in row order. An empty
    sequence is not an error: it means the unposed mesh, which is the whole
    point for a prop that has no rig.
    """
    if frame_size not in FRAME_SIZES:
        raise ValueError(f"frame_size must be one of {list(FRAME_SIZES)}")
    if lighting not in LIGHTING:
        raise ValueError(f"lighting must be one of {list(LIGHTING)}")
    if not -89.0 <= elevation <= 89.0:
        raise ValueError("elevation must be between -89 and 89 degrees")
    if yaws < 1:
        raise ValueError("a sheet needs at least one view direction")

    rows_in = list(poses) or [{"id": None, "name": REST_POSE_NAME}]
    width, height = yaws * frame_size, len(rows_in) * frame_size
    if max(width, height) > MAX_ATLAS_PX:
        raise ValueError(
            f"that sheet would be {width}x{height}px; the limit is {MAX_ATLAS_PX}px "
            "-- use a smaller frame size or fewer poses"
        )

    angles = yaw_angles(yaws)
    cells: list[Cell] = []
    for row, pose in enumerate(rows_in):
        for column, yaw in enumerate(angles):
            cells.append(
                Cell(
                    index=len(cells),
                    row=row,
                    column=column,
                    x=column * frame_size,
                    y=row * frame_size,
                    pose=pose.get("id"),
                    pose_name=str(pose.get("name") or REST_POSE_NAME),
                    yaw=yaw,
                    frame=0,
                )
            )
    return Plan(
        frame_size=frame_size,
        columns=yaws,
        rows=len(rows_in),
        yaws=angles,
        elevation=float(elevation),
        lighting=lighting,
        poses=tuple(dict(p) for p in rows_in),
        cells=tuple(cells),
    )


def pack(sheet: Plan, frames: Mapping[int, Path], out_png: Path) -> Path:
    """Composite the rendered frames into one RGBA atlas.

    A missing or wrong-sized frame raises rather than silently leaving a hole:
    a sheet with an invisible gap in it looks like a modelling problem, and the
    user would go looking in the wrong place.
    """
    from PIL import Image

    size = sheet.frame_size
    atlas = Image.new("RGBA", (sheet.width, sheet.height), (0, 0, 0, 0))
    try:
        for cell in sheet.cells:
            path = frames.get(cell.index)
            if path is None or not path.exists():
                raise ValueError(f"no rendered frame for cell {cell.index}")
            with Image.open(path) as frame:
                frame = frame.convert("RGBA")
                if frame.size != (size, size):
                    frame = frame.resize((size, size), Image.LANCZOS)
                atlas.paste(frame, (cell.x, cell.y))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(out_png, "PNG")
    finally:
        atlas.close()
    return out_png


def sidecar(
    sheet: Plan,
    *,
    sheet_id: str,
    source_job: str,
    image: str,
    created: float,
    name: str = "",
) -> dict[str, Any]:
    """The engine-neutral description of the atlas next to it.

    Engine-neutral means no Godot ``AtlasTexture``, no Unity ``SpriteMetaData``
    -- just pixel rectangles and what each one shows, in the plainest JSON that
    can carry it. Anything more opinionated would have to be rewritten for the
    second engine anyone tries.
    """
    return {
        "version": SHEET_VERSION,
        "id": sheet_id,
        "name": name or sheet_id,
        "source_job": source_job,
        "created": created,
        "image": image,
        "frame_size": sheet.frame_size,
        "columns": sheet.columns,
        "rows": sheet.rows,
        "width": sheet.width,
        "height": sheet.height,
        "elevation": sheet.elevation,
        "lighting": sheet.lighting,
        # Degrees clockwise from the front view, which is -Y in the Blender
        # axes the templates and the rig are expressed in.
        "yaws": list(sheet.yaws),
        "poses": [
            {"id": p.get("id"), "name": str(p.get("name") or REST_POSE_NAME)}
            for p in sheet.poses
        ],
        "cells": [c.as_dict(sheet.frame_size) for c in sheet.cells],
    }
