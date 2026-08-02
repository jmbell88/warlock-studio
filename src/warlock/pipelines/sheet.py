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


# --- animation ---------------------------------------------------------------
#
# An animated clip is not a new format: it is more cells whose ``frame`` is above
# zero, which is exactly what the flat cells list in the sidecar was built for.
# Interpolation happens here, on the host, for the same reason the grid does --
# it is pure arithmetic, it must be testable without a GPU, and the browser
# preview and the Blender renderer must agree about what frame 3 of a clip is.

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)   # XYZW, three.js order

# Below this the two rotations are parallel and the slerp denominator collapses;
# a straight lerp is then both correct and stable.
SLERP_LINEAR_THRESHOLD = 0.9995

MAX_CLIP_FRAMES = 32


def slerp(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    """Shortest-arc spherical interpolation between two XYZW quaternions."""
    import math

    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:
        # q and -q are the same rotation but interpolate the long way round.
        # Without this a walk cycle counter-rotates through its own midpoint.
        bx, by, bz, bw, dot = -bx, -by, -bz, -bw, -dot
    if dot > SLERP_LINEAR_THRESHOLD:
        out = [
            ax + (bx - ax) * t,
            ay + (by - ay) * t,
            az + (bz - az) * t,
            aw + (bw - aw) * t,
        ]
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        wa = math.sin((1.0 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta
        out = [ax * wa + bx * wb, ay * wa + by * wb, az * wa + bz * wb, aw * wa + bw * wb]
    norm = sum(v * v for v in out) ** 0.5 or 1.0
    return [v / norm for v in out]


def interpolate(
    pose_a: Mapping[str, Any], pose_b: Mapping[str, Any], frames: int
) -> list[dict[str, Any]]:
    """``frames`` pose records stepping from A toward B.

    The last frame stops *short* of B rather than landing on it, so a clip that
    loops back to A does not hold a duplicate frame at the seam. A bone posed in
    only one of the two ends interpolates from rest, which is what the worker's
    _reset_pose already means by an omitted bone.
    """
    if not 1 <= frames <= MAX_CLIP_FRAMES:
        raise ValueError(f"a clip must be 1-{MAX_CLIP_FRAMES} frames")
    bones_a = pose_a.get("bones") or {}
    bones_b = pose_b.get("bones") or {}
    names = sorted(set(bones_a) | set(bones_b))
    name = f"{pose_a.get('name', 'A')} -> {pose_b.get('name', 'B')}"
    out: list[dict[str, Any]] = []
    for i in range(frames):
        t = i / frames
        out.append(
            {
                # The row's identity is the clip, not the frame: the worker
                # re-poses per cell within a clip row, which is the one place
                # the group-by-row optimisation does not apply.
                "id": f"{pose_a.get('id') or ''}:{pose_b.get('id') or ''}",
                "name": f"{name} #{i}",
                "frame": i,
                "bones": {
                    bone: slerp(
                        bones_a.get(bone, IDENTITY_QUAT),
                        bones_b.get(bone, IDENTITY_QUAT),
                        t,
                    )
                    for bone in names
                },
            }
        )
    return out


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
                    # A record produced by interpolate() carries its own frame;
                    # an ordinary saved pose has none and defaults to 0.
                    frame=int(pose.get("frame", 0)),
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


def measure_trim(image: Any) -> dict[str, int] | None:
    """The alpha bounding box within one rendered frame, or None if it is empty.

    Handed to importers that pack tightly: a 128px cell whose subject occupies
    40x90 of it wastes most of its texture, and the trim rectangle is what lets
    a packer reclaim that without re-rendering. Measured rather than computed
    from the bbox because the silhouette, not the bounding volume, is what a
    packer cares about.
    """
    alpha = image.convert("RGBA").getchannel("A")
    box = alpha.getbbox()
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def pack(
    sheet: Plan, frames: Mapping[int, Path], out_png: Path
) -> dict[int, dict[str, int] | None]:
    """Composite the rendered frames into one RGBA atlas.

    Returns each cell's alpha bounding box, measured here because every frame
    is already open and decoded at this point.

    A missing or wrong-sized frame raises rather than silently leaving a hole:
    a sheet with an invisible gap in it looks like a modelling problem, and the
    user would go looking in the wrong place.
    """
    from PIL import Image

    size = sheet.frame_size
    atlas = Image.new("RGBA", (sheet.width, sheet.height), (0, 0, 0, 0))
    trims: dict[int, dict[str, int] | None] = {}
    try:
        for cell in sheet.cells:
            path = frames.get(cell.index)
            if path is None or not path.exists():
                raise ValueError(f"no rendered frame for cell {cell.index}")
            with Image.open(path) as frame:
                frame = frame.convert("RGBA")
                if frame.size != (size, size):
                    frame = frame.resize((size, size), Image.LANCZOS)
                trims[cell.index] = measure_trim(frame)
                atlas.paste(frame, (cell.x, cell.y))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        atlas.save(out_png, "PNG")
    finally:
        atlas.close()
    return trims


def sidecar(
    sheet: Plan,
    *,
    sheet_id: str,
    source_job: str,
    image: str,
    created: float,
    name: str = "",
    pivot: tuple[float, float] | None = None,
    trims: Mapping[int, dict[str, int] | None] | None = None,
) -> dict[str, Any]:
    """The engine-neutral description of the atlas next to it.

    Engine-neutral means no Godot ``AtlasTexture``, no Unity ``SpriteMetaData``
    -- just pixel rectangles and what each one shows, in the plainest JSON that
    can carry it. Anything more opinionated would have to be rewritten for the
    second engine anyone tries.
    """
    # The projected ground origin, in pixels within a cell. Identical for every
    # cell by construction: the camera is framed once from the rest bbox and
    # only spins, so the subject's origin lands in the same place in every
    # direction. That stability is the property an engine needs to place a
    # sprite without it drifting as the character turns.
    px, py = (
        pivot if pivot is not None else (sheet.frame_size / 2.0, float(sheet.frame_size))
    )
    cells = []
    for c in sheet.cells:
        entry = c.as_dict(sheet.frame_size)
        entry["pivot_x"] = px
        entry["pivot_y"] = py
        entry["trim"] = (trims or {}).get(c.index)
        cells.append(entry)
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
        "cells": cells,
    }
