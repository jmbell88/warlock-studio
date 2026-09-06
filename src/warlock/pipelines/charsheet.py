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
    "CAMERA_PRESETS",
    "COLUMNS",
    "DEFAULT_CAMERA_PRESET",
    "DIRECTION_PRESETS",
    "DIRECTIONS",
    "LAYOUT_VERSION",
    "MAX_CELLS",
    "MAX_FRAMES",
    "MOVEMENT_MIN_FRAMES",
    "WARN_CELLS",
    "SIZES",
    "LayoutSpec",
    "MovementSpec",
    "TroupeCell",
    "animation_block",
    "check_frame_counts",
    "check_subset",
    "frame_table",
    "plan",
    "point_in_cell",
    "resolve_layout",
    "spans",
    "subset_indices",
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

#: ``(key, label, elevation)`` -- the camera angles a character sheet may be
#: framed from, as a literal table for the same reason ``ANIMATIONS`` is one.
#:
#: ``isometric`` is 30.0 because that is :data:`sheet.DEFAULT_ELEVATION`, the
#: elevation every sheet this program has ever rendered used; naming it
#: "Isometric" rather than "2:1 dimetric" is deliberate, because "isometric" is
#: what players and every engine's tileset documentation call that projection
#: and a technically-correct label nobody searches for is a label nobody finds.
#:
#: ``top_down`` stops at 60 rather than reaching a true 90 overhead: a humanoid
#: seen straight down is a pair of shoulders and a hat brim -- it reads as a
#: blob, and every "top-down" sprite anyone actually ships is tilted.
#:
#: **The table lives in ``pipelines`` rather than in ``service.troupe``**
#: because the worker frames from it too, and ``pipelines`` is the one layer
#: both the door and the worker can see -- ``service`` is importable in neither
#: the Blender process nor the worker.
CAMERA_PRESETS: tuple[tuple[str, str, float], ...] = (
    ("three_quarter_top_down", "3/4 top-down", 35.0),
    ("isometric", "Isometric", 30.0),
    ("side", "Side", 0.0),
    ("top_down", "Top-down", 60.0),
)

DEFAULT_CAMERA_PRESET = "three_quarter_top_down"
LAYOUT_VERSION = 2
WARN_CELLS = 256
MAX_CELLS = 512
MAX_FRAMES = sheet.MAX_CLIP_FRAMES

_DIRECTIONS_16: tuple[tuple[str, float], ...] = (
    ("front", 0.0),
    ("front_front_left", 22.5),
    ("front_left", 45.0),
    ("left_front_left", 67.5),
    ("left", 90.0),
    ("left_back_left", 112.5),
    ("back_left", 135.0),
    ("back_back_left", 157.5),
    ("back", 180.0),
    ("back_back_right", 202.5),
    ("back_right", 225.0),
    ("right_back_right", 247.5),
    ("right", 270.0),
    ("right_front_right", 292.5),
    ("front_right", 315.0),
    ("front_front_right", 337.5),
)

DIRECTION_PRESETS: dict[int, tuple[tuple[str, float], ...]] = {
    1: (_DIRECTIONS_16[0],),
    4: tuple(_DIRECTIONS_16[i] for i in (0, 4, 8, 12)),
    8: tuple(_DIRECTIONS_16[i] for i in range(0, 16, 2)),
    16: _DIRECTIONS_16,
}

_ANIMATION_BY_NAME = {a[0]: a for a in ANIMATIONS}
MOVEMENT_MIN_FRAMES = {name: 1 for name, *_rest in ANIMATIONS}


@dataclass(frozen=True, slots=True)
class MovementSpec:
    name: str
    frames: int
    loop: bool
    duration_ms: int
    directions: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """A validated, immutable Troupe frame-table snapshot."""

    version: int
    columns: int
    movements: tuple[MovementSpec, ...]

    @property
    def cell_count(self) -> int:
        return sum(m.frames * len(m.directions) for m in self.movements)

    @property
    def directions(self) -> tuple[tuple[str, float], ...]:
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for movement in self.movements:
            for direction in movement.directions:
                if direction[0] not in seen:
                    seen.add(direction[0])
                    out.append(direction)
        return tuple(out)

    def as_dict(self) -> dict[str, Any]:
        runs = []
        index = 0
        movements = []
        for movement in self.movements:
            directions = [
                {"key": name, "label": name.replace("_", " ").title(), "yaw": yaw}
                for name, yaw in movement.directions
            ]
            movements.append(
                {
                    "key": movement.name,
                    "label": movement.name.title(),
                    "frames": movement.frames,
                    "loop": movement.loop,
                    "duration_ms": movement.duration_ms,
                    "directions": directions,
                }
            )
            for direction, yaw in movement.directions:
                runs.append(
                    {
                        "movement": movement.name,
                        "direction": direction,
                        "yaw": yaw,
                        "start": index,
                        "end": index + movement.frames - 1,
                    }
                )
                index += movement.frames
        return {
            "version": self.version,
            "columns": self.columns,
            "movements": movements,
            "runs": runs,
            "cell_count": self.cell_count,
        }


def resolve_layout(payload: Mapping[str, Any] | None = None) -> LayoutSpec:
    """Validate a v2 request/snapshot; absence is the immutable legacy layout."""

    if not payload:
        movements = tuple(
            MovementSpec(name, frames, loop, ms, DIRECTIONS)
            for name, frames, loop, ms in ANIMATIONS
        )
        return LayoutSpec(LAYOUT_VERSION, COLUMNS, movements)
    raw_version = payload.get("version")
    version = LAYOUT_VERSION if raw_version is None else int(raw_version)
    if version != LAYOUT_VERSION:
        raise ValueError(f"Troupe layout version {version} is not supported")
    raw_movements = payload.get("movements")
    if not isinstance(raw_movements, Sequence) or isinstance(raw_movements, (str, bytes)):
        raise ValueError("a Troupe layout needs at least one movement")
    movements: list[MovementSpec] = []
    seen: set[str] = set()
    for raw in raw_movements:
        if not isinstance(raw, Mapping):
            raise ValueError("every Troupe movement must be an object")
        name = str(raw.get("key") or raw.get("name") or "").strip()
        if name not in _ANIMATION_BY_NAME:
            raise ValueError(f"{name!r} is not a Troupe movement")
        if name in seen:
            raise ValueError(f"two Troupe movements are named {name!r}")
        seen.add(name)
        _name, default_frames, loop, ms = _ANIMATION_BY_NAME[name]
        raw_frames = raw.get("frames")
        frames = default_frames if raw_frames is None else int(raw_frames)
        minimum = MOVEMENT_MIN_FRAMES[name]
        if not minimum <= frames <= sheet.MAX_CLIP_FRAMES:
            raise ValueError(
                f"{name} must have {minimum}-{sheet.MAX_CLIP_FRAMES} frames"
            )
        raw_directions = raw.get("directions", raw.get("direction_preset", 8))
        if isinstance(raw_directions, Sequence) and not isinstance(
            raw_directions, (str, bytes)
        ):
            if any(not isinstance(d, Mapping) for d in raw_directions):
                raise ValueError("every Troupe direction must be an object")
            directions = tuple(
                (
                    str(d.get("key") or d.get("name") or ""),
                    float(d.get("yaw")),
                )
                for d in raw_directions
            )
            if directions not in DIRECTION_PRESETS.values():
                raise ValueError("directions must use the 1, 4, 8, or 16 direction preset")
        else:
            try:
                directions = DIRECTION_PRESETS[int(raw_directions)]
            except (KeyError, TypeError, ValueError):
                raise ValueError("directions must be 1, 4, 8, or 16") from None
        movements.append(MovementSpec(name, frames, loop, ms, directions))
    if not movements:
        raise ValueError("a Troupe layout needs at least one movement")
    raw_columns = payload.get("columns")
    columns = COLUMNS if raw_columns is None else int(raw_columns)
    if columns != COLUMNS:
        raise ValueError(f"Troupe sheets use exactly {COLUMNS} columns")
    result = LayoutSpec(LAYOUT_VERSION, columns, tuple(movements))
    if result.cell_count > MAX_CELLS:
        raise ValueError(f"a Troupe sheet may contain at most {MAX_CELLS} cells")
    return result


@dataclass(frozen=True, slots=True)
class TroupeCell:
    index: int
    animation: str
    direction: str
    yaw: float
    frame: int


def frames_per_direction(layout: LayoutSpec | Mapping[str, Any] | None = None) -> int:
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    return sum(m.frames for m in resolved.movements)


def frame_table(
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[TroupeCell, ...]:
    """Every cell of a character sheet, in pack-and-play order."""
    out: list[TroupeCell] = []
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    for movement in resolved.movements:
        for direction, yaw in movement.directions:
            for frame in range(movement.frames):
                out.append(
                    TroupeCell(
                        index=len(out),
                        animation=movement.name,
                        direction=direction,
                        yaw=yaw,
                        frame=frame,
                    )
                )
    return tuple(out)


def spans(
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str, int, int, bool], ...]:
    """``(animation, direction, start, end, loop)`` per contiguous run."""
    out = []
    index = 0
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    for movement in resolved.movements:
        for direction, _yaw in movement.directions:
            out.append(
                (
                    movement.name,
                    direction,
                    index,
                    index + movement.frames - 1,
                    movement.loop,
                )
            )
            index += movement.frames
    return tuple(out)


def check_subset(
    subset: Sequence[Mapping[str, Any]] | None,
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Normalise a re-render request to ``((animation, direction), ...)``.

    A subset names whole *runs* -- one animation seen from one direction --
    because a run is the unit a person judges and re-authors. Half a walk cycle
    re-rendered against the other half is not a thing anybody wants and would be
    a seam by construction.

    Refused by name rather than filtered, which is ``check_frame_counts``' rule
    one function up: a request naming an animation this sheet does not carry is
    a mistake about the sheet, and silently rendering the rest of it would leave
    the user looking for a change that never happened.

    **A subset naming every run is refused too**, and that one is not
    pedantry: it is a full render taking the slower path, with a copy step and
    a pinned palette it does not need. The ordinary door is right there.
    """
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    available = {(animation, direction) for animation, direction, *_ in spans(resolved)}
    if not subset:
        raise ValueError("name at least one animation and direction to re-render")
    wanted: list[tuple[str, str]] = []
    for entry in subset:
        animation = str((entry or {}).get("animation") or "")
        direction = str((entry or {}).get("direction") or "")
        if not animation or not direction:
            raise ValueError("each run needs an animation and a direction")
        if animation not in {name for name, _d in available}:
            raise ValueError(f"{animation} is not an animation on this sheet")
        if (animation, direction) not in available:
            raise ValueError(f"this sheet has no {animation} facing {direction}")
        if (animation, direction) in wanted:
            raise ValueError(f"{animation} facing {direction} is named twice")
        wanted.append((animation, direction))
    if len(wanted) == len(available):
        raise ValueError(
            "that is every run on the sheet -- build a new sheet instead of "
            "re-rendering all of it"
        )
    return tuple(wanted)


def subset_indices(
    subset: Sequence[Mapping[str, Any]] | None,
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """The cell indices a subset covers, ascending.

    Expanded from ``spans`` rather than from arithmetic of its own, so this
    answer and ``sheetscope.runs``' Inker-side answer come from the two copies
    ``tests/troupe/test_troupe_geometry_agreement.py`` already owns rather than
    from a third nothing owns.
    """
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    wanted = set(check_subset(subset, resolved))
    out: list[int] = []
    for animation, direction, start, end, _loop in spans(resolved):
        if (animation, direction) in wanted:
            out.extend(range(start, end + 1))
    return tuple(sorted(out))


def check_frame_counts(
    records: Mapping[str, Sequence[Any]],
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> None:
    """Refuse a set of expanded clips that does not fill the frame table.

    By name and with both numbers, rather than by padding or truncating: a
    seven-frame walk laid into an eight-frame table renders one cell of some
    other animation, and the user would go looking at the rig.
    """
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    for movement in resolved.movements:
        got = len(records.get(movement.name) or ())
        if got != movement.frames:
            raise ValueError(
                f"the {movement.name} clip expands to {got} frames and the table wants "
                f"{movement.frames}"
            )
    extra = sorted(set(records) - {m.name for m in resolved.movements})
    if extra:
        raise ValueError(f"not Troupe animations: {extra}")


def plan(
    records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    frame_size: int = 128,
    elevation: float = sheet.DEFAULT_ELEVATION,
    lighting: str = "flat",
    layout: LayoutSpec | Mapping[str, Any] | None = None,
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
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    check_frame_counts(records, resolved)

    table = frame_table(resolved)
    rows = (len(table) + resolved.columns - 1) // resolved.columns
    sheet.check_atlas_size(resolved.columns * frame_size, rows * frame_size)

    cells: list[sheet.Cell] = []
    poses: list[dict[str, Any]] = []
    for cell in table:
        record = records[cell.animation][cell.frame]
        column, row = cell.index % resolved.columns, cell.index // resolved.columns
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
        columns=resolved.columns,
        rows=rows,
        # Every direction the sheet contains, in table order -- ``Plan.yaws``
        # is the sheet's set of view directions, and here it is not the same
        # thing as "one per column" the way it is for a pose-per-row sheet.
        yaws=tuple(y for _n, y in resolved.directions),
        elevation=float(elevation),
        lighting=lighting,
        poses=tuple(poses),
        cells=tuple(cells),
    )


def animation_block(
    layout: LayoutSpec | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    resolved = layout if isinstance(layout, LayoutSpec) else resolve_layout(layout)
    table = frame_table(resolved)
    durations = {m.name: m.duration_ms for m in resolved.movements}
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
            for animation, direction, start, end, loop in spans(resolved)
        ],
    }


def point_in_cell(
    point: tuple[float, float] | None, frame_size: int
) -> tuple[float, float] | None:
    """A point the worker projected at ``RENDER_SIZE``, in *cell* pixels.

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

    **Generalised from ``pivot_in_cell`` on 2026-09-05**, when the worker began
    projecting *sockets* as well as the ground origin: they are the same render
    at the same size and want the same conversion, and a second copy of it
    would put a composited flame sixteen cells away from the hand rather than
    the feet.

    Here rather than inline in ``_q_troupe`` because this module is the
    filesystem-free half of the character sheet -- it decides what cell 137
    depicts and never reads a file -- which is what makes the arithmetic
    testable at all.
    """
    if point is None:
        return None
    scale = float(frame_size) / float(RENDER_SIZE)
    return (float(point[0]) * scale, float(point[1]) * scale)


def pivot_in_cell(
    pivot: tuple[float, float] | None, frame_size: int
) -> tuple[float, float] | None:
    """The sheet's ground origin, in cell pixels. :func:`point_in_cell` by its
    older name, kept because "pivot" is what the sidecar field is called."""
    return point_in_cell(pivot, frame_size)
