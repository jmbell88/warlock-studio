"""The immutable legacy Troupe layout used by pre-v2 sheets.

New sheets carry a resolved layout snapshot in their sidecar. This table remains
the compatibility authority for sheets written before that block existed.

The table itself lives in ``data/layout.json`` and carries a version, for the
reason every stored corpus in this repo does: a sheet on disk was laid out by
some version of this table, and a sheet whose tags no longer match its cells is
a document nobody can open. The code here only interprets it.

**Cell order is grouped by ``(animation, direction)`` and dense.** Grouped
because an Inker ``Tag`` is a contiguous span of timeline frames and timeline
frame *i* is cell *i*; dense because the atlas is then exactly ``columns x
(total frames / columns)`` with no empty cells, which is what keeps a 128px
character inside ``MAX_ATLAS_PX`` as a single atlas. The grid is therefore
*not* "one row per animation" -- a row can straddle two directions, and the
sidecar names every cell, so nothing has to infer it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = [
    "TROUPE_SPEC_VERSION",
    "Animation",
    "Cell",
    "Direction",
    "Spec",
    "load",
]

TROUPE_SPEC_VERSION = 1

_DATA = Path(__file__).resolve().parent / "data" / "layout.json"


@dataclass(frozen=True, slots=True)
class Animation:
    """One clip in the frame table."""

    name: str
    frames: int
    loop: bool
    keys: int          # unique authored keyframes; documentation, not geometry
    duration_ms: int


@dataclass(frozen=True, slots=True)
class Direction:
    name: str
    yaw: float         # degrees clockwise from the front view, sheet.py's axes


@dataclass(frozen=True, slots=True)
class Cell:
    """One sprite, and what it depicts."""

    index: int
    row: int
    column: int
    animation: str
    direction: str
    yaw: float
    frame: int         # frame number within this (animation, direction) run


@dataclass(frozen=True, slots=True)
class Spec:
    version: int
    animations: tuple[Animation, ...]
    directions: tuple[Direction, ...]
    sizes: tuple[int, ...]
    render_size: int
    elevation: float
    columns: int

    # -- derived ------------------------------------------------------------

    @property
    def frames_per_direction(self) -> int:
        """The 32 of ``32 x 8 directions = 256``."""
        return sum(a.frames for a in self.animations)

    @property
    def cell_count(self) -> int:
        return self.frames_per_direction * len(self.directions)

    @property
    def rows(self) -> int:
        count, columns = self.cell_count, self.columns
        return count // columns + (1 if count % columns else 0)

    def animation(self, name: str) -> Animation:
        for a in self.animations:
            if a.name == name:
                return a
        raise KeyError(f"{name!r} is not a Troupe animation")

    def direction(self, name: str) -> Direction:
        for d in self.directions:
            if d.name == name:
                return d
        raise KeyError(f"{name!r} is not a Troupe direction")

    def atlas_size(self, size: int) -> tuple[int, int]:
        """``(width, height)`` of the sheet at logical cell size ``size``."""
        return (self.columns * size, self.rows * size)

    def exact_sizes(self) -> tuple[int, ...]:
        """The rungs a whole-integer stride reaches from ``render_size``.

        ``pixel.downscale_grid`` is integer-stride only, so 16/32/64/128 out of
        512 are exact and 24/48/96 are not -- those go through the documented
        single-NEAREST-resize fallback instead. Naming the difference here
        stops each caller rediscovering it.
        """
        return tuple(s for s in self.sizes if self.render_size % s == 0)

    def cells(self) -> tuple[Cell, ...]:
        """Every cell, in the order a sheet is packed and a timeline plays."""
        out: list[Cell] = []
        for anim in self.animations:
            for d in self.directions:
                for frame in range(anim.frames):
                    index = len(out)
                    out.append(
                        Cell(
                            index=index,
                            row=index // self.columns,
                            column=index % self.columns,
                            animation=anim.name,
                            direction=d.name,
                            yaw=d.yaw,
                            frame=frame,
                        )
                    )
        return tuple(out)

    def spans(self) -> tuple[tuple[str, str, int, int, bool], ...]:
        """``(animation, direction, start, end, loop)`` per contiguous run.

        The one thing a tag needs and the one thing an importer needs, produced
        once so the Inker handoff and the sidecar's ``animation`` block cannot
        disagree about where ``walk_left`` starts.
        """
        out = []
        index = 0
        for anim in self.animations:
            for d in self.directions:
                out.append(
                    (anim.name, d.name, index, index + anim.frames - 1, anim.loop)
                )
                index += anim.frames
        return tuple(out)


def _parse(payload: dict) -> Spec:
    version = int(payload["version"])
    if version != TROUPE_SPEC_VERSION:
        raise ValueError(
            f"layout.json is version {version}; this build reads {TROUPE_SPEC_VERSION}"
        )
    animations = tuple(
        Animation(
            name=str(a["name"]),
            frames=int(a["frames"]),
            loop=bool(a["loop"]),
            keys=int(a["keys"]),
            duration_ms=int(a["duration_ms"]),
        )
        for a in payload["animations"]
    )
    directions = tuple(
        Direction(name=str(d["name"]), yaw=float(d["yaw"]))
        for d in payload["directions"]
    )
    if not animations:
        raise ValueError("a spec with no animations lays out nothing")
    if not directions:
        raise ValueError("a spec needs at least one direction")
    if len({a.name for a in animations}) != len(animations):
        raise ValueError("two animations share a name")
    if len({d.name for d in directions}) != len(directions):
        raise ValueError("two directions share a name")
    for a in animations:
        if a.frames < 1:
            raise ValueError(f"{a.name!r} has no frames")
        if a.duration_ms < 1:
            raise ValueError(f"{a.name!r} has no duration")
    return Spec(
        version=version,
        animations=animations,
        directions=directions,
        sizes=tuple(int(s) for s in payload["sizes"]),
        render_size=int(payload["render_size"]),
        elevation=float(payload["elevation"]),
        columns=int(payload["columns"]),
    )


@lru_cache(maxsize=1)
def load() -> Spec:
    """The shipped frame table. Cached: it is immutable and read every plan."""
    return _parse(json.loads(_DATA.read_text(encoding="utf-8")))
