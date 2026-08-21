"""A reader for Universal-LPC "full" sheets -- validation material, not art.

Troupe is an alternative to ULPC, not a consumer of it. This module exists for
two narrow reasons:

1. It turns the measurements taken off ``examples/*.png`` into regression
   oracles that cost nothing to keep passing -- the 352-cell frame table, the
   N/W/S/E direction order, the lossless W/E mirror away from the face.
2. It lets a user bring their own LPC art in as filler while a character is
   being built. **No ULPC art ships**; the example sheets are CC-BY-SA/GPL and
   stay out of the package and out of any training set.

Nothing here decodes pixels beyond slicing a regular 64px grid: this is a
layout table plus ``PIL.Image.crop``. The layout is the published "full" sheet
and is written as a literal table for ``spritesynth``'s reason -- so the order a
cell index means is readable in one glance and cannot be changed by a clever
arithmetic edit.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CELL",
    "DIRECTIONS",
    "LAYOUT",
    "SHEET_SIZE",
    "Row",
    "cells",
    "crop",
    "read",
    "row_frame_counts",
]

CELL = 64
SHEET_SIZE = (832, 3456)   # 13 columns x 54 rows, measured off both examples

#: The four directions a ULPC row block steps through, in row order. Confirmed
#: empirically rather than assumed: walk rows 9 and 11 are near-exact mirrors
#: (mean absolute difference 0.25, against 17+ for every other pairing), row 8
#: has no iris-coloured pixels at all, and row 10 has exactly twice as many as
#: either side. That is a back row, two profiles and a front row -- N, W, S, E.
DIRECTIONS = ("north", "west", "south", "east")


@dataclass(frozen=True, slots=True)
class Row:
    """One block of the sheet: an animation, its frame count, and its handing."""

    animation: str
    frames: int
    directional: bool   # False for the two single-direction rows (hurt, climb)


#: The published "full" layout, top to bottom. ``(7,8,9,6,13) x 4`` + ``6 + 6``
#: + ``(2,5,3,3,8,2,13,6) x 4`` = ``172 + 12 + 168 = 352`` occupied cells in a
#: 13 x 54 grid, which is exactly what both example sheets measure.
LAYOUT: tuple[Row, ...] = (
    Row("spellcast", 7, True),
    Row("thrust", 8, True),
    Row("walk", 9, True),
    Row("slash", 6, True),
    Row("shoot", 13, True),
    Row("hurt", 6, False),
    Row("climb", 6, False),
    Row("idle", 2, True),
    Row("jump", 5, True),
    Row("sit", 3, True),
    Row("emote", 3, True),
    Row("run", 8, True),
    Row("combat_idle", 2, True),
    Row("backslash", 13, True),
    Row("halfslash", 6, True),
)


def row_frame_counts() -> tuple[int, ...]:
    """Occupied cells per sheet row, top to bottom -- the direct oracle.

    A directional block is the same frame count four times over; the two
    single-direction rows contribute themselves once.
    """
    out: list[int] = []
    for row in LAYOUT:
        out.extend([row.frames] * (len(DIRECTIONS) if row.directional else 1))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class UlpcCell:
    """One named cell of a ULPC sheet, and where it is."""

    index: int
    row: int
    column: int
    animation: str
    direction: str
    frame: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        """The ``(left, upper, right, lower)`` crop box, in pixels."""
        x, y = self.column * CELL, self.row * CELL
        return (x, y, x + CELL, y + CELL)


def cells() -> tuple[UlpcCell, ...]:
    """All 352 cells, named, in top-to-bottom left-to-right sheet order."""
    return tuple(_walk())


def _walk() -> Iterator[UlpcCell]:
    index = 0
    row_index = 0
    for row in LAYOUT:
        directions = DIRECTIONS if row.directional else (row.animation,)
        for direction in directions:
            for frame in range(row.frames):
                yield UlpcCell(
                    index=index,
                    row=row_index,
                    column=frame,
                    animation=row.animation,
                    # A single-direction row is not facing nowhere; it is a
                    # side-on animation with no other view, so it is named for
                    # itself rather than borrowed one of the four compass
                    # words it does not mean.
                    direction=direction,
                    frame=frame,
                )
                index += 1
            row_index += 1


def read(image: Any) -> dict[tuple[str, str, int], Any]:
    """``(animation, direction, frame) -> RGBA cell image``.

    Refuses a sheet that is not the "full" layout rather than slicing whatever
    it was handed: a differently-generated sheet decoded on this table would
    hand back a walk frame labelled as a thrust, which is worse than an error.
    """
    if tuple(image.size) != SHEET_SIZE:
        raise ValueError(
            f"a ULPC full sheet is {SHEET_SIZE[0]}x{SHEET_SIZE[1]}px "
            f"and this is {image.size[0]}x{image.size[1]}"
        )
    rgba = image.convert("RGBA")
    return {
        (c.animation, c.direction, c.frame): rgba.crop(c.box) for c in cells()
    }


def crop(image: Any, animation: str, direction: str, frame: int) -> Any:
    """One named cell, without decoding the other 351."""
    if tuple(image.size) != SHEET_SIZE:
        raise ValueError(
            f"a ULPC full sheet is {SHEET_SIZE[0]}x{SHEET_SIZE[1]}px "
            f"and this is {image.size[0]}x{image.size[1]}"
        )
    for c in cells():
        if (c.animation, c.direction, c.frame) == (animation, direction, frame):
            return image.convert("RGBA").crop(c.box)
    raise KeyError(f"no {animation!r} {direction!r} frame {frame}")
