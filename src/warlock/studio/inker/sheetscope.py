"""Where a correction goes on a character sheet: the addressing, and only that.

A Troupe sheet opens in Inker (``sheetin.document_from_sheet``) as one track
whose timeline is the atlas read in cell order -- frame *i* is cell *i* -- and
one ``Tag`` per ``(animation, direction)`` run, named ``walk_left`` by
``pipelines.charsheet.animation_block``. Every cell is one canvas of the same
size, which is the fact this module and ``_doc_sheet`` rest on: a pixel at
``(x, y)`` on one cell *means* the same place on every other, so a patch, a
recolour or a mirror transfers by coordinates alone with nothing to register.

**The compass direction lives only in the tag's name.** ``Tag`` records a
playback direction (forward/reverse/pingpong) and nothing about the compass,
and the sixteen-direction names carry underscores of their own
(``front_front_left``), so a name is parsed by its longest known suffix rather
than split on the first underscore. The table is a copy of
``charsheet._DIRECTIONS_16``'s names, because this package may not import
``warlock.pipelines`` (``tests/inker/test_inker_imports.py``); the parity test
in ``tests/inker/test_sheetscope.py`` is what keeps the copy honest.

Pure, index-returning, and it writes nothing. The frames it names are handed
to ``_doc_sheet.SheetOps``, which is the one door onto the document.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DIRECTIONS_16",
    "SCOPES",
    "SCOPE_LABELS",
    "Run",
    "counterpart",
    "frames_for",
    "has_sheet",
    "locate",
    "opposite",
    "parse_tag_name",
    "runs",
]

#: The sixteen compass names in yaw order, ``charsheet._DIRECTIONS_16`` minus
#: the angles. Index *i* faces ``i * 22.5`` degrees; the mirror of index *i* is
#: index ``(16 - i) % 16``, which is what :func:`opposite` computes.
DIRECTIONS_16: tuple[str, ...] = (
    "front",
    "front_front_left",
    "front_left",
    "left_front_left",
    "left",
    "left_back_left",
    "back_left",
    "back_back_left",
    "back",
    "back_back_right",
    "back_right",
    "right_back_right",
    "right",
    "right_front_right",
    "front_right",
    "front_front_right",
)

#: Longest first, so ``walk_front_front_left`` is read as ``front_front_left``
#: rather than as ``left`` with an unparseable animation ``walk_front_front``.
_BY_LENGTH: tuple[str, ...] = tuple(sorted(DIRECTIONS_16, key=len, reverse=True))

#: The scopes a correction can be sent to, in the order the strip offers them.
SCOPES: tuple[str, ...] = ("directions", "direction", "animation", "sheet", "explicit")

SCOPE_LABELS: dict[str, str] = {
    "directions": "This frame, every direction",
    "direction": "Every frame of this direction",
    "animation": "Every cell of this animation",
    "sheet": "Every cell of the sheet",
    "explicit": "The selected range",
}


@dataclass(frozen=True)
class Run:
    """One ``(animation, direction)`` span of the timeline, inclusive."""

    animation: str
    direction: str
    start: int
    end: int
    tag_index: int

    @property
    def frames(self) -> int:
        return self.end - self.start + 1


def parse_tag_name(name: str) -> tuple[str, str] | None:
    """``("walk", "front_left")`` from ``"walk_front_left"``; None if no
    compass name ends it or nothing precedes the underscore."""
    text = str(name or "")
    for direction in _BY_LENGTH:
        suffix = "_" + direction
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)], direction
    return None


def runs(tags: Sequence[Any]) -> list[Run]:
    """The parseable tags as runs, in timeline order. Unparseable tags are
    simply not sheet structure -- a user's own ``hit`` tag beside the sheet's
    is left alone rather than refused."""
    found: list[Run] = []
    for index, tag in enumerate(tags):
        parsed = parse_tag_name(getattr(tag, "name", ""))
        if parsed is None:
            continue
        start, end = int(tag.start), int(tag.end)
        if end < start or start < 0:
            continue
        found.append(Run(parsed[0], parsed[1], start, end, index))
    found.sort(key=lambda run: (run.start, run.tag_index))
    return found


def has_sheet(tags: Sequence[Any]) -> bool:
    return bool(runs(tags))


def locate(sheet: Sequence[Run], frame: int) -> tuple[Run, int] | None:
    """The run holding ``frame`` and the frame's offset inside it."""
    for run in sheet:
        if run.start <= int(frame) <= run.end:
            return run, int(frame) - run.start
    return None


def frames_for(
    sheet: Sequence[Run],
    frame: int,
    scope: str,
    explicit: Iterable[int] = (),
    *,
    frame_count: int | None = None,
) -> list[int]:
    """The target frames of a correction made on ``frame``, in timeline order.

    The source is never in the list -- it already carries the correction --
    and nothing appears twice. ``directions`` is the same offset in every
    other run of the same animation, skipping a run too short to have that
    offset (a ragged table is legal); ``direction`` is the rest of the source's
    own run; ``animation`` is every other cell of every run of that animation;
    ``sheet`` is every other cell of every run. ``explicit`` takes the caller's
    frame list, clamped to ``frame_count`` when given, so a range selection
    can be a scope too.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}")
    source = int(frame)
    if scope == "explicit":
        picked = sorted({int(f) for f in explicit})
        if frame_count is not None:
            picked = [f for f in picked if 0 <= f < frame_count]
        return [f for f in picked if f != source]
    here = locate(sheet, source)
    if here is None:
        return []
    run, offset = here
    out: list[int] = []
    for other in sheet:
        if scope == "directions":
            if other.animation != run.animation or offset >= other.frames:
                continue
            out.append(other.start + offset)
        elif scope == "direction":
            if other is not run:
                continue
            out.extend(range(other.start, other.end + 1))
        elif scope == "animation":
            if other.animation != run.animation:
                continue
            out.extend(range(other.start, other.end + 1))
        else:  # sheet
            out.extend(range(other.start, other.end + 1))
    seen: set[int] = {source}
    ordered: list[int] = []
    for f in sorted(out):
        if f in seen:
            continue
        seen.add(f)
        ordered.append(f)
    return ordered


def opposite(direction: str) -> str | None:
    """The compass name facing the mirror of ``direction``, or None for the
    two that are their own mirror (``front``, ``back``) and for a name that is
    not a compass direction at all."""
    try:
        index = DIRECTIONS_16.index(str(direction))
    except ValueError:
        return None
    mirror = (16 - index) % 16
    return None if mirror == index else DIRECTIONS_16[mirror]


def counterpart(sheet: Sequence[Run], frame: int) -> int | None:
    """The frame at the same offset in the mirror direction of ``frame``'s
    run, or None when there is no such direction on the sheet."""
    here = locate(sheet, int(frame))
    if here is None:
        return None
    run, offset = here
    wanted = opposite(run.direction)
    if wanted is None:
        return None
    for other in sheet:
        if other.animation == run.animation and other.direction == wanted:
            return other.start + offset if offset < other.frames else None
    return None
