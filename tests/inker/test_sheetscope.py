"""Where a sheet correction goes, and nothing about what it writes.

The one fact the module rests on is that the compass direction lives only in
the tag's *name*, so the first assertions are about reading names -- including
the sixteen-direction ones whose own underscores would defeat a naive split.
The parity test at the end is what lets ``sheetscope`` hold a copy of the
direction table rather than importing ``pipelines`` (which the import pin
forbids).
"""

from __future__ import annotations

import pytest

from warlock.pipelines import charsheet
from warlock.studio.inker import sheetscope
from warlock.studio.inker.sheetin import span_tags

EIGHT = [name for name, _yaw in charsheet.DIRECTION_PRESETS[8]]


def _table(animations=("idle", "walk", "run"), directions=EIGHT, frames=(4, 6, 6)):
    """A dense table in cell order, one tag per (animation, direction)."""
    spans = []
    start = 0
    for animation, count in zip(animations, frames, strict=True):
        for direction in directions:
            spans.append(
                {
                    "name": f"{animation}_{direction}",
                    "start": start,
                    "end": start + count - 1,
                    "loop": True,
                }
            )
            start += count
    return span_tags(spans)


def test_a_tag_name_splits_on_its_longest_known_suffix():
    assert sheetscope.parse_tag_name("walk_front") == ("walk", "front")
    assert sheetscope.parse_tag_name("walk_front_left") == ("walk", "front_left")
    assert sheetscope.parse_tag_name("walk_front_front_left") == (
        "walk",
        "front_front_left",
    )
    assert sheetscope.parse_tag_name("left_front_left_left") == (
        "left_front_left",
        "left",
    )


def test_a_name_that_is_not_sheet_structure_is_left_alone():
    assert sheetscope.parse_tag_name("hit") is None
    assert sheetscope.parse_tag_name("front") is None
    assert sheetscope.parse_tag_name("_front") is None
    assert sheetscope.parse_tag_name("walk_sideways") is None


def test_runs_come_back_in_timeline_order_and_skip_foreign_tags():
    tags = _table()
    tags.insert(0, span_tags([{"name": "hit", "start": 10, "end": 12}])[0])
    sheet = sheetscope.runs(tags)
    assert len(sheet) == 24
    assert [run.start for run in sheet] == sorted(run.start for run in sheet)
    assert sheet[0] == sheetscope.Run("idle", "front", 0, 3, 1)
    assert sheetscope.has_sheet(tags)
    assert not sheetscope.has_sheet(tags[:1])


def test_locate_gives_the_run_and_the_offset():
    sheet = sheetscope.runs(_table())
    run, offset = sheetscope.locate(sheet, 4 * 8 + 6 + 2)
    assert (run.animation, run.direction, offset) == ("walk", "front_left", 2)
    assert sheetscope.locate(sheet, 999) is None


def test_the_directions_scope_is_the_same_offset_in_every_other_direction():
    sheet = sheetscope.runs(_table())
    walk_front_2 = 32 + 2
    picked = sheetscope.frames_for(sheet, walk_front_2, "directions")
    assert picked == [32 + 6 * d + 2 for d in range(1, 8)]
    assert walk_front_2 not in picked


def test_the_direction_scope_is_the_rest_of_the_run():
    sheet = sheetscope.runs(_table())
    assert sheetscope.frames_for(sheet, 34, "direction") == [32, 33, 35, 36, 37]


def test_the_animation_and_sheet_scopes_exclude_only_the_source():
    sheet = sheetscope.runs(_table())
    animation = sheetscope.frames_for(sheet, 34, "animation")
    assert animation == [f for f in range(32, 32 + 48) if f != 34]
    whole = sheetscope.frames_for(sheet, 34, "sheet")
    assert whole == [f for f in range(0, 128) if f != 34]


def test_a_ragged_table_skips_the_directions_that_run_short():
    tags = span_tags(
        [
            {"name": "walk_front", "start": 0, "end": 5},
            {"name": "walk_left", "start": 6, "end": 8},
            {"name": "walk_right", "start": 9, "end": 14},
        ]
    )
    sheet = sheetscope.runs(tags)
    assert sheetscope.frames_for(sheet, 4, "directions") == [13]


def test_the_explicit_scope_is_clamped_deduped_and_without_the_source():
    sheet = sheetscope.runs(_table())
    picked = sheetscope.frames_for(
        sheet, 3, "explicit", [9, 3, 1, 9, -2, 500], frame_count=128
    )
    assert picked == [1, 9]


def test_an_unknown_scope_is_refused_by_name():
    with pytest.raises(ValueError, match="scope"):
        sheetscope.frames_for([], 0, "everywhere")


def test_a_frame_outside_every_run_reaches_nothing():
    sheet = sheetscope.runs(_table())
    assert sheetscope.frames_for(sheet, 500, "sheet") == []


@pytest.mark.parametrize(
    ("direction", "mirror"),
    [
        ("left", "right"),
        ("right", "left"),
        ("front_left", "front_right"),
        ("back_left", "back_right"),
        ("front_front_left", "front_front_right"),
        ("left_back_left", "right_back_right"),
        ("front", None),
        ("back", None),
        ("sideways", None),
    ],
)
def test_the_mirror_of_a_direction(direction, mirror):
    assert sheetscope.opposite(direction) == mirror


def test_the_counterpart_is_the_same_offset_in_the_mirror_direction():
    sheet = sheetscope.runs(_table())
    # walk: front 32-37, front_left 38-43, left 44-49, back_left 50-55,
    # back 56-61, back_right 62-67, right 68-73, front_right 74-79.
    assert sheetscope.counterpart(sheet, 44 + 3) == 68 + 3
    assert sheetscope.counterpart(sheet, 68 + 3) == 44 + 3
    assert sheetscope.counterpart(sheet, 32) is None
    assert sheetscope.counterpart(sheet, 500) is None


def test_a_four_direction_sheet_pairs_left_with_right():
    four = [name for name, _yaw in charsheet.DIRECTION_PRESETS[4]]
    sheet = sheetscope.runs(_table(("walk",), four, (4,)))
    assert sheetscope.counterpart(sheet, 4) == 12
    assert sheetscope.counterpart(sheet, 0) is None


def test_the_direction_table_is_the_pipelines_own():
    """The copy is pinned here so ``sheetscope`` never has to import
    ``pipelines`` -- ``tests/inker/test_inker_imports.py`` is the wall."""
    assert tuple(n for n, _y in charsheet._DIRECTIONS_16) == sheetscope.DIRECTIONS_16
    for index, (name, yaw) in enumerate(charsheet._DIRECTIONS_16):
        mirror = sheetscope.opposite(name)
        if mirror is None:
            assert yaw in (0.0, 180.0)
            continue
        assert dict(charsheet._DIRECTIONS_16)[mirror] == (360.0 - yaw) % 360.0
        assert index == sheetscope.DIRECTIONS_16.index(name)


def test_every_scope_has_a_label():
    assert set(sheetscope.SCOPE_LABELS) == set(sheetscope.SCOPES)
