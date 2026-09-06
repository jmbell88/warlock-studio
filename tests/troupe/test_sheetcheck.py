"""The structural check over a finished sheet: clipped, blank, missing, and
whether the sidecar says what the atlas is.

Pure arithmetic over a plan, a trims map and a sidecar dict -- the same bargain
the module under test makes, so none of this needs Blender, PIL or a worker.
"""

from __future__ import annotations

from warlock.pipelines import charsheet, sheetcheck
from warlock.pipelines import sheet as sheetlib


def _plan(frame_size=64):
    return sheetlib.plan([], frame_size=frame_size, yaws=4)


def test_a_trim_touching_the_frame_edge_is_a_clip():
    """The measurable form of "the jump apex is cut off": the alpha bounding
    box runs flush against the cell edge, meaning the subject continued past
    the frame the camera drew. The trim one pixel in is the control -- a sheet
    that is merely tightly framed is not a defect."""
    plan = _plan()
    inside = {"x": 1, "y": 1, "w": 62, "h": 62}
    edge = {"x": 4, "y": 0, "w": 20, "h": 24}
    trims = {c.index: dict(inside) for c in plan.cells}
    trims[2] = edge
    assert sheetcheck.clipped_cells(plan, trims) == [2]
    assert sheetcheck.clipped_cells(plan, {c.index: dict(inside) for c in plan.cells}) == []


def test_a_missing_trim_is_blank_and_never_a_clip():
    """``measure_trim`` answers ``None`` for a frame with no opaque pixel, and
    calling that a clip would send the reframe retry -- a whole second render
    -- after a sheet with no subject in it at all. A cell the map has no entry
    for is a third thing again: never measured, which is plumbing rather than
    pixels."""
    plan = _plan()
    trims: dict[int, dict[str, int] | None] = {
        c.index: {"x": 1, "y": 1, "w": 62, "h": 62} for c in plan.cells
    }
    trims[1] = None
    del trims[3]
    assert sheetcheck.blank_cells(plan, trims) == [1]
    assert sheetcheck.clipped_cells(plan, trims) == []
    assert sheetcheck.missing_frames(plan, trims) == [3]


def test_every_cell_must_be_named_by_exactly_one_animation_tag():
    """A gap in the tag coverage is a frame no engine plays and an overlap is
    one it plays twice, and both reach the user as a stutter with nothing in
    the file to point at. ``charsheet.animation_block``'s own output is the
    control: it covers ``0..n-1`` exactly once by construction."""
    layout = charsheet.resolve_layout(
        {"version": 2, "movements": [{"key": "idle", "frames": 3, "directions": 1}]}
    )
    block = charsheet.animation_block(layout)
    meta = {
        "image": "sheet.png",
        "animation": block,
        "cells": [{"index": i, "w": 32, "h": 32, "pivot_x": 16, "pivot_y": 32} for i in range(3)],
        "troupe": layout.as_dict(),
    }
    assert sheetcheck.metadata_findings(meta) == []

    holed = dict(meta)
    holed["animation"] = {"frames": block["frames"], "tags": [dict(block["tags"][0], end=1)]}
    assert any("no tag" in finding for finding in sheetcheck.metadata_findings(holed))


def test_a_pivot_outside_its_cell_is_a_finding():
    """The pivot is documented as pixels *within* a cell, and the way it goes
    wrong is a render-size number written into a cell-size field -- a 32px cell
    carrying ``(256, 470)``, which puts the sprite's feet a long way below the
    sprite. ``charsheet.point_in_cell`` exists to convert it; this notices when
    something did not."""
    good = {
        "image": "sheet.png",
        "cells": [{"index": 0, "w": 32, "h": 32, "pivot_x": 16.0, "pivot_y": 32.0}],
    }
    assert sheetcheck.metadata_findings(good) == []

    bad = {
        "image": "sheet.png",
        "cells": [{"index": 0, "w": 32, "h": 32, "pivot_x": 256.0, "pivot_y": 470.0}],
    }
    findings = sheetcheck.metadata_findings(bad)
    assert findings and "pivot" in findings[0]
    verdict = sheetcheck.validate(_plan(), {}, bad)
    assert verdict["ok"] is False
    assert sheetcheck.describe(verdict)
