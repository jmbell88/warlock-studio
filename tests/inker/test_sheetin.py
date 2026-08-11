"""Adopting a generated atlas as a document, and the round trip back out.

``sheetin`` and ``sheetout`` are inverses, and the highest-value test here is
the one that says so: adopt a sidecar plus a PNG, export it again, and the
atlas that comes back is the one that went in. Everything either half could get
wrong -- a cell sliced from the wrong rectangle, a row that means the wrong
direction, a wrap where a fixed grid belongs -- shows up as a pixel in the
wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import sheetin, sheetout
from warlock.studio.inker.animation import DIRECTION_ORDER, DirectionalLayout

CELL = 8


def _cells(kind: str) -> list[dict[str, int]]:
    layout = DirectionalLayout.of(kind)
    out = []
    for index in range(layout.frame_count):
        row, col, *_ = layout.cell(index)
        out.append(
            {"x": col * CELL, "y": row * CELL, "w": CELL, "h": CELL, "index": index}
        )
    return out


def _atlas(kind: str) -> np.ndarray:
    """One marker pixel per cell, at a value only that cell can have."""
    layout = DirectionalLayout.of(kind)
    atlas = np.zeros((layout.rows * CELL, layout.columns * CELL, 4), dtype=np.uint8)
    for index, cell in enumerate(_cells(kind)):
        atlas[cell["y"] : cell["y"] + CELL, cell["x"] : cell["x"] + CELL] = (
            10 + index,
            20,
            30,
            255,
        )
    return atlas


# --- slicing ----------------------------------------------------------------


@pytest.mark.parametrize("kind", ("turnaround", "walk"))
def test_every_cell_lands_on_its_own_frame(kind):
    doc = sheetin.document_from_atlas(_atlas(kind), _cells(kind), kind)
    anim = doc.anim
    assert len(anim.frames) == DirectionalLayout.of(kind).frame_count
    for index, frame in enumerate(anim.frames):
        cel = anim.cel(anim.tracks[0].uid, frame.uid)
        assert cel is not None
        assert int(cel.pixels[0, 0, 0]) == 10 + index


def test_the_document_is_the_size_of_one_cell():
    doc = sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "walk")
    assert doc.stack.size == (CELL, CELL)


def test_no_frame_shares_memory_with_the_atlas_or_another_frame():
    """A view would keep the whole atlas alive behind every frame and, worse,
    let a stroke on one cell appear on another with no link to explain it."""
    atlas = _atlas("walk")
    doc = sheetin.document_from_atlas(atlas, _cells("walk"), "walk")
    cels = list(doc.anim.unique_cel_layers())
    assert len(cels) == 16
    cels[0].pixels[:] = 7
    assert int(atlas[0, 0, 0]) == 10
    assert int(cels[1].pixels[0, 0, 0]) == 11


# --- tags and layout --------------------------------------------------------


def test_a_walk_sheet_arrives_with_one_looping_tag_per_direction():
    doc = sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "walk")
    tags = doc.anim.tags
    assert [t.name for t in tags] == [f"walk_{d}" for d in DIRECTION_ORDER]
    assert [(t.start, t.end) for t in tags] == [(0, 3), (4, 7), (8, 11), (12, 15)]
    assert all(t.loop for t in tags)


def test_playback_loops_inside_one_direction():
    """The whole reason tags are enough: no change to the animation engine."""
    doc = sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "walk")
    assert doc.anim.loop_range(5) == (4, 7, True)


def test_a_turnaround_is_tagless():
    """Four still views are not a cycle; tagging them would put four one-frame
    loops in the timeline that mean nothing to play."""
    doc = sheetin.document_from_atlas(_atlas("turnaround"), _cells("turnaround"), "turnaround")
    assert doc.anim.tags == []


def test_the_layout_rides_along():
    doc = sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "walk")
    assert doc.anim.layout == DirectionalLayout("walk")


def test_every_frame_gets_the_default_duration():
    from warlock.studio.inker.animation import DEFAULT_DURATION_MS

    doc = sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "walk")
    assert {f.duration_ms for f in doc.anim.frames} == {DEFAULT_DURATION_MS}


# --- what it refuses --------------------------------------------------------


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="not a sprite sheet layout"):
        sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "isometric")


def test_a_cell_count_that_does_not_fill_the_grid_is_refused():
    with pytest.raises(ValueError, match="15"):
        sheetin.document_from_atlas(_atlas("walk"), _cells("walk")[:15], "walk")


def test_a_rectangle_off_the_edge_of_the_atlas_is_refused():
    cells = _cells("turnaround")
    cells[-1] = dict(cells[-1], x=cells[-1]["x"] + 1)
    with pytest.raises(ValueError, match="outside the"):
        sheetin.document_from_atlas(_atlas("turnaround"), cells, "turnaround")


def test_cells_of_different_sizes_are_refused():
    cells = _cells("turnaround")
    cells[0] = dict(cells[0], w=CELL - 1)
    with pytest.raises(ValueError, match="the same size"):
        sheetin.document_from_atlas(_atlas("turnaround"), cells, "turnaround")


# --- the tab it becomes -----------------------------------------------------


def test_an_adopted_document_is_unsaved_but_clean():
    """No path, so the first Ctrl+S is a Save As -- a draft on disk is where
    this document came from, not its file -- and an empty history, so closing
    it immediately prompts about nothing."""
    doc = sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "walk")
    assert doc.path is None
    assert doc.file_format == "ora"
    assert not doc.history.can_undo


# --- the round trip ---------------------------------------------------------


@pytest.mark.parametrize("kind", ("turnaround", "walk"))
def test_adopting_then_exporting_reproduces_the_atlas(kind):
    atlas = _atlas(kind)
    doc = sheetin.document_from_atlas(atlas, _cells(kind), kind)
    image, plan, extra = sheetout.from_document(doc)
    try:
        out = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    finally:
        image.close()
    assert out.shape == atlas.shape
    assert np.array_equal(out, atlas)
    # And the grid is the sheet's own, not a wrap that happened to fit.
    layout = DirectionalLayout.of(kind)
    assert (plan.columns, plan.rows) == (layout.columns, layout.rows)
    assert extra["animation"]["layout"] == {
        "kind": kind,
        "directions": list(DIRECTION_ORDER),
    }


def test_the_exported_cells_carry_their_direction_and_frame():
    doc = sheetin.document_from_atlas(_atlas("walk"), _cells("walk"), "walk")
    _image, plan, _extra = sheetout.from_document(doc)
    layout = DirectionalLayout.of("walk")
    for index, cell in enumerate(plan.cells):
        row, col, direction, yaw, frame = layout.cell(index)
        assert (cell.row, cell.column) == (row, col)
        assert (cell.pose_name, cell.yaw, cell.frame) == (direction, float(yaw), frame)
