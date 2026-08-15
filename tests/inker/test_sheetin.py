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


# --- a typed grid (C13c) -----------------------------------------------------
#
# The other door in: an image from anywhere, sliced on numbers the user typed.
# The arithmetic is worth its own tests for one reason -- the last column and
# the last row carry no trailing padding, and every naive division gets that
# wrong in the direction that silently drops a frame.


def test_a_plain_grid_is_row_major_with_no_gaps():
    rects = sheetin.grid_rects((64, 32), (32, 32))
    assert rects == [(0, 0, 32, 32), (32, 0, 32, 32)]


def test_the_last_column_carries_no_trailing_padding():
    """4 cells of 32 with 2px between them is 4*32 + 3*2 = 134, not 4*34. A
    134-wide sheet holds four columns, and dividing by 34 finds three."""
    rects = sheetin.grid_rects((134, 32), (32, 32), padding=(2, 0))
    assert len(rects) == 4
    assert [x for x, _y, _w, _h in rects] == [0, 34, 68, 102]


def test_the_last_row_carries_no_trailing_padding_either():
    rects = sheetin.grid_rects((32, 100), (32, 32), padding=(0, 2))
    assert len(rects) == 3
    assert [y for _x, y, _w, _h in rects] == [0, 34, 68]


def test_an_offset_shifts_the_whole_grid_and_costs_it_room():
    rects = sheetin.grid_rects((70, 32), (32, 32), offset=(4, 0))
    assert len(rects) == 2
    assert rects[0] == (4, 0, 32, 32)


def test_a_count_takes_the_first_n_cells_row_major():
    rects = sheetin.grid_rects((64, 64), (32, 32), count=3)
    assert rects == [(0, 0, 32, 32), (32, 0, 32, 32), (0, 32, 32, 32)]


def test_a_count_past_the_grids_capacity_is_refused_by_name():
    with pytest.raises(ValueError, match="4 cells and 9 were asked for"):
        sheetin.grid_rects((64, 64), (32, 32), count=9)


def test_a_cell_bigger_than_the_image_is_refused_by_name():
    with pytest.raises(ValueError, match="does not fit"):
        sheetin.grid_rects((16, 16), (32, 32))


def test_a_zero_or_negative_cell_is_refused():
    with pytest.raises(ValueError, match="positive size"):
        sheetin.grid_rects((64, 64), (0, 32))


def test_negative_offsets_and_padding_are_refused():
    with pytest.raises(ValueError, match="offset"):
        sheetin.grid_rects((64, 64), (32, 32), offset=(-1, 0))
    with pytest.raises(ValueError, match="padding"):
        sheetin.grid_rects((64, 64), (32, 32), padding=(0, -2))


def test_a_count_of_zero_is_refused_rather_than_meaning_all():
    """The popup spells "all" as None. Zero arriving here is a typed number and
    a sheet of no frames is not a document."""
    with pytest.raises(ValueError, match="at least one frame"):
        sheetin.grid_rects((64, 64), (32, 32), count=0)


def _striped(width: int, height: int) -> np.ndarray:
    """An atlas whose every column is a different red, so a mis-sliced cell is
    visible as the wrong number rather than as the wrong shape."""
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., 3] = 255
    out[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
    return out


def test_a_grid_document_slices_the_cells_it_says_it_will():
    atlas = _striped(64, 32)
    doc = sheetin.document_from_grid(atlas, (32, 32))
    assert doc.anim is not None
    assert len(doc.anim.frames) == 2
    assert doc.size == (32, 32)
    cels = list(doc.anim.unique_cel_layers())
    assert np.array_equal(cels[0].pixels, atlas[0:32, 0:32])
    assert np.array_equal(cels[1].pixels, atlas[0:32, 32:64])


def test_a_grid_document_carries_no_directional_layout():
    """A layout is a claim that these cells are four named directions in a
    fixed grid -- something the generator knows and a typed cell size does
    not."""
    doc = sheetin.document_from_grid(_striped(64, 32), (32, 32))
    assert doc.anim.layout is None
    assert doc.anim.tags == []


def test_a_grid_document_is_unsaved_but_clean():
    doc = sheetin.document_from_grid(_striped(64, 32), (32, 32))
    assert doc.path is None
    assert doc.file_format == "ora"
    assert len(doc.history) == 0


def test_a_grid_document_copies_rather_than_viewing_the_atlas():
    """A view would keep the whole atlas alive behind every frame and make two
    overlapping cells share pixels -- a stroke on one frame appearing on
    another with no link to explain it."""
    atlas = _striped(64, 32)
    doc = sheetin.document_from_grid(atlas, (32, 32))
    cel = next(doc.anim.unique_cel_layers())
    cel.pixels[0, 0] = (9, 9, 9, 9)
    assert not np.array_equal(atlas[0, 0], np.array([9, 9, 9, 9], dtype=np.uint8))
