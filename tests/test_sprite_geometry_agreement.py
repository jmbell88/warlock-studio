"""The one owner of the agreement between the two copies of the sheet grid.

``pipelines.spritesynth`` decides where a cell is in a generated atlas;
``studio.inker.animation.DirectionalLayout`` decides what timeline frame *i*
means when that atlas is opened for editing. They cannot share code: the inker
package imports nothing outward (``tests/inker/test_sheetout.py`` pins the
exact set), and ``pipelines`` modules run inside worker and Blender processes
where ``studio`` is not importable at all.

So the two hold the same table twice, and this file is the only place the
copies are checked against each other. If it fails, a sprite draft generated
with the back view bottom-right will open in Inker as frame 2 -- silently, and
only visible once somebody plays the animation.
"""

from __future__ import annotations

import pytest

from warlock.pipelines import spritesynth as ss
from warlock.studio.inker import animation as anim


def test_the_two_modules_name_the_same_directions_in_the_same_order():
    assert anim.DIRECTION_ORDER == ss.DIRECTION_ORDER


def test_the_two_modules_agree_on_every_yaw():
    assert anim.DIRECTION_YAWS == ss.DIRECTION_YAWS


def test_the_two_modules_offer_the_same_sheet_types():
    assert set(anim.SHEET_KINDS) == set(ss.GEOMETRY) == set(ss.SHEET_TYPES)


@pytest.mark.parametrize("kind", sorted(ss.SHEET_TYPES))
def test_the_grids_agree(kind):
    geom = ss.geometry(kind)
    layout = anim.DirectionalLayout.of(kind)
    assert layout is not None
    assert (layout.columns, layout.rows) == (geom.columns, geom.rows)
    assert layout.frames_per_direction == geom.frames_per_direction
    assert layout.frame_count == len(geom.cells)


@pytest.mark.parametrize("kind", sorted(ss.SHEET_TYPES))
def test_cell_for_cell_the_two_say_the_same_thing(kind):
    """The one that matters: index *i* of the sidecar is frame *i* of the
    timeline, and both have to think it is the same row, column, direction,
    yaw and per-direction frame number."""
    geom = ss.geometry(kind)
    layout = anim.DirectionalLayout.of(kind)
    for index, cell in enumerate(geom.cells):
        assert layout.cell(index) == (
            cell.row,
            cell.col,
            cell.name,
            cell.yaw,
            cell.frame,
        )


def test_a_layout_refuses_an_index_past_its_grid():
    layout = anim.DirectionalLayout.of("turnaround")
    with pytest.raises(IndexError):
        layout.cell(4)
