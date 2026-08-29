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

Three tables are held twice now rather than one: the direction names and yaws,
the sheet kinds and their grids, and the per-action frame counts. The last is
the newest and the easiest to get wrong, because ``spritesynth`` derives its
kinds from a table of actions and ``animation`` derives its from a copy of that
table -- so a walk that is eight frames on one side and six on the other would
give the editor a grid four cells short of the sheet it was handed.
"""

from __future__ import annotations

import pytest

from warlock.pipelines import spritesynth as ss
from warlock.studio.inker import animation as anim

ALL_KINDS = sorted(set(ss.SHEET_TYPES) | set(ss.PLANNED_KINDS))


def _geometry(kind):
    """The geometry for a kind of either era, at whatever size it plans at."""
    return ss.sheet_geometry(kind)


def test_the_two_modules_name_the_same_directions_in_the_same_order():
    assert anim.DIRECTION_ORDER == ss.DIRECTION_ORDER


def test_the_two_modules_agree_on_every_yaw():
    assert anim.DIRECTION_YAWS == ss.DIRECTION_YAWS


def test_the_two_modules_agree_on_all_eight_directions_and_their_yaws():
    assert anim.SPRITE_YAWS == ss.DIRECTION_YAWS_8
    assert anim.SPRITE_DIRECTIONS == ss.SPRITE_DIRECTIONS
    assert anim.DIRECTION_COUNTS == ss.DIRECTION_COUNTS


def test_the_two_modules_agree_on_how_many_frames_each_action_has():
    assert anim.ACTION_FRAMES == ss.ACTION_FRAMES


def test_the_two_modules_offer_the_same_sheet_kinds():
    assert set(anim.SHEET_KINDS) == set(ss.SHEET_TYPES) | set(ss.PLANNED_KINDS)


def test_the_legacy_kinds_are_still_the_only_ones_with_a_fixed_atlas():
    """``GEOMETRY`` is the atlas table and must not quietly grow: everything in
    it is generated in one 1024px frame, which is the thing eight directions
    cannot be."""
    assert set(ss.GEOMETRY) == set(ss.SHEET_TYPES)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_the_grids_agree(kind):
    geom = _geometry(kind)
    layout = anim.DirectionalLayout.of(kind)
    assert layout is not None
    assert (layout.columns, layout.rows) == (geom.columns, geom.rows)
    assert layout.frames_per_direction == geom.frames_per_direction
    assert layout.frame_count == len(geom.cells)
    assert layout.directions == geom.directions


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_cell_for_cell_the_two_say_the_same_thing(kind):
    """The one that matters: index *i* of the sidecar is frame *i* of the
    timeline, and both have to think it is the same row, column, direction,
    yaw and per-direction frame number."""
    geom = _geometry(kind)
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


def test_walk_and_walk4_are_two_different_sheets_on_both_sides():
    """The near-collision the two tables have to agree about. Legacy ``walk``
    is a four-frame cycle; ``walk4`` is the action table's eight-frame one over
    the same four directions. Aliasing either onto the other would halve or
    double a stored cycle, so both modules carry both."""
    assert ss.geometry("walk").frames_per_direction == 4
    assert ss.plan_kind("walk4").frames_per_direction == 8
    assert anim.DirectionalLayout.of("walk").frames_per_direction == 4
    assert anim.DirectionalLayout.of("walk4").frames_per_direction == 8


def test_an_older_build_loses_the_grid_and_not_the_document():
    """The tolerance the whole widening leans on: ``of`` answers None for a
    kind it does not carry, so a build that predates these names opens a
    ``walk8`` draft as an ordinary animation rather than refusing it."""
    assert anim.DirectionalLayout.of("gallop12") is None
    assert anim.DirectionalLayout.of(None) is None
