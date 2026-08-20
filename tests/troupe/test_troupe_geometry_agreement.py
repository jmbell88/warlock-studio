"""The one owner of the agreement between the two copies of Troupe's table.

``pipelines.charsheet`` decides where a cell is in a rendered character sheet;
``studio.troupe.spec`` decides what that cell means to the studio, and the
Inker handoff built on it. They cannot share code -- ``studio/troupe`` imports
nothing outward (``test_troupe_imports.py`` pins the empty set), and
``pipelines`` modules run inside worker and Blender processes where ``studio``
is not importable at all.

So the two hold the same table twice, and this file is the only place the
copies are checked against each other. This is
``tests/test_sprite_geometry_agreement.py``'s argument at its second instance;
if it fails, a character rendered with ``walk_left`` at cell 40 opens in Inker
tagged as something else -- silently, and only visible once somebody plays it.
"""

from __future__ import annotations

import pytest

from warlock.pipelines import charsheet as cs
from warlock.studio.troupe import spec as troupe_spec


@pytest.fixture(scope="module")
def spec():
    return troupe_spec.load()


def test_the_two_modules_name_the_same_animations_in_the_same_order(spec):
    assert [a[0] for a in cs.ANIMATIONS] == [a.name for a in spec.animations]


def test_the_two_modules_agree_on_every_frame_count_and_loop_flag(spec):
    assert [(a[1], a[2]) for a in cs.ANIMATIONS] == [
        (a.frames, a.loop) for a in spec.animations
    ]


def test_the_two_modules_agree_on_every_duration(spec):
    assert [a[3] for a in cs.ANIMATIONS] == [a.duration_ms for a in spec.animations]


def test_the_two_modules_name_the_same_directions_at_the_same_yaws(spec):
    assert list(cs.DIRECTIONS) == [(d.name, d.yaw) for d in spec.directions]


def test_the_two_modules_agree_on_the_grid_and_the_ladder(spec):
    assert spec.columns == cs.COLUMNS
    assert spec.sizes == cs.SIZES
    assert spec.render_size == cs.RENDER_SIZE
    assert cs.frames_per_direction() == spec.frames_per_direction


def test_cell_for_cell_the_two_say_the_same_thing(spec):
    """The one that matters: cell *i* of the atlas is timeline frame *i*, and
    both copies have to think it is the same animation, direction, yaw and
    per-direction frame number."""
    theirs = spec.cells()
    ours = cs.frame_table()
    assert len(ours) == len(theirs) == spec.cell_count
    for mine, other in zip(ours, theirs, strict=True):
        assert (mine.index, mine.animation, mine.direction, mine.yaw, mine.frame) == (
            other.index,
            other.animation,
            other.direction,
            other.yaw,
            other.frame,
        )


def test_the_two_modules_produce_identical_spans(spec):
    assert cs.spans() == spec.spans()


def test_a_tag_covers_exactly_its_span(spec):
    """The sidecar's tags and the studio's spans are one table read twice."""
    tags = cs.animation_block()["tags"]
    assert len(tags) == len(spec.spans())
    for tag, (animation, direction, start, end, loop) in zip(
        tags, spec.spans(), strict=True
    ):
        assert tag["name"] == f"{animation}_{direction}"
        assert (tag["start"], tag["end"], tag["loop"]) == (start, end, loop)
