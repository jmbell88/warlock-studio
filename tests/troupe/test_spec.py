"""The Troupe frame table, asserted rather than assumed."""

from __future__ import annotations

import pytest

from warlock.studio.troupe import spec as troupe_spec


@pytest.fixture(scope="module")
def spec():
    return troupe_spec.load()


def test_the_five_animations_are_the_ones_the_program_committed_to(spec):
    assert [a.name for a in spec.animations] == ["idle", "walk", "run", "attack", "jump"]


def test_the_frame_counts_and_loop_flags_are_the_frame_table(spec):
    assert [(a.name, a.frames, a.loop) for a in spec.animations] == [
        ("idle", 4, True),
        ("walk", 8, True),
        ("run", 8, True),
        ("attack", 6, False),
        ("jump", 6, False),
    ]


def test_thirty_two_frames_per_direction_and_two_hundred_and_fifty_six_cells(spec):
    assert spec.frames_per_direction == 32
    assert len(spec.directions) == 8
    assert spec.cell_count == 256


def test_the_eight_directions_are_evenly_spaced_and_agree_with_the_four(spec):
    """The 8-direction table must not contradict the shipped 4-direction one:
    front/left/right/back keep the yaws ``spritesynth`` and ``inker.animation``
    already give them, or a Troupe sheet and a spritesynth sheet would mean
    different things by "left"."""
    yaws = {d.name: d.yaw for d in spec.directions}
    assert [d.yaw for d in spec.directions] == [i * 45.0 for i in range(8)]
    assert {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}.items() <= (
        yaws.items()
    )


def test_the_atlas_fits_in_one_texture_at_every_rung(spec):
    from warlock.pipelines import sheet

    for size in spec.sizes:
        w, h = spec.atlas_size(size)
        assert max(w, h) <= sheet.MAX_ATLAS_PX, (size, w, h)


def test_at_128px_the_atlas_is_the_one_the_plan_measured(spec):
    assert spec.rows == 32
    assert spec.atlas_size(128) == (1024, 4096)


def test_only_the_power_of_two_rungs_divide_the_render_exactly(spec):
    assert spec.render_size == 512
    assert spec.exact_sizes() == (16, 32, 64, 128)


def test_every_cell_is_placed_once_and_the_grid_is_dense(spec):
    cells = spec.cells()
    assert len(cells) == spec.cell_count
    assert [c.index for c in cells] == list(range(len(cells)))
    assert {(c.row, c.column) for c in cells} == {
        (i // spec.columns, i % spec.columns) for i in range(len(cells))
    }


def test_every_animation_direction_run_is_contiguous(spec):
    """The property Inker tags depend on: a tag is a span, so the frames of one
    (animation, direction) must be consecutive cell indices."""
    cells = spec.cells()
    for animation, direction, start, end, loop in spec.spans():
        run = cells[start : end + 1]
        assert {c.animation for c in run} == {animation}
        assert {c.direction for c in run} == {direction}
        assert [c.frame for c in run] == list(range(len(run)))
        assert loop is spec.animation(animation).loop


def test_the_spans_tile_the_whole_sheet_with_no_gap_or_overlap(spec):
    covered: list[int] = []
    for _, _, start, end, _ in spec.spans():
        covered.extend(range(start, end + 1))
    assert covered == list(range(spec.cell_count))


def test_the_loader_is_cached_so_a_plan_does_not_re_read_the_table():
    assert troupe_spec.load() is troupe_spec.load()


def test_a_table_from_a_future_version_is_refused_by_name():
    with pytest.raises(ValueError, match="version 99"):
        troupe_spec._parse({"version": 99})


@pytest.mark.parametrize(
    "mangle, message",
    [
        (lambda p: p.update(animations=[]), "no animations"),
        (lambda p: p.update(directions=[]), "at least one direction"),
        (lambda p: p["animations"][0].update(frames=0), "has no frames"),
        (lambda p: p["animations"][0].update(duration_ms=0), "has no duration"),
        (
            lambda p: p["animations"].append(dict(p["animations"][0])),
            "share a name",
        ),
    ],
)
def test_a_self_contradictory_table_is_refused(mangle, message):
    import copy
    import json
    from pathlib import Path

    payload = json.loads(
        Path(troupe_spec._DATA).read_text(encoding="utf-8")
    )
    payload = copy.deepcopy(payload)
    mangle(payload)
    with pytest.raises(ValueError, match=message):
        troupe_spec._parse(payload)
