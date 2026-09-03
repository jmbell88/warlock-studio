"""The animation scores: pure numpy, synthetic atlases, no window.

Each test builds the sheet that would produce exactly one kind of complaint
and checks the score says that and nothing else -- because a scorer that
flags a good sheet is one the user learns to ignore.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.troupe import qa

CELL = 16
COLUMNS = 8


def _layout(movements):
    """``movements``: ``[(key, frames, directions, loop)]`` -> a layout dict of
    the shape ``troupe_mode.preview_layout`` returns, plus dense runs."""
    out_movements, runs, start = [], [], 0
    for key, frames, directions, loop in movements:
        out_movements.append(
            {
                "key": key,
                "frames": frames,
                "loop": loop,
                "directions": [{"key": d} for d in directions],
            }
        )
        for direction in directions:
            runs.append(
                {"movement": key, "direction": direction, "start": start, "end": start + frames - 1}
            )
            start += frames
    return {"version": 2, "movements": out_movements, "runs": runs, "cell_count": start}


def _atlas(cell_count: int, painter):
    rows = (cell_count + COLUMNS - 1) // COLUMNS
    atlas = np.zeros((rows * CELL, COLUMNS * CELL, 4), dtype=np.uint8)
    for index in range(cell_count):
        x0, y0, x1, y1 = qa.cell_box(index, COLUMNS, CELL, CELL)
        painter(index, atlas[y0:y1, x0:x1])
    return atlas


def _figure(crop, *, shift=0, colour=(40, 60, 200, 255)):
    crop[2:6, 6 + shift : 10 + shift] = (200, 150, 120, 255)
    crop[6:13, 7 + shift : 9 + shift] = colour
    crop[7:9, 3 + shift : 13 + shift] = colour


def _score(layout, painter):
    atlas = _atlas(int(layout["cell_count"]), painter)
    return qa.score_sheet(atlas, layout, columns=COLUMNS, frame_w=CELL, frame_h=CELL)


def test_a_static_sheet_scores_zero_everywhere():
    layout = _layout([("walk", 4, ("front", "left", "back", "right"), True)])
    score = _score(layout, lambda _i, crop: _figure(crop))
    assert len(score.cells) == 16
    assert score.worst is None
    assert score.flagged == 0
    for cell in score.cells:
        assert cell.flags == ()
        assert all(value == 0.0 for value in cell.metrics.values())
        assert qa.level(cell) == qa.LEVEL_OK


def test_a_blank_frame_is_flagged_and_so_is_the_frame_after_it():
    layout = _layout([("walk", 4, ("front",), True)])

    def paint(index, crop):
        if index != 2:
            _figure(crop)

    score = _score(layout, paint)
    by = score.lookup()
    assert "blank" in by[("walk", "front", 2)].flags
    assert "shape" in by[("walk", "front", 2)].flags
    assert "shape" in by[("walk", "front", 3)].flags
    assert by[("walk", "front", 1)].flags == ()
    assert qa.level(by[("walk", "front", 2)]) == qa.LEVEL_BAD
    assert score.worst is not None and score.worst[2] == 2


def test_a_mirrored_direction_has_no_drift():
    layout = _layout([("walk", 3, ("left", "right"), True)])

    def paint(index, crop):
        _figure(crop, shift=1)
        if index >= 3:
            crop[:] = crop[:, ::-1]

    score = _score(layout, paint)
    for cell in score.cells:
        assert cell.metrics["drift_w"] == 0.0
        assert cell.metrics["drift_h"] == 0.0
        assert cell.metrics["drift_occupancy"] == 0.0


def test_a_direction_that_grew_drifts_from_the_others():
    layout = _layout([("walk", 1, ("front", "left", "back", "right"), True)])

    def paint(index, crop):
        _figure(crop)
        if index == 1:
            crop[13:16, 2:14] = (40, 60, 200, 255)  # a much taller, wider sprite

    score = _score(layout, paint)
    by = score.lookup()
    assert "drift" in by[("walk", "left", 0)].flags
    assert by[("walk", "front", 0)].metrics["drift_h"] == 0.0


def test_the_loop_seam_is_measured_only_on_a_cycle():
    def paint(index, crop):
        _figure(crop, shift=4 if index == 3 else 0)

    cyclic = _score(_layout([("walk", 4, ("front",), True)]), paint)
    first = cyclic.lookup()[("walk", "front", 0)]
    assert first.metrics["seam_delta"] > 0.5
    assert "seam" in first.flags

    one_shot = _score(_layout([("attack", 4, ("front",), False)]), paint)
    first = one_shot.lookup()[("attack", "front", 0)]
    assert "seam_delta" not in first.metrics
    assert "seam" not in first.flags


def test_a_recoloured_frame_flickers():
    layout = _layout([("walk", 3, ("front",), True)])

    def paint(index, crop):
        _figure(crop, colour=(200, 40, 40, 255) if index == 1 else (40, 60, 200, 255))

    score = _score(layout, paint)
    by = score.lookup()
    assert "flicker" in by[("walk", "front", 1)].flags
    assert by[("walk", "front", 1)].metrics["shape_delta"] == 0.0
    assert by[("walk", "front", 0)].flags == ()


def test_a_shifted_frame_jitters_but_keeps_its_shape_metrics_honest():
    layout = _layout([("walk", 2, ("front",), True)])

    def paint(index, crop):
        _figure(crop, shift=3 if index == 1 else 0)

    score = _score(layout, paint)
    moved = score.lookup()[("walk", "front", 1)]
    assert moved.metrics["centroid_jitter"] > 0.1
    assert "jitter" in moved.flags
    assert moved.metrics["foot_jitter"] == 0.0


def test_a_run_past_the_atlas_is_skipped_rather_than_raised_on():
    layout = _layout([("walk", 4, ("front",), True), ("run", 4, ("front",), True)])
    layout["runs"].append({"movement": "cast", "direction": "front", "start": 800, "end": 803})
    score = _score(layout, lambda _i, crop: _figure(crop))
    assert {c.animation for c in score.cells} == {"walk", "run"}


def test_scoring_is_deterministic():
    layout = _layout([("walk", 4, ("front", "left"), True)])
    rng = np.random.default_rng(7)

    def paint(index, crop):
        _figure(crop, shift=int(rng.integers(0, 3)))

    atlas = _atlas(8, paint)
    a = qa.score_sheet(atlas, layout, columns=COLUMNS, frame_w=CELL, frame_h=CELL)
    b = qa.score_sheet(atlas, layout, columns=COLUMNS, frame_w=CELL, frame_h=CELL)
    assert a == b


def test_every_metric_has_a_threshold_and_a_flag():
    assert set(qa.THRESHOLDS) == set(qa.METRICS)
    assert set(qa._FLAG_OF) == set(qa.METRICS)
    assert set(qa._FLAG_OF.values()) | {"blank"} == set(qa.FLAGS)
    for warn, bad in qa.THRESHOLDS.values():
        assert 0 < warn < bad
