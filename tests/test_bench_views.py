"""The 8-view grid. Pure arithmetic -- no Blender, no GPU."""

from __future__ import annotations

from warlock.bench import views as views_mod


def test_a_view_plan_is_one_row_of_eight():
    plan = views_mod.view_plan()
    assert plan.rows == 1
    assert plan.columns == views_mod.VIEW_COUNT
    assert len(plan.cells) == views_mod.VIEW_COUNT
    assert plan.frame_size == views_mod.VIEW_SIZE


def test_the_views_are_evenly_spaced_around_the_subject():
    yaws = [c.yaw for c in views_mod.view_plan().cells]
    assert yaws[0] == 0.0
    assert yaws == sorted(yaws)
    assert all(abs((b - a) - 45.0) < 1e-6 for a, b in zip(yaws, yaws[1:], strict=False))


def test_column_zero_is_the_matched_view_by_construction():
    """The offset is added before Blender sees a cell, so column 0 *is* the
    camera-matched view and the other seven are its turntable."""
    cells = views_mod.worker_cells(views_mod.view_plan(), yaw_offset=137.0)
    assert cells[0]["yaw"] == 137.0
    assert cells[0]["column"] == 0
    assert cells[4]["yaw"] == (137.0 + 180.0) % 360.0


def test_the_offset_wraps_rather_than_exceeding_a_turn():
    cells = views_mod.worker_cells(views_mod.view_plan(), yaw_offset=350.0)
    assert all(0.0 <= c["yaw"] < 360.0 for c in cells)


def test_every_cell_carries_an_empty_pose():
    """An unrigged prop is the common case; the worker's pose calls are all
    guarded on `armature is not None`, and an absent bones key is not."""
    for cell in views_mod.worker_cells(views_mod.view_plan()):
        assert cell["bones"] == {}


def test_the_lighting_is_pinned_rather_than_configurable():
    """DINOv2 measures style as much as identity, so the numbers are only
    meaningful A-against-B -- and only while the lighting is identical across
    the two runs. A recipe field here would let them differ invisibly."""
    assert views_mod.VIEW_LIGHTING == "lit"
    assert views_mod.view_plan().lighting == views_mod.VIEW_LIGHTING
    assert "lighting" not in views_mod.view_plan.__annotations__


def test_the_camera_constants_are_declared_uncalibrated():
    """bench calibrate has not been run: nothing in this repo has established
    that TRELLIS aligns a reconstruction to its input camera."""
    from pathlib import Path

    assert "UNCALIBRATED" in Path(views_mod.__file__).read_text("utf-8")


def test_reference_view_is_none_when_nothing_was_rendered(tmp_path):
    assert views_mod.reference_view(tmp_path) is None


def test_view_paths_come_back_in_cell_order(tmp_path):
    for i in (3, 0, 10, 1):
        (tmp_path / f"{i:04d}.png").write_bytes(b"x")
    assert [p.stem for p in views_mod.view_paths(tmp_path)] == ["0000", "0001", "0003", "0010"]
    assert views_mod.reference_view(tmp_path).stem == "0000"
