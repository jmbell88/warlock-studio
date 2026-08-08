"""The inspector's reference image, its watertight line, and the two
"open this somewhere I can edit it" affordances.

All three are pure functions for the reason ``seam_verdict`` is one: the sizing
arithmetic, the wording and the gates are the feature, and none of them should
need a GL context to assert.
"""

from __future__ import annotations

from warlock.studio.panes import inspector
from warlock.studio.tokens import sp

# --- 9d: the reference fills the pane, and keeps its shape ------------------


def test_a_wide_reference_fills_the_width_and_keeps_its_aspect():
    # The bug: imgui.image(..., (THUMB_SIZE, THUMB_SIZE)) drew every reference
    # as a 96-square whatever its shape, so a 2:1 sheet came out squashed.
    width, height = inspector.reference_fit((1024, 512), 290.0)

    assert width == 290.0
    assert abs(width / height - 2.0) < 1e-6


def test_a_tall_reference_is_capped_by_height_not_by_width():
    # Without the cap a 1:4 portrait at full pane width is over a thousand
    # pixels tall and pushes every section under it off the screen.
    width, height = inspector.reference_fit((512, 2048), 290.0)

    assert height == inspector.reference_max_height()
    assert width < 290.0
    assert abs(height / width - 4.0) < 1e-6


def test_the_reference_never_returns_a_degenerate_box():
    # A zero in either axis is a texture we should still be able to draw a
    # placeholder for rather than an imgui image of size zero.
    width, height = inspector.reference_fit((0, 0), 290.0)

    assert width > 0 and height > 0


def test_the_thumbnail_constant_is_scaled_like_every_other_size():
    # Raw design px in a scaled UI is the bug the library already fixed by
    # wrapping its own use in sp().
    assert inspector.reference_max_height() == sp(inspector.THUMB_SIZE * 3)


# --- the carried-over bug: the panel must not contradict the badge ----------


def _welded(**kw):
    return {"triangles": 100, **kw}


def test_the_watertight_line_prefers_the_welded_flag():
    # Package 0c moved the badge onto welded_watertight -- measured on a welded
    # copy, because meshreport loads with process=False and every xatlas UV
    # seam split reads as a boundary edge. The panel kept printing the raw
    # flag, so it said "watertight: False" directly under a badge saying
    # "watertight".
    report = _welded(watertight=False, welded_watertight=True)

    assert inspector.watertight_value(report) is True
    assert "True" in (inspector.watertight_line(report) or "")


def test_a_row_recorded_before_the_change_still_reads_its_raw_flag():
    report = _welded(watertight=True)

    assert inspector.watertight_value(report) is True
    assert inspector.watertight_line(report) == "watertight: True"


def test_the_welded_line_says_so_and_the_legacy_one_does_not():
    # The label carries which measurement it is: claiming "welded" over a row
    # that only ever had the raw flag would be the same overclaim in reverse.
    assert "weld" in (inspector.watertight_line(_welded(welded_watertight=False)) or "")
    assert "weld" not in (inspector.watertight_line(_welded(watertight=False)) or "")


def test_a_report_with_no_watertight_reading_says_nothing():
    assert inspector.watertight_value(_welded()) is None
    assert inspector.watertight_line(_welded()) is None
    assert inspector.watertight_line(None) is None


def test_the_inspector_and_the_badge_cannot_disagree_about_watertightness():
    """Both halves read the same expression, over the whole matrix.

    ``vectors.py`` writes it as ``report.get("welded_watertight",
    report.get("watertight"))`` and ``widgets.quality_badge`` keys the badge on
    ``welded_watertight``; this pins the panel to the same answer for every
    combination either flag can be in.
    """
    import inspect

    from warlock.studio import widgets

    # If the badge ever stops keying on the welded flag, this pin is the place
    # the two halves are reconciled rather than left to drift apart again.
    assert "welded_watertight" in inspect.getsource(widgets.quality_badge)
    for welded in (True, False, None):
        for raw in (True, False, None):
            report = _welded()
            if welded is not None:
                report["welded_watertight"] = welded
            if raw is not None:
                report["watertight"] = raw
            expected = report.get("welded_watertight", report.get("watertight"))
            assert inspector.watertight_value(report) == expected


# --- 9e/9f: the post-generation affordances ---------------------------------


def test_a_finished_mesh_can_be_opened_in_clay():
    job = {"id": "0123456789ab", "stage": "model", "status": "done", "files": ["model.glb"]}

    assert inspector.can_edit_in_clay(job) is True


def test_a_job_with_no_mesh_cannot_be_opened_in_clay():
    # The same gate the library's overflow menu applies: build.wblk is never
    # listed, so model.glb is what says a mesh exists to import.
    assert inspector.can_edit_in_clay({"files": ["input.png"], "status": "done"}) is False
    assert inspector.can_edit_in_clay({"files": ["model.glb"], "status": "running"}) is False
    assert inspector.can_edit_in_clay(None) is False
