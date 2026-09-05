"""One asset, one vocabulary: the inspector describes a job the way Home and
the Library do, and keeps the raw metadata one foldout down.

The 2026-09-05 consistency review found the inspector printing
``15009c54aa81 - image - model`` over ``1788587399.0856822`` for the row Home
had just captioned "just now".
"""

from __future__ import annotations

import inspect

from warlock.studio import widgets
from warlock.studio.panes import inspector, landing

JOB = {
    "id": "15009c54aa81",
    "kind": "image",
    "stage": "model",
    "status": "done",
    "created_at": 1788587399.0856822,
}


def test_home_and_widgets_share_one_clock():
    assert landing.ago is widgets.ago


def test_the_when_line_is_relative_not_a_float():
    when, _ = widgets.asset_summary_lines(JOB, now=JOB["created_at"] + 30)
    assert when == "just now"
    assert "1788587399" not in when


def test_the_id_and_raw_stamp_are_in_the_details_not_the_headline():
    when, details = widgets.asset_summary_lines(JOB, now=JOB["created_at"] + 90000)
    assert when == "yesterday"
    assert details[0] == "id: 15009c54aa81"
    assert "kind: image" in details and "stage: model" in details
    assert any(line.startswith("created: ") and ":" in line[9:] for line in details)
    assert not any("1788587399.0856822" in line for line in details)


def test_an_unstamped_job_has_no_when_line_and_no_created_detail():
    when, details = widgets.asset_summary_lines({"id": "x", "stage": "reference"})
    assert when == ""
    assert not any(line.startswith("created") for line in details)


def test_the_inspector_no_longer_hand_rolls_its_metadata_line():
    source = inspect.getsource(inspector._meta)
    assert "widgets.asset_summary(job)" in source
    assert "created_at" not in source
    assert "job.get('kind')" not in source
