"""A filtered-to-empty panel says so, in all six places that have a filter.

The report that filed this said ``widgets.list_filter`` was "the one place to
fix all five". It was wrong twice: there are six lists, not five, and the helper
cannot fix any of them on its own -- every caller owns its loop and its own idea
of what a row is, so only the caller can count what survived the filter. Hence a
second helper that takes the count, and a source scan (the
``test_notifications`` empty-state idiom) that none of the six quietly stops
calling it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from warlock.studio import widgets

STUDIO = Path(inspect.getfile(widgets)).resolve().parent

#: Every pane drawing a ``list_filter``. ``pose_panel`` holds two of them --
#: the library list and the saved list -- which is why this is a file list and
#: the count assertion below is separate.
FILTERED = (
    "main.py",  # Review's sweep list
    "panes/clay_outliner.py",
    # The layers panel is deleted and its filter moved with the list it
    # filtered -- the timeline's track column (W2.5a).
    "panes/inker_timeline.py",
    "panes/pose_panel.py",
    "panes/poser_library.py",
    # The one list in the app that grows without bound: an atlas's sources are
    # whatever has been dropped on it, which is hundreds by the end of a sheet.
    # Packwright's *items* pane is deliberately not here -- it is a result list
    # -- and neither is Troupe's cast, which is below list_filter's own
    # self-hiding threshold.
    "panes/packwright_sources.py",
    # Plotter's tileset tab strip (W4.4). The one filtered list whose rows are
    # *tabs*: the count passed to ``no_matches`` deliberately excludes the set
    # in hand, which is never filtered out, so a query matching nothing still
    # says so with one tab still on screen.
    "panes/plotter_tileset.py",
    # Plotter's Objects dock. The list that most needs one: a map's objects are
    # whatever has been placed on it, which is hundreds by the end of a level,
    # and "where is the door I named" is the question the dock exists for.
    "panes/plotter_objects.py",
)


@pytest.mark.parametrize("relative", FILTERED)
def test_every_filtered_list_reports_a_filter_that_matched_nothing(relative):
    assert "no_matches(" in (STUDIO / relative).read_text(encoding="utf-8"), relative


def test_every_list_filter_call_site_is_listed_above():
    """The other direction: a filtered list added without the row would be
    exactly the panel-looks-lost bug again, in a new pane."""
    found = {
        str(path.relative_to(STUDIO)).replace("\\", "/")
        for path in STUDIO.rglob("*.py")
        if "list_filter(" in path.read_text(encoding="utf-8")
    }
    assert found - {"widgets.py"} == set(FILTERED)


def test_the_two_pose_lists_each_got_their_own_row():
    """``pose_panel`` filters twice -- library poses and saved poses -- and a
    single call would leave one of them still drawing nothing."""
    source = (STUDIO / "panes" / "pose_panel.py").read_text(encoding="utf-8")
    assert source.count("widgets.list_filter(") == 2
    assert source.count("widgets.no_matches(") == 2


def test_the_row_is_drawn_only_for_a_filter_that_matched_nothing(monkeypatch):
    """The gate, directly: an unfiltered empty list is the ``empty_state``
    case and must not get this row instead."""
    drawn: list[str] = []
    monkeypatch.setattr(widgets, "muted", drawn.append)

    widgets.no_matches("", 0)  # nothing typed: the list is simply empty
    widgets.no_matches("", 4)
    widgets.no_matches("hip", 2)  # the filter matched
    assert drawn == []

    widgets.no_matches("hip", 0)
    assert drawn == ["Nothing matches the filter."]
