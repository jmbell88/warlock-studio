"""Which tilesets the picker's tab strip shows, and which it may never hide.

The picker was a combo: two clicks and a read to change tileset, on a control
reached dozens of times a minute while painting. It is a tab strip now, with a
filter box over it for the map that has a dozen sets -- and a filter over a
*selection* control has one rule that has to hold or it is worse than no filter
at all: it must never hide the thing you are painting with.

Hiding it would either leave the picker below the strip drawing a tileset that
no tab names, or -- worse -- hand the selection to whichever tab imgui chose
instead, changing the brush because somebody typed in a search box.
"""

from __future__ import annotations

from warlock.studio.plotter_state import visible_tilesets

NAMES = ["Grass", "Dungeon walls", "props", "GRASS interior"]


def test_no_filter_shows_every_tileset():
    assert visible_tilesets(NAMES, "", 0) == [0, 1, 2, 3]


def test_a_blank_query_is_the_same_as_no_filter():
    """``widgets.list_filter`` clears its query when the list is too short to be
    worth searching, so whitespace and an empty box must not hide rows."""
    assert visible_tilesets(NAMES, "   ", 2) == [0, 1, 2, 3]


def test_the_filter_matches_anywhere_in_the_name_and_ignores_case():
    assert visible_tilesets(NAMES, "grass", 0) == [0, 3]
    assert visible_tilesets(NAMES, "WALL", 1) == [1]
    assert visible_tilesets(NAMES, "op", 2) == [2]


def test_the_set_in_hand_is_never_filtered_out():
    """The rule the whole function exists for."""
    assert visible_tilesets(NAMES, "grass", 1) == [0, 1, 3]
    # Even when nothing at all matches, the strip still names what is selected.
    assert visible_tilesets(NAMES, "zzz", 2) == [2]


def test_the_kept_set_stays_in_index_order():
    """A tab strip that reordered itself under a filter would move the tab under
    the pointer between one keystroke and the next."""
    assert visible_tilesets(NAMES, "s", 1) == [0, 1, 2, 3]
    assert visible_tilesets(NAMES, "dungeon", 3) == [1, 3]


def test_an_empty_map_has_an_empty_strip():
    assert visible_tilesets([], "", 0) == []
    assert visible_tilesets([], "anything", 0) == []
