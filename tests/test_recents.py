"""The one merged most-recently-used list.

Four document modes each kept their own ``recent: list[str]``; merging them at
draw time is impossible rather than untidy, because four bare path lists carry
an order within themselves and none between them. These tests are about the
ordering that replaced them, and about the migration off the four old keys.
"""

from __future__ import annotations

from typing import Any

from warlock.studio import recents


class FakeSettings:
    """The two methods :mod:`.recents` uses, and nothing else."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = dict(data or {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


def test_remember_puts_the_newest_first_across_kinds():
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "clay", "a.wblk", when=100.0)
    recents.remember(s, "inker", "b.ora", when=200.0)
    recents.remember(s, "plotter", "c.wmap", when=150.0)
    assert [(e.kind, e.path) for e in recents.entries(s)] == [
        ("inker", "b.ora"),
        ("plotter", "c.wmap"),
        ("clay", "a.wblk"),
    ]


def test_a_kind_filter_is_a_slice_of_the_same_order():
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "clay", "a.wblk", when=100.0)
    recents.remember(s, "inker", "b.ora", when=200.0)
    recents.remember(s, "clay", "c.wblk", when=300.0)
    assert recents.paths(s, "clay") == ["c.wblk", "a.wblk"]


def test_reopening_a_file_moves_it_rather_than_duplicating_it():
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "clay", "a.wblk", when=100.0)
    recents.remember(s, "clay", "b.wblk", when=200.0)
    recents.remember(s, "clay", "a.wblk", when=300.0)
    assert recents.paths(s, "clay") == ["a.wblk", "b.wblk"]


def test_the_same_path_under_two_kinds_is_two_rows():
    """Dedupe is on ``(kind, path)``: activating them does two different
    things, so they are two entries however alike the strings look."""
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "clay", "shared.dat", when=100.0)
    recents.remember(s, "plotter", "shared.dat", when=200.0)
    assert len(recents.entries(s)) == 2


def test_the_bound_is_per_kind_so_one_busy_mode_cannot_evict_the_others():
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "inker", "keep.ora", when=1.0)
    for index in range(recents.MAX_RECENT + 5):
        recents.remember(s, "clay", f"f{index}.wblk", when=100.0 + index)
    assert len(recents.paths(s, "clay")) == recents.MAX_RECENT
    assert recents.paths(s, "inker") == ["keep.ora"]


def test_forget_drops_a_path_that_no_longer_opens():
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "clay", "gone.wblk", when=100.0)
    recents.remember(s, "clay", "here.wblk", when=200.0)
    recents.forget(s, "clay", "gone.wblk")
    assert recents.paths(s, "clay") == ["here.wblk"]
    # A path that was never there is not an error and writes nothing.
    recents.forget(s, "clay", "never.wblk")
    assert recents.paths(s, "clay") == ["here.wblk"]


def test_none_is_a_no_op_rather_than_a_row():
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "clay", None)
    recents.forget(s, "clay", None)
    assert recents.entries(s) == []


def test_an_unknown_kind_is_refused_rather_than_stored():
    """The kinds are the four document modes. A row under a fifth would be a
    Resume entry Home has no way to open."""
    s = FakeSettings({recents.SETTING: []})
    recents.remember(s, "3d", "job", when=1.0)
    assert recents.entries(s) == []


def test_unstamped_rows_sort_last_in_their_stored_order():
    """``None`` is not a time at one end of a scale: filing the unstamped rows
    under "newest" would be a wrong answer rather than a partial one, and the
    order they *do* have is the one they were stored in."""
    s = FakeSettings(
        {
            recents.SETTING: [
                {"kind": "clay", "path": "old-a"},
                {"kind": "clay", "path": "old-b"},
                {"kind": "clay", "path": "stamped", "when": 50.0},
            ]
        }
    )
    assert recents.paths(s, "clay") == ["stamped", "old-a", "old-b"]


def test_it_migrates_the_four_legacy_per_mode_lists_once():
    s = FakeSettings(
        {
            "inker": {"recent": ["a.ora", "b.ora"], "swatches": []},
            "clay": {"recent": ["c.wblk"]},
            "plotter": {"recent": ["d.wmap"]},
            "packwright": {"recent": ["e.wpack"]},
        }
    )
    found = recents.entries(s)
    assert {(e.kind, e.path) for e in found} == {
        ("inker", "a.ora"),
        ("inker", "b.ora"),
        ("clay", "c.wblk"),
        ("plotter", "d.wmap"),
        ("packwright", "e.wpack"),
    }
    # Each kind keeps the MRU order it was stored in -- the only order those
    # rows have, since none of them carries a clock.
    assert recents.paths(s, "inker") == ["a.ora", "b.ora"]
    # Written once: the migrated list is what answers from now on.
    assert recents.SETTING in s.data
    s.data["inker"] = {"recent": ["never-read.ora"]}
    assert "never-read.ora" not in recents.paths(s, "inker")


def test_the_legacy_keys_are_left_alone_so_an_older_build_still_reads_them():
    s = FakeSettings({"clay": {"recent": ["c.wblk"]}})
    recents.entries(s)
    assert s.data["clay"] == {"recent": ["c.wblk"]}


def test_a_malformed_stored_row_is_skipped_rather_than_fatal():
    s = FakeSettings(
        {
            recents.SETTING: [
                "not a dict",
                {"kind": "clay"},
                {"path": "no-kind"},
                {"kind": "clay", "path": ""},
                {"kind": "clay", "path": "good", "when": "not a number"},
            ]
        }
    )
    found = recents.entries(s)
    assert [(e.kind, e.path, e.when) for e in found] == [("clay", "good", None)]


def test_it_imports_nothing_but_stdlib():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio" / "recents.py"
    ).read_text(encoding="utf-8")
    for banned in ("imgui", "moderngl", "pygame", "from ..service", "from . import"):
        assert banned not in source
