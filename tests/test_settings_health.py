"""Settings -> Health: the surface that says *which* check failed.

The checks were readable in one place, a popup, and when that popup was removed
nothing inherited the list. A fatal check still reaches the error banner; a
non-fatal one -- Blender missing so rigging is unavailable, a style LoRA whose
file has been moved -- became a count on Home and a tooltip, and `warlock
doctor` in a terminal was the only way to read what it actually said.

These assert the wording rather than the drawing, which is why the rows are
built by a pure function: the half of the old popup that was never covered was
exactly this half, and it had to be read off a screenshot.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock.studio import icons, theme
from warlock.studio.panes import app_settings
from warlock.studio.state import AppState


def _check(name: str, ok: bool, *, fatal: bool = False, detail: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, ok=ok, fatal=fatal, detail=detail)


# --- the rows ---------------------------------------------------------------


def test_a_non_fatal_failure_names_itself_and_says_why():
    """The regression this page exists for. A warning that only ever rendered
    as "1 thing needs attention" is a number the reader cannot act on."""
    rows = app_settings.health_rows(
        [_check("Blender (rigging)", False, detail="bpy is not installed on Python 3.12")]
    )
    assert [row.name for row in rows] == ["Blender (rigging)"]
    assert rows[0].detail == "bpy is not installed on Python 3.12"
    assert not rows[0].ok


def test_the_three_states_are_three_colours():
    """Fatal and non-fatal are not the same red. A warning drawn in the error
    colour is an install that looks broken because Blender is absent."""
    rows = {
        row.name: row
        for row in app_settings.health_rows(
            [
                _check("trellis", True),
                _check("style lora", False),
                _check("trellis port", False, fatal=True),
            ]
        )
    }
    # By name rather than by position: the rows come back in failure-first
    # order, and this is about the three states rather than about the order.
    assert (rows["trellis"].colour, rows["trellis"].glyph) == (theme.OK, icons.CHECK)
    assert (rows["style lora"].colour, rows["style lora"].glyph) == (theme.WARN, icons.CIRCLE_X)
    assert (rows["trellis port"].colour, rows["trellis port"].glyph) == (theme.ERR, icons.CIRCLE_X)


def test_the_failures_come_first_and_the_fatal_one_leads():
    """S140's rule, for the same reason ``config_table`` has it: a healthy
    install runs past thirty checks, so a reader who arrived by clicking "2
    things need attention" was being asked to find two rows in a wall of green."""
    rows = app_settings.health_rows(
        [
            _check("passing one", True),
            _check("a warning", False),
            _check("passing two", True),
            _check("the fatal one", False, fatal=True),
        ]
    )
    assert [row.name for row in rows] == [
        "the fatal one",
        "a warning",
        "passing one",
        "passing two",
    ]


def test_a_passing_check_is_not_reordered_by_its_fatal_flag():
    """``fatal`` says what happens *if* a check fails, so passing rows carry it
    too. Sorting on it without ``ok`` first would shuffle the green band."""
    rows = app_settings.health_rows(
        [
            _check("trellis-server.exe", True, fatal=True),
            _check("free disk space", True),
            _check("job database", True, fatal=True),
        ]
    )
    assert [row.name for row in rows] == [
        "trellis-server.exe",
        "free disk space",
        "job database",
    ]


def test_a_check_missing_its_attributes_still_renders():
    """Also asked of the static checks a partially-started runtime carries, so
    a row is never the reason the page cannot draw."""
    rows = app_settings.health_rows([SimpleNamespace()])
    assert rows[0].name == "" and rows[0].detail == ""
    assert rows[0].colour == theme.WARN  # absent is not passing


# --- the summary line -------------------------------------------------------


def test_no_checks_yet_is_not_an_all_clear():
    """The first poll lands a moment after the window opens. An empty list
    reported as "everything checks out" is the one wrong answer available."""
    assert app_settings.health_summary([]) == "No checks have run yet."


def test_the_summary_counts_only_the_failures():
    rows = app_settings.health_rows(
        [_check("a", True), _check("b", False), _check("c", False, fatal=True)]
    )
    assert app_settings.health_summary(rows) == "2 of 3 need attention."


def test_all_passing_says_so_plainly():
    rows = app_settings.health_rows([_check("a", True), _check("b", True)])
    assert app_settings.health_summary(rows) == "Everything checks out."


# --- Copy details -----------------------------------------------------------


def test_the_report_marks_the_failures_for_pasting():
    """What goes in a bug report. The pass/fail word is first on the line so a
    reader skimming a pasted block sees the shape without reading the names."""
    rows = app_settings.health_rows(
        [_check("trellis", True, detail="1.2.0"), _check("bpy", False, detail="not installed")]
    )
    # Failure-first, as the list on screen is: what a reader pastes and what
    # they were looking at should not be two different orders.
    assert app_settings.health_report(rows) == "FAIL bpy: not installed\nok trellis: 1.2.0"


def test_an_empty_report_is_empty_rather_than_a_stray_newline():
    assert app_settings.health_report([]) == ""


# --- the dismissed banner (F59) ---------------------------------------------


def test_dismissing_an_error_moves_it_here_rather_than_deleting_it():
    """A worker that died is reported through ``state.errors`` and through no
    doctor row at all, so the banner holds the only copy. Dismiss moves it; for
    as long as nothing read the list it moved it out of existence."""
    state = AppState()
    state.note_error("the worker exited: MemoryError")
    state.dismiss_errors()
    assert state.errors == []
    assert state.dismissed_errors == ["the worker exited: MemoryError"]


def test_dismissing_twice_does_not_double_the_record():
    state = AppState()
    state.note_error("the same failure")
    state.dismiss_errors()
    state.note_error("the same failure")
    state.dismiss_errors()
    assert state.dismissed_errors == ["the same failure"]
