"""Section H: toast levels, the history, and the sweep summary.

The rules worth pinning are the ones a later edit would quietly undo: that an
unknown level degrades rather than raises, that the history outlives the toast,
and that a sweep raises exactly one notice however many units it has.
"""

from __future__ import annotations

import inspect
import time
from types import SimpleNamespace

import pytest

from warlock.studio import jobs_cache, widgets
from warlock.studio.state import TOAST_LEVELS, TOAST_LOG_MAX, TOAST_STICKY, AppState

# --- H68: levels -------------------------------------------------------------


def test_the_four_levels_are_ordered_by_reading_time():
    """The ladder is how long the message takes to read and act on, not a
    severity score -- which is why success and info share a dwell."""
    assert TOAST_LEVELS["info"] == TOAST_LEVELS["success"]
    assert TOAST_LEVELS["info"] < TOAST_LEVELS["warn"] < TOAST_LEVELS["error"]


def test_an_unknown_level_is_an_ordinary_notice():
    """The same rule ``action`` follows, and what lets a new level be raised
    from the calling side before this file has heard of it."""
    state = AppState()
    state.toast("something", "kremlin")
    assert state.toasts[0].ttl == TOAST_LEVELS["info"]
    colour, glyph = widgets.toast_style("kremlin")
    assert glyph == ""


def test_every_level_has_a_style_and_only_the_sticky_ones_can_be_closed():
    for level in TOAST_LEVELS:
        colour, glyph = widgets.toast_style(level)
        assert isinstance(colour, int)
        # info is the plain one and deliberately wears no glyph.
        assert (glyph != "") == (level != "info")
    assert {"warn", "error"} == TOAST_STICKY


def test_no_toast_glyph_needs_a_font_the_atlas_does_not_have():
    """The icon range is a pinned lucide subset; a codepoint outside it renders
    as the missing-glyph box."""
    from warlock.studio import icons

    known = {value for name, value in vars(icons).items() if name.isupper()}
    for level in TOAST_LEVELS:
        _colour, glyph = widgets.toast_style(level)
        assert glyph == "" or glyph in known


# --- H67: the history --------------------------------------------------------


def test_the_history_outlives_the_toast():
    """A toast is the app's only channel for "that worked", and it is gone in
    four seconds -- which for anything that happens while the user is looking
    elsewhere means it never happened."""
    state = AppState()
    state.toast("first", "success")
    state.toasts[0].born -= 100  # long expired
    state.expire_toasts()
    assert state.toasts == []
    assert [entry.text for entry in state.toast_log] == ["first"]


def test_the_history_is_newest_first_and_capped():
    state = AppState()
    for index in range(TOAST_LOG_MAX + 10):
        state.toast(f"m{index}")
    assert len(state.toast_log) == TOAST_LOG_MAX
    assert state.toast_log[0].text == f"m{TOAST_LOG_MAX + 9}"


def test_the_history_keeps_the_level_not_just_the_text():
    """A list of strings could not say whether a message was a success or a
    refusal, which is most of what the reader is checking."""
    state = AppState()
    state.toast("bad", "error")
    assert state.toast_log[0].level == "error"


def test_the_history_is_never_persisted():
    """It is a record of *this* run; one restored from disk would be a list of
    things that are no longer true."""
    from warlock.studio import main

    assert 'settings.set("toast_log"' not in inspect.getsource(main)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(0.0, "0s"), (12.4, "12s"), (59.9, "59s"), (60, "1m"), (3599, "59m"), (7200, "2h")],
)
def test_the_history_stamps_are_relative_and_coarse(seconds, text):
    from warlock.studio.main import _ago

    assert _ago(seconds) == text


# --- H69: the visible cap ----------------------------------------------------


def test_toasts_over_the_cap_are_counted_not_dropped():
    state = AppState()
    for index in range(widgets.TOAST_VISIBLE + 3):
        state.toast(f"m{index}")
    assert len(state.toasts) == widgets.TOAST_VISIBLE + 3


def test_hovering_pauses_the_clock_rather_than_recording_a_pause():
    """Age is read from ``born`` in four places; a second field saying "but not
    that age" is how two of them come to disagree."""
    source = inspect.getsource(widgets.toasts)
    assert "toast.born += delta" in source
    assert "is_window_hovered" in source


# --- N108 / N109: the actions ------------------------------------------------


def test_every_action_name_has_a_label():
    """A name with no entry draws nothing, so an action the UI has not learned
    degrades to a plain toast rather than to an unlabelled button."""
    assert set(widgets.TOAST_ACTIONS) >= {"log", "show", "review"}
    assert all(widgets.TOAST_ACTIONS.values())


def test_the_action_argument_is_a_field_and_not_spliced_into_the_name():
    """``show:abc123`` would make every such toast's action unrecognised, and
    therefore unlabelled and undrawn."""
    state = AppState()
    state.toast("done", "success", action="show", action_arg="job-1")
    entry = state.toasts[0]
    assert entry.action == "show" and entry.action_arg == "job-1"
    assert entry.action in widgets.TOAST_ACTIONS


def _unit(job_id: str, status: str, sweep: str = "s1") -> dict:
    return {"id": job_id, "status": status, "sweep_id": sweep}


def test_a_sweep_says_nothing_until_every_loaded_unit_has_finished():
    jobs = [_unit("a", "done"), _unit("b", "running")]
    assert jobs_cache.sweep_summary(jobs, "s1") is None


def test_a_finished_sweep_is_one_notice_with_the_tally():
    jobs = [_unit("a", "done"), _unit("b", "done"), _unit("c", "error")]
    text, level = jobs_cache.sweep_summary(jobs, "s1")
    assert text == "Sweep finished - 2 done, 1 refused."
    assert level == "success"


def test_a_sweep_that_produced_nothing_is_a_warning():
    jobs = [_unit("a", "error"), _unit("b", "error")]
    text, level = jobs_cache.sweep_summary(jobs, "s1")
    assert level == "warn"
    assert "2 refused" in text


def test_a_sweep_counts_cancels_separately():
    jobs = [_unit("a", "done"), _unit("b", "cancelled")]
    text, _level = jobs_cache.sweep_summary(jobs, "s1")
    assert text == "Sweep finished - 1 done, 1 cancelled."


def test_a_sweep_summary_ignores_other_sweeps_and_ordinary_jobs():
    jobs = [
        _unit("a", "done"),
        _unit("b", "running", sweep="s2"),
        {"id": "c", "status": "queued", "sweep_id": None},
    ]
    assert jobs_cache.sweep_summary(jobs, "s1") == ("Sweep finished - 1 done.", "success")


def test_an_unknown_sweep_says_nothing():
    assert jobs_cache.sweep_summary([], "s1") is None


def test_the_announce_path_routes_a_sweep_unit_to_the_summary():
    """One toast per sweep, not one per unit -- a twenty-unit sweep otherwise
    raises twenty notices and buries the one that matters."""
    from warlock.studio import main

    source = inspect.getsource(main.App._refresh)
    assert "sweep_summary" in source
    assert source.index("sweep_summary") < source.index("transition_message(job, previous)")


# --- H70 / H71: drops --------------------------------------------------------


def test_the_drop_flash_fades_and_belongs_to_one_slot():
    state = SimpleNamespace(drop_flash_slot="2d-ref", drop_flash_at=time.monotonic())
    assert widgets.drop_flash(state, "2d-ref") > 0.5
    assert widgets.drop_flash(state, "3d-source") == 0.0
    state.drop_flash_at = time.monotonic() - widgets.DROP_FLASH_SECONDS - 1
    assert widgets.drop_flash(state, "2d-ref") == 0.0


def test_a_never_dropped_on_slot_is_not_outlined():
    assert widgets.drop_flash(AppState(), "2d-ref") == 0.0


def test_a_drop_opens_the_section_it_landed_in():
    """In 2D the reference block is collapsed by default, so the drop used to
    be accepted with nothing on screen moving at all."""
    from warlock.studio import main

    source = inspect.getsource(main.App._on_drop)
    assert 'widgets.request_open("2d/reference")' in source
    assert '_flash_drop("2d-ref")' in source


def test_the_wrong_file_refusal_says_what_this_mode_would_have_done():
    """One sentence for both modes was wrong in 2D, where a dropped image is a
    conditioning reference and never a mesh."""
    from warlock.studio import main

    source = inspect.getsource(main.App._on_drop)
    body = source.split("DROPPABLE_IMAGES", 1)[1]
    assert "condition this generation on it" in body
    assert "start a mesh from it" in body


def test_the_accepted_suffixes_are_stated_once():
    """The refusal message and every accept path have to agree about what a
    drop may be.

    Asserted as "no route spells the suffixes out" rather than as a mention
    count: five modes now accept a drop and each says what one would have done
    *there*, so the constant is legitimately read several times. What must not
    happen is one of them growing its own list, which is how ``.webp`` comes to
    be droppable in three modes and refused in the fourth.
    """
    from warlock.studio import main

    assert ".webp" in main.DROPPABLE_IMAGES
    source = inspect.getsource(main.App._on_drop)
    assert "DROPPABLE_IMAGES" in source
    for suffix in main.DROPPABLE_IMAGES:
        assert f'"{suffix}"' not in source, f"{suffix} is spelled out in _on_drop"


# --- H72 / H73 / H74: the empty states ---------------------------------------


def test_a_missing_thumbnail_says_what_kind_of_thing_is_coming():
    from warlock.studio import icons
    from warlock.studio.panes import library

    assert library.thumb_glyph({"kind": "text", "stage": "model"}) == icons.BOX
    assert library.thumb_glyph({"kind": "text", "stage": "tile"}) == icons.GRID
    assert library.thumb_glyph({"kind": "rig", "stage": "rig"}) == icons.BONE
    # A failed card is about the failure, not about what it would have been.
    assert (
        library.thumb_glyph({"kind": "text", "stage": "model", "status": "error"})
        == icons.CIRCLE_ALERT
    )


def test_every_mode_that_draws_the_viewport_has_its_own_placeholder():
    """Inker had none and fell through to the mesh sentence, so an empty canvas
    pane advised picking a finished reference (H74).

    Create is keyed per *stage* (wave 5): one mode with two viewports has two
    empty states, and answering both with one sentence is the same defect this
    test was written for."""
    from warlock.studio import create_stages, modes
    from warlock.studio.panes import overlay

    for key in modes.WORK_MODES:
        if key == create_stages.MODE:
            for stage in create_stages.STAGES:
                assert f"{key}/{stage}" in overlay.PLACEHOLDERS, stage
            continue
        assert key in overlay.PLACEHOLDERS, key
    for icon, title, hint in overlay.PLACEHOLDERS.values():
        assert icon and title and hint
        assert all(ord(ch) < 0x100 for ch in title + hint)


def test_the_six_lists_that_lacked_an_empty_state_have_one():
    """H73, as a list rather than as six separate assertions: what is being
    pinned is that none of them silently goes back to drawing nothing."""
    from pathlib import Path

    root = Path(inspect.getfile(widgets)).resolve().parent
    for relative in (
        "panes/inker_layers.py",
        "panes/profiles_panel.py",
        "panes/sheet_panel.py",
        "panes/pose_panel.py",
        "manual/render.py",
        "main.py",  # Review's sweep list
    ):
        assert "empty_state(" in (root / relative).read_text(encoding="utf-8"), relative


def test_request_open_is_a_one_shot():
    """``once`` cannot serve this: it fires the first time a section is drawn
    in a session, and a dropped file lands long after that."""
    source = inspect.getsource(widgets.header)
    assert "_OPEN_REQUESTS" in source
    assert "discard" in source
