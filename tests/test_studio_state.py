"""UI logic that has nothing to do with imgui.

The action ladder, the filters, the ETA smoothing, the status-transition
diffing and the settings file. All of it is the part of a UI that can actually
be wrong in a way a screenshot would not show.
"""

from __future__ import annotations

import json

import pytest

from warlock.studio import settings as settingslib
from warlock.studio.jobs_cache import JobsCache, transition_message
from warlock.studio.state import (
    AppState,
    Eta,
    Filters,
    format_bytes,
    format_duration,
    primary_action,
)


def job(**kwargs):
    base = {
        "id": "0123456789ab",
        "kind": "text",
        "stage": "model",
        "status": "done",
        "files": ["model.glb"],
        "prompt": "a barrel",
        "name": "",
        "tags": "",
        "favorite": 0,
    }
    return {**base, **kwargs}


# --- the action ladder ------------------------------------------------------


def test_a_running_job_offers_only_cancel():
    assert primary_action(job(status="running")) == "cancel"
    assert primary_action(job(status="queued")) == "cancel"


def test_a_failed_job_offers_a_retry():
    assert primary_action(job(status="error")) == "retry"


def test_a_finished_reference_offers_the_mesh_it_exists_for():
    assert primary_action(job(stage="reference", files=["input.png"])) == "promote"


def test_a_reference_with_no_image_offers_nothing():
    assert primary_action(job(stage="reference", files=[])) is None


def test_a_finished_mesh_offers_a_rig_until_it_has_one():
    assert primary_action(job()) == "rig"
    assert primary_action(job(files=["model.glb", "rig.glb"])) == "open"


def test_a_rig_job_is_never_offered_a_rig():
    assert primary_action(job(kind="rig")) == "open"


def test_without_blender_a_mesh_just_opens():
    assert primary_action(job(), rigging_available=False) == "open"


def test_a_cancelled_job_offers_nothing():
    assert primary_action(job(status="cancelled")) is None


# --- filters ----------------------------------------------------------------


def test_a_text_filter_searches_name_prompt_tags_and_id():
    f = Filters(text="barrel")
    assert f.matches(job(prompt="a wooden barrel"))
    assert f.matches(job(prompt="x", name="Barrel v2"))
    assert f.matches(job(prompt="x", tags="barrel,prop"))
    assert not f.matches(job(prompt="a sword", name="", tags=""))


def test_a_text_filter_is_case_and_space_insensitive():
    assert Filters(text="  BARREL ").matches(job(prompt="a barrel"))


def test_the_kind_filter_tells_a_reference_from_a_mesh():
    """Both are text jobs; only the stage says which is which."""
    assert Filters(kind="reference").matches(job(stage="reference"))
    assert not Filters(kind="reference").matches(job(stage="model"))
    assert Filters(kind="model").matches(job(stage="model"))
    assert Filters(kind="rig").matches(job(kind="rig", stage="model"))


def test_favourites_only_hides_everything_else():
    f = Filters(favorites_only=True)
    assert f.matches(job(favorite=1))
    assert not f.matches(job(favorite=0))


def test_filters_compose():
    f = Filters(text="barrel", status="done", favorites_only=True)
    assert f.matches(job(prompt="a barrel", status="done", favorite=1))
    assert not f.matches(job(prompt="a barrel", status="error", favorite=1))


# --- app state --------------------------------------------------------------


def test_prompt_history_is_most_recent_first_and_deduplicated():
    state = AppState()
    for prompt in ("a", "b", "a"):
        state.remember_prompt(prompt)
    assert state.history == ["a", "b"]


def test_prompt_history_is_bounded():
    state = AppState()
    for i in range(30):
        state.remember_prompt(f"prompt {i}")
    assert len(state.history) == 20
    assert state.history[0] == "prompt 29"


def test_blank_prompts_are_not_remembered():
    state = AppState()
    state.remember_prompt("   ")
    assert state.history == []


def test_changing_the_selection_ends_a_comparison():
    """A comparison is between the selection and something else; keeping it
    would compare two jobs neither of which was just clicked."""
    state = AppState()
    state.select("aaaaaaaaaaaa")
    state.comparing = "bbbbbbbbbbbb"
    state.select("cccccccccccc")
    assert state.comparing is None


def test_reselecting_the_same_job_keeps_the_comparison():
    state = AppState()
    state.select("aaaaaaaaaaaa")
    state.comparing = "bbbbbbbbbbbb"
    state.select("aaaaaaaaaaaa")
    assert state.comparing == "bbbbbbbbbbbb"


def test_an_error_toast_outlives_an_informational_one():
    state = AppState()
    state.toast("done", "info")
    state.toast("boom", "error")
    assert state.toasts[0].ttl < state.toasts[1].ttl


# --- the ETA ----------------------------------------------------------------


def test_no_estimate_until_the_job_is_warm_and_underway():
    eta = Eta()
    assert eta.update("a", percent=3.0, elapsed=20.0, cold=False) is None
    assert eta.update("a", percent=50.0, elapsed=2.0, cold=False) is None
    assert eta.update("a", percent=50.0, elapsed=20.0, cold=True) is None
    assert eta.update("a", percent=100.0, elapsed=20.0, cold=False) is None


def test_the_first_estimate_is_the_raw_one():
    assert Eta().update("a", percent=50.0, elapsed=10.0, cold=False) == pytest.approx(10.0)


def test_later_estimates_are_smoothed_towards_the_new_one():
    eta = Eta()
    eta.update("a", percent=50.0, elapsed=10.0, cold=False)  # 10 s remaining
    smoothed = eta.update("a", percent=50.0, elapsed=30.0, cold=False)  # 30 s raw
    # 0.7 * 10 + 0.3 * 30
    assert smoothed == pytest.approx(16.0)


def test_a_new_job_starts_the_estimate_over():
    eta = Eta()
    eta.update("a", percent=50.0, elapsed=10.0, cold=False)
    assert eta.update("b", percent=50.0, elapsed=100.0, cold=False) == pytest.approx(100.0)


def test_durations_read_as_a_clock():
    assert format_duration(45) == "45s"
    assert format_duration(130) == "2m 10s"
    assert format_duration(3700) == "1h 01m"
    assert format_duration(None) == ""


def test_byte_counts_read_as_sizes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(5 * 1024**3) == "5.0 GB"


# --- transitions ------------------------------------------------------------


def test_only_terminal_transitions_are_worth_a_toast():
    assert transition_message(job(status="running"), "queued") is None
    assert transition_message(job(status="done"), "running")[1] == "info"
    assert transition_message(job(status="error", error="boom"), "running")[1] == "error"


def test_a_cancelled_queued_job_is_not_announced():
    """Cancelling something that never started is not news; the row vanishing
    from the queue already says it."""
    assert transition_message(job(status="cancelled"), "queued") is None
    assert transition_message(job(status="cancelled"), "running") is not None


def test_a_long_name_is_shortened_for_the_toast():
    text, _level = transition_message(job(status="done", name="x" * 80), "running")
    assert len(text) < 60


def test_the_cache_reports_a_transition_only_once(svc):
    from warlock.service import jobs as svc_jobs

    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    cache = JobsCache(svc)
    seen = []
    cache.tick(lambda j, prev: seen.append((j["id"], prev)))
    assert seen == []  # the first read has nothing to compare against

    svc.store.set_status(job_id, "done")
    cache.invalidate()
    cache.tick(lambda j, prev: seen.append((j["id"], prev)))
    assert seen == [(job_id, "queued")]

    cache.invalidate()
    cache.tick(lambda j, prev: seen.append((j["id"], prev)))
    assert len(seen) == 1


def test_the_cache_does_not_re_read_on_every_frame(svc):
    cache = JobsCache(svc)
    assert cache.tick() is True
    assert cache.tick() is False


def test_the_cache_finds_the_job_worth_narrating(svc):
    from warlock.service import jobs as svc_jobs

    done = svc_jobs.create_job(svc, kind="text", prompt="old")["id"]
    svc.store.set_status(done, "done")
    running = svc_jobs.create_job(svc, kind="text", prompt="new")["id"]
    svc.store.set_status(running, "running")

    cache = JobsCache(svc)
    cache.tick()
    assert cache.active["id"] == running


def test_the_cache_survives_a_read_that_fails(svc):
    cache = JobsCache(svc)

    def boom(*_a, **_k):
        raise RuntimeError("database is locked")

    svc.store.list = boom
    assert cache.tick() is False
    assert "locked" in cache.error


# --- settings ---------------------------------------------------------------


def test_settings_round_trip(tmp_path):
    s = settingslib.Settings.load(tmp_path)
    s.set("mode", "3d")
    assert s.flush() is True
    assert settingslib.Settings.load(tmp_path).get("mode") == "3d"


def test_saving_is_debounced_but_a_flush_is_not(tmp_path):
    s = settingslib.Settings.load(tmp_path)
    s.set("mode", "3d")
    assert s.tick() is True  # the first tick writes
    s.set("mode", "2d")
    assert s.tick() is False  # ...and the next is inside the debounce window
    assert s.flush() is True  # but exit must not lose the last edit
    assert settingslib.Settings.load(tmp_path).get("mode") == "2d"


def test_setting_the_same_value_is_not_a_change(tmp_path):
    s = settingslib.Settings.load(tmp_path)
    s.set("mode", "2d")
    s.flush()
    s.set("mode", "2d")
    assert s.flush() is False


def test_a_corrupt_settings_file_is_ignored_rather_than_fatal(tmp_path):
    (tmp_path / settingslib.FILENAME).write_text("{not json")
    assert settingslib.Settings.load(tmp_path).data == {}


def test_a_settings_file_from_another_version_is_ignored(tmp_path):
    (tmp_path / settingslib.FILENAME).write_text(json.dumps({"version": 99, "data": {"x": 1}}))
    assert settingslib.Settings.load(tmp_path).data == {}


def test_the_seed_never_persists():
    """Remembering last session's seed silently reproduces last session's
    output, which reads as 'generate is broken'."""
    form = {"prompt": "a barrel", "seed": 1234}
    assert settingslib.sanitise_form(form) == {"prompt": "a barrel"}


def test_restoring_a_form_keeps_only_known_keys_of_the_right_type():
    defaults = {"prompt": "", "count": 1, "seed": 42}
    restored = settingslib.restore_form(
        defaults, {"prompt": "a sword", "count": "three", "seed": 999, "bogus": 1}
    )
    assert restored == {"prompt": "a sword", "count": 1, "seed": 42}


def test_a_settings_write_is_atomic(tmp_path):
    """A half-written file takes the next launch down with it."""
    s = settingslib.Settings.load(tmp_path)
    s.set("mode", "3d")
    s.flush()
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".settings.")]
    assert leftovers == []
