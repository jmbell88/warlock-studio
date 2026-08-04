"""UI logic that has nothing to do with imgui.

The action ladder, the filters, the ETA smoothing, the status-transition
diffing and the settings file. All of it is the part of a UI that can actually
be wrong in a way a screenshot would not show.
"""

from __future__ import annotations

import json

import pytest

from warlock.studio import settings as settingslib
from warlock.studio import state as statelib
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


def test_the_cache_exposes_how_much_history_it_is_not_showing(svc):
    """The window used to be a hard 200 with no way to know it was a window --
    filters then silently applied to a truncated set."""
    from warlock.service import jobs as svc_jobs

    for i in range(5):
        svc_jobs.create_job(svc, kind="text", prompt=f"j{i}")
    cache = JobsCache(svc, limit=2)
    cache.tick()
    assert len(cache.jobs) == 2
    assert cache.total == 5

    cache.load_more()
    cache.tick()
    assert len(cache.jobs) == 5
    assert cache.total == 5


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


@pytest.mark.parametrize("data", [[], None, "x", 3])
def test_a_settings_file_whose_data_is_not_a_map_is_ignored(tmp_path, data):
    """The version gate passed and the first get() crashed on a list."""
    (tmp_path / settingslib.FILENAME).write_text(
        json.dumps({"version": settingslib.VERSION, "data": data})
    )
    s = settingslib.Settings.load(tmp_path)
    assert s.data == {}
    assert s.get("mode", "2d") == "2d"


def test_a_settings_file_from_another_version_is_ignored(tmp_path):
    (tmp_path / settingslib.FILENAME).write_text(json.dumps({"version": 99, "data": {"x": 1}}))
    assert settingslib.Settings.load(tmp_path).data == {}


def test_a_pre_rename_settings_file_migrates_paint_to_inker(tmp_path):
    """Renaming the mode must not cost the user their swatches.

    The rename happens under version 1 rather than as a version bump: ``load``
    discards the whole file on a mismatch, so a bump would wipe every setting.
    """
    (tmp_path / settingslib.FILENAME).write_text(
        json.dumps(
            {
                "version": settingslib.VERSION,
                "data": {"mode": "paint", "paint": {"swatches": ["#ff0000"], "recent": ["a.png"]}},
            }
        )
    )
    s = settingslib.Settings.load(tmp_path)
    assert s.get("mode") == "inker"
    assert s.get("inker") == {"swatches": ["#ff0000"], "recent": ["a.png"]}
    assert "paint" not in s.data


def test_the_paint_migration_never_overwrites_an_existing_inker_block(tmp_path):
    (tmp_path / settingslib.FILENAME).write_text(
        json.dumps(
            {
                "version": settingslib.VERSION,
                "data": {"paint": {"swatches": ["#old"]}, "inker": {"swatches": ["#new"]}},
            }
        )
    )
    assert settingslib.Settings.load(tmp_path).get("inker") == {"swatches": ["#new"]}


def test_ui_scale_round_trips_and_a_junk_value_cannot_brick_the_window(tmp_path):
    from warlock.studio.main import UI_SCALE_RANGE, _ui_scale

    s = settingslib.Settings.load(tmp_path)
    assert _ui_scale(s) == 1.0  # nothing stored
    s.set("ui_scale", 1.25)
    s.flush()
    assert _ui_scale(settingslib.Settings.load(tmp_path)) == 1.25
    for junk in ("wide", None, 99.0, -3.0):
        s.set("ui_scale", junk)
        assert UI_SCALE_RANGE[0] <= _ui_scale(s) <= UI_SCALE_RANGE[1]


def test_only_the_work_modes_are_worth_persisting():
    """Home, the Manual, Clay and Settings are places you pass through:
    restoring into one on the next launch would hide the work."""
    from warlock.studio import modes

    assert modes.WORK_MODES.issubset(modes.KEYS)
    assert set(modes.KEYS) - modes.WORK_MODES
    assert modes.KEYS[0] == "home"
    assert AppState().mode == "home"
    assert set(modes.KEYS) == {k for k, _l, _i in modes.MODES}


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


# --- profiles ---------------------------------------------------------------


def test_a_fresh_form_rolls_its_own_seed():
    """The seed is not persisted, so a constant default meant every launch
    opened on the same one and a first Generate reproduced last week's image."""
    seeds = {statelib.default_form_2d()["seed"] for _ in range(8)}
    assert len(seeds) > 1
    assert all(0 <= s <= 2**31 - 1 for s in seeds)


def test_a_profile_captures_style_and_not_the_generation():
    from warlock.studio import profiles

    form = statelib.default_form_2d()
    form.update(prompt="a barrel", genre="fantasy", base_model="turbo", count=4)
    captured = profiles.capture(form)
    assert captured["genre"] == "fantasy"
    assert captured["base_model"] == "turbo"
    for volatile in ("prompt", "seed", "seed_locked", "count"):
        assert volatile not in captured


def test_a_profile_carries_the_look_fields_and_not_the_per_asset_ones():
    """A profile is a house style. Dragging 'category' or 'mood' along with it
    would make switching profiles quietly rewrite what the asset *is*."""
    from warlock.studio import profiles

    form = statelib.default_form_2d()
    form.update(
        genre="fantasy",
        art_style="painterly",
        setting="dungeon",
        palette="muted",
        category="weapon",
        material="wood",
        condition="worn",
        emissive="none",
        rarity="common",
        silhouette="tall",
        mood="grim",
    )
    captured = profiles.capture(form)
    assert set(profiles.TAXONOMY) <= set(captured)
    for per_asset in (
        "category", "material", "condition", "emissive", "rarity", "silhouette", "mood",
    ):
        assert per_asset not in captured


def test_a_profile_saved_with_per_asset_fields_is_inert_when_applied():
    """Old profiles on disk keep their extra keys; apply() must ignore them
    rather than needing a migration."""
    from warlock.studio import profiles

    form = statelib.default_form_2d()
    form["mood"] = "cheerful"
    profiles.apply(form, {"genre": "fantasy", "mood": "grim", "category": "weapon"})
    assert form["genre"] == "fantasy"
    assert form["mood"] == "cheerful"
    assert form["category"] == statelib.default_form_2d()["category"]


def test_applying_a_profile_leaves_the_generation_fields_alone():
    from warlock.studio import profiles

    form = statelib.default_form_2d()
    form.update(prompt="a barrel", seed=7, count=4)
    profiles.apply(form, {"genre": "fantasy", "prompt": "hijacked", "bogus": 1})
    assert form["genre"] == "fantasy"
    assert (form["prompt"], form["seed"], form["count"]) == ("a barrel", 7, 4)
    assert "bogus" not in form


def test_profiles_round_trip_through_settings(tmp_path):
    from warlock.studio import profiles

    s = settingslib.Settings.load(tmp_path)
    profiles.save_profile(s, "props", {"genre": "fantasy"})
    profiles.set_active(s, "props")
    assert profiles.active_fields(s) == {"genre": "fantasy"}
    s.flush()

    reloaded = settingslib.Settings.load(tmp_path)
    assert profiles.list_profiles(reloaded) == {"props": {"genre": "fantasy"}}

    profiles.delete_profile(reloaded, "props")
    assert profiles.list_profiles(reloaded) == {}
    # Deleting the active one clears the pointer: a stale name would make the
    # next New 2D Image apply nothing while claiming a profile was on.
    assert profiles.get_active(reloaded) is None


# --- conditioning -----------------------------------------------------------


def test_a_conditioning_scale_survives_a_restart_as_a_float():
    """restore_form gates on `type(value) is type(default)`, so every scale in
    the default form has to be a float literal -- an int default would make a
    persisted 0.6 fail to restore, silently reverting the slider."""
    defaults = statelib.default_form_2d()
    for key in ("ip_scale", "control_scale", "control_end"):
        assert isinstance(defaults[key], float)
    restored = settingslib.restore_form(defaults, {"ip_scale": 0.95})
    assert restored["ip_scale"] == 0.95


def test_an_int_over_a_float_scale_is_dropped():
    defaults = statelib.default_form_2d()
    assert settingslib.restore_form(defaults, {"ip_scale": 1})["ip_scale"] == defaults["ip_scale"]


def test_the_reference_path_never_persists():
    """A remembered path to a file that has since moved would silently
    condition next week's generation on nothing."""
    form = dict(statelib.default_form_2d(), ref_path="D:/pictures/knight.png")
    assert "ref_path" not in settingslib.sanitise_form(form)
    restored = settingslib.restore_form(
        statelib.default_form_2d(), {"ref_path": "D:/pictures/knight.png"}
    )
    assert restored["ref_path"] == ""


def test_the_conditioning_pickers_are_not_plain_guidance_combos():
    """They live in the Reference section and are hidden until there is an
    image to condition on, so the generic loop must not draw them too."""
    fields = statelib.guidance_fields()
    assert "ip_adapter" not in fields
    assert "control" not in fields
