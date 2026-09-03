"""The keyword mapper and the diff validator: deterministic, clamped, honest."""

from __future__ import annotations

from warlock.studio.inker import flourish
from warlock.studio.inker.flourish import keywords, presets, prims


def _fireball():
    return presets.load("fireball")


def _layer(rec, kind, name=None):
    for layer in rec.layers:
        if layer.kind == kind and (name is None or layer.name == name):
            return layer
    raise KeyError(kind)


def test_a_colour_word_repaints_every_coloured_layer():
    rec, notes = keywords.apply(_fireball(), "blue")
    core = _layer(rec, "core", "Core")
    assert core.params["color_inner"] == keywords.COLOURS["blue"][0]
    assert core.params["color_outer"] == keywords.COLOURS["blue"][1]
    assert _layer(rec, "flame").params["color_tip"] == keywords.COLOURS["blue"][1]
    assert any("blue" in n for n in notes)


def test_a_colour_before_a_kind_word_repaints_only_that_kind():
    before = _fireball()
    rec, _ = keywords.apply(before, "green flames")
    assert _layer(rec, "flame").params["color_tip"] == keywords.COLOURS["green"][1]
    assert _layer(rec, "core", "Core").params == _layer(before, "core", "Core").params


def test_more_and_no_change_counts_and_visibility():
    before = _fireball()
    rec, notes = keywords.apply(before, "more sparks, no smoke")
    for name in ("Embers", "Sparks"):
        after = _layer(rec, "particles", name).params["count"]
        assert after > _layer(before, "particles", name).params["count"]
    assert all(not layer.visible for layer in rec.layers if layer.kind == "smoke")
    assert any(n.startswith("hid") for n in notes)


def test_scale_words_scale_the_right_parameters_and_clamp():
    before = _fireball()
    rec, _ = keywords.apply(before, "bigger and faster")
    core_before = _layer(before, "core", "Core")
    core_after = _layer(rec, "core", "Core")
    assert core_after.params["radius"] > core_before.params["radius"]
    sparks_before = _layer(before, "particles", "Sparks")
    sparks_after = _layer(rec, "particles", "Sparks")
    assert sparks_after.params["speed"] > sparks_before.params["speed"]
    huge, _ = keywords.apply(rec, " ".join(["huge"] * 40))
    hi = prims.params_of("core")["radius"].hi
    assert flourish.Curve.from_json(_layer(huge, "core", "Core").params["radius"]).at(0.0) <= hi


def test_longer_and_shorter_change_the_phases():
    before = _fireball()
    rec, _ = keywords.apply(before, "longer")
    assert sum(p.frames for p in rec.phases) > before.frame_count
    rec, _ = keywords.apply(before, "shorter")
    assert sum(p.frames for p in rec.phases) < before.frame_count


def test_unknown_words_change_nothing_and_say_so():
    before = _fireball()
    rec, notes = keywords.apply(before, "make it sparkle like a disco")
    assert rec == before
    assert len(notes) == 1 and "colours" in notes[0]
    rec, notes = keywords.apply(before, "")
    assert rec == before and notes == ["Nothing to apply."]


def test_the_result_is_always_a_clamped_recipe():
    rec, _ = keywords.apply(_fireball(), "colder, huge, brighter, wilder, more embers")
    assert rec == flourish.clamp(rec)


def test_apply_diff_lands_named_parameters_and_clamps_them():
    before = _fireball()
    rec, notes = keywords.apply_diff(
        before,
        {
            "seed": 42,
            "fps": 999,
            "layers": {"Core": {"radius": 99999, "color_outer": "#00FF00", "wibble": 1}},
            "phases": {"projectile": {"frames": 4, "loop": False}},
            "hide": ["Smoke", "Nope"],
        },
    )
    assert rec.seed == 42 and rec.fps == 120
    core = _layer(rec, "core", "Core")
    assert core.params["radius"] == prims.params_of("core")["radius"].hi
    assert core.params["color_outer"] == "#00FF00"
    assert rec.phase_named("projectile").frames == 4
    assert not rec.phase_named("projectile").loop
    assert not _layer(rec, "smoke", "Smoke").visible
    assert any("wibble" in n for n in notes)
    assert any("Nope" in n for n in notes)
    assert rec == flourish.clamp(rec)


def test_apply_diff_refuses_garbage_without_raising():
    before = _fireball()
    rec, notes = keywords.apply_diff(before, "not a dict")
    assert rec == before and "JSON object" in notes[0]
    rec, notes = keywords.apply_diff(before, {"layers": {"Core": "nope"}, "seed": "x"})
    assert rec == before or rec == flourish.clamp(before)
    assert any("not a mapping" in n for n in notes)
    assert any("seed is not a number" in n for n in notes)


def test_the_model_view_has_no_uids_and_names_every_range():
    view = keywords.describe_for_model(_fireball())
    assert "uid" not in str(view)
    assert view["layers"][0]["name"]
    core = next(layer for layer in view["layers"] if layer["kind"] == "core")
    assert core["params"]["radius"]["range"] == [0.0, 512.0]
    assert "choices" in core["params"]["color_inner"] or "value" in core["params"]["color_inner"]
    assert keywords.DIFF_SCHEMA.startswith("{")
