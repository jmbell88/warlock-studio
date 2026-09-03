"""The recipe codec: clamping on load, refusals, uids, and the round trip."""

from __future__ import annotations

import json

import pytest
from _recipes import FIREBALL

from warlock.studio.inker import flourish
from warlock.studio.inker.flourish import prims
from warlock.studio.inker.flourish import recipe as R


def test_a_loaded_recipe_round_trips_through_json():
    rec = flourish.from_dict(FIREBALL)
    again = flourish.loads(flourish.dumps(rec))
    assert again == rec
    assert json.loads(flourish.dumps(rec))["version"] == flourish.SCHEMA_VERSION


def test_loading_fills_every_parameter_with_its_default():
    rec = flourish.from_dict(FIREBALL)
    for layer in rec.layers:
        assert set(layer.params) == set(prims.params_of(layer.kind))


def test_loading_clamps_out_of_range_values():
    raw = {
        "size": [100000, 4],
        "supersample": 99,
        "fps": 0,
        "mode": "watercolour",
        "phases": [{"name": "p", "frames": 100000}],
        "layers": [
            {
                "kind": "particles",
                "params": {"count": 99999, "color_start": "not a colour", "emission": "trickle"},
                "blend": "multiply",
                "opacity": 7,
            }
        ],
    }
    rec = flourish.from_dict(raw)
    assert (rec.width, rec.height) == (R.MAX_SIZE, 8)
    assert rec.supersample == R.MAX_SUPERSAMPLE
    assert rec.fps == 1
    assert rec.mode == "painterly"
    assert rec.phases[0].frames == R.MAX_FRAMES_PER_PHASE
    layer = rec.layers[0]
    assert layer.params["count"] == prims.params_of("particles")["count"].hi
    assert layer.params["color_start"] == prims.params_of("particles")["color_start"].default
    assert layer.params["emission"] == "burst"
    assert layer.blend == "normal"
    assert layer.opacity == 1.0


def test_an_unknown_primitive_kind_is_refused_not_kept():
    with pytest.raises(ValueError, match="not a primitive"):
        flourish.from_dict({"layers": [{"kind": "lensflare"}]})


def test_a_newer_schema_is_refused():
    with pytest.raises(ValueError, match="newer"):
        flourish.from_dict({"version": flourish.SCHEMA_VERSION + 1})


def test_unknown_parameters_and_phases_are_dropped():
    rec = flourish.from_dict(
        {
            "phases": [{"name": "a", "frames": 2}],
            "layers": [{"kind": "core", "params": {"wibble": 3}, "phases": ["a", "zzz"]}],
        }
    )
    assert "wibble" not in rec.layers[0].params
    assert rec.layers[0].phases == ("a",)


def test_uids_are_kept_when_unique_and_reissued_when_not():
    rec = flourish.from_dict(
        {"layers": [{"kind": "core", "uid": 40}, {"kind": "core", "uid": 40}, {"kind": "flash"}]}
    )
    uids = [layer.uid for layer in rec.layers]
    assert uids[0] == 40
    assert len(set(uids)) == 3


def test_bump_uids_gives_fresh_identities_for_a_second_insert():
    rec = flourish.from_dict(FIREBALL)
    twice = R.bump_uids(rec)
    assert {each.uid for each in rec.layers}.isdisjoint({each.uid for each in twice.layers})


def test_reserve_uids_moves_the_counter_past_a_loaded_recipe():
    rec = flourish.from_dict({"layers": [{"kind": "core", "uid": 1_000_000}]})
    R.reserve_uids(rec)
    assert flourish.new_uid() > 1_000_000


def test_phase_arithmetic():
    rec = flourish.from_dict(FIREBALL)
    assert rec.frame_count == 17
    phase, index, within = rec.phase_at(3)
    assert (phase.name, index, within) == ("projectile", 1, 0)
    assert rec.phase_start(2) == 8
    assert rec.phase_named("impact").frames == 2
    assert rec.phase_named("nope") is None
    with pytest.raises(IndexError):
        rec.phase_at(17)


def test_with_param_clamps_and_replace_layer_keeps_order():
    rec = flourish.from_dict(FIREBALL)
    core = next(each for each in rec.layers if each.kind == "core")
    changed = core.with_param("radius", 99999)
    assert changed.params["radius"] == prims.params_of("core")["radius"].hi
    with pytest.raises(KeyError):
        core.with_param("wibble", 1)
    swapped = rec.replace_layer(changed)
    assert [each.uid for each in swapped.layers] == [each.uid for each in rec.layers]
    assert swapped.layer(core.uid) == changed


def test_a_layer_with_no_phases_is_active_everywhere_and_invisible_never():
    layer = R.Layer(uid=1, kind="core")
    assert layer.active_in("anything")
    assert not R.Layer(uid=1, kind="core", visible=False).active_in("anything")
    assert not R.Layer(uid=1, kind="core", phases=("a",)).active_in("b")
