"""Sweep specs: one axis varied at a time against a fixed baseline."""

from __future__ import annotations

import pytest

from warlock import guidance
from warlock.bench import recipe as recipe_mod
from warlock.bench import sweep as sweep_mod


def _raw(**overrides):
    raw = {
        "key": "x",
        "recipe_key": "baseline-turbo-raw",
        "item": {
            "id": "sweep-chest",
            "category": "prop",
            "prompt": "a treasure chest",
            "guidance": {"style_lora": "render3d"},
        },
        "axes": [{"param": "lora_weight", "values": [0.6, 0.9, 1.2]}],
        "seeds": [7, 42],
    }
    raw.update(overrides)
    return raw


# --- loading / parsing --------------------------------------------------------


def test_the_shipped_lora_weight_spec_loads():
    spec = sweep_mod.load("lora-weight-v1")
    assert spec.key == "lora-weight-v1"
    assert spec.recipe_key == "baseline-turbo-raw"
    assert spec.item.id
    assert spec.axes[0].param == "lora_weight"
    assert spec.seeds == (7, 42, 1234, 20260805)


def test_an_unknown_sweep_says_what_there_is():
    with pytest.raises(ValueError, match="lora-weight-v1"):
        sweep_mod.load("nope")


def test_a_sweep_needs_at_least_one_seed(tmp_path):
    raw = _raw(seeds=[])
    with pytest.raises(ValueError, match="seed"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_a_sweep_needs_at_least_one_axis(tmp_path):
    raw = _raw(axes=[])
    with pytest.raises(ValueError, match="axis"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_a_sweep_needs_a_recipe_key(tmp_path):
    raw = _raw(recipe_key="")
    with pytest.raises(ValueError, match="recipe_key"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_the_item_is_validated_exactly_as_a_suite_item(tmp_path):
    raw = _raw(item={"id": "a", "category": "prop", "prompt": "x", "guidance": {"bogus": "y"}})
    with pytest.raises(ValueError, match="bogus"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_an_item_with_no_prompt_is_refused(tmp_path):
    raw = _raw(item={"id": "a", "category": "prop", "prompt": "", "guidance": {}})
    with pytest.raises(ValueError, match="prompt"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_an_unknown_axis_param_is_refused(tmp_path):
    raw = _raw(axes=[{"param": "bogus_param", "values": [1, 2]}])
    with pytest.raises(ValueError, match="bogus_param"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_an_axis_needs_values(tmp_path):
    raw = _raw(axes=[{"param": "lora_weight", "values": []}])
    with pytest.raises(ValueError, match="lora_weight"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_a_duplicate_axis_param_is_refused(tmp_path):
    raw = _raw(
        axes=[
            {"param": "lora_weight", "values": [0.5]},
            {"param": "lora_weight", "values": [0.7]},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_a_guidance_axis_value_is_validated_through_normalize(tmp_path):
    raw = _raw(axes=[{"param": "material", "values": ["stonee"]}])
    with pytest.raises(ValueError, match="material"):
        sweep_mod.parse(raw, tmp_path / "s.json")


def test_a_server_axis_loads_but_the_runner_refuses_it(tmp_path):
    """Accepted at parse -- the file can be written and reviewed now -- but
    unit_kwargs refuses it: running one means restarting trellis-server
    between groups, which is phase 2."""
    raw = _raw(axes=[{"param": "trellis_band", "values": [1, 2]}])
    spec = sweep_mod.parse(raw, tmp_path / "s.json")
    assert spec.axes[0].param == "trellis_band"

    recipe = recipe_mod.load(spec.recipe_key)
    unit = sweep_mod.SweepUnit(param="trellis_band", value=2, seed=7)
    with pytest.raises(ValueError, match="server-config|phase 2"):
        sweep_mod.unit_kwargs(spec, recipe, unit)


def test_a_kwarg_tier_axis_is_accepted(tmp_path):
    raw = _raw(axes=[{"param": "custom_triangles", "values": [5000, 8000]}])
    spec = sweep_mod.parse(raw, tmp_path / "s.json")
    assert spec.axes[0].param == "custom_triangles"


# --- unit keys -----------------------------------------------------------------


def test_the_baseline_unit_key_names_only_the_seed():
    unit = sweep_mod.SweepUnit(param=None, value=None, seed=42)
    assert unit.key == "baseline--s42"


def test_a_varied_unit_key_names_the_param_and_value():
    unit = sweep_mod.SweepUnit(param="lora_weight", value=0.6, seed=42)
    assert unit.key == "lora_weight=0.6--s42"


def test_a_unit_key_is_filesystem_safe():
    unit = sweep_mod.SweepUnit(param="negative_prompt", value="a/b c:d*e", seed=1)
    assert "/" not in unit.key
    assert ":" not in unit.key
    assert " " not in unit.key


def test_the_slug_truncates_a_long_value():
    long_value = "x" * 200
    result = sweep_mod.slug(long_value)
    assert len(result) <= sweep_mod.SLUG_MAX


# --- planning --------------------------------------------------------------


def test_a_sweep_plans_baseline_plus_one_change_per_unit(tmp_path):
    """One baseline unit per seed, plus one unit per (axis value, seed) that
    differs from the baseline -- never a cross product of multiple axes."""
    raw = _raw(
        axes=[
            {"param": "lora_weight", "values": [0.6, 0.9, 1.2]},
            {"param": "condition", "values": ["worn", "pristine"]},
        ],
        seeds=[7, 42],
    )
    spec = sweep_mod.parse(raw, tmp_path / "s.json")
    units = sweep_mod.plan_units(spec)

    baseline_units = [u for u in units if u.param is None]
    assert len(baseline_units) == 2  # one per seed

    lora_units = [u for u in units if u.param == "lora_weight"]
    # 0.9 is render3d's own default_weight -- the baseline for this item --
    # so plan_units skips it, leaving 0.6 and 1.2 (2 values x 2 seeds).
    assert len(lora_units) == 4
    assert {u.value for u in lora_units} == {0.6, 1.2}

    condition_units = [u for u in units if u.param == "condition"]
    assert len(condition_units) == 4  # both values differ from the (unset) baseline


def test_plan_units_skips_a_value_equal_to_the_baseline(tmp_path):
    raw = _raw(axes=[{"param": "lora_weight", "values": [0.6, 0.9, 1.2]}], seeds=[7])
    spec = sweep_mod.parse(raw, tmp_path / "s.json")
    units = sweep_mod.plan_units(spec)
    varied = [u for u in units if u.param == "lora_weight"]
    assert 0.9 not in {u.value for u in varied}
    assert {u.value for u in varied} == {0.6, 1.2}


# --- unit_kwargs / tier landing ----------------------------------------------


def test_a_guidance_tier_axis_lands_in_guidance_fields(tmp_path):
    raw = _raw(axes=[{"param": "condition", "values": ["pristine"]}], seeds=[1])
    spec = sweep_mod.parse(raw, tmp_path / "s.json")
    recipe = recipe_mod.load(spec.recipe_key)
    unit = sweep_mod.SweepUnit(param="condition", value="pristine", seed=1)

    kwargs = sweep_mod.unit_kwargs(spec, recipe, unit)

    assert kwargs["guidance_fields"]["condition"] == "pristine"
    assert "condition" not in kwargs


def test_a_kwarg_tier_axis_lands_as_a_top_level_kwarg(tmp_path):
    raw = _raw(axes=[{"param": "lora_weight", "values": [0.6]}], seeds=[1])
    spec = sweep_mod.parse(raw, tmp_path / "s.json")
    recipe = recipe_mod.load(spec.recipe_key)
    unit = sweep_mod.SweepUnit(param="lora_weight", value=0.6, seed=1)

    kwargs = sweep_mod.unit_kwargs(spec, recipe, unit)

    assert kwargs["lora_weight"] == 0.6
    assert "lora_weight" not in kwargs.get("guidance_fields", {})


def test_the_baseline_unit_kwargs_are_exactly_the_recipes_own(tmp_path):
    spec = sweep_mod.load("lora-weight-v1")
    recipe = recipe_mod.load(spec.recipe_key)
    baseline = sweep_mod.SweepUnit(param=None, value=None, seed=7)

    kwargs = sweep_mod.unit_kwargs(spec, recipe, baseline)
    expected = recipe_mod.job_kwargs(recipe, spec.item, 7, stage="model")
    assert kwargs == expected


def test_unit_kwargs_is_exactly_what_create_job_takes(tmp_path):
    """The harness never assembles a params dict by hand -- a benchmark that
    submitted differently from the app would measure a path no user reaches."""
    import inspect

    from warlock.service import jobs as svc_jobs

    spec = sweep_mod.load("lora-weight-v1")
    recipe = recipe_mod.load(spec.recipe_key)
    accepted = set(inspect.signature(svc_jobs.create_job).parameters) - {"svc"}
    for unit in sweep_mod.plan_units(spec):
        if unit.param in sweep_mod.SERVER_AXES:
            continue
        kwargs = sweep_mod.unit_kwargs(spec, recipe, unit)
        assert set(kwargs) <= accepted


def test_every_shipped_sweep_is_valid():
    for key in sweep_mod.available():
        spec = sweep_mod.load(key)
        assert spec.key == key
        assert spec.label
        recipe_mod.load(spec.recipe_key)  # the recipe it names must itself exist


def test_lora_weight_v1_is_the_shipped_sweep():
    assert "lora-weight-v1" in sweep_mod.available()


def test_every_axis_value_the_guidance_tier_accepts_normalizes(tmp_path):
    spec = sweep_mod.load("lora-weight-v1")
    for axis in spec.axes:
        if axis.param in guidance.form_fields():
            for value in axis.values:
                guidance.normalize({axis.param: value})
