"""Sweep specs: one axis varied at a time against a fixed baseline."""

from __future__ import annotations

import json

import pytest

from warlock import guidance
from warlock.bench import manifest as manifest_mod
from warlock.bench import recipe as recipe_mod
from warlock.bench import runner as runner_mod
from warlock.bench import sweep as sweep_mod
from warlock.config import Config


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


# --- running -------------------------------------------------------------------


@pytest.fixture
def config(tmp_path):
    return Config(
        data_dir=tmp_path / "assets",
        db_path=tmp_path / "assets" / "jobs.sqlite",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
        t2i_model_root=tmp_path / "t2i",
        bench_dir=tmp_path / "bench",
    )


_SMALL_RAW = {
    "key": "small",
    "label": "small",
    "recipe_key": "baseline-turbo-raw",
    "item": {
        "id": "sweep-widget",
        "category": "prop",
        "prompt": "a small widget",
        "guidance": {},
    },
    "axes": [{"param": "custom_triangles", "values": [5000]}],
    "seeds": [1],
}


def _write_spec(sweep_dir, raw, name="small"):
    sweep_dir.mkdir(parents=True, exist_ok=True)
    path = sweep_dir / f"{name}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture
def sweep_dir(tmp_path, monkeypatch):
    directory = tmp_path / "sweeps"
    monkeypatch.setattr(sweep_mod, "SWEEP_DIR", directory)
    _write_spec(directory, _SMALL_RAW)
    return directory


def test_plan_sweep_marks_the_manifest_with_the_sweep_section(config, sweep_dir):
    spec = sweep_mod.load("small")
    recipe = recipe_mod.load(spec.recipe_key)
    run_dir, todo, doc = sweep_mod.plan_sweep(config, spec, recipe, started="20260805-120000")

    assert run_dir.name == "20260805-120000-sweep-small"
    assert len(todo) == 2  # baseline + the one differing custom_triangles value
    assert doc["sweep"]["key"] == "small"
    assert doc["sweep"]["axes"] == [{"param": "custom_triangles", "values": [5000]}]
    assert doc["suite"]["key"] == "small"
    assert doc["suite"]["file"] != "missing"


def test_a_server_axis_sweep_is_refused_before_planning(config, sweep_dir):
    raw = dict(_SMALL_RAW, key="server", axes=[{"param": "trellis_band", "values": [1, 2]}])
    _write_spec(sweep_dir, raw, "server")
    spec = sweep_mod.load("server")
    recipe = recipe_mod.load(spec.recipe_key)

    with pytest.raises(ValueError, match="phase 2|server-config"):
        sweep_mod.plan_sweep(config, spec, recipe, started="t")


def test_a_server_axis_sweep_is_refused_before_the_runtime_starts(config, sweep_dir):
    raw = dict(_SMALL_RAW, key="server", axes=[{"param": "trellis_band", "values": [1, 2]}])
    _write_spec(sweep_dir, raw, "server")

    with pytest.raises(ValueError, match="phase 2|server-config"):
        sweep_mod.run_sweep(config, spec_key="server", started="t")


def test_run_sweep_writes_a_manifest_a_jsonl_and_one_dir_per_unit(
    config, sweep_dir, fake_pipelines
):
    run_dir = sweep_mod.run_sweep(config, spec_key="small", started="20260805-120000")

    assert (run_dir / "manifest.json").exists()
    manifest = manifest_mod.read_manifest(run_dir)
    assert manifest["sweep"]["key"] == "small"

    records = runner_mod.read_items(run_dir)
    keys = {r["key"] for r in records}
    assert keys == {"baseline--s1", "custom_triangles=5000--s1"}
    assert all(r["status"] == "done" for r in records)
    assert all("param" in r and "value" in r for r in records)
    baseline = next(r for r in records if r["key"] == "baseline--s1")
    assert baseline["param"] is None
    varied = next(r for r in records if r["key"] == "custom_triangles=5000--s1")
    assert varied["param"] == "custom_triangles"
    assert varied["value"] == 5000
    for record in records:
        item_dir = run_dir / "items" / record["key"]
        assert (item_dir / "job.json").exists()
        assert (item_dir / "model.glb").exists()


def test_a_resume_only_reruns_what_did_not_finish(config, sweep_dir, fake_pipelines, monkeypatch):
    run_dir = sweep_mod.run_sweep(config, spec_key="small", started="20260805-120000")
    records = runner_mod.read_items(run_dir)
    assert len(records) == 2

    # Simulate one unit having failed, so a resume must retry only that one.
    lines = (run_dir / runner_mod.ITEMS_FILE).read_text("utf-8").splitlines()
    lines[-1] = json.dumps({**json.loads(lines[-1]), "status": "error", "error": "boom"})
    (run_dir / runner_mod.ITEMS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert runner_mod.completed(run_dir) == {"baseline--s1"}

    seen_job_ids: list[str] = []
    from warlock.service import jobs as svc_jobs

    real_create = svc_jobs.create_job

    def watching(svc, **kwargs):
        out = real_create(svc, **kwargs)
        seen_job_ids.append(out["id"])
        return out

    monkeypatch.setattr(svc_jobs, "create_job", watching)
    sweep_mod.run_sweep(config, spec_key="small", started="20260805-130000", resume=run_dir)

    # Only the one retryable unit was resubmitted.
    assert len(seen_job_ids) == 1
    assert runner_mod.completed(run_dir) == {"baseline--s1", "custom_triangles=5000--s1"}


def test_resume_against_a_changed_spec_file_is_refused(config, sweep_dir, fake_pipelines):
    run_dir = sweep_mod.run_sweep(config, spec_key="small", started="20260805-120000")

    # The spec file on disk changes -- e.g. a value under the same key edited
    # in place -- so its fingerprint moves and a resume must be refused.
    changed = dict(_SMALL_RAW, axes=[{"param": "custom_triangles", "values": [9999]}])
    _write_spec(sweep_dir, changed, "small")

    with pytest.raises(manifest_mod.NotResumable):
        sweep_mod.run_sweep(config, spec_key="small", started="20260805-130000", resume=run_dir)


def test_views_are_attempted_for_a_done_unit_with_a_mesh(
    config, sweep_dir, fake_pipelines, monkeypatch
):
    calls: list = []

    def fake_render(item_dir):
        calls.append(item_dir)
        return True

    monkeypatch.setattr(runner_mod, "_render_views", fake_render)
    run_dir = sweep_mod.run_sweep(config, spec_key="small", started="20260805-120000")

    records = runner_mod.read_items(run_dir)
    assert all(r["views"] is True for r in records)
    assert len(calls) == 2


def test_a_view_render_failure_is_non_fatal_and_recorded(
    config, sweep_dir, fake_pipelines, monkeypatch
):
    def failing(item_dir):
        return False

    monkeypatch.setattr(runner_mod, "_render_views", failing)
    run_dir = sweep_mod.run_sweep(config, spec_key="small", started="20260805-120000")

    records = runner_mod.read_items(run_dir)
    assert all(r["status"] == "done" for r in records)
    assert all(r["views"] is False for r in records)
