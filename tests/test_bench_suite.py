"""The suite and recipe files are data, and this is what keeps them honest."""

from __future__ import annotations

import json

import pytest

from warlock import guidance
from warlock.bench import recipe as recipe_mod
from warlock.bench import suite as suite_mod

# The v1 item ids, frozen. A suite file is never edited in place -- a run's
# manifest records its fingerprint, so changing a prompt after the fact
# silently invalidates every comparison made against it. Changing the set
# means core-v2.json, and this test is what says so out loud.
CORE_V1_IDS = tuple(
    f"{category}-{n:02d}"
    for category in ("prop", "weapon", "character", "vehicle", "environment")
    for n in range(1, 9)
)


def test_the_default_suite_loads():
    s = suite_mod.load()
    assert s.key == "core-v1"
    assert len(s.items) == 40
    assert len(s.seeds) == 4


def test_the_v1_item_ids_are_frozen():
    assert tuple(i.id for i in suite_mod.load("core-v1").items) == CORE_V1_IDS


def test_every_category_is_evenly_covered():
    by_category = suite_mod.load().by_category()
    assert set(by_category) == set(suite_mod.CATEGORIES)
    assert {len(v) for v in by_category.values()} == {8}


def test_every_guidance_key_is_one_the_app_accepts():
    """A typo here would otherwise fail 160 submits one at a time, two hours
    into a run."""
    known = set(guidance.form_fields())
    for item in suite_mod.load().items:
        assert set(item.guidance) <= known
        # And the values, not just the keys.
        guidance.normalize(item.guidance)


def test_every_item_has_a_prompt_and_a_category():
    for item in suite_mod.load().items:
        assert item.prompt.strip()
        assert item.category in suite_mod.CATEGORIES


def test_an_unknown_suite_says_what_there_is():
    with pytest.raises(ValueError, match="core-v1"):
        suite_mod.load("nope")


def test_a_duplicate_item_id_is_refused(tmp_path):
    raw = {
        "seeds": [1],
        "items": [
            {"id": "a", "prompt": "x", "category": "prop"},
            {"id": "a", "prompt": "y", "category": "prop"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        suite_mod.parse(raw, tmp_path / "s.json")


def test_an_unknown_guidance_field_is_refused(tmp_path):
    raw = {"seeds": [1], "items": [{"id": "a", "prompt": "x", "guidance": {"bogus": "y"}}]}
    with pytest.raises(ValueError, match="bogus"):
        suite_mod.parse(raw, tmp_path / "s.json")


def test_a_suite_needs_seeds_and_items(tmp_path):
    with pytest.raises(ValueError, match="seed"):
        suite_mod.parse({"items": [{"id": "a", "prompt": "x"}]}, tmp_path / "s.json")
    with pytest.raises(ValueError, match="item"):
        suite_mod.parse({"seeds": [1]}, tmp_path / "s.json")


def test_filtering_by_category_and_id():
    s = suite_mod.load()
    assert len(s.filter(categories=("prop",))) == 8
    assert [i.id for i in s.filter(ids=("prop-01",))] == ["prop-01"]


# --- recipes ----------------------------------------------------------------


@pytest.mark.parametrize("key", recipe_mod.available())
def test_every_shipped_recipe_is_valid(key):
    """A recipe naming a removed LoRA should fail here, not on a run's 47th
    job."""
    r = recipe_mod.load(key)
    assert r.key == key
    assert r.label


def test_the_baseline_recipe_exists():
    assert "baseline-turbo-raw" in recipe_mod.available()


def test_a_recipe_naming_an_unknown_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="base_model"):
        recipe_mod.parse({"key": "x", "guidance": {"base_model": "gone"}}, tmp_path / "r.json")


def test_job_kwargs_is_exactly_what_create_job_takes():
    """The harness never assembles a params dict by hand: a benchmark that
    submitted differently from the app would measure a path no user reaches."""
    import inspect

    from warlock.service import jobs as svc_jobs

    s, r = suite_mod.load(), recipe_mod.load("sdxl-hyper-render3d")
    kwargs = recipe_mod.job_kwargs(r, s.items[0], 42)
    accepted = set(inspect.signature(svc_jobs.create_job).parameters) - {"svc"}
    assert set(kwargs) <= accepted


def test_the_item_wins_over_the_recipe_on_a_shared_field():
    """The recipe sets the backend; the item describes the subject. A suite
    that asks for stone must not have it overwritten."""
    s = suite_mod.load()
    item = next(i for i in s.items if i.guidance.get("material"))
    r = recipe_mod.parse({"key": "x", "guidance": {"material": "gold"}}, None)
    fields = recipe_mod.job_kwargs(r, item, 1)["guidance_fields"]
    assert fields["material"] == item.guidance["material"]


def test_the_backend_comes_from_the_recipe():
    s = suite_mod.load()
    kwargs = recipe_mod.job_kwargs(recipe_mod.load("playground-fidelity"), s.items[0], 7)
    assert kwargs["guidance_fields"]["base_model"] == "playground"
    assert kwargs["seed"] == 7
    assert kwargs["output"] == "reference"


def test_a_lora_weight_rides_only_with_a_lora():
    s = suite_mod.load()
    plain = recipe_mod.job_kwargs(recipe_mod.load("baseline-turbo-raw"), s.items[0], 1)
    assert "lora_weight" not in plain
    styled = recipe_mod.job_kwargs(recipe_mod.load("sdxl-hyper-render3d"), s.items[0], 1)
    assert styled["lora_weight"] == 0.9


def test_the_model_stage_asks_for_a_mesh():
    s = suite_mod.load()
    kwargs = recipe_mod.job_kwargs(
        recipe_mod.load("baseline-turbo-raw"), s.items[0], 1, stage="model"
    )
    assert kwargs["output"] == "model"


def test_suite_and_recipe_files_are_json_and_stay_that_way():
    for path in list(suite_mod.SUITE_DIR.glob("*.json")) + list(
        recipe_mod.RECIPE_DIR.glob("*.json")
    ):
        json.loads(path.read_text("utf-8"))
