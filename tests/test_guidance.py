from __future__ import annotations

import pytest

from warlock import guidance, models


def test_defaults_fill_in_when_nothing_is_chosen():
    params = guidance.normalize({})
    assert params["platform"] == guidance.DEFAULT_PLATFORM
    assert params["resolution"] == guidance.PLATFORMS[guidance.DEFAULT_PLATFORM].resolution
    assert params["size_m"] == guidance.DEFAULT_SIZE_M
    # Optional fields stay absent rather than being stored as empty strings.
    assert "genre" not in params
    assert "art_style" not in params
    assert "category" not in params


def test_platform_supplies_the_geometry_resolution():
    assert guidance.normalize({"platform": "mobile"})["resolution"] == 512
    assert guidance.normalize({"platform": "hero"})["resolution"] == 1536


def test_explicit_resolution_overrides_the_platform_preset():
    params = guidance.normalize({"platform": "mobile", "resolution": 1536})
    assert params["resolution"] == 1536
    assert params["platform"] == "mobile"


def test_category_supplies_a_default_size():
    assert guidance.normalize({"category": "character"})["size_m"] == 1.8
    assert guidance.normalize({"category": "consumable"})["size_m"] == 0.15


def test_explicit_size_wins_over_the_category_default():
    assert guidance.normalize({"category": "character", "size_m": "0.5"})["size_m"] == 0.5


@pytest.mark.parametrize("field", ["genre", "art_style", "category", "platform"])
def test_unknown_values_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        guidance.normalize({field: "nonsense"})


@pytest.mark.parametrize("value", ["", None])
def test_blank_means_unspecified_not_invalid(value):
    assert guidance.normalize({"genre": value}) == guidance.normalize({})


def test_size_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="between"):
        guidance.normalize({"size_m": 0.0})
    with pytest.raises(ValueError, match="between"):
        guidance.normalize({"size_m": 1000})


def test_non_numeric_size_is_rejected():
    with pytest.raises(ValueError, match="number"):
        guidance.normalize({"size_m": "big"})


def test_compose_prompt_orders_fragments_after_the_user_text():
    params = guidance.normalize(
        {"genre": "scifi", "art_style": "lowpoly", "category": "weapon", "platform": "mobile"}
    )
    composed = guidance.compose_prompt("a plasma rifle", params)
    assert composed.startswith("a plasma rifle, ")
    positions = [
        composed.index(guidance.CATEGORIES["weapon"].prompt),
        composed.index(guidance.GENRES["scifi"].prompt),
        composed.index(guidance.ART_STYLES["lowpoly"].prompt),
        composed.index(guidance.PLATFORMS["mobile"].prompt),
    ]
    assert positions == sorted(positions)


def test_compose_prompt_with_no_guidance_is_just_the_prompt():
    assert guidance.compose_prompt("  a barrel  ", {}) == "a barrel"


def test_compose_prompt_ignores_stale_values():
    """Params can come from a job row written before a key was renamed; a
    slightly less specific prompt beats failing an otherwise valid job."""
    assert guidance.compose_prompt("a barrel", {"genre": "retired"}) == "a barrel"


def test_catalog_covers_every_field_and_is_json_safe():
    import json

    catalog = guidance.catalog()
    assert set(catalog["fields"]) == {
        "genre", "art_style", "category", "platform", "base_model", "style_lora",
    }
    assert all(o["resolution"] for o in catalog["fields"]["platform"])
    assert all(o["default_size_m"] for o in catalog["fields"]["category"])
    assert all(o["default_weight"] for o in catalog["fields"]["style_lora"])
    json.dumps(catalog)


# --- model selection --------------------------------------------------------


def test_base_model_defaults_and_is_always_present():
    # The worker must never have to guess a checkpoint, so unlike genre or
    # category this key is written even when the request omits it.
    assert guidance.normalize({})["base_model"] == models.DEFAULT_BASE_MODEL


def test_base_model_is_carried_through():
    assert guidance.normalize({"base_model": "sdxl"})["base_model"] == "sdxl"


def test_unknown_base_model_is_rejected():
    with pytest.raises(ValueError, match="unknown base_model"):
        guidance.normalize({"base_model": "midjourney"})


def test_unknown_style_lora_is_rejected():
    with pytest.raises(ValueError, match="unknown style_lora"):
        guidance.normalize({"style_lora": "nope"})


def test_style_lora_is_absent_unless_chosen():
    params = guidance.normalize({})
    assert "style_lora" not in params
    # A weight with no LoRA would read as "a LoRA at 0.9" on rerun.
    assert "lora_weight" not in params


def test_style_lora_brings_its_own_default_weight():
    params = guidance.normalize({"style_lora": "ps1"})
    assert params["style_lora"] == "ps1"
    assert params["lora_weight"] == models.STYLE_LORAS["ps1"].default_weight


def test_bg_removal_defaults_to_auto_and_rejects_unknown():
    assert guidance.normalize({})["bg_removal"] == "auto"
    assert guidance.normalize({"bg_removal": "birefnet"})["bg_removal"] == "birefnet"
    with pytest.raises(ValueError):
        guidance.normalize({"bg_removal": "magic"})


def test_explicit_lora_weight_overrides_the_default():
    assert guidance.normalize({"style_lora": "ps1", "lora_weight": 0.5})["lora_weight"] == 0.5


@pytest.mark.parametrize("weight", [-0.1, 1.6, 99])
def test_lora_weight_out_of_range_is_rejected(weight):
    with pytest.raises(ValueError, match="lora_weight must be between"):
        guidance.normalize({"style_lora": "ps1", "lora_weight": weight})


def test_lora_weight_must_be_a_number():
    with pytest.raises(ValueError, match="lora_weight must be a number"):
        guidance.normalize({"style_lora": "ps1", "lora_weight": "strong"})


def test_model_selection_never_leaks_into_the_prompt():
    """A checkpoint name is not creative direction. The LoRA's trigger words
    are prepended in text2image.generate(), next to PROMPT_TEMPLATE, so they
    must not appear here either."""
    params = guidance.normalize({"base_model": "sdxl", "style_lora": "render3d"})
    prompt = guidance.compose_prompt("a barrel", params)
    # The default platform still contributes its fragment; nothing model-facing does.
    assert prompt == guidance.compose_prompt("a barrel", {"platform": params["platform"]})
    for token in ("sdxl", "render3d", "3d render", "Hyper"):
        assert token.lower() not in prompt.lower()
