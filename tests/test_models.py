"""Registry integrity. Every entry here is a manual download the user has to
perform from a string this module prints, so a typo in a filename or a repo id
is not a code bug the tests would otherwise catch -- it surfaces as a download
that lands in the wrong place, or a LoRA that silently never loads."""

from __future__ import annotations

import pytest

from warlock import models


def test_keys_match_their_table_entries():
    for key, spec in models.BASE_MODELS.items():
        assert spec.key == key
    for key, lora in models.STYLE_LORAS.items():
        assert lora.key == key


def test_base_and_style_keys_do_not_collide():
    # They share the guidance _lookup namespace and, more importantly, LoRA
    # adapter names on one pipeline.
    assert not set(models.BASE_MODELS) & set(models.STYLE_LORAS)
    assert models.BASE_LORA_ADAPTER not in models.STYLE_LORAS


def test_default_base_model_exists():
    assert models.DEFAULT_BASE_MODEL in models.BASE_MODELS


@pytest.mark.parametrize("key", sorted(models.BASE_MODELS))
def test_base_model_is_downloadable_and_runnable(key):
    spec = models.BASE_MODELS[key]
    assert spec.download.strip(), "a missing model must be able to say how to get it"
    assert "hf download" in spec.download
    assert spec.dir_name in spec.download, "the command must land in the dir we look in"
    assert spec.steps >= 1
    assert spec.image_size in (512, 768, 1024)
    assert spec.guidance_scale >= 0.0


def test_base_lora_download_covers_the_lora_too():
    # A base model whose step-distillation LoRA is missing raises rather than
    # producing noise, so its download string has to fetch both halves.
    for spec in models.BASE_MODELS.values():
        if spec.base_lora is not None:
            assert spec.base_lora in spec.download


@pytest.mark.parametrize("key", sorted(models.STYLE_LORAS))
def test_style_lora_is_downloadable(key):
    lora = models.STYLE_LORAS[key]
    assert lora.filename.endswith(".safetensors")
    assert lora.filename in lora.download
    # Everything resolves against <t2i_model_root>/loras, so the command must
    # put the file there and not in a per-repo subdirectory.
    assert "--local-dir models/loras" in lora.download
    assert lora.trigger, "the trained trigger words are what make the LoRA fire"
    assert models.LORA_WEIGHT_MIN <= lora.default_weight <= models.LORA_WEIGHT_MAX


def test_catalog_is_json_safe_and_covers_both_tables():
    import json

    catalog = models.catalog()
    assert {o["key"] for o in catalog["base_model"]} == set(models.BASE_MODELS)
    assert {o["key"] for o in catalog["style_lora"]} == set(models.STYLE_LORAS)
    json.dumps(catalog)
