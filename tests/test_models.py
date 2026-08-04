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


def test_sdxl_cfg_reuses_sdxl_weights_without_the_distillation_lora():
    """It exists for a sampler contract, not for new weights: full CFG so the
    negative prompt is encoded at all, and no Hyper-SD because a 4-step
    distilled base gives a ControlNet nothing to steer."""
    cfg, hyper = models.BASE_MODELS["sdxl_cfg"], models.BASE_MODELS["sdxl"]
    assert cfg.dir_name == hyper.dir_name
    assert cfg.base_lora is None
    assert cfg.scheduler is None
    assert cfg.guidance_scale > 1.0
    assert cfg.steps >= 20


@pytest.mark.parametrize("key", sorted(models.BASE_MODELS))
def test_a_controlnet_capable_base_runs_with_real_guidance(key):
    spec = models.BASE_MODELS[key]
    if spec.controlnet:
        assert spec.guidance_scale > 1.0


@pytest.mark.parametrize("key", sorted(models.IP_ADAPTERS))
def test_ip_adapter_is_downloadable(key):
    spec = models.IP_ADAPTERS[key]
    assert "hf download" in spec.download
    assert spec.dir_name in spec.download
    assert spec.weight_name in spec.download
    # Weights without the vision encoder load fine and then fail at the first
    # call, so the command has to fetch both.
    assert spec.image_encoder_dir in spec.download
    assert spec.download.count("hf download") == 2


@pytest.mark.parametrize("key", sorted(models.CONTROLNETS))
def test_controlnet_is_downloadable_and_names_a_real_preprocessor(key):
    from warlock.pipelines import control

    spec = models.CONTROLNETS[key]
    assert "hf download" in spec.download
    assert spec.dir_name in spec.download
    assert spec.preprocessor in control.PREPROCESSORS or spec.preprocessor == "depth"


def test_encoder_folder_is_posix_even_on_windows():
    for spec in models.IP_ADAPTERS.values():
        assert "\\" not in spec.encoder_folder


def test_conditioning_keys_do_not_collide_with_the_other_tables():
    assert not set(models.IP_ADAPTERS) & set(models.STYLE_LORAS)
    assert not set(models.CONTROLNETS) & set(models.BASE_MODELS)


def test_controlnet_bases_is_non_empty_and_real():
    bases = models.controlnet_bases()
    assert bases
    assert all(b in models.BASE_MODELS for b in bases)


def test_catalog_stays_json_safe():
    import json

    json.dumps(models.catalog())
