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


def test_pixel_base_reuses_sdxl_weights_with_lcm_not_hyper():
    """Same weights as the other SDXL 1.0 entries -- it exists for a sampler
    contract (LCM at 8 steps, guidance 1.0), which is the pixel-art-xl author's
    documented recipe and the arm the preset names until a bench says otherwise.
    """
    pixel, hyper = models.BASE_MODELS["pixel"], models.BASE_MODELS["sdxl"]
    assert pixel.dir_name == hyper.dir_name
    assert pixel.scheduler == "lcm"
    assert pixel.base_lora is not None and "lcm" in pixel.base_lora.lower()
    # <= 1.0 keeps it out of cfg_bases(): there is no negative branch to steer.
    assert pixel.guidance_scale <= 1.0
    assert "pixel" not in models.cfg_bases()
    assert not pixel.controlnet


def test_pixel_lora_weight_matches_the_author_recipe():
    lora = models.STYLE_LORAS["pixelxl"]
    assert lora.default_weight == 1.2
    assert lora.default_weight != models.DEFAULT_LORA_WEIGHT


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


def test_cfg_bases_are_exactly_the_ones_that_encode_a_negative_prompt():
    # text2image only encodes the negative prompt when guidance_scale > 1.0,
    # so this list is what the UI may present the field as live for.
    bases = models.cfg_bases()
    assert bases
    assert all(models.BASE_MODELS[b].guidance_scale > 1.0 for b in bases)
    assert all(
        m.key in bases for m in models.BASE_MODELS.values() if m.guidance_scale > 1.0
    )
    assert "turbo" not in bases


def test_catalog_stays_json_safe():
    import json

    json.dumps(models.catalog())


def test_every_named_scheduler_is_one_text2image_can_build():
    """A BaseModel naming a scheduler _scheduler() does not know fails at load
    time with the weights already in VRAM, which is the most expensive place to
    find out. diffusers is an optional extra, so build the ones that are
    nameable and only assert the name is known when it is absent."""
    pytest.importorskip("diffusers")
    from diffusers import (
        DDIMScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        LCMScheduler,
    )

    from warlock.pipelines import text2image

    base = DDIMScheduler()
    built = {
        m.scheduler: text2image._scheduler(m.scheduler, base)
        for m in models.BASE_MODELS.values()
        if m.scheduler
    }
    assert isinstance(built["lcm"], LCMScheduler)
    assert built["ddim_trailing"].config.timestep_spacing == "trailing"
    # Both distillation arms need trailing spacing, and neither says so at
    # runtime -- the failure is a washed-out image, not an error.
    assert isinstance(built["euler_trailing"], EulerDiscreteScheduler)
    assert built["euler_trailing"].config.timestep_spacing == "trailing"
    assert isinstance(built["dpm_karras"], DPMSolverMultistepScheduler)
    assert built["dpm_karras"].config.use_karras_sigmas
    assert built["dpm_karras"].config.algorithm_type == "dpmsolver++"
    assert isinstance(built["deis"], DEISMultistepScheduler)


@pytest.mark.parametrize("key", sorted(models.BASE_MODELS))
def test_family_is_one_text2image_can_sample(key):
    """The mirror of the scheduler test above: a family with no sample path
    fails at generate time with the checkpoint already loaded."""
    assert models.BASE_MODELS[key].family in models.FAMILIES


@pytest.mark.parametrize("key", sorted(models.BASE_MODELS))
def test_residency_is_one_load_knows_and_costs_something(key):
    spec = models.BASE_MODELS[key]
    assert spec.residency in (models.RESIDENT, models.OFFLOAD)
    assert spec.vram_gib > 0.0


@pytest.mark.parametrize("key", sorted(models.BASE_MODELS))
def test_a_probe_path_agrees_with_the_variant(key):
    """A probe naming an fp16 shard on a spec that loads without a variant --
    or the reverse -- is a doctor row that is red on a complete download."""
    spec = models.BASE_MODELS[key]
    for rel in spec.probe:
        assert rel.endswith(".safetensors")
        if spec.variant is None:
            assert "fp16" not in rel
        # A probe path must be relative and POSIX-shaped: it is joined onto the
        # model dir, and a backslash would never resolve.
        assert not rel.startswith("/") and "\\" not in rel


def test_lora_bases_are_exactly_the_sdxl_family():
    bases = models.lora_bases()
    assert bases
    assert all(models.BASE_MODELS[b].family == models.FAMILY_SDXL for b in bases)
    assert all(
        m.key in bases
        for m in models.BASE_MODELS.values()
        if m.family == models.FAMILY_SDXL
    )
    # The registry's whole point here is that it now holds more than one
    # architecture; if it did not, none of the family plumbing is exercised.
    assert set(bases) != set(models.BASE_MODELS)


def test_tile_bases_are_a_separate_question_with_the_same_answer_today():
    # Same list, deliberately not the same function -- see models.tile_bases.
    assert models.tile_bases() == models.lora_bases()


def test_flux_download_skips_the_redundant_single_file_checkpoint():
    """The repo ships a 7.75 GB single-file checkpoint beside the diffusers
    layout, and the --include "*.safetensors" shape would fetch it for
    nothing."""
    spec = models.BASE_MODELS["flux_klein"]
    assert '--exclude "flux-2-klein-base-4b.safetensors"' in spec.download
    assert spec.variant is None
    assert spec.family == models.FAMILY_FLUX2_KLEIN
    assert spec.residency == models.OFFLOAD
    # Undistilled, so CFG runs and the negative prompt is honoured -- which is
    # the entire reason -base was chosen over the distilled klein.
    assert spec.key in models.cfg_bases()
    # And every SDXL-only feature is off.
    assert not spec.controlnet
    assert spec.key not in models.lora_bases()


def test_lightning_is_a_second_distillation_arm_over_the_same_weights():
    lightning, hyper = models.BASE_MODELS["lightning"], models.BASE_MODELS["sdxl"]
    assert lightning.dir_name == hyper.dir_name
    assert lightning.base_lora != hyper.base_lora
    # Trailing spacing on both arms, for the same silent-failure reason.
    assert lightning.scheduler == "euler_trailing"
    assert lightning.steps == 4
    assert lightning.guidance_scale == 0.0
    assert "lightning" not in models.cfg_bases()
