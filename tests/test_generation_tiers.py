"""The Fast/Quality tier, and what picking Fast actually costs.

Both tiers named ``sdxl_cfg`` until 2026-08-29 -- same checkpoint, same steps,
same resolution -- so the control changed the recipe's *name* and nothing a
user could see. These are the pins that would have caught that, plus the pins
on the consequences of fixing it: Fast is ``sdxl`` (four steps, guidance 0), so
it can run neither a ControlNet nor a negative prompt, and both of those have
to be said rather than silently dropped.

Every assertion here is about *plumbing* -- which recipe resolves, what it
declares, what the door refuses. Nothing here looks at a pixel, and nothing
here would be made true by a fake.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock import generation, models
from warlock.studio.panes import settings_2d


def _request(**over):
    base = {"generation_type": "image", "prompt": "a wooden crate"}
    base.update(over)
    return generation.GenerationRequest(**base)


def _ctx():
    """A pane ctx whose config is ``None``.

    ``generation._present`` answers True for a ``None`` config -- "no config,
    no opinion about what is downloaded" -- which is exactly what these tests
    want: the resolver picks the recipe the *registry* would choose, rather
    than refusing because a throwaway ``WARLOCK_HOME`` holds no weights.
    """
    return SimpleNamespace(svc=SimpleNamespace(config=None))


# --- the two tiers are two pictures -------------------------------------------


def test_fast_and_quality_resolve_to_different_recipes():
    fast = generation.resolve_recipe(_request(quality="fast"), None)
    quality = generation.resolve_recipe(_request(quality="quality"), None)
    assert fast is not None and quality is not None
    assert fast.recipe.key != quality.recipe.key
    assert fast.base_model != quality.base_model


def test_no_two_tiers_of_one_generation_type_name_the_same_base():
    """The pin the original defect would have failed.

    A tier is a *choice*, so two tiers offered for the same asset that load the
    same checkpoint at the same size are a control with nothing behind it. This
    compares the resolved run, not the recipe key: two keys were exactly what
    the defect had.
    """
    for kind in generation.GENERATION_TYPES:
        # The *highest-ranked* recipe per tier: that is the one automatic
        # routing picks, and the lower-ranked siblings are a fallback order
        # rather than a choice offered to anybody.
        by_tier: dict[str, tuple[str, tuple[int, int]]] = {}
        for recipe in sorted(generation.RECIPES, key=lambda r: r.rank):
            if kind in recipe.generation_types:
                by_tier[recipe.quality] = (recipe.base_model, recipe.working_resolution)
        runs = list(by_tier.values())
        assert len(set(runs)) == len(runs), (
            f"{kind}: two quality tiers resolve to the same run {runs}"
        )


def test_fast_names_the_hyper_sd_recipe_over_the_shared_sdxl_weights():
    fast = generation.resolve_recipe(_request(quality="fast"), None)
    assert fast.base_model == "sdxl"
    spec = models.BASE_MODELS["sdxl"]
    # The same base weights as Quality, run differently -- which is why Fast
    # costs no second 7 GB download.
    assert spec.dir_name == models.BASE_MODELS["sdxl_cfg"].dir_name
    assert spec.steps < models.BASE_MODELS["sdxl_cfg"].steps
    assert spec.commercial


def test_both_tiers_say_what_they_trade():
    keyed = {r.key: r for r in generation.RECIPES}
    assert keyed["image_fast"].note and keyed["image_quality"].note
    note = keyed["image_fast"].note.lower()
    assert "four steps" in note
    assert "structure control" in note
    assert "negative prompt" in note


# --- the negative prompt is inert on Fast, and declared inert ------------------


def test_the_fast_recipe_reports_no_negative_prompt_support():
    fast = generation.resolve_recipe(_request(quality="fast"), None)
    assert not fast.recipe.supports_negative_prompt
    # And the declaration agrees with the registry's own derivation rather
    # than being an independent opinion about the same checkpoint.
    assert fast.base_model not in models.cfg_bases()


def test_the_form_does_not_offer_avoid_under_fast():
    form = {"asset_type": "image", "generation_type": "image", "quality": "fast"}
    assert settings_2d._negative_supported(_ctx(), form) is False
    form["quality"] = "quality"
    assert settings_2d._negative_supported(_ctx(), form) is True


def test_klein_distilled_accepts_a_saved_negative_prompt():
    """A prior full-CFG brief must not prevent a Klein run from starting."""
    request = _request(
        model_mode="advanced",
        model_override="flux_klein_distilled",
        negative_prompt="blurry, watermark",
    )
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None
    assert generation.validate_request(request, resolved) == []
    assert generation.effective_negative_prompt(request, resolved) == ""
    assert generation.request_to_legacy(request, resolved)["negative_prompt"] is None


def test_avoid_text_typed_under_quality_is_cleared_by_switching_to_fast():
    """The dead end the clear exists to prevent.

    The field is hidden under Fast, and ``validate_request`` refuses a request
    that carries text it cannot use -- so without this the Generate button
    refuses on a control that is off screen.
    """
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "quality": "fast",
        "negative_prompt": "blurry, watermark",
    }
    cleared = settings_2d.clear_for_tier(_ctx(), form)
    assert form["negative_prompt"] == ""
    assert any("Avoid" in one for one in cleared)


# --- Fast and structure control are incompatible ------------------------------


def test_fast_plus_structure_control_is_refused():
    request = _request(quality="fast", structure_control="canny")
    resolved = generation.resolve_recipe(request, None)
    issues = generation.validate_request(request, resolved)
    assert [i.field for i in issues] == ["quality"]
    message = issues[0].message
    assert "ControlNet" in message
    # A sentence the user can act on, naming a control automatic routing draws.
    assert "Quality" in message


def test_the_refusal_names_the_model_when_the_user_picked_it():
    request = _request(
        quality="fast",
        structure_control="canny",
        model_mode="advanced",
        model_override="sdxl",
    )
    resolved = generation.resolve_recipe(request, None)
    issues = generation.validate_request(request, resolved)
    assert [i.field for i in issues] == ["base_model"]


def test_quality_plus_structure_control_is_accepted():
    request = _request(quality="quality", structure_control="canny")
    resolved = generation.resolve_recipe(request, None)
    assert generation.validate_request(request, resolved) == []


def test_the_pane_hides_the_structure_picker_under_fast():
    form = {"asset_type": "image", "generation_type": "image", "quality": "fast"}
    note = settings_2d.recipe_structure_note(_ctx(), form)
    assert note is not None and "ControlNet" in note
    form["quality"] = "quality"
    assert settings_2d.recipe_structure_note(_ctx(), form) is None


def test_switching_to_fast_clears_a_structure_control():
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "quality": "fast",
        "ref_path": "ref.png",
        "control": "canny",
    }
    cleared = settings_2d.clear_for_tier(_ctx(), form)
    assert form["control"] == ""
    assert any("structure control" in one for one in cleared)


def test_advanced_mode_leaves_the_tier_clear_alone():
    """``clear_unusable`` owns that half: under Advanced the tier does not pick
    the checkpoint, so it cannot strand anything."""
    form = {
        "asset_type": "image",
        "generation_type": "image",
        "quality": "fast",
        "model_mode": "advanced",
        "base_model": "sdxl_cfg",
        "ref_path": "ref.png",
        "control": "canny",
    }
    assert settings_2d.clear_for_tier(_ctx(), form) == []
    assert form["control"] == "canny"


def test_a_form_carries_its_control_into_the_request_only_with_a_reference():
    form = {"asset_type": "image", "generation_type": "image", "control": "canny"}
    assert generation.request_from_legacy(form).structure_control == ""
    form["ref_path"] = "ref.png"
    assert generation.request_from_legacy(form).structure_control == "canny"


# --- img2img needs the SDXL family ---------------------------------------------


def _non_sdxl_base() -> str:
    return next(
        key for key, spec in models.BASE_MODELS.items()
        if spec.family != models.FAMILY_SDXL
    )


def test_validate_request_refuses_init_image_on_a_non_sdxl_base():
    """The 2026-09-05 audit, finding create-04: nothing before the queue door
    checked ``init_image`` against the base model's family. A request built
    under Advanced with a non-SDXL base and img2img ticked used to pass every
    pre-submit check here and fail only at ``guidance.normalize``, late, with
    the checkpoint already resident.
    """
    request = _request(
        model_mode="advanced",
        model_override=_non_sdxl_base(),
        references=("some/ref.png",),
        reference_mode="single",
        init_image=True,
        init_strength=0.5,
    )
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None
    issues = generation.validate_request(request, resolved)
    assert [i.field for i in issues] == ["init_image"]
    # The same two words the probe checks for in guidance.normalize's own
    # refusal, so the early and late refusals agree about what happened.
    assert "image" in issues[0].message and "start" in issues[0].message


def test_an_sdxl_base_with_init_image_is_accepted():
    request = _request(
        model_mode="advanced",
        model_override="sdxl_cfg",
        references=("some/ref.png",),
        reference_mode="single",
        init_image=True,
        init_strength=0.5,
    )
    resolved = generation.resolve_recipe(request, None)
    assert resolved is not None
    assert generation.validate_request(request, resolved) == []


def test_capability_controls_reports_img2img_only_for_the_sdxl_family():
    non_sdxl = generation.resolve_recipe(
        _request(model_mode="advanced", model_override=_non_sdxl_base()), None
    )
    sdxl = generation.resolve_recipe(
        _request(model_mode="advanced", model_override="sdxl_cfg"), None
    )
    assert non_sdxl is not None and sdxl is not None
    assert generation.capability_controls(_request(), non_sdxl)["img2img"] is False
    assert generation.capability_controls(_request(), sdxl)["img2img"] is True


# --- what the tier needs downloaded ---------------------------------------------


def test_fast_is_satisfied_by_the_base_row_alone():
    fast = next(r for r in generation.RECIPES if r.key == "image_fast")
    assert fast.required_downloads == ("base:sdxl",)
    resolved = generation.resolve_recipe(
        _request(quality="fast"), None, installed=set(fast.required_downloads)
    )
    assert resolved is not None and resolved.recipe.key == "image_fast"


def test_the_one_row_covers_the_hyper_sd_lora_too():
    """Why there is no second key: ``sdxl``'s own fetch tuple carries both the
    shared SDXL 1.0 base and the 0.8 GB Hyper-SD adapter, so naming the base
    row names the whole download."""
    from warlock import fetch

    entry = fetch.find("base:sdxl")
    assert entry is not None
    repos = {one.repo_id for one in entry.spec.fetch}
    assert "stabilityai/stable-diffusion-xl-base-1.0" in repos
    assert "ByteDance/Hyper-SD" in repos


@pytest.mark.parametrize("installed", [set(), {"base:sdxl_cfg"}])
def test_a_host_without_the_fast_row_does_not_qualify_it(installed):
    assert generation.resolve_recipe(_request(quality="fast"), None, installed=installed) is None
