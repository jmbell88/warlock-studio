from __future__ import annotations

import pytest

from warlock import guidance, models


def test_defaults_fill_in_when_nothing_is_chosen():
    params = guidance.normalize({})
    assert params["platform"] == guidance.DEFAULT_PLATFORM
    assert params["resolution"] == guidance.PLATFORMS[guidance.DEFAULT_PLATFORM].resolution
    assert params["size_m"] == guidance.DEFAULT_SIZE_M
    # Optional fields stay absent rather than being stored as empty strings.
    assert "style_lora" not in params
    assert "ip_adapter" not in params


def test_platform_supplies_the_geometry_resolution():
    assert guidance.normalize({"platform": "2d"})["resolution"] == 512
    assert guidance.normalize({"platform": "3d"})["resolution"] == 1024


def test_explicit_resolution_overrides_the_platform_preset():
    params = guidance.normalize({"platform": "2d", "resolution": 1536})
    assert params["resolution"] == 1536
    assert params["platform"] == "2d"


@pytest.mark.parametrize("field", ["platform", "base_model", "style_lora"])
def test_unknown_values_are_rejected(field):
    with pytest.raises(ValueError, match=field):
        guidance.normalize({field: "nonsense"})


@pytest.mark.parametrize("value", ["", None])
def test_blank_means_unspecified_not_invalid(value):
    assert guidance.normalize({"platform": value}) == guidance.normalize({})


def test_size_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="between"):
        guidance.normalize({"size_m": 0.0})
    with pytest.raises(ValueError, match="between"):
        guidance.normalize({"size_m": 1000})


def test_non_numeric_size_is_rejected():
    with pytest.raises(ValueError, match="number"):
        guidance.normalize({"size_m": "big"})


@pytest.mark.parametrize(
    ("stored", "canonical"),
    [("mobile", "2d"), ("desktop", "3d"), ("hero", "3d")],
)
def test_legacy_platform_keys_normalize_to_the_new_ones(stored, canonical):
    params = guidance.normalize({"platform": stored})
    assert params["platform"] == canonical
    assert params["resolution"] == guidance.PLATFORMS[canonical].resolution


# --- taxonomy-retirement tolerance -------------------------------------------
#
# The twelve taxonomy tables (and framing) were retired on 2026-08-17
# (docs/measurements/2026-08-17-taxonomy-retirement.md). Every job row already
# on disk still carries those keys, and rerun/promotion re-run normalize() over
# stored params -- so a stale key must be *ignored*, never refused.


def test_stale_taxonomy_keys_in_old_params_normalize_cleanly():
    old = {
        "art_style": "nes",
        "framing": "front_ortho",
        "category": "weapon",
        "genre": "fantasy",
        "material": "steel",
        "platform": "2d",
    }
    params = guidance.normalize(old)
    assert params["platform"] == "2d"
    for retired in ("art_style", "framing", "category", "genre", "material"):
        assert retired not in params


def test_a_stale_category_no_longer_supplies_a_size():
    # The category default died with CATEGORIES; the fallback is the constant.
    params = guidance.normalize({"category": "character"})
    assert params["size_m"] == guidance.DEFAULT_SIZE_M


def test_compose_prompt_ignores_stale_taxonomy_values():
    """Params can come from a job row written before the retirement; the
    fragments are simply gone rather than the job failing."""
    assert guidance.compose_prompt("a barrel", {"genre": "fantasy"}) == "a barrel"
    assert (
        guidance.compose_prompt("a barrel", {"art_style": "nes", "framing": "front_ortho"})
        == "a barrel"
    )


def test_compose_prompt_with_no_guidance_is_just_the_prompt():
    assert guidance.compose_prompt("  a barrel  ", {}) == "a barrel"


def test_compose_prompt_field_subset_is_inert_while_no_field_composes():
    params = guidance.normalize({"platform": "3d"})
    assert guidance.compose_prompt("x", params) == "x"
    assert guidance.compose_prompt("x", params, fields=("platform",)) == "x"
    assert guidance.compose_prompt("x", params, fields=None) == "x"


def test_catalog_publishes_the_cfg_bases_the_ui_gates_on():
    catalog = guidance.catalog()
    assert catalog["cfg_bases"] == models.cfg_bases()


def test_catalog_covers_every_field_and_is_json_safe():
    import json

    catalog = guidance.catalog()
    assert set(catalog["fields"]) == {
        "platform", "base_model", "style_lora", "ip_adapter", "control",
    }
    assert all(o["resolution"] for o in catalog["fields"]["platform"])
    assert all(o["default_weight"] for o in catalog["fields"]["style_lora"])
    json.dumps(catalog)


def test_form_fields_covers_every_table():
    assert set(guidance.form_fields()) == {
        "platform", "base_model", "style_lora", "ip_adapter", "control",
    }


# --- model selection --------------------------------------------------------


def test_base_model_defaults_and_is_always_present():
    # The worker must never have to guess a checkpoint, so unlike the optional
    # selections this key is written even when the request omits it.
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


def test_bg_removal_defaults_to_birefnet_and_rejects_unknown():
    assert guidance.normalize({})["bg_removal"] == "birefnet"
    assert guidance.normalize({"bg_removal": "birefnet"})["bg_removal"] == "birefnet"
    with pytest.raises(ValueError):
        guidance.normalize({"bg_removal": "magic"})


def test_the_bg_removal_default_is_gated_on_the_weights_being_on_disk(tmp_path):
    """The default matte. birefnet is the learned one and the only thing the
    2026-08-07 review found any signal for -- 3 accepts in 83, all of them
    birefnet, ``auto`` 0 for 80 -- but ``auto`` is still the right answer on a
    host that never downloaded ``birefnet.gguf``, where the server has nothing
    to load and falls back to a threshold cutout anyway.
    """
    assert guidance.default_bg_removal(tmp_path) == "auto"
    (tmp_path / guidance.BIREFNET_WEIGHTS).write_bytes(b"")
    assert guidance.default_bg_removal(tmp_path) == "birefnet"


def test_the_gate_and_the_doctor_row_name_the_same_file(tmp_path):
    """Two readers of one fact. The doctor reports the weights missing and the
    gate silently picks ``auto`` for it; a copy of the filename that drifted
    would let the app choose a matte the doctor says is unavailable."""
    from warlock import doctor

    config = _doctor_config(tmp_path)
    config.data_dir.mkdir(parents=True)  # run_checks measures free space on it
    config.trellis_models_dir.mkdir()
    before = _birefnet_row(doctor.run_checks(config))
    assert before.ok is False
    assert guidance.default_bg_removal(config.trellis_models_dir) == "auto"

    (config.trellis_models_dir / guidance.BIREFNET_WEIGHTS).write_bytes(b"")
    assert _birefnet_row(doctor.run_checks(config)).ok is True
    assert guidance.default_bg_removal(config.trellis_models_dir) == "birefnet"


def _doctor_config(tmp_path):
    from warlock.config import Config

    return Config(
        data_dir=tmp_path / "data",
        trellis_server_exe=tmp_path / "missing.exe",
        trellis_models_dir=tmp_path / "models",
    )


def _birefnet_row(checks):
    return next(c for c in checks if c.name.startswith("trellis: birefnet"))


def test_bg_removal_can_be_defaulted_explicitly_by_the_caller():
    """``normalize`` is pure and has no Config, so the gate is a value passed
    in. An explicit ``bg_removal`` still wins over it."""
    assert guidance.normalize({}, bg_default="auto")["bg_removal"] == "auto"
    assert (
        guidance.normalize({"bg_removal": "threshold"}, bg_default="auto")["bg_removal"]
        == "threshold"
    )


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
    assert prompt == "a barrel"
    for token in ("sdxl", "render3d", "3d render", "Hyper"):
        assert token.lower() not in prompt.lower()


def test_negative_prompt_defaults_and_is_length_capped():
    assert guidance.normalize({})["negative_prompt"] == guidance.DEFAULT_NEGATIVE_PROMPT
    assert guidance.normalize({"negative_prompt": " smooth "})["negative_prompt"] == "smooth"
    with pytest.raises(ValueError):
        guidance.normalize({"negative_prompt": "x" * 1001})


def test_explicit_empty_negative_prompt_means_none_not_the_default():
    # Missing key -> default; explicit "" -> the user opted out entirely.
    # A `if not negative:` refactor would silently collapse this back to the
    # default, so this is pinned separately from the defaults/strip/cap test.
    assert guidance.normalize({"negative_prompt": ""})["negative_prompt"] == ""


def test_no_guidance_fragment_ever_says_pixel():
    """The pixel-art profile is a LoRA trigger, not a prompt fragment.

    "pixel" in a fragment would fire on every job carrying the field that owns
    it, whether or not the pixel LoRA is loaded.
    """
    for field, table in guidance._OPTION_TABLES.items():
        for option in table.values():
            assert "pixel" not in option.prompt.lower(), f"{field}/{option.key}"


def test_no_option_is_hidden_and_no_default_is():
    """The ``hidden`` mechanism survives the retirement with no instance; a
    default the catalogue does not offer would be a form that opens on a value
    the user cannot choose again after changing it."""
    hidden = {
        (field, opt.key)
        for field, table in guidance._OPTION_TABLES.items()
        for opt in table.values()
        if opt.hidden
    }
    assert hidden == set()
    for field, table in guidance._OPTION_TABLES.items():
        default = guidance.catalog()["defaults"].get(field)
        if default:
            assert not table[default].hidden, f"{field} defaults to a hidden option"


def test_an_unknown_conditioning_selection_is_rejected():
    with pytest.raises(ValueError, match="ip_adapter"):
        guidance.normalize({"ip_adapter": "nope"})
    with pytest.raises(ValueError, match="control"):
        guidance.normalize({"control": "nope"})


def test_a_pipeline_fed_control_is_refused_at_the_door():
    """"depth" is registered for the re-texture pipeline, which renders its
    own hint from the mesh. A text job has no mesh to render one from, and
    accepting it here would fail at write_hint with the checkpoint already in
    VRAM. Belt to catalog()'s braces: the combo never offers it, but params
    arrive from more places than the combo."""
    with pytest.raises(ValueError, match="control"):
        guidance.normalize({"control": "depth", "base_model": "sdxl_cfg"})


def test_a_conditioning_scale_is_absent_without_its_selection():
    """Same rule as style_lora/lora_weight: a scale with nothing to scale
    reads as a live setting when the params are rerun."""
    params = guidance.normalize({"ip_scale": 0.9, "control_scale": 1.0})
    assert "ip_scale" not in params
    assert "control_scale" not in params
    assert "control_end" not in params


def test_a_conditioning_scale_is_carried_with_its_selection():
    params = guidance.normalize(
        {"ip_adapter": "plus", "ip_scale": 0.9, "base_model": "sdxl_cfg",
         "control": "canny", "control_scale": 1.0, "control_end": 0.5}
    )
    assert params["ip_scale"] == 0.9
    assert params["control_scale"] == 1.0
    assert params["control_end"] == 0.5


def test_conditioning_scales_default_from_their_registry_entry():
    params = guidance.normalize({"ip_adapter": "plus", "base_model": "sdxl_cfg",
                                 "control": "canny"})
    assert params["ip_scale"] == models.IP_ADAPTERS["plus"].default_scale
    assert params["control_scale"] == models.CONTROLNETS["canny"].default_scale
    assert params["control_end"] == models.CONTROLNETS["canny"].default_end


@pytest.mark.parametrize(
    "field,value",
    [("ip_scale", 99), ("control_scale", -1), ("control_end", 2.5)],
)
def test_an_out_of_range_conditioning_scale_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        guidance.normalize(
            {"ip_adapter": "plus", "control": "canny", "base_model": "sdxl_cfg",
             field: value}
        )


def test_a_control_on_a_distilled_base_is_refused_by_name():
    """A ControlNet at guidance 0 on a 4-step base fights the hint instead of
    honouring it, so this is an error rather than a silent no-op -- and the
    message has to say which bases do work."""
    with pytest.raises(ValueError, match="sdxl_cfg"):
        guidance.normalize({"control": "canny", "base_model": "turbo"})


def _mismatched_pair() -> tuple[str, str]:
    """A (base_model, style_lora) the registry does not pair, by construction.

    Written out rather than "the first non-SDXL base": once every architecture
    in the registry has some adapter, "a base that takes no LoRA" stops
    existing and a next() over it raises StopIteration.
    """
    for base in models.BASE_MODELS.values():
        for lora in models.STYLE_LORAS.values():
            if not models.lora_fits(base, lora):
                return base.key, lora.key
    raise AssertionError("the registry holds only one architecture")


def test_a_cross_family_style_lora_is_refused_by_name():
    """Stronger than the ControlNet refusal above: an adapter's tensors name
    one architecture's modules, so loading it onto another raises at load time
    with the checkpoint already in VRAM."""
    base_key, lora_key = _mismatched_pair()
    with pytest.raises(ValueError) as exc:
        guidance.normalize({"style_lora": lora_key, "base_model": base_key})
    assert "style" in str(exc.value)
    # The field is the half the app reads, and which control it names depends
    # on where the remedy is: when some adapter fits this base, the base is
    # fine and the style is wrong; when none does, the other way round.
    fitting = models.loras_by_base()[base_key]
    assert exc.value.field == ("style_lora" if fitting else "base_model")
    # And the same base without a style is fine -- the refusal is about the
    # pairing, not about the checkpoint.
    assert guidance.normalize({"base_model": base_key})["base_model"] == base_key


def test_a_base_no_adapter_fits_is_refused_naming_the_base(monkeypatch):
    """The second branch. Every base in the registry takes some style LoRA
    today, so the "none fits" answer is staged rather than found -- the
    branch exists for the registry of tomorrow and used to skip on every
    run, which is a test that guards nothing."""
    base_key, lora_key = _mismatched_pair()
    monkeypatch.setattr(models, "loras_by_base", lambda: {base_key: []})
    with pytest.raises(ValueError, match="cannot take a style LoRA") as exc:
        guidance.normalize({"style_lora": lora_key, "base_model": base_key})
    assert exc.value.field == "base_model"


def test_lora_bases_reaches_the_ui_through_the_catalog():
    catalog = guidance.catalog()
    assert catalog["lora_bases"] == models.lora_bases()


def test_loras_by_base_reaches_the_ui_through_the_catalog():
    # What the picker lists, beside what greys it.
    assert guidance.catalog()["loras_by_base"] == models.loras_by_base()


def test_the_composed_prompt_never_sees_a_conditioning_field():
    """The bit-identity rule at the prompt layer: conditioning changes what
    the pipeline is handed, never a single character of the text."""
    plain = guidance.normalize({"base_model": "sdxl_cfg"})
    conditioned = guidance.normalize(
        {"base_model": "sdxl_cfg", "ip_adapter": "plus",
         "ip_scale": 1.1, "control": "canny", "control_scale": 0.4, "control_end": 0.9}
    )
    assert guidance.compose_prompt("a crate", plain) == guidance.compose_prompt(
        "a crate", conditioned
    )


def test_an_ip_adapter_on_a_non_sdxl_base_is_refused_at_the_door():
    """The two other conditioning halves are gated by base family here; the
    IP-Adapter was not, and the only guard was ``Text2Image._conditioned``,
    which fires with the checkpoint already resident. Same rule, same door."""
    with pytest.raises(ValueError, match="ip_adapter") as excinfo:
        guidance.normalize({"base_model": "flux_klein", "ip_adapter": "plus"})
    assert excinfo.value.field == "ip_adapter"
