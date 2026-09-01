"""A style LoRA's measured strength has to reach the form.

``StyleLora.default_weight`` is measured per adapter and
``guidance.normalize`` applies it only when the caller omits ``lora_weight``.
The 2D pane always sends a number, so without a seed at selection time every
adapter ran at the flat ``DEFAULT_LORA_WEIGHT`` -- which for pixel-art-klein
is ~14x the top of its usable band, i.e. black frames.
"""

from __future__ import annotations

from warlock import models
from warlock.studio.panes import settings_2d


def test_each_adapter_reports_its_own_tuned_weight():
    assert settings_2d.lora_default_weight("pixelxl") == 1.2
    assert settings_2d.lora_default_weight("pixelklein") == 0.0625


def test_no_selection_falls_back_to_the_flat_default():
    assert settings_2d.lora_default_weight("") == models.DEFAULT_LORA_WEIGHT
    assert settings_2d.lora_default_weight("not-an-adapter") == models.DEFAULT_LORA_WEIGHT


def test_every_registered_adapter_is_reachable_through_the_helper():
    for key, spec in models.STYLE_LORAS.items():
        assert settings_2d.lora_default_weight(key) == spec.default_weight


def test_the_klein_default_sits_inside_its_usable_band():
    """The band is measured in docs/measurements/2026-08-10-pixel-art-klein.md;
    the flat default does not sit in it, which is the whole defect."""
    assert 0.02 <= settings_2d.lora_default_weight("pixelklein") <= 0.08
    assert not 0.02 <= models.DEFAULT_LORA_WEIGHT <= 0.08


def test_the_reseed_writes_the_tuned_weight_only_when_the_selection_changed():
    """The rule itself, called directly now that both forms share it."""
    form = {"style_lora": "pixelklein", "lora_weight": 0.9}
    settings_2d.reseed_lora_weight(form, "")
    assert form["lora_weight"] == 0.0625
    # A redraw with no change must not stomp a weight the user set by hand.
    form["lora_weight"] = 0.05
    settings_2d.reseed_lora_weight(form, "pixelklein")
    assert form["lora_weight"] == 0.05


def test_every_form_that_edits_a_lora_seeds_its_weight():
    """The seeds live in draw paths, so this reads the source rather than a GL
    frame. One form now that the profile editor is gone -- but still written as
    a list, because the defect it guards was a *second* form shipping 0.9
    against an adapter whose band is 0.02-0.08, silently, with no error.
    """
    import inspect

    for func in (settings_2d._lora,):
        assert "reseed_lora_weight" in inspect.getsource(func), func.__name__
