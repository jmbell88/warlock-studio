"""What the 2D pane says about a control that cannot act.

Both notes are pure functions of (catalog, form) so the wording is assertable
without a GL context -- the pane's draw code only decides where to put them.
"""

from __future__ import annotations

from types import SimpleNamespace

from warlock import guidance, models
from warlock.studio.panes import settings_2d


def _ctx():
    catalog = guidance.catalog()
    return SimpleNamespace(
        guidance=catalog,
        base_models=[(m["key"], m["label"]) for m in catalog["fields"]["base_model"]],
    )


def test_a_cfg_base_gets_no_negative_prompt_note():
    form = {"base_model": models.cfg_bases()[0]}
    assert settings_2d.negative_prompt_note(_ctx(), form) is None


def test_a_distilled_base_is_told_the_negative_prompt_is_inert():
    note = settings_2d.negative_prompt_note(_ctx(), {"base_model": "turbo"})
    assert note is not None
    assert "no effect" in note


def test_the_note_names_a_model_the_user_could_switch_to():
    # A refusal that doesn't say what would work is a dead end -- the label,
    # not the key, because the key is not what the picker shows.
    note = settings_2d.negative_prompt_note(_ctx(), {"base_model": "turbo"})
    label = models.BASE_MODELS[models.cfg_bases()[0]].label
    assert label in note


def test_an_unset_base_is_treated_as_the_default_which_is_distilled():
    # "" means "use the configured default", which is turbo -- so the field is
    # inert and saying nothing would be the same silence the note replaces.
    assert settings_2d.negative_prompt_note(_ctx(), {"base_model": ""}) is not None


def test_a_controlnet_base_gets_no_structure_note():
    form = {"base_model": models.controlnet_bases()[0]}
    assert settings_2d.structure_note(_ctx(), form) is None


def test_the_structure_note_names_the_bases_that_can_run_one():
    note = settings_2d.structure_note(_ctx(), {"base_model": "turbo"})
    assert note is not None
    for key in models.controlnet_bases():
        assert models.BASE_MODELS[key].label in note
