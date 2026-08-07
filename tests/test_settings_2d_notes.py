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


def _hint_ctx(tmp_path, prompt, doc):
    import json

    (tmp_path / "findings.json").write_text(json.dumps(doc), encoding="utf-8")
    from warlock.bench import findings as findings_lib

    findings_lib._CACHE.clear()
    return SimpleNamespace(
        svc=SimpleNamespace(config=SimpleNamespace(bench_dir=tmp_path)),
        state=SimpleNamespace(form_2d={"prompt": prompt}),
    )


def _scoped_doc(prompt):
    from warlock import vectors

    return {
        "version": 3,
        "generated": "x",
        "params": {"base_model": {"turbo": {"n": 84, "accepts": 3}}},
        "prompts": {
            vectors.prompt_hash(prompt): {
                "params": {"base_model": {"turbo": {"n": 8, "accepts": 6}}}
            }
        },
    }


def test_the_pane_scopes_its_hints_to_the_prompt_in_the_form(tmp_path):
    """TODO item 4's last mile. The pane owns the prompt, so it always knows
    its subject -- and this is the half a service-level test cannot reach: the
    form key is ``form_2d``, and reading the wrong one would silently hand
    ``prompt_hash("")`` to every lookup and pool everything forever, with no
    error and no visible difference except a wrong number.
    """
    ctx = _hint_ctx(tmp_path, "a snes rogue", _scoped_doc("a snes rogue"))
    assert settings_2d._findings_hint(ctx, "base_model", "turbo") == (
        "accept 6/8 · this subject"
    )


def test_the_pane_says_when_it_fell_back_to_every_subject(tmp_path):
    ctx = _hint_ctx(tmp_path, "a wooden crate", _scoped_doc("a snes rogue"))
    assert settings_2d._findings_hint(ctx, "base_model", "turbo") == (
        "accept 3/84 · all subjects"
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


# --- where the evidence is shown ---------------------------------------------


def test_every_mesh_setting_that_evidence_exists_for_shows_it():
    """The 3D pane hinted one control out of five, which put the evidence
    furthest from where it applies: an observation measures *geometry*, so the
    settings it speaks about most directly are exactly this pane's. A source
    scan rather than a rendered frame, for the reason the rest of this file is
    pure -- the wording is assertable, the layout is not.

    ``size_m`` is the deliberate omission: it is continuous, so "0.35" and
    "0.36" are separate buckets and the five-verdict threshold would never be
    met. ``custom_triangles`` is the same. Everything else the form submits and
    ``VECTOR_PARAMS`` names must carry a hint.
    """
    import re
    from pathlib import Path

    from warlock import vectors
    from warlock.studio.panes import settings_3d

    source = Path(settings_3d.__file__).read_text(encoding="utf-8")
    hinted = set(re.findall(r'_hint\(ctx, "(\w+)"', source))
    owned = {"platform", "profile", "size_m", "bg_removal", "reference_prep",
             "custom_triangles"}

    assert owned <= set(vectors.VECTOR_PARAMS), "the form and the vocabulary must agree"
    assert hinted == owned - {"size_m", "custom_triangles"}


def test_the_era_style_control_is_renamed_without_renaming_its_key():
    """A relabel, not a rename. ``art_style`` is what every job on disk
    recorded and what the findings and verdict buckets are keyed on, so
    renaming the key would need a ``_LEGACY_ALIASES`` entry *and* would still
    split the corpus -- a vector recorded under the old spelling is a different
    string, and evidence under it would simply stop accumulating.
    """
    assert settings_2d.field_label("art_style") == "era style"
    assert "art_style" in guidance.form_fields()
    assert "era_style" not in guidance.form_fields()
    # And the group still names the key, because that is what the form holds.
    style = dict(settings_2d.GUIDANCE_GROUPS)["Style"]
    assert "art_style" in style


def test_every_other_field_still_reads_as_its_key():
    for field in guidance.form_fields():
        if field != "art_style":
            assert settings_2d.field_label(field) == field.replace("_", " ")


def test_the_blank_option_is_named_by_the_pane_not_by_the_key():
    """The empty entry is what the combo shows until something is chosen, so a
    relabel that stopped at the heading would leave it saying "art style...".
    """
    options = settings_2d._field_options(_ctx(), "art_style")
    assert options[0] == ("", "era style...")
    assert settings_2d._field_options(_ctx(), "palette")[0] == ("", "palette...")
