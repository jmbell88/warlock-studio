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
        # Matching main.py's Ctx: the picker's own labels, which the pane needs
        # to name a style in a sentence and to list the fitting ones.
        style_loras=[(m["key"], m["label"]) for m in catalog["fields"]["style_lora"]],
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
    """Per-subject scoping's last mile. The pane owns the prompt, so it always knows
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


# --- what a change of base model clears --------------------------------------


def _mismatched_pair():
    """A (base_model, style_lora) the registry does not pair, by construction.

    Replaces a "base that takes no LoRA at all", which stops existing the
    moment every architecture in the registry has an adapter -- at which point
    a next() over it raises StopIteration rather than failing an assertion.
    """
    for base in models.BASE_MODELS.values():
        for lora in models.STYLE_LORAS.values():
            if not models.lora_fits(base, lora):
                return base.key, lora.key
    raise AssertionError("the registry holds only one architecture")


def _form(**over):
    form = {
        "prompt": "a wooden crate",
        "count": 1,
        "base_model": models.lora_bases()[0],
        "style_lora": "",
        "lora_weight": models.DEFAULT_LORA_WEIGHT,
        "ref_path": "",
        "ip_adapter": "",
        "control": "",
        "output": "reference",
    }
    form.update(over)
    return form


def test_choosing_a_base_that_cannot_use_a_lora_clears_the_style():
    """The dead end this replaces: the picker is *disabled* on a non-SDXL base,
    so the value that disables Generate is the one control the user cannot
    reach. Clearing it is the only recovery that does not require guessing
    which earlier choice to undo.
    """
    base_key, lora_key = _mismatched_pair()
    form = _form(style_lora=lora_key, lora_weight=0.4)
    form["base_model"] = base_key
    cleared = settings_2d.clear_unusable(_ctx(), form)
    assert form["style_lora"] == ""
    assert form["lora_weight"] == models.DEFAULT_LORA_WEIGHT
    assert cleared
    # And Generate is live again: that is the whole point.
    assert settings_2d.validate(form) == []


def test_switching_back_does_not_resurrect_the_cleared_style():
    """A clear is a clear. Remembering the old selection to restore it would
    make the same value reappear silently under a base the user has since
    reconfigured around.
    """
    ctx = _ctx()
    base_key, lora_key = _mismatched_pair()
    form = _form(style_lora=lora_key)
    form["base_model"] = base_key
    settings_2d.clear_unusable(ctx, form)
    form["base_model"] = models.lora_bases()[0]
    settings_2d.clear_unusable(ctx, form)
    assert form["style_lora"] == ""


def test_a_base_that_cannot_run_a_controlnet_clears_the_structure_control():
    """The sibling gate, and the sharper one: ``structure_note`` hides the
    Structure group entirely, so the stale ``control`` that validate refuses is
    not merely disabled -- it is off screen.
    """
    form = _form(
        ref_path="ref.png",
        control=guidance.catalog()["fields"]["control"][0]["key"],
        base_model=models.controlnet_bases()[0],
    )
    form["base_model"] = next(
        k for k in models.BASE_MODELS if k not in models.controlnet_bases()
    )
    settings_2d.clear_unusable(_ctx(), form)
    assert form["control"] == ""
    assert settings_2d.validate(form) == []


def test_the_negative_prompt_is_deliberately_not_cleared():
    """An inert Avoid text stays in the authored brief.

    The generation boundary removes it from a model that cannot consume it,
    rather than erasing it on a base-model change or blocking Generate.
    """
    form = _form(base_model="turbo", negative_prompt="blurry, watermark")
    settings_2d.clear_unusable(_ctx(), form)
    assert form["negative_prompt"] == "blurry, watermark"
    assert not any("negative" in p.lower() for p in settings_2d.validate(form))


def test_the_clear_is_explained_rather_than_silent():
    base_key, lora_key = _mismatched_pair()
    form = _form(style_lora=lora_key)
    form["base_model"] = base_key
    cleared = settings_2d.clear_unusable(_ctx(), form)
    assert any("style" in note.lower() and "cleared" in note.lower() for note in cleared)
    # Inside imgui's default Basic-Latin+Latin-1 atlas range.
    for note in cleared:
        assert all(ord(ch) < 0x100 for ch in note)


def test_a_restored_form_is_not_rewritten_merely_by_being_opened():
    """The one that catches a per-frame implementation. A form restored with a
    style picked under another base must keep it until the *user* changes the
    base: rewriting it on the frame the pane opens destroys a selection nobody
    touched, and does it before the note explaining it can be read.

    A source scan because the guard is in draw code: the call site must sit
    under the ``!= before`` comparison and nowhere else.
    """
    import re
    from pathlib import Path

    source = Path(settings_2d.__file__).read_text(encoding="utf-8")
    lines = source.splitlines()
    calls = [
        i
        for i, line in enumerate(lines)
        if "clear_unusable(" in line and not line.lstrip().startswith("def ")
    ]
    assert len(calls) == 1, "one call site, or the guard is not the only path"
    # Inside the guard's block, not necessarily on the line under it: changing
    # the base does other work in the same branch. Walk up to the comparison
    # and require the call to be indented beneath it, which is the actual rule
    # -- "only reachable when the base changed" -- rather than a line offset
    # that any statement added to the branch would break.
    call = calls[0]
    guard = next(
        (
            i
            for i in range(call - 1, max(call - 8, -1), -1)
            if re.search(r'if form\["base_model"\] != before:', lines[i])
        ),
        None,
    )
    assert guard is not None, "the call site is not under the base-model guard"

    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    assert _indent(lines[call]) > _indent(lines[guard])
    # Nothing between the guard and the call may leave the branch.
    for line in lines[guard + 1 : call]:
        assert not line.strip() or _indent(line) > _indent(lines[guard])
    # And nothing else in the pane reaches it: the notes stay pure reads.
    base_key, lora_key = _mismatched_pair()
    form = _form(style_lora=lora_key, base_model=base_key)
    ctx = _ctx()
    settings_2d.lora_note(ctx, form)
    settings_2d.lora_filter_note(ctx, form)
    settings_2d.lora_options(ctx, form)
    settings_2d.structure_note(ctx, form)
    settings_2d.validate(form)
    assert form["style_lora"] != ""


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


def test_every_field_reads_as_its_key():
    # FIELD_LABELS emptied with the taxonomy (it carried art_style); every
    # surviving field's on-screen name is its key with the underscores out.
    for field in guidance.form_fields():
        assert settings_2d.field_label(field) == field.replace("_", " ")


# --- which styles the picker offers ------------------------------------------


def _synth_ctx(by_base, loras):
    """A catalog the registry does not have to hold, so the wording is
    independent of what ships today."""
    catalog = dict(guidance.catalog())
    catalog["loras_by_base"] = by_base
    catalog["lora_bases"] = [k for k, v in by_base.items() if v]
    return SimpleNamespace(
        guidance=catalog,
        base_models=[(k, k.upper()) for k in by_base],
        style_loras=loras,
    )


def test_the_picker_lists_only_the_styles_fitted_to_the_chosen_base():
    ctx = _synth_ctx({"a": ["one"], "b": ["two"]}, [("one", "One"), ("two", "Two")])
    keys = [k for k, _ in settings_2d.lora_options(ctx, {"base_model": "a"})]
    assert keys == ["one"]


def test_a_stale_selection_stays_listed_and_is_marked():
    """widgets.combo falls back to index 0 for a value it cannot find, so
    dropping the entry would draw "no style LoRA" over a selection validate is
    refusing -- the dead end clear_unusable exists to prevent, by another
    door."""
    ctx = _synth_ctx({"a": ["one"], "b": ["two"]}, [("one", "One"), ("two", "Two")])
    options = settings_2d.lora_options(ctx, {"base_model": "a", "style_lora": "two"})
    assert ("two" in [k for k, _ in options])
    label = next(label for key, label in options if key == "two")
    assert "not fitted" in label


def test_the_disabled_note_fires_only_for_a_base_no_style_fits():
    ctx = _synth_ctx({"a": ["one"], "b": []}, [("one", "One")])
    assert settings_2d.lora_note(ctx, {"base_model": "a"}) is None
    note = settings_2d.lora_note(ctx, {"base_model": "b"})
    assert note is not None and "architecture" in note
    # And it names a base the user can actually pick, by its own label.
    assert "A" in note


def test_the_filter_note_names_what_is_listed_and_defers_when_nothing_fits():
    ctx = _synth_ctx({"a": ["one"], "b": []}, [("one", "One"), ("two", "Two")])
    narrowed = settings_2d.lora_filter_note(ctx, {"base_model": "a"})
    assert narrowed is not None and "One" in narrowed
    # lora_note owns the empty case; saying it twice is one control saying two
    # things under a disabled combo.
    assert settings_2d.lora_filter_note(ctx, {"base_model": "b"}) is None


def test_the_filter_note_is_silent_when_the_whole_list_is_offered():
    ctx = _synth_ctx({"a": ["one", "two"]}, [("one", "One"), ("two", "Two")])
    assert settings_2d.lora_filter_note(ctx, {"base_model": "a"}) is None


def test_every_note_stays_inside_the_default_atlas_range():
    ctx = _ctx()
    for base in models.BASE_MODELS:
        form = _form(base_model=base)
        for note in (settings_2d.lora_note(ctx, form), settings_2d.lora_filter_note(ctx, form)):
            if note is not None:
                assert all(ord(ch) < 0x100 for ch in note)


def test_the_catalog_and_the_registry_agree_about_which_loras_fit():
    """The pane's draw path reads ``ctx.guidance["loras_by_base"]`` while
    ``validate`` asks ``models.loras_by_base()`` directly -- one map behind two
    doors, so the two answers are pinned equal here."""
    assert guidance.catalog()["loras_by_base"] == models.loras_by_base()


def test_no_fold_machinery_survives_the_flat_form():
    """The flat form has no folds, so a refusal can never name a control
    nothing draws -- the machinery that guaranteed that is gone with the folds
    themselves."""
    for name in ("folds_to_open", "folded_fields", "MORE_KEY", "ADVANCED_KEY"):
        assert not hasattr(settings_2d, name)
