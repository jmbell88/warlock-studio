"""The sweep campaigns under ``scripts/``, validated without a GPU.

Why these are tested at all, when the scripts they live in are one-off
submitters. A campaign's cost is measured in hours of card time and its value
is measured in whether the corpus it produces answers the question it was built
for -- so the two failure modes worth catching are *before* the run, not
during it. Both are cheap to catch here and expensive to catch at 3 a.m.:

**A typo'd axis param.** ``sweeps.unit_kwargs`` raises ``Invalid`` for a param
that is neither a guidance field nor a ``KWARG_AXES`` entry, but only when the
plan is submitted -- so a misspelled axis is caught, and caught in a way that
refuses the whole campaign after the user has closed the app to run it.

**A plan that does not measure what it says.** `TODO.md` §2's whole finding
about Sweep B is that it varied four axes around a baseline pinned at the one
value that dominates the verdict, and so measured nothing. The re-baseline plan
exists to not repeat that, which is a property of its ``base`` dict and is
therefore assertable: ``bg_removal`` must be ``birefnet`` and must not be an
axis.

Nothing here writes a row. ``sweeps._validate`` is the same admission
``create_sweep`` runs, and running it against the ``svc`` fixture is exactly
what the scripts' own ``--dry-run`` does.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from warlock import guidance, vectors
from warlock.service import sweeps as sweeps_mod

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _campaign(name: str):
    return importlib.import_module(name)


CAMPAIGNS = (
    "sweep_rogue",
    "sweep_confirm",
    "sweep_rebaseline",
    "sweep_props",
    "sweep_zelda_style",
    "sweep_zelda_klein",
)


@pytest.mark.parametrize("name", CAMPAIGNS)
def test_every_campaigns_plans_pass_the_admission_they_will_be_submitted_under(name, svc):
    """The ``--dry-run`` path, as a test: one bad unit refuses the whole run."""
    module = _campaign(name)
    for plan in module.PLANS:
        units = sweeps_mod.expand(plan)
        assert units, f"{plan.label} plans no units"
        assert len(units) <= sweeps_mod.MAX_UNITS
        sweeps_mod._validate(svc, plan, units)


@pytest.mark.parametrize(
    "name",
    (
        "sweep_confirm",
        "sweep_rebaseline",
        "sweep_props",
        "sweep_zelda_style",
        "sweep_zelda_klein",
    ),
)
def test_a_campaign_states_the_matte_it_runs_rather_than_inheriting_one(name, svc):
    """A sweep off an unstated baseline is not reproducible: ``bg_removal``'s
    own default is host-gated (``guidance.default_bg_removal`` reads the weights
    directory), so a plan that leaves it out measures a different baseline on a
    machine that never downloaded ``birefnet.gguf``."""
    module = _campaign(name)
    for plan in module.PLANS:
        assert plan.base.get("bg_removal"), plan.label


def test_the_rogue_sweep_did_not_state_it_and_that_is_the_whole_lesson(svc):
    """Pinned rather than repaired, in both plans and for two different costs.

    The depiction plan left ``bg_removal`` to the host default, so all 45 units
    ran under ``auto`` -- the value that dominates the verdict, held at its bad
    one while four axes varied. That is the floor effect §2 records.

    The render plan left it unstated *and* varied it, which was harmless when
    the default was ``auto`` and is now a refusal: the default has since flipped
    to ``birefnet``, so its ``birefnet`` arm and its baseline are the same job
    and ``_validate`` says so. Under the ``svc`` fixture the weights directory
    is empty, which is why it still validates here -- the fixture pins the
    *old* host, and that is the point. Editing either plan would misrepresent
    what ran and change nothing: the corpus is already on disk.
    """
    module = _campaign("sweep_rogue")
    render = next(plan for plan in module.PLANS if "render" in plan.label)
    depiction = next(plan for plan in module.PLANS if "depiction" in plan.label)

    assert "bg_removal" not in depiction.base
    assert "bg_removal" not in {axis.param for axis in depiction.axes}
    assert "bg_removal" not in render.base
    assert "bg_removal" in {axis.param for axis in render.axes}


# --- can an axis arm actually be read? ---------------------------------------


# Every (campaign, axis param) whose arms differ from their baseline in more than
# one vector key, and therefore record no matched pair at all. Each entry is a
# decision, not a bug -- but it has to be *declared* here, because the failure is
# silent: the units run, the meshes are graded, and the axis simply never appears
# in ``findings.json``'s comparisons.
#
# All three are the same mechanism. ``guidance.normalize`` emits ``lora_weight``
# only alongside ``style_lora``, so an arm that adds or removes an adapter moves
# two keys at once and there is no way to hold the second one still.
UNPAIRABLE_BY_DESIGN = {
    ("sweep_rogue", "style_lora"),
    ("sweep_rebaseline", "style_lora"),
    ("sweep_zelda_style", "style_lora"),
}


def _unit_vector(svc, plan, unit):
    """The config vector a finished unit would be judged under.

    Mirrors ``sweeps._check_unit``'s own composition rather than restating it:
    ``normalize``'s output for everything it is handed, plus the ``KWARG_AXES``
    it never sees, through the same ``vectors.config_vector`` the worker and
    ``record_verdict`` use. Reading the params any other way here would be a
    second spelling of what a verdict is filed against.
    """
    kwargs = sweeps_mod.unit_kwargs(plan, unit)
    params = guidance.normalize(
        {
            **kwargs["guidance_fields"],
            **{k: kwargs.get(k) for k in sweeps_mod.NORMALIZED_KWARGS},
        },
        bg_default=guidance.default_bg_removal(svc.config.trellis_models_dir),
    )
    params.update(
        {
            k: kwargs[k]
            for k in sweeps_mod.KWARG_AXES
            if k in kwargs and k not in sweeps_mod.NORMALIZED_KWARGS
        }
    )
    return vectors.config_vector({"params": params, "stage": plan.stage})


@pytest.mark.parametrize("name", CAMPAIGNS)
def test_every_axis_arm_pairs_with_its_baseline_or_says_here_that_it_cannot(name, svc):
    """``findings._one_key_diff`` is what makes two rows a matched pair, and it
    returns ``None`` unless the two vectors disagree in exactly one key. An arm
    that moves two keys records no pair, no ``grade_delta`` and no entry -- and
    nothing anywhere says so.

    That already cost a reading. ``sweep_rebaseline``'s ``style_lora`` axis has
    this shape, so the ``style_lora`` row in
    ``docs/measurements/2026-08-09-rebaseline.md`` was assembled by hand off the
    grades while every other row in that table came out of ``comparisons``. The
    document does not distinguish them. An unpairable arm is legitimate -- it is
    still five meshes a human looks at -- but the run has to know it is buying a
    hand count rather than a comparison, which is what this list is for.
    """
    module = _campaign(name)
    for plan in module.PLANS:
        baselines = {u.seed: u for u in sweeps_mod.expand(plan) if not u.overrides}
        for unit in sweeps_mod.expand(plan):
            if not unit.overrides:
                continue
            (param,) = unit.overrides
            mine = _unit_vector(svc, plan, unit)
            theirs = _unit_vector(svc, plan, baselines[unit.seed])
            moved = {k for k in mine.keys() | theirs.keys() if mine.get(k) != theirs.get(k)}
            declared = (name, param) in UNPAIRABLE_BY_DESIGN
            if declared:
                assert len(moved) > 1, f"{name} {unit.label} pairs after all; drop the exemption"
            else:
                assert moved == {param}, f"{name} {unit.label} moves {sorted(moved)}"


# --- the blind confirm -------------------------------------------------------


def test_the_confirm_is_two_arms_and_nothing_else(svc):
    """`TODO.md` §2 run 1: birefnet against auto, one knob, labels hidden. Any
    third axis would make it a sweep again and cost the clean comparison."""
    module = _campaign("sweep_confirm")
    (plan,) = module.PLANS

    assert [axis.param for axis in plan.axes] == ["bg_removal"]
    assert plan.base["bg_removal"] == "birefnet"
    assert plan.axes[0].values == ("auto",)


def test_the_confirm_is_the_size_the_plan_asked_for(svc):
    """8-12 units: cheap next to a 3.7 h sweep, and enough for the 2x2 the
    original review had only four rows of."""
    module = _campaign("sweep_confirm")
    (plan,) = module.PLANS
    units = sweeps_mod.expand(plan)

    assert 8 <= len(units) <= 12
    # Matched pairs are two units at the *same* seed, so every seed must carry
    # both arms or the pair count silently halves.
    assert len(units) == 2 * len(plan.seeds)
    for seed in plan.seeds:
        arms = {u.label for u in units if u.seed == seed}
        assert arms == {"baseline", "bg_removal=auto"}


def test_the_confirm_runs_the_same_subject_the_signal_came_from(svc):
    """A confirm of a finding on a different prompt confirms nothing: the
    corpus is scoped by ``prompt_hash`` for exactly this reason."""
    rogue = _campaign("sweep_rogue")
    confirm = _campaign("sweep_confirm")

    assert confirm.PLANS[0].prompt == rogue.PROMPT
    assert set(confirm.PLANS[0].seeds) <= set(rogue.SEEDS)


# --- the re-baseline ---------------------------------------------------------


def test_the_re_baseline_pins_the_variable_that_dominated_the_verdict(svc):
    """The Sweep B lesson: ``bg_removal`` was held at its bad value while four
    axes varied, which is a floor effect and not a null result. Here it is the
    baseline and no longer an axis."""
    module = _campaign("sweep_rebaseline")
    (plan,) = module.PLANS

    assert plan.base["bg_removal"] == "birefnet"
    assert "bg_removal" not in {axis.param for axis in plan.axes}


def test_the_re_baseline_carries_the_framing_axis(svc):
    """What Night 1's 0e made expressible, and the reason this run supersedes
    the old render sweep rather than repeating it."""
    module = _campaign("sweep_rebaseline")
    (plan,) = module.PLANS
    framing = next(axis for axis in plan.axes if axis.param == "framing")

    from warlock import guidance

    assert plan.base.get("framing", guidance.DEFAULT_FRAMING) == "three_quarter"
    assert framing.values == ("front_ortho",)
    # Framing is a *camera* claim about a character; a battery of framings on a
    # prop measures nothing anyone asked about.
    assert plan.base["category"] == "character"


def test_the_re_baseline_varies_how_the_reference_is_drawn_and_nothing_else(svc):
    """The render/depiction seam ``sweep_rogue`` split along. Depiction axes
    (palette, silhouette, condition, mood) come *after* this run establishes a
    workable accept rate -- fanning out first is the mistake §2 records."""
    module = _campaign("sweep_rebaseline")
    (plan,) = module.PLANS

    assert {axis.param for axis in plan.axes} == {"base_model", "style_lora", "framing"}


# --- the tier-qualification corpus -------------------------------------------


def test_the_prop_corpus_measures_nothing(svc):
    """`TODO.md` §3 needs meshes worth keeping, not a contrast. An axis here
    would make it a sweep, and a sweep spends its units on a question nobody
    asked -- the qualification wants several seeds of *one* configuration so
    that some of them come out well."""
    module = _campaign("sweep_props")

    for plan in module.PLANS:
        assert plan.axes == ()
        assert plan.vectors == ()
        units = sweeps_mod.expand(plan)
        assert len(units) == len(plan.seeds)
        assert {u.label for u in units} == {"baseline"}


def test_the_prop_corpus_names_the_shapes_qualify_tiers_looks_for(svc):
    """``qualify_tiers._warn_about_the_corpus`` substring-matches ``WANTED_SHAPES``
    against a job's name or prompt, so a subject whose prompt does not contain
    its own word is a corpus that warns about itself. This is why the rock is
    "a large mossy rock" and not core-v1's "a large mossy boulder"."""
    from qualify_tiers import WANTED_SHAPES

    module = _campaign("sweep_props")
    prompts = [plan.prompt.lower() for plan in module.PLANS]

    for shape in WANTED_SHAPES:
        assert any(shape in prompt for prompt in prompts), shape


def test_the_prop_corpus_carries_one_asymmetric_humanoid(svc):
    """§4's handedness check cannot be read off a bilaterally symmetric subject:
    a mirrored skeleton looks perfectly plausible in a still, which is the whole
    failure mode. ``pose2d`` fits only ``humanoid``, so no prop can answer it."""
    module = _campaign("sweep_props")
    figures = [plan for plan in module.PLANS if plan.base.get("category") == "character"]

    (figure,) = figures
    # The asymmetry has to be in the prompt, and it has to distinguish a side by
    # name -- "asymmetric" alone would leave which side to chance, and the
    # measurement document has to state the expected side before looking.
    assert "right" in figure.prompt and "left" in figure.prompt


def test_the_prop_corpus_runs_at_the_re_baselines_strongest_marginals(svc):
    """`TODO.md` §2's re-baseline named no winner on any axis and could not have
    -- ``baseline s23`` was refused at the gate, leaving four usable matched
    pairs against a bar written as 5-0. Choosing anyway is legitimate here and
    would not be in a finding: a tier qualification needs a configuration that
    produces meshes worth keeping, not the best one. This pins that the choice
    is the *measured* strongest marginal rather than drifting back to a guess."""
    module = _campaign("sweep_props")

    assert module.WINNER["base_model"] == "playground"  # 4/5
    assert module.WINNER["style_lora"] == "render3d"  # 3/4
    # front_ortho lost 1-2 and accepted 1/5 against the baseline's 2/4.
    assert module.WINNER["framing"] == "three_quarter"


def test_the_prop_corpus_states_its_framing_rather_than_inheriting_it(svc):
    """``framing`` is an *axis* in the run next door, so its value here is not
    something a later reader can recover from ``DEFAULT_FRAMING`` -- that
    default is exactly what the re-baseline may move."""
    module = _campaign("sweep_props")

    for plan in module.PLANS:
        assert plan.base.get("framing"), plan.label


# --- the SNES/Zelda prop style sweep -----------------------------------------


def test_the_zelda_sweep_runs_the_same_axes_on_both_subjects(svc):
    """This is the whole reason it is two plans rather than one.

    ``findings._comparisons`` keys an entry on ``(param, low, high)`` and
    accumulates across sweeps, so identical axes on two prompts pool 5 + 5 pairs
    into one entry spanning two subjects -- the first axis reading here to clear
    ``comparison_lines``' ``min_pairs=5`` with any breadth. Diverging the axis
    tuples would silently halve every contrast back to a single-subject claim,
    which is exactly what ``prompt_hash`` scoping means when it is not planned
    for.
    """
    module = _campaign("sweep_zelda_style")
    assert len(module.PLANS) == 2
    axes = {tuple((a.param, a.values) for a in plan.axes) for plan in module.PLANS}
    assert len(axes) == 1
    # And two distinct prompts, or there is one prompt_hash and no breadth at all.
    assert len({plan.prompt for plan in module.PLANS}) == 2


def test_the_zelda_sweep_varies_depiction_over_a_pinned_render_half(svc):
    """The re-baseline's gate is what licenses fanning out here (19 accepts of
    41, against a pre-registered 12), and its Amendment 2 is where the pinned
    half comes from. ``base_model`` is deliberately not an axis: it is not one
    variable -- ``models.cfg_bases()`` gates the negative branch on
    ``guidance_scale > 1.0``, so ``DEFAULT_NEGATIVE_PROMPT``'s "multiple objects"
    clause is inert on a 4-step arm and a composition refusal there means
    something different from one on ``playground``."""
    module = _campaign("sweep_zelda_style")
    params = {axis.param for axis in module.AXES}

    assert params == {"art_style", "genre", "condition", "style_lora"}
    assert "base_model" not in params
    for plan in module.PLANS:
        assert plan.base["base_model"] == "playground"
        assert plan.base["style_lora"] == "render3d"
        # Stated though it is the default: framing is an axis in the run next
        # door, so its value here is not recoverable from ``DEFAULT_FRAMING``.
        assert plan.base["framing"] == "three_quarter"
        assert plan.stage == "model"


def test_the_zelda_sweep_asks_the_question_its_taxonomy_cannot_state(svc):
    """``snes`` gives saturation and ``nes`` gives "clean dark outlines"; ALTTP
    is both and no key carries both, so the run's job is to find out which
    fragment lands closer. ``ps2`` is in as the control rather than as a
    hypothesis: it is the app's own default prop recipe
    (``guidance.PRESETS["handpainted_prop"]``), so if it wins, none of this was
    worth doing and that is a result."""
    module = _campaign("sweep_zelda_style")
    art_style = next(a for a in module.AXES if a.param == "art_style")

    assert module.BASE["art_style"] == "snes"
    assert set(art_style.values) == {"nes", "ps2"}
    preset = next(p for p in guidance.PRESETS if p["key"] == "handpainted_prop")
    assert preset["fields"]["art_style"] == "ps2"
    # And that preset's own prompt is a wooden crate bound with iron, which is
    # this sweep's first subject -- the control is the app's default answer to
    # very nearly this question.
    assert "crate" in preset["prompt"]


def test_the_zelda_sweeps_subjects_state_their_own_singularity(svc):
    """``reference.subject_mask`` is a corner flood fill rather than a matte, so
    anything the background reaches *through* a subject splits into components
    and ``MIN_SECOND_COMPONENT`` refuses the job -- and a refusal is a
    measurement, never re-run by ``sweep_refill``, so it permanently costs that
    subject a seed. ``PROMPT_TEMPLATE``'s own "no other objects" was not enough
    for the 17 refusals of 2026-08-07; the fix then was to state the constraint
    positively, and the ``sweep_props`` rock states it again as an appositive.
    Both subjects here are one closed mass and say so."""
    module = _campaign("sweep_zelda_style")

    for plan in module.PLANS:
        assert "a single" in plan.prompt, plan.label


def test_the_zelda_sweep_fits_one_submit_per_subject(svc):
    """Six configurations at five seeds, twice. The seeds are ``sweep_rogue``'s,
    imported rather than restated: a matched pair is two units at the *same*
    seed, and two lists that drift halve every contrast."""
    module = _campaign("sweep_zelda_style")
    from sweep_rogue import SEEDS

    total = 0
    for plan in module.PLANS:
        units = sweeps_mod.expand(plan)
        assert plan.seeds == SEEDS
        assert len(units) == 6 * len(SEEDS) <= sweeps_mod.MAX_UNITS
        total += len(units)
    assert total == 60


# --- the FLUX.2 klein sibling ------------------------------------------------


def test_the_klein_sweep_could_not_have_been_an_arm_of_the_other_one(svc):
    """Not a preference -- an admission refusal. ``models.lora_fits`` is
    family-gated, so a ``base_model=flux_klein_distilled`` arm against a base
    carrying ``render3d`` is refused by ``guidance.normalize`` inside
    ``_check_unit``, and all-or-nothing admission would take the whole 60-unit
    campaign with it. This pins the mechanism, so that a later reader who
    wonders why the checkpoint question is split across two sweeps finds the
    answer rather than assuming it was untidiness."""
    style = _campaign("sweep_zelda_style")

    with pytest.raises(ValueError) as caught:
        guidance.normalize(
            {**style.BASE, "base_model": "flux_klein_distilled"}, bg_default="birefnet"
        )
    assert "flux2klein" in str(caught.value)
    assert getattr(caught.value, "field", None) == "style_lora"


def test_the_klein_sweep_shares_its_subjects_bucket_and_differs_only_in_the_render_half(svc):
    """The whole instrument. ``findings._per_prompt`` scopes on a byte-exact
    sha1 of the raw prompt, so importing the prompts rather than retyping them
    is what puts these rows in the *same* two per-subject buckets as the
    playground run -- which is the only comparison available, since
    ``_comparisons`` groups on ``sweep_id`` and no matched pair spans two
    sweeps. Retyping one comma would silently open a third bucket and the two
    marginals would never meet."""
    style = _campaign("sweep_zelda_style")
    klein = _campaign("sweep_zelda_klein")

    assert [p.prompt for p in klein.PLANS] == [p.prompt for p in style.PLANS]
    assert [p.seeds for p in klein.PLANS] == [p.seeds for p in style.PLANS]

    # Depiction identical, render half deliberately different, and no adapter --
    # pixelklein is a pixel-art LoRA and flat pixel art is poor trellis input.
    render_half = {"base_model", "style_lora", "lora_weight"}
    assert {
        k: v for k, v in klein.BASE.items() if k not in render_half
    } == {k: v for k, v in style.BASE.items() if k not in render_half}
    assert klein.BASE["base_model"] == "flux_klein_distilled"
    assert "style_lora" not in klein.BASE
    assert "lora_weight" not in klein.BASE


def test_the_klein_sweep_measures_nothing_by_contrast_and_says_so(svc):
    """No axes, for the ``sweep_props`` reason: a contrast would need matched
    pairs this design cannot form, so the units go to the marginal and the
    meshes instead."""
    klein = _campaign("sweep_zelda_klein")

    total = 0
    for plan in klein.PLANS:
        units = sweeps_mod.expand(plan)
        assert plan.axes == ()
        assert plan.vectors == ()
        assert {u.label for u in units} == {"baseline"}
        assert len(units) == len(plan.seeds)
        total += len(units)
    # Ten, and the number is chosen rather than left over: the playground run's
    # own baseline arm is five units per subject, so this is exactly matched to
    # the thing it is being read against. More seeds here would be units with no
    # counterpart on the other side of the only comparison available.
    style = _campaign("sweep_zelda_style")
    baseline_units = sum(
        1 for plan in style.PLANS for u in sweeps_mod.expand(plan) if not u.overrides
    )
    assert total == baseline_units == 10


# --- refilling a sweep that lost units ---------------------------------------


def test_a_refill_re_queues_a_cancelled_unit_but_never_a_refusal(svc):
    """A long sweep loses units to a cancel or an app shutdown, and neither is a
    measurement. A composition-gate refusal *is* one -- it writes an observation
    carrying ``refused_<code>``, and the whole reason the worker records a
    terminal ``error`` is that the 2026-08-07 sweep's 17 refusals wrote nothing
    and so flattered every checkpoint that fails most often. Re-rolling one
    would replace a real negative with whatever the next seed drew."""
    import sweep_refill

    plan = _campaign("sweep_confirm").PLANS[0]
    result = sweeps_mod.create_sweep(svc, plan)
    units = svc.store.sweep_jobs(result["id"])
    svc.store.set_status(units[0]["id"], "cancelled")
    svc.store.set_status(units[1]["id"], "error", "interrupted by shutdown")
    svc.store.set_status(
        units[2]["id"], "error", "There is more than one object in the reference."
    )

    lost, refused = sweep_refill.lost_units(svc, result["id"])

    assert units[0]["sweep_unit"] in lost
    assert units[1]["sweep_unit"] in lost
    assert units[2]["sweep_unit"] in refused
    assert units[2]["sweep_unit"] not in lost


def test_a_refill_plans_from_the_sweeps_own_stored_spec(svc):
    """Never a restatement of the settings. ``rerun_job`` cannot serve here --
    it copies params only, deliberately, so ``sweep_id`` can never leak onto a
    reroll -- which means a refill has to plan the unit itself, and the only
    honest source for that is the spec the sweep was created with."""
    import sweep_refill

    plan = _campaign("sweep_rebaseline").PLANS[0]
    result = sweeps_mod.create_sweep(svc, plan)

    recovered = sweep_refill.plan_from_spec(svc.store.get_sweep(result["id"]))

    assert recovered.base == plan.base
    assert recovered.seeds == plan.seeds
    assert {a.param for a in recovered.axes} == {a.param for a in plan.axes}
    assert recovered.stage == plan.stage
    # And it therefore expands to exactly the same units, which is what lets a
    # refilled label be looked up rather than parsed.
    assert [sweeps_mod.unit_label(u) for u in sweeps_mod.expand(recovered)] == [
        sweeps_mod.unit_label(u) for u in sweeps_mod.expand(plan)
    ]
