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

from warlock.service import sweeps as sweeps_mod

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _campaign(name: str):
    return importlib.import_module(name)


CAMPAIGNS = ("sweep_rogue", "sweep_confirm", "sweep_rebaseline", "sweep_props")


@pytest.mark.parametrize("name", CAMPAIGNS)
def test_every_campaigns_plans_pass_the_admission_they_will_be_submitted_under(name, svc):
    """The ``--dry-run`` path, as a test: one bad unit refuses the whole run."""
    module = _campaign(name)
    for plan in module.PLANS:
        units = sweeps_mod.expand(plan)
        assert units, f"{plan.label} plans no units"
        assert len(units) <= sweeps_mod.MAX_UNITS
        sweeps_mod._validate(svc, plan, units)


@pytest.mark.parametrize("name", ("sweep_confirm", "sweep_rebaseline", "sweep_props"))
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


def test_the_prop_corpus_states_its_framing_rather_than_inheriting_it(svc):
    """``framing`` is an *axis* in the run next door, so its value here is not
    something a later reader can recover from ``DEFAULT_FRAMING`` -- that
    default is exactly what the re-baseline may move."""
    module = _campaign("sweep_props")

    for plan in module.PLANS:
        assert plan.base.get("framing"), plan.label
