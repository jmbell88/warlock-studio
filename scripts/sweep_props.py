"""Queue the tier-qualification corpus: three props and one asymmetric figure.

**This measures nothing, and that is what it is for.** `TODO.md` §3 needs meshes
a human kept, on three named subjects, before a gltfpack tier can be qualified;
`TODO.md` §4 needs one *asymmetric* humanoid before the rig's handedness can be
checked. Neither is a question about settings, so there are no axes here at all
-- four no-axis plans, five seeds each, twenty units. ``expand`` on a plan with
no axes returns one baseline unit per seed, and ``_check_unit``'s canonical key
includes the seed, so the five are five distinct jobs rather than one refused
five times.

**Run this after ``sweep_rebaseline.py``'s verdicts, never before.** ``WINNER``
below is the whole point: a corpus generated at settings nobody chose qualifies
a tier against meshes that do not represent what the app produces. §2's
re-baseline is what fills it in, and the go/no-go on that run (>=12 accepts of
50) is what says the settings are worth building a corpus at.

Why a campaign script and not ``python -m warlock.bench run``. ``bench``'s
``core-v1`` suite genuinely carries all three subjects and labels them as the
qualification subjects in its own notes -- but ``recipe.Recipe`` has **no
``bg_removal`` field**, so a bench run would take the matte from
``guidance.default_bg_removal`` and therefore from whichever weights the host
happens to hold. That is precisely what ``sweep_rebaseline`` refuses ("a
baseline that depends on which weights a machine happens to hold is not a
baseline") and what ``tests/test_campaign_specs.py`` asserts against, and it
would land in the one run whose entire premise is "at the winning settings".
Two smaller things point the same way: the rock is category ``environment``, so
the obvious ``--filter prop,weapon`` selects sixteen items and no rock at all;
and ``qualify_tiers._warn_about_the_corpus`` substring-matches its subject names
against the prompt, which core-v1's "a large mossy boulder" fails.

**The prompts are written to be qualified against.** ``WANTED_SHAPES`` in
``scripts/qualify_tiers.py`` is ``("chest", "sword", "rock")`` and it matches on
the job's name or prompt, so each word appears in the prompt that asks for it.
The three are also deliberately unlike each other in the way the tier check
cares about: a chest is multi-material (wood and iron) and is the subject that
can lose a material assignment, a sword is thin and is the one a simplify pass
mangles first, and a rock is a single untextured material and is the control --
``tiercheck`` reports a source that was *already* missing something in ``notes``
beside the verdict rather than failing it, so the rock's poverty is stated
rather than mistaken for a loss.

**The fourth plan is not a prop and is not for §3.** A rig's handedness cannot
be read off a bilaterally symmetric subject: a mirrored skeleton looks perfectly
plausible in a still, which is the entire failure mode. ``pose2d`` fits only the
``humanoid`` template, so no prop can answer it either. The asymmetry is named
in the prompt, and it must be named in ``docs/measurements/`` **before the mesh
is looked at** -- writing down "the pauldron is on the subject's right" after
seeing the rig is how you talk yourself into whichever answer appeared.

    uv run python scripts/sweep_props.py --dry-run   # plan and validate only
    uv run python scripts/sweep_props.py             # write the rows
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _campaign  # noqa: E402
from sweep_rogue import SEEDS  # noqa: E402

from warlock.service import sweeps as sweeps_mod  # noqa: E402

# Filled in from the re-baseline's verdicts. Left at the re-baseline's own
# baseline arm so the script is runnable and testable today -- but that is a
# placeholder, not a recommendation, and the two entries below carry the same
# rule the other campaigns state: ``bg_removal`` and ``framing`` are written out
# rather than inherited, because a default that is host-gated (birefnet.gguf's
# presence) or that is an *axis* in the run next door is not something a later
# reader can recover from the spec.
WINNER = {
    "base_model": "sdxl",
    "style_lora": None,
    "bg_removal": "birefnet",
    "framing": "three_quarter",
    "platform": "3d",
    "reference_prep": True,
}

# (label, prompt, the fields that make this subject itself)
SUBJECTS = (
    (
        "chest",
        "a treasure chest with a domed lid, iron banding and a heavy lock",
        {"category": "prop", "genre": "fantasy", "setting": "medieval",
         "material": "wood", "condition": "worn"},
    ),
    (
        "sword",
        "a straight longsword with a crossguard and a leather-wrapped grip",
        {"category": "weapon", "genre": "fantasy", "setting": "medieval",
         "material": "steel", "condition": "worn", "silhouette": "elongated"},
    ),
    (
        "rock",
        "a large mossy rock, a single weathered boulder",
        {"category": "environment", "genre": "fantasy", "setting": "medieval",
         "material": "stone", "condition": "overgrown"},
    ),
    (
        # Not a qualification subject. See the module docstring: this exists so
        # that §4 has a mesh whose two sides are distinguishable in a still.
        "asymmetric figure",
        "a knight standing in a neutral pose, a single large pauldron on the "
        "right shoulder only, the left shoulder bare, a satchel on the left hip",
        {"category": "character", "genre": "fantasy", "setting": "medieval",
         "material": "steel", "condition": "worn", "silhouette": "bulky"},
    ),
)

PLANS = tuple(
    sweeps_mod.SweepPlan(
        label=f"tier corpus - {label}",
        prompt=prompt,
        # ``WINNER`` first so a subject may override it -- nothing does today,
        # and the ordering is what keeps that true rather than accidental.
        base={**{k: v for k, v in WINNER.items() if v is not None}, **fields},
        seeds=SEEDS,
        # No axes on purpose. A tier is qualified against meshes worth keeping,
        # not against a contrast, and every unit here is the same job at a
        # different seed.
        axes=(),
        stage="model",
    )
    for label, prompt, fields in SUBJECTS
)


if __name__ == "__main__":
    raise SystemExit(_campaign.main(PLANS, __doc__.splitlines()[0]))
