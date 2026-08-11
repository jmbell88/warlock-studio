"""Queue the SNES/Zelda prop style sweep: two subjects, four axes, five seeds.

**The taxonomy cannot say "A Link to the Past", and that is the question.**
``art_style=snes`` contributes "vivid saturated colours, bold simple shapes" and
``art_style=nes`` contributes "flat colour shading, bold readable silhouette,
clean dark outlines". ALTTP is both halves at once and no key carries both, so
which fragment lands closer on a prop is a real question with a real answer.
This run asks it, and asks whether the answer costs mesh quality.

**Two subjects, identical axes, and the identity is load-bearing.**
``findings._comparisons`` keys an entry on ``(param, low, high)`` and accumulates
across sweeps, so two plans carrying the *same* ``AXES`` pool 5 + 5 matched pairs
into one entry spanning two prompts. Findings are otherwise scoped per
``prompt_hash`` -- a byte-exact sha1 of the raw prompt, with no alias table --
so an answer measured on one crate is a claim about crates. Measuring on a crate
*and* a pot costs exactly the same units and is the first axis reading in this
repo's history to clear ``comparison_lines``' ``min_pairs=5`` with any breadth
at all.

**The baseline is ``sweep_props.WINNER``'s render half, not a fresh guess.**
``playground`` + ``render3d`` + ``birefnet`` are the strongest marginals the
2026-08-09 re-baseline produced (Amendment 2 of
``docs/measurements/2026-08-09-rebaseline.md``, which is careful to call them
marginals rather than a winner, and so is this). Holding them fixed is what lets
every unit of this run vary *depiction* only -- the fan-out the re-baseline's
own gate licensed by clearing 19 accepts of 41.

**One arm cannot pair, and it is declared rather than discovered.**
``guidance.normalize`` writes ``lora_weight`` only alongside ``style_lora``, so
the ``style_lora=""`` arm differs from a LoRA-bearing baseline in *two* vector
keys and ``findings._one_key_diff`` returns ``None`` for it: no pair, no entry,
silently. That already happened once -- ``sweep_rebaseline.py``'s ``style_lora``
axis had the same shape and its row in that run's measurement document was
assembled by hand with nothing in the tree saying so. Here it is stated in the
pre-registration, asserted in ``tests/test_campaign_specs.py``, and read by hand
off the grades.

**Two questions, two instruments, never merged.** ``grade`` keeps its
established meaning -- is this mesh usable -- so these rows stay comparable with
every row already in the corpus and ``findings.hint`` stays honest. *Style* is a
tag: ``on-style``/``wrong-style`` are legal at any grade since migration 10, so
a +4 mesh tagged ``wrong-style`` is exactly the row this run needs to record.
Grading style into the number would redefine ``USABLE_GRADE`` for everybody.

The decision rules -- Gate A on whether to build the corpus at all, Gate B on
which arms are adopted -- are pre-registered in
``docs/measurements/2026-08-10-zelda-props.md`` and were written before a unit
was queued. Review this sweep **with Blind on**.

    uv run python scripts/sweep_zelda_style.py --dry-run   # plan and validate
    uv run python scripts/sweep_zelda_style.py             # write the rows
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

# What both plans hold fixed: the reading of "16-bit game prop" that is not in
# question, plus the render half at the re-baseline's strongest marginals.
#
# Three absences are deliberate and each would cost something:
#
# ``setting`` -- ``genre=fantasy`` already contributes "high fantasy, medieval,
# ornate craftsmanship", so ``setting=medieval`` would put one word in the
# composed prompt twice against guidance.py's own brevity rule. (That
# "ornate craftsmanship" is itself anti-ALTTP is what the ``cartoon`` arm is
# for.)
#
# ``mood`` -- a mood fragment on an inanimate crate is close to inert, and the
# tempting value here is ``whimsical`` ("whimsical playful mood"), which would
# assert playfulness in the baseline and so weaken the ``genre=cartoon`` arm
# against a contrast it half-contains.
#
# ``silhouette`` -- a crate and a pot are already what they are; the phase-2
# corpus states one for the subjects whose shape is a claim (tree, bush).
#
# ``palette=vibrant`` is kept even though it says what ``snes`` says, and the
# redundancy is the point: on the ``nes`` arm, which asserts no colour at all,
# it is the only fragment holding the palette. That makes it a confound between
# the two art_style arms, which is why it is written here rather than left out.
BASE = {
    # the style hypothesis under test
    "art_style": "snes",
    "palette": "vibrant",
    "genre": "fantasy",
    "condition": "worn",
    "category": "prop",
    "platform": "3d",
    # Stated though it is the default: framing is an axis in the run next door,
    # and an axis whose baseline is implicit is one nobody can read back off the
    # spec afterwards.
    "framing": "three_quarter",
    # the render half -- marginals, not a winner; see the module docstring
    "base_model": "playground",
    "style_lora": "render3d",
    "lora_weight": 0.9,
    # Stated rather than inherited: the default is host-gated on birefnet.gguf,
    # and a baseline that depends on which weights a machine happens to hold is
    # not a baseline.
    "bg_removal": "birefnet",
    "reference_prep": True,
}

# (label, prompt, the fields that make this subject itself)
#
# Both prompts restate singularity as an appositive, which is the 2026-08-07
# PROMPT_TEMPLATE edit's lesson applied per subject: the template's "no other
# objects" lives in a clause, and ``reference.subject_mask`` is a corner flood
# fill rather than a matte, so anything the background reaches *through* the
# subject splits into components and MIN_SECOND_COMPONENT refuses the job.
# A crate and a closed pot are the safe pair -- one closed mass each, the
# ``sweep_props`` chest precedent -- which is why the risky subjects (a flower,
# a branchy tree) are in the phase-2 corpus behind wording that was designed
# against the same gate rather than in the run whose baseline must survive.
SUBJECTS = (
    (
        "crate",
        "a wooden crate with iron corner banding, a single closed box",
        {"material": "wood", "size_m": 0.9},
    ),
    (
        "pot",
        "a ceramic pot with a rounded body and a narrow neck, a single closed jar",
        {"material": "ceramic", "size_m": 0.5},
    ),
)

# One line per arm, because a unit costs three minutes of card time:
#
# ``nes``      the outline half of ALTTP the base cannot say.
# ``ps2``      "hand-painted texture style" -- the app's own default prop recipe
#              (guidance.PRESETS["handpainted_prop"]), and therefore the control
#              that says whether any of this is worth doing.
# ``cartoon``  "playful cartoon design, exaggerated friendly proportions"
#              against fantasy's "ornate craftsmanship".
# ``pristine`` the untested claim that 16-bit props carry no surface weathering.
#              Simultaneously a mesh hypothesis: less micro-detail is better
#              trellis input.
# ``""``       no style adapter. The empty string is dropped by ``unit_kwargs``,
#              which removes render3d's "3d style, 3d render" trigger -- the one
#              thing in the pipeline actively fighting a flat look. This is the
#              arm that cannot pair; see the module docstring.
#
# Deliberately absent: ``base_model``. It is not one variable -- playground is
# 25 steps at CFG 3.0, sdxl is Hyper-SD at 4 steps and guidance 0.0, and
# ``models.cfg_bases()`` gates the negative branch on guidance > 1.0, so the
# DEFAULT_NEGATIVE_PROMPT's "multiple objects" clause is inert on a 4-step arm
# and a composition refusal there means something different. A win on that axis
# is uninterpretable as a style claim, and the re-baseline's null on it was
# structurally unsatisfiable rather than measured.
AXES = (
    sweeps_mod.Axis("art_style", ("nes", "ps2")),
    sweeps_mod.Axis("genre", ("cartoon",)),
    sweeps_mod.Axis("condition", ("pristine",)),
    sweeps_mod.Axis("style_lora", ("",)),
)

PLANS = tuple(
    sweeps_mod.SweepPlan(
        label=f"zelda style - {label}",
        prompt=prompt,
        # BASE first so a subject may override it -- nothing does today, and the
        # ordering is what keeps that true rather than accidental.
        base={**BASE, **fields},
        # Imported from sweep_rogue, never restated: a matched pair is two units
        # at the *same* seed, and a per-plan seed list would silently halve the
        # comparison count the moment the two lists drifted.
        seeds=SEEDS,
        axes=AXES,
        stage="model",
    )
    for label, prompt, fields in SUBJECTS
)


if __name__ == "__main__":
    raise SystemExit(_campaign.main(PLANS, __doc__.splitlines()[0]))
