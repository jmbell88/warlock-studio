"""Queue the two "SNES rogue adventurer" sweeps onto the live job queue.

**This run has already happened** -- 2026-08-07, 100 units, reviewed the same
day. It is kept because the two later campaigns are defined against it: they
reuse its prompt, its seeds and its fixed base, and the tension left in that
base (see below) is still unresolved. `TODO.md` §2 records what it found. The
successors are ``sweep_confirm.py`` (the blind matte confirm) and
``sweep_rebaseline.py`` (the render sweep re-run over a learned matte); the
depiction half is deliberately not re-specced yet, and §2 says why.

A headless *submitter*, not a runner: it writes 100 ``queued`` rows through
``service.sweeps.create_sweep`` and exits. See ``_campaign.py``, which is that
half, and run it with the app closed.

**It will not submit again on a host that has ``birefnet.gguf``, and that is
the rule the successors follow, demonstrated.** Neither plan states
``bg_removal``, so the baseline takes whatever ``guidance.default_bg_removal``
resolves against the weights directory -- which was ``auto`` when this ran and
is ``birefnet`` now. PLAN_A's ``bg_removal=birefnet`` axis therefore duplicates
its own baseline today, and ``sweeps._validate`` refuses the pair by canonical
key rather than spending an hour drawing one picture twice. It is left exactly
as it ran: what the corpus means depends on what the script *did*, and the
refusal is more informative than a repair would be.

Why two sweeps and not one. ``MAX_UNITS`` is 64, and 100 units is over it. The
split is along the seam the units already have rather than an arbitrary 50/50:
sweep A varies *how the reference is drawn* (checkpoint, style LoRA, matting),
sweep B varies *what the prompt describes* (silhouette, palette, condition,
mood). Splitting costs nothing that matters, because ``findings.comparisons``
pairs two rows only when they share a ``sweep_id`` and a seed and differ in
exactly one key -- a pair never spans two sweeps anyway, and each plan here
carries its own baseline at every seed, so pairs form inside both halves.
Raising ``MAX_UNITS`` instead would buy only cross-group pairs that an OFAT
fan-out does not produce.

Why these axes. ``jobs`` is empty, the eight stored verdicts are all rejects
against an unrelated ``environment``/``wood`` prompt, and ``bench/findings.json``
is a stub still carrying the legacy ``platform: "pc"``. There is no prior
evidence for a *character* prompt at all, which argues for breadth over depth:
ten configurations at five seeds each, twice, rather than two configurations at
twenty-five.

One tension is deliberately left in the base rather than edited out. The
composed baseline prompt reads "... grim dark mood, **vivid saturated colours**,
bold simple shapes, ..." -- ``art_style=snes`` contributes that phrase, and it
argues against the brief's black/silver/blue. That is what this taxonomy means
by SNES, so sweep B's ``palette`` axis (a ``steel`` baseline against ``mono``,
``muted`` and ``vibrant``) measures the tension instead of resolving it by
guess.

Seeds are fixed and shared by both plans, because a matched pair is two units
at the *same* seed: a per-sweep seed list would silently halve the comparison
count.

    uv run python scripts/sweep_rogue.py --dry-run   # plan and validate only
    uv run python scripts/sweep_rogue.py             # write the rows
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _campaign  # noqa: E402

from warlock.service import sweeps as sweeps_mod  # noqa: E402

PROMPT = "a SNES-era rogue adventurer with black and silver and blue color schemes"

# Five seeds, shared by both plans -- see the module docstring on why they may
# not diverge.
SEEDS = (11, 23, 42, 77, 101)

# What both sweeps hold fixed: the reading of the brief that is not in
# question. ``platform`` here is the *geometry* one (the 3D pane's), which is
# the half a model-stage unit runs on.
COMMON = {
    "art_style": "snes",
    "category": "character",
    "genre": "fantasy",
    "platform": "3d",
    "silhouette": "slender",
    "condition": "worn",
    "setting": "medieval",
    "mood": "grim",
    "reference_prep": True,
}

PLAN_A = sweeps_mod.SweepPlan(
    label="rogue - render",
    prompt=PROMPT,
    base={**COMMON, "base_model": "sdxl"},
    seeds=SEEDS,
    axes=(
        # sdxl is the base, so it is not repeated here: ``expand`` skips an
        # axis value equal to the base rather than planning a second copy of
        # the baseline.
        sweeps_mod.Axis("base_model", ("turbo", "playground", "sdxl_cfg", "pixel")),
        # No LoRA in the base, so all four are a real contrast against "none".
        # ps1 and pixelxl are included precisely because they are the tempting
        # reading of "SNES-era" and flat pixel art is poor trellis input -- the
        # sweep should measure that rather than assume it.
        sweeps_mod.Axis("style_lora", ("render3d", "redmond3d", "ps1", "pixelxl")),
        sweeps_mod.Axis("bg_removal", ("birefnet",)),
    ),
    stage="model",
)

PLAN_B = sweeps_mod.SweepPlan(
    label="rogue - depiction",
    prompt=PROMPT,
    base={
        **COMMON,
        "base_model": "sdxl",
        # Stated rather than left to a default: a sweep off an unstated
        # baseline is not reproducible, and ``lora_weight`` is dropped by
        # ``guidance.normalize`` entirely when no adapter is selected.
        "style_lora": "render3d",
        "lora_weight": 0.9,
        "palette": "steel",
    },
    seeds=SEEDS,
    axes=(
        sweeps_mod.Axis("silhouette", ("bulky", "compact", "angular")),
        sweeps_mod.Axis("palette", ("mono", "muted", "vibrant")),
        sweeps_mod.Axis("condition", ("pristine", "ancient")),
        sweeps_mod.Axis("mood", ("sinister",)),
    ),
    stage="model",
)

PLANS = (PLAN_A, PLAN_B)


if __name__ == "__main__":
    raise SystemExit(_campaign.main(PLANS, __doc__.splitlines()[0]))
