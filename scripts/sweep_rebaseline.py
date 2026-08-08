"""Queue the re-baselined render sweep: birefnet as the baseline, plus framing.

`TODO.md` §2 run 2, and it is the second of the two runs on purpose -- run
``sweep_confirm.py`` first and look at its accept rate. **The rule this campaign
exists to obey: establish a baseline that produces acceptable output at a
workable rate before fanning out.** Sweep B spent roughly half of a 3.7 h run
measuring nothing because ``bg_removal`` was pinned at ``auto`` -- the variable
that dominates the verdict -- while four depiction axes varied. That is a floor
effect, not a null result: one-factor-at-a-time around a baseline that fails
~96% of the time has no headroom to detect an improvement in anything.

So this run varies **how the reference is drawn** and nothing else, over a
baseline whose matte is the learned one:

- ``base_model``, four checkpoints against ``sdxl``. Unsettled, and settled by
  this run rather than by the old one: refusal rate and mesh quality ranked the
  checkpoints *oppositely* in the rogue sweep (``playground`` and ``redmond3d``
  refused 0/5 each; ``sdxl_cfg`` refused 3/5 but its survivors were flawless),
  and every one of them ran under ``auto``, so all of them were being judged
  through the same defect. The re-run says it itself now: a refusal is an
  observation, so ``findings.json`` carries ``refused_multi_object`` as a
  per-checkpoint rate and the hint under the base-model select reads it.
- ``style_lora``, four adapters against none. ``ps1`` and ``pixelxl`` are in
  again for the reason ``sweep_rogue`` gives: they are the tempting reading of
  "SNES-era", flat pixel art is poor trellis input, and the sweep should measure
  that rather than assume it.
- ``framing``, the axis Night 1's 0e made expressible: a front orthographic
  plate against the global 3/4 view. It is a *camera* claim and cannot
  contradict the T-pose fragment ``category=character`` already carries, and it
  is here rather than in a depiction run because it changes how the subject is
  presented to trellis rather than what the subject is.

**The depiction axes are deliberately absent.** Silhouette, palette, condition
and mood come after this run reports a workable accept rate -- fanning out first
is the exact mistake §2 records. `TODO.md` §5 is the one depiction question
already written down (``art_style=snes`` contributes "vivid saturated colours"
against a black/silver/blue brief), and it waits on the same gate.

**Two measurement notes for whoever reads the verdicts.**

``PROMPT_TEMPLATE`` moved since the rogue sweep (``PROMPT_VERSION`` 3 -> 4), so
a unit here is not comparable with one from that run on the prompt axis either.
All the changes are deliberate and all land before this run, which is the right
order -- a sweep around a broken base measures the brokenness -- but this is the
first corpus in which any of them is measured.

And ``hole_worst`` must not be used to rank the output. AUC(hole_worst ->
reject) = 0.115 on the existing corpus: not weakly informative, *backwards*,
because a slab has no holes and ``meshaudit`` scores the dominant failure mode
as perfect. The verdicts are human, and the measurement doc this run owes
supersedes ``docs/measurements/2026-08-04-hole-rate-baseline.md``.

    uv run python scripts/sweep_rebaseline.py --dry-run   # plan and validate
    uv run python scripts/sweep_rebaseline.py             # write the rows
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _campaign  # noqa: E402
from sweep_rogue import COMMON, PROMPT, SEEDS  # noqa: E402

from warlock.service import sweeps as sweeps_mod  # noqa: E402

PLAN = sweeps_mod.SweepPlan(
    label="rogue - render (birefnet baseline)",
    prompt=PROMPT,
    base={
        **COMMON,
        "base_model": "sdxl",
        # The change this whole run is named for. Stated rather than inherited:
        # the default is host-gated on birefnet.gguf, and a baseline that
        # depends on which weights a machine happens to hold is not a baseline.
        "bg_removal": "birefnet",
        # Also stated, though it is the default: framing is an axis here, and an
        # axis whose baseline is implicit is one nobody can read back off the
        # spec afterwards.
        "framing": "three_quarter",
    },
    seeds=SEEDS,
    axes=(
        sweeps_mod.Axis("base_model", ("turbo", "playground", "sdxl_cfg", "pixel")),
        sweeps_mod.Axis("style_lora", ("render3d", "redmond3d", "ps1", "pixelxl")),
        sweeps_mod.Axis("framing", ("front_ortho",)),
    ),
    stage="model",
)

PLANS = (PLAN,)


if __name__ == "__main__":
    raise SystemExit(_campaign.main(PLANS, __doc__.splitlines()[0]))
