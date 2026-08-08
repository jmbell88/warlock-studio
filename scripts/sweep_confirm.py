"""Queue the blind ``bg_removal`` confirm: birefnet against auto, one knob.

**Run this before anything is built on the matte finding.** `TODO.md` §2 run 1.
The 2026-08-07 review produced exactly one signal in 84 verdicts -- all three
accepts were ``bg_removal=birefnet`` and ``auto`` went 0 for 80 -- and three
things make it worth acting on: the matched pairs are byte-identical upstream
(the same ``input.png`` hash, so nothing about the *picture* differs and only
the reconstruction matte does), the dominant failure tag disappears rather than
merely thinning, and the accepts are spread across review positions 46, 48 and
83 rather than clustered where a tiring reviewer drifts.

What is nevertheless weak about it, and what this run fixes:

- **The clean 2x2 on its own is Fisher p=0.14.** The p ~ 4e-5 figure comes from
  using all 80 ``auto`` units as controls, which is legitimate but leans on
  their comparability -- they varied checkpoint, LoRA and four depiction axes.
  Ten units with one knob lean on nothing.
- **The review was unblinded and single-reviewer,** and the app shows a unit's
  arm in the unit list and the verdict header. Turn **Blind** on in Review
  before judging these: it renames every unit to an id prefix and reorders them
  by a digest of the job id, because ``sweeps.expand`` enqueues the baseline
  first and position would otherwise name the arm as plainly as the label.

Why it is this shape. One axis, two values, five seeds -- ten units, which is
inside the 8-12 the plan asks for and roughly twenty minutes of card time
against the 3.7 h the original sweep cost. The subject and the seeds are
``sweep_rogue``'s, deliberately: the corpus is scoped by ``prompt_hash``, so a
confirm on a different prompt confirms nothing about this finding, and reusing
the seeds means the new pairs sit beside the old ones.

Everything else is pinned at the values the original render sweep's baseline
used, ``base_model=sdxl`` included. That is not a claim that sdxl is the right
checkpoint -- §2 is explicit that no checkpoint has been chosen and that the
re-baseline run is what settles it -- it is the opposite: holding the rest fixed
at the arm the signal was found under is what makes this a confirm rather than a
second sweep.

    uv run python scripts/sweep_confirm.py --dry-run   # plan and validate only
    uv run python scripts/sweep_confirm.py             # write the rows
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
    label="bg_removal confirm (blind)",
    prompt=PROMPT,
    base={
        **COMMON,
        "base_model": "sdxl",
        # Stated, not inherited. ``guidance.default_bg_removal`` is gated on
        # ``birefnet.gguf`` being in the weights directory, so a plan that left
        # this out would run the *fallback* matte on a host that never
        # downloaded it -- and this run's entire subject is which matte ran.
        "bg_removal": "birefnet",
    },
    seeds=SEEDS,
    # The one knob. ``expand`` skips an axis value equal to the base, so
    # ``birefnet`` is not repeated here: the baseline unit *is* the birefnet arm.
    axes=(sweeps_mod.Axis("bg_removal", ("auto",)),),
    stage="model",
)

PLANS = (PLAN,)


if __name__ == "__main__":
    raise SystemExit(_campaign.main(PLANS, __doc__.splitlines()[0]))
