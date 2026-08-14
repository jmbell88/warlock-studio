"""Queue the mesh-regression check: two re-baseline replicas and one rock.

**This measures one thing: whether 2026-08-13's pipeline still produces what
2026-08-09's did.** The tier-qualification corpus went 0 of 20
(``docs/measurements/2026-08-13-tier-qualification.md``) at settings composed
from a re-baseline that accepted 19 of 41, and no subject exists on both sides
of that gap -- the re-baseline's rate was measured on the SNES rogue alone. So
R1 and R2 re-run the rogue at two of the re-baseline's own configurations,
imported from ``sweep_rogue`` rather than restated so the replica cannot drift
from what it replicates, and T1 re-runs the tier rock the same way for the
artifacts the cleaned library no longer holds. The decision rules live in
``docs/measurements/2026-08-13-mesh-regression-check.md`` and were written
before this script was run.

Seeds are the re-baseline's own minus 23 for R1/R2 (its baseline s23 was
refused at the composition gate, so a replica of it measures the gate, not the
mesh), and the tier run's 11/42 for T1 -- 42 drew the only non-negative grade
of that corpus, 11 a -5, so the pair brackets it.

    uv run python scripts/sweep_regression_check.py --dry-run
    uv run python scripts/sweep_regression_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _campaign  # noqa: E402
from sweep_props import SUBJECTS, WINNER  # noqa: E402
from sweep_rogue import COMMON, PROMPT  # noqa: E402

from warlock.service import sweeps as sweeps_mod  # noqa: E402

# The re-baseline's base, spelled the way sweep_rebaseline.py spells it: the
# matte and the framing stated rather than inherited, for its own reason.
_REBASELINE_BASE = {
    **COMMON,
    "bg_removal": "birefnet",
    "framing": "three_quarter",
}

_ROCK = next(s for s in SUBJECTS if s[0] == "rock")

PLANS = (
    sweeps_mod.SweepPlan(
        label="regression check - R1 rogue on sdxl",
        prompt=PROMPT,
        base={**_REBASELINE_BASE, "base_model": "sdxl"},
        seeds=(11, 42, 77),
        axes=(),
        stage="model",
    ),
    sweeps_mod.SweepPlan(
        label="regression check - R2 rogue on playground",
        prompt=PROMPT,
        base={**_REBASELINE_BASE, "base_model": "playground"},
        seeds=(11, 42, 77),
        axes=(),
        stage="model",
    ),
    sweeps_mod.SweepPlan(
        label="regression check - T1 tier rock",
        prompt=_ROCK[1],
        base={**{k: v for k, v in WINNER.items() if v is not None}, **_ROCK[2]},
        seeds=(11, 42),
        axes=(),
        stage="model",
    ),
)


if __name__ == "__main__":
    raise SystemExit(_campaign.main(PLANS, __doc__.splitlines()[0]))
