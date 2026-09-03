"""Queue the trellis guidance-strength / token-budget sweep over a subject list.

The instrument ``docs/measurements/2026-09-02-trellis-guidance-sweep.md``
pre-registers: for each subject, one sweep of six units -- the omitted rung
(the exe's own defaults, which are ``gss = gsh = 7.5`` and ``max_tokens =
49152`` in ``include/trellis_args.h`` at v0.6.0, not printed by ``--help``),
``trellis_gss`` at 0.7x and 1.4x, ``trellis_gsh`` at 0.7x and 1.4x, and
``trellis_max_tokens`` at 98304. OFAT, not a cross.

One sweep *per subject* because a ``SweepPlan`` carries exactly one prompt
(``campaign_props.py`` says why a corpus is not a sweep). Every unit is also
tagged, so ``scripts/hole_audit_vs_grade.py --tag <tag>`` tabulates the whole
campaign in one table with the unit label beside each row.

A submitter, not a runner, like ``_campaign.py``.

    uv run python scripts/campaign_guidance.py --subjects holed.txt --tag guidance-v1 --dry-run
    uv run python scripts/campaign_guidance.py --subjects holed.txt --tag guidance-v1

``--subjects`` is one bare prompt per line (blank and ``#`` lines skipped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _campaign  # noqa: E402

from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402
from warlock.service import jobs as jobs_mod  # noqa: E402
from warlock.service import sweeps as sweeps_mod  # noqa: E402
from warlock.service.core import WarlockService  # noqa: E402
from warlock.service.errors import ServiceError  # noqa: E402

#: The exe's own defaults at v0.6.0 (``include/trellis_args.h``). The omitted
#: rung passes nothing and therefore runs exactly these; the neighbours are
#: relative to them.
EXE_GSS = 7.5
EXE_GSH = 7.5
EXE_MAX_TOKENS = 49152

RATIOS = (0.7, 1.4)


def read_subjects(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"no subjects file at {path}")
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if text and not text.startswith("#"):
            out.append(text)
    if not out:
        raise SystemExit(f"{path} names no subjects")
    return out


def plan_for(prompt: str, seed: int) -> sweeps_mod.SweepPlan:
    return sweeps_mod.SweepPlan(
        label=f"guidance: {prompt[:40]}",
        prompt=prompt,
        seeds=(seed,),
        axes=(
            sweeps_mod.Axis("trellis_gss", tuple(round(EXE_GSS * r, 3) for r in RATIOS)),
            sweeps_mod.Axis("trellis_gsh", tuple(round(EXE_GSH * r, 3) for r in RATIOS)),
            sweeps_mod.Axis("trellis_max_tokens", (EXE_MAX_TOKENS * 2,)),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subjects = read_subjects(args.subjects)
    plans = [plan_for(p, args.seed) for p in subjects]

    config = Config()
    db_path = Path(config.db_path)
    _campaign.require_no_live_writer(db_path)
    store = JobStore(db_path)
    svc = WarlockService(config, store)
    try:
        total = 0
        for plan in plans:
            try:
                total += _campaign.plan_and_validate(svc, plan)
            except ServiceError as exc:
                print(f"refused: {plan.label}: {exc.message}", file=sys.stderr)
                return 1
        print(f"{len(plans)} subjects, {total} units -> {db_path}")
        if args.dry_run:
            print("dry run: nothing written")
            return 0
        for plan in plans:
            result = sweeps_mod.create_sweep(svc, plan)
            for job in store.sweep_jobs(result["id"]):
                jobs_mod.update_job(svc, job["id"], {"tags": [args.tag]})
            print(f"queued {result['units']} units as sweep {result['id']} ({plan.label})")
        print(f"\n{total} jobs queued, tagged {args.tag!r}.")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
