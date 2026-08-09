"""Re-queue the units of a sweep that produced no terminal measurement.

A long sweep loses units. Closing the app mid-run leaves them ``error`` with
"interrupted by shutdown"; cancelling one leaves it ``cancelled``. Neither is a
measurement, and both leave a hole in the matched pairs -- the 2026-08-09
re-baseline lost seven units that way, including the whole ``base_model=pixel``
arm, so that axis had no data at all rather than a weak result.

**Why this cannot be ``rerun_job``.** A reroll copies *params* and nothing else,
deliberately: ``sweep_id``/``sweep_unit`` are columns rather than params keys
precisely so membership can never leak onto a reroll. That is the right rule and
it makes a reroll useless here -- the replacement would be an ordinary library
job, invisible to the sweep, and ``findings.comparisons`` pairs two rows only
when they share a ``sweep_id``. So the units are re-queued *into the sweep they
belong to*, through ``create_job``'s own ``sweep_id``/``sweep_unit`` kwargs,
which is exactly what ``create_sweep`` does internally.

**The spec is read back from the sweep row, never restated here.** ``sweeps``
stores the base, seeds, axes, vectors and stage it was created with, so the
refilled unit is planned by the same ``expand`` from the same plan and validated
by the same ``_validate``. A hand-written replacement would be a second spelling
of the run's settings and would drift on the first edit.

**What is deliberately not re-run: a genuine refusal.** A unit that errored at
the composition gate *is* a measurement -- it writes an observation carrying
``refused`` and ``refused_<code>``, and the whole reason the worker records a
terminal ``error`` is that the 17 refusals of the 2026-08-07 sweep wrote nothing
and flattered every checkpoint that fails most often. Re-rolling one would
replace a real negative with whatever the next seed happened to draw. Only
``cancelled`` and shutdown-interrupted units are refilled, and ``--list`` shows
which is which before anything is written.

    uv run python scripts/sweep_refill.py --list &lt;sweep_id&gt;
    uv run python scripts/sweep_refill.py --dry-run &lt;sweep_id&gt;
    uv run python scripts/sweep_refill.py &lt;sweep_id&gt;
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _campaign  # noqa: E402

from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402
from warlock.service import jobs as jobs_mod  # noqa: E402
from warlock.service import sweeps as sweeps_mod  # noqa: E402
from warlock.service.core import WarlockService  # noqa: E402
from warlock.service.errors import ServiceError  # noqa: E402

# What marks a row as lost rather than measured. A shutdown is matched on the
# message the worker writes when the loop is torn down mid-job; anything else
# that errored reached a real refusal or a real failure and stays.
SHUTDOWN = "interrupted by shutdown"


def plan_from_spec(sweep: dict) -> sweeps_mod.SweepPlan:
    """The ``SweepPlan`` this sweep was created from, out of its stored spec."""
    spec = sweep["spec"] if isinstance(sweep["spec"], dict) else json.loads(sweep["spec"])
    return sweeps_mod.SweepPlan(
        label=sweep["label"],
        prompt=sweep["prompt"],
        base=dict(spec.get("base") or {}),
        seeds=tuple(spec.get("seeds") or ()),
        axes=tuple(
            sweeps_mod.Axis(a["param"], tuple(a["values"])) for a in spec.get("axes") or ()
        ),
        vectors=tuple(dict(v) for v in spec.get("vectors") or ()),
        stage=spec.get("stage") or "model",
    )


def lost_units(svc: WarlockService, sweep_id: str) -> tuple[list[str], list[str]]:
    """-> (unit labels with no measurement, unit labels genuinely refused)."""
    lost: list[str] = []
    refused: list[str] = []
    for job in svc.store.sweep_jobs(sweep_id):
        status = job["status"]
        if status == "cancelled" or (status == "error" and SHUTDOWN in (job["error"] or "")):
            lost.append(job["sweep_unit"])
        elif status == "error":
            refused.append(job["sweep_unit"])
    return lost, refused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_id")
    parser.add_argument("--list", action="store_true", help="show what would be refilled")
    parser.add_argument("--dry-run", action="store_true", help="plan and validate, write nothing")
    args = parser.parse_args()

    config = Config()
    db_path = Path(config.db_path)
    _campaign.require_no_live_writer(db_path)

    store = JobStore(db_path)
    svc = WarlockService(config, store)
    try:
        sweep = store.get_sweep(args.sweep_id)
        if sweep is None:
            print(f"no such sweep: {args.sweep_id}", file=sys.stderr)
            return 1
        plan = plan_from_spec(sweep)
        lost, refused = lost_units(svc, args.sweep_id)

        print(f"{sweep['label']} ({args.sweep_id})")
        for label in sorted(refused):
            print(f"  keeping   {label}  (a refusal is a measurement)")
        for label in sorted(lost):
            print(f"  refilling {label}")
        if not lost:
            print("nothing to refill")
            return 0

        # Planned by the same expand, so a refilled unit's settings are the ones
        # the sweep was created with rather than a restatement of them.
        wanted = {sweeps_mod.unit_label(u): u for u in sweeps_mod.expand(plan)}
        missing = [label for label in lost if label not in wanted]
        if missing:
            print(f"not in the sweep's own plan: {missing}", file=sys.stderr)
            return 1

        units = [wanted[label] for label in lost]
        try:
            for unit in units:
                sweeps_mod._check_unit(svc, plan, unit)
        except ServiceError as exc:
            print(f"refused: {exc.message}", file=sys.stderr)
            return 1
        print(f"\n{len(units)} unit(s) validated")

        if args.list or args.dry_run:
            print("nothing written")
            return 0

        for unit in units:
            job = jobs_mod.create_job(
                svc,
                **sweeps_mod.unit_kwargs(plan, unit),
                sweep_id=args.sweep_id,
                sweep_unit=sweeps_mod.unit_label(unit),
            )
            print(f"  queued {sweeps_mod.unit_label(unit)} as {job['id']}")
        print(f"\n{len(units)} jobs queued into sweep {args.sweep_id}.")
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
