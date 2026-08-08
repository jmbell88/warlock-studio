"""The submitter every sweep campaign under ``scripts/`` shares.

A campaign script is a *submitter*, not a runner: it writes ``queued`` rows
through ``service.sweeps.create_sweep`` and exits. The app's worker drains them
the next time Warlock is launched, and Review mode lists the sweeps for the
verdict loop. This is the ``worker=None`` path ``service/core.py`` documents --
"a test (or a headless tool) can exercise the pure-DB half without standing up
the GPU queue" -- so nothing here touches the card, the trellis port, or the
event loop.

It was ``sweep_rogue.py``'s ``main`` until there were three campaigns. Two
things in it are load-bearing and were the reason to share rather than copy.

**All-or-nothing across the whole campaign, not just per sweep.**
``create_sweep`` gives one sweep that guarantee; a campaign with two plans wants
it across both, or sweep A queues fifty jobs and sweep B is then refused,
leaving a corpus nobody asked for. So every unit of every plan is validated
before a single row is written.

**Refuse to run while another process is writing the database.** ``JobStore``
opens a plain rollback-journal connection with sqlite3's default five-second
busy timeout, so a second process writing while the app is mid-commit is a
narrow but real way to lose a submit.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from warlock.config import Config  # noqa: E402
from warlock.db import JobStore  # noqa: E402
from warlock.service import sweeps as sweeps_mod  # noqa: E402
from warlock.service.core import WarlockService  # noqa: E402
from warlock.service.errors import ServiceError  # noqa: E402


def require_no_live_writer(db_path: Path) -> None:
    """Refuse to run while another process is writing the job database.

    ``BEGIN IMMEDIATE`` takes sqlite's reserved lock, which is exactly what a
    running app's ``JobStore`` holds mid-commit. It cannot detect an *idle*
    app, so this is a guard against the dangerous case rather than a proof the
    coast is clear -- hence the message says what to do rather than claiming it.
    """
    if not db_path.exists():
        return
    probe = sqlite3.connect(db_path, timeout=2.0)
    try:
        probe.execute("BEGIN IMMEDIATE")
        probe.rollback()
    except sqlite3.OperationalError as exc:
        raise SystemExit(
            f"{db_path} is locked by another process ({exc}).\n"
            "Close Warlock Studio and run this again: two processes writing "
            "one rollback-journal sqlite file is how a submit gets lost."
        ) from exc
    finally:
        probe.close()


def plan_and_validate(svc: WarlockService, plan: sweeps_mod.SweepPlan) -> int:
    """-> the unit count, having run the same admission ``create_sweep`` will."""
    units = sweeps_mod.expand(plan)
    sweeps_mod._validate(svc, plan, units)
    return len(units)


def main(plans: tuple[sweeps_mod.SweepPlan, ...], description: str = "") -> int:
    """Validate every plan, then queue them. ``--dry-run`` stops after the first
    half, which is what the campaign spec tests assert."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and validate every sweep, write nothing",
    )
    args = parser.parse_args()

    config = Config()
    db_path = Path(config.db_path)
    require_no_live_writer(db_path)

    store = JobStore(db_path)
    svc = WarlockService(config, store)
    try:
        counts = []
        for plan in plans:
            try:
                count = plan_and_validate(svc, plan)
            except ServiceError as exc:
                print(f"refused: {plan.label}: {exc.message}", file=sys.stderr)
                return 1
            counts.append(count)
            print(f"{plan.label}: {count} units planned")
        total = sum(counts)
        print(f"total: {total} units across {len(plans)} sweep(s) -> {db_path}")

        if args.dry_run:
            print("dry run: nothing written")
            return 0

        for plan in plans:
            # Reported per sweep rather than as one line at the end: each
            # create_sweep is independently all-or-nothing, so if the second
            # fails the first is genuinely queued and the user needs its id.
            result = sweeps_mod.create_sweep(svc, plan)
            print(f"queued {result['units']} units as sweep {result['id']} ({plan.label})")
        print(
            f"\n{total} jobs queued. Launch Warlock Studio; the worker drains them "
            "and Review mode lists the sweep(s)."
        )
    finally:
        store.close()
    return 0
