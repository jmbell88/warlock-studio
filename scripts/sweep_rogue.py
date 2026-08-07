"""Queue the two "SNES rogue adventurer" sweeps onto the live job queue.

A headless *submitter*, not a runner. It writes 100 ``queued`` rows through
``service.sweeps.create_sweep`` and exits; the app's worker drains them the
next time Warlock is launched, and Review mode lists both sweeps for the
verdict loop. This is the ``worker=None`` path ``service/core.py`` documents --
"a test (or a headless tool) can exercise the pure-DB half without standing up
the GPU queue" -- so nothing here touches the card, the trellis port, or the
event loop. Run it with the app closed all the same: ``JobStore`` opens a plain
rollback-journal connection with sqlite3's default five-second busy timeout, so
a second process writing while the app is mid-commit is a narrow but real way
to lose a submit. ``_require_no_live_writer`` probes for exactly that.

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


def _require_no_live_writer(db_path: Path) -> None:
    """Refuse to run while another process is writing the job database.

    ``BEGIN IMMEDIATE`` takes sqlite's reserved lock, which is exactly what a
    running app's ``JobStore`` holds mid-commit. It cannot detect an *idle*
    app, so this is a guard against the dangerous case rather than a proof the
    app is closed -- hence the message says what to do rather than claiming the
    coast is clear.
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


def _plan_and_validate(svc: WarlockService, plan: sweeps_mod.SweepPlan) -> int:
    """-> the unit count, having run the same admission ``create_sweep`` will.

    Every unit of *both* plans is checked before either is written, so the
    all-or-nothing guarantee ``create_sweep`` gives one sweep extends across
    the pair: the failure this prevents is sweep A queueing fifty jobs and
    sweep B then being refused, which is a corpus nobody asked for.
    """
    units = sweeps_mod.expand(plan)
    sweeps_mod._validate(svc, plan, units)
    return len(units)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan and validate both sweeps, write nothing",
    )
    args = parser.parse_args()

    config = Config()
    db_path = Path(config.db_path)
    _require_no_live_writer(db_path)

    store = JobStore(db_path)
    svc = WarlockService(config, store)
    try:
        counts = []
        for plan in PLANS:
            try:
                count = _plan_and_validate(svc, plan)
            except ServiceError as exc:
                print(f"refused: {plan.label}: {exc.message}", file=sys.stderr)
                return 1
            counts.append(count)
            print(f"{plan.label}: {count} units planned")
        total = sum(counts)
        print(f"total: {total} units across {len(PLANS)} sweeps -> {db_path}")

        if args.dry_run:
            print("dry run: nothing written")
            return 0

        for plan in PLANS:
            # Reported per sweep rather than as one line at the end: each
            # create_sweep is independently all-or-nothing, so if the second
            # fails the first is genuinely queued and the user needs its id.
            result = sweeps_mod.create_sweep(svc, plan)
            print(f"queued {result['units']} units as sweep {result['id']} ({plan.label})")
        print(
            f"\n{total} jobs queued. Launch Warlock Studio; the worker drains them "
            "and Review mode lists both sweeps."
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
