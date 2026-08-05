"""``python -m warlock.bench <subcommand>``.

Deliberately not part of ``cli.py``. That module's flat
``choices=["doctor", "sweep"]`` would have to become a subparser tree to host
this, and ``warlock`` is the user-facing entry point while everything here is
a developer tool.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import Any

from ..config import get_config
from . import calibrate as calibrate_mod
from . import metrics as metrics_mod
from . import recipe as recipe_mod
from . import runner as runner_mod
from . import suite as suite_mod


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _seeds(value: str) -> tuple[int, ...]:
    """Seeds, bounded here rather than at submit.

    ``create_job`` enforces 0..2**31-1 and raises. That raise used to arrive
    inside the run loop with no handler -- after the Runtime was stood up and
    the GPU worker started -- so a typo'd seed aborted the whole run instead of
    the argument parse.
    """
    from ..service.validation import MAX_SEED

    out = []
    for raw in _csv(value):
        seed = int(raw)
        if not 0 <= seed <= MAX_SEED:
            raise argparse.ArgumentTypeError(f"seed {seed} is outside 0..{MAX_SEED}")
        out.append(seed)
    return tuple(out)


def _angles(value: str) -> tuple[float, ...]:
    """Elevations are degrees and the sweep consumes floats. This used to reuse
    the seed parser, so ``--elevations 22.5`` was an argparse error for a value
    the code widens back to float one line later."""
    return tuple(float(v) for v in _csv(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warlock.bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("suites", help="list the available suites")
    sub.add_parser("recipes", help="list the available recipes")

    show = sub.add_parser("suite", help="describe one suite")
    show.add_argument("key", nargs="?", default=suite_mod.DEFAULT_SUITE)

    run = sub.add_parser("run", help="generate a suite under a recipe")
    run.add_argument("--recipe", required=True)
    run.add_argument("--suite", default=suite_mod.DEFAULT_SUITE)
    run.add_argument(
        "--stage",
        default="reference",
        choices=("reference", "model"),
        help="reference is ~10-45 min for 160 jobs; model is 6-8 h of GPU",
    )
    run.add_argument("--seeds", type=_seeds, default=(), help="override the suite's seeds")
    run.add_argument("--filter", type=_csv, default=(), dest="categories")
    run.add_argument("--ids", type=_csv, default=())
    run.add_argument("--limit", type=int, default=0, help="first N items after filtering")
    run.add_argument(
        "--render",
        action="store_true",
        help="render 8 views per finished mesh (+1-1.5 h); required by `score`",
    )
    run.add_argument(
        "--keep-source",
        action="store_true",
        help="also copy source.glb into each item dir (large; nothing measures it)",
    )
    run.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="an existing run directory; its own item/seed selection is reused",
    )
    run.add_argument("--dry-run", action="store_true")

    score = sub.add_parser("score", help="score a finished run's rendered views")
    score.add_argument("run_dir", type=Path, help="a run directory under bench/runs")
    score.add_argument(
        "--against",
        type=Path,
        default=None,
        help="a second scored run to compare against -- the A/B",
    )

    cal = sub.add_parser(
        "calibrate", help="sweep yaw/elevation over finished meshes to find the matched view"
    )
    cal.add_argument("--job", action="append", default=[], help="repeatable; job ids")
    cal.add_argument("--all", action="store_true", help="every finished mesh in the data dir")
    cal.add_argument("--yaws", type=int, default=calibrate_mod.DEFAULT_YAWS)
    cal.add_argument("--elevations", type=_angles, default=calibrate_mod.DEFAULT_ELEVATIONS)

    prune = sub.add_parser("prune", help="delete all but the newest runs")
    prune.add_argument("--keep", type=int, default=3)

    purge = sub.add_parser(
        "purge",
        help=(
            "free a finished run's disk space: delete its meshes, references and "
            "rendered views, keeping its items, manifest and scores"
        ),
    )
    purge.add_argument("run_dir", type=Path, nargs="*", help="run directories under bench/runs")
    purge.add_argument("--all", action="store_true", help="every run under bench/runs")

    return parser


def _finished_meshes(config) -> list[Path]:
    root = Path(config.data_dir)
    if not root.exists():
        return []
    return sorted(
        d for d in root.iterdir() if (d / "model.glb").exists() and (d / "input.png").exists()
    )


def _calibrate(config, args) -> int:
    if args.all or not args.job:
        job_dirs = _finished_meshes(config)
    else:
        job_dirs = [Path(config.job_dir(j)) for j in args.job]
    if not job_dirs:
        print("no finished meshes found; run `bench run --stage model` first")
        return 1

    out_root = Path(config.bench_dir) / "calibrate"
    results = []
    for job_dir in job_dirs:
        print(f"{job_dir.name}:")
        try:
            result = calibrate_mod.sweep_job(
                job_dir,
                out_root / job_dir.name,
                yaws=args.yaws,
                elevations=tuple(float(e) for e in args.elevations),
                config=config,
                on_event=print,
            )
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue
        results.append(result)
        for metric in metrics_mod.available(config):
            if metric in result:
                print(calibrate_mod.format_table(result, metric))

    if not results:
        return 1
    path = calibrate_mod.write_report(out_root, results, config)
    print()
    for v in calibrate_mod.verdict_lines(results, config):
        print(v)
    print(f"\nwrote {path}")
    return 0


def _score(config, args) -> int:
    from . import score as score_mod

    if not args.run_dir.is_dir():
        print(f"no such run directory: {args.run_dir}")
        return 1
    try:
        doc = score_mod.score_run(args.run_dir, config, on_event=print)
    except ValueError as exc:
        print(str(exc))
        return 1

    print()
    for line in score_mod.summary_lines(doc):
        print(line)
    if not doc["summary"]["scored"]:
        print("\nNothing was scored. A run needs --stage model and --render.")

    if args.against is not None:
        try:
            other = score_mod.read_scores(args.against)
        except (OSError, ValueError):
            print(f"\n{args.against} has no scores.json; score it first")
            return 1
        print(f"\n{other.get('run')} -> {doc.get('run')}:")
        for line in score_mod.compare_lines(other, doc):
            print(line)
    print(f"\nwrote {args.run_dir / score_mod.FILENAME}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    config = get_config()

    if args.command == "suites":
        for key in suite_mod.available():
            s = suite_mod.load(key)
            print(f"{key}  {len(s.items)} items x {len(s.seeds)} seeds  {s.label}")
        return 0

    if args.command == "recipes":
        for key in recipe_mod.available():
            print(f"{key}  {recipe_mod.load(key).label}")
        return 0

    if args.command == "suite":
        s = suite_mod.load(args.key)
        print(f"{s.key}: {s.label}")
        print(f"seeds: {list(s.seeds)}")
        for category, items in s.by_category().items():
            print(f"  {category} ({len(items)})")
            for item in items:
                tags = f"  [{', '.join(item.tags)}]" if item.tags else ""
                print(f"    {item.id}  {item.prompt}{tags}")
        return 0

    if args.command == "score":
        return _score(config, args)

    if args.command == "calibrate":
        return _calibrate(config, args)

    if args.command == "prune":
        removed = runner_mod.prune_runs(config, args.keep)
        print(f"removed {len(removed)} run(s)")
        return 0

    if args.command == "purge":
        return _purge(config, args)

    return _run(config, args)


def _run(config, args) -> int:
    started = _timestamp()
    suite = suite_mod.load(args.suite)
    recipe = recipe_mod.load(args.recipe)
    # A category that matches nothing selects zero items and runs happily,
    # reporting nothing wrong. Say so instead.
    unknown = sorted(set(args.categories) - set(suite.by_category()))
    if unknown:
        print(f"unknown categor{'y' if len(unknown) == 1 else 'ies'} {unknown}")
        print(f"{args.suite} has: {sorted(suite.by_category())}")
        return 2
    selection: dict[str, Any] = {
        "categories": args.categories,
        "ids": args.ids,
        "limit": args.limit,
        "seeds": args.seeds,
    }
    if args.resume is not None:
        # Same rule ``run`` applies: a resume inherits the original selection.
        # Planning the preview from this invocation's flags would print a
        # different plan from the one about to execute.
        from . import manifest as manifest_mod

        try:
            stored = manifest_mod.read_manifest(args.resume)
        except (OSError, ValueError) as exc:
            # A path that is not a run directory is the common typo here, and
            # a bare FileNotFoundError traceback says nothing about what the
            # argument was supposed to be.
            raise SystemExit(
                f"{args.resume}: not a bench run directory "
                f"({manifest_mod.FILENAME} is unreadable: {exc})"
            ) from None
        selection = {
            "categories": (),
            "ids": tuple(stored.get("items") or ()),
            "limit": 0,
            "seeds": tuple(stored.get("seeds") or ()),
        }
    run_dir, todo, doc = runner_mod.plan_run(
        config, suite, recipe,
        stage=args.stage, started=started, run_dir=args.resume, **selection,
    )
    print(f"{len(todo)} job(s): {len(doc['items'])} items x {len(doc['seeds'])} seeds")
    print(f"stage: {args.stage}   run dir: {run_dir}")
    if args.dry_run:
        for unit in todo:
            print(f"  {unit.key}  {unit.item.prompt}")
        return 0

    try:
        out = runner_mod.run(
            config,
            suite_key=args.suite, recipe_key=args.recipe, stage=args.stage,
            started=started, categories=args.categories, ids=args.ids,
            limit=args.limit, seeds=args.seeds, render=args.render,
            keep_source=args.keep_source, resume=args.resume, on_event=print,
        )
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    print(f"wrote {out}")
    return 0


def _mib(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MiB"


def _purge(config, args) -> int:
    """Free disk space, keeping every measurement.

    Deliberately explicit: a run directory or ``--all``, never a "newest N"
    default like ``prune``'s. Purging is not reversible and the thing it costs
    -- re-scoring a run with a new metric, and looking at its meshes in Review
    -- is not obvious from the subcommand's name.
    """
    targets = [Path(p) for p in args.run_dir]
    if args.all:
        root = Path(config.bench_dir) / "runs"
        targets = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not targets:
        print("nothing to purge; name a run directory or pass --all")
        return 1

    total = 0
    for run_dir in targets:
        if not run_dir.is_dir():
            print(f"{run_dir}: no such run directory")
            continue
        if runner_mod.is_purged(run_dir):
            print(f"{run_dir.name}: already purged")
            continue
        doc = runner_mod.purge_media(run_dir)
        total += int(doc["freed_bytes"])
        print(f"{run_dir.name}: freed {_mib(doc['freed_bytes'])} ({len(doc['files'])} files)")
    print(f"freed {_mib(total)} in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
