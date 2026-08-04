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
    return tuple(int(v) for v in _csv(value))


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
        "--render", action="store_true", help="render 8 views per finished mesh (+1-1.5 h)"
    )
    run.add_argument("--resume", type=Path, default=None, help="an existing run directory")
    run.add_argument("--dry-run", action="store_true")

    cal = sub.add_parser(
        "calibrate", help="sweep yaw/elevation over finished meshes to find the matched view"
    )
    cal.add_argument("--job", action="append", default=[], help="repeatable; job ids")
    cal.add_argument("--all", action="store_true", help="every finished mesh in the data dir")
    cal.add_argument("--yaws", type=int, default=calibrate_mod.DEFAULT_YAWS)
    cal.add_argument("--elevations", type=_seeds, default=calibrate_mod.DEFAULT_ELEVATIONS)

    prune = sub.add_parser("prune", help="delete all but the newest runs")
    prune.add_argument("--keep", type=int, default=3)
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
                print(f"    {item.id}  {item.prompt}")
        return 0

    if args.command == "calibrate":
        return _calibrate(config, args)

    if args.command == "prune":
        removed = runner_mod.prune_runs(config, args.keep)
        print(f"removed {len(removed)} run(s)")
        return 0

    return _run(config, args)


def _run(config, args) -> int:
    started = _timestamp()
    suite = suite_mod.load(args.suite)
    recipe = recipe_mod.load(args.recipe)
    run_dir, todo, doc = runner_mod.plan_run(
        config, suite, recipe,
        stage=args.stage, started=started, categories=args.categories,
        ids=args.ids, limit=args.limit, seeds=args.seeds, run_dir=args.resume,
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
            resume=args.resume, on_event=print,
        )
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
