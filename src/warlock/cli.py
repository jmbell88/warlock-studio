"""Entry point: `warlock` opens the desktop app.

`warlock doctor` checks dependencies and configuration; `warlock sweep`
measures mesh quality across trellis-server's --band values.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Warlock Studio — local AI 3D asset generator")
    parser.add_argument(
        "command", nargs="?", choices=["doctor", "sweep"], default=None,
        help="omit to open the app; 'doctor' checks dependencies and configuration; "
             "'sweep' measures mesh quality across trellis --band values",
    )
    # sweep only. Kept as plain options rather than a subparser so the
    # no-command default (open the app) stays exactly as it was.
    parser.add_argument(
        "--image", type=Path,
        help="sweep: reference PNG to generate from, e.g. assets/<job-id>/input.png",
    )
    parser.add_argument(
        "--bands", default=None,
        help="sweep: comma-separated band values; 'auto' means the exe's own heuristic",
    )
    parser.add_argument("--seed", type=int, default=42, help="sweep: fixed seed for every band")
    parser.add_argument(
        "--resolution", type=int, default=1024, help="sweep: geometry resolution per generation"
    )
    parser.add_argument(
        "--audit-resolution", type=int, default=1024,
        help="sweep: silhouette resolution for the hole measurement",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("sweep"), help="sweep: output directory for GLBs and logs"
    )
    args = parser.parse_args()

    if args.command == "doctor":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        _run_doctor()
        return
    if args.command == "sweep":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        _run_sweep(args)
        return

    # No basicConfig on the app path: studio.main._setup_logging owns the root
    # logger there, and configuring it first is what silently cost us the file
    # log (basicConfig is a no-op once the root has handlers).

    # Imported here, not at module scope: doctor and sweep must keep working on
    # a machine with no display and no GL, and importing the app pulls in
    # pygame and moderngl.
    from .studio.main import run

    raise SystemExit(run())


def _run_sweep(args: argparse.Namespace) -> None:
    import asyncio

    from . import sweep as sweep_mod
    from .config import get_config

    if args.image is None:
        raise SystemExit(
            "sweep needs --image: point it at a reference PNG, e.g. "
            "assets/<job-id>/input.png from a job whose mesh you want to improve"
        )
    try:
        bands = sweep_mod.parse_bands(args.bands or sweep_mod.DEFAULT_BANDS)
    except ValueError as exc:
        raise SystemExit(f"bad --bands: {exc}") from exc

    rows = asyncio.run(
        sweep_mod.sweep(
            get_config(),
            args.image,
            bands,
            args.out,
            seed=args.seed,
            resolution=args.resolution,
            audit_resolution=args.audit_resolution,
        )
    )
    sweep_mod.print_table(rows, args.audit_resolution)


def _run_doctor() -> None:
    from .config import effective, get_config
    from .doctor import run_checks

    config = get_config()
    checks = run_checks(config)
    for check in checks:
        status = "OK" if check.ok else ("FATAL" if check.fatal else "WARN")
        print(f"[{status}] {check.name}: {check.detail}")
    # After the checks, not before (S140): the checks are the answer and this is
    # the context for it. A host whose rows disagree with the documentation
    # usually disagrees because something in its environment says so, and the
    # env column is what makes that visible without asking the user to dump
    # their shell.
    print("\nEffective configuration:")
    for setting in effective(config):
        where = f"  <- {setting.env}" if setting.from_env else ""
        print(f"  {setting.name} = {setting.value}{where}")
    if any(not c.ok and c.fatal for c in checks):
        raise SystemExit(1)
