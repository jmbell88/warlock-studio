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
    parser.add_argument(
        "--verify", action="store_true",
        help="doctor: re-hash every installed model against what its download "
             "recorded. Reads gigabytes and takes minutes, which is why it is a "
             "flag and never runs on startup",
    )
    args = parser.parse_args()

    if args.command == "doctor":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        _run_doctor(verify=args.verify)
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
    try:
        from .studio.main import run
    except ImportError as exc:
        # ``studio`` is an *extra*, so a base install has this command on its
        # PATH and none of what it needs. The failure was an import traceback
        # naming moderngl or pygame -- accurate, and useless to somebody who
        # installed a thing and ran it (DST-03). The remedy is one line and it
        # is the same line the README opens with.
        raise SystemExit(
            f"Warlock Studio's window could not be loaded: {exc}\n\n"
            f"The app's renderer is an optional extra. Install it with:\n\n"
            f"    uv sync --extra studio --extra text2image --extra rig\n\n"
            f"`warlock doctor` and `warlock sweep` work without it."
        ) from exc

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


def _run_doctor(*, verify: bool = False) -> None:
    from .config import effective, get_config
    from .doctor import run_checks

    config = get_config()
    checks = run_checks(config)
    for check in checks:
        # Four words, not three. A row that is merely not downloaded yet is
        # the ordinary state of a fresh install, and printing WARN against it
        # told a user with nothing wrong that something was wrong. The exit
        # code below keys on ``fatal`` alone, so SETUP rows leave it at 0.
        if check.ok:
            status = "OK"
        elif check.fatal:
            status = "FATAL"
        elif check.pending_install:
            status = "SETUP"
        else:
            status = "WARN"
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
    corrupt = _run_verify(config) if verify else False
    if corrupt or any(not c.ok and c.fatal for c in checks):
        raise SystemExit(1)


def _run_verify(config: object) -> bool:
    """The integrity pass. -> whether anything is actually corrupt.

    Behind a flag and never at startup (the C29/C30 lesson): it re-reads every
    installed weight file, which is gigabytes and minutes, and nothing about
    opening the app is a request for that. ``suspect_files`` is the cheap probe
    that runs where a probe belongs.

    "unknown" is printed and is not a failure. A directory downloaded before
    manifests carried digests, or by hand with the pasted ``uvx hf download``
    line, has nothing to compare against -- and reporting that as a problem
    would light up every install that predates this and teach the reader that
    the check is noise.
    """
    from . import fetch

    print("\nIntegrity (re-hashed against each download's own manifest):")
    results = fetch.verify_all(config)  # type: ignore[arg-type]
    if not results:
        print("  nothing to check: no model directory carries a manifest")
        return False
    label = {
        fetch.VERIFY_OK: "OK",
        fetch.VERIFY_UNKNOWN: "UNKNOWN",
        fetch.VERIFY_BAD: "CORRUPT",
    }
    for result in results:
        print(f"  [{label[result.status]}] {result.dest.name}: {result.detail}")
    return any(r.status == fetch.VERIFY_BAD for r in results)
