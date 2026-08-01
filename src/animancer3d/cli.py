"""Entry point: `animancer3d` starts the local server; `animancer3d doctor` checks setup."""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Animancer3D — local AI 3D asset generator")
    parser.add_argument(
        "command", nargs="?", choices=["doctor"], default=None,
        help="omit to start the server; 'doctor' checks dependencies and configuration",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.command == "doctor":
        _run_doctor()
        return

    import uvicorn

    from .app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)


def _run_doctor() -> None:
    from .config import get_config
    from .doctor import run_checks

    config = get_config()
    checks = run_checks(config)
    for check in checks:
        status = "OK" if check.ok else ("FATAL" if check.fatal else "WARN")
        print(f"[{status}] {check.name}: {check.detail}")
    if any(not c.ok and c.fatal for c in checks):
        raise SystemExit(1)
