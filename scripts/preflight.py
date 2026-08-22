"""The release gate: run this before a commit that changes the version.

TST-03. Nothing enforced the guards this project already had. REL-01 -- a
``pyproject.toml`` at 0.0.22 against a ``CHANGELOG.md`` and a ``__version__``
at 0.0.21 -- is precisely what ``tests/test_changelog.py`` and
``tests/test_offline.py`` were written to stop, and it shipped because nobody
ran them. Two tests existed, were correct, and were not executed.

This remains the local release command.  CI runs the same cheap guards and
non-GPU suite on Windows so a forgotten local invocation cannot publish a
known-bad tree.

Four checks, in the order that fails cheapest first:

1. **Version lockstep**, read directly rather than through pytest -- it is the
   one failure that has actually happened, and it should be the first line of
   output rather than one assertion inside a six-minute run.
2. **ruff**, seconds.
3. **The non-GPU suite**, which includes the docs and manual gates.
4. A reminder about the **GPU lane**, which is opt-in and cannot run
   unattended -- reported, never silently skipped, because "the suite passed"
   must not come to mean "the parts that need a card were not run".

Exit code 0 means every check passed. Anything else names what did not.

    uv run python scripts/preflight.py
    uv run python scripts/preflight.py --fast   # skip the suite
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fail(what: str, detail: str) -> bool:
    print(f"FAIL  {what}\n      {detail}")
    return False


def _ok(what: str, detail: str = "") -> bool:
    print(f"ok    {what}{f'  ({detail})' if detail else ''}")
    return True


def check_versions() -> bool:
    """The three places a version is written, and they must agree.

    Read with regexes rather than by importing the package: this has to work
    before an install, and the failure it guards is *textual* -- one file edited
    and two forgotten.
    """
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    found = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    if found is None:
        return _fail("version lockstep", "no version in pyproject.toml")
    declared = found.group(1)

    init = (ROOT / "src" / "warlock" / "__init__.py").read_text(encoding="utf-8")
    runtime = re.search(r'^__version__ = "([^"]+)"', init, re.MULTILINE)
    if runtime is None:
        return _fail("version lockstep", "no __version__ in src/warlock/__init__.py")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(r"^## (\S+)", changelog, re.MULTILINE)
    if heading is None:
        return _fail("version lockstep", "no version heading in CHANGELOG.md")

    versions = {
        "pyproject.toml": declared,
        "src/warlock/__init__.py": runtime.group(1),
        "CHANGELOG.md": heading.group(1),
    }
    if len(set(versions.values())) != 1:
        return _fail(
            "version lockstep",
            "; ".join(f"{where} says {value}" for where, value in versions.items()),
        )
    return _ok("version lockstep", declared)


def _run(what: str, argv: list[str]) -> bool:
    print(f"...   {what}")
    result = subprocess.run(argv, cwd=ROOT)
    if result.returncode != 0:
        return _fail(what, f"{' '.join(argv)} exited {result.returncode}")
    return _ok(what)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the test suite; the version and lint checks still run",
    )
    args = parser.parse_args(argv)

    results = [check_versions(), _run("ruff", ["uv", "run", "ruff", "check", "."])]
    if args.fast:
        print("skip  test suite (--fast)")
    else:
        results.append(_run("test suite", ["uv", "run", "pytest", "-q"]))

    # Never a pass or a fail, always a line. The gpu marker is excluded by
    # default (TST-02) and these tests need a provisioned card, so the honest
    # thing is to say they did not run rather than to let a green preflight
    # imply they did.
    print(
        "note  the gpu lane is not part of this gate: run "
        "`uv run pytest -m gpu -n 0` on a provisioned machine before changing "
        "model loading, VRAM accounting or conditioning"
    )

    if all(results):
        print("\npreflight passed")
        return 0
    print(f"\npreflight FAILED ({sum(1 for r in results if not r)} check(s))")
    return 1


if __name__ == "__main__":
    sys.exit(main())
