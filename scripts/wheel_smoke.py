"""Install the built wheel into a clean venv and prove it imports.

Extracted from ``.github/workflows/windows-ci.yml`` on 2026-09-05, when
``scripts/rebuild.ps1`` began running the same gate locally. The assertions
used to be a here-doc inside the YAML, and a *copy* of them in the local
driver would have been a local pass that stopped meaning anything the month
CI's copy changed -- the workflow's own comments record one assertion (a
literal mode count) that was wrong and sat there unnoticed because the step
above it failed first. So there is one implementation and both callers run it.

The wheel "built" by ``uv build`` was never proven to import, let alone run --
so a packaging mistake (a missing force-include, a module outside
``packages``) shipped green. Installed into a *clean* venv rather than the
development one, which is the only arrangement where the source tree cannot
stand in for a file the wheel forgot.

    uv run python scripts/wheel_smoke.py            # newest dist/*.whl
    uv run python scripts/wheel_smoke.py <wheel>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Run inside the clean venv, from a directory that is not the checkout, so
# `import warlock` cannot resolve to `src/` by accident.
SMOKE = """
import warlock, warlock.models, warlock.doctor, warlock.changelog
from warlock.studio import modes

assert warlock.models.BASE_MODELS, 'the registry is empty'
# ``modes`` is imported to prove it *loads* from a wheel, not to count it. A
# literal here is a number no test can keep in sync: it said 11 against a
# twelve-mode tree, so this step could only ever fail once anything reached
# it. The count is asserted where it can be derived from ``modes.MODES``:
# ``tests/manual/test_docs.py`` and ``tests/test_studio_state.py``.
assert modes.KEYS, 'the rail is empty'
# The two force-includes, which are exactly what a wheel drops silently: the
# manual tree and the changelog.
from importlib.resources import files
assert files('warlock').joinpath('CHANGELOG.md').is_file()
assert files('warlock').joinpath('manual/00-index.md').is_file()
print('wheel import smoke test passed')
"""


def _run(argv: list[str], label: str = "", **kwargs: object) -> None:
    # ``label`` because the assertion body is 15 lines and echoing it once as an
    # argument and again as output is noise around the one line that matters.
    print("+", label or " ".join(str(a) for a in argv), flush=True)
    subprocess.run(argv, check=True, **kwargs)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wheel",
        nargs="?",
        type=Path,
        help="the wheel to test; defaults to the newest dist/*.whl",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        default=ROOT / "build" / "smoke-venv",
        help="where to build the clean environment (default: build/smoke-venv)",
    )
    args = parser.parse_args(argv)

    wheel = args.wheel
    if wheel is None:
        wheels = sorted((ROOT / "dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
        if not wheels:
            print("no wheel in dist/; run `uv build` first", file=sys.stderr)
            return 1
        wheel = wheels[-1]
    if not wheel.is_file():
        print(f"no such wheel: {wheel}", file=sys.stderr)
        return 1

    venv = args.venv
    # A venv carrying a previous release is not a clean venv: `uv pip install`
    # would leave the older warlock's files behind wherever the new wheel does
    # not overwrite them, which is precisely the missing-file case this tests.
    if venv.exists():
        shutil.rmtree(venv)
    _run(["uv", "venv", str(venv)])
    python = venv / "Scripts" / "python.exe"
    if not python.is_file():  # non-Windows layout, for the odd local run
        python = venv / "bin" / "python"
    _run(["uv", "pip", "install", "--python", str(python), str(wheel)])

    with tempfile.TemporaryDirectory() as elsewhere:
        _run(
            [str(python), "-c", SMOKE],
            label=f"{python} -c <smoke assertions>  (cwd {elsewhere})",
            cwd=elsewhere,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
