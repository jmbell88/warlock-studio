"""What ``studio/tour/`` is allowed to reach for: nothing.

The fourth instance of the pin ``studio/inker/``, ``clay/``, ``plotter/`` and
``packwright/`` each carry, and the strictest of them -- a tour is *data*, so
this package has no outward imports at all, not even a shared engine.

That is what makes the rules about a tour assertable headlessly: a test can walk
every step of every tour, check every anchor and every chapter it names, and
never need a GL context. The moment this package imports imgui to ask where a
control is, or ``service`` to ask what a job is doing, those tests need an app
to run against and stop being run.

The drawing half is ``studio/panes/tour.py``, and it may import whatever a pane
imports.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import tour

ENGINE = Path(tour.__file__).parent
PACKAGE = "warlock.studio.tour"

#: Every module outside the package that any file in it may name. Empty, and
#: that is the assertion rather than an oversight.
OUTWARD_IMPORTS: set[tuple[str, str]] = set()

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}


def _modules() -> list[Path]:
    return sorted(p for p in ENGINE.glob("*.py") if not p.name.startswith("_test"))


def _imports(path: Path) -> set[str]:
    """Every module this file names, absolute and relative alike."""

    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import stays inside the package
                continue
            if node.module:
                found.add(node.module)
    return found


def test_the_sweep_finds_the_package():
    """Guard on the guard: an empty module list would pass everything below."""

    assert _modules(), "no modules found under studio/tour"


def test_nothing_here_imports_a_renderer():
    for path in _modules():
        for name in _imports(path):
            root = name.split(".")[0]
            assert root not in BANNED_ROOTS, (
                f"{path.name} imports {name}; studio/tour draws nothing and must "
                "stay importable with no window"
            )


def test_nothing_here_reaches_outside_the_package():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _imports(path)
        if name.startswith("warlock") and not name.startswith(PACKAGE)
    }
    assert found == OUTWARD_IMPORTS, (
        "studio/tour's outward imports changed. A tour is data: if a step now "
        "needs something from the app to describe itself, that belongs in the "
        "pane that draws it, where it costs no headless test."
    )


def test_the_package_imports_with_nothing_else_loaded():
    """The whole point, stated as the thing it buys.

    Importable in a bare interpreter -- no window, no GL context, no service --
    which is what lets every other test in this directory exist.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import warlock.studio.tour as t; print(len(t.TOURS))"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) >= 1
