"""What ``studio/troupe/`` is allowed to reach for, pinned exactly.

The ``tests/inker/test_sheetout.py`` rule at its fourth instance, after
``packwright`` and ``plotter``: a headless engine imports no window, no
``service`` and no queue, so it can be tested without a GPU, driven from a
worker, and read by somebody who does not have to learn the app to follow it.

Troupe's outward set is *empty* today, deliberately. It owns a frame table and
a layout; the moment it needs an atlas ceiling or a trim rectangle it reaches
for ``pipelines.sheet`` the way ``packwright.layout`` does -- and this file is
where that is written down rather than discovered.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import troupe

ENGINE = Path(troupe.__file__).parent
PACKAGE = "warlock.studio.troupe"

OUTWARD_IMPORTS: set[tuple[str, str]] = set()

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}
LAZY_ONLY = {"PIL", "numpy"}


def _outward(path: Path) -> set[str]:
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            elif node.level >= 2:
                base = PACKAGE.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _module_level(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


def _modules() -> list[Path]:
    return sorted(ENGINE.glob("*.py"))


def test_there_are_modules_to_check():
    assert len(_modules()) >= 3


def test_the_engine_never_imports_a_window():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue():
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


def test_the_engine_never_imports_the_raster_editor():
    """Troupe *produces* a document for Inker to open; the handoff lives on the
    Inker side (``sheetin``), which is what keeps the frame table readable
    without dragging the editor in behind it."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.studio.inker"), (
                f"{path.name} imports {name}"
            )


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_pillow_and_numpy_are_never_imported_at_module_scope():
    """The spec is arithmetic and the reader is ``crop``; neither should cost a
    numpy import to read a frame count."""
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports eagerly"


def test_the_shipped_layout_table_is_part_of_the_package():
    assert (ENGINE / "data" / "layout.json").is_file()


def test_every_module_imports():
    from warlock.studio.troupe import spec, ulpc  # noqa: F401
