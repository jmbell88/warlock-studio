"""What ``studio/clay/`` is allowed to reach for, pinned exactly.

Clay was the second of the four pure packages and the last to get this test,
which is the only reason it reads as a formality: the imports were already
clean, so nothing here is a fix. What it buys is the same thing the other three
pins buy -- the package's whole claim is that every rule it has about geometry
is assertable headlessly, and that claim is worth exactly as much as the imports
staying honest, so a *new* outward import is a failing test and a deliberate
decision rather than something found in a review three months later.

Written the same way as ``tests/plotter/test_plotter_imports.py`` on purpose.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import clay

ENGINE = Path(clay.__file__).parent
PACKAGE = "warlock.studio.clay"

#: ``(module, imported name)`` for every import that leaves the package.
#: :mod:`~warlock.studio.undo` is the history engine the raster editor already
#: shares -- it was extracted out of the raster editor *for* Clay, so a pure
#: Clay reaching back into the raster editor for its own history would defeat
#: the move. ``viewer.gltf`` and ``viewer.math3d`` are the ``sheetout.py``
#: argument: a Clay material **is** a ``gltf.Material`` rather than a parallel
#: type that would need a conversion function and a place for the two to drift,
#: and quaternions are XYZW in exactly one place in this project.
OUTWARD_IMPORTS = {
    ("document.py", "warlock.studio.undo"),
    ("drag.py", "warlock.studio.viewer"),
    ("document.py", "warlock.studio.viewer"),
    ("edits.py", "warlock.studio.undo"),
    ("glbimport.py", "warlock.studio.viewer"),
    ("ops.py", "warlock.studio.viewer"),
    ("serialize.py", "warlock.studio.viewer"),
}

#: Which modules of the viewer, since the entry above is recorded at package
#: granularity the way ``test_packwright_imports`` records ``pipelines``. The
#: viewer package also holds the GL-side loader and the renderer's programs, and
#: reaching for one of those is the import this pin exists to catch.
VIEWER_MODULES = {"gltf", "math3d"}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: Imported inside the functions that need it, never at module scope. Only the
#: serializer touches a pixel at all, and a top-level Pillow import would put a
#: tenth of a second onto importing ``mesh``.
LAZY_ONLY = {"PIL"}


def _outward(path: Path) -> set[str]:
    """Absolute module names this file imports from outside its own package."""
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            elif node.level >= 2:
                # Level 1 is a sibling inside the package. Level 2+ climbs out
                # of it, which is exactly what this is measuring.
                base = PACKAGE.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _module_level(path: Path) -> set[str]:
    """Only the imports at the top of the file, not the ones inside functions."""
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
    """A glob that matched nothing would make every test below vacuously
    pass."""
    assert len(_modules()) >= 12


def test_the_engine_never_imports_a_window():
    """No imgui, no moderngl, no pygame -- which is what makes every rule this
    package has about geometry assertable in a test like this one."""
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue_or_the_pipelines():
    """Neither runs anywhere a mesh op does, and dragging either in would put
    torch and a job queue behind a test of what an extrude does to a UV."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock.pipelines"), f"{path.name} imports {name}"


def test_the_engine_never_imports_the_other_pure_packages():
    """Four packages that are pure for the same reason are not four packages
    that may reach for each other: the raster editor's undo lives in
    ``studio.undo`` precisely so Clay does not have to import the raster
    editor."""
    for path in _modules():
        for name in _outward(path):
            for other in ("inker", "plotter", "packwright"):
                assert f"warlock.studio.{other}" not in name, f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_only_two_modules_of_the_viewer_are_reached_for():
    """The allowlist above is at package granularity; this says *which*
    modules, the way ``test_packwright_imports`` does for ``pipelines``."""
    reached = {
        alias.name
        for path in _modules()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("viewer")
        for alias in node.names
    }
    assert reached == VIEWER_MODULES


def test_pillow_is_never_imported_at_module_scope():
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports PIL eagerly"


def test_the_package_imports_with_no_optional_dependency_present():
    """Importing every module is the cheapest possible smoke test that the
    lazy-import rule above is actually being followed."""
    from warlock.studio.clay import (  # noqa: F401
        adjacency,
        diagnose,
        document,
        drag,
        earclip,
        edits,
        elements,
        glbimport,
        mesh,
        ops,
        ops_bevel,
        ops_dissolve,
        ops_subdiv,
        ops_topo,
        pick,
        primitives,
        serialize,
        topo,
        uv,
    )
