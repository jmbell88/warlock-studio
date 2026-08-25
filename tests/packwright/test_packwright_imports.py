"""What ``studio/packwright/`` is allowed to reach for, pinned exactly.

Four outward imports, and every one of them is the same argument: reach for the
module that *owns* a definition rather than restating it. ``pipelines.sheet``
owns the atlas ceiling and what "trim" means; ``plotter.tsx`` owns the ``.tsx``
format and ``tilegrid.tileset`` the type it is written from; ``studio.undo`` owns
history. Restating any of those is how two answers to one question appear and
then drift.

``studio.inker`` is pointedly *not* one of them, and the test below says so from
the other side: ``sources`` takes a document and asks it for frames rather than
importing the editor that owns the type.

This is the ``tests/inker/test_sheetout.py`` pin, third instance.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import packwright

ENGINE = Path(packwright.__file__).parent
PACKAGE = "warlock.studio.packwright"

OUTWARD_IMPORTS = {
    # The shared bounded zip reader. One rule for four container doors, and a
    # leaf for ``tilegrid``/``undo``'s reason exactly: the ``file_size`` sum
    # each of these carried is written by whoever wrote the archive, and a
    # fourth private copy of a security bound is a copy that stops agreeing.
    ("wpack.py", "warlock.studio.zipguard"),
    # The shared history engine, as headless as this package is.
    ("document.py", "warlock.studio.undo"),
    # The authority on how big an atlas may be before an engine refuses it,
    # and on where a sprite's alpha stops. Recorded at package granularity, as
    # ``test_sheetout`` records it; the test below says which module.
    ("layout.py", "warlock.pipelines"),
    # The one .tsx writer in the repo. A second one is how a published format
    # comes to have two dialects.
    ("tsxout.py", "warlock.studio.tilegrid.tileset"),
    ("tsxout.py", "warlock.studio.plotter.tsx"),
    # The one RGBA-to-PNG encoder; the ``tsxout`` argument again. Four
    # byte-identical copies existed and all four sit on a determinism path, so
    # a compression setting added to one would make "two exports are
    # byte-identical" a claim about which writer ran.
    ("compose.py", "warlock.studio.plotter.pngio"),
    ("wpack.py", "warlock.studio.plotter.pngio"),
    # ``frozen_rgba``, over the edge ``tsxout`` had already established. A
    # sprite and a tileset image obey one immutability rule and used to hold
    # two byte-identical copies of it.
    ("sources.py", "warlock.studio.tilegrid.tileset"),
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}
LAZY_ONLY = {"PIL"}


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
    assert len(_modules()) >= 9


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
            # ``warlock._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name, and importing one of those
            # would drag torch behind a headless test as surely as importing
            # ``queue`` itself.
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_only_layout_reaches_into_pipelines_and_only_for_the_sheet_module():
    """``pipelines`` is a large package that runs inside worker and Blender
    processes. The allowlist above is at module granularity; this says *which*
    module, the way ``test_sheetout`` does."""
    reached = {
        alias.name
        for path in _modules()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("pipelines")
        for alias in node.names
    }
    assert reached == {"sheet"}


def test_only_the_source_enumerator_reaches_for_the_raster_editor():
    """And it does so through duck typing rather than an import: ``sources``
    takes *a document* and asks it for frames. Reaching for ``inker`` at import
    time would drag the whole raster editor in for a packer that is routinely
    handed loose PNG files and no document at all."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.studio.inker"), f"{path.name} imports {name}"


def test_pillow_is_never_imported_at_module_scope():
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports PIL eagerly"


def test_every_module_imports():
    from warlock.studio.packwright import (  # noqa: F401
        compose,
        document,
        layout,
        maxrects,
        sources,
        texturepacker,
        trim,
        tsxout,
        wpack,
    )
