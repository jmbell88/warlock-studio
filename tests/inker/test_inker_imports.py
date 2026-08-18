"""What ``studio/inker/`` is allowed to reach for, pinned exactly.

The package's whole claim is that every rule it has -- about pixels, layers,
selections, undo -- is assertable headlessly. That claim is only worth anything
while the imports stay honest, so a *new* outward import is a failing test and a
deliberate decision rather than something that turns up in a review three months
later.

**This is the oldest of the four package pins and it was the weakest**, which is
the reason it now has a file of its own. It lived at the bottom of
``test_sheetout.py`` -- a file about grid arithmetic -- and carried three of the
seven checks its siblings in ``tests/plotter``, ``tests/packwright`` and
``tests/clay`` carry. Missing were the vacuous-pass guard, the lazy-Pillow rule,
the never-reach-a-sibling-engine rule and the import smoke test. A pin that
would pass against an empty directory is not a pin, and it is the one failure
mode none of the assertions below can report on their own.

``test_sheetout.py`` keeps the one check that is genuinely about *it*: which
module inside ``pipelines`` its export reaches for.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import inker

# Located through the package rather than by climbing ``__file__``, which is how
# the other three do it: a moved test file must not silently start pinning a
# directory that does not exist.
ENGINE = Path(inker.__file__).parent
PACKAGE = "warlock.studio.inker"

#: ``(module, imported name)`` for every import that leaves the package.
#:
#: ``sheetout`` is the newest and the one with an argument attached: it reaches
#: for the *authority* on the sprite-sheet format, so ``version: 1`` cannot come
#: to mean two subtly different documents. The others predate it -- the shared
#: undo engine and the native kernel loader, both as headless as this package is.
#:
#: ``tiles.py``'s two entries are the tile model reaching for the *other*
#: shared leaf, ``warlock.studio.tilegrid`` -- the gid word and the sliced-atlas
#: type. Same reasoning as ``undo.py``'s entry above: a shared leaf is not a
#: sibling engine (``SIBLING_PACKAGES`` says so explicitly), so this is a
#: deliberate, permanent addition rather than a drift to chase down later.
OUTWARD_IMPORTS = {
    ("anim_edits.py", "warlock.studio.undo"),
    ("composite.py", "warlock.native"),
    ("dither.py", "warlock.native"),
    ("selection.py", "warlock.native"),
    ("sheetout.py", "warlock.pipelines"),
    ("tiles.py", "warlock.studio.tilegrid"),
    ("tiles.py", "warlock.studio.tilegrid.tileset"),
    ("undo.py", "warlock.studio.undo"),
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: Imported inside the functions that need it, never at module scope. Pillow
#: costs a tenth of a second to import and most of this package never decodes a
#: pixel -- the rule ``tests/plotter`` and ``tests/packwright`` already state.
LAZY_ONLY = {"PIL"}

#: The other pure packages. Each is a peer, not a dependency: an edge between
#: two of them is how "four independent engines" quietly becomes one.
#: ``packwright -> plotter`` is the single declared exception in the tree --
#: shrunk by the ``tilegrid`` promotion to just ``tsx`` and ``pngio``, since the
#: gid word, the sliced atlas and the blob collapse moved to the shared leaf
#: both packages now reach for instead -- and it runs the other way, so nothing
#: here may point at any of them. ``warlock.studio.tilegrid`` is not one of
#: these: it is a shared leaf, not a sibling engine, and this package is free to
#: import it.
SIBLING_PACKAGES = (
    "warlock.studio.clay",
    "warlock.studio.plotter",
    "warlock.studio.packwright",
)


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
                    # ``from ... import native``: with no module part the names
                    # are themselves modules, so they are the thing being
                    # reached for and the package alone would say nothing.
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _module_level(path: Path) -> set[str]:
    """Top-level import roots only -- what importing this module actually costs."""
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
    """The guard every assertion below needs and none of them can make.

    Each of them is a ``for`` over ``_modules()``, so all four pass against an
    empty list -- a renamed package, a moved test file, a glob that stops
    matching, and the pin reports success while checking nothing at all.
    """
    assert len(_modules()) >= 10


def test_the_engine_never_imports_a_window():
    """``studio/inker/`` stays headless, which is what makes every rule it has
    about pixels assertable in a test like this one. The rule has been true
    since the package was written and was held by a docstring, which is not a
    test."""
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue():
    """The one pin this file did not have, and its three siblings did. Both
    spellings: ``warlock.queue`` is the front door and ``warlock._q_*`` are the
    worker halves behind it, and importing either would put a job queue and
    torch under a headless test of what a brush does to a pixel.

    ``pipelines`` is deliberately not named here, unlike in the Clay and
    Plotter pins -- this suite's allowlist already decides what the raster
    editor may reach for there, and a second rule would be a second answer.
    """
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


def test_the_engine_never_imports_the_other_pure_packages():
    """Four engines, not one with four front doors."""
    for path in _modules():
        for name in _outward(path):
            for sibling in SIBLING_PACKAGES:
                assert not name.startswith(sibling), f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_pillow_is_never_imported_at_module_scope():
    """``ora`` and ``sheetout`` encode images and the rest never touch one, so a
    top-level Pillow import would put that cost on importing ``composite``."""
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports PIL eagerly"


def test_the_package_imports_with_no_optional_dependency_present():
    """Importing every module is the cheapest possible smoke test that the
    lazy-import rule above is actually being followed."""
    from warlock.studio.inker import (  # noqa: F401
        _doc_ranges,
        _doc_slices,
        anim_edits,
        animation,
        asein,
        brush,
        composite,
        dither,
        document,
        filters,
        gifout,
        gpl,
        gradient,
        groups,
        index_plane,
        indexed,
        layers,
        ora,
        selection,
        sheetin,
        sheetout,
        slices,
        textstamp,
        tiles,
        tiling,
        transform,
        undo,
    )
