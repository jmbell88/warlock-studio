"""``studio/tilegrid/`` imports nothing under ``warlock``, by construction.

The second shared leaf after ``studio/undo.py``: plotter, packwright and inker
all import it, none owns it. A leaf that reached back into any of them would
turn "shared vocabulary" into a dependency cycle waiting for an import order to
expose it, so this is a property pin in the ``tests/clay/test_undo_move.py``
style -- it asserts *no relative import climbs out of the package and no
absolute import starts with* ``warlock``, rather than pinning today's exact
import list the way the other three packages' pins do. A new stdlib or
third-party import here is legitimate and must not fail this test.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import tilegrid

ENGINE = Path(tilegrid.__file__).parent
PACKAGE = "warlock.studio.tilegrid"


def _modules() -> list[Path]:
    return sorted(ENGINE.glob("*.py"))


def test_there_are_modules_to_check() -> None:
    assert len(_modules()) >= 4  # __init__, blob, gid, tileset


def test_the_leaf_imports_nothing_under_warlock() -> None:
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("warlock"), (
                        f"{path.name} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                # Level 1 (``from . import blob``) is a sibling inside this
                # very package -- ``tileset.py`` importing ``blob`` -- and
                # level 0 with an absolute module is the only other shape
                # these files use. Level 2+ would climb out of the package,
                # which is exactly what a shared leaf must never do.
                assert node.level <= 1, (
                    f"{path.name} has a relative import that climbs out: "
                    f"{'.' * node.level}{node.module or ''}"
                )
                if node.level == 0:
                    assert not (node.module or "").startswith("warlock"), (
                        f"{path.name} imports {node.module}"
                    )


def test_the_leaf_lives_at_its_new_home() -> None:
    assert tilegrid.__name__ == "warlock.studio.tilegrid"


def test_the_public_names_are_present() -> None:
    for name in (
        "RGBA",
        "TerrainSpec",
        "Tileset",
        "TilesetRef",
        "blob",
        "colour_text",
        "frozen_rgba",
        "gid",
        "repolish",
        "rgba_colour",
        "slicing",
    ):
        assert hasattr(tilegrid, name), name
