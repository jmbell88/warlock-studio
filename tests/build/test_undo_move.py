"""The undo engine is a studio-level module, and inker re-exports it.

Build mode needs ``Edit``/``CompoundEdit``/``UndoStack`` with its own edit
types, so the engine cannot live inside the raster editor. What these tests pin
is that the extraction was a *move*: the names inker imports are the very
objects ``studio.undo`` defines, not copies that could drift apart -- two
serial counters in particular would hand out the same numbers twice, and
``head`` compares serials to decide whether a document is unsaved.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import undo as engine
from warlock.studio.inker import undo as inker_undo

ENGINE_NAMES = (
    "Edit",
    "CompoundEdit",
    "UndoStack",
    "_serials",
    "UNDO_BYTES",
    "UNDO_MAX_DEPTH",
    "UNDO_MIN_DEPTH",
)


def test_the_engine_lives_outside_the_raster_editor() -> None:
    for name in ENGINE_NAMES:
        assert hasattr(engine, name), name
    assert engine.__name__ == "warlock.studio.undo"


def test_the_engine_imports_nothing_from_the_raster_editor() -> None:
    # Read the imports rather than the prose: the module head talks about the
    # raster editor at length, and it is the dependency, not the mention, that
    # would put Build mode behind a pixel buffer.
    source = engine.__file__
    assert source is not None
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    assert imported == {"__future__", "itertools", "dataclasses", "typing"}


def test_inker_re_exports_the_same_objects_not_copies() -> None:
    for name in ENGINE_NAMES:
        assert getattr(inker_undo, name) is getattr(engine, name), name


def test_the_raster_edit_types_stayed_behind() -> None:
    for name in (
        "PatchEdit",
        "LayerAddEdit",
        "LayerRemoveEdit",
        "LayerMoveEdit",
        "LayerPropsEdit",
        "SelectionEdit",
        "ReplayEdit",
        "_pack",
        "_unpack",
    ):
        assert hasattr(inker_undo, name), name
        assert not hasattr(engine, name), name
