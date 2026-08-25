"""The undo engine is a studio-level module, and inker re-exports it.

Clay needs ``Edit``/``CompoundEdit``/``UndoStack`` with its own edit
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
    "UNDO_HARD_BYTES",
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
    # would put Clay behind a pixel buffer.
    source = engine.__file__
    assert source is not None
    # And assert the property, not today's exact import list: a new stdlib
    # import in the engine is legitimate and must not fail this test, whereas
    # *any* relative import is a dependency on the package the engine was
    # extracted out of.
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("warlock.studio.inker"), alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"relative import: {'.' * node.level}{node.module}"
            assert not (node.module or "").startswith("warlock.studio.inker"), node.module


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
