"""What ``studio/sirens/`` is allowed to reach for, pinned exactly.

Three outward imports, and the interesting part of this file is the *absences*.

``studio.undo`` owns history and ``zipguard``/``npyguard`` own the two container
bounds. Nothing else: this package decides what a song sounds like, and every
other question belongs to somebody who already answers it.

**``pygame`` is the absence that matters.** Sirens is the only mode in the app
whose output is audio, so it is the only one with a reason to want a sound
device -- and the whole point of keeping it out is that a machine with no audio
hardware can still open, edit, render and export a song. That machine is CI, and
it is why this suite can test the synthesiser at all. ``studio/sirens_audio.py``
is the one module in the repo that touches ``pygame.mixer``, and it is not here.

**``scipy`` is the other one.** ``scipy.signal`` has a decimator and this package
writes its own (``voices._lowpass``). That is not reinvention: the bar for this
engine is that one document renders byte-identical samples, and a filter whose
coefficients come from a dependency is a filter that can change under a
``uv sync``. Six lines of sinc are the price of the guarantee.

This is the ``tests/packwright/test_packwright_imports.py`` pin, fifth instance.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import sirens

ENGINE = Path(sirens.__file__).parent
PACKAGE = "warlock.studio.sirens"

OUTWARD_IMPORTS = {
    # The shared history engine, as headless as this package is.
    ("document.py", "warlock.studio.undo"),
    ("edits.py", "warlock.studio.undo"),
    # The two container doors. A ``.wsng`` is a zip of ``.npy`` members, which
    # is the exact pair ``.wmap`` and ``.wblk`` already go through -- the
    # archive's directory cannot see a lie one format down inside a member.
    ("wsng.py", "warlock.studio.zipguard"),
    ("wsng.py", "warlock.studio.npyguard"),
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: Not banned because it is heavy -- it is a core dependency and Clay uses it --
#: but because this package's output has to be reproducible. See the docstring.
DETERMINISM_ROOTS = {"scipy"}

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
    assert len(_modules()) >= 8


def test_the_engine_never_imports_a_window():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_audio_device_is_not_in_the_audio_engine():
    """Stated separately from the window pin above, because it is a different
    argument. The other four engines exclude pygame to stay off the frame
    thread; this one excludes it so that a machine with no sound card can still
    render a song to a file. ``sirens_audio`` is where the device lives."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("pygame"), f"{path.name} imports {name}"


def test_the_filter_is_not_borrowed_from_scipy():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & DETERMINISM_ROOTS), f"{path.name} imports {roots & DETERMINISM_ROOTS}"


def test_the_engine_never_imports_the_service_layer():
    for path in _modules():
        for name in _outward(path):
            assert "warlock.service" not in name, f"{path.name} imports {name}"


def test_the_engine_never_imports_the_queue():
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("warlock.queue"), f"{path.name} imports {name}"
            assert not name.startswith("warlock._q"), f"{path.name} imports {name}"


def test_the_engine_never_imports_another_editor():
    """A song has nothing to say about a bitmap, a mesh or a tile map, and none
    of those four packages has anything to say about a song."""
    for path in _modules():
        for name in _outward(path):
            for other in ("inker", "clay", "plotter", "packwright", "tilegrid"):
                assert f"warlock.studio.{other}" not in name, f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS


def test_pillow_is_never_imported_at_module_scope():
    for path in _modules():
        assert not (_module_level(path) & LAZY_ONLY), f"{path.name} imports PIL eagerly"


def test_every_module_imports():
    from warlock.studio.sirens import (  # noqa: F401
        document,
        edits,
        instruments,
        notes,
        synth,
        voices,
        wavout,
        wsng,
    )
