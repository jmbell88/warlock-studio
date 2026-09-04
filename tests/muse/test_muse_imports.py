"""What ``studio/muse/`` is allowed to reach for, pinned exactly.

``tests/sirens/test_sirens_imports.py``'s pin, sixth instance -- and the
interesting difference is that the outward set here is **empty**. Sirens reaches
for ``studio.undo`` and the two container guards because it owns a document;
Muse owns none, so a take is a job row and this package computes a picture and
a pair of sample offsets and answers nothing else.

**``scipy`` is banned for ``wavout``'s reason, one module further out.** An
exported loop is an artifact: the samples at its seam do not exist in the take,
this code writes them, and ``wavout``'s byte-identity rule reaches every one. A
filter whose coefficients come from a dependency is a filter that changes under
a ``uv sync``, which would make two exports of the same take and the same
markers differ.

Landed *before* ``loops.py``, deliberately: the algorithm is written under the
pin rather than retrofitted into it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from warlock.studio import muse

ENGINE = Path(muse.__file__).parent
PACKAGE = "warlock.studio.muse"

#: Empty, and that is the claim. See the module docstring.
OUTWARD_IMPORTS: set[tuple[str, str]] = set()

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}

#: Not banned because it is heavy -- it is a core dependency and Clay uses it --
#: but because what this package writes ends up in a file. See the docstring.
DETERMINISM_ROOTS = {"scipy"}


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


def _modules() -> list[Path]:
    return sorted(ENGINE.glob("*.py"))


def test_there_are_modules_to_check():
    assert len(_modules()) >= 1


def test_the_engine_never_imports_a_window():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_audio_device_is_not_in_the_audio_arithmetic():
    """Muse is one of the two modes whose output is sound, so it is one of the
    two with a reason to want a device -- and that is exactly why this pin
    exists. ``sirens_audio`` owns the one reserved channel; nothing here may
    reach past it, or the app would have a second door onto one sound card."""
    for path in _modules():
        for name in _outward(path):
            assert not name.startswith("pygame"), f"{path.name} imports {name}"


def test_the_crossfade_is_not_borrowed_from_scipy():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & DETERMINISM_ROOTS), (
            f"{path.name} imports {roots & DETERMINISM_ROOTS}"
        )


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
    """A take has nothing to say about a bitmap, a mesh, a tile map or a song.

    ``sirens`` is on this list beside the other four, which is the one that
    would be tempting: ``wavout`` lives there and Muse's *exporter* uses it. But
    that exporter is ``studio/muse_io.py``, outside this package -- and the
    reason is that ``wavout`` is a RIFF encoder misfiled under ``sirens/``
    because Sirens was its first caller, not a fact about what a take is.
    """
    for path in _modules():
        for name in _outward(path):
            for other in ("inker", "clay", "plotter", "packwright", "sirens", "tilegrid"):
                assert f"warlock.studio.{other}" not in name, f"{path.name} imports {name}"


def test_the_only_outward_imports_are_the_ones_written_down():
    found = {
        (path.name, name)
        for path in _modules()
        for name in _outward(path)
        if name.split(".")[0] == "warlock"
    }
    assert found == OUTWARD_IMPORTS
