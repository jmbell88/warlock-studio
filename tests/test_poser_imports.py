"""What Poser's four pure modules are allowed to reach for, pinned exactly.

The ``tests/inker/test_sheetout.py`` pin, fifth instance -- with one structural
departure the others do not need. Clay, Inker, Plotter and Packwright each own a
*package*, so their pins glob a directory. Poser owns no package: its pure half
is two modules at the root of ``warlock`` (``poselib``, ``rigging``) and two
inside the viewer (``pose``, ``bonelines``), and the rest of it is panes. So the
four are named, and a tripwire below fails if one of them ever moves.

They are pinned for the same reason the packages are, plus one of their own:
``rigging`` is the host half of a Blender subprocess and ``poselib`` is what a
service module reads a stored pose through. Both claim in their own docstrings
to be usable with no studio at all, and the two subprocess checks at the bottom
make that claim executable instead of merely stated.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import warlock

ROOT = Path(warlock.__file__).parent

#: ``relative path -> the package a relative import inside it resolves against``.
MODULES = {
    "poselib.py": "warlock",
    "rigging.py": "warlock",
    "studio/viewer/pose.py": "warlock.studio.viewer",
    "studio/viewer/bonelines.py": "warlock.studio.viewer",
}

#: Every import that names something under ``warlock``, per module. Viewer
#: siblings are listed too -- unlike the package pins there is no "inside the
#: package" to be exempt, and the point of naming ``gltf`` and ``math3d`` here
#: is that the list is the whole dependency, not the part that left a directory.
OUTWARD_IMPORTS = {
    # The storage half of the library reaches for ids, validation and the
    # skeleton templates. Nothing else: a stored pose is decidable with no
    # service, no studio and no Blender, which is what test_poselib.py stands on.
    "poselib.py": {"warlock.rigging"},
    # The kill-on-close job object, because ``run_worker`` spawns Blender. The
    # stdlib ``queue`` and ``subprocess`` imports are not ``warlock.queue`` --
    # the AST records the bare names, which is why this set stays empty of them.
    "rigging.py": {"warlock.winjob"},
    # The editor: rotations and mirroring from the storage half, matrices and
    # the node graph from the viewer's own.
    "studio/viewer/pose.py": {
        "warlock.poselib",
        "warlock.rigging",
        "warlock.studio.viewer.gltf",
        "warlock.studio.viewer.math3d",
    },
    # The GPU half, so it reaches only for viewer siblings.
    "studio/viewer/bonelines.py": {
        "warlock.studio.viewer.markers",
        "warlock.studio.viewer.math3d",
        "warlock.studio.viewer.render",
    },
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "pygame", "OpenGL", "glfw"}

#: moderngl is banned everywhere except ``bonelines``, which *is* the GPU half:
#: it builds the line buffers, and a draw list is not expressible without the
#: context type. The other three are asserted headlessly and may never gain it.
MODERNGL_ALLOWED = {"studio/viewer/bonelines.py"}

# No LAZY_ONLY section here, deliberately: the package pins have one because
# their modules encode PNGs, and none of these four touches Pillow at all. A
# lazy-import test over four modules that never import it would pass forever
# without measuring anything.


def _outward(rel: str) -> set[str]:
    """Absolute module names this file imports, relative levels resolved."""
    package = MODULES[rel]
    found: set[str] = set()
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            else:
                base = package.rsplit(".", node.level - 1)[0]
                if node.module:
                    found.add(f"{base}.{node.module}")
                else:
                    found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _run(stubs: tuple[str, ...], imports: str) -> subprocess.CompletedProcess:
    """Import something in a fresh interpreter with modules stubbed to None.

    ``sys.modules[name] = None`` makes any attempt to import it raise, so a
    hidden import fails loudly rather than succeeding on a developer machine
    that happens to have the whole studio installed. A subprocess rather than a
    reload, the ``test_rigging_stays_importable_with_no_bpy_anywhere`` rule:
    reloading mints new function objects and breaks the identity other tests
    assert about ``mirror_quaternion``.
    """
    script = (
        "import sys\n"
        f"for name in {stubs!r}: sys.modules[name] = None\n"
        f"import {imports}\n"
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


def test_the_four_modules_are_all_still_there():
    """A named-module pin goes vacuous by a rename rather than by a bad glob."""
    for rel in MODULES:
        assert (ROOT / rel).is_file(), f"{rel} moved; the pin below now measures nothing"


def test_none_of_them_imports_a_window():
    for rel in MODULES:
        roots = {name.split(".")[0] for name in _outward(rel)}
        assert not (roots & BANNED_ROOTS), f"{rel} imports {roots & BANNED_ROOTS}"


def test_only_the_gpu_half_imports_moderngl():
    for rel in MODULES:
        roots = {name.split(".")[0] for name in _outward(rel)}
        has = "moderngl" in roots
        assert has == (rel in MODERNGL_ALLOWED), f"{rel}: moderngl={has}"


def test_none_of_them_imports_the_service_layer():
    """The dependency runs the other way: ``service/poses.py`` reads a record
    through ``poselib``. An import back would be a cycle and, worse, would put
    a store-wide lock behind a function that is documented as pure."""
    for rel in MODULES:
        for name in _outward(rel):
            assert "warlock.service" not in name, f"{rel} imports {name}"


def test_none_of_them_imports_the_queue_or_the_pipelines():
    for rel in MODULES:
        for name in _outward(rel):
            assert not name.startswith("warlock.queue"), f"{rel} imports {name}"
            assert not name.startswith("warlock.pipelines"), f"{rel} imports {name}"
            # ``warlock._q_*`` too: the queue's worker halves are the same
            # dependency wearing a different name, and importing one of those
            # would drag torch behind a headless test as surely as importing
            # ``queue`` itself.
            assert not name.startswith("warlock._q"), f"{rel} imports {name}"


def test_the_only_warlock_imports_are_the_ones_written_down():
    for rel in MODULES:
        found = {name for name in _outward(rel) if name.split(".")[0] == "warlock"}
        assert found == OUTWARD_IMPORTS[rel], rel


def test_the_storage_half_imports_with_no_studio_at_all():
    """``poselib``'s docstring says a stored pose is decidable without the app;
    ``rigging``'s host half is imported by a service module that never draws.
    Both claims, executed."""
    proc = _run(
        ("imgui", "imgui_bundle", "moderngl", "pygame", "warlock.studio"),
        "warlock.poselib, warlock.rigging",
    )
    assert proc.returncode == 0, proc.stderr


def test_the_viewer_half_imports_with_no_imgui_and_no_service():
    """moderngl is left real -- ``bonelines`` genuinely needs it. What is being
    measured is that the renderer still knows nothing about panels or the
    business layer, which is the split ``viewer/__init__`` claims."""
    proc = _run(
        ("imgui", "imgui_bundle", "pygame", "warlock.service"),
        "warlock.studio.viewer.pose, warlock.studio.viewer.bonelines",
    )
    assert proc.returncode == 0, proc.stderr
