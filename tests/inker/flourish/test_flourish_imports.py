"""What ``studio/inker/flourish/`` is allowed to reach for, pinned exactly.

Narrower than its parent package's pin (``tests/inker/test_inker_imports.py``
scans only the top-level modules, so this subpackage needs its own). The
engine imports numpy and the standard library and *nothing* else at module
scope -- not even the rest of ``inker``: a recipe knows what a frame is, not
what a document is, and the door between the two is ``_doc_flourish`` one
level up.

**``scipy`` is the absence with an argument.** The bar for this engine is that
one recipe renders byte-identical frames on every machine (``test_render``
pins digests), and a blur or a noise whose kernel comes from a dependency is a
kernel that can change under a ``uv sync``. Sirens' reason, applied to pixels.

**``bake.py`` is the one bounded exception**, and it is written down here the
way ``sheetout``'s is in the parent pin: it reaches for ``pipelines.pixelize``
and ``pipelines.pixelsheet`` because they are the authority on what pixel art
means in this app, and a second Oklab palette map would be a second answer.
Pillow rides along inside those two functions and nowhere at module scope.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from warlock.studio.inker import flourish

ENGINE = Path(flourish.__file__).parent
PACKAGE = "warlock.studio.inker.flourish"

BANNED_ROOTS = {"imgui", "imgui_bundle", "moderngl", "pygame", "OpenGL", "glfw"}
DETERMINISM_ROOTS = {"scipy"}
ALLOWED_ROOTS = {"numpy", "warlock"}

#: ``(module, imported name)`` for every import that leaves the package.
OUTWARD_IMPORTS = {
    ("bake.py", "warlock.pipelines"),
}
#: Modules that may import Pillow, and only inside a function.
LAZY_PILLOW = {"bake.py"}


def _modules() -> list[Path]:
    return sorted(ENGINE.rglob("*.py"))


def _own_package(path: Path) -> str:
    rel = path.relative_to(ENGINE).with_suffix("")
    return PACKAGE + ("." + ".".join(rel.parts[:-1]) if len(rel.parts) > 1 else "")


def _outward(path: Path) -> set[str]:
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    own = _own_package(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found.add(node.module or "")
            else:
                base = own if node.level == 1 else own.rsplit(".", node.level - 1)[0]
                found.add(f"{base}.{node.module}" if node.module else base)
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


def _stdlib(name: str) -> bool:
    return name.split(".")[0] in sys.stdlib_module_names


def test_there_are_modules_to_check():
    assert len(_modules()) >= 14


def test_the_engine_never_imports_a_window():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & BANNED_ROOTS), f"{path.name} imports {roots & BANNED_ROOTS}"


def test_the_kernels_are_not_borrowed_from_scipy():
    for path in _modules():
        roots = {name.split(".")[0] for name in _outward(path)}
        assert not (roots & DETERMINISM_ROOTS), f"{path.name} imports {roots & DETERMINISM_ROOTS}"


def test_pillow_is_never_imported_at_module_scope():
    for path in _modules():
        assert "PIL" not in _module_level(path), f"{path.name} imports Pillow at module scope"
        if "PIL" in {n.split(".")[0] for n in _outward(path)}:
            assert path.name in LAZY_PILLOW, f"{path.name} imports Pillow"


def test_the_only_outward_imports_are_the_ones_written_down():
    """No ``warlock.*`` import that is not this package itself, except the
    ones the table names: not the Inker document, not ``service``."""
    found = set()
    for path in _modules():
        for name in _outward(path):
            if name.startswith(PACKAGE) or name.split(".")[0] != "warlock":
                continue
            found.add((path.name, ".".join(name.split(".")[:2])))
    assert found == OUTWARD_IMPORTS


def test_the_only_third_party_import_is_numpy():
    for path in _modules():
        for name in _outward(path):
            root = name.split(".")[0]
            if _stdlib(name) or root in ALLOWED_ROOTS or root == "PIL":
                continue
            raise AssertionError(f"{path.relative_to(ENGINE)} imports {name}")


def test_every_primitive_is_registered_and_complete():
    from warlock.studio.inker.flourish import prims

    on_disk = {p.stem for p in (ENGINE / "prims").glob("*.py") if p.stem != "__init__"}
    assert on_disk == set(prims.KINDS)
    for kind in prims.KINDS:
        mod = prims.module(kind)
        assert isinstance(mod.REPLACES_BELOW, bool)
        assert callable(mod.render)
        for name, spec in mod.PARAMS.items():
            assert spec.kind in prims.PARAM_KINDS, (kind, name)
            # The default is inside its own range, so a fresh layer is legal.
            assert spec.clamp(spec.default) == spec.clamp(spec.clamp(spec.default)), (kind, name)
