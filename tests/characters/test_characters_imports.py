"""What ``warlock.characters`` is allowed to reach for, pinned exactly.

The ``tests/test_poser_imports.py`` pin, sixth instance, with the structural
difference that package's docstring anticipates: this one owns a *package with
subpackages*, so the module list is a glob of the package root plus a glob of
each archetype subpackage -- and the set of archetype subpackages is itself
written down, so adding the quadruped is a deliberate edit to this file rather
than a silent widening of the pin.

Why it exists at all. ``characters`` is imported by the door to answer "what can
we make", by the recipe path to validate a request, and eventually by a worker
process to build one. None of those three may drag imgui, moderngl, pygame,
bpy, torch, ``service`` or the queue in behind it -- and the boolean kernel
(trimesh, manifold3d) may not come in behind a *registry lookup* either, which
is why Clay's imports inside the generator are pinned to function scope rather
than merely allowed.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import warlock

ROOT = Path(warlock.__file__).parent
PACKAGE = ROOT / "characters"

#: The archetype subpackages -- the four body plans. Named rather than globbed
#: on purpose: a new subpackage that quietly inherited this pin would inherit an
#: allowlist written for a different generator's dependencies, so adding a body
#: plan is a deliberate edit here and ``test_the_archetype_packages_are_the_ones
#: _written_down`` is what makes it one.
ARCHETYPE_PACKAGES = {"humanoid", "quadruped", "winged", "amorphous"}

#: Every archetype generator reaches for exactly the same set, and that is the
#: point rather than a coincidence: each is the *same pipeline* over different
#: parameters -- primitives, boolean union, weld, one Catmull-Clark level, weld,
#: manifoldness asserted -- so a generator that grew an import its siblings did
#: not would be a body plan that had stopped being built the shared way.
_GENERATOR_IMPORTS = {
    "warlock.rigging",
    "warlock.studio.clay.adjacency",
    "warlock.studio.clay.document",
    "warlock.studio.clay.elements",
    "warlock.studio.clay.mesh",
    "warlock.studio.clay.ops_boolean",
    "warlock.studio.clay.ops_subdiv",
    "warlock.studio.clay.ops_topo",
    "warlock.studio.clay.primitives",
    "warlock.studio.clay.topo",
    "warlock.studio.viewer.glbwrite",
    "warlock.studio.viewer.gltf",
}

#: ``relative path -> every ``warlock.*`` name it imports``, exactly.
OUTWARD_IMPORTS: dict[str, set[str]] = {
    # Nothing at all: a refusal type that imported anything would be a refusal
    # type that could not be raised from the module that needed it most.
    "errors.py": set(),
    # The registries are data. They read no file and know no format.
    "family.py": set(),
    # The frame table, and only the frame table: what a recipe expands into is
    # ``charsheet``'s arithmetic, and a second copy here would be a second
    # opinion about what cell 137 depicts.
    "recipe.py": {"warlock.pipelines.charsheet"},
    # **Nothing outward at all**, and that is the resolver's whole claim: it
    # turns a sentence into controls with a fixed vocabulary, so it must be
    # decidable with no pipeline, no service and no model behind it. The camera
    # keys it emits are literals here rather than read from
    # ``charsheet.CAMERA_PRESETS``, and the cost of that second spelling is
    # paid by a **two-way** pin in ``tests/characters/test_resolve.py``: every
    # key the vocabulary emits is a real preset, *and* every preset is askable
    # in words. One direction alone would let a preset the table gained become
    # a framing nobody can ask for.
    "resolve.py": set(),
    # The glTF pair to write with, and ``rigging`` to check the baked skeleton
    # against the template it claims to fit -- through the same
    # ``validate_joints`` a hand-corrected rig comes in by.
    "instantiate.py": {
        "warlock.rigging",
        "warlock.studio.viewer.glbwrite",
        "warlock.studio.viewer.gltf",
    },
    # **The one outward edge that leaves the app's own back end.** A theme
    # declares ``effects=("embers",)`` and Flourish is the thing in this repo
    # that can draw a flame; the alternative was a second flame renderer inside
    # ``characters``, which is how two flames come to look different. Both
    # imports are **function scope** --
    # ``test_flourish_is_only_ever_imported_inside_a_function`` is the pin --
    # because ``characters`` is what the door imports to answer "what can we
    # make" and Flourish drags numpy and ten primitive modules in behind it.
    "effects.py": {
        "warlock.studio.inker.flourish.recipe",
        "warlock.studio.inker.flourish.render",
    },
    "__init__.py": set(),
    "humanoid/__init__.py": set(),
    # ``rigging`` for the template it grows the body around, the glTF pair to
    # bake with, and Clay -- the last of which is function-scope only, which
    # ``test_clay_is_only_ever_imported_inside_a_function`` is what enforces.
    "humanoid/generate.py": set(_GENERATOR_IMPORTS),
    "quadruped/__init__.py": set(),
    "quadruped/generate.py": set(_GENERATOR_IMPORTS),
    "winged/__init__.py": set(),
    "winged/generate.py": set(_GENERATOR_IMPORTS),
    "amorphous/__init__.py": set(),
    "amorphous/generate.py": set(_GENERATOR_IMPORTS),
}

BANNED_ROOTS = {"imgui", "imgui_bundle", "pygame", "moderngl", "OpenGL", "glfw", "bpy", "torch"}

LAZY_ONLY = {"warlock.studio.clay", "warlock.studio.inker.flourish"}


def _package_for(rel: str) -> str:
    parts = rel.split("/")[:-1]
    return ".".join(["warlock", "characters", *parts])


def _imports(rel: str) -> list[tuple[str, bool]]:
    """``(absolute module name, at module scope)`` for every import in a file."""
    package = _package_for(rel)
    tree = ast.parse((PACKAGE / rel).read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)) and isinstance(
                node, (ast.Import, ast.ImportFrom)
            ):
                top.add(id(child))
    found: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        module_scope = id(node) in top
        if isinstance(node, ast.Import):
            found.extend((alias.name, module_scope) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                root = package.rsplit(".", node.level - 1)[0]
                base = f"{root}.{node.module}" if node.module else root
            # ``from ..pipelines import charsheet`` names a *module* and
            # ``from .errors import CharacterError`` names a class, and the AST
            # cannot tell them apart -- so the deepest name that is really a
            # module wins. Without this the pin could only ever say "it imports
            # something from pipelines", which is not a dependency anyone can
            # review.
            for alias in node.names:
                found.append((_deepest(base, alias.name), module_scope))
    return found


def _deepest(base: str, name: str) -> str:
    candidate = f"{base}.{name}" if base else name
    try:
        if importlib.util.find_spec(candidate) is not None:
            return candidate
    except (ImportError, AttributeError, ValueError):
        pass
    return base


def _outward(rel: str) -> set[str]:
    """``warlock.*`` imports that leave the ``characters`` package."""
    return {
        name
        for name, _scope in _imports(rel)
        if name.split(".")[0] == "warlock" and not name.startswith("warlock.characters")
    }


def test_the_pin_covers_every_module_in_the_package():
    """A glob pin goes vacuous by a new file, not by a bad pattern."""
    found = {str(p.relative_to(PACKAGE)).replace("\\", "/") for p in PACKAGE.glob("*.py")}
    for sub in ARCHETYPE_PACKAGES:
        found |= {
            str(p.relative_to(PACKAGE)).replace("\\", "/")
            for p in (PACKAGE / sub).glob("*.py")
        }
    assert found == set(OUTWARD_IMPORTS), (
        "a module joined or left warlock.characters; the pin below now measures "
        "something other than the package"
    )


def test_the_archetype_packages_are_the_ones_written_down():
    """Adding a body plan is an edit here, on purpose: a new subpackage that
    quietly inherited this pin would inherit an allowlist written for a
    different generator's dependencies."""
    found = {p.name for p in PACKAGE.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
    assert found == ARCHETYPE_PACKAGES


@pytest.mark.parametrize("rel", sorted(OUTWARD_IMPORTS))
def test_no_module_imports_a_window_or_a_tensor(rel):
    roots = {name.split(".")[0] for name, _scope in _imports(rel)}
    assert not (roots & BANNED_ROOTS), f"{rel} imports {sorted(roots & BANNED_ROOTS)}"


@pytest.mark.parametrize("rel", sorted(OUTWARD_IMPORTS))
def test_no_module_imports_the_service_layer_or_the_queue(rel):
    """The dependency runs the other way: a service module reads a recipe
    through this package. An import back would be a cycle, and it would put a
    store-wide lock behind a function documented as pure."""
    for name, _scope in _imports(rel):
        assert not name.startswith("warlock.service"), f"{rel} imports {name}"
        assert not name.startswith("warlock.queue"), f"{rel} imports {name}"
        assert not name.startswith("warlock._q"), f"{rel} imports {name}"


@pytest.mark.parametrize("rel", sorted(OUTWARD_IMPORTS))
def test_the_only_warlock_imports_are_the_ones_written_down(rel):
    assert _outward(rel) == OUTWARD_IMPORTS[rel], rel


def test_clay_is_only_ever_imported_inside_a_function():
    """Clay drags trimesh and manifold3d in, and the only caller that needs them
    is the authoring script. A module-scope import would put the boolean
    kernel's import time on every cold start of the app, to answer a registry
    lookup that never touches geometry."""
    for rel in OUTWARD_IMPORTS:
        for name, module_scope in _imports(rel):
            if any(name.startswith(prefix) for prefix in LAZY_ONLY):
                assert not module_scope, f"{rel} imports {name} at module scope"


def test_flourish_is_only_ever_imported_inside_a_function():
    """Clay's rule, second instance, for the same cost in a different package.

    ``effects`` is the only module here that reaches Flourish, and Flourish is
    numpy plus ten primitive modules. A module-scope import would put all of it
    on the cold start of a door whose entire job is to answer a registry lookup
    -- and ``characters`` is imported by the door, by the recipe validator and
    by the resolver, none of which will ever draw a flame.
    """
    for rel in OUTWARD_IMPORTS:
        for name, module_scope in _imports(rel):
            if name.startswith("warlock.studio.inker.flourish"):
                assert not module_scope, f"{rel} imports {name} at module scope"


def test_the_effects_module_is_importable_with_no_studio_behind_it():
    """The pin above says the import is lazy; this says the laziness is worth
    something. ``import warlock.characters.effects`` on a machine with no
    window library must succeed, because that is what a queue worker's parent
    process does when it reads the registry to build a spec."""
    proc = _run(
        ("imgui", "imgui_bundle", "moderngl", "pygame", "bpy", "torch", "warlock.service"),
        "warlock.characters.effects",
    )
    assert proc.returncode == 0, proc.stderr


def _run(stubs: tuple[str, ...], imports: str) -> subprocess.CompletedProcess:
    """Import something in a fresh interpreter with modules stubbed to None.

    ``sys.modules[name] = None`` makes any attempt to import it raise, so a
    hidden import fails loudly rather than succeeding on a developer machine
    that happens to have the whole studio installed. A subprocess rather than a
    reload, for ``test_rigging_stays_importable_with_no_bpy_anywhere``'s reason.
    """
    script = (
        "import sys\n"
        f"for name in {stubs!r}: sys.modules[name] = None\n"
        f"import {imports}\n"
    )
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


def test_the_package_imports_with_no_studio_and_no_service():
    proc = _run(
        ("imgui", "imgui_bundle", "moderngl", "pygame", "bpy", "torch", "warlock.service"),
        "warlock.characters",
    )
    assert proc.returncode == 0, proc.stderr


def test_instantiating_needs_no_window_and_no_boolean_kernel():
    """The path a job actually takes: read a baked mesh, displace it, write it.
    trimesh and manifold3d are stubbed out alongside imgui because a character
    that could not be built on a machine without them would make the whole bake
    -- the reason the assets are checked in -- pointless."""
    proc = _run(
        ("imgui", "imgui_bundle", "moderngl", "pygame", "bpy", "torch",
         "warlock.service", "trimesh", "manifold3d"),
        "warlock.characters.instantiate",
    )
    assert proc.returncode == 0, proc.stderr
