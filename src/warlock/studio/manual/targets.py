"""Context help: which manual chapter a pane's (?) opens.

Pure data, importable headlessly -- the docs integrity test validates every
entry against the real chapters and anchors.
"""

from __future__ import annotations

HELP_TARGETS: dict[str, tuple[str, str | None]] = {
    "settings-2d": ("02-generating-references", None),
    "settings-3d": ("03-generating-meshes", None),
    "library": ("08-library-and-jobs", None),
    "inspector": ("03-generating-meshes", "exports"),
    "retarget": ("03-generating-meshes", "triangle-budget"),
    "pose": ("04-rigging-and-posing", "posing"),
    "sheet": ("05-sprite-sheets", None),
    "inker-tools": ("06-inker", "tools"),
    "inker-layers": ("06-inker", "layers"),
    "inker-bridge": ("06-inker", "pipeline-bridges"),
    "build-tools": ("07-build", "transforming"),
    "build-props": ("07-build", "materials"),
    "build-outliner": ("07-build", "adding-a-primitive"),
    "build-bridge": ("07-build", "the-two-ways-out"),
    "profiles": ("08-library-and-jobs", "profiles"),
    "app-settings": ("11-configuration", "in-app-settings"),
}
