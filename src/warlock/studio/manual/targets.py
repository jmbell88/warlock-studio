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
    "clay-tools": ("07-clay", "transforming"),
    "clay-props": ("07-clay", "materials"),
    "clay-outliner": ("07-clay", "adding-a-primitive"),
    "clay-bridge": ("07-clay", "the-two-ways-out"),
    "profiles": ("08-library-and-jobs", "profiles"),
    "review": ("16-review", None),
    "app-settings": ("11-configuration", "in-app-settings"),
}
