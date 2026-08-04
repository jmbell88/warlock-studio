"""Context help: which manual chapter a pane's (?) opens.

Pure data, importable headlessly -- the docs integrity test validates every
entry against the real chapters and anchors.
"""

from __future__ import annotations

HELP_TARGETS: dict[str, tuple[str, str | None]] = {
    "settings-2d": ("02-generating-references", None),
    "settings-3d": ("03-generating-meshes", None),
    "library": ("07-library-and-jobs", None),
    "inspector": ("03-generating-meshes", "exports"),
    "retarget": ("03-generating-meshes", "triangle-budget"),
    "pose": ("04-rigging-and-posing", "posing"),
    "sheet": ("05-sprite-sheets", None),
    "inker-tools": ("06-paint", "tools"),
    "inker-layers": ("06-paint", "layers"),
    "inker-bridge": ("06-paint", "pipeline-bridges"),
    "profiles": ("07-library-and-jobs", "profiles"),
}
