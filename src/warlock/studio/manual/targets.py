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
    "inker-timeline": ("06-inker", "animation"),
    "clay-tools": ("07-clay", "transforming"),
    "clay-props": ("07-clay", "materials"),
    "clay-outliner": ("07-clay", "adding-a-primitive"),
    "clay-bridge": ("07-clay", "the-two-ways-out"),
    "profiles": ("08-library-and-jobs", "profiles"),
    "review": ("16-review", None),
    "app-settings": ("11-configuration", "in-app-settings"),
    # The chooser the app opens on (F56/O118): the one pane a first run
    # certainly sees, and the only one that had no way into the manual at all.
    "home": ("01-overview", None),
}

# Where "something is wrong and I do not know what" goes.
#
# Deliberately *not* a HELP_TARGETS entry, though it is the same shape. That
# dict is the pane-(?)-button map and is asserted against the call sites in both
# directions (``test_help_button_call_sites_match_help_targets``) precisely so a
# dead button or dead data fails a test -- and the three surfaces that lead here
# are a red banner, a popup and a Home row, none of which is a pane with a (?).
# Named once rather than spelled at each of the three, so a chapter that moved
# does not have to be found in three places (F57).
TROUBLESHOOTING: tuple[str, str | None] = ("12-troubleshooting", None)
