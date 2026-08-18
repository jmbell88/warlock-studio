"""Context help: which manual chapter a pane's (?) opens.

Pure data, importable headlessly -- the docs integrity test validates every
entry against the real chapters and anchors.
"""

from __future__ import annotations

HELP_TARGETS: dict[str, tuple[str, str | None]] = {
    "settings-2d": ("03-generating-references", None),
    "settings-3d": ("04-generating-meshes", None),
    # The Rig stage's own column (the UI redesign, wave 5). Rigging was three
    # buttons in three places and no pane of its own, so it had no (?) either.
    "settings-rig": ("05-rigging-and-posing", "rigging-a-mesh"),
    "library": ("13-library-and-jobs", None),
    "inspector": ("04-generating-meshes", "exports"),
    "retarget": ("04-generating-meshes", "triangle-budget"),
    "retexture": ("04-generating-meshes", "surface-texture"),
    "pose": ("05-rigging-and-posing", "posing"),
    "poser-library": ("06-poser", "the-pose-library"),
    "poser-controls": ("06-poser", "posing-a-skeleton"),
    "sheet": ("07-sprite-sheets", None),
    "sprites": ("07-sprite-sheets", "from-a-single-drawing"),
    "inker-tools": ("08-inker", "tools"),
    "inker-layers": ("08-inker", "layers"),
    "inker-bridge": ("08-inker", "pipeline-bridges"),
    "inker-timeline": ("09-inker-animation", "the-timeline"),
    # Found by the O118 coverage sweep: two panes a user reads and neither had
    # a way into the chapter that describes it.
    "inker-colors": ("08-inker", "colour"),
    "inker-preview": ("09-inker-animation", "preview"),
    "candidates": ("04-generating-meshes", "candidates"),
    "clay-tools": ("10-clay", "transforming"),
    "clay-props": ("10-clay", "materials"),
    "clay-outliner": ("10-clay", "adding-a-primitive"),
    "clay-bridge": ("10-clay", "the-two-ways-out"),
    "plotter-tools": ("11-plotter", "tools"),
    "plotter-tileset": ("11-plotter", "tilesets"),
    "plotter-terrain": ("11-plotter", "generating-a-ground-set"),
    "plotter-layers": ("11-plotter", "layers"),
    "plotter-bridge": ("11-plotter", "files"),
    "packwright-sources": ("12-packwright", "sources"),
    "packwright-settings": ("12-packwright", "settings"),
    "packwright-items": ("12-packwright", "when-it-does-not-fit"),
    "packwright-bridge": ("12-packwright", "exporting"),
    "profiles": ("14-profiles", None),
    "review": ("15-review", None),
    "app-settings": ("19-app-settings", None),
    # The chooser the app opens on (F56/O118): the one pane a first run
    # certainly sees, and the only one that had no way into the manual at all.
    "home": ("02-home", None),
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
TROUBLESHOOTING: tuple[str, str | None] = ("20-troubleshooting", None)
