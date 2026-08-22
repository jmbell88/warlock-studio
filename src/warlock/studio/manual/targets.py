"""Context help: which manual chapter a pane's (?) opens.

Pure data, importable headlessly -- the docs integrity test validates every
entry against the real chapters and anchors.
"""

from __future__ import annotations

HELP_TARGETS: dict[str, tuple[str, str | None]] = {
    "settings-2d": ("03-generating-references", None),
    # The Sheet output's own block. Its own entry rather than sharing
    # ``settings-2d``: the chapter is long and the two settings it explains --
    # why the grid is not a control, and why the sprite arm makes two rows --
    # are the questions a user has while looking at that section.
    "settings-sheet": ("03-generating-references", "sheets"),
    "settings-3d": ("04-generating-meshes", None),
    # The Rig stage's own column (the UI redesign, wave 5). Rigging was three
    # buttons in three places and no pane of its own, so it had no (?) either.
    "settings-rig": ("06-rigging-and-posing", "rigging-a-mesh"),
    "library": ("15-library-and-jobs", None),
    "inspector": ("04-generating-meshes", "exports"),
    "retarget": ("04-generating-meshes", "triangle-budget"),
    "retexture": ("04-generating-meshes", "surface-texture"),
    "pose": ("06-rigging-and-posing", "posing"),
    "poser-library": ("07-poser", "the-pose-library"),
    "poser-controls": ("07-poser", "posing-a-skeleton"),
    "poser-clips": ("07-poser", "editing-clips"),
    "sheet": ("08-sprite-sheets", None),
    "sprites": ("08-sprite-sheets", "from-a-single-drawing"),
    "inker-tools": ("09-inker", "tools"),
    "inker-timeline": ("10-inker-animation", "the-timeline"),
    # Found by the O118 coverage sweep: two panes a user reads and neither had
    # a way into the chapter that describes it.
    "inker-colors": ("09-inker", "colour"),
    # The tile panel (Wave 3). Its own anchor rather than sharing
    # ``inker-tools``: a tilemap layer is a different *kind* of layer, and the
    # questions asked in front of this panel -- what Manual/Auto/Stack do, what
    # a flag bit is, why an old build drops the tiles -- are all in that one
    # section.
    "inker-tiles": ("09-inker", "tilemap-layers"),
    "inker-preview": ("10-inker-animation", "preview"),
    "candidates": ("04-generating-meshes", "candidates"),
    # The viewport toolbar. The ~5k-LOC subsystem in the middle of the window
    # was chrome as far as this map was concerned -- exempted in
    # ``tests/manual/test_coverage.py`` for having "no titled section" -- while
    # being the one thing on screen at every stage of Create and in Review.
    "overlay": ("05-the-3d-viewport", "the-toolbar"),
    "clay-tools": ("11-clay", "transforming"),
    "clay-props": ("11-clay", "materials"),
    "clay-outliner": ("11-clay", "adding-a-primitive"),
    "clay-bridge": ("11-clay", "the-two-ways-out"),
    "plotter-tools": ("12-plotter", "tools"),
    "plotter-tileset": ("12-plotter", "tilesets"),
    "plotter-layers": ("12-plotter", "layers"),
    "plotter-bridge": ("12-plotter", "files"),
    "packwright-sources": ("13-packwright", "sources"),
    "packwright-settings": ("13-packwright", "settings"),
    "packwright-items": ("13-packwright", "when-it-does-not-fit"),
    "packwright-bridge": ("13-packwright", "exporting"),
    # Troupe's four panes. Four entries rather than one chapter-wide target
    # because the questions differ per pane: what a sheet *is* is the cast's
    # question, what the options mean is the form's, why the preview never
    # stops is the centre's, and what a stray-pixel count means is the sheet
    # panel's.
    "troupe-characters": ("14-troupe", "what-a-character-sheet-contains"),
    "troupe-settings": ("14-troupe", "making-a-character"),
    "troupe-preview": ("14-troupe", "watching-it"),
    "troupe-sheets": ("14-troupe", "the-sheet-panel"),
    "troupe-bridge": ("14-troupe", "taking-it-somewhere"),
    "profiles": ("16-profiles", None),
    "review": ("17-review", None),
    "app-settings": ("21-app-settings", None),
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
TROUBLESHOOTING: tuple[str, str | None] = ("22-troubleshooting", None)
