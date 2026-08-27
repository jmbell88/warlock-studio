"""Context help: which manual chapter a pane's (?) opens.

Pure data, importable headlessly -- the docs integrity test validates every
entry against the real chapters and anchors.
"""

from __future__ import annotations

HELP_TARGETS: dict[str, tuple[str, str | None]] = {
    "settings-2d": ("22-generating-references", None),
    # The Sheet output's own block. Its own entry rather than sharing
    # ``settings-2d``: the chapter is long and the two settings it explains --
    # why the grid is not a control, and why the sprite arm makes two rows --
    # are the questions a user has while looking at that section.
    "settings-sheet": ("22-generating-references", "sheets"),
    "settings-3d": ("23-generating-meshes", None),
    # The Rig stage's own column (the UI redesign, wave 5). Rigging was three
    # buttons in three places and no pane of its own, so it had no (?) either.
    "settings-rig": ("25-rigging-and-posing", "rigging-a-mesh"),
    "library": ("35-library-and-jobs", None),
    "inspector": ("23-generating-meshes", "exports"),
    "retarget": ("23-generating-meshes", "triangle-budget"),
    "retexture": ("23-generating-meshes", "surface-texture"),
    "pose": ("25-rigging-and-posing", "posing"),
    "poser-library": ("26-poser", "the-pose-library"),
    "poser-controls": ("26-poser", "posing-a-skeleton"),
    "poser-clips": ("26-poser", "editing-clips"),
    "sheet": ("27-sprite-sheets", None),
    "sprites": ("27-sprite-sheets", "from-a-single-drawing"),
    "inker-timeline": ("29-inker-animation", "the-timeline"),
    # Found by the O118 coverage sweep: two panes a user reads and neither had
    # a way into the chapter that describes it.
    "inker-colors": ("28-inker", "colour"),
    # The tile panel (Wave 3). Its own anchor rather than sharing
    # ``inker-tools``: a tilemap layer is a different *kind* of layer, and the
    # questions asked in front of this panel -- what Manual/Auto/Stack do, what
    # a flag bit is, why an old build drops the tiles -- are all in that one
    # section.
    "inker-tiles": ("28-inker", "tilemap-layers"),
    "inker-preview": ("29-inker-animation", "preview"),
    # The toolbox. It had no entry for as long as it was a 90 px rail with no
    # room for a heading to hang a (?) beside; it is a sidebar pane now, so it
    # points at the section that lists every tool and its letter.
    "inker-tools": ("28-inker", "tools"),
    # The slider surface under the palette. Its own anchor rather than sharing
    # ``inker-colors``: the questions asked in front of it -- which of the two
    # colours am I editing, why do the sliders move a palette entry in an
    # indexed document, what does the hex field accept -- are all in that one
    # section.
    "inker-picker": ("28-inker", "the-colour-picker"),
    # The four ways a drawing leaves the Inker. The bridges section is where
    # the *directions* are explained, which is what somebody standing in front
    # of a greyed "Revert to original" is asking about.
    "inker-generate": ("28-inker", "pipeline-bridges"),
    "candidates": ("23-generating-meshes", "candidates"),
    # The viewport toolbar. The ~5k-LOC subsystem in the middle of the window
    # was chrome as far as this map was concerned -- exempted in
    # ``tests/manual/test_coverage.py`` for having "no titled section" -- while
    # being the one thing on screen at every stage of Create and in Review.
    "overlay": ("24-the-3d-viewport", "the-toolbar"),
    "clay-tools": ("30-clay", "transforming"),
    "clay-props": ("30-clay", "materials"),
    "clay-outliner": ("30-clay", "adding-a-primitive"),
    "clay-bridge": ("30-clay", "the-two-ways-out"),
    "plotter-tools": ("31-plotter", "tools"),
    # The sheet over the centre pane: three titled tabs a user interacts with,
    # so an exemption would be false.
    "plotter-tileset-editor": ("31-plotter", "tilesets"),
    "plotter-tileset": ("31-plotter", "tilesets"),
    "plotter-layers": ("31-plotter", "layers"),
    "plotter-bridge": ("31-plotter", "files"),
    "packwright-sources": ("32-packwright", "sources"),
    "packwright-settings": ("32-packwright", "settings"),
    "packwright-items": ("32-packwright", "when-it-does-not-fit"),
    "packwright-bridge": ("32-packwright", "exporting"),
    # Sirens' six panes, pointed at the chapter that now exists. Every one of
    # them sat on ``20-overview#the-modes`` between phases 2 and 5 -- a
    # placeholder rather than a missing button, because a (?) that appears
    # later is one users learn to look for later. Six entries rather than one
    # chapter-wide target for Troupe's reason: the question differs per pane.
    # What a render is and why Play is greyed is the transport's; what an order
    # list is *for* is the order panel's; which number a cell holds is the
    # instrument list's; what a release point splits is the envelope editor's;
    # why an effect keeps its own tempo is the effects panel's; and what a
    # folder of WAVs contains is the bridge's.
    "sirens-transport": ("34-sirens", "playing-it"),
    "sirens-orders": ("34-sirens", "patterns-and-the-order"),
    "sirens-instruments": ("34-sirens", "instruments"),
    "sirens-envelopes": ("34-sirens", "the-envelope-editor"),
    "sirens-effects": ("34-sirens", "sound-effects"),
    "sirens-bridge": ("34-sirens", "exporting-the-audio"),
    # Troupe's four panes. Four entries rather than one chapter-wide target
    # because the questions differ per pane: what a sheet *is* is the cast's
    # question, what the options mean is the form's, why the preview never
    # stops is the centre's, and what a stray-pixel count means is the sheet
    # panel's.
    "troupe-characters": ("33-troupe", "what-a-character-sheet-contains"),
    "troupe-settings": ("33-troupe", "making-a-character"),
    "troupe-preview": ("33-troupe", "watching-it"),
    "troupe-sheets": ("33-troupe", "the-sheet-panel"),
    "troupe-bridge": ("33-troupe", "taking-it-somewhere"),
    "profiles": ("36-profiles", None),
    "review": ("37-review", None),
    "app-settings": ("41-app-settings", None),
    # The chooser the app opens on (F56/O118): the one pane a first run
    # certainly sees, and the only one that had no way into the manual at all.
    "home": ("21-home", None),
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
TROUBLESHOOTING: tuple[str, str | None] = ("42-troubleshooting", None)
