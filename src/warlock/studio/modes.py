"""The top-level modes, as data.

One list, in the order the switch draws them. Deliberately data and nothing
else -- the dispatch that turns ``state.mode`` into a pane stays hand-coded in
:mod:`.main`, because "``state.mode`` is the only thing that decides what a
pane shows" is only true while there is exactly one place doing the deciding.
A table of callbacks here would be a second one.

The module imports :mod:`.icons` and nothing else, so it stays importable from
anywhere without dragging imgui in.
"""

from __future__ import annotations

from . import icons

# (key, label, icon). The key is what lands in ``AppState.mode``.
# **The order is the grouping**, and it is contiguous on purpose: the two ways
# in (Home, the Manual), then every workspace, then the three places that are
# about the program and its shelves rather than about a piece of work. It was
# not contiguous before -- Settings sat eighth, between Review and Plotter,
# because it was appended when it was added and the positional Alt+digits were
# already in people's hands, and Plotter and Packwright were then appended after
# *it* for the same reason. Nothing is typed positionally any more, so the list
# is free to say what it means, and ``GROUP_BREAKS`` (derived) collapses from
# four breaks to the two real ones.
MODES: list[tuple[str, str, str]] = [
    ("home", "Home", icons.HOUSE),
    ("manual", "Manual", icons.BOOK_OPEN),
    ("2d", "2D", icons.IMAGE),
    ("3d", "3D", icons.BOX),
    ("inker", "Inker", icons.PEN_TOOL),
    ("clay", "Clay", icons.RULER),
    ("poser", "Poser", icons.PERSON_STANDING),
    ("review", "Review", icons.CIRCLE_CHECK),
    ("plotter", "Plotter", icons.GRID),
    ("packwright", "Packwright", icons.LAYERS),
    ("settings", "Settings", icons.SETTINGS),
    # Real modes rather than sub-views of Home. They were tiles on the chooser
    # and a ``state.landing_view`` enum behind it, which is what a destination
    # looks like when there is nowhere to put it; Home stopped being a tile
    # grid, so they went where everything else already was. The two glyphs are
    # the ones ``landing._SUBVIEW_ICONS`` already assigned them -- moved, not
    # re-picked, because a screen the user has seen should not change its
    # pictures for a refactor.
    ("library", "Library", icons.FOLDER_OPEN),
    ("profiles", "Profiles", icons.SLIDERS),
]

# The modes that own a viewport or a form, and so have work in them. Home, the
# Manual and Settings are places you pass through: they have no form to
# submit and no viewport to frame, which is why they take no keyboard
# shortcuts at all.
WORK_MODES = frozenset(
    {"2d", "3d", "inker", "clay", "poser", "review", "plotter", "packwright"}
)

# The subset that draws the *asset* viewport, and therefore the only modes
# whose selection is worth loading a mesh for. Inker and Clay each own their
# own centre pane -- Clay's draws a live document rather than a file, so
# ``_sync_viewer`` has nothing to do for it and returns early. Review is not
# here either, and deliberately: it *borrows* the shared viewer, but for a
# sweep unit's mesh rather than for the library selection, so leaving it out is
# what stops ``_sync_viewer`` reloading the selected asset over it.
VIEWPORT_MODES = frozenset({"2d", "3d"})

# Neither one pane nor the asset viewport: a mode that fills the window with
# its own three-column workspace. Inker, Clay, Poser, Review, Plotter and
# Packwright are the six; Library and Profiles are single panes, not
# workspaces, and join Home/Manual/Settings there. The three categories
# partition KEYS exactly -- which matters because ``_build_ui``'s dispatch ends
# in a bare ``else``, so an unlisted mode would draw one of these rather than
# fail.
WORKSPACE_MODES = frozenset({"inker", "clay", "poser", "review", "plotter", "packwright"})

KEYS = tuple(key for key, _label, _icon in MODES)

# After which segment indices the switch leaves a wider gap (UX.md Phase 2).
# A flat row of segments said that Manual and Settings were peers of the
# creative workspaces; a gap says they are not, and says it in the *spacing*,
# so ``MODES``' order is untouched.
#
# **Derived from where the category changes, never written out** -- which is
# what let the reorder above be a reorder and nothing else. While the places
# were scattered through the list this rendered as four breaks, because that is
# the honest picture of "places are not workspaces" against an order that did
# not group them; now that ``MODES`` is contiguous it renders as the two real
# ones, with nothing here to edit. A hand-written index would have had to be
# rewritten, and until somebody did it would have put a gap in the middle of the
# workspaces and called it a grouping.
GROUP_BREAKS: frozenset[int] = frozenset(
    index
    for index, ((key, _l, _i), (nxt, _l2, _i2)) in enumerate(zip(MODES, MODES[1:], strict=False))
    if (key in WORK_MODES) != (nxt in WORK_MODES)
)

# **There is no positional Alt+digit binding, and there deliberately is not.**
# It existed while there were ten modes and ten digits, on the argument that the
# binding was the picture on screen rather than a second table. That argument
# stopped holding the moment Library and Profiles became modes: twelve segments
# against ten digits means either two modes with no key, or a second table
# saying which two -- and the second table is exactly what the positional
# scheme existed to avoid. So mode switching is a mouse action and a palette
# (Ctrl+K) action, and the digits go back to the workspace modes that were
# already reaching for them.


# Deliberately *not* in MODES. Quitting is an action, not a place: it never
# lands in ``AppState.mode``, it has no pane, and the three categories above
# have to partition KEYS exactly (``_build_ui``'s dispatch ends in a bare
# ``else``).
#
# It used to be spliced onto the end of the switch as an eleventh segment, and
# it no longer is (UX.md Phase 2): a destructive action living inside the
# control you navigate with is one click from every mode, and the
# unconditional confirm in front of it was a mitigation rather than a fix. It
# is drawn in the header's right-hand strip now, beside the two other controls
# that are about the program rather than about the work. The tuple survives
# because the name and the glyph are still one fact and the strip reads them
# from here; the ``(key, label, icon)`` shape survives with it, even though
# nothing splices it any more, because it is what makes that obvious.
QUIT: tuple[str, str, str] = ("quit", "Quit", icons.POWER)
