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
MODES: list[tuple[str, str, str]] = [
    ("home", "Home", icons.HOUSE),
    ("manual", "Manual", icons.BOOK_OPEN),
    ("2d", "2D", icons.IMAGE),
    ("3d", "3D", icons.BOX),
    ("inker", "Inker", icons.PEN_TOOL),
    ("build", "Build", icons.RULER),
    ("clay", "Clay", icons.EGG),
    ("settings", "Settings", icons.SETTINGS),
]

# The modes that own a viewport or a form, and so have work in them. Home, the
# Manual, Clay and Settings are places you pass through: they have no form to
# submit and no viewport to frame, which is why they take no keyboard
# shortcuts at all.
WORK_MODES = frozenset({"2d", "3d", "inker", "build"})

# The subset that draws the *asset* viewport, and therefore the only modes
# whose selection is worth loading a mesh for. Inker and Build each own their
# own centre pane -- Build's draws a live document rather than a file, so
# ``_sync_viewer`` has nothing to do for it and returns early.
VIEWPORT_MODES = frozenset({"2d", "3d"})

# Neither one pane nor the asset viewport: a mode that fills the window with
# its own three-column workspace. Inker and Build are the two, and the three
# categories partition KEYS exactly -- which matters because ``_build_ui``'s
# dispatch ends in a bare ``else``, so an unlisted mode would draw one of these
# rather than fail.
WORKSPACE_MODES = frozenset({"inker", "build"})

KEYS = tuple(key for key, _label, _icon in MODES)
