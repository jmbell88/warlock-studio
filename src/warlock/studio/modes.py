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
    ("clay", "Clay", icons.RULER),
    ("review", "Review", icons.CIRCLE_CHECK),
    ("settings", "Settings", icons.SETTINGS),
]

# The modes that own a viewport or a form, and so have work in them. Home, the
# Manual and Settings are places you pass through: they have no form to
# submit and no viewport to frame, which is why they take no keyboard
# shortcuts at all.
WORK_MODES = frozenset({"2d", "3d", "inker", "clay", "review"})

# The subset that draws the *asset* viewport, and therefore the only modes
# whose selection is worth loading a mesh for. Inker and Clay each own their
# own centre pane -- Clay's draws a live document rather than a file, so
# ``_sync_viewer`` has nothing to do for it and returns early. Review is not
# here either, and deliberately: it *borrows* the shared viewer, but for a
# sweep unit's mesh rather than for the library selection, so leaving it out is
# what stops ``_sync_viewer`` reloading the selected asset over it.
VIEWPORT_MODES = frozenset({"2d", "3d"})

# Neither one pane nor the asset viewport: a mode that fills the window with
# its own three-column workspace. Inker, Clay and Review are the three, and
# the three categories partition KEYS exactly -- which matters because
# ``_build_ui``'s dispatch ends in a bare ``else``, so an unlisted mode would
# draw one of these rather than fail.
WORKSPACE_MODES = frozenset({"inker", "clay", "review"})

KEYS = tuple(key for key, _label, _icon in MODES)

# Alt+1..8, positionally: the nth segment of the switch is the nth digit, so
# the binding is the picture on screen rather than a second table to keep in
# agreement with it.
#
# **Alt, not Ctrl.** Mode switching is the one binding that has to fire in
# every mode -- including Inker and Clay, whose ``handle_key`` consumes
# everything unconditionally -- so it is checked before them, which means it
# takes whatever it names away from them for good. Inker already binds Ctrl+0
# and Ctrl+1 to fit and 100% zoom, and Clay's axis views want Ctrl+1/3/7, so
# Ctrl+digit was a key the workspace modes were already using. Alt+digit is
# used by nothing here.
def mode_for_digit(digit: int) -> str | None:
    """``1`` -> ``"home"``, ``8`` -> ``"settings"``; ``None`` past the end."""
    if 1 <= digit <= len(MODES):
        return MODES[digit - 1][0]
    return None


def digit_for_mode(key: str) -> int | None:
    """The inverse, for the shortcut list. ``None`` for a mode that is not in
    the switch (there is none today; ``QUIT`` is not a mode)."""
    for index, (mode_key, _label, _icon) in enumerate(MODES, start=1):
        if mode_key == key:
            return index
    return None

# Drawn in the switch, deliberately *not* in MODES. Quitting is an action, not
# a place: it never lands in ``AppState.mode``, it has no pane, and the three
# categories above have to partition KEYS exactly (``_build_ui``'s dispatch
# ends in a bare ``else``). Same tuple shape only so the switch can splice it
# onto the end of the list it already builds.
QUIT: tuple[str, str, str] = ("quit", "Quit", icons.POWER)
