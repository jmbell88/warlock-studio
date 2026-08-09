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
    ("plotter", "Plotter", icons.GRID),
    ("packwright", "Packwright", icons.LAYERS),
]

# The modes that own a viewport or a form, and so have work in them. Home, the
# Manual and Settings are places you pass through: they have no form to
# submit and no viewport to frame, which is why they take no keyboard
# shortcuts at all.
WORK_MODES = frozenset({"2d", "3d", "inker", "clay", "review", "plotter", "packwright"})

# The subset that draws the *asset* viewport, and therefore the only modes
# whose selection is worth loading a mesh for. Inker and Clay each own their
# own centre pane -- Clay's draws a live document rather than a file, so
# ``_sync_viewer`` has nothing to do for it and returns early. Review is not
# here either, and deliberately: it *borrows* the shared viewer, but for a
# sweep unit's mesh rather than for the library selection, so leaving it out is
# what stops ``_sync_viewer`` reloading the selected asset over it.
VIEWPORT_MODES = frozenset({"2d", "3d"})

# Neither one pane nor the asset viewport: a mode that fills the window with
# its own three-column workspace. Inker, Clay, Review, Plotter and Packwright
# are the five, and the three categories partition KEYS exactly -- which
# matters because ``_build_ui``'s dispatch ends in a bare ``else``, so an
# unlisted mode would draw one of these rather than fail.
WORKSPACE_MODES = frozenset({"inker", "clay", "review", "plotter", "packwright"})

KEYS = tuple(key for key, _label, _icon in MODES)

# After which segment indices the switch leaves a wider gap (UX.md Phase 2).
# Ten flat segments said that Manual and Settings were peers of the five
# creative workspaces; a gap says they are not, and says it in the *spacing*,
# so ``MODES``' order is untouched and every Alt+N position is exactly what it
# was.
#
# **Derived from where the category changes, never written out.** The bullet
# this comes from describes one gap, between the places (Home, Manual,
# Settings) and the workspaces -- which is one gap only if the places are
# contiguous, and in this list they are not: Settings sits eighth, between
# Review and Plotter, because it was appended when it was added and the digits
# were already in people's hands. So the honest rendering of "places are not
# workspaces" against *this* order is a break at each transition, and the day
# the order does group them the set collapses to a single break with nothing to
# edit. A hand-written index would instead have put one gap in the middle of
# the workspaces and called it a grouping.
GROUP_BREAKS: frozenset[int] = frozenset(
    index
    for index, ((key, _l, _i), (nxt, _l2, _i2)) in enumerate(zip(MODES, MODES[1:], strict=False))
    if (key in WORK_MODES) != (nxt in WORK_MODES)
)

# Alt+1..9 and Alt+0, positionally: the nth segment of the switch is the nth
# digit, so the binding is the picture on screen rather than a second table to
# keep in agreement with it. The tenth slot is Alt+0 for the reason every
# application with ten of anything uses 0 for the tenth -- see
# :func:`digit_key_label`, which is where that spelling lives so the palette
# hint and the shortcuts popup cannot disagree about it.
#
# **Alt, not Ctrl.** Mode switching is the one binding that has to fire in
# every mode -- including Inker and Clay, whose ``handle_key`` consumes
# everything unconditionally -- so it is checked before them, which means it
# takes whatever it names away from them for good. Inker already binds Ctrl+0
# and Ctrl+1 to fit and 100% zoom, and Clay's axis views want Ctrl+1/3/7, so
# Ctrl+digit was a key the workspace modes were already using. Alt+digit is
# used by nothing here.
def mode_for_digit(digit: int) -> str | None:
    """``1`` -> ``"home"``, ``10`` -> the tenth mode; ``None`` past the end.

    Positional and 1-based throughout. That 10 is typed as ``0`` is a *keyboard*
    fact and lives at the keyboard layer, not here -- ``mode_for_digit(0)`` is
    still None, because there is no zeroth segment.
    """
    if 1 <= digit <= len(MODES):
        return MODES[digit - 1][0]
    return None


def digit_key_label(digit: int) -> str:
    """What the user actually presses for slot ``digit``: ``"0"`` for the tenth.

    One spelling, in one place. The palette's hint and the shortcuts popup both
    render this, and a second copy is how one of them comes to advertise
    "Alt+10" -- a key no keyboard has.
    """
    return "0" if digit == 10 else str(digit)


def digit_for_mode(key: str) -> int | None:
    """The inverse, for the shortcut list. ``None`` for a mode that is not in
    the switch (there is none today; ``QUIT`` is not a mode)."""
    for index, (mode_key, _label, _icon) in enumerate(MODES, start=1):
        if mode_key == key:
            return index
    return None

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
