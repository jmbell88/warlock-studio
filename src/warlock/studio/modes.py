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
    ("clay", "Clay", icons.EGG),
    ("settings", "Settings", icons.SETTINGS),
]

# The modes that are worth remembering between runs: the ones that own a
# viewport or a form, and so have work in them. Home, the Manual, Clay and
# Settings are places you pass through -- restoring into one of them on the
# next launch would hide whatever was actually being made.
WORK_MODES = frozenset({"2d", "3d", "inker"})

KEYS = tuple(key for key, _label, _icon in MODES)
