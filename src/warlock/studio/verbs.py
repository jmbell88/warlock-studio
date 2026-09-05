"""The cross-workspace verbs, by effect.

Every button that hands an asset from one part of the app to another is one
of four gestures, and the 2026-09-05 consistency review found them worded
five ways across nine panes -- "Export to library" here, "Export to the
library" there, the same "Open in Inker" as a button, a small button and a
menu row. The wording is fixed here and the panes ask for it.

- :func:`open_in` -- the asset opens *for editing* in that workspace.
- :func:`add_to` -- the asset joins an open document there as a *source*.
- :func:`send_to` -- the asset starts a *process* there (Troupe rigs and
  renders; nothing is opened for editing).
- :data:`EXPORT_TO_LIBRARY` -- the document is *published* as a library asset.

The module imports :mod:`.modes` and nothing else, so anything can use it --
including ``state`` and ``inker_ops``, which draw nothing.
"""

from __future__ import annotations

from . import modes

_LABELS: dict[str, str] = {key: label for key, label, _icon in modes.MODES}


def _mode(key: str) -> str:
    return _LABELS[key]


def open_in(mode: str) -> str:
    """``Open in Inker``: opens for editing, in place -- not a copy."""
    return f"Open in {_mode(mode)}"


def add_to(mode: str, what: str = "") -> str:
    """``Add to Packwright`` / ``Add to Plotter as a tileset``: joins a document
    there as a source. ``what`` is the optional clause a menu row has room for."""
    label = f"Add to {_mode(mode)}"
    return f"{label} {what}" if what else label


def send_to(mode: str) -> str:
    """``Send to Troupe``: starts that workspace's process on the asset."""
    return f"Send to {_mode(mode)}"


#: Publishing a document as a library asset. The article is part of it: three
#: panes said "the library" and one said "library", and the odd one read as a
#: different place.
EXPORT_TO_LIBRARY = "Export to the library"
