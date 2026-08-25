"""One "is this file small enough to open" question, for every mode that opens one.

``plotter_io`` and ``packwright_io`` each carried a private ``_within_ceiling``
-- correct, well argued, and wired only to the door it sat next to. Clay had
none at all, though ``service.files.MAX_CLAY_SOURCE_BYTES`` has existed since
the format did and is applied at exactly one place, the *upload*; Inker read a
``.aseprite`` and two ``.gpl`` files with no ceiling either.

That is ``zipguard``'s finding again: the number exists, the rule exists, and
the rule holds at the call sites that remember it. A shared helper cannot make
a caller remember, but it can make remembering cost one line and make the scan
test that checks for it possible to write -- which is the pair
``atomic.staged`` and its ``dialogs.save_file`` scan already form for writes.

**A ``ServiceError``, not a ``ValueError``.** This is the *mode's* refusal about
a file the user picked, not an engine's about a document's contents, and its
text reaches the user verbatim through the task classifier. The service layer is
imported inside the function so a mode pays nothing for it until it opens a
file, and the ceiling is read at call time so a test lowers it rather than
building half a gigabyte.
"""

from __future__ import annotations

from pathlib import Path


def within_ceiling(path: Path, ceiling: int, *, field: str = "file") -> Path:
    """Refuse a file past *ceiling* bytes, before a byte of it is read.

    Returns the path so a caller reads ``within_ceiling(p, N).read_bytes()`` in
    one expression -- the shape both private copies already had, kept because a
    helper whose result is easy to drop on the floor is a helper that gets
    called and ignored.
    """
    from ..service.errors import TooLarge

    if Path(path).stat().st_size > ceiling:
        raise TooLarge(
            f"{Path(path).name} is past the {ceiling} bytes this build will open",
            field=field,
        )
    return Path(path)
