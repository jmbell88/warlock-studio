"""What counts as an image file, in one place.

Pure in the way :mod:`.modes` is -- stdlib only, no imgui, no pygame -- so
every module that has to answer "may this file be opened here" can import it
without dragging a window in.

There were five hand-written spellings of this list before: the drop router's
tuple, the shared picker's filter, Inker's two, Packwright's and Plotter's. Two
of them had already drifted -- the Packwright and Plotter filters *accepted*
``.jpeg`` and ``.bmp`` while their labels advertised neither, which is the worst
version of the drift because the user is told a file is unsupported by the very
dialog that would have opened it.

:func:`pattern` is the one a filter list wants. This file used to claim there
were two valid shapes and that neither was wrong; there is one, and the other
is a bug. portable-file-dialogs consumes ``filters`` **two at a time** -- a
label, then that label's space-separated patterns, then the next label -- so a
label followed by one entry per glob is not "these patterns under this label":
the second glob becomes the first row's only pattern and the third becomes the
*next row's label*. Plotter's "Add a tileset" was built that way and opened on
a row advertising six formats while filtering on ``*.tsx`` alone, which made a
folder of PNGs look empty. ``pfd.all_files_filter()`` returning
``["All files", "*"]`` is the shape the library states for itself; a scan test
now holds every ``*_FILTER`` in the app to it.
"""

from __future__ import annotations

from collections.abc import Iterable

# Every raster format the app will open, load or accept on a drop. Lower case
# and dotted, so a ``Path.suffix.lower()`` compares directly.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def _globs(suffixes: Iterable[str] = IMAGE_SUFFIXES) -> list[str]:
    """``[".png", ...]`` -> ``["*.png", ...]``, one entry per pattern.

    Private, and that is the point: splatting this into a filter list beside a
    label is exactly the mis-pairing described above. It exists only to be
    joined by :func:`pattern`.
    """
    return [f"*{suffix}" for suffix in suffixes]


def pattern(suffixes: Iterable[str] = IMAGE_SUFFIXES) -> str:
    """The globs as one space-separated string -- what a pfd filter row takes."""
    return " ".join(_globs(suffixes))


def describe(name: str, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> str:
    """``"Images"`` -> ``"Images (*.png *.jpg *.jpeg *.webp *.bmp)"``.

    Derived rather than written beside the patterns, because a label and a
    pattern list maintained separately is exactly how a dialog comes to refuse
    what it accepts.
    """
    return f"{name} ({pattern(suffixes)})"
