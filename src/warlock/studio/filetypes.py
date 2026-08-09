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

The two shapes below both exist because portable-file-dialogs takes a filter as
a name followed by patterns, and the app writes that two ways: one
space-separated pattern string, or one entry per glob. Neither is wrong and
both are in use, so this owns the *suffixes* and renders whichever shape a call
site already had.
"""

from __future__ import annotations

from collections.abc import Iterable

# Every raster format the app will open, load or accept on a drop. Lower case
# and dotted, so a ``Path.suffix.lower()`` compares directly.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def globs(suffixes: Iterable[str] = IMAGE_SUFFIXES) -> list[str]:
    """``[".png", ...]`` -> ``["*.png", ...]``, one entry per pattern."""
    return [f"*{suffix}" for suffix in suffixes]


def pattern(suffixes: Iterable[str] = IMAGE_SUFFIXES) -> str:
    """The same globs as one space-separated string."""
    return " ".join(globs(suffixes))


def describe(name: str, suffixes: Iterable[str] = IMAGE_SUFFIXES) -> str:
    """``"Images"`` -> ``"Images (*.png *.jpg *.jpeg *.webp *.bmp)"``.

    Derived rather than written beside the patterns, because a label and a
    pattern list maintained separately is exactly how a dialog comes to refuse
    what it accepts.
    """
    return f"{name} ({pattern(suffixes)})"
