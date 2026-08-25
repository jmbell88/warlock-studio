"""Writing onto a path that may already hold a good file.

``inker_mode._write_atomic``, promoted. It was factored out for three writers
in one module and the argument it was factored out *for* is not about Inker at
all: there are twenty-two ``dialogs.save_file`` sites in ``studio/``, every one
of them writes to a destination **the user picked**, and a destination the user
picked is one they may well have picked before. ``Path.write_bytes`` truncates
before it writes a byte, so a crash, a full disk or a yanked drive halfway
through an export destroys the file that was there and leaves nothing in its
place -- and the file that was there is, by construction, one the user cared
enough about to name twice.

So: stage beside the destination, then rename. ``os.replace`` is atomic within
one filesystem, and the staging name is a dotfile *sibling* precisely so the
two share one -- a temp in ``%TEMP%`` would make the rename a copy across
volumes and give back the window this exists to close. On failure the staging
file is unlinked in a ``finally``, ``service/derive.py``'s ``_staged`` rule:
nothing sweeps a dotfile, so one left behind is left behind for good.

**A leaf.** Stdlib only at module scope -- Pillow is imported inside
:func:`save_image` -- and it imports nothing from ``studio``. Half the writers
that need it are panes and half are modes, and a helper that lived in either
would be a helper the other one imported through a mode.

The rule is held by a scan test (``tests/test_atomic_writes.py``): a function
that opens a save dialog may not then write to the path it was given by any
other route.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: Suffix -> the Pillow format name to save under.
#:
#: Needed because the staging file is called ``.thing.png.tmp``, and Pillow
#: infers the format from the *extension* of the name it is handed. Saving a
#: PNG to a ``.tmp`` with no explicit format is an ``unknown file extension``
#: error, so :func:`save_image` resolves the format from the real destination
#: before the rename ever happens. Built from Pillow's own registry at call
#: time rather than hand-listed here; this maps only the two we spell by hand
#: often enough to be worth not paying an import for.
_FORMATS = {".png": "PNG", ".gif": "GIF"}


@contextlib.contextmanager
def staged(path: Path) -> Iterator[Path]:
    """Yield a staging path beside *path*, renamed onto it on a clean exit.

    The primitive the rest of this module is written in, and the one to reach
    for directly when the writer takes a *path* rather than giving you bytes --
    ``gifout.write_gif`` is the case that motivated exposing it. An exception
    inside the block propagates with the destination untouched.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def write_bytes(path: Path, data: bytes) -> None:
    """``Path.write_bytes``, staged."""
    with staged(path) as tmp:
        tmp.write_bytes(data)


def write_text(path: Path, text: str, *, encoding: str = "utf-8", newline: Any = None) -> None:
    """``Path.write_text``, staged.

    ``newline`` is passed through rather than defaulted away because one caller
    needs ``""`` and needs it for a reason worth keeping: Python's text mode
    rewrites every ``\\n`` as ``os.linesep``, which turns a JASC palette's
    already-correct CRLF into CR CR LF. See ``inker_mode._write_palette``.
    """
    with staged(path) as tmp:
        tmp.write_text(text, encoding=encoding, newline=newline)


def save_image(path: Path, image: Any, format: str | None = None, **kwargs: Any) -> None:
    """``Image.save``, staged.

    The format is resolved from *path*'s real suffix and passed explicitly,
    because the file Pillow is actually handed is called ``.name.tmp`` and it
    would otherwise be asked to infer an encoder from ``.tmp``. A suffix
    Pillow's registry does not know raises here, before anything is written,
    rather than at the ``save``.
    """
    path = Path(path)
    if format is None:
        suffix = path.suffix.lower()
        format = _FORMATS.get(suffix)
        if format is None:
            from PIL import Image

            Image.init()
            format = Image.registered_extensions().get(suffix)
        if format is None:
            raise ValueError(f"no image format for {path.name!r}")
    with staged(path) as tmp:
        image.save(tmp, format, **kwargs)
