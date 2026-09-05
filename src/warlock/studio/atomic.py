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
import secrets
from collections.abc import Iterator, Mapping
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


def _tmp_name(name: str) -> str:
    """A dotfile temp name unique to this call (M03).

    ``f".{name}.tmp"`` used to be the whole name -- fixed, and shared by every
    concurrent writer of the same destination. Packwright and Plotter both key
    an export task by *tab*, not by destination, so two tabs exporting under
    one basename staged into the same file at once: the later writer's
    ``os.replace`` could land on top of the earlier one's rename, and the
    earlier writer's own cleanup (``tmp.unlink`` in the ``finally``) could then
    delete the *later* writer's still-live staging file out from under it --
    the ``FileNotFoundError`` this was reported as. A token per call is
    ``service/files.py``'s ``_staged_write`` fix, ported here for the same
    reason it was needed there.
    """
    return f".{name}.{secrets.token_hex(4)}.tmp"


@contextlib.contextmanager
def staged(path: Path) -> Iterator[Path]:
    """Yield a staging path beside *path*, renamed onto it on a clean exit.

    The primitive the rest of this module is written in, and the one to reach
    for directly when the writer takes a *path* rather than giving you bytes --
    ``gifout.write_gif`` is the case that motivated exposing it. An exception
    inside the block propagates with the destination untouched.
    """
    path = Path(path)
    tmp = path.with_name(_tmp_name(path.name))
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def staged_set(files: Mapping[Path, bytes]) -> None:
    """Write a set of files that belong together: stage all, then replace each.

    ``plotter_io`` and ``packwright_io`` each carried this loop -- a dotfile
    temporary per target, every file staged before any is replaced, the
    temporaries unlinked in a ``finally`` -- and the second was written by
    porting a fix into the first, which is what a shared leaf exists to stop.
    An encode that raises on the third file leaves the first two untouched.

    What this does not close is the window *between* the replaces: two of
    three can land and the process die before the third. Stated rather than
    papered over -- closing it needs a directory-level transaction no
    filesystem here offers, and staging into a temporary directory and renaming
    that would take the user's chosen name off the file they picked.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for target, blob in files.items():
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(_tmp_name(target.name))
            tmp.write_bytes(blob)
            staged.append((tmp, target))
        for tmp, target in staged:
            os.replace(tmp, target)
    finally:
        for tmp, _target in staged:
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
