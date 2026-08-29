"""Palette files on disk: what is available, and what one contains.

A palette is a *file the user dropped in a directory*, not a registry entry, so
there is nothing here to add a palette to -- which is the point. Lospec ships
`.hex`, GIMP ships `.gpl`, and both are plain text; ``pipelines/pixel`` parses
them and this module is only the directory half plus the errors a caller can
show.

Reads happen on whatever thread asked, and every read is a small text file, so
there is no lock and no cache here. The staleness question a derived artifact
asks is answered by the content digest, never by the filename -- editing a
palette in place is the normal way to work on one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..pipelines import pixel
from .errors import Invalid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

SUFFIXES = (".hex", ".gpl")

# ``.pal`` (JASC/Paint Shop Pro) and ``.txt`` (Paint.NET) are real palette
# formats and are deliberately *absent* rather than forgotten:
# ``pipelines.pixel.parse_palette`` has a reader for neither, and this tuple is
# what both the listing and the load are keyed on, so a file with either suffix
# is not refused -- it never appears at all. The picker's helper advertised both
# until 2026-08-29, which is why :data:`SUFFIX_HELP` below is derived from this
# tuple and not written out beside it.
#
# Adding them is a small job rather than a research one: ``studio/inker/gpl.py``
# already parses a JASC ``.pal`` for the Inker's own import, so the honest fix
# may well be a parser in ``pipelines.pixel`` and two more entries here rather
# than a shorter sentence. Until that happens the sentence stays true by
# construction.

#: The one line a palette picker draws under itself, naming exactly what the
#: loader accepts. Derived, so a format added above cannot be one a form forgets
#: to mention -- and a format the loader drops cannot go on being advertised.
SUFFIX_HELP = f"Files in the palette folder: {', '.join(SUFFIXES)}."


def available(config: Config) -> list[str]:
    """Every palette the directory offers, by stem, sorted.

    A missing directory is an empty list and not an error: the whole feature is
    optional, and the pane that lists this draws "none" rather than a failure.
    """
    directory = Path(config.palette_dir)
    if not directory.is_dir():
        return []
    return sorted(
        {p.stem for p in directory.iterdir() if p.suffix.lower() in SUFFIXES}
    )


def _path(config: Config, name: str) -> Path:
    directory = Path(config.palette_dir)
    for suffix in SUFFIXES:
        candidate = directory / f"{name}{suffix}"
        # Resolved and re-checked against the directory: ``name`` reaches here
        # from a request, and "../../etc/hosts" is a palette name as far as
        # string concatenation is concerned.
        if candidate.is_file() and candidate.resolve().parent == directory.resolve():
            return candidate
    choices = ", ".join(available(config)) or "none installed"
    raise Invalid(f"unknown palette {name!r} (available: {choices})", field="palette")


def load(config: Config, name: str) -> tuple[str, tuple[pixel.RGB, ...], str]:
    """``(name, colours, digest)`` for one palette.

    The digest is of the colours rather than of the bytes, so reformatting a
    file does not re-derive every artifact that used it while changing one
    channel does.
    """
    path = _path(config, name)
    try:
        colors = pixel.parse_palette(path.read_text("utf-8"), path.suffix)
    except OSError as exc:
        raise Invalid(f"could not read palette {name!r}: {exc}", field="palette") from exc
    except ValueError as exc:
        # The file is there and unreadable *as a palette* -- which is a fact
        # about its contents, and naming the file is the only thing that makes
        # it fixable.
        raise Invalid(f"{path.name} is not a valid palette: {exc}", field="palette") from exc
    return (name, colors, pixel.palette_digest(colors))
