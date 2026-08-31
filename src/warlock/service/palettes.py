"""Palette files on disk: what is available, and what one contains.

A palette is a *file the user dropped in a directory*, not a registry entry, so
there is nothing here to add a palette to -- which is the point. Lospec ships
`.hex`, GIMP ships `.gpl`, Paint Shop Pro ships `.pal` and Paint.NET ships
`.txt`, and all four are plain text; ``pipelines/pixel`` parses them and this
module is only the directory half plus the errors a caller can show.

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

#: Every suffix a palette in the directory may wear: Lospec ``.hex``, GIMP
#: ``.gpl``, JASC/Paint Shop Pro ``.pal`` and Paint.NET ``.txt``.
#:
#: Keyed on by both the listing and the load, so a suffix missing here is a
#: file that is not refused -- it never appears at all. That is why this tuple
#: and ``pipelines.pixel.PARSERS`` are asserted against each other rather than
#: kept by hand: a suffix offered here with no reader behind it would list a
#: file the loader then refuses, and a reader with no suffix here is one
#: nothing can reach. ``.pal`` and ``.txt`` joined on 2026-08-30, when the
#: readers landed; before that this comment said why they were absent.
#:
#: Adobe's ``.aco``/``.act`` are still out, and for a different reason: they are
#: binary, this codebase does not write readers for formats it cannot verify
#: against a real file, and nobody has established that the tools these
#: palettes come from write them.
SUFFIXES = (".hex", ".gpl", ".pal", ".txt")

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
