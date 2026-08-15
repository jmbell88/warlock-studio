"""GIMP palette files: the one interchange format for a row of swatches.

``.gpl`` is what GIMP, Krita, Inkscape, Aseprite and every palette site on the
internet read and write, which is the whole argument for it over anything
tidier. It is a header line, some optional ``Key: value`` lines, ``#``
comments, and then one ``R G B [name]`` row per colour.

Two decisions are worth stating because they are losses.

**Alpha does not survive.** The format has three channels and no fourth, so an
exported swatch is written opaque and an imported one arrives opaque. A private
extension would round-trip here and be ignored by every other reader, which is
the opposite of the reason to use this format at all.

**The reader is tolerant in the way** ``ora.py`` **is.** A malformed row is
skipped rather than failing the file: a palette that opens missing one colour
is a palette the user still has, and the alternative is a download from an
unknown source taking the whole import with it. A file with no readable rows at
all *is* refused, because "imported nothing" reported as success is worse.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "JASC_HEADER",
    "dumps",
    "dumps_for",
    "dumps_jasc",
    "parse",
    "parse_any",
    "parse_jasc",
]

RGBA = tuple[int, int, int, int]

HEADER = "GIMP Palette"

#: The two lines every JASC ``.pal`` starts with: the magic and a version. The
#: version has been ``0100`` since Paint Shop Pro shipped it and no reader in
#: the wild checks it, so it is written verbatim and not parsed.
JASC_HEADER = "JASC-PAL"
JASC_VERSION = "0100"


def parse(text: str) -> list[RGBA]:
    """Every colour in a ``.gpl``, in file order. Raises on a file with none.

    The header is checked but a missing one is not fatal: plenty of palettes in
    the wild start straight in on the numbers, and a three-integer line is
    unambiguous enough to read without being told.
    """
    out: list[RGBA] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == HEADER:
            continue
        head = stripped.split(None, 3)
        if len(head) < 3:
            continue
        try:
            r, g, b = (int(part) for part in head[:3])
        except ValueError:
            # A "Name:" or "Columns:" line, or a row this reader cannot make
            # sense of. Both are skipped by the same branch on purpose: the
            # distinction is not one the caller can act on.
            continue
        out.append((_byte(r), _byte(g), _byte(b), 255))
    if not out:
        raise ValueError("no colours in this palette file")
    return out


def _byte(value: int) -> int:
    return max(0, min(255, int(value)))


def dumps(colours: Sequence[RGBA], name: str = "Warlock") -> str:
    """A ``.gpl`` for *colours*. Alpha is dropped; see the module docstring.

    ``Columns: 0`` means "let the reader decide", which is the honest answer:
    the swatch row here wraps to whatever the panel is wide, so it has no
    column count to declare.
    """
    lines = [HEADER, f"Name: {name}", "Columns: 0", "#"]
    for colour in colours:
        r, g, b = (_byte(c) for c in tuple(colour)[:3])
        lines.append(f"{r:3d} {g:3d} {b:3d}\t#{r:02x}{g:02x}{b:02x}")
    return "\n".join(lines) + "\n"


# --- JASC .pal ---------------------------------------------------------------
#
# The other format worth reading, and the *only* other one: ``.pal`` is three
# unrelated files under one extension -- JASC's text form, Microsoft's RIFF
# binary form, and a raw 768-byte dump -- and only the first is unambiguous
# enough to read without guessing. Aseprite, Paint Shop Pro, GraphicsGale and
# every pixel-art palette site write the JASC one, and a RIFF ``.pal`` opened
# here is refused with a message rather than read as noise.
#
# It lives beside ``.gpl`` rather than in a module of its own because they are
# the same thing -- a header and a row of integers per colour -- and because the
# alpha loss, the clamping and the skip-a-bad-row tolerance are decisions this
# file has already made and should not make twice.


def parse_jasc(text: str) -> list[RGBA]:
    """Every colour in a JASC ``.pal``. Raises on a file that is not one.

    The magic line *is* checked, unlike ``.gpl``'s: three integers on a line are
    unambiguous on their own, but "``.pal``" is not -- a RIFF or raw palette
    read as text produces plausible-looking garbage, and quietly importing 256
    wrong colours is worse than a refusal.

    The declared count is read past and not enforced. It disagrees with the rows
    in enough files in the wild that trusting it would drop real colours, and
    the rows are the palette.
    """
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0].strip().upper() != JASC_HEADER:
        raise ValueError("not a JASC palette file")
    out: list[RGBA] = []
    for line in lines[1:]:
        if not line or line.startswith(";"):
            continue
        parts = line.split()
        if len(parts) < 3:
            # The version and the count both land here, as does a blank-ish
            # line: one branch, because the distinction is not one the caller
            # can act on -- the same rule ``parse`` follows.
            continue
        try:
            r, g, b = (int(part) for part in parts[:3])
        except ValueError:
            continue
        out.append((_byte(r), _byte(g), _byte(b), 255))
    if not out:
        raise ValueError("no colours in this palette file")
    return out


def dumps_jasc(colours: Sequence[RGBA]) -> str:
    """A JASC ``.pal`` for *colours*, CRLF-terminated. Alpha is dropped.

    CRLF because every file of this format in the wild has them and some readers
    of it are old enough to care; the reader here strips whitespace either way.
    """
    lines = [JASC_HEADER, JASC_VERSION, str(len(colours))]
    for colour in colours:
        r, g, b = (_byte(c) for c in tuple(colour)[:3])
        lines.append(f"{r} {g} {b}")
    return "\r\n".join(lines) + "\r\n"


def parse_any(text: str) -> list[RGBA]:
    """Read a palette in whichever of the two formats it is in.

    Sniffed on the magic line rather than on the suffix, because the suffix is
    what a download got renamed to and the first line is what the file is.
    """
    if text.lstrip().upper().startswith(JASC_HEADER):
        return parse_jasc(text)
    return parse(text)


def dumps_for(suffix: str, colours: Sequence[RGBA], name: str = "Warlock") -> str:
    """Serialise for a filename's suffix; ``.gpl`` for anything unrecognised.

    A default rather than a refusal: the export dialog appends the suffix the
    user picked in the filter, and a typed name with no suffix at all has to
    produce a file.
    """
    return dumps_jasc(colours) if suffix.lower() == ".pal" else dumps(colours, name)
