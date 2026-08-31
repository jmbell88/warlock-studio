"""Palette files: a row of swatches, in the four formats worth reading.

``.gpl`` is the one that matters most -- GIMP, Krita, Inkscape, Aseprite and
every palette site on the internet read and write it, which is the whole
argument for it over anything tidier. It is a header line, some optional
``Key: value`` lines, ``#`` comments, and then one ``R G B [name]`` row per
colour. The other three are here because a palette a user already owns arrives
in whichever one its author used: JASC ``.pal``, Lospec ``.hex`` and Paint.NET
``.txt``.

Two decisions are worth stating because they are losses.

**Alpha does not survive three of the four.** ``.gpl``, ``.pal`` and ``.hex``
have three channels and no fourth, so a swatch exported to one is written
opaque and one imported from it arrives opaque. A private extension would
round-trip here and be ignored by every other reader, which is the opposite of
the reason to use these formats at all. Paint.NET's ``.txt`` is the exception:
it has a real alpha channel and this module keeps it.

**The reader is tolerant in the way** ``ora.py`` **is.** A malformed row is
skipped rather than failing the file: a palette that opens missing one colour
is a palette the user still has, and the alternative is a download from an
unknown source taking the whole import with it. A file with no readable rows at
all *is* refused, because "imported nothing" reported as success is worse.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = [
    "JASC_HEADER",
    "dumps",
    "dumps_for",
    "dumps_hex",
    "dumps_jasc",
    "dumps_txt",
    "parse",
    "parse_any",
    "parse_hex",
    "parse_jasc",
    "parse_txt",
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
# ``.pal`` is three unrelated files under one extension -- JASC's text form,
# Microsoft's RIFF binary form, and a raw 768-byte dump -- and only the first is
# unambiguous enough to read without guessing. Aseprite, Paint Shop Pro,
# GraphicsGale and every pixel-art palette site write the JASC one,
# and a RIFF ``.pal`` opened
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
    # A UTF-8 BOM survives ``errors="replace"`` decoding as U+FEFF, which is
    # not whitespace to ``strip`` -- left in place it hides the magic line and
    # a legitimate file is refused as "not a JASC palette".
    lines = [line.strip() for line in text.lstrip("﻿").splitlines()]
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


# --- the hex-digit pair: Lospec .hex and Paint.NET .txt -----------------------
#
# Both are a bare column of hex digits with no magic line at all, which is why
# they are read together: what separates them is the *width* of a row. Six
# digits is Lospec's ``rrggbb``; eight is Paint.NET's ``aarrggbb``, alpha
# first. Nothing else about either file says which it is, and a six-digit row
# means the same colour under either reading, so the width is enough.
#
# ``pipelines/pixel.py`` has its own copy of both readers, and of
# :func:`parse_jasc`, because the layering forbids it importing this package
# and forbids this package importing it. ``tests/inker/test_palette_formats.py``
# feeds the same fixture bytes to both and asserts the same colours come back,
# so the two cannot drift apart quietly -- which is the only thing that makes
# the duplication survivable.

#: One Lospec row: six hex digits, optionally introduced by a ``#``.
_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")

#: One Paint.NET row: eight hex digits, ``aarrggbb``, alpha first.
_ARGB_RE = re.compile(r"^#?([0-9a-fA-F]{8})$")


def _is_comment(line: str) -> bool:
    """Whether a hex-column line is a comment rather than a colour.

    ``;`` is Paint.NET's own comment marker, ``//`` and ``#!`` are what the
    palette sites put their name and URL behind. A bare ``#`` prefix is *not* a
    comment here: it is how half the world writes a colour.
    """
    return line.startswith((";", "//")) or line.startswith("#!")


def parse_hex(text: str) -> list[RGBA]:
    """A Lospec ``.hex``: one ``rrggbb`` per line. Raises on a file with none.

    Ported from ``pipelines.pixel.parse_hex`` line for line, including the
    strictness: blank lines and comments are skipped, and a line that looks
    like it is *trying* to be a colour and is not raises rather than being
    dropped. That is the opposite of :func:`parse`'s tolerance and it is the
    right rule here -- a ``.gpl`` row carries a name the reader can fail to
    understand, and a hex row carries nothing but the colour, so a row this
    cannot read is a file it has misunderstood.
    """
    out: list[RGBA] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _is_comment(line):
            continue
        match = _HEX_RE.match(line)
        if match is None:
            raise ValueError(f"not a hex colour: {line!r}")
        value = match.group(1)
        out.append(
            (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)
        )
    if not out:
        raise ValueError("no colours in this palette file")
    return out


def dumps_hex(colours: Sequence[RGBA]) -> str:
    """A Lospec ``.hex`` for *colours*. Alpha is dropped; nothing else is here.

    No header and no comment line: the format is the column, every reader of
    one takes it bare, and a comment is the one thing a round trip through a
    stricter reader than ours could trip over.
    """
    return "".join(
        "{:02x}{:02x}{:02x}\n".format(*(_byte(c) for c in tuple(colour)[:3]))
        for colour in colours
    )


def parse_txt(text: str) -> list[RGBA]:
    """A Paint.NET ``.txt``: one ``aarrggbb`` per line. Raises on a file with none.

    **Alpha survives this one.** It is the only palette format here with a
    fourth channel, and dropping it to match the other three would be throwing
    away information the file actually carries; a six-digit row -- which some
    writers emit -- is opaque, as it is everywhere else.
    """
    out: list[RGBA] = []
    for raw in text.lstrip("﻿").splitlines():
        line = raw.strip()
        if not line or _is_comment(line):
            continue
        wide = _ARGB_RE.match(line)
        if wide is not None:
            value = wide.group(1)
            out.append(
                (
                    int(value[2:4], 16),
                    int(value[4:6], 16),
                    int(value[6:8], 16),
                    int(value[0:2], 16),
                )
            )
            continue
        narrow = _HEX_RE.match(line)
        if narrow is None:
            raise ValueError(f"not a hex colour: {line!r}")
        value = narrow.group(1)
        out.append(
            (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)
        )
    if not out:
        raise ValueError("no colours in this palette file")
    return out


#: What Paint.NET itself writes at the top of one, kept because its own reader
#: is happy without it and every human opening the file is not.
TXT_HEADER = (
    "; Paint.NET Palette File\n"
    "; Lines that start with a semicolon are comments\n"
    "; Colours are written as 8-digit hex numbers: aarrggbb\n"
)


def dumps_txt(colours: Sequence[RGBA]) -> str:
    """A Paint.NET ``.txt`` for *colours*, alpha and all."""
    rows = "".join(
        "{:02x}{:02x}{:02x}{:02x}\n".format(
            _byte(tuple(colour)[3] if len(tuple(colour)) > 3 else 255),
            *(_byte(c) for c in tuple(colour)[:3]),
        )
        for colour in colours
    )
    return TXT_HEADER + rows


def _hex_column(text: str) -> str:
    """``"hex"``, ``"txt"`` or ``""`` for text that is not a column of hex.

    A file is the eight-digit format the moment one row is eight digits wide:
    six-digit rows read the same either way, so a mixed file loses nothing by
    being read as the wider one, and a file with no eight-digit row at all has
    nothing to gain from it.
    """
    seen = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _is_comment(line):
            continue
        if _ARGB_RE.match(line) is not None:
            return "txt"
        if _HEX_RE.match(line) is None:
            return ""
        seen = "hex"
    return seen


def parse_any(text: str) -> list[RGBA]:
    """Read a palette in whichever of the four formats it is in.

    Sniffed on the *content* rather than on the suffix, because the suffix is
    what a download got renamed to and the bytes are what the file is. Only
    JASC has a magic line to sniff; the two hex columns are told apart from
    ``.gpl`` -- and from each other -- by the shape of every row in them, which
    is the only thing there is to go on and is unambiguous: a ``.gpl`` row is
    decimal triples or a ``Key: value`` line, and neither is six hex digits on
    a line of its own.
    """
    # The BOM before the whitespace: ``lstrip()`` strips whitespace and U+FEFF
    # is not one, so a BOM'd JASC file failed the sniff and fell into the
    # ``.gpl`` reader, which refuses it as "no colours".
    if text.lstrip("﻿").lstrip().upper().startswith(JASC_HEADER):
        return parse_jasc(text)
    column = _hex_column(text.lstrip("﻿"))
    if column == "txt":
        return parse_txt(text)
    if column == "hex":
        return parse_hex(text)
    return parse(text)


#: Which serialiser each suffix asks for. ``.gpl`` is not in it because it is
#: the default -- see :func:`dumps_for`.
_WRITERS = {
    ".pal": lambda colours, name: dumps_jasc(colours),
    ".hex": lambda colours, name: dumps_hex(colours),
    ".txt": lambda colours, name: dumps_txt(colours),
}


def dumps_for(suffix: str, colours: Sequence[RGBA], name: str = "Warlock") -> str:
    """Serialise for a filename's suffix; ``.gpl`` for anything unrecognised.

    A default rather than a refusal: the export dialog appends the suffix the
    user picked in the filter, and a typed name with no suffix at all has to
    produce a file.
    """
    writer = _WRITERS.get(suffix.lower())
    return writer(colours, name) if writer is not None else dumps(colours, name)
