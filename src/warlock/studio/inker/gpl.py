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

__all__ = ["dumps", "parse"]

RGBA = tuple[int, int, int, int]

HEADER = "GIMP Palette"


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
