"""Loading the ``.aseprite`` round-trip corpus.

The shape mirrors ``tests/plotter/_corpus.py``: a fixture directory, a
``MANIFEST`` of the stems the gate requires, and a loader that keeps the gate
test from having to know the directory's layout. There is only one file per
fixture here rather than two -- ``.aseprite`` has no JSON sibling the way a
Tiled map has ``.tmx``/``.tmj`` -- so there is no ``pairs()`` to intersect two
globs with; ``available()`` is that function's whole job, singular.
"""

from __future__ import annotations

from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "aseprite"

# The fixture stems the corpus gate requires. Adding a file means adding its
# stem here in the same commit, or nothing tests it -- the ``_corpus.py``
# rule, restated.
MANIFEST: tuple[str, ...] = (
    "rgb-still",
    "rgb-animated-linked-tags",
    "grayscale-animated",
    "indexed-duplicate-colours",
    "indexed-transparent-nonzero",
    "groups-nested",
    "slices-pivot-ninepatch",
    "tilemap-rgb",
    "tilemap-indexed",
    "spare-tileset",
    "palette-constrained-rgb",
)

#: Fixtures whose *committed* bytes deliberately embody a documented loss and
#: so are expected to trip a reader warning on every read. Empty today: the
#: one candidate, ``palette-constrained-rgb``, loses its palette silently (no
#: warning fires -- see ``asein.document_from_aseprite``, which only installs
#: a palette at indexed depth and says nothing when it declines to at RGB
#: depth). Kept as a named exception point rather than assumed permanently
#: empty, so a fixture that does need one has somewhere to say so instead of
#: the gate quietly loosening for everybody.
EXPECTED_WARNINGS: dict[str, tuple[str, ...]] = {}


def available() -> list[str]:
    """Stems actually present in the fixture directory, sorted."""
    if not FIXTURE_DIR.is_dir():
        return []
    return sorted(path.stem for path in FIXTURE_DIR.glob("*.aseprite"))


def read(stem: str) -> bytes:
    """One fixture's bytes, by stem."""
    return (FIXTURE_DIR / f"{stem}.aseprite").read_bytes()
