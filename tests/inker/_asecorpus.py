"""Loading the ``.aseprite`` round-trip corpus.

The shape mirrors ``tests/plotter/_corpus.py``: a fixture directory, a
``MANIFEST`` of the stems the gate requires, and a loader that keeps the gate
test from having to know the directory's layout. There is only one file per
fixture here rather than two -- ``.aseprite`` has no JSON sibling the way a
Tiled map has ``.tmx``/``.tmj`` -- so there is no ``pairs()`` to intersect two
globs with; ``available()`` is that function's whole job, singular.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from warlock.studio.inker.document import Document
from warlock.studio.inker.tiles import strip

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


# --- pinned builders -----------------------------------------------------
#
# Two of the eleven fixtures exist specifically to pin a shape that only the
# document's *funnel* -- not a hand-built array -- ever produces: a fully
# transparent pixel carrying a colour that was never grey (the eraser leaves
# the colour it was drawn in behind it), and a strip pixel at an alpha the
# palette cannot represent as a whole number of visible-or-not (a soft
# eraser/dab in ``auto`` tile behaviour). A fixture file on its own cannot
# prove either shape was ever actually reached -- only the *builder that
# produced it* can, which is why these two are checked in rather than run
# once by a throwaway script the way the other nine were. The gate test
# asserts both: that the shape is present in the builder's own output before
# a single byte is written, and that writing it reproduces the committed
# file exactly -- so a change to the funnel, the writer, or the fixture
# itself that broke the pin would fail here on every run, not just at
# generation time.
#
# Both builders happen to be their *own* fixed point -- ``aseprite_bytes``
# applied once already equals what a second write-read-write trip would
# produce -- unlike ``palette-constrained-rgb`` (see that fixture's
# ``FIXTURES.md`` entry), so the committed bytes are the builder's raw
# output, not a post-normalisation second pass.

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)


def _tile(colour: tuple[int, int, int, int], w: int = 4, h: int = 4) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _tileset(*colours: tuple[int, int, int, int], name: str = "tiles"):
    """A vertical strip whose local id 0 is the required blank tile."""
    blank = np.zeros((4, 4, 4), dtype=np.uint8)
    built = strip(np.stack([blank, *[_tile(c) for c in colours]], axis=0))
    return replace(built, name=name)


def build_grayscale_animated() -> Document:
    """Two frames, each independently painted blue and then erased, *before*
    the document is converted to grayscale.

    ``convert_to_grayscale`` validates and luma-converts what is *visible*
    but leaves an already-invisible pixel's stored colour exactly as the
    eraser left it (``test_aseout.py``'s
    ``test_erasing_before_converting_to_grayscale_still_saves`` is the
    still-document proof of the same rule) -- so each frame needs its own
    paint-then-erase, not a shared one, or the second frame's cel would come
    from ``add_frame``'s blank default and never carry the shape at all.
    """
    doc = Document.blank(8, 8)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = BLUE
    doc.invalidate_all()
    doc.ensure_animation()
    doc.set_active_layer(0)
    doc.begin_stroke((0, 0), BLUE, size=4, hardness=1.0, mode="erase")
    doc.stroke_to((7, 7))
    doc.end_stroke()
    doc.add_frame()
    doc.set_current_frame(1)
    doc.set_active_layer(0)
    doc.write_colour((0, 0, 8, 8), BLUE, np.ones((8, 8), dtype=np.float32))
    doc.begin_stroke((7, 0), BLUE, size=3, hardness=1.0, mode="erase")
    doc.stroke_to((0, 7))
    doc.end_stroke()
    doc.set_current_frame(0)
    doc.convert_to_grayscale()
    return doc


def build_tilemap_indexed() -> Document:
    """A half-coverage dab onto an ``auto``-behaviour tilemap cel, which reaches
    the tileset strip with a soft, non-``{0, 255}`` alpha pixel -- the same
    construct ``test_aseout.py``'s ``test_a_half_coverage_dab_on_a_tile_still_saves``
    exercises against a document built fresh each run."""
    doc = Document.blank(8, 8)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([(0, 0, 0, 0), RED, GREEN], transparent=0)
    slot = doc.add_tileset(_tileset(RED, GREEN))
    cel = doc.add_tilemap_layer(slot.uid, name="Tiles")
    doc.place_tiles(cel.uid, (0, 0), np.array([[1, 2]], dtype=np.uint32))
    doc.set_active_layer(doc.stack.index_of(cel.uid))
    doc.tile_behavior = "auto"
    doc.write_colour((0, 4, 4, 8), GREEN, np.full((4, 4), 0.5, dtype=np.float32))
    return doc


#: The fixtures a builder pins, by stem -- the ones ``test_aseprite_corpus.py``
#: additionally checks for the pre-write construct and for
#: ``aseprite_bytes(builder()) == read(stem)``.
BUILDERS = {
    "grayscale-animated": build_grayscale_animated,
    "tilemap-indexed": build_tilemap_indexed,
}
