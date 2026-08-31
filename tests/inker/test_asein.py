"""Reading somebody else's ``.aseprite``, built byte by byte in this file.

Every fixture here is ``struct.pack``ed rather than checked in, the rule
``tests/plotter`` set: a binary blob in the tree is a fixture nobody can read a
diff of, and one that has to be *fetched* is a test that fails on a machine with
no network. Building them here means each test says, in its own body, exactly
which chunk it is about -- and a chunk this reader refuses can be written down
in three lines, which is what makes one named test per refusal affordable.

The builders below are the format, and only as much of it as these tests need:
a 128-byte header, frames of length-prefixed chunks, and one function per chunk
type. They are deliberately *not* a writer -- nothing here rounds a document
back out to Aseprite, because this import is read-only.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from warlock.studio.inker import asein
from warlock.studio.inker.tiles import TilemapCel
from warlock.studio.tilegrid import gid

MAGIC = 0xA5E0
FRAME_MAGIC = 0xF1FA

RGBA_DEPTH = 32
GRAY_DEPTH = 16
INDEXED_DEPTH = 8


# --- the format, as builders -------------------------------------------------


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _chunk(kind: int, payload: bytes) -> bytes:
    return struct.pack("<IH", len(payload) + 6, kind) + payload


def _frame(chunks: list[bytes], duration: int = 100, *, new: bool = False) -> bytes:
    """One frame header and its chunks.

    ``new`` picks which of the two chunk-count fields carries the number. The
    format has both -- a legacy WORD and a DWORD added when a frame could hold
    more than 65535 chunks -- under the rule "0xFFFF in the old one means read
    the new one, and 0 in the new one means read the old one". A real Aseprite
    1.3 file writes the DWORD, so a fixture that only ever fills the WORD tests
    the *fallback* branch alone: a two-byte misread of the modern field would
    land on the two padding bytes, come back 0, fall through to the legacy
    count and pass every test in this file. Both shapes are built for exactly
    that reason.
    """
    body = b"".join(chunks)
    old_count = 0xFFFF if new else len(chunks)
    new_count = len(chunks) if new else 0
    return (
        struct.pack(
            "<IHHHHI", len(body) + 16, FRAME_MAGIC, old_count, duration, 0, new_count
        )
        + body
    )


def _header(
    frames: int,
    width: int,
    height: int,
    depth: int = RGBA_DEPTH,
    *,
    magic: int = MAGIC,
    transparent: int = 0,
    flags: int = 1,
    colours: int = 0,
) -> bytes:
    head = struct.pack(
        "<IHHHHHIHIIB3sHBBhhHH",
        0,
        magic,
        frames,
        width,
        height,
        depth,
        flags,
        100,
        0,
        0,
        transparent,
        b"\0\0\0",
        colours,
        1,
        1,
        0,
        0,
        0,
        0,
    )
    return head + b"\0" * 84


def _file(header: bytes, frames: list[bytes]) -> bytes:
    body = header + b"".join(frames)
    return struct.pack("<I", len(body)) + body[4:]


def _layer(
    name: str = "Layer",
    *,
    kind: int = 0,
    child: int = 0,
    blend: int = 0,
    opacity: int = 255,
    visible: bool = True,
    editable: bool = True,
    background: bool = False,
    reference: bool = False,
    continuous: bool = False,
    tileset: int | None = None,
) -> bytes:
    flags = (
        (1 if visible else 0)
        | (2 if editable else 0)
        | (8 if background else 0)
        | (16 if continuous else 0)
        | (64 if reference else 0)
    )
    body = (
        struct.pack("<HHHHHHB3s", flags, kind, child, 0, 0, blend, opacity, b"\0\0\0")
        + _string(name)
    )
    if tileset is not None:
        # The one field a tilemap layer (kind 2) carries past its name: the
        # tileset chunk id it draws through.
        body += struct.pack("<I", tileset)
    return _chunk(0x2004, body)


def _tileset_chunk(
    tileset_id: int,
    tile_w: int,
    tile_h: int,
    tiles: list[bytes],
    *,
    name: str = "tiles",
    flags: int = 2,
    base_index: int = 1,
) -> bytes:
    """The 0x2023 chunk: ``tiles`` is a list of already-flattened RGBA byte
    strings, one per tile, stacked vertically in local-id order (tile 0
    first) exactly as this reader's own strip layout expects."""
    body = (
        struct.pack("<IIIHHh", tileset_id, flags, len(tiles), tile_w, tile_h, base_index)
        + b"\0" * 14
        + _string(name)
    )
    if flags & 2:
        raw = b"".join(tiles)
        compressed = zlib.compress(raw)
        body += struct.pack("<I", len(compressed)) + compressed
    return _chunk(0x2023, body)


def _tilemap_cel(
    layer: int,
    refs: np.ndarray,
    *,
    x: int = 0,
    y: int = 0,
    bits: int = 32,
    id_mask: int = 0x1FFFFFFF,
    x_mask: int = 0x80000000,
    y_mask: int = 0x40000000,
    d_mask: int = 0x20000000,
) -> bytes:
    grid_h, grid_w = refs.shape
    payload = refs.astype("<u4").tobytes()
    compressed = zlib.compress(payload)
    body = (
        struct.pack("<HhhBHh5s", layer, x, y, 255, 3, 0, b"\0" * 5)
        + struct.pack("<HHH", grid_w, grid_h, bits)
        + struct.pack("<IIII", id_mask, x_mask, y_mask, d_mask)
        + b"\0" * 10
        + compressed
    )
    return _chunk(0x2005, body)


def _cel(
    layer: int,
    pixels: bytes,
    width: int,
    height: int,
    *,
    x: int = 0,
    y: int = 0,
    opacity: int = 255,
    compressed: bool = False,
    z: int = 0,
) -> bytes:
    kind = 2 if compressed else 0
    data = zlib.compress(pixels) if compressed else pixels
    return _chunk(
        0x2005,
        struct.pack("<HhhBHh5s", layer, x, y, opacity, kind, z, b"\0" * 5)
        + struct.pack("<HH", width, height)
        + data,
    )


def _linked_cel(
    layer: int, frame: int, *, x: int = 0, y: int = 0, opacity: int = 255
) -> bytes:
    return _chunk(
        0x2005,
        struct.pack("<HhhBHh5s", layer, x, y, opacity, 1, 0, b"\0" * 5)
        + struct.pack("<H", frame),
    )


def _raw_cel(layer: int, kind: int, tail: bytes = b"") -> bytes:
    """A cel of an arbitrary type, for the refusals."""
    return _chunk(
        0x2005, struct.pack("<HhhBHh5s", layer, 0, 0, 255, kind, 0, b"\0" * 5) + tail
    )


def _tags(entries: list[tuple[str, int, int, int, int]]) -> bytes:
    body = struct.pack("<H8s", len(entries), b"\0" * 8)
    for name, start, end, direction, repeat in entries:
        body += (
            struct.pack("<HHBH6s3sB", start, end, direction, repeat, b"\0" * 6, b"\0\0\0", 0)
            + _string(name)
        )
    return _chunk(0x2018, body)


def _palette(colours: list[tuple[int, int, int, int]]) -> bytes:
    body = struct.pack("<III8s", len(colours), 0, len(colours) - 1, b"\0" * 8)
    for red, green, blue, alpha in colours:
        body += struct.pack("<HBBBB", 0, red, green, blue, alpha)
    return _chunk(0x2019, body)


def _old_palette(colours: list[tuple[int, int, int]], *, six_bit: bool = False) -> bytes:
    body = struct.pack("<HBB", 1, 0, len(colours) % 256)
    for red, green, blue in colours:
        body += struct.pack("<BBB", red, green, blue)
    return _chunk(0x0011 if six_bit else 0x0004, body)


def _slice(
    name: str,
    keys: list[tuple[int, tuple[int, int, int, int]]],
    *,
    centre: tuple[int, int, int, int] | None = None,
    pivot: tuple[int, int] | None = None,
) -> bytes:
    flags = (1 if centre is not None else 0) | (2 if pivot is not None else 0)
    body = struct.pack("<III", len(keys), flags, 0) + _string(name)
    for frame, (x, y, width, height) in keys:
        body += struct.pack("<IiiII", frame, x, y, width, height)
        if centre is not None:
            body += struct.pack("<iiII", *centre)
        if pivot is not None:
            body += struct.pack("<ii", *pivot)
    return _chunk(0x2022, body)


def _rgba(width: int, height: int, colour: tuple[int, int, int, int]) -> bytes:
    return bytes(colour) * (width * height)


def _one_layer_file(
    *, cel: bytes | None = None, extra: list[bytes] | None = None, size: int = 2
) -> bytes:
    """The smallest useful document: one layer, one frame, one flat cel."""
    pixels = _rgba(size, size, (10, 20, 30, 255)) if cel is None else b""
    chunks = [_layer("Art")]
    chunks.append(cel if cel is not None else _cel(0, pixels, size, size))
    chunks.extend(extra or [])
    return _file(_header(1, size, size), [_frame(chunks)])


# --- what opens --------------------------------------------------------------


def test_a_single_frame_file_opens_as_a_still_document():
    doc, warnings = asein.document_from_aseprite(_one_layer_file())
    assert doc.anim is None
    assert doc.size == (2, 2)
    assert doc.stack[0].name == "Art"
    assert tuple(doc.stack[0].pixels[0, 0]) == (10, 20, 30, 255)
    assert warnings == []


def test_an_imported_document_owns_no_file():
    """Read-only, structurally: the reader cannot write the format it read, so
    a path would let one Ctrl+S put ORA bytes over the source."""
    doc, _ = asein.document_from_aseprite(_one_layer_file())
    assert doc.path is None
    assert doc.file_format == "ora"
    assert doc.history.head == 0


def test_a_frame_counts_its_chunks_in_either_field():
    """The modern DWORD and the legacy WORD, asserted to give one document.

    A real Aseprite 1.3 file fills the DWORD and puts 0xFFFF in the WORD, and
    every other fixture in this file does the opposite -- so without this one,
    the modern read is never exercised at all and a two-byte misread of it
    would fall through to the legacy count and go unnoticed.
    """
    art = _rgba(2, 2, (7, 7, 7, 255))

    def build(new):
        return _file(
            _header(2, 2, 2),
            [
                _frame([_layer("Art"), _cel(0, art, 2, 2)], duration=40, new=new),
                _frame([_cel(0, _rgba(2, 2, (8, 8, 8, 255)), 2, 2)], new=new),
            ],
        )

    legacy = asein.parse(build(False))
    modern = asein.parse(build(True))
    assert legacy == modern
    assert [layer.name for layer in modern.layers] == ["Art"]
    assert modern.durations == [40, 100]
    assert len(modern.cels) == 2

    doc, warnings = asein.document_from_aseprite(build(True))
    assert doc.anim is not None
    assert len(doc.anim.tracks) == 1
    assert warnings == []


def test_a_compressed_cel_reads_the_same_as_a_raw_one():
    raw, _ = asein.document_from_aseprite(_one_layer_file())
    packed, _ = asein.document_from_aseprite(
        _one_layer_file(cel=_cel(0, _rgba(2, 2, (10, 20, 30, 255)), 2, 2, compressed=True))
    )
    assert np.array_equal(raw.stack[0].pixels, packed.stack[0].pixels)


def test_a_cel_is_placed_at_its_own_offset_on_a_canvas_sized_plane():
    data = _file(
        _header(1, 4, 4),
        [_frame([_layer("Art"), _cel(0, _rgba(2, 2, (255, 0, 0, 255)), 2, 2, x=1, y=2)])],
    )
    doc, warnings = asein.document_from_aseprite(data)
    pixels = doc.stack[0].pixels
    assert tuple(pixels[2, 1]) == (255, 0, 0, 255)
    assert tuple(pixels[0, 0]) == (0, 0, 0, 0)
    assert warnings == []


def test_a_multi_frame_file_opens_as_an_animation_with_its_durations():
    art = _rgba(2, 2, (1, 2, 3, 255))
    data = _file(
        _header(2, 2, 2),
        [
            _frame([_layer("Art"), _cel(0, art, 2, 2)], duration=40),
            _frame([_cel(0, _rgba(2, 2, (4, 5, 6, 255)), 2, 2)], duration=250),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    assert doc.anim is not None
    assert [f.duration_ms for f in doc.anim.frames] == [40, 250]
    assert len(doc.anim.tracks) == 1


def test_a_linked_cel_arrives_as_one_object_in_two_slots():
    """The mapping that makes this import worth more than exported PNGs: a link
    in the file is a link in the document, so an edit shows on both frames."""
    art = _rgba(2, 2, (9, 9, 9, 255))
    data = _file(
        _header(3, 2, 2),
        [
            _frame([_layer("Art"), _cel(0, art, 2, 2)]),
            _frame([_linked_cel(0, 0)]),
            _frame([_linked_cel(0, 1)]),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    anim = doc.anim
    track, frames = anim.tracks[0].uid, anim.frames
    first = anim.cels[(track, frames[0].uid)]
    assert anim.cels[(track, frames[1].uid)] is first
    # Through a chain: frame 2 links to frame 1, which links to frame 0.
    assert anim.cels[(track, frames[2].uid)] is first
    assert anim.is_linked(track, frames[2].uid)


def test_layer_properties_come_across():
    data = _file(
        _header(1, 2, 2),
        [
            _frame(
                [
                    _layer("Under", opacity=128, visible=False, editable=False),
                    _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2),
                    _layer("Over", blend=1),
                    _cel(1, _rgba(2, 2, (2, 2, 2, 255)), 2, 2),
                ]
            )
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    under, over = doc.stack[0], doc.stack[1]
    assert (under.name, under.visible, under.locked) == ("Under", False, True)
    assert under.opacity == pytest.approx(128 / 255)
    assert (over.name, over.blend) == ("Over", "multiply")


@pytest.mark.parametrize(
    ("index", "mode"),
    list(enumerate(asein._BLEND_BY_INDEX)),
)
def test_every_aseprite_blend_mode_has_one_of_ours(index, mode):
    """Nineteen modes each way, since C6 added the seven that were missing --
    so no file loses a mode on the way in and none is approximated."""
    from warlock.studio.inker import composite as cp

    assert mode in cp.BLEND_MODES
    doc, warnings = asein.document_from_aseprite(
        _file(
            _header(1, 2, 2),
            [_frame([_layer("Art", blend=index), _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2)])],
        )
    )
    assert doc.stack[0].blend == mode
    assert warnings == []


def test_tags_arrive_with_their_spans_directions_and_repeats():
    art = _rgba(1, 1, (1, 1, 1, 255))
    data = _file(
        _header(4, 1, 1),
        [
            _frame(
                [
                    _layer("Art"),
                    _cel(0, art, 1, 1),
                    _tags([("walk", 0, 1, 0, 0), ("swing", 2, 3, 2, 3)]),
                ]
            ),
            _frame([_cel(0, art, 1, 1)]),
            _frame([_cel(0, art, 1, 1)]),
            _frame([_cel(0, art, 1, 1)]),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    walk, swing = doc.anim.tags
    assert (walk.name, walk.start, walk.end, walk.direction) == ("walk", 0, 1, "forward")
    # Aseprite's zero repeat is "forever", which is this model's zero as well.
    assert (walk.repeat, walk.loop) == (0, True)
    # A finite repeat needs ``loop`` True beside it or it stops on its first
    # pass -- the C7 mapping stated in the animation model.
    assert (swing.direction, swing.repeat, swing.loop) == ("pingpong", 3, True)


def test_an_indexed_file_carries_its_palette_and_its_transparent_index():
    colours = [(0, 0, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255)]
    data = _file(
        _header(1, 2, 1, INDEXED_DEPTH, transparent=0),
        [_frame([_layer("Art"), _palette(colours), _cel(0, bytes([0, 1]), 2, 1)])],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.palette == colours
    # Truly indexed: the file's index planes are kept as the record rather than
    # flattened through the table, so the slots the file names survive.
    assert doc.is_indexed and not doc.is_palette_locked
    assert doc.stack[0].indices.tolist() == [[0, 1]]
    doc.check_materialized()
    # Index 0 is the header's transparent index, so it reads as nothing.
    assert tuple(doc.stack[0].pixels[0, 0]) == (0, 0, 0, 0)
    assert tuple(doc.stack[0].pixels[0, 1]) == (255, 0, 0, 255)
    assert warnings == []


def test_the_transparent_index_is_a_colour_on_a_background_layer():
    colours = [(7, 8, 9, 255), (255, 0, 0, 255)]
    data = _file(
        _header(1, 1, 1, INDEXED_DEPTH, transparent=0),
        [
            _frame(
                [
                    _layer("Background", background=True),
                    _palette(colours),
                    _cel(0, bytes([0]), 1, 1),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert tuple(doc.stack[0].pixels[0, 0]) == (7, 8, 9, 255)
    # And the picture is kept by the feature paying for itself. Aseprite renders
    # the transparent index two ways -- as its colour on a background layer, as
    # nothing everywhere else -- and one materialisation cannot do both. So the
    # slot is *duplicated*: a new entry with the same colour is appended and the
    # background's pixels re-pointed at it, which only an index plane makes
    # possible. The document stays consistently indexed and the user sees a real
    # swatch they can recolour.
    assert doc.is_indexed
    assert doc.palette == [*colours, (7, 8, 9, 255)]
    assert doc.stack[0].indices.tolist() == [[2]]
    assert doc.transparent_index == 0
    assert warnings == []
    doc.check_materialized()


def test_a_duplicate_swatch_survives_the_import_as_two_slots():
    """The thing an RGBA round trip cannot promise, and the reason this reader
    stopped flattening: slots 1 and 3 are the same red, and after the import the
    two pixels painted in them are still distinguishable."""
    colours = [(0, 0, 0, 255), (200, 20, 20, 255), (20, 200, 20, 255), (200, 20, 20, 255)]
    data = _file(
        _header(1, 4, 1, INDEXED_DEPTH, transparent=0),
        [_frame([_layer("Art"), _palette(colours), _cel(0, bytes([0, 1, 2, 3]), 4, 1)])],
    )
    doc, _ = asein.document_from_aseprite(data)

    assert doc.stack[0].indices.tolist() == [[0, 1, 2, 3]]
    assert tuple(doc.stack[0].pixels[0, 1]) == tuple(doc.stack[0].pixels[0, 3])
    doc.recolour_slot(3, (5, 5, 250, 255))
    assert tuple(doc.stack[0].pixels[0, 1]) == (200, 20, 20, 255)
    assert tuple(doc.stack[0].pixels[0, 3]) == (5, 5, 250, 255)


def test_an_indexed_linked_cel_shares_one_index_plane():
    colours = [(0, 0, 0, 255), (255, 0, 0, 255)]
    data = _file(
        _header(2, 2, 1, INDEXED_DEPTH, transparent=0),
        [
            _frame([_layer("Art"), _palette(colours), _cel(0, bytes([0, 1]), 2, 1)]),
            _frame([_linked_cel(0, 0)]),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    cels = list(doc.anim.unique_cel_layers())
    assert len(cels) == 1 and cels[0].indices is not None
    doc.check_materialized()


def test_a_grayscale_file_opens_as_a_grayscale_document():
    """The expansion the reader performs is exact, so the mode is what is left
    to carry -- and it is the mode that makes the *next* stroke stay grey."""
    data = _file(
        _header(1, 2, 1, GRAY_DEPTH),
        [_frame([_layer("Art"), _cel(0, bytes([90, 255, 200, 128]), 2, 1)])],
    )
    doc, _ = asein.document_from_aseprite(data)

    assert doc.color_mode == "grayscale" and doc.is_grayscale
    assert tuple(doc.stack[0].pixels[0, 0]) == (90, 90, 90, 255)
    assert tuple(doc.stack[0].pixels[0, 1]) == (200, 200, 200, 128)

    doc.begin_stroke((0, 0), (200, 20, 20, 255), size=1, nib="pixel")
    doc.end_stroke()
    r, g, b, _a = doc.stack[0].pixels[0, 0]
    assert r == g == b


def test_an_rgba_file_does_not_become_indexed_by_carrying_a_palette():
    """Aseprite writes a palette into every file it saves. Adopting it on an
    RGBA drawing would put the whole editor into indexed mode -- a constraint on
    every write -- for a table nobody asked to be constrained by."""
    doc, _ = asein.document_from_aseprite(
        _one_layer_file(extra=[_palette([(1, 2, 3, 255)])])
    )
    assert doc.palette is None
    assert not doc.is_indexed


def test_an_old_palette_chunk_is_read_when_it_is_the_only_one():
    data = _file(
        _header(1, 1, 1, INDEXED_DEPTH, transparent=255),
        [
            _frame(
                [
                    _layer("Art"),
                    _old_palette([(63, 0, 32)], six_bit=True),
                    _cel(0, bytes([0]), 1, 1),
                ]
            )
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    assert doc.palette == [(255, 0, 129, 255)]


def test_a_modern_palette_chunk_supersedes_the_old_one():
    data = _file(
        _header(1, 1, 1, INDEXED_DEPTH, transparent=255),
        [
            _frame(
                [
                    _layer("Art"),
                    _old_palette([(255, 255, 255)]),
                    _palette([(1, 2, 3, 128)]),
                    _cel(0, bytes([0]), 1, 1),
                ]
            )
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    assert doc.palette == [(1, 2, 3, 128)]


def test_a_per_frame_palette_warns_and_the_final_table_is_used():
    """Divergence 20: one table per document. Per-frame palettes are pre-1.0
    legacy the format merely tolerates, and losing them silently repaints
    frames the file coloured differently -- so the reader says so, and the
    final table wins."""
    data = _file(
        _header(2, 1, 1, INDEXED_DEPTH, transparent=255),
        [
            _frame(
                [_layer("Art"), _palette([(10, 20, 30, 255)]), _cel(0, bytes([0]), 1, 1)]
            ),
            _frame([_palette([(200, 100, 50, 255)]), _cel(0, bytes([0]), 1, 1)]),
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.palette == [(200, 100, 50, 255)], "the final table is used"
    assert "per-frame palettes are not kept; the final table is used" in warnings


def test_a_per_frame_old_palette_warns_the_same_way():
    """The pre-1.0 chunk pair is where per-frame palettes actually live, and the
    placeholder rows the old decoder appends must not trip the warning."""
    data = _file(
        _header(2, 1, 1, INDEXED_DEPTH, transparent=255),
        [
            _frame([_layer("Art"), _old_palette([(1, 2, 3)]), _cel(0, bytes([0]), 1, 1)]),
            _frame([_old_palette([(9, 8, 7)]), _cel(0, bytes([0]), 1, 1)]),
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.palette == [(9, 8, 7, 255)]
    assert "per-frame palettes are not kept; the final table is used" in warnings


def test_restating_an_identical_palette_does_not_warn():
    """The warning is about a table that *changed*: a file that repeats the same
    entries has one palette as far as this import is concerned, and a warning on
    it would teach the user to ignore the one that matters."""
    colours = [(1, 2, 3, 255)]
    data = _file(
        _header(2, 1, 1, INDEXED_DEPTH, transparent=255),
        [
            _frame([_layer("Art"), _palette(colours), _cel(0, bytes([0]), 1, 1)]),
            _frame([_palette(colours), _cel(0, bytes([0]), 1, 1)]),
        ],
    )
    _doc, warnings = asein.document_from_aseprite(data)
    assert warnings == []


def test_a_grayscale_file_is_converted_rather_than_refused():
    """The one conversion this reader performs, and it is exact: Aseprite draws
    a grey as ``(v, v, v)`` too, so replicating the channel loses nothing."""
    data = _file(
        _header(1, 2, 1, GRAY_DEPTH),
        [_frame([_layer("Art"), _cel(0, bytes([200, 255, 40, 128]), 2, 1)])],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert tuple(doc.stack[0].pixels[0, 0]) == (200, 200, 200, 255)
    assert tuple(doc.stack[0].pixels[0, 1]) == (40, 40, 40, 128)
    assert warnings == []


def test_group_layers_become_groups_with_their_nesting():
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Outer", kind=1),
                    _layer("Inner", kind=1, child=1),
                    _layer("Art", child=2),
                    _cel(2, _rgba(1, 1, (1, 1, 1, 255)), 1, 1),
                    _layer("Loose"),
                    _cel(3, _rgba(1, 1, (2, 2, 2, 255)), 1, 1),
                ]
            )
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    names = {node.name for node in doc.groups.values()}
    assert names == {"Outer", "Inner"}
    outer = next(uid for uid, node in doc.groups.items() if node.name == "Outer")
    inner = next(uid for uid, node in doc.groups.items() if node.name == "Inner")
    art = doc.stack[0]
    assert doc.stack[0].name == "Art" and doc.stack[1].name == "Loose"
    assert doc.group_of[art.uid] == inner
    assert doc.group_of[inner] == outer
    assert doc.stack[1].uid not in doc.group_of

    from warlock.studio.inker import groups as gp

    gp.check(doc.groups, doc.group_of, doc.member_uids())


def test_a_group_holding_nothing_is_dropped_rather_than_kept_empty():
    """``groups.check`` refuses an empty group, and an Aseprite file can hold
    one exactly as an ORA whose PNGs went missing can."""
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Empty", kind=1),
                    _layer("Art"),
                    _cel(1, _rgba(1, 1, (1, 1, 1, 255)), 1, 1),
                ]
            )
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    assert doc.groups == {}


def test_groups_survive_onto_an_animated_document_by_track():
    art = _rgba(1, 1, (1, 1, 1, 255))
    data = _file(
        _header(2, 1, 1),
        [
            _frame([_layer("Folder", kind=1), _layer("Art", child=1), _cel(1, art, 1, 1)]),
            _frame([_cel(1, art, 1, 1)]),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    folder = next(iter(doc.groups))
    assert doc.group_of[doc.anim.tracks[0].uid] == folder


def test_slices_arrive_with_their_pivots_centres_and_per_frame_keys():
    art = _rgba(8, 8, (1, 1, 1, 255))
    data = _file(
        _header(2, 8, 8),
        [
            _frame(
                [
                    _layer("Art"),
                    _cel(0, art, 8, 8),
                    _slice(
                        "hit",
                        [(0, (1, 1, 4, 4)), (1, (2, 2, 4, 4))],
                        centre=(1, 1, 2, 2),
                        pivot=(2, 3),
                    ),
                ]
            ),
            _frame([_cel(0, art, 8, 8)]),
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    entry = doc.slices[0]
    assert entry.name == "hit"
    assert entry.bounds == (1, 1, 5, 5)
    assert entry.pivot == (2.0, 3.0)
    assert entry.center == (1, 1, 3, 3)
    # The second key runs to the end of the timeline in Aseprite, which is one
    # frame here, so it is one override.
    assert entry.at(doc.anim.frames[1].uid).bounds == (2, 2, 6, 6)
    assert entry.at(doc.anim.frames[0].uid).bounds == (1, 1, 5, 5)
    assert warnings == []


def test_a_still_document_keeps_a_slice_without_a_timeline_to_key_it_to():
    doc, _ = asein.document_from_aseprite(
        _one_layer_file(extra=[_slice("box", [(0, (0, 0, 2, 2))])])
    )
    assert doc.slices[0].bounds == (0, 0, 2, 2)
    assert doc.slices[0].keys == {}


def test_a_file_of_nothing_but_folders_opens_empty_rather_than_refusing():
    data = _file(_header(1, 3, 3), [_frame([_layer("Folder", kind=1)])])
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.size == (3, 3)
    assert len(doc.stack) == 1
    assert any("no drawable layers" in line for line in warnings)


# --- one named test per refusal ----------------------------------------------


def test_a_file_that_is_not_aseprite_is_refused_by_name():
    data = _file(_header(1, 1, 1, magic=0x1234), [_frame([_layer("Art")])])
    with pytest.raises(ValueError, match="not an Aseprite file"):
        asein.document_from_aseprite(data)


def test_a_file_too_short_to_have_a_header_is_refused_by_name():
    with pytest.raises(ValueError, match="too short"):
        asein.document_from_aseprite(b"\0" * 12)


def test_a_colour_depth_this_build_cannot_read_is_refused_by_name():
    data = _file(_header(1, 1, 1, 64), [_frame([_layer("Art")])])
    with pytest.raises(ValueError, match="colour depth of 64 bits"):
        asein.document_from_aseprite(data)


def test_a_file_with_no_frames_is_refused_by_name():
    data = _file(_header(0, 1, 1), [])
    with pytest.raises(ValueError, match="no frames"):
        asein.document_from_aseprite(data)


def test_a_canvas_with_no_size_is_refused_by_name():
    data = _file(_header(1, 0, 0), [_frame([_layer("Art")])])
    with pytest.raises(ValueError, match="not one to draw on"):
        asein.document_from_aseprite(data)


def test_a_frame_that_is_not_a_frame_is_refused_by_name():
    body = _frame([_layer("Art")])
    broken = body[:4] + struct.pack("<H", 0x1234) + body[6:]
    with pytest.raises(ValueError, match="frame 0 of this .aseprite is not a frame"):
        asein.document_from_aseprite(_file(_header(1, 1, 1), [broken]))


def test_a_file_that_ends_mid_structure_is_refused_by_name():
    data = _one_layer_file()
    with pytest.raises(ValueError, match="ends in the middle|runs past"):
        asein.document_from_aseprite(data[:-6])


def test_a_tilemap_layer_opens_bound_to_its_tileset():
    """Wave 3 chunk 3.5: what was a refusal is now a reader. A tilemap layer
    opens as a ``TilemapCel`` bound to the tileset its own trailing field
    names, and the document carries the tileset itself."""
    blank = _rgba(2, 2, (0, 0, 0, 0))
    red = _rgba(2, 2, (255, 0, 0, 255))
    data = _file(
        _header(1, 2, 2),
        [
            _frame(
                [
                    _tileset_chunk(7, 2, 2, [blank, red]),
                    _layer("Map", kind=2, tileset=7),
                    _tilemap_cel(0, np.array([[1]], dtype=np.uint32)),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert len(doc.tilesets) == 1
    cel = doc.stack[0]
    assert isinstance(cel, TilemapCel)
    assert cel.tileset_uid == doc.tilesets[0].uid
    assert cel.refs.tolist() == [[1]]
    assert tuple(cel.pixels[0, 0]) == (255, 0, 0, 255)
    assert warnings == []


def test_a_tileset_chunk_is_read_into_the_documents_tileset_list():
    """A tileset survives an import even when nothing draws through it yet --
    ``ora``'s "not garbage" rule, restated for the reader that predates it."""
    blank = _rgba(1, 1, (0, 0, 0, 0))
    green = _rgba(1, 1, (0, 255, 0, 255))
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Art"),
                    _cel(0, _rgba(1, 1, (1, 1, 1, 255)), 1, 1),
                    _tileset_chunk(3, 1, 1, [blank, green], name="ground"),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert len(doc.tilesets) == 1
    slot = doc.tilesets[0]
    assert slot.tileset.name == "ground"
    assert slot.tileset.tile_count == 2
    assert tuple(slot.tileset.tile_pixels(1)[0, 0]) == (0, 255, 0, 255)
    assert warnings == []


def test_a_tilemap_cel_decodes_its_refs_grid():
    blank = _rgba(2, 2, (0, 0, 0, 0))
    art_a = _rgba(2, 2, (10, 20, 30, 255))
    art_b = _rgba(2, 2, (40, 50, 60, 255))
    data = _file(
        _header(1, 4, 2),
        [
            _frame(
                [
                    _tileset_chunk(1, 2, 2, [blank, art_a, art_b]),
                    _layer("Map", kind=2, tileset=1),
                    _tilemap_cel(0, np.array([[1, 2]], dtype=np.uint32)),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    cel = doc.stack[0]
    assert isinstance(cel, TilemapCel)
    assert cel.refs.tolist() == [[1, 2]]
    assert tuple(cel.pixels[0, 0]) == (10, 20, 30, 255)
    assert tuple(cel.pixels[0, 2]) == (40, 50, 60, 255)
    assert warnings == []


def test_a_cel_type_this_build_does_not_know_is_refused_by_name():
    data = _file(_header(1, 1, 1), [_frame([_layer("Art"), _raw_cel(0, 9)])])
    with pytest.raises(ValueError, match="cel type 9"):
        asein.document_from_aseprite(data)


def test_a_layer_kind_this_build_does_not_know_is_refused_by_name():
    data = _file(_header(1, 1, 1), [_frame([_layer("Odd", kind=7)])])
    with pytest.raises(ValueError, match="kind this build does not know"):
        asein.document_from_aseprite(data)


def test_a_linked_cel_that_links_to_itself_is_refused_by_name():
    """Frame 1's cel names frame 1: a chain with nowhere to end. Refused rather
    than followed, because this is somebody else's file and a cycle walked is a
    hang. The *dangling* link -- a chain that ends at a slot holding no cel --
    is the test below, and the two messages are pinned separately so neither
    can start standing in for the other."""
    data = _file(
        _header(2, 1, 1),
        [
            _frame([_layer("Art"), _cel(0, _rgba(1, 1, (1, 1, 1, 255)), 1, 1)]),
            _frame([_linked_cel(0, 1)]),
        ],
    )
    with pytest.raises(ValueError, match="links to itself"):
        asein.document_from_aseprite(data)


def test_a_link_to_a_frame_with_no_cel_on_that_layer_is_refused_by_name():
    art = _rgba(1, 1, (1, 1, 1, 255))
    data = _file(
        _header(3, 1, 1),
        [
            _frame([_layer("Under"), _layer("Over"), _cel(0, art, 1, 1)]),
            _frame([_cel(0, art, 1, 1)]),
            # ``Over`` has no cel anywhere, so frame 1 of it is nothing to link
            # to -- and a document with a hole where a cel was is the outcome
            # worth refusing.
            _frame([_linked_cel(1, 1)]),
        ],
    )
    with pytest.raises(ValueError, match="has no cel on that layer"):
        asein.document_from_aseprite(data)


def test_a_cel_naming_a_layer_that_is_not_there_is_refused_by_name():
    data = _file(
        _header(1, 1, 1),
        [_frame([_layer("Art"), _cel(5, _rgba(1, 1, (1, 1, 1, 255)), 1, 1)])],
    )
    with pytest.raises(ValueError, match="names layer 5"):
        asein.document_from_aseprite(data)


def test_a_cel_that_will_not_decompress_is_refused_by_name():
    body = struct.pack("<HhhBHh5s", 0, 0, 0, 255, 2, 0, b"\0" * 5)
    body += struct.pack("<HH", 1, 1) + b"not zlib at all"
    data = _file(_header(1, 1, 1), [_frame([_layer("Art"), _chunk(0x2005, body)])])
    with pytest.raises(ValueError, match="will not decompress"):
        asein.document_from_aseprite(data)


def test_a_cel_whose_pixels_do_not_fill_its_rectangle_is_refused_by_name():
    data = _file(
        _header(1, 4, 4),
        [_frame([_layer("Art"), _cel(0, _rgba(1, 1, (1, 1, 1, 255)), 4, 4)])],
    )
    with pytest.raises(ValueError, match="bytes where its 4x4 rectangle needs"):
        asein.document_from_aseprite(data)


def test_an_indexed_file_with_no_palette_is_refused_by_name():
    data = _file(
        _header(1, 1, 1, INDEXED_DEPTH),
        [_frame([_layer("Art"), _cel(0, bytes([0]), 1, 1)])],
    )
    with pytest.raises(ValueError, match="no palette to read it with"):
        asein.document_from_aseprite(data)


def test_an_index_outside_the_palette_is_refused_by_name():
    data = _file(
        _header(1, 1, 1, INDEXED_DEPTH, transparent=255),
        [_frame([_layer("Art"), _palette([(1, 2, 3, 255)]), _cel(0, bytes([9]), 1, 1)])],
    )
    with pytest.raises(ValueError, match="names palette entry 9"):
        asein.document_from_aseprite(data)


def test_an_external_file_tileset_is_refused_by_name():
    """Unlike every other divergence in this reader, an external tileset's
    pixels are not somewhere else in *this* file -- they are not in it at
    all, so a tilemap layer bound to it would draw nothing."""
    blank = _rgba(1, 1, (0, 0, 0, 0))
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Art"),
                    _tileset_chunk(1, 1, 1, [blank], flags=1 | 2),
                ]
            )
        ],
    )
    with pytest.raises(ValueError, match="links an external file"):
        asein.document_from_aseprite(data)


def test_a_tilemap_cel_at_other_than_32_bits_per_tile_is_refused_by_name():
    blank = _rgba(1, 1, (0, 0, 0, 0))
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _tileset_chunk(1, 1, 1, [blank]),
                    _layer("Map", kind=2, tileset=1),
                    _tilemap_cel(0, np.zeros((1, 1), dtype=np.uint32), bits=16),
                ]
            )
        ],
    )
    with pytest.raises(ValueError, match="16 bits per tile"):
        asein.document_from_aseprite(data)


def test_a_non_tile_aligned_tilemap_cel_offset_is_refused_by_name():
    """Unreachable from Aseprite's own writer, which always places a tilemap
    cel on a tile boundary -- but somebody else's file is not bound by that,
    and there is no partial cell for an unaligned offset to land in."""
    blank = _rgba(2, 2, (0, 0, 0, 0))
    art = _rgba(2, 2, (1, 2, 3, 255))
    data = _file(
        _header(1, 4, 4),
        [
            _frame(
                [
                    _tileset_chunk(1, 2, 2, [blank, art]),
                    _layer("Map", kind=2, tileset=1),
                    _tilemap_cel(0, np.array([[1]], dtype=np.uint32), x=1, y=0),
                ]
            )
        ],
    )
    with pytest.raises(ValueError, match="not a multiple of its 2x2 tile size"):
        asein.document_from_aseprite(data)


def test_a_tilemap_cel_naming_an_undefined_tileset_is_refused_by_name():
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Map", kind=2, tileset=99),
                    _tilemap_cel(0, np.zeros((1, 1), dtype=np.uint32)),
                ]
            )
        ],
    )
    with pytest.raises(ValueError, match="names a tileset this .aseprite does not define"):
        asein.document_from_aseprite(data)


# --- warned about, and the document still opens -------------------------------


def test_a_colour_profile_is_a_warning_rather_than_a_refusal():
    """Divergence 3: sRGB-assumed bytes end to end, so the profile is the only
    thing lost and the pixels are exactly as stored."""
    profile = _chunk(0x2007, struct.pack("<HHI8s", 2, 0, 0, b"\0" * 8) + b"\0" * 4)
    doc, warnings = asein.document_from_aseprite(_one_layer_file(extra=[profile]))
    assert doc.size == (2, 2)
    assert any("colour profile" in line for line in warnings)


def test_user_data_is_a_warning_rather_than_a_refusal():
    """Divergence 14: tags are names and numbers; timeline colours and user
    data are chrome."""
    user = _chunk(0x2020, struct.pack("<I", 1) + _string("notes"))
    _doc, warnings = asein.document_from_aseprite(_one_layer_file(extra=[user]))
    assert any("user data" in line for line in warnings)


def test_a_tileset_base_index_other_than_one_warns_rather_than_refuses():
    """``base_index`` is Aseprite's own display-only numbering in its tileset
    panel; it changes no id this reader stores."""
    blank = _rgba(1, 1, (0, 0, 0, 0))
    data = _file(
        _header(1, 1, 1),
        [_frame([_layer("Art"), _tileset_chunk(1, 1, 1, [blank], base_index=0)])],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert len(doc.tilesets) == 1
    assert any("start somewhere other than 1" in line for line in warnings)


def test_a_tilesets_own_user_data_warns_generically_not_per_tile():
    """The spec's own convention: "After the Tileset chunk, it could be
    followed by a user data chunk (empty or not) and then all the user data
    chunks of the tiles ordered by tile index" -- so the *first* ``USER_DATA``
    chunk in a run after a tileset chunk belongs to the tileset itself, an
    ordinary owner, and earns its own warning rather than the per-tile one.

    Both sentences narrowed on 2026-08-30 when divergence 14 was partially
    retired: a layer's, a cel's and a tag's user data are *kept* now, so the
    old shared "user data and timeline colours are not kept" line would be
    false about three of the five owners. A tileset's own note still drops,
    and says so about itself."""
    blank = _rgba(1, 1, (0, 0, 0, 0))
    user = _chunk(0x2020, struct.pack("<I", 1) + _string("notes"))
    data = _file(
        _header(1, 1, 1),
        [_frame([_layer("Art"), _tileset_chunk(1, 1, 1, [blank]), user])],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert len(doc.tilesets) == 1
    assert any("a tileset's own user data was dropped" in line for line in warnings)
    assert not any("per-tile properties" in line for line in warnings)


def test_per_tile_user_data_warns_that_the_tiles_are_kept():
    """The tileset's own user-data chunk (the first in the run) is followed
    by two per-tile ones (the second and third) -- only those earn the
    per-tile warning, and however many tiles carry one, it is one line."""
    blank = _rgba(1, 1, (0, 0, 0, 0))
    tileset_own = _chunk(0x2020, struct.pack("<I", 1) + _string("tileset notes"))
    tile_a = _chunk(0x2020, struct.pack("<I", 1) + _string("tile 0 notes"))
    tile_b = _chunk(0x2020, struct.pack("<I", 1) + _string("tile 1 notes"))
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Art"),
                    _tileset_chunk(1, 1, 1, [blank]),
                    tileset_own,
                    tile_a,
                    tile_b,
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert len(doc.tilesets) == 1
    assert warnings.count("per-tile properties are not kept; the tiles are") == 1
    assert any("a tileset's own user data was dropped" in line for line in warnings)


def test_an_empty_user_data_chunk_says_nothing():
    """Aseprite 1.3 writes one after the tags chunk for every tag whether or
    not anything was put in it. Warning about those would put a message on
    screen for almost every tagged file and teach the reader to ignore it."""
    empty = _chunk(0x2020, struct.pack("<I", 0))
    _doc, warnings = asein.document_from_aseprite(_one_layer_file(extra=[empty]))
    assert warnings == []


def test_a_per_cel_opacity_on_a_still_file_is_a_warning_rather_than_a_refusal():
    """All that is left of divergence 1, retired 2026-08-30.

    Per-cel opacity is kept now -- in ``Animation.cel_opacity``, which is part
    of the *grid*. A one-frame ``.aseprite`` opens as a still document
    (divergence 22: ``Document.anim is None``), so there is no grid for the
    byte to land on and it is said out loud rather than folded into the layer's
    own opacity, which is a different number.
    """
    cel = _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2, opacity=128)
    doc, warnings = asein.document_from_aseprite(_one_layer_file(cel=cel))
    assert doc.anim is None
    assert any("per-cel opacity" in line for line in warnings)
    # And the layer keeps the opacity the *layer* chunk declared, untouched.
    assert doc.stack[0].opacity == 1.0


def test_a_cel_z_index_on_a_still_document_is_warned_rather_than_refused():
    """The last of divergence 12, which was retired on 2026-08-30.

    A z-index is an offset within the *grid*'s stack, and a one-frame file
    opens as a still document with no grid -- divergence 22 -- so this is the
    one shape of file where the field still has nowhere to land. Said out
    loud rather than folded into the layer order, which would rewrite what the
    next save calls the stack.
    """
    cel = _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2, z=3)
    doc, warnings = asein.document_from_aseprite(_one_layer_file(cel=cel))
    assert doc.anim is None
    assert any("z-index" in line for line in warnings)


def test_a_cel_reaching_past_the_canvas_is_cropped_with_a_warning():
    data = _file(
        _header(1, 2, 2),
        [_frame([_layer("Art"), _cel(0, _rgba(2, 2, (5, 6, 7, 255)), 2, 2, x=1, y=1)])],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert tuple(doc.stack[0].pixels[1, 1]) == (5, 6, 7, 255)
    assert any("past the canvas" in line for line in warnings)


def test_a_linked_cel_drawn_at_its_own_offset_is_unlinked_where_it_stands():
    """Aseprite shares a cel's *position* along with its image, so this is a
    file no version of it writes -- but sharing the object would draw the
    pixels in the wrong place, so the link is what gives way rather than the
    picture."""
    art = _rgba(2, 2, (4, 5, 6, 255))
    data = _file(
        _header(2, 4, 4),
        [
            _frame([_layer("Art"), _cel(0, art, 2, 2)]),
            _frame([_linked_cel(0, 0, x=1, y=1)]),
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    anim = doc.anim
    track = anim.tracks[0].uid
    first = anim.cels[(track, anim.frames[0].uid)]
    second = anim.cels[(track, anim.frames[1].uid)]
    assert second is not first
    assert not anim.is_linked(track, anim.frames[1].uid)
    assert tuple(first.pixels[0, 0]) == (4, 5, 6, 255)
    assert tuple(second.pixels[0, 0]) == (0, 0, 0, 0)
    assert tuple(second.pixels[1, 1]) == (4, 5, 6, 255)
    assert any("unlinked to keep it there" in line for line in warnings)
    # The move stays inside the canvas, so nothing is cropped and the reader
    # must not say it was.
    assert not any("past the canvas" in line for line in warnings)


def test_a_cel_on_a_group_layer_is_dropped_with_a_warning():
    """A group holds no pixels here and none in Aseprite either, so a cel
    naming one is a file that disagrees with itself. Dropped rather than
    refused: the drawing's own layers are all still present and correct."""
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Folder", kind=1),
                    _cel(0, _rgba(1, 1, (9, 9, 9, 255)), 1, 1),
                    _layer("Art"),
                    _cel(1, _rgba(1, 1, (1, 2, 3, 255)), 1, 1),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert [layer.name for layer in doc.stack] == ["Art"]
    assert tuple(doc.stack[0].pixels[0, 0]) == (1, 2, 3, 255)
    # The folder held nothing in the end, so it is pruned like any empty group.
    assert doc.groups == {}
    assert any("on a group layer was dropped" in line for line in warnings)


def test_a_ping_pong_from_the_far_end_warns_and_plays_from_the_near_one():
    art = _rgba(1, 1, (1, 1, 1, 255))
    data = _file(
        _header(2, 1, 1),
        [
            _frame([_layer("Art"), _cel(0, art, 1, 1), _tags([("swing", 0, 1, 3, 0)])]),
            _frame([_cel(0, art, 1, 1)]),
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.anim.tags[0].direction == "pingpong"
    assert any("far end" in line for line in warnings)


def test_a_reference_layer_keeps_the_visibility_the_file_states():
    """The VISIBLE bit wins, on a reference layer like any other.

    This used to force ``visible = False`` on the argument that Aseprite opens
    reference layers hidden. It does -- but ``aseout`` writes ``visible``
    verbatim beside the REFERENCE flag, so a file can perfectly well say one is
    showing, and the override meant a user who toggled one on and saved got it
    back hidden on the next load with nothing said. The warning stays: naming
    the layer is the useful half, and it never depended on lying about the bit.
    """
    def sprite(visible: bool) -> bytes:
        return _file(
            _header(1, 1, 1),
            [
                _frame(
                    [
                        _layer("Trace", reference=True, visible=visible),
                        _cel(0, _rgba(1, 1, (1, 1, 1, 255)), 1, 1),
                        _layer("Art"),
                        _cel(1, _rgba(1, 1, (2, 2, 2, 255)), 1, 1),
                    ]
                )
            ],
        )

    shown, warnings = asein.document_from_aseprite(sprite(True))
    assert shown.stack[0].visible is True
    assert any("reference layer" in line for line in warnings)

    hidden, _warnings = asein.document_from_aseprite(sprite(False))
    assert hidden.stack[0].visible is False


def test_a_chunk_this_build_does_not_read_is_named_in_a_warning():
    _doc, warnings = asein.document_from_aseprite(
        _one_layer_file(extra=[_chunk(0x2099, b"\0" * 4)])
    )
    assert any("0x2099" in line for line in warnings)


def test_a_tag_over_a_single_frame_is_dropped_with_a_warning():
    _doc, warnings = asein.document_from_aseprite(
        _one_layer_file(extra=[_tags([("idle", 0, 0, 0, 0)])])
    )
    assert any("single frame" in line for line in warnings)


def test_a_warning_is_one_line_per_kind_however_many_times_it_happens():
    """A file with two hundred cels carrying user data has one thing wrong with
    it as far as this import is concerned.

    **Re-pointed twice, and the reason is worth keeping.** It was measured on
    the per-cel *opacity* warning until divergence 1 was retired on 2026-08-30
    and the byte started being kept; it moved to the *z-index* warning, and
    divergence 12 was retired the same day, so that one is kept now too. What
    is left that genuinely repeats once per cel chunk is the **cel-extra**
    chunk (0x2006) -- a cel's precise sub-pixel bounds, which this package has
    no room for because a layer here is canvas-sized. Three of them in one
    file is one line.
    """
    art = _rgba(1, 1, (1, 1, 1, 255))
    extra = _chunk(0x2006, b"\0" * 16)
    data = _file(
        _header(3, 1, 1),
        [
            _frame([_layer("Art"), _cel(0, art, 1, 1), extra]),
            _frame([_cel(0, art, 1, 1), extra]),
            _frame([_cel(0, art, 1, 1), extra]),
        ],
    )
    _doc, warnings = asein.document_from_aseprite(data)
    assert len(warnings) == len(set(warnings))
    assert sum("precise bounds" in line for line in warnings) == 1


# --- tilemaps: Wave 3 chunk 3.5 -----------------------------------------------


def test_flags_remap_under_permuted_declared_masks_lands_on_our_bits():
    """The masks are a field in the file, not a constant this reader assumes.
    A straight cast of the raw uint32 would get this wrong: under these
    (synthetic) masks the tile id lives in the low nibble and the x-flip bit
    sits where a straight reading of ``gid``'s own layout would call part of
    the id, so passing without doing the mask arithmetic would be a coincidence
    that this permutation is built to rule out."""
    blank = _rgba(1, 1, (0, 0, 0, 0))
    art = _rgba(1, 1, (9, 9, 9, 255))
    raw = np.array([[0x11]], dtype=np.uint32)  # id=1 (low nibble), x-flip (bit 4)
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _tileset_chunk(1, 1, 1, [blank, art]),
                    _layer("Map", kind=2, tileset=1),
                    _tilemap_cel(
                        0,
                        raw,
                        id_mask=0x0F,
                        x_mask=0x10,
                        y_mask=0x20,
                        d_mask=0x40,
                    ),
                ]
            )
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    cel = doc.stack[0]
    assert int(cel.refs[0, 0]) == gid.compose(1, flip_h=True)


def test_refs_match_a_hand_built_grid_including_flip_bits():
    blank = _rgba(2, 2, (0, 0, 0, 0))
    tile = np.zeros((2, 2, 4), dtype=np.uint8)
    tile[0, 0] = (255, 0, 0, 255)
    tile[0, 1] = (0, 255, 0, 255)
    ref_value = gid.compose(1, flip_h=True)
    data = _file(
        _header(1, 2, 2),
        [
            _frame(
                [
                    _tileset_chunk(1, 2, 2, [blank, tile.tobytes()]),
                    _layer("Map", kind=2, tileset=1),
                    _tilemap_cel(0, np.array([[ref_value]], dtype=np.uint32)),
                ]
            )
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    cel = doc.stack[0]
    assert cel.refs.tolist() == [[ref_value]]
    # Flipped horizontally: the tile's own (0, 0) red lands at canvas (0, 1).
    assert tuple(cel.pixels[0, 1]) == (255, 0, 0, 255)
    assert tuple(cel.pixels[0, 0]) == (0, 255, 0, 255)


def test_a_linked_tilemap_cel_arrives_as_one_object_in_two_frames():
    """The mapping that makes this import worth more than exported PNGs,
    restated for a tilemap cel: linking it links ``refs`` for free, because a
    ``TilemapCel`` is still a ``Layer``."""
    blank = _rgba(1, 1, (0, 0, 0, 0))
    art = _rgba(1, 1, (5, 5, 5, 255))
    data = _file(
        _header(2, 1, 1),
        [
            _frame(
                [
                    _tileset_chunk(1, 1, 1, [blank, art]),
                    _layer("Map", kind=2, tileset=1),
                    _tilemap_cel(0, np.array([[1]], dtype=np.uint32)),
                ]
            ),
            _frame([_linked_cel(0, 0)]),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    anim = doc.anim
    track = anim.tracks[0].uid
    first = anim.cels[(track, anim.frames[0].uid)]
    second = anim.cels[(track, anim.frames[1].uid)]
    assert isinstance(first, TilemapCel)
    assert second is first
    assert anim.is_linked(track, anim.frames[1].uid)


# --- the parse on its own ------------------------------------------------------


def test_the_parse_is_pure_data_and_keeps_the_cels_as_the_file_holds_them():
    sprite = asein.parse(_one_layer_file())
    assert (sprite.width, sprite.height, sprite.depth) == (2, 2, 32)
    assert [layer.name for layer in sprite.layers] == ["Art"]
    assert sprite.durations == [100]
    assert sprite.cels[0].width == 2
    assert sprite.warnings == []


def test_aseprites_prefer_linked_cels_flag_arrives_as_continuous():
    """Bit 16 is Aseprite's "continuous layer": new cels start from the last
    drawing rather than blank. Aseprite *links* them and we *copy*, which is
    the nearest honest reading -- the alternative is dropping the flag and
    silently giving the user blank frames where they drew one pose.
    """
    data = _file(
        _header(2, 2, 2),
        [
            _frame(
                [
                    _layer("Held", continuous=True),
                    _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2),
                ]
            ),
            _frame([]),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    assert doc.anim is not None
    assert [t.continuous for t in doc.anim.tracks] == [True]


def test_a_plain_aseprite_layer_is_not_continuous():
    data = _file(
        _header(2, 2, 2),
        [
            _frame(
                [
                    _layer("Plain"),
                    _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2),
                ]
            ),
            _frame([]),
        ],
    )
    doc, _ = asein.document_from_aseprite(data)
    assert [t.continuous for t in doc.anim.tracks] == [False]


def test_a_transparent_index_past_the_palette_is_clamped_before_the_fill():
    """The header byte is whatever the writing tool put there, and a file
    naming a slot past the end of its own palette still has to open.

    ``_install_indexed`` clamped it to 0 and ``_build_cels`` did not, so the
    canvas *around* a tight cel was filled with an index the document's table
    has no row for. ``index_plane.materialize`` clips rather than raises, so
    the surround opened as an opaque block of the **last** swatch instead of
    transparency -- and ``check_materialized`` passed, because both planes
    agreed about the wrong answer, so a save wrote it out.

    A 2x2 canvas with a 1x1 cel in the corner is the smallest shape that has a
    surround at all; real Aseprite writes tight cels by default.
    """
    colours = [(0, 0, 0, 255), (255, 0, 0, 255)]
    data = _file(
        _header(1, 2, 2, INDEXED_DEPTH, transparent=200),
        [_frame([_layer("Art"), _palette(colours), _cel(0, bytes([1]), 1, 1)])],
    )
    doc, _warnings = asein.document_from_aseprite(data)

    assert doc.transparent_index == 0
    plane = doc.stack[0].indices
    # Every cell the cel does not cover holds the clamped slot, not 200.
    assert plane[0, 1] == 0 and plane[1, 0] == 0 and plane[1, 1] == 0
    doc.check_materialized()
    # And it materialises as nothing, rather than as the last swatch.
    assert tuple(doc.stack[0].pixels[1, 1]) == (0, 0, 0, 0)
