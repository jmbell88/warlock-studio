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
) -> bytes:
    flags = (
        (1 if visible else 0)
        | (2 if editable else 0)
        | (8 if background else 0)
        | (64 if reference else 0)
    )
    return _chunk(
        0x2004,
        struct.pack("<HHHHHHB3s", flags, kind, child, 0, 0, blend, opacity, b"\0\0\0")
        + _string(name),
    )


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


def _linked_cel(layer: int, frame: int, *, x: int = 0, y: int = 0) -> bytes:
    return _chunk(
        0x2005,
        struct.pack("<HhhBHh5s", layer, x, y, 255, 1, 0, b"\0" * 5)
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
    # Palette-constrained RGB for now: the reader flattens the file's index
    # planes through its table. Wave 1.4 makes this ``is_indexed`` and keeps the
    # planes, at which point the duplicate-swatch fixture below becomes exact.
    assert doc.is_palette_locked
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
    doc, _ = asein.document_from_aseprite(data)
    assert tuple(doc.stack[0].pixels[0, 0]) == (7, 8, 9, 255)


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


def test_a_tilemap_layer_is_refused_by_name():
    """Its pixels live in a tileset, so a document built without one draws
    nothing where the layer was -- the change in meaning a refusal is for."""
    data = _file(_header(1, 1, 1), [_frame([_layer("Map", kind=2)])])
    with pytest.raises(ValueError, match="is a tilemap"):
        asein.document_from_aseprite(data)


def test_a_tileset_chunk_is_refused_by_name():
    data = _file(_header(1, 1, 1), [_frame([_layer("Art"), _chunk(0x2023, b"\0" * 8)])])
    with pytest.raises(ValueError, match="uses tilesets"):
        asein.document_from_aseprite(data)


def test_a_tilemap_cel_is_refused_by_name():
    data = _file(
        _header(1, 1, 1),
        [_frame([_layer("Art"), _raw_cel(0, 3, struct.pack("<HH", 1, 1) + b"\0" * 4)])],
    )
    with pytest.raises(ValueError, match="tilemap cel"):
        asein.document_from_aseprite(data)


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


def test_an_empty_user_data_chunk_says_nothing():
    """Aseprite 1.3 writes one after the tags chunk for every tag whether or
    not anything was put in it. Warning about those would put a message on
    screen for almost every tagged file and teach the reader to ignore it."""
    empty = _chunk(0x2020, struct.pack("<I", 0))
    _doc, warnings = asein.document_from_aseprite(_one_layer_file(extra=[empty]))
    assert warnings == []


def test_a_per_cel_opacity_is_a_warning_rather_than_a_refusal():
    """Divergence 1: opacity is a track property here."""
    cel = _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2, opacity=128)
    _doc, warnings = asein.document_from_aseprite(_one_layer_file(cel=cel))
    assert any("per-cel opacity" in line for line in warnings)


def test_a_cel_z_index_is_a_warning_rather_than_a_refusal():
    """Divergence 12: track order *is* stack order, which the compositor and
    the native stack kernel both assume."""
    cel = _cel(0, _rgba(2, 2, (1, 1, 1, 255)), 2, 2, z=3)
    _doc, warnings = asein.document_from_aseprite(_one_layer_file(cel=cel))
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


def test_a_reference_layer_opens_hidden_with_a_warning():
    data = _file(
        _header(1, 1, 1),
        [
            _frame(
                [
                    _layer("Trace", reference=True),
                    _cel(0, _rgba(1, 1, (1, 1, 1, 255)), 1, 1),
                    _layer("Art"),
                    _cel(1, _rgba(1, 1, (2, 2, 2, 255)), 1, 1),
                ]
            )
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.stack[0].visible is False
    assert any("reference layer" in line for line in warnings)


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
    it as far as this import is concerned."""
    art = _rgba(1, 1, (1, 1, 1, 255))
    user = _chunk(0x2020, struct.pack("<I", 0))
    data = _file(
        _header(3, 1, 1),
        [
            _frame([_layer("Art"), _cel(0, art, 1, 1, opacity=10), user]),
            _frame([_cel(0, art, 1, 1, opacity=20), user]),
            _frame([_cel(0, art, 1, 1, opacity=30), user]),
        ],
    )
    _doc, warnings = asein.document_from_aseprite(data)
    assert len(warnings) == len(set(warnings))
    assert sum("per-cel opacity" in line for line in warnings) == 1


# --- the parse on its own ------------------------------------------------------


def test_the_parse_is_pure_data_and_keeps_the_cels_as_the_file_holds_them():
    sprite = asein.parse(_one_layer_file())
    assert (sprite.width, sprite.height, sprite.depth) == (2, 2, 32)
    assert [layer.name for layer in sprite.layers] == ["Art"]
    assert sprite.durations == [100]
    assert sprite.cels[0].width == 2
    assert sprite.warnings == []
