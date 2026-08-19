"""A document as an ``.aseprite`` file. :mod:`.asein` read the other way.

This is the writer that module's docstring said did not exist, and the reason
it exists now is the same one that made the reader worth having: a linked cel,
a group and a tag map onto this format field for field, so a document written
here and opened in Aseprite is the *same document* rather than a flattened
picture of one. The read-only stance it enforced -- ``path = None``, so the
first Ctrl+S is a Save As -- was a consequence of there being nothing to write
back with, and this module is what retires it.

**The writer is the reader's mirror, deliberately and literally.** Every field
order below is the one :mod:`.asein` consumes, every constant is imported from
it rather than restated, and the automated gate is a round trip through it --
``tests/inker/test_aseout.py`` writes a document, reads it back and asserts the
planes, the flags, the palette and the share structure are the same ones. A
golden binary in the tree would pin these bytes against nothing, since a writer
and its own golden file can be wrong together forever. What a round trip cannot
reach -- whether *real* Aseprite agrees -- is a human's pass with the app
installed, which is the Tiled-fixtures precedent and is Wave 5's own gate.

Four decisions worth reading:

**Cels are written full-canvas at (0, 0), never cropped to their ink.** A tight
rectangle is what Aseprite itself stores and it would make smaller files, but
this package's layers *are* canvas-sized (``layers.py`` says why), so a crop
would be arithmetic invented at the door with a clip warning waiting on the
other side of it. Full-canvas costs bytes zlib mostly gives back and cannot be
off by a pixel.

**An indexed document writes its index planes, never its pixels.** Re-quantising
the RGBA materialisation would collapse two palette slots holding the same
colour into whichever one a nearest-match happened to pick -- which is exactly
the identity :mod:`.index_plane` exists to keep and :func:`asein._decode_indices`
exists to read back. The plane is the record; it is written as the record.

**A link is written as a link.** ``cels[(t, f1)] is cels[(t, f2)]`` is this
model's spelling of a linked cel, so the first frame holding an object gets the
pixels and every later slot gets a type-1 cel naming that *frame index*. Sharing
survives the trip in both directions and an edit in Aseprite still shows on
every frame the cel occupies.

**What is lost is lost silently, and written down elsewhere.** There is no
warning channel out of a function that returns bytes, and inventing one would
put a toast on a save that did exactly what the user asked. Three things drop:
``alpha_lock`` (Aseprite has no bit for it -- an editing aid, not picture data),
a group's opacity (Aseprite's UI offers a group none, and ``asein._group_tree``
reads one back as 1.0 whatever byte is stored, so writing anything but 255 would
be a number nobody could ever read), and an empty group (a group is a *run* of
the layer list here, so one with no members has no run to write). Each is a line
in ``docs/ASEPRITE_INTEROP.md``.

Refusals are by name and are the format's own limits, not this build's: more
frames or layers than a 16-bit count can hold, a palette past 256, a canvas past
65535 on a side, a tag repeating more times than its own WORD can say, a colour
mode with no depth to write it at, and -- the one that is about a *corrupted*
document rather than a big one -- a grayscale document holding a pixel where
``r != g != b``. The write funnel enforces greyness on every stroke, so a
violation means the constraint was bypassed, and the alternative to refusing is
throwing two channels away without a word.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .animation import DEFAULT_DURATION_MS
from .asein import (
    _BLEND_BY_INDEX,
    _CEL,
    _CEL_COMPRESSED,
    _CEL_LINKED,
    _FRAME_MAGIC,
    _GRAYSCALE,
    _INDEXED,
    _LAYER,
    _LAYER_CONTINUOUS,
    _LAYER_EDITABLE,
    _LAYER_GROUP,
    _LAYER_IMAGE,
    _LAYER_VISIBLE,
    _MAGIC,
    _OLD_PALETTE_256,
    _PALETTE,
    _RGBA,
    _TAG_DIRECTIONS,
    _TAGS,
)
from .index_plane import MAX_COLOURS

__all__ = ["ASEPRITE_SUFFIX", "aseprite_bytes", "write_aseprite"]

#: What this writer produces. The reader opens ``.ase`` as well -- both
#: spellings are the same format -- but a *save* picks one, and the modern one
#: is what Aseprite itself has written since 1.0.
ASEPRITE_SUFFIX = ".aseprite"

#: Every count this format spends a WORD on. Frames, layers, a tag's repeat and
#: a canvas side are all bounded by it, and each is refused by its own name
#: rather than silently truncated into a file that opens as a different sprite.
_MAX_U16 = 0xFFFF

#: The header's own size, and the offset the first frame starts at.
_HEADER_SIZE = 128

#: ``Document.color_mode`` onto the format's bits per pixel. The three the
#: reader knows, which is not a coincidence: a mode this pair cannot round trip
#: is a mode neither half should pretend to have.
_DEPTHS = {"rgb": _RGBA, "grayscale": _GRAYSCALE, "indexed": _INDEXED}

#: :data:`asein._BLEND_BY_INDEX`, inverted. Built from that tuple rather than
#: written out, because two hand-maintained copies of nineteen names are two
#: chances to file ``divide`` under ``subtract``'s number and produce files that
#: only this build reads correctly.
_BLEND_INDEX = {name: index for index, name in enumerate(_BLEND_BY_INDEX)}

#: :data:`asein._TAG_DIRECTIONS`, inverted -- *first* occurrence wins. The
#: fourth entry there is ping-pong-reverse read as an ordinary ping-pong, so
#: taking the last occurrence would write every swing out as the direction this
#: build has no spelling for and cannot read back the same way.
_DIRECTION_INDEX: dict[str, int] = {}
for _index, _name in enumerate(_TAG_DIRECTIONS):
    _DIRECTION_INDEX.setdefault(_name, _index)


# --- the format, as primitives -----------------------------------------------


class _Writer:
    """A growing buffer with the format's own words on it.

    :class:`asein._Reader`'s mirror, and small for the same reason that one is:
    the format is a handful of little-endian integers and a length-prefixed
    string, so the whole of the encoding is six methods and every chunk builder
    below reads as the spec's own field list.
    """

    __slots__ = ("out",)

    def __init__(self) -> None:
        self.out = bytearray()

    def raw(self, data: bytes) -> None:
        self.out += data

    def u8(self, value: int) -> None:
        self.out += struct.pack("<B", int(value))

    def u16(self, value: int) -> None:
        self.out += struct.pack("<H", int(value))

    def i16(self, value: int) -> None:
        self.out += struct.pack("<h", int(value))

    def u32(self, value: int) -> None:
        self.out += struct.pack("<I", int(value))

    def i32(self, value: int) -> None:
        self.out += struct.pack("<i", int(value))

    def string(self, text: str) -> None:
        raw = str(text).encode("utf-8")
        self.u16(len(raw))
        self.out += raw

    def bytes(self) -> bytes:
        return bytes(self.out)


def _string(text: str) -> bytes:
    writer = _Writer()
    writer.string(text)
    return writer.bytes()


def _chunk(kind: int, payload: bytes) -> bytes:
    """One chunk: a DWORD size **including these six bytes**, then the type.

    The inclusive size is the field the reader checks against the frame's own
    end, so getting it wrong is not a chunk that reads short -- it is every
    later chunk in the frame landing on the wrong byte.
    """
    return struct.pack("<IH", len(payload) + 6, kind) + payload


def _frame(chunks: list[bytes], duration: int) -> bytes:
    """One frame header and its chunks. The size counts the 16-byte header.

    Both chunk-count fields are filled the way Aseprite 1.3 fills them: the
    legacy WORD gets ``0xFFFF`` (its "read the other one" value) and the DWORD
    carries the number. The reader takes ``new_count or old_count``, so this is
    the shape that exercises the modern field.

    **An empty frame is the exception and it is not cosmetic.** A frame with no
    chunks -- every track's slot on it empty, which the sparse grid allows --
    would write 0 into the DWORD, and ``new_count or old_count`` would then fall
    through to the legacy field and read 0xFFFF as sixty-five thousand chunks
    that are not there. So a frame with nothing in it says zero in both.
    """
    body = b"".join(chunks)
    return (
        struct.pack(
            "<IHHHHI",
            len(body) + 16,
            _FRAME_MAGIC,
            _MAX_U16 if chunks else 0,
            int(duration),
            0,
            len(chunks),
        )
        + body
    )


def _header(
    *, frames: int, width: int, height: int, depth: int, transparent: int, colours: int
) -> bytes:
    """The 128 bytes everything else hangs off.

    **Bit 0 of the flags is not optional.** It is "layer opacity has a valid
    value", and a file written without it opens with every layer fully opaque
    whatever byte the layer chunk stored -- a picture that composites wrong,
    silently, which is the one failure a writer must not be able to produce by
    omission.

    The declared file size is left at zero here and patched in
    :func:`aseprite_bytes` once the length is known; the reader warns that a
    file may be truncated when the two disagree.
    """
    return (
        struct.pack(
            "<IHHHHHIHIIB3sHBBhhHH",
            0,  # patched: the file's own size
            _MAGIC,
            frames,
            width,
            height,
            depth,
            1,  # flags: bit 0, layer opacity is valid
            DEFAULT_DURATION_MS,  # the deprecated whole-sprite speed
            0,
            0,
            transparent,
            b"\0\0\0",
            colours,
            1,  # pixel width and height: this app is square-pixels-only
            1,
            0,  # grid x, y, width, height -- Aseprite's own defaults
            0,
            16,
            16,
        )
        + b"\0" * 84
    )


# --- the layer list ----------------------------------------------------------


@dataclass
class _Row:
    """One line of the emitted layer list -- a group row or a drawable one.

    Groups and layers are the same chunk in this format, which is why they are
    one type here: the difference is a kind field and a child level, and keeping
    them apart would mean two walks over an order that has to interleave.
    """

    name: str
    child_level: int = 0
    group: bool = False
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    blend: str = "normal"
    continuous: bool = False


def _members(doc) -> list[tuple[int, _Row]]:
    """``(member uid, row)`` bottom-first -- the stack, or the track list.

    The uid is the *group tree's* key and so is the track's uid on an animated
    document, never the materialised cel's: a placeholder layer carries a uid of
    its own, and keying membership on it would un-group every empty slot. That
    is ``Document.member_uids``' trap, restated at the one place outside the
    document that has to walk the same two shapes.
    """
    anim = getattr(doc, "anim", None)
    if anim is None:
        return [
            (
                layer.uid,
                _Row(
                    name=layer.name,
                    visible=bool(layer.visible),
                    locked=bool(layer.locked),
                    opacity=float(layer.opacity),
                    blend=str(layer.blend),
                ),
            )
            for layer in doc.stack
        ]
    return [
        (
            track.uid,
            _Row(
                name=track.name,
                visible=bool(track.visible),
                locked=bool(track.locked),
                opacity=float(track.opacity),
                blend=str(track.blend),
                continuous=bool(track.continuous),
            ),
        )
        for track in anim.tracks
    ]


def _rows(doc) -> tuple[list[_Row], list[int]]:
    """The layer list in file order, and where each stack row landed in it.

    Aseprite nests by *indentation*: a group is a row, and its members are the
    rows after it at a deeper child level. That is sound here for exactly the
    reason :func:`asein._group_tree` can read it back -- a group's leaves are
    contiguous in stack order, which ``groups.py`` maintains -- so each group is
    opened once, on the first member that is inside it, and never reopened.

    The second return value is what the cel chunks need and is the arithmetic
    most easily got wrong: **a cel's layer index counts group rows too**, so a
    group above a track shifts every index below it.

    A member naming a group that is not in ``doc.groups`` is flattened rather
    than refused -- ``resolve`` skips a dangling parent the same way, and a
    layer with a broken ancestry is still a layer.
    """
    from . import groups as gp

    rows: list[_Row] = []
    index_of: list[int] = []
    open_uids: list[int] = []
    for uid, row in _members(doc):
        chain = [
            guid
            for guid in reversed(gp.ancestry(doc.group_of, uid))
            if guid in doc.groups
        ]
        shared = 0
        while (
            shared < len(open_uids)
            and shared < len(chain)
            and open_uids[shared] == chain[shared]
        ):
            shared += 1
        del open_uids[shared:]
        for guid in chain[shared:]:
            node = doc.groups[guid]
            rows.append(
                _Row(
                    name=node.name,
                    child_level=len(open_uids),
                    group=True,
                    visible=bool(node.visible),
                    locked=bool(node.locked),
                )
            )
            open_uids.append(guid)
        index_of.append(len(rows))
        row.child_level = len(open_uids)
        rows.append(row)
    return rows, index_of


def _opacity_byte(value: float) -> int:
    return max(0, min(255, int(round(float(value) * 255.0))))


def _layer_chunk(row: _Row) -> bytes:
    """The 0x2004 chunk.

    Two inversions live here and both are one-bit mistakes that would cost a
    document its meaning. Aseprite stores the *editable* bit where this package
    stores the refusal, so the two are each other's complement -- writing the
    lock straight through would unlock every locked layer and lock every free
    one. And a group's opacity byte is written as 255 on purpose: the reader
    hands every group 1.0 whatever is stored, so any other value would be a
    number nobody can ever read back.
    """
    flags = (
        (_LAYER_VISIBLE if row.visible else 0)
        | (0 if row.locked else _LAYER_EDITABLE)
        | (_LAYER_CONTINUOUS if row.continuous else 0)
    )
    if row.group:
        blend, opacity = 0, 255
    else:
        if row.blend not in _BLEND_INDEX:
            raise ValueError(
                f"the layer {row.name!r} uses the blend mode {row.blend!r}, which"
                " this format has no number for"
            )
        blend, opacity = _BLEND_INDEX[row.blend], _opacity_byte(row.opacity)
    body = struct.pack(
        "<HHHHHHB3s",
        flags,
        _LAYER_GROUP if row.group else _LAYER_IMAGE,
        row.child_level,
        0,  # the default width and height, ignored by the format's own account
        0,
        blend,
        opacity,
        b"\0\0\0",
    )
    return _chunk(_LAYER, body + _string(row.name))


# --- cels --------------------------------------------------------------------


def _plane(layer, mode: str, size: tuple[int, int]) -> bytes:
    """One cel's pixels at the sprite's colour depth. Inverts ``_decode_plane``.

    The grayscale check is a full comparison of two channel pairs rather than a
    strided sample, and it is cheap where it matters: two vectorised passes over
    a plane that is about to be handed to zlib, which costs an order of
    magnitude more. A sample would turn a refusal into a coin flip on exactly
    the document that needs it -- one stray coloured pixel out of a million is
    the shape a bypassed funnel leaves behind.
    """
    width, height = size
    pixels = layer.pixels
    if (pixels.shape[1], pixels.shape[0]) != (width, height):
        raise ValueError(
            f"the cel {layer.name!r} is {pixels.shape[1]}x{pixels.shape[0]} on a"
            f" {width}x{height} canvas, and a cel is canvas-sized here"
        )
    if mode == "rgb":
        return np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()
    if mode == "grayscale":
        grey = pixels[..., 0]
        if not (
            np.array_equal(grey, pixels[..., 1]) and np.array_equal(grey, pixels[..., 2])
        ):
            raise ValueError(
                f"the layer {layer.name!r} holds a pixel that is not grey, which a"
                " grayscale .aseprite has nowhere to put"
            )
        pair = np.empty((height, width, 2), dtype=np.uint8)
        pair[..., 0] = grey
        pair[..., 1] = pixels[..., 3]
        return pair.tobytes()
    indices = getattr(layer, "indices", None)
    if indices is None:
        raise ValueError(
            f"the layer {layer.name!r} is on an indexed document and carries no"
            " index plane to write"
        )
    return np.ascontiguousarray(indices, dtype=np.uint8).tobytes()


def _cel_chunk(layer_index: int, plane: bytes, size: tuple[int, int]) -> bytes:
    """A type-2 (zlib) cel, full canvas at the origin.

    Opacity 255 and z-index 0, both deliberately: opacity is a track property
    here (divergence 1) and track order *is* stack order (divergence 12), so
    writing anything else would be inventing a value the reader would then warn
    about on the way back in.
    """
    width, height = size
    body = struct.pack(
        "<HhhBHh5s", layer_index, 0, 0, 255, _CEL_COMPRESSED, 0, b"\0" * 5
    )
    return _chunk(
        _CEL, body + struct.pack("<HH", width, height) + zlib.compress(plane)
    )


def _link_chunk(layer_index: int, frame_index: int) -> bytes:
    """A type-1 cel: this slot holds the cel that frame ``frame_index`` does.

    The field is a **frame** index, not a cel index -- the reader resolves it as
    ``(this cel's layer, that frame)``, which is why the link is only ever
    written within one track.
    """
    body = struct.pack("<HhhBHh5s", layer_index, 0, 0, 255, _CEL_LINKED, 0, b"\0" * 5)
    return _chunk(_CEL, body + struct.pack("<H", frame_index))


# --- tags and palettes -------------------------------------------------------


def _tags_chunk(tags, frames: int) -> bytes:
    """The 0x2018 chunk, one entry per tag.

    Spans are clamped into the timeline rather than refused: a tag reaching past
    the last frame is already what ``loop_range`` clamps at playback, so writing
    the clamped span is writing what the document plays. The repeat count is the
    opposite case and is refused, because a count past a WORD would wrap to a
    small number and silently stop a clip that was set to run.

    The three colour bytes are zero -- a timeline colour is user data this
    package does not model (divergence 14), and the reader warns about a
    non-zero one, so zero is both the honest value and the one that keeps a
    round trip clean.
    """
    body = struct.pack("<H8s", len(tags), b"\0" * 8)
    last = max(0, frames - 1)
    for tag in tags:
        start = max(0, min(int(tag.start), last))
        end = max(start, min(int(tag.end), last))
        repeat = int(getattr(tag, "repeat", 0) or 0)
        if repeat > _MAX_U16:
            raise ValueError(
                f"the tag {tag.name!r} repeats {repeat} times and an .aseprite"
                f" stores at most {_MAX_U16}"
            )
        body += struct.pack(
            "<HHBH6s3sB",
            start,
            end,
            _DIRECTION_INDEX.get(tag.direction, 0),
            repeat,
            b"\0" * 6,
            b"\0\0\0",
            0,
        ) + _string(tag.name)
    return _chunk(_TAGS, body)


def _palette_chunk(palette: list[tuple[int, ...]]) -> bytes:
    """The modern 0x2019 chunk: the whole table, with alpha."""
    body = struct.pack("<III8s", len(palette), 0, len(palette) - 1, b"\0" * 8)
    for entry in palette:
        red, green, blue, alpha = (int(v) for v in entry)
        body += struct.pack("<HBBBB", 0, red, green, blue, alpha)
    return _chunk(_PALETTE, body)


def _old_palette_chunk(palette: list[tuple[int, ...]]) -> bytes:
    """The pre-1.0 0x0004 chunk, written *beside* the modern one.

    Aseprite's own practice, and the reason it is worth copying is stated in
    :func:`asein._read_old_palette`: a reader that knows only this one still
    gets the colours, and one that knows both prefers the modern chunk, so the
    alpha this one cannot carry is never the table anybody ends up using.

    A 256-colour table writes its count as the byte ``0``, which is this
    packet's own spelling of 256.
    """
    body = struct.pack("<HBB", 1, 0, len(palette) % 256)
    for entry in palette:
        red, green, blue = (int(v) for v in entry[:3])
        body += struct.pack("<BBB", red, green, blue)
    return _chunk(_OLD_PALETTE_256, body)


# --- the file ----------------------------------------------------------------


def aseprite_bytes(doc) -> bytes:
    """``doc`` as a whole ``.aseprite`` file. Pure: no filesystem, no document.

    Split from :func:`write_aseprite` for ``ora_bytes``' reason -- a save that
    goes through a service hands bytes to a task thread and never names a path
    -- and because every refusal below is testable without one.
    """
    width, height = doc.size
    if not 1 <= width <= _MAX_U16 or not 1 <= height <= _MAX_U16:
        raise ValueError(
            f"a canvas of {width}x{height} is past the {_MAX_U16} an .aseprite"
            " stores a side in"
        )
    mode = str(getattr(doc, "color_mode", "rgb"))
    depth = _DEPTHS.get(mode)
    if depth is None:
        raise ValueError(
            f"a document in {mode!r} is not one this writer has a colour depth for"
        )

    anim = getattr(doc, "anim", None)
    frames = 1 if anim is None else len(anim.frames)
    if frames > _MAX_U16:
        raise ValueError(
            f"an .aseprite holds at most {_MAX_U16} frames, not {frames}"
        )
    rows, index_of = _rows(doc)
    if len(rows) > _MAX_U16:
        raise ValueError(
            f"an .aseprite holds at most {_MAX_U16} layers, not {len(rows)}"
        )

    palette = [tuple(entry) for entry in (getattr(doc, "palette", None) or ())]
    if len(palette) > MAX_COLOURS:
        raise ValueError(
            f"an .aseprite palette holds at most {MAX_COLOURS} colours, not"
            f" {len(palette)}"
        )
    if depth == _INDEXED and not palette:
        raise ValueError(
            "an indexed document with no palette has nothing to write its"
            " pixels through"
        )
    transparent = int(getattr(doc, "transparent_index", 0)) if depth == _INDEXED else 0
    if not 0 <= transparent <= 255:
        raise ValueError(
            f"the transparent index {transparent} is not a slot an .aseprite"
            " header can name"
        )

    size = (width, height)
    per_frame: list[list[bytes]] = [[] for _ in range(frames)]
    head = per_frame[0]
    if palette:
        # Both, and in this order: the old chunk first because a reader that
        # only knows it stops at the first table it understands, the modern one
        # after because ours -- and Aseprite's -- lets the later chunk win.
        head.append(_old_palette_chunk(palette))
        head.append(_palette_chunk(palette))
    head.extend(_layer_chunk(row) for row in rows)
    if anim is not None and anim.tags:
        head.append(_tags_chunk(anim.tags, frames))

    if anim is None:
        for position, layer in enumerate(doc.stack):
            per_frame[0].append(
                _cel_chunk(index_of[position], _plane(layer, mode, size), size)
            )
    else:
        for position, track in enumerate(anim.tracks):
            # Per track, because a link names a frame on *this* layer: the same
            # object standing in two tracks (which nothing in the editor
            # produces) has to be written twice rather than linked across.
            first: dict[int, int] = {}
            for index, frame in enumerate(anim.frames):
                layer = anim.cels.get((track.uid, frame.uid))
                if layer is None:
                    # An absent cel is an absent chunk. A full-canvas
                    # transparent one would read back as a slot somebody drew
                    # in, which is a hole in the grid's own sparseness.
                    continue
                at = first.get(id(layer))
                if at is None:
                    first[id(layer)] = index
                    per_frame[index].append(
                        _cel_chunk(index_of[position], _plane(layer, mode, size), size)
                    )
                else:
                    per_frame[index].append(_link_chunk(index_of[position], at))

    body = bytearray(
        _header(
            frames=frames,
            width=width,
            height=height,
            depth=depth,
            transparent=transparent,
            colours=len(palette),
        )
    )
    for index, chunks in enumerate(per_frame):
        duration = (
            DEFAULT_DURATION_MS if anim is None else int(anim.frames[index].duration_ms)
        )
        body += _frame(chunks, duration)
    return struct.pack("<I", len(body)) + bytes(body[4:])


def write_aseprite(doc, path) -> None:
    """The same file, on disk. Blocking; callers encode on a task thread.

    Through a temporary and a replace, ``write_ora``'s idiom: a save that dies
    partway must leave the previous file intact rather than a truncated sprite
    where the user's work was. Every refusal fires inside
    :func:`aseprite_bytes`, so a document this writer cannot store never gets as
    far as opening a file.
    """
    path = Path(path)
    data = aseprite_bytes(doc)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
