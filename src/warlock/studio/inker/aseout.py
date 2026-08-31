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
every frame the cel occupies. A tilemap cel links the same way and for free:
it is a ``Layer`` too, so one object in two slots is one chunk and one link.

**A tilemap layer is written as a tilemap layer, never as its picture.** A
:class:`~.tiles.TilemapCel`'s ``pixels`` are a *materialisation* of its refs
over a tileset, so flattening them at the door would hand back a file that
draws correctly and can never be edited as tiles again. So the tilesets go out
as 0x2023 chunks (embedded strips, ids in ``doc.tilesets`` order), the layer as
kind 2 carrying its tileset id, and the cel as a type-3 grid of references
under **explicitly declared** masks -- see :func:`_tilemap_cel_chunk` for why
declaring them is the part that matters. The one place the two models genuinely
disagree is an *indexed* document: this package keeps every strip RGBA (the
Wave 3 divergence) where the format stores tileset pixels at the sprite's own
depth, so an indexed document's strips are resolved back through its palette on
the way out. That resolution is ``index_plane.resolve``'s own rule -- alpha
decides first, a visible pixel is placed by its RGB among the slots that are
not the transparent one -- with the nearest match replaced by an exact one, so
a strip authored here always places and one that arrived from outside the
document is refused by name rather than silently repainted.

**What is lost is lost silently, and written down elsewhere.** There is no
warning channel out of a function that returns bytes, and inventing one would
put a toast on a save that did exactly what the user asked. Five things drop:
``alpha_lock`` (Aseprite has no bit for it -- an editing aid, not picture data),
a group's opacity (Aseprite's UI offers a group none, and ``asein._group_tree``
reads one back as 1.0 whatever byte is stored, so writing anything but 255 would
be a number nobody could ever read), an empty group (a group is a *run* of
the layer list here, so one with no members has no run to write), a slice
pivot's fractional part (the format's field is a signed DWORD), and a slice
key's *disagreement* about whether it carries a pivot or a nine-patch centre --
the format stores that presence once per slice, so the first value the slice
carries becomes every unkeyed frame's (:func:`_slice_chunk`; no zero is ever
invented for one). A sixth is not a loss of this writer's making but is worth
the same line: an indexed document's tileset strip is stored one byte per
pixel, so a strip pixel's own alpha becomes its palette slot's -- the same
normalisation every other plane in an indexed document already went through.
Each is a line in ``docs/COMPAT.md``'s Inker/Aseprite part.

Refusals are by name and are the format's own limits, not this build's: more
frames or layers than a 16-bit count can hold, a palette past 256, a canvas past
65535 on a side, a tile past 65535 on a side, a tag repeating more times than
its own WORD can say, a colour mode with no depth to write it at, and -- the
ones that are about a *broken* document rather than a big one -- a grayscale
document holding a **visible** pixel where ``r != g != b``, a tileset strip
holding a *visible* colour an indexed document's palette has no drawable slot
for (either none at all, or only its transparent slot -- the one state
``indexed.snap`` and ``index_plane.resolve`` disagree about), and a tilemap
layer whose binding or whose cel disagrees with itself (a tileset the document
does not have, a cel with no refs, a cel bound to a different atlas than its
layer). The write funnel enforces greyness on every stroke, so a violation is
the constraint having been bypassed, and the alternative to refusing is throwing
two channels away without a word. Visible, because the funnel deliberately
leaves the dead RGB under alpha 0 alone -- see :func:`_plane`, which is where an
unmasked version of that check would have refused to save an ordinary erased
drawing.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..tilegrid import gid
from .animation import DEFAULT_DURATION_MS
from .asein import (
    _BLEND_BY_INDEX,
    _CEL,
    _CEL_COMPRESSED,
    _CEL_LINKED,
    _CEL_TILEMAP,
    _FRAME_MAGIC,
    _GRAYSCALE,
    _INDEXED,
    _LAYER,
    _LAYER_BACKGROUND,
    _LAYER_CONTINUOUS,
    _LAYER_EDITABLE,
    _LAYER_GROUP,
    _LAYER_IMAGE,
    _LAYER_REFERENCE,
    _LAYER_TILEMAP,
    _LAYER_VISIBLE,
    _MAGIC,
    _OLD_PALETTE_256,
    _PALETTE,
    _RGBA,
    _SLICE,
    _TAG_DIRECTIONS,
    _TAGS,
    _TILESET,
)
from .index_plane import MAX_COLOURS, OPAQUE_THRESHOLD

__all__ = ["aseprite_bytes", "write_aseprite"]

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

#: The 0x2023 flags this writer sets: the tileset's pixels are **in this file**
#: (2), and **tile ID 0 is the empty tile** (4). Bit 0 is the external-file
#: link :func:`asein._read_tileset` refuses by name, and nothing here can
#: produce one -- a ``TilesetSlot`` holds a whole frozen atlas, not a path.
_TILESET_EMBEDDED = 2
#: Flag 4 states this package's own convention (``tiles.py``: gid 0 *is* the
#: required blank tile). Without it, real Aseprite treats the file as the
#: pre-release layout whose empty cell is 0xFFFFFFFF -- so a re-save there
#: could store erased cells as a value :func:`asein._remap_tile_refs`'s mask
#: arithmetic would read as a huge id wearing every flag.
_TILESET_ZERO_EMPTY = 4

#: Aseprite's own tile numbering in its tileset panel starts at 1. It changes
#: no id either half of this pair stores, and the reader *warns* about any
#: other value, so this is both the honest number and the one that keeps a
#: round trip's warning list empty.
_TILESET_BASE_INDEX = 1

#: A tilemap cel's bits per tile. The only width this build reads
#: (:func:`asein._read_cel` refuses the rest by name) and the only one
#: :data:`gid.DTYPE` has room for.
_TILE_BITS = 32

#: The slice chunk's two flags, in the spec's own order.
_SLICE_NINE_PATCH = 1
_SLICE_PIVOT = 2


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

    **Both count fields carry the number**, which is Aseprite's own practice:
    the DWORD always, and the legacy WORD saturated at ``0xFFFF`` -- its "read
    the other one" value, and the only thing it can say about a count it cannot
    hold. The reader takes ``new_count or old_count``, so the modern field is
    still the one that decides.

    Writing ``0xFFFF`` into the WORD unconditionally is the near miss, and it is
    a real one: a frame with **no chunks at all** -- every track's slot on it
    empty, which the sparse grid allows -- puts 0 in the DWORD, and
    ``new_count or old_count`` then falls through to the legacy field and reads
    the sentinel as sixty-five thousand chunks that are not there. Saturating
    rather than sentinelling makes that case correct at the root instead of by a
    special case, and leaves the WORD readable by anything that only knows it.
    """
    body = b"".join(chunks)
    return (
        struct.pack(
            "<IHHHHI",
            len(body) + 16,
            _FRAME_MAGIC,
            min(len(chunks), _MAX_U16),
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
    #: The two layer types of 6.5. Written into the chunk's own flags, which is
    #: the whole point of having them as a model: divergence #6 said this build
    #: had "no real background-layer type to carry the flag on", and it has one.
    background: bool = False
    reference: bool = False
    locked: bool = False
    opacity: float = 1.0
    blend: str = "normal"
    continuous: bool = False
    #: The ``doc.tilesets`` uid this row is bound to, or ``None`` for an
    #: ordinary raster row. Kept beside :attr:`tileset` -- the *file's* id for
    #: the same slot -- because the two answer different questions: the file id
    #: goes in the chunk, and the uid is what a cel's own binding is compared
    #: against before its refs are written under this row's tileset.
    tileset_uid: int | None = None
    tileset: int | None = None


def _tileset_uid_of(doc, track) -> int | None:
    """Which tileset a track draws through, or ``None`` if it is raster.

    ``Track.tileset_uid`` is the authority -- ``add_tilemap_layer`` sets it
    before any cel exists, and ``_ensure_cel_for`` reads it to decide what to
    autovivify -- but a track carrying tilemap cels with the field unset would
    otherwise have its whole grid written out as flattened pixels and reopen
    as a raster layer. So the cels are the fallback, not the source.
    """
    if track.tileset_uid is not None:
        return int(track.tileset_uid)
    anim = doc.anim
    for frame in anim.frames:
        cel = anim.cels.get((track.uid, frame.uid))
        found = getattr(cel, "tileset_uid", None)
        if cel is not None and getattr(cel, "refs", None) is not None and found:
            return int(found)
    return None


def _still_tileset_uid(layer) -> int | None:
    """A still layer's own tileset binding, or ``None`` for an ordinary raster
    layer -- refused **by name** rather than left to crash.

    ``TilemapCel.tileset_uid`` defaults to ``0`` and every construction site in
    this package (``ora.py``, ``asein.py``, ``document.py``, ``_doc_tiles.py``)
    sets it to a real slot uid before the object is ever reachable, so this is
    belt-and-suspenders against a layer built by hand -- a test, a future
    caller -- carrying ``refs`` with no binding at all. Without the check,
    ``int(None)`` raises a bare ``TypeError`` naming nothing; this is
    ``_rows``' own refusal for a *track's* dangling tileset binding, one layer
    earlier, for a still document's layers instead of an animated document's
    tracks.
    """
    if getattr(layer, "refs", None) is None:
        return None
    uid = layer.tileset_uid
    if uid is None:
        raise ValueError(
            f"the tilemap layer {layer.name!r} holds a cel with no tileset"
            " binding at all"
        )
    return int(uid)


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
                    background=bool(getattr(layer, "background", False)),
                    reference=bool(getattr(layer, "reference", False)),
                    locked=bool(layer.locked),
                    opacity=float(layer.opacity),
                    blend=str(layer.blend),
                    tileset_uid=_still_tileset_uid(layer),
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
                background=bool(getattr(track, "background", False)),
                reference=bool(getattr(track, "reference", False)),
                locked=bool(track.locked),
                opacity=float(track.opacity),
                blend=str(track.blend),
                continuous=bool(track.continuous),
                tileset_uid=_tileset_uid_of(doc, track),
            ),
        )
        for track in anim.tracks
    ]


def _rows(doc, tileset_ids: dict[int, int]) -> tuple[list[_Row], list[int]]:
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

    ``tileset_ids`` maps a slot uid onto the id its 0x2023 chunk was written
    under, and a binding it has no entry for is refused **by name** rather
    than flattened: a dangling group parent costs a layer its folder, where a
    dangling tileset would write a layer whose whole picture is a lookup into
    a table the file does not contain.
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
        if row.tileset_uid is not None:
            found = tileset_ids.get(row.tileset_uid)
            if found is None:
                raise ValueError(
                    f"the tilemap layer {row.name!r} draws through a tileset"
                    " this document does not have"
                )
            row.tileset = found
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

    A tilemap row is the same chunk with a different kind and **one trailing
    DWORD after the name** -- the tileset it draws through. After, because
    that is exactly where the format puts it and where :func:`asein._read_layer`
    looks for it; a byte earlier and every later chunk in the frame is read at
    the wrong offset.
    """
    flags = (
        (_LAYER_VISIBLE if row.visible else 0)
        | (0 if row.locked else _LAYER_EDITABLE)
        | (_LAYER_CONTINUOUS if row.continuous else 0)
        | (_LAYER_BACKGROUND if row.background else 0)
        | (_LAYER_REFERENCE if row.reference else 0)
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
    if row.group:
        kind = _LAYER_GROUP
    elif row.tileset is None:
        kind = _LAYER_IMAGE
    else:
        kind = _LAYER_TILEMAP
    body = struct.pack(
        "<HHHHHHB3s",
        flags,
        kind,
        row.child_level,
        0,  # the default width and height, ignored by the format's own account
        0,
        blend,
        opacity,
        b"\0\0\0",
    )
    body += _string(row.name)
    if kind == _LAYER_TILEMAP:
        body += struct.pack("<I", row.tileset)
    return _chunk(_LAYER, body)


# --- pixels, at the sprite's own depth ---------------------------------------


def _grey_pair(pixels: np.ndarray, what: str) -> bytes:
    """RGBA pixels as a grayscale file's ``(value, alpha)`` pairs.

    Shared by a cel's plane and a tileset's strip, because the sprite's depth
    says the same thing about every byte in either one -- and because the
    refusal below must read the same either way, differing only in what it
    names. ``what`` is the whole subject phrase ("the layer 'Ink'"), so the
    sentence is built once here rather than twice at the call sites.
    """
    grey = pixels[..., 0]
    visible = pixels[..., 3] > 0
    if not (
        np.array_equal(grey[visible], pixels[..., 1][visible])
        and np.array_equal(grey[visible], pixels[..., 2][visible])
    ):
        raise ValueError(
            f"{what} holds a visible pixel that is not grey, which a grayscale"
            " .aseprite has nowhere to put"
        )
    pair = np.empty(pixels.shape[:2] + (2,), dtype=np.uint8)
    pair[..., 0] = grey
    pair[..., 1] = pixels[..., 3]
    return pair.tobytes()


def _visible_lookup(palette: list[tuple[int, ...]], transparent: int) -> dict[int, int]:
    """``{packed RGB: slot}`` over the slots a **visible** pixel can land in.

    RGB and not RGBA, and every slot but the transparent one -- which is
    :func:`index_plane.resolve`'s own candidate set, deliberately, because that
    function is what every other plane in an indexed document is resolved
    through and a strip must not be resolved through a different rule. Alpha is
    left out because an indexed sprite has no per-pixel alpha at all: a slot's
    alpha is the slot's, so a pixel's own is not something to match on (it is
    :func:`_resolved_indices`' threshold instead).

    The lowest slot wins a duplicate colour. Deterministic (two saves of an
    unchanged document are byte-identical, ``_slice_json``'s reason) and
    lossless either way: both slots materialise to the same pixel, so the
    picture is the same whichever is written -- which is exactly why a *cel* is
    written from its index plane instead, where the slot is the record.

    A palette whose only colour *is* the transparent slot leaves nothing to
    match against, so the whole table becomes the candidate set:
    :func:`index_plane.resolve` makes the same allowance for the same
    degenerate document, and refusing to save one would be a refusal over a
    picture that has one colour and one meaning for it.
    """
    table: dict[int, int] = {}
    for index, entry in enumerate(palette):
        if index == transparent and len(palette) > 1:
            continue
        red, green, blue = (int(value) for value in tuple(entry)[:3])
        table.setdefault(red | (green << 8) | (blue << 16), index)
    return table


def _resolved_indices(
    pixels: np.ndarray, palette: list[tuple[int, ...]], transparent: int, what: str
) -> bytes:
    """RGBA pixels as the palette slots an indexed sprite stores them in.

    **This is ``index_plane.resolve``'s rule, with its nearest match replaced
    by an exact one**, and both halves of that sentence are load-bearing.

    *Its rule*, because the strip is the one plane in an indexed document that
    is still RGBA (the Wave 3 divergence) and it is constrained by
    ``indexed.snap``, which -- unlike ``resolve`` -- leaves alpha alone and
    returns a fully transparent pixel verbatim. So an ordinary gesture leaves
    real pixels a naive RGBA equality cannot place: a half-coverage dab is
    ``(0, 255, 0, 128)``, and an eraser leaves the colour it cut behind at
    ``(255, 0, 0, 0)``. Both are funnel-legal, and demanding exact RGBA made
    an ordinary document unsaveable over a pixel nobody can see. Alpha decides
    first, at :data:`index_plane.OPAQUE_THRESHOLD` -- below it the pixel *is*
    the hole, whatever colour is left under it -- and a visible pixel is placed
    by its RGB alone. That is precisely what a raster plane in the same
    document does, so the strip normalises the way every other plane does:
    ``lut[resolve(strip)]``, which the round-trip tests compare through.

    *Exact*, because a nearest match is only right where the pixels were
    authored here. A strip that arrived from **outside** the document -- a
    ``.tsx``, Plotter, a tile sheet -- can hold a visible colour this palette
    has no slot for, and snapping it would silently repaint somebody else's
    atlas in this document's colours. So that one case is named, with the
    colour that could not be placed.

    The second refusal is narrower and says so: a visible colour that only the
    **transparent** slot holds. ``indexed.snap`` matches against every slot
    including that one, where ``resolve`` excludes it, so this is the one state
    the two constraints disagree about -- and there is no honest answer to
    give, since writing that slot would turn a pixel the user just painted into
    a hole on the way back in.
    """
    hole = transparent if 0 <= transparent < len(palette) else 0
    out = np.full(pixels.shape[:2], hole, dtype=np.uint8)
    visible = pixels[..., 3] >= OPAQUE_THRESHOLD
    if not visible.any():
        return out.tobytes()

    table = _visible_lookup(palette, hole)
    rgb = pixels[..., :3][visible]
    packed = (
        rgb[:, 0].astype(np.uint32)
        | (rgb[:, 1].astype(np.uint32) << np.uint32(8))
        | (rgb[:, 2].astype(np.uint32) << np.uint32(16))
    )
    unique = np.unique(packed)
    missing = [int(value) for value in unique if int(value) not in table]
    if missing:
        value = missing[0]
        red, green, blue = value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF
        shown = f"#{red:02x}{green:02x}{blue:02x}"
        if tuple(int(v) for v in palette[hole][:3]) == (red, green, blue):
            raise ValueError(
                f"{what} holds the visible colour {shown}, which is this"
                " document's transparent slot's own and so has no slot that"
                " draws"
            )
        raise ValueError(
            f"{what} holds the visible colour {shown}, which this indexed"
            " document's palette has no slot for"
        )
    slots = np.array([table[int(value)] for value in unique], dtype=np.uint8)
    out[visible] = slots[np.searchsorted(unique, packed)]
    return out.tobytes()


def _at_depth(
    pixels: np.ndarray,
    mode: str,
    what: str,
    palette: list[tuple[int, ...]] | None,
    transparent: int,
) -> bytes:
    """``pixels`` at the sprite's colour depth. :func:`asein._decode_plane`'s
    inverse, and the one used for a *tileset strip* -- a cel goes through
    :func:`_plane`, which writes an indexed layer's index plane rather than
    re-resolving its pixels."""
    if mode == "rgb":
        return np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()
    if mode == "grayscale":
        return _grey_pair(pixels, what)
    if not palette:  # pragma: no cover - refused far sooner
        raise ValueError(f"{what} has no palette to write its pixels through")
    return _resolved_indices(pixels, palette, transparent, what)


# --- cels --------------------------------------------------------------------


def _plane(layer, mode: str, size: tuple[int, int]) -> bytes:
    """One cel's pixels at the sprite's colour depth. Inverts ``_decode_plane``.

    The grayscale check is a full comparison of two channel pairs rather than a
    strided sample, and it is cheap where it matters: two vectorised passes over
    a plane that is about to be handed to zlib, which costs an order of
    magnitude more. A sample would turn a refusal into a coin flip on exactly
    the document that needs it -- one stray coloured pixel out of a million is
    the shape a violation has.

    **It is masked to the visible pixels, and that is not a softening of it.**
    ``indexed.grayscale`` returns a fully transparent pixel *verbatim* (its own
    docstring says why: there is no colour under alpha 0 to convert, and
    rewriting it would make a no-op write look like an edit), and the eraser
    cuts alpha while leaving the RGB it was drawn in behind. So paint blue,
    erase it, convert to grayscale, and the document is entirely funnel-legal
    while carrying blue under alpha 0 -- an ordinary document an unmasked check
    would refuse to save. What the mask costs is stated at the write: the two
    stored channels are ``(value, alpha)``, so that dead blue is written as its
    red channel alone and reads back as ``(v, v, v, 0)``. Invisible either way,
    and the round-trip tests compare through exactly that normalisation.
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
        return _grey_pair(pixels, f"the layer {layer.name!r}")
    indices = getattr(layer, "indices", None)
    if indices is None:
        raise ValueError(
            f"the layer {layer.name!r} is on an indexed document and carries no"
            " index plane to write"
        )
    return np.ascontiguousarray(indices, dtype=np.uint8).tobytes()


def _ase_opacity(alpha: float) -> int:
    """A per-cel opacity as the format's byte, 0-255.

    One function because three chunk writers need the same rounding and the
    reader divides by 255 -- ``round`` and not ``int``, so 0.5 writes 128 and
    reads back as 128/255, which is the nearest the format can hold rather than
    a value a floor would drift a step darker on every save.
    """
    return max(0, min(255, round(max(0.0, min(1.0, float(alpha))) * 255.0)))


def _cel_chunk(
    layer_index: int, plane: bytes, size: tuple[int, int], opacity: int = 255
) -> bytes:
    """A type-2 (zlib) cel, full canvas at the origin.

    ``opacity`` is the slot's own (``Animation.cel_opacity``; divergence 1,
    retired 2026-08-30) and defaults to the 255 every undimmed cel writes, so a
    document that never used the feature is byte-for-byte the file it was.
    Z-index stays 0 deliberately: track order *is* stack order here (divergence
    12), so writing anything else would invent a value the reader warns about on
    the way back in.
    """
    width, height = size
    body = struct.pack(
        "<HhhBHh5s", layer_index, 0, 0, opacity, _CEL_COMPRESSED, 0, b"\0" * 5
    )
    return _chunk(
        _CEL, body + struct.pack("<HH", width, height) + zlib.compress(plane)
    )


def _link_chunk(layer_index: int, frame_index: int, opacity: int = 255) -> bytes:
    """A type-1 cel: this slot holds the cel that frame ``frame_index`` does.

    The field is a **frame** index, not a cel index -- the reader resolves it as
    ``(this cel's layer, that frame)``, which is why the link is only ever
    written within one track.

    **It carries an opacity of its own**, and that is what the parameter is
    for: the format gives a linked chunk its own opacity byte, and so does this
    package -- ``Animation.cel_opacity`` is keyed by *slot*, not by ``Layer`` --
    so the two slots of a link round-trip two different numbers over one image.
    """
    body = struct.pack(
        "<HhhBHh5s", layer_index, 0, 0, opacity, _CEL_LINKED, 0, b"\0" * 5
    )
    return _chunk(_CEL, body + struct.pack("<H", frame_index))


def _tilemap_cel_chunk(
    layer_index: int, refs: np.ndarray, opacity: int = 255
) -> bytes:
    """A type-3 cel: a grid of tile references, not pixels.

    Whole-canvas at the origin, for :func:`_cel_chunk`'s reason one level up --
    :attr:`~.tiles.TilemapCel.refs` *is* the canvas's grid here, so there is no
    tight rectangle to crop to and a tile offset invented at the door would
    have ``_build_tilemap_cel``'s "not a multiple of its tile size" refusal
    waiting on the other side of it.

    **The four masks are written out rather than assumed**, and that is the
    field this chunk exists to get right. ``asein._remap_tile_refs`` maps a
    stored word onto our bits using the masks *the chunk itself declares*, so a
    writer that left them zero would have every flipped tile read back as a
    plain one -- and one that wrote a different layout than it packed would
    turn a mirrored tile into a tile id of two billion. Ours are
    :mod:`..tilegrid.gid`'s own four, which are also Aseprite's defaults, so
    the remap is the identity in both directions and the file is one real
    Aseprite reads unchanged.
    """
    grid_h, grid_w = int(refs.shape[0]), int(refs.shape[1])
    body = struct.pack(
        "<HhhBHh5s", layer_index, 0, 0, opacity, _CEL_TILEMAP, 0, b"\0" * 5
    )
    body += struct.pack("<HHH", grid_w, grid_h, _TILE_BITS)
    body += struct.pack("<IIII", gid.GID_MASK, gid.FLIP_H, gid.FLIP_V, gid.FLIP_D)
    body += b"\0" * 10
    # ``<u4`` and not ``gid.DTYPE``: the grid is stored little-endian whatever
    # the machine writing it is, which is the one place the two spellings of
    # uint32 are not the same bytes.
    grid = np.ascontiguousarray(refs, dtype="<u4")
    return _chunk(_CEL, body + zlib.compress(grid.tobytes()))


def _drawn_cel(
    layer_index: int,
    row: _Row,
    layer,
    mode: str,
    size: tuple[int, int],
    opacity: int = 255,
) -> bytes:
    """The chunk one occupied slot writes -- pixels, or tile references.

    A raster cel on a row declared ``_LAYER_TILEMAP`` is refused **by name**
    rather than written as pixels: the layer chunk has already said this layer
    draws through a tileset, so ``_build_cels`` would route the cel through
    ``_build_tilemap_cel``, find no refs on it and hand back an empty grid --
    a drawing that opens blank. The pair of them disagreeing is a broken
    document, and this is the cheapest place to say so.
    """
    if row.tileset is None:
        return _cel_chunk(layer_index, _plane(layer, mode, size), size, opacity)
    refs = getattr(layer, "refs", None)
    if refs is None:
        raise ValueError(
            f"the tilemap layer {row.name!r} holds a cel with no tile grid,"
            " and a tilemap layer's cels are tilemap cels"
        )
    bound = getattr(layer, "tileset_uid", None)
    if bound is not None and int(bound) != row.tileset_uid:
        # One layer, one tileset: the format has a single DWORD for it, so a
        # cel bound elsewhere would silently redraw through this row's atlas.
        raise ValueError(
            f"the tilemap layer {row.name!r} holds a cel bound to a different"
            " tileset, and an .aseprite layer draws through exactly one"
        )
    return _tilemap_cel_chunk(layer_index, refs, opacity)


# --- tilesets ----------------------------------------------------------------


def _strip_bytes(
    ts, mode: str, palette: list[tuple[int, ...]] | None, transparent: int
) -> bytes:
    """One tileset's tiles as the vertical strip the format stores.

    Rebuilt tile by tile through ``Tileset.tile_pixels`` rather than handed
    straight off ``ts.pixels``, and the copy is worth its cost: an atlas that
    came from a ``.tsx`` may have several columns, a margin and a spacing,
    where an ``.aseprite`` tileset is *always* one column with neither -- so
    writing the image verbatim would put a file's own layout into a field that
    declares a different one, and every tile after the first would be read
    from the wrong pixels. For an inker-native strip this is the identity.

    ``count < 1`` is refused by name before ``np.concatenate`` is reached: a
    real :class:`~..tilegrid.tileset.Tileset` cannot actually hold zero tiles
    (its own ``__post_init__`` requires at least one column and one row, and
    ``tiles.blank_strip``/``.strip`` both start a tileset at its required blank
    tile 0), so this is defensive rather than reachable through the studio --
    the alternative, an empty ``np.concatenate([])``, is a bare ``ValueError``
    naming nothing, which is exactly the failure mode a refusal here exists to
    replace.
    """
    count = int(ts.tile_count)
    if count < 1:
        raise ValueError(f"the tileset {ts.name!r} holds no tiles to write")
    stack = np.concatenate(
        [np.asarray(ts.tile_pixels(index)) for index in range(count)], axis=0
    )
    return _at_depth(stack, mode, f"the tileset {ts.name!r}", palette, transparent)


def _tileset_chunk(
    tileset_id: int, ts, mode: str, palette: list[tuple[int, ...]] | None,
    transparent: int,
) -> bytes:
    """The 0x2023 chunk: one embedded vertical-strip atlas.

    The pixels are stored at the **sprite's** depth, not at the model's: an
    8-bit sprite carries an 8-bit strip. Every strip is RGBA in this package
    (the Wave 3 divergence), so an indexed document's strips are resolved back
    through its palette here -- :func:`asein._decode_tileset` reads them out
    through the same table, so the round trip is exact and an off-palette
    pixel is refused by name rather than nearest-matched (see
    :func:`_resolved_indices`).
    """
    tile_w, tile_h = int(ts.tile_w), int(ts.tile_h)
    if tile_w > _MAX_U16 or tile_h > _MAX_U16:
        raise ValueError(
            f"the tileset {ts.name!r} has {tile_w}x{tile_h} tiles and an"
            f" .aseprite stores at most {_MAX_U16} on a side"
        )
    raw = _strip_bytes(ts, mode, palette, transparent)
    compressed = zlib.compress(raw)
    body = struct.pack(
        "<IIIHHh",
        tileset_id,
        _TILESET_EMBEDDED | _TILESET_ZERO_EMPTY,
        int(ts.tile_count),
        tile_w,
        tile_h,
        _TILESET_BASE_INDEX,
    ) + b"\0" * 14
    body += _string(ts.name)
    body += struct.pack("<I", len(compressed)) + compressed
    return _chunk(_TILESET, body)


# --- slices ------------------------------------------------------------------


def _slice_runs(entry, frames) -> list[tuple[int, object]]:
    """``(frame index, key)`` wherever this slice's rectangle *changes*.

    The two models key differently and this is the whole of the conversion. A
    key in this format is valid **from** its frame until the next one; a key
    here overrides exactly one frame. So the emission is run-length over the
    resolved key: frame 0 always, and any later frame whose ``Slice.at``
    differs from the one before it.

    The near miss is the frame that goes *back* to the base. Emitting only the
    frames this model holds an override for would leave that frame inside the
    override's run, and the rectangle would stay changed to the end of the
    clip -- so a return is a run of its own and gets its own key, which is
    what makes ``asein._slices_for`` rebuild the same picture frame for frame.
    """
    runs: list[tuple[int, object]] = []
    previous = None
    for index, frame in enumerate(frames):
        key = entry.at(frame.uid)
        if previous is None or key != previous:
            runs.append((index, key))
        previous = key
    return runs


def _first_set(base, keyed: list):
    """The first value that is not ``None`` -- the slice's own, then its keys'.

    Both the flag test and the per-key fallback in :func:`_slice_chunk`, said
    once: they must be the *same* question, or the flag can be set with nothing
    to write under it.
    """
    if base is not None:
        return base
    return next((value for value in keyed if value is not None), None)


def _slice_chunk(entry, runs: list[tuple[int, object]]) -> bytes:
    """The 0x2022 chunk, one per slice. Inverts :func:`asein._read_slice`.

    The bounds are re-derived on the way out: this package stores a rectangle
    as ``x0 y0 x1 y1`` with the far edge exclusive (``slices.py``'s stated
    convention) and the format stores an origin and a size, so the width is a
    subtraction here and an addition on the way back in.

    **The two flags are a property of the slice, not of a key**, which is the
    one place the models genuinely disagree: ``SliceKey`` carries its pivot and
    its nine-patch centre per frame and may hold neither. The union decides the
    flags -- a slice that carries a pivot anywhere declares one -- and a key
    that lacks what the chunk declares is written with **the first value the
    slice carries** -- its own where it has one, and otherwise the first key's.

    *Never a zero*, and that is the rule rather than a preference: a zero pivot
    is a real pivot as far as the reader is concerned, so filling with one
    would move a sprite's anchor to its corner on every frame nobody keyed --
    metadata invented at the door, which is the thing this docstring is here to
    forbid. Because the flag is set only when *something* carries the datum,
    a fallback always exists and the zero branch does not need to exist.

    What is lost is the *distinction* between "this frame has no pivot of its
    own" and "this frame has the slice's pivot" -- they resolve to the same
    answer through ``Slice.at`` either way -- and, in the base-lacks-key-has
    direction, the base gains the pivot the key set. Losing it the other way
    (declaring the flag only when the base carries it, and dropping a key-only
    pivot entirely) was the alternative and is worse: it throws away a value
    the user explicitly set, where this keeps every value and only widens where
    it applies. One coherent rule -- *a slice's pivot is the first one it
    carries, and every key without one inherits it* -- and no invented number.

    **A zero-size rectangle is written as zero, deliberately, not clamped to
    one pixel.** ``max(0, ...)`` only ever bites when a bound this package
    stores is genuinely empty, and the one way that happens is a slice
    ``document_from_aseprite`` built straight from a *hidden* key in a foreign
    file (``asein._read_slice`` -- a zero-size key is Aseprite's own way of
    keying a slice to nothing, and this reader keeps the rectangle rather than
    inventing a hidden state, with a warning). Clamping to 1x1 here would turn
    that faithfully-carried "nothing" into a fabricated one-pixel box the next
    reader has no reason to think is meaningful; writing zero instead makes
    Aseprite's own reader treat the key exactly as hidden again, which is the
    state the source file actually declared. Every rectangle this package's own
    editing funnel produces is at least 1x1 (``transform.clamp_rect``'s own
    floor), so an ordinary drawing never reaches this branch at all.
    """
    fallback_centre = _first_set(entry.center, [key.center for _, key in runs])
    fallback_pivot = _first_set(entry.pivot, [key.pivot for _, key in runs])
    flags = (_SLICE_NINE_PATCH if fallback_centre is not None else 0) | (
        _SLICE_PIVOT if fallback_pivot is not None else 0
    )
    body = struct.pack("<III", len(runs), flags, 0) + _string(entry.name)
    for index, key in runs:
        x0, y0, x1, y1 = key.bounds
        body += struct.pack(
            "<IiiII", index, x0, y0, max(0, x1 - x0), max(0, y1 - y0)
        )
        if fallback_centre is not None:
            cx0, cy0, cx1, cy1 = (
                key.center if key.center is not None else fallback_centre
            )
            body += struct.pack(
                "<iiII", cx0, cy0, max(0, cx1 - cx0), max(0, cy1 - cy0)
            )
        if fallback_pivot is not None:
            px, py = key.pivot if key.pivot is not None else fallback_pivot
            # Rounded, because the format's field is a signed DWORD and this
            # model's pivot is a float. A fractional pivot is the one thing a
            # slice loses here, and it loses at most half a pixel.
            body += struct.pack("<ii", round(px), round(py))
    return _chunk(_SLICE, body)


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
        if repeat == 0 and not tag.loop:
            # This model's "loop flag decides, and it says no" has no zero to
            # hand Aseprite: its own repeat==0 means forever (divergence #16),
            # and its reader (this package's asein.py) hard-codes loop=True on
            # the way back in, so a bare 0 here would round-trip a "play once"
            # tag into one that never stops. Aseprite's own "play once" is
            # repeat=1, which reads back here as loop=True/repeat=1 -- and
            # ``animation.advance`` forces ``loop`` True under any positive
            # repeat anyway, stopping after the count regardless of the flag --
            # so the two are behaviourally identical on both ends.
            repeat = 1
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


def _document_planes(doc, anim) -> list:
    """Every layer whose pixels this writer is about to encode, once each.

    The same enumeration the cel loops below make -- ``doc.stack`` on a still
    document, the unique cel objects on an animated one -- so a colour this
    finds is a colour that is genuinely in the file, and one it misses is one
    that is not.
    """
    if anim is None:
        return list(doc.stack)
    seen: dict[int, object] = {}
    for layer in anim.cels.values():
        if layer is not None:
            seen.setdefault(id(layer), layer)
    return list(seen.values())


def _derived_palette(doc, anim) -> list[tuple[int, ...]]:
    """A swatch table for a document that carries none, from its own pixels.

    **Aseprite writes a palette into every file it saves and this writer did
    not** -- the Aseprite parity programme's Wave 5 owed item, and the
    only part of it this repository can close on its own. A file with no
    ``0x2019`` chunk is tolerated by this reader and, by reading of the format,
    by Aseprite's own pre-1.0 fallback; but a third-party importer that expects
    the chunk every real ``.aseprite`` carries had nothing to read, and that is
    a parity gap rather than a matter of taste.

    **Nothing is invented.** The obvious alternative -- writing Aseprite's own
    default table -- would mean reciting thirty-two colours from memory into a
    file format, which is exactly the kind of unmeasured claim this repository
    refuses (the tablet-pressure spike is the standing precedent). Every entry
    here is a colour that is actually painted somewhere in the document, so the
    file offers the user the art's own colours to paint with and asserts nothing
    about what Aseprite would have chosen.

    Fully transparent pixels are not colours and are left out; a document with
    no visible pixel at all gets the single transparent entry instead of an
    empty table, so that **every** file this writer produces carries a palette
    chunk rather than every file but one shape of blank.

    Capped at :data:`MAX_COLOURS`, most-used first and ties broken by the colour
    value, then emitted in colour order. Both halves are deterministic on
    purpose: this runs on a document with no stored palette, so it is re-derived
    identically on every save, which is what keeps a re-imported file's second
    save byte-identical to its first (the corpus's fixed-point property -- the
    reader drops a palette on a non-indexed file, so the writer has to be able
    to rebuild the same one from the pixels alone).
    """
    packed: list[np.ndarray] = []
    for layer in _document_planes(doc, anim):
        pixels = getattr(layer, "pixels", None)
        if pixels is None or pixels.size == 0:
            continue
        flat = np.asarray(pixels, dtype=np.uint8).reshape(-1, 4)
        flat = flat[flat[:, 3] != 0]
        if not flat.size:
            continue
        # Packed into one uint32 per pixel rather than counted as rows.
        # ``np.unique(..., axis=0)`` sorts a 2-D array lexicographically and is
        # the wrong tool at canvas sizes -- 2.0s on a single 1024x1024 layer,
        # which a save pays per layer. The pack is exact (four bytes into four
        # byte-lanes, unpacked by the inverse shifts below), so this is a
        # faster spelling of the same answer rather than an approximation of it.
        wide = flat.astype(np.uint32)
        packed.append(
            (wide[:, 0] << 24) | (wide[:, 1] << 16) | (wide[:, 2] << 8) | wide[:, 3]
        )
    if not packed:
        # Every pixel in this document *is* the transparent one, so it is the
        # only honest swatch -- and one entry is what keeps the chunk present.
        return [(0, 0, 0, 0)]
    # One unique over the whole document, not one per layer and a merge: a
    # colour's rank is its share of the *file*, and per-layer tables would have
    # to be summed back together anyway.
    codes, seen = np.unique(np.concatenate(packed), return_counts=True)
    if codes.size > MAX_COLOURS:
        # ``argpartition`` for the cut and a sort only over what survives it --
        # the ranking is a means to the cap, so ordering all 16 million distinct
        # colours a photographic import can hold would be work thrown away.
        # ``-seen`` then ``codes`` reproduces the (count desc, colour asc)
        # tie-break exactly, and the tie-break has to be total or the table
        # would differ between two saves of the same document.
        keep = np.argpartition(-seen, MAX_COLOURS)[: MAX_COLOURS + 1]
        # **Widened to every colour tied at the cut before the sort.**
        # ``argpartition`` guarantees only the pivot's own position, so *which*
        # of the colours sharing the cutoff count land inside that window is
        # unspecified -- and a lower-coded colour that should have won the
        # "ties broken by colour value" rule was sometimes dropped for a
        # higher-coded one with the same count (~3.5% of randomized trials with
        # boundary ties). Taking the full band at the boundary count costs one
        # comparison over ``seen`` and makes the tie-break total again, which is
        # what the docstring above already claimed.
        floor = int(seen[keep].min())
        keep = np.flatnonzero(seen >= floor)
        order = sorted(keep.tolist(), key=lambda i: (-int(seen[i]), int(codes[i])))
        codes = codes[order[:MAX_COLOURS]]
    return sorted(
        (
            int(code >> 24) & 0xFF,
            int(code >> 16) & 0xFF,
            int(code >> 8) & 0xFF,
            int(code) & 0xFF,
        )
        for code in codes.tolist()
    )


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
    tilesets = list(getattr(doc, "tilesets", None) or ())
    # Ids are slot *positions*: the reader keys its own table on the id a
    # chunk declares and hands the slots back in that table's insertion order,
    # so numbering them 0..N-1 in list order is what makes ``doc.tilesets``
    # come back in the order it went out -- and makes a second save of a
    # re-imported document byte-identical.
    tileset_ids = {slot.uid: index for index, slot in enumerate(tilesets)}
    rows, index_of = _rows(doc, tileset_ids)
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
    #: An RGB or grayscale document need not carry a palette, and until now one
    #: that did not wrote no palette chunk at all -- see :func:`_derived_palette`
    #: for why that was a parity gap and why the table is built from the
    #: document's own pixels rather than from a default recited from memory.
    #: Below the indexed refusal above, deliberately: an indexed document with
    #: no palette is refused, never quietly given one derived from its pixels,
    #: because there its palette is what the *stored* indices mean.
    derived = not palette
    if derived:
        palette = _derived_palette(doc, anim)
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
    # Before the layer chunks that name them, which is not a requirement --
    # the reader resolves a layer's tileset id after the whole file is parsed
    # -- but is the order the file reads in.
    head.extend(
        _tileset_chunk(
            index,
            slot.tileset,
            mode,
            palette if depth == _INDEXED else None,
            transparent,
        )
        for index, slot in enumerate(tilesets)
    )
    head.extend(_layer_chunk(row) for row in rows)
    if anim is not None and anim.tags:
        head.append(_tags_chunk(anim.tags, frames))
    for entry in getattr(doc, "slices", None) or ():
        runs = (
            [(0, entry.at(None))] if anim is None else _slice_runs(entry, anim.frames)
        )
        if runs:
            head.append(_slice_chunk(entry, runs))

    if anim is None:
        for position, layer in enumerate(doc.stack):
            per_frame[0].append(
                _drawn_cel(
                    index_of[position], rows[index_of[position]], layer, mode, size
                )
            )
    else:
        for position, track in enumerate(anim.tracks):
            # Per track, because a link names a frame on *this* layer: the same
            # object standing in two tracks (which nothing in the editor
            # produces) has to be written twice rather than linked across.
            row = rows[index_of[position]]
            first: dict[int, int] = {}
            for index, frame in enumerate(anim.frames):
                layer = anim.cels.get((track.uid, frame.uid))
                if layer is None:
                    # An absent cel is an absent chunk. A full-canvas
                    # transparent one would read back as a slot somebody drew
                    # in, which is a hole in the grid's own sparseness.
                    continue
                # Per *slot* and not per layer: a linked cel is one object in
                # two slots and the two may carry two different opacities,
                # which is exactly what ``cel_opacity`` is keyed by slot for.
                alpha = _ase_opacity(anim.cel_alpha(track.uid, frame.uid))
                at = first.get(id(layer))
                if at is None:
                    first[id(layer)] = index
                    per_frame[index].append(
                        _drawn_cel(index_of[position], row, layer, mode, size, alpha)
                    )
                else:
                    per_frame[index].append(
                        _link_chunk(index_of[position], at, alpha)
                    )

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
