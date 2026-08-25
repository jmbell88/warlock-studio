"""An Aseprite file as an editable document. Read in, not written back *here*.

``sheetin``'s sibling: another door that turns somebody else's file into one of
these documents, and shaped like that one deliberately -- a pure function from
bytes to a ``Document``, every refusal named, and the result *unsaved but
clean* so the first Ctrl+S is a Save As. That last part used to be the whole
of what "read-only" meant, back when nothing in the package could write this
format at all; ``aseout.write_aseprite`` retired that half, and Inker's Save
As can now put an edited drawing back into ``.aseprite`` on purpose (see
``inker_mode.save_as``). What did **not** change is this module's own
behaviour: it still only reads, and the document it hands back still drops
its source path -- not because writing is impossible any more, but because an
import is not the moment to decide where a save goes. A document that kept
its source path would let the very next Ctrl+S write over the file it came
from with whatever the editor's default happens to be, silently, before the
user has touched a pixel. The path comes back deliberately empty so that
decision -- ORA, or now Aseprite -- is the Save As dialog's to ask.

The format is a 128-byte header, then one frame after another, each a list of
length-prefixed chunks. Everything below is stdlib -- ``struct`` and ``zlib``
-- because that is all the format needs and reaching for a decoder would put a
dependency on the package that pins itself as headless.

**What is refused and what is warned about is the decision worth reading.**
Anything that would change what the pixels *mean* is a refusal, by name: a
colour depth this build cannot read, a cel type nobody here knows, a link
pointing at a cel that is not in the file. A document assembled out of any of
those is one the user edits for ten minutes before finding out, which is the
argument ``sheetin`` makes for refusing a mis-registered atlas. Tilemap
layers, tilesets and tilemap cels (Wave 3 chunk 3.5) used to be three more
refusals in this list -- the invariant that governs the change is that a
refusal only ever moves because this reader learned to model the thing, and
the refusals that replaced them are exactly as named: a tileset linking an
external file (its pixels are not in this file at all), a tilemap cel that is
not 32 bits per tile, and a tilemap cel sitting at an offset that is not a
multiple of its own tile size.

Everything *cosmetic* is a warning and the document opens: a colour profile
(this app is sRGB-assumed end to end), user data and timeline colours, a cel
opacity (opacity is a track property here), a cel z-index (track order **is**
stack order, which the compositor and the native kernel both assume). Those
are declared divergences rather than gaps, so refusing on them would refuse
almost every file Aseprite writes.

Three mappings are exact and are the reason this import is worth having at all.
An Aseprite **linked cel** is two frames sharing one image, which is exactly
this package's "two slots holding one object" -- so a link arrives as a link
and an edit to it still shows on every frame it occupies. A **tag** is a named
inclusive span with a direction and a repeat count, field for field. And a
**group layer** is a contiguous run of the layer list at a deeper child level,
which is precisely the invariant ``groups.py`` maintains -- a group's leaves are
contiguous in stack order -- so the tree maps over with nothing invented.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..tilegrid import gid
from ..tilegrid.tileset import Tileset
from . import index_plane as ixp
from .animation import Animation, Frame, Tag, Track
from .layers import Layer, LayerStack
from .slices import Slice, SliceKey
from .tiles import TilemapCel, TilesetSlot, grid_shape, materialize

__all__ = [
    "ASEPRITE_SUFFIXES",
    "AseCel",
    "AseLayer",
    "AseSlice",
    "AseTileset",
    "Sprite",
    "document_from_aseprite",
    "parse",
]

#: What this reader opens. Both spellings are the same format -- Aseprite
#: writes ``.aseprite`` and its own older releases wrote ``.ase``.
ASEPRITE_SUFFIXES = (".aseprite", ".ase")

_MAGIC = 0xA5E0
_FRAME_MAGIC = 0xF1FA

# Chunk types, in the spec's own numbering. The ones with no branch below are
# listed anyway, because "a chunk this build ignores" and "a chunk nobody has
# ever heard of" deserve different warnings -- the second is how a file written
# by a newer Aseprite announces itself.
_OLD_PALETTE_256 = 0x0004
_OLD_PALETTE_64 = 0x0011
_LAYER = 0x2004
_CEL = 0x2005
_CEL_EXTRA = 0x2006
_COLOR_PROFILE = 0x2007
_EXTERNAL_FILES = 0x2008
_MASK = 0x2016
_PATH = 0x2017
_TAGS = 0x2018
_PALETTE = 0x2019
_USER_DATA = 0x2020
_SLICE = 0x2022
_TILESET = 0x2023

#: Aseprite's blend modes by their stored number, mapped onto ours. The lists
#: are the same nineteen modes -- C6 added the seven this package was missing
#: for exactly this reason -- so nothing here is approximated and no file loses
#: a mode on the way in. ``add`` is the one spelling difference: Aseprite calls
#: it "addition".
_BLEND_BY_INDEX = (
    "normal",
    "multiply",
    "screen",
    "overlay",
    "darken",
    "lighten",
    "color-dodge",
    "color-burn",
    "hard-light",
    "soft-light",
    "difference",
    "exclusion",
    "hue",
    "saturation",
    "color",
    "luminosity",
    "add",
    "subtract",
    "divide",
)

#: Tag directions by their stored number. Ping-pong-reverse is the fourth and
#: this build has no spelling for it -- a swing that starts on its *back* leg --
#: so it arrives as an ordinary ping-pong with a warning, which plays the same
#: frames in the same order starting from the other end.
_TAG_DIRECTIONS = ("forward", "reverse", "pingpong", "pingpong")

# Layer chunk flags.
_LAYER_VISIBLE = 1
_LAYER_EDITABLE = 2
_LAYER_BACKGROUND = 8
#: Aseprite's "prefer linked cels", which is what its continuous layer is
#: stored as. Ours is a *copy* rather than a link -- see ``Track.continuous``
#: -- which is the nearest honest reading of it: the alternative is dropping
#: the flag and silently giving the user blank frames where they drew a pose
#: once and expected it held.
_LAYER_CONTINUOUS = 16
_LAYER_REFERENCE = 64

# Layer chunk types.
_LAYER_IMAGE = 0
_LAYER_GROUP = 1
_LAYER_TILEMAP = 2

# Cel chunk types.
_CEL_RAW = 0
_CEL_LINKED = 1
_CEL_COMPRESSED = 2
_CEL_TILEMAP = 3

#: Colour depths, in bits per pixel.
_RGBA = 32
_GRAYSCALE = 16
_INDEXED = 8

RGBA = tuple[int, int, int, int]

#: Bytes per pixel, keyed by the sprite's declared colour depth. Read by the
#: three bounded-decompress call sites as well as ``_decode``/``_decode_tileset``,
#: which is why it is a module constant rather than the literal each of those
#: used to spell out.
_PER_PIXEL = {_RGBA: 4, _GRAYSCALE: 2, _INDEXED: 1}

#: The absolute ceiling on any one chunk's unpacked pixels, over and above the
#: per-chunk arithmetic below. ``ora.MAX_DECOMPRESSED_BYTES`` verbatim and for
#: its reason -- an ``.aseprite`` is explicitly *anyone's* file, routinely
#: downloaded from asset sites -- and a second line of defence rather than a
#: duplicate: a cel declares its own rectangle, but the rectangle is two u16s,
#: so an honest-looking 65535x65535 RGBA cel still asks for 17 GiB before a
#: single byte is checked. Read from module globals at call time so a test can
#: lower it rather than building a gigabyte.
MAX_DECOMPRESSED_BYTES = 1 << 30


def _inflate(raw: bytes, expected: int, what: str) -> bytes:
    """Unpack one chunk's payload, refusing anything past what it declares.

    **Bounded, and the bound is the chunk's own arithmetic** --
    ``plotter/tmx.py``'s ``_decompress`` verbatim, and this reader was the one
    door in the tree still without it. A bare ``zlib.decompress`` allocates
    whatever the *stream* says rather than whatever the header says: a few
    kilobytes of crafted or simply corrupt ``.aseprite`` inflates to gigabytes,
    and the read that discovers this is the one that has already exhausted
    memory. The size checks this reader already carried run in ``_decode``,
    which is far too late -- they describe the wreckage rather than preventing
    it.

    One byte past ``expected`` is already a file that does not describe itself,
    so the tail is checked rather than the output quietly truncated: that is
    the same refusal ``_decode`` would have raised, only now before the
    allocation instead of after it.
    """
    if expected < 0 or expected > MAX_DECOMPRESSED_BYTES:
        raise ValueError(
            f"{what} declares {expected} bytes of pixels, past the"
            f" {MAX_DECOMPRESSED_BYTES} this build will unpack"
        )
    engine = zlib.decompressobj()
    try:
        out = engine.decompress(raw, expected + 1)
    except zlib.error as exc:
        raise ValueError(f"{what} will not decompress: {exc}") from exc
    if len(out) > expected or engine.unconsumed_tail:
        raise ValueError(
            f"{what} unpacks past the {expected} bytes its own header declares"
        )
    return out


class _Reader:
    """A cursor over bytes that refuses rather than returning short.

    Every ``take`` past the end is a truncated file, and the one thing a parser
    of somebody else's format must never do with one is guess: a short read
    silently becomes a zero, which becomes a layer index, which becomes pixels
    on the wrong row.
    """

    __slots__ = ("at", "data")

    def __init__(self, data: bytes, at: int = 0) -> None:
        self.data = data
        self.at = at

    def take(self, count: int) -> bytes:
        end = self.at + count
        if count < 0 or end > len(self.data):
            raise ValueError(
                "this .aseprite ends in the middle of a structure it declared"
            )
        out = self.data[self.at : end]
        self.at = end
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return int(struct.unpack("<H", self.take(2))[0])

    def i16(self) -> int:
        return int(struct.unpack("<h", self.take(2))[0])

    def u32(self) -> int:
        return int(struct.unpack("<I", self.take(4))[0])

    def i32(self) -> int:
        return int(struct.unpack("<i", self.take(4))[0])

    def string(self) -> str:
        # ``replace`` and not ``strict``: a layer name is a label, and a byte
        # sequence that is not quite UTF-8 must cost a mangled name rather than
        # the file.
        return self.take(self.u16()).decode("utf-8", "replace")

    def rest(self) -> bytes:
        return self.take(len(self.data) - self.at)


@dataclass
class AseLayer:
    """One row of the layer list, in file order (bottom first).

    ``child_level`` is how the format spells nesting: a layer deeper than the
    one before it is inside it. Groups are layers too, which is why the cel
    chunks' layer indices count them.
    """

    name: str = "Layer"
    visible: bool = True
    locked: bool = False
    opacity: float = 1.0
    blend: str = "normal"
    group: bool = False
    background: bool = False
    #: Aseprite's reference layer -- an underlay to trace over. Kept as read
    #: rather than folded into ``visible``, which is a different fact: the
    #: layer opens hidden *because* an export omits it, and the flag says what
    #: kind of layer it is.
    reference: bool = False
    continuous: bool = False
    child_level: int = 0
    #: The tileset chunk id this layer is bound to, or ``None`` for every
    #: layer kind but a tilemap one -- ``_LAYER_TILEMAP``'s own trailing
    #: DWORD, kept as the file's own id rather than resolved to anything yet,
    #: for the reason ``AseCel.data`` stays undecoded: resolving it needs the
    #: tileset table, which is not final until the whole file has been read.
    tileset: int | None = None


@dataclass
class AseTileset:
    """One vertical-strip atlas, still at the sprite's own colour depth.

    The tileset's own ``AseCel.data``: the zlib layer is undone at parse
    time (a truncated stream is a chunk that lied about its own length, the
    same refusal a cel's compressed pixels earn), but the *colour* decode --
    RGBA passthrough, grayscale expansion, or a trip through the palette --
    waits for :func:`document_from_aseprite`, because an indexed strip's
    transparent index is not known until the palette chunk, wherever in the
    file it landed, has been folded into :func:`_final_palette`.
    """

    name: str = "tiles"
    tile_w: int = 1
    tile_h: int = 1
    count: int = 0
    data: bytes = b""


@dataclass
class AseCel:
    """One cel, still as the file holds it -- tight rectangle, own offset."""

    layer: int
    frame: int
    x: int = 0
    y: int = 0
    kind: int = _CEL_RAW
    width: int = 0
    height: int = 0
    data: bytes = b""
    link: int = 0
    #: A tilemap cel's ``(height, width)`` grid of already-remapped
    #: :mod:`..tilegrid.gid` values -- ``width``/``height`` above hold the
    #: grid's own shape *in tiles* for this one kind, not pixels, since a
    #: tilemap cel is never routed through :func:`_decode`. ``None`` for
    #: every other cel kind.
    refs: np.ndarray | None = None


@dataclass
class AseSlice:
    """A named rectangle and the frames it changes on.

    Keyed by frame *index* rather than by uid, because a parse has no timeline
    to hand out uids -- the ORA reader's ``warlock.json`` slices are stored the
    same way and for the same reason.
    """

    name: str = "Slice"
    keys: list[tuple[int, SliceKey]] = field(default_factory=list)


@dataclass
class Sprite:
    """Everything the file said, before any of it is a document.

    Split from :func:`document_from_aseprite` for ``grid_rects``' reason: the
    arithmetic and the refusals are the valuable part and they are testable
    without building a canvas.
    """

    width: int = 0
    height: int = 0
    depth: int = _RGBA
    transparent_index: int = 0
    layers: list[AseLayer] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)
    cels: list[AseCel] = field(default_factory=list)
    tags: list[Tag] = field(default_factory=list)
    slices: list[AseSlice] = field(default_factory=list)
    palette: list[RGBA] | None = None
    warnings: list[str] = field(default_factory=list)
    #: Keyed by the file's own tileset chunk id (0x2023's own DWORD, not a
    #: position) -- an ``AseLayer.tileset`` names one of these.
    tilesets: dict[int, AseTileset] = field(default_factory=dict)


class _Parse:
    """The mutable half of a parse: the sprite being filled and its warnings."""

    def __init__(self) -> None:
        self.sprite = Sprite()
        self.palette: dict[int, RGBA] = {}
        self.palette_size = 0
        self.old_palette: list[RGBA] | None = None
        self.frame = 0
        #: ``None`` outside a run of ``USER_DATA`` chunks following a
        #: ``_TILESET`` chunk; otherwise how many of that run have already
        #: been consumed. The spec's own convention: "After the Tileset
        #: chunk, it could be followed by a user data chunk (empty or not)
        #: and then all the user data chunks of the tiles ordered by tile
        #: index" -- so the *first* ``USER_DATA`` chunk in the run is the
        #: tileset's own (an ordinary owner, the generic warning), and only
        #: the *second and later* ones are per-tile. Every other owner
        #: (layer, cel, tag, slice) already says its own piece through the
        #: divergence warnings at the chunk that names it, so this state
        #: exists only to tell those two tileset-run cases apart.
        self.tileset_ud_run: int | None = None

    def warn(self, text: str) -> None:
        """One line per *kind* of thing dropped, not one per occurrence.

        A file with two hundred cels carrying user data has one thing wrong
        with it as far as this import is concerned, and two hundred identical
        toasts is a way of saying nothing.
        """
        if text not in self.sprite.warnings:
            self.sprite.warnings.append(text)


def parse(data: bytes) -> Sprite:
    """The file, as data. Refuses by name; never repairs.

    Pure: no filesystem, no document, no canvas. What comes back is what the
    chunks said, with the tight cel rectangles still compressed exactly as they
    arrived -- decoding them is the document's business, because whether an
    index means transparent depends on the layer it is on.
    """
    if len(data) < 128:
        raise ValueError("this file is too short to be an Aseprite file")
    head = _Reader(data)
    declared = head.u32()
    if head.u16() != _MAGIC:
        raise ValueError("this is not an Aseprite file")
    state = _Parse()
    sprite = state.sprite
    frames = head.u16()
    sprite.width = head.u16()
    sprite.height = head.u16()
    sprite.depth = head.u16()
    flags = head.u32()
    head.u16()  # deprecated whole-sprite speed; the per-frame duration wins
    head.u32()
    head.u32()
    sprite.transparent_index = head.u8()

    if sprite.depth not in (_RGBA, _GRAYSCALE, _INDEXED):
        raise ValueError(
            f"a colour depth of {sprite.depth} bits is not one this build reads"
        )
    if not frames:
        raise ValueError("an Aseprite file with no frames has nothing to open")
    if sprite.width < 1 or sprite.height < 1:
        raise ValueError(
            f"a canvas of {sprite.width}x{sprite.height} is not one to draw on"
        )
    if declared and declared != len(data):
        state.warn(
            "this .aseprite declares a size other than the file's; it may be "
            "truncated"
        )
    # Bit 1 is "layer opacity has a valid value". Off means every layer is
    # fully opaque whatever byte is stored, which is what an Aseprite before
    # 1.0 wrote -- reading the byte anyway would open those files with their
    # layers at random opacities.
    opacity_valid = bool(flags & 1)

    at = 128
    for index in range(frames):
        at = _read_frame(state, data, at, index, opacity_valid)

    sprite.palette = _final_palette(state)
    return sprite


def _read_frame(
    state: _Parse, data: bytes, at: int, index: int, opacity_valid: bool
) -> int:
    """One frame's header and its chunks. -> where the next frame starts."""
    r = _Reader(data, at)
    size = r.u32()
    if r.u16() != _FRAME_MAGIC:
        raise ValueError(f"frame {index} of this .aseprite is not a frame")
    old_count = r.u16()
    duration = r.u16()
    r.take(2)
    new_count = r.u32()
    count = new_count or old_count
    end = at + size
    if size < 16 or end > len(data):
        raise ValueError(f"frame {index} of this .aseprite runs past the end of it")

    state.sprite.durations.append(duration)
    state.frame = index
    cursor = r.at
    for _ in range(count):
        chunk = _Reader(data, cursor)
        chunk_size = chunk.u32()
        kind = chunk.u16()
        if chunk_size < 6 or cursor + chunk_size > end:
            raise ValueError(
                f"a chunk in frame {index} of this .aseprite runs past the frame"
            )
        _read_chunk(state, kind, data[cursor + 6 : cursor + chunk_size], opacity_valid)
        cursor += chunk_size
    return end


def _read_chunk(state: _Parse, kind: int, payload: bytes, opacity_valid: bool) -> None:
    r = _Reader(payload)
    if kind == _LAYER:
        _read_layer(state, r, opacity_valid)
    elif kind == _CEL:
        _read_cel(state, r)
    elif kind == _TAGS:
        _read_tags(state, r)
    elif kind == _PALETTE:
        _read_palette(state, r)
    elif kind in (_OLD_PALETTE_256, _OLD_PALETTE_64):
        _read_old_palette(state, r, kind == _OLD_PALETTE_64)
    elif kind == _SLICE:
        _read_slice(state, r)
    elif kind == _COLOR_PROFILE:
        _read_color_profile(state, r)
    elif kind == _TILESET:
        _read_tileset(state, r)
    elif kind == _USER_DATA:
        # Aseprite 1.3 writes one of these after the tags chunk for *every*
        # tag, whether or not anything was ever put in it, so an empty one is
        # not something the file lost -- warning about it would put a message
        # on the screen for almost every tagged file and teach the user to
        # ignore the one that matters. Divergence 14 is about the ones with
        # content in them.
        if r.u32():
            if state.tileset_ud_run is not None and state.tileset_ud_run > 0:
                # The spec's own convention (see ``tileset_ud_run``'s
                # docstring): the tileset's own user data, if any, is the
                # *first* chunk in the run, so only the second and later
                # consecutive ones are per-tile -- which Wave 3 deliberately
                # does not model (deferred by the Aseprite parity
                # programme's own non-goals). Named
                # separately from the ordinary user-data warning because "the
                # tiles are kept" is true here and is not the generic
                # sentence's claim.
                state.warn("per-tile properties are not kept; the tiles are")
            else:
                state.warn("user data and timeline colours are not kept; the drawing is")
        if state.tileset_ud_run is not None:
            state.tileset_ud_run += 1
    elif kind == _CEL_EXTRA:
        state.warn("a cel's precise bounds were dropped; its pixels were not")
    elif kind == _EXTERNAL_FILES:
        state.warn("references to external files were dropped")
    elif kind in (_MASK, _PATH):
        state.warn("a saved mask or path was dropped; selections do not travel")
    else:
        state.warn(f"a chunk of type 0x{kind:04x} is not one this build reads")
    # The run tracked by ``tileset_ud_run`` starts fresh at every tileset
    # chunk, continues silently through the ``USER_DATA`` branch above (which
    # already advanced it), and ends at anything else.
    if kind == _TILESET:
        state.tileset_ud_run = 0
    elif kind != _USER_DATA:
        state.tileset_ud_run = None


def _read_tileset(state: _Parse, r: _Reader) -> None:
    """The 0x2023 chunk: one vertical-strip atlas, still at the sprite's own
    colour depth.

    An external-file tileset is refused by name rather than warned about --
    unlike every other divergence in this reader, its pixels are not
    *somewhere else in this file*, they are not in this file at all, and a
    tilemap layer bound to it would draw nothing. ``base_index`` is
    Aseprite's own UI numbering tiles from 1 in its palette-like tileset
    panel; it changes no id this reader stores, so a value other than 1 is a
    warning rather than a refusal.
    """
    tileset_id = r.u32()
    flags = r.u32()
    count = r.u32()
    tile_w = r.u16()
    tile_h = r.u16()
    base_index = r.i16()
    r.take(14)
    name = r.string()
    if flags & 1:
        raise ValueError(
            f"the tileset {name!r} links an external file, which this build"
            " cannot open"
        )
    if not flags & 2:
        raise ValueError(
            f"the tileset {name!r} holds no embedded pixel data this build"
            " can read"
        )
    if count < 1:
        raise ValueError(f"the tileset {name!r} holds no tiles")
    if base_index != 1:
        state.warn(
            "a tileset's tile numbers start somewhere other than 1 in"
            " Aseprite's own panel; this is display-only and every tile id"
            " this reader stores is unaffected"
        )
    length = r.u32()
    compressed = r.take(length)
    # The strip is ``count`` tiles stacked vertically, still at the sprite's
    # depth -- ``_decode_tileset``'s arithmetic exactly, hoisted ahead of the
    # allocation it is meant to bound.
    raw = _inflate(
        compressed,
        count * tile_w * tile_h * _PER_PIXEL[state.sprite.depth],
        f"the tileset {name!r}",
    )
    state.sprite.tilesets[tileset_id] = AseTileset(
        name=name or "tiles", tile_w=tile_w, tile_h=tile_h, count=count, data=raw
    )


def _read_layer(state: _Parse, r: _Reader, opacity_valid: bool) -> None:
    flags = r.u16()
    kind = r.u16()
    child_level = r.u16()
    r.u16()  # default width, ignored by the format's own account
    r.u16()
    blend = r.u16()
    opacity = r.u8()
    r.take(3)
    name = r.string()

    if kind not in (_LAYER_IMAGE, _LAYER_GROUP, _LAYER_TILEMAP):
        raise ValueError(f"layer {name!r} is of a kind this build does not know")
    # A tilemap layer's own trailing field: the tileset it draws through.
    # Read here, immediately after the name, because that is exactly where
    # the format puts it -- only for this one layer kind.
    tileset = r.u32() if kind == _LAYER_TILEMAP else None
    if flags & _LAYER_REFERENCE:
        # A reference layer is an underlay to trace over and Aseprite's own
        # export leaves it out, so opening it *visible* is what would change
        # the picture. It arrives hidden rather than dropped: the pixels are in
        # the file, the user may well want them, and a checkbox is a cheaper way
        # back than reopening in Aseprite would be.
        state.warn(
            f"the reference layer {name!r} opens hidden, as an export leaves it out"
        )
    mode = "normal"
    if kind in (_LAYER_IMAGE, _LAYER_TILEMAP):
        if blend < len(_BLEND_BY_INDEX):
            mode = _BLEND_BY_INDEX[blend]
        else:
            state.warn(
                f"layer {name!r} uses a blend mode this build does not have; it"
                " opens as normal"
            )
    state.sprite.layers.append(
        AseLayer(
            name=name or "Layer",
            # The VISIBLE bit as written, including on a reference layer. This
            # used to be overridden to False for those, on the reasoning that
            # Aseprite opens them hidden -- but ``aseout`` writes ``visible``
            # verbatim beside the REFERENCE flag, so a file *can* say a
            # reference layer is showing, and forcing it hidden here threw away
            # the user's own toggle on the next load with nothing said.
            visible=bool(flags & _LAYER_VISIBLE),
            # Aseprite stores the *editable* bit; ours is the refusal, so the
            # two are each other's inverse.
            locked=not (flags & _LAYER_EDITABLE),
            opacity=(opacity / 255.0) if opacity_valid else 1.0,
            blend=mode,
            group=kind == _LAYER_GROUP,
            background=bool(flags & _LAYER_BACKGROUND),
            reference=bool(flags & _LAYER_REFERENCE),
            continuous=bool(flags & _LAYER_CONTINUOUS),
            child_level=child_level,
            tileset=tileset,
        )
    )


def _read_cel(state: _Parse, r: _Reader) -> None:
    layer = r.u16()
    x = r.i16()
    y = r.i16()
    opacity = r.u8()
    kind = r.u16()
    z_index = r.i16()
    r.take(5)

    if opacity != 255:
        # Divergence 1: opacity is a track property here, per-cel was skipped.
        state.warn("per-cel opacity is not kept; the layer's opacity is")
    if z_index:
        # Divergence 12: track order *is* stack order, which the compositor and
        # the stack kernel both assume.
        state.warn("a cel's z-index was dropped; layer order is stacking order")

    cel = AseCel(layer=layer, frame=state.frame, x=x, y=y, kind=kind)
    if kind == _CEL_LINKED:
        cel.link = r.u16()
    elif kind in (_CEL_RAW, _CEL_COMPRESSED):
        cel.width = r.u16()
        cel.height = r.u16()
        raw = r.rest()
        if kind == _CEL_COMPRESSED:
            # ``_decode``'s arithmetic, hoisted ahead of the allocation.
            raw = _inflate(
                raw,
                cel.width * cel.height * _PER_PIXEL[state.sprite.depth],
                f"a cel on layer {layer}",
            )
        cel.data = raw
    elif kind == _CEL_TILEMAP:
        grid_w = r.u16()
        grid_h = r.u16()
        bits = r.u16()
        if bits != 32:
            raise ValueError(
                f"a tilemap cel on layer {layer} uses {bits} bits per tile,"
                " which this build cannot open"
            )
        id_mask = r.u32()
        x_mask = r.u32()
        y_mask = r.u32()
        d_mask = r.u32()
        r.take(10)
        raw = r.rest()
        wanted = grid_w * grid_h * 4
        decompressed = _inflate(raw, wanted, f"a tilemap cel on layer {layer}")
        if len(decompressed) != wanted:
            # Only the *short* case survives ``_inflate`` now; the long one is
            # refused before the bytes exist.
            raise ValueError(
                f"a tilemap cel on layer {layer} holds {len(decompressed)} bytes"
                f" where its {grid_w}x{grid_h} grid needs {wanted}"
            )
        cel.width, cel.height = grid_w, grid_h
        tiles = np.frombuffer(decompressed, dtype="<u4").reshape(grid_h, grid_w)
        if (tiles == np.uint32(0xFFFFFFFF)).any():
            # The format's *other* empty: when a tileset's "tile ID 0 is empty"
            # flag is off (rare, pre-release Aseprite builds), an erased cell is
            # stored as 0xFFFFFFFF rather than 0. Left alone, the mask
            # arithmetic below would read it as a huge id wearing every flag --
            # so it is translated to this model's own empty, said out loud.
            state.warn("cells stored as 0xffffffff were read as empty")
            tiles = np.where(tiles == np.uint32(0xFFFFFFFF), np.uint32(0), tiles)
        cel.refs = _remap_tile_refs(tiles, id_mask, x_mask, y_mask, d_mask)
    else:
        raise ValueError(f"cel type {kind} is not one this build knows")
    state.sprite.cels.append(cel)


def _remap_tile_refs(
    tiles: np.ndarray, id_mask: int, x_mask: int, y_mask: int, d_mask: int
) -> np.ndarray:
    """A tilemap cel's raw uint32 grid, their declared bit layout onto ours.

    Numerically the identity on every file Aseprite writes today -- its own
    default masks are :data:`~..tilegrid.gid.GID_MASK`/``FLIP_H``/``FLIP_V``/
    ``FLIP_D`` bit for bit -- but written as mask arithmetic on the masks
    *this chunk itself declared* rather than a straight cast, because the
    masks are a field in the file and not a constant this reader assumes: a
    future Aseprite laying its flag bits out differently still lands on our
    encoding correctly, rather than silently scrambling it.
    """
    raw = np.asarray(tiles, dtype=gid.DTYPE)
    out = raw & gid.DTYPE(id_mask)
    for mask, bit in ((x_mask, gid.FLIP_H), (y_mask, gid.FLIP_V), (d_mask, gid.FLIP_D)):
        if mask:
            out = np.where((raw & gid.DTYPE(mask)) != 0, out | gid.DTYPE(bit), out)
    return out.astype(gid.DTYPE)


def _read_tags(state: _Parse, r: _Reader) -> None:
    count = r.u16()
    r.take(8)
    for _ in range(count):
        start = r.u16()
        end = r.u16()
        direction = r.u8()
        repeat = r.u16()
        r.take(6)
        colour = r.take(3)
        r.take(1)
        name = r.string()
        if direction == 3:
            state.warn(
                f"the tag {name!r} plays ping-pong from its far end; it opens"
                " playing ping-pong from its near one"
            )
        if colour != b"\0\0\0":
            state.warn("user data and timeline colours are not kept; the drawing is")
        state.sprite.tags.append(
            Tag(
                name=name or "tag",
                start=start,
                end=end,
                # ``repeat`` decides on its own once it is above zero (C7), and
                # ``loop`` has to be True beside it or a tag set to play three
                # times would stop on its first pass. Aseprite's zero is
                # "forever", which is this model's zero as well.
                loop=True,
                direction=_TAG_DIRECTIONS[direction]
                if direction < len(_TAG_DIRECTIONS)
                else "forward",
                repeat=repeat,
            )
        )


def _read_palette(state: _Parse, r: _Reader) -> None:
    size = r.u32()
    first = r.u32()
    last = r.u32()
    r.take(8)
    state.palette_size = max(state.palette_size, size)
    for index in range(first, last + 1):
        flags = r.u16()
        red, green, blue, alpha = r.u8(), r.u8(), r.u8(), r.u8()
        if flags & 1:
            r.string()
        entry = (red, green, blue, alpha)
        if state.palette.get(index, entry) != entry:
            # Divergence 20: one table per document. A later chunk rewriting an
            # entry an earlier one set is a per-frame palette -- pre-1.0 legacy
            # the format merely tolerates -- and the final table wins, but
            # silently repainting frames the file coloured differently is not
            # something to do without saying so.
            state.warn("per-frame palettes are not kept; the final table is used")
        state.palette[index] = entry


def _read_old_palette(state: _Parse, r: _Reader, six_bit: bool) -> None:
    """The pre-1.0 palette chunks, kept only until a modern one turns up.

    Both are read because a file can carry the old one alone -- and both are
    *superseded* by ``0x2019`` when it is present, which is the format's own
    instruction and matters because Aseprite writes both into every file it
    saves. Preferring the old one would cost every palette its alpha.

    A later chunk changing an entry an earlier one set is a **per-frame
    palette**, which is divergence 20's pre-1.0 legacy: warned about against
    the table as it stood when this chunk began (the placeholder rows the loop
    below appends are not "set" and must not trip it), and the final table is
    used.
    """
    packets = r.u16()
    previous: list[RGBA] = list(state.old_palette or [])
    table: list[RGBA] = list(previous)
    index = 0
    for _ in range(packets):
        index += r.u8()
        count = r.u8() or 256
        for _ in range(count):
            red, green, blue = r.u8(), r.u8(), r.u8()
            if six_bit:
                red, green, blue = (
                    (red * 255) // 63,
                    (green * 255) // 63,
                    (blue * 255) // 63,
                )
            while len(table) <= index:
                table.append((0, 0, 0, 255))
            entry = (red, green, blue, 255)
            if index < len(previous) and previous[index] != entry:
                state.warn("per-frame palettes are not kept; the final table is used")
            table[index] = entry
            index += 1
    state.old_palette = table


def _read_slice(state: _Parse, r: _Reader) -> None:
    count = r.u32()
    flags = r.u32()
    r.u32()
    name = r.string()
    entry = AseSlice(name=name or "Slice")
    for _ in range(count):
        frame = r.u32()
        x, y = r.i32(), r.i32()
        width, height = r.u32(), r.u32()
        centre = None
        pivot = None
        if flags & 1:
            cx, cy = r.i32(), r.i32()
            cw, ch = r.u32(), r.u32()
            centre = (cx, cy, cx + cw, cy + ch)
        if flags & 2:
            pivot = (float(r.i32()), float(r.i32()))
        if not width or not height:
            # Aseprite hides a slice on a frame by keying it to nothing. There
            # is no hidden state here -- a slice is a note about the drawing --
            # so the rectangle stays where it was and the user is told.
            state.warn(
                f"the slice {name!r} is hidden on some frames; it stays visible here"
            )
        entry.keys.append(
            (
                frame,
                SliceKey(
                    bounds=(x, y, x + width, y + height), pivot=pivot, center=centre
                ),
            )
        )
    if entry.keys:
        state.sprite.slices.append(entry)


def _read_color_profile(state: _Parse, r: _Reader) -> None:
    kind = r.u16()
    flags = r.u16()
    if kind == 2 or flags & 1:
        # Divergence 3: sRGB-assumed bytes end to end. Saying so is the whole
        # of what can be done about it -- the pixels are read exactly as
        # stored, which is right for an sRGB profile and approximate for
        # anything else.
        state.warn("a colour profile was dropped; this app assumes sRGB")


def _final_palette(state: _Parse) -> list[RGBA] | None:
    """The colour table, modern chunk first. None when the file carries none."""
    if state.palette:
        size = max(state.palette_size, max(state.palette) + 1)
        return [state.palette.get(i, (0, 0, 0, 255)) for i in range(size)]
    return state.old_palette or None


# --- from the parse to a document --------------------------------------------


def _lut(palette: list[RGBA], transparent: int | None) -> np.ndarray:
    """A 256-entry RGBA lookup for an indexed cel.

    ``transparent`` is the header's transparent index, and it is passed as
    None for a background layer: Aseprite draws that index as its palette
    colour there and as nothing everywhere else, which is the one place the
    background flag changes pixels rather than chrome.
    """
    table = np.zeros((256, 4), dtype=np.uint8)
    table[:, 3] = 255
    for index, colour in enumerate(palette[:256]):
        table[index] = colour
    if transparent is not None and 0 <= transparent < 256:
        table[transparent] = (0, 0, 0, 0)
    return table


def _decode_plane(
    data: bytes, width: int, height: int, sprite: Sprite, lut: np.ndarray | None
) -> np.ndarray:
    """``data`` -- already known to be exactly one plane's worth of bytes at
    the sprite's own colour depth -- as ``(height, width, 4)`` uint8 RGBA.

    The shared half of :func:`_decode` and :func:`_decode_tileset`: whether
    the bytes are one cel's tight rectangle or a whole vertical tile strip,
    the sprite's depth says the same thing about every byte in either one --
    RGBA passes through, grayscale replicates across three channels, indexed
    goes through the palette. The *size* check stays with each caller, since
    "a cel on layer 3" and "the tileset 'ground'" name what came up short
    differently; this function only ever sees bytes already proven to be the
    right length.
    """
    depth = sprite.depth
    flat = np.frombuffer(data, dtype=np.uint8)
    if depth == _RGBA:
        return flat.reshape(height, width, 4).copy()
    if depth == _GRAYSCALE:
        # The one conversion this reader performs, and it is exact: a grey is
        # rendered as ``(v, v, v)`` by Aseprite too, so replicating the channel
        # loses nothing and costs the file no refusal. Divergence 2 is about
        # what this package *stores*, and it stores RGBA either way.
        pair = flat.reshape(height, width, 2)
        out = np.empty((height, width, 4), dtype=np.uint8)
        out[..., 0] = pair[..., 0]
        out[..., 1] = pair[..., 0]
        out[..., 2] = pair[..., 0]
        out[..., 3] = pair[..., 1]
        return out
    if lut is None:  # pragma: no cover - the caller refuses this one chunk sooner
        raise ValueError("this indexed .aseprite carries no palette to read it with")
    indices = flat.reshape(height, width)
    top = int(indices.max()) if indices.size else 0
    if top >= len(sprite.palette or ()):
        raise ValueError(
            f"this plane names palette entry {top} and the palette has"
            f" {len(sprite.palette or ())} colours"
        )
    return lut[indices]


def _decode(cel: AseCel, sprite: Sprite, lut: np.ndarray | None) -> np.ndarray:
    """One cel's tight rectangle as ``(h, w, 4)`` uint8."""
    width, height = cel.width, cel.height
    if width < 1 or height < 1:
        return np.zeros((0, 0, 4), dtype=np.uint8)
    per_pixel = _PER_PIXEL[sprite.depth]
    wanted = width * height * per_pixel
    if len(cel.data) != wanted:
        raise ValueError(
            f"a cel on layer {cel.layer} holds {len(cel.data)} bytes where its"
            f" {width}x{height} rectangle needs {wanted}"
        )
    return _decode_plane(cel.data, width, height, sprite, lut)


def _decode_tileset(ase: AseTileset, sprite: Sprite, lut: np.ndarray | None) -> np.ndarray:
    """One tileset's raw strip bytes as its RGBA pixels -- ``_decode``'s own
    logic, over a whole vertical strip instead of one cel's tight rectangle.

    The transparent index is drawn as transparent here rather than as a
    background layer's opaque colour: a tileset is not a layer and carries no
    background flag, so the ordinary (non-background) reading is the only one
    that applies -- the same ``lut`` an ordinary layer's cel decodes through.
    """
    width, height = ase.tile_w, ase.tile_h * ase.count
    per_pixel = _PER_PIXEL[sprite.depth]
    wanted = width * height * per_pixel
    if len(ase.data) != wanted:
        raise ValueError(
            f"the tileset {ase.name!r} holds {len(ase.data)} bytes where its"
            f" {ase.count} {ase.tile_w}x{ase.tile_h} tiles need {wanted}"
        )
    return _decode_plane(ase.data, width, height, sprite, lut)


def _decode_indices(cel: AseCel, sprite: Sprite) -> np.ndarray | None:
    """One cel's tight rectangle as a raw ``(h, w) uint8`` index plane.

    The *record*, where :func:`_decode` produces the materialisation. Reading
    both is what stops this importer flattening away the thing an indexed file
    knows and an RGBA plane cannot say: which **slot** each pixel is in. A file
    whose palette holds the same brown twice arrives with the two browns still
    distinguishable, so a later recolour reaches the pixels of one and not the
    other -- and a save puts the file back the way it came.

    None for a non-indexed depth, which is how the caller asks the question.
    """
    if sprite.depth != _INDEXED:
        return None
    width, height = cel.width, cel.height
    if width < 1 or height < 1:
        return np.zeros((0, 0), dtype=np.uint8)
    # Length is already checked by ``_decode``, which every caller runs first.
    return np.frombuffer(cel.data, dtype=np.uint8).reshape(height, width).copy()


def _place(
    tight: np.ndarray, size: tuple[int, int], offset: tuple[int, int], fill: int = 0
) -> tuple[np.ndarray, bool]:
    """A tight rectangle on a canvas-sized plane. -> ``(plane, was clipped)``.

    Clipped rather than refused: Aseprite keeps the pixels a cel has pushed
    off the canvas and draws none of them, so what is lost here is invisible in
    the file too -- but it *is* lost, since layers are canvas-sized in this
    package (``layers.py`` states why), so the caller warns.
    """
    width, height = size
    # ``fill`` rather than ``zeros``: an index plane's empty room is the
    # document's transparent index, which is only slot 0 by coincidence -- the
    # same argument ``transform.resize_canvas`` makes for its own ``fill``.
    shape = (height, width) if tight.ndim == 2 else (height, width, 4)
    plane = np.full(shape, int(fill), dtype=np.uint8)
    tall, wide = tight.shape[:2]
    x, y = offset
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + wide), min(height, y + tall)
    if x1 > x0 and y1 > y0:
        plane[y0:y1, x0:x1] = tight[y0 - y : y1 - y, x0 - x : x1 - x]
    clipped = x < 0 or y < 0 or x + wide > width or y + tall > height
    return plane, clipped


def _resolve_link(cels: dict[tuple[int, int], AseCel], key: tuple[int, int]) -> AseCel:
    """Follow a chain of linked cels to the one holding pixels.

    A chain rather than a single hop because nothing in the format forbids one,
    and a cycle is refused by name rather than hung on -- this is somebody
    else's file.
    """
    seen: set[tuple[int, int]] = set()
    cel = cels[key]
    while cel.kind == _CEL_LINKED:
        if key in seen:
            raise ValueError(
                f"a linked cel on layer {key[0]} of this .aseprite links to itself"
            )
        seen.add(key)
        key = (cel.layer, cel.link)
        found = cels.get(key)
        if found is None:
            raise ValueError(
                f"a cel on frame {cel.frame} links to frame {cel.link}, which"
                f" has no cel on that layer"
            )
        cel = found
    return cel


def _tileset_slot_for(
    layer: AseLayer, tileset_slots: dict[int, TilesetSlot]
) -> TilesetSlot:
    """The slot a tilemap layer's declared tileset id resolves to.

    A layer's ``tileset`` field is the file's own id, not an index into
    anything already built, so this is the one place that lookup happens --
    for a track's binding and for every one of its cels alike.
    """
    slot = tileset_slots.get(layer.tileset)
    if slot is None:
        raise ValueError(
            f"layer {layer.name!r} names a tileset this .aseprite does not define"
        )
    return slot


def _build_tilemap_cel(
    cel: AseCel,
    layer: AseLayer,
    sprite: Sprite,
    tileset_slots: dict[int, TilesetSlot],
    warn: Callable[[str], None],
) -> TilemapCel:
    """One tilemap cel's refs, placed at their own tile offset on a
    canvas-sized grid, and materialized.

    There is no "tight rectangle" for a tile grid the way there is for
    pixels: :attr:`TilemapCel.refs` is always the whole canvas's grid shape,
    so placing a cel is pasting its own grid into that one at
    ``(x // tile_w, y // tile_h)`` -- which is exactly why a non-tile-aligned
    offset is refused rather than clipped: there is no cell for it to land in
    partway through.
    """
    slot = _tileset_slot_for(layer, tileset_slots)
    ts = slot.tileset
    tile_w, tile_h = ts.tile_w, ts.tile_h
    if cel.x % tile_w or cel.y % tile_h:
        raise ValueError(
            f"a tilemap cel on layer {cel.layer} sits at ({cel.x}, {cel.y}),"
            f" which is not a multiple of its {tile_w}x{tile_h} tile size"
        )
    size = (sprite.width, sprite.height)
    grid_h, grid_w = grid_shape(size, tile_w, tile_h)
    refs = gid.empty_layer(grid_w, grid_h)
    src = cel.refs if cel.refs is not None else np.zeros((0, 0), dtype=gid.DTYPE)
    tx, ty = cel.x // tile_w, cel.y // tile_h
    src_h, src_w = src.shape
    x0, y0 = max(0, tx), max(0, ty)
    x1, y1 = min(grid_w, tx + src_w), min(grid_h, ty + src_h)
    if x1 > x0 and y1 > y0:
        refs[y0:y1, x0:x1] = src[y0 - ty : y1 - ty, x0 - tx : x1 - tx]
    if tx < 0 or ty < 0 or tx + src_w > grid_w or ty + src_h > grid_h:
        warn("a cel reaching past the canvas was cropped to it")
    if tile_w != tile_h and (refs & gid.DTYPE(gid.FLIP_D)).any():
        # The refs door's own mask (``_doc_tiles._strip_diagonal``), applied
        # to what the file carries: a diagonal flip of a non-square tile has
        # the wrong footprint, and a commit over one reads neighbour pixels
        # back into the atlas. The placement lands unturned, said out loud.
        warn("diagonal flips on a non-square tileset were dropped")
        refs = refs & gid.DTYPE(0xFFFFFFFF ^ gid.FLIP_D)
    pixels = materialize(refs, ts, size)
    return TilemapCel(pixels=pixels, refs=refs, tileset_uid=slot.uid, name=layer.name)


def _transparent_slot(sprite: Any) -> int:
    """The header's transparent index, clamped into the palette. One answer.

    The byte is whatever the writing tool put there, and a file naming a slot
    past the end of its own palette is a file this reader still has to open.
    ``_install_indexed`` clamped it and ``_build_cels`` did not, so the index
    planes were *filled* with a slot the document's table has no row for:
    ``index_plane.materialize`` clips rather than raises, so the off-cel
    surround opened as an opaque block of the **last** swatch instead of
    transparency, ``check_materialized`` passed because both planes agreed,
    and a save wrote the corruption out.

    One function, called by both, because two clamps are how they came to
    disagree in the first place.
    """
    transparent = int(sprite.transparent_index)
    if not 0 <= transparent < len(sprite.palette or []):
        return 0
    return transparent


def _build_cels(
    sprite: Sprite, warn: Callable[[str], None], tileset_slots: dict[int, TilesetSlot]
) -> dict[tuple[int, int], Layer]:
    """Every slot's ``Layer``, with linked slots holding **one** object.

    That sharing is the whole reason this import is worth more than an export
    of PNGs: ``cels[(t, f1)] is cels[(t, f2)]`` *is* a link in this model, so an
    Aseprite link arrives as a link and an edit to it shows on every frame it
    occupies with nothing to propagate. A tilemap cel is exactly the same
    story one level up: it is a :class:`~.tiles.TilemapCel`, still a
    ``Layer``, so linking it links its ``refs`` for free.
    """
    by_slot = {(cel.layer, cel.frame): cel for cel in sprite.cels}
    size = (sprite.width, sprite.height)
    palette = sprite.palette or []
    transparent = _transparent_slot(sprite)
    luts: dict[bool, np.ndarray] = {}
    if sprite.depth == _INDEXED:
        luts[False] = _lut(palette, transparent)
        luts[True] = _lut(palette, None)

    tight: dict[tuple[int, int], np.ndarray] = {}
    tight_indices: dict[tuple[int, int], np.ndarray] = {}
    made: dict[tuple[int, int], Layer] = {}
    for key, cel in by_slot.items():
        if cel.kind == _CEL_LINKED:
            continue
        if not 0 <= cel.layer < len(sprite.layers):
            # Not a warning: a cel naming a layer that is not in the file is
            # pixels with nowhere to go, and dropping them silently is how a
            # drawing opens missing a limb.
            raise ValueError(
                f"a cel names layer {cel.layer} and this .aseprite has"
                f" {len(sprite.layers)}"
            )
        layer = sprite.layers[cel.layer]
        if layer.group:
            warn("a cel on a group layer was dropped; a group holds no pixels")
            continue
        if layer.tileset is not None:
            made[key] = _build_tilemap_cel(cel, layer, sprite, tileset_slots, warn)
            continue
        pixels = _decode(cel, sprite, luts.get(bool(layer.background)))
        tight[key] = pixels
        plane, clipped = _place(pixels, size, (cel.x, cel.y))
        if clipped:
            warn("a cel reaching past the canvas was cropped to it")
        slots = _decode_indices(cel, sprite)
        indices = None
        if slots is not None:
            tight_indices[key] = slots
            indices, _ = _place(slots, size, (cel.x, cel.y), transparent)
        made[key] = Layer(
            pixels=plane,
            name=layer.name,
            # The two layer types this build gained in 6.5. Read rather than
            # dropped now that there is somewhere for them to land -- and a
            # reference layer's own ``visible`` was already forced off by the
            # reader, which is the honest reading of an export that omits it.
            background=bool(layer.background),
            reference=bool(getattr(layer, "reference", False)),
            indices=indices,
        )

    for key, cel in by_slot.items():
        if cel.kind != _CEL_LINKED:
            continue
        source = _resolve_link(by_slot, key)
        source_key = (source.layer, source.frame)
        if source_key not in made:
            raise ValueError(
                f"a cel on frame {cel.frame} links to frame {cel.link}, which"
                f" has no cel on that layer"
            )
        if (cel.x, cel.y) == (source.x, source.y):
            # The ordinary case, and the valuable one: one object in two slots.
            made[key] = made[source_key]
            continue
        if isinstance(made[source_key], TilemapCel):
            # Aseprite shares a cel's position along with its image the same
            # way for a tilemap cel as for a raster one, so this combination
            # is a file no version of it writes -- and unlike a raster cel's
            # tight rectangle, a tile grid has no pixel-level crop to re-paste
            # at a different offset, only a re-placement in tile units this
            # reader does not attempt for something unreachable in practice.
            raise ValueError(
                "a linked tilemap cel drawn at its own offset is not something"
                " this build can place"
            )
        # A link that moved. Aseprite shares a cel's position along with its
        # image, so this is a file no version of it writes -- but sharing the
        # object would draw the pixels in the wrong place, so the link is what
        # gives way rather than the picture.
        warn("a linked cel drawn at its own offset was unlinked to keep it there")
        plane, clipped = _place(tight[source_key], size, (cel.x, cel.y))
        if clipped:
            warn("a cel reaching past the canvas was cropped to it")
        indices = None
        if source_key in tight_indices:
            indices, _ = _place(tight_indices[source_key], size, (cel.x, cel.y), transparent)
        made[key] = Layer(
            pixels=plane, name=sprite.layers[cel.layer].name, indices=indices
        )
    return made


def _install_indexed(doc, sprite: Sprite, warn: Callable[[str], None]) -> None:
    """Make the opened document truly indexed, keeping the file's slot identity.

    The planes are already on the layers; what is left is the colour state and
    one genuine conflict between the two models.

    **The conflict.** Aseprite draws the transparent index as its palette colour
    on a *background* layer and as nothing everywhere else -- the one place that
    flag changes pixels rather than chrome. This package has no background layer
    (divergence 6: it has a document ``matte``) and exactly one transparent
    index, so a single materialisation cannot render the same slot two ways.

    **The resolution is the feature paying for itself.** The identity an index
    plane keeps is precisely what makes the fix possible: append a *duplicate*
    of the transparent slot's colour and re-point the background layers' pixels
    at it. The picture is byte-identical to what Aseprite shows, the document is
    consistently indexed, and the new slot is a real one the user can see and
    recolour. It costs one palette entry, and only when a background layer
    actually paints in that slot.

    A palette already at 256 has no room, so there the pixels become holes and
    the loss is warned about by name rather than left to be discovered.
    """
    palette = [tuple(colour) for colour in (sprite.palette or [])]
    if not palette:  # pragma: no cover - refused far earlier
        return
    transparent = _transparent_slot(sprite)

    backgrounds = {
        index for index, layer in enumerate(sprite.layers) if layer.background
    }
    planes = [
        layer
        for layer in (doc.stack if doc.anim is None else doc.anim.unique_cel_layers())
        if layer.indices is not None
    ]
    # Which layers are backgrounds is known by *row*, and the layers here are
    # addressed by object, so the link is the name the row gave them. Names are
    # unique per Aseprite file in practice and a collision costs only the
    # duplicate-slot repair, never a pixel of an ordinary layer.
    background_names = {sprite.layers[index].name for index in backgrounds}
    opaque = [
        layer
        for layer in planes
        if layer.name in background_names and bool((layer.indices == transparent).any())
    ]

    if opaque and len(palette) < ixp.MAX_COLOURS:
        spare = len(palette)
        palette.append(palette[transparent])
        for layer in opaque:
            layer.indices[layer.indices == transparent] = spare
    elif opaque:
        warn(
            "a background layer painted in the transparent index reads as"
            " transparent here; the palette is full, so no slot could be spared"
        )

    doc.palette = palette
    doc.color_mode = "indexed"
    doc.transparent_index = transparent
    table = doc._index_lut()
    for layer in planes:
        doc._rematerialize(layer, table, notify=False)
    doc.invalidate_all()
    # Re-asked after the materialisation, not before: the caller computed it off
    # the RGBA planes this method has just rewritten, and on a file with a
    # background layer that is exactly the difference between an opaque document
    # and one with holes in it.
    from .document import matte_for

    # Always inferred: .aseprite has no field for the matte and this reader
    # invents none, so absence here is genuinely "nobody said" -- unlike an
    # .ora, which stores the user's answer (``ora.MATTE_ATTR``).
    doc.matte = matte_for(doc.composite)


def _group_tree(sprite: Sprite, member_uids: dict[int, int]) -> tuple[dict, dict]:
    """``({group uid: GroupNode}, {member uid: group uid})`` from child levels.

    The format nests by *indentation*: a layer whose child level is deeper than
    the one before it is inside it, and a group's members are therefore a
    contiguous run of the layer list -- which is exactly the invariant
    ``groups.py`` maintains over the stack, so nothing has to be rearranged.
    """
    from .groups import GroupNode

    nodes: dict[int, GroupNode] = {}
    group_of: dict[int, int] = {}
    open_groups: list[tuple[int, int]] = []  # (child level, group uid)
    for index, layer in enumerate(sprite.layers):
        while open_groups and open_groups[-1][0] >= layer.child_level:
            open_groups.pop()
        parent = open_groups[-1][1] if open_groups else None
        if layer.group:
            node = GroupNode(
                name=layer.name,
                visible=layer.visible,
                # A group in Aseprite carries no opacity of its own -- the UI
                # offers none -- so inventing one from the stored byte would
                # dim a whole folder for a value nobody set.
                opacity=1.0,
                locked=layer.locked,
            )
            nodes[node.uid] = node
            if parent is not None:
                group_of[node.uid] = parent
            open_groups.append((layer.child_level, node.uid))
            continue
        uid = member_uids.get(index)
        if uid is not None and parent is not None:
            group_of[uid] = parent
    return nodes, group_of


def _slices_for(
    sprite: Sprite, frames: list[Frame], warn: Callable[[str], None]
) -> list[Slice]:
    """``Document.slices``, with Aseprite's run-to-the-end keys resolved.

    A key in that format is valid *from* its frame until the next one, where a
    key here overrides one frame. So the first key becomes the slice itself and
    every frame whose applicable key is a later one gets an override -- which is
    the same picture, stored the way this model stores it.
    """
    out: list[Slice] = []
    for entry in sprite.slices:
        keys = sorted(entry.keys, key=lambda pair: pair[0])
        first_frame, base = keys[0]
        if first_frame:
            warn(
                f"the slice {entry.name!r} starts partway through the timeline;"
                " it is shown from the first frame here"
            )
        overrides: dict[int, SliceKey] = {}
        for index, frame in enumerate(frames):
            applies = base
            for at, key in keys:
                if at <= index:
                    applies = key
            if applies is not base:
                overrides[frame.uid] = applies
        out.append(
            Slice(
                name=entry.name,
                bounds=base.bounds,
                pivot=base.pivot,
                center=base.center,
                keys=overrides,
            )
        )
    return out


def document_from_aseprite(
    data: bytes, *, budget: int | None = None
) -> tuple[Any, list[str]]:
    """An Aseprite file as a document, plus what was dropped on the way in.

    The warnings come back rather than being logged here, for the reason every
    refusal in ``sheetin`` is a ``ValueError`` rather than a print: this package
    is headless and has no toast to raise, so whoever opened the file decides
    how to say what happened to it.

    The document is **unsaved but clean** and its format is ``ora``: an import
    never arrives pointing at its own source file, so the path is deliberately
    dropped and the first Ctrl+S is a Save As. ``aseout.write_aseprite`` means
    that dialog can now choose ``.aseprite`` too, but that is the *save's*
    choice, made once the user has actually looked at the document -- not
    this function's, and not the reason ``format="ora"`` is stamped here.
    """
    from .document import Document, matte_for
    from .ora import _install_groups
    from .undo import UNDO_BYTES, UndoStack

    sprite = parse(data)

    def warn(text: str) -> None:
        """``_Parse.warn``'s rule, one phase later: one line per kind."""
        if text not in sprite.warnings:
            sprite.warnings.append(text)

    if sprite.depth == _INDEXED and not sprite.palette:
        raise ValueError("this indexed .aseprite carries no palette to read it with")

    # Wave 3 chunk 3.5: every tileset the file declared, decoded now that the
    # palette (if this is an indexed file) is final -- ``sprite.tilesets``
    # preserves the order the 0x2023 chunks arrived in, so a spare, unbound
    # tileset lands in ``doc.tilesets`` in the same place it would if a
    # tilemap layer had used it (``ora``'s "not garbage" rule, restated for
    # the reader that predates it).
    tileset_lut = (
        _lut(sprite.palette or [], sprite.transparent_index)
        if sprite.depth == _INDEXED
        else None
    )
    tileset_slots: dict[int, TilesetSlot] = {}
    for tileset_id, ase_tileset in sprite.tilesets.items():
        pixels = _decode_tileset(ase_tileset, sprite, tileset_lut)
        real = Tileset(
            name=ase_tileset.name,
            pixels=pixels,
            tile_w=ase_tileset.tile_w,
            tile_h=ase_tileset.tile_h,
        )
        tileset_slots[tileset_id] = TilesetSlot(tileset=real)

    width, height = sprite.width, sprite.height
    image_rows = [i for i, layer in enumerate(sprite.layers) if not layer.group]
    made = _build_cels(sprite, warn, tileset_slots)

    if not image_rows:
        # Every ORA this package opens has at least one layer and so does every
        # document it makes; a file of nothing but folders becomes one empty
        # drawing rather than a refusal.
        warn("this .aseprite has no drawable layers; it opens empty")
        sprite.layers.append(AseLayer(name="Background"))
        image_rows = [len(sprite.layers) - 1]

    frames = [Frame(duration_ms=duration) for duration in sprite.durations]
    animated = len(frames) > 1
    history = UndoStack(UNDO_BYTES if budget is None else budget)

    if animated:
        tracks = {
            index: Track(
                name=sprite.layers[index].name,
                opacity=sprite.layers[index].opacity,
                visible=sprite.layers[index].visible,
                blend=sprite.layers[index].blend,
                locked=sprite.layers[index].locked,
                continuous=sprite.layers[index].continuous,
                tileset_uid=(
                    _tileset_slot_for(sprite.layers[index], tileset_slots).uid
                    if sprite.layers[index].tileset is not None
                    else None
                ),
            )
            for index in image_rows
        }
        cels = {
            (tracks[row].uid, frames[frame].uid): layer
            for (row, frame), layer in made.items()
            if row in tracks and frame < len(frames)
        }
        anim = Animation(
            tracks=[tracks[index] for index in image_rows],
            frames=frames,
            cels=cels,
            tags=list(sprite.tags),
        )
        doc = Document(
            stack=LayerStack(
                anim.layers_for(anim.frames[0], (width, height)), len(image_rows) - 1
            ),
            history=history,
            anim=anim,
            slices=_slices_for(sprite, frames, warn),
        )
        member_uids = {index: tracks[index].uid for index in image_rows}
    else:
        # Divergence 22: a one-frame file has nowhere to be animated in this
        # model (``Document.anim is None`` for a still document), where
        # Aseprite's own document is always a timeline and a "still" sprite is
        # simply one with a single frame in it. A tag declared over that lone
        # frame is dropped -- there is nothing here for it to play across.
        if sprite.tags:
            warn("a tag over a single frame was dropped; there is nothing to play")
        layers: list[Layer] = []
        for index in image_rows:
            row = sprite.layers[index]
            layer = made.get((index, 0))
            if layer is None:
                if row.tileset is not None:
                    # A tilemap layer with no cel at all -- the same shape
                    # ``add_tilemap_layer`` builds for a brand-new one, not a
                    # plain empty ``Layer``.
                    slot = _tileset_slot_for(row, tileset_slots)
                    grid_h, grid_w = grid_shape(
                        (width, height), slot.tileset.tile_w, slot.tileset.tile_h
                    )
                    refs = gid.empty_layer(grid_w, grid_h)
                    layer = TilemapCel(
                        pixels=materialize(refs, slot.tileset, (width, height)),
                        refs=refs,
                        tileset_uid=slot.uid,
                        name=row.name,
                    )
                else:
                    layer = Layer.empty(width, height, row.name)
            layer.name = row.name
            layer.opacity = row.opacity
            layer.visible = row.visible
            layer.blend = row.blend
            layer.locked = row.locked
            layers.append(layer)
        doc = Document(
            stack=LayerStack(layers, len(layers) - 1),
            history=history,
            slices=_slices_for(sprite, frames, warn),
        )
        member_uids = {index: layers[at].uid for at, index in enumerate(image_rows)}

    doc.tilesets.extend(tileset_slots.values())

    nodes, group_of = _group_tree(sprite, member_uids)
    # ``ora``'s installer and not a second one: what it does past assigning the
    # two dictionaries is prune the groups nothing landed in, and an Aseprite
    # file can hold an empty folder exactly as an ORA can hold one whose PNGs
    # are missing. One mechanism, one set of rules about what a group is.
    _install_groups(doc, (nodes, group_of), {})

    # Always inferred: .aseprite has no field for the matte and this reader
    # invents none, so absence here is genuinely "nobody said" -- unlike an
    # .ora, which stores the user's answer (``ora.MATTE_ATTR``).
    doc.matte = matte_for(doc.composite)
    if sprite.depth == _INDEXED:
        # Only for an indexed file. Aseprite writes a palette into every
        # document it saves, RGBA ones included, and adopting that table on an
        # RGBA drawing would silently put the *whole editor* into indexed mode
        # for a table nobody asked to be constrained by.
        _install_indexed(doc, sprite, warn)
    elif sprite.depth == _GRAYSCALE:
        # The expansion ``_decode`` performs is exact -- Aseprite renders a grey
        # as ``(v, v, v)`` too -- so a grayscale file opens as a grayscale
        # *document*, not as an RGB one that happens to be grey. The mode is
        # what makes the next stroke stay grey. Divergence 2's grayscale half:
        # behaviour parity, storage divergence, lossless round trip.
        doc.color_mode = "grayscale"
    doc.file_format = "ora"
    doc.path = None
    return doc, list(sprite.warnings)
