"""An Aseprite file as an editable document. Read in, never written back.

``sheetin``'s sibling: another door that turns somebody else's file into one of
these documents, and shaped like that one deliberately -- a pure function from
bytes to a ``Document``, every refusal named, and the result *unsaved but
clean* so the first Ctrl+S is a Save As. That last part is not a convenience
here, it is the whole of what "read-only" means: this module can read a
``.aseprite`` and cannot write one, so a document that kept its source path
would let one Ctrl+S write ORA bytes over the file it came from.

The format is a 128-byte header, then one frame after another, each a list of
length-prefixed chunks. Everything below is stdlib -- ``struct`` and ``zlib``
-- because that is all the format needs and reaching for a decoder would put a
dependency on the package that pins itself as headless.

**What is refused and what is warned about is the decision worth reading.**
Anything that would change what the pixels *mean* is a refusal, by name: a
colour depth this build cannot read, a tilemap layer (its pixels live in a
tileset, and importing the layer without one draws nothing), a cel type nobody
here knows, a link pointing at a cel that is not in the file. A document
assembled out of any of those is one the user edits for ten minutes before
finding out, which is the argument ``sheetin`` makes for refusing a
mis-registered atlas.

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

from . import index_plane as ixp
from .animation import Animation, Frame, Tag, Track
from .layers import Layer, LayerStack
from .slices import Slice, SliceKey

__all__ = [
    "ASEPRITE_SUFFIXES",
    "AseCel",
    "AseLayer",
    "AseSlice",
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
    continuous: bool = False
    child_level: int = 0


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


class _Parse:
    """The mutable half of a parse: the sprite being filled and its warnings."""

    def __init__(self) -> None:
        self.sprite = Sprite()
        self.palette: dict[int, RGBA] = {}
        self.palette_size = 0
        self.old_palette: list[RGBA] | None = None
        self.frame = 0

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
        # Refused rather than warned: a tileset is only ever referenced by a
        # tilemap layer, whose pixels are undrawable without it.
        raise ValueError(
            "this .aseprite uses tilesets, which this build cannot open"
        )
    elif kind == _USER_DATA:
        # Aseprite 1.3 writes one of these after the tags chunk for *every*
        # tag, whether or not anything was ever put in it, so an empty one is
        # not something the file lost -- warning about it would put a message
        # on the screen for almost every tagged file and teach the user to
        # ignore the one that matters. Divergence 14 is about the ones with
        # content in them.
        if r.u32():
            state.warn("user data and timeline colours are not kept; the drawing is")
    elif kind == _CEL_EXTRA:
        state.warn("a cel's precise bounds were dropped; its pixels were not")
    elif kind == _EXTERNAL_FILES:
        state.warn("references to external files were dropped")
    elif kind in (_MASK, _PATH):
        state.warn("a saved mask or path was dropped; selections do not travel")
    else:
        state.warn(f"a chunk of type 0x{kind:04x} is not one this build reads")


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

    if kind == _LAYER_TILEMAP:
        raise ValueError(
            f"layer {name!r} is a tilemap, which this build cannot open"
        )
    if kind not in (_LAYER_IMAGE, _LAYER_GROUP):
        raise ValueError(f"layer {name!r} is of a kind this build does not know")
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
    if kind == _LAYER_IMAGE:
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
            visible=bool(flags & _LAYER_VISIBLE),
            # Aseprite stores the *editable* bit; ours is the refusal, so the
            # two are each other's inverse.
            locked=not (flags & _LAYER_EDITABLE),
            opacity=(opacity / 255.0) if opacity_valid else 1.0,
            blend=mode,
            group=kind == _LAYER_GROUP,
            background=bool(flags & _LAYER_BACKGROUND),
            continuous=bool(flags & _LAYER_CONTINUOUS),
            child_level=child_level,
        )
    )
    if flags & _LAYER_REFERENCE:
        state.sprite.layers[-1].visible = False


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
            try:
                raw = zlib.decompress(raw)
            except zlib.error as exc:
                raise ValueError(
                    f"a cel on layer {layer} will not decompress: {exc}"
                ) from exc
        cel.data = raw
    elif kind == _CEL_TILEMAP:
        raise ValueError(
            "this .aseprite has a tilemap cel, which this build cannot open"
        )
    else:
        raise ValueError(f"cel type {kind} is not one this build knows")
    state.sprite.cels.append(cel)


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
        state.palette[index] = (red, green, blue, alpha)


def _read_old_palette(state: _Parse, r: _Reader, six_bit: bool) -> None:
    """The pre-1.0 palette chunks, kept only until a modern one turns up.

    Both are read because a file can carry the old one alone -- and both are
    *superseded* by ``0x2019`` when it is present, which is the format's own
    instruction and matters because Aseprite writes both into every file it
    saves. Preferring the old one would cost every palette its alpha.
    """
    packets = r.u16()
    table: list[RGBA] = list(state.old_palette or [])
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
            table[index] = (red, green, blue, 255)
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


def _decode(cel: AseCel, sprite: Sprite, lut: np.ndarray | None) -> np.ndarray:
    """One cel's tight rectangle as ``(h, w, 4)`` uint8."""
    width, height = cel.width, cel.height
    if width < 1 or height < 1:
        return np.zeros((0, 0, 4), dtype=np.uint8)
    depth = sprite.depth
    per_pixel = {_RGBA: 4, _GRAYSCALE: 2, _INDEXED: 1}[depth]
    wanted = width * height * per_pixel
    if len(cel.data) != wanted:
        raise ValueError(
            f"a cel on layer {cel.layer} holds {len(cel.data)} bytes where its"
            f" {width}x{height} rectangle needs {wanted}"
        )
    flat = np.frombuffer(cel.data, dtype=np.uint8)
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
            f"a cel names palette entry {top} and the palette has"
            f" {len(sprite.palette or ())} colours"
        )
    return lut[indices]


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


def _build_cels(sprite: Sprite, warn: Callable[[str], None]) -> dict[tuple[int, int], Layer]:
    """Every slot's ``Layer``, with linked slots holding **one** object.

    That sharing is the whole reason this import is worth more than an export
    of PNGs: ``cels[(t, f1)] is cels[(t, f2)]`` *is* a link in this model, so an
    Aseprite link arrives as a link and an edit to it shows on every frame it
    occupies with nothing to propagate.
    """
    by_slot = {(cel.layer, cel.frame): cel for cel in sprite.cels}
    size = (sprite.width, sprite.height)
    palette = sprite.palette or []
    transparent = sprite.transparent_index
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
        made[key] = Layer(pixels=plane, name=layer.name, indices=indices)

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
    transparent = sprite.transparent_index
    if not 0 <= transparent < len(palette):
        transparent = 0

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

    The document is **unsaved but clean** and its format is ``ora``: this
    reader cannot write the format it just read, so the path is deliberately
    dropped and the first Ctrl+S is a Save As into a format that can hold what
    was imported.
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

    width, height = sprite.width, sprite.height
    image_rows = [i for i, layer in enumerate(sprite.layers) if not layer.group]
    made = _build_cels(sprite, warn)

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
        if sprite.tags:
            warn("a tag over a single frame was dropped; there is nothing to play")
        layers: list[Layer] = []
        for index in image_rows:
            row = sprite.layers[index]
            layer = made.get((index, 0)) or Layer.empty(width, height, row.name)
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

    nodes, group_of = _group_tree(sprite, member_uids)
    # ``ora``'s installer and not a second one: what it does past assigning the
    # two dictionaries is prune the groups nothing landed in, and an Aseprite
    # file can hold an empty folder exactly as an ORA can hold one whose PNGs
    # are missing. One mechanism, one set of rules about what a group is.
    _install_groups(doc, (nodes, group_of), {})

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
