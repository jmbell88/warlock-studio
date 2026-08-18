"""What goes into an atlas: one immutable RGBA image with a stable key.

``key`` is the sprite's identity and the packer's total order. It has to be
stable across a repack -- the whole determinism contract in :mod:`.maxrects`
bottoms out in sorting by it -- so it is derived from *where the sprite came
from* rather than from its name: names legitimately repeat, and two layers
called "Layer 1" packing into one slot would be a silent data loss.

**An Inker document is enumerated, never interpreted.** An animated document
gives one sprite per frame, through ``Document.frame_flat``, which is the same
flatten the timeline plays and the onion skin draws -- so a packed frame is
pixel-identical to what the user was looking at. A still document gives one
sprite per layer, hidden layers included: the pane chooses what to include, and
an enumerator that silently dropped rows would make "why is my sprite missing"
a question about two places at once.

**Metadata arrives as plain data and is frozen here.** A sprite can carry a
pivot and a list of named rectangles, which is what lets an atlas say where a
sprite is placed from and which part of a UI panel stretches. Packwright reads
them through one duck-typed method -- ``Document.sprite_meta_for_frame`` -- and
never learns what a frame, a per-frame key or a ``Slice`` is: the resolving is
the drawing editor's business and happens on its side of the call, and what
crosses is dicts and tuples that :func:`sprite_meta` turns into the frozen types
below. That is why this module imports nothing from the raster editor, which its
own package pin requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Shared rather than copied. This was a byte-identical second spelling of
# ``tilegrid.tileset``'s helper, differing only in the noun in its error
# message, which is now a parameter. The edge is already pinned in both
# directions -- ``tsxout`` imports the same module for the .tsx writer -- so
# this adds no dependency the package did not already have.
from ..tilegrid.tileset import frozen_rgba


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """One named rectangle on a sprite, in **source-image** coordinates.

    Source-image and not trimmed: a slice describes the picture the artist drew,
    and trimming is something the packer did afterwards. A consumer that wants
    the trimmed frame already has ``spriteSourceSize`` to subtract, and baking
    the trim in here would make the numbers wrong the moment trimming is turned
    off.

    ``center`` is the stretchable middle of a nine-slice panel, as
    ``(x, y, w, h)`` -- the same spelling as the four fields above it, so a
    reader has one rectangle convention rather than two.
    """

    name: str
    x: int
    y: int
    w: int
    h: int
    pivot: tuple[float, float] | None = None
    center: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class SpriteMeta:
    """Everything about a sprite that is not its pixels.

    A frozen record with defaults, so a sprite that has none is exactly the
    sprite this packer has always had -- and so the layout, the sidecars and the
    document format each get to be additive rather than versioned.
    """

    pivot: tuple[float, float] | None = None
    slices: tuple[SliceSpec, ...] = ()

    def __bool__(self) -> bool:
        return self.pivot is not None or bool(self.slices)


#: The shared "nothing to say" instance. Frozen, so sharing is safe, and one
#: object rather than a default factory keeps ``Sprite`` cheap to build in the
#: overwhelmingly common case.
EMPTY_META = SpriteMeta()


def _point(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    x, y = value  # a wrong length raises here, which is what the caller catches
    return (float(x), float(y))


def _rect(value: Any) -> tuple[int, int, int, int] | None:
    """``(x, y, w, h)``, or None. The unpack is the length check, and it has to
    be one: a three-element ``center`` would sail through a comprehension and
    fail much later, in a sidecar writer that spreads it into four arguments."""
    if value is None:
        return None
    x, y, w, h = value
    return (int(x), int(y), int(w), int(h))


def sprite_meta(raw: Any) -> SpriteMeta:
    """Plain dicts from a document into the frozen record above.

    The coercion lives on *this* side of the call deliberately. The producer is
    free to be a raster editor with per-frame keys or a loose PNG with nothing
    at all, and the packer's types stay the packer's -- which is what keeps this
    package importable with no editor present and its import pin honest.

    Anything missing is simply absent: this is a read of somebody else's data,
    and a metadata field that will not parse must cost the pivot rather than the
    sprite. ``.wpack``'s reader is the one place that refuses instead, because
    there a malformed field means a file that is wrong about itself.
    """
    if not isinstance(raw, dict):
        return EMPTY_META
    slices = []
    for entry in raw.get("slices") or ():
        try:
            slices.append(
                SliceSpec(
                    name=str(entry.get("name", "")),
                    x=int(entry["x"]),
                    y=int(entry["y"]),
                    w=int(entry["w"]),
                    h=int(entry["h"]),
                    pivot=_point(entry.get("pivot")),
                    center=_rect(entry.get("center")),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    try:
        pivot = _point(raw.get("pivot"))
    except (TypeError, ValueError, IndexError):
        pivot = None
    return SpriteMeta(pivot=pivot, slices=tuple(slices))


@dataclass(frozen=True)
class Sprite:
    """One image to be packed. ``name`` is for humans; ``key`` is identity."""

    key: str
    name: str
    pixels: np.ndarray
    #: Trailing and defaulted, so every existing construction site -- and every
    #: caller that builds one from a loose file -- is unchanged.
    meta: SpriteMeta = field(default=EMPTY_META)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixels", frozen_rgba(self.pixels, "a sprite"))
        if not self.key:
            raise ValueError("a sprite needs a key")

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def frame_key(prefix: str, index: int) -> str:
    """One animation frame's key. Zero-padded so a lexical sort is a temporal
    one -- which matters because the packer's canonical order is by key."""
    return f"{prefix}#frame{index:04d}"


def layer_key(prefix: str, index: int, name: str) -> str:
    """One layer's key. The *index* is in it as well as the name, because two
    layers may legitimately share a name and two sprites may not share a key."""
    return f"{prefix}#layer{index:02d}:{name}"


def _meta_of(doc: Any, frame_uid: Any) -> SpriteMeta:
    """One frame's metadata, if the document has any to give.

    Duck-typed rather than isinstance-checked: the packer takes sprites from
    Inker documents, from loose files and from test doubles, and a document
    without the method is one with nothing to say -- not an error.
    """
    read = getattr(doc, "sprite_meta_for_frame", None)
    return EMPTY_META if read is None else sprite_meta(read(frame_uid))


def sprites_from_document(doc: Any, *, prefix: str) -> list[Sprite]:
    """Every sprite an Inker document offers, in its own natural order.

    Animated: one per frame, bottom to top of the timeline. Still: one per
    layer, in stack order, which is bottom-first exactly as the layers panel
    shows it upside down.

    Metadata comes off the document per *frame*, and every layer of a still
    document gets the same one: a slice is a rectangle on the canvas rather than
    on a layer, so a still document's layers share it -- which is also what a
    nine-slice panel drawn on two layers means.
    """
    anim = getattr(doc, "anim", None)
    if anim is not None and anim.frames:
        out = []
        for index, frame in enumerate(anim.frames):
            flat = doc.frame_flat(frame.uid)
            if flat is None:
                # A frame the document declines to flatten is not a frame we
                # can pack; skipping it silently would leave a clip one cell
                # short with nothing to say which one.
                raise ValueError(f"frame {index + 1} of {prefix!r} could not be flattened")
            out.append(
                Sprite(
                    key=frame_key(prefix, index),
                    name=f"{prefix} {index + 1}",
                    pixels=flat,
                    meta=_meta_of(doc, frame.uid),
                )
            )
        return out

    meta = _meta_of(doc, None)
    return [
        Sprite(
            key=layer_key(prefix, index, layer.name),
            name=layer.name or f"{prefix} {index + 1}",
            pixels=layer.pixels,
            meta=meta,
        )
        for index, layer in enumerate(doc.stack)
    ]


def sprite_from_image(pixels: Any, *, key: str, name: str = "") -> Sprite:
    """A loose image file, already decoded. Here rather than in the host so the
    RGBA check and the copy happen in exactly one place."""
    return Sprite(key=key, name=name or key, pixels=pixels)


# --- already-made tile sheets -------------------------------------------------


def tile_key(prefix: str, column: int, row: int) -> str:
    """One grid cell's key. Row-major and zero-padded, so the lexical order the
    packer sorts into is the reading order of the sheet it came from."""
    return f"{prefix}#tile{row:03d}x{column:03d}"


def tileset_occupancy(pixels: Any, *, tile: tuple[int, int]) -> np.ndarray:
    """A ``(rows, columns)`` bool grid: which full cells hold any opaque pixel.

    The one answer both the popup's preview and the import compute their counts
    from -- two implementations of "is this cell empty" is one disagreement
    between the number promised and the number added. Full cells only: a
    remainder strip narrower than a tile is outside the grid, which is what
    Tiled does with the same sheet.
    """
    tile_w, tile_h = int(tile[0]), int(tile[1])
    if tile_w < 1 or tile_h < 1:
        raise ValueError("a tile must be at least 1 x 1")
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("a tile sheet must be RGBA, shaped (h, w, 4)")
    rows, columns = array.shape[0] // tile_h, array.shape[1] // tile_w
    if rows < 1 or columns < 1:
        return np.zeros((0, 0), dtype=bool)
    alpha = array[: rows * tile_h, : columns * tile_w, 3]
    return alpha.reshape(rows, tile_h, columns, tile_w).any(axis=(1, 3))


def sprites_from_tileset(
    pixels: Any, *, tile: tuple[int, int], prefix: str, name: str = ""
) -> list[Sprite]:
    """Slice an already-made tile sheet into one sprite per *occupied* cell.

    An empty cell -- no opaque pixel anywhere in it -- is dropped rather than
    packed, which is the point of re-packing a sheet: what comes out is the
    tiles, on a smaller canvas. Dropping is safe here where it is not for
    animation frames (see the module head): a tileset's cells carry no temporal
    order, and the sheet the pack writes is a new sheet, not an edit of the old
    one. A sheet whose every pixel is opaque -- the usual decode of an RGB
    file -- simply keeps every cell.
    """
    occupied = tileset_occupancy(pixels, tile=tile)
    kept = int(occupied.sum())
    from .layout import MAX_SPRITES

    if kept > MAX_SPRITES:
        raise ValueError(
            f"that slicing makes {kept} tiles; the packer's ceiling is {MAX_SPRITES}"
        )
    tile_w, tile_h = int(tile[0]), int(tile[1])
    array = np.asarray(pixels)
    base = name or prefix
    out: list[Sprite] = []
    for row in range(occupied.shape[0]):
        for column in range(occupied.shape[1]):
            if not occupied[row, column]:
                continue
            y0, x0 = row * tile_h, column * tile_w
            out.append(
                Sprite(
                    key=tile_key(prefix, column, row),
                    name=f"{base} r{row + 1}c{column + 1}",
                    pixels=array[y0 : y0 + tile_h, x0 : x0 + tile_w],
                )
            )
    return out
