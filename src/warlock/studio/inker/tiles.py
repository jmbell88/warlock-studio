"""The tile model: a mutable holder for a frozen atlas, and a cel derived from it.

Two decisions carry the module. **A tileset is edited by frozen-replace, never
in place** -- ``tilegrid.Tileset`` is frozen and its ``pixels`` are read-only
(see :mod:`..tilegrid.tileset`), because the pane's texture cache holds the
strip array it last uploaded and compares identity with ``is``
(``inker_textures.tileset_texture``); an in-place edit would hand back the same
array and redraw nothing. :class:`TilesetSlot` is the thing that *can* change identity across
an edit while still answering to a stable name: the ``uid`` undo addresses,
the ``tileset`` a whole new frozen object each time :func:`grow`,
:func:`shrink` or :func:`with_tiles` runs.

**A tilemap cel's picture is derived, not drawn.** :attr:`TilemapCel.refs` --
a ``(grid_h, grid_w)`` plane of :mod:`..tilegrid.gid`-encoded cell references
-- and the tileset it is bound to are authoritative; :attr:`TilemapCel.pixels`
is the canvas-sized RGBA :func:`materialize` of the two, kept in sync at edit
time by callers above this module. Because ``pixels`` stays honest RGBA, every
existing reader of a ``Layer`` -- the compositor, the flatten cache, the ORA
writer, onion skin, thumbnails -- needs no new case; only the code that
*writes* a tilemap cel needs to know it is one.

Inker tilesets are Aseprite-style **vertical strips**: one column
(``image_w == tile_w``), no spacing, no margin. Local tile id 0 is a real,
required-blank tile, so a ref of 0 and a ref naming local id 0 render
identically -- ``gid.EMPTY == 0`` names both "empty" and "the blank tile" on
purpose, which is what makes ``.aseprite`` import an index-preserving copy and
``.tsx`` export a plain strip (Wave 3, chunks 3.5/3.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace

import numpy as np

from ..tilegrid import gid
from ..tilegrid.tileset import Tileset
from . import composite as cp
from .layers import Layer, new_uid

__all__ = [
    "TilemapCel",
    "TilesetSlot",
    "blank_strip",
    "canonical",
    "content_key",
    "grid_shape",
    "grow",
    "materialize",
    "oriented",
    "shrink",
    "strip",
    "with_tiles",
]


@dataclass
class TilesetSlot:
    """A mutable name over an immutable :class:`~..tilegrid.tileset.Tileset`.

    The frozen tileset is replaced whole on every edit -- see the module
    docstring -- so the identity undo and every track binding actually
    address is the ``uid`` here, never the tileset object, which changes on
    every :func:`grow`/:func:`shrink`/:func:`with_tiles`.
    """

    tileset: Tileset
    uid: int = field(default_factory=new_uid)


@dataclass
class TilemapCel(Layer):
    """A cel whose ``pixels`` are a materialization of ``refs`` over a tileset.

    Subclasses :class:`~.layers.Layer` rather than growing a case onto it,
    which is what lets ``Animation.cels`` keep its ``dict[..., Layer]`` typing
    unchanged -- a tilemap cel *is* a layer, linking it links ``refs`` for
    free (two dict slots holding the same object), and every whole-grid walk
    that already exists needs no new branch to find one.

    ``refs`` and ``tileset_uid`` both default so the dataclass composes
    cleanly onto ``Layer``'s own all-but-``pixels``-defaulted fields -- there
    is no field-ordering conflict to work around. ``refs`` defaulting to
    ``None`` is not "optional", though: :meth:`__post_init__` refuses a
    ``None`` or malformed plane immediately, the same way ``Layer`` refuses a
    malformed ``pixels``. A default is only how Python lets a dataclass field
    exist at all; every real instance is validated.
    """

    refs: np.ndarray | None = None
    tileset_uid: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.refs is None or self.refs.dtype != gid.DTYPE or self.refs.ndim != 2:
            raise ValueError("a tilemap cel holds a (grid_h, grid_w) uint32 refs plane")

    @property
    def plane_bytes(self) -> int:
        """``Layer.plane_bytes`` plus the refs plane.

        The whole of the spec's ``extra_nbytes`` intent, folded into the hook
        that already exists: ``undo._plane_bytes`` asks a layer for its own
        ``plane_bytes`` precisely so a subclass with a third plane could
        answer for itself, and every cost site that charges a layer already
        goes through it.
        """
        return int(super().plane_bytes) + int(self.refs.nbytes)

    def copy(self, *, name: str | None = None, uid: int | None = None) -> TilemapCel:
        """A duplicate with its own ``refs`` -- ``Layer.copy``'s reason,
        extended: a history snapshot or a duplicated cel that shared ``refs``
        with the original would let one's tile edits redraw the other.

        Built on ``Layer.copy`` rather than beside it, so the name-default and
        the ``uid``-override rule (history snapshots keep the original
        identity) stay in exactly one place.
        """
        base = super().copy(name=name, uid=uid)
        # Every ``Layer`` field taken from *base* by name rather than re-listed
        # here: the hand-written list this replaces carried nine of the eleven,
        # so ``background`` and ``reference`` were dropped by every history
        # snapshot and every duplicate of a tilemap cel. A list that has to be
        # extended whenever ``Layer`` grows a field is a list that will be
        # forgotten again.
        return TilemapCel(
            **{spec.name: getattr(base, spec.name) for spec in fields(Layer)},
            refs=self.refs.copy(),
            tileset_uid=self.tileset_uid,
        )


def strip(tiles: np.ndarray) -> Tileset:
    """A vertical-strip :class:`Tileset` from a stack of tiles.

    ``tiles`` is ``(N, tile_h, tile_w, 4)`` uint8 -- N individual tile images,
    local id order -- and is reshaped into the one-column strip image the
    engine stores (``(N * tile_h, tile_w, 4)``; ``Tileset.columns`` comes out
    to 1 because ``image_w == tile_w``).

    ``tiles[0]`` must already be blank: local id 0 is the required "no tile"
    slot the whole model leans on (see the module docstring), and the useful
    moment to say a caller got that wrong is construction, not the first
    draw that silently shows whatever tile 0 happened to be.
    """
    array = np.asarray(tiles)
    if array.ndim != 4 or array.shape[3] != 4:
        raise ValueError("a tile stack is (N, tile_h, tile_w, 4)")
    count, tile_h, tile_w = int(array.shape[0]), int(array.shape[1]), int(array.shape[2])
    if count < 1:
        raise ValueError("a tileset holds at least its blank tile")
    if array[0].any():
        raise ValueError("tile 0 of a strip must be blank")
    pixels = array.reshape(count * tile_h, tile_w, 4)
    return Tileset(name="tiles", pixels=pixels, tile_w=tile_w, tile_h=tile_h)


def blank_strip(tile_w: int, tile_h: int) -> Tileset:
    """A one-tile strip holding only the required blank tile."""
    return Tileset(name="tiles", pixels=cp.empty(tile_w, tile_h), tile_w=tile_w, tile_h=tile_h)


def grow(ts: Tileset, tiles: np.ndarray) -> Tileset:
    """``ts`` with ``tiles`` appended -- a frozen-replace, never a mutation.

    ``tiles`` is ``(N, tile_h, tile_w, 4)``, matching ``ts``'s own tile size
    exactly: growing a strip with a tile of a different shape would leave the
    image no longer sliceable at ``ts.tile_w``/``ts.tile_h``, and the useful
    moment to refuse that is here, not at the first misaligned draw.
    """
    array = np.asarray(tiles)
    if (
        array.ndim != 4
        or array.shape[1] != ts.tile_h
        or array.shape[2] != ts.tile_w
        or array.shape[3] != 4
    ):
        raise ValueError(f"a grow batch is (N, {ts.tile_h}, {ts.tile_w}, 4)")
    added = array.reshape(-1, ts.tile_w, 4)
    return replace(ts, pixels=np.concatenate([ts.pixels, added], axis=0))


def shrink(ts: Tileset, count: int) -> Tileset:
    """The undo of :func:`grow`: truncate to the first ``count`` tiles.

    ``count`` is bounded below at 1 -- the blank tile at local id 0 is never
    optional, so an undo cannot walk a strip back past it -- and above at the
    tileset's own ``tile_count``, since a larger count names tiles this
    tileset does not have.
    """
    count = int(count)
    if count < 1 or count > ts.tile_count:
        raise ValueError(f"shrink keeps 1..{ts.tile_count} tiles, not {count}")
    return replace(ts, pixels=ts.pixels[: count * ts.tile_h])


def with_tiles(ts: Tileset, tiles: list[tuple[int, np.ndarray]]) -> Tileset:
    """``ts`` with each named tile's content replaced -- a frozen-replace.

    Each entry is ``(local_id, pixels)`` at exactly that tile's own size;
    every edit lands on one private, writable copy of the strip image before
    it becomes the next frozen ``Tileset``, so a caller editing several tiles
    in one gesture pays for one copy, not one per tile.
    """
    pixels = ts.pixels.copy()
    for local_id, tile in tiles:
        x, y, w, h = ts.tile_rect(local_id)
        patch = np.asarray(tile)
        if patch.shape != (h, w, 4):
            raise ValueError(f"tile {local_id} must be {w}x{h} RGBA")
        pixels[y : y + h, x : x + w] = patch
    return replace(ts, pixels=pixels)


def oriented(tile: np.ndarray, raw: int) -> np.ndarray:
    """One tile turned the way a cell holding ``raw`` draws it.

    Transpose-then-mirror, ``plotter/render.py``'s own order and the reason
    the two engines agree bit-for-bit on every one of the eight square
    symmetries. A *view* wherever numpy can give one -- :func:`materialize`
    copies out of it into the canvas and never writes through it.
    """
    if raw & gid.FLIP_D:
        tile = np.transpose(tile, (1, 0, 2))
    if raw & gid.FLIP_H:
        tile = tile[:, ::-1]
    if raw & gid.FLIP_V:
        tile = tile[::-1, :]
    return tile


def canonical(tile: np.ndarray, raw: int) -> np.ndarray:
    """The exact inverse of :func:`oriented`: a drawn tile back to atlas order.

    This is what makes a pixel edit on a *flipped* placement land on the
    tileset in canonical orientation, which is the only orientation the atlas
    stores. Each of the three flags is an involution, so the inverse of
    "D then H then V" is the same three applied in the opposite order -- and
    the two functions are written as a mirrored pair, next to each other,
    because the failure mode of getting this wrong is a mirrored tile silently
    replacing a correct one everywhere it is placed.
    """
    if raw & gid.FLIP_V:
        tile = tile[::-1, :]
    if raw & gid.FLIP_H:
        tile = tile[:, ::-1]
    if raw & gid.FLIP_D:
        tile = np.transpose(tile, (1, 0, 2))
    return tile


def materialize(refs: np.ndarray, ts: Tileset, size: tuple[int, int]) -> np.ndarray:
    """The canvas ``refs`` and ``ts`` describe, as ``(H, W, 4)`` uint8.

    Per cell: the local id is ``refs`` masked by :data:`gid.GID_MASK`, and a
    local id this tileset does not have is treated as 0 -- blank -- rather
    than raising out of the middle of a draw (a stale ref surviving a
    tileset shrink is exactly this case, and it should draw as absent tile,
    not crash the frame). The three transform flags are applied by
    :func:`oriented`, whose exact inverse :func:`canonical` is what a pixel
    edit on a flipped placement goes back through. Each tile is pasted at
    ``(col * tile_w, row * tile_h)`` and
    cropped wherever that runs past the canvas edge -- the ordinary case
    whenever ``size`` is not an exact multiple of the tile size, since
    :func:`grid_shape` always rounds up.
    """
    width, height = int(size[0]), int(size[1])
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    grid_h, grid_w = refs.shape
    tile_w, tile_h = ts.tile_w, ts.tile_h
    for row in range(grid_h):
        y0 = row * tile_h
        if y0 >= height:
            break
        for col in range(grid_w):
            x0 = col * tile_w
            if x0 >= width:
                break
            raw = int(refs[row, col])
            # ``refs`` is uint32 and the mask keeps it non-negative, so the
            # only way to be out of range is past the end.
            local = raw & gid.GID_MASK
            if local >= ts.tile_count:
                local = 0
            tile = oriented(ts.tile_pixels(local), raw)
            h = min(tile.shape[0], height - y0)
            w = min(tile.shape[1], width - x0)
            canvas[y0 : y0 + h, x0 : x0 + w] = tile[:h, :w]
    return canvas


def grid_shape(size: tuple[int, int], tile_w: int, tile_h: int) -> tuple[int, int]:
    """``(grid_h, grid_w)`` for a canvas of ``size`` at this tile size, ceil-divided.

    Rounds up rather than down: a canvas that is not an exact multiple of the
    tile size still needs a partial row/column of cells to cover it, and
    :func:`materialize` crops the overhang rather than leaving a gap.
    """
    width, height = int(size[0]), int(size[1])
    return (math.ceil(height / int(tile_h)), math.ceil(width / int(tile_w)))


def content_key(tile: np.ndarray) -> bytes:
    """The hash-dedup key for one tile's content: its raw bytes."""
    return tile.tobytes()
