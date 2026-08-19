"""Undo steps for tile edits: tileset content, the refs door and the list.

Every rule ``inker/undo.py`` states applies here unchanged, and ``anim_edits.py``
restates the two worth repeating: **address by uid, never by position** -- a
``TileRefsEdit`` names its cel by uid so an undo issued after the stack has
been reordered still lands on the cel the placement was made to, and a
``TilesetListEdit`` names its slot's position but holds the *object* rather
than rebuilding one, which is what lets a track binding recorded before a
remove survive the undo that puts the slot back. **Hold the object, do not
copy it** -- the same reason: a redo that minted a fresh ``TilesetSlot`` would
answer to a uid nothing else points at any more.

Every hook these edits call lives on ``Document`` (via ``TileOps`` in
``_doc_tiles.py``) and does the same three things in the same order: rebuild
the frozen tileset (or the refs plane), re-materialize whatever pixels are
derived from it, and invalidate. Recording that work here as well as there
would be a second place for the two to drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..undo import Edit

__all__ = [
    "LayerSwapEdit",
    "TileRefsEdit",
    "TilesetGrowEdit",
    "TilesetListEdit",
    "TilesetPatchEdit",
    "TrackTilesetEdit",
]


@dataclass
class TilesetPatchEdit(Edit):
    """Exact tile-sized crops of one tileset's strip, before and after.

    ``tiles`` is a list of ``(local_id, before, after)`` triples, one per tile
    actually edited in one gesture -- an Auto-mode stroke that dirtied three
    tiles at once is one Ctrl+Z, not three. Each crop is tile-sized, never
    atlas-scale: the whole strip is rebuilt by ``with_tiles`` on the way in
    (see ``TileOps._apply_tileset_tiles``), so recording the atlas here as well
    would double the bytes for nothing the undo needs.
    """

    tileset_uid: int
    tiles: list[tuple[int, np.ndarray, np.ndarray]]

    def __post_init__(self) -> None:
        # Owned outright, for ``PatchEdit``'s reason spelled at length there: a
        # caller handing over an array it goes on writing to would quietly turn
        # the recorded "before" into the "after".
        self.tiles = [
            (int(local_id), before.copy(), after.copy()) for local_id, before, after in self.tiles
        ]
        self.cost = sum(before.nbytes + after.nbytes for _local_id, before, after in self.tiles)

    def undo(self, doc: Any) -> None:
        doc._apply_tileset_tiles(
            self.tileset_uid, [(local_id, before) for local_id, before, _after in self.tiles]
        )

    def redo(self, doc: Any) -> None:
        doc._apply_tileset_tiles(
            self.tileset_uid, [(local_id, after) for local_id, _before, after in self.tiles]
        )


@dataclass
class TilesetGrowEdit(Edit):
    """An appended strip of fresh tiles. Undo truncates the strip back off.

    ``added`` -- the ``(N, tile_h, tile_w, 4)`` batch that was appended -- is
    kept outright rather than only its count, because Stack mode content-hashes
    a freshly appended tile at the moment it is added; a redo that grew the
    strip with different pixels than the gesture actually minted would answer
    a later hash lookup with the wrong tile.
    """

    tileset_uid: int
    added: np.ndarray

    def __post_init__(self) -> None:
        self.added = np.asarray(self.added).copy()
        self.cost = int(self.added.nbytes)

    def undo(self, doc: Any) -> None:
        # Undo needs only the count to truncate back to -- the pixels that
        # were there before the grow are already the tileset's own, and the
        # hook rebuilds the strip by ``shrink`` rather than by any copy held
        # here.
        #
        # **Derived from the tileset as it stands, not recorded.** An undo can
        # only run with this grow's own tiles still on the end of the strip
        # (anything appended after it is a later step, undone first), so the
        # count before the grow is exactly "what is there now, minus what this
        # step added". Handing ``added.shape[0]`` straight through -- as this
        # did -- truncated to the number of tiles *appended* instead: a
        # one-tile Stack-mode grow onto a two-tile set undid to a *one*-tile
        # set, silently destroying the tileset's only real tile.
        slot = doc.tileset_slot(self.tileset_uid)
        doc._apply_tileset_grow(
            self.tileset_uid, int(slot.tileset.tile_count) - int(self.added.shape[0])
        )

    def redo(self, doc: Any) -> None:
        doc._apply_tileset_grow(self.tileset_uid, self.added)


@dataclass
class TileRefsEdit(Edit):
    """One tile-unit rectangle of one cel's refs, before and after.

    ``rect`` is ``(x0, y0, x1, y1)`` in *tile* units, not pixels -- the same
    door :meth:`~._doc_tiles.TileOps.place_tiles` walks, so a rectangle of
    cells is exact whatever the tile size is. Addressed by ``layer_uid`` and
    never by position, ``PatchEdit``'s own reason: an undo issued after the
    stack has been reordered must still land on the cel the placement was made
    to.
    """

    layer_uid: int
    rect: tuple[int, int, int, int]
    before: np.ndarray
    after: np.ndarray

    def __post_init__(self) -> None:
        self.before = self.before.copy()
        self.after = self.after.copy()
        self.cost = int(self.before.nbytes + self.after.nbytes)

    def _put(self, doc: Any, values: np.ndarray) -> None:
        doc._apply_refs(self.layer_uid, self.rect, values)

    def undo(self, doc: Any) -> None:
        self._put(doc, self.before)

    def redo(self, doc: Any) -> None:
        self._put(doc, self.after)


@dataclass
class TilesetListEdit(Edit):
    """One slot of ``Document.tilesets``, before and after: add or remove.

    The slot object itself is held rather than rebuilt -- ``LayerAddEdit``'s
    reason exactly: a track or a cel binds a tileset by ``TilesetSlot.uid``,
    and a redo that minted a fresh slot would strand every binding recorded
    against the one this step actually holds. **Minted once, at op time** --
    the unlink rule: ``add_tileset``/``remove_tileset`` build the slot (or find
    it) before this edit is constructed, and every undo/redo re-inserts that
    same object rather than a copy of it.

    ``before``/``after`` is ``None`` for the "did not exist" side of an add or
    a remove.
    """

    index: int
    before: Any
    after: Any

    def __post_init__(self) -> None:
        # A tileset's own pixels are a few dozen KiB at most -- bookkeeping
        # for the byte budget rather than a figure eviction meaningfully acts
        # on -- but the slot is what a binding addresses, and losing it loses
        # the tileset a track or a cel is pointing at, which is worth costing
        # at all rather than reporting a free zero.
        self.cost = sum(
            int(slot.tileset.pixels.nbytes)
            for slot in (self.before, self.after)
            if slot is not None
        )

    def undo(self, doc: Any) -> None:
        doc._apply_tileset_slot(self.index, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_tileset_slot(self.index, self.after)


@dataclass
class TrackTilesetEdit(Edit):
    """One track's ``tileset_uid``, before and after: bind or unbind.

    Its own type rather than a ``TrackPropsEdit`` entry, because the binding is
    not a display property: it decides what ``_ensure_cel_for`` autovivifies on
    that track, which is a *structural* fact the conversions
    (``convert_layer_to_tilemap``/``_to_raster``) fold into their compound
    beside the cel replacements. Animated documents only -- a still document
    has no tracks, and its tilemap layer carries ``tileset_uid`` on the cel
    itself, which a :class:`LayerSwapEdit` moves whole.

    ``None`` is "bound to nothing", both as a before and as an after.
    """

    track_uid: int
    before: int | None
    after: int | None

    def undo(self, doc: Any) -> None:
        doc._apply_track_tileset(self.track_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_track_tileset(self.track_uid, self.after)


@dataclass
class LayerSwapEdit(Edit):
    """One still document's layer replaced by another wearing the same uid.

    The conversions' still-document half. A ``CelSetEdit`` cannot serve here --
    there is no grid to set a slot of -- and add+remove cannot either: the two
    objects share a uid, so a compound of ``LayerRemoveEdit`` and
    ``LayerAddEdit`` would have both halves addressing the same name and would
    depend on the order they happen to resolve in.

    Both objects are *held*, ``LayerAddEdit``'s rule: the replacement is minted
    once, at op time, so a patch recorded against it survives an undo and a
    redo. Charged for both, since whichever is off the stack is pinned by this
    step alone.
    """

    layer_uid: int
    before: Any
    after: Any

    def __post_init__(self) -> None:
        self.cost = sum(
            int(layer.plane_bytes) for layer in (self.before, self.after) if layer is not None
        )

    def undo(self, doc: Any) -> None:
        doc._apply_layer_swap(self.layer_uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_layer_swap(self.layer_uid, self.after)
