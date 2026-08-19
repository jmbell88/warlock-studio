"""Tilesets and the refs door: the document-level half of the tile model.

``tiles.py`` is the pure model -- the mutable ``TilesetSlot`` holder, the
derived ``TilemapCel``, and the free functions that build and edit a
vertical-strip atlas. What is here is everything that turns those into
*document* operations: the tileset list (``add_tileset``/``remove_tileset``),
the one entry point that writes a cel's refs (``place_tiles``), and the
``_apply_*`` hooks ``tile_edits.py``'s edit types call in both directions.

**Every hook ends the same way**: rebuild the frozen tileset (or write the
refs plane directly), then re-materialize whatever pixels are derived from it,
then invalidate. ``_rematerialize_tileset`` is the one walk that does the
middle two steps for *every* cel a tileset edit can reach -- two layers
sharing one tileset, or one tilemap cel linked across several frames -- so an
Auto-mode edit made while frame 3 is on screen still repaints frame 7's cache
the moment frame 7 is looked at. ``_repaint_tiles`` is the smaller piece both
that walk and a single ``place_tiles`` call share: given one cel and a
tile-unit rectangle, re-derive exactly that rectangle of pixels from refs and
say so.

**Geometry has no refs-aware permutation yet.** A flip or a rotation would
have to turn ``refs`` by the same eight-symmetry algebra the tileset flags
already carry (Chunk 3.7); until it does, ``_refuse_tilemaps`` is what every
whole-canvas ``_replay`` op that has not been taught calls before it touches
anything -- refuse by name rather than let ``pixels`` and ``refs`` quietly
disagree, the Wave 3 risk the whole tile suite exists to catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import composite as cp
from .anim_edits import TrackAddEdit
from .animation import Track
from .tile_edits import TileRefsEdit, TilesetListEdit
from .tiles import TilemapCel, TilesetSlot, grid_shape, grow, materialize, shrink, with_tiles
from .undo import CompoundEdit, LayerAddEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .document import Document

__all__ = ["TileOps"]


class TileOps:
    """Tilesets, tilemap layers and the refs door, mixed into
    :class:`~.document.Document`."""

    # -- the tileset list -----------------------------------------------------

    def tileset_slot(self: Document, uid: int) -> Any:
        """The slot a uid names. ``KeyError`` on an unknown one, ``stack.by_uid``'s
        own rule: a caller asking for a tileset this document does not have is a
        bug upstream, not a case to paper over."""
        for slot in self.tilesets:
            if slot.uid == uid:
                return slot
        raise KeyError(uid)

    def _tileset_bound(self: Document, uid: int) -> bool:
        """Whether any track or any cel still points at tileset *uid*.

        A track binds a tileset the instant its ``tileset_uid`` is set, before
        any cel has ever been drawn on it -- an empty slot answering "not in
        use" would let ``remove_tileset`` pull the tileset out from under a
        track that has drawn nothing yet but is still, structurally, bound to
        it.
        """
        if self.anim is not None:
            if any(track.tileset_uid == uid for track in self.anim.tracks):
                return True
            layers: Any = self.anim.unique_cel_layers()
        else:
            layers = self.stack
        return any(
            isinstance(layer, TilemapCel) and layer.tileset_uid == uid for layer in layers
        )

    def add_tileset(self: Document, tileset: Any) -> Any:
        """Append a new :class:`~.tiles.TilesetSlot` holding *tileset*, and
        return it."""
        slot = TilesetSlot(tileset=tileset)
        index = len(self.tilesets)
        self._apply_tileset_slot(index, slot)
        self.history.push(TilesetListEdit(index, None, slot))
        return slot

    def remove_tileset(self: Document, uid: int) -> bool:
        """Drop a tileset slot. ``False`` for an unknown uid; refused by name
        while a track or a cel still binds it."""
        index = next((i for i, slot in enumerate(self.tilesets) if slot.uid == uid), None)
        if index is None:
            return False
        if self._tileset_bound(uid):
            raise ValueError(
                f"tileset {uid} is still bound by a track or a cel and cannot be removed"
            )
        slot = self.tilesets[index]
        self._apply_tileset_slot(index, None)
        self.history.push(TilesetListEdit(index, slot, None))
        return True

    # -- tilemap layers ---------------------------------------------------------

    def add_tilemap_layer(self: Document, tileset_uid: int, name: str = "Tilemap") -> Any:
        """A new tilemap layer bound to *tileset_uid*, all-zero refs.

        On an animated document this is a new track -- ``add_layer``'s own
        shape, a ``TrackAddEdit`` with no cels at all, since the grid is sparse
        and the first :meth:`place_tiles` call autovivifies exactly the one cel
        it needs. On a still document, which has no tracks, the tilemap layer
        *is* the cel: a lone ``TilemapCel`` inserted directly, ``add_layer``'s
        still branch with a different layer type.
        """
        self.commit_floating()
        slot = self.tileset_slot(tileset_uid)
        width, height = self.size
        grid_h, grid_w = grid_shape((width, height), slot.tileset.tile_w, slot.tileset.tile_h)
        # Read before the insert, ``add_layer``'s own reason: a new row that
        # stayed outside the group the active row is in would land in the
        # middle of that group's span and break its contiguity.
        parent = self._parent_of_active()
        if self.anim is not None:
            index = self.stack.active_index + 1
            track = Track(name=name, tileset_uid=tileset_uid)
            self._put_track(index, track, {})
            self._push_with_inheritance(
                TrackAddEdit(index, track, {}, pinned=False), parent
            )
            return self.stack[self.stack.active_index]
        cel = TilemapCel(
            pixels=cp.empty(width, height),
            refs=np.zeros((grid_h, grid_w), dtype=np.uint32),
            tileset_uid=tileset_uid,
            name=name,
        )
        index = self.stack.insert(self.stack.active_index + 1, cel)
        self.invalidate_all()
        self._push_with_inheritance(LayerAddEdit(index, cel), parent)
        return cel

    def place_tiles(
        self: Document, layer_uid: int, origin: tuple[int, int], patch: np.ndarray
    ) -> bool:
        """Write a rectangle of tile refs onto one cel. The single refs door --
        the stamp tool, tile flood fill and every future placement gesture all
        end here.

        ``origin`` is ``(tx, ty)`` in *tile* units; ``patch`` is a 2-D uint32
        refs plane, clipped to the cel's grid wherever it runs past an edge.
        Autovivifies the cel first (``_ensure_cel_for``), exactly as a stroke
        does, so drawing tiles on an empty frame is legal. A write that lands
        entirely off-grid or changes nothing pushes no step and discards the
        cel it may have just autovivified -- the funnel's own no-op rule,
        applied to refs instead of pixels.
        """
        self._ensure_cel_for(layer_uid)
        layer = self.layer_by_uid(layer_uid)
        if not isinstance(layer, TilemapCel):
            raise ValueError("place_tiles targets a tilemap layer")
        patch = np.asarray(patch, dtype=np.uint32)
        if patch.ndim != 2:
            raise ValueError("a tile patch is a 2-D refs plane")
        tx, ty = int(origin[0]), int(origin[1])
        grid_h, grid_w = layer.refs.shape
        x0, y0 = max(0, tx), max(0, ty)
        x1, y1 = min(grid_w, tx + patch.shape[1]), min(grid_h, ty + patch.shape[0])
        if x1 <= x0 or y1 <= y0:
            self._discard_pending_cel()
            return False
        rect = (x0, y0, x1, y1)
        clipped = patch[y0 - ty : y1 - ty, x0 - tx : x1 - tx]
        before = layer.refs[y0:y1, x0:x1].copy()
        if np.array_equal(before, clipped):
            self._discard_pending_cel()
            return False
        self._apply_refs(layer_uid, rect, clipped)
        # The pending-cel machinery ``_commit_patch`` uses: an autovivified cel
        # rides into the same ``CompoundEdit`` as the refs write, so placing a
        # tile on an empty frame is one Ctrl+Z rather than two.
        pending, self._pending_cels = self._pending_cels, []
        edit: Any = TileRefsEdit(layer_uid, rect, before, clipped)
        self.history.push(edit if not pending else CompoundEdit([*pending, edit]))
        return True

    # -- undo/redo re-entry -----------------------------------------------------

    def _apply_tileset_tiles(
        self: Document, tileset_uid: int, tiles: list[tuple[int, np.ndarray]]
    ) -> None:
        """Rebuild one tileset's strip with the named tiles' content replaced,
        and re-materialize every cel bound to it. The ``TilesetPatchEdit`` hook."""
        slot = self.tileset_slot(tileset_uid)
        slot.tileset = with_tiles(slot.tileset, tiles)
        self._rematerialize_tileset(tileset_uid)

    def _apply_tileset_grow(self: Document, tileset_uid: int, added_or_count: Any) -> None:
        """Append or truncate one tileset's strip, and re-materialize every cel
        bound to it. The ``TilesetGrowEdit`` hook -- redo hands the appended
        batch, undo hands the count to truncate back to."""
        slot = self.tileset_slot(tileset_uid)
        if isinstance(added_or_count, (int, np.integer)):
            slot.tileset = shrink(slot.tileset, int(added_or_count))
        else:
            slot.tileset = grow(slot.tileset, np.asarray(added_or_count))
        self._rematerialize_tileset(tileset_uid)

    def _apply_refs(
        self: Document, layer_uid: int, rect: tuple[int, int, int, int], values: np.ndarray
    ) -> None:
        """Write one tile-unit rectangle of one cel's refs, and re-derive its
        pixels. The ``TileRefsEdit`` hook, run in either direction."""
        layer = self.layer_by_uid(layer_uid)
        x0, y0, x1, y1 = rect
        layer.refs[y0:y1, x0:x1] = values
        self._repaint_tiles(layer, rect)

    def _apply_tileset_slot(self: Document, index: int, slot: Any) -> None:
        """Insert or remove one slot of ``self.tilesets`` by position. The
        ``TilesetListEdit`` hook, run in either direction.

        ``None`` removes what is at ``index``; a slot is always *inserted*
        rather than assigned in place -- ``list.insert`` is what makes both an
        ``add`` (appending past the end) and the undo of a ``remove`` (putting
        a slot back into a list that has already closed the gap behind it)
        the same call, where an index-assignment would silently overwrite
        whatever a remove's undo is trying to make room for.
        """
        if slot is None:
            del self.tilesets[index]
        else:
            self.tilesets.insert(index, slot)
        self.rev += 1

    def _rematerialize_tileset(self: Document, tileset_uid: int) -> None:
        """THE one invalidation walk: every ``TilemapCel`` bound to this
        tileset, across every distinct cel in the document -- not the current
        frame's stack -- rebuilt from refs and invalidated.

        Walking ``self.anim.unique_cel_layers()`` rather than the visible
        stack is the whole point: an Auto-mode edit made while frame 3 is on
        screen has to reach frame 7's cel too, or that frame's cache goes on
        showing the tile before it was edited until something else happens to
        touch it.
        """
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        for layer in layers:
            if isinstance(layer, TilemapCel) and layer.tileset_uid == tileset_uid:
                grid_h, grid_w = layer.refs.shape
                self._repaint_tiles(layer, (0, 0, grid_w, grid_h))

    def _repaint_tiles(
        self: Document, layer: Any, tile_rect: tuple[int, int, int, int]
    ) -> None:
        """Re-derive the pixels a tile-unit rectangle covers, from refs, and
        invalidate them. Shared by :meth:`_apply_refs` (one cel, the rectangle
        just written) and :meth:`_rematerialize_tileset` (every bound cel, the
        whole grid) -- both are "pixels are a materialization of refs", only at
        different scales.
        """
        slot = self.tileset_slot(layer.tileset_uid)
        tx0, ty0, tx1, ty1 = tile_rect
        tile_w, tile_h = slot.tileset.tile_w, slot.tileset.tile_h
        width, height = layer.size
        px0, py0 = tx0 * tile_w, ty0 * tile_h
        px1, py1 = min(width, tx1 * tile_w), min(height, ty1 * tile_h)
        if px1 <= px0 or py1 <= py0:
            return
        refs_crop = layer.refs[ty0:ty1, tx0:tx1]
        layer.pixels[py0:py1, px0:px1] = materialize(
            refs_crop, slot.tileset, (px1 - px0, py1 - py0)
        )
        self.invalidate((px0, py0, px1, py1), layer_uid=layer.uid)

    # -- geometry refusal ---------------------------------------------------

    def _refuse_tilemaps(self: Document, verb: str) -> None:
        """Refuse *verb* outright when the document holds a tilemap layer.

        Called by every whole-canvas geometry op that has not yet been taught
        refs-aware permutation (Chunk 3.7). A track bound to a tileset counts
        even before any cel has been drawn on it -- the structural binding is
        what a later autovivify would turn into a ``TilemapCel``, and a flip
        that ran before that happened would leave nothing wrong *yet*, but
        would be one autovivify away from it.
        """
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        has_tilemap = any(isinstance(layer, TilemapCel) for layer in layers)
        has_bound_track = self.anim is not None and any(
            track.tileset_uid is not None for track in self.anim.tracks
        )
        if has_tilemap or has_bound_track:
            raise ValueError(f"a {verb} of a tilemap layer is not yet modeled")
