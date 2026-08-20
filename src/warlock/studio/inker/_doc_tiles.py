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

**Geometry is taught one op and refuses the rest.** A canvas resize
translates the picture by whole cells and pads or crops, which is a pure
pad/crop of ``refs`` with every flag bit left alone -- so ``_tile_regrid``
models it, and refuses by name when the offset is not a whole number of tiles.
A flip or a rotation would instead have to turn ``refs`` by the same
eight-symmetry algebra the tileset flags already carry, which nothing here
does; until something does, ``_refuse_tilemaps`` is what every whole-canvas
``_replay`` op that has not been taught calls before it touches anything --
refuse by name rather than let ``pixels`` and ``refs`` quietly disagree, the
Wave 3 risk the whole tile suite exists to catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..tilegrid import gid
from . import composite as cp
from . import indexed as ix
from .anim_edits import CelSetEdit, TrackAddEdit
from .animation import Track
from .layers import Layer
from .tile_edits import (
    LayerSwapEdit,
    TileRefsEdit,
    TilesetGrowEdit,
    TilesetListEdit,
    TilesetPatchEdit,
    TrackTilesetEdit,
)
from .tiles import (
    TilemapCel,
    TilesetSlot,
    canonical,
    content_key,
    grid_shape,
    grow,
    materialize,
    oriented,
    shrink,
    strip,
    with_tiles,
)
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

    def _will_be_tilemap(self: Document, layer_uid: int) -> bool:
        """Whether *layer_uid* already is, or would autovivify into, a
        ``TilemapCel`` -- decided without materializing anything.

        The funnel discipline every sibling door follows: validate first,
        autovivify second. On an animated document a placeholder's eventual
        type is a property of the *track* underneath it (``tileset_uid`` set
        or not), which ``_ensure_cel_for`` already reads without needing a
        real cel to exist first -- so this asks the same question the same
        way, rather than calling ``_ensure_cel_for`` to find out and having to
        undo the autovivify on a refusal.
        """
        if self.anim is None:
            try:
                return isinstance(self.stack.by_uid(layer_uid), TilemapCel)
            except KeyError:
                return False
        try:
            index = self.stack.index_of(layer_uid)
        except KeyError:
            # Not on the current frame at all -- an off-frame cel, addressed
            # the way undo re-entry does. ``_ensure_cel_for`` never touches
            # this case either (it only acts on the current stack), so the
            # type check alone is the whole answer.
            try:
                return isinstance(self.layer_by_uid(layer_uid), TilemapCel)
            except KeyError:
                return False
        existing = self.stack[index]
        if not self.anim.is_placeholder(existing):
            return isinstance(existing, TilemapCel)
        return index < len(self.anim.tracks) and self.anim.tracks[index].tileset_uid is not None

    def _refuse_stale_ids(
        self: Document, layer_uid: int, patch: np.ndarray, origin: tuple[int, int]
    ) -> None:
        """Refuse a patch naming a tile the bound tileset does not have.

        **Blank today, a different tile tomorrow.** :func:`~.tiles.materialize`
        draws an out-of-range ref as blank rather than raising -- which is the
        right answer for a ref that has *become* stale (a tileset shrank under
        one, and a frame must not crash for it) and exactly the wrong thing to
        allow at the door: a write that stores an id past the end looks correct
        until the tileset grows past it, at which point every one of those
        cells silently starts drawing whatever tile landed on that number, with
        nothing in the document to say it was ever wrong.

        Read through ``GID_MASK``, because the flag bits are not part of the
        id, and run **before** :meth:`_ensure_cel_for`, the funnel discipline
        every sibling check here follows: a refusal must not leave an
        autovivified cel behind with no undo step to take it back out.

        Silent when the binding cannot be resolved -- an impossible state that
        :meth:`_will_be_tilemap` has already answered for, and one this method
        has no better answer to invent for than the layer's own.
        """
        tileset_uid = self.tileset_uid_of(layer_uid)
        if tileset_uid is None or not patch.size:
            return
        try:
            ts = self.tileset_slot(tileset_uid).tileset
        except KeyError:  # pragma: no cover - a dangling binding
            return
        # Only the cells the write will actually land on. The clip is the same
        # arithmetic ``place_tiles`` applies to ``layer.refs`` below, taken off
        # the *canvas* rather than the cel so it can run before the autovivify:
        # a stale id in a region the grid clips away is never written, and a
        # patch straddling the canvas edge should not be refused for the part
        # of it that does not exist.
        grid_h, grid_w = grid_shape(self.size, ts.tile_w, ts.tile_h)
        tx, ty = int(origin[0]), int(origin[1])
        x0, y0 = max(0, tx), max(0, ty)
        x1 = min(grid_w, tx + patch.shape[1])
        y1 = min(grid_h, ty + patch.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        patch = patch[y0 - ty : y1 - ty, x0 - tx : x1 - tx]
        count = ts.tile_count
        worst = int((patch & gid.GID_MASK).max())
        if worst >= count:
            raise ValueError(
                f"tile {worst} is outside this layer's tileset, which holds "
                f"0..{count - 1}"
            )

    def _strip_diagonal(self: Document, layer_uid: int, patch: np.ndarray) -> np.ndarray:
        """*patch* with ``FLIP_D`` masked off wherever the bound tileset's tiles
        are not square. The documented mask at the refs door.

        A diagonal flip is a transpose, and a transpose of a non-square tile is
        a ``(tile_w, tile_h)`` picture asked to fill a ``(tile_h, tile_w)``
        cell -- Tiled itself defines the flag only for square tiles.
        :func:`~.tiles.materialize` would paste the wrong footprint over a
        neighbouring cell, and worse, ``_commit_tilemap_patch`` would read the
        neighbour's canvas pixels back through :func:`~.tiles.canonical` into
        the canonical tile: a permanent, propagating corruption of the atlas.
        So the flag is stripped where placements enter rather than answered
        for downstream; the import doors (``ora._read_tiles``,
        ``asein._build_tilemap_cel``) apply the same mask to what a file
        carries.

        A mask rather than a refusal, deliberately: the stamp's D toggle is
        offered per-*session*, not per-tileset, and ``place_tiles`` refuses by
        raising -- a raise out of a canvas press is a window down, not a
        refusal a user can meet. The placement lands unturned instead.
        """
        if not patch.size or not (patch & gid.DTYPE(gid.FLIP_D)).any():
            return patch
        tileset_uid = self.tileset_uid_of(layer_uid)
        if tileset_uid is None:
            return patch
        try:
            ts = self.tileset_slot(tileset_uid).tileset
        except KeyError:  # pragma: no cover - a dangling binding
            return patch
        if ts.tile_w == ts.tile_h:
            return patch
        return patch & gid.DTYPE(0xFFFFFFFF ^ gid.FLIP_D)

    def place_tiles(
        self: Document, layer_uid: int, origin: tuple[int, int], patch: np.ndarray
    ) -> bool:
        """Write a rectangle of tile refs onto one cel. The single refs door --
        the stamp tool, tile flood fill and every future placement gesture all
        end here.

        ``origin`` is ``(tx, ty)`` in *tile* units; ``patch`` is a 2-D uint32
        refs plane, clipped to the cel's grid wherever it runs past an edge.
        All three validations below run **before** ``_ensure_cel_for`` -- a
        plain track's placeholder must be refused without ever materializing a
        ``Layer`` for it, or the refusal would leave a real cel behind with no
        undo step to take it back out, and a ``CelSetEdit`` sitting orphaned
        in ``_pending_cels`` ready to attach itself to whatever the *next*
        successful write happens to be. Once all three pass, the cel is
        autovivified exactly as a stroke's is, so drawing tiles on an empty
        frame is legal; a write that then lands entirely off-grid or changes
        nothing pushes no step and discards the cel it just autovivified --
        the funnel's own no-op rule, applied to refs instead of pixels.
        """
        if not self._will_be_tilemap(layer_uid):
            raise ValueError("place_tiles targets a tilemap layer")
        patch = np.asarray(patch, dtype=np.uint32)
        if patch.ndim != 2:
            raise ValueError("a tile patch is a 2-D refs plane")
        patch = self._strip_diagonal(layer_uid, patch)
        self._refuse_stale_ids(layer_uid, patch, origin)
        self._ensure_cel_for(layer_uid)
        layer = self.layer_by_uid(layer_uid)
        if not isinstance(layer, TilemapCel):  # pragma: no cover - defensive
            raise ValueError("place_tiles targets a tilemap layer")
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

    # -- the funnel's tilemap branch ---------------------------------------------

    def _tile_hash_index(self: Document, tileset_uid: int) -> dict[bytes, int]:
        """``content_key -> lowest local id`` for one tileset. A derived cache.

        Keyed on the *identity* of the strip image, which is exactly right
        because a tileset is edited by frozen-replace: ``grow``, ``shrink`` and
        ``with_tiles`` all hand back a new ``Tileset`` over a new array, so a
        changed atlas is always a changed array.

        The array itself is held beside the index rather than its ``id()``,
        which is the whole difference between this and the recycled-id bug
        class: a freed array's address is reused, so a stamp that outlived its
        array could answer "unchanged" for a completely different atlas. Holding
        the array makes that impossible by construction and costs one strip per
        tileset -- a few dozen KiB.

        Never serialized, and rebuilt on demand: the lowest id wins a tie, so a
        tileset that holds a tile twice always dedups onto the earlier one.
        """
        slot = self.tileset_slot(tileset_uid)
        pixels = slot.tileset.pixels
        hit = self._tile_hashes.get(tileset_uid)
        if hit is not None and hit[0] is pixels:
            return hit[1]
        index: dict[bytes, int] = {}
        for local in range(slot.tileset.tile_count):
            index.setdefault(content_key(slot.tileset.tile_pixels(local)), local)
        self._tile_hashes[tileset_uid] = (pixels, index)
        return index

    def _tile_span(
        self: Document, layer: Any, rect: tuple[int, int, int, int], tile_w: int, tile_h: int
    ) -> tuple[int, int, int, int] | None:
        """The tile-unit rectangle a pixel rectangle touches, clipped to the grid."""
        grid_h, grid_w = layer.refs.shape
        x0, y0, x1, y1 = rect
        tx0, ty0 = max(0, x0 // tile_w), max(0, y0 // tile_h)
        tx1 = min(grid_w, -(-x1 // tile_w))
        ty1 = min(grid_h, -(-y1 // tile_h))
        if tx1 <= tx0 or ty1 <= ty0:
            return None
        return (tx0, ty0, tx1, ty1)

    def _commit_tilemap_patch(
        self: Document, layer: Any, rect: tuple[int, int, int, int], before: np.ndarray
    ) -> None:
        """Route a committed pixel write on a tilemap cel back onto its tileset.

        ``_commit_patch``'s tilemap branch, and the whole of the mode handling
        for a tilemap layer -- the divert returns before every one of the
        funnel's own three, so all three are answered here or nowhere.
        *Indexed* is answered by construction: the cel carries no index plane
        and the strip stays RGBA (the recorded divergence -- a tilemap layer
        materializes RGBA, and an index plane over a picture that is itself
        derived would be a third description of the same pixels). *Grayscale*
        and the *palette snap* are applied to the tile below, before the dedup
        key.

        The tool has already written RGBA over ``rect``. Per touched cell, the
        drawn tile is reconstructed -- the current atlas tile turned by the
        placement's flags, with the *visible* part of the canvas laid over it,
        so a partial cell at the canvas edge keeps the tile content the crop
        never showed -- and then put back into canonical orientation by
        :func:`~.tiles.canonical`. What happens to that content is the
        behaviour:

        * **manual** never modifies the tileset, so the write is simply
          re-derived away and nothing is pushed. The invalidate still happens,
          ``_commit_indexed_patch``'s no-op branch's reason: the tool's raw
          write is on screen and has just been rolled back.
        * **auto** edits the tile in place -- every placement of it, on every
          frame and every layer, follows through ``_rematerialize_tileset``.
          Two exceptions re-point instead: content that already exists
          elsewhere in the atlas (the dedup check, first), and an edit to local
          id 0, which is the required-blank slot and cannot be written.
        * **stack** only ever appends: unseen content becomes a new tile and
          the touched refs move to it; no existing tile is modified.

        ``before`` is deliberately unused, ``_commit_indexed_patch``'s reason
        one level along: the *tileset* is the record here, and each tile's
        before-state is still sitting in the atlas at this moment because
        nothing but this method writes one.

        Two cells of one gesture that resolve to the same tile in Auto mode are
        resolved last-writer-wins, and then reconciled: the re-materialisation
        walk repaints every placement from the tileset, so the cel is left
        agreeing with its refs whichever cell won.
        """
        slot = self.tileset_slot(layer.tileset_uid)
        ts = slot.tileset
        tile_w, tile_h = ts.tile_w, ts.tile_h
        span = self._tile_span(layer, rect, tile_w, tile_h)
        if span is None:  # pragma: no cover - defensive: a clipped rect is on-grid
            self._discard_pending_cel()
            return
        tx0, ty0, tx1, ty1 = span
        behavior = str(self.tile_behavior)
        if behavior not in ("auto", "stack"):
            # Manual -- and anything else the field has been set to. Never
            # touching the tileset is the safe reading of a behaviour this
            # method does not know, and it is the field's own default.
            self._repaint_tiles(layer, span)
            self._discard_pending_cel()
            return
        stacking = behavior == "stack"
        width, height = layer.size
        wx0, wy0, wx1, wy1 = rect
        lookup = dict(self._tile_hash_index(layer.tileset_uid))
        count = ts.tile_count
        refs_before = layer.refs[ty0:ty1, tx0:tx1].copy()
        refs_after = refs_before.copy()
        patches: dict[int, np.ndarray] = {}
        #: local id -> the ``lookup`` key its in-place patch claimed this
        #: gesture, so a second patch of the same tile can retire the first
        #: key before claiming its own. See the bookkeeping note below.
        patched_keys: dict[int, bytes] = {}
        added: list[np.ndarray] = []
        for ty in range(ty0, ty1):
            for tx in range(tx0, tx1):
                raw = int(layer.refs[ty, tx])
                local = raw & gid.GID_MASK
                if local >= ts.tile_count:
                    # A stale ref draws as blank, so that is what it edits from
                    # -- ``materialize``'s own rule, restated on the way back.
                    local = 0
                current = ts.tile_pixels(local)
                drawn = oriented(current, raw).copy()
                px, py = tx * tile_w, ty * tile_h
                rows = min(drawn.shape[0], height - py)
                cols = min(drawn.shape[1], width - px)
                if rows <= 0 or cols <= 0:  # pragma: no cover - grid_shape ceils
                    continue
                drawn[:rows, :cols] = layer.pixels[py : py + rows, px : px + cols]
                # **The colour modes, applied here.** ``_commit_patch``'s own
                # two constraints, in its own order (two ``if``s and not an
                # ``elif``, because a grayscale document with a palette gets
                # both). The divert at the head of the funnel returns *before*
                # those branches, so this is the only place either can reach a
                # tilemap layer -- without it a blue stroke on a grayscale
                # document put ``[0, 0, 255, 255]`` into the **tileset**, and
                # every placement of that tile drew it.
                #
                # Constrained before the dedup key and before any tileset
                # write, mirroring the funnel's order for the funnel's reason:
                # the key, the no-op test and the recorded before/after then
                # all describe the pixels the document actually ends up with,
                # and an undo followed by a redo lands on the same colours.
                #
                # Scoped to ``rect`` rather than to the whole cell, because
                # ``rect`` is the funnel's scope too: a tile may legitimately
                # hold off-palette content an import brought in, and touching
                # one pixel of one placement is no reason to rewrite the rest
                # of it everywhere it is placed.
                #
                # Applied before :func:`~.tiles.canonical` only for the
                # coordinates' sake -- both constraints are per-pixel, so they
                # commute with all eight of the square symmetries.
                if self.color_mode == "grayscale" or self.palette:
                    cx0, cy0 = max(wx0, px) - px, max(wy0, py) - py
                    cx1, cy1 = min(wx1, px + cols) - px, min(wy1, py + rows) - py
                    if cx1 > cx0 and cy1 > cy0:
                        region = drawn[cy0:cy1, cx0:cx1]
                        if self.color_mode == "grayscale":
                            region = ix.grayscale(region)
                        if self.palette:
                            region = ix.snap(region, self.palette)
                        drawn[cy0:cy1, cx0:cx1] = region
                tile = np.ascontiguousarray(canonical(drawn, raw))
                if np.array_equal(tile, current):
                    continue
                key = content_key(tile)
                target = lookup.get(key)
                if target is None and (stacking or local == 0):
                    target = count
                    count += 1
                    added.append(tile)
                    lookup[key] = target
                if target is None:
                    # The atlas entry this id answers to has moved; keep the
                    # working index honest for the rest of the gesture. Two
                    # rules make that bookkeeping rather than a bare ``pop``.
                    # The retired key is the one this tile actually holds in
                    # the index -- its content when the gesture opened, or the
                    # key an *earlier patch this gesture* claimed for it -- and
                    # it is dropped only when the index maps it to this tile:
                    # the index maps content to the lowest id holding it, so
                    # ``current``'s key may name a different tile with the same
                    # content, and popping that shared key would send the next
                    # matching cell to a redundant append. Leaving the earlier
                    # patch's key live would be the opposite failure -- a later
                    # cell matching the first content would point at a tile
                    # whose final content is something else.
                    previous = patched_keys.get(local, content_key(current))
                    if lookup.get(previous) == local:
                        del lookup[previous]
                    patches[local] = tile
                    lookup[key] = local
                    patched_keys[local] = key
                    continue
                # The flags stay on. The content that matched is the *canonical*
                # tile, so the cell still has to be turned the same way to draw
                # what the user just painted. Local id 0 is the exception only
                # because every orientation of blank is blank.
                refs_after[ty - ty0, tx - tx0] = np.uint32(
                    0 if target == 0 else target | (raw & gid.FLAG_MASK)
                )
        edits: list[Any] = []
        if added:
            edits.append(TilesetGrowEdit(slot.uid, np.stack(added, axis=0)))
        if patches:
            edits.append(
                TilesetPatchEdit(
                    slot.uid,
                    [(lid, ts.tile_pixels(lid), tile) for lid, tile in sorted(patches.items())],
                )
            )
        if not np.array_equal(refs_before, refs_after):
            edits.append(TileRefsEdit(layer.uid, span, refs_before, refs_after))
        if not edits:
            # Nothing the model can record: put the raw write back and say so,
            # exactly as manual mode does.
            self._repaint_tiles(layer, span)
            self._discard_pending_cel()
            return
        # Applied through the steps themselves rather than beside them -- the
        # grow must land before the refs that name the appended ids, and every
        # hook ends in the re-materialisation walk that leaves ``pixels``
        # agreeing with ``refs`` again.
        for edit in edits:
            edit.redo(self)
        pending, self._pending_cels = self._pending_cels, []
        step: Any = (
            edits[0] if len(edits) == 1 and not pending else CompoundEdit([*pending, *edits])
        )
        self.history.push(step)

    # -- conversions -------------------------------------------------------------

    def _track_of_row(self: Document, layer_uid: int) -> Any:
        """The track a stack row belongs to, or ``None`` on a still document."""
        if self.anim is None:
            return None
        return self.anim.tracks[self.stack.index_of(layer_uid)]

    def _slots_of_track(self: Document, track: Any) -> list[tuple[Any, Any]]:
        """``(frame, cel)`` for every frame this track actually holds a cel on.

        Every *slot*, not every distinct cel: a linked cel appears in several
        and each one has to be re-registered, or the conversion would unlink
        the frames it replaced.
        """
        anim = self.anim
        assert anim is not None
        return [
            (frame, anim.cels[(track.uid, frame.uid)])
            for frame in anim.frames
            if (track.uid, frame.uid) in anim.cels
        ]

    def convert_layer_to_tilemap(
        self: Document, layer_uid: int, tile_w: int, tile_h: int
    ) -> bool:
        """Turn a raster layer into a tilemap one, cutting its own art into the
        tileset it is then bound to.

        The whole track, not the visible cel: a tilemap layer is a *track*
        property on an animated document, so every cel of it is cut and every
        cut dedups against the same growing atlas -- two frames drawing the same
        16x16 chunk share one tile, which is the point of the mode. Local id 0
        is reserved blank and every transparent cell maps onto it, so an empty
        area of the drawing costs one ref and no tile.

        Replacement cels keep their **uid** and every layer property, so a patch
        recorded before the conversion still addresses the same row afterwards;
        and a cel linked across frames is replaced *once* and re-registered in
        every slot it occupied, so the link survives.

        A canvas that is not a whole number of tiles across is covered by
        ``grid_shape``'s ceil, and the partial cut is padded transparent -- the
        materialisation crops the overhang straight back off, which is what
        makes the round trip through :meth:`convert_layer_to_raster` bit-exact.

        ``False`` rather than a raise for the three ordinary "nothing to do"
        answers: an unknown uid, a layer that is already a tilemap, and a
        content-locked one (``write_locked``'s door, the same refusal every
        other write earns).
        """
        tile_w, tile_h = int(tile_w), int(tile_h)
        if tile_w < 1 or tile_h < 1:
            raise ValueError("a tile size is at least 1x1")
        try:
            index = self.stack.index_of(layer_uid)
        except KeyError:
            return False
        if self._will_be_tilemap(layer_uid) or self.write_locked(self.stack[index]):
            return False
        self.commit_floating()
        width, height = self.size
        grid_h, grid_w = grid_shape((width, height), tile_w, tile_h)
        track = self._track_of_row(layer_uid)
        slots = (
            [(None, self.stack[self.stack.index_of(layer_uid)])]
            if track is None
            else self._slots_of_track(track)
        )
        blank = np.zeros((tile_h, tile_w, 4), dtype=np.uint8)
        tiles: list[np.ndarray] = [blank]
        lookup: dict[bytes, int] = {content_key(blank): 0}
        refs_for: dict[int, np.ndarray] = {}
        for _frame, cel in slots:
            if id(cel) in refs_for:
                continue
            refs = np.zeros((grid_h, grid_w), dtype=gid.DTYPE)
            for row in range(grid_h):
                for col in range(grid_w):
                    cut = blank.copy()
                    py, px = row * tile_h, col * tile_w
                    rows, cols = min(tile_h, height - py), min(tile_w, width - px)
                    cut[:rows, :cols] = cel.pixels[py : py + rows, px : px + cols]
                    key = content_key(cut)
                    local = lookup.get(key)
                    if local is None:
                        local = len(tiles)
                        tiles.append(cut)
                        lookup[key] = local
                    refs[row, col] = local
            refs_for[id(cel)] = refs
        tileset = strip(np.stack(tiles, axis=0))
        slot = TilesetSlot(tileset=tileset)
        edits: list[Any] = [TilesetListEdit(len(self.tilesets), None, slot)]
        if track is not None:
            edits.append(TrackTilesetEdit(track.uid, track.tileset_uid, slot.uid))
        replacements: dict[int, Any] = {}
        for frame, cel in slots:
            new = replacements.get(id(cel))
            if new is None:
                refs = refs_for[id(cel)]
                new = TilemapCel(
                    pixels=materialize(refs, tileset, (width, height)),
                    refs=refs,
                    tileset_uid=slot.uid,
                    name=cel.name,
                    opacity=cel.opacity,
                    visible=cel.visible,
                    blend=cel.blend,
                    alpha_lock=cel.alpha_lock,
                    locked=cel.locked,
                    uid=cel.uid,
                )
                replacements[id(cel)] = new
                first = True
            else:
                first = False
            edits.append(
                CelSetEdit(track.uid, frame.uid, cel, new, pinned=first)
                if track is not None
                else LayerSwapEdit(cel.uid, cel, new)
            )
        for edit in edits:
            edit.redo(self)
        self.history.push(CompoundEdit(edits))
        self.invalidate_all()
        return True

    def convert_layer_to_raster(self: Document, layer_uid: int) -> bool:
        """Turn a tilemap layer back into an ordinary one. Lossless by
        construction: ``pixels`` is already the materialisation, so the
        replacement copies it and there is nothing to re-render.

        The tileset is *left in the document* rather than removed with the
        layer: nothing binds it afterwards, so ``remove_tileset`` will now take
        it, and a user converting back and forth to reach the pixel tools does
        not lose the atlas they authored in between.
        """
        try:
            index = self.stack.index_of(layer_uid)
        except KeyError:
            return False
        if not self._will_be_tilemap(layer_uid) or self.write_locked(self.stack[index]):
            return False
        self.commit_floating()
        track = self._track_of_row(layer_uid)
        slots = (
            [(None, self.stack[self.stack.index_of(layer_uid)])]
            if track is None
            else self._slots_of_track(track)
        )
        edits: list[Any] = []
        replacements: dict[int, Any] = {}
        for frame, cel in slots:
            if not isinstance(cel, TilemapCel):  # pragma: no cover - defensive
                continue
            new = replacements.get(id(cel))
            if new is None:
                pixels = cel.pixels.copy()
                new = Layer(
                    pixels=pixels,
                    # ``_resolved_plane``: a layer minted inside an indexed
                    # document is born with its plane, or the save records RGBA
                    # under a mode that promises slots. It rewrites ``pixels``
                    # in place to the materialisation, so the two agree from
                    # the moment the layer exists.
                    indices=self._resolved_plane(pixels),
                    name=cel.name,
                    opacity=cel.opacity,
                    visible=cel.visible,
                    blend=cel.blend,
                    alpha_lock=cel.alpha_lock,
                    locked=cel.locked,
                    uid=cel.uid,
                )
                replacements[id(cel)] = new
                first = True
            else:
                first = False
            edits.append(
                CelSetEdit(track.uid, frame.uid, cel, new, pinned=first)
                if track is not None
                else LayerSwapEdit(cel.uid, cel, new)
            )
        if track is not None and track.tileset_uid is not None:
            edits.append(TrackTilesetEdit(track.uid, track.tileset_uid, None))
        if not edits:
            return False
        for edit in edits:
            edit.redo(self)
        self.history.push(CompoundEdit(edits))
        self.invalidate_all()
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
            # The dedup cache goes with the slot -- it is keyed by uid and
            # rebuilt on demand, so a removed tileset's entry is only a strip
            # image held alive for nothing. An undo re-inserting the slot
            # rebuilds the index on the next ask.
            self._tile_hashes.pop(self.tilesets[index].uid, None)
            del self.tilesets[index]
        else:
            self.tilesets.insert(index, slot)
        self.rev += 1

    def _apply_track_tileset(self: Document, track_uid: int, value: int | None) -> None:
        """Bind or unbind one track's tileset. The ``TrackTilesetEdit`` hook.

        Ends in ``_anim_changed`` rather than a bare ``rev`` bump: the binding
        is what ``_ensure_cel_for`` reads to decide a placeholder's eventual
        type, so the current frame's view has to be rebuilt from it.
        """
        anim = self.anim
        assert anim is not None
        for track in anim.tracks:
            if track.uid == track_uid:
                track.tileset_uid = value
                break
        self._anim_changed()

    def _apply_layer_swap(self: Document, layer_uid: int, layer: Any) -> None:
        """Replace one still document's layer by uid. The ``LayerSwapEdit`` hook."""
        self.stack.layers[self.stack.index_of(layer_uid)] = layer
        self.invalidate_all()

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

    # -- what the UI asks about the active row ------------------------------

    def active_tilemap_uid(self: Document) -> int | None:
        """The active row's uid when it is (or would autovivify into) a tilemap
        cel, else ``None``.

        The panes' one question, asked once here rather than three times in
        three files: the toolbox greys the tile tool by it, the tools pane
        shows the behaviour toggle by it, and the canvas decides whether a
        press stamps or refuses by it. Through :meth:`_will_be_tilemap` rather
        than an ``isinstance``, so a bound track whose current frame still
        holds a placeholder answers the same as one that has been drawn on --
        which is what :meth:`place_tiles` will do with it.
        """
        if not len(self.stack):
            return None
        uid = self.stack.active.uid
        return uid if self._will_be_tilemap(uid) else None

    def tileset_uid_of(self: Document, layer_uid: int) -> int | None:
        """Which tileset one row is (or would autovivify) bound to, or ``None``.

        Read off the *cel* when there is one and off the track otherwise, for
        :meth:`_will_be_tilemap`'s reason: a placeholder carries no binding of
        its own and the track under it is where the answer lives. One
        implementation, because three callers ask it -- the panes through
        :meth:`active_tileset_uid`, and :meth:`place_tiles`, which has to know
        *before* it autovivifies anything.
        """
        try:
            layer = self.layer_by_uid(layer_uid)
        except KeyError:  # pragma: no cover - defensive
            layer = None
        if isinstance(layer, TilemapCel):
            return int(layer.tileset_uid)
        try:
            track = self._track_of_row(layer_uid)
        except KeyError:  # pragma: no cover - off-frame rows have no track row
            return None
        return None if track is None else track.tileset_uid

    def active_tileset_uid(self: Document) -> int | None:
        """Which tileset the active row is bound to, or ``None``. The panes'
        question, which is :meth:`tileset_uid_of` about the active row."""
        uid = self.active_tilemap_uid()
        return None if uid is None else self.tileset_uid_of(uid)

    # -- geometry: the canvas resize, and the refusals that remain ----------

    def _bound_tile_sizes(self: Document) -> list[tuple[int, int]]:
        """``(tile_w, tile_h)`` for every tileset something in this document is
        bound to. Empty when nothing is."""
        return [
            (slot.tileset.tile_w, slot.tileset.tile_h)
            for slot in self.tilesets
            if self._tileset_bound(slot.uid)
        ]

    def _tile_regrid(
        self: Document, size: tuple[int, int], offset: tuple[int, int]
    ) -> Any:
        """The per-cel refs re-grid a canvas resize needs, or ``None``.

        **The one geometry op that is modelled rather than refused.** A resize
        moves no pixel *within* a tile: the picture is translated by whole
        cells and padded or cropped at the edges, which is exactly a pad/crop
        of ``refs`` and leaves every cell's flag bits alone. The flips, the
        quarter turns, the scale and the crop are still refused by
        :meth:`_refuse_tilemaps` because their permutation would have to turn
        those flags by the eight-symmetry algebra, and nothing here does that
        yet.

        A **non-tile-aligned offset** is refused by name rather than rounded:
        rounding it would move the picture somewhere the user did not ask for,
        and the alternative -- re-cutting every cell at the new phase -- is a
        different operation (it rewrites the tileset) wearing a resize's name.
        Checked against *every* bound tileset, and before anything is written,
        so a document with two tile sizes cannot be half-resized.

        Returns a callable taking one layer, which
        :meth:`~.document.Document._map_planes` applies to each distinct cel
        after the pixels have been mapped; it ignores anything that is not a
        tilemap cel, so the whole tilemap case costs the ordinary path one
        ``isinstance``.
        """
        if not self._holds_tilemap():
            return None
        ox, oy = int(offset[0]), int(offset[1])
        for tile_w, tile_h in self._bound_tile_sizes():
            if ox % tile_w or oy % tile_h:
                raise ValueError(
                    "a canvas resize of a tilemap layer needs a tile-aligned "
                    f"offset; ({ox}, {oy}) is not a whole number of "
                    f"{tile_w}x{tile_h} tiles"
                )

        def regrid(layer: Any) -> None:
            if not isinstance(layer, TilemapCel):
                return
            ts = self.tileset_slot(layer.tileset_uid).tileset
            grid_h, grid_w = grid_shape(size, ts.tile_w, ts.tile_h)
            plane = np.zeros((grid_h, grid_w), dtype=gid.DTYPE)
            tx, ty = ox // ts.tile_w, oy // ts.tile_h
            old_h, old_w = layer.refs.shape
            x0, y0 = max(0, tx), max(0, ty)
            x1, y1 = min(grid_w, tx + old_w), min(grid_h, ty + old_h)
            if x1 > x0 and y1 > y0:
                plane[y0:y1, x0:x1] = layer.refs[y0 - ty : y1 - ty, x0 - tx : x1 - tx]
            layer.refs = plane
            # Re-derived rather than kept from the mapped pixels, the module's
            # own rule: a partial cell at the *old* canvas edge was cropped,
            # and a grown canvas has room to draw the rest of that tile.
            layer.pixels = materialize(plane, ts, size)

        return regrid

    # -- geometry refusal ---------------------------------------------------

    def _holds_tilemap(self: Document) -> bool:
        """Whether anything in this document is, or would autovivify into, a
        tilemap cel.

        A track bound to a tileset counts even before any cel has been drawn on
        it -- the structural binding is what a later autovivify would turn into
        a ``TilemapCel``, and an op that ran before that happened would leave
        nothing wrong *yet*, but would be one autovivify away from it.
        """
        layers = self.stack if self.anim is None else self.anim.unique_cel_layers()
        if any(isinstance(layer, TilemapCel) for layer in layers):
            return True
        return self.anim is not None and any(
            track.tileset_uid is not None for track in self.anim.tracks
        )

    def _refuse_tilemaps(self: Document, verb: str) -> None:
        """Refuse *verb* outright when the document holds a tilemap layer.

        Called by every whole-canvas geometry op that has not yet been taught
        refs-aware permutation (Chunk 3.7).
        """
        if self._holds_tilemap():
            raise ValueError(f"a {verb} of a tilemap layer is not yet modeled")

    def _refuse_tilemap_convert(self: Document, verb: str) -> None:
        """Refuse a whole-document colour-mode conversion beside a tilemap layer.

        The three that rewrite every plane -- to indexed, to grayscale, onto a
        palette -- would leave a tilemap cel's ``pixels`` saying something its
        ``refs`` do not, because the picture is *derived* and the tileset is
        what a conversion would have to convert. Refused at the door and not
        after the first plane, which is what the ``_replay`` snapshot would
        otherwise have already half-written.

        ``convert_to_rgb`` is deliberately not among them: it drops index
        planes and rewrites no pixel, and a document that could not leave the
        mode it was in when the layer was added would simply be wedged.
        """
        if self._holds_tilemap():
            raise ValueError(
                f"a tilemap layer cannot be {verb} with the document; "
                "convert it to a raster layer first"
            )

    def _refuse_tilemap_layer(self: Document, layer_uid: int, verb: str) -> None:
        """Refuse *verb* on one layer that is (or would autovivify into) a
        tilemap cel.

        The per-layer sibling of :meth:`_refuse_tilemaps`, for the ops that
        write **one** layer's pixels and then reach ``_patch_edit_for``: merging
        onto it, lifting or cutting out of it. Each of those writes ``pixels``
        *before* the funnel's refusal can fire, so a guard there alone leaves a
        half-written layer behind an exception. A document merely holding a
        tilemap layer somewhere else is no reason to refuse any of them.
        """
        if self._will_be_tilemap(layer_uid):
            raise ValueError(f"{verb} a tilemap layer is not yet modeled")
