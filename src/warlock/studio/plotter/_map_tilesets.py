"""Which tilesets a map carries, what a gid resolves to, and which lattice.

The projection lives here rather than beside the placement arithmetic, and that
is the seam rather than an oversight: ``_map_project`` answers *where a cell is*
given a lattice, while changing which lattice a document is on is an undoable
edit -- and one that can arrive welded to a tileset, since a set drawn for one
lattice is only usable on a map that is on it. Splitting the two would put half
of one step in each file.

Public method plus private hook throughout, ``_map_layers``' rule: the hooks are
the tileset list's only mutators and the public methods are the only things that
record, so an undone step never pushes a step of its own.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from ..tilegrid import gid as gidlib
from ..tilegrid.tileset import Tileset, TilesetRef
from . import project
from .edits import (
    Edit,
    ProjectionEdit,
    TileMetaEdit,
    TilesetAddEdit,
    TilesetRemoveEdit,
    TilesetReplaceEdit,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


class TilesetOps:
    """Tilesets, gid resolution and the projection, mixed into ``MapDoc``."""

    # -- resolution ------------------------------------------------------------

    def resolve(self: MapDoc, cell: int) -> tuple[Tileset, int] | None:
        """``(tileset, local id)`` for one cell, or ``None`` for an empty one.

        Takes the *encoded* value, flags and all, so callers never have to
        remember to mask -- forgetting to is how a flipped tile comes to render
        as an out-of-range id.
        """
        tile_id = int(cell) & gidlib.GID_MASK
        if tile_id == 0:
            return None
        for ref in self.tilesets:
            if ref.holds(tile_id):
                return ref.tileset, ref.local(tile_id)
        return None

    def ref_for(self: MapDoc, tile_id: int) -> TilesetRef | None:
        masked = int(tile_id) & gidlib.GID_MASK
        for ref in self.tilesets:
            if ref.holds(masked):
                return ref
        return None

    @property
    def next_firstgid(self: MapDoc) -> int:
        """Where the next tileset's ids would begin.

        Contiguous, and 1-based because zero means "no tile" in every Tiled
        file ever written.
        """
        return 1 if not self.tilesets else self.tilesets[-1].last_gid + 1

    # -- changes ---------------------------------------------------------------

    def add_tileset(self: MapDoc, tileset: Tileset, *, source: str = "") -> TilesetRef:
        ref = TilesetRef(firstgid=self.next_firstgid, tileset=tileset, source=source)
        self.history.push(TilesetAddEdit(ref=ref))
        self._attach_tileset(ref)
        return ref

    def tileset_usage(self: MapDoc, index: int) -> tuple[int, str]:
        """How many cells hold a gid from this tileset, and where. -> (count, layer).

        Pure, and the whole of the refusal's evidence: a removal that silently
        blanked forty-two cells is the failure this exists to stop, and a count
        with no layer name is a refusal the user cannot act on.
        """
        import numpy as np

        if index < 0 or index >= len(self.tilesets):
            return (0, "")
        ref = self.tilesets[index]
        low = int(ref.firstgid)
        high = int(ref.last_gid)
        worst_count, worst_name = 0, ""
        total = 0
        for layer in self.layers:
            data = getattr(layer, "data", None)
            if data is None:
                continue
            ids = gidlib.tile_ids(np.asarray(data))
            held = int(np.count_nonzero((ids >= low) & (ids <= high)))
            total += held
            if held > worst_count:
                worst_count, worst_name = held, layer.name
        return (total, worst_name)

    def remove_tileset(self: MapDoc, index: int) -> None:
        """Take a tileset out of the map, refusing while anything still uses it.

        **The refusal is the feature.** A gid is a firstgid plus a local id, so
        dropping a tileset out from under painted cells does not clear them --
        it renumbers what they mean. The message names the tileset, the count
        and the layer, because "it is still in use" is a sentence the user
        cannot act on.

        Survivors keep their firstgids. A hole in gid space is legal: ``wmap``
        requires only that they increase.
        """
        if index < 0 or index >= len(self.tilesets):
            raise IndexError(f"no tileset {index}")
        used, where = self.tileset_usage(index)
        ref = self.tilesets[index]
        if used:
            name = ref.tileset.name or "this tileset"
            raise ValueError(
                f"{name} is still used by {used} cell(s)"
                + (f" on {where}" if where else "")
            )
        self.history.push(TilesetRemoveEdit(index=index, ref=ref))
        self._detach_tileset(ref)

    def set_tile_meta(self: MapDoc, index: int, local_id: int, meta: Any) -> bool:
        """Set (or clear) one tile's metadata on one tileset, undoably.

        ``None`` -- or an empty record -- removes the entry, which is the sparse
        store's own rule: a tile whose metadata is all defaults is a tile with
        no metadata, and keeping the record would grow a ``<tile>`` element every
        time somebody opened a class field and closed it again.

        The tileset is rebuilt rather than written through, which is
        :class:`~..tilegrid.tileset.Tileset`'s standing rule: it is frozen
        because the UI keys its texture upload on identity.
        """
        if index < 0 or index >= len(self.tilesets):
            raise IndexError(f"no tileset {index}")
        current = self.tilesets[index].tileset.tiles.get(int(local_id))
        wanted = None if meta is None or meta.is_empty else meta
        if current == wanted:
            return False
        self.history.push(
            TileMetaEdit(
                index=int(index),
                local_id=int(local_id),
                before=current,
                after=wanted,
            )
        )
        self._apply_tile_meta(index, local_id, wanted)
        return True

    def _apply_tile_meta(self: MapDoc, index: int, local_id: int, meta: Any) -> None:
        """The hook :class:`~.edits.TileMetaEdit` calls back into."""
        ref = self.tilesets[int(index)]
        self.tilesets[int(index)] = dataclasses.replace(
            ref, tileset=ref.tileset.with_meta(int(local_id), meta)
        )
        # The palette draws from the tileset object, so the epoch has to move or
        # a class typed into the form would not appear until something else did.
        self.tileset_epoch += 1

    def replace_tileset(self: MapDoc, index: int, tileset: Tileset) -> TilesetRef:
        """Swap one tileset's art, keeping its ids and its declared terrains.

        What the Inker polish round trip comes back through. A different tile
        count is **refused by name**, because that is exactly the case which
        would renumber: same count and same ``firstgid`` means every cell
        already painted keeps its tile and simply redraws.

        The terrains come from the *old* set unless the new one declares its
        own, so an atlas that went out to a paint program and came back as plain
        pixels does not silently stop being a terrain set.
        """
        at = int(index)
        if at < 0 or at >= len(self.tilesets):
            raise IndexError(f"no tileset {at} in this map")
        old = self.tilesets[at]
        if tileset.tile_count != old.tileset.tile_count:
            raise ValueError(
                f"{old.tileset.name} holds {old.tileset.tile_count} tiles and the "
                f"replacement holds {tileset.tile_count}; the map's cells are numbered "
                "against the old count, so this would renumber them"
            )
        if not tileset.terrains and old.tileset.terrains:
            tileset = dataclasses.replace(tileset, terrains=old.tileset.terrains)
        ref = TilesetRef(firstgid=old.firstgid, tileset=tileset, source=old.source)
        self.history.push(TilesetReplaceEdit(index=at, before=old, after=ref))
        self._swap_tileset(at, ref)
        return ref

    def set_projection(
        self: MapDoc, projection: str, *, adding: Tileset | None = None, source: str = ""
    ) -> None:
        """Change the lattice, optionally adopting a tileset in the same step.

        ``adding`` exists for the case where the two arrive together: two steps
        would leave a Ctrl+Z on a map whose only tileset is drawn for the
        lattice it is no longer on.

        No door pairs them today. The one that did was the ground generator,
        deleted on 2026-08-18, and a map's projection is now chosen in the
        new-map dialog instead. The pairing is kept rather than unpicked because
        it is the *correct* composition for any door that carries a lattice with
        its art -- a Tiled ``.tsx`` that declares one is the obvious next -- and
        re-deriving it later means re-deriving the one-undo-step argument too.
        """
        want = project.check(projection)
        steps: list[Edit] = []
        if want != self.projection:
            steps.append(ProjectionEdit(before=self.projection, after=want))
            self._set_projection(want)
        if adding is not None:
            ref = TilesetRef(firstgid=self.next_firstgid, tileset=adding, source=source)
            steps.append(TilesetAddEdit(ref=ref))
            self._attach_tileset(ref)
        if len(steps) == 1:
            self.history.push(steps[0])
        else:
            self.compound(steps)

    # -- the hooks the edits call back into ------------------------------------

    def _attach_tileset(self: MapDoc, ref: TilesetRef) -> None:
        self.tilesets.append(ref)
        self.tileset_epoch += 1

    def _insert_tileset(self: MapDoc, index: int, ref: TilesetRef) -> None:
        """Put a removed set back where it was. The undo half of a removal.

        By index rather than by append: the list is in firstgid order and a set
        put back at the end would be out of it, which ``wmap`` refuses on the
        next save.
        """
        self.tilesets.insert(max(0, min(int(index), len(self.tilesets))), ref)
        self.tileset_epoch += 1

    def _detach_tileset(self: MapDoc, ref: TilesetRef) -> None:
        for index, entry in enumerate(self.tilesets):
            if entry is ref:
                del self.tilesets[index]
                self.tileset_epoch += 1
                return
        raise KeyError("that tileset is not in this map")

    def _set_projection(self: MapDoc, projection: str) -> None:
        self.projection = project.check(projection)

    def _swap_tileset(self: MapDoc, index: int, ref: TilesetRef) -> None:
        self.tilesets[int(index)] = ref
        self.tileset_epoch += 1
