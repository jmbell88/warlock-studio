"""The layer stack: what is on it, which one is active, and how it is reordered.

Split from ``tilemap.py`` for the raster editor's reason -- one document class
answering to seven unrelated questions is seven reasons to open one file -- and
along the same seam: everything here is *about the list*, and nothing here
writes a gid.

The pairing that runs through the module is public method plus private hook.
``remove_layer`` pushes a :class:`~.edits.LayerRemoveEdit` and then calls
``_detach_layer``; undoing that edit calls ``_attach_layer`` directly. So the
hooks are the layer list's only mutators and the public methods are the only
things that record, which is what keeps an undone step from pushing a step of
its own. The hooks resolve through the MRO, so ``edits.py`` reaches them on the
composed :class:`~.tilemap.MapDoc` exactly as it did when they were defined
there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import gid as gidlib
from ._map_model import DRAW_ORDERS, Layer, ObjectLayer, TileLayer, new_uid
from .edits import LayerAddEdit, LayerMoveEdit, LayerPropsEdit, LayerRemoveEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


class LayerOps:
    """The layer list and its undoable changes, mixed into ``MapDoc``."""

    # -- lookup ----------------------------------------------------------------

    def layer(self: MapDoc, uid: int) -> Layer | None:
        for entry in self.layers:
            if entry.uid == uid:
                return entry
        return None

    def index_of(self: MapDoc, uid: int) -> int:
        for index, entry in enumerate(self.layers):
            if entry.uid == uid:
                return index
        raise KeyError(f"no layer {uid}")

    def tile_layers(self: MapDoc) -> list[TileLayer]:
        return [entry for entry in self.layers if isinstance(entry, TileLayer)]

    def active(self: MapDoc) -> Layer | None:
        return None if self.active_layer is None else self.layer(self.active_layer)

    def set_active_layer(self: MapDoc, uid: int | None) -> None:
        """View state; pushes no step, exactly as ``set_active_layer`` does in
        the raster editor and for the same reason."""
        self.active_layer = None if uid is None else int(uid)

    # -- changes ---------------------------------------------------------------

    def add_tile_layer(self: MapDoc, name: str = "", *, index: int | None = None) -> TileLayer:
        layer = TileLayer(
            uid=new_uid(),
            name=name or f"Tiles {len(self.tile_layers()) + 1}",
            data=gidlib.empty_layer(self.width, self.height),
        )
        return self._add_layer(layer, index)

    def add_object_layer(
        self: MapDoc, name: str = "", *, index: int | None = None
    ) -> ObjectLayer:
        layer = ObjectLayer(uid=new_uid(), name=name or "Objects")
        return self._add_layer(layer, index)

    def _add_layer(self: MapDoc, layer: Any, index: int | None) -> Any:
        at = len(self.layers) if index is None else max(0, min(int(index), len(self.layers)))
        self.history.push(LayerAddEdit(layer=layer, index=at))
        self._attach_layer(layer, at)
        self.active_layer = layer.uid
        return layer

    def remove_layer(self: MapDoc, uid: int) -> None:
        index = self.index_of(uid)
        layer = self.layers[index]
        self.history.push(LayerRemoveEdit(layer=layer, index=index))
        self._detach_layer(layer)

    def move_layer(self: MapDoc, uid: int, to_index: int) -> None:
        before = self.index_of(uid)
        after = max(0, min(int(to_index), len(self.layers) - 1))
        if before == after:
            return
        self.history.push(
            LayerMoveEdit(layer_uid=int(uid), before_index=before, after_index=after)
        )
        self._relocate(uid, after)

    def set_layer_props(self: MapDoc, uid: int, **values: Any) -> None:
        """Rename, hide, fade or re-key a layer.

        Only the keys handed in are touched, and a call that changes none of
        them pushes nothing.
        """
        layer = self.layer(uid)
        if layer is None:
            raise KeyError(f"no layer {uid}")
        before = layer.snapshot()
        after = {**before, **{k: v for k, v in values.items() if k in before}}
        if after == before:
            return
        # Refused before the push, not inside ``_apply_layer_props``: that hook
        # is also the undo path, and a step that raises halfway through leaves
        # the stack describing a change the document never made.
        if after.get("draworder", "topdown") not in DRAW_ORDERS:
            raise ValueError(
                f"a draw order is one of {list(DRAW_ORDERS)}, not {after['draworder']!r}"
            )
        self.history.push(LayerPropsEdit(layer_uid=int(uid), before=before, after=after))
        self._apply_layer_props(uid, after)

    # -- the hooks the edits call back into ------------------------------------

    def _attach_layer(self: MapDoc, layer: Any, index: int) -> None:
        self.layers.insert(index, layer)

    def _detach_layer(self: MapDoc, layer: Any) -> None:
        for index, entry in enumerate(self.layers):
            if entry is layer:
                del self.layers[index]
                break
        else:
            raise KeyError("that layer is not in this map")
        if self.active_layer == layer.uid:
            # By uid, so the fallback is a layer that still exists rather than
            # whichever one slid into the removed one's index.
            self.active_layer = self.layers[-1].uid if self.layers else None

    def _relocate(self: MapDoc, uid: int, to_index: int) -> None:
        index = self.index_of(uid)
        layer = self.layers.pop(index)
        self.layers.insert(max(0, min(int(to_index), len(self.layers))), layer)

    def _apply_layer_props(self: MapDoc, uid: int, values: dict[str, Any]) -> None:
        layer = self.layer(uid)
        if layer is None:
            raise KeyError(f"no layer {uid}")
        # Every key ``snapshot`` reports has to be assigned here. ``set_layer_props``
        # filters its kwargs through the snapshot, so a new field is *recorded*
        # in the edit for free -- but undo and redo replay through this hook, so
        # a field missing from it is one the user can set and never take back.
        layer.name = str(values["name"])
        layer.visible = bool(values["visible"])
        layer.opacity = float(values["opacity"])
        layer.locked = bool(values["locked"])
        # Only object layers have one, and a snapshot only carries the keys its
        # own layer kind reports -- so this is asked of the values rather than
        # of the layer, which keeps one hook serving both kinds.
        if "draworder" in values:
            layer.draworder = str(values["draworder"])
        layer.properties = dict(values["properties"])
