"""Objects on an object layer: adding, removing and re-describing them.

Addressed by their own uid *within a named layer*, never by index, which is the
travelling rule :mod:`.edits` states: deleting the object above one must not
retarget an edit to it.

Every method opens with the same two lines -- fetch the layer, refuse if it is
not an :class:`~._map_model.ObjectLayer` -- and that repetition is deliberate
rather than factored away. Each of the five is a separate public door, the
refusal names which uid was wrong, and a shared helper returning "the layer or
None" would put the ``KeyError`` a caller sees one frame further from the call
that earned it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._map_model import OBJECT_KINDS, MapObject, ObjectLayer
from .edits import ObjectAddEdit, ObjectPropsEdit, ObjectRemoveEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


class ObjectOps:
    """Object placement and metadata, mixed into :class:`~.tilemap.MapDoc`."""

    def add_object(
        self: MapDoc, layer_uid: int, obj: MapObject, *, index: int | None = None
    ) -> MapObject:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        at = len(layer.objects) if index is None else max(0, min(int(index), len(layer.objects)))
        self.history.push(ObjectAddEdit(layer_uid=int(layer_uid), obj=obj, index=at))
        self._attach_object(layer_uid, obj, at)
        return obj

    def remove_object(self: MapDoc, layer_uid: int, obj_uid: int) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        for index, obj in enumerate(layer.objects):
            if obj.uid == obj_uid:
                self.history.push(
                    ObjectRemoveEdit(layer_uid=int(layer_uid), obj=obj, index=index)
                )
                self._detach_object(layer_uid, obj)
                return
        raise KeyError(f"no object {obj_uid}")

    def set_object(self: MapDoc, layer_uid: int, obj_uid: int, **values: Any) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        obj = next((o for o in layer.objects if o.uid == obj_uid), None)
        if obj is None:
            raise KeyError(f"no object {obj_uid}")
        before = obj.snapshot()
        after = {**before, **{k: v for k, v in values.items() if k in before}}
        if after == before:
            return
        if after["kind"] not in OBJECT_KINDS:
            raise ValueError(f"an object is one of {list(OBJECT_KINDS)}")
        self.history.push(
            ObjectPropsEdit(
                layer_uid=int(layer_uid), obj_uid=int(obj_uid), before=before, after=after
            )
        )
        self._apply_object_props(layer_uid, obj_uid, after)

    # -- the hooks the edits call back into ------------------------------------

    def _attach_object(self: MapDoc, layer_uid: int, obj: MapObject, index: int) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        layer.objects.insert(index, obj)

    def _detach_object(self: MapDoc, layer_uid: int, obj: MapObject) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        for index, entry in enumerate(layer.objects):
            if entry is obj:
                del layer.objects[index]
                return
        raise KeyError("that object is not on this layer")

    def _apply_object_props(
        self: MapDoc, layer_uid: int, obj_uid: int, values: dict[str, Any]
    ) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        obj = next((o for o in layer.objects if o.uid == obj_uid), None)
        if obj is None:
            raise KeyError(f"no object {obj_uid}")
        obj.name = str(values["name"])
        obj.kind = str(values["kind"])
        obj.x, obj.y = float(values["x"]), float(values["y"])
        obj.w, obj.h = float(values["w"]), float(values["h"])
        obj.obj_class = str(values["obj_class"])
        obj.visible = bool(values["visible"])
        obj.properties = dict(values["properties"])
