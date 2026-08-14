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
        # ``0`` is "unassigned" -- every caller that builds a fresh object
        # leaves it at the dataclass default, and this is the one place that
        # mints a real one. A caller that already set one (adopted from an
        # imported file, say) keeps it: this is an "at creation" assignment,
        # not an overwrite.
        if obj.id == 0:
            obj.id = self.next_object_id
            self.next_object_id += 1
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

    # -- the object edit session -----------------------------------------------
    #
    # ``_map_paint``'s stroke session, for objects. A drag mutates the object
    # live and pushes nothing; one ``ObjectPropsEdit`` over the whole gesture
    # lands at release. Without it a drag across the canvas would push one step
    # per frame -- the same defect the stroke session exists to answer, and the
    # same shape of answer rather than a second mechanism.

    @property
    def editing_object(self: MapDoc) -> bool:
        return self._object_edit is not None

    def begin_object_edit(self: MapDoc, layer_uid: int, obj_uid: int) -> None:
        """Open a session on one object. Re-opening closes the previous one."""
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        obj = next((o for o in layer.objects if o.uid == obj_uid), None)
        if obj is None:
            raise KeyError(f"no object {obj_uid}")
        if self._object_edit is not None:
            self.end_object_edit()
        self._object_edit = {
            "layer_uid": int(layer_uid),
            "obj_uid": int(obj_uid),
            "before": obj.snapshot(),
        }

    def end_object_edit(self: MapDoc) -> bool:
        """Close the session and push one step. ``False`` if nothing moved.

        Idempotent for ``end_stroke``'s reason, and it is the same list of
        recovery paths: a release can be missed to focus loss, a tab switch, Esc
        or a save beginning mid-drag, and none of those should have to know
        whether a drag was open. A click that never moved the object finds no
        diff and so pushes nothing, which is the no-op rule reaching a gesture.
        """
        session, self._object_edit = self._object_edit, None
        if session is None:
            return False
        layer = self.layer(session["layer_uid"])
        if not isinstance(layer, ObjectLayer):
            return False
        obj = next((o for o in layer.objects if o.uid == session["obj_uid"]), None)
        if obj is None:
            # Deleted mid-drag. The remove is its own step and already on the
            # stack; there is nothing left to describe.
            return False
        after = obj.snapshot()
        if after == session["before"]:
            return False
        self.history.push(
            ObjectPropsEdit(
                layer_uid=session["layer_uid"],
                obj_uid=session["obj_uid"],
                before=session["before"],
                after=after,
            )
        )
        return True

    def place_object(self: MapDoc, **values: Any) -> None:
        """Move or resize inside an open session, writing no history.

        ``set_object``'s live half, and it deliberately refuses outside a
        session rather than silently falling back to the undoable path: a caller
        that forgot to open one would otherwise push a step per frame and only
        be caught by counting the undo stack.
        """
        if self._object_edit is None:
            raise RuntimeError("no object edit is open")
        session = self._object_edit
        layer = self.layer(session["layer_uid"])
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {session['layer_uid']}")
        obj = next((o for o in layer.objects if o.uid == session["obj_uid"]), None)
        if obj is None:
            raise KeyError(f"no object {session['obj_uid']}")
        after = {**obj.snapshot(), **{k: v for k, v in values.items() if k in obj.snapshot()}}
        if after["kind"] not in OBJECT_KINDS:
            raise ValueError(f"an object is one of {list(OBJECT_KINDS)}")
        self._apply_object_props(session["layer_uid"], session["obj_uid"], after)

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
