"""Objects on an object layer: adding, removing and re-describing them.

Addressed by their own uid *within a named layer*, never by index, which is the
travelling rule :mod:`.edits` states: deleting the object above one must not
retarget an edit to it.

Every method opens with the same two lines -- fetch the layer, refuse if it is
not an :class:`~._map_model.ObjectLayer` -- and that repetition is deliberate
rather than factored away. Each one is a separate public door, the
refusal names which uid was wrong, and a shared helper returning "the layer or
None" would put the ``KeyError`` a caller sees one frame further from the call
that earned it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ._map_model import MapObject, ObjectLayer, merged_object_values
from .edits import ObjectAddEdit, ObjectPropsEdit, ObjectRemoveEdit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


def object_bounds(obj: MapObject) -> tuple[float, float, float, float]:
    """The object's axis-aligned box in *map* pixels, rotation included.

    Headless on purpose: a marquee is a hit test over geometry, and the canvas
    is only where the rectangle is drawn. The corners are turned about
    ``(obj.x, obj.y)`` by the same clockwise rule the renderer uses -- a box
    taken from ``w``/``h`` alone would miss a rotated object the user can
    plainly see inside the band.
    """
    points = getattr(obj.shape, "points", None)
    if points:
        local = list(points)
    elif obj.w or obj.h:
        local = [(0.0, 0.0), (obj.w, 0.0), (obj.w, obj.h), (0.0, obj.h)]
    else:
        # A point (and any other extentless shape) is its own origin.
        local = [(0.0, 0.0)]
    angle = math.radians(float(obj.rotation))
    cosine, sine = math.cos(angle), math.sin(angle)
    turned = [
        (obj.x + x * cosine - y * sine, obj.y + x * sine + y * cosine) for x, y in local
    ]
    xs = [x for x, _ in turned]
    ys = [y for _, y in turned]
    return (min(xs), min(ys), max(xs), max(ys))


def objects_in_rect(
    objects: Iterable[MapObject], rect: tuple[float, float, float, float]
) -> list[int]:
    """Uids of every object whose box meets ``rect`` -- in list order.

    **Intersects rather than contains**, which is Tiled's rubber band: a
    selection that only took objects entirely inside the band would be unable
    to pick up anything bigger than the viewport.
    """
    x0, y0, x1, y1 = rect
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
    found: list[int] = []
    for obj in objects:
        bx0, by0, bx1, by1 = object_bounds(obj)
        if bx0 <= x1 and bx1 >= x0 and by0 <= y1 and by1 >= y0:
            found.append(int(obj.uid))
    return found


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

    def remove_objects(self: MapDoc, layer_uid: int, obj_uids: Iterable[int]) -> int:
        """Remove several objects in one undo step. -> how many went.

        :meth:`reorder_object`'s shape rather than a loop over
        :meth:`remove_object` at the call site: N presses of Ctrl+Z to undo one
        Delete is the defect ``compound`` exists to prevent, and it is the same
        ``ObjectRemoveEdit`` either way -- no new kind of step.

        **Highest index first**, so each recorded index is the one the object
        actually sat at; the compound undoes in reverse, which re-inserts them
        bottom-up and restores the draw order exactly.
        """
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        wanted = {int(uid) for uid in obj_uids}
        found = [
            (index, obj)
            for index, obj in enumerate(layer.objects)
            if obj.uid in wanted
        ]
        steps = []
        for index, obj in reversed(found):
            steps.append(
                ObjectRemoveEdit(layer_uid=int(layer_uid), obj=obj, index=index)
            )
            self._detach_object(layer_uid, obj)
        self.compound(steps)
        return len(steps)

    def reorder_object(self: MapDoc, layer_uid: int, obj_uid: int, delta: int) -> bool:
        """Move an object one place in its layer's draw order. -> whether it moved.

        Order *is* draw order inside an object layer, and Tiled's Raise/Lower
        are exactly this. One compound step, out of the remove/add pair that
        already exists: expressing it as two steps would put a state on the
        undo stack in which the object does not exist, which the user never
        saw and Ctrl+Z would stop at.
        """
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        index = next(
            (at for at, obj in enumerate(layer.objects) if obj.uid == obj_uid), None
        )
        if index is None:
            raise KeyError(f"no object {obj_uid}")
        wanted = index + int(delta)
        if not 0 <= wanted < len(layer.objects):
            return False
        obj = layer.objects[index]
        steps = [
            ObjectRemoveEdit(layer_uid=int(layer_uid), obj=obj, index=index),
            ObjectAddEdit(layer_uid=int(layer_uid), obj=obj, index=wanted),
        ]
        self._detach_object(layer_uid, obj)
        self._attach_object(layer_uid, obj, wanted)
        self.compound(steps)
        return True

    def set_object(self: MapDoc, layer_uid: int, obj_uid: int, **values: Any) -> None:
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        obj = next((o for o in layer.objects if o.uid == obj_uid), None)
        if obj is None:
            raise KeyError(f"no object {obj_uid}")
        before = obj.snapshot()
        # Geometry is reconciled *before* the no-op test, not after: a ``w=``
        # that resizes nothing has to compare equal, and a ``shape=`` that
        # agrees with the size already stored has to as well.
        after = merged_object_values(before, values)
        if after == before:
            return
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

    # -- the group edit session ------------------------------------------------
    #
    # The session above, over several objects, and deliberately a *parallel*
    # mechanism rather than a generalisation of it: ``begin_object_edit`` stores
    # one uid and ``end_object_edit`` pushes one ``ObjectPropsEdit``, and every
    # single-object gesture (move, resize, rotate, vertex) reads that shape.
    # Widening it to a list would have made four working gestures pay for a
    # fifth. This one holds a list, moves every member and closes into a
    # ``compound`` of the *same* ``ObjectPropsEdit`` -- one undo step, no new
    # kind of step, and the two sessions never both open because the canvas
    # arms exactly one drag kind.

    @property
    def editing_group(self: MapDoc) -> bool:
        return self._group_edit is not None

    def begin_group_edit(self: MapDoc, layer_uid: int, obj_uids: Iterable[int]) -> None:
        """Open a session over several objects on one layer.

        Every member is snapshotted **before** anything is stored, so a uid that
        is not on the layer refuses the whole call rather than opening a session
        over a partial group.
        """
        layer = self.layer(layer_uid)
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {layer_uid}")
        members = []
        for uid in dict.fromkeys(int(value) for value in obj_uids):
            obj = next((o for o in layer.objects if o.uid == uid), None)
            if obj is None:
                raise KeyError(f"no object {uid}")
            members.append({"obj_uid": uid, "before": obj.snapshot()})
        if self._group_edit is not None:
            self.end_group_edit()
        self._group_edit = {"layer_uid": int(layer_uid), "members": members}

    def move_group(self: MapDoc, dx: float, dy: float) -> None:
        """Translate every member by one offset, writing no history.

        ``place_object``'s twin, and the offset is **from where each object
        stood when the session opened** rather than from where it stands now:
        an incremental step would accumulate the rounding of every frame of a
        drag, and a drag is hundreds of frames long.
        """
        if self._group_edit is None:
            raise RuntimeError("no group edit is open")
        session = self._group_edit
        layer = self.layer(session["layer_uid"])
        if not isinstance(layer, ObjectLayer):
            raise KeyError(f"no object layer {session['layer_uid']}")
        for member in session["members"]:
            obj = next((o for o in layer.objects if o.uid == member["obj_uid"]), None)
            if obj is None:
                # Deleted mid-drag, exactly as ``end_object_edit`` allows.
                continue
            before = member["before"]
            after = merged_object_values(
                obj.snapshot(),
                {"x": float(before["x"]) + float(dx), "y": float(before["y"]) + float(dy)},
            )
            self._apply_object_props(session["layer_uid"], member["obj_uid"], after)

    def end_group_edit(self: MapDoc) -> bool:
        """Close the session and push **one** step. ``False`` if nothing moved.

        Idempotent, and forgiving of a member that vanished, for
        :meth:`end_object_edit`'s reasons -- it is closed at the same
        chokepoints and can be reached by the same missed release.
        """
        session, self._group_edit = self._group_edit, None
        if session is None:
            return False
        layer = self.layer(session["layer_uid"])
        if not isinstance(layer, ObjectLayer):
            return False
        steps = []
        for member in session["members"]:
            obj = next((o for o in layer.objects if o.uid == member["obj_uid"]), None)
            if obj is None:
                continue
            after = obj.snapshot()
            if after == member["before"]:
                continue
            steps.append(
                ObjectPropsEdit(
                    layer_uid=session["layer_uid"],
                    obj_uid=member["obj_uid"],
                    before=member["before"],
                    after=after,
                )
            )
        if not steps:
            return False
        # ``compound`` even for a single member: the gesture is a group drag
        # whichever way it ended, and a step whose shape depended on how many
        # objects happened to move would be one the history panel labels two
        # different ways for one action.
        self.compound(steps)
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
        after = merged_object_values(obj.snapshot(), values)
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
        # The shape, never the ``kind``/``w``/``h`` echo beside it: those are
        # derived, and every dict that reaches this hook came through
        # ``merged_object_values``, which made them agree.
        obj.shape = values["shape"]
        obj.x, obj.y = float(values["x"]), float(values["y"])
        obj.rotation = float(values["rotation"])
        obj.opacity = float(values["opacity"])
        obj.obj_class = str(values["obj_class"])
        obj.visible = bool(values["visible"])
        obj.properties = dict(values["properties"])
