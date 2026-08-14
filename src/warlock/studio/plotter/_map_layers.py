"""The layer tree: what is on it, which one is active, and how it is reordered.

Split from ``tilemap.py`` for the raster editor's reason -- one document class
answering to seven unrelated questions is seven reasons to open one file -- and
along the same seam: everything here is *about the list*, and nothing here
writes a gid.

**``doc.layers`` is the root list, not the layer list.** A
:class:`~._map_model.GroupLayer` holds children of its own, so every lookup here
walks rather than scans, and ``index_of`` answers a layer's position *within its
own parent* -- which is the only index a reorder can act on. The one funnel is
:meth:`LayerOps.children_of`: hand it ``None`` for the root list or a group's
uid for that group's children, and it is the only place either is reached for.
An index alone is therefore no longer an address, which is why
:class:`~.edits.LayerMoveEdit` records a ``(parent_uid, index)`` pair on each
side: reparenting and reordering are the same gesture and must be one step.

The pairing that runs through the module is public method plus private hook.
``remove_layer`` pushes a :class:`~.edits.LayerRemoveEdit` and then calls
``_detach_layer``; undoing that edit calls ``_attach_layer`` directly. So the
hooks are the layer tree's only mutators and the public methods are the only
things that record, which is what keeps an undone step from pushing a step of
its own. The hooks resolve through the MRO, so ``edits.py`` reaches them on the
composed :class:`~.tilemap.MapDoc` exactly as it did when they were defined
there.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from . import gid as gidlib
from ._map_model import (
    DRAW_ORDERS,
    GroupLayer,
    ImageLayer,
    Layer,
    ObjectLayer,
    TileLayer,
    new_uid,
    normalize_layer_values,
)
from .edits import LayerAddEdit, LayerMoveEdit, LayerPropsEdit, LayerRemoveEdit
from .tileset import rgba_colour

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc

# "The parent you already have". ``None`` is a real parent -- the root list --
# so a nullable default could not distinguish "move to the root" from "leave it
# where it is", and those are two different gestures.
_KEEP = object()


def _walk(
    layers: list[Layer], parent_uid: int | None, depth: int
) -> Iterator[tuple[Layer, int | None, int, int]]:
    """``(layer, parent_uid, index, depth)`` depth-first, in paint order.

    Depth-first *pre-order*: a group is yielded before its children, and its
    children before the next layer beside it. That is paint order -- a group's
    contents draw over what is under the group and beneath what follows it --
    so every consumer that wants the flat picture gets it by filtering, and
    nothing has to re-derive an order.
    """
    for index, layer in enumerate(layers):
        yield layer, parent_uid, index, depth
        if isinstance(layer, GroupLayer):
            yield from _walk(layer.children, layer.uid, depth + 1)


class LayerOps:
    """The layer tree and its undoable changes, mixed into ``MapDoc``."""

    # -- lookup ----------------------------------------------------------------

    def walk(self: MapDoc) -> Iterator[tuple[Layer, int | None, int, int]]:
        """Every layer in the document, depth-first, with where it sits."""
        return _walk(self.layers, None, 0)

    def all_layers(self: MapDoc) -> list[Layer]:
        """Every layer, groups included, in depth-first paint order."""
        return [entry for entry, _parent, _index, _depth in self.walk()]

    def children_of(self: MapDoc, parent_uid: int | None) -> list[Layer]:
        """The live list one parent's children are stored in.

        ``None`` is the root list. **The list itself, not a copy** -- this is
        the funnel every mutator inserts into and deletes from, and handing back
        a copy would make three of them silently do nothing.
        """
        if parent_uid is None:
            return self.layers
        group = self.layer(int(parent_uid))
        if not isinstance(group, GroupLayer):
            raise KeyError(f"no group layer {parent_uid}")
        return group.children

    def _locate(self: MapDoc, uid: int) -> tuple[Layer, int | None, int] | None:
        """``(layer, parent_uid, index)``, or ``None``. Every lookup's engine."""
        for entry, parent_uid, index, _depth in self.walk():
            if entry.uid == uid:
                return entry, parent_uid, index
        return None

    def layer(self: MapDoc, uid: int) -> Layer | None:
        found = self._locate(uid)
        return None if found is None else found[0]

    def index_of(self: MapDoc, uid: int) -> int:
        """Where a layer sits **within its own parent's list**.

        Not a position in some flattened enumeration: an index is only ever
        used to reorder, and reordering happens among siblings.
        """
        found = self._locate(uid)
        if found is None:
            raise KeyError(f"no layer {uid}")
        return found[2]

    def parent_uid_of(self: MapDoc, uid: int) -> int | None:
        """The group a layer is in, or ``None`` when it sits at the root."""
        found = self._locate(uid)
        if found is None:
            raise KeyError(f"no layer {uid}")
        return found[1]

    def tile_layers(self: MapDoc) -> list[TileLayer]:
        """The tile leaves, depth-first, in paint order.

        The shape every caller already had -- the ``.wmap`` array enumeration,
        the flat renderer, the minimap -- which is why the tree landed as a walk
        behind this name rather than as a new one beside it.
        """
        return [entry for entry in self.all_layers() if isinstance(entry, TileLayer)]

    def tree_shape(self: MapDoc) -> tuple[tuple[int, int], ...]:
        """A cheap fingerprint of the tree's *structure*: uid and depth, in order.

        For a cache key that has to notice a layer moving. ``len(self.layers)``
        was that key while the stack was flat and cannot be now: adding a layer
        inside a group, or dragging one out of it, leaves the root list exactly
        as long as it was.
        """
        return tuple((entry.uid, depth) for entry, _parent, _index, depth in self.walk())

    def active(self: MapDoc) -> Layer | None:
        return None if self.active_layer is None else self.layer(self.active_layer)

    def set_active_layer(self: MapDoc, uid: int | None) -> None:
        """View state; pushes no step, exactly as ``set_active_layer`` does in
        the raster editor and for the same reason."""
        self.active_layer = None if uid is None else int(uid)

    # -- changes ---------------------------------------------------------------

    def add_tile_layer(
        self: MapDoc, name: str = "", *, index: int | None = None, parent_uid: int | None = None
    ) -> TileLayer:
        layer = TileLayer(
            uid=new_uid(),
            id=self._mint_layer_id(),
            name=name or f"Tiles {len(self.tile_layers()) + 1}",
            data=gidlib.empty_layer(self.width, self.height),
        )
        return self._add_layer(layer, index, parent_uid)

    def add_object_layer(
        self: MapDoc, name: str = "", *, index: int | None = None, parent_uid: int | None = None
    ) -> ObjectLayer:
        layer = ObjectLayer(uid=new_uid(), id=self._mint_layer_id(), name=name or "Objects")
        return self._add_layer(layer, index, parent_uid)

    def add_group_layer(
        self: MapDoc, name: str = "", *, index: int | None = None, parent_uid: int | None = None
    ) -> GroupLayer:
        """A new, empty group. Nothing moves into it -- that is a reparent.

        Deliberately not "group the selection": which layers a group is made
        *of* is an authoring gesture with a selection model behind it, and the
        document's job is only to be able to hold the result.
        """
        layer = GroupLayer(uid=new_uid(), id=self._mint_layer_id(), name=name or "Group")
        return self._add_layer(layer, index, parent_uid)

    def add_image_layer(
        self: MapDoc,
        name: str = "",
        *,
        pixels: Any = None,
        source: str = "",
        repeat_x: bool = False,
        repeat_y: bool = False,
        tint: Any = None,
        index: int | None = None,
        parent_uid: int | None = None,
    ) -> ImageLayer:
        layer = ImageLayer(
            uid=new_uid(),
            id=self._mint_layer_id(),
            name=name or "Image",
            source=str(source),
            repeat_x=bool(repeat_x),
            repeat_y=bool(repeat_y),
            **({} if pixels is None else {"pixels": np.asarray(pixels)}),
            **({} if tint is None else {"tint": tint}),
        )
        return self._add_layer(layer, index, parent_uid)

    def _mint_layer_id(self: MapDoc) -> int:
        """The next persistent layer id, and advance past it.

        One namespace for every layer kind -- ``next_layer_id`` does not care
        which -- matching the uid counter's own reason: nothing downstream
        should have to ask which kind of layer an id belongs to.
        """
        minted = self.next_layer_id
        self.next_layer_id += 1
        return minted

    def _add_layer(
        self: MapDoc, layer: Any, index: int | None, parent_uid: int | None = None
    ) -> Any:
        siblings = self.children_of(parent_uid)
        at = len(siblings) if index is None else max(0, min(int(index), len(siblings)))
        self.history.push(LayerAddEdit(layer=layer, index=at, parent_uid=parent_uid))
        self._attach_layer(layer, at, parent_uid)
        self.active_layer = layer.uid
        return layer

    def remove_layer(self: MapDoc, uid: int) -> None:
        """Take a layer out of the tree, with everything under it.

        A group travels as one object, so its whole subtree is one step: the
        edit holds the group, the group holds its children, and the cost the
        byte budget sees is every array underneath it.
        """
        found = self._locate(uid)
        if found is None:
            raise KeyError(f"no layer {uid}")
        layer, parent_uid, index = found
        self.history.push(LayerRemoveEdit(layer=layer, index=index, parent_uid=parent_uid))
        self._detach_layer(layer)

    def move_layer(
        self: MapDoc, uid: int, to_index: int, *, parent_uid: Any = _KEEP
    ) -> None:
        """Reorder within a parent, or move to a different one, in one step.

        One method rather than a ``move`` and a ``reparent``, because they are
        one gesture: a drag in the layer pane both changes the parent and lands
        at a position, and expressing that as two steps would put a state on the
        undo stack the user never saw.
        """
        found = self._locate(uid)
        if found is None:
            raise KeyError(f"no layer {uid}")
        layer, before_parent, before_index = found
        after_parent = (
            before_parent
            if parent_uid is _KEEP
            else (None if parent_uid is None else int(parent_uid))
        )
        if after_parent is not None:
            target = self.layer(after_parent)
            if not isinstance(target, GroupLayer):
                raise KeyError(f"no group layer {after_parent}")
            # A group moved inside itself is a cycle: the subtree would leave
            # the document with nothing holding it and every walk after that
            # would either miss it or never end. Refused rather than clamped,
            # because there is no nearby position that is what was asked for.
            if after_parent == uid or _contains(layer, after_parent):
                raise ValueError("a group cannot be moved inside itself")
        siblings = self.children_of(after_parent)
        # The layer is still in the tree while this clamps, so its own slot
        # counts as a position only when it is not about to leave the list.
        limit = len(siblings) - 1 if after_parent == before_parent else len(siblings)
        after_index = max(0, min(int(to_index), max(0, limit)))
        if (after_parent, after_index) == (before_parent, before_index):
            return
        self.history.push(
            LayerMoveEdit(
                layer_uid=int(uid),
                before=(before_parent, before_index),
                after=(after_parent, after_index),
            )
        )
        self._relocate(uid, (after_parent, after_index))

    def set_layer_props(self: MapDoc, uid: int, **values: Any) -> None:
        """Rename, hide, fade, offset, tint or re-key a layer.

        Only the keys handed in are touched, and a call that changes none of
        them pushes nothing.
        """
        layer = self.layer(uid)
        if layer is None:
            raise KeyError(f"no layer {uid}")
        before = layer.snapshot()
        # Coerced and refused *before* the no-op test, so that a caller spelling
        # an unchanged value differently -- a tint as a list, an opacity as an
        # int -- still compares equal and still pushes nothing. This is also
        # what stops ``set_layer_props`` accepting a tint the constructor would
        # refuse; see ``normalize_layer_values``.
        after = normalize_layer_values(
            {**before, **{k: v for k, v in values.items() if k in before}}
        )
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

    def _attach_layer(
        self: MapDoc, layer: Any, index: int, parent_uid: int | None = None
    ) -> None:
        siblings = self.children_of(parent_uid)
        siblings.insert(max(0, min(int(index), len(siblings))), layer)

    def _detach_layer(self: MapDoc, layer: Any) -> None:
        for entry, parent_uid, index, _depth in self.walk():
            if entry is layer:
                del self.children_of(parent_uid)[index]
                break
        else:
            raise KeyError("that layer is not in this map")
        # By uid and by *lookup*, not by comparing against the removed layer's
        # own uid: removing a group takes its whole subtree with it, so the
        # active layer may be something several levels down that this call has
        # just made unreachable. The fallback is therefore "is what I was on
        # still here", and the answer when it is not is a layer that exists.
        if self.active_layer is not None and self.layer(self.active_layer) is None:
            remaining = self.all_layers()
            self.active_layer = remaining[-1].uid if remaining else None

    def _relocate(self: MapDoc, uid: int, target: tuple[int | None, int]) -> None:
        parent_uid, index = target
        found = self._locate(uid)
        if found is None:
            raise KeyError(f"no layer {uid}")
        layer, from_parent, from_index = found
        del self.children_of(from_parent)[from_index]
        siblings = self.children_of(parent_uid)
        siblings.insert(max(0, min(int(index), len(siblings))), layer)

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
        layer.class_name = str(values["class_name"])
        # Through the validator rather than a bare ``tuple``, even though
        # ``set_layer_props`` has already refused anything this could reject:
        # this hook is reachable from an edit, and a field whose only guard is
        # one call site is a field that stops being guarded the day a second one
        # appears. It cannot raise on a value that came from a snapshot.
        layer.tint = rgba_colour(values["tint"], "a layer tint")
        layer.offset_x = float(values["offset_x"])
        layer.offset_y = float(values["offset_y"])
        layer.parallax_x = float(values["parallax_x"])
        layer.parallax_y = float(values["parallax_y"])
        # Only some kinds have these, and a snapshot only carries the keys its
        # own layer kind reports -- so they are asked of the values rather than
        # of the layer, which keeps one hook serving all four.
        if "draworder" in values:
            layer.draworder = str(values["draworder"])
        if "source" in values:
            layer.source = str(values["source"])
            layer.repeat_x = bool(values["repeat_x"])
            layer.repeat_y = bool(values["repeat_y"])
        layer.properties = dict(values["properties"])


def _contains(group: Any, uid: int) -> bool:
    """Whether ``uid`` is anywhere under ``group``. ``False`` for a non-group."""
    for child in getattr(group, "children", ()) or ():
        if child.uid == uid or _contains(child, uid):
            return True
    return False
