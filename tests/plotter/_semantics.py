"""What a map *is*, minus everything that is not the document.

The comparator every Plotter↔Tiled gate is built on. Two documents are the
same map when these facts match, and the three things deliberately absent are
the three that are not properties of the document:

- **uids**, minted per process and meaningless across a save;
- **byte encodings**, so a CSV layer and a zlib layer compare equal -- which
  is the entire point of reading the same fixture twice, once as ``.tmx`` and
  once as ``.tmj``;
- **float spelling**, because a coordinate that went out through a text format
  and came back must not fail a gate over its last bit.

Everything else is in, and the test module beside this one has one case per
field to keep it that way: a comparator with a blind spot turns every gate
built on it into a test that cannot fail.

In ``tests/`` rather than in the engine because it is a *test* vocabulary --
the engine has no reason to be able to fingerprint itself, and putting it in
``plotter/`` would add a module to a package whose import set is pinned.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import numpy as np

# Coordinates and opacities are compared to this many decimals. Six is well
# inside what a Tiled file writes and well outside float noise.
_PLACES = 6


def _num(value: Any) -> float:
    """A float as the comparator sees it: rounded, and never negative zero."""
    return round(float(value), _PLACES) + 0.0


def _digest(array: np.ndarray) -> str:
    """A short, stable fingerprint of an array's bytes.

    Hashed rather than inlined because an atlas is thousands of ints and a
    failing assert has to stay readable. ``shape`` travels beside every call
    site so a size mismatch still reports as a size mismatch.
    """
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:16]


def _property_value(value: Any) -> Any:
    """A recursively JSON-shaped property value, including Tiled 1.12 lists."""
    if hasattr(value, "type") and hasattr(value, "value"):
        return _prop(value)
    if isinstance(value, dict):
        return {name: _property_value(value[name]) for name in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_property_value(item) for item in value]
    if isinstance(value, float):
        return _num(value)
    return value


def _prop(value: Any) -> Any:
    """One custom property. Handles both spellings the codebase carries.

    ``properties`` is typed ``dict[str, Any]`` and holds ``tsx.Prop`` in
    practice; a plain value is accepted rather than refused so this never
    becomes the reason a test cannot express a document.
    """
    kind = getattr(value, "type", None)
    if kind is None:
        return ["untyped", _property_value(value)]
    raw = value.value
    propertytype = getattr(value, "propertytype", "")
    return [kind, propertytype, _property_value(raw)]


def _props(properties: dict[str, Any]) -> dict[str, Any]:
    return {name: _prop(properties[name]) for name in sorted(properties)}


def _terrain_facts(spec: Any) -> list[Any]:
    return [spec.name, list(spec.fill), list(spec.outline)]


def _tileset_facts(ref: Any) -> dict[str, Any]:
    tileset = ref.tileset
    return {
        "firstgid": int(ref.firstgid),
        # ``ref.source`` is deliberately absent. Path spelling is packaging,
        # not tileset semantics: Plotter emits a portable, collision-free
        # ``tilesets/`` bundle even when the Tiled source kept the TSX beside
        # the map.
        #
        # **What that omission also hides, stated so it is a decision and not a
        # blind spot**: an *embedded* atlas has no source at all going in and
        # comes back out as an external ``.tsx``, because both exporters write
        # every tileset as a ``source=`` reference. The atlas, its properties
        # and its terrains survive intact -- which is what this comparison is
        # for -- but "embedded" does not, and ``docs/PLOTTER_COMPAT.md``'s
        # `embedded atlas tilesets` row says so rather than leaving the reader
        # to infer it from a passing round trip.
        "name": tileset.name,
        "class_name": tileset.class_name,
        "grid": [tileset.grid_orientation, tileset.grid_width, tileset.grid_height],
        "transformations": list(tileset.transformations),
        "tile_w": int(tileset.tile_w),
        "tile_h": int(tileset.tile_h),
        "spacing": int(tileset.spacing),
        "margin": int(tileset.margin),
        "columns": int(tileset.columns),
        "rows": int(tileset.rows),
        "image_shape": list(np.asarray(tileset.pixels).shape),
        "image": _digest(tileset.pixels),
        "properties": _props(tileset.properties),
        "terrains": [_terrain_facts(spec) for spec in tileset.terrains],
    }


def _plain(value: Any) -> Any:
    """One field of a shape, in the JSON-shaped, float-tolerant form."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _num(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _geometry_facts(shape: Any) -> dict[str, Any]:
    """Every field of an object's shape, read off the dataclass.

    Enumerated rather than listed, which is the opposite of how the rest of
    this module works and is right exactly here: a shape is a frozen dataclass
    of plain values, the union will gain more, and a
    hand-written list would be a blind spot per shape rather than one for the
    whole file. ``kind``, ``w`` and ``h`` are already above -- this is what
    those three cannot say: a polygon's vertices, a tile object's gid, a text
    object's styling.
    """
    return {
        field.name: _plain(getattr(shape, field.name))
        for field in dataclasses.fields(shape)
    }


def _object_facts(obj: Any) -> dict[str, Any]:
    return {
        # Tiled's persistent id, and *not* the ``uid`` beside it: a uid is
        # minted per process and is one of the three things this module
        # promises to be blind to, while an id is an ordinary document field
        # that survives every save and is what an ``object``-typed custom
        # property references. Two maps whose objects wear different ids are
        # not the same map, because a property pointing at ``7`` finds a
        # different thing in each.
        "id": int(obj.id),
        "name": obj.name,
        "kind": obj.kind,
        "x": _num(obj.x),
        "y": _num(obj.y),
        "w": _num(obj.w),
        "h": _num(obj.h),
        "rotation": _num(obj.rotation),
        "opacity": _num(obj.opacity),
        # Named ``geometry`` and not ``shape`` because a tile *layer*'s facts
        # already spend that word on its array's dimensions.
        "geometry": _geometry_facts(obj.shape),
        "obj_class": obj.obj_class,
        "visible": bool(obj.visible),
        "properties": _props(obj.properties),
    }


def _layer_facts(layer: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        # See ``_object_facts`` -- the same field, the same argument.
        "id": int(layer.id),
        "name": layer.name,
        "class_name": str(layer.class_name),
        "blend_mode": str(layer.blend_mode),
        "visible": bool(layer.visible),
        "opacity": _num(layer.opacity),
        "locked": bool(layer.locked),
        "tint": [int(part) for part in layer.tint],
        "offset": [_num(layer.offset_x), _num(layer.offset_y)],
        "parallax": [_num(layer.parallax_x), _num(layer.parallax_y)],
        "properties": _props(layer.properties),
    }
    # Which kind, asked of what the layer *has* rather than by importing the
    # classes -- this module is a vocabulary, not a consumer of the model, and
    # the order matters: a group has children, an image has pixels, an object
    # layer has objects, and a tile layer is what is left.
    if hasattr(layer, "children"):
        facts["type"] = "group"
        # Nested, because the shape of the tree is a fact about the document:
        # two maps with the same layers under different parents are not the
        # same map, and a flattened list would say they were.
        facts["layers"] = [_layer_facts(child) for child in layer.children]
    elif hasattr(layer, "pixels"):
        facts["type"] = "image"
        facts["source"] = str(layer.source)
        facts["repeat"] = [bool(layer.repeat_x), bool(layer.repeat_y)]
        facts["shape"] = list(np.asarray(layer.pixels).shape)
        facts["image"] = _digest(layer.pixels)
    elif hasattr(layer, "objects"):
        facts["type"] = "object"
        facts["draworder"] = str(layer.draworder)
        facts["color"] = layer.color
        facts["objects"] = [_object_facts(obj) for obj in layer.objects]
    else:
        facts["type"] = "tile"
        facts["shape"] = [int(layer.height), int(layer.width)]
        facts["cells"] = _digest(layer.data)
    return facts


def doc_facts(doc: Any) -> dict[str, Any]:
    """Everything a :class:`MapDoc` is, as a JSON-shaped dict.

    Two documents compare equal exactly when they are the same map. Layer and
    tileset *order* is significant and preserved -- paint order and firstgid
    allocation are both facts about the document -- while property order is
    not, and is sorted away.

    ``next_layer_id``/``next_object_id`` are a fourth deliberate absence,
    alongside the three in the module docstring, and the reason is the one
    ``tilemap.py`` states for the counters themselves: they are monotone and
    never decremented, so a map that had a layer added and undone carries a
    higher counter than the identical map that never did. That is a fact about
    the *editing session*, not about the map -- the ids in use are the same
    either way -- and a comparator sensitive to it would call a document
    different from itself after a Ctrl+Z. The ids the counters issued are in,
    per layer and per object; the counters are not.
    """
    return {
        "projection": doc.projection,
        "infinite": bool(doc.infinite),
        # Only on an infinite map. A finite one's corner is (0, 0) by
        # definition, so including it there would compare a constant.
        "origin": [int(doc.origin_x), int(doc.origin_y)] if doc.infinite else None,
        "width": int(doc.width),
        "height": int(doc.height),
        "tile_w": int(doc.tile_w),
        "tile_h": int(doc.tile_h),
        "renderorder": doc.renderorder,
        "backgroundcolor": doc.backgroundcolor,
        "class_name": str(doc.class_name),
        "parallax_origin": [_num(value) for value in doc.parallax_origin],
        "skew": [int(doc.skew_x), int(doc.skew_y)],
        "stagger": [doc.stagger_axis, doc.stagger_index, int(doc.hex_side)],
        "properties": _props(doc.properties),
        "tilesets": [_tileset_facts(ref) for ref in doc.tilesets],
        "layers": [_layer_facts(layer) for layer in doc.layers],
    }
