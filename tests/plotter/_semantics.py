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


def _prop(value: Any) -> Any:
    """One custom property. Handles both spellings the codebase carries.

    ``properties`` is typed ``dict[str, Any]`` and holds ``tsx.Prop`` in
    practice; a plain value is accepted rather than refused so this never
    becomes the reason a test cannot express a document.
    """
    kind = getattr(value, "type", None)
    if kind is None:
        return ["untyped", value if not isinstance(value, float) else _num(value)]
    raw = getattr(value, "value")
    return [kind, _num(raw) if kind == "float" else raw]


def _props(properties: dict[str, Any]) -> dict[str, Any]:
    return {name: _prop(properties[name]) for name in sorted(properties)}


def _terrain_facts(spec: Any) -> list[Any]:
    return [spec.name, list(spec.fill), list(spec.outline)]


def _tileset_facts(ref: Any) -> dict[str, Any]:
    tileset = ref.tileset
    return {
        "firstgid": int(ref.firstgid),
        "source": ref.source,
        "name": tileset.name,
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


def _object_facts(obj: Any) -> dict[str, Any]:
    return {
        "name": obj.name,
        "kind": obj.kind,
        "x": _num(obj.x),
        "y": _num(obj.y),
        "w": _num(obj.w),
        "h": _num(obj.h),
        "obj_class": obj.obj_class,
        "visible": bool(obj.visible),
        "properties": _props(obj.properties),
    }


def _layer_facts(layer: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "name": layer.name,
        "visible": bool(layer.visible),
        "opacity": _num(layer.opacity),
        "locked": bool(layer.locked),
        "properties": _props(layer.properties),
    }
    objects = getattr(layer, "objects", None)
    if objects is None:
        facts["type"] = "tile"
        facts["shape"] = [int(layer.height), int(layer.width)]
        facts["cells"] = _digest(layer.data)
    else:
        facts["type"] = "object"
        facts["objects"] = [_object_facts(obj) for obj in objects]
    return facts


def doc_facts(doc: Any) -> dict[str, Any]:
    """Everything a :class:`MapDoc` is, as a JSON-shaped dict.

    Two documents compare equal exactly when they are the same map. Layer and
    tileset *order* is significant and preserved -- paint order and firstgid
    allocation are both facts about the document -- while property order is
    not, and is sorted away.
    """
    return {
        "projection": doc.projection,
        "width": int(doc.width),
        "height": int(doc.height),
        "tile_w": int(doc.tile_w),
        "tile_h": int(doc.tile_h),
        "renderorder": doc.renderorder,
        "backgroundcolor": doc.backgroundcolor,
        "properties": _props(doc.properties),
        "tilesets": [_tileset_facts(ref) for ref in doc.tilesets],
        "layers": [_layer_facts(layer) for layer in doc.layers],
    }
