"""``.tmx`` and ``.tmj`` -- Tiled's map, read and written.

Import and export both go through here, and the governing rule is that **a
feature this editor does not model is refused by name rather than dropped**.
Tiled's format is much larger than an orthogonal stamp-and-fill editor: hex and
isometric grids, infinite chunked layers, group and image layers, five object
shapes this cannot draw, Wang sets, per-tile animation. Loading such a file and
quietly keeping the half we understand would be fine right up to the moment the
user saved, at which point the other half is gone. So the reader raises
:class:`~.tsx.TiledUnsupported`, whose message names the feature and says what to
do about it, and ``tests/plotter/test_tmx_refusals.py`` has one case per entry.

**Gid payloads are reinterpreted, never re-derived.** A base64 layer is decoded
straight into little-endian ``uint32`` and reshaped, so the three transform
flags in the top bits survive a round trip bit-exactly without anything here
having to know what they mean. Every nonzero cell is then checked against the
declared ``firstgid`` ranges: a gid no tileset answers for is a corrupt or
hand-edited file, and accepting it produces a map with invisible tiles that
cannot be repainted because nothing knows what they were.

**Reading accepts five encodings; writing emits one.** CSV, base64 raw,
base64+zlib, base64+gzip and Tiled's older ``<tile>``-element form all read;
everything written from here is CSV (or, for ``.tmj``, a plain JSON array).
Round-tripping the compression a file happened to arrive in would make the
output depend on the input in a way nothing needs, and CSV is the form a diff
can show. ``zstd`` is refused rather than supported, which is what keeps this
package's dependency set to numpy and the standard library.

**Loading is split so this module stays pure.** Resolving a relative image or
``.tsx`` path means touching a filesystem, so the two loaders are callbacks the
host supplies: ``tsx_loader(source) -> Tileset`` and
``image_loader(source) -> RGBA array``.
"""

from __future__ import annotations

import base64
import gzip
import io
import itertools
import json
import re
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Callable
from typing import Any

import numpy as np

from . import gid as gidlib
from .tilemap import MapDoc, MapObject, ObjectLayer, TileLayer, new_uid
from .tileset import Tileset, TilesetRef
from .tsx import (
    TILED_VERSION,
    TSX_VERSION,
    Prop,
    TiledUnsupported,
    check_tileset_features,
    read_properties,
    to_bytes,
    tsx_bytes,
    write_properties,
)

# Re-exported deliberately: ``TiledUnsupported`` is defined in :mod:`.tsx`
# because the dependency runs that way (a map reads external tilesets), but it
# is *about* Tiled interop as a whole and every caller meets it here first.
__all__ = [
    "TiledUnsupported",
    "read_tmj",
    "read_tmx",
    "tmj_export",
    "tmx_export",
]

# Imported from :mod:`.tsx` rather than restated, which is the same argument
# this package makes everywhere else: neither number is a fact about maps or
# about tilesets, both are facts about the Tiled release this build targets, and
# two copies would let a bump move one of them and write an export whose ``.tmx``
# and ``.tsx`` claim different versions of one editor. ``MAP_VERSION`` keeps its
# own name because that is what the attribute is called on a ``<map>``; it is
# ``TSX_VERSION`` by construction, the format version rather than the file type's.
MAP_VERSION = TSX_VERSION

TilesetLoader = Callable[[str], Tileset]
ImageLoader = Callable[[str], Any]

# What a compression attribute may say. ``zstd`` is a real Tiled option and is
# deliberately absent: supporting it means a third-party wheel in a package
# whose whole claim is numpy and the standard library.
_COMPRESSIONS = ("", "zlib", "gzip")
_ENCODINGS = ("", "csv", "base64")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# --- shared refusals ----------------------------------------------------------


def _refuse_object_shape(has: Callable[[str], bool], where: str) -> None:
    """The four shapes and two references an object may be that this cannot be.

    One function for both formats, because the list is the *model's* limit and
    not the syntax's -- if XML and JSON disagreed about it, one of them would be
    accepting a file the editor cannot draw.
    """
    for name, label in (
        ("ellipse", "ellipse objects"),
        ("polygon", "polygon objects"),
        ("polyline", "polyline objects"),
        ("text", "text objects"),
    ):
        if has(name):
            raise TiledUnsupported(label, where)


# --- XML reading --------------------------------------------------------------


def _root(data: bytes, expect: str) -> ET.Element:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"this is not a readable Tiled {expect} file: {exc}") from exc
    if root.tag != expect:
        raise ValueError(f"expected a <{expect}> document, found <{root.tag}>")
    return root


def _check_map(root: ET.Element) -> None:
    orientation = root.get("orientation", "orthogonal")
    if orientation != "orthogonal":
        raise TiledUnsupported(f"a {orientation} map", "Plotter draws orthogonal maps only")
    if root.get("infinite", "0") not in ("0", "false"):
        raise TiledUnsupported(
            "an infinite map", "save it with a fixed size in Tiled's map properties"
        )
    if root.find("group") is not None:
        raise TiledUnsupported("group layers", "flatten them in Tiled first")
    if root.find("imagelayer") is not None:
        raise TiledUnsupported("image layers")


def _decode_payload(
    text: str, encoding: str, compression: str, width: int, height: int
) -> np.ndarray:
    """One layer's gids from its encoded text, as a ``(h, w)`` uint32 array."""
    if encoding not in _ENCODINGS:
        raise TiledUnsupported(f"layer data encoded as {encoding!r}")
    if compression not in _COMPRESSIONS:
        detail = (
            "zstd needs a third-party decoder; re-save the map with zlib, gzip or CSV"
            if compression == "zstd"
            else ""
        )
        raise TiledUnsupported(f"{compression}-compressed layer data", detail)

    if encoding == "csv":
        values = [int(piece) for piece in text.replace("\n", "").split(",") if piece.strip()]
        flat = np.array(values, dtype=gidlib.DTYPE)
    else:
        raw = base64.b64decode("".join(text.split()))
        if compression == "zlib":
            raw = zlib.decompress(raw)
        elif compression == "gzip":
            raw = gzip.decompress(raw)
        # Little-endian explicitly: the format says so, and a big-endian host
        # reading native order would silently byte-swap every flag.
        flat = np.frombuffer(raw, dtype="<u4").astype(gidlib.DTYPE)

    if flat.size != width * height:
        raise ValueError(
            f"a layer declares {width}x{height} cells and carries {flat.size}"
        )
    return np.ascontiguousarray(flat.reshape(height, width))


def _xml_tile_elements(node: ET.Element, width: int, height: int) -> np.ndarray:
    """Tiled's oldest layer form: one ``<tile gid=...>`` per cell.

    Still selectable in Tiled as the "XML" tile-layer format, so refusing it
    would refuse a file a user legitimately exported. Written by nothing here.
    """
    values = [int(tile.get("gid", 0) or 0) for tile in node.findall("tile")]
    flat = np.array(values, dtype=gidlib.DTYPE)
    if flat.size != width * height:
        raise ValueError(f"a layer declares {width}x{height} cells and carries {flat.size}")
    return np.ascontiguousarray(flat.reshape(height, width))


def _check_offsets(node: ET.Element, name: str) -> None:
    for attr in ("offsetx", "offsety"):
        if float(node.get(attr, 0) or 0) != 0.0:
            raise TiledUnsupported("layer pixel offsets", f"layer {name!r}")


def _read_tmx_tilesets(
    root: ET.Element, *, image_loader: ImageLoader, tsx_loader: TilesetLoader
) -> list[TilesetRef]:
    refs: list[TilesetRef] = []
    for node in root.findall("tileset"):
        firstgid = int(node.get("firstgid", 1) or 1)
        source = node.get("source")
        if source:
            refs.append(
                TilesetRef(firstgid=firstgid, tileset=tsx_loader(source), source=source)
            )
            continue
        # Embedded: the same element a .tsx holds, minus the file around it.
        check_tileset_features(node)
        image = node.find("image")
        path = (image.get("source") or "").strip() if image is not None else ""
        if not path:
            raise TiledUnsupported(
                "an embedded tileset image", "Plotter needs an <image source=...> path"
            )
        refs.append(
            TilesetRef(
                firstgid=firstgid,
                tileset=Tileset(
                    name=node.get("name") or "tileset",
                    pixels=image_loader(path),
                    tile_w=int(node.get("tilewidth", 0) or 0),
                    tile_h=int(node.get("tileheight", 0) or 0),
                    spacing=int(node.get("spacing", 0) or 0),
                    margin=int(node.get("margin", 0) or 0),
                    properties=read_properties(node),
                ),
            )
        )
    return refs


def _read_tmx_object(node: ET.Element) -> MapObject:
    name = node.get("name", "")
    where = f"object {node.get('id', '?')}"
    if node.get("template"):
        raise TiledUnsupported("object templates", where)
    if node.get("gid"):
        raise TiledUnsupported("tile objects", where)
    if float(node.get("rotation", 0) or 0) != 0.0:
        raise TiledUnsupported("rotated objects", where)
    _refuse_object_shape(lambda tag: node.find(tag) is not None, where)
    kind = "point" if node.find("point") is not None else "rect"
    return MapObject(
        uid=new_uid(),
        name=name,
        kind=kind,
        x=float(node.get("x", 0) or 0),
        y=float(node.get("y", 0) or 0),
        w=float(node.get("width", 0) or 0),
        h=float(node.get("height", 0) or 0),
        obj_class=node.get("class") or node.get("type") or "",
        visible=node.get("visible", "1") not in ("0", "false"),
        properties=read_properties(node),
    )


def read_tmx(
    data: bytes, *, image_loader: ImageLoader, tsx_loader: TilesetLoader
) -> MapDoc:
    """A ``.tmx``'s bytes as a :class:`~.tilemap.MapDoc`.

    Built by *construction* rather than through the document's own mutators,
    which would push one undo step per layer and open every file already dirty.
    """
    root = _root(data, "map")
    _check_map(root)

    doc = MapDoc(
        width=int(root.get("width", 1) or 1),
        height=int(root.get("height", 1) or 1),
        tile_w=int(root.get("tilewidth", 1) or 1),
        tile_h=int(root.get("tileheight", 1) or 1),
    )
    doc.renderorder = root.get("renderorder", "right-down")
    doc.backgroundcolor = root.get("backgroundcolor")
    doc.properties = read_properties(root)
    doc.tilesets = _read_tmx_tilesets(
        root, image_loader=image_loader, tsx_loader=tsx_loader
    )

    # Document order is stacking order, bottom first -- which is why this walks
    # the root's children rather than ``findall`` per tag, and is the one place
    # the two layer kinds have to be read by one loop.
    for node in root:
        if node.tag == "layer":
            name = node.get("name", "")
            _check_offsets(node, name)
            payload = node.find("data")
            if payload is None:
                raise ValueError(f"tile layer {name!r} carries no <data>")
            encoding = payload.get("encoding", "")
            if not encoding and payload.find("tile") is not None:
                cells = _xml_tile_elements(payload, doc.width, doc.height)
            else:
                cells = _decode_payload(
                    payload.text or "",
                    encoding,
                    payload.get("compression", ""),
                    doc.width,
                    doc.height,
                )
            doc.layers.append(
                TileLayer(
                    uid=new_uid(),
                    name=name,
                    data=cells,
                    visible=node.get("visible", "1") not in ("0", "false"),
                    opacity=float(node.get("opacity", 1) or 1),
                    properties=read_properties(node),
                )
            )
        elif node.tag == "objectgroup":
            name = node.get("name", "")
            _check_offsets(node, name)
            doc.layers.append(
                ObjectLayer(
                    uid=new_uid(),
                    name=name,
                    objects=[_read_tmx_object(o) for o in node.findall("object")],
                    visible=node.get("visible", "1") not in ("0", "false"),
                    opacity=float(node.get("opacity", 1) or 1),
                    properties=read_properties(node),
                )
            )

    _finish(doc)
    return doc


# --- JSON reading -------------------------------------------------------------


def _json_properties(entries: Any) -> dict[str, Prop]:
    out: dict[str, Prop] = {}
    for entry in entries or []:
        name = str(entry.get("name", ""))
        if not name:
            continue
        kind = str(entry.get("type", "string"))
        # ``Prop`` refuses an unknown type by name, so the JSON side needs no
        # list of its own -- one place decides what a property may be.
        out[name] = Prop(type=kind, value=entry.get("value"))
    return out


def _json_object(entry: dict[str, Any]) -> MapObject:
    where = f"object {entry.get('id', '?')}"
    if entry.get("template"):
        raise TiledUnsupported("object templates", where)
    if entry.get("gid"):
        raise TiledUnsupported("tile objects", where)
    if float(entry.get("rotation", 0) or 0) != 0.0:
        raise TiledUnsupported("rotated objects", where)
    _refuse_object_shape(lambda tag: bool(entry.get(tag)), where)
    return MapObject(
        uid=new_uid(),
        name=str(entry.get("name", "")),
        kind="point" if entry.get("point") else "rect",
        x=float(entry.get("x", 0) or 0),
        y=float(entry.get("y", 0) or 0),
        w=float(entry.get("width", 0) or 0),
        h=float(entry.get("height", 0) or 0),
        obj_class=str(entry.get("class") or entry.get("type") or ""),
        visible=bool(entry.get("visible", True)),
        properties=_json_properties(entry.get("properties")),
    )


def read_tmj(
    data: bytes, *, image_loader: ImageLoader, tsx_loader: TilesetLoader
) -> MapDoc:
    """The JSON spelling of the same map. Every refusal above applies here."""
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"this is not a readable Tiled JSON map: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("a Tiled JSON map is an object")

    orientation = str(payload.get("orientation", "orthogonal"))
    if orientation != "orthogonal":
        raise TiledUnsupported(f"a {orientation} map", "Plotter draws orthogonal maps only")
    if payload.get("infinite"):
        raise TiledUnsupported(
            "an infinite map", "save it with a fixed size in Tiled's map properties"
        )

    doc = MapDoc(
        width=int(payload.get("width", 1) or 1),
        height=int(payload.get("height", 1) or 1),
        tile_w=int(payload.get("tilewidth", 1) or 1),
        tile_h=int(payload.get("tileheight", 1) or 1),
    )
    doc.renderorder = str(payload.get("renderorder", "right-down"))
    doc.backgroundcolor = payload.get("backgroundcolor")
    doc.properties = _json_properties(payload.get("properties"))

    for entry in payload.get("tilesets", []):
        firstgid = int(entry.get("firstgid", 1) or 1)
        source = entry.get("source")
        if source:
            if str(source).lower().endswith(".tsj"):
                raise TiledUnsupported(
                    "an external .tsj tileset", "re-save the tileset as .tsx in Tiled"
                )
            doc.tilesets.append(
                TilesetRef(firstgid=firstgid, tileset=tsx_loader(source), source=source)
            )
            continue
        if entry.get("tiles") or entry.get("grid"):
            raise TiledUnsupported("an image-collection tileset", str(entry.get("name", "")))
        if entry.get("wangsets"):
            raise TiledUnsupported("Wang sets / terrain brushes")
        image = str(entry.get("image", ""))
        if not image:
            raise TiledUnsupported("an embedded tileset image", "Plotter needs an image path")
        doc.tilesets.append(
            TilesetRef(
                firstgid=firstgid,
                tileset=Tileset(
                    name=str(entry.get("name", "tileset")),
                    pixels=image_loader(image),
                    tile_w=int(entry.get("tilewidth", 0) or 0),
                    tile_h=int(entry.get("tileheight", 0) or 0),
                    spacing=int(entry.get("spacing", 0) or 0),
                    margin=int(entry.get("margin", 0) or 0),
                    properties=_json_properties(entry.get("properties")),
                ),
            )
        )

    for entry in payload.get("layers", []):
        kind = str(entry.get("type", ""))
        name = str(entry.get("name", ""))
        if kind == "group":
            raise TiledUnsupported("group layers", "flatten them in Tiled first")
        if kind == "imagelayer":
            raise TiledUnsupported("image layers")
        if float(entry.get("offsetx", 0) or 0) or float(entry.get("offsety", 0) or 0):
            raise TiledUnsupported("layer pixel offsets", f"layer {name!r}")
        if kind == "tilelayer":
            if entry.get("chunks"):
                raise TiledUnsupported("chunked (infinite) layer data", f"layer {name!r}")
            raw = entry.get("data")
            if isinstance(raw, str):
                cells = _decode_payload(
                    raw,
                    "base64",
                    str(entry.get("compression", "") or ""),
                    doc.width,
                    doc.height,
                )
            else:
                flat = np.array(list(raw or []), dtype=gidlib.DTYPE)
                if flat.size != doc.width * doc.height:
                    raise ValueError(
                        f"a layer declares {doc.width}x{doc.height} cells "
                        f"and carries {flat.size}"
                    )
                cells = np.ascontiguousarray(flat.reshape(doc.height, doc.width))
            doc.layers.append(
                TileLayer(
                    uid=new_uid(),
                    name=name,
                    data=cells,
                    visible=bool(entry.get("visible", True)),
                    opacity=float(entry.get("opacity", 1) or 1),
                    properties=_json_properties(entry.get("properties")),
                )
            )
        elif kind == "objectgroup":
            doc.layers.append(
                ObjectLayer(
                    uid=new_uid(),
                    name=name,
                    objects=[_json_object(o) for o in entry.get("objects", [])],
                    visible=bool(entry.get("visible", True)),
                    opacity=float(entry.get("opacity", 1) or 1),
                    properties=_json_properties(entry.get("properties")),
                )
            )
        elif kind:
            raise TiledUnsupported(f"{kind} layers", f"layer {name!r}")

    _finish(doc)
    return doc


def _finish(doc: MapDoc) -> None:
    """Validate every gid, then hand back a document that reads clean."""
    for layer in doc.tile_layers():
        if layer.data.shape != (doc.height, doc.width):
            raise ValueError(
                f"layer {layer.name!r} is {layer.data.shape[1]}x{layer.data.shape[0]}, "
                f"but the map is {doc.width}x{doc.height}"
            )
        ids = np.unique(gidlib.tile_ids(layer.data))
        for tile_id in ids.tolist():
            if tile_id and doc.ref_for(tile_id) is None:
                raise ValueError(
                    f"layer {layer.name!r} uses tile {tile_id}, which none of this "
                    "map's tilesets accounts for"
                )
    doc.active_layer = doc.layers[0].uid if doc.layers else None
    doc.history.clear()
    doc.saved_head = doc.history.head


# --- writing ------------------------------------------------------------------


def _stem(index: int, name: str) -> str:
    """A filesystem-safe, collision-free name for one tileset's two files.

    Indexed as well as sanitised: two tilesets may legitimately share a name,
    and two files that then share one would silently be the same file.
    """
    safe = _SAFE.sub("-", name).strip("-") or "tileset"
    return f"{index:02d}-{safe}"


def _png_bytes(pixels: np.ndarray) -> bytes:
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels), "RGBA").save(out, "PNG")
    return out.getvalue()


def _csv(data: np.ndarray) -> str:
    """Tiled's own CSV spelling: one row per line, commas throughout."""
    rows = [",".join(str(int(value)) for value in row) for row in data]
    return "\n" + ",\n".join(rows) + "\n"


def _tileset_files(doc: MapDoc) -> tuple[dict[str, bytes], list[str]]:
    """The external ``.tsx``/``.png`` pair per tileset, and their map-relative
    paths. Shared by both exporters so a ``.tmx`` and a ``.tmj`` reference the
    same files by the same names."""
    files: dict[str, bytes] = {}
    paths: list[str] = []
    for index, ref in enumerate(doc.tilesets):
        stem = _stem(index, ref.tileset.name)
        tsx_path = f"tilesets/{stem}.tsx"
        files[f"tilesets/{stem}.png"] = _png_bytes(ref.tileset.pixels)
        # The image name inside the .tsx is relative to the .tsx, which sits
        # beside it -- not to the map.
        files[tsx_path] = tsx_bytes(ref.tileset, image_name=f"{stem}.png")
        paths.append(tsx_path)
    return files, paths


def tmx_export(doc: MapDoc) -> dict[str, bytes]:
    """The whole map as a mapping of relative path to bytes.

    A mapping rather than one blob because TMX has no portable way to embed an
    image: a tileset is a ``.tsx`` plus a ``.png`` beside the map. The caller
    writes them; deciding *where* is not this module's business.
    """
    files, tsx_paths = _tileset_files(doc)
    root = ET.Element(
        "map",
        {
            "version": MAP_VERSION,
            "tiledversion": TILED_VERSION,
            "orientation": "orthogonal",
            "renderorder": doc.renderorder,
            "width": str(doc.width),
            "height": str(doc.height),
            "tilewidth": str(doc.tile_w),
            "tileheight": str(doc.tile_h),
            "infinite": "0",
        },
    )
    if doc.backgroundcolor:
        root.set("backgroundcolor", str(doc.backgroundcolor))
    object_count = sum(
        len(layer.objects) for layer in doc.layers if isinstance(layer, ObjectLayer)
    )
    root.set("nextlayerid", str(len(doc.layers) + 1))
    root.set("nextobjectid", str(object_count + 1))
    write_properties(root, doc.properties)

    for ref, path in zip(doc.tilesets, tsx_paths, strict=True):
        ET.SubElement(root, "tileset", {"firstgid": str(ref.firstgid), "source": path})

    # Ids are minted here rather than carried on the document: they are a
    # property of the *file*, and a uid minted per process would leak this
    # session's history into every exported map.
    layer_ids = itertools.count(1)
    object_ids = itertools.count(1)
    for layer in doc.layers:
        common = {"id": str(next(layer_ids)), "name": layer.name}
        if isinstance(layer, TileLayer):
            node = ET.SubElement(
                root, "layer", {**common, "width": str(doc.width), "height": str(doc.height)}
            )
        else:
            node = ET.SubElement(root, "objectgroup", common)
        if layer.opacity != 1.0:
            node.set("opacity", repr(float(layer.opacity)))
        if not layer.visible:
            node.set("visible", "0")
        write_properties(node, layer.properties)
        if isinstance(layer, TileLayer):
            data = ET.SubElement(node, "data", {"encoding": "csv"})
            data.text = _csv(layer.data)
        else:
            for obj in layer.objects:
                attrs = {"id": str(next(object_ids))}
                if obj.name:
                    attrs["name"] = obj.name
                if obj.obj_class:
                    attrs["class"] = obj.obj_class
                attrs["x"] = repr(float(obj.x))
                attrs["y"] = repr(float(obj.y))
                if obj.kind == "rect":
                    attrs["width"] = repr(float(obj.w))
                    attrs["height"] = repr(float(obj.h))
                if not obj.visible:
                    attrs["visible"] = "0"
                entry = ET.SubElement(node, "object", attrs)
                write_properties(entry, obj.properties)
                if obj.kind == "point":
                    ET.SubElement(entry, "point")

    files["map.tmx"] = to_bytes(root)
    return files


def _json_props(props: dict[str, Prop]) -> list[dict[str, Any]]:
    return [
        {"name": name, "type": props[name].type, "value": props[name].value}
        for name in sorted(props)
    ]


def tmj_export(doc: MapDoc) -> dict[str, bytes]:
    """The JSON spelling. Same external tilesets, same names, same bytes."""
    files, tsx_paths = _tileset_files(doc)
    layer_ids = itertools.count(1)
    object_ids = itertools.count(1)

    layers: list[dict[str, Any]] = []
    for layer in doc.layers:
        entry: dict[str, Any] = {
            "id": next(layer_ids),
            "name": layer.name,
            "opacity": float(layer.opacity),
            "visible": bool(layer.visible),
            "x": 0,
            "y": 0,
        }
        if layer.properties:
            entry["properties"] = _json_props(layer.properties)
        if isinstance(layer, TileLayer):
            entry.update(
                {
                    "type": "tilelayer",
                    "width": doc.width,
                    "height": doc.height,
                    "data": [int(v) for v in layer.data.reshape(-1)],
                }
            )
        else:
            objects = []
            for obj in layer.objects:
                record: dict[str, Any] = {
                    "id": next(object_ids),
                    "name": obj.name,
                    "type": obj.obj_class,
                    "x": float(obj.x),
                    "y": float(obj.y),
                    "rotation": 0,
                    "visible": bool(obj.visible),
                }
                if obj.kind == "point":
                    record["point"] = True
                    record["width"] = 0
                    record["height"] = 0
                else:
                    record["width"] = float(obj.w)
                    record["height"] = float(obj.h)
                if obj.properties:
                    record["properties"] = _json_props(obj.properties)
                objects.append(record)
            entry.update({"type": "objectgroup", "draworder": "topdown", "objects": objects})
        layers.append(entry)

    payload: dict[str, Any] = {
        "type": "map",
        "version": MAP_VERSION,
        "tiledversion": TILED_VERSION,
        "orientation": "orthogonal",
        "renderorder": doc.renderorder,
        "infinite": False,
        "width": doc.width,
        "height": doc.height,
        "tilewidth": doc.tile_w,
        "tileheight": doc.tile_h,
        "nextlayerid": len(doc.layers) + 1,
        "nextobjectid": sum(
            len(layer.objects) for layer in doc.layers if isinstance(layer, ObjectLayer)
        )
        + 1,
        "tilesets": [
            {"firstgid": ref.firstgid, "source": path}
            for ref, path in zip(doc.tilesets, tsx_paths, strict=True)
        ],
        "layers": layers,
    }
    if doc.backgroundcolor:
        payload["backgroundcolor"] = str(doc.backgroundcolor)
    if doc.properties:
        payload["properties"] = _json_props(doc.properties)

    files["map.tmj"] = (json.dumps(payload, indent=2) + "\n").encode()
    return files
