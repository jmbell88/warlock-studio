"""``.tmx`` and ``.tmj`` -- Tiled's map, read and written.

Import and export both go through here, and the governing rule is that **a
feature this editor does not model is refused by name rather than dropped**.
Tiled's format is much larger than an orthogonal stamp-and-fill editor: hex and
isometric grids, infinite chunked layers, group and image layers, five object
shapes this cannot draw, Wang sets, per-tile animation. Loading such a file and
quietly keeping the half we understand would be fine right up to the moment the
user saved, at which point the other half is gone. So the reader raises
:class:`~.props.TiledUnsupported`, whose message names the feature and says what to
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

from . import blob, project
from . import gid as gidlib
from .pngio import png_bytes
from .props import (
    TiledUnsupported,
    read_json_properties,
    read_properties,
    write_json_properties,
    write_properties,
)
from .tilemap import MapDoc, MapObject, ObjectLayer, TileLayer, new_uid
from .tileset import TerrainSpec, Tileset, TilesetRef
from .tsx import (
    TILED_VERSION,
    TSX_VERSION,
    check_tileset_features,
    read_wangsets,
    read_wangsets_json,
    to_bytes,
    tsx_bytes,
    xml_root,
)

# Re-exported deliberately: ``TiledUnsupported`` is defined in :mod:`.props`
# because the property model is the package's leaf, but it is *about* Tiled
# interop as a whole and every caller meets it here first.
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

# Tiled's hexagonal 120-degree rotation flag. See the note in :mod:`.gid` for
# why the constant lives here and not beside the other three.
_HEX_ROTATE = gidlib.DTYPE(0x10000000)


# --- shared refusals ----------------------------------------------------------


def _refuse_infinite(infinite: bool) -> None:
    """One sentence, both formats -- see :func:`_refuse_object_shape`."""
    if infinite:
        raise TiledUnsupported(
            "an infinite map", "save it with a fixed size in Tiled's map properties"
        )


def _refuse_container_layers(has: Callable[[str], bool]) -> None:
    """The two layer kinds that hold other layers or a picture instead of tiles.

    ``has`` is the accessor because XML asks the *document* (a ``<group>``
    anywhere under the root) and JSON asks each *layer* (its ``type``): the
    predicate differs, the model's limit does not.
    """
    if has("group"):
        raise TiledUnsupported("group layers", "flatten them in Tiled first")
    if has("imagelayer"):
        raise TiledUnsupported("image layers")


def _refuse_layer_offsets(offset: Callable[[str], float], name: str) -> None:
    if offset("offsetx") or offset("offsety"):
        raise TiledUnsupported("layer pixel offsets", f"layer {name!r}")


def _refuse_wangsets(recognised: tuple[TerrainSpec, ...] | None) -> tuple[TerrainSpec, ...]:
    """**Recognise or refuse**, and the asymmetry is the point.

    Tiled's Wang model is strictly larger than this one -- corner-only and
    edge-only sets, up to 255 colours, tile assignments that need not form a
    blob at all -- so adopting a foreign one would be the silent half-read the
    whole reader exists to prevent. What is recognised is exactly what
    ``tsx.write_wangsets`` emits, which keeps the reader and the writer
    symmetric: every file this writes, this reads.

    Shared by the XML and JSON tileset readers because the *decision* is one
    decision. The JSON side used to refuse every wangset outright, so a ``.tmj``
    carrying a set this build had itself written was turned away by a sentence
    that did not say what about it was wrong.
    """
    if recognised is None:
        raise TiledUnsupported(
            "Wang sets / terrain brushes",
            f"Plotter models one blob set: {blob.TILE_COUNT} tiles per terrain colour, "
            "in mask order",
        )
    return recognised


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


# The sentence each shape the writers cannot spell is refused under -- the
# *same* sentence the readers refuse it with, because it is one limit with two
# doors and not two features that happen to share a name. Keyed by the shape
# kind ``MapObject.kind`` reports; the two absent keys are the two a writer can
# spell.
_UNWRITABLE_SHAPES = {
    "ellipse": "ellipse objects",
    "polygon": "polygon objects",
    "polyline": "polyline objects",
    "text": "text objects",
    "tile": "tile objects",
}


def _refuse_unwritable_objects(doc: MapDoc) -> None:
    """What the *document* can hold and neither writer can yet spell.

    The mirror of the reader's object refusals, at the other door and for the
    same reason. The model gained rotation, a draw order and five more shapes
    before the exporters gained any way to emit them, so without this an export
    would drop a rotation, flatten an index-ordered layer to ``topdown`` --
    which changes which object is drawn on top, not merely an attribute -- and
    write an ellipse as a rectangle with no size. A silent half-*write* is
    worse than the half-read this ledger already forbids: the user still has
    their document, and the file they just handed to an engine is quietly wrong
    with nothing anywhere saying so.

    Both writers, one function, exactly as ``_refuse_object_shape`` serves both
    readers -- if the two lists drifted, one format would be writing a file the
    other could not read back. Both doors flip together in M3, when the writers
    learn to emit these and the reader stops refusing them.
    """
    for layer in doc.layers:
        if not isinstance(layer, ObjectLayer):
            continue
        if layer.draworder != "topdown":
            raise TiledUnsupported(
                "an index-ordered object layer", f"layer {layer.name!r}", exporting=True
            )
        for obj in layer.objects:
            where = f"object {obj.name or obj.uid}"
            if obj.rotation:
                raise TiledUnsupported("rotated objects", where, exporting=True)
            label = _UNWRITABLE_SHAPES.get(obj.kind)
            if label is not None:
                raise TiledUnsupported(label, where, exporting=True)


# --- XML reading --------------------------------------------------------------


def _check_orientation(orientation: str) -> str:
    """Accept what this draws, refuse the rest by name.

    Isometric left this list when the editor learned to draw one -- the refusal
    was never about the word, it was about not silently half-reading a map whose
    cells this could not place. Staggered and hexagonal stay, which is also what
    keeps ``gid``'s missing hex-rotation bit honest: a file that could set it
    never gets past here.
    """
    if orientation not in project.PROJECTIONS:
        raise TiledUnsupported(
            f"a {orientation} map",
            f"Plotter draws {' and '.join(project.PROJECTIONS)} maps",
        )
    return orientation


def _check_map(root: ET.Element) -> None:
    _check_orientation(root.get("orientation", "orthogonal"))
    _refuse_infinite(root.get("infinite", "0") not in ("0", "false"))
    _refuse_container_layers(lambda tag: root.find(tag) is not None)


def _gid_array(values: Any, width: int, height: int) -> np.ndarray:
    """A layer's cells from a sequence of plain numbers, checked then cast.

    **``int64`` first, and the range check before the cast.** Building the array
    as ``uint32`` directly is what numpy does silently and wrongly here: a
    hand-edited ``-1`` arrives as 4294967295 and an entry past the id space
    wraps, so a corrupt file used to be refused two steps later under a sentence
    about a tile no tileset accounts for -- a true statement about the wrong
    problem, and one that names no way to fix the file.

    Three callers share it (CSV, the ``<tile>``-element form and the TMJ raw
    list) because the three are one question asked in three syntaxes; two copies
    of the range test is how one spelling comes to accept what the others refuse.
    """
    try:
        flat = np.fromiter(values, dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("a layer's data holds something that is not a number") from exc
    if flat.size != width * height:
        raise ValueError(f"a layer declares {width}x{height} cells and carries {flat.size}")
    if bool(((flat < 0) | (flat > 0xFFFFFFFF)).any()):
        raise ValueError("a layer holds a tile id outside the unsigned 32-bit range")
    return np.ascontiguousarray(flat.astype(gidlib.DTYPE).reshape(height, width))


def _decompress(raw: bytes, compression: str, expected: int) -> bytes:
    """Unpack one layer's payload, refusing anything past what its size declares.

    **Bounded, and the bound is the layer's own arithmetic.** ``zlib.decompress``
    on a hostile payload allocates whatever the stream says: a few hundred bytes
    of archive is enough to ask for gigabytes, and the read that discovers this
    is the one that has already exhausted memory. A layer of ``w * h`` cells is
    exactly ``w * h * 4`` bytes, so one more than that is already a file that
    does not describe the map around it -- which is why the tail is checked
    rather than the output simply truncated.
    """
    if compression == "zlib":
        engine = zlib.decompressobj()
        out = engine.decompress(raw, expected + 1)
        if len(out) > expected or engine.unconsumed_tail:
            raise ValueError(
                f"a layer's compressed data unpacks past the {expected} bytes its size declares"
            )
        return out
    if compression == "gzip":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
            out = fh.read(expected + 1)
        if len(out) > expected:
            raise ValueError(
                f"a layer's compressed data unpacks past the {expected} bytes its size declares"
            )
        return out
    return raw


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
        pieces = [piece for piece in text.replace("\n", "").split(",") if piece.strip()]
        try:
            values = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise ValueError("a layer's CSV data holds something that is not a number") from exc
        return _gid_array(values, width, height)

    raw = _decompress(
        base64.b64decode("".join(text.split())), compression, width * height * 4
    )
    # Little-endian explicitly: the format says so, and a big-endian host
    # reading native order would silently byte-swap every flag. Already unsigned,
    # so this path needs no range check -- four bytes *are* a uint32.
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
    try:
        values = [int(tile.get("gid", 0) or 0) for tile in node.findall("tile")]
    except ValueError as exc:
        raise ValueError("a layer's data holds something that is not a number") from exc
    return _gid_array(values, width, height)


def _check_offsets(node: ET.Element, name: str) -> None:
    _refuse_layer_offsets(lambda attr: float(node.get(attr, 0) or 0), name)


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
        wangsets = node.find("wangsets")
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
                    # ``check_tileset_features`` already refused an unrecognised
                    # set; an embedded one that *is* recognised is a terrain set
                    # and used to be accepted and then dropped, which left the
                    # tool greyed out on a map whose atlas plainly declares it.
                    terrains=() if wangsets is None else _refuse_wangsets(
                        read_wangsets(wangsets)
                    ),
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
        id=int(node.get("id", 0) or 0),
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


def _adopt_object_space(doc: MapDoc) -> None:
    """Move every object from Tiled's coordinate space into this map's.

    A no-op for an orthogonal map, where the two spaces are the same. Applied to
    the whole document at once, after the layers are on it, rather than inside
    the two object readers -- they are handed one element and know nothing about
    the map's size, and the conversion needs its height.
    """
    if not doc.isometric:
        return
    for layer in doc.layers:
        if isinstance(layer, ObjectLayer):
            for obj in layer.objects:
                obj.x, obj.y = project.object_to_pixels(doc._lattice(), obj.x, obj.y)


def _object_xy(doc: MapDoc, obj: MapObject) -> tuple[float, float]:
    """One object's position in Tiled's space, for the two writers."""
    return project.object_from_pixels(doc._lattice(), obj.x, obj.y)
def read_tmx(
    data: bytes, *, image_loader: ImageLoader, tsx_loader: TilesetLoader
) -> MapDoc:
    """A ``.tmx``'s bytes as a :class:`~.tilemap.MapDoc`.

    Built by *construction* rather than through the document's own mutators,
    which would push one undo step per layer and open every file already dirty.
    """
    root = xml_root(data, "map")
    _check_map(root)

    doc = MapDoc(
        width=int(root.get("width", 1) or 1),
        height=int(root.get("height", 1) or 1),
        tile_w=int(root.get("tilewidth", 1) or 1),
        tile_h=int(root.get("tileheight", 1) or 1),
        projection=root.get("orientation", "orthogonal"),
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
                    id=int(node.get("id", 0) or 0),
                    name=name,
                    data=cells,
                    visible=node.get("visible", "1") not in ("0", "false"),
                    opacity=float(node.get("opacity", 1) or 1),
                    # Absent means unlocked, which is what every file written
                    # before this existed says by saying nothing.
                    locked=node.get("locked", "0") not in ("0", "false"),
                    properties=read_properties(node),
                )
            )
        elif node.tag == "objectgroup":
            name = node.get("name", "")
            _check_offsets(node, name)
            doc.layers.append(
                ObjectLayer(
                    uid=new_uid(),
                    id=int(node.get("id", 0) or 0),
                    name=name,
                    objects=[_read_tmx_object(o) for o in node.findall("object")],
                    visible=node.get("visible", "1") not in ("0", "false"),
                    opacity=float(node.get("opacity", 1) or 1),
                    # Absent means unlocked, which is what every file written
                    # before this existed says by saying nothing.
                    locked=node.get("locked", "0") not in ("0", "false"),
                    properties=read_properties(node),
                )
            )

    _finish(
        doc,
        next_layer_id=_optional_int(root.get("nextlayerid")),
        next_object_id=_optional_int(root.get("nextobjectid")),
    )
    _adopt_object_space(doc)
    return doc


# --- JSON reading -------------------------------------------------------------


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
        id=int(entry.get("id", 0) or 0),
        name=str(entry.get("name", "")),
        kind="point" if entry.get("point") else "rect",
        x=float(entry.get("x", 0) or 0),
        y=float(entry.get("y", 0) or 0),
        w=float(entry.get("width", 0) or 0),
        h=float(entry.get("height", 0) or 0),
        obj_class=str(entry.get("class") or entry.get("type") or ""),
        visible=bool(entry.get("visible", True)),
        properties=read_json_properties(entry.get("properties")),
    )


def _read_tmj_tilesets(
    payload: dict[str, Any], *, image_loader: ImageLoader, tsx_loader: TilesetLoader
) -> list[TilesetRef]:
    """``_read_tmx_tilesets`` over the JSON spelling, refusal for refusal."""
    refs: list[TilesetRef] = []
    for entry in payload.get("tilesets", []):
        firstgid = int(entry.get("firstgid", 1) or 1)
        source = entry.get("source")
        if source:
            if str(source).lower().endswith(".tsj"):
                raise TiledUnsupported(
                    "an external .tsj tileset", "re-save the tileset as .tsx in Tiled"
                )
            refs.append(
                TilesetRef(firstgid=firstgid, tileset=tsx_loader(source), source=source)
            )
            continue
        # ``tiles`` or ``grid`` is an image collection: every tile its own file.
        # Kept ahead of the wangset question because it decides whether there is
        # one sliced atlas at all, which everything below assumes.
        if entry.get("tiles") or entry.get("grid"):
            raise TiledUnsupported("an image-collection tileset", str(entry.get("name", "")))
        wangsets = entry.get("wangsets")
        terrains: tuple[TerrainSpec, ...] = (
            () if not wangsets else _refuse_wangsets(read_wangsets_json(wangsets))
        )
        image = str(entry.get("image", ""))
        if not image:
            raise TiledUnsupported("an embedded tileset image", "Plotter needs an image path")
        refs.append(
            TilesetRef(
                firstgid=firstgid,
                tileset=Tileset(
                    name=str(entry.get("name", "tileset")),
                    pixels=image_loader(image),
                    tile_w=int(entry.get("tilewidth", 0) or 0),
                    tile_h=int(entry.get("tileheight", 0) or 0),
                    spacing=int(entry.get("spacing", 0) or 0),
                    margin=int(entry.get("margin", 0) or 0),
                    properties=read_json_properties(entry.get("properties")),
                    terrains=terrains,
                ),
            )
        )
    return refs


def _read_tmj_layers(payload: dict[str, Any], doc: MapDoc) -> None:
    """Append every layer, in the order the file lists them -- which is stacking
    order, bottom first, exactly as the XML reader's walk of the root is."""
    for entry in payload.get("layers", []):
        kind = str(entry.get("type", ""))
        name = str(entry.get("name", ""))
        # Both accessors are bound to *this* iteration's entry rather than
        # closing over the loop variable: they are called immediately, but a
        # closure over a name the loop rebinds is the bug that shape invites.
        _refuse_container_layers(lambda tag, kind=kind: kind == tag)
        _refuse_layer_offsets(
            lambda attr, entry=entry: float(entry.get(attr, 0) or 0), name
        )
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
                cells = _gid_array(list(raw or []), doc.width, doc.height)
            doc.layers.append(
                TileLayer(
                    uid=new_uid(),
                    id=int(entry.get("id", 0) or 0),
                    name=name,
                    data=cells,
                    visible=bool(entry.get("visible", True)),
                    opacity=float(entry.get("opacity", 1) or 1),
                    locked=bool(entry.get("locked", False)),
                    properties=read_json_properties(entry.get("properties")),
                )
            )
        elif kind == "objectgroup":
            doc.layers.append(
                ObjectLayer(
                    uid=new_uid(),
                    id=int(entry.get("id", 0) or 0),
                    name=name,
                    objects=[_json_object(o) for o in entry.get("objects", [])],
                    visible=bool(entry.get("visible", True)),
                    opacity=float(entry.get("opacity", 1) or 1),
                    locked=bool(entry.get("locked", False)),
                    properties=read_json_properties(entry.get("properties")),
                )
            )
        elif kind:
            raise TiledUnsupported(f"{kind} layers", f"layer {name!r}")


def read_tmj(
    data: bytes, *, image_loader: ImageLoader, tsx_loader: TilesetLoader
) -> MapDoc:
    """The JSON spelling of the same map. Every refusal above applies here.

    Split into the same two halves the XML reader has -- tilesets, then layers
    -- so the two formats are one shape read twice rather than two readers that
    happen to agree today.
    """
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"this is not a readable Tiled JSON map: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("a Tiled JSON map is an object")

    orientation = _check_orientation(str(payload.get("orientation", "orthogonal")))
    _refuse_infinite(bool(payload.get("infinite")))

    doc = MapDoc(
        width=int(payload.get("width", 1) or 1),
        height=int(payload.get("height", 1) or 1),
        tile_w=int(payload.get("tilewidth", 1) or 1),
        tile_h=int(payload.get("tileheight", 1) or 1),
        projection=orientation,
    )
    doc.renderorder = str(payload.get("renderorder", "right-down"))
    doc.backgroundcolor = payload.get("backgroundcolor")
    doc.properties = read_json_properties(payload.get("properties"))
    doc.tilesets = _read_tmj_tilesets(
        payload, image_loader=image_loader, tsx_loader=tsx_loader
    )
    _read_tmj_layers(payload, doc)

    _finish(
        doc,
        next_layer_id=_optional_int(payload.get("nextlayerid")),
        next_object_id=_optional_int(payload.get("nextobjectid")),
    )
    _adopt_object_space(doc)
    return doc


def _optional_int(value: Any) -> int | None:
    """A map-level ``nextlayerid``/``nextobjectid``, or ``None`` if absent.

    Distinct from the ``int(x.get(k, 0) or 0)`` idiom used for a layer or
    object's own ``id`` throughout this module: there, ``0`` and "absent" are
    the same fact (unassigned). Here they are not -- a file that omits the
    attribute needs the *computed* minimum, not zero, so ``_finish`` has to be
    able to tell "wrote nothing" apart from "wrote 0".
    """
    return None if value is None else int(value)


def _finish(
    doc: MapDoc, *, next_layer_id: int | None = None, next_object_id: int | None = None
) -> None:
    """Validate every gid, adopt the persistent-id counters, then hand back a
    document that reads clean.

    ``next_layer_id``/``next_object_id`` are the map's declared
    ``nextlayerid``/``nextobjectid`` when the file wrote them, else ``None``.
    Either way the counters are never set *below* one past the highest id
    actually seen -- a declared value that undershoots the ids present (a
    hand-edited or otherwise corrupt file) would let the very next allocation
    collide with an id already in the document, which is the one thing this
    counter exists to make impossible.
    """
    for layer in doc.tile_layers():
        if layer.data.shape != (doc.height, doc.width):
            raise ValueError(
                f"layer {layer.name!r} is {layer.data.shape[1]}x{layer.data.shape[0]}, "
                f"but the map is {doc.width}x{doc.height}"
            )
        # Refused by name rather than left to fail as an out-of-range id: the
        # bit is Tiled's hexagonal rotation, and "a tile no tileset accounts
        # for" is a true sentence about the wrong problem.
        if bool((np.asarray(layer.data) & _HEX_ROTATE).any()):
            raise TiledUnsupported(
                "hexagonal 120-degree tile rotation", f"layer {layer.name!r}"
            )
        ids = np.unique(gidlib.tile_ids(layer.data))
        for tile_id in ids.tolist():
            if tile_id and doc.ref_for(tile_id) is None:
                raise ValueError(
                    f"layer {layer.name!r} uses tile {tile_id}, which none of this "
                    "map's tilesets accounts for"
                )
    # The same check where the *other* gid in a document lives. A tile object
    # is a cell that happens to sit on an object layer -- same 29 bits, same
    # three flags -- so a gid nothing accounts for is the same unreadable file,
    # and letting it through here would mean the rule held only for tiles that
    # happened to be on a grid.
    for layer in doc.layers:
        if not isinstance(layer, ObjectLayer):
            continue
        for obj in layer.objects:
            gid_value = getattr(obj.shape, "gid", 0)
            tile_id = gidlib.decompose(int(gid_value))[0]
            if tile_id and doc.ref_for(tile_id) is None:
                raise ValueError(
                    f"object {obj.name or obj.uid} on layer {layer.name!r} uses tile "
                    f"{tile_id}, which none of this map's tilesets accounts for"
                )

    seen_layer_ids = [layer.id for layer in doc.layers if layer.id]
    seen_object_ids = [
        obj.id
        for layer in doc.layers
        if isinstance(layer, ObjectLayer)
        for obj in layer.objects
        if obj.id
    ]
    computed_layer_id = max(seen_layer_ids, default=0) + 1
    computed_object_id = max(seen_object_ids, default=0) + 1
    # Deliberately not "adopt the declared value outright" -- real Tiled
    # trusts its own nextlayerid/nextobjectid unconditionally, but this
    # reader does not, for the reason in this function's docstring. A
    # well-formed file (declared and computed agreeing) behaves identically
    # either way; only a corrupt one sees the difference.
    doc.next_layer_id = max(next_layer_id or 0, computed_layer_id)
    doc.next_object_id = max(next_object_id or 0, computed_object_id)

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
        files[f"tilesets/{stem}.png"] = png_bytes(ref.tileset.pixels)
        # The image name inside the .tsx is relative to the .tsx, which sits
        # beside it -- not to the map.
        files[tsx_path] = tsx_bytes(ref.tileset, image_name=f"{stem}.png")
        paths.append(tsx_path)
    return files, paths


def _export_ids(
    doc: MapDoc,
) -> tuple[dict[int, int], dict[int, int], int, int]:
    """The id every layer and object writes, plus the map-level ``next*``.

    A stored id (nonzero) is used as-is -- it is what an object-typed
    property references, and this is the only place all of them are visible
    at once. A layer or object still carrying ``id == 0`` -- a hand-built
    document, or one read from a ``.wmap``, whose ids are out of scope this
    milestone -- gets one assigned *here*, sequentially from the same
    counters a fresh document starts at. That assignment is never written
    back onto the object: two exports of the same untouched document still
    agree, because nothing about writing a file is allowed to change the
    document that asked for it.

    Keyed by ``uid`` -- the process-local address every other lookup in this
    package already uses -- rather than the persistent ``id`` field this
    function is computing, which is exactly the value not yet settled for a
    zero-id entry while this runs.
    """
    layer_ids: dict[int, int] = {}
    object_ids: dict[int, int] = {}
    fallback_layer_id = itertools.count(doc.next_layer_id)
    fallback_object_id = itertools.count(doc.next_object_id)
    for layer in doc.layers:
        layer_ids[layer.uid] = layer.id or next(fallback_layer_id)
        if isinstance(layer, ObjectLayer):
            for obj in layer.objects:
                object_ids[obj.uid] = obj.id or next(fallback_object_id)
    next_layer_id = max([doc.next_layer_id, *(v + 1 for v in layer_ids.values())])
    next_object_id = max([doc.next_object_id, *(v + 1 for v in object_ids.values())])
    return layer_ids, object_ids, next_layer_id, next_object_id


def tmx_export(doc: MapDoc) -> dict[str, bytes]:
    """The whole map as a mapping of relative path to bytes.

    A mapping rather than one blob because TMX has no portable way to embed an
    image: a tileset is a ``.tsx`` plus a ``.png`` beside the map. The caller
    writes them; deciding *where* is not this module's business.
    """
    _refuse_unwritable_objects(doc)
    files, tsx_paths = _tileset_files(doc)
    root = ET.Element(
        "map",
        {
            "version": MAP_VERSION,
            "tiledversion": TILED_VERSION,
            "orientation": doc.projection,
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
    layer_ids, object_ids, next_layer_id, next_object_id = _export_ids(doc)
    root.set("nextlayerid", str(next_layer_id))
    root.set("nextobjectid", str(next_object_id))
    write_properties(root, doc.properties)

    for ref, path in zip(doc.tilesets, tsx_paths, strict=True):
        ET.SubElement(root, "tileset", {"firstgid": str(ref.firstgid), "source": path})

    for layer in doc.layers:
        common = {"id": str(layer_ids[layer.uid]), "name": layer.name}
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
        if layer.locked:
            # Written only when set, the ``visible="0"`` idiom, and here that is
            # a requirement rather than tidiness: every export of an unlocked
            # map has to stay byte-for-byte what it was, and the round-trip
            # tests pin those bytes.
            node.set("locked", "1")
        write_properties(node, layer.properties)
        if isinstance(layer, TileLayer):
            data = ET.SubElement(node, "data", {"encoding": "csv"})
            data.text = _csv(layer.data)
        else:
            for obj in layer.objects:
                attrs = {"id": str(object_ids[obj.uid])}
                if obj.name:
                    attrs["name"] = obj.name
                if obj.obj_class:
                    attrs["class"] = obj.obj_class
                obj_x, obj_y = _object_xy(doc, obj)
                attrs["x"] = repr(obj_x)
                attrs["y"] = repr(obj_y)
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


def tmj_export(doc: MapDoc) -> dict[str, bytes]:
    """The JSON spelling. Same external tilesets, same names, same bytes."""
    _refuse_unwritable_objects(doc)
    files, tsx_paths = _tileset_files(doc)
    layer_ids, object_ids, next_layer_id, next_object_id = _export_ids(doc)

    layers: list[dict[str, Any]] = []
    for layer in doc.layers:
        entry: dict[str, Any] = {
            "id": layer_ids[layer.uid],
            "name": layer.name,
            "opacity": float(layer.opacity),
            "visible": bool(layer.visible),
            "x": 0,
            "y": 0,
        }
        if layer.locked:
            # Only when set, so an unlocked map's .tmj is byte-identical to what
            # it was before locks existed. Tiled omits the key too.
            entry["locked"] = True
        if layer.properties:
            entry["properties"] = write_json_properties(layer.properties)
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
                    "id": object_ids[obj.uid],
                    "name": obj.name,
                    "type": obj.obj_class,
                    "x": _object_xy(doc, obj)[0],
                    "y": _object_xy(doc, obj)[1],
                    # Both constants are *guaranteed* by
                    # ``_refuse_unwritable_objects`` above rather than assumed:
                    # a rotated object and an index-ordered layer are refused
                    # at this door, so writing them out is a statement of what
                    # got past it and not a value being dropped.
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
                    record["properties"] = write_json_properties(obj.properties)
                objects.append(record)
            entry.update({"type": "objectgroup", "draworder": "topdown", "objects": objects})
        layers.append(entry)

    payload: dict[str, Any] = {
        "type": "map",
        "version": MAP_VERSION,
        "tiledversion": TILED_VERSION,
        "orientation": doc.projection,
        "renderorder": doc.renderorder,
        "infinite": False,
        "width": doc.width,
        "height": doc.height,
        "tilewidth": doc.tile_w,
        "tileheight": doc.tile_h,
        "nextlayerid": next_layer_id,
        "nextobjectid": next_object_id,
        "tilesets": [
            {"firstgid": ref.firstgid, "source": path}
            for ref, path in zip(doc.tilesets, tsx_paths, strict=True)
        ],
        "layers": layers,
    }
    if doc.backgroundcolor:
        payload["backgroundcolor"] = str(doc.backgroundcolor)
    if doc.properties:
        payload["properties"] = write_json_properties(doc.properties)

    files["map.tmj"] = (json.dumps(payload, indent=2) + "\n").encode()
    return files
