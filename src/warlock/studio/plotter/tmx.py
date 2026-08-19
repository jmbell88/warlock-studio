"""``.tmx`` and ``.tmj`` -- Tiled's map, read and written.

Import and export both go through here, and the governing rule is that **a
feature this editor does not model is refused by name rather than dropped**.
Tiled's format is much larger than a finite stamp-and-fill editor: staggered
and hexagonal grids, infinite chunked layers, image-collection tilesets, Wang
sets and per-tile animation. Loading such a file and
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
can show. ``zstd`` is **read but never written**: every Tiled reads zlib, so
writing zstd would buy nothing and cost a reader, and its decoder is imported
lazily at the one call site so this module still imports where the studio extra
is absent.

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
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from ..tilegrid import gid as gidlib
from ..tilegrid.tileset import Tileset, TilesetRef, colour_text
from . import project
from .pngio import png_bytes
from .props import (
    TiledUnsupported,
    read_json_properties,
    read_properties,
    write_json_properties,
    write_properties,
)
from .tilemap import (
    OPAQUE_WHITE,
    Capsule,
    Ellipse,
    GroupLayer,
    ImageLayer,
    MapDoc,
    MapObject,
    ObjectLayer,
    Point,
    Polygon,
    Polyline,
    Rect,
    Text,
    TileLayer,
    TileShape,
    new_uid,
)
from .tsx import (
    TILED_VERSION,
    TSX_VERSION,
    check_tileset_features,
    check_tileset_features_json,
    phases_from_properties,
    read_tile_meta,
    read_wangsets,
    tileset_from_json,
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

# What a compression attribute may say.
#
# **``zstd`` reads but is never written.** Every Tiled reads zlib, so writing
# zstd would buy nothing and cost a reader; the claim this row makes is
# read-without-loss. ``zstandard`` is a wheel in the studio extra rather than a
# download, so the offline invariant is untouched -- and it is imported lazily,
# at the one call site, so this module still imports where the extra is absent.
_COMPRESSIONS = ("", "zlib", "gzip", "zstd")
#: What the *writer* may emit, which is a smaller set and deliberately so.
_WRITE_COMPRESSIONS = ("", "zlib", "gzip")
_ENCODINGS = ("", "csv", "base64")

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Tiled's hexagonal 120-degree rotation flag. See the note in :mod:`.gid` for
# why the constant lives here and not beside the other three.
_HEX_ROTATE = gidlib.DTYPE(0x10000000)


# --- shared refusals ----------------------------------------------------------


def _refuse_infinite(infinite: bool) -> None:
    """Keep finite-only storage explicit at both Tiled reader doors."""
    if infinite:
        raise TiledUnsupported(
            "an infinite map", "save it with a fixed size in Tiled's map properties"
        )


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
    if compression == "zstd":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - the extra is installed
            # A plain ``ValueError`` and deliberately **not** a named refusal:
            # zstd is supported, and what is missing is an install. A refusal
            # name here would be a name the compat ledger has no refused row
            # for, which is the drift that gate exists to catch.
            raise ValueError(
                "reading zstd layer data needs the studio extra's decoder"
            ) from exc
        # ``max_output_size`` is the same bound the two above use and for the
        # same reason: a few hundred bytes of archive can declare gigabytes, and
        # the read that discovers it is the one that already exhausted memory.
        try:
            out = zstandard.ZstdDecompressor().decompress(
                raw, max_output_size=expected + 1
            )
        except zstandard.ZstdError as exc:
            raise ValueError(f"a layer's zstd data could not be read: {exc}") from exc
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
        raise TiledUnsupported(f"{compression}-compressed layer data")

    if encoding == "csv":
        pieces = [piece for piece in text.replace("\n", "").split(",") if piece.strip()]
        try:
            values = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise ValueError("a layer's CSV data holds something that is not a number") from exc
        return _gid_array(values, width, height)

    if encoding != "base64":
        # ``""`` is in ``_ENCODINGS`` because a ``<data>`` holding ``<tile>``
        # elements has no encoding attribute -- but that form is routed to
        # ``_xml_tile_elements`` before this function is reached, so an empty
        # encoding arriving *here* means a ``<data>`` with text and no way
        # given to read it. Tiled writes no such element. It used to fall into
        # the base64 branch below and fail as a base64 error, or, for text that
        # happens to decode, succeed into a layer of garbage gids -- a sentence
        # about an encoding the file never claimed either way.
        raise ValueError(
            "a tile layer's <data> declares no encoding and carries no <tile> "
            "elements, so there is nothing to read it as"
        )
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


def _tiled_colour(value: Any, what: str = "a Tiled colour") -> tuple[int, int, int, int]:
    """Tiled's ``#RRGGBB``/``#AARRGGBB`` spelling as internal RGBA."""
    text = str(value or "").strip()
    if not text:
        return OPAQUE_WHITE
    digits = text[1:] if text.startswith("#") else text
    try:
        if len(digits) == 6:
            red, green, blue = (int(digits[index : index + 2], 16) for index in (0, 2, 4))
            return red, green, blue, 255
        if len(digits) == 8:
            alpha, red, green, blue = (
                int(digits[index : index + 2], 16) for index in (0, 2, 4, 6)
            )
            return red, green, blue, alpha
    except ValueError:
        pass
    raise ValueError(f"{what} is not #RRGGBB or #AARRGGBB: {text!r}")


def _tiled_colour_text(colour: Any) -> str:
    red, green, blue, alpha = (int(value) for value in colour)
    if alpha == 255:
        return f"#{red:02x}{green:02x}{blue:02x}"
    return f"#{alpha:02x}{red:02x}{green:02x}{blue:02x}"


def _xml_layer_common(node: ET.Element) -> dict[str, Any]:
    """Fields shared by all four XML layer elements."""
    return {
        "uid": new_uid(),
        "id": int(node.get("id", 0) or 0),
        "name": node.get("name", ""),
        "visible": node.get("visible", "1") not in ("0", "false"),
        "opacity": float(node.get("opacity", 1) or 1),
        "locked": node.get("locked", "0") not in ("0", "false"),
        "class_name": node.get("class") or node.get("type") or "",
        "blend_mode": node.get("mode", "normal"),
        "tint": _tiled_colour(node.get("tintcolor"), "a layer tint"),
        "offset_x": float(node.get("offsetx", 0) or 0),
        "offset_y": float(node.get("offsety", 0) or 0),
        "parallax_x": float(node.get("parallaxx", 1) or 1),
        "parallax_y": float(node.get("parallaxy", 1) or 1),
        "properties": read_properties(node),
    }


def _json_layer_common(entry: dict[str, Any]) -> dict[str, Any]:
    """Fields shared by all four JSON layer records."""
    return {
        "uid": new_uid(),
        "id": int(entry.get("id", 0) or 0),
        "name": str(entry.get("name", "")),
        "visible": bool(entry.get("visible", True)),
        "opacity": float(entry.get("opacity", 1) or 1),
        "locked": bool(entry.get("locked", False)),
        "class_name": str(entry.get("class", "")),
        "blend_mode": str(entry.get("mode", "normal")),
        "tint": _tiled_colour(entry.get("tintcolor"), "a layer tint"),
        "offset_x": float(entry.get("offsetx", 0) or 0),
        "offset_y": float(entry.get("offsety", 0) or 0),
        "parallax_x": float(entry.get("parallaxx", 1) or 1),
        "parallax_y": float(entry.get("parallaxy", 1) or 1),
        "properties": read_json_properties(entry.get("properties")),
    }


def _read_tmx_tilesets(
    root: ET.Element, *, image_loader: ImageLoader, tsx_loader: TilesetLoader
) -> list[TilesetRef]:
    refs: list[TilesetRef] = []
    for node in root.findall("tileset"):
        firstgid = int(node.get("firstgid", 1) or 1)
        source = node.get("source")
        if source:
            # ``.tsx`` and ``.tsj`` both go straight to the loader now: which
            # spelling a reference names is the *host*'s question, because only
            # the host reads bytes, and the engine has no way to tell them apart
            # that is not the extension it just handed over.
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
        grid = node.find("grid")
        transformations = node.find("transformations")
        props = read_properties(node)
        declared, remaining = phases_from_properties(props)
        terrains = () if wangsets is None else (read_wangsets(wangsets, declared) or ())
        refs.append(
            TilesetRef(
                firstgid=firstgid,
                tileset=Tileset(
                    name=node.get("name") or "tileset",
                    class_name=node.get("class") or node.get("type") or "",
                    pixels=image_loader(path),
                    tile_w=int(node.get("tilewidth", 0) or 0),
                    tile_h=int(node.get("tileheight", 0) or 0),
                    spacing=int(node.get("spacing", 0) or 0),
                    margin=int(node.get("margin", 0) or 0),
                    grid_orientation=(
                        "orthogonal"
                        if grid is None
                        else grid.get("orientation", "orthogonal")
                    ),
                    grid_width=(
                        int(node.get("tilewidth", 0) or 0)
                        if grid is None
                        else int(grid.get("width", node.get("tilewidth", 0)) or 0)
                    ),
                    grid_height=(
                        int(node.get("tileheight", 0) or 0)
                        if grid is None
                        else int(grid.get("height", node.get("tileheight", 0)) or 0)
                    ),
                    transformations=(
                        False,
                        False,
                        False,
                        False,
                    )
                    if transformations is None
                    else (
                        transformations.get("hflip", "0") not in ("0", "false"),
                        transformations.get("vflip", "0") not in ("0", "false"),
                        transformations.get("rotate", "0") not in ("0", "false"),
                        transformations.get("preferuntransformed", "0")
                        not in ("0", "false"),
                    ),
                    # The "phases" property is structural only on a recognised
                    # terrain set (the reader popped it; the writer re-derives
                    # it); anywhere else it is somebody's ordinary custom
                    # property and travels intact.
                    properties=remaining if terrains else props,
                    # ``check_tileset_features`` already refused an unrecognised
                    # set; an embedded one that *is* recognised is a terrain set
                    # and used to be accepted and then dropped, which left the
                    # tool greyed out on a map whose atlas plainly declares it.
                    terrains=terrains,
                    phases=declared if terrains else 1,
                    tiles=read_tile_meta(node),
                ),
            )
        )
    return refs


def _xml_points(node: ET.Element, tag: str) -> tuple[tuple[float, float], ...]:
    child = node.find(tag)
    raw = "" if child is None else child.get("points", "")
    try:
        return tuple(
            (float(pair.split(",", 1)[0]), float(pair.split(",", 1)[1]))
            for pair in raw.split()
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(f"a {tag} object has malformed points") from exc


def _read_tmx_text(node: ET.Element, width: float, height: float) -> Text:
    text = node.find("text")
    assert text is not None
    return Text(
        text=text.text or "",
        w=width,
        h=height,
        family=text.get("fontfamily", "sans-serif"),
        pixel_size=int(text.get("pixelsize", 16) or 16),
        wrap=text.get("wrap", "0") not in ("0", "false"),
        color=text.get("color", "#000000"),
        halign=text.get("halign", "left"),
        valign=text.get("valign", "top"),
        bold=text.get("bold", "0") not in ("0", "false"),
        italic=text.get("italic", "0") not in ("0", "false"),
        underline=text.get("underline", "0") not in ("0", "false"),
        strikeout=text.get("strikeout", "0") not in ("0", "false"),
        kerning=text.get("kerning", "1") not in ("0", "false"),
    )


def _read_tmx_object(node: ET.Element) -> MapObject:
    name = node.get("name", "")
    where = f"object {node.get('id', '?')}"
    if node.get("template"):
        raise TiledUnsupported("object templates", where)
    width = float(node.get("width", 0) or 0)
    height = float(node.get("height", 0) or 0)
    if node.get("gid") is not None:
        shape: Any = TileShape(gid=int(node.get("gid", 0) or 0), w=width, h=height)
    elif node.find("ellipse") is not None:
        shape = Ellipse(width, height)
    elif node.find("capsule") is not None:
        shape = Capsule(width, height)
    elif node.find("point") is not None:
        shape = Point()
    elif node.find("polygon") is not None:
        shape = Polygon(_xml_points(node, "polygon"))
    elif node.find("polyline") is not None:
        shape = Polyline(_xml_points(node, "polyline"))
    elif node.find("text") is not None:
        shape = _read_tmx_text(node, width, height)
    else:
        shape = Rect(width, height)
    return MapObject(
        uid=new_uid(),
        id=int(node.get("id", 0) or 0),
        name=name,
        shape=shape,
        x=float(node.get("x", 0) or 0),
        y=float(node.get("y", 0) or 0),
        rotation=float(node.get("rotation", 0) or 0),
        opacity=float(node.get("opacity", 1) or 1),
        obj_class=node.get("type") or node.get("class") or "",
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
    for layer in doc.all_layers():
        if isinstance(layer, ObjectLayer):
            for obj in layer.objects:
                obj.x, obj.y = project.object_to_pixels(doc._lattice(), obj.x, obj.y)


def _object_xy(doc: MapDoc, obj: MapObject) -> tuple[float, float]:
    """One object's position in Tiled's space, for the two writers."""
    return project.object_from_pixels(doc._lattice(), obj.x, obj.y)


def _read_tmx_layers(
    nodes: Any, doc: MapDoc, *, image_loader: ImageLoader
) -> list[Any]:
    """Read one XML layer list recursively, preserving paint order."""
    layers: list[Any] = []
    for node in nodes:
        if node.tag not in ("layer", "objectgroup", "imagelayer", "group"):
            continue
        common = _xml_layer_common(node)
        name = common["name"]
        if node.tag in ("layer", "objectgroup") and (
            float(node.get("x", 0) or 0) or float(node.get("y", 0) or 0)
        ):
            raise TiledUnsupported("layer tile coordinates", f"layer {name!r}")
        if node.tag == "layer":
            width = int(node.get("width", doc.width) or doc.width)
            height = int(node.get("height", doc.height) or doc.height)
            if (width, height) != (doc.width, doc.height):
                raise ValueError(
                    f"tile layer {name!r} is {width}x{height}, but the fixed map is "
                    f"{doc.width}x{doc.height}"
                )
            payload = node.find("data")
            if payload is None:
                raise ValueError(f"tile layer {name!r} carries no <data>")
            encoding = payload.get("encoding", "")
            if not encoding and payload.find("tile") is not None:
                cells = _xml_tile_elements(payload, width, height)
            else:
                cells = _decode_payload(
                    payload.text or "",
                    encoding,
                    payload.get("compression", ""),
                    width,
                    height,
                )
            layers.append(TileLayer(**common, data=cells))
        elif node.tag == "objectgroup":
            layers.append(
                ObjectLayer(
                    **common,
                    draworder=node.get("draworder", "topdown"),
                    # Validated at the reader door as well as at the setter:
                    # this value is stored as Tiled's own text and written back
                    # verbatim, so an unparseable one taken on trust here would
                    # be carried straight into the next export.
                    color=colour_text(node.get("color"), "an object layer colour"),
                    objects=[_read_tmx_object(obj) for obj in node.findall("object")],
                )
            )
        elif node.tag == "imagelayer":
            image = node.find("image")
            if image is not None and image.get("trans"):
                raise TiledUnsupported(
                    "an image layer transparent colour", f"layer {name!r}"
                )
            source = "" if image is None else str(image.get("source", "") or "")
            pixels = (
                np.zeros((0, 0, 4), dtype=np.uint8)
                if not source
                else image_loader(source)
            )
            # Deprecated image-layer x/y were pixel offsets. Folding them into
            # offset is semantics-preserving and normalizes only the encoding.
            common["offset_x"] += float(node.get("x", 0) or 0)
            common["offset_y"] += float(node.get("y", 0) or 0)
            layers.append(
                ImageLayer(
                    **common,
                    pixels=pixels,
                    source=source,
                    repeat_x=node.get("repeatx", "0") not in ("0", "false"),
                    repeat_y=node.get("repeaty", "0") not in ("0", "false"),
                )
            )
        else:
            layers.append(
                GroupLayer(
                    **common,
                    children=_read_tmx_layers(node, doc, image_loader=image_loader),
                )
            )
    return layers


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
    doc.class_name = root.get("class") or root.get("type") or ""
    doc.parallax_origin = (
        float(root.get("parallaxoriginx", 0) or 0),
        float(root.get("parallaxoriginy", 0) or 0),
    )
    doc.skew_x = int(root.get("skewx", 0) or 0)
    doc.skew_y = int(root.get("skewy", 0) or 0)
    doc.properties = read_properties(root)
    doc.tilesets = _read_tmx_tilesets(
        root, image_loader=image_loader, tsx_loader=tsx_loader
    )

    doc.layers.extend(_read_tmx_layers(root, doc, image_loader=image_loader))

    _finish(
        doc,
        next_layer_id=_optional_int(root.get("nextlayerid")),
        next_object_id=_optional_int(root.get("nextobjectid")),
    )
    _adopt_object_space(doc)
    return doc


# --- JSON reading -------------------------------------------------------------


def _json_points(entry: dict[str, Any], key: str) -> tuple[tuple[float, float], ...]:
    raw = entry.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"a {key} object has malformed points")
    try:
        return tuple((float(point["x"]), float(point["y"])) for point in raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"a {key} object has malformed points") from exc


def _json_text(entry: dict[str, Any], width: float, height: float) -> Text:
    text = entry.get("text")
    if not isinstance(text, dict):
        raise ValueError("a text object has malformed styling")
    return Text(
        text=str(text.get("text", "")),
        w=width,
        h=height,
        family=str(text.get("fontfamily", "sans-serif")),
        pixel_size=int(text.get("pixelsize", 16) or 16),
        wrap=bool(text.get("wrap", False)),
        color=str(text.get("color", "#000000")),
        halign=str(text.get("halign", "left")),
        valign=str(text.get("valign", "top")),
        bold=bool(text.get("bold", False)),
        italic=bool(text.get("italic", False)),
        underline=bool(text.get("underline", False)),
        strikeout=bool(text.get("strikeout", False)),
        kerning=bool(text.get("kerning", True)),
    )


def _json_object(entry: dict[str, Any]) -> MapObject:
    where = f"object {entry.get('id', '?')}"
    if entry.get("template"):
        raise TiledUnsupported("object templates", where)
    width = float(entry.get("width", 0) or 0)
    height = float(entry.get("height", 0) or 0)
    if "gid" in entry:
        shape: Any = TileShape(gid=int(entry.get("gid", 0) or 0), w=width, h=height)
    elif entry.get("ellipse"):
        shape = Ellipse(width, height)
    elif entry.get("capsule"):
        shape = Capsule(width, height)
    elif entry.get("point"):
        shape = Point()
    elif "polygon" in entry:
        shape = Polygon(_json_points(entry, "polygon"))
    elif "polyline" in entry:
        shape = Polyline(_json_points(entry, "polyline"))
    elif "text" in entry:
        shape = _json_text(entry, width, height)
    else:
        shape = Rect(width, height)
    return MapObject(
        uid=new_uid(),
        id=int(entry.get("id", 0) or 0),
        name=str(entry.get("name", "")),
        shape=shape,
        x=float(entry.get("x", 0) or 0),
        y=float(entry.get("y", 0) or 0),
        rotation=float(entry.get("rotation", 0) or 0),
        opacity=float(entry.get("opacity", 1) or 1),
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
            refs.append(
                TilesetRef(firstgid=firstgid, tileset=tsx_loader(source), source=source)
            )
            continue
        # **One definition**, shared with the external ``.tsj`` reader: a second
        # opinion about what a JSON tileset means is exactly how the external and
        # embedded spellings of one file come to load differently. Everything
        # this used to spell out inline -- the feature check, the grid, the
        # transformations, the wang model, the per-tile metadata, the
        # presentation -- lives in ``tsx.tileset_from_json`` now.
        # **The feature check runs first**, and the order is load-bearing: a
        # true image-collection tileset has per-tile images and *no* top-level
        # one, so checking the image first would report "an embedded tileset
        # image" for a file whose actual problem is that it is a collection.
        check_tileset_features_json(entry)
        image = str(entry.get("image", ""))
        if not image:
            raise TiledUnsupported(
                "an embedded tileset image", "Plotter needs an image path"
            )
        refs.append(
            TilesetRef(
                firstgid=firstgid,
                tileset=tileset_from_json(entry, image_loader(image)),
            )
        )
    return refs


def _read_tmj_layer_list(
    entries: Any, doc: MapDoc, *, image_loader: ImageLoader
) -> list[Any]:
    """One JSON layer list recursively, preserving paint order."""
    if not isinstance(entries, list):
        raise ValueError("a Tiled JSON layer list is not an array")
    layers: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("a Tiled JSON layer is not an object")
        kind = str(entry.get("type", ""))
        name = str(entry.get("name", ""))
        common = _json_layer_common(entry)
        if kind in ("tilelayer", "objectgroup") and (
            float(entry.get("x", 0) or 0) or float(entry.get("y", 0) or 0)
        ):
            raise TiledUnsupported("layer tile coordinates", f"layer {name!r}")
        if kind == "tilelayer":
            if entry.get("chunks"):
                raise TiledUnsupported("chunked (infinite) layer data", f"layer {name!r}")
            width = int(entry.get("width", doc.width) or doc.width)
            height = int(entry.get("height", doc.height) or doc.height)
            if (width, height) != (doc.width, doc.height):
                raise ValueError(
                    f"tile layer {name!r} is {width}x{height}, but the fixed map is "
                    f"{doc.width}x{doc.height}"
                )
            raw = entry.get("data")
            if isinstance(raw, str):
                cells = _decode_payload(
                    raw,
                    "base64",
                    str(entry.get("compression", "") or ""),
                    width,
                    height,
                )
            else:
                cells = _gid_array(list(raw or []), width, height)
            layers.append(TileLayer(**common, data=cells))
        elif kind == "objectgroup":
            layers.append(
                ObjectLayer(
                    **common,
                    draworder=str(entry.get("draworder", "topdown")),
                    # The XML reader's rule, in the other spelling. This file's
                    # standing rule is that the two paths do not drift.
                    color=colour_text(entry.get("color"), "an object layer colour"),
                    objects=[_json_object(o) for o in entry.get("objects", [])],
                )
            )
        elif kind == "imagelayer":
            if entry.get("transparentcolor"):
                raise TiledUnsupported(
                    "an image layer transparent colour", f"layer {name!r}"
                )
            source = str(entry.get("image", "") or "")
            pixels = (
                np.zeros((0, 0, 4), dtype=np.uint8)
                if not source
                else image_loader(source)
            )
            layers.append(
                ImageLayer(
                    **common,
                    pixels=pixels,
                    source=source,
                    repeat_x=bool(entry.get("repeatx", False)),
                    repeat_y=bool(entry.get("repeaty", False)),
                )
            )
        elif kind == "group":
            layers.append(
                GroupLayer(
                    **common,
                    children=_read_tmj_layer_list(
                        entry.get("layers", []), doc, image_loader=image_loader
                    ),
                )
            )
        elif kind:
            raise TiledUnsupported(f"{kind} layers", f"layer {name!r}")
        else:
            raise ValueError(f"layer {name!r} has no type")
    return layers


def _read_tmj_layers(
    payload: dict[str, Any], doc: MapDoc, *, image_loader: ImageLoader
) -> None:
    doc.layers.extend(
        _read_tmj_layer_list(payload.get("layers", []), doc, image_loader=image_loader)
    )


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
    doc.class_name = str(payload.get("class", ""))
    doc.parallax_origin = (
        float(payload.get("parallaxoriginx", 0) or 0),
        float(payload.get("parallaxoriginy", 0) or 0),
    )
    doc.skew_x = int(payload.get("skewx", 0) or 0)
    doc.skew_y = int(payload.get("skewy", 0) or 0)
    doc.properties = read_json_properties(payload.get("properties"))
    doc.tilesets = _read_tmj_tilesets(
        payload, image_loader=image_loader, tsx_loader=tsx_loader
    )
    _read_tmj_layers(payload, doc, image_loader=image_loader)

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
    if doc.infinite:
        # A *reader's* sentence. Both callers of this function are readers
        # (``read_tmx`` and ``read_tmj``), so ``exporting=True`` -- which swaps
        # in "this map uses ..., which Plotter cannot write" and drops the
        # remedy -- was the wrong half of the pair. ``_check_exportable_map``
        # below is the writer door and keeps the flag. Unreachable today, since
        # an infinite map is refused before a document is built at all; kept
        # because the check is cheap and the day it is reachable is the day the
        # sentence matters.
        raise TiledUnsupported("an infinite map")
    all_layers = doc.all_layers()
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
    for layer in all_layers:
        if not isinstance(layer, ObjectLayer):
            continue
        for obj in layer.objects:
            gid_value = int(getattr(obj.shape, "gid", 0))
            # By name here too. ``decompose`` strips the three transform flags
            # and leaves bit 28, so a tile object carrying Tiled's hexagonal
            # rotation arrived at the check below as tile 268435457 and was
            # refused as "a tile no tileset accounts for" -- a true sentence
            # about the wrong problem, which is the exact wording the layer
            # pass above exists to avoid. The rule held for tiles on a grid and
            # not for the same gid on an object layer.
            if gid_value & int(_HEX_ROTATE):
                raise TiledUnsupported(
                    "hexagonal 120-degree tile rotation",
                    f"object {obj.name or obj.uid} on layer {layer.name!r}",
                )
            tile_id = gidlib.decompose(gid_value)[0]
            if tile_id and doc.ref_for(tile_id) is None:
                raise ValueError(
                    f"object {obj.name or obj.uid} on layer {layer.name!r} uses tile "
                    f"{tile_id}, which none of this map's tilesets accounts for"
                )

    seen_layer_ids = [layer.id for layer in all_layers if layer.id]
    seen_object_ids = [
        obj.id
        for layer in all_layers
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


def _check_exportable_map(doc: MapDoc) -> None:
    """Map-level writer doors shared by XML and JSON exporters."""
    if doc.infinite:
        raise TiledUnsupported("an infinite map", exporting=True)


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
    all_layers = doc.all_layers()
    taken_layers = {layer.id for layer in all_layers if layer.id}
    taken_objects = {
        obj.id
        for layer in all_layers
        if isinstance(layer, ObjectLayer)
        for obj in layer.objects
        if obj.id
    }

    def mint(start: int, taken: set[int]) -> Any:
        for value in itertools.count(start):
            if value not in taken:
                taken.add(value)
                yield value

    fallback_layer_id = mint(doc.next_layer_id, taken_layers)
    fallback_object_id = mint(doc.next_object_id, taken_objects)
    for layer in all_layers:
        layer_ids[layer.uid] = layer.id or next(fallback_layer_id)
        if isinstance(layer, ObjectLayer):
            for obj in layer.objects:
                object_ids[obj.uid] = obj.id or next(fallback_object_id)
    next_layer_id = max([doc.next_layer_id, *(v + 1 for v in layer_ids.values())])
    next_object_id = max([doc.next_object_id, *(v + 1 for v in object_ids.values())])
    return layer_ids, object_ids, next_layer_id, next_object_id


def _image_layer_files(doc: MapDoc, files: dict[str, bytes]) -> dict[int, str]:
    """Add image-layer PNGs and return each layer uid's safe relative path."""
    paths: dict[int, str] = {}
    for index, layer in enumerate(
        entry for entry in doc.all_layers() if isinstance(entry, ImageLayer)
    ):
        if layer.pixels.size == 0:
            continue
        raw = png_bytes(layer.pixels)
        source = str(layer.source).replace("\\", "/")
        candidate = PurePosixPath(source)
        safe = bool(source) and not candidate.is_absolute() and ".." not in candidate.parts
        stem = _SAFE.sub("-", layer.name).strip("-") or "image"
        path = source if safe else f"images/{index:02d}-{stem}.png"
        if path in files and files[path] != raw:
            # The colliding path can *be* this layer's fallback -- another
            # layer's safe source spelt ``images/NN-stem.png`` -- so recomputing
            # the fallback once reproduced the same string and clobbered the
            # other layer's bytes. Bump until the name is genuinely free (or
            # already holds these exact bytes, which is sharing, not a clash).
            for bump in itertools.count(index):
                path = f"images/{bump:02d}-{stem}.png"
                if path not in files or files[path] == raw:
                    break
        files[path] = raw
        paths[layer.uid] = path
    return paths


def _xml_common_layer(node: ET.Element, layer: Any, layer_id: int) -> None:
    node.set("id", str(layer_id))
    node.set("name", layer.name)
    if layer.class_name:
        node.set("class", layer.class_name)
    if layer.opacity != 1.0:
        node.set("opacity", repr(float(layer.opacity)))
    if not layer.visible:
        node.set("visible", "0")
    if layer.locked:
        node.set("locked", "1")
    if tuple(layer.tint) != OPAQUE_WHITE:
        node.set("tintcolor", _tiled_colour_text(layer.tint))
    if layer.offset_x:
        node.set("offsetx", repr(float(layer.offset_x)))
    if layer.offset_y:
        node.set("offsety", repr(float(layer.offset_y)))
    if layer.parallax_x != 1.0:
        node.set("parallaxx", repr(float(layer.parallax_x)))
    if layer.parallax_y != 1.0:
        node.set("parallaxy", repr(float(layer.parallax_y)))
    if layer.blend_mode != "normal":
        node.set("mode", layer.blend_mode)
    write_properties(node, layer.properties)


def _points_text(points: Any) -> str:
    return " ".join(f"{repr(float(x))},{repr(float(y))}" for x, y in points)


def _xml_text(parent: ET.Element, shape: Text) -> None:
    attrs: dict[str, str] = {}
    for name, value, default in (
        ("fontfamily", shape.family, "sans-serif"),
        ("pixelsize", shape.pixel_size, 16),
        ("wrap", int(shape.wrap), 0),
        ("color", shape.color, "#000000"),
        ("halign", shape.halign, "left"),
        ("valign", shape.valign, "top"),
        ("bold", int(shape.bold), 0),
        ("italic", int(shape.italic), 0),
        ("underline", int(shape.underline), 0),
        ("strikeout", int(shape.strikeout), 0),
        ("kerning", int(shape.kerning), 1),
    ):
        if value != default:
            attrs[name] = str(value)
    node = ET.SubElement(parent, "text", attrs)
    node.text = shape.text


def _write_tmx_object(
    parent: ET.Element, doc: MapDoc, obj: MapObject, object_id: int
) -> None:
    obj_x, obj_y = _object_xy(doc, obj)
    attrs = {
        "id": str(object_id),
        "x": repr(float(obj_x)),
        "y": repr(float(obj_y)),
    }
    if obj.name:
        attrs["name"] = obj.name
    if obj.obj_class:
        # Tiled 1.10 renamed Object.class back to ``type`` for compatibility.
        attrs["type"] = obj.obj_class
    if obj.rotation:
        attrs["rotation"] = repr(float(obj.rotation))
    if obj.opacity != 1.0:
        attrs["opacity"] = repr(float(obj.opacity))
    if not obj.visible:
        attrs["visible"] = "0"
    if hasattr(obj.shape, "w"):
        attrs["width"] = repr(float(obj.w))
        attrs["height"] = repr(float(obj.h))
    if isinstance(obj.shape, TileShape):
        attrs["gid"] = str(int(obj.shape.gid))
    node = ET.SubElement(parent, "object", attrs)
    write_properties(node, obj.properties)
    if isinstance(obj.shape, Point):
        ET.SubElement(node, "point")
    elif isinstance(obj.shape, Ellipse):
        ET.SubElement(node, "ellipse")
    elif isinstance(obj.shape, Capsule):
        ET.SubElement(node, "capsule")
    elif isinstance(obj.shape, Polygon):
        ET.SubElement(node, "polygon", {"points": _points_text(obj.shape.points)})
    elif isinstance(obj.shape, Polyline):
        ET.SubElement(node, "polyline", {"points": _points_text(obj.shape.points)})
    elif isinstance(obj.shape, Text):
        _xml_text(node, obj.shape)


def _write_tmx_layers(
    parent: ET.Element,
    doc: MapDoc,
    layers: list[Any],
    layer_ids: dict[int, int],
    object_ids: dict[int, int],
    image_paths: dict[int, str],
) -> None:
    for layer in layers:
        if isinstance(layer, TileLayer):
            node = ET.SubElement(
                parent,
                "layer",
                {"width": str(doc.width), "height": str(doc.height)},
            )
        elif isinstance(layer, ObjectLayer):
            node = ET.SubElement(parent, "objectgroup", {"draworder": layer.draworder})
            if layer.color:
                node.set("color", layer.color)
        elif isinstance(layer, ImageLayer):
            node = ET.SubElement(parent, "imagelayer")
            if layer.repeat_x:
                node.set("repeatx", "1")
            if layer.repeat_y:
                node.set("repeaty", "1")
        elif isinstance(layer, GroupLayer):
            node = ET.SubElement(parent, "group")
        else:  # pragma: no cover - the closed Layer union guards this
            raise TypeError(f"unknown layer kind {type(layer).__name__}")
        _xml_common_layer(node, layer, layer_ids[layer.uid])
        if isinstance(layer, TileLayer):
            data = ET.SubElement(node, "data", {"encoding": "csv"})
            data.text = _csv(layer.data)
        elif isinstance(layer, ObjectLayer):
            for obj in layer.objects:
                _write_tmx_object(node, doc, obj, object_ids[obj.uid])
        elif isinstance(layer, ImageLayer):
            path = image_paths.get(layer.uid)
            if path:
                ET.SubElement(
                    node,
                    "image",
                    {
                        "source": path,
                        "width": str(layer.width),
                        "height": str(layer.height),
                    },
                )
        else:
            _write_tmx_layers(
                node, doc, layer.children, layer_ids, object_ids, image_paths
            )


def tmx_export(doc: MapDoc) -> dict[str, bytes]:
    """The whole map as a mapping of relative path to bytes.

    A mapping rather than one blob because TMX has no portable way to embed an
    image: a tileset is a ``.tsx`` plus a ``.png`` beside the map. The caller
    writes them; deciding *where* is not this module's business.
    """
    _check_exportable_map(doc)
    files, tsx_paths = _tileset_files(doc)
    image_paths = _image_layer_files(doc, files)
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
    if doc.class_name:
        root.set("class", str(doc.class_name))
    if doc.parallax_origin[0]:
        root.set("parallaxoriginx", repr(float(doc.parallax_origin[0])))
    if doc.parallax_origin[1]:
        root.set("parallaxoriginy", repr(float(doc.parallax_origin[1])))
    if doc.projection == project.OBLIQUE:
        root.set("skewx", str(int(doc.skew_x)))
        root.set("skewy", str(int(doc.skew_y)))
    layer_ids, object_ids, next_layer_id, next_object_id = _export_ids(doc)
    root.set("nextlayerid", str(next_layer_id))
    root.set("nextobjectid", str(next_object_id))
    write_properties(root, doc.properties)

    for ref, path in zip(doc.tilesets, tsx_paths, strict=True):
        ET.SubElement(root, "tileset", {"firstgid": str(ref.firstgid), "source": path})

    _write_tmx_layers(root, doc, doc.layers, layer_ids, object_ids, image_paths)

    files["map.tmx"] = to_bytes(root)
    return files


def _json_common_layer(layer: Any, layer_id: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": layer_id,
        "name": layer.name,
        "opacity": float(layer.opacity),
        "visible": bool(layer.visible),
    }
    if layer.locked:
        entry["locked"] = True
    if layer.class_name:
        entry["class"] = layer.class_name
    if layer.blend_mode != "normal":
        entry["mode"] = layer.blend_mode
    if tuple(layer.tint) != OPAQUE_WHITE:
        entry["tintcolor"] = _tiled_colour_text(layer.tint)
    if layer.offset_x:
        entry["offsetx"] = float(layer.offset_x)
    if layer.offset_y:
        entry["offsety"] = float(layer.offset_y)
    if layer.parallax_x != 1.0:
        entry["parallaxx"] = float(layer.parallax_x)
    if layer.parallax_y != 1.0:
        entry["parallaxy"] = float(layer.parallax_y)
    if layer.properties:
        entry["properties"] = write_json_properties(layer.properties)
    return entry


def _json_text_record(shape: Text) -> dict[str, Any]:
    return {
        "text": shape.text,
        "fontfamily": shape.family,
        "pixelsize": shape.pixel_size,
        "wrap": shape.wrap,
        "color": shape.color,
        "halign": shape.halign,
        "valign": shape.valign,
        "bold": shape.bold,
        "italic": shape.italic,
        "underline": shape.underline,
        "strikeout": shape.strikeout,
        "kerning": shape.kerning,
    }


def _json_object_record(
    doc: MapDoc, obj: MapObject, object_id: int
) -> dict[str, Any]:
    obj_x, obj_y = _object_xy(doc, obj)
    record: dict[str, Any] = {
        "id": object_id,
        "name": obj.name,
        "type": obj.obj_class,
        "x": obj_x,
        "y": obj_y,
        "width": float(obj.w),
        "height": float(obj.h),
        "rotation": float(obj.rotation),
        "opacity": float(obj.opacity),
        "visible": bool(obj.visible),
    }
    if isinstance(obj.shape, Point):
        record["point"] = True
    elif isinstance(obj.shape, Ellipse):
        record["ellipse"] = True
    elif isinstance(obj.shape, Capsule):
        record["capsule"] = True
    elif isinstance(obj.shape, Polygon):
        record["polygon"] = [
            {"x": float(x), "y": float(y)} for x, y in obj.shape.points
        ]
    elif isinstance(obj.shape, Polyline):
        record["polyline"] = [
            {"x": float(x), "y": float(y)} for x, y in obj.shape.points
        ]
    elif isinstance(obj.shape, TileShape):
        record["gid"] = int(obj.shape.gid)
    elif isinstance(obj.shape, Text):
        record["text"] = _json_text_record(obj.shape)
    if obj.properties:
        record["properties"] = write_json_properties(obj.properties)
    return record


def _write_tmj_layers(
    doc: MapDoc,
    layers: list[Any],
    layer_ids: dict[int, int],
    object_ids: dict[int, int],
    image_paths: dict[int, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for layer in layers:
        entry = _json_common_layer(layer, layer_ids[layer.uid])
        if isinstance(layer, TileLayer):
            entry.update(
                {
                    "type": "tilelayer",
                    "width": doc.width,
                    "height": doc.height,
                    "x": 0,
                    "y": 0,
                    "data": [int(value) for value in layer.data.reshape(-1)],
                }
            )
        elif isinstance(layer, ObjectLayer):
            entry.update(
                {
                    "type": "objectgroup",
                    "draworder": layer.draworder,
                    "objects": [
                        _json_object_record(doc, obj, object_ids[obj.uid])
                        for obj in layer.objects
                    ],
                    "x": 0,
                    "y": 0,
                }
            )
            if layer.color:
                entry["color"] = layer.color
        elif isinstance(layer, ImageLayer):
            entry["type"] = "imagelayer"
            path = image_paths.get(layer.uid)
            if path:
                entry.update(
                    {
                        "image": path,
                        "imagewidth": layer.width,
                        "imageheight": layer.height,
                    }
                )
            if layer.repeat_x:
                entry["repeatx"] = True
            if layer.repeat_y:
                entry["repeaty"] = True
        elif isinstance(layer, GroupLayer):
            entry.update(
                {
                    "type": "group",
                    "layers": _write_tmj_layers(
                        doc, layer.children, layer_ids, object_ids, image_paths
                    ),
                }
            )
        else:  # pragma: no cover - the closed Layer union guards this
            raise TypeError(f"unknown layer kind {type(layer).__name__}")
        out.append(entry)
    return out


def tmj_export(doc: MapDoc) -> dict[str, bytes]:
    """The JSON spelling. Same external tilesets, same names, same bytes."""
    _check_exportable_map(doc)
    files, tsx_paths = _tileset_files(doc)
    image_paths = _image_layer_files(doc, files)
    layer_ids, object_ids, next_layer_id, next_object_id = _export_ids(doc)
    layers = _write_tmj_layers(doc, doc.layers, layer_ids, object_ids, image_paths)

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
    if doc.class_name:
        payload["class"] = str(doc.class_name)
    if doc.parallax_origin[0]:
        payload["parallaxoriginx"] = float(doc.parallax_origin[0])
    if doc.parallax_origin[1]:
        payload["parallaxoriginy"] = float(doc.parallax_origin[1])
    if doc.projection == project.OBLIQUE:
        payload["skewx"] = int(doc.skew_x)
        payload["skewy"] = int(doc.skew_y)
    if doc.properties:
        payload["properties"] = write_json_properties(doc.properties)

    files["map.tmj"] = (json.dumps(payload, indent=2) + "\n").encode()
    return files
