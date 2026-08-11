"""``.wmap`` -- the Plotter document on disk, as a zip.

The ``.wblk`` shape, for the ``.wblk`` reasons: ``map.json`` is the small,
human-meaningful half that a person can read and a diff can show, and the layer
arrays are ``.npy`` members rather than nested JSON lists, because a 200x200
layer written as text is most of a megabyte of ``0,`` and its own bytes
otherwise.

**Every timestamp in the archive is fixed** at 1980-01-01, the earliest a zip
can express. A zip stamps each member with the wall clock, so an unchanged
document would otherwise produce different bytes every time it was written --
which makes the file undiffable, a content hash useless, and "has this actually
changed" unanswerable outside the app.

**The file stores indices, never uids.** A uid is minted per process and means
nothing in a file; what a document needs back is *which layer* an entry is, and
that is its position. Every layer and object therefore gets a *fresh* uid on
read, which is also why this format needs no ``reserve_uid`` step of the kind
``.wblk`` has: ``.wblk`` restores the uids it stored and so must raise the
process floor past them, whereas nothing here is ever restored onto a number the
counter has already issued.

**Tileset images are embedded**, which is the house pattern (``.wblk`` embeds
its textures, ``.ora`` its layers) and buys two things: the file is the whole
document, so it can be moved or sent without a folder of dependencies, and the
reader can validate at the door rather than discovering a missing PNG halfway
through. The accepted cost, stated rather than hidden: editing the source PNG on
disk does not propagate into a map already saved. A ``.tmx`` export is the way
out to an external, editable tileset.

**A half-read document is worse than a refused one.** A file from a newer
version, a layer whose array is the wrong shape or dtype, a tileset naming a
member the archive does not carry, a gid no tileset accounts for, tilesets whose
firstgids do not increase: each is refused, because each of them opens as a map
that looks nearly right and can be saved back over the original.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from . import gid as gidlib
from . import project
from .pngio import png_bytes
from .tilemap import (
    OBJECT_KINDS,
    MapDoc,
    MapObject,
    ObjectLayer,
    TileLayer,
    new_uid,
)
from .tileset import TerrainSpec, Tileset, TilesetRef
from .tsx import Prop

VERSION = 2
MANIFEST = "map.json"
LAYER_DIR = "layers"
TILESET_DIR = "tilesets"

_EPOCH = (1980, 1, 1, 0, 0, 0)

WMAP_SUFFIX = ".wmap"

# A zip's directory declares what each member unpacks to, and nothing makes that
# number honest -- a few kilobytes of archive can claim terabytes, and the read
# that discovers this is the one that has already exhausted memory. One gigabyte
# is far past any map this editor produces (a 512-square layer is a megabyte and
# a large atlas tens of them) and far short of a machine's RAM, so the ceiling is
# only ever hit by a file that was not written by us. The ``clay/serialize.py``
# constant verbatim, and read from module globals at call time for its reason
# too: a test lowers it rather than building a gigabyte.
MAX_DECOMPRESSED_BYTES = 1 << 30


# --- properties ---------------------------------------------------------------


def _props_json(props: dict[str, Prop]) -> dict[str, Any]:
    return {
        name: {"type": props[name].type, "value": props[name].value}
        for name in sorted(props)
    }


def _terrains_from(entry: Any) -> tuple[TerrainSpec, ...]:
    """The declared terrains of a tileset, in the order they were written.

    Order is precedence, so this reads a *list* and keeps it -- which is why the
    writer emits one rather than a dict whose keys the manifest's ``sort_keys``
    would reorder into a different map.
    """
    if not isinstance(entry, list):
        return ()
    out: list[TerrainSpec] = []
    for item in entry:
        if not isinstance(item, dict):
            continue
        out.append(
            TerrainSpec(
                name=str(item.get("name", "")),
                fill=tuple(int(part) for part in item.get("fill", (0, 0, 0, 255))),
                outline=tuple(int(part) for part in item.get("outline", (0, 0, 0, 255))),
            )
        )
    return tuple(out)


def _props_from(entry: Any) -> dict[str, Prop]:
    out: dict[str, Prop] = {}
    for name, record in (entry or {}).items():
        if not isinstance(record, dict):
            raise ValueError(f"property {name!r} is malformed")
        # ``Prop`` refuses an unknown type by name, so there is one list of
        # legal types and this is not a second copy of it.
        out[str(name)] = Prop(type=str(record.get("type", "string")), value=record.get("value"))
    return out


# --- writing ------------------------------------------------------------------


def _npy_bytes(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.lib.format.write_array(out, np.ascontiguousarray(array, dtype=gidlib.DTYPE))
    return out.getvalue()


def manifest_json(doc: MapDoc) -> str:
    """``map.json``'s text: sorted keys, indented, one entry per layer.

    Sorted and indented rather than compact because this half exists to be
    *read* -- the arrays are the reason the format is a zip, and there is no
    size argument left for minifying the small half.
    """
    tilesets = []
    for index, ref in enumerate(doc.tilesets):
        ts = ref.tileset
        tilesets.append(
            {
                "name": ts.name,
                "firstgid": int(ref.firstgid),
                "source": ref.source,
                "tile_w": ts.tile_w,
                "tile_h": ts.tile_h,
                "spacing": ts.spacing,
                "margin": ts.margin,
                "image": f"{TILESET_DIR}/{index}.png",
                "properties": _props_json(ts.properties),
                # A *list*, because a terrain's position is its precedence and
                # ``sort_keys`` below would shuffle the ranks of a dict. Written
                # even when empty, so the shape does not depend on the content.
                "terrains": [
                    {"name": entry.name, "fill": list(entry.fill), "outline": list(entry.outline)}
                    for entry in ts.terrains
                ],
            }
        )

    layers: list[dict[str, Any]] = []
    tile_index = 0
    for layer in doc.layers:
        common = {
            "name": layer.name,
            "visible": bool(layer.visible),
            "opacity": float(layer.opacity),
            "properties": _props_json(layer.properties),
        }
        if isinstance(layer, TileLayer):
            layers.append(
                {**common, "type": "tile", "data": f"{LAYER_DIR}/{tile_index}.npy"}
            )
            tile_index += 1
        else:
            layers.append(
                {
                    **common,
                    "type": "object",
                    "objects": [
                        {
                            "name": obj.name,
                            "kind": obj.kind,
                            "x": float(obj.x),
                            "y": float(obj.y),
                            "w": float(obj.w),
                            "h": float(obj.h),
                            "class": obj.obj_class,
                            "visible": bool(obj.visible),
                            "properties": _props_json(obj.properties),
                        }
                        for obj in layer.objects
                    ],
                }
            )

    payload = {
        "version": VERSION,
        "width": doc.width,
        "height": doc.height,
        "tile_w": doc.tile_w,
        "tile_h": doc.tile_h,
        "projection": doc.projection,
        "renderorder": doc.renderorder,
        "backgroundcolor": doc.backgroundcolor,
        "properties": _props_json(doc.properties),
        "tilesets": tilesets,
        "layers": layers,
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def wmap_bytes(doc: MapDoc) -> bytes:
    """The document as the bytes of a ``.wmap`` archive.

    Two saves of an unchanged document are byte-identical, which is what the
    fixed timestamps and the sorted manifest are for.
    """
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # First member, so a reader can identify the file without a full
        # directory scan and a human running ``unzip -p`` gets the readable half.
        zf.writestr(zipfile.ZipInfo(MANIFEST, _EPOCH), manifest_json(doc))
        for index, layer in enumerate(doc.tile_layers()):
            zf.writestr(
                zipfile.ZipInfo(f"{LAYER_DIR}/{index}.npy", _EPOCH),
                _npy_bytes(layer.data),
            )
        for index, ref in enumerate(doc.tilesets):
            # Stored, not deflated: a PNG is already compressed, and deflating
            # it again spends time to make it marginally bigger.
            zf.writestr(
                zipfile.ZipInfo(f"{TILESET_DIR}/{index}.png", _EPOCH),
                png_bytes(ref.tileset.pixels),
                zipfile.ZIP_STORED,
            )
    return out.getvalue()


# --- reading ------------------------------------------------------------------


def _member(zf: zipfile.ZipFile, name: str, what: str) -> bytes:
    try:
        return zf.read(name)
    except KeyError as exc:
        raise ValueError(f"this map names {what} the file does not carry ({name})") from exc


def _read_layer_array(raw: bytes, width: int, height: int, name: str) -> np.ndarray:
    try:
        array = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
    except (ValueError, EOFError) as exc:
        raise ValueError(f"layer {name!r} is unreadable: {exc}") from exc
    if array.dtype != gidlib.DTYPE:
        raise ValueError(
            f"layer {name!r} is stored as {array.dtype}, not {np.dtype(gidlib.DTYPE)}"
        )
    if array.shape != (height, width):
        raise ValueError(
            f"layer {name!r} is {array.shape[1]}x{array.shape[0]}, "
            f"but the map is {width}x{height}"
        )
    # A copy: ``read_array`` may hand back something backed by the buffer we
    # read, and a document whose layers alias a temporary is a document whose
    # next stamp writes into freed memory.
    return np.ascontiguousarray(array).copy()


def read_wmap(data: bytes) -> MapDoc:
    """A ``.wmap``'s bytes back into a :class:`~.tilemap.MapDoc`.

    The returned document has an empty history and reads clean: a file that has
    just been opened is not unsaved, and the layers are placed directly rather
    than through the mutators, which would push a step apiece.
    """
    from PIL import Image

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("this is not a Warlock map document") from exc

    # Every refusal is inside the ``with``, the two above included: raised in
    # the gap between the open and the block, they left the archive open with
    # nothing to close it but the collector. Same fix, same reason, as
    # ``clay/serialize.read_wblk``.
    with zf:
        # Before anything is read: the directory says what every member unpacks
        # to, and the cheapest place to refuse an archive that claims more than
        # this build will hold is before the first ``read``.
        claimed = sum(int(info.file_size) for info in zf.infolist())
        ceiling = MAX_DECOMPRESSED_BYTES
        if claimed > ceiling:
            raise ValueError(
                f"this map document claims {claimed} bytes unpacked, "
                f"past the {ceiling} this build will read"
            )

        try:
            manifest = json.loads(zf.read(MANIFEST))
        except (
            zipfile.BadZipFile,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("this is not a Warlock map document") from exc
        if not isinstance(manifest, dict):
            raise ValueError("this map's manifest is malformed")

        version = int(manifest.get("version", 0))
        if version > VERSION:
            raise ValueError(
                f"this map was written by a newer version of Warlock "
                f"(format {version}, this build reads {VERSION})"
            )

        doc = MapDoc(
            width=int(manifest.get("width", 1)),
            height=int(manifest.get("height", 1)),
            tile_w=int(manifest.get("tile_w", 1)),
            tile_h=int(manifest.get("tile_h", 1)),
            # A version 1 file predates projections and is orthogonal by
            # definition, which is what the default says without a branch.
            projection=str(manifest.get("projection", project.ORTHOGONAL)),
        )
        doc.renderorder = str(manifest.get("renderorder", "right-down"))
        doc.backgroundcolor = manifest.get("backgroundcolor")
        doc.properties = _props_from(manifest.get("properties"))

        previous = 0
        for entry in manifest.get("tilesets", []):
            firstgid = int(entry.get("firstgid", 1))
            if firstgid <= previous:
                # Contiguity is what ``resolve`` walks; a list that does not
                # increase means two tilesets claim one id and every cell in
                # the overlap draws whichever comes first.
                raise ValueError("this map's tilesets are numbered out of order")
            previous = firstgid
            name = str(entry.get("image", ""))
            image = Image.open(io.BytesIO(_member(zf, name, "a tileset image"))).convert(
                "RGBA"
            )
            doc.tilesets.append(
                TilesetRef(
                    firstgid=firstgid,
                    tileset=Tileset(
                        name=str(entry.get("name", "tileset")),
                        pixels=np.asarray(image, dtype=np.uint8),
                        tile_w=int(entry.get("tile_w", 1)),
                        tile_h=int(entry.get("tile_h", 1)),
                        spacing=int(entry.get("spacing", 0)),
                        margin=int(entry.get("margin", 0)),
                        properties=_props_from(entry.get("properties")),
                        terrains=_terrains_from(entry.get("terrains")),
                    ),
                    source=str(entry.get("source", "")),
                )
            )

        for entry in manifest.get("layers", []):
            kind = str(entry.get("type", ""))
            name = str(entry.get("name", ""))
            common = {
                "uid": new_uid(),
                "name": name,
                "visible": bool(entry.get("visible", True)),
                "opacity": float(entry.get("opacity", 1.0)),
                "properties": _props_from(entry.get("properties")),
            }
            if kind == "tile":
                member = str(entry.get("data", ""))
                doc.layers.append(
                    TileLayer(
                        **common,
                        data=_read_layer_array(
                            _member(zf, member, "a layer"), doc.width, doc.height, name
                        ),
                    )
                )
            elif kind == "object":
                doc.layers.append(
                    ObjectLayer(
                        **common,
                        objects=[_read_object(o) for o in entry.get("objects", [])],
                    )
                )
            else:
                raise ValueError(f"this map holds a layer of unknown kind {kind!r}")

    _validate(doc)
    doc.active_layer = doc.layers[0].uid if doc.layers else None
    doc.history.clear()
    doc.saved_head = doc.history.head
    return doc


def _read_object(entry: Any) -> MapObject:
    if not isinstance(entry, dict):
        raise ValueError("this map holds a malformed object")
    kind = str(entry.get("kind", "rect"))
    if kind not in OBJECT_KINDS:
        raise ValueError(f"this map holds an object of unknown kind {kind!r}")
    return MapObject(
        uid=new_uid(),
        name=str(entry.get("name", "")),
        kind=kind,
        x=float(entry.get("x", 0.0)),
        y=float(entry.get("y", 0.0)),
        w=float(entry.get("w", 0.0)),
        h=float(entry.get("h", 0.0)),
        obj_class=str(entry.get("class", "")),
        visible=bool(entry.get("visible", True)),
        properties=_props_from(entry.get("properties")),
    )


def _validate(doc: MapDoc) -> None:
    """Every nonzero gid resolves, or the file is refused.

    The same check :mod:`.tmx` runs, and for the same reason: a cell nothing
    accounts for draws as nothing and cannot be repainted with what it was,
    because nothing knows.
    """
    for layer in doc.tile_layers():
        for tile_id in np.unique(gidlib.tile_ids(layer.data)).tolist():
            if tile_id and doc.ref_for(tile_id) is None:
                raise ValueError(
                    f"layer {layer.name!r} uses tile {tile_id}, which none of this "
                    "map's tilesets accounts for"
                )
