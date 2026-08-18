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

**The file stores positions and Tiled ids, never process uids.** A uid is
minted per process and means nothing in a file; what a document needs back is
*which layer* an entry is, and that is its position. Every layer and object
therefore gets a *fresh* uid on read, which is also why this format needs no
``reserve_uid`` step of the kind ``.wblk`` has: ``.wblk`` restores the uids it
stored and so must raise the process floor past them, whereas nothing here is
ever restored onto a number the counter has already issued. The persistent
``id`` is the opposite and is stored verbatim -- it is an ordinary document
field with its own monotone counters, and it is what an object-typed custom
property references.

**Tileset images are embedded**, which is the house pattern (``.wblk`` embeds
its textures, ``.ora`` its layers) and buys two things: the file is the whole
document, so it can be moved or sent without a folder of dependencies, and the
reader can validate at the door rather than discovering a missing PNG halfway
through. The accepted cost, stated rather than hidden: editing the source PNG on
disk does not propagate into a map already saved. A ``.tmx`` export is the way
out to an external, editable tileset. An image *layer*'s picture is embedded the
same way and for the same reasons, as ``images/N.png``.

**Version 3 is the layer tree.** The document models what Tiled models -- a tree
of tile/object/group/image layers, each carrying a class, a tint, a pixel offset
and a parallax factor; objects carrying a persistent id, a rotation and one of
the tagged geometries -- and this is the version that stores all of it. The
manifest's ``layers`` became *recursive* (a group entry carries a ``layers``
list of its own) while the tile arrays stayed a **flat, depth-first**
``layers/N.npy`` enumeration, so the half of the container that holds megabytes
did not churn for a change to the half that holds kilobytes.

**Version 4 is the 1.12-era additions**, and it is written only when one of them
is actually present: an object layer's ``color``, an object's ``opacity``, a
layer ``blend_mode``, the ``capsule`` geometry (which takes the count to eight),
a ``list``-typed custom property, an oblique projection with its skew, the
stagger/hex-side fields, a map or tileset class, a parallax origin, and a
tileset's grid and transformations. :func:`_document_version` is the one place
that decides, and **every one of those fields has to be named in it** -- the
entries are written unconditionally, so a field the gate does not ask about
produces a file declaring version 3 while carrying a key a version 3 reader
drops in silence.

**Version 5 is phase variants**: a terrain tileset whose ``phases`` is 2 or 4
carries ``phases**2`` sub-rows per terrain, and the field is gated because it
is *not* additive -- an old reader that dropped it would divide every local id
by the wrong row count and misattribute the terrain of every painted cell. A
map whose tilesets are all ``phases == 1`` keeps writing its old version, so
existing documents re-save byte-stable.

Versions 1 and 2 are still read, through tolerant defaults rather than a branch
per version -- the ``locked`` precedent. A version 1 file predates projections
and is orthogonal by definition; a version 2 file has no tint, offset, parallax
or class and those are identity; neither stores an ``id``, so both get fresh
persistent ids minted from the document's own counters on read, exactly as
:func:`.tmx._export_ids` assigns one to an object that has none. The writer uses
version 3 until a 1.12-era field needs version 4; this is one schema with
additive defaults, not two independent writers.

**A half-read document is worse than a refused one.** A file from a newer
version, a layer whose array is the wrong shape or dtype, a tileset naming a
member the archive does not carry, a gid no tileset accounts for, tilesets whose
firstgids do not increase: each is refused, because each of them opens as a map
that looks nearly right and can be saved back over the original.
"""

from __future__ import annotations

import dataclasses
import io
import itertools
import json
import zipfile
from typing import Any, NoReturn

import numpy as np

from ..tilegrid import gid as gidlib
from ..tilegrid.tileset import TerrainSpec, Tileset, TilesetRef
from . import project
from .pngio import png_bytes
from .props import read_wmap_properties, write_wmap_properties
from .tilemap import (
    OBJECT_KINDS,
    OPAQUE_WHITE,
    SHAPE_KINDS,
    GroupLayer,
    ImageLayer,
    Layer,
    MapDoc,
    MapObject,
    ObjectLayer,
    Shape,
    TileLayer,
    TileShape,
    new_uid,
    shape_for_kind,
    shape_kind,
)

VERSION = 5
#: The 1.12-era additions (see the module docstring); written when one of them
#: is present and no tileset carries phase variants.
TILED_ERA_VERSION = 4
BASE_VERSION = 3
MANIFEST = "map.json"
LAYER_DIR = "layers"
IMAGE_DIR = "images"
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

#: Every shape by the name it answers to, which is the reverse of
#: :data:`~._map_model.SHAPE_KINDS` and the only thing this codec needs: the
#: writer asks a shape for its name and the reader asks a name for its class.
_SHAPE_BY_KIND: dict[str, type] = {name: cls for cls, name in SHAPE_KINDS}


# --- tilesets -----------------------------------------------------------------
#
# Custom properties are :mod:`.props`' job, in both directions -- this format
# used to carry a third copy of the property codec and so a third opinion about
# what a property may be.


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


# --- writing ------------------------------------------------------------------


def _npy_bytes(array: np.ndarray) -> bytes:
    out = io.BytesIO()
    np.lib.format.write_array(out, np.ascontiguousarray(array, dtype=gidlib.DTYPE))
    return out.getvalue()


class WmapUnstorable(ValueError):
    """Something the document models that this format version cannot write down.

    A *named* exception rather than a bare ``ValueError``, and the name is the
    whole point: the studio's save path has to turn a writer-door refusal into a
    toast instead of letting it take the frame thread down, and a handler catching
    ``ValueError`` would catch far more than the door -- a numpy shape complaint,
    a bad float, any genuine defect in the encoder -- and dress every one of them
    up as a polite refusal the user is meant to act on. ``TiledUnsupported``
    already exists for exactly this reason at the other door; this is its
    counterpart for ours.

    A ``ValueError`` subclass, so callers that only care that the encode was
    refused keep working unchanged.

    **Version 3 stores everything version 2 refused**, so the tree, the image
    layers and the six per-layer decorations no longer raise this -- but the
    class, and the two handlers that name it
    (``studio/plotter_io._encoded``, ``studio/plotter_mode.export_library``),
    stay exactly where they are. The remaining raise is the unknown-layer-kind
    fallthrough below, and the milestones ahead put more behind it: a document
    that becomes infinite (chunked storage, M5) or gains a layer kind before the
    container learns to hold it lands here first. Removing the plumbing in order
    to re-add it next wave is churn, and the intervening builds would crash the
    frame thread rather than toast.
    """


# --- the manifest -------------------------------------------------------------


def _shape_record(shape: Shape) -> dict[str, Any]:
    """One object's geometry, as a tagged record: its kind plus its own fields.

    Enumerated off the dataclass rather than written out per shape, which is
    the same choice ``tests/plotter/_semantics._geometry_facts`` makes and right
    for the same reason: the union has eight members and will gain more, and a
    hand-written branch per shape is a field silently dropped per shape. A
    polygon's vertices, a tile object's gid and a text object's dozen styling
    fields all travel because they are *fields*, not because this function
    knows about them.

    The kind is stored, never inferred: ``Rect`` and ``Ellipse`` carry the same
    two numbers and a reader that guessed from the shape of the record would
    turn every ellipse into a rectangle.
    """
    record: dict[str, Any] = {"kind": shape_kind(shape)}
    for entry in dataclasses.fields(shape):
        record[entry.name] = getattr(shape, entry.name)
    return record


def _object_entry(obj: MapObject) -> dict[str, Any]:
    """One object. ``kind``/``w``/``h`` are *not* written beside the shape.

    Version 2 wrote those three and nothing else, because they were the whole
    of an object's geometry. They are derived properties now
    (:class:`~._map_model.MapObject` computes them from ``shape``), and writing
    a derived echo beside the thing it is derived from is two spellings of one
    fact in a file -- the exact arrangement that lets a reader restore an
    object whose stated size disagrees with its own geometry. The reader still
    *understands* the old spelling, which is what makes a version 2 file
    readable; it simply prefers the shape when one is there.
    """
    return {
        "id": int(obj.id),
        "name": obj.name,
        "x": float(obj.x),
        "y": float(obj.y),
        "rotation": float(obj.rotation),
        "opacity": float(obj.opacity),
        "shape": _shape_record(obj.shape),
        "class": obj.obj_class,
        "visible": bool(obj.visible),
        "properties": write_wmap_properties(obj.properties),
    }


def _layer_entries(
    layers: list[Layer], tiles: Any, images: Any
) -> list[dict[str, Any]]:
    """The manifest's ``layers``, recursively, in paint order.

    ``tiles`` and ``images`` are counters shared with :func:`wmap_bytes`, and
    sharing them is the point: the member a layer *names* and the member the
    archive *carries* have to be the same string, and the cheapest way to
    guarantee that is for both sides to number in one order. That order is
    depth-first pre-order -- a group's entry, then its children, then the
    layer beside it -- which is what :meth:`~._map_layers.LayerOps.walk`
    yields and therefore what ``tile_layers()`` and ``all_layers()`` hand back.
    ``tests/plotter/test_wmap.py`` pins the two sides against each other.
    """
    out: list[dict[str, Any]] = []
    for layer in layers:
        # The kind decides first, and the shared fields are read after, so a
        # layer this version has no entry for is refused *before* anything is
        # asked of it. The other order refuses too, eventually, but by way of
        # an ``AttributeError`` on whichever decoration the new kind happens to
        # be missing -- a defect's traceback where a writer-door refusal
        # belongs.
        if isinstance(layer, TileLayer):
            # ``chunks`` is reserved beside ``data`` and deliberately absent:
            # an infinite map stores a tile layer as a sparse list of chunks
            # rather than as one dense rectangle (M5). Named here and at the
            # top level so that the day it arrives it is an addition to an
            # entry rather than a rearrangement of the container.
            specific = {"type": "tile", "data": f"{LAYER_DIR}/{next(tiles)}.npy"}
        elif isinstance(layer, ObjectLayer):
            specific = {
                "type": "object",
                "draworder": str(layer.draworder),
                "color": layer.color,
                "objects": [_object_entry(obj) for obj in layer.objects],
            }
        elif isinstance(layer, ImageLayer):
            specific = {
                "type": "image",
                "image": f"{IMAGE_DIR}/{next(images)}.png",
                # Carried verbatim and never resolved -- where the picture came
                # from is the host's problem, exactly as a ``file`` property's
                # path is. The pixels are in the archive either way, so a
                # source that no longer exists costs nothing.
                "source": str(layer.source),
                "repeat": [bool(layer.repeat_x), bool(layer.repeat_y)],
            }
        elif isinstance(layer, GroupLayer):
            specific = {
                "type": "group",
                "layers": _layer_entries(layer.children, tiles, images),
            }
        else:
            # The fifth layer kind, arriving before this format can hold it.
            # Named rather than written as something else: a group is the only
            # kind whose *contents* another kind could be flattened into, and
            # flattening is the silent half-write every refusal here exists to
            # prevent.
            raise WmapUnstorable(
                f"layer {getattr(layer, 'name', '') or 'unnamed'!r} is a "
                f"{type(layer).__name__}, which the version {VERSION} .wmap "
                f"format has no entry for"
            )
        out.append(
            {
                # Tiled's persistent id, stored verbatim. ``0`` is "never
                # assigned one" and reads back as a freshly minted id rather
                # than as a zero, which is how a version 2 file joins the
                # scheme.
                "id": int(layer.id),
                "name": layer.name,
                # ``class`` in the file, ``class_name`` on the model: the model
                # cannot spell it ``class`` because that is a Python keyword,
                # and the file has no such problem and every reason to match
                # the word Tiled uses for the same field.
                "class": str(layer.class_name),
                "blend_mode": str(layer.blend_mode),
                "visible": bool(layer.visible),
                "opacity": float(layer.opacity),
                # Written unconditionally, unlike the TMX side. This is our own
                # format with one reader, and a key that is sometimes absent is
                # a key every future reader has to have an opinion about.
                "locked": bool(layer.locked),
                "tint": [int(part) for part in layer.tint],
                # Pairs rather than four scalar keys, because each of these is
                # one value with two components -- an offset with an ``x`` and
                # no ``y`` is not a document state that exists.
                "offset": [float(layer.offset_x), float(layer.offset_y)],
                "parallax": [float(layer.parallax_x), float(layer.parallax_y)],
                "properties": write_wmap_properties(layer.properties),
                **specific,
            }
        )
    return out


def _has_list_property(props: Any) -> bool:
    """Whether any property in this block is (or contains) a ``list``.

    Recursive, because a list can sit inside a class member or inside another
    list. ``list`` is one of the 1.12-era property types, so a document
    carrying one has to declare version 4 for the same reason an object opacity
    does: the entry is written either way, and a reader that predates the type
    would take the record apart into something else.
    """
    values = props.values() if isinstance(props, dict) else props
    for prop in values or ():
        kind = getattr(prop, "type", None)
        if kind == "list":
            return True
        if kind in ("class", "list") and _has_list_property(getattr(prop, "value", None)):
            return True
    return False


def _document_version(doc: MapDoc) -> int:
    """The oldest version emitted by this writer that represents ``doc``.

    Version 3 already stores the layer tree, decorations and seven original
    shapes. Version 4 is used only when one of the Tiled 1.12 additions would
    otherwise be lost.

    **Every field the version 4 writer added has to be named here.** The entries
    are written unconditionally -- this function chooses only the number at the
    top of the manifest -- so a field the gate does not ask about is a document
    declared version 3 while carrying a key a version 3 reader drops on the
    floor, silently and with nothing in the file to say it happened.
    ``ObjectLayer.color`` was exactly that: the one decoration on the object
    arm below that nothing looked at.
    """
    if doc.infinite:
        raise WmapUnstorable(
            f"the version {VERSION} .wmap format has no sparse chunk entry for an infinite map"
        )
    # Phase variants first: 5 is the ceiling, so nothing below can override it.
    # The gate is mandatory rather than tolerant -- ``phases`` is written
    # unconditionally, and an old reader that dropped it would divide every
    # local id by the wrong row count and misattribute the terrain of every
    # painted cell, silently.
    if any(ref.tileset.phases > 1 for ref in doc.tilesets):
        return VERSION
    if (
        doc.class_name
        or tuple(doc.parallax_origin) != (0.0, 0.0)
        or doc.projection == project.OBLIQUE
        or doc.skew_x
        or doc.skew_y
        or doc.stagger_axis != "y"
        or doc.stagger_index != "odd"
        or doc.hex_side
    ):
        return TILED_ERA_VERSION
    if any(
        ref.tileset.class_name
        or (
            ref.tileset.grid_orientation,
            ref.tileset.grid_width,
            ref.tileset.grid_height,
        )
        != ("orthogonal", ref.tileset.tile_w, ref.tileset.tile_h)
        or any(ref.tileset.transformations)
        # The same question the map, layer and object arms ask of theirs:
        # ``manifest_json`` writes each tileset's properties unconditionally,
        # and a ``.tsx`` import is a door a list walks in through.
        or _has_list_property(ref.tileset.properties)
        for ref in doc.tilesets
    ):
        return TILED_ERA_VERSION
    if _has_list_property(doc.properties):
        return TILED_ERA_VERSION
    for layer in doc.all_layers():
        if not isinstance(layer, (TileLayer, ObjectLayer, ImageLayer, GroupLayer)):
            continue
        if layer.blend_mode != "normal" or _has_list_property(layer.properties):
            return TILED_ERA_VERSION
        if isinstance(layer, ObjectLayer):
            # ``color`` before the objects, because it is the layer's own field
            # and an empty object layer still carries it.
            if layer.color:
                return TILED_ERA_VERSION
            for obj in layer.objects:
                if obj.opacity != 1.0 or shape_kind(obj.shape) == "capsule":
                    return TILED_ERA_VERSION
                if _has_list_property(obj.properties):
                    return TILED_ERA_VERSION
    return BASE_VERSION


def manifest_json(doc: MapDoc) -> str:
    """``map.json``'s text: sorted keys, indented, one entry per layer.

    Sorted and indented rather than compact because this half exists to be
    *read* -- the arrays are the reason the format is a zip, and there is no
    size argument left for minifying the small half. ``sort_keys`` reaches
    every level, nested layer entries and class properties included, which is
    what keeps a dict's insertion order out of the file.
    """
    tilesets = []
    for index, ref in enumerate(doc.tilesets):
        ts = ref.tileset
        tilesets.append(
            {
                "name": ts.name,
                "class": ts.class_name,
                "grid": [ts.grid_orientation, ts.grid_width, ts.grid_height],
                "transformations": list(ts.transformations),
                "firstgid": int(ref.firstgid),
                "source": ref.source,
                "tile_w": ts.tile_w,
                "tile_h": ts.tile_h,
                "spacing": ts.spacing,
                "margin": ts.margin,
                "image": f"{TILESET_DIR}/{index}.png",
                "properties": write_wmap_properties(ts.properties),
                # A *list*, because a terrain's position is its precedence and
                # ``sort_keys`` below would shuffle the ranks of a dict. Written
                # even when empty, so the shape does not depend on the content.
                "terrains": [
                    {"name": entry.name, "fill": list(entry.fill), "outline": list(entry.outline)}
                    for entry in ts.terrains
                ],
                # Written unconditionally like every key here; the version gate
                # above is what keeps an old reader from ever seeing k > 1.
                "phases": int(ts.phases),
            }
        )

    payload = {
        "version": _document_version(doc),
        "width": doc.width,
        "height": doc.height,
        "tile_w": doc.tile_w,
        "tile_h": doc.tile_h,
        # Reserved for M5 and written now, false, on purpose. An infinite map
        # has no fixed ``width``/``height`` rectangle at all: its tile layers
        # are sparse ``chunks`` (see the per-layer note in ``_layer_entries``),
        # and the two keys have to be *reserved* rather than invented later, or
        # every reader between now and then treats their absence as a fact.
        # ``read_wmap`` refuses a file that sets it, for the same reason it
        # refuses a newer version: this build cannot draw one.
        "infinite": False,
        "projection": doc.projection,
        "class": str(doc.class_name),
        "parallax_origin": [float(value) for value in doc.parallax_origin],
        "skew": [int(doc.skew_x), int(doc.skew_y)],
        "stagger": [doc.stagger_axis, doc.stagger_index, int(doc.hex_side)],
        "renderorder": doc.renderorder,
        "backgroundcolor": doc.backgroundcolor,
        # Document state, and the reason a re-save after an add-then-undo
        # differs from the prior bytes in these two fields alone: the counters
        # are monotone and never decremented, because an object-typed property
        # may go on naming an id whose object was deleted. Tiled behaves
        # identically.
        "next_layer_id": int(doc.next_layer_id),
        "next_object_id": int(doc.next_object_id),
        "properties": write_wmap_properties(doc.properties),
        "tilesets": tilesets,
        "layers": _layer_entries(doc.layers, itertools.count(), itertools.count()),
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def wmap_bytes(doc: MapDoc) -> bytes:
    """The document as the bytes of a ``.wmap`` archive.

    Two saves of an unchanged document are byte-identical, which is what the
    fixed timestamps and the sorted manifest are for.

    The tile arrays and the image-layer pictures are enumerated **flat and
    depth-first**, matching the numbering :func:`_layer_entries` wrote into the
    manifest: the tree lives in the small half of the file, and the half that
    holds megabytes stayed exactly the sequence of members it was in version 2.
    """
    # The manifest first, and outside the archive: it is the one part of the
    # encode that can refuse (an unknown layer kind), and a refusal raised with
    # nothing half-written behind it is ``read_wmap``'s "every refusal is
    # inside the ``with``" rule from the other side.
    manifest = manifest_json(doc)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        # First member, so a reader can identify the file without a full
        # directory scan and a human running ``unzip -p`` gets the readable half.
        zf.writestr(zipfile.ZipInfo(MANIFEST, _EPOCH), manifest)
        for index, layer in enumerate(doc.tile_layers()):
            zf.writestr(
                zipfile.ZipInfo(f"{LAYER_DIR}/{index}.npy", _EPOCH),
                _npy_bytes(layer.data),
            )
        for index, image in enumerate(_image_layers(doc)):
            # Stored, not deflated, for the tileset PNG's reason below.
            zf.writestr(
                zipfile.ZipInfo(f"{IMAGE_DIR}/{index}.png", _EPOCH),
                png_bytes(image.pixels),
                zipfile.ZIP_STORED,
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


def _image_layers(doc: MapDoc) -> list[ImageLayer]:
    """The image leaves, depth-first, in paint order.

    ``tile_layers()``' twin, and not a method beside it because the document
    has no other caller for one: the renderers reach an image layer through
    :func:`.scene.resolve`, which yields every drawable leaf at once.
    """
    return [layer for layer in doc.all_layers() if isinstance(layer, ImageLayer)]


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


def _read_picture(zf: zipfile.ZipFile, name: str, what: str) -> np.ndarray:
    from PIL import Image

    image = Image.open(io.BytesIO(_member(zf, name, what))).convert("RGBA")
    return np.asarray(image, dtype=np.uint8)


def _two(entry: dict[str, Any], key: str, default: Any) -> Any:
    """One two-component layer field, or its default. The tolerance rule.

    Absent is the version 2 case and takes the identity default -- the
    ``locked`` precedent. *Present and malformed* is not the same thing and is
    refused: a version 2 file says nothing about an offset, whereas a file
    carrying ``"offset": [3]`` says something this reader cannot honour, and
    guessing at the missing half is the half-read the format refuses.
    """
    if key not in entry:
        return default
    raw = entry[key]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        _malformed(f"a layer's {key}")
    return raw


def _pair(entry: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    """``offset`` and ``parallax``: two floats."""
    raw = _two(entry, key, default)
    return (float(raw[0]), float(raw[1]))


def _three(entry: dict[str, Any], key: str, default: Any) -> Any:
    """One three-component metadata field, or its backwards-compatible default."""
    if key not in entry:
        return default
    raw = entry[key]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        _malformed(f"a tileset's {key}")
    return raw


def _flags(entry: dict[str, Any], key: str) -> tuple[bool, bool]:
    """``repeat``: two booleans, which is a different type and so its own door.

    Its own function rather than ``bool(...)`` over :func:`_pair` at the call
    site, because the two spellings do not agree on what a value *is*:
    ``float("")`` raises where ``bool("")`` is ``False``, so routing a flag pair
    through the float reader would refuse some malformed files and silently
    accept others.
    """
    raw = _two(entry, key, (False, False))
    return (bool(raw[0]), bool(raw[1]))


def _malformed(what: str) -> NoReturn:
    """Typed ``NoReturn`` so a caller cannot mistake it for a value producer.

    It was reached as ``return _malformed(...)``, which reads as a fallback and
    is not one -- and a checker had no way to tell the difference.
    """
    raise ValueError(f"this map holds {what} that is malformed")


def _read_layers(
    entries: Any, zf: zipfile.ZipFile, doc: MapDoc
) -> list[Layer]:
    """One list of manifest layer entries, as layers. Recursive through groups.

    The depth this will descend is bounded, but *not* by the JSON parser having
    already survived the same nesting: one frame here is far larger than one
    frame of the scanner, so a manifest that parsed can still exhaust the stack
    on the way back out. :func:`read_wmap` catches that at the call site and
    turns it into the same ``ValueError`` every other unreadable manifest gets
    -- :func:`_read_shape`'s re-raise, one level up.
    """
    if not isinstance(entries, list):
        _malformed("a layer list")
    out: list[Layer] = []
    for entry in entries:
        if not isinstance(entry, dict):
            _malformed("a layer")
        kind = str(entry.get("type", ""))
        name = str(entry.get("name", ""))
        common: dict[str, Any] = {
            "uid": new_uid(),
            # Zero means "this file never stored one" -- every version 1 and 2
            # file, and any version 3 layer written before it was minted one --
            # and ``_assign_ids`` below turns it into a real one once the whole
            # tree is visible at once.
            "id": int(entry.get("id", 0) or 0),
            "name": name,
            "visible": bool(entry.get("visible", True)),
            "opacity": float(entry.get("opacity", 1.0)),
            # Tolerant, so a version 2 file written before locks existed
            # opens unlocked rather than being refused.
            "locked": bool(entry.get("locked", False)),
            "class_name": str(entry.get("class", "")),
            "blend_mode": str(entry.get("blend_mode", "normal")),
            # Straight to the constructor, which runs it through
            # ``rgba_colour`` and refuses a colour that is not one by name.
            "tint": entry.get("tint", OPAQUE_WHITE),
            "properties": read_wmap_properties(entry.get("properties")),
        }
        common["offset_x"], common["offset_y"] = _pair(entry, "offset", (0.0, 0.0))
        common["parallax_x"], common["parallax_y"] = _pair(entry, "parallax", (1.0, 1.0))
        if kind == "tile":
            member = str(entry.get("data", ""))
            out.append(
                TileLayer(
                    **common,
                    data=_read_layer_array(
                        _member(zf, member, "a layer"), doc.width, doc.height, name
                    ),
                )
            )
        elif kind == "object":
            out.append(
                ObjectLayer(
                    **common,
                    # Refused by ``ObjectLayer`` itself when it is not one of
                    # the two, so there is one list of legal draw orders.
                    draworder=str(entry.get("draworder", "topdown")),
                    color=entry.get("color"),
                    objects=[_read_object(o) for o in entry.get("objects", [])],
                )
            )
        elif kind == "image":
            repeat_x, repeat_y = _flags(entry, "repeat")
            out.append(
                ImageLayer(
                    **common,
                    pixels=_read_picture(
                        zf, str(entry.get("image", "")), "an image layer's picture"
                    ),
                    source=str(entry.get("source", "")),
                    repeat_x=repeat_x,
                    repeat_y=repeat_y,
                )
            )
        elif kind == "group":
            out.append(
                GroupLayer(**common, children=_read_layers(entry.get("layers", []), zf, doc))
            )
        else:
            raise ValueError(f"this map holds a layer of unknown kind {kind!r}")
    return out


def _read_shape(entry: dict[str, Any]) -> Shape:
    """One object's geometry, from either spelling.

    A version 3 record is a tagged ``shape`` block and is built by *name*: the
    kind picks the class and the class's own fields pick which keys are read,
    so a field the record does not carry takes the shape's default rather than
    being refused, and a key the shape has no field for is ignored rather than
    exploding. Versions 1 and 2 had no such block and stored ``kind``/``w``/``h``
    instead, which :func:`~._map_model.shape_for_kind` turns into the ``Rect``
    or ``Point`` that was the whole of what those versions could hold.
    """
    record = entry.get("shape")
    if not isinstance(record, dict):
        kind = str(entry.get("kind", "rect"))
        if kind not in OBJECT_KINDS:
            raise ValueError(f"this map holds an object of unknown kind {kind!r}")
        return shape_for_kind(kind, entry.get("w", 0.0), entry.get("h", 0.0))
    kind = str(record.get("kind", ""))
    cls = _SHAPE_BY_KIND.get(kind)
    if cls is None:
        raise ValueError(f"this map holds an object of unknown kind {kind!r}")
    names = {field.name for field in dataclasses.fields(cls)}
    try:
        return cls(**{key: value for key, value in record.items() if key in names})
    except (TypeError, ValueError) as exc:
        # ``ValueError`` is already the shapes' own refusal (a negative extent,
        # a polygon of two points); ``TypeError`` is what a corrupt record
        # produces -- a ``points`` that is a string, a ``gid`` that is a list --
        # and it would otherwise leave this reader through a door that is not
        # ``ValueError``, which every caller of it is written against.
        raise ValueError(f"this map holds a malformed {kind} object: {exc}") from exc


def _read_object(entry: Any) -> MapObject:
    if not isinstance(entry, dict):
        raise ValueError("this map holds a malformed object")
    return MapObject(
        uid=new_uid(),
        id=int(entry.get("id", 0) or 0),
        name=str(entry.get("name", "")),
        x=float(entry.get("x", 0.0)),
        y=float(entry.get("y", 0.0)),
        rotation=float(entry.get("rotation", 0.0)),
        opacity=float(entry.get("opacity", 1.0)),
        shape=_read_shape(entry),
        obj_class=str(entry.get("class", "")),
        visible=bool(entry.get("visible", True)),
        properties=read_wmap_properties(entry.get("properties")),
    )


def _assign_ids(doc: MapDoc, manifest: dict[str, Any]) -> None:
    """Every layer and object holds a persistent id, and the counters clear them.

    A stored id (nonzero) is kept as-is -- it is what an object-typed property
    references, and rewriting it would break every reference into the file that
    carried it. A zero is a version 1 or 2 entry, which had no such field, and
    it is minted one *here*, from the document's own counters, sequentially in
    paint order. :func:`.tmx._export_ids` does exactly this at the export door
    and for exactly this reason; the two rules have to agree or a map that
    travelled out through Tiled and back would come home with different ids
    than the same map saved and reopened.

    The counters then advance past everything actually in use, whatever the
    file claimed: a hand-edited manifest whose ``next_object_id`` sits below an
    id it also carries would otherwise reissue that number to the next object
    the user drew.

    **Minting skips what is already taken**, which is the same defect one step
    earlier and not covered by the advance above. A file understating its
    counter *while also mixing* stored and absent ids -- ``next_object_id: 1``
    over objects wearing ``0`` and ``1`` -- would hand the zero the number the
    other one already has, and two objects sharing an id is exactly what an
    object-typed property cannot survive: the reference resolves to whichever
    is found first. Skipping rather than renumbering, because the stored id is
    the one something outside this file may already name.
    """
    doc.next_layer_id = max(1, int(manifest.get("next_layer_id", 1)))
    doc.next_object_id = max(1, int(manifest.get("next_object_id", 1)))
    layers = doc.all_layers()
    objects = [
        obj for layer in layers if isinstance(layer, ObjectLayer) for obj in layer.objects
    ]
    layer_ids = _minter(doc.next_layer_id, {layer.id for layer in layers if layer.id})
    object_ids = _minter(doc.next_object_id, {obj.id for obj in objects if obj.id})
    for layer in layers:
        if not layer.id:
            layer.id = next(layer_ids)
    for obj in objects:
        if not obj.id:
            obj.id = next(object_ids)
    doc.next_layer_id = max([doc.next_layer_id, *(layer.id + 1 for layer in layers)])
    doc.next_object_id = max([doc.next_object_id, *(obj.id + 1 for obj in objects)])


def _minter(start: int, taken: set[int]) -> Any:
    """Ids from ``start`` upward, never one this file already spends."""
    for value in itertools.count(start):
        if value not in taken:
            taken.add(value)
            yield value


def read_wmap(data: bytes) -> MapDoc:
    """A ``.wmap``'s bytes back into a :class:`~.tilemap.MapDoc`.

    Versions 1, 2 and 3 all open here, through tolerant defaults rather than a
    reader per version; see the module docstring. The returned document has an
    empty history and reads clean: a file that has just been opened is not
    unsaved, and the layers are placed directly rather than through the
    mutators, which would push a step apiece.
    """
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
            # The manifest's ``layers`` is a *tree* since version 3, so a file
            # can now be nested past what the parser will descend. That is the
            # parser refusing to build the object, not a document this reader
            # half-understood, so it belongs with the other "this is not a map"
            # answers rather than escaping as a bare ``RecursionError``.
            RecursionError,
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
        if manifest.get("infinite", False):
            # The key is reserved and written ``false``; a file that sets it
            # stores its tile layers as sparse chunks and has no dense
            # rectangle for ``_read_layer_array`` to check against, so reading
            # one as if it were dense is the half-read this format refuses.
            # M5 turns this into an acceptance, the way isometric turned the
            # projection refusal into one.
            raise ValueError(
                "this map is infinite, which this build cannot open "
                "(chunked tile storage is not implemented yet)"
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
        doc.class_name = str(manifest.get("class", ""))
        doc.parallax_origin = _pair(manifest, "parallax_origin", (0.0, 0.0))
        skew = _two(manifest, "skew", (0, 0))
        doc.skew_x, doc.skew_y = int(skew[0]), int(skew[1])
        stagger = _three(manifest, "stagger", ("y", "odd", 0))
        doc.stagger_axis = str(stagger[0])
        doc.stagger_index = str(stagger[1])
        doc.hex_side = int(stagger[2])
        doc.properties = read_wmap_properties(manifest.get("properties"))

        previous = 0
        for entry in manifest.get("tilesets", []):
            firstgid = int(entry.get("firstgid", 1))
            if firstgid <= previous:
                # Contiguity is what ``resolve`` walks; a list that does not
                # increase means two tilesets claim one id and every cell in
                # the overlap draws whichever comes first.
                raise ValueError("this map's tilesets are numbered out of order")
            previous = firstgid
            grid = _three(
                entry,
                "grid",
                ("orthogonal", int(entry.get("tile_w", 1)), int(entry.get("tile_h", 1))),
            )
            transformations = entry.get("transformations", [False] * 4)
            if not isinstance(transformations, (list, tuple)) or len(transformations) != 4:
                _malformed("a tileset's transformations")
            doc.tilesets.append(
                TilesetRef(
                    firstgid=firstgid,
                    tileset=Tileset(
                        name=str(entry.get("name", "tileset")),
                        class_name=str(entry.get("class", "")),
                        grid_orientation=str(grid[0]),
                        grid_width=int(grid[1]),
                        grid_height=int(grid[2]),
                        transformations=tuple(bool(value) for value in transformations),
                        pixels=_read_picture(
                            zf, str(entry.get("image", "")), "a tileset image"
                        ),
                        tile_w=int(entry.get("tile_w", 1)),
                        tile_h=int(entry.get("tile_h", 1)),
                        spacing=int(entry.get("spacing", 0)),
                        margin=int(entry.get("margin", 0)),
                        properties=read_wmap_properties(entry.get("properties")),
                        terrains=_terrains_from(entry.get("terrains")),
                        # Absent in every file before version 5, where 1 is
                        # what the absence meant.
                        phases=int(entry.get("phases", 1) or 1),
                    ),
                    source=str(entry.get("source", "")),
                )
            )

        try:
            doc.layers.extend(_read_layers(manifest.get("layers", []), zf, doc))
        except RecursionError as exc:
            # A tree deep enough to exhaust the stack on the way *out* of the
            # parser, which is a file this reader cannot hold rather than a
            # defect in it -- and it has to leave through ``ValueError`` like
            # every other refusal, because that is the door the studio's open
            # path is written against.
            raise ValueError(
                "this map's layers are nested deeper than this build can read"
            ) from exc

    _assign_ids(doc, manifest)
    _validate(doc)
    doc.active_layer = doc.layers[0].uid if doc.layers else None
    doc.history.clear()
    doc.saved_head = doc.history.head
    return doc


# Tiled's hexagonal 120-degree rotation flag, probed by name here for the same
# reason ``tmx._finish`` probes it at the other door (see the note in
# :mod:`.gid` for why the constant lives at the doors and not beside the other
# three): the masks in :mod:`.gid` deliberately span bit 28, so a hand-edited
# file carrying it read as a tile id 268435456 too large and was refused as "a
# tile no tileset accounts for" -- a true sentence about the wrong problem.
_HEX_ROTATE = gidlib.DTYPE(0x10000000)


def _validate(doc: MapDoc) -> None:
    """Every nonzero gid resolves, or the file is refused.

    The same check :mod:`.tmx` runs, and for the same reason: a cell nothing
    accounts for draws as nothing and cannot be repainted with what it was,
    because nothing knows. A tile *object* carries a gid in exactly the same
    encoding -- flags in the top three bits -- so it is checked in the same
    pass: a dangling gid is the same unreadable file wherever it sits. And in
    both halves the hexagonal-rotation bit is refused *by name* first, exactly
    as :func:`.tmx._finish` refuses it, rather than being left to fail as an
    out-of-range id.
    """
    for layer in doc.tile_layers():
        if bool((np.asarray(layer.data) & _HEX_ROTATE).any()):
            raise ValueError(
                f"layer {layer.name!r} uses hexagonal 120-degree tile rotation, "
                "which Plotter does not support"
            )
        for tile_id in np.unique(gidlib.tile_ids(layer.data)).tolist():
            if tile_id and doc.ref_for(tile_id) is None:
                raise ValueError(
                    f"layer {layer.name!r} uses tile {tile_id}, which none of this "
                    "map's tilesets accounts for"
                )
    for layer in doc.all_layers():
        if not isinstance(layer, ObjectLayer):
            continue
        for obj in layer.objects:
            if not isinstance(obj.shape, TileShape):
                continue
            if int(obj.shape.gid) & int(_HEX_ROTATE):
                raise ValueError(
                    f"object {obj.name!r} uses hexagonal 120-degree tile "
                    "rotation, which Plotter does not support"
                )
            tile_id = int(gidlib.decompose(obj.shape.gid)[0])
            if tile_id and doc.ref_for(tile_id) is None:
                raise ValueError(
                    f"object {obj.name!r} uses tile {tile_id}, which none of this "
                    "map's tilesets accounts for"
                )
