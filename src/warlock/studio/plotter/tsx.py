"""``.tsx`` -- Tiled's external tileset, read and written.

This is the **only** reader and the only writer of the format in the repo:
Packwright's grid packer emits a ``.tsx`` beside its atlas, and it does so by
building a :class:`~.tileset.Tileset` and calling :func:`tsx_bytes` rather than
assembling XML of its own. A second writer of a published format is how one
version number comes to mean two subtly different documents.

**Loading is split in two so this module stays pure.** A ``.tsx`` names its
image by relative path, and resolving that path means touching a filesystem.
:func:`tsx_source` answers "what image does this file want" and the caller
fetches it; :func:`read_tsx` takes the bytes and the decoded pixels together.

**An unsupported feature is refused by name.** Tiled's format is far larger than
what an orthogonal stamp-and-fill editor models, and the alternative to refusing
is loading a file, silently dropping half of it, and writing that back over the
user's work. :class:`TiledUnsupported` is a ``ValueError`` subclass so a caller
that only wants "this did not load" needs no new except clause, and it carries
the feature's name so the message can say which one.

:class:`Prop`, :class:`TiledUnsupported` and the XML property codec used to
live here -- one property model per syntax, three of them, each deciding for
itself what a property may be. They live in :mod:`.props` now, the package's
leaf, and are **re-exported from this module**: ``tmx``, ``wmap``, the panes
and the tests all say ``from .tsx import Prop``, and the move was about having
one model rather than about moving where anybody looks for it.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from .. import xmlguard
from ..tilegrid import blob
from ..tilegrid.tileset import (
    TerrainSpec,
    TileEllipse,
    TileFrame,
    TileMeta,
    TilePolygon,
    TileRect,
    Tileset,
    compose_collection,
)
from ..tilegrid.wang import WANG_KINDS, WangColour, WangSet
from .props import (
    PROPERTY_TYPES,
    Prop,
    TiledUnsupported,
    json_number,
    read_json_properties,
    read_properties,
    write_properties,
)

log = logging.getLogger(__name__)

# What this build writes. Tiled accepts anything it recognises; these are the
# values a current Tiled writes and so the values least likely to surprise it.
#
# ``TILED_VERSION`` was held *behind* the target in ``docs/COMPAT.md`` for as
# long as the claim it makes was unproven: it says a real Tiled of that version
# opens what we write, and nothing in this repository can establish that -- a
# fixture we authored ourselves proves a round trip against ourselves. It was
# bumped once, in the same change that deleted the sentence saying not to, and
# that bump was reverted because the gate was never satisfied.
#
# **The gate was satisfied on 2026-08-29** and the constant moved with it. A
# Plotter-exported map (a 150x150 orthogonal three-layer map with an external
# ``tilesets/*.tsx``) was opened and worked on in Tiled 1.12.x, and a map that
# Tiled 1.12.2 itself wrote -- two external tilesets, one CSV layer, 640x360 --
# reads here without complaint. ``docs/COMPAT.md`` records both directions and
# what they do *not* cover, which is every construct past orthogonal CSV
# ground: flip bits, objects, properties, and the ``.tmj`` spelling.
TSX_VERSION = "1.10"
TILED_VERSION = "1.12.2"

#: Re-exported, not merely imported -- see the module docstring. Listed so a
#: linter reads the names as public rather than as unused imports.
__all__ = [
    "PROPERTY_TYPES",
    "TILED_VERSION",
    "TSX_VERSION",
    "Prop",
    "TiledUnsupported",
    "check_tileset_features",
    "phases_from_properties",
    "read_properties",
    "read_tsx",
    "read_wangsets",
    "read_wangsets_json",
    "to_bytes",
    "tsx_bytes",
    "tsx_element",
    "tsx_source",
    "write_properties",
    "write_wangsets",
    "xml_root",
]


# --- reading ------------------------------------------------------------------


def xml_root(data: bytes, expect: str) -> ET.Element:
    """The one XML door for both Tiled readers: parse, then check the root tag.

    Public and shared with :mod:`.tmx` rather than copied into it, because the
    two copies were byte-identical and the *refusal* below is the reason that
    matters: a second door is a door with no lock on it.

    **A DTD is refused, and the depth is capped, by the parser itself.** Both
    used to be one line here: a substring probe of the first 4 KiB, argued on
    the grounds that a DTD is part of the *prolog* and 4 KiB is far past any
    prolog. **Both halves of that were reproduced false on this machine.** A
    UTF-16 document spells the declaration ``<\\x00!\\x00D\\x00...``, so the
    byte substring never matches and expat parses the entities anyway; and a
    comment is legal prolog content, so five thousand bytes of ``<!-- -->`` put
    the declaration past byte 4096 in an otherwise ordinary file. The refusal is
    :mod:`..xmlguard`'s now, on the parser's own DOCTYPE event, which is a
    property of the parser object the way a bound is a property of
    ``zipguard.BoundedZip``.

    The depth cap that came with it is the half this door never had at all.
    ``ET.fromstring`` will build a 20,001-deep tree out of 300 KB, and the
    shallower case is the worse one: a document nested a few hundred deep opens
    and then blows the stack in ``scene.resolve`` and the layers pane, once a
    frame, on the frame thread.
    """
    root = xmlguard.fromstring(data, f"this Tiled {expect} file")
    if root.tag != expect:
        raise ValueError(f"expected a <{expect}> document, found <{root.tag}>")
    return root


def check_tileset_features(root: ET.Element) -> None:
    """Refuse everything a ``<tileset>`` element may hold that this cannot draw.

    Public because :mod:`.tmx` applies it to an *embedded* tileset -- the same
    element without a file around it -- and a second copy of the list is how the
    embedded and external paths come to accept different files.
    """
    # A Wang set that is *not* this editor's blob preset used to be refused
    # here. It is now read as data (:func:`read_wang_model`) and painted by
    # constraint matching, so the recognise-or-refuse asymmetry is over -- see
    # ``tilegrid/wang.py``. The preset is still recognised first and still keeps
    # its positional terrain rows and its byte-identical export.
    if root.find("terraintypes") is not None:
        raise TiledUnsupported("terrain types")
    # Object alignment, render size, fill mode, background colour and the tile
    # offset are all modelled now -- see the presentation block on
    # :class:`~..tilegrid.tileset.Tileset`. What remains refused is a tileset
    # with no single atlas at all.
    # A tileset with no top-level ``<image>`` is an image collection, and it is
    # modelled now (:class:`~..tilegrid.tileset.Collection`) rather than refused.
    # What the reader still needs is *pixels*, which only the host can fetch --
    # so the collection path goes through ``collection_sources`` and the loader,
    # and this check has nothing left to refuse.
    for tile in root.findall("tile"):
        where = f"tile {tile.get('id', '?')}"
        # Class, probability, animation, collision and custom properties are
        # all modelled now (:class:`~..tilegrid.tileset.TileMeta`); what remains
        # refused here is what this editor genuinely cannot hold.
        if tile.get("terrain") is not None:
            # Tiled's own pre-Wang terrain types, which Tiled itself is
            # retiring. A refusal is the honest state.
            raise TiledUnsupported("per-tile terrain assignment", where)


def check_tileset_features_json(entry: dict[str, Any]) -> None:
    """``check_tileset_features`` over Tiled's JSON tileset spelling.

    Covers the same two checks that read straight off the tileset object --
    ``terrains`` (JSON's pre-1.5 ``<terraintypes>``) and the four per-tile
    refusals -- so the JSON embedded-tileset path in :mod:`.tmx` refuses the
    same file the XML path does, by the same names. Wangsets and the
    top-level ``image`` question are decided elsewhere in that caller, each
    with its own JSON-specific recognise-or-refuse logic already in place, so
    this stays to the part the two formats read identically: one dict, not an
    element, but the same four keys per tile.
    """
    if entry.get("terrains"):
        raise TiledUnsupported("terrain types")
    for tile in entry.get("tiles") or ():
        if not isinstance(tile, dict):
            continue
        where = f"tile {tile.get('id', '?')}"
        if "terrain" in tile:
            raise TiledUnsupported("per-tile terrain assignment", where)


def tsx_source(data: bytes) -> str:
    """The image path a ``.tsx`` names, relative to the ``.tsx`` itself.

    Separate from :func:`read_tsx` so this module never opens a file: the
    caller resolves the path, reads and decodes it, and hands the pixels back.
    """
    root = xml_root(data, "tileset")
    check_tileset_features(root)
    image = root.find("image")
    source = (image.get("source") or "").strip() if image is not None else ""
    if not source:
        raise TiledUnsupported(
            "an embedded tileset image", "Plotter needs an <image source=...> path"
        )
    return source


#: The eight ``wangid`` slots, in Tiled's own order -- top, top-right, right,
#: bottom-right, bottom, bottom-left, left, top-left. It happens to be exactly
#: :data:`blob.NEIGHBOURS`' clockwise-from-north order, which is why translating
#: between the two is a list comprehension rather than a lookup table.
_WANG_BITS: tuple[int, ...] = (
    blob.N,
    blob.NE,
    blob.E,
    blob.SE,
    blob.S,
    blob.SW,
    blob.W,
    blob.NW,
)


def _hex_rgba(value: str) -> tuple[int, int, int, int]:
    text = (value or "").strip().lstrip("#")
    if len(text) == 8:  # Tiled writes #aarrggbb when an alpha is present
        text = text[2:] + text[:2]
    elif len(text) == 6:
        text = text + "ff"
    else:
        return (0, 0, 0, 255)
    try:
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4, 6))  # type: ignore[return-value]
    except ValueError:
        return (0, 0, 0, 255)


def _expected_wangids(colours: int, phases: int = 1) -> dict[int, str]:
    """``{tile id: wangid}`` for the one set this editor models.

    Extracted so the XML and JSON readers ask the *same* question: what is
    recognised is precisely what :func:`write_wangsets` emits, and two copies of
    that table is how one spelling comes to adopt a set the other refuses.

    Every one of a terrain's ``phases**2`` sub-rows carries the same wangid per
    case -- Tiled treats equal wangids as random alternatives, so its terrain
    brush keeps working on an exported phase set.
    """
    want: dict[int, str] = {}
    for index in range(colours):
        wangids = [
            ",".join(str(index + 1 if mask & bit else 0) for bit in _WANG_BITS)
            for mask in blob.BLOB_MASKS
        ]
        for sub in range(phases * phases):
            base = (index * phases * phases + sub) * blob.TILE_COUNT
            for case, wangid in enumerate(wangids):
                want[base + case] = wangid
    return want


def phases_from_properties(props: dict[str, Prop]) -> tuple[int, dict[str, Prop]]:
    """The declared phase count, popped out of a properties mapping.

    ``(phases, the mapping without the key)``. Absent means 1 -- every file
    written before phases existed. Malformed comes back as 0, which no reader
    recognises, so the tileset is refused rather than half-read. Popped rather
    than left in place, because the writer re-derives the property from the
    ``phases`` field on every export -- carrying both would double the key.
    """
    prop = props.get("phases")
    if prop is None:
        return 1, props
    rest = {key: value for key, value in props.items() if key != "phases"}
    try:
        return int(prop.value), rest
    except (TypeError, ValueError):
        return 0, rest


def _terrains_from_colours(colours: list[tuple[str, str]]) -> tuple[TerrainSpec, ...]:
    """``(name, #rrggbb)`` pairs as terrains, outline derived.

    The outline colour is **not** carried by the format and is derived on the
    way back in. Lossless where it matters: a terrain's two colours are read by
    the *generator* and by nothing else, since a tileset that already has pixels
    renders from them.
    """
    out = []
    for index, (name, colour) in enumerate(colours):
        fill = _hex_rgba(colour)
        outline = (*(part * 3 // 5 for part in fill[:3]), fill[3])
        out.append(
            TerrainSpec(name=name or f"Terrain {index + 1}", fill=fill, outline=outline)
        )
    return tuple(out)


def read_wangsets_json(entries: Any, phases: int = 1) -> tuple[TerrainSpec, ...] | None:
    """:func:`read_wangsets` over Tiled's JSON spelling of the same block.

    Same contract, same table, same ``None``-rather-than-exception rule -- one
    model, two syntaxes. The JSON schema names the parts differently (``colors``
    for ``<wangcolor>``, ``wangtiles`` for ``<wangtile>``, and a ``wangid`` that
    is a list of eight numbers rather than a comma-joined string), and
    reconciling those spellings is all this function is.
    """
    if not isinstance(entries, list) or len(entries) != 1:
        return None
    wangset = entries[0]
    if not isinstance(wangset, dict) or wangset.get("type") != "mixed":
        return None
    colours = wangset.get("colors")
    if not isinstance(colours, list) or not colours:
        return None
    if not all(isinstance(colour, dict) for colour in colours):
        return None
    if phases not in (1, 2, 4):
        return None
    tiles = wangset.get("wangtiles")
    expected = len(colours) * phases * phases * blob.TILE_COUNT
    if not isinstance(tiles, list) or len(tiles) != expected:
        return None

    want = _expected_wangids(len(colours), phases)
    for tile in tiles:
        if not isinstance(tile, dict) or not isinstance(tile.get("wangid"), list):
            return None
        try:
            tileid = int(tile["tileid"])
            wangid = ",".join(str(int(part)) for part in tile["wangid"])
        except (KeyError, TypeError, ValueError):
            return None
        if want.get(tileid) != wangid:
            return None

    return _terrains_from_colours(
        [(str(colour.get("name", "")), str(colour.get("color", ""))) for colour in colours]
    )


def read_wangsets(node: ET.Element, phases: int = 1) -> tuple[TerrainSpec, ...] | None:
    """A ``<wangsets>`` block as this editor's terrains, or ``None``.

    ``None`` rather than an exception, so the caller owns the refusal sentence
    and there is one place that says "Wang sets" to a user.

    **Recognise-or-refuse, and the asymmetry is the point.** Tiled's model is
    strictly larger than this one -- corner-only and edge-only sets, up to 255
    colours, arbitrary tile assignments that need not form a blob at all -- so
    adopting a foreign one would be exactly the silent half-read the reader
    exists to prevent. What is recognised is precisely what :func:`write_wangsets`
    emits, which keeps the reader and the writer symmetric: every file this
    writes, this reads.

    The outline colour is **not** carried by the format and is derived on the
    way back in. That is lossless where it matters: a terrain's two colours are
    read by the *generator* and by nothing else, since a tileset that already
    has pixels renders from them.
    """
    sets = node.findall("wangset")
    if len(sets) != 1:
        return None
    wangset = sets[0]
    if wangset.get("type") != "mixed":
        return None
    colours = wangset.findall("wangcolor")
    if not colours:
        return None
    if phases not in (1, 2, 4):
        return None
    tiles = wangset.findall("wangtile")
    if len(tiles) != len(colours) * phases * phases * blob.TILE_COUNT:
        return None

    want = _expected_wangids(len(colours), phases)
    for tile in tiles:
        try:
            tileid = int(tile.get("tileid", ""))
        except ValueError:
            return None
        if want.get(tileid) != (tile.get("wangid") or "").replace(" ", ""):
            return None

    return _terrains_from_colours(
        [(colour.get("name") or "", colour.get("color", "")) for colour in colours]
    )


def write_wangsets(
    parent: ET.Element, terrains: tuple[TerrainSpec, ...], phases: int = 1
) -> None:
    """Describe a terrain set as a Tiled Wang set.

    Derived from ``terrains`` on every write rather than stored, so it cannot
    drift from the atlas it describes -- and a generated set opened in Tiled
    arrives with a working terrain brush rather than as 235 anonymous tiles.

    A phase set writes every sub-row's tiles with the *same* wangid per case:
    Tiled treats equal wangids as random alternatives, so its brush still
    paints (randomising phases, which ``docs/COMPAT.md`` accepts),
    while exported *layers* carry concrete gids and travel exactly.
    """
    if not terrains:
        return
    node = ET.SubElement(parent, "wangsets")
    wangset = ET.SubElement(node, "wangset", {"name": "Terrain", "type": "mixed", "tile": "-1"})
    for entry in terrains:
        ET.SubElement(
            wangset,
            "wangcolor",
            {
                "name": entry.name,
                "color": "#{:02x}{:02x}{:02x}".format(*entry.fill[:3]),
                "tile": "-1",
                "probability": "1",
            },
        )
    for tileid, wangid in sorted(_expected_wangids(len(terrains), phases).items()):
        ET.SubElement(wangset, "wangtile", {"tileid": str(tileid), "wangid": wangid})


def _wang_model_of(
    root: ET.Element, terrains: tuple[TerrainSpec, ...]
) -> tuple[WangSet, ...]:
    """Foreign Wang sets, or nothing when the blob preset was recognised."""
    if terrains:
        return ()
    node = root.find("wangsets")
    return () if node is None else read_wang_model(node)



def read_wang_model(node: ET.Element) -> tuple[WangSet, ...]:
    """Every ``<wangset>`` under a ``<wangsets>``, as data.

    **Recognise-or-refuse ends here, deliberately.** Until now a Wang set was
    accepted only when it matched this editor's own blob table exactly and
    refused otherwise, which meant a Tiled user's corner set could not be opened
    at all. Now every set is read; :func:`read_wangsets` still runs first and
    still recognises the preset, so a set this editor generated keeps its
    positional terrain rows and its byte-identical export.

    A malformed wangid is skipped rather than refused: it is one tile of one set,
    and losing a whole map over it would be the half-read trade run backwards.
    """
    out: list[WangSet] = []
    for wangset in node.findall("wangset"):
        colours = [
            WangColour(
                name=colour.get("name") or "",
                colour=colour.get("color") or "#ffffff",
                probability=float(colour.get("probability", 1.0) or 1.0),
                # The representative tile and the custom class, read verbatim.
                # The writer emitted ``tile="-1"`` unconditionally and dropped
                # the class, so a foreign set came home with every swatch reset
                # and its class gone -- a silent edit to somebody's file, which
                # is the one thing an interop reader must not do.
                tile=_int_attr(colour, "tile", -1),
                klass=colour.get("class") or "",
            )
            for colour in wangset.findall("wangcolor")
        ]
        tiles: dict[int, tuple[int, ...]] = {}
        for tile in wangset.findall("wangtile"):
            try:
                local = int(tile.get("tileid", ""))
                slots = tuple(
                    int(part)
                    for part in (tile.get("wangid") or "").replace(" ", "").split(",")
                )
            except ValueError:
                continue
            if len(slots) != 8 or any(s < 0 or s > len(colours) for s in slots):
                continue
            tiles[local] = slots
        kind = wangset.get("type", "mixed")
        out.append(
            WangSet(
                name=wangset.get("name") or "Terrain",
                kind=kind if kind in WANG_KINDS else "mixed",
                colours=tuple(colours),
                tiles=tiles,
                tile=_int_attr(wangset, "tile", -1),
                klass=wangset.get("class") or "",
            )
        )
    return tuple(out)


def _int_attr(node: Any, name: str, default: int) -> int:
    """One integer attribute, or *default* if it is absent or junk."""
    try:
        return int(node.get(name, default))
    except (TypeError, ValueError):
        return default


def read_wang_model_json(entries: Any) -> tuple[WangSet, ...]:
    """:func:`read_wang_model` over Tiled's JSON spelling."""
    if not isinstance(entries, list):
        return ()
    out: list[WangSet] = []
    for wangset in entries:
        if not isinstance(wangset, dict):
            continue
        colours = [
            WangColour(
                name=str(colour.get("name") or ""),
                colour=str(colour.get("color") or "#ffffff"),
                probability=json_number(colour, "probability", 1.0),
                tile=int(colour.get("tile", -1) or -1),
                klass=str(colour.get("class") or ""),
            )
            for colour in (wangset.get("colors") or ())
            if isinstance(colour, dict)
        ]
        tiles: dict[int, tuple[int, ...]] = {}
        for tile in wangset.get("wangtiles") or ():
            if not isinstance(tile, dict):
                continue
            slots = tuple(int(part) for part in (tile.get("wangid") or ()))
            if len(slots) != 8 or any(s < 0 or s > len(colours) for s in slots):
                continue
            tiles[int(tile.get("tileid", 0) or 0)] = slots
        kind = str(wangset.get("type", "mixed"))
        out.append(
            WangSet(
                name=str(wangset.get("name") or "Terrain"),
                kind=kind if kind in WANG_KINDS else "mixed",
                colours=tuple(colours),
                tiles=tiles,
                tile=int(wangset.get("tile", -1) or -1),
                klass=str(wangset.get("class") or ""),
            )
        )
    return tuple(out)


def write_wang_model(parent: ET.Element, wangsets: tuple[WangSet, ...]) -> None:
    """Data-driven sets back out, sorted so two writes are byte identical.

    Only reached for a tileset that has no blob preset: :func:`write_wangsets`
    handles that one and its output must stay byte-identical, which is the
    determinism pin the whole exporter is held to.
    """
    if not wangsets:
        return
    node = ET.SubElement(parent, "wangsets")
    for wangset in wangsets:
        attrs = {
            "name": wangset.name,
            "type": wangset.kind,
            # What the file said, not ``-1``: this is the tile Tiled shows as
            # the set's representative, and rewriting it is an edit to
            # somebody's document made on the way through.
            "tile": str(int(wangset.tile)),
        }
        if wangset.klass:
            attrs["class"] = wangset.klass
        element = ET.SubElement(node, "wangset", attrs)
        for colour in wangset.colours:
            colour_attrs = {
                "name": colour.name,
                "color": colour.colour,
                "tile": str(int(colour.tile)),
                "probability": _number(colour.probability),
            }
            if colour.klass:
                colour_attrs["class"] = colour.klass
            ET.SubElement(element, "wangcolor", colour_attrs)
        for local in sorted(wangset.tiles):
            ET.SubElement(
                element,
                "wangtile",
                {
                    "tileid": str(local),
                    "wangid": ",".join(str(slot) for slot in wangset.tiles[local]),
                },
            )



def _terrains_of(root: ET.Element, phases: int = 1) -> tuple[TerrainSpec, ...]:
    node = root.find("wangsets")
    return () if node is None else (read_wangsets(node, phases) or ())


# --- per-tile metadata --------------------------------------------------------
#
# The plotter's own object shapes and tilegrid's collision records are converted
# into each other **here**, at the codec, and nowhere else: the shared leaf
# imports nothing under ``warlock``, so it cannot know what a ``Polygon`` is,
# and the plotter should not grow a second vocabulary for the same rectangle.


def _collision_from(group: ET.Element) -> tuple[Any, ...]:
    """A ``<objectgroup>`` inside a ``<tile>`` as tilegrid collision records.

    Only the three shapes the tilegrid records can hold. Anything else in the
    group -- a text object, a tile object -- is skipped rather than refused: it
    is not collision, and a file that carries one is not a file this editor
    cannot open. Two things Tiled's collision editor *can* author are dropped
    by name into the log rather than silently: a point or polyline member (the
    records have no shape for either), and a shape's ``rotation`` (the records
    carry none, so it would not survive to the export either way). Both are
    stated in ``docs/COMPAT.md`` beside the round-trip claim.
    """
    out: list[Any] = []
    for obj in group.findall("object"):
        x = float(obj.get("x", 0) or 0)
        y = float(obj.get("y", 0) or 0)
        w = float(obj.get("width", 0) or 0)
        h = float(obj.get("height", 0) or 0)
        if float(obj.get("rotation", 0) or 0):
            log.warning(
                "collision shape %s: rotation is not modeled and was dropped",
                obj.get("id", "?"),
            )
        polygon = obj.find("polygon")
        if polygon is not None:
            points = tuple(
                (float(px), float(py))
                for px, py in (
                    pair.split(",") for pair in (polygon.get("points") or "").split()
                )
            )
            out.append(TilePolygon(x=x, y=y, points=points))
        elif obj.find("ellipse") is not None:
            out.append(TileEllipse(x=x, y=y, w=w, h=h))
        elif obj.find("point") is not None or obj.find("polyline") is not None:
            log.warning(
                "collision shape %s: point and polyline outlines are not "
                "modeled and were dropped",
                obj.get("id", "?"),
            )
        else:
            out.append(TileRect(x=x, y=y, w=w, h=h))
    return tuple(out)


def _collision_element(parent: ET.Element, shapes: tuple[Any, ...]) -> None:
    group = ET.SubElement(parent, "objectgroup", {"draworder": "index", "id": "1"})
    for index, shape in enumerate(shapes, start=1):
        attrs = {"id": str(index), "x": _number(shape.x), "y": _number(shape.y)}
        if isinstance(shape, TilePolygon):
            obj = ET.SubElement(group, "object", attrs)
            ET.SubElement(
                obj,
                "polygon",
                {
                    "points": " ".join(
                        f"{_number(px)},{_number(py)}" for px, py in shape.points
                    )
                },
            )
            continue
        attrs["width"] = _number(shape.w)
        attrs["height"] = _number(shape.h)
        obj = ET.SubElement(group, "object", attrs)
        if isinstance(shape, TileEllipse):
            ET.SubElement(obj, "ellipse")


def _number(value: float) -> str:
    """Tiled's own spelling: an integral value has no decimal point.

    Not cosmetic -- it is what keeps our output diff-clean against a file Tiled
    wrote for the same tileset, which is the bar the whole writer is held to.
    """
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def colour_key(pixels: np.ndarray, colour: str | None) -> np.ndarray:
    """``pixels`` with every occurrence of ``colour`` made fully transparent.

    Tiled's ``trans``/``transparentcolor``: a tileset drawn on a solid backdrop
    says which colour that backdrop is rather than carrying an alpha channel.
    Applied **at decode**, so nothing downstream ever sees the key colour and no
    renderer needs to know a tileset had one.

    The *tileset* row only. Tiled's image-layer twin is deprecated and stays
    refused, which is the honest state for something Tiled is retiring.
    """
    if not colour:
        return pixels
    text = str(colour).lstrip("#")
    if len(text) == 8:  # #AARRGGBB -- the alpha is not part of the match
        text = text[2:]
    try:
        if len(text) != 6:
            raise ValueError(text)
        key = tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        # A plain ``ValueError`` and deliberately **not** ``TiledUnsupported``:
        # the feature is supported, and what is wrong is the *value*. A
        # refusal-by-name would put a name in the source that the compat ledger
        # has no refused row for, which is exactly the drift that gate exists to
        # catch. Framed rather than raised bare, so the message says what.
        raise ValueError(
            f"a tileset transparent colour is #RRGGBB, not {colour!r}"
        ) from exc
    out = np.array(pixels, dtype=np.uint8)
    if out.ndim != 3 or out.shape[2] != 4:
        return out
    matched = (
        (out[:, :, 0] == key[0]) & (out[:, :, 1] == key[1]) & (out[:, :, 2] == key[2])
    )
    out[matched, 3] = 0
    return out


def presentation_of(root: ET.Element) -> dict[str, Any]:
    """The five presentation fields off a ``<tileset>`` element."""
    offset = root.find("tileoffset")
    return {
        "offset_x": 0 if offset is None else int(offset.get("x", 0) or 0),
        "offset_y": 0 if offset is None else int(offset.get("y", 0) or 0),
        "object_alignment": root.get("objectalignment", "unspecified"),
        "render_size": root.get("tilerendersize", "tile"),
        "fill_mode": root.get("fillmode", "stretch"),
        "background": root.get("backgroundcolor") or None,
    }


def presentation_of_json(entry: dict[str, Any]) -> dict[str, Any]:
    """:func:`presentation_of` over Tiled's JSON tileset spelling."""
    offset = entry.get("tileoffset") or {}
    if not isinstance(offset, dict):
        offset = {}
    return {
        "offset_x": int(offset.get("x", 0) or 0),
        "offset_y": int(offset.get("y", 0) or 0),
        "object_alignment": str(entry.get("objectalignment", "unspecified")),
        "render_size": str(entry.get("tilerendersize", "tile")),
        "fill_mode": str(entry.get("fillmode", "stretch")),
        "background": entry.get("backgroundcolor") or None,
    }


def write_presentation(root: ET.Element, ts: Tileset) -> None:
    """The five fields back onto a ``<tileset>``, omitting every default.

    Omitting the defaults is not cosmetic: the writer is held to producing a
    file that diffs cleanly against one Tiled wrote for the same tileset, and
    Tiled omits them.
    """
    if ts.object_alignment != "unspecified":
        root.set("objectalignment", ts.object_alignment)
    if ts.render_size != "tile":
        root.set("tilerendersize", ts.render_size)
    if ts.fill_mode != "stretch":
        root.set("fillmode", ts.fill_mode)
    if ts.background:
        root.set("backgroundcolor", ts.background)
    if ts.offset_x or ts.offset_y:
        ET.SubElement(
            root, "tileoffset", {"x": str(ts.offset_x), "y": str(ts.offset_y)}
        )


def collection_sources(root: ET.Element) -> dict[int, str]:
    """``{local id: image path}`` for an image-collection ``<tileset>``.

    Empty for an ordinary sliced atlas, which is how every caller tells the two
    apart without asking twice. The paths are relative to the tileset file and
    are **not** resolved here: this package never touches a filesystem, so the
    host resolves and decodes them and hands the pixels back.
    """
    if root.find("image") is not None:
        return {}
    out: dict[int, str] = {}
    for tile in root.findall("tile"):
        image = tile.find("image")
        source = (image.get("source") or "").strip() if image is not None else ""
        if source:
            out[int(tile.get("id", 0) or 0)] = source
    return out


def collection_sources_json(entry: dict[str, Any]) -> dict[int, str]:
    """:func:`collection_sources` over Tiled's JSON tileset spelling."""
    if entry.get("image"):
        return {}
    out: dict[int, str] = {}
    for tile in entry.get("tiles") or ():
        if isinstance(tile, dict) and tile.get("image"):
            out[int(tile.get("id", 0) or 0)] = str(tile["image"])
    return out


def write_collection(root: ET.Element, ts: Tileset, names: dict[int, str]) -> None:
    """Per-tile ``<image>`` elements for a collection, by local id.

    ``names`` is what the host is writing each tile's pixels *as*; the codec
    cannot invent them for the reason it cannot read them.
    """
    if ts.collection is None:
        return
    by_id = {tile.get("id"): tile for tile in root.findall("tile")}
    for slot, local in enumerate(ts.collection.ids):
        source = names.get(local)
        if not source:
            continue
        tile = by_id.get(str(local))
        if tile is None:
            tile = ET.SubElement(root, "tile", {"id": str(local)})
        width, height = ts.collection.sizes[slot]
        ET.SubElement(
            tile,
            "image",
            {"source": source, "width": str(width), "height": str(height)},
        )



def read_tile_meta(root: ET.Element) -> dict[int, TileMeta]:
    """Every ``<tile>`` block's metadata, by local id. Sparse."""
    out: dict[int, TileMeta] = {}
    for tile in root.findall("tile"):
        local = int(tile.get("id", 0) or 0)
        animation = tile.find("animation")
        group = tile.find("objectgroup")
        meta = TileMeta(
            class_name=tile.get("class") or tile.get("type") or "",
            properties=read_properties(tile),
            probability=float(tile.get("probability", 1.0) or 1.0),
            animation=()
            if animation is None
            else tuple(
                TileFrame(
                    local_id=int(frame.get("tileid", 0) or 0),
                    duration_ms=int(frame.get("duration", 100) or 100),
                )
                for frame in animation.findall("frame")
            ),
            collision=() if group is None else _collision_from(group),
        )
        if not meta.is_empty:
            out[local] = meta
    return out


def write_tile_meta(root: ET.Element, tiles: dict[int, TileMeta]) -> None:
    """One ``<tile>`` block per tile that carries anything, in id order.

    Sorted, because the writer's determinism pin compares bytes: a dict's
    insertion order is a property of how the tileset was built, not of what it
    holds.
    """
    for local in sorted(tiles):
        meta = tiles[local]
        if meta.is_empty:
            continue
        tile = ET.SubElement(root, "tile", {"id": str(local)})
        if meta.class_name:
            tile.set("class", meta.class_name)
        if meta.probability != 1.0:
            tile.set("probability", _number(meta.probability))
        write_properties(tile, meta.properties)
        if meta.animation:
            animation = ET.SubElement(tile, "animation")
            for frame in meta.animation:
                ET.SubElement(
                    animation,
                    "frame",
                    {
                        "tileid": str(frame.local_id),
                        "duration": str(frame.duration_ms),
                    },
                )
        if meta.collision:
            _collision_element(tile, meta.collision)


def read_tile_meta_json(entry: dict[str, Any]) -> dict[int, TileMeta]:
    """:func:`read_tile_meta` over Tiled's JSON tileset spelling."""
    out: dict[int, TileMeta] = {}
    for tile in entry.get("tiles") or ():
        local = int(tile.get("id", 0) or 0)
        group = tile.get("objectgroup") or {}
        shapes: list[Any] = []
        for obj in group.get("objects") or ():
            x = float(obj.get("x", 0) or 0)
            y = float(obj.get("y", 0) or 0)
            if obj.get("polygon"):
                shapes.append(
                    TilePolygon(
                        x=x,
                        y=y,
                        points=tuple(
                            (float(point["x"]), float(point["y"]))
                            for point in obj["polygon"]
                        ),
                    )
                )
            elif obj.get("ellipse"):
                shapes.append(
                    TileEllipse(
                        x=x,
                        y=y,
                        w=float(obj.get("width", 0) or 0),
                        h=float(obj.get("height", 0) or 0),
                    )
                )
            elif not obj.get("point") and not obj.get("polyline"):
                shapes.append(
                    TileRect(
                        x=x,
                        y=y,
                        w=float(obj.get("width", 0) or 0),
                        h=float(obj.get("height", 0) or 0),
                    )
                )
        meta = TileMeta(
            class_name=tile.get("class") or tile.get("type") or "",
            properties=read_json_properties(tile.get("properties")),
            probability=json_number(tile, "probability", 1.0),
            animation=tuple(
                TileFrame(
                    local_id=int(json_number(frame, "tileid", 0)),
                    duration_ms=int(json_number(frame, "duration", 100)),
                )
                for frame in (tile.get("animation") or ())
            ),
            collision=tuple(shapes),
        )
        if not meta.is_empty:
            out[local] = meta
    return out


def read_tsx(data: bytes, image: Any) -> Tileset:
    """A ``.tsx``'s bytes plus its decoded image, as a :class:`Tileset`.

    ``image`` is the atlas for an ordinary tileset and a **mapping of local id
    to decoded pixels** for an image collection -- which is what
    :func:`collection_sources` told the host to fetch. One parameter rather than
    two because a tileset is one or the other and never both, and a second
    parameter would be a second thing every caller has to decide about.
    """
    return tileset_from_element(xml_root(data, "tileset"), image)


def tileset_from_element(root: ET.Element, image: Any) -> Tileset:
    """One XML ``<tileset>`` element, external or embedded, as a :class:`Tileset`.

    :func:`tileset_from_json`'s twin, and here for the same reason it is: the
    embedded spelling in :mod:`.tmx` had its own copy of this construction, and
    the copy had drifted. It dropped the five presentation fields, the foreign
    Wang model and the image's ``trans`` colour key -- all three of which the
    *JSON* twin of the same map read -- so one map saved both ways loaded as two
    different tilesets, and several `docs/COMPAT.md` rows were claims about this
    function rather than about Tiled (the 2026-09-02 review, section 7).

    ``image`` is the atlas, or a **mapping of local id to decoded pixels** for
    an image collection; resolving either means touching a filesystem, which
    this package deliberately cannot.
    """
    check_tileset_features(root)
    grid = root.find("grid")
    transformations = root.find("transformations")
    props = read_properties(root)
    declared, remaining = phases_from_properties(props)
    terrains = _terrains_of(root, declared)
    # The property is structural only on a recognised terrain set; anywhere
    # else "phases" is somebody's ordinary custom property and travels intact.
    phases = declared if terrains else 1
    picture = root.find("image")
    if isinstance(image, dict):
        atlas, collection = compose_collection(image)
    else:
        atlas, collection = (
            colour_key(image, None if picture is None else picture.get("trans")),
            None,
        )
    return Tileset(
        name=root.get("name") or "tileset",
        class_name=root.get("class") or root.get("type") or "",
        pixels=atlas,
        collection=collection,
        **presentation_of(root),
        tile_w=int(root.get("tilewidth", 0) or 0),
        tile_h=int(root.get("tileheight", 0) or 0),
        spacing=int(root.get("spacing", 0) or 0),
        margin=int(root.get("margin", 0) or 0),
        grid_orientation=(
            "orthogonal" if grid is None else grid.get("orientation", "orthogonal")
        ),
        grid_width=(
            int(root.get("tilewidth", 0) or 0)
            if grid is None
            else int(grid.get("width", root.get("tilewidth", 0)) or 0)
        ),
        grid_height=(
            int(root.get("tileheight", 0) or 0)
            if grid is None
            else int(grid.get("height", root.get("tileheight", 0)) or 0)
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
            transformations.get("preferuntransformed", "0") not in ("0", "false"),
        ),
        properties=remaining if terrains else props,
        terrains=terrains,
        phases=phases,
        tiles=read_tile_meta(root),
        # Only when the preset was *not* recognised: a blob set is already
        # modelled by ``terrains`` and holding it twice would let the two drift.
        wangsets=_wang_model_of(root, terrains),
    )


def read_tsj(data: bytes, image: np.ndarray) -> Tileset:
    """A ``.tsj``'s bytes plus its decoded image, as a :class:`Tileset`.

    :func:`read_tsx`'s JSON twin, and deliberately built out of the *same*
    pieces the embedded-JSON-tileset path in :mod:`.tmx` already uses --
    ``check_tileset_features_json``, ``read_json_properties``,
    ``read_wangsets_json``, ``read_tile_meta_json``, ``presentation_of_json``.
    A second opinion about what a JSON tileset means is exactly how the external
    and embedded spellings of one file come to load differently.
    """
    import json

    entry = json.loads(data.decode("utf-8"))
    if not isinstance(entry, dict):
        raise ValueError("a .tsj holds a tileset object")
    return tileset_from_json(entry, image)


def tsj_source(data: bytes) -> str:
    """The image path a ``.tsj`` names, relative to the ``.tsj`` itself."""
    import json

    entry = json.loads(data.decode("utf-8"))
    if not isinstance(entry, dict):
        raise ValueError("a .tsj holds a tileset object")
    source = str(entry.get("image", "") or "")
    if not source:
        # Only reached for a tileset with no per-tile images either -- the host
        # asks :func:`collection_sources_json` first and never gets here for a
        # real collection. So this is "no pixels at all", which is the same
        # sentence the XML side gives.
        raise TiledUnsupported(
            "an embedded tileset image", "Plotter needs an image path"
        )
    return source


def tileset_from_json(entry: dict[str, Any], image: np.ndarray) -> Tileset:
    """One JSON tileset object, external or embedded, as a :class:`Tileset`.

    The single definition both spellings go through. ``image`` is already
    decoded, because resolving a path means touching a filesystem and this
    package deliberately cannot.
    """
    check_tileset_features_json(entry)
    grid = entry.get("grid") or {}
    if not isinstance(grid, dict):
        raise ValueError("a tileset object grid is not an object")
    transformations = entry.get("transformations") or {}
    if not isinstance(transformations, dict):
        raise ValueError("tileset transformations are not an object")
    props = read_json_properties(entry.get("properties"))
    declared, remaining = phases_from_properties(props)
    wangsets = entry.get("wangsets")
    terrains = () if not wangsets else (read_wangsets_json(wangsets, declared) or ())
    if isinstance(image, dict):
        atlas, collection = compose_collection(image)
    else:
        atlas, collection = colour_key(image, entry.get("transparentcolor")), None
    return Tileset(
        name=str(entry.get("name", "tileset")),
        class_name=str(entry.get("class") or entry.get("type") or ""),
        pixels=atlas,
        collection=collection,
        tile_w=int(entry.get("tilewidth", 0) or 0),
        tile_h=int(entry.get("tileheight", 0) or 0),
        spacing=int(entry.get("spacing", 0) or 0),
        margin=int(entry.get("margin", 0) or 0),
        grid_orientation=str(grid.get("orientation", "orthogonal")),
        grid_width=int(grid.get("width", entry.get("tilewidth", 0)) or 0),
        grid_height=int(grid.get("height", entry.get("tileheight", 0)) or 0),
        transformations=(
            bool(transformations.get("hflip", False)),
            bool(transformations.get("vflip", False)),
            bool(transformations.get("rotate", False)),
            bool(transformations.get("preferuntransformed", False)),
        ),
        properties=remaining if terrains else props,
        terrains=terrains,
        phases=declared if terrains else 1,
        tiles=read_tile_meta_json(entry),
        wangsets=() if terrains else read_wang_model_json(wangsets),
        **presentation_of_json(entry),
    )



# --- writing ------------------------------------------------------------------


def tsx_element(
    ts: Tileset, *, image_name: str, collection_names: dict[int, str] | None = None
) -> ET.Element:
    """The ``<tileset>`` element, for embedding in a ``.tmx`` or writing alone."""
    root = ET.Element(
        "tileset",
        {
            "version": TSX_VERSION,
            "tiledversion": TILED_VERSION,
            "name": ts.name,
            "tilewidth": str(ts.tile_w),
            "tileheight": str(ts.tile_h),
        },
    )
    if ts.class_name:
        root.set("class", ts.class_name)
    write_presentation(root, ts)
    # Tiled omits both when they are zero; matching that keeps our output
    # diff-clean against a file Tiled wrote for the same tileset.
    if ts.spacing:
        root.set("spacing", str(ts.spacing))
    if ts.margin:
        root.set("margin", str(ts.margin))
    if (ts.grid_orientation, ts.grid_width, ts.grid_height) != (
        "orthogonal",
        ts.tile_w,
        ts.tile_h,
    ):
        ET.SubElement(
            root,
            "grid",
            {
                "orientation": ts.grid_orientation,
                "width": str(ts.grid_width),
                "height": str(ts.grid_height),
            },
        )
    if any(ts.transformations):
        hflip, vflip, rotate, prefer = ts.transformations
        ET.SubElement(
            root,
            "transformations",
            {
                "hflip": "1" if hflip else "0",
                "vflip": "1" if vflip else "0",
                "rotate": "1" if rotate else "0",
                "preferuntransformed": "1" if prefer else "0",
            },
        )
    root.set("tilecount", str(ts.tile_count))
    # Tiled writes ``columns="0"`` for a collection, because it has no atlas to
    # count across -- and a reader that saw a real number would slice one.
    root.set("columns", "0" if ts.is_collection else str(ts.columns))
    if not ts.is_collection:
        ET.SubElement(
            root,
            "image",
            {
                "source": image_name,
                "width": str(ts.image_w),
                "height": str(ts.image_h),
            },
        )
    props = ts.properties
    if ts.terrains and ts.phases > 1:
        # Re-derived from the field on every write -- the reader popped it out
        # of the stored properties, so this is the one spelling of the fact.
        props = {**props, "phases": Prop("int", ts.phases)}
    write_properties(root, props)
    if ts.terrains and ts.wangsets:
        # **The exclusion, asserted rather than assumed.** The two writers
        # below each return early only on *their own* emptiness, so a tileset
        # carrying both would emit two ``<wangsets>`` blocks -- which Tiled's
        # schema does not allow and which breaks this exporter's
        # byte-identical pin. It used to be true by construction because
        # nothing but ``_wang_model_of`` (which drops the foreign model
        # whenever the preset is present) ever produced the field; the Terrain
        # tab is a second producer, and it refuses the combination at its own
        # door for the same reason. This is the belt to that pair of braces,
        # and it can only fire on a document written by a build that had the
        # tab and not the refusal -- so it is a named refusal the save door
        # already toasts, not a crash.
        raise TiledUnsupported(
            "a tileset carrying both a generated terrain set and a hand-authored "
            "Wang set",
            "delete the hand-authored set in the tileset editor's Terrain tab",
            exporting=True,
        )
    write_wangsets(root, ts.terrains, ts.phases if ts.terrains else 1)
    # Mutually exclusive, by the refusal above and by the Terrain tab's --
    # so a tileset never writes two ``<wangsets>`` blocks and the blob preset's
    # output is byte-identical to what it always was.
    write_wang_model(root, ts.wangsets)
    # After the wang sets, matching Tiled's own element order -- the writer is
    # held to producing a file that diffs cleanly against one Tiled wrote.
    write_tile_meta(root, ts.tiles)
    # After the per-tile metadata, so a tile that carries both gets one element
    # holding both -- which is what Tiled writes and what a diff expects.
    write_collection(root, ts, collection_names or {})
    return root


def to_bytes(root: ET.Element) -> bytes:
    """One element tree as the bytes of a file, indented and declared.

    Shared by both writers so a ``.tmx`` and a ``.tsx`` are formatted the same
    way, and so the declaration is spelled once.
    """
    ET.indent(root, space=" ")
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}\n'.encode()


def tsx_bytes(
    ts: Tileset, *, image_name: str, collection_names: dict[int, str] | None = None
) -> bytes:
    """A standalone ``.tsx`` file.

    ``collection_names`` is what the host is writing each tile's pixels as, and
    is required for a collection: the codec cannot invent a filename for the
    same reason it cannot read one.
    """
    return to_bytes(
        tsx_element(ts, image_name=image_name, collection_names=collection_names)
    )
