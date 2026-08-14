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

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from . import blob
from .props import (
    PROPERTY_TYPES,
    Prop,
    TiledUnsupported,
    read_properties,
    write_properties,
)
from .tileset import TerrainSpec, Tileset

# What this build writes. Tiled accepts anything it recognises; these are the
# values a current Tiled writes and so the values least likely to surprise it.
TSX_VERSION = "1.10"
TILED_VERSION = "1.10.2"

#: Re-exported, not merely imported -- see the module docstring. Listed so a
#: linter reads the names as public rather than as unused imports.
__all__ = [
    "PROPERTY_TYPES",
    "TILED_VERSION",
    "TSX_VERSION",
    "Prop",
    "TiledUnsupported",
    "check_tileset_features",
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

    **A DTD is refused before the parser sees it.** ``ExpatParser`` expands
    internal entities, so the billion-laughs shape -- ten nested entities each
    referencing the previous one ten times -- turns a few hundred bytes of file
    into gigabytes of string inside ``fromstring``, and no ceiling downstream
    ever gets a turn. Tiled writes no DTD, so nothing legitimate is lost.

    The probe is a substring of the first 4 KiB rather than a parse, because the
    declaration is part of the *prolog*: XML requires it before the root element,
    a prolog is a version declaration, optional comments and processing
    instructions, and 4 KiB is far past any of those and far short of the
    document body. Uppercased, since ``<!doctype`` is equally legal.
    """
    if b"<!DOCTYPE" in data[:4096].upper():
        raise ValueError("this file declares a DTD, which Plotter does not read")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"this is not a readable Tiled {expect} file: {exc}") from exc
    if root.tag != expect:
        raise ValueError(f"expected a <{expect}> document, found <{root.tag}>")
    return root


def check_tileset_features(root: ET.Element) -> None:
    """Refuse everything a ``<tileset>`` element may hold that this cannot draw.

    Public because :mod:`.tmx` applies it to an *embedded* tileset -- the same
    element without a file around it -- and a second copy of the list is how the
    embedded and external paths come to accept different files.
    """
    node = root.find("wangsets")
    if node is not None and read_wangsets(node) is None:
        raise TiledUnsupported(
            "Wang sets / terrain brushes",
            f"Plotter models one blob set: {blob.TILE_COUNT} tiles per terrain colour, "
            "in mask order",
        )
    if root.find("terraintypes") is not None:
        raise TiledUnsupported("terrain types")
    if root.find("image") is None:
        raise TiledUnsupported(
            "an image-collection tileset",
            "every tile is its own file; Plotter needs one sliced atlas",
        )
    for tile in root.findall("tile"):
        where = f"tile {tile.get('id', '?')}"
        if tile.find("animation") is not None:
            raise TiledUnsupported("per-tile animation", where)
        if tile.find("image") is not None:
            raise TiledUnsupported("an image-collection tileset", where)
        if tile.find("objectgroup") is not None:
            raise TiledUnsupported("per-tile collision shapes", where)
        if tile.find("properties") is not None:
            raise TiledUnsupported("per-tile custom properties", where)


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


def _expected_wangids(colours: int) -> dict[int, str]:
    """``{tile id: wangid}`` for the one set this editor models.

    Extracted so the XML and JSON readers ask the *same* question: what is
    recognised is precisely what :func:`write_wangsets` emits, and two copies of
    that table is how one spelling comes to adopt a set the other refuses.
    """
    want: dict[int, str] = {}
    for index in range(colours):
        for case, mask in enumerate(blob.BLOB_MASKS):
            want[index * blob.TILE_COUNT + case] = ",".join(
                str(index + 1 if mask & bit else 0) for bit in _WANG_BITS
            )
    return want


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


def read_wangsets_json(entries: Any) -> tuple[TerrainSpec, ...] | None:
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
    tiles = wangset.get("wangtiles")
    if not isinstance(tiles, list) or len(tiles) != len(colours) * blob.TILE_COUNT:
        return None

    want = _expected_wangids(len(colours))
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


def read_wangsets(node: ET.Element) -> tuple[TerrainSpec, ...] | None:
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
    tiles = wangset.findall("wangtile")
    if len(tiles) != len(colours) * blob.TILE_COUNT:
        return None

    want = _expected_wangids(len(colours))
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


def write_wangsets(parent: ET.Element, terrains: tuple[TerrainSpec, ...]) -> None:
    """Describe a terrain set as a Tiled Wang set.

    Derived from ``terrains`` on every write rather than stored, so it cannot
    drift from the atlas it describes -- and a generated set opened in Tiled
    arrives with a working terrain brush rather than as 235 anonymous tiles.
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
    for index in range(len(terrains)):
        for case, mask in enumerate(blob.BLOB_MASKS):
            ET.SubElement(
                wangset,
                "wangtile",
                {
                    "tileid": str(index * blob.TILE_COUNT + case),
                    "wangid": ",".join(
                        str(index + 1 if mask & bit else 0) for bit in _WANG_BITS
                    ),
                },
            )


def _terrains_of(root: ET.Element) -> tuple[TerrainSpec, ...]:
    node = root.find("wangsets")
    return () if node is None else (read_wangsets(node) or ())


def read_tsx(data: bytes, image: np.ndarray) -> Tileset:
    """A ``.tsx``'s bytes plus its decoded image, as a :class:`Tileset`."""
    root = xml_root(data, "tileset")
    check_tileset_features(root)
    return Tileset(
        name=root.get("name") or "tileset",
        pixels=image,
        tile_w=int(root.get("tilewidth", 0) or 0),
        tile_h=int(root.get("tileheight", 0) or 0),
        spacing=int(root.get("spacing", 0) or 0),
        margin=int(root.get("margin", 0) or 0),
        properties=read_properties(root),
        terrains=_terrains_of(root),
    )


# --- writing ------------------------------------------------------------------


def tsx_element(ts: Tileset, *, image_name: str) -> ET.Element:
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
    # Tiled omits both when they are zero; matching that keeps our output
    # diff-clean against a file Tiled wrote for the same tileset.
    if ts.spacing:
        root.set("spacing", str(ts.spacing))
    if ts.margin:
        root.set("margin", str(ts.margin))
    root.set("tilecount", str(ts.tile_count))
    root.set("columns", str(ts.columns))
    ET.SubElement(
        root,
        "image",
        {
            "source": image_name,
            "width": str(ts.image_w),
            "height": str(ts.image_h),
        },
    )
    write_properties(root, ts.properties)
    write_wangsets(root, ts.terrains)
    return root


def to_bytes(root: ET.Element) -> bytes:
    """One element tree as the bytes of a file, indented and declared.

    Shared by both writers so a ``.tmx`` and a ``.tsx`` are formatted the same
    way, and so the declaration is spelled once.
    """
    ET.indent(root, space=" ")
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}\n'.encode()


def tsx_bytes(ts: Tileset, *, image_name: str) -> bytes:
    """A standalone ``.tsx`` file."""
    return to_bytes(tsx_element(ts, image_name=image_name))
