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

:class:`Prop` and the two property codecs live here rather than in :mod:`.tmx`
because the dependency runs that way: a map reads external tilesets, so
``tmx`` imports ``tsx`` and not the reverse. Properties are typed explicitly
rather than inferred from the Python value, because ``color`` and ``string`` are
both ``str`` and a round trip that guessed would silently retype every colour a
user set in Tiled.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np

from .tileset import Tileset

# What this build writes. Tiled accepts anything it recognises; these are the
# values a current Tiled writes and so the values least likely to surprise it.
TSX_VERSION = "1.10"
TILED_VERSION = "1.10.2"

PROPERTY_TYPES = ("string", "int", "float", "bool", "color")


class TiledUnsupported(ValueError):
    """A Tiled file using a feature this editor does not model.

    ``feature`` is the name to put in front of the user; the message already
    contains it, and the attribute exists so a test can assert on the feature
    rather than on the sentence around it.
    """

    def __init__(self, feature: str, detail: str = "") -> None:
        self.feature = feature
        tail = f" ({detail})" if detail else ""
        super().__init__(
            f"this file uses {feature}, which Plotter does not support{tail}. "
            "Open it in Tiled and remove or flatten that feature first."
        )


@dataclass(frozen=True)
class Prop:
    """One typed custom property, as Tiled stores it."""

    type: str
    value: Any

    def __post_init__(self) -> None:
        if self.type not in PROPERTY_TYPES:
            raise TiledUnsupported(
                f"a custom property of type {self.type!r}",
                f"supported types are {', '.join(PROPERTY_TYPES)}",
            )


# --- properties ---------------------------------------------------------------


def _parse_value(kind: str, text: str) -> Any:
    if kind == "bool":
        return text.strip().lower() == "true"
    if kind == "int":
        return int(float(text))
    if kind == "float":
        return float(text)
    return str(text)


def read_properties(parent: ET.Element | None) -> dict[str, Prop]:
    """The ``<properties>`` child of an element, as a mapping.

    An element with no properties gives an empty dict, and so does one with an
    empty ``<properties>`` block -- the two are the same document.
    """
    if parent is None:
        return {}
    node = parent.find("properties")
    if node is None:
        return {}
    out: dict[str, Prop] = {}
    for entry in node.findall("property"):
        name = entry.get("name")
        if not name:
            continue
        kind = entry.get("type", "string")
        if kind not in PROPERTY_TYPES:
            raise TiledUnsupported(f"a custom property of type {kind!r}", f"property {name!r}")
        # Tiled puts a multi-line string in the element's text instead of in
        # the attribute, so the attribute is preferred and the text is the
        # fallback rather than the other way round.
        raw = entry.get("value")
        if raw is None:
            raw = entry.text or ""
        out[name] = Prop(type=kind, value=_parse_value(kind, raw))
    return out


def _value_text(prop: Prop) -> str:
    if prop.type == "bool":
        return "true" if prop.value else "false"
    if prop.type == "int":
        return str(int(prop.value))
    if prop.type == "float":
        return repr(float(prop.value))
    return str(prop.value)


def write_properties(parent: ET.Element, props: dict[str, Prop]) -> None:
    """Append a ``<properties>`` block, or nothing at all when there are none.

    Written in sorted name order rather than in whatever order they were read.
    The output is canonical on purpose -- two saves of an unchanged document
    have to be byte-identical, and a dict's order is not a property of the
    document.
    """
    if not props:
        return
    node = ET.SubElement(parent, "properties")
    for name in sorted(props):
        prop = props[name]
        entry = ET.SubElement(node, "property", {"name": name})
        # Tiled omits type="string", and matching that keeps a file written
        # here diff-clean against the same file written there.
        if prop.type != "string":
            entry.set("type", prop.type)
        entry.set("value", _value_text(prop))


# --- reading ------------------------------------------------------------------


def _root(data: bytes, expect: str) -> ET.Element:
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
    if root.find("wangsets") is not None:
        raise TiledUnsupported("Wang sets / terrain brushes")
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
    root = _root(data, "tileset")
    check_tileset_features(root)
    image = root.find("image")
    source = (image.get("source") or "").strip() if image is not None else ""
    if not source:
        raise TiledUnsupported(
            "an embedded tileset image", "Plotter needs an <image source=...> path"
        )
    return source


def read_tsx(data: bytes, image: np.ndarray) -> Tileset:
    """A ``.tsx``'s bytes plus its decoded image, as a :class:`Tileset`."""
    root = _root(data, "tileset")
    check_tileset_features(root)
    return Tileset(
        name=root.get("name") or "tileset",
        pixels=image,
        tile_w=int(root.get("tilewidth", 0) or 0),
        tile_h=int(root.get("tileheight", 0) or 0),
        spacing=int(root.get("spacing", 0) or 0),
        margin=int(root.get("margin", 0) or 0),
        properties=read_properties(root),
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
