"""``.tsx`` -- the one tileset reader and writer in the repo.

Packwright's grid packer emits through :func:`tsx_bytes` rather than assembling
XML of its own, so this file is also what stops that output and Plotter's from
drifting into two dialects of one format.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from warlock.studio.plotter import tsx
from warlock.studio.plotter.tileset import Tileset


def _pixels(w: int = 64, h: int = 64) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def _tileset(**kw) -> Tileset:
    return Tileset(name=kw.pop("name", "terrain"), pixels=_pixels(), tile_w=16, tile_h=16, **kw)


# --- round trip ---------------------------------------------------------------


def test_a_tileset_round_trips_through_its_own_writer():
    source = _tileset(spacing=0, margin=0)
    data = tsx.tsx_bytes(source, image_name="terrain.png")
    assert tsx.tsx_source(data) == "terrain.png"
    back = tsx.read_tsx(data, _pixels())
    assert (back.name, back.tile_w, back.tile_h) == ("terrain", 16, 16)
    assert (back.columns, back.rows, back.tile_count) == (4, 4, 16)


def test_spacing_and_margin_survive_and_are_omitted_when_zero():
    """Tiled omits both at zero, and matching that keeps a file written here
    diff-clean against the same tileset written there."""
    plain = tsx.tsx_bytes(_tileset(), image_name="a.png").decode()
    assert "spacing" not in plain and "margin" not in plain

    # 2 + 3*10 + 2*4 + 2 == 42 across, so a valid slicing.
    spaced = Tileset(name="t", pixels=_pixels(42, 42), tile_w=10, tile_h=10, spacing=4, margin=2)
    data = tsx.tsx_bytes(spaced, image_name="a.png")
    back = tsx.read_tsx(data, _pixels(42, 42))
    assert (back.spacing, back.margin) == (4, 2)


def test_the_declared_counts_match_the_slicing():
    root = ET.fromstring(tsx.tsx_bytes(_tileset(), image_name="a.png"))
    assert root.get("tilecount") == "16"
    assert root.get("columns") == "4"
    image = root.find("image")
    assert image is not None
    assert (image.get("width"), image.get("height")) == ("64", "64")


def test_two_writes_of_one_tileset_are_byte_identical():
    ts = _tileset()
    assert tsx.tsx_bytes(ts, image_name="a.png") == tsx.tsx_bytes(ts, image_name="a.png")


# --- properties ---------------------------------------------------------------


def test_every_property_type_round_trips_with_its_type_intact():
    """``color`` and ``string`` are both ``str``, which is why the type is
    stored rather than inferred -- a round trip that guessed would silently
    retype every colour a user set in Tiled."""
    props = {
        "name": tsx.Prop("string", "grass"),
        "cost": tsx.Prop("int", 7),
        "drag": tsx.Prop("float", 0.5),
        "solid": tsx.Prop("bool", True),
        "tint": tsx.Prop("color", "#ff00ff00"),
    }
    ts = Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16, properties=props)
    back = tsx.read_tsx(tsx.tsx_bytes(ts, image_name="a.png"), _pixels())
    assert back.properties == props


def test_a_string_property_is_written_without_a_type_attribute():
    ts = Tileset(
        name="t",
        pixels=_pixels(),
        tile_w=16,
        tile_h=16,
        properties={"n": tsx.Prop("string", "x")},
    )
    text = tsx.tsx_bytes(ts, image_name="a.png").decode()
    assert 'name="n"' in text and 'type="string"' not in text


def test_properties_are_written_in_sorted_order():
    """Canonical output: two saves of an unchanged document have to be
    byte-identical, and a dict's order is not a property of the document."""
    ts = Tileset(
        name="t",
        pixels=_pixels(),
        tile_w=16,
        tile_h=16,
        properties={"z": tsx.Prop("int", 1), "a": tsx.Prop("int", 2)},
    )
    text = tsx.tsx_bytes(ts, image_name="a.png").decode()
    assert text.index('name="a"') < text.index('name="z"')


def test_a_multiline_string_in_the_element_text_is_read():
    """Tiled puts a multi-line value in the element body rather than in the
    attribute, so the attribute is preferred and the text is the fallback."""
    data = b"""<?xml version="1.0" encoding="UTF-8"?>
<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 <properties><property name="note">line one
line two</property></properties>
</tileset>"""
    back = tsx.read_tsx(data, _pixels())
    assert back.properties["note"].value == "line one\nline two"


def test_an_unknown_property_type_is_refused_by_name():
    data = b"""<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 <properties><property name="who" type="object" value="3"/></properties>
</tileset>"""
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tsx.read_tsx(data, _pixels())
    assert "object" in str(exc.value)


# --- refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "feature"),
    [
        ("<wangsets/>", "Wang sets"),
        ("<terraintypes/>", "terrain types"),
        ('<tile id="0"><animation/></tile>', "per-tile animation"),
        ('<tile id="0"><image source="x.png"/></tile>', "image-collection"),
        ('<tile id="0"><objectgroup/></tile>', "per-tile collision"),
        ('<tile id="0"><properties/></tile>', "per-tile custom properties"),
    ],
)
def test_an_unsupported_tileset_feature_is_refused_and_named(body, feature):
    data = f"""<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 {body}
</tileset>""".encode()
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tsx.read_tsx(data, _pixels())
    assert feature in str(exc.value)


def test_an_image_collection_with_no_atlas_at_all_is_refused():
    data = b'<tileset name="t" tilewidth="16" tileheight="16"><grid/></tileset>'
    with pytest.raises(tsx.TiledUnsupported, match="image-collection"):
        tsx.tsx_source(data)


def test_a_file_that_is_not_a_tileset_is_refused_plainly():
    with pytest.raises(ValueError, match="expected a <tileset>"):
        tsx.read_tsx(b"<map/>", _pixels())
    with pytest.raises(ValueError, match="not a readable"):
        tsx.read_tsx(b"not xml at all", _pixels())


def test_tiled_unsupported_is_a_value_error():
    """So a caller that only wants "this did not load" needs no new except
    clause."""
    assert issubclass(tsx.TiledUnsupported, ValueError)
    assert tsx.TiledUnsupported("hex maps").feature == "hex maps"
