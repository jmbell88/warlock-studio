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
from warlock.studio.tilegrid.tileset import Tileset


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


def test_a_tileset_class_round_trips_through_its_own_writer():
    source = _tileset(class_name="BiomeAtlas")
    data = tsx.tsx_bytes(source, image_name="terrain.png")
    assert b'class="BiomeAtlas"' in data
    assert tsx.read_tsx(data, _pixels()).class_name == "BiomeAtlas"


def test_tileset_transformations_round_trip_through_the_writer():
    source = _tileset(transformations=(True, False, True, True))
    data = tsx.tsx_bytes(source, image_name="terrain.png")
    assert b'<transformations hflip="1" vflip="0" rotate="1"' in data
    assert b'preferuntransformed="1"' in data
    assert tsx.read_tsx(data, _pixels()).transformations == (
        True,
        False,
        True,
        True,
    )


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


def test_an_object_property_on_a_tileset_loads_now_that_plotter_models_one():
    """This case used to assert ``type="object"`` was refused. It is one of
    the four types the property model gained, so the case is *flipped* rather
    than deleted -- the refusal moved because the model did."""
    data = b"""<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 <properties><property name="who" type="object" value="3"/></properties>
</tileset>"""
    assert tsx.read_tsx(data, _pixels()).properties == {"who": tsx.Prop("object", 3)}


def test_the_new_property_types_round_trip_through_a_tileset():
    """``tsx`` is the second writer of the XML property block and the one
    Packwright emits, so the types it can carry are asserted here too rather
    than assumed from the map side."""
    props = {
        "art": tsx.Prop("file", "art/atlas.png"),
        "owner": tsx.Prop("object", 4),
        "npc": tsx.Prop("class", {"hp": tsx.Prop("int", 3)}, propertytype="NPC"),
    }
    ts = Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16, properties=props)
    assert tsx.read_tsx(tsx.tsx_bytes(ts, image_name="a.png"), _pixels()).properties == props


def test_a_property_type_outside_tileds_nine_is_refused_by_name():
    data = b"""<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 <properties><property name="who" type="vector2" value="1,2"/></properties>
</tileset>"""
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tsx.read_tsx(data, _pixels())
    assert "vector2" in str(exc.value)


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
        ('<tile id="0" class="Water"/>', "per-tile class"),
        ('<tile id="0" probability="0.5"/>', "per-tile probability"),
        ('<tile id="0" terrain="0,,,"/>', "per-tile terrain assignment"),
        ('<tileoffset x="1" y="0"/>', "tileset tile offset"),
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


@pytest.mark.parametrize(
    ("attribute", "feature"),
    [
        ('objectalignment="center"', "tileset object alignment"),
        ('tilerendersize="grid"', "tileset render size"),
        ('fillmode="preserve-aspect-fit"', "tileset fill mode"),
        ('backgroundcolor="#ff00ff"', "tileset background colour"),
    ],
)
def test_an_unsupported_tileset_attribute_is_refused_and_named(attribute, feature):
    data = f'''<tileset name="t" tilewidth="16" tileheight="16" {attribute}>
 <image source="a.png" width="64" height="64"/>
</tileset>'''.encode()
    with pytest.raises(tsx.TiledUnsupported, match=feature):
        tsx.read_tsx(data, _pixels())


def test_a_tileset_image_transparent_colour_is_refused():
    data = b'''<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64" trans="ff00ff"/>
</tileset>'''
    with pytest.raises(tsx.TiledUnsupported, match="transparent colour"):
        tsx.read_tsx(data, _pixels())


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


# --- phase variants -------------------------------------------------------------


def _terrain_tileset(k: int, terrains: int = 2) -> Tileset:
    from warlock.studio.tilegrid import blob
    from warlock.studio.tilegrid.tileset import TerrainSpec

    tile = 8
    specs = tuple(
        TerrainSpec(f"T{i}", (10 * i, 200, 0, 255), (0, 90, 0, 255))
        for i in range(terrains)
    )
    return Tileset(
        name="ground",
        pixels=_pixels(blob.TILE_COUNT * tile, terrains * k * k * tile),
        tile_w=tile,
        tile_h=tile,
        terrains=specs,
        phases=k,
    )


@pytest.mark.parametrize("k", [1, 4])
def test_a_phased_terrain_set_round_trips(k):
    """Recognise-or-refuse symmetry at both phase counts: every file this
    writes, this reads, with the phase count intact."""
    source = _terrain_tileset(k)
    data = tsx.tsx_bytes(source, image_name="ground.png")
    back = tsx.read_tsx(data, np.asarray(source.pixels))
    assert back.phases == k
    assert len(back.terrains) == 2
    assert [entry.name for entry in back.terrains] == ["T0", "T1"]
    # The property is the field's spelling, not a second stored fact.
    assert "phases" not in back.properties


def test_the_phases_property_is_written_only_when_it_says_something():
    data = tsx.tsx_bytes(_terrain_tileset(1), image_name="g.png")
    assert b'"phases"' not in data and b"phases" not in data
    data4 = tsx.tsx_bytes(_terrain_tileset(4), image_name="g.png")
    assert b'name="phases"' in data4


def test_a_wangset_whose_count_disagrees_with_its_phases_is_refused():
    """A phases property over a classic-count wangset is a foreign file."""
    data = tsx.tsx_bytes(_terrain_tileset(2), image_name="g.png")
    text = data.decode()
    hacked = text.replace('value="2"', 'value="4"').encode()
    with pytest.raises(tsx.TiledUnsupported):
        tsx.read_tsx(hacked, _pixels())


def test_a_phases_property_on_an_ordinary_tileset_is_just_a_property():
    source = _tileset(properties={"phases": tsx.Prop("int", 3)})
    data = tsx.tsx_bytes(source, image_name="t.png")
    back = tsx.read_tsx(data, _pixels())
    assert back.phases == 1
    assert back.properties == {"phases": tsx.Prop("int", 3)}
