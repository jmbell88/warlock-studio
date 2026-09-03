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
        ("<terraintypes/>", "terrain types"),
        # Deprecated pre-Wang terrain indices. Tiled itself is retiring them and
        # a refusal is the honest state; everything else a ``<tile>`` can carry
        # is modelled now and has an acceptance case below.
        ('<tile id="0" terrain="0,,,"/>', "per-tile terrain assignment"),
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


def test_a_tileset_with_no_pixels_at_all_is_still_refused():
    """A collection is modelled now, but a tileset with neither an atlas nor
    per-tile images has no pixels at all -- a file this reader cannot draw."""
    data = b'<tileset name="t" tilewidth="16" tileheight="16"><grid/></tileset>'
    with pytest.raises(tsx.TiledUnsupported, match="embedded tileset image"):
        tsx.tsx_source(data)


# --- image collections ------------------------------------------------------------


COLLECTION = b'''<tileset name="props" tilewidth="16" tileheight="16" tilecount="3" columns="0">
 <tile id="0"><image source="rock.png" width="16" height="16"/></tile>
 <tile id="5"><image source="tree.png" width="16" height="32"/></tile>
 <tile id="9"><image source="sign.png" width="16" height="16"/></tile>
</tileset>'''


def _tile_image(width: int, height: int, value: int) -> np.ndarray:
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., :3] = value
    out[..., 3] = 255
    return out


def _collection_pixels() -> dict:
    return {
        0: _tile_image(16, 16, 10),
        5: _tile_image(16, 32, 20),
        9: _tile_image(16, 16, 30),
    }


def test_the_sources_of_a_collection_are_reported_for_the_host_to_fetch():
    """This package never touches a filesystem, so the host resolves and decodes
    and hands the pixels back."""
    root = tsx.xml_root(COLLECTION, "tileset")
    assert tsx.collection_sources(root) == {0: "rock.png", 5: "tree.png", 9: "sign.png"}


def test_a_sliced_atlas_reports_no_collection_sources():
    """Which is how every caller tells the two apart without asking twice."""
    root = tsx.xml_root(
        b'<tileset name="t" tilewidth="16" tileheight="16">'
        b'<image source="a.png" width="32" height="32"/></tileset>',
        "tileset",
    )
    assert tsx.collection_sources(root) == {}


def test_a_collection_keeps_its_sparse_ids():
    ts = tsx.read_tsx(COLLECTION, _collection_pixels())
    assert ts.is_collection
    assert ts.tile_count == 3
    assert ts.collection.ids == (0, 5, 9)
    # The highest *id*, not the count: a range from the count would refuse a gid
    # the file legitimately carries.
    assert ts.max_local_id == 9


def test_each_collection_tile_keeps_its_own_size_and_pixels():
    ts = tsx.read_tsx(COLLECTION, _collection_pixels())
    assert ts.tile_size(0) == (16, 16)
    assert ts.tile_size(5) == (16, 32), "an oversized tile keeps its own height"
    assert ts.tile_pixels(5).shape == (32, 16, 4)
    assert int(ts.tile_pixels(0)[0, 0, 0]) == 10
    assert int(ts.tile_pixels(9)[0, 0, 0]) == 30


def test_the_packing_padding_is_outside_every_tiles_own_rectangle():
    """Which is what lets ``uv`` and ``tile_pixels`` ignore the packing."""
    ts = tsx.read_tsx(COLLECTION, _collection_pixels())
    for local in (0, 5, 9):
        assert (ts.tile_pixels(local)[..., 3] == 255).all(), local


def test_a_tile_the_collection_lacks_is_refused_rather_than_wrapped():
    ts = tsx.read_tsx(COLLECTION, _collection_pixels())
    with pytest.raises(IndexError):
        ts.tile_rect(4)


def test_the_palette_grid_hands_out_only_real_tiles():
    """A cell past the end of the last row is padding, and handing it out as a
    brush would give the map a gid that answers to nothing."""
    ts = tsx.read_tsx(COLLECTION, _collection_pixels())
    found = [
        ts.local_at(column, row)
        for row in range(ts.rows)
        for column in range(ts.columns)
    ]
    assert sorted(v for v in found if v is not None) == [0, 5, 9]
    assert found.count(None) == ts.columns * ts.rows - 3


def test_a_collection_writes_no_atlas_image_and_zero_columns():
    """Tiled's own spelling: a collection has no atlas to count across, and a
    reader that saw a real column count would slice one."""
    ts = tsx.read_tsx(COLLECTION, _collection_pixels())
    written = tsx.tsx_bytes(ts, image_name="unused.png")
    assert b'columns="0"' in written
    assert b"<image source=\"unused.png\"" not in written


def test_a_collection_cannot_be_a_terrain_set():
    """Terrain roles are positional over a 47-column atlas; a collection has no
    positions to read them off."""
    from warlock.studio.tilegrid.tileset import TerrainSpec, compose_collection

    atlas, collection = compose_collection(_collection_pixels())
    with pytest.raises(ValueError, match="cannot be a terrain set"):
        Tileset(
            name="x",
            pixels=atlas,
            tile_w=16,
            tile_h=16,
            collection=collection,
            terrains=(TerrainSpec(name="a", fill=(1, 1, 1, 255), outline=(0, 0, 0, 255)),),
        )


def test_composing_an_empty_collection_is_refused():
    from warlock.studio.tilegrid.tileset import compose_collection

    with pytest.raises(ValueError, match="at least one tile"):
        compose_collection({})


def test_composing_a_collection_past_the_pixel_ceiling_is_refused():
    """Every tile image is bounded by ``pixelguard``; the *atlas* was not.

    Ten thousand ``<tile>`` entries all naming one ordinary 1024px file is
    under a megabyte of ``.tsx`` and a 42 GB ``np.zeros``, which is the shape
    this ceiling exists for -- nothing in such a file is individually oversized.
    The fixture shares one array between every id rather than building ten
    thousand of them, because the refusal happens before any of them is read.
    """
    from warlock.studio.tilegrid.tileset import compose_collection

    one = _tile_image(1024, 1024, 7)
    images = dict.fromkeys(range(10_000), one)
    with pytest.raises(ValueError, match="past the .* pixels this build will allocate"):
        compose_collection(images)


def test_the_collection_ceiling_refuses_before_it_copies_anything(monkeypatch):
    """The sentence above -- "before any of them is read" -- was not true.

    ``compose_collection`` built ``frames`` through ``frozen_rgba``, which
    *copies*, and only then measured the pack and refused. So the guard against
    one 42 GB allocation was reached only after making a larger one out of the
    same inputs: ten thousand ids sharing one 1024px image is 40 GB of copies
    on the way to a refusal about 42 GB. It passed on a workstation with the
    memory to absorb that and died on a CI runner with a ``MemoryError`` over
    4 MiB, which is the wrong way round for a ceiling.

    Pinned by making the copy itself fail: if anything is read before the
    refusal, this raises ``AssertionError`` rather than ``ValueError``.
    """
    from warlock.studio.tilegrid import tileset

    def _refuse_to_copy(*_args, **_kwargs):
        raise AssertionError("a frame was copied before the ceiling refused")

    monkeypatch.setattr(tileset, "frozen_rgba", _refuse_to_copy)

    one = _tile_image(64, 64, 7)
    images = dict.fromkeys(range(100_000), one)
    with pytest.raises(ValueError, match="past the .* pixels this build will allocate"):
        tileset.compose_collection(images)


def test_an_ordinary_collection_still_composes():
    """The ceiling must not be felt by a real collection of images."""
    from warlock.studio.tilegrid.tileset import compose_collection

    atlas, collection = compose_collection(_collection_pixels())
    assert collection.cell_w == 16 and collection.cell_h == 32
    assert atlas.shape[2] == 4


@pytest.mark.parametrize(
    ("attribute", "field", "value"),
    [
        ('objectalignment="center"', "object_alignment", "center"),
        ('tilerendersize="grid"', "render_size", "grid"),
        ('fillmode="preserve-aspect-fit"', "fill_mode", "preserve-aspect-fit"),
        ('backgroundcolor="#ff00ff"', "background", "#ff00ff"),
    ],
)
def test_a_presentation_attribute_round_trips(attribute, field, value):
    """The ledger rule's acceptance half for four rows that moved together:
    read, written and back, with the writer omitting every default so the
    output still diffs cleanly against a file Tiled wrote."""
    data = f"""<tileset name="t" tilewidth="16" tileheight="16" {attribute}>
 <image source="a.png" width="64" height="64"/>
</tileset>""".encode()
    ts = tsx.read_tsx(data, _pixels())
    assert getattr(ts, field) == value
    again = tsx.read_tsx(tsx.tsx_bytes(ts, image_name="a.png"), _pixels())
    assert getattr(again, field) == value


def test_a_tile_offset_round_trips():
    data = b"""<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 <tileoffset x="3" y="-5"/>
</tileset>"""
    ts = tsx.read_tsx(data, _pixels())
    assert (ts.offset_x, ts.offset_y) == (3, -5)
    again = tsx.read_tsx(tsx.tsx_bytes(ts, image_name="a.png"), _pixels())
    assert (again.offset_x, again.offset_y) == (3, -5)


def test_a_tileset_with_default_presentation_writes_none_of_it():
    data = b"""<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
</tileset>"""
    written = tsx.tsx_bytes(tsx.read_tsx(data, _pixels()), image_name="a.png")
    for absent in (b"objectalignment", b"tilerendersize", b"fillmode", b"tileoffset"):
        assert absent not in written


def test_an_invalid_presentation_value_is_refused():
    data = b"""<tileset name="t" tilewidth="16" tileheight="16" tilerendersize="huge">
 <image source="a.png" width="64" height="64"/>
</tileset>"""
    with pytest.raises(ValueError, match="render size"):
        tsx.read_tsx(data, _pixels())


def test_a_tileset_image_transparent_colour_becomes_alpha():
    """Applied at *decode*, so nothing downstream ever sees the key colour and
    no renderer needs to know the tileset had one."""
    import numpy as np

    data = b'''<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64" trans="ff00ff"/>
</tileset>'''
    pixels = _pixels()
    keyed = np.array(pixels)
    keyed[0, 0, :3] = (255, 0, 255)
    ts = tsx.read_tsx(data, keyed)
    assert int(ts.pixels[0, 0, 3]) == 0, "the key colour became a hole"
    assert int(ts.pixels[1, 1, 3]) == 255, "and nothing else moved"


def test_a_transparent_colour_that_is_not_a_colour_is_refused():
    data = b'''<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64" trans="nonsense"/>
</tileset>'''
    # A plain ValueError rather than a named refusal: the feature is supported
    # and it is the *value* that is wrong, so there is no compat row to name.
    with pytest.raises(ValueError, match="transparent colour is"):
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


def test_a_wangset_whose_count_disagrees_with_its_phases_is_not_the_preset():
    """A phases property over a classic-count wangset is a foreign file. It is
    no longer *refused* -- the general model reads it as data -- but it is
    emphatically not recognised as this editor's blob preset, which is the part
    that would have been wrong to accept."""
    data = tsx.tsx_bytes(_terrain_tileset(2), image_name="g.png")
    text = data.decode()
    hacked = text.replace('value="2"', 'value="4"').encode()
    back = tsx.read_tsx(hacked, _pixels())
    assert back.terrains == (), "not adopted as a terrain set"
    assert back.wangsets, "but read as data rather than dropped"


def test_a_phases_property_on_an_ordinary_tileset_is_just_a_property():
    source = _tileset(properties={"phases": tsx.Prop("int", 3)})
    data = tsx.tsx_bytes(source, image_name="t.png")
    back = tsx.read_tsx(data, _pixels())
    assert back.phases == 1
    assert back.properties == {"phases": tsx.Prop("int", 3)}


# --- per-tile metadata (the six rows that moved) ------------------------------


def _with_tiles(body: str) -> str:
    return f"""<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 {body}
</tileset>"""


def test_a_tile_class_and_probability_are_read() -> None:
    ts = tsx.read_tsx(
        _with_tiles('<tile id="3" class="Water" probability="0.25"/>').encode(),
        _pixels(),
    )
    meta = ts.meta_of(3)
    assert meta.class_name == "Water"
    assert meta.probability == 0.25
    assert ts.meta_of(0).is_empty, "the store is sparse"


def test_a_tile_animation_is_read_in_order() -> None:
    ts = tsx.read_tsx(
        _with_tiles(
            '<tile id="1"><animation>'
            '<frame tileid="1" duration="120"/>'
            '<frame tileid="2" duration="80"/>'
            "</animation></tile>"
        ).encode(),
        _pixels(),
    )
    frames = ts.meta_of(1).animation
    assert [(f.local_id, f.duration_ms) for f in frames] == [(1, 120), (2, 80)]


def test_tile_collision_shapes_are_read() -> None:
    ts = tsx.read_tsx(
        _with_tiles(
            '<tile id="2"><objectgroup>'
            '<object id="1" x="1" y="2" width="8" height="9"/>'
            '<object id="2" x="0" y="0" width="4" height="4"><ellipse/></object>'
            '<object id="3" x="2" y="2"><polygon points="0,0 4,0 4,4"/></object>'
            "</objectgroup></tile>"
        ).encode(),
        _pixels(),
    )
    shapes = ts.meta_of(2).collision
    assert [type(s).__name__ for s in shapes] == ["TileRect", "TileEllipse", "TilePolygon"]
    assert shapes[0].w == 8.0
    assert shapes[2].points == ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))


def test_tile_custom_properties_are_read() -> None:
    ts = tsx.read_tsx(
        _with_tiles(
            '<tile id="4"><properties>'
            '<property name="hp" type="int" value="7"/>'
            "</properties></tile>"
        ).encode(),
        _pixels(),
    )
    assert ts.meta_of(4).properties["hp"].value == 7


def test_per_tile_metadata_round_trips_through_the_writer() -> None:
    """The ledger rule's acceptance half: read, write, read, and the model is
    the same on both sides."""
    source = _with_tiles(
        '<tile id="2" class="Water" probability="0.5">'
        '<properties><property name="hp" type="int" value="7"/></properties>'
        '<animation><frame tileid="2" duration="120"/></animation>'
        '<objectgroup><object id="1" x="1" y="2" width="8" height="9"/></objectgroup>'
        "</tile>"
    )
    first = tsx.read_tsx(source.encode(), _pixels())
    again = tsx.read_tsx(tsx.tsx_bytes(first, image_name="a.png"), _pixels())
    assert again.meta_of(2) == first.meta_of(2)


def test_a_tileset_with_no_tile_metadata_writes_no_tile_elements() -> None:
    """The sparse store's own rule -- a round trip must not grow elements
    nobody authored."""
    ts = tsx.read_tsx(_with_tiles("").encode(), _pixels())
    assert ts.tiles == {}
    assert b"<tile " not in tsx.tsx_bytes(ts, image_name="a.png")


def test_a_foreign_wangset_round_trips_as_data():
    """The ledger rule's acceptance half: read, written and back, with the set
    intact. A blob preset takes the other path and its export is unchanged."""
    source = b'''<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 <wangsets>
  <wangset name="Edges" type="edge" tile="-1">
   <wangcolor name="road" color="#808080" tile="-1" probability="3"/>
   <wangtile tileid="0" wangid="1,0,1,0,0,0,0,0"/>
   <wangtile tileid="5" wangid="0,0,1,0,1,0,0,0"/>
  </wangset>
 </wangsets>
</tileset>'''
    first = tsx.read_tsx(source, _pixels())
    assert first.terrains == ()
    assert len(first.wangsets) == 1

    again = tsx.read_tsx(tsx.tsx_bytes(first, image_name="a.png"), _pixels())
    assert again.wangsets == first.wangsets


def test_a_blob_preset_still_writes_the_wangset_it_always_did():
    """The determinism pin: the preset's export is byte-identical, which is what
    lets the general model land beside it rather than on top of it."""
    source = _terrain_tileset(1)
    first = tsx.tsx_bytes(source, image_name="g.png")
    back = tsx.read_tsx(first, _pixels(w=source.image_w, h=source.image_h))
    assert back.terrains, "still recognised as the preset"
    assert back.wangsets == (), "and not held twice"
    assert tsx.tsx_bytes(back, image_name="g.png") == first


def test_a_malformed_wangid_is_skipped_rather_than_refusing_the_file():
    """One tile of one set; losing a whole map over it would be the half-read
    trade run backwards."""
    source = b'''<tileset name="t" tilewidth="16" tileheight="16">
 <image source="a.png" width="64" height="64"/>
 <wangsets>
  <wangset name="Broken" type="corner" tile="-1">
   <wangcolor name="a" color="#ff0000" tile="-1" probability="1"/>
   <wangtile tileid="0" wangid="0,1,0,1,0,1,0,1"/>
   <wangtile tileid="1" wangid="nonsense"/>
   <wangtile tileid="2" wangid="0,9,0,0,0,0,0,0"/>
  </wangset>
 </wangsets>
</tileset>'''
    ts = tsx.read_tsx(source, _pixels())
    assert set(ts.wangsets[0].tiles) == {0}
