"""TMX and TMJ: the round trip, and the four encodings that must agree.

The property this file exists for is **bit-exactness**. Flags live in the top
three bits of every cell, nothing between the reader and the renderer strips
them, and a single lost bit is a tile that draws the wrong way round in an
exported map and nowhere else -- which is to say, in a place nobody looks until
the map is in an engine.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import re
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest
from PIL import Image

from warlock.studio.plotter import tmx, tsx
from warlock.studio.plotter.tilemap import MapDoc, MapObject, ObjectLayer, TileLayer, new_uid
from warlock.studio.tilegrid import gid
from warlock.studio.tilegrid.tileset import Tileset


def _pixels(w: int = 64, h: int = 64) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    array[0, 0] = (7, 8, 9, 255)
    return array


def _doc() -> MapDoc:
    doc = MapDoc(6, 4, 16, 16)
    doc.add_tileset(Tileset(name="terrain", pixels=_pixels(), tile_w=16, tile_h=16))
    tiles = doc.add_tile_layer("Ground")
    cells = np.zeros((4, 6), gid.DTYPE)
    cells[0, 0] = gid.compose(1)
    cells[1, 1] = gid.compose(2, flip_h=True)
    cells[2, 2] = gid.compose(3, flip_v=True, flip_d=True)
    cells[3, 3] = gid.compose(16, flip_h=True, flip_v=True, flip_d=True)
    doc.write_region(tiles.uid, 0, 0, cells)
    doc.set_layer_props(tiles.uid, opacity=0.5)

    objects = doc.add_object_layer("Things")
    doc.add_object(
        objects.uid,
        MapObject(
            uid=new_uid(),
            name="spawn",
            kind="point",
            x=17.5,
            y=3.0,
            properties={"team": tsx.Prop("int", 2)},
        ),
    )
    doc.add_object(
        objects.uid,
        MapObject(uid=new_uid(), name="zone", kind="rect", x=0, y=0, w=32, h=48,
                  obj_class="Trigger", visible=False),
    )
    doc.properties = {"theme": tsx.Prop("string", "cave")}
    doc.backgroundcolor = "#ff112233"
    return doc


def _loaders(files: dict[str, bytes]):
    """Resolve the exporter's own output, which is what a caller writing the
    files to a directory and reading them back would do."""

    def image_loader(source: str) -> np.ndarray:
        raw = files[source] if source in files else files[f"tilesets/{source}"]
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"), dtype=np.uint8)

    def tsx_loader(source: str) -> Tileset:
        raw = files[source]
        image = tsx.tsx_source(raw)
        png = files[f"tilesets/{image}"]
        return tsx.read_tsx(
            raw, np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.uint8)
        )

    return {"image_loader": image_loader, "tsx_loader": tsx_loader}


def _cells(doc: MapDoc) -> np.ndarray:
    return doc.tile_layers()[0].data


# --- round trips --------------------------------------------------------------


def test_a_tmx_round_trip_is_bit_exact_including_every_flag():
    doc = _doc()
    files = tmx.tmx_export(doc)
    back = tmx.read_tmx(files["map.tmx"], **_loaders(files))
    assert np.array_equal(_cells(back), _cells(doc))
    assert int(_cells(back)[3, 3]) == gid.compose(16, flip_h=True, flip_v=True, flip_d=True)


def test_a_tmj_round_trip_is_bit_exact_too():
    doc = _doc()
    files = tmx.tmj_export(doc)
    back = tmx.read_tmj(files["map.tmj"], **_loaders(files))
    assert np.array_equal(_cells(back), _cells(doc))


def test_the_two_formats_read_back_the_same_document():
    """One model, two spellings. If they disagreed, one of them would be
    accepting a file the editor cannot draw."""
    doc = _doc()
    xml_files, json_files = tmx.tmx_export(doc), tmx.tmj_export(doc)
    a = tmx.read_tmx(xml_files["map.tmx"], **_loaders(xml_files))
    b = tmx.read_tmj(json_files["map.tmj"], **_loaders(json_files))
    assert np.array_equal(_cells(a), _cells(b))
    assert [layer.name for layer in a.layers] == [layer.name for layer in b.layers]
    assert a.properties == b.properties
    assert a.backgroundcolor == b.backgroundcolor


def test_layers_come_back_in_stacking_order():
    """Document order is stacking order, bottom first, which is why the reader
    walks the root's children rather than doing ``findall`` per tag."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    back = tmx.read_tmx(files["map.tmx"], **_loaders(files))
    assert [layer.name for layer in back.layers] == ["Ground", "Things"]


def test_layer_and_object_detail_survives():
    doc = _doc()
    files = tmx.tmx_export(doc)
    back = tmx.read_tmx(files["map.tmx"], **_loaders(files))

    ground = back.layers[0]
    assert ground.opacity == 0.5 and ground.visible

    spawn, zone = back.layers[1].objects
    assert (spawn.kind, spawn.x, spawn.y) == ("point", 17.5, 3.0)
    assert spawn.properties == {"team": tsx.Prop("int", 2)}
    assert (zone.kind, zone.w, zone.h, zone.obj_class) == ("rect", 32.0, 48.0, "Trigger")
    assert zone.visible is False


def test_the_tileset_image_travels_as_a_png_beside_the_tsx():
    """TMX has no portable way to embed an image, so a tileset is a ``.tsx``
    plus a ``.png``, and the exporter returns the mapping rather than deciding
    where they go."""
    files = tmx.tmx_export(_doc())
    assert sorted(files) == ["map.tmx", "tilesets/00-terrain.png", "tilesets/00-terrain.tsx"]
    # The image name inside the .tsx is relative to the .tsx, which sits beside
    # it -- not to the map.
    assert tsx.tsx_source(files["tilesets/00-terrain.tsx"]) == "00-terrain.png"


def test_two_tilesets_of_one_name_do_not_collide():
    """Names legitimately repeat, and two files sharing one would silently be
    the same file."""
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(Tileset(name="same", pixels=_pixels(32, 32), tile_w=16, tile_h=16))
    doc.add_tileset(Tileset(name="same", pixels=_pixels(32, 32), tile_w=16, tile_h=16))
    files = tmx.tmx_export(doc)
    assert "tilesets/00-same.tsx" in files and "tilesets/01-same.tsx" in files


def test_an_exported_map_opens_clean():
    doc = _doc()
    files = tmx.tmx_export(doc)
    assert not tmx.read_tmx(files["map.tmx"], **_loaders(files)).dirty


def test_two_exports_of_one_document_are_byte_identical():
    doc = _doc()
    assert tmx.tmx_export(doc) == tmx.tmx_export(doc)
    assert tmx.tmj_export(doc) == tmx.tmj_export(doc)


def _classy() -> MapDoc:
    """``_doc`` plus a two-member class property, at both levels that carry
    one. Two members because one cannot be out of order, and the sorting that
    keeps a class deterministic lives *inside* a value the cases above never
    look at."""
    doc = _doc()
    npc = tsx.Prop(
        "class",
        {"zeal": tsx.Prop("int", 1), "armour": tsx.Prop("string", "plate")},
        propertytype="NPC",
    )
    doc.properties = {"boss": npc, "art": tsx.Prop("file", "art/boss.png")}
    doc.set_layer_props(doc.layers[0].uid, properties={"boss": npc})
    return doc


def test_two_exports_of_a_document_with_a_class_property_are_byte_identical():
    """Determinism reaches all the way down. The case above carries no class
    property and so exercises none of the nested sorting -- and a writer that
    is deterministic at the top level but not inside a member is a writer with
    an unsorted dict in it, which is precisely what the byte rule is for."""
    doc = _classy()
    assert tmx.tmx_export(doc) == tmx.tmx_export(doc)
    assert tmx.tmj_export(doc) == tmx.tmj_export(doc)


def test_class_members_are_written_in_sorted_order_in_both_spellings():
    """Sorted by name rather than by insertion, at every depth -- a dict's
    order is not a fact about the document. Asserted on the *bytes* rather
    than on a re-read, because a re-read into a dict would agree either
    way."""
    doc = _classy()
    text = tmx.tmx_export(doc)["map.tmx"].decode()
    assert text.index('name="armour"') < text.index('name="zeal"')

    payload = json.loads(tmx.tmj_export(doc)["map.tmj"])
    boss = next(entry for entry in payload["properties"] if entry["name"] == "boss")
    assert boss["propertytype"] == "NPC"
    assert list(boss["value"]) == ["armour", "zeal"]
    # And the properties themselves are sorted, which is what puts the class
    # ahead of the file property rather than leaving them as they were set.
    assert [entry["name"] for entry in payload["properties"]] == ["art", "boss"]


# --- persistent ids -------------------------------------------------------
#
# Tiled gives every layer and object a persistent ``id`` that survives in the
# file, distinct from the process-only ``uid`` an undo step is addressed by.
# Plotter used to mint a throwaway one at every export (``itertools.count``);
# now the document carries its own, monotone and never reused.


def _tmx_with_ids(*, layer_id: int, object_id: int, next_ids: bool = False) -> bytes:
    next_attrs = ' nextlayerid="42" nextobjectid="55"' if next_ids else ""
    return (
        '<map version="1.10" orientation="orthogonal" width="1" height="1" '
        f'tilewidth="16" tileheight="16"{next_attrs}>'
        f'<objectgroup id="{layer_id}" name="Things">'
        f'<object id="{object_id}" name="spawn" x="0" y="0"><point/></object>'
        "</objectgroup>"
        "</map>"
    ).encode()


def _bare_loaders():
    return {"image_loader": lambda s: None, "tsx_loader": lambda s: None}


def test_imported_object_ids_survive_a_tmx_round_trip():
    doc = tmx.read_tmx(_tmx_with_ids(layer_id=9, object_id=7), **_bare_loaders())
    obj = doc.layers[0].objects[0]
    assert (doc.layers[0].id, obj.id) == (9, 7)

    files = tmx.tmx_export(doc)
    back = tmx.read_tmx(files["map.tmx"], **_bare_loaders())
    assert (back.layers[0].id, back.layers[0].objects[0].id) == (9, 7)


def test_new_objects_allocate_past_the_imported_ids():
    """No declared ``nextlayerid``/``nextobjectid`` in the source file, so the
    counters fall back to one past the highest id actually seen."""
    doc = tmx.read_tmx(_tmx_with_ids(layer_id=9, object_id=7), **_bare_loaders())
    assert (doc.next_layer_id, doc.next_object_id) == (10, 8)

    new_obj = doc.add_object(doc.layers[0].uid, MapObject(uid=new_uid(), name="new", kind="point"))
    assert new_obj.id == 8
    new_layer = doc.add_tile_layer()
    assert new_layer.id == 10


def test_declared_next_ids_are_adopted_over_the_computed_minimum():
    doc = tmx.read_tmx(
        _tmx_with_ids(layer_id=1, object_id=2, next_ids=True), **_bare_loaders()
    )
    assert (doc.next_layer_id, doc.next_object_id) == (42, 55)


def test_a_declared_next_id_below_the_actual_max_id_is_not_trusted():
    """A hand-edited or otherwise corrupt file must not be allowed to make the
    very next allocation collide with an id already in the document."""
    data = (
        b'<map version="1.10" orientation="orthogonal" width="1" height="1" '
        b'tilewidth="16" tileheight="16" nextlayerid="1" nextobjectid="1">'
        b'<objectgroup id="5" name="Things">'
        b'<object id="9" name="spawn" x="0" y="0"><point/></object>'
        b"</objectgroup>"
        b"</map>"
    )
    doc = tmx.read_tmx(data, **_bare_loaders())
    assert doc.next_layer_id > 5
    assert doc.next_object_id > 9


def test_a_tmj_import_also_adopts_ids_and_next_counters():
    payload = {
        "type": "map",
        "version": "1.10",
        "orientation": "orthogonal",
        "renderorder": "right-down",
        "infinite": False,
        "width": 1,
        "height": 1,
        "tilewidth": 16,
        "tileheight": 16,
        "nextlayerid": 5,
        "nextobjectid": 6,
        "layers": [
            {
                "type": "objectgroup",
                "id": 3,
                "name": "Things",
                "draworder": "topdown",
                "objects": [
                    {"id": 4, "name": "spawn", "x": 0, "y": 0, "point": True, "visible": True}
                ],
            }
        ],
    }
    doc = tmx.read_tmj(json.dumps(payload).encode(), **_bare_loaders())
    assert (doc.layers[0].id, doc.layers[0].objects[0].id) == (3, 4)
    assert (doc.next_layer_id, doc.next_object_id) == (5, 6)


def test_a_tmx_round_trip_preserves_the_docs_own_ids():
    """``_doc()`` builds through the ops, so every layer and object already
    carries a real id by the time it is exported."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    back = tmx.read_tmx(files["map.tmx"], **_loaders(files))
    assert [layer.id for layer in back.layers] == [layer.id for layer in doc.layers]
    assert [o.id for o in back.layers[1].objects] == [o.id for o in doc.layers[1].objects]


def test_export_writes_the_docs_own_persistent_ids():
    doc = _doc()
    tiles_layer, things_layer = doc.layers
    spawn, zone = things_layer.objects
    files = tmx.tmx_export(doc)
    root = ET.fromstring(files["map.tmx"])
    assert root.find("layer").get("id") == str(tiles_layer.id)
    group = root.find("objectgroup")
    assert group.get("id") == str(things_layer.id)
    assert [o.get("id") for o in group.findall("object")] == [str(spawn.id), str(zone.id)]
    assert root.get("nextlayerid") == str(doc.next_layer_id)
    assert root.get("nextobjectid") == str(doc.next_object_id)


def test_export_assigns_but_does_not_persist_ids_for_hand_built_zero_id_docs():
    """A document built by direct construction, bypassing the ops that assign
    ids at creation -- the same shape a ``.wmap``-loaded document has, since
    ids are out of scope there this milestone -- still exports something
    valid, and exporting it twice changes nothing about the document itself."""
    layer = TileLayer(uid=new_uid(), name="l", data=gid.empty_layer(2, 2))
    other = ObjectLayer(
        uid=new_uid(),
        name="o",
        objects=[MapObject(uid=new_uid(), name="a", kind="point")],
    )
    doc = MapDoc(2, 2, 16, 16, layers=[layer, other])
    assert (layer.id, other.id, other.objects[0].id) == (0, 0, 0)

    files_a = tmx.tmx_export(doc)
    files_b = tmx.tmx_export(doc)
    assert files_a == files_b
    assert (layer.id, other.id, other.objects[0].id) == (0, 0, 0)

    root = ET.fromstring(files_a["map.tmx"])
    assert root.find("layer").get("id") == "1"
    assert root.find("objectgroup").get("id") == "2"
    assert root.find("objectgroup/object").get("id") == "1"
    assert (root.get("nextlayerid"), root.get("nextobjectid")) == ("3", "2")


# --- encodings ----------------------------------------------------------------


def _with_data(files: dict[str, bytes], payload: str) -> bytes:
    return re.sub(
        r'<data encoding="csv">.*?</data>', payload, files["map.tmx"].decode(), flags=re.S
    ).encode()


@pytest.mark.parametrize("compression", ["", "zlib", "gzip"])
def test_every_base64_variant_agrees_with_the_csv_one(compression):
    """A layer written four ways is one layer. Reinterpreting the payload as
    little-endian uint32 is what keeps the flags out of it."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    raw = _cells(doc).astype("<u4").tobytes()
    if compression == "zlib":
        raw = zlib.compress(raw)
    elif compression == "gzip":
        raw = gzip.compress(raw)
    attrs = f' compression="{compression}"' if compression else ""
    body = f'<data encoding="base64"{attrs}>{base64.b64encode(raw).decode()}</data>'
    back = tmx.read_tmx(_with_data(files, body), **_loaders(files))
    assert np.array_equal(_cells(back), _cells(doc))


def test_the_old_xml_tile_element_form_still_reads():
    """Still selectable in Tiled as the "XML" tile-layer format, so refusing it
    would refuse a file a user legitimately exported. Written by nothing here."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    tiles = "".join(f'<tile gid="{int(v)}"/>' for v in _cells(doc).reshape(-1))
    back = tmx.read_tmx(_with_data(files, f"<data>{tiles}</data>"), **_loaders(files))
    assert np.array_equal(_cells(back), _cells(doc))


def test_a_json_layer_may_arrive_base64_encoded_too():
    doc = _doc()
    files = tmx.tmj_export(doc)
    payload = json.loads(files["map.tmj"])
    raw = zlib.compress(_cells(doc).astype("<u4").tobytes())
    payload["layers"][0]["data"] = base64.b64encode(raw).decode()
    payload["layers"][0]["encoding"] = "base64"
    payload["layers"][0]["compression"] = "zlib"
    back = tmx.read_tmj(json.dumps(payload).encode(), **_loaders(files))
    assert np.array_equal(_cells(back), _cells(doc))


def test_writing_is_always_csv_whatever_was_read():
    """The output must not depend on the compression a file happened to arrive
    in, and CSV is the form a diff can show."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    root = ET.fromstring(files["map.tmx"])
    data = root.find("layer/data")
    assert data is not None and data.get("encoding") == "csv"
    assert data.get("compression") is None


def test_an_embedded_tileset_keeps_the_terrains_its_wangset_declares():
    """The XML side recognised a wangset and then dropped it, which left the
    Terrain tool greyed out on a map whose own atlas declares one. Read-side
    parity with the ``.tmj`` case in ``test_tmx_refusals``."""
    from warlock.studio.plotter import terrain as terrainlib
    from warlock.studio.tilegrid import blob

    from ._terrainset import terrain_tileset

    generated = terrain_tileset(
        terrains=terrainlib.DEFAULT_TERRAINS[:2], tile_w=8, tile_h=8
    )
    node = tsx.tsx_element(generated, image_name="atlas.png")
    embedded = ET.tostring(node, encoding="unicode").replace(
        "<tileset ", '<tileset firstgid="1" ', 1
    )
    data = (
        '<map version="1.10" orientation="orthogonal" width="1" height="1" '
        f'tilewidth="8" tileheight="8">{embedded}</map>'
    ).encode()
    doc = tmx.read_tmx(
        data,
        image_loader=lambda _s: np.zeros(
            (8 * 2, 8 * blob.TILE_COUNT, 4), dtype=np.uint8
        ),
        tsx_loader=lambda _s: None,
    )
    assert [entry.name for entry in doc.tilesets[0].tileset.terrains] == ["Grass", "Dirt"]


# --- bounded decompression and the id range -----------------------------------
#
# A payload is unpacked against the size the layer *declares*, and every plain
# list of numbers is range-checked as int64 before it is cast. Both refusals
# exist because the alternative is not a wrong picture but a wrong sentence: an
# unbounded ``zlib.decompress`` allocates whatever the stream claims, and a
# uint32 cast turns a hand-edited ``-1`` into a tile id nothing accounts for.


@pytest.mark.parametrize("compression", ["zlib", "gzip"])
def test_a_payload_that_unpacks_past_its_declared_size_is_refused(compression):
    doc = _doc()
    files = tmx.tmx_export(doc)
    fat = b"\0" * (doc.width * doc.height * 4 * 4)
    raw = zlib.compress(fat) if compression == "zlib" else gzip.compress(fat)
    body = (
        f'<data encoding="base64" compression="{compression}">'
        f"{base64.b64encode(raw).decode()}</data>"
    )
    with pytest.raises(ValueError, match="unpacks past"):
        tmx.read_tmx(_with_data(files, body), **_loaders(files))


@pytest.mark.parametrize("value", ["-1", "4294967296"])
def test_a_csv_cell_outside_the_unsigned_32_bit_range_is_refused(value):
    """A ``uint32`` cast wraps both of these silently, and the refusal that
    eventually followed named a tile id the file never wrote."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    row = ",".join(["0"] * 6)
    first = ",".join([value] + ["0"] * 5)
    body = f'<data encoding="csv">\n{first},\n{row},\n{row},\n{row}\n</data>'
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        tmx.read_tmx(_with_data(files, body), **_loaders(files))


def test_csv_that_is_not_numbers_is_refused_as_such():
    doc = _doc()
    files = tmx.tmx_export(doc)
    body = '<data encoding="csv">\nnope,0,0,0,0,0\n</data>'
    with pytest.raises(ValueError, match="not a number"):
        tmx.read_tmx(_with_data(files, body), **_loaders(files))


def test_a_negative_entry_in_a_json_layer_is_refused_too():
    """One model, two spellings -- the TMJ raw list goes through the same
    range check as the CSV one."""
    doc = _doc()
    files = tmx.tmj_export(doc)
    payload = json.loads(files["map.tmj"])
    payload["layers"][0]["data"] = [-1] + [0] * (doc.width * doc.height - 1)
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        tmx.read_tmj(json.dumps(payload).encode(), **_loaders(files))


# --- validation ---------------------------------------------------------------


def test_a_gid_no_tileset_accounts_for_is_refused():
    """Accepting it produces a map with invisible tiles that cannot be
    repainted with what they were, because nothing knows."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    body = '<data encoding="csv">\n999,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0\n</data>'
    with pytest.raises(ValueError, match="accounts for"):
        tmx.read_tmx(_with_data(files, body), **_loaders(files))


def test_a_flipped_gid_still_validates_against_its_range():
    """The mask has to come off before the range check, or every flipped tile
    in a legitimate file reads as out of range."""
    doc = _doc()
    files = tmx.tmx_export(doc)
    value = gid.compose(1, flip_h=True, flip_v=True, flip_d=True)
    row = ",".join(["0"] * 6)
    first = ",".join([str(value)] + ["0"] * 5)
    body = f'<data encoding="csv">\n{first},\n{row},\n{row},\n{row}\n</data>'
    back = tmx.read_tmx(_with_data(files, body), **_loaders(files))
    assert int(_cells(back)[0, 0]) == value


def test_a_layer_with_the_wrong_number_of_cells_is_refused():
    doc = _doc()
    files = tmx.tmx_export(doc)
    with pytest.raises(ValueError, match="carries"):
        tmx.read_tmx(_with_data(files, '<data encoding="csv">1,2,3</data>'), **_loaders(files))


def test_a_file_that_is_not_a_map_is_refused_plainly():
    with pytest.raises(ValueError, match="expected a <map>"):
        tmx.read_tmx(b"<tileset/>", image_loader=lambda s: None, tsx_loader=lambda s: None)
    with pytest.raises(ValueError, match="not a readable"):
        tmx.read_tmj(b"{ not json", image_loader=lambda s: None, tsx_loader=lambda s: None)


# --- locks --------------------------------------------------------------------


def test_an_unlocked_export_carries_no_locked_attribute_at_all():
    """The byte-identity guarantee, and the reason the attribute is written only
    when set: every .tmx and .tmj this editor has ever produced must still come
    out exactly as it did."""
    doc = _doc()
    assert b"locked" not in tmx.tmx_export(doc)["map.tmx"]
    assert b"locked" not in tmx.tmj_export(doc)["map.tmj"]


def test_a_lock_survives_both_formats():
    doc = _doc()
    for layer in doc.layers:
        doc.set_layer_props(layer.uid, locked=True)

    xml_files = tmx.tmx_export(doc)
    assert b'locked="1"' in xml_files["map.tmx"]
    back = tmx.read_tmx(xml_files["map.tmx"], **_loaders(xml_files))
    assert all(layer.locked for layer in back.layers)

    json_files = tmx.tmj_export(doc)
    assert b'"locked": true' in json_files["map.tmj"]
    back = tmx.read_tmj(json_files["map.tmj"], **_loaders(json_files))
    assert all(layer.locked for layer in back.layers)


def test_a_foreign_file_with_no_locked_attribute_opens_unlocked():
    doc = _doc()
    files = tmx.tmx_export(doc)
    back = tmx.read_tmx(files["map.tmx"], **_loaders(files))
    assert all(layer.locked is False for layer in back.layers)


def test_only_the_locked_layer_carries_the_attribute():
    """One locked layer must not stamp the key onto its neighbours."""
    doc = _doc()
    doc.set_layer_props(doc.layers[0].uid, locked=True)
    files = tmx.tmx_export(doc)
    assert files["map.tmx"].count(b'locked="1"') == 1
    back = tmx.read_tmx(files["map.tmx"], **_loaders(files))
    assert [layer.locked for layer in back.layers] == [True, False]


def test_an_image_layer_name_collision_never_clobbers_the_other_layers_bytes():
    """Layer B's fallback name is exactly layer A's declared safe source, so
    the collision branch used to recompute the *same* string and overwrite A's
    bytes -- the export showed one layer's picture on both."""
    from warlock.studio.plotter.pngio import png_bytes

    doc = MapDoc(4, 4, 16, 16)
    a_pixels = _pixels(8, 8)
    b_pixels = _pixels(8, 8)
    b_pixels[1, 1] = (11, 22, 33, 255)
    doc.add_image_layer("first", pixels=a_pixels, source="images/01-image.png")
    doc.add_image_layer("image", pixels=b_pixels, source="")

    files = tmx.tmx_export(doc)
    root = ET.fromstring(files["map.tmx"])
    sources = [
        node.find("image").get("source") for node in root.findall("imagelayer")
    ]
    assert len(set(sources)) == 2, "the two layers must reference two files"
    assert files[sources[0]] == png_bytes(a_pixels)
    assert files[sources[1]] == png_bytes(b_pixels)
