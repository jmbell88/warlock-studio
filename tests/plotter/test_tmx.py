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

from warlock.studio.plotter import gid, tmx, tsx
from warlock.studio.plotter.tilemap import MapDoc, MapObject, new_uid
from warlock.studio.plotter.tileset import Tileset


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
