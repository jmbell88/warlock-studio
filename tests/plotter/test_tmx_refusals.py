"""One case per Tiled feature this editor refuses.

The doctrine, stated as a test matrix: a file using something Plotter does not
model is refused *by name*, not loaded with half of it silently dropped. The
drop is invisible right up to the moment the user saves, at which point the
other half is gone.

Every case asserts the message names the feature, because a refusal that does
not say what to remove sends the user to a forum rather than to Tiled.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from warlock.studio.plotter import tilemap, tmx, tsx
from warlock.studio.plotter.tileset import Tileset


def _pixels() -> np.ndarray:
    array = np.zeros((32, 32, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def _tsx_loader(_source: str) -> Tileset:
    return Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16)


def _image_loader(_source: str) -> np.ndarray:
    return _pixels()


LOADERS = {"image_loader": _image_loader, "tsx_loader": _tsx_loader}

_EMPTY_LAYER = (
    '<layer id="1" name="L" width="2" height="2">'
    '<data encoding="csv">0,0,\n0,0</data></layer>'
)


def _map(attrs: str = "", body: str = _EMPTY_LAYER) -> bytes:
    return (
        f'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        f'tilewidth="16" tileheight="16" {attrs}>'
        f'<tileset firstgid="1" source="t.tsx"/>{body}</map>'
    ).encode()


def _refuses(data: bytes, feature: str) -> None:
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmx(data, **LOADERS)
    assert feature in str(exc.value)


# --- the map itself -----------------------------------------------------------


@pytest.mark.parametrize("orientation", ["staggered", "hexagonal"])
def test_a_staggered_or_hexagonal_map_is_refused(orientation):
    """The two grids left in the list, and the reason ``gid`` carries no
    hexagonal rotation bit: a file that could set one never gets past here."""
    data = _map().replace(b'orientation="orthogonal"', f'orientation="{orientation}"'.encode())
    _refuses(data, orientation)


def test_an_isometric_map_loads_now_that_plotter_draws_one():
    """Isometric left the refusal list *because the editor learned to draw it*,
    which is the only reason a refusal is ever allowed to move. Asserted here,
    beside the two that stayed, so the removal reads as a decision rather than
    as a case somebody deleted."""
    data = _map().replace(b'orientation="orthogonal"', b'orientation="isometric"')
    doc = tmx.read_tmx(data, **LOADERS)
    assert doc.projection == "isometric"
    assert b'orientation="isometric"' in tmx.tmx_export(doc)["map.tmx"]


def test_an_infinite_map_is_refused():
    _refuses(_map(attrs='infinite="1"'), "infinite map")


def test_a_chunked_json_layer_is_refused():
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 2,
        "height": 2,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [{"type": "tilelayer", "name": "L", "chunks": [{"data": [0]}]}],
    }
    with pytest.raises(tsx.TiledUnsupported, match="chunked"):
        tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)


# --- layers -------------------------------------------------------------------


def test_group_layers_are_refused():
    _refuses(_map(body=_EMPTY_LAYER + '<group id="9" name="G"/>'), "group layers")


def test_image_layers_are_refused():
    _refuses(_map(body=_EMPTY_LAYER + '<imagelayer id="9" name="I"/>'), "image layers")


@pytest.mark.parametrize("attr", ["offsetx", "offsety"])
def test_a_layer_with_a_pixel_offset_is_refused(attr):
    body = _EMPTY_LAYER.replace('name="L"', f'name="L" {attr}="8"')
    _refuses(_map(body=body), "layer pixel offsets")


def test_zstd_layer_data_is_refused_with_the_remedy_named():
    """Refused rather than supported, which is what keeps this package's
    dependency set to numpy and the standard library."""
    body = (
        '<layer id="1" name="L" width="2" height="2">'
        '<data encoding="base64" compression="zstd">AAAA</data></layer>'
    )
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmx(_map(body=body), **LOADERS)
    assert "zstd" in str(exc.value) and "zlib" in str(exc.value)


def test_an_unknown_encoding_is_refused():
    body = (
        '<layer id="1" name="L" width="2" height="2">'
        '<data encoding="rot13">x</data></layer>'
    )
    _refuses(_map(body=body), "rot13")


def test_an_unknown_json_layer_type_is_refused():
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 2,
        "height": 2,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [{"type": "somethingnew", "name": "L"}],
    }
    with pytest.raises(tsx.TiledUnsupported, match="somethingnew"):
        tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)


# --- objects ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "feature"),
    [
        ("<ellipse/>", "ellipse objects"),
        ('<polygon points="0,0 1,1"/>', "polygon objects"),
        ('<polyline points="0,0 1,1"/>', "polyline objects"),
        ('<text>hi</text>', "text objects"),
    ],
)
def test_an_object_shape_that_cannot_be_drawn_is_refused(shape, feature):
    body = f'<objectgroup id="2" name="O"><object id="1" x="0" y="0">{shape}</object></objectgroup>'
    _refuses(_map(body=_EMPTY_LAYER + body), feature)


def test_a_tile_object_is_refused():
    body = '<objectgroup id="2" name="O"><object id="1" gid="1" x="0" y="0"/></objectgroup>'
    _refuses(_map(body=_EMPTY_LAYER + body), "tile objects")


def test_a_templated_object_is_refused():
    body = (
        '<objectgroup id="2" name="O">'
        '<object id="1" template="tree.tx" x="0" y="0"/></objectgroup>'
    )
    _refuses(_map(body=_EMPTY_LAYER + body), "object templates")


def test_a_rotated_object_is_refused():
    """An unrotated outline drawn for a rotated object is a *wrong* picture,
    and a wrong picture is worse than a refusal. Contrast a hidden object,
    which is modelled: hiding it changes nothing about where it is."""
    body = (
        '<objectgroup id="2" name="O">'
        '<object id="1" x="0" y="0" rotation="45"/></objectgroup>'
    )
    _refuses(_map(body=_EMPTY_LAYER + body), "rotated objects")


@pytest.mark.parametrize(
    ("shape", "feature"),
    [
        ({"ellipse": True}, "ellipse objects"),
        ({"polygon": [{"x": 0, "y": 0}]}, "polygon objects"),
        ({"polyline": [{"x": 0, "y": 0}]}, "polyline objects"),
        ({"text": {"text": "hi"}}, "text objects"),
        ({"gid": 3}, "tile objects"),
        ({"template": "tree.tj"}, "object templates"),
        ({"rotation": 90}, "rotated objects"),
    ],
)
def test_the_json_reader_refuses_exactly_the_same_object_shapes(shape, feature):
    """One model, two spellings. If the two lists drifted, one format would
    accept a file the editor cannot draw."""
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 2,
        "height": 2,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [
            {"type": "objectgroup", "name": "O", "objects": [{"id": 1, "x": 0, "y": 0, **shape}]}
        ],
    }
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)
    assert feature in str(exc.value)


# --- tilesets -----------------------------------------------------------------


def test_an_embedded_image_collection_tileset_is_refused():
    data = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        b'tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" name="c" tilewidth="16" tileheight="16">'
        b'<tile id="0"><image source="a.png"/></tile></tileset></map>'
    )
    _refuses(data, "image-collection")


def test_an_embedded_tileset_with_wangsets_is_refused():
    data = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        b'tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" name="c" tilewidth="16" tileheight="16">'
        b'<image source="a.png" width="32" height="32"/><wangsets/></tileset></map>'
    )
    _refuses(data, "Wang sets")


def _tmj(tileset: dict) -> bytes:
    return json.dumps(
        {
            "type": "map",
            "orientation": "orthogonal",
            "width": 2,
            "height": 2,
            "tilewidth": 16,
            "tileheight": 16,
            "tilesets": [{"firstgid": 1, **tileset}],
            "layers": [],
        }
    ).encode()


def _blob_wangset(colours: list[dict]) -> dict:
    """The set ``tsx.write_wangsets`` emits, in Tiled's JSON spelling."""
    from warlock.studio.plotter import blob as bloblib

    tiles = []
    for index in range(len(colours)):
        for case, mask in enumerate(bloblib.BLOB_MASKS):
            tiles.append(
                {
                    "tileid": index * bloblib.TILE_COUNT + case,
                    "wangid": [
                        index + 1 if mask & bit else 0
                        for bit in (
                            bloblib.N,
                            bloblib.NE,
                            bloblib.E,
                            bloblib.SE,
                            bloblib.S,
                            bloblib.SW,
                            bloblib.W,
                            bloblib.NW,
                        )
                    ],
                }
            )
    return {"name": "Terrain", "type": "mixed", "colors": colours, "wangtiles": tiles}


def test_a_tmj_wangset_this_build_wrote_is_recognised_rather_than_refused():
    """Recognise-or-refuse, the rule the XML side already followed. The JSON
    side used to refuse *every* wangset, so a ``.tmj`` carrying a set this build
    had itself written was turned away."""
    from warlock.studio.plotter import blob as bloblib

    colours = [{"name": "Grass", "color": "#6a994e"}, {"name": "Sand", "color": "#d6c384"}]
    doc = tmx.read_tmj(
        _tmj(
            {
                "name": "ground",
                "image": "a.png",
                "tilewidth": 16,
                "tileheight": 16,
                "wangsets": [_blob_wangset(colours)],
            }
        ),
        # One row per terrain, 47 columns of blob cases: the geometry a terrain
        # set declares and ``Tileset`` refuses a declaration that denies.
        image_loader=lambda _s: np.zeros(
            (16 * 2, 16 * bloblib.TILE_COUNT, 4), dtype=np.uint8
        ),
        tsx_loader=_tsx_loader,
    )
    assert [entry.name for entry in doc.tilesets[0].tileset.terrains] == ["Grass", "Sand"]
    assert doc.tilesets[0].tileset.is_terrain_set


def test_a_foreign_tmj_wangset_is_refused_and_says_what_is_modelled():
    """Corner-only sets, 255 colours, tiles that form no blob: adopting one
    would be the silent half-read the reader exists to prevent."""
    from warlock.studio.plotter import blob as bloblib

    foreign = {
        "name": "Corners",
        "type": "corner",
        "colors": [{"name": "Grass", "color": "#6a994e"}],
        "wangtiles": [{"tileid": 0, "wangid": [1, 0, 0, 0, 0, 0, 0, 0]}],
    }
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(
            _tmj({"name": "g", "image": "a.png", "wangsets": [foreign]}), **LOADERS
        )
    assert "Wang sets" in str(exc.value)
    assert str(bloblib.TILE_COUNT) in str(exc.value)


def test_an_external_tsj_tileset_is_refused_with_the_remedy():
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 2,
        "height": 2,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [{"firstgid": 1, "source": "t.tsj"}],
        "layers": [],
    }
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)
    assert ".tsj" in str(exc.value) and ".tsx" in str(exc.value)


def test_an_external_tsj_tileset_on_the_xml_path_is_refused_with_the_remedy():
    """The TMX spelling of the same file: ``<tileset source="x.tsj"/>``. Before
    this refusal existed, the XML path fell through to ``tsx_loader`` and died
    with the host's generic "not a readable tileset" -- the right outcome, but
    the wrong sentence, and one that does not say to re-save as ``.tsx``."""
    data = _map().replace(b'source="t.tsx"', b'source="t.tsj"')
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmx(data, **LOADERS)
    assert ".tsj" in str(exc.value) and ".tsx" in str(exc.value)


# --- embedded-JSON tileset feature checks --------------------------------------
#
# The XML embedded-tileset path runs every one of these through
# ``check_tileset_features``; the JSON embedded-tileset path used to run a
# single blanket check (``entry.get("tiles") or entry.get("grid")``) that
# mislabelled per-tile animation/collision/properties as "an image-collection
# tileset", and let ``terrains`` (the JSON spelling of ``<terraintypes>``) slip
# past into an unrelated ``ValueError`` from the tile-size arithmetic further
# down. Each case below pins the correct, XML-matching refusal.


def test_a_tmj_embedded_tileset_with_per_tile_animation_is_refused():
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [{"id": 0, "animation": [{"tileid": 0, "duration": 100}]}],
        }
    )
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(data, **LOADERS)
    assert "per-tile animation" in str(exc.value)


def test_a_tmj_embedded_tileset_with_a_per_tile_image_is_refused_as_a_collection():
    data = _tmj(
        {
            "name": "g",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [{"id": 0, "image": "0.png"}],
        }
    )
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(data, **LOADERS)
    assert "image-collection" in str(exc.value)


def test_a_tmj_embedded_tileset_with_per_tile_collision_is_refused():
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [{"id": 0, "objectgroup": {"objects": []}}],
        }
    )
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(data, **LOADERS)
    assert "per-tile collision" in str(exc.value)


def test_a_tmj_embedded_tileset_with_per_tile_properties_is_refused():
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [{"id": 0, "properties": [{"name": "p", "type": "bool", "value": True}]}],
        }
    )
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(data, **LOADERS)
    assert "per-tile custom properties" in str(exc.value)


def test_a_tmj_embedded_tileset_with_terrain_types_is_refused():
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "terrains": [{"name": "Grass", "tile": 0}],
        }
    )
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(data, **LOADERS)
    assert "terrain types" in str(exc.value)


def test_a_tmj_embedded_tileset_with_a_grid_key_is_not_refused():
    """``grid`` marks an isometric tileset's own rendering grid, not an image
    collection. ``check_tileset_features`` -- the XML path's version of these
    checks -- has no test for it at all, so an XML embedded tileset carrying
    ``<grid/>`` beside its ``<image>`` already loads cleanly. The JSON path
    used to refuse on ``grid`` alone (the old blanket ``entry.get("tiles") or
    entry.get("grid")`` check), which is a case the two per-format readers
    were never supposed to disagree about. Asserted here, beside the
    refusals that stayed, so the removal reads as a decision rather than a
    case somebody deleted -- the same pattern
    ``test_an_isometric_map_loads_now_that_plotter_draws_one`` follows above."""
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "grid": {"orientation": "isometric", "width": 16, "height": 8},
        }
    )
    doc = tmx.read_tmj(data, **LOADERS)
    assert doc.tilesets[0].tileset.name == "g"


# --- properties ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "raw", "expected"),
    [("file", "art/hero.png", "art/hero.png"), ("object", "7", 7)],
)
def test_a_file_or_object_property_loads_now_that_plotter_models_one(kind, raw, expected):
    """These two left the refusal list *because the model gained a place to
    put them*, which is the only reason a refusal is ever allowed to move --
    the isometric case above, applied to properties. Flipped in place rather
    than deleted, so the removal reads as a decision."""
    body = _EMPTY_LAYER.replace(
        "<data", f'<properties><property name="p" type="{kind}" value="{raw}"/></properties><data'
    )
    doc = tmx.read_tmx(_map(body=body), **LOADERS)
    assert doc.layers[0].properties["p"] == tsx.Prop(kind, expected)
    again = tmx.read_tmx(tmx.tmx_export(doc)["map.tmx"], **LOADERS)
    assert again.layers[0].properties == doc.layers[0].properties


def test_a_class_property_loads_with_its_members_and_its_type_name():
    """The recursive one. ``propertytype`` names a class declared in a Tiled
    *project*, which this editor never reads -- so it is carried verbatim
    rather than validated, and every member carries its own type."""
    block = (
        '<properties><property name="npc" type="class" propertytype="NPC">'
        '<properties><property name="hp" type="int" value="3"/>'
        '<property name="tint" type="color" value="#ff00ff00"/></properties>'
        "</property></properties>"
    )
    doc = tmx.read_tmx(_map(body=_EMPTY_LAYER.replace("<data", block + "<data")), **LOADERS)
    npc = doc.layers[0].properties["npc"]
    assert npc.propertytype == "NPC"
    assert npc.value == {"hp": tsx.Prop("int", 3), "tint": tsx.Prop("color", "#ff00ff00")}
    again = tmx.read_tmx(tmx.tmx_export(doc)["map.tmx"], **LOADERS)
    assert again.layers[0].properties == doc.layers[0].properties


def test_the_new_property_types_survive_the_json_round_trip_too():
    """One model, two syntaxes: what the XML reader accepts the JSON reader
    accepts, and both writers write it back. The ``color`` member is left out
    of the class here on purpose -- Tiled's JSON stores class members
    untyped, and ``test_props.py`` pins that one documented loss."""
    block = (
        '<properties><property name="art" type="file" value="a/b.png"/>'
        '<property name="target" type="object" value="4"/>'
        '<property name="npc" type="class" propertytype="NPC">'
        '<properties><property name="hp" type="int" value="3"/></properties>'
        "</property></properties>"
    )
    doc = tmx.read_tmx(_map(body=_EMPTY_LAYER.replace("<data", block + "<data")), **LOADERS)
    again = tmx.read_tmj(tmx.tmj_export(doc)["map.tmj"], **LOADERS)
    assert again.layers[0].properties == doc.layers[0].properties


def test_a_property_type_outside_tileds_nine_is_still_refused():
    body = _EMPTY_LAYER.replace(
        "<data", '<properties><property name="p" type="vector2" value="1,2"/></properties><data'
    )
    _refuses(_map(body=body), "vector2")


def test_a_list_property_is_refused_because_tiled_has_no_syntax_for_one():
    """Modelled in the document and stored in a ``.wmap``, refused at the
    Tiled door: Tiled 1.12.2 has no list-valued property, so there is nothing
    to write that it would read back, and inventing a syntax would produce a
    file only this editor can open. See ``tests/plotter/test_props.py``."""
    body = _EMPTY_LAYER.replace(
        "<data", '<properties><property name="bag" type="list" value="1"/></properties><data'
    )
    _refuses(_map(body=body), "list-valued custom property")


def test_an_embedded_tileset_image_with_no_path_is_refused():
    data = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        b'tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" name="c" tilewidth="16" tileheight="16">'
        b"<image/></tileset></map>"
    )
    _refuses(data, "embedded tileset image")


# --- the XML door -------------------------------------------------------------
#
# Not ``TiledUnsupported``: a DTD is not a Tiled feature this editor declines to
# model, it is a shape the parser must never be handed. ``ExpatParser`` expands
# internal entities, so the refusal has to happen before ``fromstring`` rather
# than under any ceiling downstream.

_LAUGHS = (
    '<!DOCTYPE map [<!ENTITY a "AAAAAAAAAA">'
    '<!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
    '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]>'
)


def test_a_tmx_declaring_a_dtd_is_refused_before_it_is_parsed():
    data = (_LAUGHS + '<map version="1.10" orientation="orthogonal" width="1" '
            'height="1" tilewidth="16" tileheight="16">&c;</map>').encode()
    with pytest.raises(ValueError, match="DTD"):
        tmx.read_tmx(data, **LOADERS)


def test_a_tsx_declaring_a_dtd_is_refused_by_the_same_door():
    """One door for both readers, which is the point of sharing it: a second
    copy is a copy with no lock on it."""
    data = (_LAUGHS + '<tileset name="t" tilewidth="16" tileheight="16">'
            '<image source="a.png"/></tileset>').encode()
    with pytest.raises(ValueError, match="DTD"):
        tsx.tsx_source(data)


# --- the writer door ----------------------------------------------------------
#
# The document models rotation, a draw order and five object shapes that
# neither exporter can yet spell, so the same list is refused on the way *out*.
# A silent half-write is worse than the half-read every case above forbids: the
# user keeps their document, and the file they just handed to an engine is
# quietly wrong with nothing saying so. Both doors flip together in M3.


def _exportable_doc() -> tilemap.MapDoc:
    """A map that exports cleanly, so each case below changes one thing."""
    doc = tilemap.MapDoc(2, 2, 16, 16)
    doc.add_tileset(Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16))
    doc.add_tile_layer("L")
    doc.add_object_layer("O")
    return doc


def _refuses_export(doc: tilemap.MapDoc, feature: str) -> None:
    """Both writers, one assertion: a refusal only one of them made would let
    a user reach the same broken file by picking the other format."""
    for export in (tmx.tmx_export, tmx.tmj_export):
        with pytest.raises(tsx.TiledUnsupported) as exc:
            export(doc)
        assert exc.value.feature == feature
        assert feature in str(exc.value)
        assert "this map uses" in str(exc.value), "an export refusal is not about a file"


def test_the_control_map_exports_from_both_writers():
    """The control, first: everything below refuses, so this is what says the
    writer door is not simply refusing everything."""
    doc = _exportable_doc()
    doc.add_object(doc.layers[1].uid, tilemap.MapObject(uid=tilemap.new_uid(), kind="rect"))
    assert tmx.tmx_export(doc) and tmx.tmj_export(doc)


def test_exporting_a_rotated_object_is_refused():
    doc = _exportable_doc()
    doc.add_object(
        doc.layers[1].uid,
        tilemap.MapObject(uid=tilemap.new_uid(), kind="rect", rotation=45.0),
    )
    _refuses_export(doc, "rotated objects")


@pytest.mark.parametrize(
    ("shape", "feature"),
    [
        (tilemap.Ellipse(4, 4), "ellipse objects"),
        (tilemap.Polygon(((0, 0), (4, 0), (4, 4))), "polygon objects"),
        (tilemap.Polyline(((0, 0), (4, 0))), "polyline objects"),
        (tilemap.Text("hi"), "text objects"),
        (tilemap.TileShape(gid=1, w=16, h=16), "tile objects"),
    ],
)
def test_exporting_a_shape_no_writer_can_spell_is_refused(shape, feature):
    """The same five sentences the readers refuse these with, out of one table
    -- two doors on one limit, not two features that share a name."""
    doc = _exportable_doc()
    doc.add_object(doc.layers[1].uid, tilemap.MapObject(uid=tilemap.new_uid(), shape=shape))
    _refuses_export(doc, feature)


def test_exporting_an_index_ordered_object_layer_is_refused():
    """Not merely an attribute lost: ``"index"`` means the list order *is* the
    stacking order, so flattening it to ``"topdown"`` changes which object is
    drawn on top."""
    doc = _exportable_doc()
    doc.set_layer_props(doc.layers[1].uid, draworder="index")
    _refuses_export(doc, "an index-ordered object layer")


def test_a_default_object_layer_still_exports():
    """The other half of the draworder case: ``"topdown"`` is the default and
    must not have become a refusal."""
    doc = _exportable_doc()
    assert doc.layers[1].draworder == "topdown"
    assert tmx.tmx_export(doc) and tmx.tmj_export(doc)


# --- what is *not* refused ----------------------------------------------------


def test_a_hidden_object_is_modelled_rather_than_refused():
    body = (
        '<objectgroup id="2" name="O">'
        '<object id="1" name="ghost" x="1" y="2" visible="0"/></objectgroup>'
    )
    doc = tmx.read_tmx(_map(body=_EMPTY_LAYER + body), **LOADERS)
    obj = doc.layers[1].objects[0]
    assert obj.visible is False and (obj.x, obj.y) == (1.0, 2.0)


def test_a_plain_orthogonal_map_loads():
    """The control: everything above refuses, so this says the refusals are not
    simply refusing everything."""
    doc = tmx.read_tmx(_map(), **LOADERS)
    assert (doc.width, doc.height) == (2, 2)
    assert [layer.name for layer in doc.layers] == ["L"]
