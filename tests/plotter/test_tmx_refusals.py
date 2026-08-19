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

from warlock.studio.plotter import tilemap, tmx, tsx, wmap
from warlock.studio.tilegrid.tileset import Tileset


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
def test_a_staggered_or_hexagonal_map_loads_now_that_plotter_projects_one(
    orientation,
):
    """Both left the refusal list the only way a refusal is ever allowed to
    move: the editor learned to place their cells. Asserted here, where the
    refusal used to be, so the removal reads as a decision."""
    data = _map().replace(
        b'orientation="orthogonal"', f'orientation="{orientation}"'.encode()
    )
    doc = tmx.read_tmx(data, **LOADERS)
    assert doc.projection == orientation


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


def test_group_layers_are_modelled_recursively():
    body = '<group id="9" name="G">' + _EMPTY_LAYER + "</group>"
    doc = tmx.read_tmx(_map(body=body), **LOADERS)
    assert isinstance(doc.layers[0], tilemap.GroupLayer)
    assert isinstance(doc.layers[0].children[0], tilemap.TileLayer)


def test_an_empty_image_layer_is_modelled():
    doc = tmx.read_tmx(
        _map(body=_EMPTY_LAYER + '<imagelayer id="9" name="I"/>'), **LOADERS
    )
    assert isinstance(doc.layers[1], tilemap.ImageLayer)
    assert doc.layers[1].pixels.shape == (0, 0, 4)


@pytest.mark.parametrize("attr", ["offsetx", "offsety"])
def test_a_layer_with_a_pixel_offset_is_modelled(attr):
    body = _EMPTY_LAYER.replace('name="L"', f'name="L" {attr}="8"')
    layer = tmx.read_tmx(_map(body=body), **LOADERS).layers[0]
    assert getattr(layer, "offset_x" if attr == "offsetx" else "offset_y") == 8.0


def test_zstd_layer_data_reads_the_same_cells_as_its_zlib_twin():
    """Read but never written: every Tiled reads zlib, so writing zstd would
    buy nothing and cost a reader. The claim the row makes is read-without-loss,
    and this is that claim."""
    import base64
    import zlib

    import numpy as np
    import zstandard

    from warlock.studio.tilegrid import gid

    cells = np.array([[1, 2], [3, 4]], gid.DTYPE)
    raw = cells.astype("<u4").tobytes()

    def _map_with(payload: bytes, compression: str) -> bytes:
        return _map(
            body=(
                '<layer id="1" name="L" width="2" height="2">'
                f'<data encoding="base64" compression="{compression}">'
                f"{base64.b64encode(payload).decode()}</data></layer>"
            )
        )

    zlibbed = tmx.read_tmx(_map_with(zlib.compress(raw), "zlib"), **LOADERS)
    zstdded = tmx.read_tmx(
        _map_with(zstandard.ZstdCompressor().compress(raw), "zstd"), **LOADERS
    )
    assert np.array_equal(
        zstdded.tile_layers()[0].data, zlibbed.tile_layers()[0].data
    )


def test_an_unknown_compression_is_still_refused():
    body = (
        '<layer id="1" name="L" width="2" height="2">'
        '<data encoding="base64" compression="brotli">AAAA</data></layer>'
    )
    with pytest.raises(tsx.TiledUnsupported, match="brotli"):
        tmx.read_tmx(_map(body=body), **LOADERS)


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
    ("shape", "kind"),
    [
        ("<ellipse/>", "ellipse"),
        ('<capsule/>', "capsule"),
        ('<polygon points="0,0 1,1 2,0"/>', "polygon"),
        ('<polyline points="0,0 1,1"/>', "polyline"),
        ('<text>hi</text>', "text"),
    ],
)
def test_every_xml_object_shape_is_modelled(shape, kind):
    body = f'<objectgroup id="2" name="O"><object id="1" x="0" y="0">{shape}</object></objectgroup>'
    doc = tmx.read_tmx(_map(body=_EMPTY_LAYER + body), **LOADERS)
    assert doc.layers[1].objects[0].kind == kind


def test_a_tile_object_is_modelled():
    body = '<objectgroup id="2" name="O"><object id="1" gid="1" x="0" y="0"/></objectgroup>'
    doc = tmx.read_tmx(_map(body=_EMPTY_LAYER + body), **LOADERS)
    assert doc.layers[1].objects[0].kind == "tile"


def test_a_templated_object_is_refused():
    body = (
        '<objectgroup id="2" name="O">'
        '<object id="1" template="tree.tx" x="0" y="0"/></objectgroup>'
    )
    _refuses(_map(body=_EMPTY_LAYER + body), "object templates")


def test_a_rotated_object_is_modelled():
    body = (
        '<objectgroup id="2" name="O">'
        '<object id="1" x="0" y="0" rotation="45"/></objectgroup>'
    )
    doc = tmx.read_tmx(_map(body=_EMPTY_LAYER + body), **LOADERS)
    assert doc.layers[1].objects[0].rotation == 45.0


@pytest.mark.parametrize(
    ("shape", "kind"),
    [
        ({"ellipse": True}, "ellipse"),
        ({"capsule": True}, "capsule"),
        ({"polygon": [{"x": 0, "y": 0}, {"x": 1, "y": 1}, {"x": 2, "y": 0}]}, "polygon"),
        ({"polyline": [{"x": 0, "y": 0}, {"x": 1, "y": 1}]}, "polyline"),
        ({"text": {"text": "hi"}}, "text"),
        ({"gid": 3}, "tile"),
        ({"rotation": 90}, "rect"),
    ],
)
def test_the_json_reader_models_the_same_object_shapes(shape, kind):
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
    doc = tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)
    assert doc.layers[0].objects[0].kind == kind
    if "rotation" in shape:
        assert doc.layers[0].objects[0].rotation == 90.0


def test_a_templated_object_is_refused_on_the_json_path_too():
    """The XML half of this pair is above. This file's header says the two
    lists must not drift, and this one had no JSON case at all -- so a reader
    that learned to ignore ``template`` in ``.tmj`` alone would have loaded a
    map with an object whose real geometry lives in a file nothing opened."""
    payload = {
        "type": "map",
        "orientation": "orthogonal",
        "width": 2,
        "height": 2,
        "tilewidth": 16,
        "tileheight": 16,
        "tilesets": [{"firstgid": 1, "source": "t.tsx"}],
        "layers": [
            {
                "type": "objectgroup",
                "name": "O",
                "objects": [{"id": 1, "x": 0, "y": 0, "template": "tree.tx"}],
            }
        ],
    }
    with pytest.raises(tmx.TiledUnsupported) as excinfo:
        tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)
    assert excinfo.value.feature == "object templates"


def test_a_tile_object_with_the_hex_rotation_bit_is_refused_by_name():
    """The same probe the layer-data pass makes, on the other gid in a
    document. ``gid.decompose`` strips the three transform flags and leaves bit
    28, so without this the object arrived at the accounting check as tile
    268435457 and was refused as "a tile no tileset accounts for" -- true, and
    about the wrong problem."""
    hex_gid = 1 | 0x10000000
    body = (
        f'<objectgroup id="2" name="O"><object id="1" gid="{hex_gid}" '
        'x="0" y="0"/></objectgroup>'
    )
    _refuses(_map(body=_EMPTY_LAYER + body), "hexagonal 120-degree tile rotation")


# --- tilesets -----------------------------------------------------------------


def test_an_embedded_image_collection_tileset_is_read():
    """Modelled rather than refused: the host fetches each tile's image and the
    engine composes them into one backing atlas."""
    data = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        b'tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" name="c" tilewidth="16" tileheight="16">'
        b'<tile id="0"><image source="a.png"/></tile>'
        b'<tile id="3"><image source="a.png"/></tile></tileset>'
        b'<layer id="1" name="L" width="2" height="2">'
        b'<data encoding="csv">1,4,0,0</data></layer></map>'
    )
    doc = tmx.read_tmx(data, **LOADERS)
    ts = doc.tilesets[0].tileset
    assert ts.is_collection
    assert ts.collection.ids == (0, 3)
    # The sparse id is reachable: gid 4 is local 3, which a count-derived range
    # would have refused.
    assert doc.tilesets[0].holds(4)


def test_an_embedded_tileset_with_an_empty_wangsets_block_is_read():
    """Recognise-or-refuse ended with the general model: a set that is not this
    editor's blob preset is read as data rather than turned away, and an empty
    block is simply no sets at all."""
    data = (
        b'<map version="1.10" orientation="orthogonal" width="2" height="2" '
        b'tilewidth="16" tileheight="16">'
        b'<tileset firstgid="1" name="c" tilewidth="16" tileheight="16">'
        b'<image source="a.png" width="32" height="32"/><wangsets/></tileset></map>'
    )
    doc = tmx.read_tmx(data, **LOADERS)
    ts = doc.tilesets[0].tileset
    assert ts.terrains == ()
    assert ts.wangsets == ()


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
    from warlock.studio.tilegrid import blob as bloblib

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
    from warlock.studio.tilegrid import blob as bloblib

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


def test_a_foreign_tmj_wangset_is_read_as_data():
    """Corner-only sets, 255 colours, tiles that form no blob -- all of it is
    modelled now, and painted by constraint matching rather than by the blob
    collapse. The preset keeps its own positional path untouched."""
    foreign = {
        "name": "Corners",
        "type": "corner",
        "colors": [{"name": "Grass", "color": "#6a994e", "probability": 2}],
        "wangtiles": [{"tileid": 0, "wangid": [0, 1, 0, 1, 0, 1, 0, 1]}],
    }
    doc = tmx.read_tmj(
        _tmj(
            {
                "name": "g",
                "image": "a.png",
                "tilewidth": 16,
                "tileheight": 16,
                "wangsets": [foreign],
            }
        ),
        **LOADERS,
    )
    ts = doc.tilesets[0].tileset
    assert ts.terrains == (), "a foreign set is not the blob preset"
    assert len(ts.wangsets) == 1
    got = ts.wangsets[0]
    assert (got.name, got.kind) == ("Corners", "corner")
    assert got.colours[0].probability == 2.0
    assert got.tiles == {0: (0, 1, 0, 1, 0, 1, 0, 1)}


def test_an_external_tsj_tileset_goes_to_the_loader_like_a_tsx():
    """Which spelling a reference names is the *host*'s question -- only the
    host reads bytes, and the engine has no way to tell them apart that is not
    the extension it is passing over."""
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
    doc = tmx.read_tmj(json.dumps(payload).encode(), **LOADERS)
    assert doc.tilesets[0].source == "t.tsj"


def test_an_external_tsj_tileset_on_the_xml_path_goes_to_the_loader_too():
    """The TMX spelling of the same file: ``<tileset source="x.tsj"/>``. Both
    spellings hand the reference straight over, and both remember it."""
    data = _map().replace(b'source="t.tsx"', b'source="t.tsj"')
    doc = tmx.read_tmx(data, **LOADERS)
    assert doc.tilesets[0].source == "t.tsj"


# --- embedded-JSON tileset feature checks --------------------------------------
#
# The XML embedded-tileset path runs every one of these through
# ``check_tileset_features``; the JSON embedded-tileset path used to run a
# single blanket check (``entry.get("tiles") or entry.get("grid")``) that
# mislabelled per-tile animation/collision/properties as "an image-collection
# tileset", and let ``terrains`` (the JSON spelling of ``<terraintypes>``) slip
# past into an unrelated ``ValueError`` from the tile-size arithmetic further
# down. Each case below pins the correct, XML-matching answer -- which for the
# three per-tile metadata cases is now an *acceptance*, since the model landed:
# the ledger rule says a refusal only moves alongside the reader, the writer and
# the swapped case, and these are the swapped cases.


def test_a_tmj_embedded_tileset_with_per_tile_animation_is_read():
    """Moved from refused to accepted with the per-tile metadata model."""
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [{"id": 0, "animation": [{"tileid": 1, "duration": 120}]}],
        }
    )
    doc = tmx.read_tmj(data, **LOADERS)
    frames = doc.tilesets[0].tileset.meta_of(0).animation
    assert [(f.local_id, f.duration_ms) for f in frames] == [(1, 120)]


def test_a_tmj_embedded_image_collection_is_read():
    data = _tmj(
        {
            "name": "g",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [{"id": 0, "image": "0.png"}, {"id": 7, "image": "0.png"}],
        }
    )
    doc = tmx.read_tmj(data, **LOADERS)
    ts = doc.tilesets[0].tileset
    assert ts.is_collection
    assert ts.collection.ids == (0, 7)


def test_a_tmj_tileset_with_no_pixels_at_all_is_still_refused():
    data = _tmj({"name": "g", "tilewidth": 16, "tileheight": 16})
    with pytest.raises(tsx.TiledUnsupported) as exc:
        tmx.read_tmj(data, **LOADERS)
    assert "embedded tileset image" in str(exc.value)


def test_a_tmj_embedded_tileset_with_per_tile_collision_is_read():
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [
                {
                    "id": 0,
                    "objectgroup": {
                        "objects": [{"id": 1, "x": 1, "y": 2, "width": 8, "height": 9}]
                    },
                }
            ],
        }
    )
    doc = tmx.read_tmj(data, **LOADERS)
    shapes = doc.tilesets[0].tileset.meta_of(0).collision
    assert len(shapes) == 1
    assert (shapes[0].x, shapes[0].w) == (1.0, 8.0)


def test_a_tmj_embedded_tileset_with_per_tile_properties_is_read():
    data = _tmj(
        {
            "name": "g",
            "image": "a.png",
            "tilewidth": 16,
            "tileheight": 16,
            "tiles": [{"id": 0, "properties": [{"name": "p", "type": "bool", "value": True}]}],
        }
    )
    doc = tmx.read_tmj(data, **LOADERS)
    assert doc.tilesets[0].tileset.meta_of(0).properties["p"].value is True


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


def test_a_tiled_112_list_property_round_trips_recursively():
    body = _EMPTY_LAYER.replace(
        "<data",
        '<properties><property name="bag" type="list">'
        '<item type="int" value="1"/><item type="string" value="torch"/>'
        '<item type="list"><item type="bool" value="true"/></item>'
        "</property></properties><data",
    )
    doc = tmx.read_tmx(_map(body=body), **LOADERS)
    assert doc.layers[0].properties["bag"] == tsx.Prop(
        "list",
        [
            tsx.Prop("int", 1),
            tsx.Prop("string", "torch"),
            tsx.Prop("list", [tsx.Prop("bool", True)]),
        ],
    )
    again = tmx.read_tmx(tmx.tmx_export(doc)["map.tmx"], **LOADERS)
    assert again.layers[0].properties == doc.layers[0].properties


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
# Every core object spelling the document models must leave through both writer
# doors and return through the matching reader without a weaker rectangle being
# substituted for it.


def _exportable_doc() -> tilemap.MapDoc:
    """A map that exports cleanly, so each case below changes one thing."""
    doc = tilemap.MapDoc(2, 2, 16, 16)
    doc.add_tileset(Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16))
    doc.add_tile_layer("L")
    doc.add_object_layer("O")
    return doc


def test_the_control_map_exports_from_both_writers():
    """The control, first: everything below refuses, so this is what says the
    writer door is not simply refusing everything."""
    doc = _exportable_doc()
    doc.add_object(doc.layers[1].uid, tilemap.MapObject(uid=tilemap.new_uid(), kind="rect"))
    assert tmx.tmx_export(doc) and tmx.tmj_export(doc)


def test_both_writers_round_trip_rotation_opacity_shapes_and_index_order():
    doc = _exportable_doc()
    doc.set_layer_props(doc.layers[1].uid, draworder="index")
    shapes = (
        tilemap.Rect(4, 5),
        tilemap.Point(),
        tilemap.Ellipse(4, 5),
        tilemap.Capsule(4, 5),
        tilemap.Polygon(((0, 0), (4, 0), (4, 4))),
        tilemap.Polyline(((0, 0), (4, 0))),
        tilemap.Text("hi", pixel_size=12, color="#ff112233"),
        tilemap.TileShape(gid=1, w=16, h=16),
    )
    for index, shape in enumerate(shapes):
        doc.add_object(
            doc.layers[1].uid,
            tilemap.MapObject(
                uid=tilemap.new_uid(),
                name=f"shape-{index}",
                x=float(index),
                rotation=15.0,
                opacity=0.75,
                shape=shape,
            ),
        )

    for export, read, filename in (
        (tmx.tmx_export, tmx.read_tmx, "map.tmx"),
        (tmx.tmj_export, tmx.read_tmj, "map.tmj"),
    ):
        files = export(doc)
        again = read(files[filename], **LOADERS)
        layer = next(
            layer for layer in again.all_layers() if isinstance(layer, tilemap.ObjectLayer)
        )
        assert layer.draworder == "index"
        assert [obj.shape for obj in layer.objects] == list(shapes)
        assert {obj.rotation for obj in layer.objects} == {15.0}
        assert {obj.opacity for obj in layer.objects} == {0.75}


def test_a_default_object_layer_still_exports():
    """The other half of the draworder case: ``"topdown"`` is the default and
    must not have become a refusal."""
    doc = _exportable_doc()
    assert doc.layers[1].draworder == "topdown"
    assert tmx.tmx_export(doc) and tmx.tmj_export(doc)


def test_an_infinite_document_cannot_be_silently_exported_as_finite():
    doc = tilemap.MapDoc(2, 2, 16, 16, infinite=True)
    doc.add_tile_layer()
    assert doc.infinite is True
    for export in (tmx.tmx_export, tmx.tmj_export):
        with pytest.raises(tmx.TiledUnsupported) as exc:
            export(doc)
        assert exc.value.feature == "an infinite map"
        assert exc.value.exporting is True
    with pytest.raises(wmap.WmapUnstorable, match="infinite map"):
        wmap.wmap_bytes(doc)


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
