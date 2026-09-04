"""The 2026-09-02 review's section 7: Plotter and Packwright.

The entries were struck from the findings file as they were built, per the
repository's rule that a built thing is deleted rather than ticked; this is
what keeps them fixed. Interop-shaped ones are here beside the corpus tests
they belong with; the pane-shaped ones assert the decision function rather than
a draw, which is this repository's pattern for a surface that cannot be driven
headlessly.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter.tilemap import MapDoc

# --- a tileset is "in use" wherever a gid of it is ------------------------------


def test_a_tileset_painted_inside_a_group_counts_as_used():
    """``tileset_usage`` walked ``doc.layers`` -- the *root* list -- so a tile
    layer inside a folder was invisible to it and ``remove_tileset`` renumbered
    every one of those cells."""
    from warlock.studio.tilegrid.tileset import Tileset

    doc = MapDoc(4, 4, 16, 16)
    pixels = np.zeros((16, 32, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    doc.add_tileset(Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16))
    ref = doc.tilesets[0]

    group = doc.add_group_layer()
    inner = doc.add_tile_layer(parent_uid=group.uid)
    inner.data[0, 0] = ref.firstgid

    used, where = doc.tileset_usage(0)
    assert used == 1
    assert where == inner.name
    with pytest.raises(ValueError, match="still used"):
        doc.remove_tileset(0)


def test_a_tile_object_counts_as_used_too():
    """A ``TileShape`` holds a gid exactly as a cell does."""
    from warlock.studio.plotter.tilemap import MapObject, TileShape, new_uid
    from warlock.studio.tilegrid.tileset import Tileset

    doc = MapDoc(4, 4, 16, 16)
    pixels = np.zeros((16, 32, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    doc.add_tileset(Tileset(name="t", pixels=pixels, tile_w=16, tile_h=16))
    ref = doc.tilesets[0]

    layer = doc.add_object_layer()
    doc.add_object(
        layer.uid,
        MapObject(
            uid=new_uid(), name="crate", shape=TileShape(gid=ref.firstgid, w=16, h=16)
        ),
    )

    used, _where = doc.tileset_usage(0)
    assert used == 1


# --- what a Tiled file says survives the round trip -----------------------------


def test_a_foreign_wang_set_keeps_its_representative_tile_and_class():
    """The writer emitted ``tile="-1"`` unconditionally and dropped the class,
    so a foreign set came home with every swatch reset -- a silent edit to
    somebody's file."""
    import xml.etree.ElementTree as ET

    from warlock.studio.plotter import tsx

    xml = (
        '<tileset><wangsets>'
        '<wangset name="Cliffs" type="corner" tile="7" class="terrain">'
        '<wangcolor name="Grass" color="#00ff00" tile="3" probability="1" class="soft"/>'
        '<wangtile tileid="0" wangid="0,1,0,1,0,1,0,1"/>'
        '</wangset></wangsets></tileset>'
    )
    sets = tsx.read_wang_model(ET.fromstring(xml).find("wangsets"))

    assert sets[0].tile == 7
    assert sets[0].klass == "terrain"
    assert sets[0].colours[0].tile == 3
    assert sets[0].colours[0].klass == "soft"

    out = ET.Element("tileset")
    tsx.write_wang_model(out, sets)
    written = out.find("wangsets/wangset")
    assert written.get("tile") == "7"
    assert written.get("class") == "terrain"
    assert written.find("wangcolor").get("tile") == "3"
    assert written.find("wangcolor").get("class") == "soft"


def test_the_deprecated_image_layer_offsets_fold_on_both_spellings():
    """One map saved by an older Tiled read with its image in place from a
    ``.tmx`` and at the origin from the ``.tmj`` beside it."""
    import inspect

    from warlock.studio.plotter import tmx

    body = inspect.getsource(tmx._read_tmj_layer_list)
    assert 'common["offset_x"] += json_number(entry, "x", 0)' in body


def test_an_image_layer_cannot_overwrite_the_map_document():
    """The export writes ``map.tmx`` into the same dict, and a source that
    spelt it replaced the map with PNG bytes."""
    from warlock.studio.plotter import tmx

    doc = MapDoc(4, 4, 16, 16)
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    doc.add_image_layer("sneaky", pixels=pixels, source="map.tmx")

    files = tmx.tmx_export(doc)
    assert files["map.tmx"].startswith(b"<?xml") or b"<map" in files["map.tmx"]
    assert any(name.startswith("images/") for name in files)


def test_an_unknown_stagger_value_is_said_out_loud(caplog):
    """Both fall back silently, and on a staggered map the fallback moves every
    other row half a tile: the map opens looking wrong with nothing saying a
    value was not understood."""
    import xml.etree.ElementTree as ET

    from warlock.studio.plotter import tmx

    node = ET.fromstring('<map staggeraxis="diagonal" staggerindex="middle"/>')
    with caplog.at_level("WARNING"):
        out = tmx._offset_fields(node)

    assert out["stagger_axis"] == "y"
    assert out["stagger_index"] == "odd"
    assert "diagonal" in caplog.text
    assert "middle" in caplog.text


# --- Packwright -----------------------------------------------------------------


def test_a_sprite_key_carries_no_directory(tmp_path):
    """It was ``str(path)``, written into the ``.wpack`` -- so a shared atlas
    document carried the author's directory layout, and the same file added on
    two machines was two sprites."""
    from warlock.studio.packwright.sources import file_key

    one = tmp_path / "barrel.png"
    other = tmp_path / "sub" / "barrel.png"
    other.parent.mkdir()

    assert file_key(one).startswith("barrel#")
    assert str(tmp_path) not in file_key(one)
    # Stable per file -- which is what makes re-adding an edited PNG a
    # replacement rather than a second copy.
    assert file_key(one) == file_key(tmp_path / "barrel.png")
    # And two files of the same name in different folders stay apart.
    assert file_key(one) != file_key(other)


def test_a_quoted_boolean_in_a_hand_edited_manifest_is_read_as_written():
    """``bool("false")`` is True, and this file is hand-editable."""
    from warlock.studio.packwright.wpack import _json_bool

    assert _json_bool("false", True) is False
    assert _json_bool("true", False) is True
    assert _json_bool(False, True) is False
    assert _json_bool(None, True) is True
    assert _json_bool("nonsense", True) is True


def test_the_size_search_can_reach_a_limit_that_is_not_a_power_of_two():
    """The doubling walked past a 1500px ceiling, so a set that fits in 1500
    square was refused as "does not fit in a 1500px atlas"."""
    from warlock.studio.packwright.layout import _candidate_sizes

    sizes = _candidate_sizes(area=1_600_000, floor_w=200, floor_h=200, limit=1500)

    assert (1500, 1500) in sizes
    assert all(w <= 1500 and h <= 1500 for w, h in sizes)


def test_a_pivot_can_be_set_cleared_and_undone():
    """It was modelled end to end and could only be *set* by importing an Inker
    document that already carried one."""
    from warlock.studio.packwright.document import PackDoc
    from warlock.studio.packwright.sources import Sprite

    doc = PackDoc()
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    source = doc.add_source(Sprite(key="a", name="a", pixels=pixels))

    doc.set_pivot(source.uid, (4.0, 8.0))
    assert doc.source(source.uid).sprite.meta.pivot == (4.0, 8.0)

    doc.undo()
    assert doc.source(source.uid).sprite.meta.pivot is None

    # Clearing is a real answer and is not the same as the centre.
    doc.set_pivot(source.uid, (4.0, 4.0))
    doc.set_pivot(source.uid, None)
    assert doc.source(source.uid).sprite.meta.pivot is None
    # A no-op pushes nothing.
    head = doc.history.head
    doc.set_pivot(source.uid, None)
    assert doc.history.head == head
