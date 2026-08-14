"""``.wmap`` -- the project file.

Two properties carry this format. **Two saves of an unchanged document are
byte-identical**, which is what the fixed zip timestamps and the sorted manifest
buy: a file that changes every time it is written is undiffable, its content
hash useless, and "has this actually changed" unanswerable outside the app. And
**a half-read document is refused**, because every way of half-reading one ends
with the user saving it back over the original.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from warlock.studio.plotter import gid, tsx, wmap
from warlock.studio.plotter.tilemap import (
    Ellipse,
    MapDoc,
    MapObject,
    Point,
    Polygon,
    Polyline,
    Rect,
    Text,
    TileShape,
    new_uid,
)
from warlock.studio.plotter.tileset import Tileset

from ._semantics import doc_facts

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "wmap"


def _pixels(w: int = 64, h: int = 64) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    array[3, 5] = (10, 20, 30, 255)
    return array


def _doc() -> MapDoc:
    doc = MapDoc(6, 4, 16, 24)
    doc.add_tileset(
        Tileset(name="terrain", pixels=_pixels(), tile_w=16, tile_h=16,
                properties={"kind": tsx.Prop("string", "outdoor")})
    )
    tiles = doc.add_tile_layer("Ground")
    cells = np.zeros((4, 6), gid.DTYPE)
    cells[0, 0] = gid.compose(1)
    cells[2, 3] = gid.compose(9, flip_h=True, flip_d=True)
    doc.write_region(tiles.uid, 0, 0, cells)
    doc.set_layer_props(tiles.uid, opacity=0.25, visible=False)

    objects = doc.add_object_layer("Things")
    doc.add_object(
        objects.uid,
        MapObject(uid=new_uid(), name="spawn", kind="point", x=1.5, y=2.5,
                  properties={"team": tsx.Prop("int", 1)}),
    )
    doc.add_object(
        objects.uid,
        MapObject(uid=new_uid(), name="zone", kind="rect", x=3, y=4, w=5, h=6,
                  obj_class="Trigger", visible=False),
    )
    doc.properties = {"theme": tsx.Prop("string", "cave"), "hard": tsx.Prop("bool", True)}
    doc.backgroundcolor = "#ff000000"
    doc.renderorder = "right-up"
    doc.mark_saved()
    return doc


# --- round trip ---------------------------------------------------------------


def test_everything_survives_a_round_trip():
    doc = _doc()
    back = wmap.read_wmap(wmap.wmap_bytes(doc))

    assert (back.width, back.height, back.tile_w, back.tile_h) == (6, 4, 16, 24)
    assert back.renderorder == "right-up"
    assert back.backgroundcolor == "#ff000000"
    assert back.properties == doc.properties

    ref = back.tilesets[0]
    assert ref.firstgid == 1
    assert ref.tileset.name == "terrain"
    assert ref.tileset.properties == {"kind": tsx.Prop("string", "outdoor")}
    assert np.array_equal(ref.tileset.pixels, doc.tilesets[0].tileset.pixels)

    ground = back.layers[0]
    assert (ground.name, ground.opacity, ground.visible) == ("Ground", 0.25, False)
    assert np.array_equal(ground.data, doc.tile_layers()[0].data)
    assert ground.data.dtype == gid.DTYPE

    spawn, zone = back.layers[1].objects
    assert (spawn.name, spawn.kind, spawn.x, spawn.y) == ("spawn", "point", 1.5, 2.5)
    assert spawn.properties == {"team": tsx.Prop("int", 1)}
    assert (zone.obj_class, zone.visible, zone.w, zone.h) == ("Trigger", False, 5.0, 6.0)


def test_a_flipped_cell_survives_bit_for_bit():
    doc = _doc()
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert int(back.tile_layers()[0].data[2, 3]) == gid.compose(9, flip_h=True, flip_d=True)


def test_a_restored_document_opens_clean_and_undoable_only_from_here():
    """A file that has just been opened is not unsaved, which is why the reader
    places layers directly rather than through the mutators."""
    back = wmap.read_wmap(wmap.wmap_bytes(_doc()))
    assert not back.dirty
    assert not back.history.can_undo
    assert back.active_layer == back.layers[0].uid


def test_a_restored_document_gets_fresh_uids():
    """The file stores indices, never uids -- a uid is minted per process and
    means nothing in a file. Which is also why this format needs no
    ``reserve_uid``: nothing is ever restored onto a number already issued."""
    doc = _doc()
    data = wmap.wmap_bytes(doc)
    a, b = wmap.read_wmap(data), wmap.read_wmap(data)
    assert {layer.uid for layer in a.layers}.isdisjoint({layer.uid for layer in b.layers})
    assert {layer.uid for layer in a.layers}.isdisjoint({layer.uid for layer in doc.layers})


def test_a_restored_layer_does_not_alias_the_archive_buffer():
    doc = _doc()
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    layer = back.tile_layers()[0]
    assert layer.data.flags.writeable
    layer.data[0, 0] = 4  # must not explode, must not touch anything else
    assert int(layer.data[0, 0]) == 4


# --- byte identity ------------------------------------------------------------


def test_two_saves_of_one_document_are_byte_identical():
    doc = _doc()
    assert wmap.wmap_bytes(doc) == wmap.wmap_bytes(doc)


def test_a_reopened_document_saves_back_to_the_same_bytes():
    """The stronger form: a round trip is a fixed point, so opening a file and
    saving it without touching anything leaves the file alone."""
    data = wmap.wmap_bytes(_doc())
    assert wmap.wmap_bytes(wmap.read_wmap(data)) == data


def test_every_member_is_stamped_at_the_epoch():
    with zipfile.ZipFile(io.BytesIO(wmap.wmap_bytes(_doc()))) as zf:
        assert zf.namelist()[0] == wmap.MANIFEST  # readable half first
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_the_manifest_is_sorted_and_readable():
    payload = json.loads(wmap.manifest_json(_doc()))
    assert payload["version"] == wmap.VERSION
    assert list(payload) == sorted(payload)
    assert payload["layers"][0]["data"] == "layers/0.npy"
    assert payload["tilesets"][0]["image"] == "tilesets/0.png"


def test_a_png_is_stored_rather_than_deflated():
    """A PNG is already compressed; deflating it again spends time to make it
    marginally bigger."""
    with zipfile.ZipFile(io.BytesIO(wmap.wmap_bytes(_doc()))) as zf:
        info = zf.getinfo("tilesets/0.png")
        assert info.compress_type == zipfile.ZIP_STORED


# --- refusals -----------------------------------------------------------------


def _rewrite(doc: MapDoc, mutate) -> bytes:
    """Rebuild an archive with a mutated manifest, keeping every other member."""
    original = wmap.wmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(original)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members[wmap.MANIFEST])
    mutate(manifest)
    members[wmap.MANIFEST] = json.dumps(manifest).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    return out.getvalue()


def test_an_archive_claiming_more_than_the_ceiling_is_refused(monkeypatch):
    """A few kilobytes of zip can claim terabytes, and the read that discovers
    that is the one which has already exhausted memory. The constant is lowered
    rather than a gigabyte being built, which is why it is read at call time."""
    data = wmap.wmap_bytes(_doc())
    monkeypatch.setattr(wmap, "MAX_DECOMPRESSED_BYTES", 16)
    with pytest.raises(ValueError) as exc:
        wmap.read_wmap(data)
    assert "bytes unpacked" in str(exc.value) and "16" in str(exc.value)


def test_a_file_from_a_newer_version_is_refused():
    data = _rewrite(_doc(), lambda m: m.update(version=wmap.VERSION + 1))
    with pytest.raises(ValueError, match="newer version"):
        wmap.read_wmap(data)


def test_a_missing_layer_member_is_refused():
    def drop(manifest):
        manifest["layers"][0]["data"] = "layers/99.npy"

    with pytest.raises(ValueError, match="does not carry"):
        wmap.read_wmap(_rewrite(_doc(), drop))


def test_a_missing_tileset_image_is_refused():
    def drop(manifest):
        manifest["tilesets"][0]["image"] = "tilesets/99.png"

    with pytest.raises(ValueError, match="does not carry"):
        wmap.read_wmap(_rewrite(_doc(), drop))


def test_a_layer_of_the_wrong_shape_is_refused():
    def shrink(manifest):
        manifest["width"] = 3

    with pytest.raises(ValueError, match="but the map is"):
        wmap.read_wmap(_rewrite(_doc(), shrink))


def test_a_gid_no_tileset_accounts_for_is_refused():
    doc = _doc()
    layer = doc.tile_layers()[0]
    layer.data[0, 1] = 500
    with pytest.raises(ValueError, match="accounts for"):
        wmap.read_wmap(wmap.wmap_bytes(doc))


def test_tilesets_numbered_out_of_order_are_refused():
    """Contiguity is what ``resolve`` walks; a list that does not increase means
    two tilesets claim one id and every cell in the overlap draws whichever
    comes first."""
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(Tileset(name="a", pixels=_pixels(32, 32), tile_w=16, tile_h=16))
    doc.add_tileset(Tileset(name="b", pixels=_pixels(32, 32), tile_w=16, tile_h=16))

    def collide(manifest):
        manifest["tilesets"][1]["firstgid"] = 1

    with pytest.raises(ValueError, match="out of order"):
        wmap.read_wmap(_rewrite(doc, collide))


def test_an_unknown_layer_kind_is_refused():
    """``"image"`` used to be the example here and is a real kind since version
    3, which is the shape of every refusal in this package: one leaves the list
    exactly when the editor learns to model the thing."""
    def rename(manifest):
        manifest["layers"][0]["type"] = "hologram"

    with pytest.raises(ValueError, match="unknown kind"):
        wmap.read_wmap(_rewrite(_doc(), rename))


def test_an_unknown_object_kind_is_refused():
    """Through the tagged ``shape`` record, which is where an object's kind
    lives since version 3 -- and ``"ellipse"`` is no longer the example, because
    the document models all seven of Tiled's geometries now."""
    def rename(manifest):
        manifest["layers"][1]["objects"][0]["shape"]["kind"] = "wobble"

    with pytest.raises(ValueError, match="unknown kind"):
        wmap.read_wmap(_rewrite(_doc(), rename))


def test_an_unknown_object_kind_in_the_version_2_spelling_is_refused():
    """The other door into the same refusal: a version 2 object had no shape
    record, only a ``kind``, and the two kinds it could hold are the two the
    editor can author."""
    def downgrade(manifest):
        entry = manifest["layers"][1]["objects"][0]
        del entry["shape"]
        entry["kind"] = "ellipse"

    with pytest.raises(ValueError, match="unknown kind"):
        wmap.read_wmap(_rewrite(_doc(), downgrade))


def test_a_wrongly_typed_layer_array_is_refused():
    """A signed layer stores ``FLIP_H`` as a negative number, and every
    comparison after that is an argument about two's complement."""
    doc = _doc()
    original = wmap.wmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(original)) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.zeros((4, 6), np.int32))
    members["layers/0.npy"] = buffer.getvalue()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    with pytest.raises(ValueError, match="stored as int32"):
        wmap.read_wmap(out.getvalue())


def test_something_that_is_not_an_archive_at_all_is_refused():
    with pytest.raises(ValueError, match="not a Warlock map"):
        wmap.read_wmap(b"just some bytes")


# --- projection and terrains --------------------------------------------------
#
# ``.wmap`` is the only format that carries a terrain set as this editor means
# it, so these two are what stop a generated map from reopening as 235
# anonymous tiles.


def test_a_maps_projection_survives_a_save():
    doc = MapDoc(5, 4, 32, 16, projection="isometric")
    doc.add_tile_layer("Ground")
    assert wmap.read_wmap(wmap.wmap_bytes(doc)).projection == "isometric"


def test_a_version_one_file_still_reads_as_orthogonal():
    """A file written before projections existed is orthogonal by definition,
    which the default says without a branch."""
    doc = MapDoc(3, 3, 16, 16)
    doc.add_tile_layer("Ground")
    data = wmap.wmap_bytes(doc)
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for item in src.namelist():
            payload = src.read(item)
            if item == wmap.MANIFEST:
                manifest = json.loads(payload)
                manifest["version"] = 1
                manifest.pop("projection", None)
                payload = json.dumps(manifest).encode()
            dst.writestr(item, payload)
    assert wmap.read_wmap(out.getvalue()).projection == "orthogonal"


def test_a_terrain_set_survives_a_save():
    from warlock.studio.plotter import tilegen

    doc = MapDoc(4, 4, 8, 8)
    doc.add_tileset(tilegen.generate(tilegen.GenSpec(tile_w=8, tile_h=8)))
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    terrains = back.tilesets[0].tileset.terrains
    assert [entry.name for entry in terrains] == [
        entry.name for entry in tilegen.DEFAULT_TERRAINS
    ]
    # Order is precedence, so it is a list in the manifest and must not be
    # reordered by the manifest's own sorted keys.
    assert terrains == tilegen.DEFAULT_TERRAINS


def test_a_map_with_terrains_still_saves_byte_identically_twice():
    from warlock.studio.plotter import tilegen

    doc = MapDoc(4, 4, 8, 8, projection="isometric")
    doc.add_tileset(tilegen.generate(tilegen.GenSpec(tile_w=8, tile_h=8)))
    doc.add_tile_layer("Ground")
    once = wmap.wmap_bytes(doc)
    assert wmap.wmap_bytes(wmap.read_wmap(once)) == once


def test_an_ordinary_tileset_saves_an_empty_terrain_list():
    """Written even when empty, so the manifest's shape does not depend on its
    content."""
    doc = MapDoc(3, 3, 8, 8)
    doc.add_tileset(
        Tileset(name="plain", pixels=np.zeros((8, 8, 4), dtype=np.uint8), tile_w=8, tile_h=8)
    )
    manifest = json.loads(wmap.manifest_json(doc))
    assert manifest["tilesets"][0]["terrains"] == []


def test_a_lock_survives_a_round_trip():
    doc = _doc()
    doc.set_layer_props(doc.layers[0].uid, locked=True)
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert back.layers[0].locked is True
    assert back.layers[1].locked is False


def test_a_file_written_before_locks_existed_opens_unlocked():
    """Tolerant read of a key that is simply absent -- which is what every
    version 2 file written before this says by saying nothing."""
    data = wmap.wmap_bytes(_doc())
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        manifest = json.loads(zf.read(wmap.MANIFEST))
    assert all("locked" in entry for entry in manifest["layers"]), "we write it"

    # Strip the key back out and rebuild the archive, which is what an older
    # writer would have produced.
    for entry in manifest["layers"]:
        del entry["locked"]
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            payload = (
                json.dumps(manifest).encode() if info.filename == wmap.MANIFEST
                else src.read(info.filename)
            )
            dst.writestr(info, payload)
    back = wmap.read_wmap(out.getvalue())
    assert all(layer.locked is False for layer in back.layers)


def test_layer_properties_set_through_the_editor_survive_a_round_trip():
    """The model has carried these since the format did; T8 added the way in,
    so the way out is worth pinning beside it."""
    doc = _doc()
    doc.set_layer_props(doc.layers[0].uid, properties={"kind": tsx.Prop("string", "floor")})
    doc.set_map_properties({"theme": tsx.Prop("string", "crypt")})
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert back.layers[0].properties == {"kind": tsx.Prop("string", "floor")}
    assert back.properties == {"theme": tsx.Prop("string", "crypt")}


#: Every type the model gained, including the one Tiled has no syntax for.
#: ``.wmap`` is the *complete* codec -- it has one reader and no foreign schema
#: to fit inside, which is why ``list`` lives here and is refused at the Tiled
#: door rather than the other way round.
_NEW_TYPES = {
    "art": tsx.Prop("file", "art/hero.png"),
    "target": tsx.Prop("object", 7),
    "none": tsx.Prop("object", 0),
    "npc": tsx.Prop(
        "class",
        {"hp": tsx.Prop("int", 3), "tint": tsx.Prop("color", "#ff00ff00")},
        propertytype="NPC",
    ),
    "bag": tsx.Prop("list", [tsx.Prop("int", 1), tsx.Prop("string", "two")]),
}


def test_the_recursive_property_types_survive_a_round_trip_at_every_level():
    doc = _doc()
    doc.set_map_properties(dict(_NEW_TYPES))
    doc.set_layer_props(doc.layers[0].uid, properties=dict(_NEW_TYPES))
    objects = doc.layers[1]
    doc.set_object(objects.uid, objects.objects[0].uid, properties=dict(_NEW_TYPES))
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert back.properties == _NEW_TYPES
    assert back.layers[0].properties == _NEW_TYPES
    assert back.layers[1].objects[0].properties == _NEW_TYPES


def test_two_saves_of_a_document_with_recursive_properties_are_byte_identical():
    """Determinism reaches all the way down: class members are written
    sorted, at every depth, so a nested dict's order is not in the file."""
    doc = _doc()
    doc.set_map_properties(dict(_NEW_TYPES))
    assert wmap.wmap_bytes(doc) == wmap.wmap_bytes(doc)


def test_a_property_that_gained_nothing_is_stored_as_version_2_stored_it():
    """The *property record* is unchanged, which is the claim this test has
    always made. The container's version moved to 3 for the layer tree, not for
    the properties: a plain string still writes the two keys version 2 wrote,
    so the recursive types cost a document that uses none of them nothing."""
    doc = _doc()
    raw = wmap.wmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        manifest = json.loads(zf.read(wmap.MANIFEST))
    assert manifest["version"] == 3
    assert manifest["properties"]["theme"] == {"type": "string", "value": "cave"}


# --- version 3: the layer tree ------------------------------------------------
#
# Version 3 is what the document outgrew version 2 into: a tree of four layer
# kinds, each carrying a class, a tint, a pixel offset and a parallax factor;
# objects carrying a persistent id, a rotation and one of seven tagged
# geometries; and the two id counters. The two properties at the top of this
# file are unchanged and every test below is written against them -- the tree
# lives in ``map.json``, so the half of the archive that holds megabytes is the
# same flat sequence of members it was in version 2.

#: Every recursive property type, at every level the format stores properties.
_RICH_PROPS = {
    "art": tsx.Prop("file", "art/hero.png"),
    "target": tsx.Prop("object", 7),
    "npc": tsx.Prop(
        "class",
        {"hp": tsx.Prop("int", 3), "tint": tsx.Prop("color", "#ff00ff00")},
        propertytype="NPC",
    ),
    "bag": tsx.Prop("list", [tsx.Prop("int", 1), tsx.Prop("string", "two")]),
}

#: All seven of Tiled's geometries, one object apiece. The tile object's gid is
#: composed with a flip so the flags are proven to ride in the number rather
#: than being dropped on the way through the manifest.
_SHAPES = (
    Rect(3.0, 4.0),
    Point(),
    Ellipse(5.0, 6.0),
    Polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0))),
    Polyline(((0.0, 0.0), (2.0, 5.5))),
    TileShape(gid.compose(9, flip_h=True, flip_d=True), 16.0, 16.0),
    Text("hello", 40.0, 12.0, family="serif", pixel_size=11, bold=True, halign="center"),
)


def _gate_doc() -> MapDoc:
    """Milestone 2's exit gate, as a document.

    Nested groups, an image layer, all seven object shapes, rotation, an
    ``"index"`` draw order, decorated layers at two depths, and the four
    property types version 2's codec could not spell. Everything the milestone
    taught the model, in one map -- which is the point: a gate assembled from
    seven small documents would pass while the *combination* was broken.
    """
    doc = MapDoc(6, 4, 16, 24)
    doc.add_tileset(
        Tileset(name="terrain", pixels=_pixels(), tile_w=16, tile_h=16,
                properties={"kind": tsx.Prop("string", "outdoor")})
    )

    outer = doc.add_group_layer("Outer")
    inner = doc.add_group_layer("Inner", parent_uid=outer.uid)
    doc.set_layer_props(outer.uid, tint=(255, 128, 0, 200), offset_x=3.5, class_name="Deco")
    doc.set_layer_props(inner.uid, parallax_x=0.5, parallax_y=0.25, opacity=0.75)

    # A tile leaf *inside* the nested group, and another at the root, so the
    # flat depth-first ``.npy`` numbering has something to get wrong.
    buried = doc.add_tile_layer("Buried", parent_uid=inner.uid)
    cells = np.zeros((4, 6), gid.DTYPE)
    cells[1, 2] = gid.compose(9, flip_v=True)
    doc.write_region(buried.uid, 0, 0, cells)

    doc.add_image_layer(
        "Sky",
        pixels=_pixels(8, 6),
        source="art/sky.png",
        repeat_x=True,
        parent_uid=inner.uid,
        tint=(200, 200, 255, 255),
    )

    ground = doc.add_tile_layer("Ground")
    doc.write_region(ground.uid, 0, 0, cells)
    doc.set_layer_props(ground.uid, locked=True, offset_y=-2.0, properties=dict(_RICH_PROPS))

    things = doc.add_object_layer("Things")
    doc.set_layer_props(things.uid, draworder="index")
    for index, shape in enumerate(_SHAPES):
        doc.add_object(
            things.uid,
            MapObject(
                uid=new_uid(),
                name=f"obj{index}",
                x=1.5 + index,
                y=2.5,
                shape=shape,
                rotation=45.0 + index,
                obj_class="Trigger" if index % 2 else "",
                visible=index != 3,
                properties=dict(_RICH_PROPS) if index == 0 else {},
            ),
        )

    doc.set_map_properties(dict(_RICH_PROPS))
    doc.backgroundcolor = "#ff102030"
    doc.renderorder = "right-up"
    doc.mark_saved()
    return doc


def test_the_milestone_gate_document_survives_a_round_trip():
    """**The wave's gate.** Nested groups, an image layer, all seven shapes,
    rotation, ``draworder="index"``, decorated layers and the four recursive
    property types, compared with the same ``doc_facts`` every Plotter-Tiled
    gate is built on -- which is uid-blind by construction and therefore the
    right comparator for a format that mints fresh uids on every read."""
    doc = _gate_doc()
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert doc_facts(back) == doc_facts(doc)


def test_the_gate_document_saves_byte_identically_twice():
    doc = _gate_doc()
    assert wmap.wmap_bytes(doc) == wmap.wmap_bytes(doc)


def test_the_gate_document_is_a_fixed_point_of_a_round_trip():
    """The stronger form of the same rule, and the one that catches a field
    read back differently from how it was written: opening the file and saving
    it without touching anything leaves the bytes alone."""
    data = wmap.wmap_bytes(_gate_doc())
    assert wmap.wmap_bytes(wmap.read_wmap(data)) == data


def test_the_manifests_layer_entries_are_recursive():
    manifest = json.loads(wmap.manifest_json(_gate_doc()))
    outer = manifest["layers"][0]
    assert outer["type"] == "group"
    inner = outer["layers"][0]
    assert inner["type"] == "group"
    assert [entry["type"] for entry in inner["layers"]] == ["tile", "image"]


def test_the_tile_arrays_stay_a_flat_depth_first_enumeration():
    """The container's shape is the point: the tree went into ``map.json``,
    which is kilobytes, and the members that hold megabytes are numbered
    exactly as they were in version 2 -- ``layers/0.npy``, ``layers/1.npy``, in
    paint order, whatever the nesting."""
    doc = _gate_doc()
    manifest = json.loads(wmap.manifest_json(doc))
    buried = manifest["layers"][0]["layers"][0]["layers"][0]
    ground = manifest["layers"][1]
    assert buried["data"] == "layers/0.npy"
    assert ground["data"] == "layers/1.npy"
    with zipfile.ZipFile(io.BytesIO(wmap.wmap_bytes(doc))) as zf:
        assert "layers/0.npy" in zf.namelist() and "layers/1.npy" in zf.namelist()


def test_every_member_the_manifest_names_is_in_the_archive_and_the_reverse():
    """The manifest and the archive number their members in two separate
    loops, and the only thing keeping them in step is that both walk the tree
    depth-first. A mismatch is a file whose own reader refuses it on the line
    about a member the archive does not carry, so it is pinned in both
    directions rather than trusted."""
    doc = _gate_doc()
    data = wmap.wmap_bytes(doc)
    manifest = json.loads(wmap.manifest_json(doc))

    named: set[str] = set()

    def walk(entries):
        for entry in entries:
            for key in ("data", "image"):
                if key in entry:
                    named.add(entry[key])
            walk(entry.get("layers", []))

    walk(manifest["layers"])
    named.update(ts["image"] for ts in manifest["tilesets"])
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        carried = set(zf.namelist()) - {wmap.MANIFEST}
    assert named == carried


def test_an_image_layers_picture_is_stored_rather_than_deflated():
    """A PNG is already compressed, the tileset atlas's rule, and an image
    layer's picture is a PNG for the same reason a tileset's is: the file is
    the whole document, so it can be sent without a folder of dependencies."""
    doc = _gate_doc()
    with zipfile.ZipFile(io.BytesIO(wmap.wmap_bytes(doc))) as zf:
        assert zf.getinfo("images/0.png").compress_type == zipfile.ZIP_STORED


def test_every_object_shape_survives_a_round_trip_field_for_field():
    """``kind``/``w``/``h`` cannot say what a polygon's vertices are, what a
    tile object's gid is, or how a text object is styled. The shape record is
    enumerated off the dataclass so that a field added to a shape travels
    without this codec learning about it, and this is what pins that."""
    doc = _gate_doc()
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert [obj.shape for obj in back.layers[-1].objects] == list(_SHAPES)


def test_a_tile_objects_flip_flags_ride_in_its_gid():
    """One number, flags in the top three bits, exactly as a cell carries
    them -- so a mirrored tile object is not silently unmirrored by the trip
    through JSON."""
    back = wmap.read_wmap(wmap.wmap_bytes(_gate_doc()))
    tile_object = back.layers[-1].objects[5]
    assert tile_object.shape.gid == gid.compose(9, flip_h=True, flip_d=True)


def test_an_object_does_not_store_its_derived_kind_or_size_beside_its_shape():
    """Two spellings of one fact in a file is the arrangement that lets a
    reader restore an object whose stated size disagrees with its geometry.
    ``kind``/``w``/``h`` are properties of the shape now, so only the shape is
    written."""
    manifest = json.loads(wmap.manifest_json(_gate_doc()))
    entry = manifest["layers"][-1]["objects"][0]
    assert entry["shape"] == {"kind": "rect", "w": 3.0, "h": 4.0}
    assert "kind" not in entry and "w" not in entry and "h" not in entry


def test_an_objects_rotation_survives_a_round_trip():
    """The field the ``.tmx`` door still refuses by name. Our own format has to
    hold it, or that refusal's "save it as ``.wmap``" has nowhere to send the
    user."""
    back = wmap.read_wmap(wmap.wmap_bytes(_gate_doc()))
    assert [obj.rotation for obj in back.layers[-1].objects] == [
        45.0 + index for index in range(len(_SHAPES))
    ]


def test_a_draw_order_survives_a_round_trip():
    back = wmap.read_wmap(wmap.wmap_bytes(_gate_doc()))
    assert back.layers[-1].draworder == "index"


def test_a_draw_order_that_is_not_one_of_the_two_is_refused():
    def scramble(manifest):
        manifest["layers"][-1]["draworder"] = "sideways"

    with pytest.raises(ValueError, match="draw order"):
        wmap.read_wmap(_rewrite(_gate_doc(), scramble))


def test_every_layer_decoration_survives_a_round_trip():
    back = wmap.read_wmap(wmap.wmap_bytes(_gate_doc()))
    outer = back.layers[0]
    assert (outer.class_name, tuple(outer.tint), outer.offset_x) == (
        "Deco", (255, 128, 0, 200), 3.5
    )
    inner = outer.children[0]
    assert (inner.parallax_x, inner.parallax_y, inner.opacity) == (0.5, 0.25, 0.75)


def test_the_persistent_ids_survive_a_round_trip():
    """A stored id is what an ``object``-typed property references, so a read
    that reissued them would leave every such property pointing somewhere
    else."""
    doc = _gate_doc()
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert [layer.id for layer in back.all_layers()] == [
        layer.id for layer in doc.all_layers()
    ]
    assert [obj.id for obj in back.layers[-1].objects] == [
        obj.id for obj in doc.layers[-1].objects
    ]
    assert (back.next_layer_id, back.next_object_id) == (
        doc.next_layer_id, doc.next_object_id
    )


def test_a_read_still_mints_fresh_uids_for_a_nested_document():
    """The rule the tree did not change: ids are stored, uids never are."""
    data = wmap.wmap_bytes(_gate_doc())
    a, b = wmap.read_wmap(data), wmap.read_wmap(data)
    assert {layer.uid for layer in a.all_layers()}.isdisjoint(
        {layer.uid for layer in b.all_layers()}
    )
    assert {layer.id for layer in a.all_layers()} == {
        layer.id for layer in b.all_layers()
    }


def test_the_counters_advance_past_an_id_a_manifest_understated():
    """Whatever the file claimed, the counters end past everything in use: a
    hand-edited manifest whose ``next_object_id`` sits below an id it carries
    would otherwise reissue that number to the next object the user draws."""
    def understate(manifest):
        manifest["next_object_id"] = 1
        manifest["next_layer_id"] = 1

    back = wmap.read_wmap(_rewrite(_gate_doc(), understate))
    assert back.next_layer_id > max(layer.id for layer in back.all_layers())
    assert back.next_object_id > max(obj.id for obj in back.layers[-1].objects)


def test_the_recursive_property_types_survive_at_every_level_of_the_tree():
    doc = _gate_doc()
    back = wmap.read_wmap(wmap.wmap_bytes(doc))
    assert back.properties == _RICH_PROPS
    assert back.layers[1].properties == _RICH_PROPS
    assert back.layers[-1].objects[0].properties == _RICH_PROPS


# --- version 3: what is reserved and what is refused --------------------------


def test_the_infinite_key_is_reserved_and_written_false():
    """Named now so that the day chunked storage arrives it is an addition to
    the container rather than a rearrangement of it. Every reader between now
    and then would otherwise read its absence as a fact."""
    manifest = json.loads(wmap.manifest_json(_gate_doc()))
    assert manifest["infinite"] is False


def test_a_file_claiming_to_be_infinite_is_refused():
    """An infinite map has no dense rectangle for the layer-shape check to
    compare against, so reading one as if it had is exactly the half-read this
    format refuses. M5 turns this into an acceptance."""
    def claim(manifest):
        manifest["infinite"] = True

    with pytest.raises(ValueError, match="infinite"):
        wmap.read_wmap(_rewrite(_gate_doc(), claim))


def test_a_malformed_offset_pair_is_refused_while_an_absent_one_defaults():
    """Absent and malformed are not the same statement. A version 2 file says
    nothing about an offset; ``[3]`` says something this reader cannot
    honour."""
    def half(manifest):
        manifest["layers"][0]["offset"] = [3]

    with pytest.raises(ValueError, match="malformed"):
        wmap.read_wmap(_rewrite(_gate_doc(), half))

    def drop(manifest):
        del manifest["layers"][0]["offset"]

    assert wmap.read_wmap(_rewrite(_gate_doc(), drop)).layers[0].offset_x == 0.0


def test_a_malformed_shape_record_is_refused_as_a_value_error():
    """A corrupt record would otherwise leave this reader through a
    ``TypeError``, which is not the door every caller of it is written
    against."""
    def wreck(manifest):
        manifest["layers"][-1]["objects"][3]["shape"]["points"] = "not points"

    with pytest.raises(ValueError, match="malformed polygon"):
        wmap.read_wmap(_rewrite(_gate_doc(), wreck))


def test_a_tile_object_naming_a_gid_no_tileset_accounts_for_is_refused():
    """A tile object's gid is the same encoding a cell's is, so it is the same
    unreadable file wherever it sits."""
    def dangle(manifest):
        manifest["layers"][-1]["objects"][5]["shape"]["gid"] = 500

    with pytest.raises(ValueError, match="accounts for"):
        wmap.read_wmap(_rewrite(_gate_doc(), dangle))


def test_a_group_holding_a_layer_of_unknown_kind_is_refused():
    """The refusal reaches into the tree, not only across the root list."""
    def rename(manifest):
        manifest["layers"][0]["layers"][0]["layers"][0]["type"] = "hologram"

    with pytest.raises(ValueError, match="unknown kind"):
        wmap.read_wmap(_rewrite(_gate_doc(), rename))


# --- the committed version 1 and version 2 fixtures ---------------------------
#
# See ``fixtures/wmap/FIXTURES.md`` for how these were produced and how to
# reproduce them. They are the only cases here written against bytes this build
# did not write, which is the whole of their value: every other case in this
# file asserts that our reader agrees with our writer, and that cannot catch
# the reader quietly requiring a key the old writer never emitted.


def test_the_wmap_fixture_directory_and_its_recipe_exist():
    assert FIXTURE_DIR.is_dir()
    assert (FIXTURE_DIR / "FIXTURES.md").is_file()


@pytest.mark.parametrize("stem", ["v1", "v2"])
def test_an_older_fixture_opens_with_identity_decorations_and_fresh_ids(stem):
    """The tolerant-defaults contract, from the only direction that proves it:
    neither file carries a class, a tint, an offset, a parallax factor or an
    id, and every one of those has to arrive as the value that changes
    nothing."""
    doc = wmap.read_wmap((FIXTURE_DIR / f"{stem}.wmap").read_bytes())
    ground, things = doc.layers
    assert [layer.name for layer in doc.layers] == ["Ground", "Things"]
    for layer in doc.layers:
        assert layer.class_name == ""
        assert tuple(layer.tint) == (255, 255, 255, 255)
        assert (layer.offset_x, layer.offset_y) == (0.0, 0.0)
        assert (layer.parallax_x, layer.parallax_y) == (1.0, 1.0)
    # No file before version 3 stored an id, so every one is minted here, in
    # paint order, from the counters a fresh document starts at.
    assert [layer.id for layer in doc.layers] == [1, 2]
    assert [obj.id for obj in things.objects] == [1, 2]
    assert (doc.next_layer_id, doc.next_object_id) == (3, 3)
    assert things.draworder == "topdown"
    assert ground.opacity == 0.5
    assert int(ground.data[2, 3]) == gid.compose(2, flip_h=True, flip_d=True)
    assert doc.properties == {"theme": tsx.Prop("string", "cave")}
    assert things.objects[0].properties == {"team": tsx.Prop("int", 1)}


def test_the_version_1_fixture_opens_as_an_orthogonal_unlocked_map():
    """Version 1 predates projections *and* locks, and both defaults have to
    say so without a branch."""
    doc = wmap.read_wmap((FIXTURE_DIR / "v1.wmap").read_bytes())
    assert doc.projection == "orthogonal"
    assert all(layer.locked is False for layer in doc.layers)


def test_the_version_2_fixture_keeps_what_version_2_did_store():
    """The other half of the same statement: tolerance is not amnesia. A
    version 2 file stored a projection and a lock, and both come back."""
    doc = wmap.read_wmap((FIXTURE_DIR / "v2.wmap").read_bytes())
    assert doc.projection == "isometric"
    assert doc.layers[0].locked is True


@pytest.mark.parametrize("stem", ["v1", "v2"])
def test_an_older_fixture_reads_its_objects_from_the_version_2_spelling(stem):
    """Neither file has a ``shape`` record -- an object was a ``kind`` and two
    numbers -- and the two kinds those versions could hold are exactly the two
    the editor can author."""
    doc = wmap.read_wmap((FIXTURE_DIR / f"{stem}.wmap").read_bytes())
    spawn, zone = doc.layers[1].objects
    assert (spawn.kind, spawn.x, spawn.y) == ("point", 1.5, 2.5)
    assert spawn.shape == Point()
    assert zone.shape == Rect(5.0, 6.0)
    assert (zone.obj_class, zone.visible) == ("Trigger", False)
    assert all(obj.rotation == 0.0 for obj in doc.layers[1].objects)


@pytest.mark.parametrize("stem", ["v1", "v2"])
def test_an_older_fixture_saves_forward_as_version_3(stem):
    """Reading an old file and saving it writes the current version -- there is
    one writer, and a format that emitted whichever version a document happened
    to need would be three writers to keep in step. The document is unchanged
    by the trip, which is what makes the upgrade safe to do on the user's own
    file."""
    doc = wmap.read_wmap((FIXTURE_DIR / f"{stem}.wmap").read_bytes())
    data = wmap.wmap_bytes(doc)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert json.loads(zf.read(wmap.MANIFEST))["version"] == 3
    assert doc_facts(wmap.read_wmap(data)) == doc_facts(doc)
