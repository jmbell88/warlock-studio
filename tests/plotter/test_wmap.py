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

import numpy as np
import pytest

from warlock.studio.plotter import gid, tsx, wmap
from warlock.studio.plotter.tilemap import MapDoc, MapObject, new_uid
from warlock.studio.plotter.tileset import Tileset


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
    def rename(manifest):
        manifest["layers"][0]["type"] = "image"

    with pytest.raises(ValueError, match="unknown kind"):
        wmap.read_wmap(_rewrite(_doc(), rename))


def test_an_unknown_object_kind_is_refused():
    def rename(manifest):
        manifest["layers"][1]["objects"][0]["kind"] = "ellipse"

    with pytest.raises(ValueError, match="unknown kind"):
        wmap.read_wmap(_rewrite(_doc(), rename))


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
