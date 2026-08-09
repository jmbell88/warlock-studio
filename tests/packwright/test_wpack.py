"""The document and its project file.

The document's own rules are the ones every editor here shares -- uid-addressed
history, a no-op step is not pushed, dirty is a comparison -- plus one of its
own: **a key may appear once**, because two sprites under one key pack into one
slot and the loser is simply missing.

The file's rule is the third instance of the byte-identity one: epoch-stamped
members and a sorted manifest, so a round trip is a fixed point.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from warlock.studio.packwright import wpack
from warlock.studio.packwright.document import PackDoc
from warlock.studio.packwright.sources import Sprite


def _sprite(key: str, w: int = 8, h: int = 6) -> Sprite:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (int(key[-1]) * 20, 40, 60, 255)
    return Sprite(key=key, name=f"name-{key}", pixels=pixels)


def _doc(count: int = 4) -> PackDoc:
    doc = PackDoc()
    for i in range(count):
        doc.add_source(_sprite(f"s{i}"))
    doc.set_settings(mode="maxrects", padding=4, extrude=2, power_of_two=False)
    doc.rename_source(doc.sources[1].uid, "hero")
    doc.mark_saved()
    return doc


# --- the document -------------------------------------------------------------


def test_a_duplicate_key_is_refused():
    doc = PackDoc()
    doc.add_source(_sprite("s0"))
    with pytest.raises(ValueError, match="already holds"):
        doc.add_source(_sprite("s0"))


def test_a_rename_does_not_change_identity():
    """``key`` is derived from where a sprite came from. Renaming the third
    layer of a document does not make it a different layer -- and the pack
    order, which sorts on the key, must not move."""
    doc = PackDoc()
    source = doc.add_source(_sprite("s0"))
    doc.rename_source(source.uid, "something else")
    assert source.key == "s0"
    assert doc.sprites()[0].key == "s0"
    assert doc.sprites()[0].name == "something else"


def test_sources_come_out_in_canonical_key_order():
    doc = PackDoc()
    for key in ("s2", "s0", "s1"):
        doc.add_source(_sprite(key))
    assert [s.key for s in doc.sprites()] == ["s0", "s1", "s2"]


def test_a_rename_to_the_same_name_pushes_nothing():
    doc = PackDoc()
    source = doc.add_source(_sprite("s0"))
    doc.rename_source(source.uid, "x")
    head = doc.history.head
    doc.rename_source(source.uid, "x")
    assert doc.history.head == head


def test_re_applying_the_same_settings_pushes_nothing():
    doc = PackDoc()
    doc.set_settings(padding=6)
    head = doc.history.head
    doc.set_settings(padding=6)
    assert doc.history.head == head


def test_dirty_is_a_comparison_and_undo_can_clear_it():
    doc = PackDoc()
    doc.add_source(_sprite("s0"))
    doc.mark_saved()
    doc.set_settings(padding=8)
    assert doc.dirty
    doc.undo()
    assert not doc.dirty


def test_a_removal_undoes_to_the_same_source_object():
    doc = _doc()
    source = doc.sources[1]
    doc.remove_source(source.uid)
    assert doc.source(source.uid) is None
    doc.undo()
    assert doc.source(source.uid) is source
    assert doc.source(source.uid).name == "hero"


def test_an_undone_add_still_costs_its_pixels():
    """It holds the only reference to a full sprite's pixels; a zero cost hides
    megabytes from the byte budget entirely."""
    doc = PackDoc()
    sprite = _sprite("s0", 64, 64)
    doc.add_source(sprite)
    assert doc.history.bytes >= sprite.pixels.nbytes


def test_the_layout_is_derived_rather_than_stored():
    """Which is what makes a re-export of an unchanged document byte-identical
    with nothing to invalidate."""
    doc = _doc()
    assert doc.layout() == doc.layout()
    before = doc.layout()
    doc.set_settings(padding=8)
    assert doc.layout() != before


def test_removing_a_source_is_visible_in_the_next_layout():
    doc = _doc()
    doc.remove_source(doc.sources[0].uid)
    assert len(doc.layout().frames) == 3


# --- the file -----------------------------------------------------------------


def test_everything_survives_a_round_trip():
    doc = _doc()
    back = wpack.read_wpack(wpack.wpack_bytes(doc))

    assert back.settings == doc.settings
    assert [s.key for s in back.sprites()] == [s.key for s in doc.sprites()]
    assert [s.name for s in back.sprites()] == [s.name for s in doc.sprites()]
    for a, b in zip(back.sprites(), doc.sprites(), strict=True):
        assert np.array_equal(a.pixels, b.pixels)


def test_a_reopened_document_produces_the_same_layout():
    doc = _doc()
    assert wpack.read_wpack(wpack.wpack_bytes(doc)).layout() == doc.layout()


def test_a_restored_document_opens_clean():
    back = wpack.read_wpack(wpack.wpack_bytes(_doc()))
    assert not back.dirty
    assert not back.history.can_undo


def test_two_saves_of_one_document_are_byte_identical():
    doc = _doc()
    assert wpack.wpack_bytes(doc) == wpack.wpack_bytes(doc)


def test_a_reopened_document_saves_back_to_the_same_bytes():
    data = wpack.wpack_bytes(_doc())
    assert wpack.wpack_bytes(wpack.read_wpack(data)) == data


def test_every_member_is_stamped_at_the_epoch():
    with zipfile.ZipFile(io.BytesIO(wpack.wpack_bytes(_doc()))) as zf:
        assert zf.namelist()[0] == wpack.MANIFEST
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert zf.getinfo("sources/0.png").compress_type == zipfile.ZIP_STORED


def test_the_manifest_holds_no_atlas_and_no_layout():
    """Storing either would let the file disagree with the settings beside it,
    and the packer is deterministic so there is nothing to gain."""
    payload = json.loads(wpack.manifest_json(_doc()))
    assert set(payload) == {"version", "settings", "sources"}
    assert list(payload) == sorted(payload)


# --- refusals -----------------------------------------------------------------


def _rewrite(doc: PackDoc, mutate) -> bytes:
    with zipfile.ZipFile(io.BytesIO(wpack.wpack_bytes(doc))) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    manifest = json.loads(members[wpack.MANIFEST])
    mutate(manifest)
    members[wpack.MANIFEST] = json.dumps(manifest).encode()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)
    return out.getvalue()


def test_a_file_from_a_newer_version_is_refused():
    with pytest.raises(ValueError, match="newer version"):
        wpack.read_wpack(_rewrite(_doc(), lambda m: m.update(version=wpack.VERSION + 1)))


def test_a_missing_source_image_is_refused():
    def drop(manifest):
        manifest["sources"][0]["image"] = "sources/99.png"

    with pytest.raises(ValueError, match="does not carry"):
        wpack.read_wpack(_rewrite(_doc(), drop))


def test_a_duplicate_key_in_the_file_is_refused():
    def collide(manifest):
        manifest["sources"][1]["key"] = manifest["sources"][0]["key"]

    with pytest.raises(ValueError, match="twice"):
        wpack.read_wpack(_rewrite(_doc(), collide))


def test_a_source_with_no_key_is_refused():
    def blank(manifest):
        manifest["sources"][0]["key"] = ""

    with pytest.raises(ValueError, match="no key"):
        wpack.read_wpack(_rewrite(_doc(), blank))


def test_settings_that_would_bleed_are_refused_at_the_door():
    """``PackSettings`` validates on construction, so a hand-edited manifest is
    refused here rather than producing an atlas that bleeds at some zoom
    levels."""

    def bleed(manifest):
        manifest["settings"]["padding"] = 1
        manifest["settings"]["extrude"] = 4

    with pytest.raises(ValueError, match="twice extrude"):
        wpack.read_wpack(_rewrite(_doc(), bleed))


def test_something_that_is_not_an_archive_is_refused():
    with pytest.raises(ValueError, match="not a Warlock atlas"):
        wpack.read_wpack(b"just some bytes")
