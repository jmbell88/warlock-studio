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


# --- power-of-two re-baselines on a bare mode switch ------------------------


def test_a_mode_switch_alone_re_resolves_power_of_two_to_the_new_modes_default():
    doc = PackDoc()
    assert doc.settings.mode == "grid" and doc.settings.power_of_two is False
    doc.set_settings(mode="maxrects")
    assert doc.settings.power_of_two is True


def test_switching_back_to_grid_re_resolves_to_false():
    doc = PackDoc()
    doc.set_settings(mode="maxrects")
    assert doc.settings.power_of_two is True
    doc.set_settings(mode="grid")
    assert doc.settings.power_of_two is False


def test_an_explicit_power_of_two_in_the_same_call_as_a_mode_change_wins():
    """The re-baseline only fires when the caller left ``power_of_two``
    unsaid -- naming it in the same call is exactly as authoritative as it
    always was."""
    doc = PackDoc()
    doc.set_settings(mode="maxrects", power_of_two=False)
    assert doc.settings.power_of_two is False


def test_a_toggle_after_a_mode_switch_still_sticks():
    """The re-baseline is a one-time reset at the moment of the switch, not
    a standing rule that fights every edit after it."""
    doc = PackDoc()
    doc.set_settings(mode="maxrects")
    assert doc.settings.power_of_two is True
    doc.set_settings(power_of_two=False)
    assert doc.settings.power_of_two is False
    # And a second, unrelated settings edit does not re-baseline it again --
    # only a change to ``mode`` itself does.
    doc.set_settings(padding=6)
    assert doc.settings.power_of_two is False


def test_a_mode_no_op_does_not_re_baseline_power_of_two():
    """Passing the *current* mode is not a switch -- ``set_settings``
    already pushes nothing for a no-op edit, and a stray re-baseline would
    have made this one look like a change when it is not."""
    doc = PackDoc()
    doc.set_settings(power_of_two=True)
    head = doc.history.head
    doc.set_settings(mode="grid")  # already grid
    assert doc.settings.power_of_two is True
    assert doc.history.head == head


# --- a mode switch away from grid clears a stale columns value --------------


def test_switching_from_grid_to_maxrects_clears_a_set_columns():
    doc = PackDoc()
    doc.set_settings(columns=4)
    assert doc.settings.columns == 4
    doc.set_settings(mode="maxrects")
    assert doc.settings.columns is None


def test_an_explicit_columns_in_the_same_call_as_the_switch_still_refuses():
    """Naming ``columns`` in the same call as the switch always wins over the
    auto-clear -- but 4 columns is still not a MaxRects setting, so
    ``PackSettings`` refuses the combination exactly as it always did."""
    doc = PackDoc()
    doc.set_settings(columns=4)
    with pytest.raises(ValueError, match="columns only applies to a grid pack"):
        doc.set_settings(mode="maxrects", columns=4)
    assert doc.settings.columns == 4  # the refused edit never applied


def test_switching_grid_to_grid_is_a_no_op_that_leaves_columns_alone():
    doc = PackDoc()
    doc.set_settings(columns=4)
    head = doc.history.head
    doc.set_settings(mode="grid")  # already grid
    assert doc.settings.columns == 4
    assert doc.history.head == head


def test_a_mode_switch_with_no_columns_set_pushes_nothing_extra():
    doc = PackDoc()
    doc.set_settings(mode="maxrects")
    assert doc.settings.columns is None


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


def test_addressing_a_source_that_is_not_there_is_a_refusal_not_a_key_error():
    """A ``KeyError`` reached the pygame key dispatch as a crash: the mode
    frames a ``ValueError`` into a toast and lets everything else through to
    the log. A uid goes stale for ordinary reasons -- a selection that survived
    an undo is the one that bit."""
    doc = _doc()
    for call in (
        lambda: doc.remove_source(9999),
        lambda: doc.index_of(9999),
        lambda: doc.rename_source(9999, "x"),
    ):
        with pytest.raises(ValueError, match="not in this pack"):
            call()


def test_a_name_that_would_mean_something_else_in_the_sidecar_is_refused():
    """The value lands verbatim in the TexturePacker ``filename`` and in a
    ``.tsx``, both of which are read by other programs."""
    doc = _doc()
    uid = doc.sources[0].uid
    for bad in ("sub/hero", "sub\\hero", "hero\nname"):
        with pytest.raises(ValueError, match="path separator or a control"):
            doc.rename_source(uid, bad)
    with pytest.raises(ValueError, match="at most 64 characters"):
        doc.rename_source(uid, "h" * 65)


def test_an_empty_name_stays_legal_because_it_clears_the_override():
    doc = _doc()
    source = doc.sources[1]
    assert source.name == "hero"
    doc.rename_source(source.uid, "")
    assert source.name == source.sprite.name


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


# --- columns / json_schema: additive settings fields ---------------------


def test_default_columns_and_schema_are_absent_from_the_manifest():
    """The ``_meta_json`` rule, applied to settings: a document that never
    touched either field produces exactly the bytes it always did, so an
    older build's ``.get``-based read of a file this build wrote still opens
    it as the document it already knows how to describe."""
    payload = json.loads(wpack.manifest_json(_doc()))
    assert "columns" not in payload["settings"]
    assert "json_schema" not in payload["settings"]


def test_an_explicit_columns_and_schema_round_trip():
    doc = PackDoc()
    for i in range(4):
        doc.add_source(_sprite(f"s{i}"))
    doc.set_settings(columns=2, json_schema="hash")
    doc.mark_saved()
    payload = json.loads(wpack.manifest_json(doc))
    assert payload["settings"]["columns"] == 2
    assert payload["settings"]["json_schema"] == "hash"
    back = wpack.read_wpack(wpack.wpack_bytes(doc))
    assert back.settings.columns == 2
    assert back.settings.json_schema == "hash"


def test_a_manifest_that_never_heard_of_columns_or_schema_opens_at_the_defaults():
    """A ``.wpack`` written before this task -- or by hand -- has neither key
    at all, not even ``null``; the reader's ``.get`` default is what a
    genuinely older file needs, not just a value round-tripped through this
    build."""
    data = _rewrite(_doc(), lambda m: m["settings"].pop("columns", None))
    back = wpack.read_wpack(data)
    assert back.settings.columns is None
    assert back.settings.json_schema == "array"


def test_an_old_grid_document_keeps_its_stored_power_of_two_true():
    """``power_of_two`` has always been written unconditionally (unlike the
    additive ``columns``/``json_schema``), so every file this app has ever
    produced carries an explicit value -- including one saved as a grid pack
    before a grid's default flipped to off. Reading it back uses the stored
    value, not the new mode-resolved default."""
    data = _rewrite(_doc(), lambda m: m["settings"].update(mode="grid", power_of_two=True))
    back = wpack.read_wpack(data)
    assert back.settings.mode == "grid"
    assert back.settings.power_of_two is True


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


def _rewrite_member(doc: PackDoc, name: str, blob: bytes) -> bytes:
    """One member's bytes replaced, the manifest left alone.

    The companion to :func:`_rewrite`: a hostile ``.wpack`` is as likely to
    carry a manifest that is fine and a member that is not.
    """
    with zipfile.ZipFile(io.BytesIO(wpack.wpack_bytes(doc))) as zf:
        members = {entry: zf.read(entry) for entry in zf.namelist()}
    members[name] = blob
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        for entry, data in members.items():
            zf.writestr(entry, data)
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


# --- the doors ----------------------------------------------------------------


def test_an_archive_claiming_more_than_the_ceiling_is_refused(monkeypatch):
    """A zip's directory declares what it unpacks to and nothing makes that
    honest, so the refusal has to come off the directory rather than off the
    read that has already exhausted memory. The constant is lowered rather than
    a gigabyte being built, which is why it is read at call time."""
    data = wpack.wpack_bytes(_doc())
    monkeypatch.setattr(wpack, "MAX_DECOMPRESSED_BYTES", 16)
    with pytest.raises(ValueError, match="bytes unpacked"):
        wpack.read_wpack(data)


def test_more_sources_than_one_atlas_will_take_are_refused(monkeypatch):
    """Asked off the manifest, before any of them is decoded: a file listing a
    million sources is a million PNG decodes ahead of the refusal that was
    always going to come out of the packer anyway."""
    from warlock.studio.packwright import layout as lay

    data = wpack.wpack_bytes(_doc(4))
    monkeypatch.setattr(lay, "MAX_SPRITES", 2)
    with pytest.raises(ValueError, match="is the most this build will open"):
        wpack.read_wpack(data)


@pytest.mark.parametrize("version", [[], "abc", {"a": 1}])
def test_a_junk_version_is_a_refusal_rather_than_a_type_error(version):
    """``int([])`` is a ``TypeError``, which is not a ``ValueError`` and so
    reaches the user as the generic log-pointer toast."""
    with pytest.raises(ValueError, match="malformed"):
        wpack.read_wpack(_rewrite(_doc(), lambda m: m.update(version=version)))


def test_a_junk_sources_list_is_a_refusal():
    with pytest.raises(ValueError, match="malformed"):
        wpack.read_wpack(_rewrite(_doc(), lambda m: m.update(sources=7)))


def test_junk_settings_are_a_refusal_rather_than_a_value_error_from_int():
    def junk(manifest):
        manifest["settings"]["padding"] = "wide"

    with pytest.raises(ValueError, match="settings are malformed"):
        wpack.read_wpack(_rewrite(_doc(), junk))


def test_a_member_that_is_not_an_image_is_refused_by_name():
    """``UnidentifiedImageError`` is an ``OSError``; without this clause it
    escaped the ``ValueError`` contract every caller of this function reads."""
    data = _rewrite_member(_doc(), "sources/1.png", b"not a png at all")
    with pytest.raises(ValueError, match="sources/1.png is not an image"):
        wpack.read_wpack(data)


def test_a_source_image_past_the_pixel_ceiling_is_refused(monkeypatch):
    """The byte ceiling is about the archive; one 60000-square PNG deflates to
    very little and decodes to fourteen gigabytes."""
    data = wpack.wpack_bytes(_doc())
    monkeypatch.setattr(wpack, "MAX_SOURCE_PIXELS", 4)
    with pytest.raises(ValueError, match="the limit is 4 pixels"):
        wpack.read_wpack(data)


def test_a_refusal_does_not_leave_the_archive_open(monkeypatch):
    """Every refusal moved inside the ``with``: they used to sit in the gap
    between ``ZipFile(...)`` and the block, where the one thing that block
    exists to do left the archive to the collector."""
    closed: list[bool] = []
    real_close = zipfile.ZipFile.close

    def note(self):
        closed.append(True)
        real_close(self)

    data = _rewrite(_doc(), lambda m: m.update(version=wpack.VERSION + 1))
    monkeypatch.setattr(zipfile.ZipFile, "close", note)
    with pytest.raises(ValueError, match="newer version"):
        wpack.read_wpack(data)
    assert closed
