"""``warlock.json``: what an ``.ora`` carries about slices, and what it does not.

The member is additive in the strict sense -- written only when there is
something to write, read with ``.get``, versioned so a future shape cannot be
mistaken for this one -- and the tests below are mostly about the *negative*
half of that: an old file opens with no slices, a slice-less save produces no
member at all, and a member that is wrong about itself costs the document its
slices and nothing else.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from warlock.studio.inker import ora
from warlock.studio.inker.document import Document
from warlock.studio.inker.slices import SliceKey


def _doc() -> Document:
    doc = Document.blank(16, 12)
    doc.stack.active.pixels[2:6, 2:6] = (255, 0, 0, 255)
    return doc


def _sliced() -> Document:
    doc = _doc()
    doc.add_slice((2, 3, 10, 9), name="body", pivot=(4.0, 6.0), center=(2, 2, 6, 4))
    doc.add_slice((0, 0, 4, 4), name="plain")
    return doc


def _members(doc: Document) -> set[str]:
    with zipfile.ZipFile(BytesIO(ora.ora_bytes(doc))) as zf:
        return set(zf.namelist())


def _payload(doc: Document) -> dict:
    with zipfile.ZipFile(BytesIO(ora.ora_bytes(doc))) as zf:
        return json.loads(zf.read(ora.WARLOCK_MEMBER))


def _rewritten(doc: Document, member: bytes, path: Path) -> Document:
    """The same archive with ``warlock.json`` replaced, on disk."""
    original = ora.ora_bytes(doc)
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(original)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = member if info.filename == ora.WARLOCK_MEMBER else src.read(info.filename)
            dst.writestr(info, data, info.compress_type)
    path.write_bytes(out.getvalue())
    return ora.read_ora(path)


# --- writing ------------------------------------------------------------------


def test_a_document_with_no_slices_writes_no_member_at_all():
    """Which is what keeps every archive this build wrote before slices existed
    byte-for-byte what it was."""
    assert ora.WARLOCK_MEMBER not in _members(_doc())


def test_a_document_with_slices_writes_one():
    assert ora.WARLOCK_MEMBER in _members(_sliced())


def test_the_member_is_versioned_and_holds_rectangles_engines_recognise():
    payload = _payload(_sliced())
    assert payload["version"] == ora.WARLOCK_VERSION
    body = payload["slices"][0]
    assert body["name"] == "body"
    assert body["bounds"] == {"x": 2, "y": 3, "w": 8, "h": 6}
    assert body["pivot"] == {"x": 4.0, "y": 6.0}
    assert body["center"] == {"x": 2, "y": 2, "w": 4, "h": 2}


def test_an_optional_field_is_absent_rather_than_null():
    """Small, and more usefully stable: a slice that is a plain rectangle writes
    the same three keys it would have written before pivots existed."""
    plain = _payload(_sliced())["slices"][1]
    assert set(plain) == {"name", "bounds"}


def test_two_saves_of_a_sliced_document_are_byte_identical():
    doc = _sliced()
    assert ora.ora_bytes(doc) == ora.ora_bytes(doc)


def test_keys_are_written_by_frame_index_and_sorted_by_it():
    doc = _doc()
    doc.add_frame()
    doc.add_frame()
    entry = doc.add_slice((0, 0, 4, 4))
    # Keyed back to front, so the sort is doing something.
    for index in (2, 0):
        doc.set_slice_key(
            entry.uid,
            doc.anim.frames[index].uid,
            key=SliceKey(bounds=(index, index, index + 2, index + 2)),
        )
    keys = _payload(doc)["slices"][0]["keys"]
    assert [record["frame"] for record in keys] == [0, 2]


def test_a_key_whose_frame_has_gone_is_skipped_rather_than_failing_the_save():
    doc = _doc()
    doc.add_frame()
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, doc.anim.frames[1].uid)
    doc.remove_frame(1)
    # Nothing purges the key -- the accepted leak ``_placeholder_uids`` takes --
    # so the writer is where it stops.
    assert doc.slices[0].keys
    assert "keys" not in _payload(doc)["slices"][0]


def test_a_still_documents_keys_are_not_written():
    """There are no frame indices to write them against."""
    doc = _doc()
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice(entry.uid, keys={999: SliceKey(bounds=(1, 1, 2, 2))})
    assert "keys" not in _payload(doc)["slices"][0]


# --- reading ------------------------------------------------------------------


def test_a_still_document_round_trips_its_slices(tmp_path: Path):
    doc = _sliced()
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)
    assert [(s.name, s.bounds, s.pivot, s.center) for s in back.slices] == [
        ("body", (2, 3, 10, 9), (4.0, 6.0), (2, 2, 6, 4)),
        ("plain", (0, 0, 4, 4), None, None),
    ]


def test_an_animated_document_round_trips_its_keys(tmp_path: Path):
    doc = _doc()
    doc.add_frame()
    doc.add_frame()
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, doc.anim.frames[2].uid, key=SliceKey(bounds=(5, 5, 9, 9)))
    path = tmp_path / "b.ora"
    ora.write_ora(doc, path)

    back = ora.read_ora(path)
    assert back.anim is not None and len(back.anim.frames) == 3
    # Uids are per process, so what round-trips is the *index*: the key that was
    # on frame 3 is on frame 3.
    assert back.slices[0].at(back.anim.frames[2].uid).bounds == (5, 5, 9, 9)
    assert back.slices[0].at(back.anim.frames[0].uid).bounds == (0, 0, 4, 4)


def test_a_file_written_before_slices_existed_opens_with_none(tmp_path: Path):
    doc = _doc()
    path = tmp_path / "old.ora"
    ora.write_ora(doc, path)
    assert ora.read_ora(path).slices == []


@pytest.mark.parametrize(
    "member",
    [
        b"not json at all",
        json.dumps({"version": 99, "slices": []}).encode(),
        json.dumps({"version": 1, "slices": [{"name": "x"}]}).encode(),
        json.dumps({"version": 1, "slices": [{"bounds": "wide"}]}).encode(),
        json.dumps({"version": 1, "slices": [{"bounds": {"x": 0, "y": 0, "w": 2}}]}).encode(),
        json.dumps({"version": 1, "slices": "some"}).encode(),
    ],
)
def test_a_malformed_member_costs_the_slices_and_never_the_file(
    tmp_path: Path, member: bytes
):
    """The bargain the rest of this reader makes: a file that opens slightly
    wrong is a file the user still has. The pixels are all present and correct
    whatever this member says about itself."""
    back = _rewritten(_sliced(), member, tmp_path / "bad.ora")
    assert back.slices == []
    assert len(back.stack) == 1
    assert int(back.stack[0].pixels[3, 3, 0]) == 255


def test_one_bad_slice_drops_the_whole_member_rather_than_half_of_it(tmp_path: Path):
    """Half a slice list is the outcome worth avoiding: a nine-slice panel that
    quietly lost its centre still exports and stretches wrong in the game."""
    member = json.dumps(
        {
            "version": 1,
            "slices": [
                {"name": "good", "bounds": {"x": 0, "y": 0, "w": 4, "h": 4}},
                {"name": "bad", "bounds": {"x": 0, "y": 0, "w": 4}},
            ],
        }
    ).encode()
    assert _rewritten(_sliced(), member, tmp_path / "half.ora").slices == []


def test_a_key_naming_no_such_frame_is_dropped_and_the_slice_kept(tmp_path: Path):
    doc = _doc()
    doc.add_frame()
    doc.add_slice((0, 0, 4, 4), name="k")
    member = json.dumps(
        {
            "version": 1,
            "slices": [
                {
                    "name": "k",
                    "bounds": {"x": 0, "y": 0, "w": 4, "h": 4},
                    "keys": [{"frame": 9, "bounds": {"x": 1, "y": 1, "w": 2, "h": 2}}],
                }
            ],
        }
    ).encode()
    back = _rewritten(doc, member, tmp_path / "ghost.ora")
    assert [s.name for s in back.slices] == ["k"]
    assert back.slices[0].keys == {}


def test_keys_are_dropped_when_the_grid_read_fell_back_flat(tmp_path: Path):
    """An index into a timeline that was not restored names nothing, so it is
    discarded with a line in the log rather than guessed at."""
    doc = _doc()
    doc.add_frame()
    entry = doc.add_slice((0, 0, 4, 4))
    doc.set_slice_key(entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(5, 5, 9, 9)))

    original = ora.ora_bytes(doc)
    out = BytesIO()
    with zipfile.ZipFile(BytesIO(original)) as src, zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            data = (
                b"{not json}"
                if info.filename == ora.ANIMATION_MEMBER
                else src.read(info.filename)
            )
            dst.writestr(info, data, info.compress_type)
    path = tmp_path / "flat.ora"
    path.write_bytes(out.getvalue())

    back = ora.read_ora(path)
    assert back.anim is None
    assert [s.name for s in back.slices] == [entry.name]
    assert back.slices[0].keys == {}


# --- crash recovery -----------------------------------------------------------


def test_a_journal_copy_carries_the_slices():
    """The journal encodes a drawing through ``ora_bytes``, so it rides along --
    asserted rather than trusted, because "it uses the same writer" is exactly
    the kind of claim that stops being true in one edit."""
    from warlock.studio import inker_mode
    from warlock.studio.inker_state import InkerDoc

    doc = _sliced()
    raw = inker_mode._journal_encode(InkerDoc(doc=doc))
    with zipfile.ZipFile(BytesIO(raw)) as zf:
        payload = json.loads(zf.read(ora.WARLOCK_MEMBER))
    assert [entry["name"] for entry in payload["slices"]] == ["body", "plain"]
