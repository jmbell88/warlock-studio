"""The recorded render: where it comes from, and that it survives a save.

The base is the third picture a three-way merge needs. Without it a
re-rendered sheet can only be compared against what is on screen, and "this
cell differs" cannot be told from "somebody spent an afternoon on this cell".
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import ora, sheetmerge
from warlock.studio.inker.sheetin import document_from_grid, document_from_sheet

CELL = 8
FRAMES = 4


def _atlas(shade: int = 0) -> np.ndarray:
    atlas = np.zeros((CELL, FRAMES * CELL, 4), dtype=np.uint8)
    atlas[..., 3] = 255
    for i in range(FRAMES):
        atlas[:, i * CELL : (i + 1) * CELL, 0] = 30 + i * 40 + shade
    return atlas


def _cells():
    return [{"x": i * CELL, "y": 0, "w": CELL, "h": CELL} for i in range(FRAMES)]


def _anim():
    return {
        "tags": [{"name": "walk_front", "start": 0, "end": FRAMES - 1, "loop": True}],
        "frames": [],
    }


def _sheet(shade: int = 0, source=None):
    return document_from_sheet(_atlas(shade), _cells(), _anim(), source=source)


def _cel_pixels(doc, frame: int):
    anim = doc.anim
    return anim.cels[(anim.tracks[0].uid, anim.frames[frame].uid)].pixels


# -- recording ----------------------------------------------------------------


def test_a_rendered_sheet_arrives_with_its_render_recorded():
    doc = _sheet(source={"job": "abc", "sheet": "def"})
    assert doc.sheet_base is not None
    assert len(doc.sheet_base.digests) == FRAMES
    assert doc.sheet_base.source == {"job": "abc", "sheet": "def"}
    assert doc.sheet_base.conflicts == set()


def test_the_digests_are_of_the_cells_and_tell_them_apart():
    doc = _sheet()
    digests = doc.sheet_base.digests
    assert len(set(digests.values())) == FRAMES, "four different cells, four digests"
    for i, frame in enumerate(doc.anim.frames):
        assert digests[frame.uid] == sheetmerge.cell_digest(_cel_pixels(doc, i))


def test_the_base_is_keyed_by_uid_so_a_reorder_cannot_slide_it():
    """The undo stack's rule: every edit addresses by uid, never by index. A
    digest that travelled by index would land on the wrong cell the moment a
    frame moved."""
    doc = _sheet()
    assert set(doc.sheet_base.digests) == {frame.uid for frame in doc.anim.frames}


def test_a_sheet_with_no_source_still_records_its_render():
    """Provenance is optional; the digests are not. A document that cannot say
    which sheet it came from can still tell an edit from a render."""
    doc = _sheet()
    assert doc.sheet_base is not None
    assert doc.sheet_base.source == {}


def test_the_grid_importer_records_nothing():
    """A recorded base is a claim that a re-renderer exists behind the door.
    "A sheet from anywhere" has none, so it gets none -- and a base it did not
    earn would let a merge overwrite work against a render that never made
    these pixels."""
    doc = document_from_grid(_atlas(), (CELL, CELL))
    assert getattr(doc, "sheet_base", None) is None


def test_an_ordinary_document_has_no_base_at_all():
    from warlock.studio.inker.document import Document

    assert Document.blank(8, 8).sheet_base is None


# -- persistence --------------------------------------------------------------


def test_the_recorded_render_survives_a_save_and_reopen(tmp_path):
    doc = _sheet(source={"job": "abc", "sheet": "def"})
    flagged = doc.anim.frames[2].uid
    doc.sheet_base.conflicts.add(flagged)

    path = tmp_path / "sheet.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)

    assert back.sheet_base is not None
    assert back.sheet_base.source == {"job": "abc", "sheet": "def"}
    # Uids are minted per process, so the *values* travel and the addresses are
    # re-derived from the indices in the file.
    assert sorted(back.sheet_base.digests.values()) == sorted(doc.sheet_base.digests.values())
    assert back.sheet_base.digests[back.anim.frames[2].uid] == doc.sheet_base.digests[flagged]
    assert back.sheet_base.conflicts == {back.anim.frames[2].uid}


def test_writing_the_same_document_twice_is_byte_identical(tmp_path):
    """``.ora`` writing is pinned byte-identical, and a set's iteration order
    is not a promise -- which is why the block sorts everything it writes."""
    doc = _sheet()
    doc.sheet_base.conflicts.update(
        {doc.anim.frames[3].uid, doc.anim.frames[1].uid}
    )
    first, second = tmp_path / "a.ora", tmp_path / "b.ora"
    ora.write_ora(doc, first)
    ora.write_ora(ora.read_ora(first), second)

    assert _animation_json(first) == _animation_json(second)


def _animation_json(path) -> bytes:
    import zipfile

    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("animation.json"))
        return archive.read(name)


def test_a_document_with_no_base_writes_no_sheet_key(tmp_path):
    """Additive and written only when set -- ``groups``' rule. An ordinary
    document's ``animation.json`` has to stay what it was."""
    doc = _sheet()
    doc.sheet_base = None
    path = tmp_path / "plain.ora"
    ora.write_ora(doc, path)

    assert b'"sheet"' not in _animation_json(path)
    assert ora.read_ora(path).sheet_base is None


def test_a_block_written_by_a_future_algorithm_leaves_no_base(tmp_path):
    """Digests we cannot recompute are worse than none: every cell would
    classify as edited and the merge would refuse to take anything, silently.
    The document itself must open whole regardless."""
    import json
    import zipfile

    doc = _sheet()
    path = tmp_path / "sheet.ora"
    ora.write_ora(doc, path)

    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    key = next(n for n in members if n.endswith("animation.json"))
    payload = json.loads(members[key])
    payload["sheet"]["algorithm"] = "sha3-512-from-the-future"
    members[key] = json.dumps(payload, indent=2).encode("utf-8")

    broken = tmp_path / "broken.ora"
    with zipfile.ZipFile(broken, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)

    back = ora.read_ora(broken)
    assert back.sheet_base is None
    assert len(back.anim.frames) == FRAMES, "the timeline must survive intact"
