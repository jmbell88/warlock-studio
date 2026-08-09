"""Two saves of an unchanged ``.ora`` are byte-identical.

A zip stamps every member with the wall clock, so a document nobody had touched
produced a different file each time it was written -- which makes a save look
like a change to anything that hashes, diffs or syncs one. The three younger
formats in this repo (``.wblk``, ``.wmap``, ``.wpack``) all fix their members at
1980-01-01 and pin it with a test; ``.ora`` was the odd one out, and the only
member it fixed was ``mimetype``, by accident of a bare ``ZipInfo``'s default.

That accident is also the evidence that a fixed stamp is safe against a foreign
reader: ``mimetype`` is the first member every ORA reader touches, it is
specified to be first and stored, and it has carried this exact timestamp since
this writer was written. The OpenRaster spec has nothing to say about
modification times at all, and 1980-01-01 is not an arbitrary choice but the
floor -- MS-DOS date fields cannot express anything earlier and ``zipfile``
raises below it.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from warlock.studio.inker import ora
from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc() -> Document:
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[2:6, 2:6] = RED
    doc.add_layer(name="Ink")
    doc.stack.active.pixels[0:3, 0:3] = BLUE
    return doc


def _animated() -> Document:
    doc = _doc()
    doc.add_frame(copy=True)
    doc.stack.active.pixels[4:8, 4:8] = BLUE
    return doc


def test_two_saves_of_an_unchanged_document_are_byte_identical():
    doc = _doc()
    assert ora.ora_bytes(doc) == ora.ora_bytes(doc)


def test_two_saves_of_an_unchanged_animated_document_are_byte_identical():
    """The animated writer takes a different branch and writes two more kinds of
    member -- the cel PNGs and ``animation.json`` -- so it needs its own case."""
    doc = _animated()
    assert ora.ora_bytes(doc) == ora.ora_bytes(doc)


def test_every_member_carries_the_fixed_epoch():
    """Named rather than merely implied by the equality above: a writer that
    stamped the wall clock at a *whole second* resolution would pass a
    back-to-back comparison and fail an hour later."""
    with zipfile.ZipFile(BytesIO(ora.ora_bytes(_animated()))) as zf:
        stamps = {info.filename: info.date_time for info in zf.infolist()}
    assert stamps, "an archive with no members would make this vacuous"
    for name, stamp in stamps.items():
        assert stamp == ora._EPOCH, name


def test_the_members_are_still_compressed():
    """``writestr`` takes the compression off a ``ZipInfo`` when it is handed
    one, and a bare ``ZipInfo`` says stored -- so fixing the timestamps is
    exactly the change that could silently stop compressing every layer."""
    with zipfile.ZipFile(BytesIO(ora.ora_bytes(_doc()))) as zf:
        by_name = {info.filename: info for info in zf.infolist()}
    assert by_name["mimetype"].compress_type == zipfile.ZIP_STORED
    for name, info in by_name.items():
        if name == "mimetype":
            continue
        assert info.compress_type == zipfile.ZIP_DEFLATED, name


def test_mimetype_is_first_and_stored_uncompressed():
    """The one member the spec is strict about: it is a magic number read at a
    fixed offset."""
    with zipfile.ZipFile(BytesIO(ora.ora_bytes(_doc()))) as zf:
        names = zf.namelist()
        assert names[0] == "mimetype"
        assert zf.read("mimetype") == b"image/openraster"


def test_a_fixed_stamp_still_reads_back(tmp_path: Path):
    """The whole risk of the change, asserted rather than assumed: a zip reader
    must not care what the timestamp says."""
    doc = _doc()
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)
    assert len(back.stack) == len(doc.stack)
    assert np.array_equal(back.stack[0].pixels, doc.stack[0].pixels)
    assert np.array_equal(back.stack[1].pixels, doc.stack[1].pixels)


def test_an_animated_file_still_reads_back_animated(tmp_path: Path):
    doc = _animated()
    path = tmp_path / "b.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)
    assert back.anim is not None
    assert len(back.anim.frames) == len(doc.anim.frames)


def test_a_changed_document_writes_a_different_file():
    """So the equality tests cannot pass by the writer emitting a constant."""
    doc = _doc()
    first = ora.ora_bytes(doc)
    doc.stack.active.pixels[7, 7] = RED
    assert ora.ora_bytes(doc) != first
