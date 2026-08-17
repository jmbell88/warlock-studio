"""Indexed and grayscale documents on disk: P-PNG planes and a `color` block.

Three claims, and each has its own section below.

*The archive is the record.* Index planes are written as PNG mode-P images --
``PLTE`` for the table, ``tRNS`` for its alpha, the pixel bytes are the plane
verbatim -- so the file is self-describing and every other reader in the world
decodes it correctly by accident.

*Nothing about an RGB document changed.* The colour block is written only when
there is one, so a plain document's archive is byte-for-byte what this writer
produced before any of this existed.

*Degradation is a ladder, not a cliff.* A missing block, a broken block, RGBA
planes under an indexed block -- each costs a little and never the file.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from warlock.studio.inker import Document
from warlock.studio.inker import index_plane as ixp
from warlock.studio.inker.ora import WARLOCK_MEMBER, read_ora, write_ora

BLACK = (0, 0, 0, 255)
RED = (200, 20, 20, 255)
GREEN = (20, 200, 20, 255)
#: A *translucent* swatch, which is the entry that proves the table's alpha
#: survives the ``.gpl`` projection's inability to carry it.
GHOST = (10, 10, 240, 128)
PALETTE = [BLACK, RED, GREEN, RED, GHOST]


def _doc(width: int = 6, height: int = 4) -> Document:
    doc = Document.blank(width, height)
    doc.stack.active.pixels[:, :] = RED
    doc.convert_to_indexed(PALETTE)
    # One pixel in the duplicate red, one in the hole, one translucent: between
    # them they exercise identity, transparency and palette alpha.
    doc.stack.active.indices[0, 0] = 3
    doc.stack.active.indices[1, 1] = 0
    doc.stack.active.indices[2, 2] = 4
    doc._rematerialize(doc.stack.active)
    doc.check_materialized()
    return doc


# --- the round trip ---------------------------------------------------------


def test_an_indexed_document_round_trips_index_exactly(tmp_path):
    doc = _doc()
    path = tmp_path / "indexed.ora"
    write_ora(doc, path)

    back = read_ora(path)
    back.check_materialized()

    assert back.is_indexed and back.color_mode == "indexed"
    assert back.transparent_index == doc.transparent_index
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices)
    assert np.array_equal(back.stack.active.pixels, doc.stack.active.pixels)
    # Index-exact means the *duplicate* survived, which no RGBA round trip can
    # promise: slots 1 and 3 are the same red.
    assert back.stack.active.indices[0, 0] == 3
    assert back.stack.active.indices[0, 1] == 1


def test_the_palette_keeps_its_alpha_which_the_gpl_projection_cannot_carry(tmp_path):
    doc = _doc()
    path = tmp_path / "alpha.ora"
    write_ora(doc, path)

    back = read_ora(path)
    assert back.palette == PALETTE, "including the translucent swatch and the hole's colour"


def test_the_transparent_slot_keeps_its_colour_and_its_opacity(tmp_path):
    """Its ``tRNS`` byte is forced to zero on the way out -- it has to be, or the
    hole is not a hole to any other reader -- so reading it back literally would
    be reading the writer's own requirement as though it were user data. The RGB
    is what matters and it survives untouched."""
    doc = _doc()
    path = tmp_path / "hole.ora"
    write_ora(doc, path)

    with zipfile.ZipFile(path) as zf, Image.open(zf.open("data/layer0.png")) as im:
        assert im.mode == "P"
        assert bytes(im.info["transparency"])[0] == 0

    assert read_ora(path).palette[0] == BLACK


def test_saving_twice_is_byte_identical(tmp_path):
    doc = _doc()
    write_ora(doc, tmp_path / "one.ora")
    write_ora(doc, tmp_path / "two.ora")
    assert (tmp_path / "one.ora").read_bytes() == (tmp_path / "two.ora").read_bytes()


def test_open_and_resave_is_a_fixpoint(tmp_path):
    """The property that makes a save something a diff, a hash or a sync client
    can trust. It also catches every asymmetry between the writer and the
    reader, which is why it is worth having beside the field-by-field checks."""
    doc = _doc()
    write_ora(doc, tmp_path / "a.ora")
    write_ora(read_ora(tmp_path / "a.ora"), tmp_path / "b.ora")
    assert (tmp_path / "a.ora").read_bytes() == (tmp_path / "b.ora").read_bytes()


def test_an_animated_indexed_document_round_trips_with_its_links(tmp_path):
    doc = _doc()
    doc.add_frame(link=True)
    write_ora(doc, tmp_path / "clip.ora")

    back = read_ora(tmp_path / "clip.ora")
    back.check_materialized()
    assert back.is_indexed
    cels = list(back.anim.unique_cel_layers())
    assert len(cels) == 1, "one shared cel, so one shared index plane"
    assert np.array_equal(cels[0].indices, doc.stack.active.indices)


def test_a_grayscale_document_writes_ordinary_rgba_planes(tmp_path):
    """Grayscale is a constraint over RGBA storage, so there is nothing to
    encode differently -- only a mode to record."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = (90, 90, 90, 255)
    doc.color_mode = "grayscale"
    write_ora(doc, tmp_path / "gray.ora")

    with (
        zipfile.ZipFile(tmp_path / "gray.ora") as zf,
        Image.open(zf.open("data/layer0.png")) as im,
    ):
        assert im.mode == "RGBA"

    back = read_ora(tmp_path / "gray.ora")
    assert back.color_mode == "grayscale"
    assert tuple(back.stack.active.pixels[0, 0]) == (90, 90, 90, 255)


# --- the negative control ---------------------------------------------------


def test_an_rgb_archive_carries_no_colour_block_at_all(tmp_path):
    """Additive means invisible. The block is written only when it is not the
    default, which is what keeps ``WARLOCK_VERSION`` at 1 and every existing
    document's bytes exactly what they were."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    write_ora(doc, tmp_path / "plain.ora")

    with zipfile.ZipFile(tmp_path / "plain.ora") as zf:
        assert WARLOCK_MEMBER not in zf.namelist()
        with Image.open(zf.open("data/layer0.png")) as im:
            assert im.mode == "RGBA"


def test_a_palette_constrained_rgb_archive_carries_no_colour_block(tmp_path):
    """The mode that every document written before true indexing is in. Its
    bytes must not move either."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :] = RED
    doc.set_palette([BLACK, RED, GREEN])
    write_ora(doc, tmp_path / "constrained.ora")

    with zipfile.ZipFile(tmp_path / "constrained.ora") as zf:
        assert WARLOCK_MEMBER not in zf.namelist()
        assert "palette.gpl" in zf.namelist()

    back = read_ora(tmp_path / "constrained.ora")
    assert back.is_palette_locked and not back.is_indexed


def test_a_sliced_rgb_document_still_writes_slices_and_no_colour_block(tmp_path):
    """The member is shared, so the two additions have to be independent."""
    doc = Document.blank(4, 4)
    doc.add_slice((0, 0, 2, 2), name="hit")
    write_ora(doc, tmp_path / "sliced.ora")

    with zipfile.ZipFile(tmp_path / "sliced.ora") as zf:
        payload = json.loads(zf.read(WARLOCK_MEMBER))
    assert payload["slices"] and "color" not in payload


def test_an_indexed_document_with_slices_carries_both(tmp_path):
    doc = _doc()
    doc.add_slice((0, 0, 2, 2), name="hit")
    write_ora(doc, tmp_path / "both.ora")

    back = read_ora(tmp_path / "both.ora")
    assert back.is_indexed and len(back.slices) == 1


# --- the degradation ladder -------------------------------------------------


def _rewrite(path, changes):
    """Rebuild an archive with some members replaced or dropped.

    A ``None`` value drops the member. Written out rather than edited in place
    because a zip cannot be edited in place, and because rebuilding is what an
    older build of this app would do anyway -- which is the case under test.
    """
    with zipfile.ZipFile(path) as zf:
        members = [(name, zf.read(name)) for name in zf.namelist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            if name in changes:
                data = changes[name]
                if data is None:
                    continue
            zf.writestr(name, data)


def test_a_missing_colour_block_opens_as_the_picture_it_is(tmp_path):
    """The P-PNGs are still self-describing, but nothing says the document is
    indexed -- so it opens as the RGBA picture Krita would show, which is
    exactly right and exactly what an older build already did."""
    doc = _doc()
    write_ora(doc, tmp_path / "x.ora")
    _rewrite(tmp_path / "x.ora", {WARLOCK_MEMBER: None})

    back = read_ora(tmp_path / "x.ora")
    assert not back.is_indexed
    assert back.is_palette_locked, "the .gpl projection still constrains it"
    assert tuple(back.stack.active.pixels[0, 1]) == RED


def test_a_malformed_colour_block_costs_the_mode_and_not_the_file(tmp_path, caplog):
    doc = _doc()
    write_ora(doc, tmp_path / "x.ora")
    _rewrite(tmp_path / "x.ora", {WARLOCK_MEMBER: b"{ this is not json"})

    back = read_ora(tmp_path / "x.ora")
    assert not back.is_indexed
    assert tuple(back.stack.active.pixels[0, 1]) == RED


def test_an_unknown_colour_mode_is_ignored_rather_than_adopted(tmp_path):
    doc = _doc()
    write_ora(doc, tmp_path / "x.ora")
    payload = {"version": 1, "slices": [], "color": {"mode": "cmyk"}}
    _rewrite(tmp_path / "x.ora", {WARLOCK_MEMBER: json.dumps(payload).encode()})

    back = read_ora(tmp_path / "x.ora")
    assert back.color_mode == "rgb"


def test_a_transparent_index_outside_the_table_falls_back_to_zero(tmp_path):
    doc = _doc()
    write_ora(doc, tmp_path / "x.ora")
    payload = {"version": 1, "slices": [], "color": {"mode": "indexed", "transparent": 99}}
    _rewrite(tmp_path / "x.ora", {WARLOCK_MEMBER: json.dumps(payload).encode()})

    back = read_ora(tmp_path / "x.ora")
    assert back.is_indexed and back.transparent_index == 0
    back.check_materialized()


def test_rgba_planes_under_an_indexed_block_are_re_inferred(tmp_path):
    """The stated forward-compat cost, made concrete: an old build opened the
    file, re-saved it with RGBA planes, and this build opens it again. The
    picture is intact; duplicate-slot identity is gone, and slot 3's pixel comes
    back as slot 1."""
    doc = _doc()
    write_ora(doc, tmp_path / "x.ora")
    flat = Document.blank(6, 4)
    flat.stack.active.pixels[...] = doc.stack.active.pixels
    write_ora(flat, tmp_path / "flat.ora")
    with zipfile.ZipFile(tmp_path / "flat.ora") as zf:
        rgba = zf.read("data/layer0.png")
    _rewrite(tmp_path / "x.ora", {"data/layer0.png": rgba})

    back = read_ora(tmp_path / "x.ora")
    back.check_materialized()
    assert back.is_indexed
    assert back.stack.active.indices[0, 0] == 1, "the duplicate is gone"
    assert back.stack.active.indices[1, 1] == 0, "but the hole is still a hole"
    # The picture is intact where the ``.gpl`` fallback table can describe it.
    # The translucent swatch is the one exception and is the *other* half of the
    # same cost: ``.gpl`` carries no alpha, so a document that lost its P-PNGs
    # lost the table's alpha with them.
    opaque = doc.stack.active.pixels[..., 3] == 255
    assert np.array_equal(
        back.stack.active.pixels[..., :3][opaque], doc.stack.active.pixels[..., :3][opaque]
    )


def test_an_indexed_block_with_no_table_anywhere_opens_as_rgb(tmp_path):
    """An indexed document with nothing to index onto is not a document."""
    doc = _doc()
    write_ora(doc, tmp_path / "x.ora")
    flat = Document.blank(6, 4)
    flat.stack.active.pixels[...] = doc.stack.active.pixels
    write_ora(flat, tmp_path / "flat.ora")
    with zipfile.ZipFile(tmp_path / "flat.ora") as zf:
        rgba = zf.read("data/layer0.png")
    _rewrite(tmp_path / "x.ora", {"data/layer0.png": rgba, "palette.gpl": None})

    back = read_ora(tmp_path / "x.ora")
    assert back.color_mode == "rgb" and back.palette is None


def test_an_old_reader_sees_the_right_pixels(tmp_path):
    """The whole reason a P-PNG was chosen over a container of our own: every
    decoder in the world honours ``PLTE`` + ``tRNS``, so a build that has never
    heard of index planes -- and Krita, and GIMP -- opens the file correctly."""
    doc = _doc()
    write_ora(doc, tmp_path / "x.ora")

    with (
        zipfile.ZipFile(tmp_path / "x.ora") as zf,
        Image.open(zf.open("data/layer0.png")) as im,
    ):
        seen = np.asarray(im.convert("RGBA"), dtype=np.uint8)

    want = doc.stack.active.pixels
    visible = want[..., 3] > 0
    assert np.array_equal(seen[..., 3], want[..., 3])
    assert np.array_equal(seen[..., :3][visible], want[..., :3][visible])


def test_the_writer_keeps_slots_nothing_is_painted_in(tmp_path):
    """Pillow's PNG optimiser drops unused palette entries and renumbers the
    plane -- which is exactly the identity this member exists to preserve. A
    slot the user authored and has not used yet is still a slot."""
    doc = _doc()
    doc.add_slot((7, 7, 7, 255))
    write_ora(doc, tmp_path / "x.ora")

    back = read_ora(tmp_path / "x.ora")
    assert len(back.palette) == len(PALETTE) + 1
    assert back.palette[-1] == (7, 7, 7, 255)
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices)


# --- crash recovery ---------------------------------------------------------


def test_the_crash_journal_carries_the_indices(tmp_path):
    """The journal's payload is ``ora_bytes(doc)``, so recovery becomes honest
    for free -- but "for free" is a claim worth a test, because it is only true
    while the ORA writer is the one that carries index planes."""
    from warlock.studio import inker

    doc = _doc()
    blob = inker.ora_bytes(doc)
    path = tmp_path / "recovered.ora"
    path.write_bytes(blob)

    back = read_ora(path)
    back.check_materialized()
    assert back.is_indexed
    assert np.array_equal(back.stack.active.indices, doc.stack.active.indices)


# --- the P-PNG encoder itself -----------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 255, 256])
def test_every_legal_palette_size_encodes_and_decodes(tmp_path, count):
    from warlock.studio.inker.ora import _png_indexed, _read_indexed_png

    palette = [(i, (i * 7) % 256, (i * 13) % 256, 255) for i in range(count)]
    plane = (np.arange(16).reshape(4, 4) % count).astype(np.uint8)
    data = _png_indexed(plane, palette, 0)

    got = _read_indexed_png(data, (4, 4))
    assert got is not None
    indices, table = got
    assert np.array_equal(indices, plane)
    assert len(table) >= count
    assert table[0][3] == 0
    assert [tuple(c[:3]) for c in table[:count]] == [c[:3] for c in palette]


def test_reading_an_rgba_png_as_an_index_plane_answers_none():
    from warlock.studio.inker.ora import _png, _read_indexed_png

    data = _png(np.zeros((4, 4, 4), np.uint8))
    assert _read_indexed_png(data, (4, 4)) is None


def test_a_plane_of_the_wrong_size_is_refused_rather_than_placed():
    """A P-PNG whose dimensions do not match the canvas is a file we did not
    write. Placing it would guess an offset; answering None re-infers from the
    pixels, which the RGBA path has already placed correctly."""
    from warlock.studio.inker.ora import _png_indexed, _read_indexed_png

    data = _png_indexed(np.zeros((2, 2), np.uint8), [BLACK, RED], 0)
    assert _read_indexed_png(data, (4, 4)) is None


def test_the_encoder_and_index_plane_agree_about_the_hole():
    from warlock.studio.inker.ora import _png_indexed, _read_indexed_png

    plane = np.asarray([[0, 1]], dtype=np.uint8)
    got = _read_indexed_png(_png_indexed(plane, [BLACK, RED], 0), (2, 1))
    assert got is not None
    indices, table = got
    rendered = ixp.materialize(indices, ixp.lut(table, 0))
    assert tuple(rendered[0, 0]) == (0, 0, 0, 0)
    assert tuple(rendered[0, 1]) == RED
