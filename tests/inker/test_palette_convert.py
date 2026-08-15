"""Converting a document onto a palette, with a dither or without one.

``set_palette`` is the ``method="nearest"`` case of this and nothing else, which
is the property most of these assert: whatever the arithmetic, a conversion is
one undo step across every layer and every frame, it carries the table with it,
it survives links, and it ignores the selection because it is a change of mode
rather than a write.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import dither
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
GREY = (128, 128, 128, 255)
RAMP = [BLACK, GREY, WHITE]


def _ramp_doc(width: int = 32, height: int = 8) -> Document:
    doc = Document.blank(width, height)
    row = np.linspace(0, 255, width).astype(np.uint8)
    doc.stack.active.pixels[..., :3] = row[None, :, None]
    doc.stack.active.pixels[..., 3] = 255
    return doc


def _used(pixels: np.ndarray) -> set[tuple[int, int, int]]:
    visible = pixels[..., 3] > 0
    return {tuple(int(c) for c in rgb) for rgb in pixels[..., :3][visible]}


# --- the conversion ----------------------------------------------------------


@pytest.mark.parametrize("method", dither.METHODS)
def test_a_conversion_puts_the_document_on_the_table(method):
    doc = _ramp_doc()
    assert doc.convert_to_palette(RAMP, method) is True
    assert doc.palette == RAMP
    assert _used(doc.stack.active.pixels) <= {c[:3] for c in RAMP}


def test_nearest_is_exactly_what_set_palette_does():
    """One path, so the two cannot drift: an indexed document and a
    nearest-converted one are the same bytes and the same history depth."""
    indexed = _ramp_doc()
    converted = _ramp_doc()
    indexed.set_palette(RAMP)
    converted.convert_to_palette(RAMP, "nearest")
    assert np.array_equal(indexed.stack.active.pixels, converted.stack.active.pixels)
    assert len(indexed.history) == len(converted.history)


def test_a_dither_is_not_the_nearest_conversion():
    plain = _ramp_doc()
    dithered = _ramp_doc()
    plain.convert_to_palette(RAMP, "nearest")
    dithered.convert_to_palette(RAMP, "bayer4")
    assert not np.array_equal(plain.stack.active.pixels, dithered.stack.active.pixels)


def test_a_conversion_is_one_undo_step_that_restores_the_pixels_and_the_table():
    doc = _ramp_doc()
    before = doc.stack.active.pixels.copy()
    depth = len(doc.history)

    doc.convert_to_palette(RAMP, "floyd-steinberg")
    doc.undo()

    assert len(doc.history) == depth
    assert doc.palette is None
    assert np.array_equal(doc.stack.active.pixels, before)


def test_redoing_a_dithered_conversion_lands_on_the_same_pixels():
    """Redo *replays* rather than storing a second copy, which is only sound
    because a conversion is a pure function of the document and the table --
    every method here is deterministic, dithers included."""
    doc = _ramp_doc()
    doc.convert_to_palette(RAMP, "bayer8")
    expected = doc.stack.active.pixels.copy()

    doc.undo()
    doc.redo()

    assert np.array_equal(doc.stack.active.pixels, expected)
    assert doc.palette == RAMP


def test_every_layer_is_converted_in_the_one_step():
    doc = _ramp_doc()
    doc.add_layer(name="Second")
    doc.stack.active.pixels[:] = (200, 200, 200, 255)

    doc.convert_to_palette(RAMP, "nearest")

    assert _used(doc.stack[0].pixels) <= {c[:3] for c in RAMP}
    assert _used(doc.stack[1].pixels) == {WHITE[:3]}


def test_a_linked_cel_is_converted_once_and_stays_linked():
    doc = _ramp_doc(8, 8)
    doc.add_frame(copy=False, link=True)
    assert doc.anim is not None
    shared = doc.anim.cels
    assert len({id(layer) for layer in shared.values()}) == 1

    doc.convert_to_palette(RAMP, "bayer4")
    doc.undo()

    assert len({id(layer) for layer in doc.anim.cels.values()}) == 1


def test_the_selection_is_ignored_and_survives():
    """A change of mode, not a write: converting only the marquee would leave
    the pixels outside it off the palette they are now declared to be on."""
    doc = _ramp_doc()
    mask = np.zeros((8, 32), dtype=np.uint8)
    mask[:, 8:16] = 255
    doc.select(SelectionMask(mask), "replace")

    doc.convert_to_palette(RAMP, "bayer4")

    assert _used(doc.stack.active.pixels) <= {c[:3] for c in RAMP}
    assert np.array_equal(doc.mask.mask, mask)


def test_converting_to_the_same_table_by_nearest_does_nothing():
    doc = _ramp_doc()
    doc.set_palette(RAMP)
    depth = len(doc.history)
    assert doc.convert_to_palette(RAMP, "nearest") is False
    assert len(doc.history) == depth


def test_re_dithering_onto_the_same_table_is_a_real_request():
    """How a user compares two matrices -- and it does move pixels, so it must
    not be swallowed by the same-table shortcut."""
    doc = _ramp_doc()
    doc.convert_to_palette(RAMP, "nearest")
    assert doc.convert_to_palette(RAMP, "bayer2") is True


def test_an_empty_table_is_refused():
    with pytest.raises(ValueError):
        _ramp_doc().convert_to_palette([], "nearest")


def test_the_document_is_left_snapped_so_the_next_write_moves_nothing():
    """The property the whole mode rests on: after a dither, every visible pixel
    is exactly on a swatch, so ``_commit_patch``'s snap is a no-op and a stroke
    does not quietly re-flatten the dithering around it."""
    doc = _ramp_doc()
    doc.convert_to_palette(RAMP, "bayer8")
    untouched = doc.stack.active.pixels.copy()

    doc.write_colour((0, 0, 1, 1), GREY, np.ones((1, 1), dtype=np.float32))

    assert np.array_equal(doc.stack.active.pixels[:, 1:], untouched[:, 1:])


# --- a palette out of the document's own pixels ------------------------------


def test_a_palette_from_the_document_indexes_it_to_its_own_colours():
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:, :2] = (10, 20, 30, 255)
    doc.stack.active.pixels[:, 2:] = (200, 210, 220, 255)

    assert doc.palette_from_document(8) is True

    assert doc.palette == [(10, 20, 30, 255), (200, 210, 220, 255)]
    assert _used(doc.stack.active.pixels) == {(10, 20, 30), (200, 210, 220)}


def test_a_palette_from_the_document_is_cut_to_the_budget():
    doc = _ramp_doc(64, 4)
    assert doc.palette_from_document(6) is True
    assert 1 <= len(doc.palette) <= 6
    assert _used(doc.stack.active.pixels) <= {c[:3] for c in doc.palette}


def test_a_palette_from_the_document_can_dither_onto_itself():
    doc = _ramp_doc(64, 4)
    assert doc.palette_from_document(4, "floyd-steinberg") is True
    assert _used(doc.stack.active.pixels) <= {c[:3] for c in doc.palette}


def test_a_linked_cel_is_weighed_once_when_the_palette_is_built():
    """``unique_cel_layers``: weighting the median cut by how many frames a cel
    happens to appear on is not a fact about the drawing."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:] = (10, 20, 30, 255)
    doc.add_frame(copy=False, link=True)
    doc.add_frame(copy=False, link=True)

    assert len(doc._palette_planes()) == 1


def test_building_a_palette_is_one_undo_step_like_any_other_conversion():
    doc = _ramp_doc(16, 4)
    before = doc.stack.active.pixels.copy()

    doc.palette_from_document(4)
    doc.undo()

    assert doc.palette is None
    assert np.array_equal(doc.stack.active.pixels, before)
