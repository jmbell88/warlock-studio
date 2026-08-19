"""Grouped as an *indexed* conversion: the invariants the mode is built on.

Grouped enters indexed mode through the same door every other method does, so
what is asserted here is not the arithmetic (``test_dither_grouped.py`` has
that) but that the arithmetic did not cost the document any of the four things
an index plane exists to guarantee: the planes agree with the pixels, the file
round-trips, no opaque pixel is parked on the transparent slot, and the whole
mode change is one undo step.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import Document
from warlock.studio.inker import index_plane as ixp
from warlock.studio.inker.ora import read_ora, write_ora

BLACK = (0, 0, 0, 255)
MID = (120, 120, 120, 255)
WHITE = (255, 255, 255, 255)
PALETTE = [BLACK, MID, WHITE]


def _band(width: int = 12, height: int = 3) -> Document:
    """A narrow grey band -- the case nearest collapses and grouped does not."""
    doc = Document.blank(width, height)
    doc.stack.active.pixels[..., 3] = 255
    for x in range(width):
        doc.stack.active.pixels[:, x, :3] = 140 + x * 5
    return doc


def test_a_grouped_indexed_conversion_holds_the_materialisation() -> None:
    doc = _band()
    assert doc.convert_to_indexed(PALETTE, "grouped") is True
    doc.check_materialized()


def test_a_grouped_indexed_document_round_trips_through_ora(tmp_path) -> None:
    doc = _band()
    doc.convert_to_indexed(PALETTE, "grouped")
    path = tmp_path / "grouped.ora"
    write_ora(doc, path)
    back = read_ora(path)
    assert back.palette == doc.palette
    assert np.array_equal(back.stack[0].indices, doc.stack[0].indices)


def test_no_opaque_pixel_lands_on_the_transparent_slot() -> None:
    """The transparent slot is not a candidate by construction: the grouped
    table is built over the candidate slots and never sees it."""
    doc = _band()
    doc.convert_to_indexed(PALETTE, "grouped", transparent=0)
    plane = doc.stack[0]
    opaque = plane.pixels[..., 3] >= ixp.OPAQUE_THRESHOLD
    assert opaque.any()
    assert 0 not in set(np.unique(plane.indices[opaque]).tolist())


def test_entering_indexed_mode_by_grouping_is_one_undo_step() -> None:
    doc = _band()
    before = doc.stack.active.pixels.copy()
    doc.convert_to_indexed(PALETTE, "grouped")
    assert doc.is_indexed
    assert doc.undo() is True
    assert not doc.is_indexed
    assert np.array_equal(doc.stack.active.pixels, before)


def test_the_grouping_is_built_before_the_planes_are_rewritten() -> None:
    """The resolve loop writes each layer's pixels through ``materialize``, so a
    table built inside it would group the later layers against a document that
    no longer exists. Two layers on disjoint ranges is where that shows."""
    doc = Document.blank(4, 4)
    doc.stack.active.pixels[:] = (20, 20, 20, 255)
    doc.add_layer(name="Light")
    doc.stack.active.pixels[:] = (230, 230, 230, 255)

    doc.convert_to_indexed(PALETTE, "grouped")
    dark, light = doc.stack[0], doc.stack[1]
    # Slot 0 is the hole and never a candidate, so the two targets in the
    # running are MID and WHITE -- and the two layers take one each rather than
    # both landing on whichever is absolutely nearer.
    assert set(np.unique(dark.pixels[..., 0]).tolist()) == {120}
    assert set(np.unique(light.pixels[..., 0]).tolist()) == {255}
