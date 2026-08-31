"""Per-frame palettes: the drawing holds still and the colours move.

**Divergence 20, retired.** The reader used to warn and keep the final table,
which silently repainted every frame the file had coloured differently.

The design decision worth stating, because everything else follows from it:
``Layer.pixels`` goes on being the *document* table's materialisation and the
frame's table is an **override beside it** (``LayerStack.frame_pixels``, keyed
by layer uid). So ``check_materialized``'s invariant is untouched, a stroke
writes where it always wrote, and both file writers read what they always read
-- only what is *shown* changes, which is all a palette swap ever was.

Rewriting the shared ``Layer`` in place would have been five lines and a trap:
a linked cel is one object in several frames, so materialising it for the
onion-skinned neighbour would repaint the frame being edited. The test that
would have caught that is
``test_flattening_another_frame_does_not_repaint_this_one``.

The model stores **whole tables**; the format stores deltas that apply from
their own frame onward. The chain is resolved once at read time and computed
back out at write time, and both directions are pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio import inker_ops, inker_state
from warlock.studio.inker import asein, aseout, ora
from warlock.studio.inker.document import Document

HOLE = (0, 0, 0, 0)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)
GREEN = (0, 255, 0, 255)


def _indexed(frames: int = 2) -> Document:
    """A 2x2 indexed document, every pixel on slot 1, with ``frames`` frames."""
    doc = Document.blank(2, 2)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([HOLE, RED], "nearest", transparent=0)
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    doc.set_current_frame(0)
    return doc


def _at(doc: Document, x: int = 0, y: int = 0):
    return tuple(np.round(doc.stack.composite_region((x, y, x + 1, y + 1))[0, 0] * 255))


# --- the model ---------------------------------------------------------------


def test_a_fresh_document_has_no_per_frame_tables():
    doc = _indexed()
    assert doc.anim.frame_palettes == {}
    assert doc.has_frame_palettes is False
    # The gate the whole feature hangs off: an ordinary composite must not pay
    # for a feature nobody used.
    assert doc.stack.frame_pixels is None


def test_palette_for_falls_back_to_the_documents_own():
    doc = _indexed()
    assert doc.palette_for(doc.anim.frames[1]) == [HOLE, RED]


def test_setting_one_frames_table_leaves_the_others_alone():
    doc = _indexed()
    assert doc.set_frame_palette([HOLE, BLUE], 1) is True
    assert doc.palette_for(doc.anim.frames[0]) == [HOLE, RED]
    assert doc.palette_for(doc.anim.frames[1]) == [HOLE, BLUE]
    assert doc.palette == [HOLE, RED], "the document's own table does not move"


def test_the_frames_own_table_is_what_it_composites_through():
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.set_current_frame(1)
    assert _at(doc) == (0.0, 0.0, 255.0, 255.0)
    doc.set_current_frame(0)
    assert _at(doc) == (255.0, 0.0, 0.0, 255.0)


def test_the_index_planes_are_not_touched():
    """The whole point: slot 4 stays slot 4 and becomes a different colour."""
    doc = _indexed()
    before = doc.stack[0].indices.copy()
    doc.set_frame_palette([HOLE, BLUE], 1)
    assert np.array_equal(doc.stack[0].indices, before)


def test_the_materialisation_invariant_still_holds():
    """``Layer.pixels`` is the *document* table's materialisation, still."""
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.set_current_frame(1)
    doc.check_materialized()


def test_setting_the_table_a_frame_already_has_changes_nothing():
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    head = doc.history.head
    assert doc.set_frame_palette([HOLE, BLUE], 1) is False
    assert doc.history.head == head


def test_clearing_puts_the_frame_back_on_the_documents_table():
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.set_current_frame(1)
    assert doc.clear_frame_palette() is True
    assert doc.palette_for() == [HOLE, RED]
    assert doc.has_frame_palettes is False


def test_a_still_document_is_refused():
    """A door that quietly did nothing would read as a bug in the palette
    rather than as the document being the wrong shape."""
    doc = Document.blank(2, 2)
    assert doc.set_frame_palette([HOLE, BLUE]) is False


def test_a_frame_index_that_is_not_there_is_refused():
    doc = _indexed()
    assert doc.set_frame_palette([HOLE, BLUE], 9) is False


def test_a_dead_frames_entry_does_not_turn_the_feature_on():
    """Entries are kept when a frame goes -- ``cel_opacity``'s rule -- so the
    gate has to walk the live frames rather than test the dict."""
    doc = _indexed(frames=3)
    doc.set_frame_palette([HOLE, BLUE], 2)
    doc.remove_frame(2)
    assert doc.anim.frame_palettes != {}, "the entry is kept for the undo"
    assert doc.has_frame_palettes is False


# --- undo --------------------------------------------------------------------


def test_one_undo_step_and_it_puts_the_colours_back():
    doc = _indexed()
    head = doc.history.head
    doc.set_frame_palette([HOLE, BLUE], 1)
    assert doc.history.head == head + 1
    doc.set_current_frame(1)
    assert _at(doc) == (0.0, 0.0, 255.0, 255.0)
    doc.undo()
    assert _at(doc) == (255.0, 0.0, 0.0, 255.0)
    assert doc.has_frame_palettes is False


def test_redo_brings_it_back():
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.undo()
    doc.redo()
    doc.set_current_frame(1)
    assert _at(doc) == (0.0, 0.0, 255.0, 255.0)


def test_the_edit_holds_its_own_copy_of_the_table():
    """The live list is the document's own object otherwise, and a later
    recolour would edit the history's idea of what came before."""
    doc = _indexed()
    table = [HOLE, BLUE]
    doc.set_frame_palette(table, 1)
    table[1] = GREEN
    assert doc.palette_for(doc.anim.frames[1]) == [HOLE, BLUE]


# --- the aliasing hazard the override exists to avoid ------------------------


def test_flattening_another_frame_does_not_repaint_this_one():
    """A linked cel is **one** ``Layer`` object in two frames.

    Materialising it in place for the onion-skinned neighbour would repaint the
    frame being edited, and it would look right until the next composite. The
    override map is what makes this test pass.
    """
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.set_current_frame(0)
    other = doc.frame_stack(doc.anim.frames[1])
    flat = other.composite_region((0, 0, 1, 1))
    assert tuple(np.round(flat[0, 0] * 255)) == (0.0, 0.0, 255.0, 255.0)
    # And frame 0 is still red afterwards, which is the half that used to break.
    assert _at(doc) == (255.0, 0.0, 0.0, 255.0)


def test_a_stroke_shows_in_the_frames_own_colours():
    """The override is patched over the dirty rectangle, or the dab would go on
    showing the pixels from before it.

    Three slots, because the interesting case needs a colour the stroke can
    move *to*: every pixel starts on slot 1, which frame 1 paints blue, and the
    dab moves one pixel to slot 2, which both tables paint green. Without the
    patch that pixel would still read blue -- the override plane as it stood
    before the stroke.
    """
    doc = Document.blank(2, 2)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([HOLE, RED, GREEN], "nearest", transparent=0)
    doc.ensure_animation()
    # Its own cel rather than a link, so the dab below is frame 1's alone.
    doc.add_frame(copy=True)
    doc.set_frame_palette([HOLE, BLUE, GREEN], 1)
    doc.set_current_frame(1)
    assert _at(doc) == (0.0, 0.0, 255.0, 255.0)

    doc.write_colour((0, 0, 1, 1), GREEN, np.ones((1, 1), dtype=np.float32))
    assert _at(doc) == (0.0, 255.0, 0.0, 255.0), "the dab landed on slot 2"
    assert _at(doc, 1, 0) == (0.0, 0.0, 255.0, 255.0), "the rest is still blue"


def test_palette_constrained_rgb_gets_no_override():
    """There the table is a rule applied to writes, not a lookup the pixels
    come through, so a different table has nothing to re-derive."""
    doc = Document.blank(2, 2)
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.set_palette([HOLE, RED])
    doc.ensure_animation()
    doc.add_frame(link=True)
    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.set_current_frame(1)
    assert doc.frame_pixels_for() is None
    assert _at(doc) == (255.0, 0.0, 0.0, 255.0)


# --- .aseprite round trip ----------------------------------------------------


def test_aseprite_round_trips_a_per_frame_table():
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    back, warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert warnings == []
    assert back.palette == [HOLE, RED]
    assert back.palette_for(back.anim.frames[1]) == [HOLE, BLUE]


def test_a_frame_that_goes_back_to_the_base_emits_its_own_chunk():
    """Compared against the *previous* frame rather than the base, or frame 2
    would silently inherit frame 1's colours."""
    doc = _indexed(frames=3)
    doc.set_frame_palette([HOLE, BLUE], 1)
    back, _warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert back.palette_for(back.anim.frames[2]) == [HOLE, RED]


def test_a_document_with_no_per_frame_table_writes_what_it_always_wrote():
    """The determinism pin: no extra chunk on any frame."""
    doc = _indexed(frames=3)
    plain = aseout.aseprite_bytes(doc)
    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.clear_frame_palette(1)
    assert aseout.aseprite_bytes(doc) == plain


# --- .ora round trip ---------------------------------------------------------


def test_ora_round_trips_a_per_frame_table(tmp_path):
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 1)
    path = tmp_path / "cycle.ora"
    ora.write_ora(doc, path)

    back = Document.load(path)
    assert back.palette_for(back.anim.frames[1]) == [HOLE, BLUE]
    # Frame 0 has no override, so it is the document's own table -- whatever
    # ORA's palette storage preserved of it. That storage drops alpha (slot 0
    # reads back opaque black), which is a limitation this feature neither
    # introduced nor repairs: ``index_plane.lut`` forces the transparent slot
    # to a hole regardless, so the picture is unaffected either way.
    assert back.palette_for(back.anim.frames[0]) == back.palette


def test_an_ora_with_no_per_frame_table_is_byte_identical(tmp_path):
    doc = _indexed(frames=3)
    plain = tmp_path / "plain.ora"
    ora.write_ora(doc, plain)
    first = plain.read_bytes()

    doc.set_frame_palette([HOLE, BLUE], 1)
    doc.clear_frame_palette(1)
    again = tmp_path / "again.ora"
    ora.write_ora(doc, again)
    assert again.read_bytes() == first


# --- the two menu rows -------------------------------------------------------


def _ops():
    return {op.name: op for op in inker_ops.OPS}


def test_the_row_is_offered_on_an_indexed_animated_frame():
    doc = _indexed()
    tab = inker_state.InkerDoc(doc=doc, title="t")
    state = inker_state.InkerState()
    assert _ops()["frame_palette"].enabled(state, tab) is True
    assert _ops()["clear_frame_palette"].enabled(state, tab) is False


def test_the_rows_swap_once_the_frame_has_a_table():
    doc = _indexed()
    doc.set_frame_palette([HOLE, BLUE], 0)
    tab = inker_state.InkerDoc(doc=doc, title="t")
    state = inker_state.InkerState()
    assert _ops()["frame_palette"].enabled(state, tab) is False
    assert _ops()["clear_frame_palette"].enabled(state, tab) is True


def test_the_row_is_refused_on_a_document_that_is_not_indexed():
    doc = Document.blank(2, 2)
    doc.ensure_animation()
    tab = inker_state.InkerDoc(doc=doc, title="t")
    state = inker_state.InkerState()
    assert _ops()["frame_palette"].enabled(state, tab) is False


@pytest.mark.parametrize("name", ["frame_palette", "clear_frame_palette"])
def test_neither_row_is_offered_without_a_document(name):
    assert _ops()[name].enabled(inker_state.InkerState(), None) is False
