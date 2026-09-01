"""Nine numbered stamps, and the second mouse button's first job.

Three decisions are pinned here rather than left to the code.

**Slots live on the map.** A stamp is an array of gids, and gids are numbered
against one map's firstgids -- which is exactly why ``plotter_mode._paste``
refuses a cross-document tile paste by name. They were on the *tab* until
2026-09-01, which had that property and lost every slot on a close; the document
has it and is saved.

**Bare digits recall and the chord stores.** Recall happens hundreds of times in
a session and storing nine times, so the cheap gesture goes to the frequent one
-- the opposite way round would be a hand reaching for Ctrl+Shift on every stamp.

**A stamp is undoable and dirties the map.** It is written into the ``.wmap``,
so a map with a stamp stored and not saved really does have unsaved work in it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from warlock.studio import plotter_mode, plotter_state
from warlock.studio.plotter.tilemap import MapDoc


def _session():
    doc = MapDoc(8, 8, 16, 16)
    doc.add_tile_layer()
    doc.history.clear()
    tab = plotter_state.PlotterDoc(doc=doc, title="m")
    state = plotter_state.PlotterState()
    state.add(tab)
    sent: list[tuple[str, str]] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(plotter=state),
        toast=lambda text, level="info": sent.append((text, level)),
    )
    return ctx, state, tab, sent


# --- where the slots live ----------------------------------------------------


def test_the_slots_belong_to_the_map():
    _ctx, _state, tab, _sent = _session()
    assert tab.doc.stamps == {}
    # Neither the app state nor the tab carries a copy: one home, and it is the
    # one thing a gid is numbered against.
    assert not hasattr(plotter_state.PlotterState(), "stamps")
    assert not hasattr(tab, "stamps")


def test_two_maps_do_not_share_slots():
    ctx, state, tab, _sent = _session()
    other_doc = MapDoc(4, 4, 8, 8)
    other_doc.add_tile_layer()
    other = plotter_state.PlotterDoc(doc=other_doc, title="other")
    state.add(other)
    state.brush = np.array([[1]], dtype=np.uint32)
    plotter_mode.store_stamp(ctx, state, tab, 1)

    assert other.doc.stamps == {}


# --- storing and recalling ---------------------------------------------------


def test_storing_and_recalling_a_stamp():
    ctx, state, tab, _sent = _session()
    state.brush = np.array([[3, 4]], dtype=np.uint32)
    assert plotter_mode.store_stamp(ctx, state, tab, 2)
    state.brush = None
    state.tool = "fill"
    assert plotter_mode.recall_stamp(ctx, state, tab, 2)
    assert np.array_equal(state.brush, np.array([[3, 4]], dtype=np.uint32))
    # Recall puts the tool that can put it down in hand: a loaded brush with no
    # way to use it is a gesture that stops halfway.
    assert state.tool == "stamp"


def test_a_stored_stamp_is_a_copy_and_a_recalled_one_is_too():
    """Otherwise the next capture would rewrite the slot under the user -- and
    a transform of the recalled brush would rewrite it after the fact."""

    ctx, state, tab, _sent = _session()
    state.brush = np.array([[3]], dtype=np.uint32)
    plotter_mode.store_stamp(ctx, state, tab, 1)
    state.brush[0, 0] = 9
    assert tab.doc.stamps[1].cells[0, 0] == 3

    plotter_mode.recall_stamp(ctx, state, tab, 1)
    state.brush[0, 0] = 7
    assert tab.doc.stamps[1].cells[0, 0] == 3


def test_the_stored_block_is_read_only():
    """The copy is the guard; the freeze is what makes a caller that got past
    it fail loudly rather than silently editing a saved stamp."""
    ctx, state, tab, _sent = _session()
    state.brush = np.array([[3]], dtype=np.uint32)
    plotter_mode.store_stamp(ctx, state, tab, 1)
    assert not tab.doc.stamps[1].cells.flags.writeable


def test_storing_nothing_is_refused_out_loud():
    ctx, state, tab, sent = _session()
    state.brush = None
    assert plotter_mode.store_stamp(ctx, state, tab, 1) is False
    assert sent and "no stamp" in sent[0][0].lower()


def test_recalling_an_empty_slot_says_how_to_fill_it():
    ctx, state, tab, sent = _session()
    assert plotter_mode.recall_stamp(ctx, state, tab, 5) is False
    assert sent and "Ctrl+Shift+5" in sent[0][0]


def test_storing_while_the_map_is_saving_is_refused():
    """New with the move onto the document: this pushes an undo step now, where
    it used to write a dict on the tab."""
    ctx, state, tab, sent = _session()
    state.brush = np.array([[1]], dtype=np.uint32)
    tab.saving = True

    assert plotter_mode.store_stamp(ctx, state, tab, 1) is False
    assert tab.doc.stamps == {}
    assert sent and "being written" in sent[0][0]


# --- naming, clearing, and undo ----------------------------------------------


def test_a_slot_can_be_named_and_the_name_survives_a_re_store():
    """Re-capturing a better block is not a rename, and asking the user to
    retype the name every time would make naming a slot not worth doing."""
    _ctx, _state, tab, _sent = _session()
    doc = tab.doc
    doc.set_stamp(3, np.array([[1]], dtype=np.uint32))
    assert doc.rename_stamp(3, "roof corner")

    doc.set_stamp(3, np.array([[2, 2]], dtype=np.uint32))

    assert doc.stamps[3].name == "roof corner"
    assert doc.stamps[3].cells.tolist() == [[2, 2]]


def test_renaming_an_empty_slot_is_refused_rather_than_minting_one():
    """A name with no cells behind it is a row that recalls nothing."""
    _ctx, _state, tab, _sent = _session()
    assert tab.doc.rename_stamp(4, "nothing here") is False
    assert 4 not in tab.doc.stamps


def test_storing_the_same_block_twice_pushes_nothing():
    """``TilePatchEdit``'s rule: a write that changes nothing pushes nothing,
    so re-storing an identical stamp does not dirty the map."""
    _ctx, _state, tab, _sent = _session()
    block = np.array([[1, 2]], dtype=np.uint32)
    assert tab.doc.set_stamp(1, block)
    head = tab.doc.history.head

    assert tab.doc.set_stamp(1, block) is False
    assert tab.doc.history.head == head


def test_every_stamp_verb_is_one_undo_step():
    _ctx, _state, tab, _sent = _session()
    doc = tab.doc
    doc.set_stamp(1, np.array([[5]], dtype=np.uint32))
    doc.rename_stamp(1, "wall")
    doc.clear_stamp(1)
    assert len(doc.history) == 3

    doc.undo()
    assert doc.stamps[1].name == "wall"
    doc.undo()
    assert doc.stamps[1].name == ""
    doc.undo()
    assert doc.stamps == {}


def test_clearing_an_empty_slot_does_nothing_and_says_so():
    _ctx, _state, tab, _sent = _session()
    head = tab.doc.history.head
    assert tab.doc.clear_stamp(9) is False
    assert tab.doc.history.head == head


def test_a_stored_stamp_makes_the_map_dirty():
    """Honest rather than unfortunate: the stamps are written into the file."""
    _ctx, _state, tab, _sent = _session()
    tab.doc.mark_saved()
    assert not tab.doc.dirty

    tab.doc.set_stamp(1, np.array([[1]], dtype=np.uint32))

    assert tab.doc.dirty


# --- the object reorder tests that share this file ---------------------------


def test_an_object_reorder_is_one_undo_step():
    """Order is draw order, and Tiled's Raise/Lower is exactly this. Two steps
    would put a state on the stack in which the object does not exist."""

    from warlock.studio.plotter.tilemap import MapObject, new_uid

    doc = MapDoc(8, 8, 16, 16)
    layer = doc.add_object_layer()
    first = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="rect"))
    doc.add_object(layer.uid, MapObject(uid=new_uid(), name="b", kind="rect"))
    head = doc.history.head

    assert doc.reorder_object(layer.uid, first.uid, 1) is True
    assert [obj.name for obj in doc.layer(layer.uid).objects] == ["b", "a"]
    assert doc.history.head == head + 1

    doc.undo()
    assert [obj.name for obj in doc.layer(layer.uid).objects] == ["a", "b"]


def test_a_reorder_off_either_end_does_nothing_and_says_so():
    from warlock.studio.plotter.tilemap import MapObject, new_uid

    doc = MapDoc(8, 8, 16, 16)
    layer = doc.add_object_layer()
    only = doc.add_object(layer.uid, MapObject(uid=new_uid(), name="a", kind="rect"))
    head = doc.history.head
    assert doc.reorder_object(layer.uid, only.uid, -1) is False
    assert doc.reorder_object(layer.uid, only.uid, 1) is False
    assert doc.history.head == head
