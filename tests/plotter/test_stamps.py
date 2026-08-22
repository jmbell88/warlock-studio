"""Nine numbered stamps, and the second mouse button's first job.

Two decisions are pinned here rather than left to the code.

**Slots live on the tab.** A stamp is an array of gids, and gids are numbered
against one map's firstgids -- which is exactly why ``plotter_mode._paste``
refuses a cross-document tile paste by name. Sharing them app-wide would paste
one map's tiles into another under a different tileset.

**Bare digits recall and the chord stores.** Recall happens hundreds of times
in a session and storing nine times, so the cheap gesture goes to the frequent
one -- the opposite way round would be a hand reaching for Ctrl+Shift on every
stamp.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from warlock.studio import plotter_mode, plotter_state
from warlock.studio.plotter.tilemap import MapDoc


def _session():
    doc = MapDoc(8, 8, 16, 16)
    doc.add_tile_layer()
    tab = plotter_state.PlotterDoc(doc=doc, title="m")
    state = plotter_state.PlotterState()
    state.add(tab)
    sent: list[tuple[str, str]] = []
    ctx = SimpleNamespace(
        state=SimpleNamespace(plotter=state),
        toast=lambda text, level="info": sent.append((text, level)),
    )
    return ctx, state, tab, sent


def test_the_slots_belong_to_the_tab():
    _ctx, _state, tab, _sent = _session()
    assert tab.stamps == {}
    assert not hasattr(plotter_state.PlotterState(), "stamps")


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


def test_a_stored_stamp_is_a_copy():
    """Otherwise the next capture would rewrite the slot under the user."""

    ctx, state, tab, _sent = _session()
    state.brush = np.array([[3]], dtype=np.uint32)
    plotter_mode.store_stamp(ctx, state, tab, 1)
    state.brush[0, 0] = 9
    assert tab.stamps[1][0, 0] == 3
    plotter_mode.recall_stamp(ctx, state, tab, 1)
    state.brush[0, 0] = 7
    assert tab.stamps[1][0, 0] == 3


def test_storing_nothing_is_refused_out_loud():
    ctx, state, tab, sent = _session()
    state.brush = None
    assert plotter_mode.store_stamp(ctx, state, tab, 1) is False
    assert sent and "no stamp" in sent[0][0].lower()


def test_recalling_an_empty_slot_says_how_to_fill_it():
    ctx, state, tab, sent = _session()
    assert plotter_mode.recall_stamp(ctx, state, tab, 5) is False
    assert sent and "Ctrl+Shift+5" in sent[0][0]


def test_two_tabs_do_not_share_slots():
    ctx, state, tab, _sent = _session()
    other = plotter_state.PlotterDoc(doc=MapDoc(4, 4, 8, 8), title="other")
    state.add(other)
    state.brush = np.array([[1]], dtype=np.uint32)
    plotter_mode.store_stamp(ctx, state, tab, 1)
    assert other.stamps == {}


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
