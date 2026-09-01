"""The Objects dock, and the two ways of getting to an object by name or by id.

Objects were rows inside the layer list until 2026-09-01. What replaced them is
a list of its own, and the parts worth asserting are the ones a rendered frame
would not settle: which objects the list holds and in what order, what the search
box matches, and whether a jump lands on the object or somewhere near it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import plotter_mode, plotter_state
from warlock.studio.panes import plotter_layers, plotter_objects
from warlock.studio.plotter import layer_rows
from warlock.studio.plotter.tilemap import MapDoc


class _Settings:
    """Enough of the settings store for ``recents.remember`` to write into."""

    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(plotter=None, preview={}, list_filters={})
        self.settings = _Settings()
        self.toasts: list[tuple[str, str]] = []

    def toast(self, message: str, level: str = "info") -> None:
        self.toasts.append((message, level))


def _map():
    doc = MapDoc(16, 12, 16, 16)
    doc.add_tile_layer("Ground")
    return doc


def _tab(ctx, doc):
    return plotter_mode.adopt(ctx, doc, title="Level")


def _place(doc, layer, name, *, x=0.0, y=0.0, kind="rect", obj_class=""):
    obj = plotter_layers.add_object(doc, layer, kind, x, y, 16.0, 16.0)
    doc.set_object(layer.uid, obj.uid, name=name, obj_class=obj_class)
    return doc.layer(layer.uid).objects[-1]


# --- find_object -------------------------------------------------------------


def test_find_object_translates_a_tiled_id_into_the_editors_handles():
    """The one crossing between Tiled's persistent numbering and the uids every
    handle in this package is."""
    doc = _map()
    layer = doc.add_object_layer()
    obj = _place(doc, layer, "door_1")

    assert doc.find_object(obj.id) == (layer.uid, obj.uid)


def test_find_object_answers_nothing_for_zero():
    """Zero is Tiled's spelling of an unset reference, so it must not be looked
    for -- or an object that somehow carried id 0 would become what every unset
    property points at."""
    doc = _map()
    layer = doc.add_object_layer()
    _place(doc, layer, "door_1")

    assert doc.find_object(0) is None


def test_find_object_answers_nothing_for_a_dangling_reference():
    """The ordinary case rather than a caller's mistake: ids are monotone and
    never reused, so a property goes on naming its object after the object is
    deleted. A reader that raised would turn that into a crash on the frame
    that drew it."""
    doc = _map()
    layer = doc.add_object_layer()
    obj = _place(doc, layer, "door_1")
    gone = obj.id
    doc.remove_object(layer.uid, obj.uid)

    assert doc.find_object(gone) is None


def test_find_object_reaches_inside_a_group():
    """A group is not a place objects stop existing, and on any map organised
    into folders that is where they are."""
    doc = _map()
    group = doc.add_group_layer()
    layer = doc.add_object_layer()
    doc.move_layer(layer.uid, 0, parent_uid=group.uid)
    obj = _place(doc, doc.layer(layer.uid), "in_a_folder")

    assert doc.find_object(obj.id) == (layer.uid, obj.uid)


# --- what the dock lists -----------------------------------------------------


def test_the_dock_groups_by_layer_and_reads_top_first():
    """The layer stack reads top-first, and a dock ordering its groups the
    other way round would put the same two layers in opposite orders on one
    screen."""
    doc = _map()
    lower = doc.add_object_layer()
    doc.set_layer_props(lower.uid, name="Spawns")
    upper = doc.add_object_layer()
    doc.set_layer_props(upper.uid, name="Triggers")
    _place(doc, doc.layer(lower.uid), "player")
    _place(doc, doc.layer(upper.uid), "door_1")

    groups = plotter_objects.rows(doc)

    assert [layer.name for layer, _objects in groups] == ["Triggers", "Spawns"]


def test_objects_read_down_the_page_the_way_they_stack_on_screen():
    """The last object in the list is the one drawn on top, so the dock lists
    it first -- the canvas's own order."""
    doc = _map()
    layer = doc.add_object_layer()
    _place(doc, doc.layer(layer.uid), "under")
    _place(doc, doc.layer(layer.uid), "over")

    (_layer, objects), = plotter_objects.rows(doc)

    assert [obj.name for obj in objects] == ["over", "under"]


def test_a_tile_only_map_lists_no_groups():
    assert plotter_objects.rows(_map()) == []


def test_the_dock_reaches_objects_inside_a_group():
    doc = _map()
    group = doc.add_group_layer()
    layer = doc.add_object_layer()
    doc.move_layer(layer.uid, 0, parent_uid=group.uid)
    _place(doc, doc.layer(layer.uid), "hidden_away")

    groups = plotter_objects.rows(doc)

    assert [obj.name for _layer, objects in groups for obj in objects] == [
        "hidden_away"
    ]


# --- when the pane is there at all -------------------------------------------


def test_the_pane_appears_only_once_there_is_somewhere_for_an_object_to_be():
    """On a tile-only map it would be a heading over nothing."""
    ctx = _Ctx()
    doc = _map()
    _tab(ctx, doc)

    assert plotter_objects.has_object_layer(ctx) is False
    doc.add_object_layer()
    assert plotter_objects.has_object_layer(ctx) is True


def test_the_pane_is_absent_with_nothing_open():
    ctx = _Ctx()
    assert plotter_objects.has_object_layer(ctx) is False


# --- the search box ----------------------------------------------------------


def test_the_search_matches_the_name_the_class_the_kind_and_the_id():
    doc = _map()
    layer = doc.add_object_layer()
    obj = _place(doc, doc.layer(layer.uid), "door_1", obj_class="trigger")

    assert plotter_objects.matches(obj, "door")
    assert plotter_objects.matches(obj, "trigger")
    assert plotter_objects.matches(obj, "rect")
    # The id, deliberately: an object-typed property in another object holds
    # one, so pasting it in is how you ask what a reference points at.
    assert plotter_objects.matches(obj, str(obj.id))
    assert not plotter_objects.matches(obj, "nothing like this")


def test_an_empty_search_matches_everything():
    doc = _map()
    layer = doc.add_object_layer()
    obj = _place(doc, doc.layer(layer.uid), "door_1")
    assert plotter_objects.matches(obj, "")


# --- going to one ------------------------------------------------------------


def test_go_to_object_selects_activates_and_asks_for_a_centre():
    """All three, because "go to" means all three: centring without selecting
    leaves the Properties pane showing something else, and selecting without
    activating puts a selection on a layer no tool can reach."""
    ctx = _Ctx()
    doc = _map()
    tab = _tab(ctx, doc)
    layer = doc.add_object_layer()
    obj = _place(doc, doc.layer(layer.uid), "door_1", x=128.0, y=64.0)
    state = plotter_mode.ensure(ctx)

    assert plotter_mode.go_to_object(ctx, tab, layer.uid, obj.uid)

    assert doc.active_layer == layer.uid
    assert state.selected_object == obj.uid
    assert state.centre_on == (128.0, 64.0)


def test_the_centre_is_pixels_rather_than_a_rounded_cell():
    """An object sits wherever it was placed, routinely off the grid on
    purpose. Rounding to a cell to reuse ``goto_cell`` would centre on
    somewhere it is not."""
    ctx = _Ctx()
    doc = _map()
    tab = _tab(ctx, doc)
    layer = doc.add_object_layer()
    obj = _place(doc, doc.layer(layer.uid), "spawn", x=7.5, y=3.25)

    plotter_mode.go_to_object(ctx, tab, layer.uid, obj.uid)

    assert plotter_mode.ensure(ctx).centre_on == (7.5, 3.25)


def test_go_to_object_refuses_an_object_that_is_not_there():
    ctx = _Ctx()
    doc = _map()
    tab = _tab(ctx, doc)
    layer = doc.add_object_layer()

    assert not plotter_mode.go_to_object(ctx, tab, layer.uid, 999999)
    assert not plotter_mode.go_to_object(ctx, tab, 999999, 1)


def test_go_to_object_id_takes_tileds_number():
    ctx = _Ctx()
    doc = _map()
    tab = _tab(ctx, doc)
    layer = doc.add_object_layer()
    obj = _place(doc, doc.layer(layer.uid), "door_1", x=32.0, y=16.0)

    assert plotter_mode.go_to_object_id(ctx, tab, obj.id)
    assert plotter_mode.ensure(ctx).selected_object == obj.uid


def test_go_to_object_id_refuses_a_dangling_reference_rather_than_toasting():
    """The caller is a small arrow beside a field; it greys out."""
    ctx = _Ctx()
    doc = _map()
    tab = _tab(ctx, doc)
    layer = doc.add_object_layer()
    obj = _place(doc, doc.layer(layer.uid), "door_1")
    doc.remove_object(layer.uid, obj.uid)

    assert not plotter_mode.go_to_object_id(ctx, tab, obj.id)
    assert ctx.toasts == []


def test_a_centre_request_is_dropped_when_the_document_is_left():
    """It names a pixel of the document being left, ``last_paint``'s reason."""
    ctx = _Ctx()
    first = _tab(ctx, _map())
    state = plotter_mode.ensure(ctx)
    state.centre_on = (10.0, 10.0)
    second = _tab(ctx, _map())
    state.activate(first.uid)

    assert state.centre_on is None
    assert second is not None


# --- the layer list is a list of layers now ----------------------------------


def test_an_object_layer_no_longer_folds_in_the_layer_stack():
    """One row shape is kept -- an object layer is a row with nothing under it,
    like a tile layer, which is what it is from the stack's point of view."""
    doc = _map()
    layer = doc.add_object_layer()
    _place(doc, doc.layer(layer.uid), "door_1")

    assert layer_rows.can_fold(doc.layer(layer.uid)) is False


def test_a_group_still_folds():
    doc = _map()
    group = doc.add_group_layer()
    assert layer_rows.can_fold(group) is False, "an empty group folds nothing"
    doc.move_layer(doc.tile_layers()[0].uid, 0, parent_uid=group.uid)
    assert layer_rows.can_fold(doc.layer(group.uid)) is True


@pytest.mark.parametrize("attribute", ["centre_on", "renaming_layer"])
def test_the_new_view_state_is_dropped_with_the_document(attribute):
    state = plotter_state.PlotterState()
    setattr(state, attribute, (1.0, 2.0) if attribute == "centre_on" else 7)
    state._forget_document_state()
    assert getattr(state, attribute) in (None, 0)
