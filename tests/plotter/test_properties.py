"""Plotter's Properties pane: whose fields it is showing, and the drafts it keeps.

The pane moved to the far side of the window from the layer list in wave A and
became a name/value table in wave C. Both changes turn on things that can be
asserted without a frame: what the subject line says, and where a half-typed
property name lives while it is being typed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio.panes import plotter_layers
from warlock.studio.plotter.props import Prop
from warlock.studio.plotter.tilemap import MapDoc


def _map():
    doc = MapDoc(8, 6, 16, 16)
    layer = doc.add_tile_layer()
    doc.set_active_layer(layer.uid)
    return doc, layer


def _state(**kwargs):
    base = {"selected_object": None, "selected_objects": set()}
    base.update(kwargs)
    state = SimpleNamespace(**base)
    state.select_objects = lambda values: setattr(state, "selected_objects", set(values))
    state.select_object = lambda uid: None
    return state


def _ctx():
    return SimpleNamespace(state=SimpleNamespace(preview={}))


# --- whose fields are these --------------------------------------------------


def test_the_subject_names_the_layer_when_a_layer_is_showing():
    """Until wave A this pane sat directly under the list it was about, where
    "the thing above" was answer enough. On the other side of the map it is
    not: a user reading a Name field has to be told whose name it is."""
    doc, layer = _map()
    doc.set_layer_props(layer.uid, name="Ground")

    assert plotter_layers._subject(doc, _state(), layer) == "Layer: Ground"


def test_an_unnamed_layer_still_reads_as_a_layer():
    doc, layer = _map()
    doc.set_layer_props(layer.uid, name="")
    assert plotter_layers._subject(doc, _state(), layer) == "Layer: Untitled"


def test_the_subject_names_the_object_and_its_id():
    """Two objects may share a name and the map addresses them by number, so
    the id is what makes the line unambiguous."""
    doc, _layer = _map()
    objects = doc.add_object_layer()
    obj = plotter_layers.add_object(doc, objects, "rect", 0.0, 0.0, 16.0, 16.0)
    doc.set_object(objects.uid, obj.uid, name="door_1")
    state = _state(selected_object=obj.uid, selected_objects={obj.uid})

    line = plotter_layers._subject(doc, state, objects)

    assert line == f"Object: door_1 (#{obj.uid})"


def test_an_unnamed_object_is_still_identified_by_its_id():
    """``add_object`` gives every object a default name, so this is the case a
    user reaches by clearing the Name field rather than one the door produces.
    The id is what keeps the line pointing at something."""
    doc, _layer = _map()
    objects = doc.add_object_layer()
    obj = plotter_layers.add_object(doc, objects, "point", 4.0, 4.0, 0.0, 0.0)
    doc.set_object(objects.uid, obj.uid, name="")
    state = _state(selected_object=obj.uid, selected_objects={obj.uid})

    assert plotter_layers._subject(doc, state, objects) == f"Object: Object (#{obj.uid})"


def test_a_multi_selection_reads_as_a_count():
    """The pane is a summary rather than a bulk editor there, and the line says
    so before the form does."""
    doc, layer = _map()
    state = _state(selected_objects={1, 2, 3})

    assert plotter_layers._subject(doc, state, layer) == "3 objects"


# --- the drafts the table keeps ----------------------------------------------


def test_the_new_key_draft_is_scoped_to_one_editor():
    """A layer's editor and the map's are drawn in one frame, and a name typed
    into one must not appear in the other."""
    ctx = _ctx()
    first = plotter_layers._prop_form(ctx, "plotter_layer_prop:1")
    second = plotter_layers._prop_form(ctx, "plotter_map_prop:1")

    first["name"] = "theme"

    assert second["name"] == ""
    assert plotter_layers._prop_form(ctx, "plotter_layer_prop:1")["name"] == "theme"


def test_the_draft_survives_being_reopened():
    ctx = _ctx()
    plotter_layers._prop_form(ctx, "k")["name"] = "half typed"
    assert plotter_layers._prop_form(ctx, "k")["name"] == "half typed"


def test_an_older_two_key_draft_is_filled_in_rather_than_replaced():
    """The dict parked here before wave C carried only a name and a type. A
    session upgrading mid-edit must not lose the name it was holding."""
    ctx = _ctx()
    ctx.state.preview["k"] = {"name": "cost", "type": "int"}

    form = plotter_layers._prop_form(ctx, "k")

    assert form["name"] == "cost"
    assert form["selected"] == ""
    assert form["folded"] == set()


def test_a_new_draft_starts_with_every_key_the_editor_reads():
    form = plotter_layers._prop_form(_ctx(), "k")
    assert set(form) == {"name", "type", "selected", "folded"}


# --- the tables the forms are built from -------------------------------------


def test_the_text_flags_name_real_fields_of_the_text_shape():
    """Six toggles that were six rows are one row of six checkboxes. A flag
    named here that the shape does not carry would be a box that writes an
    attribute nothing reads."""
    from warlock.studio.plotter.tilemap import Text

    shape = Text(text="hello")
    for key, label in plotter_layers.TEXT_FLAGS:
        assert hasattr(shape, key), key
        assert isinstance(getattr(shape, key), bool), key
        assert label


def test_every_addable_property_type_has_a_blank_value():
    for kind in plotter_layers.AUTHORABLE_TYPES:
        blank = plotter_layers._blank_value(kind)
        assert blank is not None or kind == "string"


@pytest.mark.parametrize("kind", ["class", "list"])
def test_a_container_property_summarises_rather_than_showing_nothing(kind):
    """A class arriving from Tiled used to look like an empty string until the
    recursive editor landed; the count is what says it carries something."""
    prop = Prop(type=kind, value={} if kind == "class" else [])
    assert plotter_layers._summary(prop)
