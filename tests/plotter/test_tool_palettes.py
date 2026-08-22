"""The toolbox belongs to the layer you are standing on.

Tiled's split, and the property that makes it safe rather than merely tidy:
``sync_tool`` is idempotent and is called at the top of *both* doors -- the
canvas's draw and the key handler -- so no frame can act with a tool the active
layer cannot host. A layer-driven palette that only synced on the switch goes
wrong on the frame *after* the switch, which is the frame the user clicks in.
"""

from __future__ import annotations

import pytest

from warlock.studio import plotter_state
from warlock.studio.plotter.tilemap import MapDoc


def _doc() -> MapDoc:
    return MapDoc(8, 8, 16, 16)


def test_the_two_palettes_are_disjoint_where_it_matters():
    tile = {key for key, _l, _k in plotter_state.TILE_TOOLS}
    objects = {key for key, _l, _k in plotter_state.OBJECT_TOOLS}
    # One tool is in both, deliberately: the object *pointer*, which is how you
    # reach an object layer from a tile one.
    assert tile & objects == {"object"}


def test_six_letters_mean_two_things():
    doc = _doc()
    tile_layer = doc.add_tile_layer()
    object_layer = doc.add_object_layer()
    tile_keys = plotter_state.tool_keys(tile_layer)
    object_keys = plotter_state.tool_keys(object_layer)
    shared = set(tile_keys) & set(object_keys)
    assert len(shared) >= 6
    # R is the rectangular select on a tile layer and insert-rectangle on an
    # object one -- Tiled's arrangement exactly.
    assert tile_keys["r"] == "select"
    assert object_keys["r"] == "object_rect"


def test_a_group_or_an_image_layer_hosts_no_gesture():
    doc = _doc()
    assert plotter_state.tools_for(doc.add_group_layer()) == ()
    assert plotter_state.tools_for(doc.add_image_layer()) == ()
    assert plotter_state.tools_for(None) == ()
    assert plotter_state.tool_keys(None) == {}


def test_sync_puts_a_legal_tool_in_hand():
    doc = _doc()
    tile_layer = doc.add_tile_layer()
    object_layer = doc.add_object_layer()
    state = plotter_state.PlotterState()
    state.tool = "fill"
    assert plotter_state.sync_tool(state, object_layer) != "fill"
    assert state.tool in {key for key, _l, _k in plotter_state.OBJECT_TOOLS}
    # The tile side is remembered rather than lost...
    assert state.remembered["tile"] == "fill"
    # ...and switching back with an *illegal* tool in hand restores it. (The
    # object pointer is legal on both, so holding that one keeps it -- which is
    # the point of the shared tool.)
    state.tool = "object_ellipse"
    assert plotter_state.sync_tool(state, tile_layer) == "fill"


def test_sync_is_idempotent():
    doc = _doc()
    layer = doc.add_object_layer()
    state = plotter_state.PlotterState()
    state.tool = "fill"
    once = plotter_state.sync_tool(state, layer)
    assert plotter_state.sync_tool(state, layer) == once


def test_each_kind_remembers_its_own_tool():
    doc = _doc()
    tile_layer = doc.add_tile_layer()
    object_layer = doc.add_object_layer()
    state = plotter_state.PlotterState()
    state.tool = "terrain"
    plotter_state.sync_tool(state, tile_layer)
    plotter_state.sync_tool(state, object_layer)
    state.tool = "object_ellipse"
    plotter_state.sync_tool(state, object_layer)
    assert plotter_state.sync_tool(state, tile_layer) == "terrain"
    assert plotter_state.sync_tool(state, object_layer) == "object_ellipse"


def test_a_group_layer_leaves_the_tool_alone():
    """Nothing is drawn on it, so there is nothing to choose for it -- and
    changing the tool on the way past would lose what the user had in hand."""

    doc = _doc()
    state = plotter_state.PlotterState()
    state.tool = "terrain"
    assert plotter_state.sync_tool(state, doc.add_group_layer()) == "terrain"


@pytest.mark.parametrize(
    ("tool", "shape"), sorted(plotter_state.OBJECT_SHAPES.items())
)
def test_an_insert_tool_writes_the_shape_it_names(tool, shape):
    """The Insert combo that used to sit in the layers pane *was* this table
    with one control in front of it."""

    doc = _doc()
    layer = doc.add_object_layer()
    state = plotter_state.PlotterState()
    state.tool = tool
    plotter_state.sync_tool(state, layer)
    assert state.object_shape == shape


def test_insert_template_is_not_offered():
    """``docs/COMPAT.md`` states object templates are a non-goal, and offering
    the tool would be the offer-then-refuse shape this codebase rejects."""

    labels = {label for _k, label, _l in plotter_state.OBJECT_TOOLS}
    assert not any("template" in label.lower() for label in labels)
