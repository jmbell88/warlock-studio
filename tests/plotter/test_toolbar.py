"""What Plotter's toolbar offers, per layer and per tool.

``bar_items`` is pure -- no imgui, no ctx, no frame -- so every combination the
row can be in is a plain assertion rather than a rendered window. That is the
whole reason it exists as a function: the old tool grid decided the same things
inline inside a draw call, where the only way to ask "does an object layer offer
Random" was to rasterise one and read the labels back.

The tier arithmetic itself belongs to ``toolbar.plan`` and is tested there; what
is pinned here is which items and fields are handed to it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import plotter_state
from warlock.studio.panes import plotter_tools

# ``plotter_state.layer_kind`` dispatches on the class *name*, which is what
# lets this file stay free of a real document: three empty types named after the
# three the tilemap defines say everything the function reads.
TileLayer = type("TileLayer", (), {})
ObjectLayer = type("ObjectLayer", (), {})
GroupLayer = type("GroupLayer", (), {})


def _state(**kwargs):
    base = {"tool": "stamp", "brush": None, "random_mode": False}
    base.update(kwargs)
    return SimpleNamespace(**base)


def _keys(items):
    return [item.key for item in items]


# --- which palette the layer hosts -------------------------------------------


def test_a_tile_layer_gets_the_transforms_and_random():
    items, fields = plotter_tools.bar_items(_state(), TileLayer())
    assert _keys(items) == ["flip_h", "flip_v", "rotate_cw", "rotate_ccw", "random"]
    assert fields == []


def test_an_object_layer_gets_no_brush_row_at_all():
    """Greyed would be worse than absent here: there is no brush on an object
    layer and never can be, so five permanently dead buttons would be five
    buttons' width of the row claimed to say nothing."""
    items, fields = plotter_tools.bar_items(_state(tool="object"), ObjectLayer())
    assert items == []
    assert fields == []


def test_a_group_layer_gets_nothing_either():
    items, fields = plotter_tools.bar_items(_state(), GroupLayer())
    assert (items, fields) == ([], [])


def test_nothing_open_is_the_same_empty_row():
    items, fields = plotter_tools.bar_items(_state(), None)
    assert (items, fields) == ([], [])


# --- the transforms need a brush ---------------------------------------------


def test_the_transforms_are_refused_with_a_reason_until_a_tile_is_picked():
    items, _fields = plotter_tools.bar_items(_state(), TileLayer())
    transforms = [item for item in items if item.key != "random"]
    assert transforms and all(not item.enabled for item in transforms)
    assert all(item.reason == plotter_tools.NO_BRUSH for item in transforms)
    # The reason is a sentence a user can act on, not a restatement.
    assert "tileset" in plotter_tools.NO_BRUSH


def test_a_brush_in_hand_arms_all_four():
    items, _fields = plotter_tools.bar_items(_state(brush=object()), TileLayer())
    transforms = [item for item in items if item.key != "random"]
    assert len(transforms) == 4
    assert all(item.enabled for item in transforms)


def test_random_is_never_refused_and_shows_its_state():
    """It is a mode rather than an action on the brush, so it is legal with an
    empty hand -- and ``selected`` is the only thing on screen that says it is
    on, since the button carries a glyph rather than a label at the icon tier."""
    off, _ = plotter_tools.bar_items(_state(), TileLayer())
    on, _ = plotter_tools.bar_items(_state(random_mode=True), TileLayer())
    assert next(i for i in off if i.key == "random").enabled
    assert not next(i for i in off if i.key == "random").selected
    assert next(i for i in on if i.key == "random").selected


# --- the tool's own field ----------------------------------------------------


def test_shape_in_hand_adds_the_shape_field_and_nothing_else_does():
    _items, fields = plotter_tools.bar_items(_state(tool="shape"), TileLayer())
    assert fields == ["shape"]
    for tool in ("stamp", "erase", "fill", "select", "wand", "pick"):
        _items, other = plotter_tools.bar_items(_state(tool=tool), TileLayer())
        assert other == [], tool


def test_terrain_in_hand_adds_the_terrain_field():
    _items, fields = plotter_tools.bar_items(_state(tool="terrain"), TileLayer())
    assert fields == ["terrain"]


def test_the_tool_field_outranks_the_transforms_when_the_row_is_short():
    """Priority is what ``toolbar.plan`` collapses by, and the order is a claim
    about what the row is *for*: the setting of the tool in your hand survives
    a narrow window, Random goes first, the transforms next."""
    items, _fields = plotter_tools.bar_items(_state(tool="shape"), TileLayer())
    by_key = {item.key: item.priority for item in items}
    assert by_key["random"] > by_key["flip_h"] > 0


# --- the tables the row is built from ----------------------------------------


def test_every_transform_names_a_real_plotter_mode_transform():
    """The bar and the keyboard press one door. A row naming a transform the
    mode does not have would be a button that silently did nothing."""
    from warlock.studio import plotter_mode

    for _key, _label, _glyph, name, _back, _tip in plotter_tools.BRUSH_TRANSFORMS:
        assert name in plotter_mode._BRUSH_TRANSFORMS


def test_every_transform_says_what_it_does_and_which_key_does_it():
    for _key, label, _glyph, _name, _back, tip in plotter_tools.BRUSH_TRANSFORMS:
        assert label and tip.endswith(")"), label
        # The letter is in the tooltip because at the icon tier the label is
        # gone and the tooltip is all there is.
        assert "(" in tip


def test_the_two_transforms_with_no_glyph_are_the_two_lucide_lacks():
    """``icons.py`` forbids guessing a codepoint and the vendored subset has
    ``flip-horizontal-2`` and ``rotate-cw`` and neither mirror. Those two draw
    as words at every tier, which ``toolbar._measure`` supports by handing a
    glyphless item the same width twice."""
    wordless = {
        key for key, _l, glyph, _n, _b, _t in plotter_tools.BRUSH_TRANSFORMS if not glyph
    }
    assert wordless == {"flip_v", "rotate_ccw"}


def test_the_view_table_names_only_attributes_the_state_really_has():
    """``view_rows`` reads and writes by attribute name, so a typo here would
    be a menu row that toggled a field nothing draws."""
    state = plotter_state.PlotterState()
    for key, label, _chord in plotter_tools.VIEW_TOGGLES:
        assert hasattr(state, key), key
        assert isinstance(getattr(state, key), bool), key
        assert label and label[0].isupper()


def test_the_view_table_carries_the_chords_that_exist_and_no_others():
    chords = {key: chord for key, _label, chord in plotter_tools.VIEW_TOGGLES}
    assert chords["grid"] == "Ctrl+G"
    assert chords["rulers"] == "Ctrl+R"
    assert chords["highlight"] == "H"
    # Two have no keyboard route, and say so with an empty string rather than
    # inventing one.
    assert chords["show_objects"] == "" and chords["minimap"] == ""


@pytest.mark.parametrize("key", [key for key, _l, _c in plotter_tools.VIEW_TOGGLES])
def test_every_view_toggle_is_a_distinct_row(key):
    keys = [row[0] for row in plotter_tools.VIEW_TOGGLES]
    assert keys.count(key) == 1


def test_the_snap_pills_are_exactly_the_modes_the_engine_knows():
    assert [key for key, _label in plotter_tools.SNAP_LABELS] == list(
        plotter_state.SNAP_MODES
    )
    for key, _label in plotter_tools.SNAP_LABELS:
        # Every pill says what Ctrl does, because a modifier nobody documents
        # is a modifier nobody finds.
        assert "Ctrl" in plotter_tools.SNAP_TIPS[key]
