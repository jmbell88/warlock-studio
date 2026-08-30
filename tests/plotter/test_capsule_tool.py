"""The capsule had a shape, a hit test, a renderer and four codecs -- and no way
in.

Every layer under the toolbox was finished: ``add_object`` has accepted
``"capsule"`` since the geometry landed, the canvas draws its outline, the hit
test has an arm for it, and ``.wmap``, ``.tmx`` and ``.tmj`` all carry it in both
directions. What was missing was one row in ``OBJECT_TOOLS``, so the only way to
author the shape was to hand-edit a file -- in an editor whose own interop
ledger lists the capsule as a thing *this* editor has and Tiled does not.

These tests walk the whole path the missing row opened: the letter reaches the
tool, the tool reaches ``state.object_shape``, the shape reaches ``add_object``,
and the object it makes survives a round trip through the format that has to
carry it.
"""

from __future__ import annotations

from warlock.studio import plotter_state
from warlock.studio.panes import plotter_layers
from warlock.studio.plotter import tmx
from warlock.studio.plotter._map_model import Capsule, shape_kind
from warlock.studio.plotter.tilemap import MapDoc


def _doc() -> MapDoc:
    return MapDoc(8, 8, 16, 16)


def test_the_object_toolbox_offers_a_capsule():
    keys = {key for key, _label, _letter in plotter_state.OBJECT_TOOLS}
    assert "object_capsule" in keys
    # Not on the tile side: a capsule is an object, and a letter that meant two
    # unrelated things would be the one divergence from Tiled's table that is
    # not Tiled's.
    assert "object_capsule" not in {key for key, _l, _k in plotter_state.TILE_TOOLS}


def test_the_letter_reaches_the_tool_and_the_tool_reaches_the_shape():
    doc = _doc()
    layer = doc.add_object_layer()
    assert plotter_state.tool_keys(layer)["c"] == "object_capsule"
    state = plotter_state.PlotterState()
    state.tool = "object_capsule"
    plotter_state.sync_tool(state, layer)
    # ``sync_tool`` is what both the canvas and the key handler call, so this is
    # the value the release will read.
    assert state.object_shape == "capsule"


def test_the_letter_is_free_on_both_palettes():
    """A letter that already meant something would have taken it away.

    Checked rather than assumed, because the four-letter move that gave the tile
    tools Tiled's spellings is exactly the kind of change that quietly reuses
    one.
    """
    doc = _doc()
    tile_keys = plotter_state.tool_keys(doc.add_tile_layer())
    assert "c" not in tile_keys


def test_the_toolbox_shape_makes_a_capsule_object():
    """The door the canvas calls on release, with the string the tool writes."""
    doc = _doc()
    layer = doc.add_object_layer()
    obj = plotter_layers.add_object(
        doc, layer, plotter_state.OBJECT_SHAPES["object_capsule"], 32.0, 48.0, 24.0, 40.0
    )
    assert isinstance(obj.shape, Capsule)
    assert shape_kind(obj.shape) == "capsule"
    assert (obj.x, obj.y, obj.w, obj.h) == (32.0, 48.0, 24.0, 40.0)
    assert [entry.uid for entry in doc.layers[-1].objects] == [obj.uid]


def test_an_authored_capsule_survives_the_format_that_carries_it():
    """The dialect claim in ``docs/COMPAT.md``, made about a capsule the *UI*
    authored rather than one a fixture was hand-written to hold."""
    doc = _doc()
    doc.add_tile_layer()
    layer = doc.add_object_layer()
    plotter_layers.add_object(doc, layer, "capsule", 16.0, 16.0, 24.0, 40.0)

    files = tmx.tmx_export(doc)

    def _missing(source: str):  # pragma: no cover - the map has no tileset
        raise AssertionError(f"no tileset was exported, so {source!r} is unexpected")

    back = tmx.read_tmx(
        files["map.tmx"], image_loader=_missing, tsx_loader=_missing
    )
    objects = [obj for target in back.layers for obj in getattr(target, "objects", [])]
    assert [shape_kind(obj.shape) for obj in objects] == ["capsule"]
    assert (objects[0].w, objects[0].h) == (24.0, 40.0)
