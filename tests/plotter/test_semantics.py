"""The comparator's own properties, tested before anything gates on it.

``doc_facts`` is the substrate every later milestone compares documents
through, so the thing that must be true of it is not "it works on this map"
but that it is *blind to what it promises to be blind to* (uids, dict order,
float spelling) and *sensitive to everything else*. A comparator that quietly
ignored a field would turn every gate built on it into a test that passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter import gid, tsx
from warlock.studio.plotter.tilemap import MapDoc, MapObject, new_uid
from warlock.studio.plotter.tileset import Tileset

from ._semantics import doc_facts


def _pixels(w: int = 64, h: int = 64) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    array[0, 0] = (7, 8, 9, 255)
    return array


def _doc() -> MapDoc:
    doc = MapDoc(6, 4, 16, 16)
    doc.add_tileset(Tileset(name="terrain", pixels=_pixels(), tile_w=16, tile_h=16))
    tiles = doc.add_tile_layer("Ground")
    cells = np.zeros((4, 6), gid.DTYPE)
    cells[0, 0] = gid.compose(1)
    cells[1, 1] = gid.compose(2, flip_h=True)
    doc.write_region(tiles.uid, 0, 0, cells)
    doc.set_layer_props(tiles.uid, opacity=0.5)
    objects = doc.add_object_layer("Things")
    doc.add_object(
        objects.uid,
        MapObject(uid=new_uid(), name="spawn", kind="point", x=17.5, y=3.0,
                  properties={"team": tsx.Prop("int", 2)}),
    )
    doc.properties = {"theme": tsx.Prop("string", "cave")}
    doc.backgroundcolor = "#ff112233"
    return doc


def test_two_documents_built_the_same_way_agree():
    assert doc_facts(_doc()) == doc_facts(_doc())


def test_the_facts_are_blind_to_uids():
    """The point of the whole function: a uid is minted per process and means
    nothing across a save, so two readings of one file must compare equal."""
    first, second = _doc(), _doc()
    uids = {layer.uid for layer in first.layers} | {layer.uid for layer in second.layers}
    assert len(uids) == 4, "the two documents really do carry different uids"
    assert doc_facts(first) == doc_facts(second)


def test_the_facts_survive_json_and_so_can_be_diffed():
    import json

    assert json.loads(json.dumps(doc_facts(_doc()))) == doc_facts(_doc())


def test_a_float_written_two_ways_compares_equal():
    """0.1 + 0.2 is not 0.3, and a document that went through a text format
    must not fail the gate over the last bit of a coordinate."""
    left, right = _doc(), _doc()
    obj = right.layers[1].objects[0]
    obj.x = 17.5 + 1e-12
    assert doc_facts(left) == doc_facts(right)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda d: setattr(d, "projection", "isometric"), id="projection"),
        pytest.param(lambda d: setattr(d, "renderorder", "left-up"), id="renderorder"),
        pytest.param(lambda d: setattr(d, "backgroundcolor", "#ffffffff"), id="background"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, name="Other"), id="layer-name"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, opacity=0.25), id="opacity"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, visible=False), id="visible"),
        pytest.param(lambda d: d.set_layer_props(d.layers[0].uid, locked=True), id="locked"),
        pytest.param(
            lambda d: d.write_region(
                d.layers[0].uid, 0, 0, np.full((1, 1), gid.compose(9), gid.DTYPE)
            ),
            id="cells",
        ),
        pytest.param(
            lambda d: setattr(d.layers[1].objects[0], "name", "other"), id="object-name"
        ),
        pytest.param(lambda d: setattr(d.layers[1].objects[0], "x", 99.0), id="object-x"),
        pytest.param(
            lambda d: setattr(d.layers[1].objects[0], "obj_class", "Spawn"), id="object-class"
        ),
        pytest.param(
            lambda d: setattr(d, "properties", {"theme": tsx.Prop("string", "forest")}),
            id="map-properties",
        ),
        pytest.param(lambda d: setattr(d, "width", 7), id="map-width"),
        pytest.param(lambda d: setattr(d.layers[1].objects[0], "kind", "rect"), id="object-kind"),
        pytest.param(
            lambda d: setattr(d.layers[1].objects[0], "visible", False), id="object-visible"
        ),
        pytest.param(
            lambda d: d.set_layer_props(
                d.layers[0].uid, properties={"foo": tsx.Prop("string", "bar")}
            ),
            id="layer-properties",
        ),
    ],
)
def test_every_field_the_comparator_claims_to_cover_actually_moves_it(mutate):
    """One case per field. A comparator is only as good as its worst blind
    spot, and the blind spot is invisible until something that should have
    failed passes."""
    before = doc_facts(_doc())
    doc = _doc()
    mutate(doc)
    assert doc_facts(doc) != before


def test_a_tilesets_firstgid_moves_the_facts():
    """``firstgid`` is allocated by ``add_tileset`` from append order and the
    ref that carries it is frozen, so there is no mutator to reach for here --
    unlike the cases above, this is a standalone test rather than a
    ``pytest.param`` lambda, built with ``dataclasses.replace`` so only
    ``firstgid`` moves and the tileset's own content and the document's
    tileset order do not."""
    import dataclasses

    doc = _doc()
    before = doc_facts(doc)
    doc.tilesets[0] = dataclasses.replace(doc.tilesets[0], firstgid=99)
    assert doc_facts(doc) != before


def test_a_tilesets_terrain_order_moves_the_facts():
    """A terrain's position in the list is precedence, per :class:`TerrainSpec`'s
    own docstring in ``tileset.py`` -- when a cell sits between two terrains,
    the earlier one wins. Reordering two terrains without changing either one
    must therefore move the facts. ``Tileset`` is frozen, so this is built as
    two documents rather than one mutated in place."""
    from warlock.studio.plotter import blob
    from warlock.studio.plotter.tileset import TerrainSpec

    terrains = (
        TerrainSpec(name="grass", fill=(0, 255, 0, 255), outline=(0, 128, 0, 255)),
        TerrainSpec(name="sand", fill=(255, 255, 0, 255), outline=(128, 128, 0, 255)),
    )

    # A terrain set is a strict grid: one column per blob case, one row per
    # terrain -- see the shape check in ``Tileset.__post_init__`` -- so the
    # atlas has to be sized for it rather than reused from ``_pixels()``.
    def _with_terrains(order: tuple[TerrainSpec, ...]) -> MapDoc:
        doc = MapDoc(6, 4, 16, 16)
        pixels = _pixels(blob.TILE_COUNT * 16, len(order) * 16)
        doc.add_tileset(
            Tileset(name="terrain", pixels=pixels, tile_w=16, tile_h=16, terrains=order)
        )
        return doc

    assert doc_facts(_with_terrains(terrains)) != doc_facts(
        _with_terrains(tuple(reversed(terrains)))
    )
    # And a sanity check that the helper itself is not accidentally always
    # different for some unrelated reason (e.g. object identity leaking in).
    assert doc_facts(_with_terrains(terrains)) == doc_facts(_with_terrains(terrains))


def test_a_different_atlas_moves_the_facts():
    """Pixels are hashed rather than inlined, so this is the test that the
    hash is of the pixels and not of, say, the shape alone."""
    doc = _doc()
    other = _pixels()
    other[5, 5] = (1, 2, 3, 255)
    swapped = MapDoc(6, 4, 16, 16)
    swapped.add_tileset(Tileset(name="terrain", pixels=other, tile_w=16, tile_h=16))
    assert doc_facts(swapped)["tilesets"] != doc_facts(doc)["tilesets"]
