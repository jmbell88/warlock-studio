"""Per-tile metadata: the model, the edit, the weights and the playback.

Two rules decide most of this file. The store is **sparse** -- a tile whose
metadata is all defaults is a tile with no metadata, so an empty record is
dropped on the way in and a round trip cannot grow elements nobody authored. And
the animation substitution is **draw-time only**: a clock that wrote gids would
dirty a saved map sixty times a second, so the document's bytes must be
identical across animation frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter import tools
from warlock.studio.plotter.tilemap import MapDoc
from warlock.studio.tilegrid import gid as gidlib
from warlock.studio.tilegrid.tileset import (
    TileEllipse,
    TileFrame,
    TileMeta,
    TilePolygon,
    TileRect,
    Tileset,
)


def _pixels(size: int = 64) -> np.ndarray:
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 3] = 255
    return out


def _doc(tiles: dict | None = None) -> MapDoc:
    doc = MapDoc(4, 4, 16, 16)
    doc.add_tileset(
        Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16, tiles=tiles or {})
    )
    doc.add_tile_layer("Ground")
    return doc


# --- the model ----------------------------------------------------------------


def test_an_empty_record_is_dropped_on_construction() -> None:
    ts = Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16, tiles={0: TileMeta()})
    assert ts.tiles == {}


def test_meta_of_never_returns_none() -> None:
    """Every caller wants to read a field off it, and an ``is not None`` at each
    of them is one place for the check to be forgotten."""
    ts = Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16)
    assert ts.meta_of(3).is_empty


def test_with_meta_returns_a_copy() -> None:
    """A ``Tileset`` is frozen because the UI keys its texture upload on
    identity; an in-place edit would move no counter."""
    ts = Tileset(name="t", pixels=_pixels(), tile_w=16, tile_h=16)
    made = ts.with_meta(1, TileMeta(class_name="Water"))
    assert made is not ts
    assert ts.tiles == {}
    assert made.meta_of(1).class_name == "Water"


def test_with_meta_of_nothing_removes_the_entry() -> None:
    ts = Tileset(
        name="t", pixels=_pixels(), tile_w=16, tile_h=16,
        tiles={1: TileMeta(class_name="Water")},
    )
    assert ts.with_meta(1, None).tiles == {}
    assert ts.with_meta(1, TileMeta()).tiles == {}


def test_a_negative_probability_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        TileMeta(probability=-1.0)


def test_a_stray_object_is_not_a_collision_shape() -> None:
    with pytest.raises(ValueError, match="collision shape"):
        TileMeta(collision=(object(),))


def test_a_frame_duration_floors_at_one() -> None:
    """A zero-duration frame makes the playback arithmetic divide by nothing."""
    assert TileFrame(local_id=0, duration_ms=0).duration_ms == 1


# --- the edit -----------------------------------------------------------------


def test_setting_tile_metadata_is_one_undo_step() -> None:
    doc = _doc()
    assert doc.set_tile_meta(0, 2, TileMeta(class_name="Water")) is True
    assert doc.tilesets[0].tileset.meta_of(2).class_name == "Water"
    assert doc.undo() is True
    assert doc.tilesets[0].tileset.meta_of(2).is_empty


def test_setting_the_same_metadata_twice_pushes_nothing() -> None:
    doc = _doc()
    doc.set_tile_meta(0, 2, TileMeta(class_name="Water"))
    depth = len(doc.history)
    assert doc.set_tile_meta(0, 2, TileMeta(class_name="Water")) is False
    assert len(doc.history) == depth


def test_clearing_tile_metadata_removes_the_entry() -> None:
    doc = _doc({2: TileMeta(class_name="Water")})
    assert doc.set_tile_meta(0, 2, None) is True
    assert doc.tilesets[0].tileset.tiles == {}


def test_setting_metadata_moves_the_tileset_epoch() -> None:
    """The palette draws from the tileset object, so a class typed into the form
    would not appear until something else moved the counter."""
    doc = _doc()
    before = doc.tileset_epoch
    doc.set_tile_meta(0, 1, TileMeta(class_name="Water"))
    assert doc.tileset_epoch > before


def test_an_out_of_range_tileset_is_refused() -> None:
    with pytest.raises(IndexError):
        _doc().set_tile_meta(5, 0, TileMeta(class_name="x"))


# --- random weights -----------------------------------------------------------


def _weights(doc: MapDoc):
    from warlock.studio.panes import plotter_canvas

    return plotter_canvas._tile_weights(doc)


def test_no_metadata_means_no_weights_at_all() -> None:
    """Short-circuits the whole thing to the uniform pick it always was."""
    assert _weights(_doc()) is None


def test_probability_zero_is_never_chosen_at_random() -> None:
    doc = _doc({0: TileMeta(probability=0.0)})
    first = doc.tilesets[0].firstgid
    brush = np.array([[first, first + 1]], gidlib.DTYPE)
    rng = np.random.default_rng(1)
    landed = set()
    for _ in range(40):
        region = tools.random_stamp(
            doc.tile_layers()[0].data, 0, 0, brush, rng=rng, weights=_weights(doc)
        )
        landed.add(int(region[2][0, 0]))
    assert landed == {first + 1}, "the zero-weight tile never lands"


def test_a_weighted_brush_favours_the_heavier_tile() -> None:
    doc = _doc({0: TileMeta(probability=0.1), 1: TileMeta(probability=10.0)})
    first = doc.tilesets[0].firstgid
    brush = np.array([[first, first + 1]], gidlib.DTYPE)
    rng = np.random.default_rng(4)
    counts = {first: 0, first + 1: 0}
    for _ in range(200):
        region = tools.random_stamp(
            doc.tile_layers()[0].data, 0, 0, brush, rng=rng, weights=_weights(doc)
        )
        counts[int(region[2][0, 0])] += 1
    assert counts[first + 1] > counts[first] * 3


def test_a_brush_weighted_entirely_to_zero_falls_back_to_uniform() -> None:
    """Refusing to place anything would be a scatter brush that silently does
    nothing."""
    doc = _doc({0: TileMeta(probability=0.0), 1: TileMeta(probability=0.0)})
    first = doc.tilesets[0].firstgid
    brush = np.array([[first, first + 1]], gidlib.DTYPE)
    choices, probabilities = tools.random_choices(brush, _weights(doc))
    assert probabilities is None
    assert set(choices.tolist()) == {first, first + 1}


def test_a_weighted_area_fill_writes_only_brush_members() -> None:
    doc = _doc({0: TileMeta(probability=0.0)})
    first = doc.tilesets[0].firstgid
    brush = np.array([[first, first + 1]], gidlib.DTYPE)
    _x, _y, region = tools.fill_rect(
        doc.tile_layers()[0].data,
        0,
        0,
        3,
        3,
        first,
        brush=brush,
        rng=np.random.default_rng(2),
        weights=_weights(doc),
    )
    assert set(np.unique(region).tolist()) == {first + 1}


# --- animation playback -------------------------------------------------------


def _animated(doc: MapDoc, value: int, clock_ms: int) -> int:
    from warlock.studio.panes import plotter_canvas

    return plotter_canvas.animated_gid(doc, value, clock_ms)


def test_the_current_frame_is_substituted_by_the_clock() -> None:
    doc = _doc({0: TileMeta(animation=(TileFrame(0, 100), TileFrame(1, 100)))})
    first = doc.tilesets[0].firstgid
    assert _animated(doc, first, 0) == first
    assert _animated(doc, first, 150) == first + 1
    assert _animated(doc, first, 250) == first, "and it loops"


def test_a_cells_flip_flags_survive_the_substitution() -> None:
    """A mirrored animated tile animates mirrored, exactly as a mirrored still
    one draws mirrored."""
    doc = _doc({0: TileMeta(animation=(TileFrame(0, 100), TileFrame(1, 100)))})
    first = doc.tilesets[0].firstgid
    value = gidlib.compose(first, flip_h=True, flip_d=True)
    got = _animated(doc, value, 150)
    tile, h, v, d = gidlib.decompose(got)
    assert tile == first + 1
    assert (h, v, d) == (True, False, True)


def test_a_tile_with_no_animation_is_returned_verbatim() -> None:
    doc = _doc({0: TileMeta(class_name="Water")})
    first = doc.tilesets[0].firstgid
    assert _animated(doc, first, 999) == first


def test_an_empty_cell_is_returned_verbatim() -> None:
    assert _animated(_doc(), 0, 999) == 0


def test_the_document_bytes_are_identical_across_animation_frames() -> None:
    """The substitution is draw-only: a clock that wrote gids would dirty a
    saved map every frame."""
    doc = _doc({0: TileMeta(animation=(TileFrame(0, 100), TileFrame(1, 100)))})
    layer = doc.tile_layers()[0]
    first = doc.tilesets[0].firstgid
    doc.write_region(layer.uid, 0, 0, np.array([[first]], gidlib.DTYPE))
    before = np.array(doc.tile_layers()[0].data)
    for clock in (0, 150, 275, 1000):
        _animated(doc, first, clock)
    assert np.array_equal(doc.tile_layers()[0].data, before)


def test_the_flat_renderer_draws_frame_one() -> None:
    """An export is a still -- the parallax precedent, where the canvas and the
    export deliberately disagree and the disagreement is stated."""
    from warlock.studio.plotter import render

    doc = _doc({0: TileMeta(animation=(TileFrame(0, 100), TileFrame(1, 100)))})
    layer = doc.tile_layers()[0]
    first = doc.tilesets[0].firstgid
    doc.write_region(layer.uid, 0, 0, np.array([[first]], gidlib.DTYPE))
    # Two renders taken at different wall-clock moments are the same bytes,
    # because the renderer has no clock at all.
    assert np.array_equal(render.render_map(doc), render.render_map(doc))


# --- collision shapes ---------------------------------------------------------


def test_collision_shapes_are_tilegrids_own_records() -> None:
    """The shared leaf imports nothing under ``warlock``, so it cannot hold a
    plotter shape -- the codec converts instead."""
    meta = TileMeta(
        collision=(
            TileRect(1.0, 2.0, 3.0, 4.0),
            TileEllipse(0.0, 0.0, 2.0, 2.0),
            TilePolygon(1.0, 1.0, ((0.0, 0.0), (2.0, 0.0))),
        )
    )
    assert [type(s).__name__ for s in meta.collision] == [
        "TileRect",
        "TileEllipse",
        "TilePolygon",
    ]
    assert meta.collision[2].points == ((0.0, 0.0), (2.0, 0.0))
