"""Slicing an atlas, and refusing one that does not slice.

The geometry is checked at construction rather than at the first draw, so these
are mostly refusals: a tileset that exists, appears in the list and renders
garbage is the outcome the validation is there to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter.tileset import Tileset, TilesetRef


def _pixels(w: int, h: int) -> np.ndarray:
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., 3] = 255
    return array


def test_a_plain_grid_slices_row_major():
    ts = Tileset(name="t", pixels=_pixels(64, 32), tile_w=16, tile_h=16)
    assert (ts.columns, ts.rows, ts.tile_count) == (4, 2, 8)
    assert ts.tile_rect(0) == (0, 0, 16, 16)
    assert ts.tile_rect(3) == (48, 0, 16, 16)
    assert ts.tile_rect(4) == (0, 16, 16, 16)  # local ids wrap row-major


def test_margin_and_spacing_follow_tileds_formula():
    """Margin is taken off both edges; each tile after the first costs its own
    width plus one spacing."""
    # margin 2 + 3 tiles of 10 + 2 gaps of 4 + margin 2 == 42
    ts = Tileset(name="t", pixels=_pixels(42, 42), tile_w=10, tile_h=10, spacing=4, margin=2)
    assert (ts.columns, ts.rows) == (3, 3)
    assert ts.tile_rect(0) == (2, 2, 10, 10)
    assert ts.tile_rect(1) == (16, 2, 10, 10)


def test_a_partial_final_column_is_not_a_tile():
    """70 pixels holds four 16px tiles and six pixels of nothing. Counting the
    remainder as a fifth tile is how a palette grows a column of garbage."""
    ts = Tileset(name="t", pixels=_pixels(70, 16), tile_w=16, tile_h=16)
    assert ts.columns == 4


def test_a_geometry_that_fits_nothing_is_refused_at_construction():
    with pytest.raises(ValueError, match="holds no"):
        Tileset(name="t", pixels=_pixels(16, 16), tile_w=32, tile_h=32)
    with pytest.raises(ValueError, match="holds no"):
        Tileset(name="t", pixels=_pixels(16, 16), tile_w=16, tile_h=16, margin=8)


def test_nonsense_geometry_is_refused():
    with pytest.raises(ValueError):
        Tileset(name="t", pixels=_pixels(32, 32), tile_w=0, tile_h=16)
    with pytest.raises(ValueError):
        Tileset(name="t", pixels=_pixels(32, 32), tile_w=16, tile_h=16, spacing=-1)


def test_a_non_rgba_image_is_refused():
    with pytest.raises(ValueError, match="RGBA"):
        Tileset(name="t", pixels=np.zeros((16, 16, 3), np.uint8), tile_w=16, tile_h=16)


def test_the_pixels_are_a_private_read_only_copy():
    """The UI invalidates its GPU upload only when the document's tileset list
    changes (``tileset_epoch``). An in-place edit anywhere would move no
    counter and leave the cache holding a live key over stale pixels."""
    source = _pixels(32, 32)
    ts = Tileset(name="t", pixels=source, tile_w=16, tile_h=16)
    assert ts.pixels is not source
    assert not ts.pixels.flags.writeable
    source[0, 0] = (1, 2, 3, 4)
    assert tuple(ts.pixels[0, 0]) == (0, 0, 0, 255)
    with pytest.raises(ValueError):
        ts.pixels[0, 0] = 1


def test_a_tile_out_of_range_raises_rather_than_wrapping():
    ts = Tileset(name="t", pixels=_pixels(32, 32), tile_w=16, tile_h=16)
    with pytest.raises(IndexError):
        ts.tile_rect(4)
    with pytest.raises(IndexError):
        ts.tile_rect(-1)


def test_uv_is_the_rect_in_texture_coordinates():
    ts = Tileset(name="t", pixels=_pixels(64, 64), tile_w=16, tile_h=16)
    assert ts.uv(0) == (0.0, 0.0, 0.25, 0.25)
    assert ts.uv(5) == (0.25, 0.25, 0.5, 0.5)


def test_firstgid_belongs_to_the_reference_not_the_tileset():
    """The same ``.tsx`` used by two maps has a different firstgid in each, so
    baking it into ``Tileset`` would make one shared object two."""
    ts = Tileset(name="t", pixels=_pixels(32, 32), tile_w=16, tile_h=16)
    a, b = TilesetRef(firstgid=1, tileset=ts), TilesetRef(firstgid=101, tileset=ts)
    assert a.tileset is b.tileset
    assert (a.last_gid, b.last_gid) == (4, 104)
    assert a.holds(4) and not a.holds(5)
    assert b.local(103) == 2


# --- phase variants -------------------------------------------------------------


def test_a_phased_terrain_set_maps_ids_both_ways():
    from warlock.studio.plotter import blob
    from warlock.studio.plotter.tileset import TerrainSpec, Tileset

    k, tile = 2, 8
    terrains = (
        TerrainSpec("A", (1, 2, 3, 255), (0, 0, 0, 255)),
        TerrainSpec("B", (4, 5, 6, 255), (0, 0, 0, 255)),
    )
    ts = Tileset(
        name="g",
        pixels=_pixels(blob.TILE_COUNT * tile, len(terrains) * k * k * tile),
        tile_w=tile,
        tile_h=tile,
        terrains=terrains,
        phases=k,
    )
    for terrain in range(2):
        for phase in range(k * k):
            local = ts.local_for(terrain, blob.FULL, phase)
            assert ts.terrain_of(local) == terrain
    with pytest.raises(IndexError):
        ts.local_for(0, blob.FULL, k * k)


def test_a_phase_count_outside_the_table_is_refused():
    from warlock.studio.plotter.tileset import Tileset

    with pytest.raises(ValueError):
        Tileset(name="g", pixels=_pixels(16, 16), tile_w=8, tile_h=8, phases=3)


def test_a_phased_set_needs_phase_squared_rows_per_terrain():
    from warlock.studio.plotter import blob
    from warlock.studio.plotter.tileset import TerrainSpec, Tileset

    with pytest.raises(ValueError):
        Tileset(
            name="g",
            pixels=_pixels(blob.TILE_COUNT * 8, 8),  # one row, four needed
            tile_w=8,
            tile_h=8,
            terrains=(TerrainSpec("A", (1, 2, 3, 255), (0, 0, 0, 255)),),
            phases=2,
        )
