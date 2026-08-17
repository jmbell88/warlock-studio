"""The textured ground set: its band geometry, and its bridge to the flat one.

Two claims carry this module. The first is :mod:`~warlock.studio.plotter.tilegen`'s
determinism promise, inherited: the same textures give the same bytes, because
nothing here decides anything with a float or an RNG. The second is the bridge --
**at a border of one pixel every case reproduces the flat generator's footprint
exactly** -- which is what keeps the two from becoming two art rules that drift
apart, and is asserted case by case rather than on a sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter import blob, groundtex, project, tilegen
from warlock.studio.plotter.tileset import TerrainSpec

ORTHO = project.ORTHOGONAL
ISO = project.ISOMETRIC


def _texture(seed: int, tile_w: int, tile_h: int) -> np.ndarray:
    """A tile-sized texture with a different byte in every channel of every
    pixel, so a test can tell which material a pixel came from."""
    count = tile_h * tile_w * 4
    flat = (np.arange(count, dtype=np.int64) * 7 + seed * 101) % 251
    return flat.astype(np.uint8).reshape(tile_h, tile_w, 4)


def _pair(terrains: int, tile_w: int, tile_h: int):
    fills = [_texture(i, tile_w, tile_h) for i in range(terrains)]
    borders = [_texture(100 + i, tile_w, tile_h) for i in range(terrains)]
    return fills, borders


# -- determinism -------------------------------------------------------------


@pytest.mark.parametrize("projection", [ORTHO, ISO])
def test_the_same_textures_give_the_same_bytes(projection):
    fills, borders = _pair(2, 16, 16)
    first = groundtex.compose_atlas(16, 16, projection, 3, fills, borders)
    second = groundtex.compose_atlas(16, 16, projection, 3, fills, borders)
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize("projection", [ORTHO, ISO])
def test_the_atlas_is_one_row_per_terrain_and_one_column_per_case(projection):
    fills, borders = _pair(3, 16, 8)
    atlas = groundtex.compose_atlas(16, 8, projection, 2, fills, borders)
    assert atlas.shape == (3 * 8, blob.TILE_COUNT * 16, 4)


# -- the bridge to the flat generator ----------------------------------------


@pytest.mark.parametrize("projection", [ORTHO, ISO])
def test_a_one_pixel_border_reproduces_the_flat_outline_exactly(projection):
    """The bridge. Every one of the forty-seven, in both projections: widening
    the outline is the only thing this module does differently."""
    spec = tilegen.GenSpec(tile_w=16, tile_h=16, projection=projection)
    for case in range(blob.TILE_COUNT):
        flat = tilegen.tile_coverage(spec, case)
        band = groundtex.band_coverage(16, 16, projection, case, 1)
        assert np.array_equal(flat, band), f"case {case} differs at border one"


def test_an_odd_isometric_tile_also_bridges():
    """The diamond's row spans are integer division, so an odd height is where a
    second spelling of them would show up first."""
    spec = tilegen.GenSpec(tile_w=17, tile_h=9, projection=ISO)
    for case in range(blob.TILE_COUNT):
        assert np.array_equal(
            tilegen.tile_coverage(spec, case),
            groundtex.band_coverage(17, 9, ISO, case, 1),
        )


# -- band geometry -----------------------------------------------------------


def test_an_interior_tile_is_entirely_fill():
    codes = groundtex.band_coverage(16, 16, ORTHO, blob.FULL, 4)
    assert (codes == groundtex.FILL).all()


def test_an_open_edge_gets_a_band_of_the_asked_for_thickness():
    codes = groundtex.band_coverage(16, 16, ORTHO, blob.LONE, 3)
    assert (codes[0:3, :] == groundtex.BORDER).all()
    assert (codes[-3:, :] == groundtex.BORDER).all()
    assert (codes[:, 0:3] == groundtex.BORDER).all()
    assert (codes[:, -3:] == groundtex.BORDER).all()
    # And the middle is untouched -- a band is a rim, not a repaint.
    assert (codes[3:-3, 3:-3] == groundtex.FILL).all()


def test_a_notched_corner_is_a_border_square_and_nothing_else():
    """Every edge has a neighbour and the north-east diagonal does not: the one
    case that separates the forty-seven from the sixteen."""
    mask = 0xFF & ~blob.NE
    case = int(blob.BLOB_INDEX[mask])
    codes = groundtex.band_coverage(16, 16, ORTHO, case, 4)
    assert (codes[0:4, -4:] == groundtex.BORDER).all()
    assert (codes == groundtex.BORDER).sum() == 16


def test_an_isometric_band_is_half_as_thick_across_the_row():
    """A diamond's edge runs at a slope, so the same visual weight is half the
    pixel count measured horizontally."""
    codes = groundtex.band_coverage(32, 16, ISO, blob.LONE, 4)
    row = codes[0]
    inside = np.flatnonzero(row != groundtex.VOID)
    assert len(inside) >= 2
    # The top row of a lone diamond is border on both ends of its span.
    assert row[inside[0]] == groundtex.BORDER
    assert row[inside[-1]] == groundtex.BORDER


def test_an_isometric_tile_is_void_outside_its_diamond():
    codes = groundtex.band_coverage(32, 16, ISO, blob.FULL, 4)
    assert (codes == groundtex.VOID).any()
    assert codes[0, 0] == groundtex.VOID
    assert codes[0, -1] == groundtex.VOID


def test_an_isometric_case_is_symmetric_about_its_vertical_axis():
    """North-east and south-west are mirror images across the diamond's vertical
    axis, and an even-height diamond has two widest rows -- marking one of them
    would leave a one-pixel wobble along an otherwise straight coastline."""
    ne_case = int(blob.BLOB_INDEX[0xFF & ~blob.NE])
    sw_case = int(blob.BLOB_INDEX[0xFF & ~blob.SW])
    ne_codes = groundtex.band_coverage(32, 16, ISO, ne_case, 4)
    sw_codes = groundtex.band_coverage(32, 16, ISO, sw_case, 4)
    assert (ne_codes == groundtex.BORDER).sum() == (sw_codes == groundtex.BORDER).sum()


# -- composition -------------------------------------------------------------


def test_a_full_tile_is_the_fill_texture_verbatim():
    """The seam promise, stated as an assertion: a FULL tile is the material at
    its own local coordinates, so two of them side by side continue it."""
    fills, borders = _pair(1, 16, 16)
    atlas = groundtex.compose_atlas(16, 16, ORTHO, 4, fills, borders)
    x0 = blob.FULL * 16
    tile = atlas[0:16, x0 : x0 + 16]
    expected = fills[0].copy()
    expected[:, :, 3] = 255
    assert np.array_equal(tile, expected)


def test_a_border_pixel_comes_from_the_border_texture():
    fills, borders = _pair(1, 16, 16)
    atlas = groundtex.compose_atlas(16, 16, ORTHO, 4, fills, borders)
    x0 = blob.LONE * 16
    tile = atlas[0:16, x0 : x0 + 16]
    assert np.array_equal(tile[0, 0, :3], borders[0][0, 0, :3])
    assert np.array_equal(tile[8, 8, :3], fills[0][8, 8, :3])


def test_each_terrain_reads_its_own_row_of_textures():
    fills, borders = _pair(3, 8, 8)
    atlas = groundtex.compose_atlas(8, 8, ORTHO, 2, fills, borders)
    for row in range(3):
        y0, x0 = row * 8, blob.FULL * 8
        assert np.array_equal(atlas[y0 : y0 + 8, x0 : x0 + 8, :3], fills[row][:, :, :3])


def test_void_stays_transparent_and_everything_else_opaque():
    fills, borders = _pair(1, 32, 16)
    atlas = groundtex.compose_atlas(32, 16, ISO, 4, fills, borders)
    x0 = blob.FULL * 32
    tile = atlas[0:16, x0 : x0 + 32]
    codes = groundtex.band_coverage(32, 16, ISO, blob.FULL, 4)
    assert (tile[codes == groundtex.VOID, 3] == 0).all()
    assert (tile[codes != groundtex.VOID, 3] == 255).all()


def test_an_rgb_texture_is_accepted_and_made_opaque():
    fills = [np.zeros((8, 8, 3), dtype=np.uint8)]
    borders = [np.full((8, 8, 3), 9, dtype=np.uint8)]
    atlas = groundtex.compose_atlas(8, 8, ORTHO, 2, fills, borders)
    assert (atlas[:, :, 3] == 255).all()


# -- phase variants -----------------------------------------------------------


def test_variants_one_is_byte_identical_to_the_classic_atlas():
    """The regression pin: k=1 must not move a single byte, or every stored
    set recomposes differently."""
    fills, borders = _pair(2, 16, 16)
    classic = groundtex.compose_atlas(16, 16, ORTHO, 3, fills, borders)
    explicit = groundtex.compose_atlas(16, 16, ORTHO, 3, fills, borders, 1)
    assert classic.tobytes() == explicit.tobytes()


def test_each_phase_is_its_own_slice_of_the_period():
    """Phase (px, py)'s FULL tile is the texture's tile-sized slice at
    (px * w, py * h) -- the whole reason a painted field reads as one larger
    surface."""
    k, tw, th = 2, 8, 8
    fills = [_texture(1, k * tw, k * th)]
    borders = [_texture(2, k * tw, k * th)]
    atlas = groundtex.compose_atlas(tw, th, ORTHO, 2, fills, borders, k)
    assert atlas.shape == (k * k * th, blob.TILE_COUNT * tw, 4)
    x0 = blob.FULL * tw
    for py in range(k):
        for px in range(k):
            y0 = (py * k + px) * th
            tile = atlas[y0 : y0 + th, x0 : x0 + tw]
            want = fills[0][py * th : (py + 1) * th, px * tw : (px + 1) * tw].copy()
            want[:, :, 3] = 255
            assert np.array_equal(tile, want), (px, py)


def test_the_full_tiles_reassemble_the_whole_fill_texture():
    """k x k FULL tiles laid out in phase order are the period, exactly."""
    k, tw, th = 4, 4, 4
    fills = [_texture(5, k * tw, k * th)]
    borders = [_texture(6, k * tw, k * th)]
    atlas = groundtex.compose_atlas(tw, th, ORTHO, 1, fills, borders, k)
    x0 = blob.FULL * tw
    rebuilt = np.zeros((k * th, k * tw, 4), dtype=np.uint8)
    for py in range(k):
        for px in range(k):
            y0 = (py * k + px) * th
            rebuilt[py * th : (py + 1) * th, px * tw : (px + 1) * tw] = atlas[
                y0 : y0 + th, x0 : x0 + tw
            ]
    want = fills[0].copy()
    want[:, :, 3] = 255
    assert np.array_equal(rebuilt, want)


def test_a_phase_texture_of_the_wrong_size_is_refused_naming_both_sizes():
    fills = [np.zeros((8, 8, 4), dtype=np.uint8)]
    borders = [np.zeros((16, 16, 4), dtype=np.uint8)]
    with pytest.raises(ValueError) as excinfo:
        groundtex.compose_atlas(8, 8, ORTHO, 2, fills, borders, 2)
    message = str(excinfo.value)
    assert "8x8" in message and "16x16" in message


def test_a_spec_validates_the_phase_count_and_the_period():
    with pytest.raises(ValueError):
        groundtex.GroundSpec(tile_w=16, tile_h=16, variants=3)
    with pytest.raises(ValueError):
        groundtex.GroundSpec(tile_w=512, tile_h=512, variants=4)
    assert groundtex.GroundSpec(tile_w=16, tile_h=16, variants=4).variants == 4


# -- refusals ----------------------------------------------------------------


def test_a_texture_of_the_wrong_size_is_refused_naming_both_sizes():
    fills = [np.zeros((7, 9, 4), dtype=np.uint8)]
    borders = [np.zeros((8, 8, 4), dtype=np.uint8)]
    with pytest.raises(ValueError) as excinfo:
        groundtex.compose_atlas(8, 8, ORTHO, 2, fills, borders)
    message = str(excinfo.value)
    assert "9x7" in message and "8x8" in message


def test_mismatched_texture_counts_are_refused():
    fills, borders = _pair(2, 8, 8)
    with pytest.raises(ValueError):
        groundtex.compose_atlas(8, 8, ORTHO, 2, fills, borders[:1])


def test_no_terrains_at_all_is_refused():
    with pytest.raises(ValueError):
        groundtex.compose_atlas(8, 8, ORTHO, 2, [], [])


# -- the default thickness ---------------------------------------------------


@pytest.mark.parametrize(
    "tile_w,tile_h,expected",
    [(32, 32, 4), (16, 16, 2), (8, 8, 2), (4, 4, 1), (128, 64, 8)],
)
def test_the_default_border_is_an_eighth_of_the_shorter_edge(tile_w, tile_h, expected):
    assert groundtex.default_border(tile_w, tile_h) == expected


@pytest.mark.parametrize("edge", range(4, 65))
def test_two_opposite_bands_never_meet_in_the_middle(edge):
    """A tile whose fill has been entirely eaten shows no material at all, and
    every one of the forty-seven cases then looks alike."""
    border = groundtex.default_border(edge, edge)
    assert border >= 1 and 2 * border < edge


# -- the spec ----------------------------------------------------------------


def test_a_spec_fills_in_the_border_when_nobody_said():
    spec = groundtex.GroundSpec(tile_w=32, tile_h=32)
    assert spec.border == groundtex.default_border(32, 32)


def test_a_spec_refuses_a_border_that_would_swallow_the_tile():
    with pytest.raises(ValueError) as excinfo:
        groundtex.GroundSpec(tile_w=8, tile_h=8, border=4)
    assert "1..3" in str(excinfo.value)


def test_a_spec_refuses_what_the_flat_generator_refuses():
    """Read off the same two constants, so the two cannot come to disagree about
    how big a set may be."""
    many = tuple(
        TerrainSpec(name=f"t{i}", fill=(1, 2, 3, 255), outline=(0, 0, 0, 255))
        for i in range(tilegen.MAX_TERRAINS + 1)
    )
    with pytest.raises(ValueError):
        groundtex.GroundSpec(terrains=many)
    with pytest.raises(ValueError):
        groundtex.GroundSpec(tile_w=tilegen.MAX_TILE_EDGE + 1)
    with pytest.raises(ValueError):
        groundtex.GroundSpec(tile_w=3)
    with pytest.raises(ValueError):
        groundtex.GroundSpec(terrains=())
    with pytest.raises(ValueError):
        groundtex.GroundSpec(projection="hexagonal")


def test_a_spec_composes_with_its_own_numbers():
    spec = groundtex.GroundSpec(terrains=tilegen.DEFAULT_TERRAINS[:2], tile_w=8, tile_h=8)
    fills, borders = _pair(2, 8, 8)
    assert spec.atlas(fills, borders).shape == (16, blob.TILE_COUNT * 8, 4)
