"""Palette conversion: what every method promises, and what tells them apart.

Three properties hold for all of them and are what the document layer above
relies on: the output is a subset of the palette (so the snap on every later
write is a no-op), alpha is byte-identical, and the same input converts the same
way twice. The rest of this file is negative controls -- a method that quietly
fell through to nearest would pass every one of the three.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import dither

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
RED = (255, 0, 0, 255)
RAMP = [BLACK, (128, 128, 128, 255), WHITE]


def _ramp_image(width: int = 32, height: int = 8) -> np.ndarray:
    """A horizontal grey ramp, fully opaque. The one input every method has a
    different answer for."""
    row = np.linspace(0, 255, width).astype(np.uint8)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., :3] = row[None, :, None]
    out[..., 3] = 255
    return out


def _palette_set(palette) -> set[tuple[int, int, int]]:
    return {tuple(c)[:3] for c in palette}


def _used(pixels: np.ndarray) -> set[tuple[int, int, int]]:
    visible = pixels[..., 3] > 0
    return {tuple(int(c) for c in rgb) for rgb in pixels[..., :3][visible]}


# --- the promises every method makes -----------------------------------------


@pytest.mark.parametrize("method", dither.METHODS)
def test_every_output_colour_is_in_the_palette(method):
    out = dither.convert(_ramp_image(), RAMP, method)
    assert _used(out) <= _palette_set(RAMP)


@pytest.mark.parametrize("method", dither.METHODS)
def test_alpha_is_byte_identical(method):
    source = _ramp_image()
    source[..., 3] = np.linspace(0, 255, source.shape[1]).astype(np.uint8)[None, :]
    out = dither.convert(source, RAMP, method)
    assert np.array_equal(out[..., 3], source[..., 3])


@pytest.mark.parametrize("method", dither.METHODS)
def test_a_transparent_pixel_is_returned_verbatim(method):
    """It has no colour to convert, and rewriting its dead RGB would make a
    no-op conversion look like an edit -- ``indexed``'s rule, unchanged."""
    source = _ramp_image()
    source[0, 0] = (7, 9, 11, 0)
    out = dither.convert(source, RAMP, method)
    assert tuple(out[0, 0]) == (7, 9, 11, 0)


@pytest.mark.parametrize("method", dither.METHODS)
def test_the_same_input_converts_the_same_way_twice(method):
    source = _ramp_image()
    assert np.array_equal(
        dither.convert(source, RAMP, method), dither.convert(source, RAMP, method)
    )


@pytest.mark.parametrize("method", dither.METHODS)
def test_the_input_is_never_written_to(method):
    source = _ramp_image()
    before = source.copy()
    dither.convert(source, RAMP, method)
    assert np.array_equal(source, before)


@pytest.mark.parametrize("method", dither.METHODS)
def test_already_palettised_input_comes_back_unchanged(method):
    """Identity on a document that is already on its table. Without it, opening
    the convert popup and pressing Apply twice would move the picture the second
    time -- and every method's error term would compound across a session."""
    source = np.zeros((8, 8, 4), dtype=np.uint8)
    source[:, :4] = BLACK
    source[:, 4:] = WHITE
    source[..., 3] = 255
    assert np.array_equal(dither.convert(source, RAMP, method), source)


@pytest.mark.parametrize("method", dither.METHODS)
def test_a_one_colour_palette_flattens_rather_than_failing(method):
    out = dither.convert(_ramp_image(), [RED], method)
    assert _used(out) == {RED[:3]}


# --- the negative controls: the methods are not each other -------------------


def test_nearest_bands_a_ramp_into_exactly_three_blocks():
    out = dither.convert(_ramp_image(), RAMP, "nearest")
    row = [tuple(int(c) for c in px) for px in out[0, :, :3]]
    # Three runs and no interleaving: the definition of banding.
    runs = [row[0]]
    for colour in row[1:]:
        if colour != runs[-1]:
            runs.append(colour)
    assert runs == [BLACK[:3], (128, 128, 128), WHITE[:3]]


@pytest.mark.parametrize("method", ("floyd-steinberg", *dither.ORDERED))
def test_a_dither_interleaves_where_nearest_bands(method):
    dithered = dither.convert(_ramp_image(), RAMP, method)
    assert not np.array_equal(dithered, dither.convert(_ramp_image(), RAMP, "nearest"))


def test_bayer2_and_bayer8_are_different_matrices_and_different_pictures():
    """The control the whole ordered path needs: a size that was ignored would
    make every ``bayerN`` the same conversion under three names."""
    source = _ramp_image()
    two = dither.convert(source, RAMP, "bayer2")
    eight = dither.convert(source, RAMP, "bayer8")
    assert not np.array_equal(two, eight)


def test_ordered_dithering_repeats_with_the_matrix():
    """A bayer2 conversion of a *flat* colour is a 2x2 tile, repeated. That is
    what "ordered" means, and it is the property that makes the noise read as
    texture instead of as dirt."""
    flat = np.zeros((8, 8, 4), dtype=np.uint8)
    flat[..., :3] = 64
    flat[..., 3] = 255
    out = dither.convert(flat, [BLACK, WHITE], "bayer2")[..., :3]
    assert np.array_equal(out[0:2, 0:2], out[2:4, 2:4])
    assert np.array_equal(out[0:2, 0:2], out[6:8, 6:8])


def test_a_half_way_colour_comes_out_an_even_chequer():
    """Centred thresholds, stated as what they buy: the mix at the midpoint is
    0.5, and half the cells of the matrix are above it."""
    flat = np.zeros((4, 4, 4), dtype=np.uint8)
    flat[..., :3] = 128
    flat[..., 3] = 255
    out = dither.convert(flat, [BLACK, WHITE], "bayer4")
    whites = int((out[..., 0] == 255).sum())
    assert whites == 8


def test_floyd_steinberg_keeps_the_average_tone_where_nearest_loses_it():
    """The reason to have it: a flat 25% grey on a black-and-white palette is a
    quarter of the pixels white, where nearest makes the whole thing black."""
    flat = np.zeros((16, 16, 4), dtype=np.uint8)
    flat[..., :3] = 64
    flat[..., 3] = 255
    out = dither.convert(flat, [BLACK, WHITE], "floyd-steinberg")
    mean = float(out[..., 0].mean())
    assert abs(mean - 64.0) < 12.0
    assert float(dither.convert(flat, [BLACK, WHITE], "nearest")[..., 0].mean()) == 0.0


def test_the_serpentine_scan_does_not_lean_the_error_to_one_side():
    """A one-directional raster scan piles its error towards one edge; the
    alternating one does not, and the two halves of a flat field carry the same
    amount of ink."""
    flat = np.zeros((32, 32, 4), dtype=np.uint8)
    flat[..., :3] = 100
    flat[..., 3] = 255
    out = dither.convert(flat, [BLACK, WHITE], "floyd-steinberg")[..., 0]
    left = int((out[:, :16] == 255).sum())
    right = int((out[:, 16:] == 255).sum())
    assert abs(left - right) <= max(8, (left + right) // 10)


# --- the matrices themselves -------------------------------------------------


@pytest.mark.parametrize("size", (2, 4, 8))
def test_a_bayer_matrix_is_a_permutation_of_its_cells(size):
    matrix = dither.bayer_matrix(size)
    assert matrix.shape == (size, size)
    cells = np.sort(matrix.reshape(-1) * size * size - 0.5)
    assert np.allclose(cells, np.arange(size * size))


def test_every_threshold_is_strictly_inside_the_unit_interval():
    """A parameter of exactly 0 must always pick the low candidate and one of
    exactly 1 the high one, or a flat region at either end of a ramp picks up a
    sprinkle of the wrong colour."""
    matrix = dither.bayer_matrix(8)
    assert matrix.min() > 0.0
    assert matrix.max() < 1.0


def test_the_matrix_is_built_by_the_recurrence_that_defines_it():
    two = dither.bayer_matrix(2) * 4.0 - 0.5
    assert np.array_equal(two, np.array([[0, 2], [3, 1]], dtype=np.float32))


def test_a_matrix_size_that_is_not_a_power_of_two_is_refused():
    with pytest.raises(ValueError):
        dither.bayer_matrix(3)


def test_the_tiling_is_anchored_at_the_origin():
    """Phase follows the canvas, not the region being written -- a ramp drawn in
    two halves has to line up with itself."""
    tiled = dither.tile_matrix(dither.bayer_matrix(4), (9, 7))
    assert tiled.shape == (9, 7)
    assert np.array_equal(tiled[0:4, 0:4], dither.bayer_matrix(4))
    assert np.array_equal(tiled[4:8, 0:4], dither.bayer_matrix(4))


# --- refusals ----------------------------------------------------------------


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError):
        dither.convert(_ramp_image(), RAMP, "atkinson")


def test_an_empty_palette_is_refused():
    with pytest.raises(ValueError):
        dither.convert(_ramp_image(), [], "bayer4")


def test_the_wrong_array_shape_is_refused():
    with pytest.raises(ValueError):
        dither.convert(np.zeros((4, 4), dtype=np.uint8), RAMP, "bayer4")


# --- building a palette from the pixels --------------------------------------


def test_a_drawing_with_few_enough_colours_gives_its_colours_back_exactly():
    """The common case: the input is already pixel art, and approximating a
    palette the user hand-authored would be the worst possible answer."""
    plane = np.zeros((4, 6, 4), dtype=np.uint8)
    plane[:, 0:2] = (10, 20, 30, 255)
    plane[:, 2:4] = (200, 30, 40, 255)
    plane[:, 4:6] = (250, 250, 250, 255)
    assert dither.build_palette([plane], 8) == [
        (10, 20, 30, 255),
        (200, 30, 40, 255),
        (250, 250, 250, 255),
    ]


def test_the_table_comes_back_sorted_by_brightness():
    plane = np.zeros((1, 3, 4), dtype=np.uint8)
    plane[0, 0] = (255, 255, 255, 255)
    plane[0, 1] = (0, 0, 0, 255)
    plane[0, 2] = (128, 128, 128, 255)
    assert [c[0] for c in dither.build_palette([plane], 8)] == [0, 128, 255]


def test_transparent_pixels_contribute_no_colours():
    """A palette describes a subject's colours; spending an entry on the dead
    RGB behind the transparency is the failure mode the rule exists to avoid."""
    plane = np.zeros((2, 2, 4), dtype=np.uint8)
    plane[0, 0] = (10, 20, 30, 255)
    plane[0, 1] = (99, 99, 99, 0)
    assert dither.build_palette([plane], 8) == [(10, 20, 30, 255)]


def test_the_cut_is_weighted_by_pixels_and_not_by_distinct_colours():
    """A thousand near-identical antialiasing colours over a handful of pixels
    must not out-vote the flat fill behind them. Sixty pixels of one red against
    four single-pixel near-blacks, cut to two: the red survives."""
    plane = np.zeros((8, 8, 4), dtype=np.uint8)
    plane[..., 3] = 255
    plane[..., :3] = (200, 30, 30)
    for i in range(4):
        plane[0, i, :3] = (i, i, i)
    table = dither.build_palette([plane], 2)
    assert len(table) == 2
    reds = [c for c in table if c[0] > 100]
    assert len(reds) == 1
    assert abs(reds[0][0] - 200) <= 2


def test_more_colours_than_the_budget_are_cut_down_to_it():
    table = dither.build_palette([_ramp_image(64, 4)], 5)
    assert 1 <= len(table) <= 5


def test_the_cut_is_deterministic():
    source = [_ramp_image(64, 4)]
    assert dither.build_palette(source, 7) == dither.build_palette(source, 7)


def test_every_plane_of_the_document_is_read():
    a = np.zeros((1, 1, 4), dtype=np.uint8)
    a[0, 0] = (10, 10, 10, 255)
    b = np.zeros((1, 1, 4), dtype=np.uint8)
    b[0, 0] = (200, 200, 200, 255)
    assert dither.build_palette([a, b], 8) == [
        (10, 10, 10, 255),
        (200, 200, 200, 255),
    ]


def test_a_document_with_nothing_visible_still_gives_a_palette():
    """An indexed document with no colours is one every visible pixel would have
    to snap to nothing, so the empty answer is one colour rather than none."""
    assert dither.build_palette([np.zeros((4, 4, 4), dtype=np.uint8)], 8) == [
        (0, 0, 0, 255)
    ]


def test_a_built_palette_never_repeats_a_swatch():
    """Two boxes can land on the same representative, and a duplicate slot is
    one the user cannot tell from its neighbour and a wasted GIF entry."""
    table = dither.build_palette([_ramp_image(200, 4)], 64)
    assert len(set(table)) == len(table)


def test_a_zero_budget_is_refused():
    with pytest.raises(ValueError):
        dither.build_palette([_ramp_image()], 0)


def test_the_built_palette_is_what_a_conversion_can_be_run_against():
    """The two halves of "palette from document" meet here: everything
    ``build_palette`` returns is a legal input to ``convert``."""
    source = _ramp_image(64, 4)
    table = dither.build_palette([source], 6)
    out = dither.convert(source, table, "bayer4")
    assert _used(out) <= _palette_set(table)
