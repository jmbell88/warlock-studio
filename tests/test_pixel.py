"""The pixel-art grid, palettes and cleanup.

The load-bearing test here is the first one: authored pixels, blown up to a
canvas the way a pixel-art generator draws them, must come back *exactly*. Every
other property in this module is in service of that one -- a phase measured to
within a pixel or a reduction sampled anywhere but a cell centre turns pixel art
back into a small photograph, and nothing downstream can recover it.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import pixel


def _authored(size: int = 24, seed: int = 7) -> Image.Image:
    """A small image of flat, hard-edged blocks -- authored pixel art."""
    rng = np.random.default_rng(seed)
    # A handful of colours, in patches, so neighbouring pixels usually agree
    # (real pixel art) but edges are hard.
    palette = np.array(
        [[20, 20, 30], [200, 60, 60], [60, 180, 90], [230, 210, 120], [255, 255, 255]],
        dtype=np.uint8,
    )
    idx = rng.integers(0, len(palette), size=(size, size))
    return Image.fromarray(palette[idx], "RGB")


def _blown_up(art: Image.Image, scale: int, phase: tuple[int, int]) -> Image.Image:
    """The authored art drawn as scale x scale blocks at a known phase, on a
    canvas whose edges carry partial cells -- which is what a generator's output
    looks like."""
    py, px = phase
    big = art.resize((art.width * scale, art.height * scale), Image.NEAREST)
    canvas = Image.new("RGB", (big.width + scale, big.height + scale), (128, 128, 128))
    canvas.paste(big, (px, py))
    return canvas


def test_exact_8x_reduction_recovers_the_authored_pixels():
    art = _authored()
    canvas = _blown_up(art, 8, (5, 3))

    grid = pixel.detect_grid(canvas)
    assert grid["scale"] == 8
    assert tuple(grid["phase"]) == (5, 3)

    small = pixel.downscale_grid(canvas, grid["scale"], tuple(grid["phase"]))
    # The reduction covers the whole frame including the partial edge cells, so
    # the authored art sits at the origin of the result.
    recovered = np.asarray(small.convert("RGB"))[: art.height, : art.width]
    assert np.array_equal(recovered, np.asarray(art))


@pytest.mark.parametrize("scale", [4, 8, 16])
def test_the_phase_is_found_at_every_offset(scale):
    art = _authored(size=12)
    for phase in ((0, 0), (1, scale - 1), (scale // 2, 2)):
        grid = pixel.detect_grid(_blown_up(art, scale, phase))
        assert grid["scale"] == scale, (scale, phase)
        assert tuple(grid["phase"]) == phase, (scale, phase)


def test_residual_is_zero_on_the_grid_and_rises_off_it():
    canvas = _blown_up(_authored(), 8, (0, 0))
    arr = np.asarray(canvas)
    on = pixel.grid_residual(arr, 8, (0, 0))
    off = pixel.grid_residual(arr, 8, (4, 4))
    assert on == pytest.approx(0.0, abs=1e-9)
    assert off > on
    assert on < pixel.GRID_RESIDUAL_MAX < off


def test_a_smooth_image_has_no_grid():
    """The common case, and the one that must keep the legacy export path: a
    gradient is not pixel art, and calling it pixel art would silently change
    every ordinary asset's reduction."""
    x = np.linspace(0, 255, 256)
    ramp = np.dstack([np.tile(x, (256, 1))] * 3).astype(np.uint8)
    grid = pixel.detect_grid(Image.fromarray(ramp, "RGB"))
    assert grid["scale"] is None


# --------------------------------------------------------------------------
# Palette files
# --------------------------------------------------------------------------


def test_hex_palette_survives_headers_and_blank_lines():
    text = "; Lospec palette\n\n#1a1c2c\nf4f4f4\n// trailing note\n"
    assert pixel.parse_hex(text) == ((26, 28, 44), (244, 244, 244))


def test_a_line_that_is_trying_to_be_a_colour_and_is_not_raises():
    # Silently dropping it would change what every mapped pixel becomes.
    with pytest.raises(ValueError):
        pixel.parse_hex("1a1c2c\nzzzz\n")


def test_gpl_needs_its_own_magic():
    good = "GIMP Palette\nName: Test\n#\n 26  28  44 dark\n244 244 244 light\n"
    assert pixel.parse_gpl(good) == ((26, 28, 44), (244, 244, 244))
    with pytest.raises(ValueError):
        pixel.parse_gpl("26 28 44\n")


def test_parse_palette_dispatches_on_suffix_and_refuses_anything_else():
    assert pixel.parse_palette("1a1c2c\n", ".HEX") == ((26, 28, 44),)
    with pytest.raises(ValueError):
        pixel.parse_palette("1a1c2c\n", ".txt")


def test_digest_ignores_formatting_and_not_colour():
    a = pixel.parse_hex("#1a1c2c\n#f4f4f4\n")
    b = pixel.parse_gpl("GIMP Palette\n26 28 44\n244 244 244\n")
    assert pixel.palette_digest(a) == pixel.palette_digest(b)
    c = pixel.parse_hex("#1a1c2d\n#f4f4f4\n")
    assert pixel.palette_digest(a) != pixel.palette_digest(c)


# --------------------------------------------------------------------------
# Colour mapping
# --------------------------------------------------------------------------


def _colours(image: Image.Image) -> set[tuple[int, int, int]]:
    arr = np.asarray(image.convert("RGBA"))
    return {tuple(int(c) for c in p) for p in arr[arr[:, :, 3] > 0][:, :3]}


@pytest.mark.parametrize("dither", [False, True])
def test_mapped_output_contains_only_palette_members(dither):
    rng = np.random.default_rng(3)
    src = Image.fromarray(
        rng.integers(0, 256, size=(16, 16, 4), dtype=np.uint8), "RGBA"
    )
    palette = ((0, 0, 0), (255, 255, 255), (200, 40, 40), (40, 90, 200))
    out = pixel.map_palette(src, palette, dither=dither)
    assert _colours(out) <= set(palette)


def test_mapping_is_perceptual_rather_than_nearest_in_rgb():
    """Shadow detail is where the two disagree and where it matters.

    A dark grey is nearer black than it is to a mid grey by raw RGB arithmetic,
    but perceptually the opposite is true -- sRGB spends most of its range on
    the darks. Mapping it to black is how a limited palette eats a whole
    shadow.
    """
    dark = (30, 30, 30)
    black = (0, 0, 0)
    mid = (70, 70, 70)
    assert np.linalg.norm(np.subtract(dark, black)) < np.linalg.norm(
        np.subtract(dark, mid)
    )
    out = pixel.map_palette(Image.new("RGBA", (2, 2), (*dark, 255)), (black, mid))
    assert _colours(out) == {mid}


def test_mapping_leaves_alpha_exactly_alone():
    src = Image.fromarray(
        np.dstack(
            [
                np.full((4, 4), 128, np.uint8),
                np.full((4, 4), 128, np.uint8),
                np.full((4, 4), 128, np.uint8),
                np.arange(16, dtype=np.uint8).reshape(4, 4) * 16,
            ]
        ),
        "RGBA",
    )
    out = pixel.map_palette(src, ((0, 0, 0), (255, 255, 255)))
    assert np.array_equal(
        np.asarray(out)[:, :, 3], np.asarray(src)[:, :, 3]
    )


def test_dither_actually_mixes_where_a_flat_map_would_not():
    """A flat mid-grey between two palette entries maps to one colour without
    dithering and to both with it -- that is the whole feature."""
    src = Image.new("RGBA", (8, 8), (128, 128, 128, 255))
    palette = ((0, 0, 0), (255, 255, 255))
    assert len(_colours(pixel.map_palette(src, palette))) == 1
    assert len(_colours(pixel.map_palette(src, palette, dither=True))) == 2


# --------------------------------------------------------------------------
# Cleanup and measurement
# --------------------------------------------------------------------------


def test_orphan_removal_takes_the_lone_pixel_and_keeps_the_pair():
    arr = np.zeros((7, 7, 4), np.uint8)
    arr[:, :, :3] = (20, 20, 20)
    arr[:, :, 3] = 255
    arr[1, 1, :3] = (255, 0, 0)          # alone
    arr[4, 4, :3] = arr[4, 5, :3] = (0, 255, 0)  # a deliberate pair
    out, removed = pixel.clean_orphans(Image.fromarray(arr, "RGBA"))
    got = np.asarray(out)
    assert removed == 1
    assert tuple(got[1, 1, :3]) == (20, 20, 20)
    assert tuple(got[4, 4, :3]) == (0, 255, 0)
    assert tuple(got[4, 5, :3]) == (0, 255, 0)


def test_a_transparent_hole_is_a_silhouette_decision_not_an_orphan():
    arr = np.zeros((7, 7, 4), np.uint8)
    arr[:, :, :3] = (20, 20, 20)
    arr[:, :, 3] = 255
    arr[3, 3, 3] = 0
    out, removed = pixel.clean_orphans(Image.fromarray(arr, "RGBA"))
    assert removed == 0
    assert np.asarray(out)[3, 3, 3] == 0


def test_snap_alpha_is_binary():
    src = Image.fromarray(
        np.dstack([np.zeros((1, 4), np.uint8)] * 3
                  + [np.array([[0, 127, 128, 255]], np.uint8)]),
        "RGBA",
    )
    got = np.asarray(pixel.snap_alpha(src))[:, :, 3]
    assert got.tolist() == [[0, 0, 255, 255]]


def test_report_and_verdict_say_something_readable():
    palette = ((0, 0, 0), (255, 255, 255))
    src = pixel.map_palette(Image.new("RGBA", (8, 8), (10, 10, 10, 255)), palette)
    rep = pixel.report(src, palette=palette, grid={"scale": 8, "residual": 0.01})
    assert rep["colors"] == 1
    assert rep["palette_fraction"] == 1.0
    sentence = pixel.verdict(rep)
    assert sentence and "8px source grid" in sentence
    # UI strings stay inside imgui's default Basic-Latin + Latin-1 atlas.
    sentence.encode("latin-1")
