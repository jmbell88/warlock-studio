"""The AI-free pixeliser, including the determinism bar Phase 1 sets.

"A fixed render and a fixed palette must reduce byte-identically" is the whole
verification story for this stage: the chain has no RNG in it, so anything that
is not reproducible is a bug rather than a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import pixelize

RAMP = ((16, 16, 24), (72, 56, 48), (168, 120, 88), (232, 216, 184))


def _render(size=64, seed=7):
    """A deterministic stand-in for a rendered frame: a soft blob on nothing,
    with a partial-alpha edge, which is exactly what EEVEE hands back."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx) / (size * 0.4)
    alpha = np.clip((1.0 - r) * 400.0, 0, 255)
    shade = np.clip(220 - r * 160 + rng.normal(0, 6, (size, size)), 0, 255)
    arr = np.dstack([shade, shade * 0.85, shade * 0.7, alpha]).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# --- reduction ---------------------------------------------------------------


def test_an_integer_stride_reduces_by_supersampling():
    small = pixelize.reduce(_render(64), (16, 16))
    assert small.size == (16, 16)


def test_a_non_dividing_target_falls_back_to_one_nearest_resize():
    """24 does not divide 64; the documented fallback must still produce the
    asked-for size rather than refusing or silently rounding."""
    assert pixelize.reduce(_render(64), (24, 24)).size == (24, 24)


def test_the_reduction_is_alpha_weighted_so_edges_do_not_darken():
    """A plain RGBA mean averages the transparent background's black into every
    silhouette edge. Weighted, an edge pixel keeps its own colour."""
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[0:2, 0:2] = (200, 180, 160, 255)   # one quadrant solid, rest empty
    src = Image.fromarray(arr, "RGBA")
    out = np.asarray(pixelize.reduce(src, (1, 1)))[0, 0]
    assert tuple(out[:3]) == (200, 180, 160)
    assert out[3] == 64                    # coverage, a quarter of 255


def test_point_mode_takes_the_cell_centre_sample():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    arr[:, :] = (10, 10, 10, 255)
    arr[2, 2] = (99, 99, 99, 255)          # the centre of a 4x stride
    src = Image.fromarray(arr, "RGBA")
    out = np.asarray(pixelize.reduce(src, (1, 1), mode="point"))[0, 0]
    assert tuple(out[:3]) == (99, 99, 99)


def test_an_unknown_reduce_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be one of"):
        pixelize.reduce(_render(16), (8, 8), mode="lanczos")


# --- outline -----------------------------------------------------------------


def _square(size=8, inset=2):
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[inset : size - inset, inset : size - inset] = (200, 200, 200, 255)
    return Image.fromarray(arr, "RGBA")


def test_an_inner_outline_cannot_change_the_silhouette():
    src = _square()
    out = pixelize.outline(src, (0, 0, 0), mode="inner")
    assert (np.asarray(out)[:, :, 3] == np.asarray(src)[:, :, 3]).all()
    assert tuple(np.asarray(out)[2, 2, :3]) == (0, 0, 0)
    assert tuple(np.asarray(out)[3, 3, :3]) == (200, 200, 200)


def test_an_outer_outline_grows_the_silhouette_by_exactly_one_pixel():
    out = np.asarray(pixelize.outline(_square(), (0, 0, 0), mode="outer"))
    assert out[1, 3, 3] == 255 and tuple(out[1, 3, :3]) == (0, 0, 0)
    assert out[0, 3, 3] == 0
    # Four-neighbour: the diagonal corner is not claimed.
    assert out[1, 1, 3] == 0


def test_no_outline_leaves_the_frame_alone():
    src = _square()
    assert (np.asarray(pixelize.outline(src, (0, 0, 0), mode="none")) == np.asarray(src)).all()


def test_the_default_outline_colour_comes_from_the_ramp():
    assert pixelize.darkest(RAMP) == (16, 16, 24)


# --- the whole chain ---------------------------------------------------------


def test_the_result_uses_only_palette_colours():
    out, report = pixelize.pixelize(_render(64), size=(16, 16), palette=RAMP)
    colours = {tuple(p) for p in np.asarray(out)[np.asarray(out)[:, :, 3] > 0][:, :3].tolist()}
    assert colours <= set(RAMP)
    assert report["colors"] == len(colours)


def test_alpha_is_binary_after_the_chain():
    out, _ = pixelize.pixelize(_render(64), size=(16, 16), palette=RAMP)
    assert set(np.unique(np.asarray(out)[:, :, 3]).tolist()) <= {0, 255}


def test_the_chain_is_byte_identical_run_to_run():
    """The Phase 1 bar, stated as a test."""
    a, ra = pixelize.pixelize(_render(64), size=(16, 16), palette=RAMP, outline_mode="inner")
    b, rb = pixelize.pixelize(_render(64), size=(16, 16), palette=RAMP, outline_mode="inner")
    assert a.tobytes() == b.tobytes()
    assert ra == rb


def test_an_outlined_frame_still_only_contains_palette_colours():
    out, _ = pixelize.pixelize(
        _render(64), size=(16, 16), palette=RAMP, outline_mode="outer"
    )
    colours = {tuple(p) for p in np.asarray(out)[np.asarray(out)[:, :, 3] > 0][:, :3].tolist()}
    assert colours <= set(RAMP)


def test_an_empty_palette_is_refused():
    with pytest.raises(ValueError, match="empty palette"):
        pixelize.pixelize(_render(16), size=(8, 8), palette=())


def test_the_report_says_whether_the_stride_was_exact():
    _, exact = pixelize.pixelize(_render(64), size=(16, 16), palette=RAMP)
    _, fallback = pixelize.pixelize(_render(64), size=(24, 24), palette=RAMP)
    assert exact["exact_stride"] is True
    assert fallback["exact_stride"] is False


# --- atlases -----------------------------------------------------------------


def _atlas(columns=2, rows=2, cell=32):
    atlas = Image.new("RGBA", (columns * cell, rows * cell), (0, 0, 0, 0))
    for row in range(rows):
        for column in range(columns):
            atlas.paste(_render(cell, seed=row * columns + column), (column * cell, row * cell))
    return atlas


def test_an_atlas_reduces_on_exact_cell_boundaries():
    out, report = pixelize.pixelize_atlas(
        _atlas(), columns=2, rows=2, cell=8, palette=RAMP
    )
    assert out.size == (16, 16)
    assert report["cell"] == 8


def test_an_atlas_whose_cells_do_not_divide_it_is_refused():
    with pytest.raises(ValueError, match="whole cells"):
        pixelize.pixelize_atlas(
            Image.new("RGBA", (33, 32)), columns=2, rows=2, cell=8, palette=RAMP
        )


def test_an_outline_never_crosses_a_cell_seam():
    """The reason the neighbourhood passes are per cell: a dense atlas has no
    gutter, so a sprite that fills its cell would otherwise outline against the
    sprite beside it."""
    cell = 4
    arr = np.zeros((cell, 2 * cell, 4), dtype=np.uint8)
    arr[:, :cell] = (200, 200, 200, 255)   # left cell full, right cell empty
    atlas = Image.fromarray(arr, "RGBA")
    out, _ = pixelize.pixelize_atlas(
        atlas, columns=2, rows=1, cell=cell, palette=RAMP, outline_mode="outer"
    )
    assert (np.asarray(out)[:, cell:, 3] == 0).all()


def test_an_atlas_is_byte_identical_run_to_run():
    a, _ = pixelize.pixelize_atlas(_atlas(), columns=2, rows=2, cell=8, palette=RAMP)
    b, _ = pixelize.pixelize_atlas(_atlas(), columns=2, rows=2, cell=8, palette=RAMP)
    assert a.tobytes() == b.tobytes()
