"""Reducing an upscaled pixel-art render back onto the grid it was drawn on.

The measurement already existed on the export path (``pipelines/pixel.py``);
what these tests pin is the numpy port inside the editor and, above all, the
two properties that make it worth having at all: it is *phase*-aware, so the
round trip is bit-exact rather than approximately right, and it is pure element
selection, so it can never mint a colour.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import transform as tf
from warlock.studio.inker.document import Document


def _art(cells: np.ndarray, scale: int, phase: tuple[int, int] = (0, 0)) -> np.ndarray:
    """``cells`` blown up ``scale`` times, offset by ``phase``, on a grey field.

    The offset is what makes the fixture worth having: art that starts at the
    origin is reduced correctly by any sampler that ignores the phase.
    """
    blown = np.repeat(np.repeat(cells, scale, axis=0), scale, axis=1)
    py, px = phase
    out = np.full(
        (blown.shape[0] + py, blown.shape[1] + px, 4), 128, dtype=np.uint8
    )
    out[..., 3] = 255
    out[py:, px:] = blown
    return out


def _cells(rows: int = 8, cols: int = 8, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((rows, cols, 4), dtype=np.uint8)
    # A handful of flat colours -- what authored pixel art actually is.
    palette = np.array(
        [[20, 20, 30], [200, 60, 40], [40, 160, 90], [230, 220, 190], [90, 90, 110]],
        dtype=np.uint8,
    )
    out[..., :3] = palette[rng.integers(0, len(palette), (rows, cols))]
    out[..., 3] = 255
    return out


def test_a_blown_up_drawing_round_trips_bit_exactly() -> None:
    cells = _cells(10, 10)
    for scale in (4, 8):
        art = _art(cells, scale)
        found = tf.detect_pixel_grid(art)
        assert found["scale"] == scale, found
        back = tf.descale(art, found["scale"], found["phase"])
        assert np.array_equal(back, cells)


def test_a_phase_offset_lattice_round_trips_bit_exactly() -> None:
    """The whole reason this is not ``scale(..., resample="nearest")``: an
    off-phase reduction samples the neighbouring cell's edge for half the art."""
    cells = _cells(10, 10)
    art = _art(cells, 8, phase=(3, 5))
    found = tf.detect_pixel_grid(art)
    assert found["scale"] == 8
    assert found["phase"] == (3, 5)
    assert np.array_equal(tf.descale(art, 8, found["phase"]), cells)


def test_a_smooth_gradient_detects_nothing() -> None:
    """An ordinary image must see no change at all -- a gradient changes as
    much mid-cell as it does at a boundary."""
    width = height = 128
    ramp = np.linspace(0, 255, width).astype(np.uint8)
    out = np.zeros((height, width, 4), dtype=np.uint8)
    out[..., :3] = ramp[None, :, None]
    out[..., 3] = 255
    assert tf.detect_pixel_grid(out)["scale"] is None


def test_the_divisor_rule_prefers_the_largest_passing_scale() -> None:
    """An image of 8px blocks is trivially also an image of 4px blocks, and
    taking the smallest would halve the size of every authored pixel."""
    art = _art(_cells(12, 12), 8)
    assert tf.detect_pixel_grid(art)["scale"] == 8


def test_the_output_values_are_a_subset_of_the_input() -> None:
    art = _art(_cells(9, 9), 6)
    back = tf.descale(art, 6, (0, 0))
    assert set(np.unique(back).tolist()) <= set(np.unique(art).tolist())


def test_the_source_array_is_untouched() -> None:
    art = _art(_cells(6, 6), 4)
    art.flags.writeable = False
    back = tf.descale(art, 4, (0, 0))
    back[:] = 7  # would raise if it aliased the read-only source
    assert art[0, 0, 0] != 7 or art[0, 0, 1] != 7


def test_a_scale_under_two_is_refused() -> None:
    art = _art(_cells(4, 4), 4)
    with pytest.raises(ValueError):
        tf.descale(art, 1, (0, 0))


def test_a_grid_that_leaves_no_cells_is_refused() -> None:
    tiny = np.zeros((3, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="no cells"):
        tf.descale(tiny, 8, (0, 0))


def test_a_tiny_image_measures_nothing() -> None:
    tiny = np.zeros((8, 8, 4), dtype=np.uint8)
    tiny[..., 3] = 255
    found = tf.detect_pixel_grid(tiny)
    # 8 // 4 = 2 cells at scale 4, below _MIN_CELLS, so nothing is measurable.
    assert found["scale"] is None


def test_descale_size_agrees_with_the_sampler() -> None:
    art = _art(_cells(7, 9), 5, phase=(2, 3))
    for scale, phase in ((5, (2, 3)), (4, (0, 0)), (6, (1, 1))):
        width, height = tf.descale_size((art.shape[1], art.shape[0]), scale, phase)
        assert tf.descale(art, scale, phase).shape[:2] == (height, width)


# --- the document operation --------------------------------------------------


def _doc_of(art: np.ndarray) -> Document:
    doc = Document.blank(art.shape[1], art.shape[0])
    doc.stack.active.pixels[:] = art
    return doc


def test_the_document_op_moves_layers_mask_and_index_planes_together() -> None:
    cells = _cells(8, 8)
    art = _art(cells, 4)
    doc = _doc_of(art)
    doc.add_layer(name="Second")
    doc.stack.active.pixels[:] = art
    doc.select_all()
    assert doc.mask is not None

    doc.descale_to_grid(4, (0, 0))

    assert doc.size == (8, 8)
    for layer in doc.stack:
        assert layer.pixels.shape[:2] == (8, 8)
        assert np.array_equal(layer.pixels, cells)
    assert doc.mask.mask.shape == (8, 8)


def test_one_undo_restores_everything_the_descale_moved() -> None:
    art = _art(_cells(8, 8), 4)
    doc = _doc_of(art)
    before = doc.stack.active.pixels.copy()
    doc.descale_to_grid(4, (0, 0))
    assert doc.undo() is True
    assert doc.size == (art.shape[1], art.shape[0])
    assert np.array_equal(doc.stack.active.pixels, before)


def test_an_indexed_document_descales_its_planes_exactly() -> None:
    """Pure element selection, so duplicate-slot identity survives it -- the
    plane is permuted, never re-inferred from colours."""
    cells = _cells(8, 8)
    art = _art(cells, 4)
    doc = _doc_of(art)
    palette = [(0, 0, 0, 0), *[tuple(int(v) for v in c) for c in
                               np.unique(cells.reshape(-1, 4), axis=0)]]
    doc.convert_to_indexed(palette)
    expected = tf.descale(doc.stack[0].indices, 4, (0, 0))

    doc.descale_to_grid(4, (0, 0))
    assert np.array_equal(doc.stack[0].indices, expected)
    doc.check_materialized()


def test_a_grid_that_leaves_no_cells_is_refused_by_the_document() -> None:
    doc = Document.blank(4, 4)
    with pytest.raises(ValueError):
        doc.descale_to_grid(16, (0, 0))
