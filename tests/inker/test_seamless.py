"""The seamless-tile toolkit: the offset that moves the seam, and the number.

Tiled mode already drew the 3x3 wrap; what it did not give was a way to *get
at* the seam or any statement about how bad it is. These two together are the
whole loop -- roll the seam into the middle, paint over it, watch the number
come down.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import tiling
from warlock.studio.inker.document import Document

# --- offset_layer ------------------------------------------------------------


def _speckled(size: int = 8, seed: int = 1) -> Document:
    doc = Document.blank(size, size)
    rng = np.random.default_rng(seed)
    doc.stack.active.pixels[..., :3] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    doc.stack.active.pixels[..., 3] = 255
    return doc


def test_four_half_wraps_are_the_identity() -> None:
    """An exact permutation: no pixel is invented and none is lost."""
    doc = _speckled()
    before = doc.stack.active.pixels.copy()
    for _ in range(4):
        assert doc.offset_layer(4, 4) is True
    assert np.array_equal(doc.stack.active.pixels, before)


def test_two_half_wraps_on_even_dimensions_are_the_identity() -> None:
    doc = _speckled()
    before = doc.stack.active.pixels.copy()
    doc.offset_layer(4, 4)
    assert not np.array_equal(doc.stack.active.pixels, before)
    doc.offset_layer(4, 4)
    assert np.array_equal(doc.stack.active.pixels, before)


def test_an_index_plane_permutes_exactly() -> None:
    """Two slots holding one colour stay two slots: the plane is moved, never
    re-inferred from the colours it materialises to."""
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[..., 3] = 255
    doc.stack.active.pixels[0, 0, :3] = (200, 20, 20)
    doc.stack.active.pixels[0, 1, :3] = (20, 200, 20)
    palette = [(0, 0, 0, 255), (200, 20, 20, 255), (20, 200, 20, 255), (200, 20, 20, 255)]
    doc.convert_to_indexed(palette, transparent=0)
    before = doc.stack[0].indices.copy()

    doc.offset_layer(3, 5)
    assert np.array_equal(doc.stack[0].indices, np.roll(before, (5, 3), axis=(0, 1)))
    doc.check_materialized()


def test_a_zero_offset_pushes_no_step() -> None:
    doc = _speckled()
    depth = len(doc.history)
    assert doc.offset_layer(0, 0) is False
    assert doc.offset_layer(8, 8) is False, "a whole canvas is a zero offset"
    assert len(doc.history) == depth


def test_an_offset_is_one_undo_step() -> None:
    doc = _speckled()
    before = doc.stack.active.pixels.copy()
    assert doc.offset_layer(3, 5) is True
    assert doc.undo() is True
    assert np.array_equal(doc.stack.active.pixels, before)


def test_a_locked_layer_refuses_the_offset() -> None:
    doc = _speckled()
    before = doc.stack.active.pixels.copy()
    doc.stack.active.locked = True
    assert doc.offset_layer(4, 4) is False
    assert np.array_equal(doc.stack.active.pixels, before)


def test_the_selection_mask_does_not_travel_with_the_layer() -> None:
    """This offsets the *layer*, not the selection. A marquee that travelled
    with it would make "offset, then paint inside the selection" mean something
    nobody asked for."""
    doc = _speckled()
    doc.select_all()
    before = doc.mask.mask.copy()
    doc.offset_layer(4, 4)
    assert np.array_equal(doc.mask.mask, before)


# --- seam_ratio --------------------------------------------------------------


def test_wrap_constructed_noise_measures_about_one() -> None:
    """Noise built to wrap has no join at all: its edge difference is just
    another adjacent pair, so the ratio sits around 1."""
    rng = np.random.default_rng(4)
    tile = np.zeros((64, 64, 4), dtype=np.uint8)
    tile[..., :3] = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    tile[..., 3] = 255
    horizontal, vertical = tiling.seam_ratio(tile)
    assert 0.5 < horizontal < 2.0
    assert 0.5 < vertical < 2.0


def test_two_flat_halves_with_a_hard_join_measure_high() -> None:
    tile = np.zeros((64, 64, 4), dtype=np.uint8)
    tile[..., 3] = 255
    # A gentle ramp so the interior is nonzero, then a hard step at the wrap.
    tile[:, :, 0] = np.arange(64, dtype=np.uint8)[None, :]
    tile[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :]
    tile[:, :, 2] = np.arange(64, dtype=np.uint8)[None, :]
    horizontal, _vertical = tiling.seam_ratio(tile)
    assert horizontal > tiling.SEAM_MAX


def test_a_flat_image_is_zero() -> None:
    """An image that is one colour tiles perfectly, and zero is the honest
    answer -- the ``inf`` arm cannot fire while the statistic is a mean."""
    flat = np.zeros((32, 32, 4), dtype=np.uint8)
    flat[..., :3] = 120
    flat[..., 3] = 255
    assert tiling.seam_ratio(flat) == (0.0, 0.0)


def test_a_vertical_seam_is_reported_on_the_vertical_axis() -> None:
    tile = np.zeros((64, 64, 4), dtype=np.uint8)
    tile[..., 3] = 255
    ramp = np.arange(64, dtype=np.uint8)
    tile[:, :, :3] = ramp[:, None, None]
    horizontal, vertical = tiling.seam_ratio(tile)
    assert vertical > tiling.SEAM_MAX
    assert horizontal < 1.0


def test_a_tiny_image_measures_nothing() -> None:
    tiny = np.zeros((4, 4, 4), dtype=np.uint8)
    tiny[..., 3] = 255
    assert tiling.seam_ratio(tiny) == (0.0, 0.0)


def test_a_non_image_array_is_refused() -> None:
    with pytest.raises(ValueError):
        tiling.seam_ratio(np.zeros((8, 8), dtype=np.uint8))


def test_the_threshold_carries_its_citation() -> None:
    """A copy at a second surface moves nothing: the same measurement document
    governs both, and the value must stay in step with it."""
    from warlock.pipelines import seam

    assert tiling.SEAM_MAX == seam.SEAM_MAX


def test_offset_autovivifies_an_empty_animated_cel():
    """The defect: ``offset_layer`` skipped ``_ensure_active_cel``, so Wrap
    1/2 on an empty animated cel wrote into the shared read-only placeholder
    plane and raised ``ValueError`` mid-gesture. Autovivified like every other
    write path, the pending cel rides the same undo step as the roll."""
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[0:2, 0:2] = (9, 8, 7, 255)
    doc.invalidate_all()
    anim = doc.ensure_animation()
    anim.tracks[0].continuous = True  # the new cel starts from frame 0's art
    art = doc.stack.active.pixels.copy()
    doc.add_frame()
    doc.set_current_frame(1)
    head = doc.history.head

    assert doc.offset_layer(4, 4) is True

    assert np.array_equal(
        doc.stack.active.pixels, np.roll(art, (4, 4), axis=(0, 1))
    )
    assert doc.history.head == head + 1, "one step: the cel and the roll together"
    doc.undo()
    track, frame = anim.tracks[0], anim.frames[1]
    assert (track.uid, frame.uid) not in anim.cels, "the undo took the cel back out"
