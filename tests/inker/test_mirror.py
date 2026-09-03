"""The arithmetic under mirror-assisted cleanup and the propagation mark."""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import mirror


def _sprite(size: int = 16) -> np.ndarray:
    """A symmetric figure: a body column, two arms, a head. Uint8 RGBA."""
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[2:6, 6:10] = (200, 150, 120, 255)  # head
    out[6:13, 7:9] = (40, 60, 200, 255)  # body
    out[7:9, 3:7] = (40, 60, 200, 255)  # left arm
    out[7:9, 9:13] = (40, 60, 200, 255)  # right arm
    return out


def test_the_face_box_is_the_top_of_the_alpha_bbox_at_full_width():
    box = mirror.face_box(_sprite(), 0.30)
    # bbox rows 2..12 (11 tall) -> round(3.3) = 3 rows; cols 3..12.
    assert box == (3, 2, 13, 5)


def test_an_empty_plane_or_a_zero_fraction_has_no_face():
    assert mirror.face_box(np.zeros((8, 8, 4), np.uint8)) is None
    assert mirror.face_box(_sprite(), 0.0) is None


def test_the_face_weight_is_open_everywhere_but_the_box():
    weight = mirror.face_weight((16, 16), (3, 2, 13, 5))
    assert weight is not None
    assert weight.dtype == np.uint8
    assert weight[2:5, 3:13].max() == 0
    assert weight[5:, :].min() == 255
    assert weight[:2, :].min() == 255
    assert mirror.face_weight((16, 16), None) is None


def test_mirrored_is_an_exact_horizontal_flip():
    sprite = _sprite()
    sprite[7, 3] = (255, 0, 0, 255)
    flipped = mirror.mirrored(sprite)
    assert tuple(flipped[7, 12]) == (255, 0, 0, 255)
    assert np.array_equal(mirror.mirrored(flipped), sprite)


def test_diff_ignores_colour_under_zero_alpha():
    a = np.zeros((4, 4, 4), np.uint8)
    b = np.zeros((4, 4, 4), np.uint8)
    b[0, 0] = (255, 255, 255, 0)
    assert not mirror.diff(a, b).any()
    b[1, 1] = (255, 255, 255, 255)
    assert mirror.diff(a, b).sum() == 1


def test_a_symmetric_sprite_with_an_asymmetric_face_differs_only_inside_the_box():
    west = _sprite()
    west[3, 7] = (0, 0, 0, 255)  # one eye, left of centre
    east = _sprite()
    east[3, 7] = (0, 0, 0, 255)  # drawn independently, the eye stays left
    weight = mirror.face_weight(west.shape[:2], mirror.face_box(west))
    outside, inside, changed = mirror.diff_report(west, east, weight)
    assert outside == 0
    assert inside == 2
    assert changed.sum() == 2


def test_a_fix_outside_the_face_shows_up_as_the_pixels_it_would_change():
    west = _sprite()
    west[10, 4] = (40, 60, 200, 255)  # a longer left arm on the west view
    east = _sprite()
    weight = mirror.face_weight(west.shape[:2], mirror.face_box(west))
    outside, inside, _changed = mirror.diff_report(west, east, weight)
    assert (outside, inside) == (1, 0)
    outside, inside, _changed = mirror.diff_report(west, east, None)
    assert (outside, inside) == (1, 0)


def test_the_mark_is_none_when_nothing_changed_and_the_changed_pixels_otherwise():
    before = _sprite()
    assert mirror.changed_weight(before, before.copy()) is None
    now = before.copy()
    now[0, 0] = (1, 2, 3, 255)
    now[15, 15] = (1, 2, 3, 255)
    weight = mirror.changed_weight(before, now)
    assert weight is not None
    assert weight.dtype == np.uint8
    assert weight.sum() == 2 * 255
    assert weight[0, 0] == 255 and weight[15, 15] == 255


def test_translate_within_moves_the_selection_and_clears_where_it_was():
    plane = np.zeros((6, 6, 4), np.uint8)
    plane[1, 1] = (9, 9, 9, 255)
    plane[4, 4] = (7, 7, 7, 255)
    weight = np.zeros((6, 6), np.uint8)
    weight[1, 1] = 255
    out = mirror.translate_within(plane, weight, 2, 1)
    assert tuple(out[2, 3]) == (9, 9, 9, 255)
    assert tuple(out[1, 1]) == (0, 0, 0, 0)
    assert tuple(out[4, 4]) == (7, 7, 7, 255)


def test_translate_within_clips_rather_than_wrapping():
    plane = np.zeros((4, 4, 4), np.uint8)
    plane[0, 3] = (9, 9, 9, 255)
    weight = np.zeros((4, 4), np.uint8)
    weight[0, 3] = 255
    out = mirror.translate_within(plane, weight, 1, 0)
    assert out[..., 3].sum() == 0


def test_shapes_are_checked_by_name():
    with pytest.raises(ValueError):
        mirror.diff(np.zeros((4, 4, 4), np.uint8), np.zeros((5, 5, 4), np.uint8))
    with pytest.raises(ValueError):
        mirror.mirrored(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        mirror.translate_within(np.zeros((4, 4, 4), np.uint8), np.zeros((3, 3), np.uint8), 1, 0)
