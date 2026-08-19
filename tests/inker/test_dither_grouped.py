"""Grouped conversion: nearest's answer to nearest's own failure.

Nearest maps every colour to the absolutely-closest swatch, so a drawing with
fifteen mid-greys and a palette with two can land all fifteen on one of them.
Grouped clusters the drawing's *own* colours and assigns groups one-to-one, so
the darker greys go to the darker target whatever the absolute distances say.
These tests pin that behaviour and the two properties that make it safe to run
on a document: the map is flat (no texture) and the table is scope-wide.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import dither

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)


def _plane(values: list[int], width: int = 1) -> np.ndarray:
    """One column of grey pixels, one row per value."""
    out = np.zeros((len(values), width, 4), dtype=np.uint8)
    out[..., 3] = 255
    for row, value in enumerate(values):
        out[row, :, :3] = value
    return out


def test_fifteen_greys_onto_two_keeps_the_contrast() -> None:
    greys = list(range(100, 175, 5))  # fifteen values, all mid-grey
    plane = _plane(greys)
    out = dither.convert(plane, [BLACK, WHITE], "grouped")
    mapped = out[:, 0, 0].tolist()

    # Both targets are used -- which is the whole point; nearest sends all
    # fifteen to whichever of black and white is closer to the middle.
    assert set(mapped) == {0, 255}
    # And the map is monotone in luma: no dark grey ends up lighter than a
    # light one.
    assert mapped == sorted(mapped)
    nearest = dither.convert(plane, [BLACK, WHITE], "nearest")[:, 0, 0].tolist()
    assert len(set(nearest)) < len(set(mapped)) or nearest != mapped


def test_colour_families_do_not_cross_map() -> None:
    """Reds land on the red target, blues on the blue one."""
    reds = [(v, 0, 0) for v in range(180, 240, 10)]
    blues = [(0, 0, v) for v in range(180, 240, 10)]
    plane = np.zeros((len(reds) + len(blues), 1, 4), dtype=np.uint8)
    plane[..., 3] = 255
    for row, colour in enumerate([*reds, *blues]):
        plane[row, 0, :3] = colour

    out = dither.convert(plane, [(255, 0, 0, 255), (0, 0, 255, 255)], "grouped")
    got = [tuple(int(v) for v in out[row, 0, :3]) for row in range(out.shape[0])]
    assert set(got[: len(reds)]) == {(255, 0, 0)}
    assert set(got[len(reds):]) == {(0, 0, 255)}


def test_grouped_is_a_flat_map_with_no_texture() -> None:
    """Every pixel of one colour gets the same answer wherever it sits --
    which is what makes flat regions stay flat."""
    plane = np.zeros((16, 16, 4), dtype=np.uint8)
    plane[..., 3] = 255
    plane[..., :3] = 130
    out = dither.convert(plane, [BLACK, WHITE], "grouped")
    assert len(np.unique(out[..., :3].reshape(-1, 3), axis=0)) == 1


def test_count_weighting_decides_the_split() -> None:
    """A thousand antialiasing colours must not out-vote the fill behind them.

    Two clusters: one is a single colour covering most of the picture, the other
    a spread of colours covering a few pixels. Grouped must not spend both of a
    two-entry palette on the spread.
    """
    plane = np.zeros((40, 1, 4), dtype=np.uint8)
    plane[..., 3] = 255
    plane[:36, 0, :3] = 20            # the fill: one colour, 36 pixels
    for row in range(36, 40):         # the spread: four colours, one pixel each
        plane[row, 0, :3] = 200 + (row - 36) * 5

    out = dither.convert(plane, [BLACK, WHITE], "grouped")
    assert np.all(out[:36, 0, 0] == 0)
    assert np.all(out[36:, 0, 0] == 255)


def test_fewer_distinct_colours_than_swatches_map_one_to_one() -> None:
    plane = _plane([10, 120, 240])
    palette = [BLACK, (80, 80, 80, 255), (170, 170, 170, 255), WHITE]
    out = dither.convert(plane, palette, "grouped")
    mapped = [tuple(int(v) for v in out[row, 0, :3]) for row in range(3)]
    # Injective: three distinct colours, three distinct targets.
    assert len(set(mapped)) == 3
    assert mapped == sorted(mapped)


def test_one_distinct_colour_takes_its_nearest_swatch() -> None:
    plane = _plane([200])
    out = dither.convert(plane, [BLACK, (190, 190, 190, 255), WHITE], "grouped")
    assert tuple(int(v) for v in out[0, 0, :3]) == (190, 190, 190)


def test_the_table_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    plane = np.zeros((24, 24, 4), dtype=np.uint8)
    plane[..., 3] = 255
    plane[..., :3] = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    palette = [(r, g, b, 255) for r, g, b in [(0, 0, 0), (255, 0, 0), (0, 255, 0),
                                              (0, 0, 255), (255, 255, 255)]]
    first = dither.grouped_table([plane], palette)
    second = dither.grouped_table([plane], palette)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_palette_order_does_not_change_the_picture() -> None:
    """A tie-free fixture converts the same however the swatches are listed --
    the assignment is by distance, not by position."""
    plane = _plane(list(range(0, 256, 17)))
    palette = [BLACK, (85, 85, 85, 255), (170, 170, 170, 255), WHITE]
    straight = dither.convert(plane, palette, "grouped")
    shuffled = dither.convert(plane, list(reversed(palette)), "grouped")
    assert np.array_equal(straight, shuffled)


def test_a_shared_table_overrides_per_plane_grouping() -> None:
    """The central correctness requirement: two layers with disjoint grey ranges
    grouped separately would each spend the whole palette on their own range, so
    the same document would use one swatch for two quite different greys."""
    dark = _plane([10, 30, 50])
    light = _plane([200, 220, 240])

    alone_dark = dither.convert(dark, [BLACK, WHITE], "grouped")
    alone_light = dither.convert(light, [BLACK, WHITE], "grouped")
    # Grouped per plane, each range uses *both* targets.
    assert set(alone_dark[:, 0, 0].tolist()) == {0, 255}
    assert set(alone_light[:, 0, 0].tolist()) == {0, 255}

    table = dither.grouped_table([dark, light], [BLACK, WHITE])
    together_dark = dither.convert(dark, [BLACK, WHITE], "grouped", table=table)
    together_light = dither.convert(light, [BLACK, WHITE], "grouped", table=table)
    # Grouped together, the dark layer is dark and the light layer is light.
    assert set(together_dark[:, 0, 0].tolist()) == {0}
    assert set(together_light[:, 0, 0].tolist()) == {255}


def test_a_colour_missing_from_the_table_snaps_to_nearest() -> None:
    """``convert`` stays total for a stale table: no off-palette pixel escapes."""
    table = dither.grouped_table([_plane([10, 240])], [BLACK, WHITE])
    stranger = _plane([10, 240, 200])
    out = dither.convert(stranger, [BLACK, WHITE], "grouped", table=table)
    assert tuple(int(v) for v in out[2, 0, :3]) == (255, 255, 255)
    assert set(np.unique(out[..., :3]).tolist()) <= {0, 255}


def test_an_empty_table_still_snaps_every_pixel() -> None:
    empty = dither.grouped_table([], [BLACK, WHITE])
    out = dither.convert(_plane([10, 240]), [BLACK, WHITE], "grouped", table=empty)
    assert out[0, 0, 0] == 0
    assert out[1, 0, 0] == 255


@pytest.mark.parametrize("method", [m for m in dither.METHODS if m != "grouped"])
def test_a_table_with_another_method_is_refused(method: str) -> None:
    table = dither.grouped_table([_plane([10, 240])], [BLACK, WHITE])
    with pytest.raises(ValueError, match="grouped"):
        dither.convert(_plane([10]), [BLACK, WHITE], method, table=table)


def test_fully_transparent_pixels_ride_through_verbatim() -> None:
    plane = _plane([10, 240])
    plane[0, 0, 3] = 0
    plane[0, 0, :3] = (7, 8, 9)
    out = dither.convert(plane, [BLACK, WHITE], "grouped")
    assert tuple(int(v) for v in out[0, 0]) == (7, 8, 9, 0)


def test_grouped_is_the_identity_on_an_already_palettized_drawing() -> None:
    """Distance-zero pairs win the assignment first, so a drawing already on the
    palette maps every colour to itself."""
    palette = [BLACK, (85, 85, 85, 255), (170, 170, 170, 255), WHITE]
    plane = np.zeros((4, 4, 4), dtype=np.uint8)
    plane[..., 3] = 255
    for row, entry in enumerate(palette):
        plane[row, :, :3] = entry[:3]
    out = dither.convert(plane, palette, "grouped")
    assert np.array_equal(out, plane)


def test_duplicate_palette_entries_stay_distinct_slots() -> None:
    plane = _plane([10, 240])
    keys, targets = dither.grouped_table([plane], [BLACK, BLACK, WHITE])
    assert keys.size == 2
    # Both distinct colours are still mapped; the duplicate did not shrink the
    # table the assignment ran over.
    assert targets.shape == (2, 3)


def test_the_indexed_table_never_targets_the_transparent_slot() -> None:
    palette = [(0, 0, 0, 0), BLACK, WHITE]
    plane = _plane([10, 120, 240])
    keys, targets = dither.grouped_index_table(plane_list := [plane], palette)
    assert keys.size == 3
    assert plane_list  # the planes argument is an iterable of planes
    indices = dither.convert_indices(
        plane, palette, "grouped", table=(keys, targets)
    )
    assert 0 not in set(np.unique(indices).tolist())
