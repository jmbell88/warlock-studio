"""Where sprites land: settings validation, the grid, and the size search.

One file for ``layout.py`` rather than one per mode: ``PackSettings``, the grid
and the MaxRects *size search* all live there, and the grid's geometry is only
interesting next to the settings that produce it. (The basename is
``test_pack_layout`` and not ``test_layout`` because the suite has no
``__init__.py`` files, so two test modules of one name collide at collection --
``tests/test_layout.py`` already exists.)

The grid's geometry is the load-bearing part. It is spelled as Tiled's margin
and spacing -- outer border and gutter both one ``padding`` -- because that is
what lets :mod:`~warlock.studio.packwright.tsxout` hand the numbers straight to
``tilegrid.tileset.Tileset`` instead of approximating them.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.pipelines import sheet as sheetlib
from warlock.studio.packwright import layout as lay
from warlock.studio.packwright.layout import PackSettings, next_pot
from warlock.studio.packwright.sources import Sprite


def _sprite(key: str, w: int, h: int, *, blank: bool = False) -> Sprite:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    if not blank:
        pixels[..., :] = (10, 20, 30, 255)
    return Sprite(key=key, name=key, pixels=pixels)


def _sprites(count: int, w: int = 10, h: int = 10) -> list[Sprite]:
    return [_sprite(f"s{i:02d}", w, h) for i in range(count)]


# --- settings -----------------------------------------------------------------


def test_padding_must_be_at_least_twice_extrude():
    """Two neighbours extrude into one gutter, so it needs twice the room.
    Getting this wrong produces bleed that only shows on a GPU with filtering
    on, at some zoom levels."""
    PackSettings(padding=4, extrude=2)
    with pytest.raises(ValueError, match="twice extrude"):
        PackSettings(padding=3, extrude=2)


def test_max_size_is_ceilinged_by_the_sprite_sheet_pipelines_limit():
    """Imported, not restated: "how big may an atlas be before an engine
    refuses it" is one question and it already has an answer."""
    PackSettings(max_size=sheetlib.MAX_ATLAS_PX)
    with pytest.raises(ValueError, match=str(sheetlib.MAX_ATLAS_PX)):
        PackSettings(max_size=sheetlib.MAX_ATLAS_PX * 2)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be"):
        PackSettings(mode="skyline")


def test_negative_settings_are_refused():
    with pytest.raises(ValueError):
        PackSettings(padding=-1)
    with pytest.raises(ValueError):
        PackSettings(max_size=0)


def test_a_gutter_wider_than_an_atlas_is_refused():
    """The sliders offer 16 and 8, so this only ever refuses a hand-edited
    manifest -- which is exactly the input with no other door in front of it.
    Each message names its own field, because a single "padding and extrude"
    sentence sends the user to look at the one that was fine."""
    PackSettings(padding=lay.MAX_PADDING, extrude=lay.MAX_PADDING // 2)
    with pytest.raises(ValueError, match="padding must be at most"):
        PackSettings(padding=lay.MAX_PADDING + 1)
    with pytest.raises(ValueError, match="extrude must be at most"):
        PackSettings(padding=lay.MAX_PADDING, extrude=lay.MAX_PADDING + 1)


def test_more_sprites_than_one_atlas_will_take_are_refused(monkeypatch):
    """In ``_measure``, which is what makes it one door: both packers and every
    caller of either come through it."""
    monkeypatch.setattr(lay, "MAX_SPRITES", 3)
    lay.layout(_sprites(3), PackSettings())
    with pytest.raises(ValueError, match="is the most one"):
        lay.layout(_sprites(4), PackSettings())
    with pytest.raises(ValueError, match="is the most one"):
        lay.layout(_sprites(4), PackSettings(mode="maxrects"))


def test_next_pot_is_the_smallest_power_of_two_at_or_above():
    assert [next_pot(n) for n in (0, 1, 2, 3, 5, 64, 65)] == [1, 1, 2, 4, 8, 64, 128]


# --- grid ---------------------------------------------------------------------


def test_a_grid_uses_one_cell_the_size_of_the_largest_trimmed_sprite():
    sprites = [_sprite("a", 10, 6), _sprite("b", 4, 20), _sprite("c", 8, 8)]
    result = lay.grid_layout(sprites, PackSettings(power_of_two=False, padding=2))
    assert (result.cell_w, result.cell_h) == (10, 20)


def test_the_grid_geometry_is_tileds_margin_and_spacing():
    """``x = padding + column * (cell + padding)`` -- which is exactly
    ``margin == spacing == padding`` in a ``.tsx``."""
    pad = 3
    result = lay.grid_layout(_sprites(4, 10, 10), PackSettings(padding=pad, power_of_two=False))
    step = result.cell_w + pad
    for index, frame in enumerate(result.frames):
        assert frame.x == pad + (index % result.columns) * step
        assert frame.y == pad + (index // result.columns) * (result.cell_h + pad)
    assert result.width == pad + result.columns * step


def test_the_grid_is_row_major_in_key_order():
    """The order is part of the layout, so it comes from the key rather than
    from a document list the user reorders."""
    sprites = [_sprite("c", 8, 8), _sprite("a", 8, 8), _sprite("b", 8, 8)]
    result = lay.grid_layout(sprites, PackSettings(power_of_two=False))
    assert [frame.key for frame in result.frames] == ["a", "b", "c"]


def test_columns_and_rows_describe_the_grid_a_tsx_reader_would_derive():
    """Not "cells that happen to be occupied". A power-of-two rounding that
    bought room for another column is a column the .tsx and this layout have to
    agree about; trailing empty cells are ordinary in a tileset, a disagreement
    about the shape of the grid is not."""
    result = lay.grid_layout(_sprites(9, 10, 10), PackSettings(padding=2, power_of_two=True))
    step_w, step_h = result.cell_w + result.padding, result.cell_h + result.padding
    assert result.columns == (result.width - result.padding) // step_w
    assert result.rows == (result.height - result.padding) // step_h
    assert result.columns * result.rows >= len(result.frames)


def test_a_power_of_two_grid_is_a_power_of_two():
    result = lay.grid_layout(_sprites(7, 13, 11), PackSettings(power_of_two=True))
    assert next_pot(result.width) == result.width
    assert next_pot(result.height) == result.height


def test_a_grid_too_large_for_the_ceiling_is_refused():
    with pytest.raises(ValueError, match="limit is"):
        lay.grid_layout(_sprites(40, 200, 200), PackSettings(max_size=256))


# --- explicit columns -----------------------------------------------------


def test_an_explicit_columns_count_drives_the_grid_instead_of_the_isqrt_search():
    result = lay.grid_layout(
        _sprites(9, 10, 10), PackSettings(padding=2, power_of_two=False, columns=2)
    )
    assert result.columns == 2
    assert result.rows == 5  # ceil(9 / 2)
    step = 10 + 2
    assert result.width == 2 + 2 * step
    assert result.height == 2 + 5 * step
    for index, frame in enumerate(result.frames):
        assert frame.x == 2 + (index % 2) * step
        assert frame.y == 2 + (index // 2) * step


def test_columns_must_be_at_least_one():
    with pytest.raises(ValueError, match="columns must be at least 1"):
        PackSettings(columns=0)
    with pytest.raises(ValueError, match="columns must be at least 1"):
        PackSettings(columns=-1)


def test_columns_on_a_maxrects_pack_is_refused_rather_than_ignored():
    """MaxRects has no uniform cell for a column count to describe -- silently
    dropping it would look like the setting worked."""
    with pytest.raises(ValueError, match="columns only applies to a grid pack"):
        PackSettings(mode="maxrects", columns=4)


def test_explicit_columns_is_not_widened_by_power_of_two_rounding():
    """The point of setting columns is exact control: a rounding-bought column
    is not a column the user asked for. Contrast with the auto search, which
    *does* re-derive (below) -- this is the "backwards for tileset authoring"
    case the setting exists to fix."""
    result = lay.grid_layout(
        _sprites(9, 10, 10), PackSettings(padding=2, power_of_two=True, columns=3)
    )
    assert result.columns == 3
    assert result.rows == 3
    assert lay.next_pot(result.width) == result.width
    assert lay.next_pot(result.height) == result.height


# --- json schema -----------------------------------------------------------


def test_json_schema_defaults_to_array():
    assert PackSettings().json_schema == "array"


def test_an_unknown_json_schema_is_refused():
    with pytest.raises(ValueError, match="json_schema must be one of"):
        PackSettings(json_schema="xml")


def test_json_schema_is_a_settings_field_a_repack_does_not_disturb():
    """It travels with ``.set_settings`` exactly like ``mode``/``trim`` --
    proven here by the dataclass round trip ``document.py`` uses."""
    from dataclasses import replace

    settings = PackSettings()
    after = replace(settings, json_schema="hash")
    assert after.json_schema == "hash"
    assert after.mode == settings.mode  # untouched fields survive replace()


def test_auto_columns_still_re_derives_from_the_rounded_width():
    """The existing, tsx-safe behaviour: unset ``columns`` is unaffected by the
    explicit-columns fix and stays byte-for-byte what it always was."""
    result = lay.grid_layout(_sprites(9, 10, 10), PackSettings(padding=2, power_of_two=True))
    step = 10 + 2
    assert result.columns == (result.width - 2) // step
    assert result.rows == (result.height - 2) // step
    assert result.columns == 5 and result.rows == 2


# --- trim inside a layout -----------------------------------------------------


def test_a_frame_records_where_its_trim_came_from():
    pixels = np.zeros((20, 20, 4), dtype=np.uint8)
    pixels[5:9, 3:11] = (255, 0, 0, 255)
    result = lay.grid_layout(
        [Sprite(key="a", name="a", pixels=pixels)], PackSettings(power_of_two=False)
    )
    frame = result.frames[0]
    assert frame.trim == (3, 5, 8, 4)
    assert (frame.w, frame.h) == (8, 4)
    assert (frame.source_w, frame.source_h) == (20, 20)
    assert frame.trimmed is True


def test_a_sprite_that_fills_its_canvas_is_not_reported_as_trimmed():
    """Saying otherwise makes every consumer do offset arithmetic for a zero
    offset."""
    result = lay.grid_layout([_sprite("a", 8, 8)], PackSettings(power_of_two=False))
    assert result.frames[0].trimmed is False


def test_a_blank_sprite_still_gets_a_frame():
    result = lay.grid_layout(
        [_sprite("a", 8, 8), _sprite("b", 8, 8, blank=True)],
        PackSettings(power_of_two=False),
    )
    assert len(result.frames) == 2
    blank = result.frame("b")
    assert blank is not None and blank.empty is True and (blank.w, blank.h) == (1, 1)


# --- maxrects size search -----------------------------------------------------


def test_the_size_search_starts_square_and_doubles_the_shorter_side():
    """Stated in the docstring, pinned here: doubling the shorter side keeps
    the atlas near-square, which is what a GPU wants and what keeps the wasted
    corner small."""
    sizes = lay._candidate_sizes(area=60 * 60, floor_w=8, floor_h=8, limit=512)
    assert sizes[:5] == [(64, 64), (128, 64), (128, 128), (256, 128), (256, 256)]


def test_the_size_search_never_starts_below_the_biggest_item():
    sizes = lay._candidate_sizes(area=100, floor_w=300, floor_h=10, limit=1024)
    assert sizes[0] == (512, 512)


def test_a_maxrects_pack_grows_until_it_fits():
    result = lay.maxrects_layout(_sprites(64, 30, 30), PackSettings(mode="maxrects"))
    assert max(frame.x + frame.w for frame in result.frames) <= result.width
    assert len(result.frames) == 64


def test_a_maxrects_pack_that_cannot_fit_raises_rather_than_truncating():
    with pytest.raises(ValueError, match="do not fit"):
        lay.maxrects_layout(
            _sprites(50, 100, 100), PackSettings(mode="maxrects", max_size=128)
        )


def test_every_sprite_has_padding_on_all_four_sides():
    """Including against the edge of the atlas, which is the case an un-offset
    MaxRects pack gets wrong -- and which extrude needs to be true or it writes
    outside the array."""
    pad = 4
    result = lay.maxrects_layout(
        _sprites(20, 12, 9), PackSettings(mode="maxrects", padding=pad, extrude=2)
    )
    for frame in result.frames:
        assert frame.x >= pad and frame.y >= pad
        assert frame.x + frame.w + pad <= result.width
        assert frame.y + frame.h + pad <= result.height


def test_a_non_power_of_two_pack_shrinks_to_what_it_used():
    tight = lay.maxrects_layout(
        _sprites(5, 10, 10), PackSettings(mode="maxrects", power_of_two=False)
    )
    padded = lay.maxrects_layout(
        _sprites(5, 10, 10), PackSettings(mode="maxrects", power_of_two=True)
    )
    assert tight.width <= padded.width and tight.height <= padded.height
    assert max(f.x + f.w for f in tight.frames) + tight.padding == tight.width


# --- dispatch -----------------------------------------------------------------


def test_layout_dispatches_on_the_settings_mode():
    sprites = _sprites(6)
    assert lay.layout(sprites, PackSettings(mode="grid")).is_grid
    assert not lay.layout(sprites, PackSettings(mode="maxrects")).is_grid


def test_packing_nothing_is_refused_in_both_modes():
    for mode in ("grid", "maxrects"):
        with pytest.raises(ValueError, match="nothing to pack"):
            lay.layout([], PackSettings(mode=mode))


def test_the_same_input_always_produces_the_same_layout():
    """The property that makes a re-export of an unchanged document
    byte-identical, asserted at the level a caller actually uses."""
    sprites = _sprites(24, 17, 13)
    for mode in ("grid", "maxrects"):
        settings = PackSettings(mode=mode)
        assert lay.layout(sprites, settings) == lay.layout(list(reversed(sprites)), settings)
