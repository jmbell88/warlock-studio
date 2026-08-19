"""A grid pack as a Tiled tileset -- and the round trip that proves it.

The payoff of the two modes being genuinely different: a grid pack *is* a
tileset, so it can be handed to Plotter or to Tiled with no conversion. The
strongest statement of that is the round trip below, which reads the ``.tsx``
back through the Plotter reader and checks the slicing lands on the sprites.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.packwright import tsxout
from warlock.studio.packwright.compose import compose
from warlock.studio.packwright.layout import PackSettings, layout
from warlock.studio.packwright.sources import Sprite
from warlock.studio.plotter import tsx as tsxmod


def _sprite(key: str, colour) -> Sprite:
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[...] = colour
    return Sprite(key=key, name=key, pixels=pixels)


def _pack(count: int = 6, **kw):
    sprites = [_sprite(f"s{i:02d}", (10 * i + 10, 40, 60, 255)) for i in range(count)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=2, **kw))
    return sprites, result, compose(sprites, result)


def test_a_grid_pack_round_trips_through_the_plotter_tsx_reader():
    _sprites, result, atlas = _pack()
    data = tsxout.grid_tsx(result, atlas, name="pack", image_name="atlas.png")
    back = tsxmod.read_tsx(data, atlas)

    assert back.name == "pack"
    assert (back.tile_w, back.tile_h) == (result.cell_w, result.cell_h)
    assert (back.margin, back.spacing) == (result.padding, result.padding)
    assert back.columns == result.columns
    assert back.rows == result.rows


def test_the_slicing_lands_exactly_on_the_packed_sprites():
    """The whole claim, checked in pixels: tile *n* of the tileset is the
    rectangle frame *n* of the layout was written into."""
    _sprites, result, atlas = _pack()
    tileset = tsxout.grid_tileset(result, atlas, name="pack")
    for index, frame in enumerate(result.frames):
        x, y, _w, _h = tileset.tile_rect(index)
        assert (x, y) == (frame.x, frame.y)
        # Every sprite here fills its cell, so the tile's top-left pixel is the
        # sprite's own colour rather than gutter.
        assert tuple(atlas[y, x]) == tuple(atlas[frame.y, frame.x])


def test_a_maxrects_pack_is_refused_rather_than_approximated():
    sprites = [_sprite(f"s{i}", (200, 0, 0, 255)) for i in range(4)]
    result = layout(sprites, PackSettings(mode="maxrects", power_of_two=False))
    atlas = compose(sprites, result)
    with pytest.raises(ValueError, match="only a grid pack is a tileset"):
        tsxout.grid_tsx(result, atlas, name="pack", image_name="atlas.png")


def test_the_image_name_is_what_the_tsx_points_at():
    _sprites, result, atlas = _pack()
    data = tsxout.grid_tsx(result, atlas, name="pack", image_name="sprites.png")
    assert tsxmod.tsx_source(data) == "sprites.png"


def test_it_writes_no_xml_of_its_own():
    """One ``.tsx`` writer in the repo, so a Packwright tileset and a Plotter
    one cannot become two dialects of the format."""
    import ast
    import inspect
    from pathlib import Path

    tree = ast.parse(Path(inspect.getfile(tsxout)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "xml" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            assert "xml" not in (node.module or "")


def test_two_writes_of_one_pack_are_byte_identical():
    _sprites, result, atlas = _pack()
    a = tsxout.grid_tsx(result, atlas, name="pack", image_name="atlas.png")
    b = tsxout.grid_tsx(result, atlas, name="pack", image_name="atlas.png")
    assert a == b


# --- power-of-two geometry -------------------------------------------------


def test_a_power_of_two_auto_grid_still_round_trips():
    """The auto column search re-derives after rounding precisely so this
    stays true: :func:`~.layout.grid_layout`'s pow2 branch and Tiled's own
    formula must never disagree about how many columns an image holds."""
    sprites = [_sprite(f"s{i:02d}", (10 * i + 10, 40, 60, 255)) for i in range(9)]
    result = layout(sprites, PackSettings(padding=2, power_of_two=True))
    atlas = compose(sprites, result)
    data = tsxout.grid_tsx(result, atlas, name="pack", image_name="atlas.png")
    back = tsxmod.read_tsx(data, atlas)
    assert back.columns == result.columns
    assert back.rows == result.rows
    tileset = tsxout.grid_tileset(result, atlas, name="pack")
    for index, frame in enumerate(result.frames):
        x, y, _w, _h = tileset.tile_rect(index)
        assert (x, y) == (frame.x, frame.y)


def test_an_explicit_columns_grid_that_pow2_rounding_widens_refuses_a_tsx():
    """Honouring an explicit column count means the grid does *not* follow
    power-of-two rounding the way the auto search does -- so the rounded
    image can carry room Tiled would read as extra columns/rows the pack
    never placed. Writing a ``.tsx`` from that would slice tiles onto the
    wrong sprites past the first row, so it is refused by name instead."""
    sprites = [_sprite(f"s{i:02d}", (10 * i + 10, 40, 60, 255)) for i in range(4)]
    result = layout(sprites, PackSettings(padding=2, power_of_two=True, columns=2))
    atlas = compose(sprites, result)
    with pytest.raises(ValueError, match="power-of-two rounding leaves Tiled reading"):
        tsxout.grid_tileset(result, atlas, name="pack")


def test_an_explicit_columns_grid_that_pow2_rounding_does_not_widen_still_exports():
    """The refusal is about the mismatch, not about ``columns`` or
    ``power_of_two`` being set at all -- when the pre-rounding width already
    sits on a power of two, rounding buys no spare room, the two stay in
    agreement, and the export proceeds."""
    sprites = [_sprite(f"s{i:02d}", (10 * i + 10, 40, 60, 255)) for i in range(9)]
    result = layout(sprites, PackSettings(padding=2, power_of_two=True, columns=3))
    atlas = compose(sprites, result)
    tileset = tsxout.grid_tileset(result, atlas, name="pack")
    assert (tileset.columns, tileset.rows) == (result.columns, result.rows)
