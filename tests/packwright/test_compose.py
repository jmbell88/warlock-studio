"""Painting the atlas, and the gutter.

Extrude is the interesting half: it exists so a filtered texture sampling just
past a sprite's edge finds that sprite's own colour, and it is wrong in a way
that only shows on a GPU at some zoom levels -- which is exactly why it is
asserted pixel by pixel here.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.packwright import compose
from warlock.studio.packwright.layout import PackSettings, layout
from warlock.studio.packwright.sources import Sprite

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _solid(key: str, w: int, h: int, colour) -> Sprite:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[...] = colour
    return Sprite(key=key, name=key, pixels=pixels)


def test_each_sprite_lands_exactly_where_its_frame_says():
    sprites = [_solid("a", 6, 4, RED), _solid("b", 6, 4, BLUE)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=2))
    atlas = compose.compose(sprites, result)
    assert atlas.shape == (result.height, result.width, 4)
    for frame, colour in ((result.frame("a"), RED), (result.frame("b"), BLUE)):
        block = atlas[frame.y : frame.y + frame.h, frame.x : frame.x + frame.w]
        assert (block == np.array(colour, np.uint8)).all()


def test_only_the_trimmed_region_is_pasted():
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    pixels[4:6, 3:7] = RED
    sprites = [Sprite(key="a", name="a", pixels=pixels)]
    result = layout(sprites, PackSettings(power_of_two=False))
    atlas = compose.compose(sprites, result)
    frame = result.frames[0]
    assert (frame.w, frame.h) == (4, 2)
    assert (atlas[frame.y : frame.y + 2, frame.x : frame.x + 4] == np.array(RED, np.uint8)).all()
    assert int(atlas[..., 3].sum()) == 4 * 2 * 255


def test_the_gutter_is_transparent_without_extrude():
    sprites = [_solid("a", 4, 4, RED), _solid("b", 4, 4, BLUE)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=2, extrude=0))
    atlas = compose.compose(sprites, result)
    assert int(atlas[..., 3].sum()) == 2 * 4 * 4 * 255


def test_extrude_replicates_the_border_outward_including_the_corners():
    """The corners come free by extruding vertically *after* the side columns
    are written, which is why the order in ``_extrude`` matters."""
    sprites = [_solid("a", 4, 4, RED)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=4, extrude=2))
    atlas = compose.compose(sprites, result)
    frame = result.frames[0]
    grown = atlas[frame.y - 2 : frame.y + frame.h + 2, frame.x - 2 : frame.x + frame.w + 2]
    assert grown.shape == (8, 8, 4)
    assert (grown == np.array(RED, np.uint8)).all()


def test_extrude_never_crosses_into_a_neighbour():
    """The whole point of ``padding >= extrude * 2``: two sprites extruding
    into one gutter must not meet."""
    sprites = [_solid(f"s{i}", 8, 8, RED if i % 2 else BLUE) for i in range(9)]
    result = layout(sprites, PackSettings(power_of_two=False, padding=4, extrude=2))
    atlas = compose.compose(sprites, result)
    for frame in result.frames:
        colour = np.array(RED if int(frame.key[1]) % 2 else BLUE, np.uint8)
        grown = atlas[frame.y - 2 : frame.y + frame.h + 2, frame.x - 2 : frame.x + frame.w + 2]
        assert (grown == colour).all(), f"{frame.key} was reached by a neighbour"


def test_extrude_stays_inside_the_atlas_in_both_modes():
    """A sprite against the edge of the atlas has to have room too, which is
    the case an un-offset MaxRects pack gets wrong -- and getting it wrong here
    is an out-of-bounds write rather than a cosmetic artefact."""
    sprites = [_solid(f"s{i:02d}", 7, 5, RED) for i in range(12)]
    for mode in ("grid", "maxrects"):
        settings = PackSettings(mode=mode, padding=4, extrude=2, power_of_two=False)
        result = layout(sprites, settings)
        atlas = compose.compose(sprites, result)  # must not raise
        assert atlas.shape[:2] == (result.height, result.width)


def test_a_blank_sprite_composites_as_nothing():
    blank = Sprite(key="b", name="b", pixels=np.zeros((6, 6, 4), np.uint8))
    sprites = [_solid("a", 4, 4, RED), blank]
    result = layout(sprites, PackSettings(power_of_two=False))
    atlas = compose.compose(sprites, result)
    assert int(atlas[..., 3].sum()) == 4 * 4 * 255


def test_a_frame_naming_a_missing_sprite_raises():
    """An atlas with an invisible gap looks like an art problem and sends the
    user looking in the wrong place -- the rule ``sheet.pack`` already follows
    for a missing rendered frame."""
    sprites = [_solid("a", 4, 4, RED)]
    result = layout(sprites, PackSettings(power_of_two=False))
    with pytest.raises(ValueError, match="no sprite for packed frame"):
        compose.compose([], result)


def test_png_bytes_round_trips_the_atlas():
    from PIL import Image

    sprites = [_solid("a", 4, 4, RED)]
    result = layout(sprites, PackSettings(power_of_two=False))
    atlas = compose.compose(sprites, result)
    import io

    decoded = np.asarray(Image.open(io.BytesIO(compose.png_bytes(atlas))).convert("RGBA"))
    assert np.array_equal(decoded, atlas)


def test_two_composites_of_one_layout_are_identical():
    sprites = [_solid(f"s{i}", 5, 5, RED) for i in range(6)]
    result = layout(sprites, PackSettings())
    assert np.array_equal(compose.compose(sprites, result), compose.compose(sprites, result))
