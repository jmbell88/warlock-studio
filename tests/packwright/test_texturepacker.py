"""The interchange sidecar.

Two things are being defended. The schema is the one every 2D engine already
has a loader for -- an atlas nobody can read is not an export -- and it is
**not** ``pipelines.sheet.sidecar``, which stays Warlock's own versioned format
with exactly one writer.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np

from warlock.studio.packwright import texturepacker
from warlock.studio.packwright.layout import PackSettings, layout
from warlock.studio.packwright.sources import Sprite


def _sprite(key: str, w: int, h: int, *, mark=None) -> Sprite:
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    if mark is None:
        pixels[...] = (255, 0, 0, 255)
    else:
        pixels[mark[1] : mark[3], mark[0] : mark[2]] = (255, 0, 0, 255)
    return Sprite(key=key, name=key, pixels=pixels)


def _layout(sprites=None):
    sprites = sprites or [_sprite("a", 8, 6), _sprite("b", 4, 4)]
    return sprites, layout(sprites, PackSettings(power_of_two=False, padding=2))


def test_the_schema_is_the_json_array_shape_engines_expect():
    _sprites, result = _layout()
    payload = texturepacker.tp_json(result, image_name="atlas.png")

    assert set(payload) == {"frames", "meta"}
    assert isinstance(payload["frames"], list)
    entry = payload["frames"][0]
    assert set(entry) == {
        "filename",
        "frame",
        "rotated",
        "trimmed",
        "spriteSourceSize",
        "sourceSize",
        "pivot",
    }
    assert set(entry["frame"]) == {"x", "y", "w", "h"}
    assert set(entry["sourceSize"]) == {"w", "h"}
    assert payload["meta"]["image"] == "atlas.png"
    assert payload["meta"]["format"] == "RGBA8888"
    assert payload["meta"]["size"] == {"w": result.width, "h": result.height}
    # A string, which is what TexturePacker writes and several loaders parse.
    assert payload["meta"]["scale"] == "1.0"


def test_a_frame_rectangle_is_the_one_the_layout_placed():
    sprites, result = _layout()
    payload = texturepacker.tp_json(result, image_name="atlas.png")
    by_name = {entry["filename"]: entry for entry in payload["frames"]}
    for frame in result.frames:
        entry = by_name[f"{frame.name}.png"]
        assert entry["frame"] == {"x": frame.x, "y": frame.y, "w": frame.w, "h": frame.h}


def test_a_trimmed_sprite_reports_where_it_came_from():
    """``spriteSourceSize`` plus ``sourceSize`` is what puts a sprite back where
    the artist drew it rather than flush against its own bounding box."""
    sprites = [_sprite("a", 20, 16, mark=(3, 5, 11, 9))]
    result = layout(sprites, PackSettings(power_of_two=False))
    entry = texturepacker.tp_json(result, image_name="a.png")["frames"][0]
    assert entry["trimmed"] is True
    assert entry["spriteSourceSize"] == {"x": 3, "y": 5, "w": 8, "h": 4}
    assert entry["sourceSize"] == {"w": 20, "h": 16}


def test_an_untrimmed_sprite_says_so():
    _sprites, result = _layout([_sprite("a", 8, 8)])
    entry = texturepacker.tp_json(result, image_name="a.png")["frames"][0]
    assert entry["trimmed"] is False
    assert entry["spriteSourceSize"] == {"x": 0, "y": 0, "w": 8, "h": 8}


def test_rotated_is_always_present_and_always_false():
    """The packer does not rotate, and the key is emitted anyway because a
    consumer reads it -- a missing one is a schema question, not an answer."""
    _sprites, result = _layout()
    for entry in texturepacker.tp_json(result, image_name="a.png")["frames"]:
        assert entry["rotated"] is False


def test_two_serializations_of_one_layout_are_byte_identical():
    """What makes a re-export of an unchanged document reproducible."""
    _sprites, result = _layout()
    first = texturepacker.tp_bytes(result, image_name="atlas.png")
    assert first == texturepacker.tp_bytes(result, image_name="atlas.png")
    assert json.loads(first)["meta"]["image"] == "atlas.png"


def test_a_name_that_already_ends_in_png_is_not_doubled():
    sprites = [Sprite(key="a", name="hero.png", pixels=np.ones((4, 4, 4), np.uint8) * 255)]
    result = layout(sprites, PackSettings(power_of_two=False))
    entry = texturepacker.tp_json(result, image_name="a.png")["frames"][0]
    assert entry["filename"] == "hero.png"


def test_this_module_never_reaches_for_the_warlock_sheet_sidecar():
    """``pipelines.sheet`` stays the sole writer of Warlock's own versioned
    sheet format, so ``version: 1`` cannot come to mean two documents. This
    module writes a *different* format and must not blur into it."""
    tree = ast.parse(Path(inspect.getfile(texturepacker)).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "sheet" not in (node.module or "")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "sheet" not in alias.name
