"""The bake: layers in painterly mode, one palette in pixel mode, facings,
tags; every preset renders in both; every engine snippet is well formed."""

from __future__ import annotations

import ast
import json

import numpy as np
import pytest
from _recipes import FIREBALL, solo

from warlock.studio.inker import flourish
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import engines, presets, prims


def _small(raw: dict, **over) -> flourish.Recipe:
    return flourish.from_dict({**raw, **over})


def test_a_painterly_bake_carries_a_cel_per_layer_per_frame():
    rec = _small(FIREBALL)
    baked = B.bake(rec)
    assert not baked.pixel
    assert baked.palette is None and baked.palette_source == "none"
    assert len(baked.facings) == 1
    facing = baked.facings[0]
    for phase in rec.phases:
        assert len(facing.composites[phase.name]) == phase.frames
        active = [each for each in rec.layers if each.active_in(phase.name)]
        assert set(facing.layers[phase.name]) == {each.uid for each in active}
        for cels in facing.layers[phase.name].values():
            assert len(cels) == phase.frames
            assert all(c.shape == (48, 48, 4) and c.dtype == np.uint8 for c in cels)


def test_the_tags_cover_the_flat_frames_exactly():
    rec = _small(FIREBALL)
    baked = B.bake(rec)
    tags = baked.tags()
    assert [t[0] for t in tags] == [p.name for p in rec.phases]
    assert tags[0][1] == 0
    assert tags[-1][2] == baked.frame_count - 1 == len(baked.flat()) - 1
    for (_, _, last, _), (_, first, _, _) in zip(tags, tags[1:], strict=False):
        assert first == last + 1
    assert [t[3] for t in tags] == [p.loop for p in rec.phases]


def test_directions_make_facings_and_their_own_tags():
    rec = _small(solo("particles", spread=10.0, gravity=0.0))
    baked = B.bake(rec, directions=4)
    assert [f.name for f in baked.facings] == ["E", "S", "W", "N"]
    assert [f.degrees for f in baked.facings] == [0.0, 90.0, 180.0, 270.0]
    names = [t[0] for t in baked.tags()]
    assert names == ["main/E", "main/S", "main/W", "main/N"]
    assert baked.frame_count == 4 * rec.frame_count
    east = baked.facings[0].composites["main"][3]
    south = baked.facings[1].composites["main"][3]
    assert not np.array_equal(east, south)


def test_direction_names_fall_back_to_angles():
    assert B.direction_names(8) == ("E", "SE", "S", "SW", "W", "NW", "N", "NE")
    assert B.direction_names(3) == ("0deg", "120deg", "240deg")
    assert B.direction_angles(3) == (0.0, 120.0, 240.0)


def test_a_pixel_bake_has_one_palette_and_no_layers():
    rec = _small(FIREBALL, mode="pixel", colors=8)
    baked = B.bake(rec)
    assert baked.pixel
    assert baked.palette_source == "derived"
    assert 1 <= len(baked.palette) <= 8
    facing = baked.facings[0]
    assert facing.layers == {}
    used: set[tuple[int, int, int]] = set()
    for phase in rec.phases:
        for cel in facing.composites[phase.name]:
            assert cel.shape == (48, 48, 4)
            # Hard alpha: pixel art has no half-covered pixels.
            assert set(np.unique(cel[..., 3])) <= {0, 255}
            opaque = cel[cel[..., 3] > 0][:, :3]
            used |= {tuple(int(v) for v in p) for p in opaque}
    assert used <= set(baked.palette)


def test_a_designed_palette_is_used_verbatim():
    rec = _small(FIREBALL, mode="pixel", palette=["#FF0000", "#00FF00", "#0000FF"])
    baked = B.bake(rec)
    assert baked.palette_source == "designed"
    assert baked.palette == ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    override = B.bake(rec, palette=((1, 2, 3),))
    assert override.palette == ((1, 2, 3),)


def test_the_bake_is_deterministic_in_both_modes():
    for mode in ("painterly", "pixel"):
        a = B.bake(_small(FIREBALL, mode=mode)).flat()
        b = B.bake(_small(FIREBALL, mode=mode)).flat()
        for x, y in zip(a, b, strict=True):
            assert np.array_equal(x, y)


def test_progress_is_reported_per_frame():
    rec = _small(solo("core"))
    seen: list[tuple[int, int]] = []
    B.bake(rec, directions=2, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(i, 12) for i in range(1, 13)]


def test_origin_and_fps_come_from_the_recipe():
    baked = B.bake(_small(solo("core")))
    assert baked.origin == (16, 16)
    assert baked.fps == 12


# -- presets ---------------------------------------------------------------------


def test_the_preset_library_has_the_launch_set():
    names = presets.names()
    assert len(names) >= 29
    for wanted in ("fireball", "explosion", "lightning_bolt", "heal", "portal", "meteor"):
        assert wanted in names
    assert presets.label("ice_nova") == "Ice nova"


def test_every_preset_loads_clamped_with_fresh_uids():
    for name in presets.names():
        rec = presets.load(name)
        assert rec == flourish.clamp(rec)
        assert len({each.uid for each in rec.layers}) == len(rec.layers)
        assert rec.frame_count > 0
        again = presets.load(name)
        assert {each.uid for each in rec.layers}.isdisjoint({each.uid for each in again.layers})
        for layer in rec.layers:
            assert layer.kind in prims.KINDS


def test_preset_files_are_tidy_json():
    for name in presets.names():
        data = json.loads(presets.path_of(name).read_text(encoding="utf-8"))
        assert data["name"]
        assert data["size"] == [128, 128]
        assert data["mode"] == "painterly"


@pytest.mark.parametrize("name", presets.names())
def test_every_preset_paints_in_both_modes(name):
    """Each preset at a quarter of its size, one direction, both modes: it
    renders without error and puts something on screen. Small so the whole
    library is a couple of seconds, not a minute."""
    rec = presets.load(name)
    raw = flourish.to_dict(rec)
    raw["size"] = [32, 32]
    raw["supersample"] = 2
    for layer in raw["layers"]:
        for key, value in list(layer["params"].items()):
            spec = prims.params_of(layer["kind"])[key]
            if spec.kind in ("curve", "float") and key in {
                "x", "y", "radius", "width", "height", "size", "spawn_radius", "speed", "gravity"
            }:
                layer["params"][key] = _quarter(value)
    for mode in ("painterly", "pixel"):
        raw["mode"] = mode
        raw["colors"] = 8
        baked = B.bake(flourish.from_dict(raw))
        assert max(int(f[..., 3].sum()) for f in baked.flat()) > 0, (name, mode)


def _quarter(value):
    if isinstance(value, (int, float)):
        return value / 4.0
    if isinstance(value, dict):
        return {**value, "keys": [[t, v / 4.0] for t, v in value["keys"]]}
    return value


def test_presets_refuse_paths():
    with pytest.raises(ValueError):
        presets.path_of("../fireball")
    with pytest.raises(KeyError):
        presets.path_of("no_such_effect")


# -- engines ---------------------------------------------------------------------


def _info() -> dict:
    return engines.describe(
        name="fireball explosion",
        image="fireball_explosion.png",
        frame_width=128,
        frame_height=128,
        frames=12,
        fps=18,
        loop=False,
        origin=(64, 64),
    )


def test_every_engine_renders_a_snippet_naming_the_sheet():
    info = _info()
    for engine in engines.ENGINES:
        text = engines.snippet(engine, info)
        assert "fireball_explosion.png" in text
        assert "128" in text and "18" in text
    with pytest.raises(ValueError):
        engines.snippet("unreal", info)


def test_the_pygame_snippet_is_valid_python_and_builds_the_animation():
    text = engines.snippet("pygame-ce", _info())
    ast.parse(text)

    class _Surface:
        def convert_alpha(self):
            return self

        def subsurface(self, rect):
            return rect

    stub = type("pygame", (), {})()
    stub.image = type("image", (), {"load": staticmethod(lambda path: _Surface())})()
    namespace: dict = {"pygame": stub}
    exec(text.replace("import pygame\n", ""), namespace)  # noqa: S102 -- the snippet under test
    anim = namespace["fireball_explosion"]
    assert len(anim.frames) == 12
    assert anim.frames[3] == (384, 0, 128, 128)
    assert anim.fps == 18 and anim.loop is False and anim.origin == (64, 64)
    anim.update(1.0)
    assert anim.done


def test_identifiers_are_safe():
    assert engines._ident("Ice nova!") == "ice_nova"  # noqa: SLF001
    assert engines._ident("2 fast") == "fx_2_fast"  # noqa: SLF001
    assert engines._ident("dark burst", pascal=True) == "DarkBurst"  # noqa: SLF001
    assert engines._ident("") == "effect"  # noqa: SLF001
