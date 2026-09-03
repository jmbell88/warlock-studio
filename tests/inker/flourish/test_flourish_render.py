"""Rendering: byte-identical, direction-aware, and each primitive does what
its name says.

The digests in ``digests.json`` are the bar. A change that alters a single
byte of a fixture render fails here, which is the point: a primitive's look
is part of a recipe's meaning, and a preset the user saved must render the
same after an upgrade. Regenerate the file deliberately with
``uv run python tests/inker/flourish/_digests.py`` when a primitive's
arithmetic is *meant* to change, and say so in the commit.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from _recipes import ASSETS, FIREBALL, solo

from warlock.studio.inker import flourish
from warlock.studio.inker.flourish import prims

DIGESTS = Path(__file__).with_name("digests.json")
render_mod = importlib.import_module("warlock.studio.inker.flourish.render")


def _digest(frames: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for f in frames:
        h.update(f.tobytes())
    return h.hexdigest()


def _frames(raw: dict, direction: float = 0.0) -> list[np.ndarray]:
    rec = flourish.from_dict(raw)
    return [
        flourish.to_uint8(flourish.render_frame(rec, f, direction, ASSETS), rec.supersample)
        for f in range(rec.frame_count)
    ]


def test_the_same_recipe_renders_the_same_bytes_twice():
    a = _frames(FIREBALL)
    b = _frames(FIREBALL)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_frames_are_logical_size_straight_alpha_uint8():
    rec = flourish.from_dict(FIREBALL)
    frame = flourish.to_uint8(flourish.render_frame(rec, 5), rec.supersample)
    assert frame.shape == (48, 48, 4)
    assert frame.dtype == np.uint8
    # Straight alpha: a fully transparent pixel carries no colour.
    clear = frame[..., 3] == 0
    assert not frame[clear][..., :3].any()


def test_a_frame_can_be_rendered_out_of_order():
    rec = flourish.from_dict(FIREBALL)
    late = flourish.render_frame(rec, 12)
    early = flourish.render_frame(rec, 4)
    assert np.array_equal(flourish.render_frame(rec, 12), late)
    assert np.array_equal(flourish.render_frame(rec, 4), early)


def test_direction_rotates_the_simulation():
    rec = flourish.from_dict(solo("particles", spread=10.0, direction=0.0, gravity=0.0))
    right = flourish.to_uint8(flourish.render_frame(rec, 3), rec.supersample)
    down = flourish.to_uint8(flourish.render_frame(rec, 3, 90.0), rec.supersample)
    assert not np.array_equal(right, down)
    # Sparks fired to the right sit right of centre; turned 90 degrees, below it.
    ys, xs = np.nonzero(right[..., 3])
    assert xs.mean() > 16
    ys, xs = np.nonzero(down[..., 3])
    assert ys.mean() > 16


def test_every_layer_reports_its_own_plane():
    rec = flourish.from_dict(FIREBALL)
    planes = flourish.render_layers(rec, 5)  # projectile
    active = {each.uid for each in rec.layers if each.active_in("projectile")}
    assert set(planes) <= active
    assert {rec.layer(uid).kind for uid in planes} >= {"core", "flame", "trail", "glow"}


def test_the_digests_match():
    if not DIGESTS.exists():
        pytest.fail(f"{DIGESTS.name} is missing; run tests/inker/flourish/_digests.py")
    expected = json.loads(DIGESTS.read_text(encoding="utf-8"))
    got = {"fireball": _digest(_frames(FIREBALL)), "fireball@90": _digest(_frames(FIREBALL, 90.0))}
    for kind in prims.KINDS:
        got[f"solo:{kind}"] = _digest(_frames(solo(kind)))
    assert got == expected


@pytest.mark.parametrize("kind", prims.KINDS)
def test_each_primitive_paints_something(kind):
    rec = flourish.from_dict(solo(kind))
    mass = [
        int(flourish.to_uint8(flourish.render_frame(rec, f, 0.0, ASSETS), 2)[..., 3].sum())
        for f in range(6)
    ]
    assert max(mass) > 0, kind


def test_the_distortion_replaces_rather_than_adds():
    plain = flourish.from_dict(solo("core"))
    shimmer = flourish.from_dict(solo("distortion", strength=3.0))
    a = flourish.to_uint8(flourish.render_frame(plain, 2), 2)
    b = flourish.to_uint8(flourish.render_frame(shimmer, 2), 2)
    assert not np.array_equal(a, b)
    # Roughly the same coverage: pixels moved, none were added. A warp is not
    # measure-preserving (measured: an 11% loss at strength 3 on an 8px core),
    # so the bound is loose; doubling or halving would still fail it.
    assert abs(int(a[..., 3].sum()) - int(b[..., 3].sum())) < 0.25 * int(a[..., 3].sum())


def test_the_glow_only_brightens():
    plain = flourish.from_dict(solo("core"))
    glowing = flourish.from_dict(solo("glow", strength=1.0, radius=4.0))
    a = flourish.render_frame(plain, 2)
    b = flourish.render_frame(glowing, 2)
    assert float(b.sum()) > float(a.sum())
    assert float((b - a).min()) >= -1e-6


def test_an_invisible_or_out_of_phase_layer_renders_nothing():
    raw = solo("core")
    raw["layers"][0]["visible"] = False
    rec = flourish.from_dict(raw)
    assert flourish.render_layers(rec, 0) == {}
    raw = solo("core")
    raw["phases"].append({"name": "other", "frames": 2})
    raw["layers"][0]["phases"] = ["other"]
    rec = flourish.from_dict(raw)
    assert flourish.render_layers(rec, 0) == {}
    assert flourish.render_layers(rec, 6) != {}


def test_an_empty_recipe_composites_to_a_clear_frame():
    rec = flourish.from_dict({"size": [8, 8], "supersample": 2})
    frame = flourish.to_uint8(flourish.render_frame(rec, 0), 2)
    assert frame.shape == (8, 8, 4)
    assert not frame.any()


def test_the_frame_context_time_axes():
    rec = flourish.from_dict(FIREBALL)
    ctx = render_mod.frame_ctx(rec, 3)  # first frame of "projectile"
    assert ctx.t == 0.0
    assert ctx.phase_time == 0.0
    assert ctx.time == pytest.approx(3 / 18)
    ctx = render_mod.frame_ctx(rec, 7)  # last frame of "projectile"
    assert ctx.t == 1.0
    one = flourish.from_dict({"phases": [{"name": "a", "frames": 1}]})
    assert render_mod.frame_ctx(one, 0).t == 0.0


def test_a_sprite_without_its_asset_renders_nothing_rather_than_a_placeholder():
    rec = flourish.from_dict(solo("sprite"))
    assert flourish.render_layers(rec, 2) == {}
    assert flourish.render_layers(rec, 2, 0.0, ASSETS) != {}


def test_textured_particles_stamp_the_texture():
    plain = flourish.from_dict(solo("particles", count=6))
    textured = flourish.from_dict(solo("particles", count=6, texture="tex", size=4.0))
    a = flourish.render_frame(plain, 2, 0.0, ASSETS)
    b = flourish.render_frame(textured, 2, 0.0, ASSETS)
    assert not np.array_equal(a, b)
    # Without the asset the textured layer falls back to discs.
    c = flourish.render_frame(textured, 2)
    d = flourish.render_frame(flourish.from_dict(solo("particles", count=6, size=4.0)), 2)
    assert np.array_equal(c, d)
