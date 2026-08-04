"""Blend arithmetic, against hand-computed values rather than against itself.

A compositor is easy to write so that it looks right and is subtly wrong -- a
missing (1-ab) term shows up only where a translucent layer meets a translucent
backdrop, which is nowhere in a screenshot and everywhere in a real document.
So the assertions here are the spec's own numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import composite as cp


def _px(r, g, b, a=1.0):
    return np.array([[[r, g, b, a]]], dtype=np.float32)


def test_normal_over_opaque_is_the_source():
    out = cp.over(_px(1, 0, 0), _px(0, 0, 1))
    assert np.allclose(out, _px(0, 0, 1))


def test_a_half_transparent_source_lerps_toward_the_backdrop():
    out = cp.over(_px(0, 0, 0), _px(1, 1, 1, 0.5))
    assert np.allclose(out[..., :3], 0.5, atol=1e-6)
    assert np.allclose(out[..., 3], 1.0)


def test_layer_opacity_scales_the_source_alpha_not_its_colour():
    out = cp.over(_px(0, 0, 0), _px(1, 1, 1), opacity=0.25)
    assert np.allclose(out[..., :3], 0.25, atol=1e-6)


def test_an_empty_source_leaves_the_backdrop_untouched():
    back = _px(0.2, 0.4, 0.6, 0.8)
    out = cp.over(back, _px(1, 1, 1, 0.0))
    assert np.allclose(out, back, atol=1e-6)


def test_compositing_onto_nothing_keeps_the_sources_colour():
    """The ao == 0 guard must not smear black into the transparent region: the
    stamp's bounding box is mostly empty and its colour still has to survive."""
    out = cp.over(_px(0, 0, 0, 0.0), _px(1, 0, 0, 0.5))
    assert np.allclose(out[..., :3], [1, 0, 0], atol=1e-6)
    assert np.allclose(out[..., 3], 0.5)


@pytest.mark.parametrize(
    "mode, cb, cs, expected",
    [
        ("multiply", 0.5, 0.5, 0.25),
        ("screen", 0.5, 0.5, 0.75),
        ("overlay", 0.25, 0.5, 0.25),  # cb <= 0.5 -> 2·cb·cs
        ("overlay", 0.75, 0.5, 0.75),  # cb  > 0.5 -> 1 - 2(1-cb)(1-cs)
        ("add", 0.6, 0.6, 1.0),  # clamped, not wrapped
        ("normal", 0.3, 0.9, 0.9),
    ],
)
def test_blend_formulas_match_the_svg_spec(mode, cb, cs, expected):
    got = cp.blend(_px(cb, cb, cb)[..., :3], _px(cs, cs, cs)[..., :3], mode)
    assert np.allclose(got, expected, atol=1e-6)


def test_an_unknown_blend_mode_is_a_programming_error():
    with pytest.raises(ValueError):
        cp.blend(_px(0, 0, 0)[..., :3], _px(1, 1, 1)[..., :3], "dodge")


def test_a_blend_only_applies_where_the_backdrop_exists():
    """B(Cb, Cs) is weighted by ab: multiply over emptiness is the source, not
    black. Getting this wrong makes every non-normal layer punch a hole."""
    out = cp.over(_px(0, 0, 0, 0.0), _px(0.5, 0.5, 0.5), mode="multiply")
    assert np.allclose(out[..., :3], 0.5, atol=1e-6)


def test_every_named_mode_has_an_ora_op_and_round_trips():
    for mode in cp.BLEND_MODES:
        assert cp.OPS_ORA[cp.ORA_OPS[mode]] == mode


# --- conversions ------------------------------------------------------------


def test_uint8_survives_a_round_trip_through_float():
    values = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(4, axis=2)
    assert np.array_equal(cp.to_uint8(cp.to_float(values)), values)


def test_a_matte_shows_through_transparency_and_a_none_matte_does_not():
    pixels = np.zeros((1, 1, 4), dtype=np.uint8)
    assert cp.flatten_onto(pixels, (255, 255, 255, 255))[0, 0].tolist() == [255, 255, 255, 255]
    assert cp.flatten_onto(pixels, None)[0, 0].tolist() == [0, 0, 0, 0]
