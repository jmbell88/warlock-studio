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
        ("darken", 0.3, 0.7, 0.3),
        ("darken", 0.7, 0.3, 0.3),
        ("lighten", 0.3, 0.7, 0.7),
        ("difference", 0.3, 0.7, 0.4),
        ("difference", 0.7, 0.3, 0.4),  # symmetric: it is |Cb - Cs|
        ("hard-light", 0.5, 0.25, 0.25),  # cs <= 0.5 -> 2·cb·cs
        ("hard-light", 0.5, 0.75, 0.75),  # cs  > 0.5 -> 1 - 2(1-cb)(1-cs)
        # The spec's three cases, in its order. An empty backdrop stays empty
        # under a full source, which is the case a naive `min(1, Cb/(1-Cs))`
        # gets wrong by dividing zero by zero.
        ("color-dodge", 0.0, 1.0, 0.0),
        ("color-dodge", 0.5, 1.0, 1.0),
        ("color-dodge", 0.4, 0.5, 0.8),
        ("color-dodge", 0.9, 0.5, 1.0),  # clamped at one
        ("color-burn", 1.0, 0.0, 1.0),
        ("color-burn", 0.5, 0.0, 0.0),
        ("color-burn", 0.6, 0.5, 0.2),  # 1 - min(1, 0.4/0.5)
        ("color-burn", 0.1, 0.5, 0.0),  # clamped at zero
        # Soft light: a mid-grey source is the identity, and either end pulls
        # the backdrop toward black or white without ever reaching it.
        ("soft-light", 0.4, 0.5, 0.4),
        ("soft-light", 0.25, 0.0, 0.25 - 0.25 * 0.75),
        ("soft-light", 0.36, 1.0, 0.36 + (0.6 - 0.36)),  # D(Cb) = sqrt(Cb)
        ("soft-light", 0.0, 1.0, 0.0),
        ("soft-light", 1.0, 0.0, 1.0),
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


def test_every_ora_op_is_the_name_krita_and_gimp_write():
    """The interop contract, spelled out rather than trusted to a pattern.

    ``.ora`` is a real exchange format and the composite-op strings are the
    whole of what another editor reads a blend mode from, so a name invented
    here would round-trip perfectly through this app and open as normal
    everywhere else. ``add`` is the one that is not derivable from ours: it is
    ``svg:plus``, a compositing operator rather than a blend mode.
    """
    assert cp.ORA_OPS == {
        "normal": "svg:src-over",
        "darken": "svg:darken",
        "multiply": "svg:multiply",
        "color-burn": "svg:color-burn",
        "lighten": "svg:lighten",
        "screen": "svg:screen",
        "color-dodge": "svg:color-dodge",
        "add": "svg:plus",
        "overlay": "svg:overlay",
        "hard-light": "svg:hard-light",
        "soft-light": "svg:soft-light",
        "difference": "svg:difference",
        "exclusion": "svg:exclusion",
        # The two the SVG set has no name for, so OpenRaster's namespacing rule
        # applies and the namespace is the application whose definition we
        # implement. See ``test_blend_full_set``.
        "subtract": "krita:subtract",
        "divide": "krita:divide",
        "hue": "svg:hue",
        "saturation": "svg:saturation",
        "color": "svg:color",
        "luminosity": "svg:luminosity",
    }
    assert set(cp.ORA_OPS) == set(cp.BLEND_MODES)


def test_every_mode_the_kernel_knows_is_a_mode_that_exists():
    """``_MODE_IDS`` is written out, which is what stops a new mode from
    silently compositing as normal -- but the numbers still have to name real
    modes, and each one has to be unique or two modes share a C case."""
    assert set(cp._MODE_IDS) <= set(cp.BLEND_MODES)
    assert len(set(cp._MODE_IDS.values())) == len(cp._MODE_IDS)
    assert cp._MODE_REPLACE not in cp._MODE_IDS.values()


def test_hard_light_is_overlay_with_the_operands_swapped():
    """Not a coincidence to be preserved -- it is how both paths are written.

    The reference calls ``blend(source, backdrop, "overlay")`` and the kernel
    calls ``blend_channel(OVERLAY, cs, cb)``, so the identity is exact rather
    than exact-to-a-rounding, and asserting it here is what stops somebody
    "simplifying" either side into a hand-expanded formula that agrees only to
    within an ulp.
    """
    rng = np.random.default_rng(4)
    cb = rng.random((16, 16, 3), dtype=np.float32)
    cs = rng.random((16, 16, 3), dtype=np.float32)
    assert np.array_equal(cp.blend(cb, cs, "hard-light"), cp.blend(cs, cb, "overlay"))


@pytest.mark.parametrize("mode", cp.BLEND_MODES)
def test_no_mode_produces_a_warning_or_a_nan_on_ordinary_pixels(mode, recwarn):
    """Every formula is fed the values it actually branches on.

    Zero and one are not edge cases here: ``1 - Cs`` is exactly zero for a
    saturated channel, which is most of a white stroke, and colour-dodge would
    divide by it. A RuntimeWarning out of a paint stroke is a bug even when the
    pixel it produces happens to be right.

    The grid carries an **RGB last axis** rather than being a bare (5, 5) plane
    of pairs: the non-separable four read all three channels to decide one, so
    they are only defined on the shape ``over`` passes them. Two of the three
    channels are permutations of the same grid and the third is a constant, so
    every pair of extremes still meets, and greys (all three equal) are in there
    too -- which is the case ``_set_sat``'s zero-span branch exists for.
    """
    values = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    grid_b, grid_s = np.meshgrid(values, values)
    cb = np.stack([grid_b, grid_b[::-1], grid_b], axis=-1).astype(np.float32)
    cs = np.stack([grid_s, grid_s, grid_s[:, ::-1]], axis=-1).astype(np.float32)
    with np.errstate(all="raise"):
        out = cp.blend(cb, cs, mode)
    assert np.all(np.isfinite(out)), mode
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# --- conversions ------------------------------------------------------------


def test_uint8_survives_a_round_trip_through_float():
    values = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(4, axis=2)
    assert np.array_equal(cp.to_uint8(cp.to_float(values)), values)


def test_a_matte_shows_through_transparency_and_a_none_matte_does_not():
    pixels = np.zeros((1, 1, 4), dtype=np.uint8)
    assert cp.flatten_onto(pixels, (255, 255, 255, 255))[0, 0].tolist() == [255, 255, 255, 255]
    assert cp.flatten_onto(pixels, None)[0, 0].tolist() == [0, 0, 0, 0]
