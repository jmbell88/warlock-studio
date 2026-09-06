"""_to_oklab's uint8 fast path must match the float reference exactly.

docs/measurements/2026-08-30-native-batch-6-candidates.md #3: a 256-entry
lookup table replaces the sRGB->linear closed form when the input is already
uint8. The float expression stays in ``_to_oklab`` unchanged, so it is the
reference these tests check the LUT path against.
"""

import numpy as np
import pytest

from warlock.pipelines import pixel


def test_uint8_path_matches_float_reference_for_random_frame():
    """A seeded random uint8 (H, W, 3) frame: LUT path == float path, exactly."""
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, size=(37, 41, 3), dtype=np.uint8)
    got = pixel._to_oklab(u8)
    want = pixel._to_oklab(u8.astype(np.float64))
    assert np.array_equal(got, want)


def test_uint8_path_matches_float_reference_for_every_channel_value():
    """Every single value 0..255, broadcast across channels, is bit-identical."""
    values = np.arange(256, dtype=np.uint8)
    u8 = np.stack([values, values, values], axis=-1)  # (256, 3): grey ramp
    got = pixel._to_oklab(u8)
    want = pixel._to_oklab(u8.astype(np.float64))
    assert np.array_equal(got, want)


def test_uint8_input_uses_the_lut_not_the_power_expression():
    """The uint8 path calls the LUT builder, and only once (cached)."""
    calls = []
    real_builder = pixel._srgb_to_linear_lut
    pixel._srgb_lut_cache = None

    def counting_builder():
        calls.append(1)
        return real_builder()

    original = pixel._srgb_to_linear_lut
    pixel._srgb_to_linear_lut = counting_builder
    try:
        u8 = np.zeros((2, 2, 3), dtype=np.uint8)
        pixel._to_oklab(u8)
        pixel._to_oklab(u8)
        assert calls == [1]
    finally:
        pixel._srgb_to_linear_lut = original
        pixel._srgb_lut_cache = None


def test_float_input_with_integral_values_still_uses_float_path():
    """A float64 array is never silently routed through the uint8 LUT path.

    ``map_palette`` widens the palette to float64 before calling
    ``_to_oklab``; even though its values are integral, the function must not
    reinterpret it as uint8 -- this asserts the float path still produces the
    same closed-form result the unmodified expression always gave.
    """
    entries = np.array([[0.0, 128.0, 255.0]], dtype=np.float64)
    got = pixel._to_oklab(entries)

    c = entries / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]
    lm = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    mm = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    sm = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(lm), np.cbrt(mm), np.cbrt(sm)
    want = np.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        axis=-1,
    )
    assert np.array_equal(got, want)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
