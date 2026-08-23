"""The float palette search: same pick as ``argmin`` over ``linalg.norm``.

Bit-identity is not quite the right phrase for a kernel whose output is an
*index*, so the bar is stated as what it is: the same index, for every row,
including the ties and including the case the sqrt exists for.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock import native
from warlock.pipelines import pixel

needs_dll = pytest.mark.skipif(not native.available(), reason="warlockc.dll not built")


def _reference(flat: np.ndarray, plab: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(flat[:, None, :] - plab[None, :, :], axis=-1)
    return np.argmin(d, axis=1).astype(np.int32)


@needs_dll
def test_the_fallback_is_genuinely_taken_when_the_seam_is_closed(monkeypatch):
    calls: list[int] = []
    real = native.palette_nearest_f64
    monkeypatch.setattr(
        native, "palette_nearest_f64", lambda *a: (calls.append(1), real(*a))[1]
    )
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (8, 8, 4), dtype=np.uint8), "RGBA"
    )
    pixel.map_palette(image, ((0, 0, 0), (255, 255, 255)))
    assert calls, "the kernel was never reached with the DLL present"

    monkeypatch.setattr(native, "available", lambda: False)
    calls.clear()
    pixel.map_palette(image, ((0, 0, 0), (255, 255, 255)))
    assert not calls, "the numpy path still called the kernel"


@needs_dll
def test_every_row_picks_the_index_argmin_would():
    rng = np.random.default_rng(0x9A1)
    flat = rng.normal(size=(20_000, 3))
    plab = rng.normal(size=(48, 3))
    out = np.empty(flat.shape[0], dtype=np.int32)
    native.palette_nearest_f64(
        np.ascontiguousarray(flat), np.ascontiguousarray(plab), out
    )
    assert np.array_equal(out, _reference(flat, plab))


@needs_dll
def test_a_tie_goes_to_the_lowest_index():
    """``argmin``'s first-minimum rule, which is what makes a palette with a
    duplicated entry behave the same either way."""
    plab = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    flat = np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = np.empty(2, dtype=np.int32)
    native.palette_nearest_f64(
        np.ascontiguousarray(flat), np.ascontiguousarray(plab), out
    )
    assert out.tolist() == [0, 0]
    assert np.array_equal(out, _reference(flat, plab))


@needs_dll
def test_the_sqrt_is_not_optimised_away_into_a_squared_comparison():
    """The trap the kernel's header names. sqrt is monotonic but it also
    *rounds*, so two distinct squared distances can land on the same double --
    at which point a norm-based argmin takes the earlier index and a
    squares-based one takes the strictly smaller, which may sit later.

    Constructed rather than sampled, because a rounded tie is rare enough that
    random data would make this a test that passed for the wrong reason. Entry
    0 sits at squared distance ``1 + 2**-52`` -- the very next double above one
    -- and entry 1 at exactly one. sqrt rounds both to 1.0 (the true difference
    is half a unit in the last place, and ties round to even), so a norm-based
    argmin sees a tie and takes index 0, while a squares-based one sees entry 1
    as strictly nearer and takes index 1.
    """
    query = np.zeros((1, 3))
    plab = np.array([[1.0, 2.0**-26, 0.0], [1.0, 0.0, 0.0]])

    squares = ((query[:, None, :] - plab[None, :, :]) ** 2).sum(axis=-1)[0]
    assert squares[0] != squares[1], "the constructed pair is not distinct"
    norms = np.linalg.norm(query[:, None, :] - plab[None, :, :], axis=-1)[0]
    assert norms[0] == norms[1], "the constructed pair does not round to a tie"
    assert int(np.argmin(squares)) == 1 and int(np.argmin(norms)) == 0

    out = np.empty(1, dtype=np.int32)
    native.palette_nearest_f64(
        np.ascontiguousarray(query), np.ascontiguousarray(plab), out
    )
    assert out.tolist() == [0], "the kernel compared squares, not norms"

    # And the ordinary case still agrees over a large random sample.
    rng = np.random.default_rng(0x5417)
    big_plab = rng.normal(size=(200, 3))
    flat = rng.normal(size=(50_000, 3))
    got = np.empty(flat.shape[0], dtype=np.int32)
    native.palette_nearest_f64(
        np.ascontiguousarray(flat), np.ascontiguousarray(big_plab), got
    )
    assert np.array_equal(got, _reference(flat, big_plab))


@needs_dll
def test_a_mapped_image_is_byte_identical_either_way(monkeypatch):
    rng = np.random.default_rng(0x11)
    image = Image.fromarray(rng.integers(0, 256, (64, 64, 4), dtype=np.uint8), "RGBA")
    palette = tuple(
        (int(r), int(g), int(b)) for r, g, b in rng.integers(0, 256, (17, 3))
    )
    got = np.asarray(pixel.map_palette(image, palette))
    monkeypatch.setattr(native, "available", lambda: False)
    want = np.asarray(pixel.map_palette(image, palette))
    assert np.array_equal(got, want)


@needs_dll
def test_dithering_maps_the_same_either_way(monkeypatch):
    """The dithered path adds an offset before the search, so it feeds the
    kernel out-of-gamut Oklab -- which is exactly where the int32 sibling's
    signed-query requirement came from and is worth asserting here too."""
    rng = np.random.default_rng(0x22)
    image = Image.fromarray(rng.integers(0, 256, (48, 48, 4), dtype=np.uint8), "RGBA")
    palette = ((0, 0, 0), (255, 255, 255), (200, 30, 40), (30, 40, 200))
    got = np.asarray(pixel.map_palette(image, palette, dither=True))
    monkeypatch.setattr(native, "available", lambda: False)
    want = np.asarray(pixel.map_palette(image, palette, dither=True))
    assert np.array_equal(got, want)
