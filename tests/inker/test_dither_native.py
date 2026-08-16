"""The Floyd-Steinberg kernel against the Python loop it replaced.

The bar is byte identity, and here that is a stronger claim than it sounds.
Error diffusion is a chain: every pixel's quantised output becomes part of its
neighbours' input, so a single differing ulp anywhere does not stay local -- it
propagates down and across the rest of the image and shows up as a *different
picture*, not as one wrong pixel. That makes this the one kernel whose parity
test barely needs to search for a hard case: any disagreement at all, on any
input, compounds into a visible one. It also means a passing large image is
worth more here than anywhere else in ``native/``, which is why the sweep ends
with one.

What is worth searching for instead is the *shapes*, and they are the serpentine
scan's: the direction flips per row, so the "pixel ahead" and "pixel behind" are
different neighbours on odd and even rows, and every clamp at a row end is a
different clamp depending on parity. Single-row, single-column and single-pixel
images are in the sweep because each removes one of the four diffusion targets
entirely.

Transparency is the other axis. Invisible pixels are neither quantised nor used
as error sinks, so a hole in the alpha changes which neighbours an error reaches
-- a fully opaque image never exercises that branch at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import native
from warlock.studio.inker import dither

needs_dll = pytest.mark.skipif(not native.available(), reason="warlockc.dll not built")


def _numpy(image: np.ndarray, table: np.ndarray, monkeypatch) -> np.ndarray:
    """The reference body, with the kernel forced off."""
    monkeypatch.setattr(native, "available", lambda: False)
    return dither._floyd_steinberg(image.copy(), table)


def _kernel(image: np.ndarray, table: np.ndarray, monkeypatch) -> np.ndarray:
    monkeypatch.setattr(native, "available", lambda: True)
    return dither._floyd_steinberg(image.copy(), table)


def _image(shape: tuple[int, int], alpha: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    height, width = shape
    out = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    if alpha == "opaque":
        out[..., 3] = 255
    elif alpha == "holes":
        out[..., 3] = np.where(rng.random(shape) < 0.4, 0, 255)
    elif alpha == "empty":
        out[..., 3] = 0
    else:  # "checker" -- every other pixel, so no error ever reaches its
        # right-hand neighbour and the row-below writes carry everything.
        rows, cols = np.indices(shape)
        out[..., 3] = np.where((rows + cols) % 2 == 0, 255, 0)
    return out


def _table(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (n, 3), dtype=np.int16)


@needs_dll
@pytest.mark.parametrize("shape", [(1, 1), (1, 40), (40, 1), (2, 2), (23, 61), (61, 23)])
@pytest.mark.parametrize("alpha", ["opaque", "holes", "empty", "checker"])
def test_the_kernel_and_the_loop_agree_byte_for_byte(shape, alpha, monkeypatch):
    image = _image(shape, alpha, seed=hash((shape, alpha)) % 2**32)
    table = _table(17, seed=5)
    reference = _numpy(image, table, monkeypatch)
    assert np.array_equal(reference, _kernel(image, table, monkeypatch))


@needs_dll
@pytest.mark.parametrize("entries", [1, 2, 3, 8, 32, 255])
def test_agreement_holds_across_palette_sizes(entries, monkeypatch):
    """A one-colour palette makes every error maximal and every argmin a tie at
    index zero; 255 is the largest a converted document can carry."""
    image = _image((31, 29), "opaque", seed=entries)
    table = _table(entries, seed=entries + 100)
    reference = _numpy(image, table, monkeypatch)
    assert np.array_equal(reference, _kernel(image, table, monkeypatch))


@needs_dll
def test_a_tied_argmin_picks_the_same_entry_on_both_paths(monkeypatch):
    """``np.argmin`` returns the *first* minimum, and a palette carrying the
    same colour twice is the only way to see which one a implementation took.
    A strict ``<`` in the kernel is what keeps that the first."""
    image = np.full((8, 8, 4), 128, dtype=np.uint8)
    image[..., 3] = 255
    table = np.array([[10, 10, 10], [200, 200, 200], [10, 10, 10]], dtype=np.int16)
    reference = _numpy(image, table, monkeypatch)
    assert np.array_equal(reference, _kernel(image, table, monkeypatch))


@needs_dll
def test_a_flat_grey_ramp_agrees(monkeypatch):
    """Real dither input, not noise: a smooth ramp is where error diffusion
    actually accumulates, and where a wrong serpentine direction shows up as a
    lean rather than as noise."""
    ramp = np.linspace(0, 255, 64, dtype=np.float64)
    image = np.zeros((48, 64, 4), dtype=np.uint8)
    image[..., :3] = ramp[None, :, None].astype(np.uint8)
    image[..., 3] = 255
    table = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.int16)
    reference = _numpy(image, table, monkeypatch)
    assert np.array_equal(reference, _kernel(image, table, monkeypatch))


@needs_dll
def test_a_large_image_agrees(monkeypatch):
    """The one that matters most, for the reason at the top of this file: a
    disagreement anywhere in a chain this long is a different picture, so a
    hundred thousand pixels of agreement is a much stronger statement than the
    same test at 8 by 8. Kept to 128 square because the *reference* is a Python
    loop at ~10 us/px and this test has to run it."""
    image = _image((128, 128), "holes", seed=99)
    table = _table(12, seed=99)
    reference = _numpy(image, table, monkeypatch)
    assert np.array_equal(reference, _kernel(image, table, monkeypatch))


@needs_dll
def test_convert_routes_through_the_kernel_and_still_lands_on_the_palette(monkeypatch):
    """The public entry point, and the invariant every method shares: every
    output colour is an entry of the palette, exactly."""
    monkeypatch.setattr(native, "available", lambda: True)
    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (24, 24, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    palette = [(0, 0, 0, 255), (255, 0, 0, 255), (0, 255, 0, 255), (255, 255, 255, 255)]

    out = dither.convert(pixels, palette, "floyd-steinberg")

    wanted = {tuple(c)[:3] for c in palette}
    seen = {tuple(v) for v in out[..., :3].reshape(-1, 3)}
    assert seen <= wanted


@needs_dll
def test_alpha_rides_through_untouched(monkeypatch):
    """The kernel is handed the alpha only as a visibility mask and never writes
    it; the reference never touched it either."""
    monkeypatch.setattr(native, "available", lambda: True)
    image = _image((16, 16), "holes", seed=7)
    before = image[..., 3].copy()
    out = dither._floyd_steinberg(image.copy(), _table(8, seed=7))
    assert np.array_equal(out[..., 3], before)


@needs_dll
def test_an_invisible_pixel_keeps_its_dead_rgb(monkeypatch):
    """Neither quantised nor diffused into: a transparent pixel's RGB is not a
    colour, and the kernel has to leave it exactly as it found it."""
    monkeypatch.setattr(native, "available", lambda: True)
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[..., :3] = 77
    image[..., 3] = 255
    image[1, 1, 3] = 0
    out = dither._floyd_steinberg(image.copy(), np.array([[0, 0, 0]], dtype=np.int16))
    assert tuple(out[1, 1]) == (77, 77, 77, 0)


def test_the_seam_refuses_a_shape_the_kernel_cannot_index(monkeypatch):
    """A refusal is a fallback and never an error -- the rule at every one of
    these seams. Non-contiguous work is the shape that would silently index
    wrongly, so it has to be turned away rather than reinterpreted."""
    monkeypatch.setattr(native, "available", lambda: True)
    work = np.zeros((4, 8, 3), dtype=np.float32)[:, ::2]
    visible = np.ones((4, 4), dtype=np.bool_)
    entries = np.zeros((2, 3), dtype=np.float32)
    assert dither._diffuse_native(work, visible, entries) is False

    # float64 work would be read as garbage floats, and an empty palette has no
    # entry to pick.
    assert dither._diffuse_native(np.zeros((4, 4, 3)), visible, entries) is False
    assert (
        dither._diffuse_native(
            np.zeros((4, 4, 3), dtype=np.float32), visible, np.zeros((0, 3), dtype=np.float32)
        )
        is False
    )


def test_the_loop_is_still_there(monkeypatch):
    """The numpy/Python reference is never deleted: it is the fallback on a
    machine with no compiler and the thing this file measures against."""
    monkeypatch.setattr(native, "available", lambda: False)
    image = _image((12, 12), "opaque", seed=1)
    out = dither._floyd_steinberg(image.copy(), _table(4, seed=1))
    assert out.shape == (12, 12, 4)
