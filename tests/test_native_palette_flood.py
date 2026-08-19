"""ABI 9's two kernels against their numpy references, bit for bit.

Bit-parity is the contract rather than a goal, and both of these are integer
kernels, so parity is reachable without any care about rounding -- which is
exactly why neither wants SIMD. Every test below runs the reference and the
kernel over the same input and asserts equality, and the module is skipped
whole when no DLL is present (``vendor/`` is gitignored, so that is the ordinary
state of a fresh checkout).
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import native

pytestmark = pytest.mark.skipif(
    not native.available(), reason="warlockc is not built in this checkout"
)


def _reference_nearest(queries: np.ndarray, palette: np.ndarray) -> np.ndarray:
    delta = queries[:, None, :].astype(np.int64) - palette[None, :, :].astype(np.int64)
    return np.argmin((delta * delta).sum(axis=2), axis=1)


def _run_nearest(queries: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.empty(queries.shape[0], dtype=np.int32)
    native.palette_nearest(
        np.ascontiguousarray(queries, dtype=np.int32),
        np.ascontiguousarray(palette, dtype=np.int32),
        out,
    )
    return out


# --- palette nearest ---------------------------------------------------------


def test_random_colours_against_a_random_palette() -> None:
    rng = np.random.default_rng(11)
    queries = rng.integers(0, 256, (4096, 3), dtype=np.int32)
    palette = rng.integers(0, 256, (37, 3), dtype=np.int32)
    assert np.array_equal(_run_nearest(queries, palette), _reference_nearest(queries, palette))


def test_a_tie_goes_to_the_lowest_index() -> None:
    """numpy ``argmin``'s first-minimum rule, which is what makes duplicate
    palette entries behave identically through either path."""
    palette = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 0]], dtype=np.int32)
    queries = np.array([[5, 0, 0], [0, 0, 0]], dtype=np.int32)
    got = _run_nearest(queries, palette)
    assert got.tolist() == [0, 0]
    assert np.array_equal(got, _reference_nearest(queries, palette))


def test_duplicate_palette_entries() -> None:
    palette = np.array([[7, 7, 7]] * 5, dtype=np.int32)
    queries = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.int32)
    assert _run_nearest(queries, palette).tolist() == [0, 0]


def test_a_one_entry_palette() -> None:
    palette = np.array([[128, 64, 32]], dtype=np.int32)
    rng = np.random.default_rng(3)
    queries = rng.integers(0, 256, (64, 3), dtype=np.int32)
    assert set(_run_nearest(queries, palette).tolist()) == {0}


def test_a_full_256_entry_palette() -> None:
    rng = np.random.default_rng(5)
    palette = rng.integers(0, 256, (256, 3), dtype=np.int32)
    queries = rng.integers(0, 256, (2048, 3), dtype=np.int32)
    assert np.array_equal(_run_nearest(queries, palette), _reference_nearest(queries, palette))


def test_negative_and_overflowing_reflection_queries() -> None:
    """``dither._ordered``'s second search is over ``2c - p1``, which spans
    -255..510 by construction. A uint8 signature would clamp exactly the search
    that exists to look at the far side, and would pass every test that only
    fed in-gamut colours."""
    rng = np.random.default_rng(9)
    palette = rng.integers(0, 256, (16, 3), dtype=np.int32)
    queries = rng.integers(-255, 511, (1024, 3), dtype=np.int32)
    assert queries.min() < 0 and queries.max() > 255
    assert np.array_equal(_run_nearest(queries, palette), _reference_nearest(queries, palette))


def test_no_queries_at_all() -> None:
    palette = np.array([[0, 0, 0]], dtype=np.int32)
    assert _run_nearest(np.zeros((0, 3), dtype=np.int32), palette).size == 0


# --- the three call sites agree either way ------------------------------------


def _both_ways(monkeypatch, run):
    """Run ``run`` with the kernel and again with ``WARLOCK_NATIVE=0``."""
    with_kernel = run()
    monkeypatch.setenv("WARLOCK_NATIVE", "0")
    native.reset()
    try:
        without = run()
    finally:
        monkeypatch.delenv("WARLOCK_NATIVE", raising=False)
        native.reset()
    return with_kernel, without


def test_snap_agrees_with_and_without_the_kernel(monkeypatch) -> None:
    from warlock.studio.inker import indexed as ix

    rng = np.random.default_rng(2)
    pixels = np.zeros((32, 32, 4), dtype=np.uint8)
    pixels[..., :3] = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[0, 0, 3] = 0  # a fully transparent pixel rides through verbatim
    palette = [tuple(int(v) for v in c) + (255,) for c in
               rng.integers(0, 256, (12, 3), dtype=np.uint8)]

    a, b = _both_ways(monkeypatch, lambda: ix.snap(pixels, palette))
    assert np.array_equal(a, b)


def test_ordered_dithering_agrees_with_and_without_the_kernel(monkeypatch) -> None:
    from warlock.studio.inker import dither

    rng = np.random.default_rng(6)
    pixels = np.zeros((24, 24, 4), dtype=np.uint8)
    pixels[..., :3] = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    pixels[..., 3] = 255
    palette = [tuple(int(v) for v in c) + (255,) for c in
               rng.integers(0, 256, (8, 3), dtype=np.uint8)]

    a, b = _both_ways(monkeypatch, lambda: dither.convert(pixels, palette, "bayer4"))
    assert np.array_equal(a, b)


def test_resolve_agrees_with_and_without_the_kernel(monkeypatch) -> None:
    from warlock.studio.inker import index_plane as ixp

    rng = np.random.default_rng(8)
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[..., :3] = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    pixels[..., 3] = 255
    pixels[0, :4, 3] = 0
    palette = [(0, 0, 0, 255)] + [
        tuple(int(v) for v in c) + (255,)
        for c in rng.integers(0, 256, (10, 3), dtype=np.uint8)
    ]
    table = ixp.lut(palette, 0)

    a, b = _both_ways(monkeypatch, lambda: ixp.resolve(pixels, table, 0))
    assert np.array_equal(a, b)


# --- the flood ---------------------------------------------------------------


def _reference_flood(match: np.ndarray, x: int, y: int) -> np.ndarray:
    """The frontier dilation, written out so the test does not depend on which
    branch ``flood_mask`` happens to take."""
    height, width = match.shape
    seen = np.zeros((height, width), dtype=bool)
    if not (0 <= x < width and 0 <= y < height) or not match[y, x]:
        return seen
    seen[y, x] = True
    while True:
        grown = np.zeros_like(seen)
        grown[:-1, :] |= seen[1:, :]
        grown[1:, :] |= seen[:-1, :]
        grown[:, :-1] |= seen[:, 1:]
        grown[:, 1:] |= seen[:, :-1]
        grown &= match & ~seen
        if not grown.any():
            return seen
        seen |= grown


def _run_flood(match: np.ndarray, x: int, y: int) -> np.ndarray:
    height, width = match.shape
    wanted = np.ascontiguousarray(match, dtype=np.uint8)
    out = np.empty((height, width), dtype=np.uint8)
    scratch = np.empty(height * width, dtype=np.int32)
    native.flood(wanted, out, scratch, (x, y))
    return out.astype(bool)


def _agree(match: np.ndarray, x: int, y: int) -> None:
    assert np.array_equal(_run_flood(match, x, y), _reference_flood(match, x, y))


def test_an_empty_map_fills_whole() -> None:
    _agree(np.ones((16, 16), dtype=bool), 0, 0)


def test_a_full_map_reaches_nothing() -> None:
    _agree(np.zeros((16, 16), dtype=bool), 3, 3)


def test_a_region_with_a_hole() -> None:
    match = np.ones((16, 16), dtype=bool)
    match[6:10, 6:10] = False
    _agree(match, 0, 0)
    _agree(match, 7, 7)


def test_diagonal_contact_does_not_connect() -> None:
    """Four-connected: a diagonal is only ever reached through one of the four
    orthogonal neighbours."""
    match = np.zeros((8, 8), dtype=bool)
    match[2, 2] = match[3, 3] = True
    got = _run_flood(match, 2, 2)
    assert got[2, 2] and not got[3, 3]
    _agree(match, 2, 2)


def test_a_serpentine_corridor() -> None:
    """The pathological case for the dilation -- path distance approaching the
    cell count -- and the reason the kernel exists at all."""
    size = 64
    match = np.zeros((size, size), dtype=bool)
    for row in range(0, size, 2):
        match[row, :] = True
        if (row // 2) % 2 == 0:
            match[row + 1, size - 1] = True
        elif row + 1 < size:
            match[row + 1, 0] = True
    _agree(match, 0, 0)


def test_the_boundary_never_wraps() -> None:
    match = np.ones((8, 8), dtype=bool)
    match[:, 4] = False
    got = _run_flood(match, 0, 0)
    assert got[:, :4].all()
    assert not got[:, 5:].any()
    _agree(match, 0, 0)


def test_an_out_of_bounds_seed_reaches_nothing() -> None:
    match = np.ones((8, 8), dtype=bool)
    for seed in ((-1, 0), (0, -1), (8, 0), (0, 8)):
        assert not _run_flood(match, *seed).any()


def test_a_non_matching_seed_reaches_nothing() -> None:
    match = np.ones((8, 8), dtype=bool)
    match[3, 3] = False
    _agree(match, 3, 3)
    assert not _run_flood(match, 3, 3).any()


def test_random_masks_agree() -> None:
    rng = np.random.default_rng(13)
    for _ in range(20):
        match = rng.random((24, 24)) > 0.35
        match[5, 5] = True
        _agree(match, 5, 5)


def test_flood_mask_agrees_with_and_without_the_kernel(monkeypatch) -> None:
    from warlock.studio.plotter import tools

    rng = np.random.default_rng(17)
    match = rng.random((32, 32)) > 0.3
    match[0, 0] = True

    a, b = _both_ways(monkeypatch, lambda: tools.flood_mask(match, 0, 0))
    assert np.array_equal(a, b)
