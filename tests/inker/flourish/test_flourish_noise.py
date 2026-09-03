"""Noise: seeded, bounded, smooth, and the same on every call."""

from __future__ import annotations

import numpy as np

from warlock.studio.inker.flourish import noise


def test_lattice_values_are_in_range_and_seed_dependent():
    ix = np.arange(64, dtype=np.int64)
    iy = np.zeros_like(ix)
    a = noise.hash_lattice(ix, iy, 1)
    b = noise.hash_lattice(ix, iy, 2)
    assert a.dtype == np.float32
    assert float(a.min()) >= 0.0 and float(a.max()) < 1.0
    assert not np.array_equal(a, b)
    assert np.array_equal(a, noise.hash_lattice(ix, iy, 1))


def test_value_noise_is_deterministic_and_bounded():
    x, y = noise.grid(64, 48, 6.0)
    a = noise.value2d(x, y, 11)
    assert a.shape == (48, 64)
    assert np.array_equal(a, noise.value2d(x, y, 11))
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_value_noise_interpolates_the_lattice():
    """At integer coordinates the noise *is* the lattice value; between, it is
    between the corners."""
    ix = np.arange(8, dtype=np.int64)
    iy = np.full_like(ix, 3)
    lattice = noise.hash_lattice(ix, iy, 5)
    sampled = noise.value2d(ix.astype(np.float32), iy.astype(np.float32), 5)
    assert np.allclose(lattice, sampled, atol=1e-6)
    mid = noise.value2d(np.asarray([2.5], dtype=np.float32), np.asarray([3.0], dtype=np.float32), 5)
    lo, hi = sorted((float(lattice[2]), float(lattice[3])))
    assert lo - 1e-6 <= float(mid[0]) <= hi + 1e-6


def test_fbm_stays_bounded_over_many_octaves():
    x, y = noise.grid(64, 64, 4.0)
    for octaves in (1, 3, 6):
        f = noise.fbm(x, y, 9, octaves=octaves)
        assert float(f.min()) >= 0.0 and float(f.max()) <= 1.0


def test_warp_with_zero_amount_is_identity_and_otherwise_moves():
    x, y = noise.grid(32, 32, 4.0)
    wx, wy = noise.warp(x, y, 1, amount=0.0)
    assert wx is x and wy is y
    wx, wy = noise.warp(x, y, 1, amount=1.0)
    assert not np.array_equal(wx, x)
    assert float(np.abs(wx - x).max()) <= 1.0 + 1e-6
