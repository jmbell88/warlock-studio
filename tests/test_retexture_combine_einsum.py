"""Regression coverage for the einsum weighted-sum in ``retexture.combine``.

docs/measurements/2026-09-06-native-batch-8-retexture.md measured the shipped
``(colours * w[..., None]).sum(axis=0)`` as 56% of ``combine`` at a TRELLIS
atlas size, and measured ``np.einsum("nhwc,nhw->hwc", colours, w)`` as a
bit-identical, 2.2x-faster replacement. The parity claim here is exact
accumulation order, not closeness: every assertion below must be
``np.array_equal``, never ``np.allclose``.
"""

import numpy as np

from warlock.pipelines import retexture


def _reference_combine(colours, weights, base, vis=None):
    """The old `combine` body, verbatim, as the parity reference."""
    w = np.where(weights >= retexture.MIN_FACING, weights, 0.0).astype(np.float32)
    if vis is not None:
        w = w * vis.astype(np.float32)
    total = w.sum(axis=0)
    safe = np.where(total > 0.0, total, 1.0)
    mixed = (colours * w[..., None]).sum(axis=0) / safe[..., None]
    return np.where((total > 0.0)[..., None], mixed, base).astype(np.float32)


def test_combine_matches_shipped_product_and_sum_with_and_without_vis():
    rng = np.random.default_rng(0)
    n = len(retexture.VIEWS)
    h = w_ = 64

    colours = rng.random((n, h, w_, 3), dtype=np.float32)
    # Straddle MIN_FACING so both the dropped and kept branches are exercised.
    weights = rng.random((n, h, w_), dtype=np.float32) * (2 * retexture.MIN_FACING)
    vis = (rng.random((n, h, w_)) > 0.3).astype(np.float32)
    base = rng.random((h, w_, 3), dtype=np.float32)

    # Force some texels to have zero total weight across every view, so the
    # base-colour fallback path is covered too.
    weights[:, 0, 0] = 0.0
    vis[:, 0, 0] = 0.0

    got_no_vis = retexture.combine(colours, weights, base, vis=None)
    ref_no_vis = _reference_combine(colours, weights, base, vis=None)
    assert np.array_equal(got_no_vis, ref_no_vis)

    got_vis = retexture.combine(colours, weights, base, vis=vis)
    ref_vis = _reference_combine(colours, weights, base, vis=vis)
    assert np.array_equal(got_vis, ref_vis)


def test_combine_matches_at_odd_non_simd_multiple_shape():
    rng = np.random.default_rng(1)
    n = 3
    h, w_ = 129, 257

    colours = rng.random((n, h, w_, 3), dtype=np.float32)
    weights = rng.random((n, h, w_), dtype=np.float32) * (2 * retexture.MIN_FACING)
    vis = (rng.random((n, h, w_)) > 0.5).astype(np.float32)
    base = rng.random((h, w_, 3), dtype=np.float32)

    # A patch with no contributor at all: zero weight and zero visibility.
    weights[:, 10:15, 20:26] = 0.0
    vis[:, 10:15, 20:26] = 0.0

    got = retexture.combine(colours, weights, base, vis=vis)
    ref = _reference_combine(colours, weights, base, vis=vis)
    assert np.array_equal(got, ref)
