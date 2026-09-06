"""The fused facing floor: same numbers as the two-pass form, computed once.

docs/measurements/2026-09-06-native-batch-9-facing-floor.md measured the
shipped ``combine``/``assemble`` pair spending 255 ms per pass at a 2048 atlas
on ``np.where(weights >= MIN_FACING, weights, 0.0)`` then ``* vis`` -- and
``assemble`` repeated the whole thing a second time for its own ``total``.
``floor_weights`` is the one masked pass both now share.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from warlock.pipelines import retexture


def _write(path, arr, mode="RGB"):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode).save(path)


def _bakes(tmp_path, n=2, size=8):
    for i in range(n):
        colour = np.zeros((size, size, 3), dtype=np.float32)
        colour[..., i % 3] = 1.0
        weight = np.zeros((size, size), dtype=np.float32)
        weight[:, i * 2 : i * 2 + 2] = 1.0
        _write(tmp_path / f"bake_{i:02d}.png", colour)
        _write(tmp_path / f"weight_{i:02d}.png", weight, "L")
    return n


def _old_two_pass_floor(weights, vis):
    """The floor as it was spelled before the fuse: two full passes."""
    floored = np.where(weights >= retexture.MIN_FACING, weights, 0.0)
    if vis is not None:
        floored = floored * vis
    return floored.astype(np.float32)


def _old_combine(colours, weights, base, vis=None):
    """``combine``'s body before it was rewritten onto ``floor_weights``."""
    w = np.where(weights >= retexture.MIN_FACING, weights, 0.0).astype(np.float32)
    if vis is not None:
        w = w * vis.astype(np.float32)
    total = w.sum(axis=0)
    safe = np.where(total > 0.0, total, 1.0)
    mixed = np.einsum("nhwc,nhw->hwc", colours, w) / safe[..., None]
    return np.where((total > 0.0)[..., None], mixed, base).astype(np.float32)


def _weights_straddling_min_facing(rng, shape):
    w = rng.random(shape, dtype=np.float32)
    # Force some texels to land exactly on MIN_FACING, both sides of it, and
    # dead on it -- the floor is a >= comparison and the boundary is the part
    # most likely to drift under a rewrite.
    flat = w.reshape(-1)
    flat[0] = retexture.MIN_FACING
    flat[1] = np.nextafter(np.float32(retexture.MIN_FACING), np.float32(0.0))
    flat[2] = np.nextafter(np.float32(retexture.MIN_FACING), np.float32(1.0))
    flat[3] = 0.0
    return w


def test_floor_weights_matches_the_shipped_two_pass_expression_with_vis():
    rng = np.random.default_rng(0xF10A)
    shape = (4, 6, 6)
    weights = _weights_straddling_min_facing(rng, shape)
    vis = rng.random(shape, dtype=np.float32)
    vis.reshape(-1)[:2] = 0.0
    vis.reshape(-1)[2:4] = 1.0

    got = retexture.floor_weights(weights, vis)
    want = _old_two_pass_floor(weights, vis)

    assert got.dtype == np.float32
    assert np.array_equal(got, want)


def test_floor_weights_matches_the_shipped_expression_with_vis_none():
    rng = np.random.default_rng(0xF10B)
    weights = _weights_straddling_min_facing(rng, (3, 5, 5))

    got = retexture.floor_weights(weights, None)
    want = _old_two_pass_floor(weights, None)

    assert got.dtype == np.float32
    assert np.array_equal(got, want)


def test_floor_weights_pins_the_float32_output_from_float64_input():
    rng = np.random.default_rng(0xF10C)
    weights = rng.random((3, 4, 4)).astype(np.float64)
    weights[0, 0, 0] = float(retexture.MIN_FACING)
    vis = rng.random((3, 4, 4)).astype(np.float64)

    got = retexture.floor_weights(weights, vis)
    # Cast to float32 first, the way ``floor_weights`` does, then run the old
    # two-pass expression -- the dtype it pins is float32 arithmetic, not
    # float64 arithmetic truncated at the end.
    want = _old_two_pass_floor(weights.astype(np.float32), vis.astype(np.float32))

    assert got.dtype == np.float32
    assert np.array_equal(got, want)


def test_combine_still_equals_its_old_two_pass_body_with_vis():
    rng = np.random.default_rng(0xC0B1)
    n, h, w = 3, 6, 6
    colours = rng.random((n, h, w, 3), dtype=np.float32)
    weights = _weights_straddling_min_facing(rng, (n, h, w))
    vis = rng.random((n, h, w), dtype=np.float32)
    base = rng.random((h, w, 3), dtype=np.float32)

    got = retexture.combine(colours, weights, base, vis=vis)
    want = _old_combine(colours, weights, base, vis=vis)

    assert np.array_equal(got, want)


def test_combine_still_equals_its_old_two_pass_body_without_vis():
    rng = np.random.default_rng(0xC0B2)
    n, h, w = 3, 6, 6
    colours = rng.random((n, h, w, 3), dtype=np.float32)
    weights = _weights_straddling_min_facing(rng, (n, h, w))
    base = rng.random((h, w, 3), dtype=np.float32)

    got = retexture.combine(colours, weights, base)
    want = _old_combine(colours, weights, base)

    assert np.array_equal(got, want)


def test_assemble_computes_the_facing_floor_only_once(tmp_path, monkeypatch):
    """This is the test that fails against the unfixed code: the shipped
    ``assemble`` computed the floor once inside ``combine`` and a second time
    for its own ``total``, so the counting wrapper below would see 2 calls."""
    n = _bakes(tmp_path)
    base = tmp_path / "base.png"
    _write(base, np.full((8, 8, 3), 0.25, np.float32))

    calls = []
    real = retexture.floor_weights

    def counting(weights, vis=None):
        calls.append(1)
        return real(weights, vis)

    monkeypatch.setattr(retexture, "floor_weights", counting)

    report = retexture.assemble(tmp_path, base, tmp_path / "out.png", count=n)

    assert report is not None
    assert len(calls) == 1
