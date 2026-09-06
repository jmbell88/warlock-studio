"""Parity tests for the fused ``_grow`` rewrite.

filters._grow used to build a per-neighbour whole-canvas ``_shift`` array
and OR it in; that allocated the canvas ~24 times per dilation step. The
fused version ORs each neighbour in place. This module spells the OLD
``_shift``-based implementation as the reference and asserts bit-identical
output (docs/measurements/2026-08-30-native-batch-6-candidates.md §2).
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import filters


def _old_shift(mask: np.ndarray, dy: int, dx: int, *, wrap: bool) -> np.ndarray:
    if wrap:
        return np.roll(mask, (dy, dx), axis=(0, 1))
    out = np.zeros_like(mask)
    height, width = mask.shape
    out[max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = mask[
        max(-dy, 0):height + min(-dy, 0), max(-dx, 0):width + min(-dx, 0)
    ]
    return out


def _old_grow(mask: np.ndarray, steps: int, corners: int, *, wrap: bool) -> np.ndarray:
    neighbours = filters._CORNERS_8 if int(corners) >= 8 else filters._CORNERS_4
    out = mask
    for _ in range(steps):
        grown = out
        for dy, dx in neighbours:
            grown = grown | _old_shift(out, dy, dx, wrap=wrap)
        out = grown
    return out


_SIZES = ((1, 1), (5, 7), (33, 17))
_STEPS = (0, 1, 3, 7)
_CORNERS = (4, 8)
_WRAP = (False, True)


def test_grow_matches_the_shift_based_reference_for_every_neighbourhood_and_wrap() -> None:
    rng = np.random.default_rng(20260830)
    for height, width in _SIZES:
        for steps in _STEPS:
            for corners in _CORNERS:
                for wrap in _WRAP:
                    mask = rng.random((height, width)) < 0.5
                    expected = _old_grow(mask, steps, corners, wrap=wrap)
                    actual = filters._grow(mask, steps, corners, wrap=wrap)
                    assert np.array_equal(actual, expected), (
                        f"mismatch at size=({height},{width}) steps={steps} "
                        f"corners={corners} wrap={wrap}"
                    )


def test_grow_does_not_mutate_its_input() -> None:
    rng = np.random.default_rng(1)
    mask = rng.random((9, 11)) < 0.5
    original = mask.copy()
    filters._grow(mask, 3, 8, wrap=False)
    assert np.array_equal(mask, original)
