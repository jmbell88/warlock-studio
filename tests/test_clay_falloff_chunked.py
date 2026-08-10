"""H10: the falloff distance search runs in bounded memory, bit-identically.

``MAX_FALLOFF_PAIRS`` caps the pair *count*, which is a time budget -- at the
cap the one-shot broadcast still materialised ``pairs x 3 x float64``, close to
a gigabyte on the frame thread at the press of a drag. The search now walks the
selected rows in chunks under a running minimum, and the bar for that rewrite
is the native-kernel bar: **bit-identical** to the broadcast it replaces, not
close. Every per-element value is computed by the same expression in the same
order; chunking only changes which of those values ``min`` compares when, and
``min`` selects rather than computes.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import drag as bd


def _reference(
    positions: np.ndarray, selected: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """The pre-chunking ``proportional_set`` body, verbatim: the one-shot
    broadcast this module's output is measured against."""
    delta = positions[None, :, :] - positions[selected][:, None, :]
    distance = np.sqrt((delta**2).sum(axis=2)).min(axis=0)
    weights = bd.falloff(distance, radius)
    weights[selected] = 1.0
    keep = np.flatnonzero(weights > 0.0)
    return keep.astype("i4"), weights[keep]


class _CountingNp:
    """The module's ``np``, counting ``minimum`` -- the running-minimum fold is
    one call per chunk after the first, so the count is the proof the chunked
    path genuinely ran rather than degenerating to one big broadcast."""

    def __init__(self) -> None:
        self.minimum_calls = 0

    def __getattr__(self, name: str):
        return getattr(np, name)

    def minimum(self, *args, **kwargs):
        self.minimum_calls += 1
        return np.minimum(*args, **kwargs)


def _mesh(rng: np.random.Generator, verts: int, sel: int) -> tuple[np.ndarray, np.ndarray]:
    positions = rng.normal(size=(verts, 3)).astype("f8")
    selected = rng.choice(verts, size=sel, replace=False).astype("i8")
    return positions, selected


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("verts", "sel"),
    [
        # Chunk-boundary shapes against FALLOFF_CHUNK_PAIRS = 7 * verts below:
        # under one chunk row-count, an exact multiple of it, one past it, and
        # a selection the chunk size does not divide.
        (50, 3),
        (50, 14),
        (50, 15),
        (50, 26),
        (200, 41),
    ],
)
def test_chunked_matches_the_direct_broadcast_bit_for_bit(monkeypatch, seed, verts, sel) -> None:
    monkeypatch.setattr(bd, "FALLOFF_CHUNK_PAIRS", 7 * verts)  # 7 selected rows per chunk
    positions, selected = _mesh(np.random.default_rng(seed), verts, sel)
    for radius in (0.5, 1.5, 10.0):
        got_verts, got_weights = bd.proportional_set(positions, selected, radius)
        want_verts, want_weights = _reference(positions, selected, radius)
        assert np.array_equal(got_verts, want_verts), (seed, radius)
        # array_equal, never allclose: the whole point is bit identity.
        assert np.array_equal(got_weights, want_weights), (seed, radius)


def test_the_chunk_loop_actually_runs_more_than_once(monkeypatch) -> None:
    """A chunk size the test forces below the selection means the fold *must*
    iterate -- and counting the ``np.minimum`` folds is the only way to see
    that from outside, since the result is identical either way."""
    counting = _CountingNp()
    monkeypatch.setattr(bd, "np", counting)
    monkeypatch.setattr(bd, "FALLOFF_CHUNK_PAIRS", 1)  # one selected row per chunk
    positions, selected = _mesh(np.random.default_rng(3), 40, 9)

    got_verts, got_weights = bd.proportional_set(positions, selected, 2.0)
    assert counting.minimum_calls >= len(selected) - 1

    want_verts, want_weights = _reference(positions, selected, 2.0)
    assert np.array_equal(got_verts, want_verts)
    assert np.array_equal(got_weights, want_weights)


def test_the_pair_budget_still_declines_rather_than_chunking_forever() -> None:
    """``MAX_FALLOFF_PAIRS`` survives as the *time* cap: bounded memory does
    not make a quadratic search fast, so past the budget the drag is still an
    ordinary hard one."""
    positions = np.zeros((10, 3))
    saved, bd.MAX_FALLOFF_PAIRS = bd.MAX_FALLOFF_PAIRS, 1
    try:
        verts, weights = bd.proportional_set(positions, np.array([0, 1]), 1.0)
    finally:
        bd.MAX_FALLOFF_PAIRS = saved
    assert verts.tolist() == [0, 1]
    assert weights.tolist() == [1.0, 1.0]


def test_the_set_comes_back_in_vertex_index_order_with_selected_at_weight_one() -> None:
    """What the docstring now states, pinned: the order is vertex-index order
    -- ``flatnonzero``'s -- and the selected vertices are *in* the set at
    weight 1, not sorted to its front."""
    positions = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]], dtype="f8"
    )
    verts, weights = bd.proportional_set(positions, np.array([3]), 1.5)
    assert verts.tolist() == [2, 3], "index order, selection last here"
    assert float(weights[1]) == 1.0
    assert 0.0 < float(weights[0]) < 1.0
