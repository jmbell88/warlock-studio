"""The traced contours against the chained ones.

Not "identical lists", and deliberately so. Loop order, starting vertex and
winding are unspecified by ``contours()``'s contract -- ``_chain`` starts from
``next(iter(set))`` -- and at a checkerboard corner, where two regions touch
diagonally and four boundary edges meet, the two implementations legitimately
partition the same edges into different loops: the tracer turns left and keeps
the regions apart, the reference resolves it by dict order.

What cannot differ is the geometry. So the bar here is the *set of unit edges*,
which is the actual content of an outline, plus the structural properties the
pane relies on: every loop closed, every step a unit step, no loop shorter than
three points. Loop count is asserted only where the two must agree -- on masks
with no diagonal corner at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import native
from warlock.studio.inker import selection as sel

needs_dll = pytest.mark.skipif(not native.available(), reason="warlockc.dll not built")


def _both(mask: sel.SelectionMask, monkeypatch):
    """The traced loops and the chained ones, for the same mask."""
    monkeypatch.setattr(native, "available", lambda: True)
    traced = mask.contours()
    monkeypatch.setattr(native, "available", lambda: False)
    chained = mask.contours()
    return traced, chained


def _edges(loops) -> set[frozenset[tuple[int, int]]]:
    out: set[frozenset[tuple[int, int]]] = set()
    for loop in loops:
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            out.add(frozenset((a, b)))
    return out


def _assert_well_formed(loops) -> None:
    """Closed, unit-stepped, and no shorter than the reference's floor.

    Not "visits each vertex once": at a pinch -- one region touching itself
    diagonally -- the left-turn rule keeps the walk on that region, so the loop
    passes through the pinch vertex twice. That is the correct outline, and it
    is why the walk ends when it *arrives back at its start* rather than after
    some vertex count.
    """
    for loop in loops:
        assert len(loop) > 2, "the reference drops loops of two points or fewer"
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, (a, b)


def _diagonal_corner(inside: np.ndarray) -> bool:
    """Whether any 2x2 window holds two diagonally-touching regions."""
    a, b = inside[:-1, :-1], inside[:-1, 1:]
    c, d = inside[1:, :-1], inside[1:, 1:]
    return bool(((a & d & ~b & ~c) | (b & c & ~a & ~d)).any())


def _check(mask: sel.SelectionMask, monkeypatch) -> None:
    """Every property the two paths must share, for one mask."""
    traced, chained = _both(mask, monkeypatch)
    _assert_well_formed(traced)
    assert _edges(traced) == _edges(chained)
    if not _diagonal_corner(mask.mask >= sel.INSIDE):
        assert len(traced) == len(chained)


# --- the shapes the app actually produces ------------------------------------


@needs_dll
def test_a_marquee_traces_the_same_edges(monkeypatch):
    _check(sel.SelectionMask.from_rect((64, 48), (7, 5, 40, 33)), monkeypatch)


@needs_dll
def test_an_ellipse_traces_the_same_edges(monkeypatch):
    _check(sel.SelectionMask.from_ellipse((64, 64), (3, 3, 61, 48)), monkeypatch)


@needs_dll
def test_a_lasso_traces_the_same_edges(monkeypatch):
    """A polygon with a concave notch and a shallow diagonal -- the staircase
    the ants are actually drawn along."""
    points = [(4, 4), (58, 9), (40, 30), (55, 52), (20, 60), (26, 33), (6, 40)]
    _check(sel.SelectionMask.from_polygon((64, 64), points), monkeypatch)


@needs_dll
def test_a_wand_result_traces_the_same_edges(monkeypatch):
    """Ragged, one-pixel features everywhere: the hardest real input, and the
    one whose loop count is least likely to survive a diagonal corner."""
    rng = np.random.default_rng(17)
    plane = rng.integers(0, 256, (48, 48), dtype=np.uint8)
    pixels = np.zeros((48, 48, 4), dtype=np.uint8)
    pixels[..., 0] = plane
    pixels[..., 3] = 255
    _check(sel.magic_wand(pixels, (24, 24), tolerance=90, contiguous=False), monkeypatch)


@needs_dll
def test_a_feathered_blob_traces_the_same_edges(monkeypatch):
    """The threshold is what the boundary is *about*: a feathered mask has a
    soft ring, and both paths have to cut it at exactly INSIDE."""
    blob = sel.SelectionMask.from_ellipse((64, 64), (12, 12, 52, 52)).feathered(6.0)
    _check(blob, monkeypatch)


# --- the edges of the lattice ------------------------------------------------


@needs_dll
def test_a_one_pixel_selection_is_four_vertices_either_way(monkeypatch):
    mask = sel.SelectionMask.from_rect((8, 8), (3, 3, 4, 4))
    traced, chained = _both(mask, monkeypatch)
    assert sorted(traced[0]) == [(3, 3), (3, 4), (4, 3), (4, 4)]
    assert sorted(traced[0]) == sorted(chained[0])


@needs_dll
def test_a_full_canvas_selection_hugs_the_border(monkeypatch):
    """What the reference's ``np.pad`` buys and the tracer gets from treating
    everything off the image as outside. Getting it wrong loses the border
    edges entirely, which looks like no selection at all."""
    mask = sel.SelectionMask.full(16, 12)
    traced, chained = _both(mask, monkeypatch)
    assert _edges(traced) == _edges(chained)
    assert len(traced) == 1 and len(traced[0]) == 2 * (16 + 12)


@needs_dll
def test_an_empty_selection_has_no_outline_on_either_path(monkeypatch):
    mask = sel.SelectionMask(np.zeros((8, 8), dtype=np.uint8))
    assert _both(mask, monkeypatch) == ([], [])


@needs_dll
@pytest.mark.parametrize("size", [(1, 1), (1, 9), (9, 1)])
def test_a_degenerate_canvas_still_traces(monkeypatch, size):
    """A one-pixel-tall mask counts its top and bottom border in the same term
    of the edge count -- the buffer is exact, so an off-by-one there is a -1
    from the kernel and a silent fall back to numpy."""
    width, height = size
    mask = sel.SelectionMask.full(width, height)
    traced, chained = _both(mask, monkeypatch)
    assert _edges(traced) == _edges(chained)
    _assert_well_formed(traced)


@needs_dll
def test_two_separate_regions_stay_two_loops(monkeypatch):
    a = sel.SelectionMask.from_rect((32, 16), (2, 2, 8, 14))
    b = sel.SelectionMask.from_rect((32, 16), (20, 2, 28, 14))
    both = a.combined(b, "add")
    traced, chained = _both(both, monkeypatch)
    assert len(traced) == len(chained) == 2
    assert _edges(traced) == _edges(chained)


@needs_dll
def test_a_hole_is_its_own_loop(monkeypatch):
    """A ring: the outer boundary and the inner one are separate loops, and the
    inner one winds the other way. Nothing consumes winding -- the ants walk
    either direction -- so only the edges are asserted."""
    outer = sel.SelectionMask.from_rect((32, 32), (4, 4, 28, 28))
    inner = sel.SelectionMask.from_rect((32, 32), (12, 12, 20, 20))
    ring = outer.combined(inner, "subtract")
    traced, chained = _both(ring, monkeypatch)
    assert len(traced) == len(chained) == 2
    assert _edges(traced) == _edges(chained)


# --- the corner the two are allowed to disagree about ------------------------


@needs_dll
def test_a_checkerboard_corner_keeps_the_edges_and_may_split_the_loops(monkeypatch):
    """Two pixels touching only at a corner. Four boundary edges meet at that
    vertex and a tracer has to pick a rule; turning left keeps each region's
    walk on its own region, which is two loops where the reference's dict order
    produced one. The edges are the same either way, and so are the ants."""
    plane = np.zeros((8, 8), dtype=np.uint8)
    plane[3, 3] = 255
    plane[4, 4] = 255
    mask = sel.SelectionMask(plane)

    traced, chained = _both(mask, monkeypatch)
    _assert_well_formed(traced)
    assert _edges(traced) == _edges(chained)
    assert len(traced) == 2
    assert sum(len(loop) for loop in traced) == sum(len(loop) for loop in chained)


@needs_dll
def test_a_checkerboard_field_keeps_every_edge(monkeypatch):
    """The pathological version: every vertex in the middle is ambiguous."""
    plane = np.indices((12, 12)).sum(axis=0) % 2 == 0
    mask = sel.SelectionMask((plane * 255).astype(np.uint8))
    traced, chained = _both(mask, monkeypatch)
    _assert_well_formed(traced)
    assert _edges(traced) == _edges(chained)


# --- and everything else -----------------------------------------------------


@needs_dll
@pytest.mark.parametrize("seed", range(50))
def test_a_random_mask_traces_the_same_edges(monkeypatch, seed):
    """Fifty of them, because the shapes above are all *connected* and the
    interesting failures -- a loop the scan starts in the wrong place, a
    boundary edge visited twice -- need several regions at once."""
    rng = np.random.default_rng(seed)
    plane = rng.random((24, 24)) < 0.45
    # Blurred so the regions are blobs rather than confetti, then thresholded
    # back: confetti is all corners and nothing else.
    plane = sel.SelectionMask((plane * 255).astype(np.uint8)).feathered(1.2)
    _check(plane, monkeypatch)


@needs_dll
def test_the_pane_gets_tuples_it_can_hash(monkeypatch):
    """``contours()`` returns lists of *tuples*: the cache keys on them and the
    behaviour tests compare them to literals, so a list of arrays would pass
    every geometric assertion here and fail at the call site."""
    monkeypatch.setattr(native, "available", lambda: True)
    loops = sel.SelectionMask.from_rect((8, 8), (2, 2, 6, 6)).contours()
    assert isinstance(loops[0][0], tuple)
    assert all(isinstance(value, int) for value in loops[0][0])
    assert len(set(loops[0])) == len(loops[0])


@needs_dll
def test_a_refused_trace_falls_back_to_the_chain(monkeypatch):
    """The kernel returns -1 rather than writing past a buffer, and -1 has to
    reach the numpy path rather than an empty selection."""
    monkeypatch.setattr(native, "available", lambda: True)
    monkeypatch.setattr(native, "contours", lambda *args: -1)
    mask = sel.SelectionMask.from_rect((16, 16), (4, 4, 12, 12))
    assert sel._contours_native(mask.mask) is None
    assert len(mask.contours()) == 1
