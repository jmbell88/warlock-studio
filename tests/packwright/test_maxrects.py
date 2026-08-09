"""The packer, and the determinism contract that makes re-export reproducible.

Correctness here is cheap to state -- nothing overlaps and everything is inside
the bin -- so most of this file is about the *other* property: that the same
sprites produce the same atlas, byte for byte, however the document happened to
be ordered. Three things guarantee it and each has a test: the canonical input
order, the positional tie-breaks, and the absence of any set or dict iteration.
"""

from __future__ import annotations

import ast
import inspect
import random

import pytest

from warlock.studio.packwright import maxrects
from warlock.studio.packwright.maxrects import Rect, order, pack


def _items(sizes):
    return [(f"s{i:02d}", w, h) for i, (w, h) in enumerate(sizes)]


def _overlaps(a, b) -> bool:
    return not (
        a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y
    )


def _check(placed, width, height, items):
    assert placed is not None
    assert {p.key for p in placed} == {key for key, _w, _h in items}
    sizes = {key: (w, h) for key, w, h in items}
    for p in placed:
        assert sizes[p.key] == (p.w, p.h)
        assert p.x >= 0 and p.x + p.w <= width
        assert p.y >= 0 and p.y + p.h <= height
    for i, a in enumerate(placed):
        for b in placed[i + 1 :]:
            assert not _overlaps(a, b), f"{a} overlaps {b}"


# --- correctness --------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes",
    [
        [(10, 10)] * 16,
        [(31, 7), (7, 31), (16, 16), (3, 3), (40, 2), (2, 40)],
        [(1, 1)] * 64,
        [(63, 63)],
        [(20, 30), (30, 20), (10, 60), (60, 10), (25, 25), (25, 25)],
    ],
)
def test_nothing_overlaps_and_everything_is_inside(sizes):
    items = order(_items(sizes))
    placed = pack(items, 128, 128)
    _check(placed, 128, 128, items)


def test_a_perfect_fit_uses_every_pixel():
    """Four quadrants of a square: any correct packer places them exactly, and
    a packer that left a gap would fail this on area alone."""
    items = order(_items([(32, 32)] * 4))
    placed = pack(items, 64, 64)
    assert placed is not None
    assert sum(p.w * p.h for p in placed) == 64 * 64
    assert {(p.x, p.y) for p in placed} == {(0, 0), (32, 0), (0, 32), (32, 32)}


# --- all or nothing -----------------------------------------------------------


def test_a_pack_that_does_not_fit_returns_none_rather_than_a_partial():
    """``layout`` responds to None by trying a larger atlas. A partial result
    would have to be told from a complete one by counting, which is how half an
    atlas gets written."""
    assert pack(order(_items([(40, 40)] * 5)), 64, 64) is None


def test_an_item_larger_than_the_bin_returns_none():
    assert pack([("big", 100, 10)], 64, 64) is None


def test_a_zero_sized_item_is_refused_rather_than_placed():
    with pytest.raises(ValueError, match="no area"):
        pack([("empty", 0, 10)], 64, 64)


# --- determinism --------------------------------------------------------------


def test_shuffled_input_packs_identically_once_ordered():
    """The property the whole contract exists for. ``order`` is applied by
    ``layout``, so a document whose sources the user dragged around still
    exports the same bytes."""
    sizes = [(3 + (i * 7) % 29, 3 + (i * 11) % 23) for i in range(40)]
    items = _items(sizes)
    baseline = pack(order(items), 256, 256)
    rng = random.Random(1234)
    for _ in range(8):
        shuffled = items[:]
        rng.shuffle(shuffled)
        assert pack(order(shuffled), 256, 256) == baseline


def test_the_canonical_order_is_biggest_first_then_by_key():
    """Biggest-first because MaxRects packs badly when a large item arrives
    after the space it needed has been fragmented; by key rather than left
    alone, because "whatever order the caller built the list in" is not an
    order."""
    items = [("b", 10, 10), ("a", 10, 10), ("c", 20, 5), ("d", 4, 4)]
    assert [key for key, _w, _h in order(items)] == ["c", "a", "b", "d"]


def test_two_equally_good_free_rectangles_break_on_position():
    """A crafted tie: after one 32x32 lands in a 64x32 bin, the leftover is a
    single rectangle -- so this pins the *scoring*, which is what decides
    between candidates when several fit equally well."""
    a = Rect(0, 10, 40, 40)
    b = Rect(0, 0, 40, 40)
    # Same leftovers; the lower y wins, and the lower x after that.
    assert maxrects._score(b, 10, 10) < maxrects._score(a, 10, 10)
    assert maxrects._score(Rect(0, 0, 40, 40), 10, 10) < maxrects._score(
        Rect(5, 0, 40, 40), 10, 10
    )


def test_best_short_side_fit_prefers_the_tighter_rectangle():
    tight = Rect(0, 0, 11, 40)
    loose = Rect(0, 0, 40, 40)
    assert maxrects._score(tight, 10, 10) < maxrects._score(loose, 10, 10)


def test_the_module_iterates_no_set_or_dict():
    """Stated in the docstring and checked here, because iteration order over
    either is exactly the kind of thing that is stable until the day it is not.

    Parsed rather than grepped: the word "set" appears in prose throughout.
    """
    tree = ast.parse(inspect.getsource(maxrects))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.Set | ast.SetComp | ast.DictComp), (
            f"maxrects builds a {type(node).__name__}"
        )
        if isinstance(node, ast.Dict):
            raise AssertionError("maxrects builds a dict")


# --- the free list ------------------------------------------------------------


def test_a_rectangle_the_placement_misses_survives_untouched():
    """``_split`` returns None for no overlap, which the caller distinguishes
    from an empty list -- a rectangle entirely consumed by the placement."""
    assert maxrects._split(Rect(50, 50, 10, 10), Rect(0, 0, 10, 10)) is None
    assert maxrects._split(Rect(0, 0, 10, 10), Rect(0, 0, 10, 10)) == []


def test_pruning_drops_a_contained_rectangle():
    """Without it the free list grows without bound and, worse, offers the same
    space several times over -- so the tie-breaks stop being a total order over
    distinct candidates."""
    free = [Rect(0, 0, 10, 10), Rect(2, 2, 4, 4), Rect(0, 0, 20, 20)]
    maxrects._prune(free)
    assert free == [Rect(0, 0, 20, 20)]


def test_no_rotation_is_ever_emitted():
    """A consumer that ignores ``rotated: true`` renders the sprite sideways
    with nothing to say why. Later, behind an option; never a default."""
    items = order(_items([(40, 8)] * 2))
    placed = pack(items, 64, 64)
    assert placed is not None
    for p in placed:
        assert (p.w, p.h) == (40, 8)
