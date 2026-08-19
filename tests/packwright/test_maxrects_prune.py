"""The free-list prune, against the shape it replaced.

The pairwise containment walk is unavoidable -- containment is a relation, not
an order -- but deleting in place cost an O(n) memmove per removal, so a pack
that split its free list into a few hundred rectangles paid O(n**3) to prune
them. What matters here is that nothing about the *answer* moved: the survivor
set and its order are the same, including the case that decides it.
"""

from __future__ import annotations

import random

from warlock.studio.packwright import maxrects
from warlock.studio.packwright.maxrects import Rect, _contains


def _reference(free: list[Rect]) -> list[Rect]:
    """The delete-in-place original, kept here as the thing to agree with."""
    free = list(free)
    i = 0
    while i < len(free):
        j = i + 1
        while j < len(free):
            if _contains(free[j], free[i]):
                del free[i]
                i -= 1
                break
            if _contains(free[i], free[j]):
                del free[j]
            else:
                j += 1
        i += 1
    return free


def _pruned(rects: list[Rect]) -> list[Rect]:
    out = list(rects)
    maxrects._prune(out)
    return out


def test_the_survivors_match_the_original_on_random_lists() -> None:
    rng = random.Random(19)
    for _ in range(200):
        rects = [
            Rect(rng.randrange(0, 12), rng.randrange(0, 12),
                 rng.randrange(1, 8), rng.randrange(1, 8))
            for _ in range(rng.randrange(2, 14))
        ]
        assert _pruned(rects) == _reference(rects)


def test_of_two_identical_rectangles_the_earlier_is_dropped() -> None:
    """The case that decides the rewrite: a naive "drop anything another rect
    contains" would drop both and change the pack."""
    same = [Rect(0, 0, 4, 4), Rect(0, 0, 4, 4)]
    assert _pruned(same) == [Rect(0, 0, 4, 4)]
    assert _pruned(same) == _reference(same)


def test_nothing_contained_means_nothing_dropped() -> None:
    apart = [Rect(0, 0, 2, 2), Rect(4, 4, 2, 2), Rect(8, 0, 3, 1)]
    assert _pruned(apart) == apart


def test_a_short_list_is_left_alone() -> None:
    assert _pruned([]) == []
    assert _pruned([Rect(0, 0, 1, 1)]) == [Rect(0, 0, 1, 1)]


def test_the_pack_is_unchanged_on_a_real_run() -> None:
    """The determinism pin the packer already leans on, restated against the
    prune: the same items in the same order produce the same atlas."""
    rng = random.Random(7)
    items = [(f"s{i}", rng.randrange(3, 20), rng.randrange(3, 20)) for i in range(60)]
    first = maxrects.pack(items, 128, 128)
    second = maxrects.pack(items, 128, 128)
    assert first == second
