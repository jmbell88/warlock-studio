"""The free-list prune, against the shape it replaced -- twice over.

The pairwise containment walk is unavoidable -- containment is a relation, not
an order -- but deleting in place cost an O(n) memmove per removal, so a pack
that split its free list into a few hundred rectangles paid O(n**3) to prune
them. Marking-and-filtering fixed that. Batch 7
(docs/measurements/2026-09-06-native-batch-7-candidates.md) went further:
after any prune no survivor contains another, so on the next placement only
pairs touching a piece the last split produced can newly fire, and
``_prune`` was restricted to exactly those pairs. Nothing about the *answer*
moved either time: the survivor set and its order are the same, including the
case that decides it.
"""

from __future__ import annotations

import random
from bisect import bisect_right

from warlock.studio.packwright import maxrects
from warlock.studio.packwright.layout import next_pot
from warlock.studio.packwright.maxrects import Placement, Rect, _contains, _fits, _score, _split


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


def _all_pairs_prune(free: list[Rect]) -> list[Rect]:
    """The shipped-before-batch-7 all-pairs prune, spelled out here rather than
    called: `maxrects._prune` no longer has this signature, and this is the
    thing the restricted version has to keep agreeing with."""
    out = list(free)
    count = len(out)
    if count < 2:
        return out
    bounds = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in out]
    dead = [False] * count
    dropped = False
    for i in range(count):
        if dead[i]:
            continue
        ix, iy, ir, ib = bounds[i]
        for j in range(i + 1, count):
            if dead[j]:
                continue
            jx, jy, jr, jb = bounds[j]
            if jx <= ix and jy <= iy and jr >= ir and jb >= ib:
                dead[i] = True
                dropped = True
                break
            if ix <= jx and iy <= jy and ir >= jr and ib >= jb:
                dead[j] = True
                dropped = True
    if dropped:
        return [rect for rect, gone in zip(out, dead, strict=True) if not gone]
    return out


def _pruned(rects: list[Rect], fresh: list[bool] | None = None) -> list[Rect]:
    out = list(rects)
    if fresh is None:
        fresh = [True] * len(out)
    maxrects._prune(out, fresh)
    return out


def test_the_survivors_match_the_original_on_random_lists() -> None:
    """All-fresh is the case where the restricted prune degenerates to the
    all-pairs one -- every pair is old-vs-new or new-vs-new because there is
    no "old"."""
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


def test_identical_rectangles_old_versus_fresh_both_orders() -> None:
    """The duplicate-drop rule has to hold whichever rectangle is the fresh
    one, since a real pack can produce the duplicate on either side."""
    same = [Rect(0, 0, 4, 4), Rect(0, 0, 4, 4)]
    assert _pruned(same, fresh=[False, True]) == [Rect(0, 0, 4, 4)]
    assert _pruned(same, fresh=[True, False]) == [Rect(0, 0, 4, 4)]


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


def test_no_fresh_entries_does_zero_work() -> None:
    """Fails against the unfixed (all-pairs) code: with every entry marked
    old, the restricted prune must not touch a contained pair at all -- the
    caller is what guarantees the postcondition (no survivor of a real pack
    ever contains another), so a list that violates it because nothing here
    is actually fresh is left exactly as it came in."""
    contained = [Rect(0, 0, 4, 4), Rect(1, 1, 2, 2)]
    assert _pruned(contained, fresh=[False, False]) == contained


def _pack_new_only(items: list[tuple[str, int, int]], width: int, height: int):
    """The restricted-prune pack, spelled out independently of `maxrects.pack`
    so the parity test below is checking two implementations, not one
    implementation against itself. Mirrors
    scripts/bench_native.py::_pack_new_only exactly."""
    free = [Rect(0, 0, int(width), int(height))]
    placed: list[Placement] = []
    free_snapshots: list[list[Rect]] = []
    for key, w, h in items:
        w, h = int(w), int(h)
        best_score = None
        best: Rect | None = None
        for candidate in free:
            if not _fits(candidate, w, h):
                continue
            score = _score(candidate, w, h)
            if best_score is None or score < best_score:
                best_score, best = score, candidate
        if best is None:
            return None, free_snapshots
        used = Rect(best.x, best.y, w, h)
        placed.append(Placement(key=key, x=used.x, y=used.y, w=w, h=h))
        remaining: list[Rect] = []
        fresh: list[bool] = []
        for candidate in free:
            pieces = _split(candidate, used)
            if pieces is None:
                remaining.append(candidate)
                fresh.append(False)
            else:
                remaining.extend(pieces)
                fresh.extend([True] * len(pieces))
        count = len(remaining)
        bounds = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in remaining]
        dead = [False] * count
        new_idx = [k for k in range(count) if fresh[k]]
        for i in range(count):
            if dead[i]:
                continue
            ix, iy, ir, ib = bounds[i]
            js = range(i + 1, count) if fresh[i] else new_idx[bisect_right(new_idx, i):]
            for j in js:
                if dead[j]:
                    continue
                jx, jy, jr, jb = bounds[j]
                if jx <= ix and jy <= iy and jr >= ir and jb >= ib:
                    dead[i] = True
                    break
                if ix <= jx and iy <= jy and ir >= jr and ib >= jb:
                    dead[j] = True
        free = [rect for rect, gone in zip(remaining, dead, strict=True) if not gone]
        free_snapshots.append(list(free))
    return placed, free_snapshots


def _all_pairs_pack(items: list[tuple[str, int, int]], width: int, height: int):
    """The pre-batch-7 shipped pack, spelled out here (all-pairs prune every
    placement) so the parity test drives both algorithms independently of
    what `maxrects.pack` currently does."""
    free = [Rect(0, 0, int(width), int(height))]
    placed: list[Placement] = []
    free_snapshots: list[list[Rect]] = []
    for key, w, h in items:
        w, h = int(w), int(h)
        best_score = None
        best: Rect | None = None
        for candidate in free:
            if not _fits(candidate, w, h):
                continue
            score = _score(candidate, w, h)
            if best_score is None or score < best_score:
                best_score, best = score, candidate
        if best is None:
            return None, free_snapshots
        used = Rect(best.x, best.y, w, h)
        placed.append(Placement(key=key, x=used.x, y=used.y, w=w, h=h))
        remaining: list[Rect] = []
        for candidate in free:
            pieces = _split(candidate, used)
            if pieces is None:
                remaining.append(candidate)
            else:
                remaining.extend(pieces)
        free = _all_pairs_prune(remaining)
        free_snapshots.append(list(free))
    return placed, free_snapshots


def test_restricted_prune_matches_all_pairs_on_a_seeded_pack() -> None:
    """The parity test: drive the old all-pairs prune and the new
    fresh-restricted one through the same seeded random pack (as the bench
    does: random.Random(7), 8..64 px, atlas side from `next_pot`), and check
    not just the final placements but the free-rect survivor list produced
    after *every* placement -- so a restriction that happened to agree only
    at the end could not slip past this."""
    rng = random.Random(7)
    items = maxrects.order(
        [(f"s{i}", rng.randrange(8, 65), rng.randrange(8, 65)) for i in range(300)]
    )
    area = sum(w * h for _key, w, h in items)
    side = next_pot(int(area ** 0.5) + 1)
    # Grow the atlas until both algorithms can actually place everything --
    # the point here is comparing the two algorithms, not the size search.
    while True:
        old_placed, old_snapshots = _all_pairs_pack(items, side, side)
        new_placed, new_snapshots = _pack_new_only(items, side, side)
        if old_placed is not None and new_placed is not None:
            break
        side *= 2

    assert old_placed == new_placed
    assert old_snapshots == new_snapshots

    # And the shipped `maxrects.pack` -- which now *is* the restricted
    # algorithm -- agrees with both.
    assert maxrects.pack(items, side, side) == old_placed
