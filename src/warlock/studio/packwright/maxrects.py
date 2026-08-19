"""MaxRects bin packing, made deterministic on purpose.

The algorithm is Jukka Jylanki's MAXRECTS-BSSF: keep a list of maximal free
rectangles, place each item into the free rectangle whose *shorter* leftover
side is smallest, then split every free rectangle the placement overlapped and
drop the ones now contained in another.

**Determinism is a contract, not a happy accident, and it is what makes an
unchanged document re-export to identical bytes.** Three things guarantee it and
all three are load-bearing:

1. The caller supplies items in a canonical order -- descending max side, then
   descending area, then ascending key. :func:`order` is that comparison, here
   rather than at the call site so there is one of it.
2. Ties between free rectangles break on ``(short leftover, long leftover,
   lowest y, lowest x)``. Without the last two, two rectangles that fit equally
   well would be chosen by list position, which the split step reorders.
3. The free list is built and walked in index order. There is no set and no dict
   anywhere in this module, because iteration order over either is exactly the
   kind of thing that is stable until the day it is not.

**All or nothing.** A pack that cannot fit every item returns ``None`` rather
than a partial result: :mod:`.layout` responds by trying a larger atlas, and a
partial layout would have to be distinguished from a complete one by counting,
which is how half an atlas gets written.

**No rotation in v1.** Consumers that ignore ``"rotated": true`` render the
sprite sideways with nothing to say why, and the JSON schema has to carry the
flag whether or not anything sets it. It is a later option, not a default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True, slots=True)
class Placement:
    key: str
    x: int
    y: int
    w: int
    h: int


def order(items: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """The canonical input order: biggest-first, ties broken by key.

    Biggest-first because MaxRects packs badly when a large item arrives after
    the space it needed has been fragmented. Ties broken by *key* rather than
    left alone, because "stable sort of whatever order the caller happened to
    build the list in" is not an order -- and the caller's list comes from a
    document whose sources the user reorders.
    """
    return sorted(items, key=lambda item: (-max(item[1], item[2]), -item[1] * item[2], item[0]))


def _fits(free: Rect, w: int, h: int) -> bool:
    return free.w >= w and free.h >= h


def _score(free: Rect, w: int, h: int) -> tuple[int, int, int, int]:
    """Best-short-side-fit, with the two positional tie-breaks appended."""
    leftover_h = free.w - w
    leftover_v = free.h - h
    return (
        min(leftover_h, leftover_v),
        max(leftover_h, leftover_v),
        free.y,
        free.x,
    )


def _split(free: Rect, used: Rect) -> list[Rect] | None:
    """The up-to-four maximal rectangles left of ``free`` after ``used`` lands.

    ``None`` means they do not overlap and ``free`` survives untouched, which
    the caller distinguishes from an empty list -- a rectangle entirely consumed
    by the placement.
    """
    if used.x >= free.right or used.right <= free.x:
        return None
    if used.y >= free.bottom or used.bottom <= free.y:
        return None

    out: list[Rect] = []
    if used.y > free.y:
        out.append(Rect(free.x, free.y, free.w, used.y - free.y))
    if used.bottom < free.bottom:
        out.append(Rect(free.x, used.bottom, free.w, free.bottom - used.bottom))
    if used.x > free.x:
        out.append(Rect(free.x, free.y, used.x - free.x, free.h))
    if used.right < free.right:
        out.append(Rect(used.right, free.y, free.right - used.right, free.h))
    return out


def _contains(outer: Rect, inner: Rect) -> bool:
    """Is *inner* wholly inside *outer*? The readable statement of the rule.

    :func:`_prune` unpacks the bounds and inlines this rather than calling it
    per pair -- four property reads per comparison over a few hundred
    rectangles was measurable -- so this stays as the definition the tests
    check the inlined form against.
    """
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.right <= outer.right
        and inner.bottom <= outer.bottom
    )


def _prune(free: list[Rect]) -> None:
    """Drop every free rectangle wholly inside another.

    Without this the list grows without bound and, worse, the same space is
    offered several times over, so the tie-breaks above stop being a total
    order over *distinct* candidates.

    **Marked and filtered rather than deleted in place, and the result is the
    same list.** The pairwise walk is unavoidable -- this is a containment
    relation, not an order -- but the old version's ``del free[i]`` inside it
    was an O(n) memmove per removal, so a pack that split its free list into a
    few hundred rectangles paid O(n**3) to prune them. The traversal order and
    the survivor set are preserved exactly, including the case that decides it:
    of two *identical* rectangles the earlier is the one dropped, because
    ``_contains(free[j], free[i])`` fires first. A naive "drop anything another
    rect contains" would drop both and change the pack.

    The bounds are unpacked once per rectangle instead of through four property
    reads per comparison, which is the rest of the win and changes nothing.
    """
    count = len(free)
    if count < 2:
        return
    bounds = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in free]
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
        free[:] = [rect for rect, gone in zip(free, dead, strict=True) if not gone]


def pack(items: list[tuple[str, int, int]], width: int, height: int) -> list[Placement] | None:
    """Place every ``(key, w, h)`` inside ``width`` x ``height``, or ``None``.

    The caller is expected to have run :func:`order` first; this does not sort,
    because the *input order is part of the result* and hiding that would make
    a caller that forgot get a silently different -- still valid -- atlas.
    """
    free = [Rect(0, 0, int(width), int(height))]
    placed: list[Placement] = []

    for key, w, h in items:
        w, h = int(w), int(h)
        if w <= 0 or h <= 0:
            raise ValueError(f"{key!r} has no area to pack")
        best_score: tuple[int, int, int, int] | None = None
        best: Rect | None = None
        for candidate in free:
            if not _fits(candidate, w, h):
                continue
            score = _score(candidate, w, h)
            if best_score is None or score < best_score:
                best_score, best = score, candidate
        if best is None:
            return None

        used = Rect(best.x, best.y, w, h)
        placed.append(Placement(key=key, x=used.x, y=used.y, w=w, h=h))

        remaining: list[Rect] = []
        for candidate in free:
            pieces = _split(candidate, used)
            if pieces is None:
                remaining.append(candidate)
            else:
                remaining.extend(pieces)
        free = remaining
        _prune(free)

    return placed
