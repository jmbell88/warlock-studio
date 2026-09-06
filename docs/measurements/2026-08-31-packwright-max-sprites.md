# Packwright's sprite ceiling: what MaxRects actually costs at 4096

2026-08-31. Machine: Windows 11 Pro 26200, Python 3.13.13, numpy and Pillow
from the project's own `uv` environment. Driven by a throwaway script that
calls `packwright.maxrects.pack` directly — the door `maxrects_layout` comes
through — with `layout.order` applied first, so what is timed is the real
search and not an unsorted worst case. Items are random 8–64 px rectangles,
seeded (`random.Random(7)`) so the run reproduces; the atlas side is the next
power of two above twice the total item area, which is roughly what
`_candidate_sizes` tries first.

## Why this document exists

`layout.MAX_SPRITES` carried this claim:

> MaxRects is quadratic in the free-rect list, so a document claiming a million
> sources is not a slow pack but a hung frame thread; 4096 is well past any
> real sheet (a 64x64 grid at the 8192px ceiling) and short of where the search
> stops answering.

The first two clauses are right. **"Short of where the search stops answering"
was never measured, and it is false.** A constant the app advertises as a
supported ceiling is exactly the kind this repo requires a document for before
it moves, so here is the measurement that moves it.

## The numbers

| sprites | atlas side | one `pack()` |
| ------: | ---------: | -----------: |
|     128 |       1024 |      0.016 s |
|     256 |       1024 |      0.110 s |
|     512 |       2048 |      0.770 s |
|    1024 |       2048 |      4.687 s |
|    2048 |       4096 |     30.431 s |
|    4096 |       4096 |    190.846 s |

Roughly 6× per doubling — super-quadratic in practice, because `_prune` is
`O(F²)` per placement and the free-rect count `F` itself grows with the number
of items placed.

**And that is one `pack()`.** `maxrects_layout` calls it once per candidate
size in the `_candidate_sizes` loop, so a document whose first candidate does
not fit pays the figure above more than once.

## What changed

`MAX_SPRITES` 4096 → **1024**.

1024 is the largest count whose single pack stays under five seconds, which is
the most a background task should take before the "packing…" label stops
reading as progress and starts reading as a hang. It is still a 32×32 grid of
tiles, comfortably past any hand-assembled atlas; the 8192 px atlas ceiling is
what actually binds for large tiles, and it is unchanged.

Packing runs on a task thread (`packwright_mode.request_pack` submits it), so
none of this ever froze the frame loop — the cost was in the user waiting.
The frame-thread half of the same feature, the tileset-import popup's dedup
preview, was a separate defect and is now cached per input rather than redone
per frame.

## What would lift the ceiling again

The free-rect search is the whole cost and it is all linear scans: `_score`
walks every free rect per item, and `_prune` compares every pair. Bucketing
free rects by position so neither has to scan the whole list is the standard
fix and would change the shape of this table rather than its constant factor.
Until someone does that, 1024 is what the search answers in reasonable time,
and this file is the evidence for the number.

## 2026-09-06: the ceiling returns to 4096

Not the bucketing this file predicted. `_prune` was re-checking every pair of
free rectangles per placement when, after any prune, no survivor contains
another -- so only pairs touching a piece the last `_split` produced can newly
fire. Restricting the loop to those pairs is exact by induction, and it is
92% of a pack's time at 1024 sprites. See
docs/measurements/2026-09-06-native-batch-7-candidates.md §1 for the full
account; the restricted-prune figures, minimum of seven runs in a fresh
process:

| sprites | `prune_new_only` |
|---|---|
| 1024 | 297 ms |
| 2048 | 977 ms |
| 4096 | 3 589 ms |

4096 at 3.6 seconds is back under this document's own five-second rule, so
`layout.MAX_SPRITES` is 4096 again.
