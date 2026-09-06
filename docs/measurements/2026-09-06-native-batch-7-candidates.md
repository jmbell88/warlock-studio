# Native kernel batch 7 — two candidates, measured. No C written, no fix landed.

> **Landed the same day** (commit after `867b05b1`): both Python fixes below are
> now the shipped code, so the bench's `shipped` and variant columns read the
> same. `MAX_SPRITES` is 4096 again. Regression tests:
> `tests/packwright/test_maxrects_prune.py`, `tests/plotter/test_wang_field_hoisted.py`.
> Default lane 18 463 passed. The tables below are the before-and-after record.

2026-09-06, at `d6f8d148`. Machine: Windows 11, numpy 2.x, Python 3.13,
`vendor/warlockc/warlockc.dll` present (ABI 10) — neither site below has a
kernel, so the DLL changes nothing here. Every figure is the **minimum of seven
runs in a fresh process** through `scripts/bench_native.py --sweep`, except the
two rows marked as taken outside the declared sweep (three runs, minimum).

Batch 6 closed with nothing on its list clearing a gate. These are the two
strongest *unbenched* shapes left in the tree, both sequential integer work
numpy cannot express and both, on paper, the ideal kernel: `packwright`'s
free-rectangle prune and `plotter`'s Wang re-fit. **Both clear their gates as
shipped, and both fall back under them after a pure-Python change that
reproduces the shipped answer exactly.** So this batch, like the last, writes
no C — and records why, so the two kernels are not re-proposed on the strength
of the shipped numbers alone.

## Summary

| # | Candidate | Gate | Shipped | After the Python fix | Verdict |
|---|---|---|---|---|---|
| B8 | `packwright/maxrects.pack` @1024 sprites | >1.0 s | **4 432 ms** | **297 ms** | **fix in Python, no kernel** — §1 |
| B9 | `plotter/terrain.paint_wang_cells`, 128-cell diagonal drag | >100 ms | **131 ms** | **68 ms** | **fix in Python, no kernel** — §2 |

Both gates were written into the bench case docstrings before the first
number was taken. New bench cases: `packwright_pack` (variants `shipped`,
`prune_new_only`) and `wang_drag` (variants `shipped`, `holds_hoisted`). Each
variant builder asserts its result equals the shipped one before anything is
timed. **Nothing under `src/` changed.**

---

## §1 — B8, `maxrects.pack`: 92 % is a prune re-checking pairs that cannot fire

`docs/measurements/2026-08-31-packwright-max-sprites.md` cut `MAX_SPRITES`
from 4096 to 1024 on a 4.7 s pack and named the shape: `_prune` is O(F²) per
placement with F growing. The profile at 1024 sprites puts a number on it —
`_prune` is **4.46 s of 4.86 s**, over 1024 calls, with the free list peaking
at 597 rectangles and averaging 329. Everything else (`_split`, `_score`, the
placement scan) is a fifth of a second combined.

| sprites | `shipped` | `prune_new_only` |
|---|---|---|
| 128 | 15.5 ms | 6.4 ms |
| 256 | 112.0 ms | 25.9 ms |
| 512 | 750.3 ms | 87.3 ms |
| 1024 | **4 432 ms** | **297 ms** |
| 2048 | (30.4 s, 2026-08-31) | 977 ms * |
| 4096 | (190.8 s, 2026-08-31) | 3 589 ms * |

\* outside the declared sweep; three runs, minimum.

**What the variant does.** After every prune no surviving free rectangle
contains another — that is the prune's postcondition. So on the next placement
the only pairs that can fire are those involving a piece `_split` just made.
`_pack_new_only` keeps the shipped pair body, its traversal order and the
"of two identical rectangles the earlier is dropped" rule verbatim, and skips
only old-versus-old pairs. It is exact by induction and the bench asserts the
placement list is equal to `pack`'s on a 256-item pack before timing.

15× at the ceiling, and **the ceiling moves**: the 2026-08-31 document set 1024
as "the largest count whose single pack stays under five seconds"; by that same
rule the restricted prune supports **4096** (3.6 s), which is where the constant
was before that document lowered it. Note that `maxrects_layout` may call
`pack` more than once per document (once per candidate size), so the
per-document figure can be a small multiple.

**Why not the kernel.** The remaining 297 ms is still a Python loop, and a C
prune of the same restricted pairs would take it to a few milliseconds. But
under the gate is under the gate, and the batch-2 rule that killed
`warlockc_blur_f32` applies: a kernel that requires rewriting the reference
must beat the *rewritten* reference, and the rewritten reference is what would
be shipped. If `MAX_SPRITES` is raised back to 4096 and 3.6 s is judged too
long, *that* is the kernel question, and it is a different gate from this one.

**Where the fix goes.** `_prune` becomes `_prune_new(free, fresh)` or the
`pack` loop tracks which entries of `remaining` came from a split; either way
the survivor rule and the parity test in `tests/packwright/` against the
shipped `_prune` are the deliverable. The `layout.MAX_SPRITES` docstring and the
2026-08-31 document's "what would lift the ceiling again" paragraph both need
updating when it lands — it is not the bucketing that paragraph predicted.

---

## §2 — B9, `paint_wang_cells`: the loop is fine, `ref.holds` is not

The 2026-09-02 review filed the per-cell Python here as "the algorithm rather
than the implementation", and the re-fit is genuinely sequential. The profile
says the sequential part is not where the time is. At a 128-cell diagonal drag
(16 256 cells re-fit, 130 558 neighbour reads), the top of the profile is
`field_of` → `TilesetRef.holds` → `last_gid` → `max_local_id` → `tile_count`
→ `columns`/`rows` → `image_w`/`image_h`: seven property hops per neighbour
read, each recomputing a fact about the tileset that cannot change during a
gesture. `constraints_from` is 0.47 s of 0.55 s cumulative under the profiler
and almost all of it is that chain.

| side | box cells | `shipped` | `holds_hoisted` |
|---|---|---|---|
| 32 | 1 024 | 8.8 ms | 4.9 ms |
| 64 | 4 096 | 32.9 ms | 17.7 ms |
| 128 | 16 384 | **131.1 ms** | **67.7 ms** |

**What the variant does.** `_wang_field_hoisted` is `terrain.wang_field` with
`ref.firstgid`, `ref.last_gid`, `wangset.tiles` and `GID_MASK` read once into
the closure, and `holds` inlined as the range test it is. Same answers by
inspection of `holds`; the bench asserts the returned region is equal to the
shipped one on the same fixture before timing.

1.9×, and under the 100 ms gate. The shape stays linear in the box (13.8× over
16× work) because it is arithmetic-bound in Python's sense — a per-cell closure
call and dict lookup — not dispatch-bound.

**Why not the kernel.** A C re-fit is the right *shape* (it is `flood.c` with a
lookup), but it would have to absorb `constraints_from`'s SHARERS table,
`wangset.matching`'s scan-and-sort by float weight, and the memo — a table the
Python side would have to flatten per call. That is a real seam for a 68 ms
gesture that is now under its gate. If the gate is ever re-asked, the honest
next Python step before it is reading the layer through `data.tolist()`-style
scalar access instead of `int(data[y, x])` per neighbour, which the profile
shows as the next cost down; it was not measured this round because it changes
how `_wang_cell`'s writes reach the field and needs its own care.

**Where the fix goes.** `terrain.wang_field` itself — hoist the four reads and
inline the range test, with a comment naming `holds`'s property chain as the
reason. `tests/plotter/test_wang_paint.py` already pins the answers; the
regression test that fails against the unfixed code is the one that counts
`last_gid` accesses across a drag.

---

## Not benched

`retexture.combine` / `dilate` / `feather_blend` — fourth batch running, same
reason as every previous one. "Bench first" is still its state.

## What this batch changed in the tree

`scripts/bench_native.py` only: two new cases, two variants, and the two
reference implementations the variants time. No `native/*.c`, no ABI bump, no
`native.py` wrapper, nothing under `src/`. The bench sits outside `testpaths`,
so the suite count is unchanged.

The ranked follow-up, for whoever lands the build half:

1. `maxrects._prune` restricted to pairs touching a fresh piece — Python,
   exact, 15× at 1024 sprites; then re-raise `MAX_SPRITES` to 4096 on the 3.6 s
   figure, updating the 2026-08-31 document (§1).
2. `terrain.wang_field` with `holds` hoisted — Python, exact, 1.9× (§2).
3. No kernel from this batch. Batch 8's candidate list is empty until
   something new is measured; `retexture.combine` is the only named item
   without a number.
