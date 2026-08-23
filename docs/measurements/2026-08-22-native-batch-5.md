# Native kernel batch 5 — what else should go to C

2026-08-22. Machine: Windows 11, MSVC (`native/build.ps1` picked `cl`), numpy
2.5.1, Python 3.13.13. Every figure is the **minimum of seven runs in a fresh
process**, which is batch 2's stated method and the only honest one: the spread
on multi-megabyte float work is ±30 %, so a mean reports on the machine's other
tenants rather than on the code. The harness is `scripts/bench_native.py`,
which is in the repo *because* the last two batches benched from scripts that
were then thrown away — see §0.

Eleven candidates were measured. **Four numpy fixes shipped, three kernels
shipped (ABI 9 → 10), two candidates were rejected on their numbers, one was
already fixed by unrelated work, and one was ruled out of scope.** The
rejections are the durable half of this document: they are what stops the
question being asked again in three months.

## Summary

| # | Candidate | Gate | Before | After | Verdict |
|---|---|---|---|---|---|
| I1 | `np.add.at`, the seven `(n, 3)` sites | any win at 20k verts | **42.5 ms** @200k | **22.9 ms** | **Tier 0 shipped, 1.9×**, §1 |
| I1b | `np.add.at`, the three 1-D counters | any win at 20k verts | **2.8 ms** @200k | 4.2 ms | **rejected — bincount is slower**, §1 |
| I1c | `viewer/scene._flat_normals` | any win | — | — | **rejected — f4 accumulator**, §1 |
| I3 | `adjacency._build`'s `unique(axis=0)` | >30 % of `_build` | **777 ms** @200k | **130 ms** | **Tier 0 shipped, 6.0×**, §2 |
| I3b | `check_manifold`'s `unique(axis=0)` | on the interactive path | — | — | **out of scope — button-driven**, §2 |
| I6 | `indexed.histogram` | >50 ms @2048², 256 entries | **11 351 ms** | **117 ms** | **Tier 0 shipped, 97×**, §3 |
| I2 | `picking.build_bvh` | >50 ms @200k tris | **1 045 ms** | **123 ms** | **kernel shipped, 8.5×**, §4 |
| B1 | `render.render_layer` | any win | **4 348 ms** @200×200×3 | **759 ms** | **kernel shipped, 5.7×**, §5 |
| B2 | `pixel.map_palette` | >500 ms @1024²×64 | **2 488 ms** | **450 ms** | **kernel shipped, 5.5×**, §6 |
| I4 | `panes/plotter_canvas._layers` | >4 ms/frame | — | — | **already fixed**, §7 |
| I7 | `inker/filters._grow` | >100 ms at r=32 | **177 ms** | — | **deferred**, §8 |
| B5 | `meshreport._welded` | >200 ms | **106 ms** @200k | — | **rejected — overflow**, §9 |
| B3 | `retexture.combine`/`dilate` | bench first | — | — | **not written**, §10 |
| B4 | `_doc_tiles` / `tiles.materialize` | >200 ms per conversion | — | — | **not measured**, §10 |

ABI 9 → 10, carried by three new symbols: `warlockc_bvh_build`,
`warlockc_blit_cells_u8`, `warlockc_palette_nearest_f64`.

---

## §0 — Why there is a harness this time

Three of batch 3's five ranked follow-ups were stale by the time this batch
read them, and none of the staleness was visible without re-measuring:
`dither._ordered` had been rewritten to cost per *distinct colour*;
`maxrects._prune` went O(n³) → O(n²) in Part J; and the "48 ms of a 49 ms frame"
picking figure had been spent by Part J's BVH. Acting on the queue as written
would have optimised three things that were already fixed.

The queue itself survived only in `NEXT_SESSION.md`, deleted untracked on
2026-08-17 — so the ranking existed nowhere in the repo at all. `scripts/bench_native.py`
is the fix: one named case per candidate, each carrying **its gate, written
down before the number was known**. A gate chosen after the fact is a
rationalisation.

It carries two triage rules forward, both earned:

**Flat-vs-linear** (batch 3). `--sweep` varies the work parameter. A cost curve
flat in it is dispatch-bound and a kernel replacing N numpy calls with one wins
enormously; a rising one is arithmetic-bound and the kernel buys the constant
factor only. This predicted Floyd–Steinberg's 76× before a line of C was
written, and in this batch it is what told `indexed.histogram` from
`inker/filters._grow`.

**The keep/reject rule** (batch 2). A kernel that leaves the fallback unchanged
need only be faster. One that requires rewriting the reference to be
deterministic must beat the *original* reference, not the new one. That is what
killed `warlockc_blur_f32`, which was bit-identical on 54/54 cases, 1.3× with
the DLL and **1.34× worse without one**.

A third rule this batch adds: **do the numpy fixes first regardless of what the
kernels say.** They help a checkout with no DLL, which is every checkout until
the installer ships — and here the single largest win in the batch was one of
them.

---

## §1 — `np.add.at`: a win on rows, a loss on scalars

`clay/mesh.accumulate` (promoted from `_accumulate`; the `face_normals`
precedent, same alias kept) already replaced `np.add.at` with `np.bincount` and
carried the correctness argument — both accumulate in *input* order, so the
float sum order and therefore the result are unchanged bit for bit. It had one
caller. Eleven sites still bypassed it.

**The `(n, 3)` scatters, at 200 000 vertices:**

| | 2k | 20k | 200k |
|---|---|---|---|
| `np.add.at` | 0.31 ms | 3.41 ms | 42.54 ms |
| `accumulate` | 0.06 ms | 1.98 ms | 22.88 ms |

Seven sites converted: `ops_subdiv` :237 (edge points), :249 (`q_sum`), :254
(`r_sum`), :261–262 (`b_sum`), and `ops_topo` :338 (`total`), :486 and :543
(`sums`). All are on Clay's rebuild-on-every-edit path, which `ClayView.sync`
keys on `id(obj.mesh)` — so every extrude, inset and element-drag commit pays
it.

`b_sum` is the one that needed care. It was **two** `np.add.at` calls into the
same accumulator, and they ran column 0 to completion before column 1 — so it
became one `accumulate` over the two columns *concatenated*. Adding two separate
per-column totals would re-associate the sum and move the last bit of a crease
position. Asserted, not assumed
(`tests/clay/test_ops_subdiv.py::test_the_border_sum_keeps_the_summation_order_the_two_scatters_had`).

### Rejected: the 1-D counters

The plan expected `hits = np.zeros(n); np.add.at(hits, verts, 1.0)` to convert
too. Measured, it goes the other way:

| | 2k | 20k | 200k |
|---|---|---|---|
| `np.add.at` | 0.01 ms | **0.19 ms** | **2.77 ms** |
| `np.bincount` | 0.01 ms | 0.28 ms | 4.20 ms |

Adding a *scalar* has a buffered fast path that adding a row does not, and
`bincount` still has to build and cast a table. All three counters stay on
`np.add.at`, with the number in a comment beside the one at `ops_topo:339`.

### Rejected: `viewer/scene._flat_normals`

Its accumulator is `np.zeros_like(positions)`, which is **f4**. The loop sums in
single precision; `bincount` returns float64 and would round differently. It is
also a rare path — trellis always writes normals — so a changed last bit on the
one attribute the shading reads is not worth it. Left as it was, with the reason
written in.

---

## §2 — The edge table: 777 ms → 130 ms

`adjacency._build:154` is described in place as "the one sort", and it was
`np.unique(pairs, axis=0, return_inverse=True)` — which goes through a
structured-void view and a lexsort.

| verts | `unique(axis=0)` | packed scalar |
|---|---|---|
| 2 000 | 4.46 ms | 0.54 ms |
| 20 000 | 57.11 ms | 8.89 ms |
| 200 000 | **777.37 ms** | **130.32 ms** |

The packed idiom already existed twice in this tree — `indexed.snap:118-122` and
`dither._ordered:459` — with the reason stated in place. `lo * stride + hi` is
order-preserving in exactly the sense the lexsort is, but **only while both
columns are non-negative and below `stride`**; both are vertex indices, so that
is asserted at the call site rather than trusted, along with the bound that
keeps `stride²` inside i8. The parity assertion is the full triple — same edge
list, same order, same inverse — because a packing that merely deduplicated
correctly but reordered would renumber every edge in the document.

### Out of scope: `check_manifold` (`:395`)

Its own docstring says **no pane calls it per frame** — the properties panel
runs it from a button and holds the result against the immutable `Mesh`. It is
also a rectangle of `arity` columns, so packing needs a running polynomial that
overflows on a large n-gon. Not on the interactive path, so not touched.

---

## §3 — `indexed.histogram`: the best effort-to-reward ratio in the batch

One full-canvas `(rgb == want).all(axis=1).sum()` **per palette entry**. On a
2048² document with a full palette that is 256 passes over four million pixels.

| entries | per-entry loop | packed, one pass |
|---|---|---|
| 16 | 802 ms | 110 ms |
| 64 | 4 256 ms | 118 ms |
| 256 | **11 351 ms** | **117 ms** |

97× at the gate size, and the *shape* changed: the cost is now per **distinct
colour** and is flat in the palette, which the flat-vs-linear sweep shows
directly (1.1× cost over 16× work). That shape is what the perf budget asserts
— a 16-entry and a 256-entry reading over the same canvas must stay within a
factor of two, which a reintroduced per-entry scan could not manage.

The one thing that needed care: `searchsorted` lands *somewhere* for every
entry, so "would sit here" has to be turned into "is here" by an equality test,
or every slot inherits a neighbour's count. The function's whole purpose is that
a zero means the slot is safe to delete.

---

## §4 — `warlockc_bvh_build`: 1 045 ms → 123 ms

`build_bvh` is an explicit stack doing ~6 small numpy calls per node, and
`BVH_LEAF = 8` means a 200k-triangle mesh is tens of thousands of iterations at
spans small enough to be pure dispatch. It reruns on **every mesh edit** —
`cached_bvh` is weak-keyed on the immutable `Mesh` — so it is squarely
interactive.

| tris | before | after |
|---|---|---|
| 2 000 | 6.74 ms | 0.76 ms |
| 20 000 | 106.25 ms | 10.02 ms |
| 200 000 | **1 045 ms** | **123 ms** |

The residue is the Python-side `positions[tris]` and its min/max, which the
kernel deliberately does not swallow.

### The licensed parity bar

**The tree cannot be reproduced bit-for-bit and this is the second kernel where
that is true.** `np.argpartition` is introselect; its permutation among *equal*
keys is unspecified, so a C median split separates coincident centroids
differently and builds a different — equally valid — tree. Asserting node-for-node
equality would be asserting a property the numpy path does not have either.
This is the `contours.c` situation exactly: the reference leaves something
unspecified, so the bar moves to what is guaranteed.

Here that is the **pick result**, and Part J already pinned the tie-break
(lowest triangle index) that makes it well-defined, chosen so the tree and the
full sweep agree. `tests/test_bvh_native.py` asserts:

- the same triangle from either tree over a 120-ray sweep, and the same triangle
  as the unnarrowed linear sweep;
- every triangle in exactly one leaf, and an interior node holding none of its
  own;
- a child's box contained in its parent's;
- a leaf above `BVH_LEAF` only where every centroid in the span coincides —
  the one legitimate oversized leaf, since no split separates them;
- the fallback genuinely taken when the seam is closed.

The boxes *are* bit-identical: min and max over doubles are exact whatever order
the triangles inside a node ended up in.

The node budget is `2 * (T / 4 + 2) + 8`, from the fact that a median split of
nine or more leaves at least four a side. The kernel returns `-1` rather than
overrunning it, and `-1` is a fall-back, not a retry — the `contours` contract.

---

## §5 — `warlockc_blit_cells_u8`: 4 348 ms → 759 ms

The largest single number available, and it confirms batch 2's stale one (3 992
ms): **4 348 ms for a 200×200×3 map of 32px tiles**, 36 µs across each of
120 000 cells. That is far past dispatch overhead — a 32-square tile is a
thousand pixels — so a C blit, at about a microsecond, is the whole difference.

| cells/side | before | after |
|---|---|---|
| 50 | 217 ms | 46 ms |
| 100 | 893 ms | 180 ms |
| 200 | **4 348 ms** | **759 ms** |

Batch 2 §7 named this kernel and left it unwritten because the partial-alpha
case it would answer had no measured corpus. That reason never covered the
**binary-alpha** case, which is what a tileset actually is — an opaque body with
a transparent surround — and which is still four seconds.

So the kernel answers `_over`'s masked-copy branch and nothing else.
`_blit_cells_native` declines for a non-normal mode, an opacity below one, mixed
tile sizes (a collection), or a tile whose alpha is not binary — and the
binary test is **one vectorised pass per distinct oriented tile**, not per cell,
which is the only reason asking is affordable. That is the same observation the
`oriented` memo already rested on: a map is the same handful of tiles repeated
thousands of times.

Two things carried forward deliberately:

**The exact-identity trap** from batch 2 §7. Where source *and* destination are
both fully clear, the full path writes rgb 0, discarding colour stored under a
zero alpha. `cells.c` reproduces it rather than tidying it up, and there is a
test whose entire subject is that quirk.

**Draw order.** The cells reach the kernel in the document's own draw order, not
grouped by tile. Grouping would be faster still and would paint an oversized
tile's overlap the wrong way round — and an oversized tile is ordinary (a 32px
map with 48px trees, anchored bottom-left as Tiled anchors it).

The restructure that made this possible — resolve the whole layer into a cell
list, then blit — also subsumes what `project.draw_order`'s per-cell tuple yield
was costing.

---

## §6 — `warlockc_palette_nearest_f64`: 2 488 ms → 450 ms

A float sibling of the shipped `warlockc_palette_nearest_i32`, which is int32
RGB and wired only into inker. This site searches in **Oklab float64** and is
chunked at `1 << 16` rows precisely because each chunk builds a `(65536, p, 3)`
difference array — about 100 MB at 64 entries — allocated, written and thrown
away to read one index per row.

| entries | before | after |
|---|---|---|
| 8 | 573 ms | 314 ms |
| 32 | 1 426 ms | 386 ms |
| 64 | **2 488 ms** | **450 ms** |

After the kernel the case is dispatch-bound (1.4× cost over 8× work): what is
left is `_to_oklab`, which the kernel deliberately does not swallow. Folding it
in would remove ~8 more full-frame float64 planes and is the obvious batch-6
candidate.

### The sqrt is part of the contract

It is tempting to compare squared distances, since sqrt is monotonic — **but
sqrt also rounds**, so two distinct squared values can land on the same double.
numpy's argmin over the rounded norms then takes the *earlier* index, while an
argmin over the exact squares takes the strictly smaller one, which may sit
later. Same ordering, different pick, and on a ramp with near-equidistant
entries that is a visibly wrong colour.

The test constructs the pair rather than hoping to sample one: entry 0 at
squared distance `1 + 2⁻⁵²` (the next double above one) and entry 1 at exactly
one. sqrt rounds both to 1.0 — the true difference is half a unit in the last
place and ties round to even — so a norm-based argmin sees a tie and takes index
0, while a squares-based one takes index 1. Ties go to the lowest index,
matching argmin and the int32 sibling.

---

## §7 — `panes/plotter_canvas._layers` was already fixed

Recorded in batch 2 §3 as an algorithmic defect rather than a kernel candidate:
per visible cell per layer it linear-scanned `refs.items()` calling
`ref.holds(tile_id)`, and the comment falsely claimed resolution was hoisted
when only the texture was.

Both are fixed in current code. `_index_memo(tab.uid, doc.tileset_epoch)` answers
which ref holds an id, the animation check is hoisted out of the cell loop, and
the comment now says exactly what the hoist covers and what `_TILESET_MEMO`
answers instead. Nothing to do; recorded so the entry is not chased again.

---

## §8 — `inker/filters._grow`: over its gate, deferred

177 ms at r=32 on a 1024² mask, against a >100 ms gate, and up to 256
whole-canvas passes (8 neighbours × 32 steps).

Deferred rather than rejected. The cost curve is **linear in the radius**
(8.3× over 8× work), so this is arithmetic-bound and a kernel buys the constant
factor rather than an order of magnitude — and the iterated single-step shape is
load-bearing, not incidental: it is what makes the 4-connected case a diamond
and the 8-connected case a square, which is the whole difference between a
rounded outline and a boxy one. A radius-at-once rewrite would be a different
filter. It also runs on a preview a user drags, not on every frame of one.
Below the other three on an interactive-first ranking, and the next batch should
take it with `_shift` fused rather than the loop transplanted.

---

## §9 — `meshreport._welded`: rejected on overflow, not on speed

106 ms at 200k vertices and 306 ms at 500k. Real meshes here are ~200k, so it
only clears its >200 ms gate at a size this pipeline does not produce.

More decisively, it packs **rounded floats**, which can be negative and whose
range is set by `diagonal * WELD_TOLERANCE` — at 1e-6 that is ±10⁶ per column,
so a three-column pack is ~10¹⁸ and sits inside i8 by less than an order of
magnitude, before any offset for the sign. The packing's soundness condition is
exactly the thing that cannot be asserted cheaply here. Left on the slow path.

---

## §10 — Not measured

**B3, `retexture.combine`/`dilate`/`feather_blend`.** Its complaint is working
set, not loop overhead: `np.stack(colours)` is `(10, 2048, 2048, 3)` float32 ≈
500 MB and `combine` multiplies it into a second 500 MB temp before reducing. A
kernel streaming over texels would remove the N-view temporary entirely. But it
is a background job already dominated by GPU bake time, so under an
interactive-first priority it ranks last — and the plan's own rule is that it
must not be written before it is benched. It was not benched, and it was not
written.

**B4, `_doc_tiles` / `tiles.materialize`.** Vectorisation was already tried here
and lost (the comment at `_doc_tiles.py:678-688` records the block-transpose
attempt). Not re-measured this round.

## Not re-litigated

Read and measured previously, not guessed: `filters._gaussian` blur (batch 2 §1,
kernel built, bit-identical on 54/54, and **reverted** — 1.3× with the DLL,
1.34× worse without one); the fused brush dab (batch 2 §2, 6.4 % of a stroke
against a 30 % gate); `plotter.tools.flood_fill` and `tmx._csv` (batch 2
§5/§10); `_doc_tiles`' block transpose; and everything in the Tier 2 list of
`docs/PLAN_NEXT.md`, deleted at `4fc9927` (`git show 4fc9927^:docs/PLAN_NEXT.md`) —
Pillow/trimesh-backed code, single-pass whole-array numpy, zero-copy accessors.

## The gap this closes retroactively

ABI 8 → 9 (`warlockc_palette_nearest_i32`, `warlockc_flood_u8`) shipped with no
measurement document; its budgets lived only in
`tests/test_perf_native_batch3.py`. Those two kernels are now covered by the
same harness (`scripts/bench_native.py`, cases `inker_histogram` and the flood's
existing budget), and batch 5's own budgets are in
`tests/test_perf_native_batch5.py` — which also adds the coverage batch 3
lacked: the Plotter render path and the Clay rebuild path, both interactive.
