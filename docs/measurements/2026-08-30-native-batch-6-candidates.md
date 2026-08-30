# Native kernel batch 6 — the four candidates, measured. No C written.

2026-08-30, at `f0d2124`. Machine: Windows 11, numpy 2.5.1, Python 3.13.13,
`vendor/warlockc/warlockc.dll` present and loading (ABI 10) — though none of the
four sites below has a kernel today, so the DLL changes nothing here. Every
figure is the **minimum of seven runs in a fresh process**, batch 2's stated
method, through `scripts/bench_native.py`.

**This batch deliberately writes no C.** It is the measurement half of batch 5's
own follow-up list — the three entries batch 5 recorded as "deferred" or "not
measured" (`docs/measurements/2026-08-22-native-batch-5.md` §8 and §10), plus one
site that had never been on any list, `clay/ops_bevel.bevel_edges`. The house
rule is that a candidate which has not been benched must not be written, and the
honest outcome of benching four candidates was that **two of the four want a
pure-numpy fix that has to land before any kernel question is even well posed,
one is over its gate and wants a kernel, and one fails its gate by a factor of
ten and should be struck off**.

## Summary

| # | Candidate | Gate | Measured | Shape | Verdict |
|---|---|---|---|---|---|
| B4 | `inker/tiles.materialize` | >200 ms per conversion | **444 ms** @3200² px, 8 px tiles | arithmetic-bound | **build** — but the Tier 0 memo first (3.1×), §1 |
| I7 | `inker/filters._grow` | >100 ms at r=32 | **175 ms** @r=32 | arithmetic-bound | **needs an algorithmic fix first** — 6.3× in numpy, §2 |
| B6 | `pipelines/pixel._to_oklab` | >200 ms residual @1024²×64 | **236 ms** | dispatch-bound *per the harness*, and the harness is being asked the wrong question here | **build**, §3 |
| B7 | `clay/ops_bevel.bevel_edges` | >1.0 s on the 200×200-quad mesh | **103 ms** @1000 edges, **579 ms** @10 000 | dispatch-bound | **defer — under its gate, and the wrong shape for C anyway**, §4 |

New bench cases: `tiles_materialize`, `oklab_fold`, `clay_bevel`; `inker_grow`
gained three variants where it had none. Nothing else in the tree changed.

---

## §0 — Two things the harness's own triage cannot see, said once

The flat-vs-linear rule prints `dispatch-bound` when the cost ratio across the
sweep is under a quarter of the **size** ratio. Twice in this batch that label is
right for the wrong reason, and pretending otherwise would be the kind of number
this document exists to stop.

**The size parameter is not always the work.** `tiles_materialize` sweeps *cells
per side*, so 4× in the parameter is 16× in the work — exactly as
`plotter_render` already does. Its verdict survives either reading (16.6× cost
is above the 4× threshold at span 4 and above it again at span 16), but the
printed span is 4 and the real one is 16.

**The swept axis is not always the dominant axis.** `oklab_fold` sweeps palette
entries because that is what makes it comparable with `pixel_map_palette`'s own
8/32/64 — and the residual conversion is 99.98 % image and 0.02 % palette, so of
course it is flat. That flatness is a statement about the palette, not about the
conversion. §3 reads it correctly.

Both are recorded in the case docstrings, not just here.

---

## §1 — B4, `inker/tiles.materialize`: over its gate, and the memo comes first

Batch 5 §10 listed this as **not measured**, on the strength of a *different*
function's failed vectorisation (`_doc_tiles.py:669-680`, the block-transpose
rewrite that "pays for the whole canvas twice to save the same per-cell copy").
That is a note about `_doc_tiles`' tile *extraction*, and it was allowed to stand
in for a measurement of `tiles.py`'s tile *placement*. It should not have been.

Sweep, cells per side, 32 px tiles, 64-tile sheet, a third of the refs carrying a
transform flag:

| cells/side | canvas | `as_shipped` | `memo_oriented` |
|---|---|---|---|
| 50 | 1600² | 12.14 ms | 8.47 ms |
| 100 | 3200² | 49.53 ms | 30.82 ms |
| 200 | 6400² | **201.23 ms** | 127.30 ms |

201 ms clears the >200 ms gate, but only at a 6400-square canvas, which is not
what anyone draws. The decisive measurement is the second one, at a **fixed
3200² canvas** — the same pixels every time, only the tile size moving:

| tile size | cells | `as_shipped` | `memo_oriented` | ratio |
|---|---|---|---|---|
| 32 px | 100² | 49.97 ms | 30.86 ms | 1.62× |
| 16 px | 200² | 131.23 ms | 54.71 ms | 2.40× |
| 8 px | 400² | **444.06 ms** | **141.79 ms** | **3.13×** |

Same canvas, 16× the cells, 8.9× the cost. The cost is **per cell and not per
pixel**, which is the whole reason a kernel is on the table: a pixel-art
document at 8 px tiles is the ordinary case here, not the extreme one, and it is
where a whole-canvas materialize costs nearly half a second.

**Where that runs.** Not on the interactive path — `_doc_tiles._repaint_tiles`
materializes only the tile-unit rectangle just written. It runs on **conversion**
(`_doc_tiles.py:712`, a layer becoming a tilemap), on `_rematerialize_tileset`
(one tile edited → every bound cel's whole grid), and in the `.aseprite` and
`.ora` readers (`asein.py:1162,1576`, `ora.py:1705`). Those are the "per
conversion" the gate names, and a tile edit rematerialising every bound cel is
the one a user can feel.

**Verdict: build — after the numpy fix.** `oriented()` is called per cell with
no memo, and there are at most `tile_count × 8` distinct answers on a canvas
asking for hundreds of thousands. Memoising on the raw gid is bit-identical (the
bench asserts array equality) and buys **3.1× at 8 px tiles with no C at all**;
it is worth more the smaller the tiles get, which is the direction this document
says the cost lives in. Batch 2's third rule — do the numpy fix first regardless
of what the kernel says, because it helps a checkout with no DLL — applies
exactly.

After the memo, 142 ms of per-cell copy remains, and *that* is the kernel
question. `warlockc_blit_cells_u8` is not a drop-in for it: it is source-over
with binary alpha where `materialize` **replaces**, alpha included, and it takes
an `(n_tiles, th, tw, 4)` atlas where the eight orientations would have to be
pre-expanded into it. A replace-mode sibling, or a mode flag, is the shape — for
the batch that writes C.

---

## §2 — I7, `inker/filters._grow`: batch 5 deferred it to C. It wanted numpy.

Batch 5 §8 deferred this and said "the next batch should take it with `_shift`
fused rather than the loop transplanted." That instruction was right and the
implied destination was wrong: the fusing alone, in numpy, is most of the win.

1024² mask, 8-connected, `wrap=False`:

| steps | `shipped` | `fused` | `separable` |
|---|---|---|---|
| 4 | 21.71 ms | 3.38 ms | 3.35 ms |
| 16 | 87.01 ms | 13.52 ms | 13.88 ms |
| 32 | **174.72 ms** | **27.76 ms** | 27.73 ms |

175 ms against a >100 ms gate, and **6.3× from fusing, with no C**.

The reason is allocation, not arithmetic. `_shift` builds a whole-canvas
`np.zeros_like` and slice-assigns into it, then `|` allocates the union — so the
shipped loop writes the canvas about 24 times per step where an in-place
`grown |= out[slice]` writes it 9 times. All three curves are linear in the
radius (8.0×/8.2×/8.3× over 8× work), so this is arithmetic-bound and always
was; batch 5 read that correctly and drew the wrong conclusion from it, because
the constant factor available was never measured.

`separable` — the 3×3 square as a 1×3 dilation then a 3×1 one, 4 whole-canvas
ORs per step instead of 8 — is **not** faster than `fused` (27.73 vs 27.76 ms,
inside the noise). It halves the ORs and doubles the copies, and on a 1 MB bool
canvas those cancel. It also only exists for the 8-connected case: the
4-connected diamond is not separable, and that flag is the whole difference
between a rounded outline and a boxy one. Recorded so it is not tried again.

**Verdict: needs an algorithmic fix first.** The fused version is bit-identical
to the shipped one on every radius tested (the bench asserts it), needs no ABI
bump and helps a checkout with no DLL. Only after it lands is the kernel
question honest — and at 27.8 ms on a preview a user drags, it very likely
answers itself.

---

## §3 — B6, `pipelines/pixel._to_oklab`: the residual is now the whole cost

Batch 5 §6 kernelised the nearest-palette search (`warlockc_palette_nearest_f64`,
2488 ms → 450 ms, 5.5×). What that leaves is the two `_to_oklab` calls the
search sits between. 1024² frame:

| entries | `residual` (frame + palette) | `palette_only` |
|---|---|---|
| 8 | 244.50 ms | 0.03 ms |
| 32 | 242.17 ms | 0.04 ms |
| 64 | **235.70 ms** | 0.04 ms |

Flat in the palette, as §0 said it would be: the palette conversion is **0.02 %**
of the residual at 64 entries. The 236 ms is the frame, and it is essentially all
of what `map_palette` still costs — the shipped kernel took the 2488 ms half and
the remaining half was never re-examined. Against a stated >200 ms gate, over.

The work is three whole-frame temporaries in a chain that numpy cannot fuse: a
float64 divide, a `np.where` over a `** 2.4` power computed **on every pixel
including the ones the branch discards**, three cube roots, and a `np.stack` of
three linear combinations. At 1024² that is roughly a dozen 8 MB arrays
allocated, written and dropped. A kernel streaming one pixel at a time would
allocate the output and nothing else, and the `** 2.4` would be evaluated on the
branch that needs it — which the numpy form structurally cannot do.

**Verdict: build.** It is the largest single remaining cost in a pipeline stage
this project already decided was worth C, at the same size and on the same
frame, and the parity bar is easy: it is a fixed-shape elementwise map with no
reduction and no ordering, so bit-identical under `/fp:precise` is a question
about `pow` and `cbrt`, not about the algorithm. A **cheaper** first move, and
the one that should be tried before the kernel for the same reason as §1 and §2:
sRGB→linear is a 256-entry lookup on uint8 input, and `rgba[:, :, :3]` is uint8
here. That removes the divide, the `where` and the `** 2.4` outright. Not
measured this round — it is a different function, and this document does not
report numbers it did not take.

---

## §4 — B7, `clay/ops_bevel.bevel_edges`: under its gate, and the wrong shape

Never on any previous batch's list. Gate stated before measuring: >1.0 s on
`tests/clay/test_scale.py`'s 200×200-quad, 40 000-face mesh — a third of that
file's `BUDGET = 3.0` s, chosen because `bevel_edges` is the one walking op with
no wall-clock budget anywhere (`tests/clay/test_ops_bevel.py` covers it
functionally only). The selection is pairwise vertex-disjoint interior edges, so
every corner stays in the one-beveled-edge row of the op's own table and the
sweep varies count and nothing else. Adjacency is built outside the clock: by
the time a user clicks Bevel it is already cached.

| beveled edges | `grid_200` (40 000 faces) | `grid_50` (2 500 faces) |
|---|---|---|
| 10 | 54.22 ms | 3.73 ms |
| 100 | 59.83 ms | 7.93 ms |
| 1 000 | **103.12 ms** | 51.91 ms |
| 5 000 | 309.54 ms | — |
| 10 000 | **578.78 ms** | — |

(The 5 000 and 10 000 rows are the same harness builder driven directly, seven
runs, minimum, outside the declared sweep — the declared sizes stop at 1 000 and
these two are the bound on the worst case.)

**It does not clear its gate.** 579 ms with a quarter of every edge in a 40k-face
mesh selected, against 1.0 s. There is no realistic selection that gets there.

The shape is the more interesting half. On the big mesh the cost is **1.9× over
100× the selection**; on the small one, 13.9× over the same 100×. Read them
together: the cost is dominated by mesh size, not by how much is beveled. That is
the two loops that walk everything — `for face in range(face_count(mesh))` and
`for corner in range(len(loops))` — running over all 40 000 faces and 160 000
corners to bevel ten edges. 54 ms of the 54 ms at ten edges is that walk. The
marginal cost of a beveled edge is about 52 µs.

**And a kernel is not what fixes it.** `bevel_edges` builds `new_loops`,
`new_uv`, `counts`, `owners` and `darts` as **dynamic Python lists and dicts**,
appended to a variable number of times per corner by a three-row table, with
`slides`/`miters` memo dicts minting vertices as it goes. That is exactly the
shape that made `clay.ops_topo.weld` a deferred candidate in batch 2 rather than
a C port (`docs/measurements/2026-08-09-native-batch-2.md:242-255`: "Not acted on
in this batch because the fix is not a kernel… Both are pure-numpy/pure-Python
changes that should be measured on their own before any C is considered"). The
fix, if one is ever wanted, is to stop walking untouched faces at all: the
corners that change are `a.vertex_corners(v)` for `v in touched`, which is
hundreds where the loop does hundreds of thousands, and everything else could be
copied through as a slice. That is a pure-Python change.

**Verdict: defer.** Under its gate by an order of magnitude at any realistic
selection, and the available win is an algorithmic one that a kernel would only
get in the way of. Recorded with its numbers so the question is answered rather
than re-asked. If the mesh ceiling here ever moves — 40k faces is `model.glb`
scale today — the 54 ms floor moves with it linearly, and the O(mesh) walk is
where to look.

---

## Not benched

**B3, `retexture.combine` / `dilate` / `feather_blend`** (`pipelines/retexture.py:285`,
`:314`, `:211`). Batch 5 §10 left it unbenched under an interactive-first
priority and it stays unbenched here, deliberately: it is the lowest-priority
item on this wave's own list, its working set is ~500 MB per temporary, and a
seven-run minimum at that size measures the machine's page cache as much as the
code. **No number is reported for it because none was taken.** Its gate remains
"bench first", which is still the correct state for it.

## What this batch changed in the tree

`scripts/bench_native.py` only: three new cases and three new variants on an
existing one, plus the fused/separable `_grow` reference implementations the
variants compare against. **No `native/*.c`, no ABI bump, no `native.py`
wrapper.** The bench cases sit outside `testpaths`, so the suite count is
unchanged.

The ranked follow-up, for whoever takes batch 6's build half:

1. `filters._grow` fused — numpy, bit-identical, 6.3×, no ABI (§2).
2. `tiles.materialize` orientation memo — numpy, bit-identical, 3.1× (§1).
3. `_to_oklab` sRGB→linear as a 256-entry uint8 LUT — numpy, needs its own
   measurement (§3).
4. Only then: a replace-mode `blit_cells` sibling (§1) and an Oklab kernel (§3).
5. Never, unless the mesh ceiling moves: `bevel_edges` in C (§4).
