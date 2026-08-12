# Native kernel batch 2 — what was converted, and what the numbers refused

2026-08-09. Machine: Windows 11, MSVC (`native/build.ps1` picked
`cl`), numpy 2.5.1, Python 3.13.13. Every figure below is the minimum of 5–7
runs in a fresh process unless stated; the spread between runs on this machine
is wide enough (±30 % on multi-megabyte float work) that only the minimum is
quoted, and any comparison is between two minima taken the same way.

The batch was gated the way `docs/PLAN_NEXT.md` (deleted in v0.0.10, recoverable
with `git show 4fc99271^:docs/PLAN_NEXT.md`) gated its own: measure first, and
record the rejections with their numbers, because a rejection carrying a number
closes a question while a rejection carrying an opinion gets re-litigated.

**Three of the ten candidates shipped. One shipped as a C kernel. The headline
candidate was built, measured and reverted**, which is the most useful thing in
this document.

## Summary

| # | Candidate | Gate | Measured | Verdict |
|---|---|---|---|---|
| M1 | `filters._gaussian` / blur preview | >16 ms/frame | 1124 ms/frame @2048² r=8 | **fixed, but not by a kernel** — see §1 |
| M2 | `brush.StrokeState._resolve` | ≥30 % of stroke | **6.4 %** | rejected, §2 |
| M3 | `plotter_canvas._layers` | >4 ms/frame | not benched headlessly | deferred, §3 |
| M4 | `selection._morph` | >100 ms/click | 127 ms @2048² r=10 | **shipped as `warlockc_morph_u8`**, §4 |
| M5 | `plotter.tools.flood_fill` | >50 ms/click | **35.9 ms** @40 k cells | rejected, §5 |
| M6 | `viewer.picking.ray_triangles` | >2 ms/frame | 46 ms @200 k tris | rejected on the *fix*, not the number, §6 |
| M7 | `plotter.render.render_map` | any win | 12 590 ms @200²×3 | **shipped, 3.3–17×**, §7 |
| M8 | `clay.ops_topo.weld` | >500 ms/click | 951 ms @18.7 k verts | deferred, §8 |
| M9 | `packwright.maxrects.pack` | >200 ms/export | 2238 ms @500 sprites | deferred, §9 |
| M10 | `plotter.tmx._csv` / `_decode_payload` | >50 ms | **4.4 ms** @200² | rejected, §10 |

ABI went 6 → 7, carried by the one new symbol (`warlockc_morph_u8`).

---

## §1 — Blur: a C kernel was written, measured, and reverted

This was the batch's headline and it is the entry worth reading.

`panes/inker_bridge.py:283` calls `Document.preview_filter` on **every frame**
the filter popup is open, deliberately, so that switching filters in the combo
repaints. The cost of that, at 2048² radius 8:

| | ms |
|---|---|
| `blur()` whole image | **1124** |
| `_gaussian` (one plane; blur runs 4) | 164 |
| `_straight` | 198 |
| `_premultiplied` | 107 |

So `_gaussian` is ~58 % of it, as expected, and the whole thing ran sixty times
a second.

**What was tried.** The plan's sequencing was: bit-identity against
`np.convolve` is impossible (it reaches `cblas_sdot`, whose summation order is
OpenBLAS's CPU dispatch and varies by machine and wheel), so rewrite the numpy
reference into a deterministic shifted-slice accumulation this repository owns,
*then* write a C kernel bit-identical to that. Both halves were built:

- The deterministic rewrite deviated from the BLAS-order original by at most
  **0.00012 levels** out of 255, changing **2 of 786 432** uint8 channel values
  (0.0003 %) — pixels sitting exactly on a `.5` rounding boundary.
- It was also **1.7× slower**: 273 ms against 164 ms. A shifted-slice form is
  33 whole-array passes over 17 MB and is memory-bound, where BLAS keeps a row
  in cache.
- `warlockc_blur_f32` was then written and was **bit-identical on 54/54 cases**
  (radii 0.5–32, shapes including 1×1, 1×N, non-square). It ran at 127 ms.

**Why it was reverted.** 127 ms against the *original* 164 ms is 1.3×, not the
order of magnitude a kernel is worth. The reason is structural rather than
fixable by better C: **bit-parity forbids vectorising the reduction**, which is
exactly what makes BLAS fast, so the kernel spends its budget buying back the
1.7× the deterministic reference gave away. Net, end to end:

| | blur @2048² r=8 |
|---|---|
| original (`np.convolve`) | 1124 ms |
| deterministic reference + kernel | **850 ms** (1.3× better) |
| deterministic reference, no DLL | **1505 ms** (1.34× *worse*) |

`vendor/` is gitignored, so the no-DLL row is what a checkout gets. A change
that is 24 % better for people with a compiler and 34 % worse for everyone else
is not an optimisation. `native/blur.c` was deleted and `_gaussian` restored to
`np.convolve` verbatim.

**What actually fixed it** is upstream and free: `Document.preview_filter` now
memoises the filtered array on `(name, params)` for the life of a session. The
snapshot `begin_filter` takes never changes, so `apply_named` is a pure function
of those two within one session. Only the expensive half is cached — the blend,
the alpha lock and the invalidate still run every frame, so switching filters
still repaints. Idle frames with the popup open went from 1124 ms to nothing.

`_straight` was also routed through the shipped `composite.to_uint8_255` kernel
and had a redundant full-plane `np.zeros_like` removed from its masked divide.
Bit-identical; worth ~2 %, kept because it deletes a duplicated expression
rather than for the speed.

**The one open thread.** A slider *drag* still pays ~1.1 s per frame, because
every drag step is a new memo key. The answer there is a reduced-resolution
preview or a debounce — policy, not arithmetic — and it is not attempted here.
A *vectorised* kernel judged against a stated tolerance rather than bit-identity
is the other option, and it would need the `contours.c` licence argument written
out first (the reference genuinely leaves summation order unspecified, so that
argument is available), plus a `/fp:fast` exception for one translation unit
against `build.ps1`'s global `/fp:precise`. Neither was in scope.

## §2 — The fused brush dab: rejected, and the standing deferral is now closed

`docs/PLAN_NEXT.md` (deleted; git history) deferred `warlockc_dab_u8` pending a
profile. Here it is —
300 dabs, ⌀64, 2048², `symmetry="none"`, total 582 ms:

| | cumulative ms | % of stroke |
|---|---|---|
| `document.invalidate` | 494.2 | **84.8** |
| ├ `native.stack_f32` | 301.0 | 51.7 |
| └ `native.to_uint8_f32` | 144.9 | 24.9 |
| `brush._stamp` | 45.3 | 7.8 |
| `brush._resolve` | 37.2 | **6.4** |

The gate was ≥30 %. `_resolve` is 6.4 %, so a kernel fusing it perfectly would
buy under 6 % of a stroke. **The deferral is closed as a rejection**, not
re-deferred: 85 % of the cost is the per-dab recomposite, and that is *already*
in shipped C kernels.

Two side findings, recorded because they cost real time to establish:

- `make_stamp` is **already** `@lru_cache(maxsize=256)` (`brush.py:71`). A
  planned "add a stamp cache" item was deleted as a no-op before it was written.
- `symmetry="xy"` costs **34.2 ms per dab** against 2.16 ms with symmetry off —
  a 16× cliff, 10.3 s for the same 300-dab stroke. The four mirrored dabs are
  invalidated as one union rectangle spanning most of the canvas, so the
  recomposite is whole-canvas. That is a dirty-rect defect, not a kernel
  candidate: the fix is invalidating four small boxes rather than their bounding
  box. **Not fixed here** — it is a behaviour change to a path nothing in this
  batch touched, and it deserves its own measurement.

## §3 — `plotter_canvas._layers`: deferred, and it is a defect not an optimisation

Per visible cell per layer it linear-scans `refs.items()` calling
`ref.holds(tile_id)`, although the comment at `:183-185` claims resolution is
hoisted — only the *texture* is. Not benchmarked: it needs a live imgui frame,
and the harness to do that honestly is larger than the fix. Recorded as an
algorithmic defect to fix on inspection with a version-counter memo, not as a
kernel candidate — the per-cell `add_image_quad` underneath it is the
irreducible imgui shape `ants.cull` already names.

## §4 — Selection grow/shrink: shipped as `warlockc_morph_u8`

`_morph` is `radius` passes of `np.pad` plus 4 or 8 full-frame
`np.maximum`/`np.minimum` with no `out=`. At 2048², radius 10: **127 ms**, past
the 100 ms gate.

| | ms |
|---|---|
| numpy reference | 127 |
| `warlockc_morph_u8` | **74** |

1.7×, bit-identical on **1728/1728** cases (7 shapes × 6 mask kinds × radii 1–16
× both directions), plus `bordered()` which runs both directions over one mask.

Kept where the blur kernel was not, and the difference is the whole rule: this
one **leaves the numpy fallback exactly as it was**, so a machine with no
compiler is unaffected. There is no trade to weigh.

Two things about it are worth writing down. The element is an **octagon** — the
passes alternate 4-neighbour and 8-neighbour, so the shape depends on the parity
of the radius — which is why this could not be handed to a library's `dilate`,
and why the parity sweep runs every radius rather than a sample. And the first
attempt was **2.3× slower than numpy** (274 ms): a clamped accessor called per
neighbour per pixel is ~40 unpredictable branches per output. Hoisting the
border handling out into clamped row indices plus two end-column lines, leaving
the interior as straight-line row loops the compiler vectorises, is what turned
it around. Integer min/max is exact, so unlike §1 vectorising cannot change the
answer — the tension that killed the blur kernel simply does not exist here.

cv2's `dilate`/`erode` was the alternative and was declined: it would be the
first cv2 import under `studio/`, it interacts with `provenance.py`'s module
map, and its morphology border default is `morphologyDefaultBorderValue()`
rather than replicate. It would also have had to reproduce the octagon as
`cross^⌈r/2⌉ ∘ box^⌊r/2⌋`, which is correct but is one more thing to keep true.

## §5 — `flood_fill`: rejected

40 k cells (200×200, the gate case): **35.9 ms**, under the 50 ms gate. 90 k
cells is 87 ms, so a very large open map is perceptible, but the stated gate is
not met and nothing ships. The cost is the numpy scalar index per neighbour;
if it is ever revisited, the fix is a flat `memoryview` walk over a precomputed
`data == target` mask — the reached set is provably identical, since the
predicate is captured once from the seed and never updated.

## §6 — `ray_triangles`: rejected on the remedy, not on the number

| tris | ms |
|---|---|
| 50 000 | 10.2 |
| 200 000 | **46.0** |

The gate (>2 ms/frame) is met many times over, and the path is worse than the
plan assumed: `pick_face`'s docstring says "what a click lands on", but
`clay_view.py:1296` also calls `pick_element` from the **mouse-move** handler to
set `hover_element`, so a heavy imported mesh pays 46 ms per motion event in
element mode.

Rejected anyway, and deliberately: C would remove ~13 numpy temporaries from an
**O(n)** algorithm. A bounding-volume hierarchy makes it O(log n), and there is
already an AABB prefilter per object to hang one off. Writing the kernel would
make the wrong algorithm 3–5× faster and remove the pressure to fix it. Recorded
as a spatial-indexing item, not a kernel item.

## §7 — `render_map`: shipped, and the first benchmark was wrong

Baseline, 200×200 × 3 layers (120 000 cells): **12 590 ms**.

Two changes, both pure numpy. `_over` gained an early-out, and the oriented tile
is memoised on `(tile_id, flags)` beside the existing tileset-lookup memo.

**The early-out's first version was gated on "every source alpha is 255" and was
close to worthless.** It measured 17× on a synthetic tileset that was opaque
everywhere — and a real tile is an opaque body with a *transparent surround*, so
on a realistic tileset it never fired at all: 13 065 ms, unchanged. The gate is
now "alpha is **binary**", which is the shape a tile actually has:

| tileset | before | after | |
|---|---|---|---|
| opaque everywhere (synthetic) | 12 590 ms | 734 ms | 17× |
| binary alpha (realistic) | 13 065 ms | **3 992 ms** | **3.3×** |
| arbitrary partial alpha | 13 027 ms | 14 519 ms | no early-out; noise |

Byte-identical on **4000** randomised cases across all three alpha regimes. The
identity argument is exact rather than approximate, and one part of it is
unobvious: where both source and destination are fully clear, the full path
writes rgb 0 — discarding whatever colour was stored under a zero alpha — so the
early-out reproduces that rather than tidying it up, and computes its mask
before the copy for that reason.

A map of genuinely partial-alpha tiles is unimproved. That is the case a
per-cell `warlockc_cell_over_u8` would answer, and it is left unwritten because
no measured corpus has such a tileset.

## §8 — `clay.ops_topo.weld`: deferred

| verts | ms |
|---|---|
| 4 900 | 284 |
| 18 769 (near `WELD_SEARCH_LIMIT` 20 000) | **951** |

Past the 500 ms gate, and the plan expected it to fail — so this is a genuine
surprise rather than a confirmation. Not acted on in this batch because the fix
is not a kernel: the cost is a Python union-find building a tuple dict key per
point per each of 27 neighbourhood offsets, and the four `np.add.at` sites
(`:338-339`, `:484-485`, `:541-542`) which `np.bincount` typically beats by
10–50×. Both are pure-numpy/pure-Python changes that should be measured on their
own before any C is considered.

## §9 — `packwright.maxrects.pack`: deferred

| sprites | ms |
|---|---|
| 200 | 141 |
| 500 | **2238** |

Also past its gate, also expected to fail, also not a kernel: the growth from
200 to 500 is ~16× for 2.5× the input, which is the O(n²) `_prune` rather than
constant-factor overhead. A C port would additionally have to re-prove the
determinism contract (no set or dict iteration, exact positional tie-breaks)
against a stored corpus. Algorithmic fix first.

## §10 — `tmx._csv` / `_decode_payload`: rejected

200×200: `_csv` **4.4 ms**, `_decode_payload` **4.3 ms**, against a 50 ms gate.
An order of magnitude under. Nothing to do.

## What shipped

- `Document.preview_filter` session memo (`inker/document.py`) — §1
- `filters._straight` through `composite.to_uint8_255`, one allocation removed — §1
- `native/morph.c` + `warlockc_morph_u8`, ABI 6 → 7 — §4
- `plotter/render._over` binary-alpha early-out and the oriented-tile memo — §7

## What was written and deleted

- `native/blur.c` and the deterministic `_gaussian` rewrite — §1. Recoverable
  from this session's history if the vectorised-with-a-tolerance route is ever
  taken; the kernel was correct and bit-identical, it was simply not worth its
  own reference's cost.
