# Native kernel batch 8 — `retexture.combine` benched at last. No C written.

2026-09-06, at `3b0f12f5`. Machine: Windows 11, numpy 2.x, Python 3.13,
`vendor/warlockc/warlockc.dll` present (ABI 10); none of the three functions
has a kernel, so the DLL changes nothing here. Every figure is the **minimum of
seven runs in a fresh process** through `scripts/bench_native.py --sweep`; the
line-level timings in §2 and the working-set figures in §3 are three-run minima
from a scratch driver and are labelled as such.

`retexture.combine` / `dilate` / `feather_blend` has been "bench first" on
every batch since batch 5 §10, and unbenched on all four. This is the number.
The bench case is `retexture_tail`: `assemble`'s CPU tail after the Blender
bake — `combine`, the coverage sum, `feather_blend`, `dilate` — over the ten
shipped `VIEWS` at a square texture, with variants for each function.

## The gate, and the mistake in it

Stated before the number: **>1.0 s at `TEXTURE_PX` (1024) with the ten shipped
views.** Measured: **424 ms.** Under, and by the house rule that is the verdict
on the kernel.

But the gate was stated at the wrong size, and that is recorded rather than
repaired. `TEXTURE_PX` is the *fallback*; `_q_sprite` bakes at
`retexture.atlas_size(model_glb)`, and a TRELLIS mesh's atlas is 2048
(`docs/measurements/2026-08-08-retexture-bake.md` is why it asks rather than
assumes). The ordinary re-texture therefore runs the **2048** row below, which
is **1.5 s** and a **1.7 GB** working set. A gate re-stated at 2048 after seeing
that would be exactly the rationalisation the bench docstrings exist to
prevent, so this document reports both readings and claims neither as a
licence: the gate as written is under, the same gate at the real size would be
over, and the honest next step is a numpy fix that is measured here and
lands before the question is asked again.

## Summary

| texture | `tail` | `combine` | `feather` | `dilate` |
|---|---|---|---|---|
| 512 | 106 ms | 64 ms | 9 ms | 17 ms |
| 1024 (`TEXTURE_PX`) | **424 ms** | 258 ms | 40 ms | 67 ms |
| 2048 (a TRELLIS atlas) | **1 533 ms** | 972 ms | 148 ms | 263 ms |

Scaling is 16× cost over 16× texels at every column: arithmetic-bound, no
dispatch to remove. `combine` is 60–63 % of the tail at every size.

## §1 — Batch 5's diagnosis was wrong: the temporary is not the cost

Batch 5 §10 named the complaint as working set — `colours * w[..., None]` is a
second (n, h, w, 3) float32 stack, 500 MB at 2048 — and sketched a kernel that
"streams over texels" to remove it. The first variant tested that claim
directly: `combine_loop` accumulates `colours[i] * w[i][..., None]` in place,
view by view, in the same order `.sum(axis=0)` folds a C-contiguous stack, so
it is bit-identical (asserted) and allocates no N-view temporary.

| texture | `combine` | `combine_loop` |
|---|---|---|
| 512 | 64.3 ms | 65.4 ms |
| 1024 | 258.5 ms | 259.7 ms |
| 2048 | 971.7 ms | 1 021.3 ms |

**No faster at any size.** Removing the temporary removes nothing, so a kernel
justified by removing it would have bought nothing either.

## §2 — Where `combine` actually spends a second (2048, three-run minima)

| line | ms |
|---|---|
| `np.where(weights >= MIN_FACING, weights, 0.0).astype(f32)` | 135 |
| `w * vis` | 120 |
| `total = w.sum(axis=0)` | 16 |
| `safe = np.where(total > 0, total, 1)` | 6 |
| **`colours * w[..., None]`** | **595** |
| `prod.sum(axis=0)` | 50 |
| `/ safe[..., None]` | 61 |
| final `where` + `astype` | 45 |

The one line is 56 % of the function, and it is slow for a specific reason:
broadcasting a **stride-0 channel axis** puts numpy on its non-SIMD inner loop,
three elements at a time, over 500 MB. That is also why `combine_loop` did not
help — it broadcasts the same way per view. Four spellings of the same
weighted sum, all bit-identical to the shipped one (asserted at 256 and 2048):

| spelling of the weighted view sum, 2048 | ms |
|---|---|
| shipped `(c * w[..., None]).sum(0)` | 664 |
| per-view fold, same broadcast | 661 |
| per-view fold with `w` repeated to three channels | 338 |
| channels-first: `(c[..., k] * w).sum(0)` per channel | 284 |
| **`np.einsum("nhwc,nhw->hwc", c, w)`** | **116** |

`einsum` contracts the view axis with an in-order accumulation per output
element — the order `.sum(axis=0)` already uses on a contiguous stack — which
is why it is bit-identical and why that identity is asserted rather than
assumed in the variant builder. As a whole-function variant:

| texture | `combine` | `combine_einsum` |
|---|---|---|
| 512 | 64.3 ms | 31.2 ms |
| 1024 | 258.5 ms | 125.9 ms |
| 2048 | 971.7 ms | **446.6 ms** |

2.2× on `combine`, which takes the 2048 tail from 1.53 s to roughly 1.0 s.

## §3 — Working set (three-run minima, `peak_wset` of the bench child)

| texture | RSS before the tail | peak working set |
|---|---|---|
| 1024 | 261 MB | 450 MB |
| 2048 | 928 MB | 1 683 MB |

The fixture itself is most of the "before": ten views of colours, weights and
visibility at 2048 are 500 + 168 + 168 MB. The tail adds ~750 MB of
temporaries on top. That is a real number for a job that runs beside a loaded
SDXL, but it is the *stack* the pipeline chose to build, not `combine`'s
temporary — `einsum` leaves the peak essentially unchanged and that is fine,
because §1 showed the peak was never the time.

## Verdict

**No kernel.** Under the gate as written; and at the size the app actually
runs, the numpy fix measured here recovers half of `combine` with no C, and
the batch-2 rule says it lands first. After it, the tail at 2048 is about a
second and split three ways — `combine` ~450 ms, `dilate` ~260, `feather`
~150 — with nothing left that a kernel would take from 100 ms to 10; each is a
handful of whole-texture passes, bandwidth-bound, and the remaining constant
factor in `combine` (`where` + `w * vis` at 255 ms) is two passes that could be
fused into one `np.multiply(..., where=)` on the numpy side before any C is
considered.

## What this batch changed in the tree

`scripts/bench_native.py` only: the `retexture_tail` case with six variants
(`tail`, `combine`, `combine_loop`, `combine_einsum`, `feather`, `dilate`), the
two reference implementations the `combine_*` variants time, and their
parity asserts. Nothing under `src/`. The bench sits outside `testpaths`.

The ranked follow-up:

1. `combine`'s weighted sum as `np.einsum("nhwc,nhw->hwc", colours, w)` —
   numpy, bit-identical, 2.2× on `combine` (§2). The regression test's
   claim is the parity against the shipped expression at a size where the
   stride-0 broadcast is exercised, and it must assert equality, not
   closeness.
2. Then, still numpy: fuse the facing floor and the visibility multiply
   (255 ms of two passes over 168 MB). Not measured this round.
3. The kernel question is closed for `retexture` unless a future gate is
   stated at the size the app runs *and* both of the above have landed.
   `dilate` and `feather_blend` were never over anything.
