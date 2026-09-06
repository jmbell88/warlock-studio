# Native kernel batch 9 — the facing floor, fused and computed once. No C.

2026-09-06, at `21481814`. Same machine and method as batch 8
(`docs/measurements/2026-09-06-native-batch-8-retexture.md`): minimum of seven
runs in a fresh process through `scripts/bench_native.py retexture_tail
--sweep`, ten shipped views, square textures.

Batch 8's second follow-up item. After the `einsum` landed, `combine`'s
remaining constant factor was two whole-stack passes that its line profile put
at 255 ms at 2048: the facing floor (`np.where(weights >= MIN_FACING, ...)`)
and the visibility multiply. Reading `assemble` for the landing showed a
second fact the profile of `combine` alone could not: **`assemble` runs the
same two passes again** to get its own coverage sum for `feather_blend` and
`dilate`. So there are two fixes, measured separately.

## Summary

| texture | `tail` (shipped) | `tail_fused` | `tail_shared` |
|---|---|---|---|
| 512 | 72 ms | 59 ms | 53 ms |
| 1024 | 291 ms | 236 ms | 212 ms |
| 2048 | **1 170 ms** | 961 ms | **861 ms** |

Both variants are asserted `np.array_equal` against the shipped tail on a
256-square fixture before anything is timed.

## §1 — `tail_fused`: one masked multiply instead of two passes

`np.multiply(weights, vis, out=zeros, where=weights >= MIN_FACING)` writes the
product where the floor holds and leaves the zero elsewhere. Identity is an
argument about two cases: where the floor fails, the shipped path multiplies
`0.0` by a `vis` in 0..1 and gets `0.0`; where it holds, both compute the same
float32 product. 18 % off the tail at 2048 with the floor still done twice.

## §2 — `tail_shared`: do it once

`combine` returns nothing but `mixed`, so `assemble` recomputed the floor to
get `total`. Splitting `combine` into `floor_weights` and `combine_floored`
lets `assemble` compute the floored stack once, sum it once, and hand it to
both. Another 100 ms at 2048, and the shape is the point: it removes a pass
rather than speeding one up.

## Verdict

**No kernel**, as batch 8 said. Land both — they are one change — and the
2048 tail is 0.86 s: `combine` ~400 ms, `dilate` ~270, `feather` ~160, every
one a few bandwidth-bound whole-texture passes. Nothing here is a Python loop
or a stride-0 broadcast any more, which is the shape a kernel wins against.
The `retexture` kernel question is closed.

## What this batch changed in the tree

`scripts/bench_native.py`: two variants on `retexture_tail` (`tail_fused`,
`tail_shared`) and the reference implementations they time.

**Landed the same day**, commit after `21481814`: `retexture.floor_weights`
(one masked pass), `combine_floored`, `combine` as the wrapper, and `assemble`
computing the floor once. Tests: `tests/test_retexture_floor.py`, whose
once-per-assemble count fails against the old code. Default lane 18 475
passed. Re-measured with the fix shipped: `combine` 507 → 330 ms at 2048.
The bench's own `tail` column still replicates `assemble`'s *old* duplicated
floor by construction (it is the before-picture), which is why it reads
999 ms rather than `tail_shared`'s 862; the shipped `assemble` is the
`tail_shared` shape.
