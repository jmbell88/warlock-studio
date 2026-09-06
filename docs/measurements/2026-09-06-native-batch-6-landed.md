# Native kernel batch 6 — the build half. Three numpy fixes landed, no C written.

2026-09-06, on top of `5676b777`. Machine: Windows 11, numpy 2.x, Python 3.13,
`vendor/warlockc/warlockc.dll` present (ABI 10). Every figure is the **minimum
of seven runs in a fresh process** through `scripts/bench_native.py --sweep`,
except the fixed-canvas materialize table, which is the same seven-run minimum
through the ad hoc driver the 2026-08-30 document used.

`docs/measurements/2026-08-30-native-batch-6-candidates.md` measured four
candidates and ranked the follow-up: three numpy fixes first, and only then the
two kernel questions (a replace-mode `blit_cells` sibling, an Oklab kernel).
This document is what happened when the three fixes were landed and the kernel
questions were re-asked against their original gates. **Both fell under their
gates, so no C was written** — the second batch in a row to end that way, and
for the reason the house rule predicts: a numpy constant factor nobody had
measured was most of the win.

## Summary

| # | Site | Gate | Before | After | Verdict on the kernel |
|---|---|---|---|---|---|
| I7 | `inker/filters._grow`, r=32, 1024² | >100 ms | 174.7 ms | **27.3 ms** | under gate 3.7× — **struck** |
| B4 | `inker/tiles.materialize`, 3200² @ 8 px | >200 ms per conversion | 444.1 ms | **137.3 ms** | under gate — **struck** |
| B6 | `pipelines/pixel._to_oklab`, 1024² frame residual | >200 ms | 240.7 ms | **163.6 ms** | under gate — **struck** |

All three are bit-identical with the code they replace, asserted by a parity
test that spells the old implementation out inside the test module:
`tests/inker/test_grow_fused.py`, `tests/inker/test_materialize_memo.py`,
`tests/test_oklab_lut.py`. The default lane (18 452 passed) and the
`WARLOCK_NATIVE=0` run of the touched files both pass. No ABI bump.

## §1 — `_grow` fused

The eight `_shift` calls per step (a whole-canvas `zeros_like`, a slice
assign, then an allocating `|`) became one `out.copy()` and eight in-place
`grown[slice] |= out[slice]` ORs, iterating the same `_CORNERS_4`/`_CORNERS_8`
tables; the wrap case is `grown |= np.roll(...)`. `_shift` had no other caller
and is deleted.

| steps | 2026-08-30 shipped | today |
|---|---|---|
| 4 | 21.71 ms | 3.36 ms |
| 16 | 87.01 ms | 13.22 ms |
| 32 | **174.72 ms** | **27.31 ms** |

6.4×, linear in the radius as before. The bench's `fused` and `separable`
variants are now within noise of `shipped` (27.97 / 27.80 ms), which is the
expected reading: they were the experiment and it has shipped.

## §2 — `materialize` memoised

`oriented(ts.tile_pixels(local), raw)` is now looked up in a per-call dict keyed
on the raw ref, flags included. The memo holds views and the loop only ever
copies out of them, so aliasing is safe by `oriented`'s existing contract.

Fixed 3200² canvas, the decisive shape:

| tile size | cells | 2026-08-30 | today | ratio |
|---|---|---|---|---|
| 32 px | 100² | 49.97 ms | 32.68 ms | 1.5× |
| 16 px | 200² | 131.23 ms | 54.61 ms | 2.4× |
| 8 px | 400² | **444.06 ms** | **137.27 ms** | **3.2×** |

The sweep (cells per side at 32 px) moved 212 → 131 ms at 200². What remains
is the per-cell slice copy, 137 ms at the worst realistic shape — under the
200 ms gate, and the replace-mode `blit_cells` sibling the earlier document
sketched is therefore not written. It stays sketched there if the gate ever
moves.

## §3 — `_to_oklab` sRGB→linear as a uint8 LUT

`map_palette` hands the frame in as `rgba[:, :, :3]` of a PIL RGBA array, which
is uint8. On uint8 input the sRGB→linear step is a 256-entry float64 table
built once from the *same* `np.where` expression over `arange(256) / 255.0`;
because the step is elementwise, `lut[x]` is bit-identical to evaluating it on
`x` and that equality is the whole parity argument. Any other dtype — including
the palette, which the caller widens to float64 — keeps the closed form. The
palette is not silently reinterpreted as integers.

| entries | 2026-08-30 residual | today |
|---|---|---|
| 8 | 244.50 ms | 163.91 ms |
| 32 | 242.17 ms | 164.77 ms |
| 64 | **235.70 ms** | **163.58 ms** |

1.5×. The remaining 164 ms is the matrix, the three cube roots and the
`np.stack` — under the 200 ms gate, so the Oklab kernel is not written. If it
ever is, its shape is unchanged from the earlier document: fixed-shape
elementwise, no reduction, parity a question about `cbrt` alone now that the
`pow` is gone.

## What this batch changed in the tree

`src/warlock/studio/inker/filters.py`, `src/warlock/studio/inker/tiles.py`,
`src/warlock/pipelines/pixel.py`, three new test files, this document and the
batch-6 sentence in `docs/INVARIANTS.md`. `scripts/bench_native.py` is
unchanged: its `fused` and `memo_oriented` variants stay as independent
references, and the three cases keep their gates so the numbers above can be
re-taken.

## Not benched

`retexture.combine` / `dilate` / `feather_blend`, for the third batch running
and for the reason the last two gave. Its state is still "bench first".

## Open kernel candidates, after this batch

Nothing on batch 6's list clears a gate any more. The two strongest unbenched
shapes in the tree are sequential integer work numpy cannot express and have no
bench case yet: `packwright/maxrects._prune` (190.8 s at 4096 sprites,
`docs/measurements/2026-08-31-packwright-max-sprites.md`, which is why
`MAX_SPRITES` is 1024) and `plotter/terrain._wang_cell`. Both want a
pre-registered gate and a `bench_native.py` case before a line of C.
