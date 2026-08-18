# Ground-set reduction and prompts: the comparison, 2026-08-17

> **The path this measured was deleted on 2026-08-18**, when tile sheets moved to
> Create and the AI ground set was retired with Plotter's generators. The
> *result* was not: the two-stage reduction, the detail clause and the negative
> prompt were carried across to `pipelines/tilesheet.py` unchanged, and this
> document is still what they cite. The argument transfers because it was never
> about terrain -- it is about a 128px block of pixel-art-LoRA output reduced to
> a 32px tile, which is exactly what a tile sheet's cell is. What did *not*
> transfer is the seamless-torus half: a cell in a grid is not a torus, so the
> seam-ratio column below constrained the old path and constrains nothing now.

**Status: run taken, decisions applied.** The decision rules below were written
before the run, in the shape [`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md)
established; the Results section records what came back and which rule fired.

This document exists because four constants the stored corpus is keyed on are
changing at once, all under `GROUND_VERSION` (which goes 1 → 2 in the same
change): the reduction sampler in `pipelines/ground.py:reduce_texture`, the
subject clause appended by `texture_prompts`, the default negative prompt the
service door applies, and — behind them — the phase-variant factor table and
possibly the `colors` default.

## The problem

A generated ground set comes back as near-solid tiles. The pixel-art LoRA
(`pixel-art-xl` at weight 1.2) draws ~8px "art pixels" at 1024, i.e. the true
art resolution of a generation is ~128×128 — and `reduce_texture` box-**means**
1024 → 32, so each output pixel averages a 4×4 block of art pixels. Averaging
uncorrelated art pixels regresses every material to its mean colour. Every
sibling pixel path point-samples (`pixelsheet.downscale`,
`spritesynth.reduce_atlas`, `pixel.downscale_grid`); only the ground path
averages.

## The arms

Prompts: **A** = current subjects, empty negative. **B** = subjects with the new
`DETAIL_CLAUSE` appended, plus `GROUND_NEGATIVE_PROMPT`. Both arms: the
`test_ground_gpu.py` TERRAINS (stone/water), seed 42, `tile=True`, pixelxl @1.2,
`sdxl_cfg`.

Reductions to 32×32, all partition-based (the torus survives each):

- `boxmean` — the current `reduce_texture`, integer block mean.
- `center` — pure centre sample of each partition block.
- `twostage4` — box-mean to 128×128 (each mid pixel ≈ one art pixel), then
  centre sample of each 4×4 group. The proposed default.
- reference — box-mean to 128×128 only (the art resolution; the ceiling).

## Decision rules, written in advance

1. **Sampler.** Ship `twostage4` unless `center` beats it on *both* per-tile
   contrast ratio and seam ratio on every texture.
2. **GPU contrast floor.** Absolute floor = half the winning arm's minimum
   per-tile std *iff* that is ≥ 3× the current arm's max; otherwise the ratio
   form: per-channel std of the reduced tile ≥ R × the same texture's 128px
   reference std, provisional R = 0.5, absolute backstop ≈ 6.0.
3. **Colours.** Default moves 32 → 64 only if palette occupancy at 32 is
   ≥ ~90% for the 2-terrain case.
4. **Phase factor table** (`variant_factor`): orthogonal k = 4 if
   max(tile) ≤ 64, 2 if ≤ 128, else 1; isometric k = 1 (bit-identical to
   today; a half-tile phase lattice is an explicit follow-up). Rationale:
   k·tile ≤ 1024 must hold for the reduction target to fit the generation, and
   1024/(4·32) = 8 ≈ the pixelxl art-pixel pitch, so at the house 32px tile the
   k=4 period is exactly the art resolution.

## Results

Per-tile contrast (mean per-channel std) as a ratio of the 128px reference,
over the 8 textures (2 arms × 2 terrains × fill/border):

| reduction | ratio range | seam-ratio range |
| --- | --- | --- |
| boxmean (current) | 0.554 – 0.779 | 0.63 – 1.59 |
| center | 0.967 – 1.108 | 0.62 – 1.55 |
| twostage4 | 0.920 – 0.984 | 0.71 – 1.50 |

- **Rule 1 → `twostage4`.** `center` won contrast on all 8 but lost seam ratio
  on 4 of 8 (e.g. A-t0-border 0.947 vs 0.934), so the "beats on both" clause
  did not fire. Two-stage also denoises single-pixel outliers that pure centre
  sampling inherits, at ~4% contrast cost.
- **Rule 2 → ratio form.** Winning-arm minimum std 19.28 (B, twostage4); half
  of that (9.64) is not ≥ 3× the current arm's max (28.97 → 86.9), so the GPU
  test asserts contrast ≥ 0.5 × the tile's own 128px reference std, floor 6.0.
  Observed twostage minimum ratio: 0.92 — wide margin.
- **Rule 3 → colours default 64.** Occupancy at 32 was 100% (32/32) in every
  arm/reduction; at 64 it stayed ≥ 95%. 32 was the binding constraint.
- Prompt arm B's tiles are visibly chunkier and its fill-vs-border
  distinctness rose (18.0 → 22.0 mean abs RGB, twostage4) with fill-vs-fill
  essentially unchanged (34.9 → 32.3). Seam ratios stayed < 1.6 everywhere —
  point sampling did not disturb the torus.
- All seam ratios far below the GPU lane's 3.0 guard, at 1024, at 32, in both
  arms, under all three reductions.

Contact sheets and the raw report: produced by the scratchpad harness
(`ground_diag.py`, not committed); the numbers above are transcribed from its
`report.json`.

## What changed, per these rules

- `reduce_texture` becomes the two-stage partition sampler (m capped at 4).
- `texture_prompts` appends `DETAIL_CLAUSE`; the service door defaults an
  absent negative prompt to `GROUND_NEGATIVE_PROMPT`.
- `colors` default 32 → 64 (`ground_options`, the pane's form defaults).
- `GROUND_VERSION` = 2 records all of it in the sidecar.
- Phase variants ship per the rule-4 table; the atlas grows k² sub-rows per
  terrain and `.wmap` gains a gated `phases` field (VERSION 5 iff k > 1).
