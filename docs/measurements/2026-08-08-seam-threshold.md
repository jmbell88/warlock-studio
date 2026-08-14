# What the seam ratio has to separate — 2026-08-08

Why `pipelines/seam.SEAM_MAX` moves from **2.0** to **3.5**, and what it is now
measured against.

## The question

`seam.report` divides the wrap seam — the mean absolute difference between an
image's first and last column — by the mean interior adjacent-column
difference. The ratio is the design: a hard number says nothing on its own,
because gravel legitimately differs by a lot between *any* two adjacent columns
while flat plaster differs by almost nothing.

`SEAM_MAX` is the line under that ratio, and it was a guess. Its comment said
"twice the picture's own grain… comfortably past the noise on a real SDXL tile
and comfortably short of an unpatched generation", and `TODO.md` (deleted; git history) carried it in
the deferred table as *uncalibrated*, owing a measurement document. It is
corpus-keyed — every `seam_report` on disk stores the threshold it was judged
against — so it does not move on an opinion.

Phase 5 Tier 1 is the reason it moves now: everything in the material generator
stands on this gate meaning something.

## The corpus

RTX 5090, `sdxl-turbo` at its registry settings (1024², 4 steps,
`guidance_scale` 0), `PROMPT_VERSION` 4, `TILE_TEMPLATE`, no taxonomy fields and
no LoRA. Two scripts, both deterministic on seeds 11/12/13:

- `scripts/calibrate_seam.py` — eight materials × three seeds × **two arms**.
  The tiled arm is the production path (`tile=True`, circular padding through
  the UNet and the VAE); the plain arm is the *identical* prompt, seed and
  checkpoint with the padding off.
- `scripts/calibrate_seam_hard.py` — eight hard-structured materials × three
  seeds, **tiled only**.

72 units: 48 tiled, 24 plain. The images are ~49 MB and are not tracked
(`docs/measurements/data/` is gitignored except its JSON); `results.json` and
`results-hard.json` carry every ratio, and both scripts reproduce the PNGs.

The contrast is the whole design. Measuring only tiles would say what a seamless
tile scores and nothing at all about what it has to be distinguished from.

## The measurement

| Population | n | min | max |
|---|---|---|---|
| tiled (both batches) | 48 | 0.454 | **2.500** |
| plain (batch 1) | 24 | 1.905 | 35.330 |

Sorted extremes, which is where the constant actually lives:

- tiled, highest five: 1.74, 1.78, 1.87, **2.24**, **2.50**
- plain, lowest five: **1.90**, 5.52, 6.19, 6.63, 7.65

Every unit above 2.0 in the tiled population was eyeballed through
`seam.wrap_preview` — rolled by half, so what was the wrap seam runs through the
centre of the frame, which is the only way to *see* what the ratio measures.

- **`mosaic-s13` (2.500)** — hexagonal ceramic tiles. The lattice is continuous
  through the centre cross. **Seamless.**
- **`corrugate-s11` (2.237)** — ribbed metal. Continuous through the centre.
  **Seamless.**
- **`wood-s11` (1.871)**, batch 1's ceiling — plank floor, continuous.
  **Seamless.**

And the two boundary cases on the other side:

- **`fabric-s11-plain` (1.905)** — an *untiled* image that scores below 2.0.
  Wrapped, it has no visible join: fine linen weave with no structure above the
  thread scale genuinely tiles by accident. This is a correct pass, not a
  miss — the ratio is not lying about it.
- **`metal-s11-plain` (5.521)**, the lowest genuinely-seamed unit. Wrapped, the
  cross through the centre is unmistakable at a glance.

## What that says

**Two legitimately seamless tiles score above 2.0**, and the mechanism is
visible in which ones they are. Both are large flat cells separated by thin hard
lines — grout, ridges. The denominator is a *mean* over every adjacent pair, so
a texture that is mostly flat interior has a tiny grain figure; the numerator is
one column, and if that column lands on a grout line it carries a full
grout-line contrast. The ratio rises with no seam existing. Batch 2 exists
because batch 1's ceiling (a plank floor at 1.87) already pointed at it.

So the boundary is not where the old comment put it. The evidence gives an
**empty band from 2.50 to 5.52** — no unit of either population lands in it —
and the geometric centre of that band is 3.72. Ratios are multiplicative, so the
geometric centre is the right middle.

**3.5** is the round value inside the band: 1.4× above the highest legitimately
seamless tile, 1.6× below the lowest visible seam. At 3.5 all 72 units are
classified correctly — every tiled unit passes, every visibly seamed unit fails,
and the one untiled-but-actually-seamless unit passes, which is the right
answer about it.

| threshold | tiled false alarms | plain flagged |
|---|---|---|
| 2.0 (old) | **2** / 48 | 23 / 24 |
| 2.5 | 1 / 48 | 23 / 24 |
| **3.5 (new)** | **0** / 48 | 23 / 24 |
| 5.0 | 0 / 48 | 23 / 24 |

The 23-of-24 column is the same `fabric-s11-plain` throughout and is not an
error at any threshold.

## What this does not settle

Anything in (2.50, 5.52) scores identically on this corpus — 3.5 is chosen for
its position in the band, not because the data distinguishes it from 4.0. What
the data does settle is that the answer is **not** 2.0, and that is what moves.

The corpus is one checkpoint. `sdxl-turbo` at 4 steps was the default when this
was measured (until 2026-08-11 — see
[`2026-08-11-default-base-model.md`](2026-08-11-default-base-model.md)), but a
CFG base at 30 steps draws harder edges, and the failure mode found here is
*about* hard edges. A tile base other than turbo is the first thing that should
re-run these scripts.

The band is empty because there is nothing in this corpus that seams *slightly*.
Circular padding either applies or it does not, so the plain arm is a proxy for
"the mechanism failed entirely" rather than for "it half worked". A future
partial failure — a ControlNet left zero-padded while the UNet wraps, say —
would land somewhere in the gap, and it would be the unit that decides between
3.5 and 4.0.

## Consequence for stored rows

`seam_report` carries its own `threshold`, and `inspector.seam_verdict` reads
the stored one rather than the live constant, so nothing already on disk is
reinterpreted. The cost is the one every corpus-keyed constant pays: tiles
generated either side of this change are judged against different numbers, and
the two `seamless` flags are not comparable. Nothing aggregates seam verdicts
today (they are not in `VECTOR_PARAMS` and no findings bucket reads them), so
the split is confined to what the inspector says about individual tiles.
