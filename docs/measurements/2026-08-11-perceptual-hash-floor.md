# What a difference hash can and cannot separate — 2026-08-11

Why `bench/metrics.HASH_FLOOR` is **0.5** and not 0, and why the number
`pixel_similarity` reports is a rescale rather than the raw bit agreement.

## The question

`metrics.perceptual_hash` reduces an image to 64 bits: an 8×9 grayscale
downscale, each pixel compared to its right-hand neighbour. Two hashes are
compared by counting agreeing bits.

The obvious thing to do with that count is divide by 64 and print it as a
percentage. That is wrong, and it is wrong in a direction that misleads:
independent bits agree half the time by chance, so two images with nothing
whatsoever in common report "about 50% similar". The floor has to be measured
before it can be subtracted, and the subtraction is what makes the output
readable as a percentage at all.

The second question is what the metric is *for*. `dino_cosine` already answers
"is this the same subject". If the hash also ranked related-but-different
images, the two would be two views of one quantity and `compare` should blend
them the way `pipelines/rank` blends composition and anchor. If it does not,
they answer disjoint questions and must be reported separately.

## The corpus

`scripts`-free; the measurement is a throwaway script over three populations,
all deterministic:

- **Synthetic transforms.** One Pillow-drawn sword (256², plain gray
  background, the same fixture shape `tests/test_bench_metrics.py` uses)
  against itself under five transforms, plus a drawn shield and a drawn circle.
- **400 unrelated pairs.** 40 uniform-random-noise 256² images, seeded
  `default_rng(0)`, every pairing among them.
- **The five real PNGs tracked in this repo** — `examples/*.png` and
  `src/warlock/assets/logo.png` — all ten pairings, plus `player.png` against a
  half-resolution round trip and against JPEG quality 35.

Raw bit agreement throughout this section; the rescale is applied at the end.

## The measurement

| pair | raw agreement |
|---|---|
| image against itself | 1.000 |
| resized 256→64→256 | 1.000 |
| JPEG re-encoded at quality 40 | 1.000 |
| brightness ×1.4 | 0.828 |
| translated 6 px | 0.719 |
| sword vs shield | 0.531 |
| sword vs circle | 0.500 |

| population | n | min | mean | p95 | max |
|---|---|---|---|---|---|
| unrelated noise pairs | 400 | 0.359 | **0.520** | 0.625 | 0.750 |
| unrelated real PNG pairs | 10 | 0.344 | 0.494 | — | 0.594 |

And the two round trips on real data:

- `player.png` vs half-resolution → **1.000**
- `player.png` vs JPEG quality 35 → **1.000**

One real pairing is worth naming: `player_sheet.png` against
`player_sheet_2.png` — two different sprite sheets generated from the *same*
drawing — scores **0.359**, at the floor. That is the correct answer to "is this
the same image" and the wrong answer to "is this the same character", and the
division of labour between this metric and the DINOv2 cosine is exactly that
distinction.

## What that says

**The floor is 0.5.** Both unrelated populations centre on it — 0.520 over 400
noise pairs, 0.494 over ten real ones — and neither ever approaches the
round-trip scores. So the reported number is

```
(agreement - 0.5) / 0.5, clipped at zero
```

which is the same rescale `pipelines/rank.py` already applies to the cosine's
−1..1 range. Unrelated lands at ~0.04, identical stays at 1.0.

**Rescaling matters most exactly where the raw number looks reassuring.** The
noise p95 is 0.625 and its max is 0.750: a raw readout would call a
worst-case pair of pure-noise images "75% similar". Rescaled, that same pair is
50%, and the *typical* unrelated pair drops from 52% to 4%.

**It is a near-duplicate detector and nothing else.** Sword vs shield (0.531)
and sword vs circle (0.500) are indistinguishable from the noise mean. Two
drawn objects that a human sorts instantly are, to this metric, as unrelated as
two random-noise plates. It cannot rank "different but related" images at all,
which is why `compare` returns `pixel_similarity` and `dino_cosine` as separate
numbers and blends nothing.

**What it does survive is the whole reason to have it.** Rescaling by 4× in
each dimension and a quality-40 JPEG both leave the hash bit-identical, on
drawn fixtures and on real assets alike. A brightness multiply of 1.4 costs
0.172 raw — still 0.66 rescaled, far above any unrelated pair — because dHash
keys on gradient *direction* rather than level. That is the property that makes
it the right tool for "did this file survive a round trip" and the wrong tool
for anything else.

## What this does not settle

The translation row (0.719 raw, 0.44 rescaled) is the soft spot. A 6-pixel
shift on a 256² image is 2.3%, and it already halves the score. That is
correct behaviour for a hash — a shifted image is not the same file — but it
means `pixel_similarity` must not be used to compare renders from
almost-the-same camera. `silhouette_iou` is the metric for that, and it is in
`compare`'s output beside it.

No cutoff is chosen here, deliberately. Nothing in this measurement says where
"similar enough" is, because that depends on what the caller is deciding, and
`compare` reports three numbers with no verdict for that reason. A threshold, if
one is ever wanted, gets its own pre-registration in the style of
`2026-08-09-judge-threshold.md`.

The corpus is drawn fixtures, synthetic noise, and five real PNGs. It
establishes the floor and the round-trip ceiling, which is what `HASH_FLOOR`
needs. It says nothing about how the metric behaves on a large library of
genuinely near-duplicate assets, which is what a duplicate-detection index
would need before it existed.
