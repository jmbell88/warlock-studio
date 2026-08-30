# Replacing the seam statistic: `dominance` — 2026-08-30

**Status: pre-registration written 2026-08-30 before any held-out image was
generated.** Everything under "The statistic", "What will be run" and "Decision
rules" was fixed first; numbers and the adjudication go under "Results", and
whichever rule fired is applied verbatim, including the boring one and
including the refusal. That ordering is the only thing that makes the answer
worth anything ([`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md)).

This document exists because a *user decision* was taken on 2026-08-30 in
response to [`2026-08-29-seam-threshold-cfg.md`](2026-08-29-seam-threshold-cfg.md)'s
R7, which recorded two readings of a red test and deliberately did not choose
between them. The choice made was reading (a), in its strongest form: **promote
the seam-versus-interior-maximum statistic into `seam.py` as the shipped
verdict, replacing the edge-energy ratio.** Per the repo rule that is a new
constant keyed on the stored corpus, so it owes this.

## What is already known, and why it may not set the number

08-29's "objective check" is the *motivating* evidence for this change. Its
counts were re-derived from the PNGs on disk on 2026-08-30 before this document
was written, and they reproduce exactly:

| population | n axes | seam exceeds every interior pair | dominance max |
|---|---|---|---|
| A+B tiled, no LoRA | 96 | **0** | 0.958 |
| A plain, no LoRA | 48 | 41 | 20.304 |
| C tiled, `pixelxl` + neg | 48 | **1** (`wood-s12` h, 1.021) | 1.021 |
| C plain, `pixelxl` + neg | 48 | 19 | 3.795 |

`data/seam-cfg-2` (arms A and B) and `data/seam-cfg-lora` (arm C) are unchanged
and were only read.

**These 240 axes are the data that suggested the statistic, so they may not also
be the data that sets its constant.** That is exactly the tuning 08-29's own R7
refused ("the threshold is not moved to make it pass, and no bound is widened
because it is inconvenient"), and it applies with more force here, not less: a
new statistic proposed *from* a corpus and then fitted *to* the same corpus is
guaranteed to look excellent and says nothing about the next tile. The
motivating table is therefore treated as read-only throughout, and the constant
is set below by construction and tested on units that do not exist yet.

One thing the motivating set does establish, and it is worth stating because it
tempers the whole exercise: **dominance is not a clean separator either.** Arm
C's plain units span 0.015 to 3.795 while its tiled units span 0.102 to 1.021,
so the populations overlap heavily. What changes is *which* error the statistic
makes. The edge-energy ratio's failures are false alarms on real tiles (20 of
24 arm-C tiled units above `SEAM_MAX`); dominance's failures are misses on
seamed pictures whose seam happens to run through flat ground (29 of 48 arm-C
plain axes below 1.0). This document is a claim about the first kind of error
and makes no claim about the second.

## The statistic

For one axis of an image, with rows and colour channels averaged over:

- **seam step** — the mean absolute difference between the first column and the
  last. Unchanged; this is the numerator the ratio already used.
- **interior maximum** — the largest single adjacent-column step in the
  picture: for each interior pair, the mean absolute difference over rows and
  channels, then the maximum over pairs.
- **dominance** = seam step ÷ interior maximum.

The horizontal and vertical axes are computed separately and the worse decides,
as now.

The question it asks is **"is the wrap seam the largest discontinuity in this
picture?"** — where the shipped ratio asks "is the wrap seam larger than this
picture's *average* discontinuity". That one word is the entire difference and
it is the whole fix: a pixel-art texture of flat cells parted by hard lines has
an average that collapses toward zero and a maximum that does not, so the ratio
inflates on exactly the population the seamless-tileset track generates, while
dominance does not. Both prior documents named that false-alarm shape; neither
had a statistic immune to it.

`interior_max <= 0` — a picture of one flat colour — returns 0.0 when the seam
step is also zero and infinity otherwise, which is the rule `_ratios` already
uses and for the reason recorded there.

## The constant, fixed here and not fitted

**`SEAM_DOMINANCE_MAX = 1.0`, by construction.**

It is the statistic's own semantics rather than a value read off a
distribution: at exactly 1.0 the wrap seam is precisely as large as the largest
step the picture already contains, so "the seam is not the worst join in the
image" is the verdict, spelled arithmetically. There is no band to centre a
value in and no geometric mean to take, and that is a feature — the number
cannot drift with a corpus because it was never derived from one.

The motivating set is *consistent* with 1.0 (144 tiled axes, one of them over,
by 2%), and this document does not treat that as evidence for 1.0 because it
would have been written the same way had those numbers been slightly different.
What the held-out corpus can do is **falsify** 1.0, and the rules below say what
happens when it does.

## The question

1. **Does 1.0 hold on tiles it has never seen?** How many wrap-preview-confirmed
   seamless tiled units, drawn at seeds no published corpus contains, score
   above 1.0?
2. **Does dominance keep the sensitivity the ratio has?** How many visibly
   seamed plain units does it catch, on the same held-out draw?

## What will be run

Three arms on `sdxl_cfg`, at **seeds 21, 22, 23** — held out by construction,
since every published corpus is 11/12/13 and seeds are the only free variable
that leaves the contrast otherwise identical:

```
uv run python scripts/calibrate_seam.py      --out docs/measurements/data/seam-dom-plain --base sdxl_cfg --model-root ~/.warlock/models --seeds 21,22,23
uv run python scripts/calibrate_seam.py      --out docs/measurements/data/seam-dom-lora  --base sdxl_cfg --model-root ~/.warlock/models --seeds 21,22,23 --lora pixelxl --negative sheet
uv run python scripts/calibrate_seam_hard.py --out docs/measurements/data/seam-dom-hard  --base sdxl_cfg --model-root ~/.warlock/models --seeds 21,22,23
```

- **Arm D (48 units)** — arm A's eight materials, tiled and plain, no LoRA.
- **Arm E (48 units)** — arm C's population: the same eight, tiled and plain,
  `pixelxl` at its default weight with `SHEET_NEGATIVE_PROMPT`. This is the
  population the shipped seamless-tileset track actually draws and the one the
  red test samples, so it is the arm that matters.
- **Arm F (24 units)** — arm B's eight hard-structured prompts, tiled only. The
  grout-and-rivet shapes that broke the ratio's denominator at 2.0.

**No new harness.** `--seeds` is one optional flag added to the two existing
scripts, defaulting to `11,12,13`, so the 08-08, 08-09 and 08-29 documents still
reproduce their corpora byte-identically from the same files. A harness that
could not express a held-out draw would be the wrong thing to preserve.

**`--out .../seam-dom-*`, never any existing directory.** The scripts write
`{material}-s{seed}-{arm}.png` and a seed collision would overwrite a corpus
three published documents reproduce, with nothing on screen to say so. This is
08-13's operational rule and it is restated because it is the one mistake that
cannot be undone. The new seeds make a collision impossible, which is belt and
braces rather than a reason to relax it.

Every tiled unit scoring above **0.8** is eyeballed through `seam.wrap_preview`
— rolled by half, so the wrap seam runs as a cross through the centre —
**before** it counts as anything, and every plain unit scoring *below* 1.0 is
eyeballed too. A number is not a seam and the absence of a number is not the
absence of one. That second sweep is new and it is the one this document owes
that 08-29 did not: a statistic is being adopted for its misses as well as its
false alarms, so the misses get looked at.

## Decision rules, written in advance

**R1 — Reproduction is not available and is not claimed.** New seeds mean new
pictures, so there is no bit-identity check to run here as 08-29 had. The
substitute is that arms D and F use the same prompts, script and checkpoint
settings as A and B: if their *population shapes* land wildly away from A and B
(either extreme moving by more than an order of magnitude), the first finding is
that something in the tile path moved and the threshold work is **suspended**
until it is explained.

**R2 — 1.0 holds.** If no wrap-preview-confirmed seamless tiled unit in D, E or
F scores above 1.0, the constant ships at 1.0 and this document is the evidence.
This is the boring rule and it is applied verbatim if it fires.

**R3 — 1.0 is exceeded by a confirmed-seamless tile.** The number does **not**
move to whatever value clears the corpus — that is the fit this document exists
to avoid. Instead the excess is *reported as a rate* and the choice is between
two stated options, decided by the size of that rate:
  - **at most 5% of tiled axes over 1.0, none by more than 1.25×** — 1.0 ships
    with the false-alarm rate recorded here as a known and quantified cost. An
    advisory verdict is allowed to be wrong 1 time in 20 provided the number is
    published; the wrap preview is one click away in the inspector and is what
    the verdict tells the reader to look at.
  - **more than 5%, or any confirmed-seamless tile above 1.25×** — the constant
    does not ship at all. R5 fires.

**R4 — Sensitivity floor.** Dominance must catch at least as many *visibly
seamed* plain units as `SEAM_MAX` does on the same held-out draw. If it catches
fewer, the change trades a false-alarm problem for a miss problem and the
honest answer is that neither statistic works; R5 fires. Both are scored on the
same pictures, so this is a paired comparison and not two separate claims.

**R5 — The refusal rule.** If R3's second branch or R4 fires, **nothing in
`seam.py` changes.** The finding is recorded as a property of the metric rather
than of the threshold, `SEAM_MAX` stays 3.5 with its existing caveats, and the
red test's disposition returns to the user with this document's numbers
attached. A statistic that needs a fitted constant to look good is the thing
both prior documents refused, and swapping which statistic gets fitted would not
be a different answer.

**R6 — One number, not one per population.** If D, E and F disagree about where
a line would sit, the constant does **not** become per-checkpoint, per-LoRA or
per-arm. One number, one stored field, one meaning of `seamless`; a
per-population threshold is several spellings of one fact and drifts the first
time a base or a style is added. This is 08-29's R6 and it is inherited
unchanged.

**R7 — What ships, if anything ships.** The edge-energy ratio is **kept and
still reported** — it is free to compute, three documents are keyed on it, and
removing it would make every stored row unreadable. What changes is which number
decides `seamless`. `report()` gains `dominance` and a `metric` field naming the
statistic that decided; `worst` continues to carry the ratio so no stored row
changes meaning.

**R8 — Stored rows and the reader.** `seam.report` stores the threshold it
judged against and `inspector.seam_verdict` reads the stored one, so the change
is prospective by construction and **nothing on disk is reinterpreted.** The new
part is that the *statistic* moves, not just its constant, so the wording must
move with it: `seam_verdict` branches on the stored `metric` and keeps saying
"edge/grain" for every row written before today, which is the only honest thing
it can say about a number computed that way. A row with no `metric` field is a
pre-change row by definition. The cost is stated rather than avoided: tiles
either side of this change are judged by different instruments and their two
`seamless` flags are not comparable. Seam verdicts are not in `VECTOR_PARAMS`
and no findings bucket aggregates them, so the split stays confined to what the
inspector says about individual tiles.

**R9 — The red test.** `tests/test_tileset_gpu.py::test_every_material_is_seamless`
is an **input to this measurement, not a target.** If R5 fires the test stays
red and this document says so. If the constant ships, the test is rewritten to
assert dominance — and it is rewritten to assert the shipped verdict rather
than a number of its own, so there is one definition of "seamless" in the tree
and not two.

## Results

Taken 2026-08-30. RTX 5090, `sdxl_cfg` at its registry settings (1024², 30
steps, `guidance_scale` 7.0, `guidance_rescale` 0.0), `PROMPT_VERSION` 7,
`--model-root ~/.warlock/models`, seeds 21/22/23. 120 units. Output in
`data/seam-dom-plain` (D), `data/seam-dom-lora` (E) and `data/seam-dom-hard`
(F); `data/seam`, `data/seam-cfg`, `data/seam-cfg-2` and `data/seam-cfg-lora`
were read and not written.

### R1: the populations are where A and B put them

Arm D's tiled dominance runs 0.075–0.940 against A+B's 0.044–0.958, and arm F's
0.038–0.923 against B's 0.080–0.958. Neither extreme moves by anything like an
order of magnitude, and the plain arms' edge-energy ratios sit in the same range
as 08-13's. Nothing in the tile path moved between 2026-08-29 and today. R1 does
not suspend.

### The specificity claim: 144 held-out tiled axes

| arm | units | axes | dominance > 1.0 | dominance max | `SEAM_MAX` flags |
|---|---|---|---|---|---|
| D tiled, no LoRA | 24 | 48 | **0** | 0.940 | 1 / 24 |
| E tiled, `pixelxl` + neg | 24 | 48 | **0** | 0.857 | **15 / 24** |
| F tiled, hard-structured | 24 | 48 | **0** | 0.923 | 2 / 24 |
| **total** | **72** | **144** | **0** | **0.940** | **18 / 72** |

**Not one held-out tiled unit scores above 1.0 on either axis.** The
pre-registered eyeball threshold of 0.8 caught sixteen units and every one was
read through the wrap cross:

- Arm D — `stone-s22` 0.940, `plaster-s23` 0.892, `plaster-s21` 0.889,
  `grass-s21` 0.859, `stone-s23` 0.845, `wood-s22` 0.830, `stone-s21` 0.812,
  `fabric-s21` 0.807. Boulder, brushed plaster, turf, gravel and vertical wood
  grain all run unbroken through both arms of the cross. **All seamless.**
- Arm E — `fabric-s22` 0.857, `fabric-s21` 0.848. Pixel weave continuous.
  **Seamless.**
- Arm F — `herring-s23` 0.923, `herring-s21` 0.902, `mosaic-s21` 0.861,
  `corrugate-s22` 0.844, `herring-s22` 0.837, `hex-s21` 0.827. Parquet chevrons,
  grout lines, ribs and hexagon joints all continuous. **All seamless.**

All eighteen units `SEAM_MAX` flags were read the same way and **every one of
them wraps.** Fifteen are arm E, which is the population the shipped
seamless-tileset track draws and the one the red test samples: pixel cobbles,
weave, blockwork and foliage, all continuous through the cross, scoring 3.55 to
7.40 on the ratio and 0.45 to 0.86 on dominance. `F mosaic-s21` reproduces the
08-08 document's own false-alarm shape exactly — ceramic tiles with grout, ratio
8.56, dominance 0.861. `D metal-s21` is the worst single case: brushed steel,
whose near-uniform vertical grain gives a collapsed mean denominator, at ratio
**13.54** and dominance 0.72.

So on this draw the shipped statistic's false-alarm rate on real tiles is
**18/72 = 25%**, and dominance's is **0/72**.

### The sensitivity claim: 48 held-out plain units

Every plain unit was read through the wrap cross, as pre-registered — including
the ones both statistics pass, which is the sweep 08-29 did not do.

| arm | units | visibly seamed | dominance catches | `SEAM_MAX` catches |
|---|---|---|---|---|
| D plain, no LoRA | 24 | **24** | 22 | 19 |
| E plain, `pixelxl` + neg | 24 | **20** | 18 | 20 |
| **total** | **48** | **44** | **40** | **39** |

Arm E's four non-seamed units are `grass-s22`, `metal-s22`, `plaster-s22` and
`wood-s22`: a featureless flat grey field in each case, with no content to join.
Under a pixel-art LoRA the un-tiled arm sometimes renders nothing at all, which
is the limit case of 08-29's "subject on a backdrop" finding. Neither statistic
flags them and neither should — there is no seam because there is no picture.
Three of the ratio's four misses are exactly these; dominance misses them too.

The disagreements, which are the whole content of that table:

- **Dominance catches, the ratio misses (5 units).** Arm D `brick-s21` (d 2.19,
  r 2.9), `brick-s22` (2.34, 3.0), `fabric-s23` (1.21, 2.1), `stone-s21` (1.49,
  2.4), `stone-s22` (2.26, 3.1) — all five blatantly quartered by the wrap, all
  five under `SEAM_MAX`.
- **The ratio catches, dominance misses (4 units).** Arm D `metal-s21` (d 0.69,
  r 31.3) and `metal-s22` (0.86, 25.4); arm E `brick-s22` (0.97, 14.3) and
  `metal-s21` (0.29, 5.7). Every one is a picture that already contains a step
  as hard as its own seam — panel edges, a plank/metal boundary, a pixel-art
  mortar line — so the seam ties or loses against the interior maximum and
  passes. `D metal-s22` is the sharpest illustration and the most uncomfortable
  one: brushed metal above, white planks below, an obvious join, dominance 0.86.

**R4 passes on the pooled held-out draw, 40 against 39, and it is close.** The
per-arm split is reported because it is the honest shape of the result:
dominance is better by three on arm D and worse by two on arm E, which is the
production population. Nothing here supports a claim that dominance is a *more
sensitive* instrument. The claim it supports is that it is not a less sensitive
one, while being enormously more specific.

### R2 fires; R3 and R5 are never reached

No wrap-preview-confirmed seamless tiled unit scores above 1.0 — none scores
above 1.0 at all, over 144 held-out axes across three populations. R3's rate
arithmetic has nothing to operate on, R4 passes, and R5 does not fire.

**`SEAM_DOMINANCE_MAX` ships at 1.0**, the value written into this document
before the first held-out unit was drawn.

### What shipped

- `seam.py` gains `_dominance` and `SEAM_DOMINANCE_MAX = 1.0`. `report()` gains
  `dominance`, `dominance_horizontal`, `dominance_vertical` and `metric`;
  `seamless` and `threshold` now come from dominance. `SEAM_MAX` and `worst` are
  **kept and still reported**, unchanged, per R7.
- `inspector.seam_verdict` branches on the stored `metric` and words a row with
  the statistic that judged it — "edge/grain" for every row written before
  today, "seam/worst join" after. Its "likely seamless" hedge stays, for the
  narrower and better-measured reason below: the miss direction.
- `tests/test_tileset_gpu.py::test_every_material_is_seamless` asserts
  `report["seamless"]` rather than a number of its own, so there is one
  definition of the word in the tree (R9). **Re-run on a card after the change:
  7 passed.**
- `tests/test_seam.py` needed one assertion rewritten, and it is worth naming
  because it is the semantic change in miniature: a picture that is one colour
  on the left half and another on the right was asserted *not* seamless. Under
  dominance it ties at exactly 1.0 and passes — correctly, because a stripe
  repeats without introducing any join it does not already contain. The test
  that meant "one hard join across the wrap" now ramps its interior transition,
  leaving the wrap the single worst step in the frame, and the stripe case is
  kept as its own test pinning the disagreement.

### R8: nothing on disk is reinterpreted

Stored rows keep their own `worst`, `seamless` and `threshold`, and gain no
fields retroactively. A row with no `metric` is a pre-2026-08-30 row by
definition and the inspector still describes it as an edge/grain ratio judged
against 3.5. The cost is the one every corpus-keyed constant pays and it is
larger here than usual because the *statistic* moved and not only its number:
tiles either side of today are judged by different instruments and their two
`seamless` flags are not comparable. Seam verdicts are not in `VECTOR_PARAMS`
and no findings bucket aggregates them, so the split stays confined to what the
inspector says about individual tiles.

### What this does not settle

**The miss direction is measured loosely and must not be quoted as a strength.**
Four of 44 visibly seamed control units passed, and the mechanism — a picture
whose interior already contains a step as hard as its seam — puts no bound on
how bad the missed seam can look. `D metal-s22` is an obvious join at 0.86. An
advisory verdict is the right place for that and the inspector still says
"likely"; a gate would not be, and nothing here argues for making one.

Arms D and F are one checkpoint and arm E is one LoRA at one weight. The finding
is about *flat-cell hard-line texture* rather than about `pixelxl`, and any
style with that shape should be expected to behave the same way.

Separately, and outside this document's question: arm E reproduced 08-29's
prompt-adherence concern twice over. Four of its 24 plain units rendered an
empty grey field, and `wood-s21` tiled came back as wood planks parted by a wide
blank band. `tile=True` and `TILE_TEMPLATE`'s "no single focal object" are not
reliably surviving a style LoRA at 1.2, and that is a question for the tileset
track rather than for the seam statistic.
