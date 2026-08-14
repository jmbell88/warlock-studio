# `SEAM_MAX` on a CFG base, 2026-08-09

**Status: run taken 2026-08-13. The refusal rule fired: `SEAM_MAX` stays 3.5,
and the finding is that the ratio does not separate the two populations on a
CFG base.** The pre-registration below is unchanged; the numbers and the
adjudication are under "Results". The *method* is inherited wholesale from
[`2026-08-08-seam-threshold.md`](2026-08-08-seam-threshold.md) and restating it
here would be ceremony. Two paragraphs are genuinely new -- the refusal rule and
the reconciliation rule -- and they are exactly what an after-the-fact write-up
would fudge.

## The question

`pipelines/seam.SEAM_MAX` was closed at **3.5** on a 72-unit corpus, and the
document that closed it names its own limitation: **one checkpoint only**,
sdxl-turbo at 4 steps. A CFG base draws harder edges, and the metric is a ratio
of edge energy across the wrap seam to edge energy inside the tile. So the
question is not "is 3.5 right" but "does the ratio still separate seamless tiles
from seamed ones when the tiles themselves are sharper".

## What will be run

The same two harnesses, on `sdxl_cfg` (30 steps, guidance 7.0, the same
`sdxl-base-1.0` weights as `sdxl`, and in `models.tile_bases()` so
`calibrate_seam.py`'s own gate admits it):

```
uv run python scripts/calibrate_seam.py      --out docs/measurements/data/seam-cfg --base sdxl_cfg
uv run python scripts/calibrate_seam_hard.py --out docs/measurements/data/seam-cfg --base sdxl_cfg
```

8 materials x 3 seeds x tiled/plain = 48, plus 8 hard-structured x 3 seeds tiled
= 24. **72 units, mirroring the turbo corpus exactly.**

**`--out .../seam-cfg`, never `.../seam`.** The two scripts share an output
directory on purpose -- the hard batch is a second half of one corpus, which is
what `calibrate_seam_hard.py`'s docstring licenses -- but they write
`{material}-s{seed}-{arm}.png` with **identical filenames across checkpoints**,
because the base model appears only inside the results JSON and never in a path.
Writing this run into `data/seam` overwrites all 125 turbo files and both result
JSONs, and the measurement that closed `SEAM_MAX` at 3.5 becomes unreproducible
with nothing on screen to say so.

## Decision rules, written in advance

**The method, restated only as far as it binds.** Sort both populations, locate
the empty band between them, take its **geometric** centre (the metric is a
ratio, so the midpoint is multiplicative), and pick a round value inside it.
Every tiled unit scoring above the incumbent 3.5 is eyeballed through
`seam.wrap_preview` -- rolled by half -- **before** it counts as a false alarm.
That is the turbo run's own procedure and it is why `mosaic-s13` at 2.500 and
`corrugate-s11` at 2.237 were correctly read as legitimately seamless grout and
ridge cases rather than misses.

**The refusal rule.** If the CFG populations *overlap* -- any tiled unit scoring
above the lowest **visibly seamed** plain unit -- **the constant does not move**,
and this document records that the ratio does not separate the two populations on
a CFG base. That is a finding about the metric rather than about the threshold,
and it is a more useful one. A re-tuned number in the middle of one population is
false precision, which is `2026-08-06-pixel-art-xl.md`'s rule applied to a
different metric.

**The reconciliation rule, and it is the reason this needs pre-registering at
all.** Two outcomes, decided now:

- If the turbo empty band and the CFG empty band **overlap**, `SEAM_MAX` moves to
  a round value in the intersection.
- If they are **disjoint**, the threshold does **not** become per-checkpoint. One
  number, one stored `threshold` field, one meaning of `seamless` -- a
  per-checkpoint threshold is two spellings of one fact and drifts the first time
  a base is added. It takes the **larger** of the two values, on the stated ground
  that for an advisory gate a false alarm on a good tile is worse than passing a
  marginal one, and this document states the cost: seamed tiles on the softer
  base go unflagged.

**Null, stated in advance.** A higher tiled ceiling on a CFG base is the
*expected* outcome and is not on its own evidence to move anything. The number
moves only if a **wrap-preview-confirmed seamless** CFG tile lands above 3.5.

**Nothing on disk is reinterpreted either way.** `seam.report` stores the
`threshold` it judged against beside `seamless`, and `inspector.seam_verdict`
reads the stored one, so a change is prospective by construction.

## Results

Taken 2026-08-13, exactly as pre-registered plus one operational correction: the
library moved to `~/.warlock` on 2026-08-11, so both commands carried
`--model-root C:\Users\<you>\.warlock\models` -- the scripts' repo-relative
default now points at an empty directory. Output went to `data/seam-cfg` as
required; the turbo corpus in `data/seam` is untouched. RTX 5090, `sdxl_cfg` at
its registry settings (1024 squared, 30 steps, `guidance_scale` 7.0),
`PROMPT_VERSION` 4 -- unchanged since the turbo corpus, so the two runs compiled
byte-identical prompts. Batch 1's 48 units took 183 s of wall clock
(3.4 s/unit after the pipe load; the hard batch's harness records no
per-unit timing).

### The populations

| population | n | min | max |
|---|---|---|---|
| tiled, batch 1 | 24 | 0.862 (`gravel-s13`) | 1.488 (`gravel-s11`) |
| tiled, hard batch | 24 | 0.733 (`circuit-s11`) | **4.288 (`mosaic-s11`)** |
| plain | 24 | **2.705 (`stone-s13`)** | 55.465 (`metal-s13`) |

Tiled highest five: `hex-s11` 1.834, `checker-s12` 2.343, and then nothing until
`mosaic-s11` at 4.288. Plain lowest six: `stone-s13` 2.705, `fabric-s11` 2.774,
`stone-s11` 3.038, `grass-s11` 3.078, `fabric-s13` 4.055, `wood-s12` 5.478.

### The adjudication, by eye as pre-registered

`mosaic-s11` -- the one tiled unit above the incumbent 3.5 -- is **seamless**
through `wrap_preview`: the tile and grout pattern flow through the wrap cross
without a break. It is the same grout-line shape that produced turbo's
`mosaic-s13` false alarm, one octave up: a CFG base draws the grout darker and
the cells flatter, so the one-column numerator lands harder. All six low plain
units were wrapped and eyeballed the same way (the harness writes previews only
for the tiled arm; the six plain previews were generated with the same
`seam.wrap_preview`): **every one is visibly seamed** -- truncated rocks,
breaking weave rows, cut-off grass tufts, chopped drape folds, plank direction
flipping at the midline.

### The refusal rule fires

A wrap-preview-confirmed seamless tiled unit (4.288) scores above the lowest
visibly seamed plain unit (2.705). The populations overlap, so per the rule
written above before any unit ran: **the constant does not move**, and this
document records that the edge-energy ratio does not separate seamless tiles
from seamed ones on a CFG base. The reconciliation rule is never reached -- there
is no CFG empty band to reconcile. The threshold sweep says the same thing as
the overlap: no value works.

| threshold | tiled flagged / 48 | plain flagged / 24 |
|---|---|---|
| 2.0 | 2 | 24 |
| 2.5 | 1 | 24 |
| **3.5 (kept)** | **1** | **20** |
| 4.5 | 0 | 19 |
| 5.0 | 0 | 19 |

Every threshold either flags the confirmed-seamless mosaic or passes four to six
visibly seamed units; on this corpus there is no zero-error value, where the
turbo corpus had a 3.0-wide band of them.

### Why the plain floor collapsed, and what that bounds

The overlap is one unit wide on the tiled side but the deeper change is plain's
floor falling from turbo's 5.52 to 2.71. The mechanism is the template split:
`generate` applies `TILE_TEMPLATE`'s "flat top-down orthographic, no single
focal object" framing only on the tiled arm, and the plain arm's
`PROMPT_TEMPLATE` on a 30-step CFG base renders a crisp *subject on a flat
backdrop* -- rocks on grey ground, grass tufts on white -- rather than a
full-frame texture. Its wrap seam then crosses mostly-flat backdrop (small
numerator) while the interior carries hard object edges (large denominator), so
a unit whose seam is *obvious to the eye as truncated shapes* scores low. On
turbo the same split existed but 4-step output is soft enough that plain stayed
texture-like. So the finding is bounded: the ratio's failure is partly the
plain arm weakening as a proxy for "the mechanism failed", not only the metric
misreading true tiles. What survives either reading: `seamless` verdicts on
`sdxl_cfg` tiles are advisory at best -- a grout-structured tile over 3.5 can be
a false alarm (`mosaic-s11`), and a partially failed tiling could land in the
2.7-4.3 region where this corpus proves nothing.

### Consequence

`SEAM_MAX` stays 3.5 and nothing on disk is reinterpreted (`seam_report` stores
its `threshold` per row). The known cost, stated by the refusal rule in advance:
on a CFG base the inspector may call a legitimately seamless grout-structured
tile a visible seam. The first structural fix worth considering is not a new
number but a better denominator -- and per the repo rule, that is a new
measurement document, not an edit to this one.
