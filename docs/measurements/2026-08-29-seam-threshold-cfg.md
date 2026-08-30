# `SEAM_MAX` on the population that actually fails — 2026-08-29

> **Followed up 2026-08-30.** R7 left the red test's disposition to the user
> and recorded two readings of it. Reading (a) was chosen, in its strongest
> form, and the statistic this document's objective check proposed was taken to
> a held-out corpus and shipped:
> [`2026-08-30-seam-dominance.md`](2026-08-30-seam-dominance.md). Nothing below
> is amended — `SEAM_MAX` still did not move, and its refusal still stands.

**Status: pre-registration written 2026-08-29 before any image was generated.**
Everything under "What will be run" and "Decision rules" was fixed first;
numbers and the adjudication go under "Results", and whichever rule fired is
applied verbatim, including the boring one. That ordering is the only thing
that makes the answer worth anything
([`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md)).

## This run's premise had to be corrected first

This was commissioned as "`SEAM_MAX` was measured on turbo and the default is
now `sdxl_cfg`, so the constant is being applied to a population it was never
measured on". **That is not true, and the correction is the reason this
document is not a repeat.**

[`2026-08-09-seam-threshold-cfg.md`](2026-08-09-seam-threshold-cfg.md) ran
exactly that: both harnesses, `--base sdxl_cfg`, 72 units, taken 2026-08-13,
with `data/seam-cfg/results.json` and `results-hard.json` still on disk. Its
**refusal rule already fired** — a wrap-preview-confirmed seamless tile
(`mosaic-s11`, 4.288) scored above the lowest visibly seamed plain unit
(`stone-s13`, 2.705), the populations overlapped, and the recorded finding is
that *the edge-energy ratio does not separate seamless tiles from seamed ones
on a CFG base*. `SEAM_MAX` stayed 3.5 as a deliberate refusal, not as an
oversight. Two live artefacts say otherwise and both are stale: `seam.py`'s own
comment still says "re-measure per checkpoint — the corpus is turbo at 4 steps",
and `test_every_material_is_seamless`'s failure message still points the reader
at the 08-08 document as "the first thing that should re-run its scripts".
Neither was updated when the CFG run was taken.

Nothing has moved under that run since. `PROMPT_VERSION` went 4 → 7, but bumps
5, 6 and 7 are each documented as leaving the tile path byte-identical (5 is
pinned as such by `tests/test_prompt.py`'s literal; 6 and 7 add
`SCENE_TEMPLATE` and `TILESHEET_TEMPLATE`, reachable only from other job
kinds). The `sdxl_cfg` registry row is unchanged — 1024², 30 steps, guidance
7.0, `guidance_rescale` 0.0 (the rescale variant is a separate row). So
re-running the two harnesses verbatim is a *reproduction*, not a new question.

**The new question is that neither corpus contains the population that is
failing.** `tests/test_tileset_gpu.py::test_every_material_is_seamless` — the
test scoring 3.74 — does not draw what `calibrate_seam.py` draws. It draws what
`_q_tileset._tile_set` draws:

| | calibrate corpus (08-08, 08-13) | tileset materials (the failing path) |
|---|---|---|
| base | `sdxl_cfg` | `sdxl_cfg` |
| LoRA | **none** | **`pixelxl` (pixel-art-xl) at its default weight** |
| negative prompt | **none** | **`tilesheet.SHEET_NEGATIVE_PROMPT`** |
| subject | bare material phrase | `tileatlas.material_subject` + `compose_prompt` |
| template | `TILE_TEMPLATE` | `TILE_TEMPLATE` |
| size | 1024² | 1024² (`MATERIAL_PX`) |

A pixel-art style LoRA is a hard-edge, flat-cell transform. That is *precisely*
the shape both prior documents name as the ratio's own false-alarm mode: the
denominator is a mean over every adjacent pair and collapses on a mostly-flat
picture, while the numerator is one column that may land on a hard line. The
commissioning spot check — four materials at 3.16, 2.95, 3.41, 3.74, all above
the turbo corpus's entire tiled maximum of 2.50 — is a sample of this third
population, and no corpus contains it. That, not the checkpoint, is the gap.

## The question

Two, and the second is the one that is actually open.

1. **Does the 2026-08-13 CFG result reproduce on today's tree?**
   `pipelines/text2image.py` has been substantially rewritten since (v0.0.24,
   v0.0.29: the tilesheet template, the `size` override, `guidance_rescale`
   plumbing). None of it should move the tile path, and a reproduction is the
   only thing that can say so.
2. **Where does the LoRA'd material population sit, and is `SEAM_MAX = 3.5` a
   defensible line for it?** This is the population the constant is applied to
   in production by the whole seamless-tileset track, and the one with a red
   test standing on it.

## What will be run

Three arms, all on `sdxl_cfg`, passed explicitly so the record is unambiguous:

```
uv run python scripts/calibrate_seam.py      --out docs/measurements/data/seam-cfg-2 --base sdxl_cfg --model-root ~/.warlock/models
uv run python scripts/calibrate_seam_hard.py --out docs/measurements/data/seam-cfg-2 --base sdxl_cfg --model-root ~/.warlock/models
uv run python scripts/calibrate_seam.py      --out docs/measurements/data/seam-cfg-lora --base sdxl_cfg --model-root ~/.warlock/models --lora pixelxl --negative sheet
```

- **Arms A (tiled/plain, 48 units) and B (hard-structured, tiled only, 24
  units)** are the 08-13 corpus re-taken: same eight + eight materials, same
  seeds 11/12/13, same script, same checkpoint settings.
- **Arm C (48 units)** is arm A's identical prompts and seeds with the
  `pixelxl` LoRA at its default weight and `SHEET_NEGATIVE_PROMPT` applied —
  the two things that separate the failing production path from the corpus.

**No new harness.** Arm C is `calibrate_seam.py` with two optional flags added,
both defaulting to off, so the 08-08 and 08-13 documents still reproduce their
own corpora byte-identically from the same file. A harness that could not
express the population under test would be the wrong thing to preserve.

**`--out .../seam-cfg-2`, never `.../seam` or `.../seam-cfg`.** The scripts
write `{material}-s{seed}-{arm}.png` with filenames identical across
checkpoints, so writing into either existing directory would destroy a corpus a
published document reproduces, with nothing on screen to say so. This is
08-13's own operational rule and it is restated because it is the one mistake
that cannot be undone.

Every tiled unit scoring above **2.0** is eyeballed through `seam.wrap_preview`
— rolled by half, so the wrap seam runs as a cross through the centre of the
frame — **before** it counts as anything. A ratio is not a seam. That is the
08-08 procedure and it is what correctly read `mosaic-s13` at 2.500 and
`corrugate-s11` at 2.237 as legitimate grout and ridge cases.

## Decision rules, written in advance

**R1 — Reproduction.** If arms A and B land materially away from 08-13's
populations (either extreme moving by more than 25%, or the overlap verdict
flipping), the *first* finding is a change in the tile path between 2026-08-13
and today, and the threshold work is **suspended** until it is explained. A
constant may not be set from a corpus whose generator quietly moved. If they
reproduce, 08-13 stands and this document inherits it.

**R2 — Clean separation.** If a population pair separates cleanly — an empty
band between the highest wrap-preview-confirmed-seamless tiled unit and the
lowest visibly-seamed plain unit — the threshold is a round value inside that
band, chosen the way 3.5 was: the band's **geometric** centre (the metric is a
ratio, so the midpoint is multiplicative) rounded to a round number that stays
inside it.

**R3 — Overlap: the refusal rule.** If the populations overlap — any
wrap-preview-confirmed seamless tiled unit scoring at or above the lowest
visibly seamed plain unit — **the constant does not move.** The finding is
recorded as a property of the metric rather than of the threshold: the
edge-energy ratio is not a usable discriminator on this population. A re-tuned
number inside one population is false precision. This is 08-13's rule and
`2026-08-06-pixel-art-xl.md`'s rule, restated because it is the outcome the
prior CFG run already reached and is therefore the likeliest one here.

**R4 — The band exists and 3.5 is already inside it.** The constant stays at
3.5 and this document is the evidence that it survived a corpus it was not
measured on. Nothing is edited but `seam.py`'s stale comment. This is the
boring rule and it is applied verbatim if it fires.

**R5 — Arm C specifically.** A LoRA'd tiled unit above 3.5 moves nothing on its
own; a *higher* ceiling under a hard-edge style LoRA is the expected outcome
and is the ratio's known failure shape, not evidence of a seam. The number
moves only if a **wrap-preview-confirmed seamless** arm-C tile lands above 3.5
**and** R2's band exists on arm C. If a confirmed-seamless arm-C tile lands
above 3.5 and there is *no* band, R3 fires and the honest consequence is that
`seamless` is not a verdict this path can make — see R7.

**R6 — Reconciliation across checkpoints and arms.** If more than one arm
yields a band and the bands intersect, `SEAM_MAX` takes a round value in the
intersection. If they are disjoint, the threshold does **not** become
per-checkpoint or per-arm — one number, one stored `threshold` field, one
meaning of `seamless`; a per-population threshold is several spellings of one
fact and drifts the first time a base or a LoRA is added. It takes the
**larger** value, on 08-13's stated ground that for an advisory gate a false
alarm on a good tile is worse than passing a marginal one, and this document
states the cost that buys: seamed tiles on the softer arms go unflagged.

**R7 — The red test.** `test_every_material_is_seamless` is an **input to this
measurement, not a target.** The threshold is not moved to make it pass, and no
bound is widened because it is inconvenient. If the honest answer leaves the
test failing, this document says so and the test's own disposition is a
*separate* decision for the user — the two candidate readings, recorded now so
neither is invented afterwards, are (a) the assertion is measuring a real
property with an instrument that R3 says does not work on this population, in
which case it should assert wrap-preview evidence rather than a ratio, or (b)
the ratio is right and the LoRA'd materials genuinely seam. (b) is settled by
eye in Results and by nothing else.

**R8 — Stored rows, if the number moves.** `seam.report` stores the
`threshold` it judged against beside `seamless`, and `inspector.seam_verdict`
reads the stored one rather than the live constant, so **any change is
prospective by construction and nothing on disk is reinterpreted.** The cost is
the one every corpus-keyed constant pays and it is stated rather than avoided:
tiles generated either side of a change are judged against different numbers
and their two `seamless` flags are not comparable. Seam verdicts are not in
`VECTOR_PARAMS` and no findings bucket aggregates them, so the split stays
confined to what the inspector says about individual tiles. If the number does
**not** move, this paragraph costs nothing — which is a reason to prefer R3 and
R4 over a marginal adjustment, and it is written here so that preference cannot
be claimed as a discovery later.

## Results

Taken 2026-08-29. RTX 5090, `sdxl_cfg` at its registry settings (1024², 30
steps, `guidance_scale` 7.0, `guidance_rescale` 0.0), `PROMPT_VERSION` 7,
`--model-root ~/.warlock/models`. **Wall clock: 3 m 09 s (arm A, 48 units,
3.5 s/unit), 1 m 43 s (arm B, 24 units), 4 m 49 s (arm C, 48 units, 5.5 s/unit
— the LoRA load and fuse is the difference). 9 m 41 s and 120 units in total.**
Output in `data/seam-cfg-2` and `data/seam-cfg-lora`; `data/seam` and
`data/seam-cfg` untouched.

### R1: the reproduction is exact

**Max absolute delta against the 2026-08-13 corpus across all 72 units:
`0.000000`.** Every ratio reproduces bit-identically, `mosaic-s11` at 4.288 and
`metal-s13-plain` at 55.465 included. The v0.0.24/v0.0.29 rewrites of
`text2image.py` did not move the tile path, and `PROMPT_VERSION` 4 → 7 is
confirmed byte-identical here rather than merely asserted in a comment. R1 does
not suspend; 08-13 stands and this document inherits it.

### The three populations

| population | n | min | max | over 3.5 |
|---|---|---|---|---|
| A+B tiled, no LoRA | 48 | 0.733 (`circuit-s11`) | 4.288 (`mosaic-s11`) | 1 |
| A plain, no LoRA | 24 | 2.705 (`stone-s13`) | 55.465 (`metal-s13`) | — |
| **C tiled, `pixelxl` + neg** | 24 | **2.885** (`wood-s13`) | **10.658** (`brick-s11`) | **20** |
| **C plain, `pixelxl` + neg** | 24 | **0.857** (`brick-s11`) | 33.888 (`brick-s13`) | — |

- A+B tiled, highest five: `circuit-s13` 1.461, `gravel-s11` 1.488, `hex-s11`
  1.834, `checker-s12` 2.343, `mosaic-s11` 4.288.
- A plain, lowest five: `stone-s13` 2.705, `fabric-s11` 2.774, `stone-s11`
  3.038, `grass-s11` 3.078, `fabric-s13` 4.055.
- C tiled, highest five: `metal-s13` 5.121, `stone-s11` 5.131, `plaster-s11`
  6.115, `plaster-s12` 7.708, `brick-s11` 10.658.
- C plain, lowest five: `brick-s11` 0.857, `grass-s11` 0.888, `plaster-s11`
  0.944, `plaster-s12` 0.978, `brick-s12` 1.249.

**Arm C is not merely overlapping — it is inverted.** The tiled minimum (2.885)
is more than three times the plain minimum (0.857), so on the production path
there is no threshold of any value that puts more seamless tiles on the pass
side than seamed ones. The commissioning spot check (3.16, 2.95, 3.41, 3.74)
lands squarely in arm C's lower half and is a representative sample of it.

### The adjudication, by eye as pre-registered

Every arm-C tiled unit is above 2.0, so all 24 were read through
`seam.wrap_preview`, cropped to the 512² centred on the wrap cross. **Not one
has a visible join.**

- `brick-s11` (10.658, the ceiling) — pixel-art brick courses run unbroken
  through both arms of the cross, mortar lines meeting mortar lines.
  **Seamless.** It is the exact shape 08-08 named as its own false-alarm mode,
  and the LoRA sharpens it: flat brick faces give a near-zero denominator while
  a 1px pure-dark mortar line gives a full-contrast numerator.
- `plaster-s12` (7.708) and `plaster-s11` (6.115) — flat grey blockwork, cells
  and joint lines continuous across the centre. **Seamless.**
- `stone-s11` (5.131), `metal-s13` (5.121) — full-frame stone and cobble
  textures, blocks flowing through the cross. **Seamless.**
- `wood-s12` (3.423) — horizontal planks running straight through the vertical
  centre line. **Seamless.** (This is the one flagged by the objective check
  below, and only barely: 2.98 against an interior maximum of 2.92.)
- `grass-s11` (4.778) — **not a texture at all.** The LoRA plus
  `SHEET_NEGATIVE_PROMPT` rendered a grass tuft on a near-white field despite
  `tile=True` and `TILE_TEMPLATE`'s "no single focal object". The wrap cross
  runs through empty white, so the denominator collapses toward zero and the
  ratio inflates on an image that has almost no content to seam. `grass-s12` is
  the same shape. **Seamless, and separately a prompt-adherence concern.**
- The remaining 17 (`brick-s12/13`, `fabric-s11/12/13`, `grass-s12/13`,
  `gravel-s11/12/13`, `metal-s11/12`, `plaster-s13`, `stone-s12/13`,
  `wood-s11/13`) were swept as one contact sheet of seam-cross crops: every one
  is a continuous surface through the centre. **No join in any of them.**

The plain arm decides the other side, and it is worse than 08-13 found. The two
lowest-scoring units in the entire arm-C corpus are **plain** and both are
blatantly seamed to the eye: `brick-s11-plain` (0.857) is a pile of bricks in
the corner of a flat grey field, sliced into four quadrants by the wrap;
`grass-s11-plain` (0.888) is the same with a grass tuft. This is 08-13's
"why the plain floor collapsed" mechanism carried to its limit — under a
pixel-art LoRA the un-tiled arm renders a *subject on a backdrop*, whose wrap
seam crosses flat empty ground (tiny numerator) while its interior carries the
subject's hard edges (large denominator). A picture that is obviously chopped
in four scores below 1.

### An objective check, since the eye is doing so much work here

The eye says every tiled unit wraps and several plain units do not, while the
ratio says the reverse. One statistic settles which is right without a
threshold: **is the wrap seam the largest discontinuity in the picture?** —
the seam's mean column difference against the *maximum* interior
adjacent-column difference, rather than against their mean.

| population | axes where the seam exceeds **every** interior pair |
|---|---|
| A+B tiled, no LoRA | **0 / 96** |
| A plain, no LoRA | 41 / 48 |
| C tiled, `pixelxl` | **1 / 48** (`wood-s12` h, 2.98 vs 2.92 — a tie) |
| C plain, `pixelxl` | 19 / 48 |

**The circular padding is working on the LoRA path.** On 47 of 48 tiled axes
the wrap column is not even the single largest step in its own image. The
inversion in the table above is a property of the *statistic*, not of the
pictures: `SEAM_MAX` divides by a mean, and a pixel-art texture of flat cells
parted by hard lines has a mean that collapses while its maximum does not. This
is recorded as evidence, not adopted — replacing the denominator is a different
constant and per the repo rule it needs its own document and its own corpus.

### R3 and R5 fire; R2, R4 and R6 are never reached

Wrap-preview-confirmed seamless tiled units (up to 10.658) score far above the
lowest visibly seamed plain unit (0.857). The populations overlap completely on
arm C, so per the rule written above before any unit ran: **the constant does
not move.** `SEAM_MAX` stays **3.5**.

R5's condition for movement — a confirmed-seamless arm-C tile above 3.5 **and**
an arm-C band — has its first half met twenty times over and its second half
not at all, which is exactly the case R5 routes to R3. There is no band on arm
C to centre a value in, so R2 is unreachable; 3.5 is not inside a band, so R4
does not apply either; and with only arm A+B yielding anything band-shaped there
is nothing for R6 to reconcile.

The threshold sweep says the same thing the inversion does:

| threshold | C tiled flagged / 24 | C plain flagged / 24 |
|---|---|---|
| 2.0 | 24 | 14 |
| 3.5 (kept) | 20 | 14 |
| 5.0 | 5 | 14 |
| 11.0 | 0 | 5 |

Every value either flags confirmed-seamless tiles or passes visibly seamed
ones, and no value does better than chance. **On the LoRA'd material path the
edge-energy ratio is not a discriminator at all** — a stronger and more useful
finding than a number would have been, and the second checkpoint-family corpus
in a row to reach it.

### R7: the red test stays red, and that is the honest answer

`tests/test_tileset_gpu.py::test_every_material_is_seamless` fails at 3.74 and
**this measurement does not fix it, because fixing it would mean tuning a
constant to a test.** Reading (a) of R7 is what the evidence supports: the test
asserts a real and important property — that the circular padding fired — using
an instrument this document has just shown does not work on the population it
draws from. Its four materials are arm C. The objective check above says the
padding *is* firing, so the test is currently red about something that is not
wrong.

Per R7 the test's disposition is the user's decision and is deliberately not
taken here. What this run supplies is the evidence for it: the assertion needs
an instrument that survives a pixel-art LoRA, and the seam-versus-interior-
maximum statistic is the candidate — 0/96 and 1/48 against 41/48 and 19/48,
on this corpus, with no threshold tuning. Adopting it is a new constant and
therefore a new document.

### R8: nothing on disk is reinterpreted

The number did not move, so R8 costs nothing — which is what it was written to
make sayable without it sounding like a discovery.

### What this does not settle

Arm C is one LoRA at one weight. `pixelxl` at 1.2 is the tileset track's own
default and therefore the population that matters, but the finding is about
*flat-cell hard-line texture*, not about that checkpoint — any style with the
same shape should be expected to behave the same way, and a smooth-render LoRA
should be expected to behave like arm A.

Separately, and outside this document's question: **arm C produced two units
(`grass-s11`, `grass-s12`) that are a single subject on a flat backdrop despite
`tile=True` and `TILE_TEMPLATE`'s explicit "no single focal object"**, and the
plain arm produced them consistently. A style LoRA at 1.2 overriding a framing
clause is a prompt-adherence question for the tileset track, and it is recorded
here only because this corpus is where it became visible.
