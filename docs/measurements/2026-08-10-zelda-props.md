# The SNES/Zelda prop style sweep, 2026-08-10

**Status: procedure written, run not yet taken.** Everything below "What will be
run" is a pre-registration — the decision rules were written before a single unit
was queued, which is the only thing that makes the answer worth anything. When the
run happens the numbers go under "Results" and whichever rule fired is applied
verbatim, including if it is the boring one. This follows
[`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md) and
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md), the two pre-registration
exemplars in this directory.

It **supersedes nothing.** It is the depiction run that
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) deferred ("the depiction
axes are deliberately absent … they come after this run reports a workable accept
rate"), taken now because that gate cleared at 19 accepts of 41 against a
pre-registered bar of 12 of 50.

## The question

**Which taxonomy vector reads as 16-bit Zelda on a game prop, and does it cost
mesh quality?**

The taxonomy cannot say "A Link to the Past", and that is the whole reason this
needs measuring rather than picking. `art_style=snes` contributes *"vivid
saturated colours, bold simple shapes"*; `art_style=nes` contributes *"flat colour
shading, bold readable silhouette, clean dark outlines"*. ALTTP is both halves at
once — saturated **and** hard-outlined — and no key in `guidance.ART_STYLES`
carries both. So the run is a choice between two deliberately imperfect fragments,
plus two depiction knobs that plausibly close the remaining gap, plus the one
model-side thing in the pipeline that fights a flat look.

Two prior beliefs are being tested rather than assumed:

- **`genre=fantasy` may be working against the brief.** Its fragment is *"high
  fantasy, medieval, ornate craftsmanship"*, and ornate craftsmanship is the
  opposite of a 16-bit prop. `genre=cartoon` (*"playful cartoon design,
  exaggerated friendly proportions"*) is the arm.
- **`condition=worn` may be too.** 16-bit props carry no surface weathering
  because there is no resolution to carry it in. `condition=pristine` is the arm,
  and it is simultaneously a mesh hypothesis: less micro-detail is better trellis
  input.

## What will be run

```
uv run python scripts/sweep_zelda_style.py --dry-run   # plan and validate
uv run python scripts/sweep_zelda_style.py             # 60 units, 2 sweeps
```

A headless submitter (`scripts/_campaign.py`); the app's worker drains the rows.
Roughly **3 hours** of card time at the 2.54 min/unit the re-baseline measured,
and probably a little more — four of the six configurations run `playground` at 25
steps, where that figure came off a 4-step `sdxl` baseline.

Two subjects (a wooden crate, a ceramic pot), six configurations, five seeds:
`baseline`, `art_style=nes`, `art_style=ps2`, `genre=cartoon`,
`condition=pristine`, `style_lora=<none>`. The base is the re-baseline's strongest
marginals for the render half — `playground` + `render3d` + `bg_removal=birefnet`
+ `framing=three_quarter` — held fixed so that every unit varies depiction only.

**`base_model` is deliberately not an axis.** It is not one variable: `playground`
is 25 steps at CFG 3.0, `sdxl` is Hyper-SD at 4 steps and guidance 0.0, and
`models.cfg_bases()` gates the negative branch on `guidance_scale > 1.0` — so
`DEFAULT_NEGATIVE_PROMPT`'s `"multiple objects"` clause is **inert** on a 4-step
arm, and a composition-gate refusal there means something different from one on
`playground`. A win on that axis would be uninterpretable as a style claim.

### Two subjects, identical axes, and why that is the design

Findings are scoped per `prompt_hash` — a byte-exact sha1 of the raw prompt, with
no alias table — so an answer measured on one crate is a claim about crates.
`findings._comparisons` keys an entry on `(param, low, high)` and accumulates
**across sweeps**, so two plans carrying the same axes pool 5 + 5 matched pairs
into one entry spanning two prompts. That costs exactly the same 60 units as one
subject at ten seeds would, and it is the first axis reading in this repo's
history to clear `comparison_lines`' `min_pairs=5` with any breadth at all.

### One arm cannot be paired, and it is declared here rather than discovered

`guidance.normalize` writes `lora_weight` only alongside `style_lora`, so the
no-adapter arm differs from the baseline in **two** vector keys and
`findings._one_key_diff` returns `None` for it. It records no pair, no
`grade_delta`, and no entry — silently.

This already cost a reading. `sweep_rebaseline.py`'s `style_lora` axis had the
same shape, so the `style_lora=render3d` row in
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) was assembled by hand off
the grades while every other row in that table came out of `comparisons`, and that
document does not distinguish them. Here the arm is kept (it is the only
model-side question worth asking — `render3d`'s trigger is literally *"3d style,
3d render"*, prepended to every composed prompt), it is **read by hand to the same
bar as everything else**, and `tests/test_campaign_specs.py` now carries an
`UNPAIRABLE_BY_DESIGN` list so the next one is a decision rather than a surprise.

## Two questions, two instruments, never merged

This is the trap specific to a *style* sweep and it is settled before the first
verdict.

- **`grade` keeps its established meaning: is this mesh usable.** Not "does it
  look like Zelda". The 11-point scale and `USABLE_GRADE = 3` are keyed into every
  row already in the corpus and into `findings.hint`, and redefining the number
  for one campaign poisons it for every reader.
- **Style is a tag.** `on-style` and `wrong-style` are legal at any grade since
  migration 10, so a +4 mesh tagged `wrong-style` is exactly the row this run
  needs to be able to record, and so is a +1 tagged `on-style`.
- **Every unit is tagged for style.** An untagged unit is missing data, not a
  neutral one, and the style rate is computed over units that carry one.

**Stated limitation.** `findings._per_prompt` keeps only `n`, `accepts`,
`accept_rate`, `wilson_low`, `graded_n` and `mean_grade` per scoped bucket — tags
are dropped. So the `on-style` rate per arm is **not** readable off
`findings.json` and must be computed from the `verdicts.reasons` column. Written
down here because otherwise the number silently does not exist when it is looked
for.

## Decision rules, written in advance

The re-baseline's own confession is the specification for these:

> a sign test sized as though ties and losses were rare … A future OFAT run
> should state its bar as a fraction of usable pairs, decided before the run and
> evaluated after the losses are known.

Taken literally. There is no significance test below, and **none is claimed**: ten
pairs cannot support one. These are effect-size bars and a coverage floor.

### Gate A — is the corpus worth building at all

Both denominators reported, the bar written against the larger:

| Overall (of 60 submitted) | Baseline arm (of its 10) | Action |
| --- | --- | --- |
| **≥ 17 graded ≥ +3** (28%) | ≥ 3 | Phase 2 runs at the vector Gate B produces |
| ≥ 17 | < 3 | Phase 2 runs at the arm with the **highest mean grade over its 10 units**, ties broken by fewest refusals — named this way now so it cannot be chosen after the pictures are in view |
| **< 17** | any | **Stop. Phase 2 is not submitted.** The problem is upstream of every axis in this run and no contrast measured through it means anything |

28% is the re-baseline's own 25%-of-50 bar carried over and rounded to this run's
unit count.

### Gate B — the arms

Each arm is 10 seed-matched pairs (5 seeds × 2 subjects), pooled into one
`comparisons` entry.

- **Report `n_u` first** — the pairs in which *both* units carry a graded verdict.
  **No arm is read below `n_u = 7` of 10.** A fraction of usable pairs, evaluated
  after the losses are known, which is the rule the re-baseline asked for. Seven
  also sits above `comparison_lines`' `min_pairs=5`, so for the first time the
  reading comes straight out of `findings.json` with no override.
- **The statistic is `comparisons[param][…]["grade_delta"]["mean"]`**, re-oriented
  before it becomes a sentence (it is a-minus-b with a = the lexicographically
  lower value).
- **Adoption bar: `|mean grade delta| ≥ 1.5`**, and no single usable pair losing
  by more than 3 grades.
  - `≥ +1.5` → the arm is **adopted** into the phase-2 vector.
  - `≤ −1.5` → **rejected**; the baseline value stands.
  - between → **null**; the baseline value stands.
  - 1.5 on an 11-point scale is a bit under one step of the scale's own meaning
    (crossing `USABLE_GRADE` from +1 to +3 is 2.0). **This is an effect-size bar,
    not a significance bar**, and stating that is the point of pre-registering it.
- **`style_lora` is read by hand** off the grades, to the same `n_u` floor and the
  same 1.5 bar, because its `comparisons` entry is empty by construction.
- **Style is scored separately and never summed into the grade delta.** An arm is
  *style-adopted* if its `on-style` rate exceeds the baseline's by **≥ 0.3
  absolute over ≥ 7 tagged units**. Where style and mesh quality disagree:
  **style decides the phase-2 vector, mesh quality decides Gate A.** Split here,
  before it is uncomfortable.
- **Refusals are reported separately and never summed into a score**
  (`2026-08-09-rebaseline.md`'s rule, kept verbatim): `playground` and `redmond3d`
  refused 0/5 each while `sdxl_cfg` refused 3/5 with flawless survivors, and
  adding the two numbers picks the wrong arm. Read `refused_multi_object` off the
  observation metrics.
- **`hole_worst` ranks nothing in this run.** AUC against these labels is
  recomputed and reported; only ≥ 0.65 **on this corpus** would restore it as
  evidence, and the 0.756 of 2026-08-09 does not carry over — it was measured on a
  *character* corpus, and the whole finding there was that the figure moves 0.64
  when an unrelated knob moves.
- **Review with Blind on.** `ReviewState.blind` renames every unit to an id prefix
  *and* reorders by a digest of the job id, because `sweeps.expand` enqueues the
  baseline first and position otherwise names the arm. The confirm sweep stated
  this; the 50-unit re-baseline forgot, and the omission is unrecoverable.

### Choosing the phase-2 vector

1. Start from the phase-1 base.
2. Apply adopted arms in descending `|delta|`, **at most two**. Adopting four at
   once produces a vector no unit in the corpus ever ran, which is exactly the
   claim `sweep_props.WINNER`'s comment is careful not to make.
3. Two adopted arms on the same taxonomy field → take the larger `|delta|`.
4. **If nothing clears the bar — the likely outcome, and it is planned for — the
   phase-2 vector is the phase-1 base, unchanged**, and the script records that it
   is a base rather than a winner.

## The FLUX.2 klein sibling — `scripts/sweep_zelda_klein.py`

Ten units, two subjects, five seeds, no axes, submitted after the sixty above
have drained. Pre-registered here for the same reason everything else is, and
with a **weaker claim than anything above it**, stated before the numbers exist.

**It is a sibling rather than an arm because it could not be an arm.**
`models.lora_fits` is family-gated, so `base_model=flux_klein_distilled` against a
base carrying `render3d` is *refused* by `guidance.normalize` inside
`_check_unit`, and all-or-nothing admission would take the whole 60-unit campaign
with it. With the adapter dropped it would still move three vector keys at once
and could never pair. There is no way to ask this question inside that sweep.

**What it can be read as.** Both prompts are imported from
`sweep_zelda_style.py`, so they hash to the same two `prompt_hash`es and these
rows land in the *same* per-subject buckets. The comparison is therefore between
two `params.base_model` marginals inside one subject's scope — **not a matched
pair**, since `_comparisons` groups on `sweep_id` and no pair spans two sweeps.
Ten units is chosen rather than left over: the playground run's baseline arm is
five units per subject, so the two sides of the only available comparison are
exactly matched.

**Why it is worth half an hour.** The single accepted row in the live corpus is
`flux_klein_distilled` at `art_style=snes` with no style adapter, graded +4 and
tagged `on-style` — the only `on-style` accept anybody has recorded. It is n=1 and
it was a *character* at `platform=2d`, so it is not a claim about props.

**Rules, fixed in advance.** No adapter (`pixelklein` is the only LoRA that fits
the family, it is a pixel-art adapter, flat pixel art is poor trellis input, and
the +4 above ran without one; adding it would move a second key against the very
marginal this exists to read). **No arm of the playground run may be adopted or
rejected on this evidence** — it is ten units against ten, unpaired and
confounded, and its only legitimate outputs are (a) a `base_model` marginal
within each subject's bucket, reported with both denominators, and (b) an
`on-style` rate over its ten units against the playground baseline's ten.
**If klein's usable rate and `on-style` rate both exceed the playground
baseline's by the Gate B margins, that is grounds for a properly powered klein
run — never for switching the phase-2 corpus to it.** Written now, because it is
exactly the inference a good-looking result would invite.

## Phase 2 — the corpus, `scripts/sweep_zelda_corpus.py`

Written and submitted **only after** Gate A fires. Four no-axis plans (tree,
flower, rock, bush) at five seeds each, 20 units, ~1 hour — the `sweep_props.py`
shape, where every unit is the same configuration at a different seed and the
deliverable is the meshes.

**What phase 2 can and cannot say.** Four new subjects are four new
`prompt_hash`es and four new scoped buckets that phase 1 never populates, so it
**cannot confirm phase 1** and its per-subject marginals are degenerate — one
vector at five seeds gives every `param: value` in that vector identical
statistics. A later reader seeing `doc["prompts"][<tree>].params.art_style.snes`
is looking at evidence about *those five meshes*, not about `snes`. What it does
buy that is real is the `on-style` and `refused_*` rates **across subjects**: a
subject-generalisation observation, which is the deliverable and is named as one.

Each prompt is worded against the composition gate, which is what actually costs
units on this subject list. `reference.subject_mask` is a corner flood fill rather
than a matte, so anything the background reaches *through* a subject splits into
components and `MIN_SECOND_COMPONENT = 0.08` refuses the job as `multi_object`;
`sweep_refill` never re-runs a refusal, because a refusal is a measurement, so
each one permanently costs that subject a seed. The flower is the dangerous
subject (SDXL's prior for "a flower" is a cluster, a bloom on a thin stem
fragments, and a pale thin subject can also fall under `MIN_OCCUPANCY = 0.04`) and
the clay pot in its prompt is the counter-measure: it connects the components and
adds mass. The tree and bush are the same mechanism through visible sky, hence
"one solid mass of foliage". Never plural, never a scene word.

The rock's prompt is **byte-identical** to `scripts/sweep_props.py`'s — same
`prompt_hash`, so its rows join that subject's existing bucket and an ALTTP rock
can be read beside a hand-painted one, rather than opening a bucket of one.

## Results

*(to be filled in after the run; the rules above are applied verbatim)*
