# TODO

Consolidated 2026-08-07 from `SUGGESTIONS.md`, `CAMERA.md` and `ANALYSIS.md`,
which this file replaces. Everything here is grounded in something observed —
the 100-unit SNES-rogue sweep (`scripts/sweep_rogue.py`, sweeps `b5c47248e13d`
"rogue - render" and `8340cd0b2f5a` "rogue - depiction") or the QA audit of the
same day — not in a reading of the code alone.

**Nothing in items 1–11 has been implemented.** Item 12 is a closed record kept
only so the deleted documents' conclusions are not lost.

**Sweep state.** Both sweeps ran to completion: 100 attempted, 83 done, 17
refused, 3.7 h GPU. All 17 refusals had one cause — more than one object in the
reference — and the occupancy and edge-runoff gates never fired once. **The 83
meshes were reviewed on 2026-08-07** (88 latest-wins verdicts in the DB, 84 of
them sweep-scoped, an 11.3 h session). **Three were accepted.** What that review
found is item 0, and it invalidates enough of the rest of this file that the
corrections are marked inline rather than left to be rediscovered.

---

## 0. Make `bg_removal=birefnet` the default, then re-run

**The review is done. This is what it found and the one thing to do about it.**

### The result

**3 accepts in 83.** All three are `bg_removal=birefnet`; `auto` went 0 for 80.
It is the only signal in the corpus — `bench/findings.json` has `bg_removal` as
the **sole** comparison with a non-zero win count (`a_wins=0, b_wins=3,
ties=1`). Every other axis, in both sweeps, is all ties: base_model, style_lora,
silhouette, palette, condition, mood.

At n=4 that would normally be a curiosity. Three things make it more:

- **The matched pairs are byte-identical upstream.** `input.png` hashes the same
  for `baseline s23` and `bg_removal=birefnet s23`, and for s42, s77 and s101.
  Same reference image, same seed, one knob — `bg_removal` is passed to
  trellis-server at reconstruction time (`trellis.py:392`), so nothing about the
  picture differs. This is a controlled A/B, not a confounded marginal.
- **The failure mode changes, not just the rate.** 58 of 80 `auto` rejects are
  tagged `broken`; 0 of 4 birefnet are (its one reject is `bad-shape`). A rate
  shift at n=4 is weak evidence; a rate shift plus the disappearance of the
  dominant failure tag is a mechanism signature.
- **It is not review drift.** The accepts land at review positions **46, 48 and
  83** of 84, with 34 consecutive rejects between the second and the third.

**The mechanism.** `doctor.py:118` — without `birefnet.gguf` matting "falls back
to a threshold cutout" — and `auto` "lets the server decide" (`trellis.py:397`).
A threshold cutout on a deliberately dark brief ("black and silver and blue")
leaves background attached, and TRELLIS reconstructs it into a solid slab.
`models/trellis2-gguf/birefnet.gguf` is present, so the learned matte was
available the whole time and simply was not being asked for.

### `hole_worst` is not weakly informative. It is backwards.

```
AUC(hole_worst predicts reject) = 0.115      (0.5 = coin flip)
rejects with hole_worst EXACTLY 0.0:  48/81 = 59%
accepts with hole_worst exactly 0.0:   0/3
median hole_worst — rejects 0.0000, accepts 0.0304
```

The accepted meshes have *more* measured holes than the median discarded one,
because a slab has no holes: `meshaudit` scores the dominant failure mode as
perfect. This is the calibration case item 8 was written to find, and it came
out inverted from what that document assumed. Anywhere below that reads a low
hole fraction as evidence of quality is wrong, and is corrected in place.

### Sweep B measured nothing, and the design is the lesson

45 units, four axes, zero accepts, every comparison a tie. `bg_removal` was
pinned at `auto` throughout, so the variable that dominates the verdict was held
fixed at its bad value while the axes under study varied. That is a floor
effect: OFAT around a baseline that fails ~96% of the time has no headroom to
detect an improvement in anything. Roughly half the GPU time bought one finding
about a knob that was an afterthought in the design.

**The rule that follows: establish a baseline that produces acceptable output at
a workable rate before fanning out.** A sweep around a broken base measures the
brokenness.

### The actions

1. Change the `bg_removal` default from `auto` to `birefnet`
   (`guidance.DEFAULT_BG_REMOVAL`), gated on `birefnet.gguf` being present —
   `doctor` already reports it, and `auto` remains correct when it is absent.
2. **Blind-confirm first if any of this is to be leaned on.** The clean 2×2 on
   its own is Fisher p=0.14; the p≈4×10⁻⁵ figure (≈8×10⁻⁴ after Bonferroni over
   ~20 blocks) comes from using all 80 `auto` units as controls, which is
   legitimate but leans on their comparability. The review was also unblinded and
   single-reviewer, and the app shows the params. 8–12 units, birefnet against
   auto, labels hidden, is cheap next to a 3.7 h sweep.
3. Re-run the render sweep with birefnet as the **baseline**, check the accept
   rate is workable, and only then re-run the depiction axes on top of it.

### What the review says about the checkpoints: nothing yet

Refusal rate and mesh quality still rank them oppositely — `playground` and
`redmond3d` refused 0/5 each; `sdxl_cfg` refused 3/5 but its survivors were
flawless, while `render3d` and `pixelxl` passed more often and produced the two
worst meshes (0.48 and 0.61 worst-view hole fraction). But every checkpoint ran
under `auto`, so all of them were being judged through the same defect. Neither
number picks a checkpoint, and the re-run is what settles it.

---

## 1. Harden the reference prompt against concept-art layouts

**Observed.** Every one of the 17 refusals was `reference.py:279` (second
component ≥ `MIN_SECOND_COMPONENT`, 8% of the largest), and they are one family
under one cause: **concept-art layouts** — character sheets (a central figure
ringed by detached armour and shield icons), turnarounds (side and front side by
side), multi-view plates. It reproduces across checkpoints, so it is a property
of the prompt, not of one model. A third unit, `baseline s23`, *passed* and drew
a full architectural background: pillars, a stone floor, a cast shadow.

**Why.** The composed prompt never asks for a single subject on plain ground.
`PROMPT_TEMPLATE` contributes "a game character, full body, standing, T-pose
neutral stance", which SDXL readily satisfies with a concept sheet.
`DEFAULT_NEGATIVE_PROMPT` does carry `multiple objects, busy background`.

**The sweep favours the positive fix.** `sdxl_cfg` — full CFG at 30 steps, where
negative adherence is *strongest* — refused most of all. That points at the
positive prompt driving the layout rather than at weak negative adherence, and
so at amending `PROMPT_TEMPLATE` over lengthening the negative.

1. Preferred: add the constraint on the positive side (`isolated on a plain
   background, single subject, no props`) so it is composed for every
   model-stage job and cannot be lost by a user editing the negative prompt.
2. Failing that: extend `DEFAULT_NEGATIVE_PROMPT` with `character sheet,
   turnaround, multiple views, item icons, reference sheet, background scenery,
   architecture`.

**The migration cost is the real decision.** `negative_prompt` is in
`VECTOR_PARAMS`, so changing it re-keys every stored vector: units recorded
before and after stop pairing in `findings.comparisons` and their marginals
split across two spellings. Same accepted cost `_LEGACY_ALIASES` documents for a
renamed taxonomy key, and it deserves the same treatment — a deliberate
decision, applied by requeuing a whole sweep, never mid-run.

## 2. Bounded reference-seed re-roll on refusal

**Observed.** Seed 11's baseline errored, so that unit is absent from sweep
`b5c47248e13d`.

**Why it costs more than one mesh.** `findings.comparisons` pairs two rows only
when they share `(sweep_id, source)` and a **seed** and differ in exactly one
key. The baseline at seed 11 was one side of *nine* prospective pairs — one per
axis value at that seed. Losing it does not cost 1/50 of the comparison power,
it costs every comparison anchored at that seed.

Let a reference refusal re-roll the *reference* seed a bounded number of times
(2–3) while holding `mesh_seed` fixed, before failing the job. `create_job`
already separates `reference_seed` from `mesh_seed`, so the machinery exists. A
unit's identity in the corpus is its config vector and its mesh seed; which of
three reference draws happened to avoid a sheet is not a setting anyone is
trying to measure.

Cheaper fallback if that is too invasive: have `service/sweeps.py` report refused
units at the end of a run, so a targeted requeue is one command rather than a
manual diff of what is missing.

## 3. Record an observation for a job refused at the reference gate

**Prerequisite for item 8.**

**Observed.** The 17 refusals produced zero `observations` rows.
`_observe_finished` records only finished model-stage jobs, so "this checkpoint
produces character sheets 60% of the time" is not in the corpus — it lives only
in the `jobs` table and dies with `prune_jobs`. It is the single most useful
thing this sweep measured about `base_model`, and `findings.json` will never
show it. A reader sees a checkpoint's accept rate *among the references that
survived*, which flatters exactly the checkpoints that fail most often.

Write an observation for a job that fails at the gate, carrying the refusal
reason as a metric (`refused_multi_object`, `refused_occupancy`,
`refused_edge`) with no mesh metrics. The `metrics` sub-object already tolerates
missing keys — every mean carries its own `counts[metric]` precisely so a
partial reading is not advertised as a whole one — so a refusal row costs
nothing structurally and turns an invisible failure into a comparable rate.

## 4. Scope findings by `prompt_hash`

**Prerequisite for item 8; promoted from nice-to-have by the judge's scope risk.**

**Observed.** Before the sweep the DB held 8 verdicts and 9 observations, **all**
against an unrelated `environment`/`wood` prompt and all rejects, and
`bench/findings.json` was a stub still carrying the legacy `platform: "pc"`
value. Those rows now confound the marginals for `base_model`, `platform` and
`negative_prompt` in a document about a character.

`verdicts` and `observations` both already carry `prompt_hash` (migration 5).
`service/findings.py` should aggregate scoped to a prompt hash, and the generate
panes should say which corpus a hint came from.

Confounding across *settings* is documented and accepted — the price of letting
daily use feed the hints. Confounding across *subjects* is not the same bargain:
what makes a good wooden crate says very little about what makes a good
character, and a hint that silently mixes them is worse than no hint.

**Now demonstrated rather than predicted.** The regenerated `findings.json`
reports `bg_removal: auto` at **n=84**, but only 80 of those are rogue sweep
units — the other 4 are the old wood-prompt rejects, silently pooled into a
marginal about a character. They happen not to change the conclusion here (all
four are rejects, and `auto` is 0 either way), which is exactly why it is worth
fixing before a case arrives where they do.

## 5. Vendor gltfpack and qualify the tiers

**Observed.** All 83 meshes: 177k–299k triangles, 0 of 83 watertight, unmoved by
any sweep axis. That is `gltfpack` being absent rather than a settings effect —
working as documented, since `vendor/gltfpack/gltfpack.exe` is missing, so `raw`
is the only profile and `Config.mesh_profile` defaults to it.

**Corrected after the review.** This item used to read "hole fraction was
essentially zero, so reconstruction quality is genuinely fine — the issue is
only budget". Both halves of that were wrong. A near-zero hole fraction was
evidence of the slab failure, not of quality (item 0), and the review accepted 3
of 83 — so budget was never the binding problem.

**The qualification corpus is not this sweep.** 80 of these 83 meshes were
rejected, and a tier test needs meshes worth keeping: the question is whether a
tier preserves UVs, both PBR maps and material assignment, which cannot be
judged on output that is already broken. Vendor the binary, but qualify the
tiers against the **re-run** (item 0), not against this corpus. The three
accepted birefnet meshes are a start and are not enough.

## 6. `MAX_UNITS = 64` measures the wrong thing

**Observed.** A 100-unit intent had to be split into two 50-unit sweeps. The
split turned out fine — arguably better, since each half has a coherent question
— but it was forced by the cap, not chosen.

Its stated rationale is time ("a model-stage unit is roughly two minutes of GPU
— 64 of them is already a couple of hours"), but it bounds one *submit*, not the
queue. Two submits of 64 are admitted without complaint, so the constant does not
cap the thing its comment is about.

Either bound total queue depth (which would enforce the stated intent), or keep
the per-submit cap and reword the comment to say what it really is — a guard
against a runaway fan-out from a mis-typed axis list, which is a legitimate and
different purpose. The present mismatch invites someone to raise the number
believing it protects them from a long run.

## 7. `art_style=snes` fights an explicit colour brief

**No code change yet — this is a note for whoever reads the sweep results.**

The brief asked for "black and silver and blue"; the composed prompt reads "…
grim dark mood, **vivid saturated colours**, bold simple shapes, …", contributed
by `ART_STYLES["snes"]`. A defensible reading of the 16-bit era, left in the
sweep's base deliberately rather than edited out, but it argues against the
stated colours.

Sweep B's `palette` axis (`steel` baseline against `mono`, `muted`, `vibrant`)
was meant to measure the tension directly. If `mono` or `muted` wins clearly,
the era fragments are over-specifying colour and should describe *shape and
shading* language only, leaving colour to `palette` and the user's own words.
That would be a real finding about `guidance.py`, and it is worth waiting for
the numbers rather than adjusting the fragment on taste.

**Still unmeasured.** Sweep B returned 0 accepts and all ties on every palette
comparison — the floor effect in item 0, not a null result about colour. The
question is untouched and rides on the re-run. Note also that a colour finding
is one of the few things `meshaudit` could never have answered even had the
meshes been sound: it is a texture judgement, so it needs human verdicts or the
image probe (item 8) regardless.

---

# The quality judge ("CAMERA")

Named for the analogy that produced it: an industrial vision system is shown good
parts and bad parts, learns the boundary, then inspects on its own. Nothing
described below exists yet.

## 8. Phase 1 — the 2D judge

**Blocked on items 3 and 4** (they are prerequisites, not neighbours) **and on a
labelling pass.**

### Why this is cheaper than it looks

The seam is already built and tested, not hypothetical:

- `verdicts.source` is a free string, never an enum, and `latest_verdicts` keys
  on `(job_id, source)`. A judge's opinion sits *beside* a human's and can never
  overwrite it (`service/verdicts.py:15-18`).
- `db.py:733 unverdicted_models(source=...)` filters per source, so a judge run
  resumes where it stopped; `tests/test_verdicts_db.py:76` already exercises
  `source="ai:demo"`.
- Verdicts are append-only, latest-wins by max rowid, and carry a
  **denormalized** `vector`, `prompt_hash` and sweep context — which is what lets
  the corpus outlive `prune_jobs` deleting the assets it was learned from.
- `models/dinov2-base` is downloaded and wired: `bench/metrics.py:145` extracts
  the normalized CLS token, `_dino_model` (`:96`) caches per `(path, device)`.
- `service/verdicts.py:38` already defines the vocabulary:
  `REASONS = ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")`.

The missing piece is **calibration**. `meshaudit` and `meshreport` measure holes,
watertightness, triangles and pivot on every job, and nothing anywhere has ever
established that `hole_worst = 0.04` means *reject*. Labelled verdicts are what
convert a measurement into a decision boundary.

**And the first calibration result is that the boundary runs the wrong way.**
Against the 84 reviewed meshes, `AUC(hole_worst → reject) = 0.115` — not
uninformative, inverted (item 0). 59% of rejects scored exactly 0.0 and no
accept did. Two consequences for anything built here. A probe must be trained on
**pixels**, never on the audit scalars, which would have to be sign-flipped to
beat a coin and would then be fitting the slab artefact rather than quality. And
`hole_worst` is disqualified as a sanity check *on* the probe: a probe that
disagrees with it is, on this evidence, more likely to be right.

### Where the factory analogy breaks

**Break 1 — two products, not one product and one blank.** In 2D mode the image
*is* the deliverable: jobs run with `output="reference"`,
`service.jobs.import_reference` mints one as a finished asset, users export it.
So the same PNG is sometimes the product and sometimes the input to the next
machine, and "good" means opposite things:

| | Good as a 2D deliverable | Good as a TRELLIS blank |
|---|---|---|
| Composition | rich, styled, dramatic | single subject, nothing else |
| Background | atmosphere, setting | plain, empty |
| Pose | expressive | neutral, T-pose |

`baseline s23` is the proof — pillars, tiled floor, cast shadow. A **better** 2D
asset than a flat T-pose on grey and a **worse** blank. One probe trained across
both label sets learns the average of two opposed objectives and is useless for
each.

**Break 2 — self-labelled data teaches the judge to imitate the rules.** It is
tempting to treat the 17 refusals as free labels. Those labels *are* the rule's
output, so a probe fitted to them can at best reproduce `reference.py` exactly,
blind spots included. `baseline s23` **passed** the rules and is still a poor
blank; no self-labelled dataset contains that case, and it is precisely the case
a learned judge exists to catch. **Labels must be human.** What makes the 2D half
attractive is not free data but *cheap* data: ~2 s to judge an image against
~15 s for a mesh (load, orbit, assess silhouette and texture) — same label count,
a fifth of the wall-clock.

### Decisions already taken

| Decision | Choice | Why |
|---|---|---|
| Authority | **Advisory only** | Files a verdict and sorts Review; never refuses, deletes or retries. The only mode in which its accuracy can be measured before it is trusted. Mirrors the `native.py` doctrine that an optimisation never replaces the reference it is checked against. |
| Labelling and training | **In-app, in Review mode** | One loop, so the judge improves as the corpus is reviewed — the analogue of the camera's teach mode. |
| Label storage | **`verdicts`, new typed `stage` column** | A real column, not a naming convention on `source`. Costs migration 6. |
| Scope | **Both 2D images and 3D meshes** | Non-negotiable requirement. |

### Three probes, one artifact pipeline

Intent, not artifact type, separates the models:

| Probe | Input | Judges | Blocked on |
|---|---|---|---|
| `image-as-product` | `reference.png` from a `reference`-stage job | Is this a good 2D asset? | a labelling pass |
| `image-as-blank` | `reference.png` from a `model`-stage job | Will this reconstruct? | a labelling pass |
| `mesh` | 8 rendered views of `model.glb` | Is this a good mesh? | **a corpus with positives in it** — see item 9 |

All three are **linear probes over frozen DINOv2 CLS embeddings** — logistic
regression on a 768-d vector. Tens to low hundreds of examples per class rather
than thousands, trains in seconds on CPU, no new dependency, weights a few KB of
`.npz`.

**The mesh probe must be max- or mean-over-8-views, never single-view.**
`bench/views.py` records a calibration over 37 finished jobs spanning every
category: the per-job argmax scattered by 330° (`silhouette_iou`) and 300°
(`dino_cosine`) against a `STABLE_YAW_SPREAD` of 30, the two metrics disagreed
with each other, and it scattered *within* categories as well as across them. The
recorded conclusion is that there is no fixed matched view at all. A single-view
mesh classifier would be learning camera pose, not quality. See
`docs/measurements/2026-08-04-view-calibration.md`.

### Architecture

- **`src/warlock/judge.py` — pure, in the `vram.py` / `memlog.py` sense.** Stdlib
  plus numpy, no imports from `service`, `queue` or `studio`, returns `None`
  rather than raising when a probe file or DINOv2 is absent. Testable headlessly,
  degrades to "unavailable" on a machine that never downloaded the weights.
- **The hand-written rules stay.** `reference.py`'s `MIN_OCCUPANCY` (0.04),
  `MIN_SECOND_COMPONENT` (0.08) and `MAX_ASPECT` (8.0) are not replaced. They
  remain the fallback when no probe is trained *and* the reference the probe's
  accuracy is reported against — the `native.py` rule that the reference
  implementation is never deleted.
- **Fully offline.** DINOv2 loads from a local path with `local_files_only=True`.
  The probe weights are ours. Nothing downloads at runtime, ever.
- **A judge failure must never fail a job.** Inference logs and swallows, exactly
  as `Worker._record_observation` does, and for the same reason.

### Data model

**Migration 6: `verdicts.stage`**, three values — `'reference' | 'blank' |
'model'`. Three rather than two, and the third is load-bearing. The instinct is a
two-value column mirroring `jobs.stage` (already exactly `'reference' | 'model'`,
`db.py:270`) and recovering intent by joining to the job. That breaks:
`prune_jobs` deletes job rows, and the sole reason `verdicts.vector` is
denormalized is that the corpus must outlive the assets. A label whose meaning
depends on a row that no longer exists is uninterpretable the moment it matters.
Existing rows backfill to `'model'` — every verdict to date is about a mesh.

**New query, `db.unlabelled_references()`.** `unverdicted_models` cannot be
reused and cannot be fixed with a parameter: it filters `status = 'done' AND
stage = 'model' AND sweep_id IS NULL` (`db.py:745-747`), excluding the two things
a labelling pass most needs — **errored jobs** (a reference refused for
multi-object is the most informative negative available) and **sweep units**
(the entire corpus this work is built on).

**`findings.py` must filter by stage.** `aggregate` runs over every source's
latest verdicts (`findings.py:103`). Matched pairs group by `(sweep_id, source)`
(`:314-323`) and stay clean, but the **marginals do not**: an image label would
count into the same accept/reject rate as a mesh verdict, and `"accept 6/8"`
under a prompt control would silently average two different questions. Mesh
findings must read `stage = 'model'` only.

### User interface

A labelling surface in Review mode, beside the existing verdict loop.

- **A thumbnail grid** of unlabelled references, `A`/`R` to label, no reason step
  — reasons are a mesh-stage concept and five classes is far more than a first
  corpus can support.
- **Texture uploads must be paced.** `ThumbnailCache` already defers release by
  one frame and never evicts an entry handed out during the current frame, but
  the pacing lesson is `viewer/sheet.StripRender`, which renders exactly **one
  cell per `step()`** because a draw plus a synchronous `read_rgba` sixteen times
  in one frame is a visible freeze. A hundred thumbnails is the same hazard,
  larger.
- **"Retrain" is a flag pumped every frame, never a direct `ctx.submit`.**
  `TaskRunner.submit` refuses a key already in flight and nothing re-arms it —
  the exact bug `AppState.findings_dirty` / `pump_findings` exists to prevent.
  Reuse that pattern.
- **Review sorts by judge score rather than filtering by it** — the filter-bubble
  guard, see item 10.
- Training runs on `TaskRunner`. A DINOv2 forward pass over a hundred images plus
  a logistic fit is seconds, and seconds on the frame thread is a freeze.

### Scope of Phase 1

Migration 6, `unlabelled_references`, the findings stage filter, `judge.py` with
embed + fit + score, the labelling grid, and both image probes. Ends with a
held-out accuracy figure. This is the whole of the value for 2D mode and the
cheapest useful gate for 3D.

## 9. Phase 2 — the mesh judge

Reuses `bench/views.py`'s 8-view render, adds the pooling adapter and the third
probe.

**No longer blocked on labelling — blocked on a corpus that contains acceptable
meshes.** The review is done and produced **3 positives against 81 negatives**.
That is not a thin corpus, it is an unusable one: a linear probe fitted to it
learns "reject" and scores 96% accuracy doing so. The risk section below flagged
17 negatives as thin; 3 positives is the worse end of the same problem.

So the ordering has changed. This phase now sits behind item 0's re-run, and the
gate on starting it is a positive count in the tens, not a total label count.
The 84 existing labels are not wasted — they are a clean negative set, and the
matched pairs (identical `input.png`, opposite verdict) are the most useful
training rows in the corpus precisely because everything except the matte is
held constant.

## 10. Phase 3 — earned authority

Only after item 11's numbers exist. Candidates, in increasing order of risk:

1. Sorting Review by score (already in Phase 1).
2. A bounded reference-seed re-roll on a predicted-bad blank — pairs with item 2
   and is cheap because `create_job` already separates `reference_seed` from
   `mesh_seed`.
3. Refusal at the gate, which should not be attempted without a measurement
   document.

### Risks to carry through all three phases

- **The filter bubble.** If a judge auto-rejects, its mistakes become invisible —
  you stop seeing what it discards and never learn it was wrong. Factories manage
  this by auditing *passed* parts on a schedule. Mitigation here is structural:
  advisory-only through Phase 2, Review **sorts** rather than filters, and a
  sampled share of the judge's own rejects keeps surfacing for human review.
- **Scope.** A probe trained on this sweep learns "good SNES rogue", not "good
  asset". Pointed at a wooden crate it is confidently wrong. This is what
  promotes item 4 from nice-to-have to prerequisite. A judge with no notion of
  subject is worse than no judge, because it is trusted.
- **Label volume — measured, and worse than feared.** The estimate here was 83
  passing and 17 refused references, with seventeen negatives called thin. The
  review came back **3 accepts and 81 rejects**, so the *positive* class is the
  scarce one and the mesh probe is unbuildable on this corpus (item 9). The
  deliberate labelling session this predicted is now confirmed necessary, and it
  has to run against output worth accepting.
- **Confounded marginals, inherited.** A verdict credits every `param: value` in
  its vector — the accepted price of letting daily use feed the hints. A judge
  trained on those labels inherits the confound. The unconfounded answers stay
  where they are: `vectors` ranked by Wilson lower bound, and `comparisons`
  recovered from sweep structure.
- **Silent staleness.** A probe is a binary artifact built from a corpus that
  keeps growing — the `vendor/warlockc` hazard exactly: an absent probe is
  obvious, a stale one quietly scoring last week's opinion is not. The `.npz`
  should carry the corpus size, the label count and a schema version, and the UI
  should say when it was trained.

## 11. What "it works" has to mean

The threshold is a constant the stored corpus is keyed on, so by this repo's own
rule it gets a **measurement document under `docs/measurements/`** before it is
baked in — the pattern `trellis_band` and `mesh_hole_max` both set.

Report, on a held-out split:

- **False-reject rate** — good assets the judge would have discarded. The number
  that decides whether it may ever gate.
- **False-accept rate** — bad assets it waves through. Cheaper: they reach Review
  anyway.
- **Agreement with `reference.py`'s rules**, on the blank probe specifically. If
  agreement is ~100%, the probe has learned to imitate the rules and has added
  nothing. Genuine value shows up as *disagreement that a human sides with the
  probe on* — `baseline s23` is the canonical test case.
- **Per-subject breakdown by `prompt_hash`**, or the scope risk is unmeasured.
- **Beat `hole_worst`, which is a floor of 0.115 AUC and therefore no floor at
  all.** Its inversion (item 0) means the honest baseline to beat is a coin
  flip, and any probe that merely correlates with the audit scalars has learned
  the slab artefact. Report the probe against the human labels directly.

### Open questions

- Does the mesh probe pool views by max or by mean? Max finds the worst angle,
  matching how `meshaudit` already reports `worst`; mean is more stable at small
  sample sizes. Decide with data, not taste.
- Binary accept/reject first, or straight to the five `REASONS` as multi-class? A
  first corpus almost certainly cannot support five classes, but the vocabulary
  exists and predicting *why* is far more actionable.
- Should the 2D probes be per-`prompt_hash` from the outset, or global with scope
  added later once the scope risk is measured rather than assumed?
- Does an `ai:` verdict feed `findings.json` at all, or only sort Review? Feeding
  it makes the corpus self-referential — the judge's own opinions become evidence
  for the hints that shape the next generation.

---

## 12. Closed — the 2026-08-07 QA audit (no action outstanding)

Recorded here only so the deleted `ANALYSIS.md` does not take its conclusions
with it. Base `master` @ `4fc9927` plus the then-uncommitted working tree. Ran in
two passes: seven layout defects fixed, then five deferred items all closed after
review. Tests went 3091 → 3110 passed, 7 skipped, lint clean; every new test was
verified red before its fix.

**One bug shape recurred seven times, and it is worth remembering:**

> `imgui.same_line()` after an item drawn at full width leaves the cursor **on**
> the content region's right edge, and an imgui child window *clips* rather than
> wraps. Whatever is drawn next is not squeezed — it is gone.

Four controls were living outside the panel: the library's favourites star, the
select-all tick, the Prune button, and every evidence hint and `(?)` in the two
generate panes. Nothing logged, nothing threw, and each looked exactly like a
feature nobody had built — the favourites feature was unusable end to end.

Fixed by `widgets.same_line_or_wrap(width)` and `widgets.hint_text(text)`, plus
two standing guards:
`test_no_pane_continues_a_line_that_has_no_room_left` (spies on `same_line` while
building all sixteen sidebar panes) and
`test_no_two_of_a_panes_icon_buttons_are_drawn_on_top_of_each_other` (collects
every icon button's rect through a real frame). The second exists because the
first cannot see a control drawn *on top of* another — which is how the restored
library `(?)` came to occupy the select-all tick's exact pixels, so pressing
select-all opened the manual.

Also fixed: `clay_ops.format_for(param)` derives a numeric format downwards from
the step and the default, because both weld distances default to `1e-4` and
imgui's implicit `"%.3f"` drew them as `0.000`; the Clay snap grid drew 1/16 m as
`0.063`; and `quality_badge(job, inline=True)` now issues its own `same_line`
only in the branches where it draws, because a `same_line` in front of a call
that draws nothing is inherited by whatever comes next.

**The one correctness bug was not a layout bug**: `CLAUDE.md` asserted something
about the filesystem that is false on Windows. A directory's mtime did **not**
move after adding a file 155 times in 200 on this machine, so `attach_files`'
`(status, job-dir mtime)` stamp could serve a stale file list permanently — the
rig-into-the-source-job case the mtime half exists for. Fixed with git's
racily-clean rule (`MTIME_RACE_NS`), written up at
`docs/measurements/2026-08-07-directory-mtime-granularity.md`, and the `CLAUDE.md`
clause corrected. The flaky test then passed 20/20.

The four remaining deferred items — the merge's weld distance being in local
units while labelled metres (now divided by the largest absolute scale
component, `max` because that is the bound holding per axis); merge silently
absorbing hidden objects (fixed in `_select_all`/`_invert` rather than in the op);
the library `(?)` restored; and the bulk bar now saying `12 selected (4 not
shown)` — are all closed and covered by tests. The Clay object merge itself was
reviewed and found sound.
