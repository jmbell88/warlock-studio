# LEFTOVERS

**The one place outstanding work is written down.** Consolidated 2026-08-08 from
the four documents it replaces — `TODO.md`, `NEXT_ROADMAP.md`, `docs/LIST.md`
and `docs/BUILD_PLAN.md` — after every item in all four was checked against the
source rather than against the documents' own claims. `git log --diff-filter=D`
finds all four if an old reference needs chasing.

**Nothing here is ticked.** A finished item is deleted, not ticked: a plan whose
boxes disagree with the tree is worse than no plan, and `../CLAUDE.md` plus the
comments at each site are the record of what shipped and why.

## What the verification found, so it is not re-run

Every item in the four documents was checked in source. What was *implemented*
is deliberately absent below; this is the summary of it, so a reader who
remembers a list item and cannot find it here knows it landed rather than that it
was lost:

- **All 44 speed items** — `A1`–`A10`, `B11`–`B28`, `C29`–`C38`, `D39`–`D42`,
  `L101`–`L103`. Two are narrower than written and both meet the intent: `A4`'s
  indexes are declared ascending (SQLite serves the `DESC` use by reverse scan),
  and `C38` backs the idle poll off 1 s → 5 s rather than removing the executor
  hop, which is deliberate — the hop is the missed-wake backstop.
- **All 38 clarity items** — `E43`–`E53`, `F54`–`F59`, `G60`–`G64`, `P120`,
  `S136`–`S140`.
- **All 43 UX items** — `H67`–`H74`, `I75`–`I84`, `J85`–`J91`, `K92`–`K100`,
  `M105`–`M107`, `N108`–`N115`, `O116`–`O119`, `P124`. Two landed narrower on
  purpose and are recorded in §15 so they are not rediscovered as defects.
- **Inker** `Ink1`–`Ink6`, `Ink8`, `Ink10`–`Ink13`; **Clay** `Clay14`–`Clay17`,
  `Clay25`.
- **The material generator and mesh re-texturing** in full.
- **The whole quality-judge code path** — migration 7 (`verdicts.stage`),
  `db.unlabelled_references`, the findings stage filter, `warlock/judge.py`,
  `service/judge.py`, and the labelling grid in Review with both image probes.
  It is unused because it is waiting on a human, not on code (§7).

Three citations in the deleted documents were stale and are corrected here rather
than carried: the command palette is `../src/warlock/studio/panes/palette.py` (with
its pure half in `../src/warlock/studio/palette.py`); the configuration chapter is
`manual/16-configuration.md`; `manual` is numbered `00-index` through
`21-extending`, twenty-two chapters. **Two of those three corrections were
themselves wrong as first written** — the chapter was given as
`14-configuration.md`, which is the *shortcuts* chapter, and the renumbering was
given as "01→19" — and both were repaired on 2026-08-09 against the tree. A
correction is not exempt from the rule it enforces.

**A bare path in this file means `src/warlock/<path>` unless it starts with
``, `../scripts`, `../tests`, `../native` or `../vendor`.** That matters because a
real top-level `../bench` output directory exists beside the `../src/warlock/bench`
package, so "`bench/findings.py`" is ambiguous on its face; where it could be
read either way below, it is written out in full.

## The section numbers are load-bearing — do not renumber

Eighteen source, test and script files cite `TODO.md §2`, `§3`, `§5`, `§7`, `§8`
and `§10` **by number** — `../src/warlock/judge.py`, `../src/warlock/service/judge.py`,
`../src/warlock/service/findings.py`, `../src/warlock/tiercheck.py`,
`../src/warlock/studio/widgets.py`, `../src/warlock/studio/review_mode.py`,
`../scripts/sweep_confirm.py`, `../scripts/sweep_rebaseline.py`,
`../scripts/sweep_rogue.py`, `../scripts/qualify_tiers.py` and eight test docstrings.
A further handful (`../src/warlock/db.py`, `../src/warlock/sweep.py`,
`../scripts/calibrate_seam.py`) name `TODO.md` without a section number, which is
the provenance case below. `§4` and `§6` are cited by nothing: §4 is carried
anyway, and §6 was deleted when its three UI decisions shipped — the two test
docstrings that still pointed at §6 now state the decision instead of citing it,
which is what a section with no live section number is owed. **A citation reading
`TODO.md §N` means this file's §N.** Every
carried section keeps the number it had, which is why the numbering has gaps
(§6's three UI decisions shipped, and §13–§16 are new). Renumbering would strand
all twenty-five at once, silently; a new section takes the next free number.

The other direction is different and needs no repair. A comment naming
`NEXT_ROADMAP`, `docs/LIST.md` or `BUILD_PLAN` — `db.py`'s "NEXT_ROADMAP A4-A7"
on the read-path indexes, four test-module docstrings, two measurement
documents — is **provenance for work that shipped**, not a pointer at a live
plan. Those items are recorded as done in the summary above and nowhere else,
which is correct: `git log --diff-filter=D` has the original wording, and
repointing them here would claim this file specifies work it only reports.

**All of §1–§10 needs a GPU, a network or a human rather than code.** §11–§14 are
code. Where anything here disagrees with a comment in the tree, the tree is
right and this file is stale — say so by editing it.

---

## 1. Verification debt — three runs, and none of them are optional

Four packages were built and tested entirely headless. Every claim below is
proven at the level of bookkeeping, arithmetic and imgui frames building, and
**unproven at the level of the thing actually working**. This has the shortest
path to an unpleasant surprise of anything in the file, and nothing under
`measurements` records any of the three.

1. **One real N-candidate promote.** Candidates 1/2/3 is proven as columns,
   seeds, admission, dissolve, filter and picker. Nothing has ever reconstructed
   three meshes. What is untested is the *feel*: three real two-minute runs
   queueing in order, the picker updating as each lands, and the viewport
   swapping between finished candidates through `_sync_viewer`'s off-thread
   parse. That last one is where a bug would actually live.
2. **One real download.** Every test stubs `snapshot_download`. Unverified:
   resume semantics, whether `allow_patterns` with an exact path behaves like
   `hf download <file>`, and whether the merged IP-Adapter pattern really fetches
   both halves. The staging-directory contract (a failure leaves nothing behind)
   was tested with a stub that raises, not with a real network drop. Also
   untested live: that the child dies with the app — verified structurally, by
   the package-wide `winjob.assign` scan, but nobody has killed the app mid-fetch.
3. **One rig of a real trellis mesh.** This is the big one. **The central
   hypothesis — that welding lets `ARMATURE_AUTO` succeed where it currently
   falls back to envelope weights — has never been tested against a real
   reconstruction.** What *is* measured is that the weld is **invisible**:
   per-loop UVs, face count and exported texture bytes come back identical across
   it. Invisible is not the same as effective. Until a real trellis mesh goes
   through it, the weld is a well-argued change with a parity proof attached, not
   a fix. (§4 is the rest of this thread.)

Also unmeasured, and cheap to fix when convenient: roughly half the `size_gib`
figures behind the downloader's disk-space refusal are estimates rather than
measured sizes (IP-Adapter, ControlNet, BiRefNet, DINOv2/ViTPose and the style
LoRAs). Understating only ever weakens the refusal, never causes a wrong one, so
this is untidiness rather than a bug — but they should not be quoted as facts.

---

## 2. The re-baseline campaign — run, and what it left open

*Both GPU sessions happened on 2026-08-09 and their results are written up in
the documents that pre-registered them:
[`2026-08-09-rebaseline.md`](measurements/2026-08-09-rebaseline.md) (the
blind-confirm and the 50-unit re-baseline, with two amendments) and
[`2026-08-09-framing-axis.md`](measurements/2026-08-09-framing-axis.md). The
narrative that used to stand here — the 2026-08-07 review, the `bg_removal`
mechanism argument, the two run plans — is deleted per this file's own rule; it
is in those documents and in `git log`. What follows is only what is still open.*

**What the campaign settled.** `bg_removal=birefnet` is the baseline: `auto` is
**0 accepts in 90 units** across two runs, and no birefnet reject carries the
`broken` tag (0 of 5 against 4 of 5, Fisher p=0.024). The re-baseline's go/no-go
**passed** — 19 accepts of 41 finished units, against a pre-registered bar of ≥12
of 50 — so a workable baseline exists and the floor effect that made Sweep B
measure nothing is gone. `hole_worst` is **no longer inverted** (AUC 0.756
against the pre-registered ≥0.65), and the inversion is now understood as an
artefact of the `auto` matte rather than a property of the metric. `framing`
did not win and nothing changes; `mesh_hole_max` is retired as a retry trigger.

**What it did not settle, and is the remaining work:**

1. **The depiction axes have still never been measured on a working baseline.**
   The re-baseline carried checkpoint, style LoRA and framing only. `palette`,
   `silhouette`, `condition` and `mood` were last run inside Sweep B's floor
   effect, which means they are untouched — this is §5's dependency and the
   whole reason §5 is still open.
2. **No checkpoint is chosen.** Every axis in the re-baseline was null, and the
   bar was *unsatisfiable* rather than merely unmet: `baseline s23` was refused
   at the composition gate, so only four matched pairs ever existed against a bar
   written as five. The marginals (`playground` 4/5, `sdxl_cfg` 3/3,
   `style_lora=render3d` 3/4) are confounded and underpowered, and
   `sweep_props.WINNER` records them as marginals rather than winners. Two arms
   are informative anyway: `style_lora=pixelxl` took 0 of 5 with every rejection
   tagged `broken`, and `base_model=turbo` at 1/5 is the weakest checkpoint that
   produced data.
3. **A future OFAT run must state its bar as a fraction of *usable* pairs.**
   Both mis-specifications in this campaign have one root cause — a sign test
   sized as though ties and losses were rare. Recorded here because the rule was
   deliberately not repaired after seeing the data it judged.
4. **Seven units produced no terminal measurement** and are re-queued into the
   same sweep by `../scripts/sweep_refill.py`, so their pairs can still form. The
   two genuine composition-gate refusals are not re-run: a refusal is a
   measurement.

**The `quality_badge` green branch is still absent, deliberately.** The
re-baseline's own rule defers it to its own measurement and does not take it:
n=41 on one subject and one prompt is not grounds for a UI that tells users a low
hole count means a good mesh. `widgets.AUDIT_UNINFORMATIVE` and the inspector's
"a solid, featureless mesh scores this too" stand unchanged. `../CLAUDE.md`'s
invariant has been rewritten to match: the claim is no longer "inverted" but
**corpus-dependent**, which is the stronger form and the one that survives the
next time the corpus moves.

---

## 3. Qualify the gltfpack tiers

*Binary present; harness present; `WINNER` filled. What is missing is a run of
`sweep_props.py` and the meshes it would leave on disk.*

`../vendor/gltfpack/gltfpack.exe` arrived on 2026-08-07, so `pipelines/optimize.py`,
the config field, the doctor check and the retarget panel's full tier list are
all live. What is *not* done is the qualification: a tier stays unqualified until
it has been run against a chest, a sword and a rock and shown to keep UVs, both
PBR maps and material assignment. `Config.mesh_profile` stays `raw` until then,
and `panes/settings_3d.PROFILES` still offers `raw` alone, with the combo
disabled and the reason stated.

**The harness exists — what is missing is the corpus to point it at.**
`../scripts/qualify_tiers.py` runs draft/standard/detailed through `optimize.run`
(the same binary and flags a job uses) and `warlock/tiercheck.py` reads both
GLBs' JSON chunks and names every loss: a primitive that lost `TEXCOORD_0`, a
material count that fell, a primitive that came back unassigned, either PBR map
gone, or an output requiring an extension the source did not. `--sheets` adds the
before/after picture through the one sheet pipeline. Every rule is "not worse
than the source" rather than "identical to an ideal", so an untextured rock is a
legitimate input; the accepted cost is that a run could pass by preserving
nothing, which is why a source already missing something is reported in the
`notes` beside the verdict.

Two things it deliberately does **not** do. It never drives
`service.jobs.optimize_job` — that rewrites `model.glb` in place and deletes the
derived artifacts describing the old mesh, so it would consume the accepted
meshes the qualification needs, and a harness that destroys its inputs can be run
once. And it does not touch `PROFILES`: on a pass, exposing the tiers is a
decision, and `Config.mesh_profile` stays `raw` regardless — the default flip is
a separate one again.

**The qualification corpus still does not exist on disk, and that is now the
whole of this section.** §2's re-baseline produced 19 accepts and took the stock
of accepted meshes with files from 1 to 20 — but **`config.data_dir` holds zero
job directories today**, against 66 `done` rows in `jobs.sqlite` and 26
model-stage accepts among 231 verdicts. The verdicts outlived the pixels exactly
as they are designed to; the meshes did not. `../bench/tiers/corpus` holds one
`source.glb` (`44593039ccee`) and nothing else. So `qualify_tiers.py` run today
still prints *no accepted meshes to qualify against* and exits 1.

Whether that loss was a prune, a `delete_sweep` or a manual clear is worth
establishing before the next corpus is generated — `service.jobs.retained_job_ids`
was added to stop precisely this, and it did not.

**The qualification corpus is not the old sweep either.** A tier test needs
meshes worth keeping: whether a tier *preserves* something cannot be judged on
output that is already broken.

**`../scripts/sweep_props.py` is what generates the corpus**, and its `WINNER` dict
is **filled in** as of the re-baseline's Amendment 2
(`base_model=playground` + `style_lora=render3d` + `bg_removal=birefnet`,
recorded there as the strongest marginals rather than as winners). So this is
now a run to take rather than a blocked one — four no-axis plans at five seeds,
twenty units:
chest, sword and rock (whose prompts contain those words, because
`qualify_tiers._warn_about_the_corpus` substring-matches `WANTED_SHAPES` against
the prompt), plus one deliberately asymmetric knight that exists only for §4.
Deliberately **not** `python -m warlock.bench run`: `core-v1` carries all three
subjects, but `recipe.Recipe` has no `bg_removal` field, so a bench run would
inherit the matte from whichever weights the host holds — in the one run whose
whole premise is "at the winning settings".

**The watertight figure has been re-measured and is `2 of 41`** — the first taken
since `meshreport` began welding by position, and the replacement for the void
"0 of 83". Triangles are 273,888–299,408 (median 288,440), unmoved by any axis
and consistent with the old run's 177k–299k. Nothing here is outstanding; it is
kept only so the void figure is not carried forward by someone who remembers it.

---

## 4. The rig questions

**Weld-before-heat is in and its parity is proven; its effectiveness is not.**
See §1.3 — the same item, listed there because it is verification debt and here
because it is the rigging thread. `_skin` runs weld → verify → unwelded heat →
envelope, with the decision carved into a bpy-free `_skin_steps` so it is
testable without Blender, and `automatic-welded` is in the `rig_meta` vocabulary
the inspector surfaces.

**Handedness is still unverified, and it is the one that would be invisible.**
The mapping takes COCO's anatomical left to the template's +X (`humanoid.json`:
"+X is the subject's left"), on the reasoning that a subject facing the camera
has their left at the larger pixel x. The image half is confirmed on a real
detection and `test_a_subjects_left_arm_lands_on_the_templates_positive_x_side`
pins the convention — but whether **trellis reconstructs with the same
handedness** is a fact about the exe, and the check used a symmetric box as its
stand-in because no reconstruction survives in `jobs.sqlite`. A mirrored skeleton
looks perfectly plausible in a still. §2's re-run produced meshes and they are no
longer on disk (§3), so this still wants a fresh one: rig an *asymmetric* subject
— `sweep_props.py`'s knight plan exists for exactly this — and look at which side
the `.L` bones came out on.
Nothing else depends on the answer, a flip is a one-line sign change, and it owes
a measurement document.

**Whether landmark placement improves the actual skin weights is unjudged**,
which is the only reason any of it matters. The detector tracks the inner edge of
bulky armour rather than the limb's centre line, visible in the overlay, and may
or may not cost anything once Blender's automatic weights run. The deformation
battery (`../src/warlock/templates/deform_qa/humanoid.json` — squat, arms overhead, elbow and
knee 90°, torso twist, rendered through the existing sheet pipeline as
`rig_qa.png`, with a thumbnail in the inspector's rig section) is the artifact for
judging this by eye. Scoring waits for the judge (§7).

Deliberately out of scope until the re-run: skeleton-conditioned *generation*
(ControlNet OpenPose, so the reference is drawn in a pose rather than measured
after the fact) and non-humanoid templates (a quadruped needs an AP-10K model and
its own mapping; `pose2d.POSE_FIT_TEMPLATES` is the extension point). The
deformation battery ships for `humanoid` only, for the same reason — a squat
means nothing to a fish.

---

## 5. `art_style=snes` fights an explicit colour brief

**No code change yet — this is a note for whoever reads the re-run's results.**

The brief asked for "black and silver and blue"; the composed prompt reads "…
grim dark mood, **vivid saturated colours**, bold simple shapes, …", contributed
by `ART_STYLES["snes"]`. A defensible reading of the 16-bit era, left in the
sweep's base deliberately rather than edited out, but it argues against the
stated colours. The 2D pane now *flags* the conflict advisorily; what it does not
do is settle whether the era fragment should be describing colour at all.

Sweep B's `palette` axis (`steel` baseline against `mono`, `muted`, `vibrant`)
was meant to measure the tension directly and returned 0 accepts and all ties —
the floor effect in §2, not a null result about colour. **The 2026-08-09
re-baseline did not carry a `palette` axis**, so this is still untouched and now
rides on §2.1, the depiction-axes run that has not been designed yet. If `mono`
or `muted` wins clearly, the era fragments are over-specifying colour and should
describe *shape and shading* language only, leaving colour to `palette` and the
user's own words.

Note a colour finding is one of the few things `meshaudit` could never answer
even had the meshes been sound: it is a texture judgement, so it needs human
verdicts or the image probe (§7) regardless.

---

# The quality judge ("CAMERA")

Named for the analogy that produced it: an industrial vision system is shown good
parts and bad parts, learns the boundary, then inspects on its own.

**The code exists and is unused.** The design was implemented as written rather
than re-designed — including the parts that read as fussy, because each of them is
the difference between a probe that means something and one that is merely
believed. The end-to-end path was verified against real DINOv2 weights (768-d CLS
embeddings, fit, save, load, score) rather than only against stubs.

**One probe now exists.** A labelling pass was taken on 2026-08-09: the corpus is
**231 human verdicts** — 61 `blank` (45 accept / 16 reject), 11 `reference`
(7 / 4) and 159 `model` — and `../bench/probe-blank.npz` is a fitted 768-d blank
probe. The `reference` probe is still unbuilt, four rejects short of
`MIN_PER_CLASS` (8) per class. No `ai:` verdict has been filed by anything.

## 7. Phase 1 — the 2D judge — one probe fitted, one still short

**Everything in this section is implemented, and the first labelling pass has
been taken.** What remains is: **~4 more `reference`-stage rejects** to reach
`MIN_PER_CLASS` and fit the second probe, and then §10's measurement document.

How to run one: open Review, and under the sweep list pick one of the two passes
under *Teach the judge*. The centre column becomes a grid; `A` is good, `R` is
bad, there is no second step, and the pass owns the keyboard while it is open so a
keypress about a picture can never be filed as a verdict about a mesh. Those two
keys are **unchanged by the 2026-08-09 grade scale**, which reaches the mesh
stage only — an image label is still one bit, and the verdict loop behind this
grid is the one that now takes `1`–`5`, `R`+digit and `0`. Retraining
happens as you go, off the frame thread. Do **both** passes over the same images
if you want both probes: they are different questions and neither answer implies
the other.

Two things to expect, both deliberate. Nothing appears for a while at the top of a
large grid — thumbnails upload one per frame, which is `StripRender`'s rule at a
hundred cells. And images marked as refused are in the list on purpose: they are
the most informative negatives available.

**Labels must be human.** It is tempting to treat the gate's own refusals as free
labels. Those labels *are* the rule's output, so a probe fitted to them can at
best reproduce `reference.py` exactly, blind spots included — and `baseline s23`
**passed** those rules and is still a poor blank, which is precisely the case a
learned judge exists to catch. What makes the 2D half attractive is not free data
but *cheap* data: ~2 s to judge an image against ~15 s for a mesh.

**And a probe must be trained on pixels, never on the audit scalars.** The
original argument was the inversion, which §2 has since overturned
(AUC 0.115 → 0.756 once the matte changed). The rule survives its own evidence
and is *strengthened* by it: a scalar whose AUC moved 0.64 because an unrelated
knob moved is measuring the corpus's dominant artefact rather than quality, and a
probe fitted to it would have learned the slab in 2026-08-07 and something else
now. It is still not a sanity check *on* a probe.

**Filing an `ai:` verdict is deliberately not part of it, and it is the one
remaining seam.** `review_mode.SOURCE_AI = "ai:dino-probe"` is a constant nothing
writes — `../tests/test_review_mode.py` asserts no row carries it, and the live
database holds 231 verdict rows (159 `model`, 61 `blank`, 11 `reference`, after
the 2026-08-09 pass), every one `source='human'` — which is the load-bearing half. The
`(job_id, source, stage)` seam is built and tested, so the day §10's document
exists this is one call. What it is waiting on is the threshold: a
probability-to-accept cut is a constant the stored corpus is then keyed on, and
that owes a measurement first —
[`2026-08-09-judge-threshold.md`](measurements/2026-08-09-judge-threshold.md),
pre-registered with an empty Results section and blocked on a labelling session.

**It is blocked on pixels as much as on a human, and that got worse rather than
better.** `config.data_dir` now holds **zero job directories** against 231
verdict rows, so *every* labelled image is a row without pixels behind it. The
fitted blank probe survives — it stores 768-d weights, not images — but nothing
can be re-labelled, re-split or held out from what is on disk today, which is
what §10 needs. `service.jobs.retained_job_ids` was added to prevent exactly
this and evidently did not; §3 carries the same finding and the same question
about what removed them.

**The held-out accuracy figure is not built, and cannot be.** It is a measurement
over labels that do not exist yet, and it is what §10 specifies — so the remaining
work here is: run a labelling session, then write the measurement document.

## 8. Phase 2 — the mesh judge

Reuses `../src/warlock/bench/views.py`'s 8-view render, adds the pooling adapter and the third
probe.

**Not blocked on labelling — blocked on a corpus that contains acceptable
meshes.** The mesh corpus is now **27 accepts against 132 rejects** (159 rows,
up from 3-against-81), so the positive count has cleared "unusable". What has
*not* changed is the spread: **every one of those 159 grades is exactly ±3**, the
two backfill values, because no reviewer has yet used the −5..+5 scale. A
regression target with no variance is not a regression target.

**The corpus is graded as of 2026-08-09, and that changes what this probe is
aimed at.** Mesh verdicts are integers −5..+5 rather than accept/reject
(`measurements/2026-08-09-grade-scale.md`), so the target here becomes **grade
regression** rather than binary classification — which is a strictly better fit
for a corpus whose problem was never the number of labels but their resolution.
It does not unblock the phase, and the gate is now stated in the scale's own
terms: **grades spanning more than the two backfilled values**. The positive
count is no longer what blocks this — resolution is. What the change does is
make the next pass worth taking, because the same afternoon of judging yields a
graded target and a tag distribution instead of one more column of rejects.

The existing labels are not wasted — they are a clean negative set, and the
matched pairs (identical `input.png`, opposite verdict) are the most useful rows
in the corpus precisely because everything except the matte is held constant.
But see §7: none of them has pixels on disk any more, so a graded pass has to be
taken over meshes that do not yet exist, which puts this behind §3's prop corpus
run rather than beside it.

**The mesh probe must be max- or mean-over-8-views, never single-view.**
`../src/warlock/bench/views.py` records a calibration over 37 finished jobs spanning every
category: the per-job argmax scattered by 330° (`silhouette_iou`) and 300°
(`dino_cosine`) against a `STABLE_YAW_SPREAD` of 30, the two metrics disagreed
with each other, and it scattered *within* categories as well as across them. The
recorded conclusion is that there is no fixed matched view at all. A single-view
mesh classifier would be learning camera pose, not quality. See
`measurements/2026-08-04-view-calibration.md`.

## 9. Phase 3 — earned authority

Only after §10's numbers exist. Candidates, in increasing order of risk:

1. A bounded reference-seed re-roll on a predicted-bad blank. The mechanism is
   already there and already on — `Config.reference_retries` is 2 and the loop
   holds `mesh_seed` fixed — so this is only a matter of letting the probe, not
   `reference.py`'s three thresholds, decide what counts as bad.
2. Refusal at the gate, which should not be attempted without a measurement
   document.

*(Sorting Review by score shipped and is deleted from this list per the rule at
the top of the file.)*

### Risks to carry through all three phases

- **The filter bubble.** If a judge auto-rejects, its mistakes become invisible —
  you stop seeing what it discards and never learn it was wrong. Factories manage
  this by auditing *passed* parts on a schedule. Mitigation here is structural:
  advisory-only through Phase 2, Review **sorts** rather than filters, and a
  sampled share of the judge's own rejects keeps surfacing for human review.
- **Scope.** A probe trained on this sweep learns "good SNES rogue", not "good
  asset". Pointed at a wooden crate it is confidently wrong. The corpus is scoped
  by `prompt_hash` now and the hints say which subject they came from; the
  *probe* is not, and giving it the same treatment is a decision this phase still
  owes. A judge with no notion of subject is worse than no judge, because it is
  trusted.
- **Confounded marginals, inherited.** A verdict credits every `param: value` in
  its vector — the accepted price of letting daily use feed the hints. A judge
  trained on those labels inherits the confound. The unconfounded answers stay
  where they are: `vectors` ranked by Wilson lower bound, and `comparisons`
  recovered from sweep structure.
- **Silent staleness.** A probe is a binary artifact built from a corpus that
  keeps growing — the `../vendor/warlockc` hazard exactly: an absent probe is
  obvious, a stale one quietly scoring last week's opinion is not. The `.npz`
  carries the corpus size, the label count and a schema version, and a retrain of
  the `blank` probe clears every score.

## 10. What "it works" has to mean

The threshold is a constant the stored corpus is keyed on, so by this repo's own
rule it gets a **measurement document under `measurements`** before it is
baked in — the pattern `trellis_band`, `mesh_hole_max` and `SEAM_MAX` all set.

Report, on a held-out split:

- **False-reject rate** — good assets the judge would have discarded. The number
  that decides whether it may ever gate.
- **False-accept rate** — bad assets it waves through. Cheaper: they reach Review
  anyway.
- **Agreement with `reference.py`'s rules**, on the blank probe specifically. If
  agreement is ~100%, the probe has learned to imitate the rules and has added
  nothing. Genuine value shows up as *disagreement that a human sides with the
  probe on* — `baseline s23` is the canonical test case. Agreement is cheap to
  compute per rule rather than in aggregate, since a refusal is an observation
  carrying `refused_<code>`.
- **Per-subject breakdown by `prompt_hash`**, or the scope risk is unmeasured.
- **Beat `hole_worst`, whose AUC is now 0.756 and was 0.115 six weeks of corpus
  earlier.** The floor is no longer a coin flip, and it is no longer stable
  either — it moved 0.64 because the matte changed (§2). Quote the floor with the
  corpus it was measured on, and report the probe against the human labels
  directly rather than against the scalar.

### Open questions

- Does the mesh probe pool views by max or by mean? Max finds the worst angle,
  matching how `meshaudit` already reports `worst`; mean is more stable at small
  sample sizes. Decide with data, not taste.
- ~~Binary accept/reject first, or straight to the five `REASONS` as
  multi-class?~~ **Answered, 2026-08-09, and by neither option.** The mesh
  verdict became an ordinal **grade, −5..+5**, with the five reasons re-cast as
  optional *tags* legal at any grade and a good vocabulary added beside them
  (`measurements/2026-08-09-grade-scale.md`). The question assumed the choice was
  between one bit and five classes, and both are the wrong shape: a bit cannot
  say how close a mesh came, and five mutually exclusive classes are not what a
  reviewer is looking at — a mesh routinely has holes *and* a bad texture, which
  is why the tags are a set rather than a label. Multi-class over the reasons is
  therefore not deferred, it is retired. What the grade buys the probes is a
  **regression target with spread**, which §8 is blocked on; the tag tallies are
  a second, cheaper signal that needs no probe at all.

  The image stages deliberately did **not** follow. `reference` and `blank` stay
  binary and keep `grade` NULL permanently: they feed binary logistic probes, so
  a grade would be thresholded straight back to a bit, and the two-key loop is
  what makes a hundred-image pass viable at all. That half of the
  pre-registration in `measurements/2026-08-09-judge-threshold.md` stands
  unchanged.
- Should the 2D probes be per-`prompt_hash` from the outset, or global with scope
  added later once the scope risk is measured rather than assumed?
- Does an `ai:` verdict feed `findings.json` at all, or only sort Review? Feeding
  it makes the corpus self-referential — the judge's own opinions become evidence
  for the hints that shape the next generation.

---

# Code, not runs

## 11. Retopo + bake prototype

*Needs §4's rigging thread; characters only. Not started —
`blender_worker.OPS` is `{"rig", "pose", "sheet", "fbx", "views", "project"}`,
and `deform.glb` appears nowhere in the tree.*

A new `blender_worker` op producing a *deformation* mesh from the immutable
source: remesh (voxel or QuadriFlow, explicitly labelled a preview/proxy), UV
unwrap (smart-project first), bake base colour / metallic-roughness / tangent
normals from the source, then rig it through the weld-before-heat path.

Output is a new labelled artifact (e.g. `deform.glb`), derived on demand under
the `_convert_locks` idiom, **never replacing `model.glb`** — the `source.glb` /
`model.glb` invariant applies unchanged. UI labels matter here: *Reconstruction*
/ *Static game-ready* / *Preview riggable* / *Deformation-ready*, so a decimated
reconstruction is never presented as real retopology. Prototype on 2–3 accepted
character meshes, with a QA note on bake fidelity before any UI default moves.

**Its bake infrastructure overlaps §14's, and the overlap is the point** —
`pipelines/retexture.py` and `blender_worker.op_views`/`op_project` already
render views, project them onto a UV atlas and swap the atlas in through `glbio`,
and `measurements/2026-08-08-retexture-bake.md` already measures how well
that works. Build the second bake on the first rather than beside it.

## 12. External backend A/B

*Last, and it gets its own spec first. Not started — SkinTokens, TokenRig and
Hunyuan3D appear only in `REPORT.md`'s research prose.*

- **SkinTokens/TokenRig** as an isolated out-of-process worker — the
  `trellis-server` / `blender_worker` pattern, kill-on-close job, weights by
  one-time manual download or through the fetcher, doctor reporting absence
  non-fatally. Run in existing-skeleton mode against Warlock's template so bone
  names and animation compatibility hold. A/B against the welded heat weights on
  a fixed rig corpus using the deformation battery, verdicts through Review.
- **Hunyuan3D 2.1** as an optional isolated reconstruction backend, same
  isolation rules, benchmarked on the curated references from §2 — human
  acceptance, silhouette, runtime, VRAM. It must fit the coexist/exclusive
  `vram.plan` machinery: a new backend declares its GiB cost in the cost table.
- **Explicitly out:** MeshAnything V2 as a retopo path (sub-1600-face target).

## 14. Texture coverage — what replaced the "dedicated texture model" tier

**The measurement is the reason, not a change of mind.**
`measurements/2026-08-08-retexture-bake.md`: the bake is **faithful** where
it has something to bake — every plank seam and bolt mark survives in the one
valid positive control — and covers only **36–37% of the atlas** over four real
reconstructions, because a perforated mesh's interior walls project onto holes in
the render and are masked out. Neither knob helps: 1024 and 2048 atlases are
indistinguishable, and front-plus-back reach 36.1% while the other four views add
1.0 pp between them. A better texture *model* would improve the third already
being painted and do nothing about the two thirds that are not. So, in order:

1. **Pack Clay's box unwrap.** The smallest of the three and the difference
   between a feature that silently does nothing on authored geometry and one that
   works. `clay/uv.py`'s `_projected` normalises by the mesh's single largest
   extent and writes every face into **one shared unit square** — faces are
   grouped by axis *pair*, so +X and −X land in the same region and a box's six
   faces stack as three overlapping groups. The module is explicit that
   overlapping islands are what a cube projection *is*, and that the primitives
   able to pack cheaply should do so in their own generator — which
   `cylinder`/`cone`/`torus`/`uv_sphere` do and `primitives.box()` does not, since
   it calls `box_unwrap` directly. **Consequence: a Clay-authored asset has UVs
   and still cannot carry a texture**, so the stated Clay25 prerequisite is only
   half met. The strength-0.85 control shows the bake is ready for it.
2. **Attack coverage.** More axis views are measured not to help. What would:
   weighting by *visibility* rather than by facing — a depth test from the
   camera, which also removes the overhang smear that `pipelines/retexture.py`
   reports as `occlusion_tested: false` and states as a limitation rather than an
   oversight — and views chosen from the mesh rather than from a fixed basis.
3. **Only then reconsider a texture model**, as a registry spec with
   family/residency declared, a `models.Fetch` deduped on
   `(repo_id, destination)`, download only through `fetch_worker`, and
   `local_files_only=True` at load — A/B'd through Review against the bake on a
   corpus whose coverage is high enough for the difference to be attributable to
   the model. It gets its own spec first.

## 15. Accepted narrowings and one open design call

Small, and written down so they are not rediscovered as defects:

- **`Ink9` is quarter turns and a mirror, not free-angle rotation.**
  `inker_state.ROTATIONS` is `(0, 90, 180, 270)` and `flipped` is a separate
  boolean. That is the decision the section it replaced argued for from the
  other side: a free angle makes every overlay in the pane a rotated quantity —
  the grid stops being two families of axis-aligned lines, the marquee preview
  stops being a rect, the transform box's handles stop being squares — and a
  half-done version silently misplaces all of them. A quarter turn maps an
  axis-aligned image rectangle onto an axis-aligned *screen* rectangle, so each
  of those stays exactly what it was and is checked by
  `test_a_quarter_turn_keeps_an_axis_aligned_rectangle_axis_aligned`. What it
  costs is drawing on a page turned to an arbitrary angle; what it buys is the
  two things canvas rotation is actually reached for. The orientation is
  orthonormal and handed to `ants.dash_segments` **separately from the zoom**,
  which is why the dash arithmetic needed no change at all: every distance in it
  is an arc length in canvas space, and turning does not stretch.
- **`Clay21` has no vertex extrude, and cannot have one.** A `Mesh` stores no
  wire edges — `edges` is derived from face corners — so an extruded lone vertex
  is a position no face uses, which `compact_vertices` drops on the way out and
  every exporter would drop again. `ops_topo.extrude_verts` therefore extrudes
  the *border* edges the selection implies, and refuses by name when it implies
  none. `extrude_edges` also takes no `offset`, where `extrude_faces` does: the
  same quantity is degenerate on the commonest case, since a closed rim's owning
  faces point radially outward all the way round and their mean is zero.
- **`K93` is a floor, not an audit.** `../tests/test_forms_and_layout.py` asserts
  each of ten dense panes carries at least one `help_marker`/`set_tooltip`. The
  original ask was per-control tooltip coverage; the floor is what shipped.
- **`M106` is configurable, not resizable.** `layout.SIDEBAR_WIDTHS` offers three
  named presets from a combo. `layout.py`'s header explicitly refuses a draggable
  splitter, so this is a decision rather than a gap.
- **`viewer/env.BACKGROUND_HEX` is the literal dark `0x0F1014`, so both 3D
  viewports stay black under the light palette.** Its comment says the colour
  "matches `tokens.BG` so the viewport and the panels around it read as one
  surface" — which the light palette breaks. Making it follow `theme.BG` means
  threading a colour through `viewer/render.draw` (which already takes one) *and*
  adding it to `Viewer.render`'s render-skip key, or a theme switch would not
  redraw. A design call, deliberately left open.

## 16. Deferred by decision — not open work

| Item | Status |
|---|---|
| `measurements/2026-08-06-pixel-art-xl.md` — Results still "not yet taken" | Unblocked: all three recipes and all weights verified present. A three-arm run settles which arm `pixel_sprite` names and where `GRID_RESIDUAL_MAX` belongs (0.05; the doc says "that number is a guess"). The re-baseline touches it only obliquely — `style_lora=pixelxl` took 0/5 with every rejection `broken`, and the `base_model=pixel` arm produced no data at all — and neither says anything about the 2D grid question this document asks. Independent of everything; good use of idle GPU time. |
| `seam.SEAM_MAX` | **Closed at 3.5** on 72 units — `measurements/2026-08-08-seam-threshold.md`, corpus from `../scripts/calibrate_seam.py` and `calibrate_seam_hard.py`. One checkpoint only (turbo at 4 steps); a CFG base draws harder edges and should re-run the scripts. The re-run is pre-registered — `measurements/2026-08-09-seam-threshold-cfg.md` — and its `--out` **must** be `docs/measurements/data/seam-cfg`: both scripts write identical filenames across checkpoints, so `.../seam` overwrites all 125 turbo files and makes the closed measurement unreproducible. |
| Fused brush dab kernel (`warlockc_dab_u8`) | Deferred on purpose; the gate is "the brush shows up in a profile first". |
| Merge-down and flatten on an animated Inker document | Refused rather than approximated (`Document.can_restructure`). Both are defined over one layer stack and an animated document has one per frame, so the honest versions are "merge these two tracks across every frame" — which has to decide what merging a linked cel with an unlinked one means — and "flatten this frame", which discards every other frame's cels. Real features; neither is v1. |
| Cel thumbnails in the Inker timeline | Dots and chain glyphs instead. A per-cel texture on a grid that can be fifty columns wide is `viewer/sheet.StripRender`'s problem at a larger scale — worth doing on a per-frame upload budget, not worth doing by accident. The `layer_thumb` stamp pattern extends to it when it is. |
| `studio/clay/ops_topo.py` hole-fill UV | Documented, deliberate approximation. |
| View-matched reference ranking | Not built on purpose; the Scattered verdict *is* the deliverable (`measurements/2026-08-04-view-calibration.md`). |

## Appendix — what a diff could not carry

Kept because it was expensive to work out and is invisible in the resulting
commits. `../CLAUDE.md` holds the load-bearing invariants; this is the residue that
does not belong there.

**The file-contention map, and why the build waves were chains.** Logical
dependency was never the scheduling constraint — file contention was.
`studio/widgets.py` was touched by the quality badge, `icon_button` and the
library kind badge, so that chain ran serially in one worktree in that order;
`panes/library.py` by four packages and `panes/inspector.py` by four more, which
is why the second wave was a chain rather than a fan-out. `../pyproject.toml`
belonged to exactly one package. Any future parallel run should be scheduled the
same way.

**A worktree needs the three `WARLOCK_*` environment variables**, because
`../vendor` and `../models` are gitignored. Without them the suite reports
427 passed / 6 skipped instead of the full count, and the six look like
regressions.

**Three departures the judge's implementation made from its own written design**,
each because following it literally left a hole. The findings stage filter is at
the **top of `aggregate`**, not inside `_marginals` — the plan was right about
`_marginals` covering both call sites and incomplete about the other two
consumers, since `vectors` would advertise an image label as a ranked mesh
configuration and `_comparisons` would pair a blank label against a mesh verdict
*on the same unit*. `latest_verdicts` groups by `(job_id, source, stage)`, which
was not stated and is not optional: without stage in the grouping, labelling one
image under both intents makes the second answer supersede the first. And the
mesh probe's stage is spelled **`model`, not `mesh`** — two names for one thing
needs a translation table, and the drift would surface as a probe trained on the
wrong population, so `judge.STAGES == verdicts.STAGES` is asserted by a test.

**The re-texture default atlas size is the mesh's own, and that is a measured
correction rather than a preference.** A flat 1024 resampled every trellis mesh's
2048 atlas down by half — including the ~63% that no view covers and that
therefore *keeps its old colour*, which is exactly the half a re-texture must
leave alone.

**The smoke suite runs at UI scale 1.0 and nothing else does.** `theme.apply`
sets `item_spacing` through `sp()`, so a grid subtracting a literal `8` for its
gaps is exact at 1.0 and short by 4.8 px per gap at 1.5 — which is the only scale
anybody runs. That single cause produced three of the four defects the
screenshot pass found (the Inker toolbox losing its fifth column, Clay its fourth
button, the outliner reserving 32 px for a 58 px button). `../scripts/screenshot_modes.py`
drives the **real** `App` and reads frames off the framebuffer, which is why it
could see them and GL smoke coverage could not; re-run it after any layout work.
