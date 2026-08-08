# TODO

The live roadmap. Consolidated 2026-08-07 from `SUGGESTIONS.md`, `CAMERA.md` and
`ANALYSIS.md`, which this file replaces; revised 2026-08-08 after the Night 1 and
Night 2 build runs. Everything here is grounded in something observed — the
100-unit SNES-rogue sweep (`scripts/sweep_rogue.py`, sweeps `b5c47248e13d`
"rogue - render" and `8340cd0b2f5a` "rogue - depiction"), the QA audit of
2026-08-07, or the two build runs since — not in a reading of the code alone.

**Two numbering schemes exist and they are not the same.** This file's `§`
numbers are its own. The overnight build plan — transcribed to
`docs/BUILD_PLAN.md`, which is a **record** of why each package was shaped as it
was, not a queue — numbers its work `Phase 0`–`Phase 9`. Where an item here
corresponds to one of those phases, it says so. Nothing in this file is named
after the phase numbers alone, because two of them collided once already, and
where the two documents disagree about what is outstanding, **this one is
right**.

**Implementation state, 2026-08-08.** Night 1 shipped fourteen packages (0a–0f,
9a–9d, 9h, 9i) and Night 2 shipped four (the matte preview, mesh candidates,
weld-before-heat rigging, and the model downloader). What those packages
concluded is recorded in `CLAUDE.md` and in the comments at each site, not here.

**Night 3 shipped the four GPU-free packages that were left**, which is why
several sections below now describe a *run to perform* rather than code to write:

- **§2's preparation.** `scripts/sweep_confirm.py` (the blind 10-unit matte
  confirm) and `scripts/sweep_rebaseline.py` (the render sweep re-run over a
  birefnet baseline, carrying the framing axis), both on a shared
  `scripts/_campaign.py` submitter, both validated headlessly by
  `tests/test_campaign_specs.py`. Review gained a **Blind** toggle, which renames
  every unit *and reorders them*, because `expand` enqueues the baseline first
  and position names the arm as plainly as a label does.
- **§3's harness.** `warlock/tiercheck.py` (pure: reads both GLBs' JSON chunks
  and names every loss) plus `scripts/qualify_tiers.py`, which defaults its
  corpus to the accepted meshes and refuses to run when there are none. Still
  unqualified, because that corpus is still empty — the run is §2's output.
- **§6, all three.** The weighting verdict now reads `rig.json` beside the
  selected mesh (mtime-cached under the racily-clean rule), `rig_qa.png` has a
  thumbnail in the rig section, and the duplicate "Open in Inker" is resolved:
  the viewport toolbar owns it where it exists, the inspector everywhere else,
  written as complements so they can be neither both nor neither.
- **§7 in full, code-complete.** Migration **7** (`verdicts.stage`),
  `db.unlabelled_references`, the findings stage filter, `warlock/judge.py`,
  `service/judge.py`, and the labelling grid in Review with both image probes.

What remains below is what is genuinely still open, and **all of it now needs a
GPU, a network or a human rather than code**: verification debt from work that
shipped without ever touching a GPU or the network (§1), the two GPU sessions
nobody has run (§2), three items waiting on their output (§3, §4, §5), and the
labelling pass the judge is now waiting for (§7–§10).

---

## 1. Verification debt — three runs, and none of them are optional

Night 2's four packages were built and tested entirely headless. Every claim
below is proven at the level of bookkeeping, arithmetic and imgui frames
building, and **unproven at the level of the thing actually working**. This is
the newest item in the file and the one with the shortest path to an unpleasant
surprise.

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
3. **One rig of a real trellis mesh.** This is the big one. **Phase 4's central
   hypothesis — that welding lets `ARMATURE_AUTO` succeed where it currently
   falls back to envelope weights — has never been tested against a real
   reconstruction.** What *is* now measured (and was not, until the Night 2
   integration run made the test actually execute) is that the weld is
   **invisible**: per-loop UVs, face count and exported texture bytes come back
   identical across it. Invisible is not the same as effective. Until a real
   trellis mesh goes through it, the weld is a well-argued change with a parity
   proof attached, not a fix.

Also unmeasured, and cheap to fix when convenient: roughly half the `size_gib`
figures behind the downloader's disk-space refusal are estimates rather than
measured sizes (IP-Adapter, ControlNet, BiRefNet, DINOv2/ViTPose and the style
LoRAs). Understating only ever weakens the refusal, never causes a wrong one, so
this is untidiness rather than a bug — but they should not be quoted as facts.

---

## 2. Two GPU sessions, and the measurement items still wait on them

*(The build plan's Phase 3. Code work is sweep-spec preparation; the GPU time is
the user's. §3, §4 and §5 are all waiting on the output.)*

The code changes these were meant to enable are done. Neither run has happened.

### What the review found

**3 accepts in 83.** All three are `bg_removal=birefnet`; `auto` went 0 for 80.
It is the only signal in the corpus — `bg_removal` is the **sole** comparison
with a non-zero win count (`a_wins=0, b_wins=3, ties=1`). Every other axis, in
both sweeps, is all ties: base_model, style_lora, silhouette, palette, condition,
mood.

At n=4 that would normally be a curiosity. Three things make it more:

- **The matched pairs are byte-identical upstream.** `input.png` hashes the same
  for `baseline s23` and `bg_removal=birefnet s23`, and for s42, s77 and s101.
  Same reference image, same seed, one knob — `bg_removal` is passed to
  trellis-server at reconstruction time, so nothing about the picture differs.
  This is a controlled A/B, not a confounded marginal.
- **The failure mode changes, not just the rate.** 58 of 80 `auto` rejects are
  tagged `broken`; 0 of 4 birefnet are (its one reject is `bad-shape`). A rate
  shift at n=4 is weak evidence; a rate shift plus the disappearance of the
  dominant failure tag is a mechanism signature.
- **It is not review drift.** The accepts land at review positions **46, 48 and
  83** of 84, with 34 consecutive rejects between the second and the third.

**The mechanism.** Without `birefnet.gguf`, matting falls back to a threshold
cutout, and `auto` lets the server decide. A threshold cutout on a deliberately
dark brief ("black and silver and blue") leaves background attached, and TRELLIS
reconstructs it into a solid slab. `models/trellis2-gguf/birefnet.gguf` is
present, so the learned matte was available the whole time and simply was not
being asked for.

**One caveat on the evidence trail, found during Night 1.** The suite was
destroying `bench/findings.json` on every run — silently, because the file is
derived and nothing recomputed it. The bench directory is pinned by a fixture
now and the file has been regenerated (299,768 bytes, and it survives a full
suite). The figures above were re-derived from the verdicts in the DB, which is
where they always lived; the denormalized `vector` column is exactly what made
that recovery possible after `prune_jobs` took the job rows.

### The two runs

1. **Blind-confirm, before any of this is leaned on.** The clean 2×2 on its own
   is Fisher p=0.14; the p≈4×10⁻⁵ figure (≈8×10⁻⁴ after Bonferroni over ~20
   blocks) comes from using all 80 `auto` units as controls, which is legitimate
   but leans on their comparability. The review was also unblinded and
   single-reviewer, and the app shows the params. 8–12 units, birefnet against
   auto, labels hidden, is cheap next to a 3.7 h sweep.
2. **Re-run the render sweep with birefnet as the baseline**, check the accept
   rate is workable, and only then re-run the depiction axes on top of it. The
   re-run now also carries the **framing axis** for character subjects, which
   Night 1's 0e made expressible (front-ortho A/T-pose against global 3/4).

Note the re-run measures more than one change: `PROMPT_TEMPLATE` moved
(`PROMPT_VERSION` 3 → 4), so a unit from the new run is not comparable with one
from the old on the prompt axis either. All the changes are deliberate and all
land before the re-run, which is the right order — a sweep around a broken base
measures the brokenness — but the re-run is the first corpus in which any of them
is measured.

**After verdicts:** write the framing measurement doc. If `front_ortho` wins for
characters, flip the per-category default *then* — that is the `PROMPT_VERSION`
4 → 5 moment, and the findings-corpus split is the documented cost.

### `hole_worst` is not weakly informative. It is backwards.

```
AUC(hole_worst predicts reject) = 0.115      (0.5 = coin flip)
rejects with hole_worst EXACTLY 0.0:  48/81 = 59%
accepts with hole_worst exactly 0.0:   0/3
median hole_worst — rejects 0.0000, accepts 0.0304
```

The accepted meshes have *more* measured holes than the median discarded one,
because a slab has no holes: `meshaudit` scores the dominant failure mode as
perfect. Anywhere that reads a low hole fraction as evidence of quality is wrong.
This supersedes `docs/measurements/2026-08-04-hole-rate-baseline.md`, which the
re-run's measurement doc should say explicitly.

Note this is `meshaudit` (the silhouette question), **not** `meshreport` (the
importer question) — see §3 for what happened to the other one.

### Sweep B measured nothing, and the design is the lesson

45 units, four axes, zero accepts, every comparison a tie. `bg_removal` was
pinned at `auto` throughout, so the variable that dominates the verdict was held
fixed at its bad value while the axes under study varied. That is a floor effect:
OFAT around a baseline that fails ~96% of the time has no headroom to detect an
improvement in anything. Roughly half the GPU time bought one finding about a
knob that was an afterthought in the design.

**The rule that follows: establish a baseline that produces acceptable output at
a workable rate before fanning out.**

### What the review says about the checkpoints: nothing yet

Refusal rate and mesh quality still rank them oppositely — `playground` and
`redmond3d` refused 0/5 each; `sdxl_cfg` refused 3/5 but its survivors were
flawless, while `render3d` and `pixelxl` passed more often and produced the two
worst meshes (0.48 and 0.61 worst-view hole fraction). But every checkpoint ran
under `auto`, so all of them were being judged through the same defect. Neither
number picks a checkpoint, and the re-run is what settles it.

**The re-run will now say this itself**, which it could not before: a refusal is
an observation, so `findings.json` carries `refused_multi_object` as a
per-checkpoint rate and the hint under the base-model select reads it.

---

## 3. Qualify the gltfpack tiers

*(The build plan's Phase 5. Binary present; corpus from §2.)*

`vendor/gltfpack/gltfpack.exe` arrived on 2026-08-07, so `pipelines/optimize.py`,
the config field, the doctor check and the retarget panel's full tier list are
all live. What is *not* done is the qualification: a tier stays unqualified until
it has been run against a chest, a sword and a rock and shown to keep UVs, both
PBR maps and material assignment. `Config.mesh_profile` stays `raw` until then,
and the generate forms still offer only `raw`.

**The harness exists — what is missing is the corpus to point it at.**
`scripts/qualify_tiers.py` runs draft/standard/detailed through `optimize.run`
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
once. And it does not touch `panes/settings_3d.PROFILES`: on a pass, exposing the
tiers is a decision, and `Config.mesh_profile` stays `raw` regardless — the
default flip is a separate one again.

Run today it prints *no accepted meshes to qualify against* and exits 1, which is
the correct answer: the accepts it wants are §2's output.

**The qualification corpus is not this sweep.** 80 of the 83 meshes were
rejected, and a tier test needs meshes worth keeping: whether a tier *preserves*
something cannot be judged on output that is already broken. Qualify against §2's
re-run. The three accepted birefnet meshes are a start and are not enough.

**Correction — the old triangle/watertight figures here were measuring the wrong
thing.** This section used to record "177k–299k triangles, **0 of 83
watertight**, unmoved by any sweep axis". The triangle counts stand. The
watertight figure does not: Night 1's 0c package found `meshreport` was counting
**xatlas UV-seam splits** as holes, so it was answering a question about the
atlas rather than about the mesh. `meshreport` now welds by position
(`WELD_TOLERANCE`, quantised onto a lattice) before judging. **Every watertight
number recorded before 2026-08-08 is void**, including the "0 of 83" that was
offered as a Phase-3 baseline. Re-measure on the re-run; do not carry the old
figure forward.

---

## 4. The rig questions

*(Rig handedness and weight quality. The build plan's Phase 4 shipped the weld
chain; these are what it did not settle.)*

**Weld-before-heat is in and its parity is proven; its effectiveness is not.**
See §1.3 — this is the same item, listed there because it is verification debt
and here because it is the rigging thread. `_skin` now runs weld → verify →
unwelded heat → envelope, with the decision carved into a bpy-free
`_skin_steps` so it is testable without Blender, and `automatic-welded` joined
the `rig_meta` vocabulary 0d surfaces.

**Handedness is still unverified, and it is the one that would be invisible.**
The mapping takes COCO's anatomical left to the template's +X (`humanoid.json`:
"+X is the subject's left"), on the reasoning that a subject facing the camera
has their left at the larger pixel x. The image half is confirmed on a real
detection and `test_a_subjects_left_arm_lands_on_the_templates_positive_x_side`
pins the convention — but whether **trellis reconstructs with the same
handedness** is a fact about the exe, and the check used a symmetric box as its
stand-in because no reconstruction survives in `jobs.sqlite`. A mirrored skeleton
looks perfectly plausible in a still. So: on the first mesh out of §2's re-run,
rig an *asymmetric* subject and look at which side the `.L` bones came out on.
Nothing else depends on the answer, and a flip is a one-line sign change.

**Whether landmark placement improves the actual skin weights is unjudged**,
which is the only reason any of it matters. The detector tracks the inner edge of
bulky armour rather than the limb's centre line, visible in the overlay, and may
or may not cost anything once Blender's automatic weights run. The deformation
battery (`templates/deform_qa/humanoid.json` — squat, arms overhead, elbow and
knee 90°, torso twist, rendered through the existing sheet pipeline as
`rig_qa.png`) is the artifact for judging this by eye. Scoring waits for the
judge (§7).

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
stated colours.

Sweep B's `palette` axis (`steel` baseline against `mono`, `muted`, `vibrant`)
was meant to measure the tension directly and returned 0 accepts and all ties —
the floor effect in §2, not a null result about colour. The question is untouched
and rides on the re-run. If `mono` or `muted` wins clearly, the era fragments are
over-specifying colour and should describe *shape and shading* language only,
leaving colour to `palette` and the user's own words.

Note a colour finding is one of the few things `meshaudit` could never answer
even had the meshes been sound: it is a texture judgement, so it needs human
verdicts or the image probe (§7) regardless.

---

## 6. Three small UI decisions that are really one — decided

All three landed together, in the same corner of the 3D inspector, and the
decisions taken were:

- **The rig-weighting verdict reads `rig.json` beside the selected mesh.**
  `inspector.rig_meta` caches it per job on the directory's mtime, under the
  racily-clean rule `files.attach_files` documents and with its constant imported
  rather than restated — a re-rig lands in a directory whose mtime may not move,
  so a stamp is only remembered once it is safely in the past. The row still wins
  where it has an answer, so a *rig* job selected in the library answers for
  itself. The rejected alternative was making `queue._rig` a second writer onto
  the model job's row: that row has one owner and `set_params` is
  last-write-wins.
- **The deformation sheet has a thumbnail** in the rig section beside the
  weighting line, gated on `rig_qa.json` for the reason `rig.glb` is gated on
  `rig.json`. Its caption says nothing scores it, because nothing does and the
  wording is the feature.
- **The duplicate "Open in Inker" is resolved by ownership**, not by removing
  either: `overlay.offers_inker` is the toolbar's gate and
  `inspector.offers_inker` is written as its **complement**, so exactly one is on
  screen for every mode. The toolbar wins where it exists (adjacency to the pixels
  is its whole advantage) and the inspector covers the rest — a reference selected
  in 3D, whose toolbar hides the button because the thing on screen is a mesh.
  This is *not* the F10 resolution: two readouts saying different amounts about
  different things is defensible, two identical buttons invoking one function on
  one job is not.

---

# The quality judge ("CAMERA")

Named for the analogy that produced it: an industrial vision system is shown good
parts and bad parts, learns the boundary, then inspects on its own.

**The code described below now exists and is unused.** Migration 7
(`verdicts.stage`), `db.unlabelled_references`, the findings stage filter,
`warlock/judge.py`, `service/judge.py` and the labelling grid in Review all
shipped, and the design was implemented as written rather than re-designed —
including the parts that read as fussy, because each of them is the difference
between a probe that means something and one that is merely believed. The
end-to-end path was verified against real DINOv2 weights (768-d CLS embeddings,
fit, save, load, score) rather than only against stubs.

**What it is blocked on is a human.** No probe exists on this machine because
nobody has labelled anything: the corpus is 3 accepts against 81 rejects, all of
them mesh verdicts. `fit` returns `None` below `MIN_PER_CLASS` of each class on
purpose. So §7 below is now a description of a *session to run*, not of code to
write, and §8's gate is unchanged: positives in the tens, from §2's re-run.

*(The build plan's Phase 6.)*

## 7. Phase 1 — the 2D judge — built, and waiting for labels

**Everything in this section is implemented.** What it is blocked on is **a
labelling pass** — and, for the mesh half, §8's corpus.

How to run one: open Review, and under the sweep list pick one of the two passes
under *Teach the judge*. The centre column becomes a grid; `A` is good, `R` is
bad, there is no reason step, and the pass owns the keyboard while it is open so a
keypress about a picture can never be filed as a verdict about a mesh. Retraining
happens as you go, off the frame thread. Do **both** passes over the same images
if you want both probes: they are different questions and neither answer implies
the other.

Two things to expect, both deliberate. Nothing appears for a while at the top of a
large grid — thumbnails upload one per frame, which is `StripRender`'s rule at a
hundred cells. And images marked as refused are in the list on purpose: they are
the most informative negatives available.

### Why this was cheaper than it looked

The seam was already built and tested, not hypothetical:

- `verdicts.source` is a free string, never an enum, and `latest_verdicts` keys
  on `(job_id, source, stage)`. A judge's opinion sits *beside* a human's and can
  never overwrite it — and, since migration 7, an answer to one question can
  never overwrite the answer to another about the same image.
- `db.unverdicted_models(source=...)` filters per source, so a judge run resumes
  where it stopped; `tests/test_verdicts_db.py` already exercises
  `source="ai:demo"`.
- Verdicts are append-only, latest-wins by max rowid, and carry a
  **denormalized** `vector`, `prompt_hash` and sweep context — which is what lets
  the corpus outlive `prune_jobs` deleting the assets it was learned from.
- `models/dinov2-base` is downloaded and wired: `bench/metrics.py` extracts the
  normalized CLS token and caches the model per `(path, device)`.
- `service/verdicts.py` already defines the vocabulary:
  `REASONS = ("holes", "bad-shape", "bad-texture", "wrong-style", "broken")`.
- The corpus is scoped per subject and carries refusals: `findings.json` v3
  breaks marginals out per `prompt_hash`, and a reference refused at the gate is
  an observation rather than a silent hole in the record.

The missing piece is **calibration**. `meshaudit` and `meshreport` measure holes,
watertightness, triangles and pivot on every job, and nothing has ever
established that `hole_worst = 0.04` means *reject*. Labelled verdicts are what
convert a measurement into a decision boundary.

**And the first calibration result is that the boundary runs the wrong way.**
`AUC(hole_worst → reject) = 0.115` — not uninformative, inverted (§2). Two
consequences. A probe must be trained on **pixels**, never on the audit scalars,
which would have to be sign-flipped to beat a coin and would then be fitting the
slab artefact rather than quality. And `hole_worst` is disqualified as a sanity
check *on* the probe: a probe that disagrees with it is, on this evidence, more
likely to be right.

### Where the factory analogy breaks

**Break 1 — two products, not one product and one blank.** In 2D mode the image
*is* the deliverable: jobs run with `output="reference"`, `import_reference`
mints one as a finished asset, users export it. So the same PNG is sometimes the
product and sometimes the input to the next machine, and "good" means opposite
things:

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
~15 s for a mesh — same label count, a fifth of the wall-clock.

### Decisions already taken

| Decision | Choice | Why |
|---|---|---|
| Authority | **Advisory only** | Files a verdict and sorts Review; never refuses, deletes or retries. The only mode in which its accuracy can be measured before it is trusted. Mirrors the `native.py` doctrine that an optimisation never replaces the reference it is checked against. |
| Labelling and training | **In-app, in Review mode** | One loop, so the judge improves as the corpus is reviewed — the analogue of the camera's teach mode. |
| Label storage | **`verdicts`, new typed `stage` column** | A real column, not a naming convention on `source`. |
| Scope | **Both 2D images and 3D meshes** | Non-negotiable requirement. |

### Three probes, one artifact pipeline

Intent, not artifact type, separates the models:

| Probe | Input | Judges | Blocked on |
|---|---|---|---|
| `image-as-product` | `reference.png` from a `reference`-stage job | Is this a good 2D asset? | a labelling pass |
| `image-as-blank` | `reference.png` from a `model`-stage job | Will this reconstruct? | a labelling pass |
| `mesh` | 8 rendered views of `model.glb` | Is this a good mesh? | **a corpus with positives in it** — see §8 |

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
  implementation is never deleted. `REFUSAL_CODES` is the vocabulary that
  comparison is spelled in.
- **Fully offline.** DINOv2 loads from a local path with `local_files_only=True`.
  The probe weights are ours. Nothing downloads at runtime — and note the model
  *downloader* does not change this: it is a separate process, reachable from one
  button, and never from the job path.
- **A judge failure must never fail a job.** Inference logs and swallows, exactly
  as `Worker._record_observation` does, and for the same reason.

### Data model

**Migration 7: `verdicts.stage`**, three values — `'reference' | 'blank' |
'model'`. **This was written down as migration 6 and is not: Night 2's candidate
columns took 6 by landing first.** Migrations are append-only and never edited
once shipped, so confirm `len(MIGRATIONS)` before writing the entry rather than
trusting this line.

Three values rather than two, and the third is load-bearing. The instinct is a
two-value column mirroring `jobs.stage` (already exactly `'reference' | 'model'`)
and recovering intent by joining to the job. That breaks: `prune_jobs` deletes
job rows, and the sole reason `verdicts.vector` is denormalized is that the
corpus must outlive the assets. A label whose meaning depends on a row that no
longer exists is uninterpretable the moment it matters. Existing rows backfill to
`'model'` — every verdict to date is about a mesh.

**New query, `db.unlabelled_references()`.** `unverdicted_models` cannot be
reused and cannot be fixed with a parameter: it filters `status = 'done' AND
stage = 'model' AND sweep_id IS NULL`, excluding the two things a labelling pass
most needs — **errored jobs** (a reference refused for multi-object is the most
informative negative available) and **sweep units** (the entire corpus this work
is built on).

**`findings.py` must filter by stage.** `aggregate` runs over every source's
latest verdicts. Matched pairs group by `(sweep_id, source)` and stay clean, but
the **marginals do not**: an image label would count into the same accept/reject
rate as a mesh verdict, and `"accept 6/8"` under a prompt control would silently
average two different questions. Mesh findings must read `stage = 'model'` only.
This applies in two places, not one — the per-subject `prompts` section is built
by the same `_marginals` helper, so the stage filter belongs *there* rather than
at either call site.

> **As built, three deliberate departures from the text above.** Each is a case
> where following the plan literally would have left a hole.
>
> 1. **The stage filter is at the top of `aggregate`, not inside `_marginals`.**
>    The reasoning above is right about `_marginals` covering both call sites and
>    incomplete about who else reads that list: `vectors` would advertise an image
>    label as a *ranked mesh configuration*, and `_comparisons` pairs rows sharing
>    a sweep and a seed whose vectors differ in one key — which a blank label and
>    a mesh verdict **on the same unit** satisfy, yielding a matched "pair"
>    comparing two different questions. One filter covers all four consumers.
> 2. **`latest_verdicts` groups by `(job_id, source, stage)`.** Not stated above
>    and not optional: without stage in the grouping, labelling the same image
>    under both intents makes the second answer supersede the first, which is
>    exactly what the column exists to prevent.
> 3. **The mesh probe's stage is spelled `model`, not `mesh`.** The probe table
>    says `mesh`; the label column says `model`. Two names for one thing needs a
>    translation table, and the drift would surface as a probe trained on the
>    wrong population, so `judge.STAGES == verdicts.STAGES` is asserted by a test.
>
> Also new and not in the plan: **`db.unlabelled_references` derives the
> population it lists from the label stage** (`LABEL_POPULATION`), because the
> probe table is explicit that the product question is about a *reference*-stage
> job's image and the blank question about a *model*-stage job's — a single
> listing would have trained each probe on the other's population. And
> `record_verdict`'s `status == 'done'` rule became a rule about **the artifact a
> verdict judges**: an image label needs the image (a refused job has one), while
> the mesh side is untouched.

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
  guard, see §9.
- Training runs on `TaskRunner`. A DINOv2 forward pass over a hundred images plus
  a logistic fit is seconds, and seconds on the frame thread is a freeze.

### Scope of Phase 1

Migration 7, `unlabelled_references`, the findings stage filter, `judge.py` with
embed + fit + score, the labelling grid, and both image probes. Ends with a
held-out accuracy figure. This is the whole of the value for 2D mode and the
cheapest useful gate for 3D.

**All of that is built; the held-out accuracy figure is not, and cannot be.** It
is a measurement over labels that do not exist yet, and it is what §10 specifies —
so the remaining Phase 1 work is: run a labelling session, then write the
measurement document. One thing was added on the way that the scope list did not
name, and it earns its place: `judge.fit` returns **`None`** rather than a probe
below `MIN_PER_CLASS` (8) of each class, so a corpus of 3 accepts against 81
rejects produces no probe at all instead of one that has learned "reject" and
scores 96% doing it. A missing probe is visible in the panel; a confident useless
one is not.

## 8. Phase 2 — the mesh judge

Reuses `bench/views.py`'s 8-view render, adds the pooling adapter and the third
probe.

**Not blocked on labelling — blocked on a corpus that contains acceptable
meshes.** The review produced **3 positives against 81 negatives**. That is not a
thin corpus, it is an unusable one: a linear probe fitted to it learns "reject"
and scores 96% accuracy doing so.

So this phase sits behind §2's re-run, and the gate on starting it is a positive
count in the tens, not a total label count. The 84 existing labels are not
wasted — they are a clean negative set, and the matched pairs (identical
`input.png`, opposite verdict) are the most useful training rows in the corpus
precisely because everything except the matte is held constant.

## 9. Phase 3 — earned authority

Only after §10's numbers exist. Candidates, in increasing order of risk:

1. Sorting Review by score (already in Phase 1).
2. A bounded reference-seed re-roll on a predicted-bad blank. The mechanism is
   already there and already on — `Config.reference_retries` is 2 and the loop
   holds `mesh_seed` fixed — so this is only a matter of letting the probe, not
   `reference.py`'s three thresholds, decide what counts as bad.
3. Refusal at the gate, which should not be attempted without a measurement
   document.

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
- **Label volume — measured, and worse than feared.** The original estimate was
  83 passing and 17 refused references, with seventeen negatives called thin. The
  review came back **3 accepts and 81 rejects**, so the *positive* class is the
  scarce one and the mesh probe is unbuildable on this corpus (§8). The
  deliberate labelling session this predicted is confirmed necessary, and it has
  to run against output worth accepting.
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

## 10. What "it works" has to mean

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
  probe on* — `baseline s23` is the canonical test case. Agreement is cheap to
  compute per rule rather than in aggregate, since a refusal is an observation
  carrying `refused_<code>`.
- **Per-subject breakdown by `prompt_hash`**, or the scope risk is unmeasured.
- **Beat `hole_worst`, which is a floor of 0.115 AUC and therefore no floor at
  all.** Its inversion (§2) means the honest baseline to beat is a coin flip, and
  any probe that merely correlates with the audit scalars has learned the slab
  artefact. Report the probe against the human labels directly.

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

# Later, and specced only in outline

## 11. Retopo + bake prototype

*(The build plan's Phase 7. Needs §4's rigging thread; characters only.)*

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

## 12. External backend A/B

*(The build plan's Phase 8. Last, and it gets its own spec first.)*

- **SkinTokens/TokenRig** as an isolated out-of-process worker — the
  `trellis-server` / `blender_worker` pattern, kill-on-close job, weights by
  one-time manual download or through the new fetcher, doctor reporting absence
  non-fatally. Run in existing-skeleton mode against Warlock's template so bone
  names and animation compatibility hold. A/B against the welded heat weights on
  a fixed rig corpus using the deformation battery, verdicts through Review.
- **Hunyuan3D 2.1** as an optional isolated reconstruction backend, same
  isolation rules, benchmarked on the curated references from §2 — human
  acceptance, silhouette, runtime, VRAM. It must fit the coexist/exclusive
  `vram.plan` machinery: a new backend declares its GiB cost in the cost table.
- **Explicitly out:** MeshAnything V2 as a retopo path (sub-1600-face target).

---

## Deferred by decision — not open work

| Item | Status |
|---|---|
| `seam.SEAM_MAX = 2.0` uncalibrated (`pipelines/seam.py`) | Open, low priority. Needs stone / plaster / gravel / fabric tiles eyeballed. Corpus-keyed, so it owes a measurement doc before it moves. |
| `docs/measurements/2026-08-06-pixel-art-xl.md` — "run not yet taken" | Unblocked: all three recipes and all weights verified present. A three-arm run settles which arm `pixel_sprite` names and where `GRID_RESIDUAL_MAX` belongs (0.05; the doc says "that number is a guess"). Independent of everything — good use of idle GPU time alongside §2. |
| Fused brush dab kernel (`warlockc_dab_u8`) | Deferred on purpose; the gate is "the brush shows up in a profile first". ABI 5, four kernels shipped. |
| `studio/clay/ops_topo.py` hole-fill UV | Documented, deliberate approximation. |
| View-matched reference ranking | Not built on purpose; the Scattered verdict *is* the deliverable (`docs/measurements/2026-08-04-view-calibration.md`). |

---

## Closed

The 2026-08-07 QA audit section that used to close this file has been removed:
every item in it was fixed, tested and recorded where it belongs — the
`same_line`-past-the-edge bug in `widgets.same_line_or_wrap` plus two standing
guards in `tests/test_studio_smoke.py`, and the Windows directory-mtime finding
in `CLAUDE.md` and
`docs/measurements/2026-08-07-directory-mtime-granularity.md`. A finished plan is
deleted rather than ticked, by this repo's own rule; `git log` keeps it, and the
code is the record.

Likewise removed: the six items this file listed as done on 2026-08-07 (the
`bg_removal` default, `PROMPT_TEMPLATE`, `reference_retries`, refusal codes, the
per-subject findings section, and the `MAX_UNITS` comment), and the eighteen
packages of the Night 1 and Night 2 build runs. `git log --graph` reads as
identifiable `--no-ff` merges, one per package.
