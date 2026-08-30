# The art-verdict programme: four questions, pre-registered — 2026-08-30

**Status: pre-registration written 2026-08-30 before any image was generated.**
Everything under "What will be run" and "Decision rules" is fixed first; numbers
and the adjudication go under "Results" in the per-question documents, and
whichever rule fired is applied verbatim, including the boring one. That
ordering is the only thing that makes the answers worth anything
([`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md),
[`2026-08-29-seam-threshold-cfg.md`](2026-08-29-seam-threshold-cfg.md)).

[`2026-08-09-grade-scale.md`](2026-08-09-grade-scale.md) puts the reason in one
line: *"Written now, this document is a commitment; written after the grades
exist, it is a rationalisation with a table in it."*

## Why now

Three capability tracks shipped on 2026-08-30 — a game-ready remesh
(`pipelines/remesh.py`), local style-LoRA training (`pipelines/lora_train.py`)
and masked regeneration inside the Inker (`studio/inker/inpaint.py`) — and none
of the three has run on a card. They join four art verdicts already owed
(`TODO.md` P3, P4/P12, P15, P16) and a default mesh profile that is still `raw`
because its tiers were never qualified (`config.py:238`).

The tree is therefore accumulating unmeasured capability faster than it is
accumulating evidence. This programme is the correction: it adds nothing, and
measures four things that already ship.

## This is the retry 2026-08-13 demanded

[`2026-08-13-tier-qualification.md`](2026-08-13-tier-qualification.md) scored
**zero usable of twenty** — four subjects (chest, sword, rock, knight) at five
seeds each — and its corpus gate fired, so no tier moved. It closes with three
instructions to whatever ran next, and this document answers all three:

1. *"A retry at different settings or seeds is a new pre-registration, not an
   amendment here."* — this is that document.
2. *"Should re-frame its subjects to be trellis-friendly and pre-register its
   bar against the graded scale."* — see the corpus and the decision rules.
3. *"The corpus assets were not retained… any future diagnosis run should plan
   around by retaining its evidence until it is written up."* — see Retention.

Its Amendment 1 also settled that there was **no code regression**, so subject
choice and the instrument are what remain. Two measured facts about subjects
carry into the corpus below: boxes failed at **0/30** on the Zelda crate, and
the rock of that run *"reconstructs as a hollow foliage lattice"*.

## What is under test

The shipped default path and nothing else: `text → sdxl_cfg → TRELLIS`, at
whatever `Config` supplies with no overrides —

- `Config.mesh_profile` = `raw` (no gltfpack tier applied)
- trellis texture resolution pinned at 512 (`config.py`; a server launch flag,
  not per-request)
- `bg_removal` = `guidance.default_bg_removal(...)`, i.e. `birefnet` when its
  weights are on disk
- no style LoRA, no IP-Adapter, no ControlNet
- seed 42 for every subject
- prompt framing from `pipelines/prompt.PROMPT_TEMPLATE`, which supplies *"a
  single subject centered on a plain light gray background, no other objects,
  3/4 perspective view, studio lighting, game asset render, full object in
  frame, no cropping, no text, no watermark"*

Corpus prompts are therefore **bare subject descriptions**. Anything this
programme could override is a departure from the configuration the verdict is
about, so `scripts/campaign_props.py` passes nothing.

## The instrument

The mesh grade scale, **cited and not restated**: integer −5..+5, model stage,
declared in [`2026-08-09-grade-scale.md`](2026-08-09-grade-scale.md). The usable
cut is `vectors.USABLE_GRADE = 3`, applied as `grade >= 3`. This programme
introduces no new threshold and moves no existing one.

**The pass is blind, and what that does here is narrower than usual.** Blinding
is a property of the session in `studio/review_mode.py`: it renames units to a
neutral id prefix and orders them by a digest of the job id. On a two-arm sweep
that hides the arm. **On this corpus it cannot hide the subject** — the mesh is
the thing being looked at and the prompt names it — so what it actually buys is
the **order randomisation**, which is not nothing: the corpus file lists eight
`easy` subjects consecutively, and grading them as a run is a real drift.

The protection that matters here is therefore not blinding at all. It is that
**the difficulty class is never written onto a job**, so the reviewer cannot be
told which subjects were predicted hard. Blinding is switched on anyway, for
the ordering; claiming more for it than that would be the kind of overstatement
this document exists to prevent.

**One reviewer, and that is a limitation not a design.** The
`bg_removal` signal that
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) chased was unblinded and
single-reviewer, which is why that campaign's rule is a small blind confirm
before anything is leaned on. The same caveat applies to every number this
programme produces and is to be repeated in each writeup rather than assumed
read.

---

## Q1 (P3) — does the shipped default produce usable props?

### What will be run

`docs/measurements/corpora/props-v1.txt`: **22 subjects at seed 42**, submitted
by `scripts/campaign_props.py` as ordinary library jobs tagged `props-v1`.

Breadth over depth, one seed per subject, and the argument is 2026-08-13's own
data: three of its four subjects scored zero accepts across *five* seeds each,
so subject dominated seed by a wide margin. Seed variance is explicitly **not**
measured here; if the result is marginal, a seed-replicate run is a new
pre-registration, not an amendment to this one.

Every subject is classed **before the run**, as a prediction of how hard it is
for a single-view reconstruction:

| class | n | what it means |
|---|---|---|
| `easy` | 8 | compact, solid, rounded, closed silhouette, strong surface texture |
| `medium` | 8 | a handle, spout or protrusion; moderate concavity |
| `hard` | 3 | thin or flat parts, open lattice, or flat-faced boxes |
| `humanoid` | 3 | Q2's subjects; a different rubric, see below |

### The control subject

`hard | a wooden treasure chest with iron banding` is in the corpus **as a
calibration point, not as a fair sample**. It is the one shape with a prior
measurement: −5 at all five seeds on 2026-08-13, and boxes at 0/30 on the Zelda
crate. Its reading is pre-committed:

- **chest scores −5 or −4** → the corpus is calibrated against the earlier run,
  and the `easy`/`medium` classes carry the finding.
- **chest scores ≥ +3** → something material changed between 2026-08-13 and
  today that nobody has identified. The comparison to that run is void, the
  headline number below is **not** to be reported against it, and finding the
  change is its own investigation before anything here is leaned on.

### What is reported

Usable-of-N (`grade >= 3`) **overall, and per class**. Both, always. An
aggregate dragged down by three subjects predicted hard in advance is not a fact
about the shipped default, and reporting only the aggregate is how a corpus
chosen for spread gets read as a product verdict.

### Decision rules

Fixed now, applied verbatim, on the 16 `easy` + `medium` subjects:

- **≥ 50% usable** → the README may state a usable-of-N figure for props at the
  shipped default, with the corpus and its date named beside it.
- **25–49% usable** → the README states the figure *and* calls generated 3D a
  draft path. No "game-ready" claim.
- **< 25% usable** → the README makes **no** positive quality claim for
  generated 3D, and says so plainly. This closes the release audit's "no
  positive quality evidence" item in the honest direction, which was always one
  of its two acceptable outcomes.

Whatever fires, `Config.mesh_profile` does **not** move here: qualifying the
gltfpack tiers is `scripts/qualify_tiers.py`'s separate bar, which needs
accepted meshes as its input. If this run produces some, that qualification
becomes possible for the first time — that is a consequence, not a goal, and it
gets its own document. (`qualify_tiers` warns when its subjects are not
recognisably *a chest, a sword and a rock*; props-v1 has a chest and a rock and
no sword, so expect that warning. Adding a sword to satisfy a harness's name
matcher would be fitting the corpus to the tool rather than to the question.)

---

## Q2 (P12) — are generated humanoids reconstructable at all?

### What will be run

The three `humanoid` subjects of the same corpus, through the same path, then a
`fit_template` pass.

### Rubric, fixed now

Judged on **limb separation** and **silhouette**, in that order. Not on face
fidelity, which does not survive sprite scale and is not what Troupe consumes.
The mesh-usability grade is *not* the instrument here — a humanoid that is a
poor prop can still be a viable Troupe input, and vice versa.

**Pre-declared and therefore not a finding:** reconstruction is single-image, so
**the back is hallucinated**. A back-side defect is expected, is recorded, and
does not count against the verdict unless it breaks the silhouette from the
sides.

### Decision rule

- **2 or 3 of 3 show separable limbs and a readable silhouette** → the
  generated-character path is viable; Troupe Phase 7 is worth planning.
- **0 or 1 of 3** → the generated-character path is not viable at the shipped
  default, and Troupe's supplied-base-mesh path (P4) is the only one worth
  investing in. This is the cheaper answer to get wrong in the safe direction,
  and it is the whole reason this question is asked before Phase 7 rather than
  after.

---

## Q3 (P15) — is a generated terrain set paintable?

### What will be run

Create → Sheet → **Terrain set**, two surfaces that ought to meet — grass into
dirt — at 32px. Then into Plotter, painted with the **Terrain** tool, looking at
the joins where the brush turns a corner and where two strokes meet. Then the
map is exported and **opened in real Tiled**.

The last step is P6 and P7's argument and not a new one: our writer and our
reader agree on the blob-47 ordering *by construction*, so a round trip through
our own two halves cannot catch an error both halves make together. A terrain
set is the case where that matters most.

### Failure modes, named in advance

Pre-named so that finding one is a result rather than an anecdote: a coverage
field that is *right* and *reads wrong*; a boundary too soft at 16px; two
materials whose scales disagree.

### Decision rule

- **A map a person would actually paint with, and Tiled opens it without
  complaint** → the terrain path is real; `docs/COMPAT.md`'s Tiled rows can
  finally make a claim about Tiled rather than about ourselves.
- **Any of the three named failure modes** → that is the finding, it gets the
  document, and the terrain set stays unrecommended until it is fixed.

Note for the writeup: `SEAM_MAX` is **not** owed a re-measurement here.
[`2026-08-29-seam-threshold-cfg.md`](2026-08-29-seam-threshold-cfg.md) already
ran that on `sdxl_cfg` and corrected the premise that it had not been.

---

## Q4 (P16) — does an eight-direction action sheet read?

### What will be run

One character, one reference, then `attack8` at 32px and `walk8` at 32px. The
walk played at 10fps in Inker, and the character turned through all eight
directions.

### Three questions, judged separately

Fixed now, and separately on purpose — a sheet can pass any two and fail the
third, and one merged verdict would hide which:

1. Does **one identity** survive all eight bands?
2. Does the **action read as the action**?
3. Does the **front row look like a different picture** from the back row it is
   a literal copy of, with only the prompt clause changed?

### Known limits — recorded, so they cannot be reported as findings

Front and back rows are copies (the convention `walk.json` and `idle8.json`
already set); `run8`'s knee-drive frames read a little like a crumple in
profile; `cast8`'s release is weaker head-on than in profile, because a forward
thrust has nowhere to go in an orthographic front view. All three are already
known. Re-discovering one is not a result.

### Decision rule

- **All three pass** → the action set is worth having, and the remaining guides
  are worth authoring. P17 (four-direction guides) becomes a live question.
- **Question 1 fails** → identity drift is the blocker and no amount of guide
  authoring fixes it; the next work is conditioning, not art.
- **Question 2 or 3 fails alone** → the specific guide is reworked; the
  programme survives.

---

## Q5 (P4/P5) — does a supplied humanoid reach a rendered sheet?

Added 2026-08-30, when a textured rigged humanoid arrived and unblocked the
entry that had been waiting on one.

### What will be run

`tests/fixtures/humanoid/cesium_man.glb` through **Send to Troupe** with
`palette=cosmos`, then the rig, then a `charsheet` job against real Blender.

### What it can and cannot settle

**P5 cleanly, P4 only partly.** P5 asks whether the `charsheet` job runs end to
end on hardware — a mechanism question, and this file answers it. P4 asks
whether the palette ramp works at sprite scale, and the answer this file
supports is weak: CesiumMan is a 3,273-vertex specification sample with a small
JPEG, not character art anyone would ship, and there is no female variant. The
art half of P4 still needs the authored or commissioned mesh.

Recorded so the writeup cannot overclaim: **a good Q5 result is evidence about
the pipeline, not about the art.**

### Three defects it already found, in code

The rig path had only ever been given TRELLIS reconstructions — no armature, no
vertex groups, no parent — and a supplied humanoid arrives with all three. Each
consequence was silent:

1. **`_skin`'s failure guard stopped working.** `_has_weights` asks whether
   *any* vertex group carries a weight, so an incoming skin answers yes before
   the new armature is bound at all. A bind producing nothing was reported as a
   clean `automatic` rig — the app says it succeeded and the character does not
   deform.
2. **Two skeletons in the export.** `_import_glb` returns the joined mesh and
   leaves the scene alone; `_export` writes the whole scene.
3. **The measurements were wrong, which is the worst.** `_import_glb` bakes the
   Y-up → Z-up rotation into the vertex data, but a skinned import parents the
   mesh to its armature and that parent still carries the rotation, so
   `matrix_world` applies it twice. Measured: **(0.505, 0.896, 1.458) against a
   true (1.138, 0.312, 1.507)** — an arm span under half its real width.
   `_rig_bones` fits the template to that box, so every joint lands wrong while
   the stature stays plausible enough to pass a glance.

`_strip_incoming_rig` fixes all three, and `tests/test_rig_supplied_mesh.py`
pins them. **The card half of Q5 is still owed** — none of this says the
resulting rig deforms well, only that it is now fitted to the right body.

---

## Protocol

**Order.** A smoke session first — remesh one existing accepted mesh at `medium`
and confirm `remesh.report_line` says *quads* rather than the decimate fallback;
one short LoRA training run well under the 800-step default, to prove the child
process, marker protocol and registration; one Inker inpaint. Finding a worker
crash there costs minutes; finding it after Q1's corpus costs the afternoon.

**The remesh half of that smoke was taken in code on 2026-08-30, and it
failed.** `tests/test_remesh_worker.py` runs `op_remesh` for real — nothing ever
had; `test_remesh.py` fakes the child at `rigging.run_worker` — and its first
run came back `method="decimate"`, `quads=0.0` on a closed, manifold UV sphere.

The cause: glTF cannot share a vertex position between two texture coordinates,
so every GLB splits its vertices at each UV seam, and quadriflow refuses
non-manifold input. Measured on that sphere — **1,106 vertices before export,
4,512 after the round trip, quadriflow answering "Remeshing failed"**. Since
every input on this path is a GLB, the quadriflow branch could never succeed:
**every remesh silently took the triangle fallback**, and a feature whose
profiles are spelled in quads shipped triangles from the day it landed.

`op_remesh` now welds the working copy before remeshing, reusing the existing
`_weld` — whose own docstring already names this root cause for the skinning
path — and the same sphere returns **479 faces, all quads**. The bake source is
deliberately left unwelded, since it is what the colour and normal passes read.

`report_line` was never wrong about this: it said *"decimated: the surface was
not manifold enough to quad"* throughout, which is why the defect was invisible
rather than misreported. The card half of the remesh smoke is still owed — a
real 300k-face reconstruction is a different input from a sphere — but it now
starts from a path that can reach quadriflow at all.

**Retention.** The library is **not** cleaned until every writeup exists. This
is 2026-08-13's own instruction, from the run whose twenty meshes and job rows
were deleted before anyone could re-inspect them. The corpus is tagged
`props-v1`; that tag is how it is found again. Note that `sweeps.cleanup_sweep`
cannot reach these rows — it refuses `RECENT_ID` and these are ordinary library
jobs — so the only thing that can destroy this evidence is a manual **Clean
library**.

**Blinding.** On, for the Q1/Q2 grading pass.

## What this programme does not measure

Stated so that its silence is not read as a result:

- **`pipelines/remesh.py`**, beyond whether it runs. It is an opt-in rework, not
  the shipped default, so it is not what Q1 is about. Qualifying it is a corpus
  question of its own and needs Q1's accepted meshes as input.
- **The gltfpack tiers.** `scripts/qualify_tiers.py`'s bar, separate document.
- **Seed variance.** One seed per subject; see Q1.
- **Time-to-usable in artist-minutes** — the metric that would actually settle
  "minimal post-editing". It requires someone repairing meshes and timing it,
  which is a different session with a different instrument. It is the natural
  successor to this programme and is deliberately not smuggled into it.
